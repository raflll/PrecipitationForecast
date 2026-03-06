#!/usr/bin/env Rscript
# ==============================================================================
# Train XGBoost Gamma Regression Models for Precipitation Forecasting
#
# R port of train_xgboostlss.py. Uses xgboost with objective="reg:gamma" for
# Gamma regression of the conditional mean, with the shape parameter estimated
# via MLE from training residuals. This gives a full Gamma(shape, rate)
# predictive distribution for each observation.
#
# Features:
#   - Bayesian hyperparameter optimization (ParBayesianOptimization, like Optuna)
#   - SHAP analysis (SHAPforxgboost)
#   - Full diagnostic suite: PIT, reliability, prediction intervals
#   - Identical visualizations to the Python pipeline
#
# Install required packages:
#   install.packages(c("arrow", "xgboost", "data.table", "ggplot2",
#                      "SHAPforxgboost", "ParBayesianOptimization",
#                      "scoringRules", "patchwork", "jsonlite"))
#
# Usage:
#   Rscript src/train_xgboost_gamma.R                  # All horizons
#   Rscript src/train_xgboost_gamma.R --horizon 7      # Single horizon
#   Rscript src/train_xgboost_gamma.R --quick           # Quick test run
# ==============================================================================

suppressPackageStartupMessages({
  library(arrow)                     # parquet I/O
  library(xgboost)                   # gradient boosting (reg:gamma)
  library(data.table)                # fast data manipulation
  library(ggplot2)                   # plotting
  library(SHAPforxgboost)            # SHAP values + plots
  library(ParBayesianOptimization)   # Bayesian HP tuning (like Optuna)
  library(scoringRules)              # CRPS for gamma
  library(jsonlite)                  # JSON I/O
})

# Optional: patchwork for combining SHAP dependence plots
has_patchwork <- requireNamespace("patchwork", quietly = TRUE)
if (has_patchwork) library(patchwork)


# ==============================================================================
# Configuration
# ==============================================================================

# Detect project root (works both from Rscript and interactive R)
script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) getwd()
)
PROJECT_ROOT <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
if (!dir.exists(file.path(PROJECT_ROOT, "data"))) {
  PROJECT_ROOT <- getwd()
}

DATA_FILE  <- file.path(PROJECT_ROOT, "data", "training", "training_dataset.parquet")
MODELS_DIR <- file.path(PROJECT_ROOT, "models")
VIS_DIR    <- file.path(PROJECT_ROOT, "visualizations")
dir.create(MODELS_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(VIS_DIR, recursive = TRUE, showWarnings = FALSE)

HORIZONS <- c(3, 7, 14, 21, 28, 35)
NON_FEATURE_COLS <- c("init_date", "horizon_days", "ground_truth_precip", "source")


# ==============================================================================
# Data Loading
# ==============================================================================

load_data <- function(horizon) {
  df <- as.data.table(read_parquet(DATA_FILE))
  df <- df[horizon_days == horizon]
  df[, init_date := as.Date(init_date)]
  setorder(df, init_date)

  # Temporal split (same as Python)
  train <- df[year(init_date) <= 2023]
  val   <- df[year(init_date) == 2024]
  test  <- df[year(init_date) == 2025]

  cat(sprintf("  Train: %d rows (%s to %s)\n", nrow(train),
              min(train$init_date), max(train$init_date)))
  cat(sprintf("  Val:   %d rows (%s to %s)\n", nrow(val),
              min(val$init_date), max(val$init_date)))
  cat(sprintf("  Test:  %d rows (%s to %s)\n", nrow(test),
              min(test$init_date), max(test$init_date)))

  list(train = train, val = val, test = test)
}

get_feature_cols <- function(df) {
  setdiff(names(df), NON_FEATURE_COLS)
}


# ==============================================================================
# Feature Category Classification (for color-coded importance plots)
# ==============================================================================

classify_feature <- function(feat) {
  if (grepl(paste0("^cum_|^mean_6h|^max_6h|^min_6h|^std_6h|^median_6h|",
                   "forecast|spread|^q[1-4]_apcp|6h|top10|^gefs_|^persistence"),
            feat)) {
    "Forecast"
  } else if (grepl("^clim_", feat)) {
    "Climatology"
  } else if (grepl(paste0("^precip_|^rainy_|^max_precip|^days_since|^thunder|",
                          "^fog|^mean_temp|^temp_|^mean_wind|^mean_gust|",
                          "^mean_sun|^sunshine|^mean_precip_h|^rain_flag|",
                          "^haze|^mean_dtr"),
                   feat)) {
    "Recent Conditions"
  } else {
    "Calendar/Cross"
  }
}

CATEGORY_COLORS <- c(
  "Forecast"          = "#2196F3",
  "Climatology"       = "#FF9800",
  "Recent Conditions" = "#4CAF50",
  "Calendar/Cross"    = "#9C27B0"
)


# ==============================================================================
# Gamma Shape Parameter Estimation (MLE)
# ==============================================================================

estimate_gamma_shape <- function(y, mu) {
  # MLE for Gamma shape given observed y and predicted means mu.
  # Solve: log(shape) - digamma(shape) = log(mean(y/mu)) - mean(log(y/mu))
  valid <- y > 0 & mu > 0
  y  <- y[valid]
  mu <- mu[valid]

  s <- log(mean(y / mu)) - mean(log(y / mu))
  if (s <= 0) return(1.0)

  # Initial estimate (method of moments)
  shape <- 0.5 / s

  # Newton-Raphson
  for (i in seq_len(50)) {
    g <- log(shape) - digamma(shape) - s
    h <- 1 / shape - trigamma(shape)
    shape_new <- shape - g / h
    if (shape_new <= 0) shape_new <- shape / 2
    if (abs(shape_new - shape) < 1e-8) break
    shape <- shape_new
  }

  shape
}


# ==============================================================================
# Model Training with Bayesian Hyperparameter Optimization
# ==============================================================================

train_model <- function(train_dt, val_dt, feature_cols, horizon, quick = FALSE) {
  X_train <- as.matrix(train_dt[, ..feature_cols])
  y_train <- train_dt$ground_truth_precip
  X_val   <- as.matrix(val_dt[, ..feature_cols])
  y_val   <- val_dt$ground_truth_precip

  dtrain <- xgb.DMatrix(X_train, label = y_train)
  dval   <- xgb.DMatrix(X_val,   label = y_val)

  n_trials   <- if (quick) 10L else 30L
  max_rounds <- if (quick) 100L else 500L
  n_init     <- min(5L, n_trials)
  n_iter     <- n_trials - n_init

  cat(sprintf("  Running Bayesian HP optimization (%d trials)...\n", n_trials))

  # Objective function for Bayesian optimization (maximize = minimize nloglik)
  xgb_score <- function(eta, max_depth, gamma_reg, subsample,
                         colsample_bytree, min_child_weight,
                         reg_alpha, reg_lambda) {
    params <- list(
      objective        = "reg:gamma",
      eval_metric      = "gamma-nloglik",
      eta              = eta,
      max_depth        = as.integer(round(max_depth)),
      gamma            = gamma_reg,
      subsample        = subsample,
      colsample_bytree = colsample_bytree,
      min_child_weight = min_child_weight,
      reg_alpha        = reg_alpha,
      reg_lambda       = reg_lambda,
      seed             = 42L,
      nthread          = max(1L, parallel::detectCores() - 1L)
    )

    cv <- xgb.cv(
      params                = params,
      data                  = dtrain,
      nrounds               = max_rounds,
      nfold                 = 3L,
      early_stopping_rounds = 20L,
      verbose               = 0L
    )

    best_score <- min(cv$evaluation_log$test_gamma_nloglik_mean)
    list(Score = -best_score, nrounds = cv$best_iteration)
  }

  bounds <- list(
    eta              = c(1e-3, 0.2),
    max_depth        = c(2L, 5L),
    gamma_reg        = c(1e-8, 40.0),
    subsample        = c(0.5, 0.9),
    colsample_bytree = c(0.3, 0.7),
    min_child_weight = c(10.0, 500.0),
    reg_alpha        = c(0.01, 50.0),
    reg_lambda       = c(0.01, 50.0)
  )

  set.seed(42)
  opt <- bayesOpt(
    FUN        = xgb_score,
    bounds     = bounds,
    initPoints = n_init,
    iters.n    = n_iter,
    iters.k    = 1L,
    acq        = "ucb",
    verbose    = 1L
  )

  # Extract best parameters
  best_idx <- which.max(opt$scoreSummary$Score)
  best_row <- opt$scoreSummary[best_idx, ]

  best_params <- list(
    objective        = "reg:gamma",
    eval_metric      = "gamma-nloglik",
    eta              = best_row$eta,
    max_depth        = as.integer(round(best_row$max_depth)),
    gamma            = best_row$gamma_reg,
    subsample        = best_row$subsample,
    colsample_bytree = best_row$colsample_bytree,
    min_child_weight = best_row$min_child_weight,
    reg_alpha        = best_row$reg_alpha,
    reg_lambda       = best_row$reg_lambda,
    seed             = 42L,
    nthread          = max(1L, parallel::detectCores() - 1L)
  )

  # Re-run CV with best params to get optimal nrounds
  cv_best <- xgb.cv(
    params                = best_params,
    data                  = dtrain,
    nrounds               = max_rounds,
    nfold                 = 3L,
    early_stopping_rounds = 20L,
    verbose               = 0L
  )
  n_rounds <- cv_best$best_iteration

  cat(sprintf("  Best: eta=%.4f, depth=%d, gamma=%.2e, subsample=%.3f, ",
              best_params$eta, best_params$max_depth,
              best_params$gamma, best_params$subsample))
  cat(sprintf("colsample=%.3f, min_child=%.1f, alpha=%.3f, lambda=%.3f\n",
              best_params$colsample_bytree, best_params$min_child_weight,
              best_params$reg_alpha, best_params$reg_lambda))
  cat(sprintf("  Optimal rounds: %d\n", n_rounds))

  # Train final model
  cat(sprintf("  Training final model (%d rounds)...\n", n_rounds))
  model <- xgb.train(
    params        = best_params,
    data          = dtrain,
    nrounds       = n_rounds,
    watchlist     = list(train = dtrain, val = dval),
    verbose       = 1L,
    print_every_n = max(1L, n_rounds %/% 5L)
  )

  # Estimate Gamma shape from training predictions
  mu_train <- predict(model, dtrain)
  shape <- estimate_gamma_shape(y_train, mu_train)
  cat(sprintf("  Estimated Gamma shape: %.4f\n", shape))

  list(model = model, params = best_params, n_rounds = n_rounds, shape = shape)
}


# ==============================================================================
# Model Evaluation
# ==============================================================================

evaluate_model <- function(model, shape, df, feature_cols, split_name) {
  X <- as.matrix(df[, ..feature_cols])
  y <- df$ground_truth_precip
  dmat <- xgb.DMatrix(X, label = y)

  # Predicted mean from xgboost
  mu <- predict(model, dmat)

  # Gamma parameters: shape (common), rate = shape/mu
  rate  <- shape / mu
  scale <- mu / shape

  # Predicted quantiles
  quant_levels <- c(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
  pred_q <- sapply(quant_levels, function(q) qgamma(q, shape = shape, rate = rate))
  colnames(pred_q) <- paste0("q", quant_levels)

  # CRPS (analytical, via scoringRules)
  crps_vals <- crps_gamma(y, shape = shape, rate = rate)
  mean_crps <- mean(crps_vals, na.rm = TRUE)

  # Negative log-likelihood (clipped for stability)
  nll <- -dgamma(y, shape = shape, rate = rate, log = TRUE)
  nll <- pmin(pmax(nll, -100), 100)
  mean_nll <- mean(nll, na.rm = TRUE)

  # Point forecast metrics
  mae  <- mean(abs(mu - y))
  bias <- mean(mu - y)

  # Coverage
  coverage_90 <- mean(y >= pred_q[, "q0.05"] & y <= pred_q[, "q0.95"])
  coverage_50 <- mean(y >= pred_q[, "q0.25"] & y <= pred_q[, "q0.75"])

  # Median absolute error
  median_ae <- median(abs(pred_q[, "q0.5"] - y))

  # Reliability at multiple nominal levels
  reliability <- list()
  for (level in c(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)) {
    lo_q <- (1 - level) / 2
    hi_q <- 1 - lo_q
    lo <- qgamma(lo_q, shape = shape, rate = rate)
    hi <- qgamma(hi_q, shape = shape, rate = rate)
    reliability[[paste0("cov_", as.integer(level * 100))]] <- mean(y >= lo & y <= hi)
  }

  metrics <- c(
    list(
      split          = split_name,
      n_samples      = length(y),
      mean_crps      = mean_crps,
      mean_nll       = mean_nll,
      mae            = mae,
      bias           = bias,
      coverage_90    = coverage_90,
      coverage_50    = coverage_50,
      median_ae      = median_ae,
      mean_observed  = mean(y),
      mean_predicted = mean(mu)
    ),
    reliability
  )

  list(metrics = metrics, mu = mu, rate = rate, scale = scale, pred_q = pred_q)
}


# ==============================================================================
# Diagnostic Plots
# ==============================================================================

plot_diagnostics <- function(model, shape, test_df, feature_cols, horizon) {
  X_test <- as.matrix(test_df[, ..feature_cols])
  y      <- test_df$ground_truth_precip
  dtest  <- xgb.DMatrix(X_test, label = y)

  mu    <- predict(model, dtest)
  rate  <- shape / mu
  scale <- mu / shape

  quant_levels <- c(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
  pred_q <- sapply(quant_levels, function(q) qgamma(q, shape = shape, rate = rate))
  colnames(pred_q) <- paste0("q", quant_levels)

  prefix <- file.path(VIS_DIR, paste0("xgboost_gamma_", horizon, "d"))

  # --- 1. Predicted vs Observed scatter ---
  max_val <- max(c(y, mu), na.rm = TRUE) * 1.1
  p1 <- ggplot(data.frame(obs = y, pred = mu), aes(x = obs, y = pred)) +
    geom_point(alpha = 0.4, size = 1) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    labs(
      x     = "Observed Precipitation (mm)",
      y     = "Predicted Mean (mm)",
      title = sprintf("XGBoost Gamma %dd: Predicted vs Observed", horizon)
    ) +
    coord_fixed(xlim = c(0, max_val), ylim = c(0, max_val)) +
    theme_minimal()
  ggsave(paste0(prefix, "_pred_vs_obs.png"), p1, width = 8, height = 6, dpi = 150)


  # --- 2. Prediction Intervals (first 100 test points) ---
  n_plot <- min(100, length(y))
  pi_df <- data.frame(
    idx = seq_len(n_plot),
    obs = y[1:n_plot],
    q05 = pred_q[1:n_plot, "q0.05"],
    q25 = pred_q[1:n_plot, "q0.25"],
    q50 = pred_q[1:n_plot, "q0.5"],
    q75 = pred_q[1:n_plot, "q0.75"],
    q95 = pred_q[1:n_plot, "q0.95"]
  )
  p2 <- ggplot(pi_df, aes(x = idx)) +
    geom_ribbon(aes(ymin = q05, ymax = q95), fill = "blue", alpha = 0.2) +
    geom_ribbon(aes(ymin = q25, ymax = q75), fill = "blue", alpha = 0.3) +
    geom_line(aes(y = q50), color = "blue") +
    geom_point(aes(y = obs), size = 1, shape = 16) +
    labs(
      x     = "Test Sample Index",
      y     = "Precipitation (mm)",
      title = sprintf("XGBoost Gamma %dd: Prediction Intervals (2025 test)", horizon)
    ) +
    theme_minimal()
  ggsave(paste0(prefix, "_intervals.png"), p2, width = 14, height = 5, dpi = 150)


  # --- 3. Feature Importance (gain-based, top 25) ---
  tryCatch({
    imp <- xgb.importance(model = model)
    imp <- imp[order(-Gain)][1:min(25, nrow(imp))]
    imp[, category := sapply(Feature, classify_feature)]
    imp[, Feature := factor(Feature, levels = rev(Feature))]

    p3 <- ggplot(imp, aes(x = Gain, y = Feature, fill = category)) +
      geom_col() +
      scale_fill_manual(values = CATEGORY_COLORS, name = "") +
      labs(
        x     = "Gain",
        title = sprintf("XGBoost Gamma %dd: Top 25 Features by Gain", horizon)
      ) +
      theme_minimal() +
      theme(legend.position = "bottom")
    ggsave(paste0(prefix, "_importance.png"), p3, width = 10, height = 8, dpi = 150)
  }, error = function(e) {
    cat(sprintf("  Warning: Could not plot feature importance: %s\n", e$message))
  })


  # --- 4. SHAP Analysis ---
  tryCatch({
    cat("    Computing SHAP values...\n")

    shap_result <- shap.values(xgb_model = model, X_train = X_test)
    shap_long   <- shap.prep(
      shap_contrib = shap_result$shap_score,
      X_train      = X_test,
      top_n        = 25L
    )

    # Beeswarm plot
    p4a <- shap.plot.summary(shap_long, dilute = FALSE) +
      ggtitle(sprintf("XGBoost Gamma %dd: SHAP Beeswarm", horizon))
    ggsave(paste0(prefix, "_shap_beeswarm.png"), p4a,
           width = 12, height = 10, dpi = 150)

    # Bar plot (mean |SHAP|)
    mean_shap <- sort(shap_result$mean_shap_score, decreasing = TRUE)
    mean_shap <- mean_shap[1:min(25, length(mean_shap))]
    bar_df <- data.frame(
      feature       = factor(names(mean_shap), levels = rev(names(mean_shap))),
      mean_abs_shap = unname(mean_shap)
    )
    p4b <- ggplot(bar_df, aes(x = mean_abs_shap, y = feature)) +
      geom_col(fill = "#2196F3") +
      labs(
        x     = "Mean |SHAP value|",
        y     = NULL,
        title = sprintf("XGBoost Gamma %dd: Mean |SHAP|", horizon)
      ) +
      theme_minimal()
    ggsave(paste0(prefix, "_shap_bar.png"), p4b,
           width = 12, height = 8, dpi = 150)

    # Dependence plots for top 4 features
    top4 <- names(mean_shap)[1:min(4, length(mean_shap))]
    dep_plots <- lapply(top4, function(feat) {
      shap.plot.dependence(
        data_long     = shap_long,
        x             = feat,
        color_feature = "auto",
        smooth        = FALSE
      ) + ggtitle(feat) + theme_minimal()
    })

    if (has_patchwork && length(dep_plots) == 4) {
      p4c <- (dep_plots[[1]] + dep_plots[[2]]) /
             (dep_plots[[3]] + dep_plots[[4]]) +
        plot_annotation(
          title = sprintf("XGBoost Gamma %dd: SHAP Dependence (Top 4)", horizon)
        )
    } else {
      # Fallback: save individual plots
      for (j in seq_along(dep_plots)) {
        ggsave(sprintf("%s_shap_dep_%s.png", prefix, top4[j]),
               dep_plots[[j]], width = 7, height = 5, dpi = 150)
      }
      p4c <- NULL
    }
    if (!is.null(p4c)) {
      ggsave(paste0(prefix, "_shap_dependence.png"), p4c,
             width = 14, height = 10, dpi = 150)
    }

    cat("    SHAP plots saved.\n")
  }, error = function(e) {
    cat(sprintf("  Warning: Could not compute SHAP values: %s\n", e$message))
    cat(sprintf("           %s\n", conditionMessage(e)))
  })


  # --- 5. PIT Histogram ---
  pit <- pgamma(y, shape = shape, rate = rate)
  p5 <- ggplot(data.frame(pit = pit), aes(x = pit)) +
    geom_histogram(aes(y = after_stat(density)), bins = 20,
                   fill = "steelblue", alpha = 0.7, color = "black") +
    geom_hline(yintercept = 1.0, color = "red", linetype = "dashed") +
    labs(
      x     = "PIT Value",
      y     = "Density",
      title = sprintf("XGBoost Gamma %dd: PIT Histogram (Calibration)", horizon)
    ) +
    theme_minimal()
  ggsave(paste0(prefix, "_pit.png"), p5, width = 6, height = 4, dpi = 150)


  # --- 6. Reliability Diagram ---
  nominal_levels <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
  actual_covs <- sapply(nominal_levels, function(level) {
    lo_q <- (1 - level) / 2
    hi_q <- 1 - lo_q
    lo <- qgamma(lo_q, shape = shape, rate = rate)
    hi <- qgamma(hi_q, shape = shape, rate = rate)
    mean(y >= lo & y <= hi)
  })

  rel_df <- data.frame(nominal = nominal_levels, actual = actual_covs)
  p6 <- ggplot(rel_df, aes(x = nominal, y = actual)) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    geom_point(color = "blue", size = 3) +
    geom_line(color = "blue") +
    labs(
      x     = "Nominal Coverage",
      y     = "Actual Coverage",
      title = sprintf("XGBoost Gamma %dd: Reliability Diagram", horizon)
    ) +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
    theme_minimal()
  ggsave(paste0(prefix, "_reliability.png"), p6, width = 6, height = 6, dpi = 150)
}


# ==============================================================================
# Main
# ==============================================================================

main <- function() {
  # Parse command-line arguments
  args <- commandArgs(trailingOnly = TRUE)
  horizon_arg <- NULL
  quick <- FALSE

  i <- 1L
  while (i <= length(args)) {
    if (args[i] == "--horizon" && i < length(args)) {
      horizon_arg <- as.integer(args[i + 1])
      i <- i + 2L
    } else if (args[i] == "--quick") {
      quick <- TRUE
      i <- i + 1L
    } else {
      i <- i + 1L
    }
  }

  horizons <- if (!is.null(horizon_arg)) horizon_arg else HORIZONS

  if (!file.exists(DATA_FILE)) {
    stop(sprintf(
      "Training dataset not found: %s\nRun: python scripts/data_processing/build_training_dataset.py",
      DATA_FILE
    ))
  }

  all_metrics <- list()

  for (horizon in horizons) {
    cat(sprintf("\n%s\n", strrep("=", 60)))
    cat(sprintf("Training %d-day Horizon Model\n", horizon))
    cat(sprintf("%s\n", strrep("=", 60)))

    # Load and split
    cat("\nLoading data...\n")
    data     <- load_data(horizon)
    train_dt <- data$train
    val_dt   <- data$val
    test_dt  <- data$test

    # Get features, drop high-NaN columns (same logic as Python)
    feature_cols <- get_feature_cols(train_dt)
    nan_pcts <- sapply(feature_cols, function(col) mean(is.na(train_dt[[col]])))
    drop_cols <- names(nan_pcts[nan_pcts > 0.5])
    if (length(drop_cols) > 0) {
      cat(sprintf("  Dropping %d high-NaN columns: %s\n",
                  length(drop_cols), paste(drop_cols, collapse = ", ")))
      feature_cols <- setdiff(feature_cols, drop_cols)
    }

    # Fill remaining NAs with 0
    for (col in feature_cols) {
      set(train_dt, which(is.na(train_dt[[col]])), col, 0)
      set(val_dt,   which(is.na(val_dt[[col]])),   col, 0)
      set(test_dt,  which(is.na(test_dt[[col]])),  col, 0)
    }

    cat(sprintf("  Features: %d\n", length(feature_cols)))

    # Train
    cat("\nTraining...\n")
    result <- train_model(train_dt, val_dt, feature_cols, horizon, quick)
    model  <- result$model
    shape  <- result$shape

    # Evaluate on val and test
    cat("\nEvaluating...\n")
    for (split_info in list(
      list(name = "val",  dt = val_dt),
      list(name = "test", dt = test_dt)
    )) {
      eval_result <- evaluate_model(
        model, shape, split_info$dt, feature_cols, split_info$name
      )
      m <- eval_result$metrics
      m$horizon_days <- horizon
      m$gamma_shape  <- shape
      all_metrics[[length(all_metrics) + 1]] <- m

      cat(sprintf("\n  %s metrics:\n", toupper(split_info$name)))
      cat(sprintf("    CRPS:        %.2f\n", m$mean_crps))
      cat(sprintf("    NLL:         %.2f\n", m$mean_nll))
      cat(sprintf("    MAE:         %.1f mm\n", m$mae))
      cat(sprintf("    Bias:        %+.1f mm\n", m$bias))
      cat(sprintf("    Coverage 90: %.1f%%\n", m$coverage_90 * 100))
      cat(sprintf("    Coverage 50: %.1f%%\n", m$coverage_50 * 100))
      cat(sprintf("    Mean obs:    %.1f mm\n", m$mean_observed))
      cat(sprintf("    Mean pred:   %.1f mm\n", m$mean_predicted))
      cat("    Reliability: ")
      for (lev in c(50, 60, 70, 80, 90, 95)) {
        key <- paste0("cov_", lev)
        if (!is.null(m[[key]])) cat(sprintf("%d%%=%.1f%% ", lev, m[[key]] * 100))
      }
      cat("\n")

      # Diagnostic plots for test set
      if (split_info$name == "test") {
        cat("\n  Generating diagnostic plots...\n")
        plot_diagnostics(model, shape, split_info$dt, feature_cols, horizon)
      }
    }

    # Save model
    model_path <- file.path(MODELS_DIR, sprintf("xgboost_gamma_%dd.json", horizon))
    xgb.save(model, model_path)
    cat(sprintf("\n  Model saved to %s\n", model_path))

    # Save params + shape
    params_to_save <- result$params
    params_to_save$gamma_shape <- shape
    params_to_save$n_rounds    <- result$n_rounds
    params_path <- file.path(MODELS_DIR,
                             sprintf("xgboost_gamma_%dd_params.json", horizon))
    writeLines(toJSON(params_to_save, auto_unbox = TRUE, pretty = TRUE),
               params_path)
  }

  # Save all metrics
  if (length(all_metrics) > 0) {
    metrics_df <- rbindlist(all_metrics, fill = TRUE)
    metrics_path <- file.path(MODELS_DIR, "all_metrics_r.csv")
    fwrite(metrics_df, metrics_path)
    cat(sprintf("\n%s\n", strrep("=", 60)))
    cat(sprintf("All metrics saved to %s\n", metrics_path))
    cat(sprintf("%s\n", strrep("=", 60)))
    print(metrics_df)
  }

  cat("\nDone!\n")
}


# Run
main()
