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
# Custom Theme (matching GAMLSS pipeline)
# ==============================================================================

theme_precip <- function(base_size = 12) {
  theme_minimal(base_size = base_size) %+replace%
    theme(
      plot.title       = element_text(face = "bold", size = base_size + 2,
                                       hjust = 0, margin = margin(b = 8)),
      plot.subtitle    = element_text(color = "grey40", size = base_size,
                                       hjust = 0, margin = margin(b = 12)),
      axis.title       = element_text(size = base_size),
      axis.text        = element_text(size = base_size - 1, color = "grey30"),
      legend.position  = "bottom",
      legend.title     = element_text(face = "bold", size = base_size - 1),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey92", linewidth = 0.4),
      plot.background  = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      strip.text       = element_text(face = "bold", size = base_size)
    )
}

COLORS <- list(
  primary    = "#1B6CA8",
  secondary  = "#E8751A",
  accent     = "#2CA02C",
  light_blue = "#A8D4F0",
  red        = "#D62728",
  purple     = "#9467BD",
  forecast   = "#2196F3",
  climate    = "#FF9800",
  recent     = "#4CAF50",
  calendar   = "#9C27B0",
  bias       = "#FF5722"
)


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

  colnames(X_train) <- feature_cols
  colnames(X_val)   <- feature_cols
  dtrain <- xgb.DMatrix(X_train, label = y_train)
  dval   <- xgb.DMatrix(X_val,   label = y_val)

  n_trials   <- if (quick) 15L else 30L
  max_rounds <- if (quick) 100L else 500L
  # ParBayesianOptimization requires initPoints > number of bounds (8 here).
  n_init     <- 10L
  n_iter     <- max(1L, n_trials - n_init)

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

#' Find a scale factor s that widens the Gamma predictive distribution so the
#' validation PIT is as close to Uniform(0,1) as possible.
#'
#' Mean-preserving variance scaling: shape_eff = shape/s^2, rate_eff = rate/s^2.
#' Keeps mean = shape/rate unchanged while multiplying variance by s^2.
#'
#' Objective options:
#'   - "ks"   : minimise KS distance between PIT and Uniform (flattens whole PIT)
#'   - "cov"  : minimise |coverage_at_target - target| (only ties one level)
find_optimal_scale_factor <- function(y, shape, rate,
                                      objective = c("ks", "cov"),
                                      target_coverage = 0.90,
                                      search_range = c(0.3, 15.0),
                                      step = 0.05) {
  objective <- match.arg(objective)
  factors <- seq(search_range[1], search_range[2], by = step)

  best_sf  <- 1.0
  best_val <- Inf
  for (sf in factors) {
    shape_adj <- shape / (sf^2)
    rate_adj  <- rate  / (sf^2)
    if (objective == "ks") {
      pit <- pgamma(y, shape = shape_adj, rate = rate_adj)
      pit <- pmin(pmax(pit, 1e-9), 1 - 1e-9)
      val <- suppressWarnings(ks.test(pit, "punif"))$statistic
    } else {
      lo_q <- (1 - target_coverage) / 2
      hi_q <- 1 - lo_q
      lo <- qgamma(lo_q, shape = shape_adj, rate = rate_adj)
      hi <- qgamma(hi_q, shape = shape_adj, rate = rate_adj)
      cov <- mean(y >= lo & y <= hi)
      val <- abs(cov - target_coverage)
    }
    if (val < best_val) {
      best_val <- val
      best_sf  <- sf
    }
  }
  best_sf
}


evaluate_model <- function(model, shape, df, feature_cols, split_name,
                           scale_factor = 1.0) {
  X <- as.matrix(df[, ..feature_cols])
  colnames(X) <- feature_cols
  y <- df$ground_truth_precip
  dmat <- xgb.DMatrix(X, label = y)

  # Predicted mean from xgboost
  mu <- predict(model, dmat)

  # Gamma parameters: shape (common), rate = shape/mu
  # Mean-preserving variance calibration: shape_eff = shape/sf^2, rate_eff = rate/sf^2
  # Keeps mean = shape/rate = mu, inflates variance by sf^2.
  rate_raw <- shape / mu
  shape_eff <- shape    / (scale_factor^2)
  rate      <- rate_raw / (scale_factor^2)
  shape     <- shape_eff
  scale     <- 1.0 / rate

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

  # Point forecast metrics — use raw mu (calibration doesn't change the point)
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

#' Given a validation PIT vector, return a function that maps a desired
#' nominal level τ to the underlying Gamma-CDF level τ_raw = Q_pit(τ).
#' Querying the Gamma at τ_raw then yields a calibrated τ-quantile on test.
build_quantile_recalibrator <- function(pit_val) {
  # Empirical inverse-CDF (quantile function) of val PIT, with endpoints pinned
  # so τ in (0,1) always maps to a value in (0,1).
  u_sorted <- sort(pit_val)
  u_sorted <- pmin(pmax(u_sorted, 1e-6), 1 - 1e-6)
  n <- length(u_sorted)
  p_grid <- (seq_len(n) - 0.5) / n  # plotting positions
  approxfun(
    x = c(0, p_grid, 1),
    y = c(0, u_sorted, 1),
    method = "linear", rule = 2, ties = "ordered"
  )
}


plot_diagnostics <- function(model, shape, test_df, feature_cols, horizon,
                             scale_factor = 1.0, val_df = NULL) {
  X_test <- as.matrix(test_df[, ..feature_cols])
  colnames(X_test) <- feature_cols
  y      <- test_df$ground_truth_precip
  dtest  <- xgb.DMatrix(X_test, label = y)

  mu    <- predict(model, dtest)
  # Apply mean-preserving variance calibration
  shape <- shape / (scale_factor^2)
  rate  <- (shape / mu)
  scale <- 1.0 / rate

  # --- Quantile recalibration (learned on validation PIT) ---
  recal <- NULL
  pit_val <- NULL
  if (!is.null(val_df)) {
    X_val_m <- as.matrix(val_df[, ..feature_cols])
    colnames(X_val_m) <- feature_cols
    mu_val  <- predict(model, xgb.DMatrix(X_val_m,
                                          label = val_df$ground_truth_precip))
    rate_val <- shape / mu_val
    pit_val  <- pgamma(val_df$ground_truth_precip, shape = shape, rate = rate_val)
    pit_val  <- pmin(pmax(pit_val, 1e-6), 1 - 1e-6)
    recal    <- build_quantile_recalibrator(pit_val)
  }

  quant_levels <- c(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
  if (!is.null(recal)) {
    # Map each nominal level to its empirical counterpart, then query Gamma
    raw_levels <- recal(quant_levels)
    pred_q <- sapply(seq_along(quant_levels), function(i) {
      qgamma(raw_levels[i], shape = shape, rate = rate)
    })
  } else {
    pred_q <- sapply(quant_levels, function(q) qgamma(q, shape = shape, rate = rate))
  }
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
    theme_precip()
  ggsave(paste0(prefix, "_pred_vs_obs.png"), p1, width = 8, height = 6, dpi = 200)


  # --- 2. Prediction Intervals (all test points, date x-axis, legend) ---
  has_dates <- "init_date" %in% names(test_df)
  n_plot    <- length(y)  # show every test sample
  if (has_dates) {
    x_vals <- as.Date(test_df$init_date[1:n_plot])
    x_lab  <- "Forecast Init Date"
  } else {
    x_vals <- seq_len(n_plot)
    x_lab  <- "Test Sample Index"
  }
  pi_df <- data.frame(
    x   = x_vals,
    obs = y[1:n_plot],
    q05 = pred_q[1:n_plot, "q0.05"],
    q25 = pred_q[1:n_plot, "q0.25"],
    q50 = pred_q[1:n_plot, "q0.5"],
    q75 = pred_q[1:n_plot, "q0.75"],
    q95 = pred_q[1:n_plot, "q0.95"]
  )
  # Sort by x so lines connect correctly
  pi_df <- pi_df[order(pi_df$x), ]

  # Map fills/colors through aes() so legends auto-build
  band_colors <- c("90% PI" = "#4C78A8", "50% PI" = "#1F4E79")
  line_colors <- c("Median forecast" = "#0B2948", "Observed" = "black")

  p2 <- ggplot(pi_df, aes(x = x)) +
    geom_ribbon(aes(ymin = q05, ymax = q95, fill = "90% PI"), alpha = 0.25) +
    geom_ribbon(aes(ymin = q25, ymax = q75, fill = "50% PI"), alpha = 0.45) +
    geom_line(aes(y = q50, color = "Median forecast"), linewidth = 0.7) +
    geom_point(aes(y = obs, color = "Observed"), size = 1.6, shape = 16, alpha = 0.85) +
    scale_fill_manual(name = "Prediction interval", values = band_colors,
                      breaks = c("90% PI", "50% PI")) +
    scale_color_manual(name = NULL, values = line_colors,
                       breaks = c("Median forecast", "Observed")) +
    labs(
      x     = x_lab,
      y     = "Cumulative precipitation (mm)",
      title = sprintf("XGBoost Gamma %dd — Prediction Intervals (2025 test set)", horizon),
      subtitle = sprintf("n = %d forecasts | 50%% and 90%% central credible bands", n_plot)
    ) +
    theme_precip() +
    theme(
      legend.position = "bottom",
      legend.box = "horizontal",
      plot.title = element_text(face = "bold"),
      plot.subtitle = element_text(color = "grey40"),
      panel.grid.minor = element_blank()
    )
  if (has_dates) {
    p2 <- p2 + scale_x_date(date_labels = "%b %Y", date_breaks = "1 month")
  }
  ggsave(paste0(prefix, "_intervals.png"), p2, width = 16, height = 6, dpi = 200)


  # --- 3. Feature Importance (gain-based, top 25) ---
  tryCatch({
    imp <- xgb.importance(feature_names = feature_cols, model = model)
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
      theme_precip() +
      theme(legend.position = "bottom")
    ggsave(paste0(prefix, "_importance.png"), p3, width = 10, height = 8, dpi = 200)
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
           width = 12, height = 10, dpi = 200)

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
      theme_precip()
    ggsave(paste0(prefix, "_shap_bar.png"), p4b,
           width = 12, height = 8, dpi = 200)

    # Dependence plots for top 4 features
    top4 <- names(mean_shap)[1:min(4, length(mean_shap))]
    dep_plots <- lapply(top4, function(feat) {
      shap.plot.dependence(
        data_long     = shap_long,
        x             = feat,
        color_feature = "auto",
        smooth        = FALSE
      ) + ggtitle(feat) + theme_precip()
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
               dep_plots[[j]], width = 7, height = 5, dpi = 200)
      }
      p4c <- NULL
    }
    if (!is.null(p4c)) {
      ggsave(paste0(prefix, "_shap_dependence.png"), p4c,
             width = 14, height = 10, dpi = 200)
    }

    cat("    SHAP plots saved.\n")
  }, error = function(e) {
    cat(sprintf("  Warning: Could not compute SHAP values: %s\n", e$message))
    cat(sprintf("           %s\n", conditionMessage(e)))
  })


  # --- 5. PIT Histogram with uniformity band + KS test ---
  pit_raw <- pgamma(y, shape = shape, rate = rate)
  pit_raw <- pmin(pmax(pit_raw, 1e-6), 1 - 1e-6)
  # If we have a recalibrator from val, apply its inverse to test PIT to
  # produce the calibrated PIT (should look uniform if recalibration works).
  pit_cal <- if (!is.null(recal)) {
    # Recalibrated PIT = empirical CDF of val PIT applied to test raw PIT.
    # Build ecdf from val PIT and evaluate at test raw PIT.
    ecdf_val <- ecdf(pit_val)
    ecdf_val(pit_raw)
  } else {
    NULL
  }

  pit_long <- if (!is.null(pit_cal)) {
    data.frame(
      pit   = c(pit_raw, pit_cal),
      which = factor(rep(c("Variance-scaled only", "Quantile-recalibrated"),
                         each = length(pit_raw)),
                     levels = c("Variance-scaled only", "Quantile-recalibrated"))
    )
  } else {
    data.frame(pit = pit_raw, which = factor("Variance-scaled only"))
  }

  # 95% pointwise binomial band for uniformity at the chosen bin count
  n_bins <- 25
  bin_width <- 1 / n_bins
  n_obs <- length(pit_raw)
  expected_count <- n_obs * bin_width
  # Expected density = 1.0; binomial 95% band on density
  band_lo_density <- qbinom(0.025, n_obs, bin_width) / (n_obs * bin_width)
  band_hi_density <- qbinom(0.975, n_obs, bin_width) / (n_obs * bin_width)

  # KS stat vs Uniform(0,1)
  ks_raw <- suppressWarnings(ks.test(pit_raw, "punif"))
  ks_lab_raw <- sprintf("KS=%.3f (p=%.3f)", ks_raw$statistic, ks_raw$p.value)
  ks_lab_cal <- if (!is.null(pit_cal)) {
    ks_c <- suppressWarnings(ks.test(pit_cal, "punif"))
    sprintf("KS=%.3f (p=%.3f)", ks_c$statistic, ks_c$p.value)
  } else NULL

  annot_df <- if (!is.null(pit_cal)) {
    data.frame(
      which = factor(c("Variance-scaled only", "Quantile-recalibrated"),
                     levels = c("Variance-scaled only", "Quantile-recalibrated")),
      label = c(ks_lab_raw, ks_lab_cal)
    )
  } else {
    data.frame(which = factor("Variance-scaled only"), label = ks_lab_raw)
  }

  p5 <- ggplot(pit_long, aes(x = pit)) +
    annotate("rect", xmin = 0, xmax = 1,
             ymin = band_lo_density, ymax = band_hi_density,
             alpha = 0.15, fill = "grey40") +
    geom_histogram(aes(y = after_stat(density)), bins = n_bins,
                   fill = "#4C78A8", alpha = 0.85, color = "white") +
    geom_hline(yintercept = 1.0, color = "#C1272D",
               linetype = "dashed", linewidth = 0.6) +
    geom_text(data = annot_df, aes(x = 0.02, y = Inf, label = label),
              hjust = 0, vjust = 1.5, size = 3.3, color = "grey20") +
    facet_wrap(~which, nrow = 1) +
    labs(
      x     = "PIT value",
      y     = "Density",
      title = sprintf("XGBoost Gamma %dd — PIT Histogram (2025 test set)", horizon),
      subtitle = paste0(
        "Grey band = 95% pointwise uniform envelope | ",
        "dashed line = ideal density (1.0)"
      )
    ) +
    theme_precip() +
    theme(
      plot.title = element_text(face = "bold"),
      plot.subtitle = element_text(color = "grey40"),
      strip.text = element_text(face = "bold"),
      panel.grid.minor = element_blank()
    )

  pit_width <- if (!is.null(pit_cal)) 11 else 6
  ggsave(paste0(prefix, "_pit.png"), p5, width = pit_width, height = 4.5, dpi = 200)


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
    theme_precip()
  ggsave(paste0(prefix, "_reliability.png"), p6, width = 6, height = 6, dpi = 200)
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

    # Calibrate variance on validation set (mean-preserving scaling of Gamma)
    cat("\nCalibrating (variance scaling on validation set)...\n")
    X_val_m <- as.matrix(val_dt[, ..feature_cols])
    colnames(X_val_m) <- feature_cols
    mu_val  <- predict(model, xgb.DMatrix(X_val_m,
                                          label = val_dt$ground_truth_precip))
    rate_val <- shape / mu_val
    scale_factor <- find_optimal_scale_factor(
      y = val_dt$ground_truth_precip, shape = shape, rate = rate_val,
      objective = "ks"  # minimise PIT KS distance from Uniform
    )
    cat(sprintf("  Optimal scale factor (KS-minimising): %.2f\n", scale_factor))

    # Evaluate on val and test
    cat("\nEvaluating...\n")
    for (split_info in list(
      list(name = "val",  dt = val_dt),
      list(name = "test", dt = test_dt)
    )) {
      eval_result <- evaluate_model(
        model, shape, split_info$dt, feature_cols, split_info$name,
        scale_factor = scale_factor
      )
      m <- eval_result$metrics
      m$horizon_days <- horizon
      m$gamma_shape  <- shape
      m$scale_factor <- scale_factor
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
        plot_diagnostics(model, shape, split_info$dt, feature_cols, horizon,
                         scale_factor = scale_factor, val_df = val_dt)
      }
    }

    # Save model
    model_path <- file.path(MODELS_DIR, sprintf("xgboost_gamma_%dd.json", horizon))
    xgb.save(model, model_path)
    cat(sprintf("\n  Model saved to %s\n", model_path))

    # Save params + shape
    params_to_save <- result$params
    params_to_save$gamma_shape  <- shape
    params_to_save$n_rounds     <- result$n_rounds
    params_to_save$scale_factor <- scale_factor
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


# Run — only when invoked as a script (not when sourced for reuse)
if (!exists("SKIP_MAIN") || !isTRUE(SKIP_MAIN)) {
  main()
}
