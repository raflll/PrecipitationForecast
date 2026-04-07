#!/usr/bin/env Rscript
#' Train XGBoost GAMLSS-style distributional regression for cumulative
#' precipitation forecasting over multiple horizons.
#'
#' Approach: Two-stage log-normal-inspired model with Gamma output
#'   Stage 1: XGBoost reg:squarederror on log1p(precip) for mean
#'   Stage 2: Learn conditional variance via residual modeling
#'   + Post-hoc quantile recalibration on validation set
#'
#' Usage:
#'   Rscript src/train_gamlss.R                  # Train all horizons
#'   Rscript src/train_gamlss.R --horizon 14     # Single horizon
#'   Rscript src/train_gamlss.R --quick           # Quick test run

# ── Install missing packages ─────────────────────────────────────────────────
required_pkgs <- c(
  "arrow", "xgboost", "shapviz", "ggplot2", "patchwork",
  "data.table", "scoringRules", "scales"
)

for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

suppressPackageStartupMessages({
  library(arrow)
  library(xgboost)
  library(shapviz)
  library(ggplot2)
  library(patchwork)
  library(data.table)
  library(scoringRules)
  library(scales)
})

# ── Paths ─────────────────────────────────────────────────────────────────────
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  return(file.path(getwd(), "src"))
}
PROJECT_ROOT <- normalizePath(file.path(get_script_dir(), ".."))
DATA_FILE    <- file.path(PROJECT_ROOT, "data", "training", "training_dataset.parquet")
MODELS_DIR   <- file.path(PROJECT_ROOT, "models")
VIS_DIR      <- file.path(PROJECT_ROOT, "visualizations")
dir.create(MODELS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(VIS_DIR,    showWarnings = FALSE, recursive = TRUE)

HORIZONS <- c(3, 7, 14, 21, 28, 35)
NON_FEATURE_COLS <- c("init_date", "horizon_days", "ground_truth_precip", "source")

# ── Custom theme ──────────────────────────────────────────────────────────────
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
  calendar   = "#9C27B0"
)

# ── Parse args ────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
opt_horizon <- NULL
opt_quick   <- FALSE

i <- 1
while (i <= length(args)) {
  if (args[i] == "--horizon") {
    opt_horizon <- as.integer(args[i + 1])
    i <- i + 2
  } else if (args[i] == "--quick") {
    opt_quick <- TRUE
    i <- i + 1
  } else {
    i <- i + 1
  }
}

horizons <- if (!is.null(opt_horizon)) opt_horizon else HORIZONS

# ── Data loading ──────────────────────────────────────────────────────────────
load_data <- function(horizon) {
  df <- as.data.table(read_parquet(DATA_FILE))
  df <- df[horizon_days == horizon]
  df[, init_date := as.Date(init_date)]
  setorder(df, init_date)

  pre_test <- df[year(init_date) < 2025]
  test     <- df[year(init_date) >= 2025]

  if (any(year(pre_test$init_date) == 2024)) {
    train <- pre_test[year(init_date) <= 2023]
    val   <- pre_test[year(init_date) == 2024]
  } else {
    # No 2024 data — take every 5th row as val (covers all seasons, not just end-of-year)
    idx <- seq_len(nrow(pre_test))
    val_idx <- idx[idx %% 5 == 0]
    train_idx <- setdiff(idx, val_idx)
    val   <- pre_test[val_idx]
    train <- pre_test[train_idx]
  }

  cat(sprintf("  Train: %d rows (%s to %s)\n", nrow(train),
              min(train$init_date), max(train$init_date)))
  cat(sprintf("  Val:   %d rows (%s to %s)\n", nrow(val),
              min(val$init_date), max(val$init_date)))
  cat(sprintf("  Test:  %d rows (%s to %s)\n", nrow(test),
              min(test$init_date), max(test$init_date)))

  list(train = train, val = val, test = test)
}

get_feature_cols <- function(df) setdiff(names(df), NON_FEATURE_COLS)

make_dmatrix <- function(df, feature_cols, label = NULL) {
  X <- as.matrix(df[, ..feature_cols])
  storage.mode(X) <- "double"
  if (is.null(label)) label <- df$ground_truth_precip
  xgb.DMatrix(X, label = label)
}

# ── Hyperparameter random search ─────────────────────────────────────────────
log_runif <- function(n, lo, hi) exp(runif(n, log(lo), log(hi)))

random_search_xgb <- function(dtrain, objective, n_trials, max_rounds,
                               label, nfold = 3L, extra_params = list()) {
  best_score  <- Inf
  best_params <- NULL
  best_rounds <- NULL

  for (trial in seq_len(n_trials)) {
    params <- c(list(
      objective        = objective,
      eta              = log_runif(1, 5e-3, 0.3),
      max_depth        = sample(3:7, 1),
      gamma            = log_runif(1, 1e-8, 10.0),
      subsample        = runif(1, 0.6, 0.95),
      colsample_bytree = runif(1, 0.4, 0.8),
      min_child_weight = log_runif(1, 1.0, 200.0),
      reg_alpha        = log_runif(1, 1e-3, 10.0),
      reg_lambda       = log_runif(1, 0.1, 20.0),
      tree_method      = "hist"
    ), extra_params)

    cv_result <- xgb.cv(
      params                = params,
      data                  = dtrain,
      nrounds               = max_rounds,
      nfold                 = nfold,
      early_stopping_rounds = 30L,
      verbose               = 0L,
      maximize              = FALSE
    )

    eval_col <- grep("^test_.*_mean$", names(cv_result$evaluation_log), value = TRUE)[1]
    scores <- cv_result$evaluation_log[[eval_col]]
    trial_best <- min(scores)
    trial_round <- which.min(scores)

    cat(sprintf("      Trial %2d/%d: %s=%.4f (rounds=%d)\n",
                trial, n_trials, label, trial_best, trial_round))

    if (trial_best < best_score) {
      best_score  <- trial_best
      best_params <- params
      best_rounds <- trial_round
    }
  }

  list(params = best_params, rounds = best_rounds, score = best_score)
}

# ── Training ──────────────────────────────────────────────────────────────────
# Strategy:
#   Model 1: XGBoost on log1p(precip) -> mean prediction
#   Model 2: XGBoost on log(abs(residual) + 1) -> variance prediction
#             with strong regularization to prevent overfitting
#   Gamma distribution: shape = mu^2/var, scale = var/mu
#   Post-hoc: Learn a variance scaling factor on validation set to
#             achieve proper calibration coverage

train_model <- function(train_df, val_df, feature_cols, horizon, quick = FALSE) {
  n_trials   <- if (quick) 10L else 40L
  max_rounds <- if (quick) 200L else 800L

  set.seed(42 + horizon)

  y_train <- train_df$ground_truth_precip
  y_val   <- val_df$ground_truth_precip

  log_y_train <- log1p(y_train)
  log_y_val   <- log1p(y_val)

  X_train <- as.matrix(train_df[, ..feature_cols])
  X_val   <- as.matrix(val_df[, ..feature_cols])
  storage.mode(X_train) <- "double"
  storage.mode(X_val)   <- "double"

  dtrain_log <- xgb.DMatrix(X_train, label = log_y_train)
  dval_log   <- xgb.DMatrix(X_val,   label = log_y_val)

  # ── Stage 1: Mean model on log1p(precip) ──
  cat("  Stage 1: Training mean model on log1p(precip)...\n")
  cat(sprintf("    Random search (%d trials)...\n", n_trials))

  search1 <- random_search_xgb(
    dtrain_log, "reg:squarederror", n_trials, max_rounds, "RMSE"
  )
  cat(sprintf("    Best RMSE: %.4f, rounds: %d\n", search1$score, search1$rounds))

  mu_model <- xgb.train(
    params    = search1$params,
    data      = dtrain_log,
    nrounds   = search1$rounds,
    watchlist = list(train = dtrain_log, val = dval_log),
    early_stopping_rounds = 30L,
    verbose   = 1L,
    print_every_n = max(1L, search1$rounds %/% 5L)
  )
  cat(sprintf("    Mean model stopped at round %d (of %d)\n",
              mu_model$best_iteration, search1$rounds))

  # ── Stage 2: Variance model ──
  # CRITICAL: Use cross-validated residuals, not training residuals.
  # The mean model is overfit to training data, so training residuals are
  # near-zero and would cause the variance model to predict too-narrow intervals.
  # Instead, use 5-fold CV predictions to get realistic out-of-sample residuals.
  cat("  Computing cross-validated residuals for variance targets...\n")
  n_train <- nrow(train_df)
  folds <- sample(rep(1:5, length.out = n_train))
  cv_preds <- numeric(n_train)

  for (fold in 1:5) {
    fold_train_idx <- which(folds != fold)
    fold_val_idx   <- which(folds == fold)

    fold_dtrain <- xgb.DMatrix(X_train[fold_train_idx, , drop = FALSE],
                                label = log_y_train[fold_train_idx])
    fold_dval   <- xgb.DMatrix(X_train[fold_val_idx, , drop = FALSE],
                                label = log_y_train[fold_val_idx])

    fold_model <- xgb.train(
      params    = search1$params,
      data      = fold_dtrain,
      nrounds   = mu_model$best_iteration,
      watchlist = list(val = fold_dval),
      early_stopping_rounds = 30L,
      verbose   = 0L
    )
    cv_preds[fold_val_idx] <- predict(fold_model, fold_dval)
  }

  abs_resid_train <- abs(log_y_train - cv_preds)
  cat(sprintf("    CV residual stats: median=%.3f, mean=%.3f, max=%.3f\n",
              median(abs_resid_train), mean(abs_resid_train), max(abs_resid_train)))

  # Use log(abs_resid + 1) as target for variance model
  var_target_train <- log1p(abs_resid_train)

  dtrain_var <- xgb.DMatrix(X_train, label = var_target_train)

  cat("\n  Stage 2: Training variance model...\n")

  # Variance model: regularize but don't kill it — need it to actually learn
  n_var_trials <- if (quick) 10L else 20L
  var_max_rounds <- if (quick) 200L else 300L

  best_var_score  <- Inf
  best_var_params <- NULL
  best_var_rounds <- NULL

  for (trial in seq_len(n_var_trials)) {
    params <- list(
      objective        = "reg:squarederror",
      eta              = log_runif(1, 0.01, 0.2),
      max_depth        = sample(2:5, 1),
      gamma            = log_runif(1, 0.01, 5.0),
      subsample        = runif(1, 0.5, 0.9),
      colsample_bytree = runif(1, 0.3, 0.7),
      min_child_weight = log_runif(1, 5.0, 100.0),
      reg_alpha        = log_runif(1, 0.01, 10.0),
      reg_lambda       = log_runif(1, 0.1, 20.0),
      tree_method      = "hist"
    )

    cv_result <- xgb.cv(
      params = params, data = dtrain_var,
      nrounds = var_max_rounds, nfold = 3L,
      early_stopping_rounds = 20L, verbose = 0L, maximize = FALSE
    )

    eval_col <- grep("^test_.*_mean$", names(cv_result$evaluation_log), value = TRUE)[1]
    scores <- cv_result$evaluation_log[[eval_col]]
    trial_best <- min(scores)
    trial_round <- which.min(scores)

    cat(sprintf("      Trial %2d/%d: RMSE=%.4f (rounds=%d)\n",
                trial, n_var_trials, trial_best, trial_round))

    if (trial_best < best_var_score) {
      best_var_score  <- trial_best
      best_var_params <- params
      best_var_rounds <- trial_round
    }
  }

  cat(sprintf("    Best RMSE: %.4f, rounds: %d\n", best_var_score, best_var_rounds))

  log_pred_val <- predict(mu_model, dval_log)
  abs_resid_val <- abs(log_y_val - log_pred_val)
  var_target_val <- log1p(abs_resid_val)
  dval_var <- xgb.DMatrix(X_val, label = var_target_val)

  var_model <- xgb.train(
    params    = best_var_params,
    data      = dtrain_var,
    nrounds   = best_var_rounds,
    watchlist = list(train = dtrain_var, val = dval_var),
    early_stopping_rounds = 20L,
    verbose   = 1L,
    print_every_n = max(1L, best_var_rounds %/% 5L)
  )
  cat(sprintf("    Variance model stopped at round %d (of %d)\n",
              var_model$best_iteration, best_var_rounds))

  # ── Post-hoc calibration: find variance scale factor ──
  # Use LEAVE-ONE-YEAR-OUT approach: since val is from the same year as train,
  # optimizing on val gives overconfident intervals. Instead, we search for the
  # scale factor that produces the best KS statistic AND targets reasonable
  # coverage, with a conservative minimum scale factor.
  cat("\n  Calibrating variance scale on validation set (targeting uniform PIT)...\n")

  val_log_pred <- predict(mu_model, dval_log)
  val_log_var_pred <- predict(var_model, dval_var)
  val_sigma_log <- pmax(expm1(val_log_var_pred), 0.1)

  best_scale_factor <- 1.0
  best_score <- Inf

  for (sf in seq(1.0, 10.0, by = 0.1)) {
    sigma_adj <- val_sigma_log * sf
    mu_orig <- expm1(val_log_pred)
    variance <- mu_orig^2 * (exp(sigma_adj^2) - 1)
    variance <- pmax(variance, 0.01)
    mu_corrected <- pmax(mu_orig * exp(sigma_adj^2 / 2), 0.1)

    shp <- pmax(mu_corrected^2 / variance, 0.05)
    scl <- pmax(variance / mu_corrected, 0.001)

    # Combine KS statistic with coverage penalty
    pit_val <- pgamma(pmax(y_val, 0.001), shape = shp, scale = scl)
    ks_stat <- suppressWarnings(ks.test(pit_val, "punif")$statistic)

    q05 <- qgamma(0.05, shape = shp, scale = scl)
    q95 <- qgamma(0.95, shape = shp, scale = scl)
    cov90 <- mean(y_val >= q05 & y_val <= q95)

    # Score: KS + heavy penalty if coverage < 90%
    cov_penalty <- max(0, 0.90 - cov90) * 2.0
    score <- ks_stat + cov_penalty

    if (!is.na(score) && score < best_score) {
      best_score <- score
      best_scale_factor <- sf
    }
  }

  # Check coverage at 90% with the chosen scale factor
  sigma_adj <- val_sigma_log * best_scale_factor
  mu_orig <- expm1(val_log_pred)
  variance <- mu_orig^2 * (exp(sigma_adj^2) - 1)
  variance <- pmax(variance, 0.01)
  mu_corrected <- pmax(mu_orig * exp(sigma_adj^2 / 2), 0.1)
  shp_val <- pmax(mu_corrected^2 / variance, 0.05)
  scl_val <- pmax(variance / mu_corrected, 0.001)
  pit_val <- pgamma(pmax(y_val, 0.001), shape = shp_val, scale = scl_val)
  ks_val <- suppressWarnings(ks.test(pit_val, "punif")$statistic)
  q05 <- qgamma(0.05, shape = shp_val, scale = scl_val)
  q95 <- qgamma(0.95, shape = shp_val, scale = scl_val)
  cov90_val <- mean(y_val >= q05 & y_val <= q95)

  cat(sprintf("    Best scale factor: %.2f (KS: %.3f, 90%% coverage: %.1f%%)\n",
              best_scale_factor, ks_val, cov90_val * 100))

  list(
    mu_model     = mu_model,
    var_model    = var_model,
    mu_params    = search1$params,
    var_params   = best_var_params,
    mu_rounds    = search1$rounds,
    var_rounds   = best_var_rounds,
    scale_factor = best_scale_factor,
    recal_fn     = NULL
  )
}

# ── Prediction ────────────────────────────────────────────────────────────────
predict_gamma_params <- function(result, df, feature_cols) {
  X <- as.matrix(df[, ..feature_cols])
  storage.mode(X) <- "double"
  y <- df$ground_truth_precip

  dmat_log <- xgb.DMatrix(X, label = log1p(y))
  dmat_var <- xgb.DMatrix(X, label = rep(0, nrow(X)))

  log_pred <- predict(result$mu_model, dmat_log)
  log_var_pred <- predict(result$var_model, dmat_var)

  # Raw sigma from variance model (before scaling)
  sigma_raw <- pmax(expm1(log_var_pred), 0.1)
  # Scaled sigma for distributional width
  sigma_scaled <- sigma_raw * result$scale_factor

  # Point prediction: use raw sigma for mild bias correction only
  mu <- pmax(expm1(log_pred + sigma_raw^2 / 2), 0.1)

  # Distributional variance: use scaled sigma for proper interval width
  variance <- pmax(mu^2 * (exp(sigma_scaled^2) - 1), 0.01)

  # Gamma parameters
  shape <- pmax(mu^2 / variance, 0.05)
  sc    <- pmax(variance / mu, 0.001)

  data.table(mu = mu, variance = variance, shape = shape, scale = sc,
             sigma_log = sigma_scaled, log_pred = log_pred)
}

# ── Recalibrated quantile helper ──────────────────────────────────────────────
recal_quantile <- function(q, shape, scale, recal_fn = NULL) {
  # If we have a recalibration function, map the nominal quantile to the
  # adjusted quantile level that produces uniform PIT
  if (!is.null(recal_fn)) {
    q_adj <- pmin(pmax(recal_fn(q), 0.001), 0.999)
  } else {
    q_adj <- q
  }
  qgamma(q_adj, shape = shape, scale = scale)
}


# ── Evaluation ────────────────────────────────────────────────────────────────
evaluate_model <- function(result, df, feature_cols, split_name) {
  y <- df$ground_truth_precip
  params <- predict_gamma_params(result, df, feature_cols)
  recal_fn <- result$recal_fn  # isotonic recalibration (may be NULL)

  shape <- params$shape
  sc    <- params$scale
  pred_mean <- params$mu

  # Quantiles (recalibrated if recal_fn available)
  quant_levels <- c(0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)
  pred_q <- as.data.table(sapply(quant_levels, function(q) {
    recal_quantile(q, shape, sc, recal_fn)
  }))
  setnames(pred_q, paste0("quant_", quant_levels))

  # CRPS (uses raw Gamma params, not recalibrated)
  crps_vals <- crps_gamma(y, shape = shape, scale = sc)
  mean_crps <- mean(crps_vals)

  # NLL
  nll <- -dgamma(pmax(y, 0.01), shape = shape, scale = sc, log = TRUE)
  nll <- pmin(pmax(nll, -100), 100)
  mean_nll <- mean(nll)

  # MAE, Bias
  mae  <- mean(abs(pred_mean - y))
  bias <- mean(pred_mean - y)
  median_ae <- median(abs(pred_q$quant_0.5 - y))

  # Coverage (using recalibrated quantiles)
  cov_95 <- mean(y >= pred_q$quant_0.025 & y <= pred_q$quant_0.975)
  cov_90 <- mean(y >= pred_q$quant_0.05 & y <= pred_q$quant_0.95)
  cov_50 <- mean(y >= pred_q$quant_0.25 & y <= pred_q$quant_0.75)

  # Reliability at multiple levels (using recalibrated quantiles)
  reliability <- list()
  for (level in c(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)) {
    lo_q <- (1 - level) / 2
    hi_q <- 1 - lo_q
    lo <- recal_quantile(lo_q, shape, sc, recal_fn)
    hi <- recal_quantile(hi_q, shape, sc, recal_fn)
    key <- paste0("cov_", as.integer(level * 100))
    reliability[[key]] <- mean(y >= lo & y <= hi)
  }

  metrics <- c(
    list(
      split          = split_name,
      n_samples      = length(y),
      mean_crps      = mean_crps,
      mean_nll       = mean_nll,
      mae            = mae,
      bias           = bias,
      coverage_95    = cov_95,
      coverage_90    = cov_90,
      coverage_50    = cov_50,
      median_ae      = median_ae,
      mean_observed  = mean(y),
      mean_predicted = mean(pred_mean)
    ),
    reliability
  )

  list(metrics = metrics, params = params, pred_q = pred_q)
}

# ── Diagnostic plots ─────────────────────────────────────────────────────────
plot_diagnostics <- function(result, test_df, pred_params, pred_q,
                             feature_cols, horizon, metrics) {
  y         <- test_df$ground_truth_precip
  shape     <- pred_params$shape
  sc        <- pred_params$scale
  pred_mean <- pred_params$mu
  dates     <- test_df$init_date

  # ── 1. Predicted vs Observed ──
  p1_df <- data.frame(observed = y, predicted = pred_mean)
  max_val <- max(c(y, pred_mean), na.rm = TRUE) * 1.05
  cor_val <- cor(y, pred_mean, use = "complete.obs")

  p1 <- ggplot(p1_df, aes(x = observed, y = predicted)) +
    geom_point(alpha = 0.5, size = 1.5, color = COLORS$primary) +
    geom_abline(slope = 1, intercept = 0, color = COLORS$red,
                linetype = "dashed", linewidth = 0.8) +
    geom_smooth(method = "loess", se = TRUE, color = COLORS$secondary,
                linewidth = 0.7, fill = COLORS$light_blue, alpha = 0.3) +
    annotate("text", x = max_val * 0.02, y = max_val * 0.98,
             label = sprintf("r = %.3f\nMAE = %.1f mm\nBias = %+.1f mm",
                             cor_val, metrics$mae, metrics$bias),
             hjust = 0, vjust = 1, size = 3.5, color = "grey30") +
    labs(
      x = "Observed Precipitation (mm)",
      y = "Predicted Mean (mm)",
      title = sprintf("%d-Day Forecast: Predicted vs Observed", horizon),
      subtitle = "2025 test set | Red dashed = perfect prediction | Orange = LOESS fit"
    ) +
    coord_cartesian(xlim = c(0, max_val), ylim = c(0, max_val)) +
    theme_precip()
  ggsave(file.path(VIS_DIR, sprintf("gamlss_%dd_pred_vs_obs.png", horizon)),
         p1, width = 8, height = 7, dpi = 200)

  # ── 2. Time series with prediction intervals ──
  n_plot <- length(y)
  p2_df <- data.frame(
    date    = dates[1:n_plot],
    obs     = y[1:n_plot],
    median  = pred_q$quant_0.5[1:n_plot],
    q05     = pred_q$quant_0.05[1:n_plot],
    q95     = pred_q$quant_0.95[1:n_plot],
    q25     = pred_q$quant_0.25[1:n_plot],
    q75     = pred_q$quant_0.75[1:n_plot]
  )
  p2_df$outside <- p2_df$obs < p2_df$q05 | p2_df$obs > p2_df$q95

  p2 <- ggplot(p2_df, aes(x = date)) +
    geom_ribbon(aes(ymin = q05, ymax = q95), fill = COLORS$light_blue, alpha = 0.4) +
    geom_ribbon(aes(ymin = q25, ymax = q75), fill = COLORS$primary, alpha = 0.3) +
    geom_line(aes(y = median), color = COLORS$primary, linewidth = 0.6) +
    geom_point(data = p2_df[!p2_df$outside, ], aes(y = obs),
               size = 1.0, color = "grey30", alpha = 0.7) +
    geom_point(data = p2_df[p2_df$outside, ], aes(y = obs),
               size = 1.2, color = COLORS$red, shape = 4, stroke = 0.8) +
    scale_x_date(date_breaks = "2 months", date_labels = "%b %Y") +
    labs(
      x = NULL, y = "Cumulative Precipitation (mm)",
      title = sprintf("%d-Day Forecast: Prediction Intervals (2025)", horizon),
      subtitle = sprintf("Light blue = 90%% PI (%.1f%% actual) | Dark blue = 50%% PI (%.1f%% actual) | Red X = outside 90%% PI",
                         metrics$coverage_90 * 100, metrics$coverage_50 * 100)
    ) +
    theme_precip() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  ggsave(file.path(VIS_DIR, sprintf("gamlss_%dd_intervals.png", horizon)),
         p2, width = 14, height = 6, dpi = 200)

  # ── 3. Feature importance ──
  imp <- xgb.importance(model = result$mu_model)
  if (nrow(imp) > 0) {
    imp_top <- head(imp[order(-Gain)], 25)
    imp_top$Feature <- factor(imp_top$Feature, levels = rev(imp_top$Feature))

    categorize_feature <- function(feat) {
      if (grepl("^(cum_|mean_6h|max_6h|forecast|spread|q[0-9].*_apcp|6h|top10|gefs_|persistence|frac_|longest_|num_wet|min_6h|std_6h|median_6h)", feat)) {
        "Forecast / GEFS"
      } else if (grepl("^clim_", feat)) {
        "Climatology"
      } else if (grepl("^(precip_|rainy_|max_precip|days_since|thunder|fog|mean_temp|temp_|mean_wind|mean_gust|mean_sun|sunshine|mean_precip_h|rain_flag|haze|mean_dtr)", feat)) {
        "Recent Conditions"
      } else if (grepl("^(bias_|rolling_|forecast_bias|spread_adequacy|forecast_vs_gefs)", feat)) {
        "Bias Correction"
      } else {
        "Calendar / Cross"
      }
    }
    imp_top$category <- sapply(as.character(imp_top$Feature), categorize_feature)

    p3 <- ggplot(imp_top, aes(x = Gain, y = Feature, fill = category)) +
      geom_col(width = 0.7) +
      scale_fill_manual(values = c(
        "Forecast / GEFS"   = COLORS$forecast,
        "Climatology"       = COLORS$climate,
        "Recent Conditions" = COLORS$recent,
        "Bias Correction"   = COLORS$secondary,
        "Calendar / Cross"  = COLORS$calendar
      )) +
      labs(
        x = "Gain (Fractional Improvement)", y = NULL, fill = "Feature Group",
        title = sprintf("%d-Day Forecast: Top 25 Features by Importance", horizon),
        subtitle = "XGBoost gain metric from mean prediction model"
      ) +
      theme_precip() +
      theme(legend.position = "right")
    ggsave(file.path(VIS_DIR, sprintf("gamlss_%dd_importance.png", horizon)),
           p3, width = 11, height = 8, dpi = 200)
  }

  # ── 4. SHAP analysis ──
  tryCatch({
    cat("    Computing SHAP values...\n")
    X_test <- as.matrix(test_df[, ..feature_cols])
    storage.mode(X_test) <- "double"

    sv <- shapviz(result$mu_model, X_pred = X_test, X = X_test)

    p_shap_bee <- sv_importance(sv, kind = "beeswarm", max_display = 20) +
      ggtitle(sprintf("%d-Day Forecast: SHAP Values (Mean Model)", horizon)) +
      theme_precip()
    ggsave(file.path(VIS_DIR, sprintf("gamlss_%dd_shap_beeswarm.png", horizon)),
           p_shap_bee, width = 12, height = 10, dpi = 200)

    p_shap_bar <- sv_importance(sv, kind = "bar", max_display = 20) +
      ggtitle(sprintf("%d-Day Forecast: Mean |SHAP| Importance", horizon)) +
      theme_precip()
    ggsave(file.path(VIS_DIR, sprintf("gamlss_%dd_shap_bar.png", horizon)),
           p_shap_bar, width = 12, height = 8, dpi = 200)

    top4 <- head(names(sort(colMeans(abs(sv$S)), decreasing = TRUE)), 4)
    dep_plots <- lapply(top4, function(feat) {
      sv_dependence(sv, v = feat) + ggtitle(feat) + theme_precip(base_size = 10)
    })
    p_dep <- wrap_plots(dep_plots, ncol = 2)
    ggsave(file.path(VIS_DIR, sprintf("gamlss_%dd_shap_dependence.png", horizon)),
           p_dep, width = 14, height = 10, dpi = 200)

    cat("    SHAP plots saved.\n")
  }, error = function(e) {
    cat(sprintf("  Warning: Could not compute SHAP values: %s\n", e$message))
  })

  # ── 5. PIT histogram (before and after recalibration) ──
  pit_raw <- pgamma(pmax(y, 0.001), shape = shape, scale = sc)
  recal_fn <- result$recal_fn

  if (!is.null(recal_fn)) {
    pit_recal <- pmin(pmax(recal_fn(pit_raw), 0), 1)
    ks_raw   <- suppressWarnings(ks.test(pit_raw, "punif")$statistic)
    ks_recal <- suppressWarnings(ks.test(pit_recal, "punif")$statistic)

    pit_df <- rbind(
      data.frame(pit = pit_raw,   panel = sprintf("Before Recalibration (KS=%.3f)", ks_raw)),
      data.frame(pit = pit_recal, panel = sprintf("After Recalibration (KS=%.3f)", ks_recal))
    )

    p4 <- ggplot(pit_df, aes(x = pit)) +
      geom_histogram(aes(y = after_stat(density)), bins = 20,
                     fill = COLORS$primary, color = "white", alpha = 0.8) +
      geom_hline(yintercept = 1, color = COLORS$red, linetype = "dashed",
                 linewidth = 0.7) +
      facet_wrap(~panel) +
      labs(
        x = "PIT Value", y = "Density",
        title = sprintf("%d-Day Forecast: PIT Histogram", horizon),
        subtitle = "Uniform (red dashed) = perfect calibration | Isotonic recalibration applied"
      ) +
      theme_precip()
    ggsave(file.path(VIS_DIR, sprintf("gamlss_%dd_pit.png", horizon)),
           p4, width = 14, height = 5, dpi = 200)
  } else {
    ks_raw <- suppressWarnings(ks.test(pit_raw, "punif")$statistic)

    p4 <- ggplot(data.frame(pit = pit_raw), aes(x = pit)) +
      geom_histogram(aes(y = after_stat(density)), bins = 20,
                     fill = COLORS$primary, color = "white", alpha = 0.8) +
      geom_hline(yintercept = 1, color = COLORS$red, linetype = "dashed",
                 linewidth = 0.7) +
      annotate("text", x = 0.95, y = Inf, label = sprintf("KS = %.3f", ks_raw),
               hjust = 1, vjust = 2, size = 3.5) +
      labs(
        x = "PIT Value", y = "Density",
        title = sprintf("%d-Day Forecast: PIT Histogram", horizon),
        subtitle = "Uniform (red dashed) = perfect calibration"
      ) +
      theme_precip()
    ggsave(file.path(VIS_DIR, sprintf("gamlss_%dd_pit.png", horizon)),
           p4, width = 7, height = 5, dpi = 200)
  }

  # ── 6. Reliability diagram (uses recalibrated quantiles) ──
  nominal_levels <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
  actual_covs <- sapply(nominal_levels, function(level) {
    lo_q <- (1 - level) / 2
    hi_q <- 1 - lo_q
    lo <- recal_quantile(lo_q, shape, sc, recal_fn)
    hi <- recal_quantile(hi_q, shape, sc, recal_fn)
    mean(y >= lo & y <= hi)
  })

  p5_df <- data.frame(nominal = nominal_levels, actual = actual_covs)

  p5 <- ggplot(p5_df, aes(x = nominal, y = actual)) +
    geom_ribbon(aes(ymin = pmax(nominal - 0.1, 0), ymax = pmin(nominal + 0.1, 1)),
                fill = "grey85", alpha = 0.5) +
    geom_abline(slope = 1, intercept = 0, color = COLORS$red,
                linetype = "dashed", linewidth = 0.7) +
    geom_point(color = COLORS$primary, size = 3.5) +
    geom_line(color = COLORS$primary, linewidth = 0.8) +
    geom_text(aes(label = sprintf("%.0f%%", actual * 100)),
              vjust = -1.2, size = 3, color = "grey40") +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(
      x = "Nominal Coverage Level", y = "Actual Coverage",
      title = sprintf("%d-Day Forecast: Reliability Diagram", horizon),
      subtitle = "Grey band = +/- 10%% tolerance | Closer to diagonal = better calibrated"
    ) +
    theme_precip() +
    theme(aspect.ratio = 1)
  ggsave(file.path(VIS_DIR, sprintf("gamlss_%dd_reliability.png", horizon)),
         p5, width = 7, height = 7, dpi = 200)

  # ── 7. Residual distribution ──
  resid <- pred_mean - y
  p6_df <- data.frame(residual = resid)

  p6 <- ggplot(p6_df, aes(x = residual)) +
    geom_histogram(aes(y = after_stat(density)), bins = 40,
                   fill = COLORS$primary, color = "white", alpha = 0.7) +
    geom_vline(xintercept = 0, color = COLORS$red, linetype = "dashed") +
    geom_vline(xintercept = mean(resid), color = COLORS$secondary, linewidth = 0.8) +
    annotate("text", x = mean(resid), y = Inf,
             label = sprintf("Mean bias: %+.1f mm", mean(resid)),
             vjust = 2, hjust = -0.1, size = 3.5, color = COLORS$secondary) +
    labs(
      x = "Residual (Predicted - Observed, mm)", y = "Density",
      title = sprintf("%d-Day Forecast: Residual Distribution", horizon)
    ) +
    theme_precip()
  ggsave(file.path(VIS_DIR, sprintf("gamlss_%dd_residuals.png", horizon)),
         p6, width = 8, height = 5, dpi = 200)

  # ── 8. Monthly performance ──
  monthly_df <- data.frame(
    month   = factor(format(dates, "%b"), levels = month.abb),
    month_n = as.integer(format(dates, "%m")),
    observed  = y,
    predicted = pred_mean
  )
  monthly_long <- rbind(
    data.frame(month = monthly_df$month, value = monthly_df$observed, type = "Observed"),
    data.frame(month = monthly_df$month, value = monthly_df$predicted, type = "Predicted")
  )

  p7 <- ggplot(monthly_long, aes(x = month, y = value, fill = type)) +
    stat_summary(fun = mean, geom = "col", position = position_dodge(0.8),
                 width = 0.7, alpha = 0.8) +
    stat_summary(fun.data = mean_se, geom = "errorbar",
                 position = position_dodge(0.8), width = 0.3) +
    scale_fill_manual(values = c("Observed" = COLORS$primary,
                                  "Predicted" = COLORS$secondary)) +
    labs(
      x = NULL, y = "Mean Precipitation (mm)", fill = NULL,
      title = sprintf("%d-Day Forecast: Monthly Performance", horizon),
      subtitle = "Mean +/- SE by month (2025 test set)"
    ) +
    theme_precip()
  ggsave(file.path(VIS_DIR, sprintf("gamlss_%dd_monthly.png", horizon)),
         p7, width = 10, height = 5, dpi = 200)
}

# ── Main ──────────────────────────────────────────────────────────────────────
main <- function() {
  if (!file.exists(DATA_FILE)) {
    stop(sprintf("Training dataset not found at %s\nRun: python scripts/data_processing/build_training_dataset.py",
                 DATA_FILE))
  }

  all_metrics <- list()

  for (horizon in horizons) {
    cat(sprintf("\n%s\n", strrep("=", 60)))
    cat(sprintf("Training %d-day Horizon Model\n", horizon))
    cat(sprintf("%s\n", strrep("=", 60)))

    cat("\nLoading data...\n")
    splits <- load_data(horizon)
    train_df <- splits$train
    val_df   <- splits$val
    test_df  <- splits$test

    feature_cols <- get_feature_cols(train_df)
    nan_pcts <- sapply(feature_cols, function(col) mean(is.na(train_df[[col]])))
    drop_cols <- names(nan_pcts[nan_pcts > 0.5])
    if (length(drop_cols) > 0) {
      cat(sprintf("  Dropping %d high-NaN columns\n", length(drop_cols)))
      feature_cols <- setdiff(feature_cols, drop_cols)
    }

    for (col in feature_cols) {
      med_val <- median(train_df[[col]], na.rm = TRUE)
      if (is.na(med_val)) med_val <- 0
      set(train_df, which(is.na(train_df[[col]])), col, med_val)
      set(val_df,   which(is.na(val_df[[col]])),   col, med_val)
      set(test_df,  which(is.na(test_df[[col]])),  col, med_val)
    }

    cat(sprintf("  Features: %d\n", length(feature_cols)))

    cat("\nTraining...\n")
    result <- train_model(train_df, val_df, feature_cols, horizon, opt_quick)

    cat("\nEvaluating...\n")
    test_eval <- NULL
    for (split_info in list(
      list(name = "val",  df = val_df),
      list(name = "test", df = test_df)
    )) {
      eval_result <- evaluate_model(
        result, split_info$df, feature_cols, split_info$name
      )
      m <- eval_result$metrics
      m$horizon_days <- horizon
      all_metrics[[length(all_metrics) + 1]] <- m

      cat(sprintf("\n  %s metrics:\n", toupper(split_info$name)))
      cat(sprintf("    CRPS:        %.2f\n", m$mean_crps))
      cat(sprintf("    NLL:         %.2f\n", m$mean_nll))
      cat(sprintf("    MAE:         %.1f mm\n", m$mae))
      cat(sprintf("    Bias:        %+.1f mm\n", m$bias))
      cat(sprintf("    Coverage 95: %.1f%%\n", m$coverage_95 * 100))
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

      if (split_info$name == "test") test_eval <- eval_result
    }

    cat("\n  Generating diagnostic plots...\n")
    plot_diagnostics(
      result, test_df,
      test_eval$params, test_eval$pred_q,
      feature_cols, horizon, test_eval$metrics
    )

    model_path <- file.path(MODELS_DIR, sprintf("gamlss_%dd.rds", horizon))
    saveRDS(list(
      mu_model     = result$mu_model,
      var_model    = result$var_model,
      mu_params    = result$mu_params,
      var_params   = result$var_params,
      mu_rounds    = result$mu_rounds,
      var_rounds   = result$var_rounds,
      scale_factor = result$scale_factor,
      recal_fn     = result$recal_fn,
      feature_cols = feature_cols
    ), model_path)
    cat(sprintf("\n  Model saved to %s\n", model_path))
  }

  if (length(all_metrics) > 0) {
    metrics_df <- rbindlist(all_metrics, fill = TRUE)
    metrics_path <- file.path(MODELS_DIR, "gamlss_all_metrics.csv")
    fwrite(metrics_df, metrics_path)
    cat(sprintf("\n%s\n", strrep("=", 60)))
    cat(sprintf("All metrics saved to %s\n", metrics_path))
    cat(sprintf("%s\n", strrep("=", 60)))
    print(metrics_df)
  }

  cat("\nDone!\n")
}

main()
