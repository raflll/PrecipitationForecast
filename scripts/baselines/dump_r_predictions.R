#!/usr/bin/env Rscript
# Load saved R models (xgboost_gamma_<h>d.json, gamlss_<h>d.rds) and dump
# per-sample Gamma predictions for val+test to CSV. Used by the cross-model
# comparison plot.

suppressPackageStartupMessages({
  library(arrow)
  library(xgboost)
  library(data.table)
  library(jsonlite)
})

# Parse args ------------------------------------------------------------------
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(horizon = 14, xgboost_out = NULL, gamlss_out = NULL)
  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--horizon") { out$horizon <- as.integer(args[i + 1]); i <- i + 2 }
    else if (args[i] == "--xgboost-out") { out$xgboost_out <- args[i + 1]; i <- i + 2 }
    else if (args[i] == "--gamlss-out") { out$gamlss_out <- args[i + 1]; i <- i + 2 }
    else i <- i + 1
  }
  out
}

cfg <- parse_args()
H   <- cfg$horizon

# Detect script dir from Rscript args, fallback to getwd()
script_path <- (function() {
  sub_args <- commandArgs(trailingOnly = FALSE)
  needle <- "--file="
  hit <- sub_args[startsWith(sub_args, needle)]
  if (length(hit) > 0) sub(needle, "", hit[1]) else NA_character_
})()
script_dir <- if (!is.na(script_path)) dirname(normalizePath(script_path)) else getwd()
REPO <- normalizePath(file.path(script_dir, "..", ".."), mustWork = FALSE)
PARQ <- file.path(REPO, "data", "training", "training_dataset.parquet")
MOD  <- file.path(REPO, "models")

NON_FEATURE_COLS <- c("init_date", "horizon_days", "ground_truth_precip", "source", "year")

# Load data -------------------------------------------------------------------
df <- as.data.table(read_parquet(PARQ))
df[, year := as.integer(format(init_date, "%Y"))]
sub <- df[horizon_days == H]
train <- sub[year <= 2023]
eval_df <- sub[year %in% c(2024, 2025)]
eval_df[, split := ifelse(year == 2024, "val", "test")]

cat(sprintf("horizon=%dd  train=%d  eval=%d\n", H, nrow(train), nrow(eval_df)))

feature_cols <- setdiff(names(df), NON_FEATURE_COLS)
# Drop high-NaN features (consistent with training script)
nan_pcts <- sapply(feature_cols, function(c) mean(is.na(train[[c]])))
feature_cols <- feature_cols[nan_pcts < 0.5]
# Median fill from training
for (c in feature_cols) {
  if (anyNA(eval_df[[c]])) {
    med <- median(train[[c]], na.rm = TRUE)
    if (is.na(med)) med <- 0
    eval_df[is.na(get(c)), (c) := med]
  }
}

X_eval <- as.matrix(eval_df[, ..feature_cols])
storage.mode(X_eval) <- "double"
colnames(X_eval) <- feature_cols
dmat <- xgb.DMatrix(X_eval, label = rep(0, nrow(X_eval)))

# Helper: Gamma quantile frame
qframe <- function(shape, scale, observed, init_date, split) {
  data.table(
    init_date = init_date,
    split     = split,
    observed  = observed,
    mean      = shape * scale,
    q05       = qgamma(0.05, shape = shape, scale = scale),
    q25       = qgamma(0.25, shape = shape, scale = scale),
    q50       = qgamma(0.50, shape = shape, scale = scale),
    q75       = qgamma(0.75, shape = shape, scale = scale),
    q95       = qgamma(0.95, shape = shape, scale = scale)
  )
}

# ---- XGBoost-Gamma ----------------------------------------------------------
xg_path <- file.path(MOD, sprintf("xgboost_gamma_%dd.json", H))
xg_par_path <- file.path(MOD, sprintf("xgboost_gamma_%dd_params.json", H))
if (!is.null(cfg$xgboost_out) && file.exists(xg_path) && file.exists(xg_par_path)) {
  cat("Loading xgboost_gamma...\n")
  m  <- xgb.load(xg_path)
  par <- fromJSON(xg_par_path)
  shape <- par$gamma_shape
  scale_factor <- par$scale_factor
  if (is.null(scale_factor)) scale_factor <- 1.0
  mu <- predict(m, dmat)
  scale <- (mu / shape) * scale_factor
  out <- qframe(rep(shape, length(mu)), scale,
                eval_df$ground_truth_precip, eval_df$init_date, eval_df$split)
  fwrite(out, cfg$xgboost_out)
  cat(sprintf("  wrote %s (%d rows)\n", cfg$xgboost_out, nrow(out)))
}

# ---- GAMLSS 2-stage ---------------------------------------------------------
gam_path <- file.path(MOD, sprintf("gamlss_%dd.rds", H))
if (!is.null(cfg$gamlss_out) && file.exists(gam_path)) {
  cat("Loading gamlss...\n")
  result <- readRDS(gam_path)
  feat_g <- result$feature_cols
  X_g <- as.matrix(eval_df[, ..feat_g])
  storage.mode(X_g) <- "double"
  colnames(X_g) <- feat_g
  dlog <- xgb.DMatrix(X_g, label = log1p(eval_df$ground_truth_precip))
  dvar <- xgb.DMatrix(X_g, label = rep(0, nrow(X_g)))

  log_pred <- predict(result$mu_model, dlog)
  log_var_pred <- predict(result$var_model, dvar)
  sigma_raw <- pmax(expm1(log_var_pred), 0.1)
  sigma_scaled <- sigma_raw * result$scale_factor
  mu <- pmax(expm1(log_pred + sigma_raw^2 / 2), 0.1)
  variance <- pmax(mu^2 * (exp(sigma_scaled^2) - 1), 0.01)
  shape <- pmax(mu^2 / variance, 0.05)
  scale <- pmax(variance / mu, 0.001)
  out <- qframe(shape, scale,
                eval_df$ground_truth_precip, eval_df$init_date, eval_df$split)
  fwrite(out, cfg$gamlss_out)
  cat(sprintf("  wrote %s (%d rows)\n", cfg$gamlss_out, nrow(out)))
}

cat("R prediction dump done.\n")
