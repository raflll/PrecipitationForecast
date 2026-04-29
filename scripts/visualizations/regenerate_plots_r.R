#!/usr/bin/env Rscript
# Regenerate diagnostic plots for a saved xgboost_gamma model without retraining.
# Loads models/xgboost_gamma_{h}d.json + _params.json, reloads the test split
# from the training parquet, and calls plot_diagnostics().
#
# Usage:
#   Rscript scripts/visualizations/regenerate_plots_r.R --horizon 14
#   Rscript scripts/visualizations/regenerate_plots_r.R            # all horizons

suppressPackageStartupMessages({
  library(arrow)
  library(xgboost)
  library(data.table)
  library(ggplot2)
  library(SHAPforxgboost)
  library(jsonlite)
})

# ── Paths ─────────────────────────────────────────────────────────────────────
args0    <- commandArgs(trailingOnly = FALSE)
script   <- sub("^--file=", "", args0[grep("^--file=", args0)][1])
SRC_DIR  <- normalizePath(file.path(dirname(script), "..", "..", "src"))
ROOT_DIR <- normalizePath(file.path(SRC_DIR, ".."))

# Source the training script to reuse plot_diagnostics(). The sentinel tells
# train_xgboost_gamma.R to skip its main() block.
SKIP_MAIN <- TRUE
sys.source(file.path(SRC_DIR, "train_xgboost_gamma.R"), envir = globalenv(),
           chdir = TRUE, keep.source = FALSE)

DATA_FILE  <- file.path(ROOT_DIR, "data", "training", "training_dataset.parquet")
MODELS_DIR <- file.path(ROOT_DIR, "models")
VIS_DIR    <- file.path(ROOT_DIR, "visualizations")

# ── Parse args ────────────────────────────────────────────────────────────────
cli <- commandArgs(trailingOnly = TRUE)
horizons <- if ("--horizon" %in% cli) {
  as.integer(cli[which(cli == "--horizon") + 1])
} else {
  c(3, 7, 14, 21, 28, 35)
}

# ── Load dataset once ─────────────────────────────────────────────────────────
cat("Loading training dataset...\n")
df <- as.data.table(read_parquet(DATA_FILE))
df[, init_date := as.POSIXct(init_date)]

NON_FEATURE_COLS <- c("init_date", "horizon_days", "ground_truth_precip", "source")

for (horizon in horizons) {
  cat(sprintf("\n%s\n", strrep("=", 60)))
  cat(sprintf("Regenerating plots for %d-day horizon\n", horizon))
  cat(sprintf("%s\n", strrep("=", 60)))

  model_path  <- file.path(MODELS_DIR, sprintf("xgboost_gamma_%dd.json", horizon))
  params_path <- file.path(MODELS_DIR, sprintf("xgboost_gamma_%dd_params.json", horizon))

  if (!file.exists(model_path) || !file.exists(params_path)) {
    cat(sprintf("  SKIP: %s or params file missing.\n", model_path))
    next
  }

  model        <- xgb.load(model_path)
  params       <- fromJSON(params_path)
  shape        <- params$gamma_shape
  cat(sprintf("  Loaded shape=%.4f\n", shape))

  # Reconstruct the train/val/test splits
  h_df <- df[horizon_days == horizon]
  h_df <- h_df[order(init_date)]
  train_dt <- h_df[format(init_date, "%Y") <= "2023"]
  val_dt   <- h_df[format(init_date, "%Y") == "2024"]
  test_dt  <- h_df[format(init_date, "%Y") == "2025"]

  # Feature columns — match training logic (drop >50% NaN columns, fill rest)
  feature_cols <- setdiff(names(h_df), NON_FEATURE_COLS)
  nan_pct <- sapply(train_dt[, ..feature_cols], function(x) mean(is.na(x)))
  drop_cols <- names(nan_pct)[nan_pct > 0.5]
  feature_cols <- setdiff(feature_cols, drop_cols)
  for (dt in list(val_dt, test_dt)) {
    for (col in feature_cols) {
      set(dt, which(is.na(dt[[col]])), col, 0)
    }
  }
  cat(sprintf("  Val samples: %d | Test samples: %d | features: %d\n",
              nrow(val_dt), nrow(test_dt), length(feature_cols)))

  # Recompute scale_factor on val using KS-minimisation objective (same as
  # training). This means plot regeneration always uses the latest calibration
  # objective, even if the saved params JSON is from an older training run.
  X_val_m  <- as.matrix(val_dt[, ..feature_cols])
  colnames(X_val_m) <- feature_cols
  mu_val   <- predict(model, xgb.DMatrix(X_val_m,
                                         label = val_dt$ground_truth_precip))
  rate_val <- shape / mu_val
  scale_factor <- find_optimal_scale_factor(
    y = val_dt$ground_truth_precip, shape = shape, rate = rate_val,
    objective = "ks"
  )
  cat(sprintf("  Recomputed scale_factor (KS-minimising): %.2f\n", scale_factor))

  plot_diagnostics(model, shape, test_dt, feature_cols, horizon,
                   scale_factor = scale_factor, val_df = val_dt)
  cat(sprintf("  Wrote plots to %s/xgboost_gamma_%dd_*.png\n", VIS_DIR, horizon))
}

cat("\nDone.\n")
