#!/usr/bin/env python3
"""
Train XGBoostLSS models to predict gamma distribution parameters for
cumulative precipitation over 6 forecast horizons (3, 7, 14, 21, 28, 35 days).

Reads the training dataset built by build_training_dataset.py, trains one
model per horizon using hyperparameter optimization, and saves models + metrics.

Usage:
    # Train all horizons with hyperparameter tuning
    python src/train_xgboostlss.py

    # Train a single horizon (faster for testing)
    python src/train_xgboostlss.py --horizon 7

    # Quick test run (fewer trials, rounds)
    python src/train_xgboostlss.py --quick

Output:
    models/xgboostlss_{horizon}d.json       — trained model
    models/xgboostlss_{horizon}d_metrics.csv — evaluation metrics
    visualizations/xgboostlss_{horizon}d_*.png — diagnostic plots
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import gamma as sp_gamma
from xgboostlss.distributions.Gamma import Gamma
from xgboostlss.model import XGBoostLSS

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "training" / "training_dataset.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
VIS_DIR = PROJECT_ROOT / "visualizations"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [3, 7, 14, 21, 28, 35]

# Columns that are NOT features (keys, target, metadata)
NON_FEATURE_COLS = ["init_date", "horizon_days", "ground_truth_precip", "source"]


def load_data(horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and split data for a specific horizon.

    Returns (train_df, val_df, test_df) with temporal split:
        Train: 2000-2023 (excluding 2022 which is missing)
        Val:   2024
        Test:  2025
    """
    df = pd.read_parquet(DATA_FILE)
    df = df[df["horizon_days"] == horizon].copy()
    df["init_date"] = pd.to_datetime(df["init_date"])
    df = df.sort_values("init_date").reset_index(drop=True)

    # Temporal split
    train = df[df["init_date"].dt.year <= 2023]
    val = df[df["init_date"].dt.year == 2024]
    test = df[df["init_date"].dt.year == 2025]

    print(f"  Train: {len(train)} rows ({train['init_date'].min().date()} to {train['init_date'].max().date()})")
    print(f"  Val:   {len(val)} rows ({val['init_date'].min().date()} to {val['init_date'].max().date()})")
    print(f"  Test:  {len(test)} rows ({test['init_date'].min().date()} to {test['init_date'].max().date()})")

    return train, val, test


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get feature column names (everything except keys, target, metadata)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def prepare_dmatrix(
    df: pd.DataFrame, feature_cols: list[str]
) -> xgb.DMatrix:
    """Create an XGBoost DMatrix from a DataFrame."""
    X = df[feature_cols].values.astype(np.float32)
    y = df["ground_truth_precip"].values.astype(np.float32)
    return xgb.DMatrix(X, label=y, feature_names=feature_cols)


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    quick: bool = False,
) -> tuple[XGBoostLSS, dict]:
    """Train an XGBoostLSS model with hyperparameter optimization."""
    dtrain = prepare_dmatrix(train_df, feature_cols)
    dval = prepare_dmatrix(val_df, feature_cols)

    xgblss = XGBoostLSS(
        Gamma(
            stabilization="MAD",
            response_fn="exp",
            loss_fn="nll",
        )
    )

    hp_dict = {
        "eta": ["float", {"low": 1e-3, "high": 0.2, "log": True}],
        "max_depth": ["int", {"low": 2, "high": 5, "log": False}],
        "gamma": ["float", {"low": 1e-8, "high": 40.0, "log": True}],
        "subsample": ["float", {"low": 0.5, "high": 0.9, "log": False}],
        "colsample_bytree": ["float", {"low": 0.3, "high": 0.7, "log": False}],
        "min_child_weight": ["float", {"low": 10.0, "high": 500, "log": True}],
        "reg_alpha": ["float", {"low": 0.01, "high": 50.0, "log": True}],
        "reg_lambda": ["float", {"low": 0.01, "high": 50.0, "log": True}],
        "booster": ["categorical", ["gbtree"]],
    }

    n_trials = 10 if quick else 30
    max_minutes = 3 if quick else 15
    num_boost_round = 100 if quick else 500

    print(f"  Running hyperparameter optimization ({n_trials} trials, {max_minutes}min max)...")
    opt_params = xgblss.hyper_opt(
        hp_dict,
        dtrain,
        num_boost_round=num_boost_round,
        nfold=3,
        early_stopping_rounds=20,
        max_minutes=max_minutes,
        n_trials=n_trials,
        silence=True,
        seed=42,
        hp_seed=42,
    )

    n_rounds = opt_params.pop("opt_rounds")
    print(f"  Best params: {json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in opt_params.items()}, indent=4)}")
    print(f"  Optimal rounds: {n_rounds}")

    print(f"  Training final model ({n_rounds} rounds)...")
    xgblss.train(
        opt_params,
        dtrain,
        num_boost_round=n_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=max(1, n_rounds // 5),
    )

    return xgblss, opt_params


def _crps_gamma_analytical(y: float, concentration: float, scale: float) -> float:
    """Analytical CRPS for a single Gamma(concentration, scale) observation.

    Uses the closed-form: CRPS = y*(2F(y) - 1) - a*s*(2*F_a1(y) - 1) - a*s/B(0.5, a)
    where F is Gamma CDF, F_a1 is Gamma(a+1, s) CDF, B is Beta function.
    """
    from scipy.special import beta as beta_fn
    a, s = concentration, scale
    F_y = sp_gamma.cdf(y, a=a, scale=s)
    F_y_a1 = sp_gamma.cdf(y, a=a + 1, scale=s)
    return y * (2 * F_y - 1) - a * s * (2 * F_y_a1 - 1 + 1.0 / (a * beta_fn(0.5, a)))


def evaluate_model(
    xgblss: XGBoostLSS,
    df: pd.DataFrame,
    feature_cols: list[str],
    split_name: str,
) -> dict:
    """Evaluate model on a dataset, return metrics dict."""
    dmat = prepare_dmatrix(df, feature_cols)
    y = df["ground_truth_precip"].values

    # Predict distribution parameters
    pred_params = xgblss.predict(dmat, pred_type="parameters")
    concentration = pred_params["concentration"].values
    rate = pred_params["rate"].values
    scale = 1.0 / rate

    # Predicted mean and variance
    pred_mean = concentration / rate
    pred_var = concentration / rate**2

    # Compute quantiles analytically from gamma params (no sampling needed)
    quantile_levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    pred_q = pd.DataFrame({
        f"quant_{q}": sp_gamma.ppf(q, a=concentration, scale=scale)
        for q in quantile_levels
    })

    # CRPS via analytical formula for gamma distribution
    # CRPS(Gamma(a, scale), y) computed using scipy
    crps_scores = np.array([
        _crps_gamma_analytical(y[i], concentration[i], scale[i])
        for i in range(len(y))
    ])
    mean_crps = float(np.mean(crps_scores))

    # Negative log-likelihood
    nll = -sp_gamma.logpdf(y, a=concentration, scale=scale)
    # Clip extreme NLL values to avoid inf issues
    nll = np.clip(nll, -100, 100)
    mean_nll = float(np.mean(nll))

    # Mean absolute error of predicted mean
    mae = float(np.mean(np.abs(pred_mean - y)))

    # Bias
    bias = float(np.mean(pred_mean - y))

    # Coverage
    coverage_90 = float(np.mean((y >= pred_q["quant_0.05"].values) & (y <= pred_q["quant_0.95"].values)))
    coverage_50 = float(np.mean((y >= pred_q["quant_0.25"].values) & (y <= pred_q["quant_0.75"].values)))

    # Median absolute deviation
    median_ae = float(np.median(np.abs(pred_q["quant_0.5"].values - y)))

    # Reliability: check coverage at multiple levels
    reliability = {}
    for level in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        lo_q = (1 - level) / 2
        hi_q = 1 - lo_q
        lo = sp_gamma.ppf(lo_q, a=concentration, scale=scale)
        hi = sp_gamma.ppf(hi_q, a=concentration, scale=scale)
        actual_cov = float(np.mean((y >= lo) & (y <= hi)))
        reliability[f"cov_{int(level*100)}"] = actual_cov

    metrics = {
        "split": split_name,
        "n_samples": len(y),
        "mean_crps": mean_crps,
        "mean_nll": mean_nll,
        "mae": mae,
        "bias": bias,
        "coverage_90": coverage_90,
        "coverage_50": coverage_50,
        "median_ae": median_ae,
        "mean_observed": float(np.mean(y)),
        "mean_predicted": float(np.mean(pred_mean)),
        **reliability,
    }

    return metrics, pred_params, pred_q


def plot_diagnostics(
    xgblss: XGBoostLSS,
    test_df: pd.DataFrame,
    pred_params: pd.DataFrame,
    pred_q: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
):
    """Generate diagnostic plots for a trained model."""
    y = test_df["ground_truth_precip"].values
    concentration = pred_params["concentration"].values
    rate = pred_params["rate"].values
    pred_mean = concentration / rate

    # --- 1. Predicted vs Actual scatter ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, pred_mean, alpha=0.4, s=15)
    max_val = max(y.max(), pred_mean.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="Perfect")
    ax.set_xlabel("Observed Precipitation (mm)")
    ax.set_ylabel("Predicted Mean (mm)")
    ax.set_title(f"XGBoostLSS {horizon}d: Predicted vs Observed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIS_DIR / f"xgboostlss_{horizon}d_pred_vs_obs.png", dpi=150)
    plt.close(fig)

    # --- 2. Prediction intervals time series (first 100 test points) ---
    n_plot = min(100, len(y))
    idx = np.arange(n_plot)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(
        idx,
        pred_q["quant_0.05"].values[:n_plot],
        pred_q["quant_0.95"].values[:n_plot],
        alpha=0.2,
        color="blue",
        label="90% PI",
    )
    ax.fill_between(
        idx,
        pred_q["quant_0.25"].values[:n_plot],
        pred_q["quant_0.75"].values[:n_plot],
        alpha=0.3,
        color="blue",
        label="50% PI",
    )
    ax.plot(idx, pred_q["quant_0.5"].values[:n_plot], "b-", linewidth=1, label="Median")
    ax.plot(idx, y[:n_plot], "ko", markersize=3, label="Observed")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Precipitation (mm)")
    ax.set_title(f"XGBoostLSS {horizon}d: Prediction Intervals (2025 test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIS_DIR / f"xgboostlss_{horizon}d_intervals.png", dpi=150)
    plt.close(fig)

    # --- 3. Feature importance (gain-based, top 25) ---
    dtest = prepare_dmatrix(test_df, feature_cols)
    try:
        booster = xgblss.booster
        importance = booster.get_score(importance_type="gain")
        if importance:
            # Handle both fN format and actual feature name formats
            sample_key = next(iter(importance))
            if sample_key.startswith("f") and sample_key[1:].isdigit():
                imp_df = pd.DataFrame([
                    {"feature": feature_cols[int(k[1:])], "gain": v}
                    for k, v in importance.items()
                    if k.startswith("f") and k[1:].isdigit() and int(k[1:]) < len(feature_cols)
                ])
            else:
                imp_df = pd.DataFrame([
                    {"feature": k, "gain": v}
                    for k, v in importance.items()
                ])
            imp_df = imp_df.sort_values("gain", ascending=True).tail(25)

            fig, ax = plt.subplots(figsize=(10, 8))
            colors = []
            for feat in imp_df["feature"]:
                if (feat.startswith("cum_") or feat.startswith("mean_6h") or
                        feat.startswith("max_6h") or "forecast" in feat or
                        "spread" in feat or (feat.startswith("q") and "_apcp" in feat) or
                        "6h" in feat or "top10" in feat or feat.startswith("gefs_") or
                        feat.startswith("persistence")):
                    colors.append("#2196F3")  # blue = forecast
                elif feat.startswith("clim_"):
                    colors.append("#FF9800")  # orange = climatology
                elif (feat.startswith("precip_") or feat.startswith("rainy_") or
                      feat.startswith("max_precip") or feat.startswith("days_since") or
                      feat.startswith("thunder") or feat.startswith("fog") or
                      feat.startswith("mean_temp") or feat.startswith("temp_") or
                      feat.startswith("mean_wind") or feat.startswith("mean_gust") or
                      feat.startswith("mean_sun") or feat.startswith("sunshine") or
                      feat.startswith("mean_precip_h") or feat.startswith("rain_flag") or
                      feat.startswith("haze") or feat.startswith("mean_dtr")):
                    colors.append("#4CAF50")  # green = recent conditions
                else:
                    colors.append("#9C27B0")  # purple = calendar/other
            ax.barh(imp_df["feature"], imp_df["gain"], color=colors)
            ax.set_xlabel("Gain")
            ax.set_title(f"XGBoostLSS {horizon}d: Top 25 Features by Gain")
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#2196F3", label="Forecast"),
                Patch(facecolor="#FF9800", label="Climatology"),
                Patch(facecolor="#4CAF50", label="Recent Conditions"),
                Patch(facecolor="#9C27B0", label="Calendar/Cross"),
            ]
            ax.legend(handles=legend_elements, loc="lower right")
            fig.tight_layout()
            fig.savefig(VIS_DIR / f"xgboostlss_{horizon}d_importance.png", dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"  Warning: Could not plot feature importance: {e}")

    # --- 3b. SHAP analysis (using XGBoost native pred_contribs) ---
    try:
        print("    Computing SHAP values...")
        X_test = test_df[feature_cols].values.astype(np.float32)
        n_features = len(feature_cols)

        # Use XGBoost's native SHAP via pred_contribs for multi-output compatibility
        contribs = booster.predict(dtest, pred_contribs=True)
        print(f"    pred_contribs shape: {contribs.shape}")

        if contribs.ndim == 3:
            # Multi-output: determine axis layout using known n_features
            if contribs.shape[1] == n_features + 1:
                # (n_samples, n_features+1, n_outputs)
                shap_conc = contribs[:, :n_features, 0]
                shap_rate = contribs[:, :n_features, 1] if contribs.shape[2] > 1 else None
            elif contribs.shape[2] == n_features + 1:
                # (n_samples, n_outputs, n_features+1)
                shap_conc = contribs[:, 0, :n_features]
                shap_rate = contribs[:, 1, :n_features] if contribs.shape[1] > 1 else None
            else:
                raise ValueError(
                    f"Unexpected contribs shape {contribs.shape} with {n_features} features"
                )
        elif contribs.ndim == 2:
            shap_conc = contribs[:, :n_features]
            shap_rate = None
        else:
            raise ValueError(f"Unexpected contribs ndim={contribs.ndim}")

        print(f"    Concentration SHAP shape: {shap_conc.shape}")

        # Create SHAP Explanation object for proper plotting
        explanation_conc = shap.Explanation(
            values=shap_conc,
            data=X_test,
            feature_names=feature_cols,
        )

        # Beeswarm plot (concentration/shape parameter)
        fig = plt.figure(figsize=(12, 10))
        shap.plots.beeswarm(explanation_conc, max_display=25, show=False)
        plt.title(f"XGBoostLSS {horizon}d: SHAP Beeswarm (concentration)")
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"xgboostlss_{horizon}d_shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        # Bar plot (mean |SHAP|)
        fig = plt.figure(figsize=(12, 8))
        shap.plots.bar(explanation_conc, max_display=25, show=False)
        plt.title(f"XGBoostLSS {horizon}d: Mean |SHAP| (concentration)")
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"xgboostlss_{horizon}d_shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        # Dependence plots for top 4 features
        mean_abs_shap = np.mean(np.abs(shap_conc), axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:4]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for idx, ax in zip(top_indices, axes.flat):
            shap.plots.scatter(explanation_conc[:, idx], ax=ax, show=False)
            ax.set_title(feature_cols[idx])
        fig.suptitle(f"XGBoostLSS {horizon}d: SHAP Dependence (Top 4)", fontsize=14)
        fig.tight_layout()
        fig.savefig(VIS_DIR / f"xgboostlss_{horizon}d_shap_dependence.png", dpi=150)
        plt.close("all")

        # Rate parameter SHAP (if multi-output)
        if shap_rate is not None:
            explanation_rate = shap.Explanation(
                values=shap_rate,
                data=X_test,
                feature_names=feature_cols,
            )
            fig = plt.figure(figsize=(12, 10))
            shap.plots.beeswarm(explanation_rate, max_display=25, show=False)
            plt.title(f"XGBoostLSS {horizon}d: SHAP Beeswarm (rate)")
            plt.tight_layout()
            plt.savefig(VIS_DIR / f"xgboostlss_{horizon}d_shap_rate.png", dpi=150, bbox_inches="tight")
            plt.close("all")

        print("    SHAP plots saved.")
    except Exception as e:
        print(f"  Warning: Could not compute SHAP values: {e}")
        import traceback
        traceback.print_exc()

    # --- 4. Calibration: PIT histogram ---
    scale = 1.0 / rate
    pit = sp_gamma.cdf(y, a=concentration, scale=scale)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pit, bins=20, density=True, alpha=0.7, edgecolor="black")
    ax.axhline(1.0, color="red", linestyle="--", label="Ideal (uniform)")
    ax.set_xlabel("PIT Value")
    ax.set_ylabel("Density")
    ax.set_title(f"XGBoostLSS {horizon}d: PIT Histogram (Calibration)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIS_DIR / f"xgboostlss_{horizon}d_pit.png", dpi=150)
    plt.close(fig)

    # --- 5. Reliability diagram ---
    nominal_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    actual_covs = []
    for level in nominal_levels:
        lo_q = (1 - level) / 2
        hi_q = 1 - lo_q
        lo = sp_gamma.ppf(lo_q, a=concentration, scale=scale)
        hi = sp_gamma.ppf(hi_q, a=concentration, scale=scale)
        actual_covs.append(float(np.mean((y >= lo) & (y <= hi))))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Perfect")
    ax.plot(nominal_levels, actual_covs, "bo-", markersize=5, label="Model")
    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Actual Coverage")
    ax.set_title(f"XGBoostLSS {horizon}d: Reliability Diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(VIS_DIR / f"xgboostlss_{horizon}d_reliability.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoostLSS precipitation models")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Train only this horizon (3/7/14/21/28/35)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run with fewer trials and rounds")
    args = parser.parse_args()

    horizons = [args.horizon] if args.horizon else HORIZONS

    if not DATA_FILE.exists():
        print(f"ERROR: Training dataset not found at {DATA_FILE}")
        print("Run: python scripts/data_processing/build_training_dataset.py")
        sys.exit(1)

    all_metrics = []

    for horizon in horizons:
        print(f"\n{'=' * 60}")
        print(f"Training {horizon}-day Horizon Model")
        print(f"{'=' * 60}")

        # Load and split
        print("\nLoading data...")
        train_df, val_df, test_df = load_data(horizon)

        # Get feature columns (drop columns that are all NaN for this horizon)
        feature_cols = get_feature_cols(train_df)
        # Drop columns that have too many NaNs (sub-window features for smaller horizons)
        nan_pcts = train_df[feature_cols].isnull().mean()
        drop_cols = nan_pcts[nan_pcts > 0.5].index.tolist()
        if drop_cols:
            print(f"  Dropping {len(drop_cols)} high-NaN columns: {drop_cols}")
            feature_cols = [c for c in feature_cols if c not in drop_cols]

        # Fill remaining NaNs with 0 (sub-window features that are partially available)
        for df in [train_df, val_df, test_df]:
            df[feature_cols] = df[feature_cols].fillna(0)

        print(f"  Features: {len(feature_cols)}")

        # Train
        print("\nTraining...")
        xgblss, opt_params = train_model(train_df, val_df, feature_cols, horizon, args.quick)

        # Evaluate on val and test (skip train — slow and less useful)
        print("\nEvaluating...")
        for split_name, split_df in [("val", val_df), ("test", test_df)]:
            metrics, pred_params, pred_q = evaluate_model(
                xgblss, split_df, feature_cols, split_name
            )
            metrics["horizon_days"] = horizon
            all_metrics.append(metrics)

            print(f"\n  {split_name.upper()} metrics:")
            print(f"    CRPS:        {metrics['mean_crps']:.2f}")
            print(f"    NLL:         {metrics['mean_nll']:.2f}")
            print(f"    MAE:         {metrics['mae']:.1f} mm")
            print(f"    Bias:        {metrics['bias']:+.1f} mm")
            print(f"    Coverage 90: {metrics['coverage_90']:.1%}")
            print(f"    Coverage 50: {metrics['coverage_50']:.1%}")
            print(f"    Mean obs:    {metrics['mean_observed']:.1f} mm")
            print(f"    Mean pred:   {metrics['mean_predicted']:.1f} mm")
            # Reliability at multiple levels
            print(f"    Reliability: ", end="")
            for lev in [50, 60, 70, 80, 90, 95]:
                key = f"cov_{lev}"
                if key in metrics:
                    print(f"{lev}%={metrics[key]:.1%} ", end="")
            print()

            # Generate plots for test set
            if split_name == "test":
                print("\n  Generating diagnostic plots...")
                plot_diagnostics(xgblss, split_df, pred_params, pred_q, feature_cols, horizon)

        # Save model
        model_path = MODELS_DIR / f"xgboostlss_{horizon}d.json"
        xgblss.save_model(str(model_path))
        print(f"\n  Model saved to {model_path}")

        # Save optimal params
        params_path = MODELS_DIR / f"xgboostlss_{horizon}d_params.json"
        with open(params_path, "w") as f:
            json.dump(opt_params, f, indent=2, default=str)

    # Save all metrics
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = MODELS_DIR / "all_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n{'=' * 60}")
        print(f"All metrics saved to {metrics_path}")
        print(f"{'=' * 60}")
        print(metrics_df.to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
