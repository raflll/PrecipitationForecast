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
from xgboostlss.distributions.Gamma import Gamma
from xgboostlss.model import XGBoostLSS

# Add src/ to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from evaluation import compute_metrics, plot_all_diagnostics, plot_calibration_comparison
from calibration import find_optimal_scale_factor
from viz_theme import DPI

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


def predict_params(xgblss: XGBoostLSS, df: pd.DataFrame, feature_cols: list[str]):
    """Get Gamma distribution parameters from model predictions."""
    dmat = prepare_dmatrix(df, feature_cols)
    pred_params = xgblss.predict(dmat, pred_type="parameters")
    return pred_params["concentration"].values, pred_params["rate"].values, dmat


def plot_shap(
    xgblss: XGBoostLSS,
    dtest: xgb.DMatrix,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
):
    """Generate SHAP analysis plots (XGBoost-specific)."""
    try:
        booster = xgblss.booster
        print("    Computing SHAP values...")
        X_test = test_df[feature_cols].values.astype(np.float32)
        n_features = len(feature_cols)

        contribs = booster.predict(dtest, pred_contribs=True)
        print(f"    pred_contribs shape: {contribs.shape}")

        if contribs.ndim == 3:
            if contribs.shape[1] == n_features + 1:
                shap_conc = contribs[:, :n_features, 0]
                shap_rate = contribs[:, :n_features, 1] if contribs.shape[2] > 1 else None
            elif contribs.shape[2] == n_features + 1:
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

        explanation_conc = shap.Explanation(
            values=shap_conc, data=X_test, feature_names=feature_cols,
        )

        fig = plt.figure(figsize=(12, 10))
        shap.plots.beeswarm(explanation_conc, max_display=25, show=False)
        plt.title(f"{horizon}-Day Forecast: SHAP Beeswarm (concentration)")
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"xgboostlss_{horizon}d_shap_beeswarm.png", dpi=DPI, bbox_inches="tight")
        plt.close("all")

        fig = plt.figure(figsize=(12, 8))
        shap.plots.bar(explanation_conc, max_display=25, show=False)
        plt.title(f"{horizon}-Day Forecast: Mean |SHAP| (concentration)")
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"xgboostlss_{horizon}d_shap_bar.png", dpi=DPI, bbox_inches="tight")
        plt.close("all")

        mean_abs_shap = np.mean(np.abs(shap_conc), axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:4]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for idx, ax in zip(top_indices, axes.flat):
            shap.plots.scatter(explanation_conc[:, idx], ax=ax, show=False)
            ax.set_title(feature_cols[idx])
        fig.suptitle(f"{horizon}-Day Forecast: SHAP Dependence (Top 4)", fontsize=14)
        fig.tight_layout()
        fig.savefig(VIS_DIR / f"xgboostlss_{horizon}d_shap_dependence.png", dpi=DPI)
        plt.close("all")

        if shap_rate is not None:
            explanation_rate = shap.Explanation(
                values=shap_rate, data=X_test, feature_names=feature_cols,
            )
            fig = plt.figure(figsize=(12, 10))
            shap.plots.beeswarm(explanation_rate, max_display=25, show=False)
            plt.title(f"{horizon}-Day Forecast: SHAP Beeswarm (rate)")
            plt.tight_layout()
            plt.savefig(VIS_DIR / f"xgboostlss_{horizon}d_shap_rate.png", dpi=DPI, bbox_inches="tight")
            plt.close("all")

        print("    SHAP plots saved.")
    except Exception as e:
        print(f"  Warning: Could not compute SHAP values: {e}")
        import traceback
        traceback.print_exc()


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
        nan_pcts = train_df[feature_cols].isnull().mean()
        drop_cols = nan_pcts[nan_pcts > 0.5].index.tolist()
        if drop_cols:
            print(f"  Dropping {len(drop_cols)} high-NaN columns: {drop_cols}")
            feature_cols = [c for c in feature_cols if c not in drop_cols]

        for df in [train_df, val_df, test_df]:
            df[feature_cols] = df[feature_cols].fillna(0)

        print(f"  Features: {len(feature_cols)}")

        # Train
        print("\nTraining...")
        xgblss, opt_params = train_model(train_df, val_df, feature_cols, horizon, args.quick)

        # Predict on val and test
        print("\nEvaluating...")
        val_conc, val_rate, _ = predict_params(xgblss, val_df, feature_cols)
        test_conc, test_rate, dtest = predict_params(xgblss, test_df, feature_cols)

        # Calibrate: find optimal scale factor on validation set
        y_val = val_df["ground_truth_precip"].values
        y_test = test_df["ground_truth_precip"].values

        print("\n  Calibrating (variance scaling on validation set)...")
        scale_factor = find_optimal_scale_factor(y_val, val_conc, val_rate)
        print(f"    Optimal scale factor: {scale_factor:.2f}")

        # Evaluate both splits with calibration
        for split_name, conc, rate, y, split_df in [
            ("val", val_conc, val_rate, y_val, val_df),
            ("test", test_conc, test_rate, y_test, test_df),
        ]:
            metrics, pred_q = compute_metrics(y, conc, rate, split_name, scale_factor)
            metrics["horizon_days"] = horizon
            metrics["scale_factor"] = scale_factor
            all_metrics.append(metrics)

            print(f"\n  {split_name.upper()} metrics (scale_factor={scale_factor:.2f}):")
            print(f"    CRPS:        {metrics['mean_crps']:.2f}")
            print(f"    NLL:         {metrics['mean_nll']:.2f}")
            print(f"    MAE:         {metrics['mae']:.1f} mm")
            print(f"    Bias:        {metrics['bias']:+.1f} mm")
            print(f"    Coverage 90: {metrics['coverage_90']:.1%}")
            print(f"    Coverage 50: {metrics['coverage_50']:.1%}")
            print(f"    Mean obs:    {metrics['mean_observed']:.1f} mm")
            print(f"    Mean pred:   {metrics['mean_predicted']:.1f} mm")
            print(f"    Reliability: ", end="")
            for lev in [50, 60, 70, 80, 90, 95]:
                key = f"cov_{lev}"
                if key in metrics:
                    print(f"{lev}%={metrics[key]:.1%} ", end="")
            print()

            # Generate plots for test set
            if split_name == "test":
                print("\n  Generating diagnostic plots...")

                # Feature importance dict
                try:
                    booster = xgblss.booster
                    raw_imp = booster.get_score(importance_type="gain")
                    if raw_imp:
                        sample_key = next(iter(raw_imp))
                        if sample_key.startswith("f") and sample_key[1:].isdigit():
                            feat_imp = {
                                feature_cols[int(k[1:])]: v
                                for k, v in raw_imp.items()
                                if k.startswith("f") and k[1:].isdigit() and int(k[1:]) < len(feature_cols)
                            }
                        else:
                            feat_imp = dict(raw_imp)
                    else:
                        feat_imp = None
                except Exception:
                    feat_imp = None

                dates = split_df["init_date"].values if "init_date" in split_df.columns else None

                plot_all_diagnostics(
                    y=y,
                    concentration=conc,
                    rate=rate,
                    pred_q=pred_q,
                    dates=dates,
                    feature_names=feature_cols,
                    feature_importance=feat_imp,
                    horizon=horizon,
                    model_name="XGBoostLSS",
                    vis_dir=VIS_DIR,
                    scale_factor=scale_factor,
                )

                # Calibration comparison (before vs after)
                plot_calibration_comparison(
                    y=y,
                    concentration=conc,
                    rate=rate,
                    scale_factor_before=1.0,
                    scale_factor_after=scale_factor,
                    horizon=horizon,
                    model_name="XGBoostLSS",
                    vis_dir=VIS_DIR,
                )

                # SHAP (XGBoost-specific)
                plot_shap(xgblss, dtest, split_df, feature_cols, horizon)

        # Save model
        model_path = MODELS_DIR / f"xgboostlss_{horizon}d.json"
        xgblss.save_model(str(model_path))
        print(f"\n  Model saved to {model_path}")

        # Save optimal params
        params_path = MODELS_DIR / f"xgboostlss_{horizon}d_params.json"
        with open(params_path, "w") as f:
            json.dump({**opt_params, "scale_factor": scale_factor}, f, indent=2, default=str)

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
