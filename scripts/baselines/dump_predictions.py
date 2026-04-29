"""
Dump per-sample Gamma predictions for all six models at one or more horizons,
so the cross-model comparison plot can show actual time-series predictions
against observations rather than only metric aggregates.

Outputs: models/predictions_<horizon>d.csv with columns
  init_date, model, observed, mean, q05, q25, q50, q75, q95
covering both the validation (2024) and test (2025) splits.

Baselines (climatology, raw_gefs, emos_gamma) are computed in pure Python
following the same recipes as run_sota_baselines.py. The three ML models
(xgboost_gamma R, gamlss R, xgboostlss Python) are loaded from disk and
predicted via their respective libraries.
"""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma as sp_gamma

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "baselines"))

from run_sota_baselines import (  # noqa: E402
    baseline_climatology,
    baseline_raw_gefs,
    fit_emos,
    _emos_predict,
    _emos_features,
    EPS,
)

EPS = 0.1
TRAIN_END_YEAR = 2023
VAL_YEAR = 2024
TEST_YEAR = 2025

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def _qframe(shape, scale):
    out = {}
    for q in QUANTILES:
        out[f"q{int(q * 100):02d}"] = sp_gamma.ppf(q, a=shape, scale=scale)
    out["mean"] = shape * scale
    return out


def predict_baselines(df: pd.DataFrame, train_df: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}

    sh, sc = baseline_climatology(df)
    out["climatology"] = _qframe(sh, sc)

    sh, sc = baseline_raw_gefs(df)
    out["raw_gefs"] = _qframe(sh, sc)

    theta = fit_emos(train_df)
    sh, sc = _emos_predict(theta, _emos_features(df))
    out["emos_gamma"] = _qframe(sh, sc)

    return out


def predict_xgboostlss(df: pd.DataFrame, horizon: int) -> dict | None:
    """Predict from saved Python xgboostlss artifacts. Only 14d is on disk."""
    model_path = REPO / "models" / f"xgboostlss_{horizon}d.json"
    params_path = REPO / "models" / f"xgboostlss_{horizon}d_params.json"
    if not model_path.exists() or not params_path.exists():
        print(f"  xgboostlss: no saved artifacts at horizon={horizon}, skipping")
        return None

    try:
        from xgboostlss.model import XGBoostLSS
        from xgboostlss.distributions.Gamma import Gamma
    except ImportError:
        print("  xgboostlss: package not available in this env, skipping")
        return None

    with open(params_path) as f:
        params = json.load(f)

    feat_cols = params.get("feature_cols")
    if feat_cols is None:
        print("  xgboostlss: feature_cols missing in params JSON, skipping")
        return None

    X = df[feat_cols].copy()
    for c in X.columns:
        med = X[c].median()
        X[c] = X[c].fillna(med if pd.notna(med) else 0.0)

    try:
        import xgboost as xgb
        dist = Gamma(stabilization="MAD", response_fn="exp", loss_fn="nll")
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        # Use distribution forward to convert raw predictions to params
        dmat = xgb.DMatrix(X.to_numpy(), feature_names=list(X.columns))
        raw = booster.predict(dmat, output_margin=True)
        # Gamma in xgboostlss has 2 params: concentration (shape), rate
        # raw is shape (n, 2)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 2)
        concentration = np.exp(raw[:, 0])
        rate = np.exp(raw[:, 1])
        scale = 1.0 / rate
        return _qframe(concentration, scale)
    except Exception as e:
        print(f"  xgboostlss: predict failed ({e}), skipping")
        return None


def predict_r_models_via_subprocess(horizon: int, out_dir: Path) -> dict[str, dict]:
    """Invoke an Rscript that loads xgboost_gamma_*.json and gamlss_*.rds,
    predicts on val+test, and writes predictions_<model>_<h>d.csv files."""
    script = REPO / "scripts" / "baselines" / "dump_r_predictions.R"
    csv_xg = out_dir / f"_r_xgboost_gamma_{horizon}d.csv"
    csv_gam = out_dir / f"_r_gamlss_{horizon}d.csv"
    csv_xg.parent.mkdir(parents=True, exist_ok=True)
    if csv_xg.exists():
        csv_xg.unlink()
    if csv_gam.exists():
        csv_gam.unlink()

    cmd = ["Rscript", str(script), "--horizon", str(horizon),
           "--xgboost-out", str(csv_xg), "--gamlss-out", str(csv_gam)]
    print(f"  R: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print("  R stderr:", result.stderr[-1000:])
        else:
            print("  R OK")
    except Exception as e:
        print(f"  R: failed ({e})")
        return {}

    out = {}
    for label, path in [("xgboost_gamma", csv_xg), ("gamlss", csv_gam)]:
        if path.exists() and path.stat().st_size > 0:
            d = pd.read_csv(path)
            d["init_date"] = pd.to_datetime(d["init_date"])
            out[label] = d
        else:
            print(f"  R: predictions for {label} not produced")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=14)
    args = parser.parse_args()
    h = args.horizon

    parquet = REPO / "data" / "training" / "training_dataset.parquet"
    df = pd.read_parquet(parquet)
    df["init_date"] = pd.to_datetime(df["init_date"])
    df["year"] = df["init_date"].dt.year

    sub = df[df["horizon_days"] == h].copy()
    train = sub[sub["year"] <= TRAIN_END_YEAR].reset_index(drop=True)
    eval_df = sub[sub["year"].isin([VAL_YEAR, TEST_YEAR])].reset_index(drop=True)

    print(f"horizon={h}d  train={len(train)}  val+test={len(eval_df)}")

    # Baseline predictions (Python)
    print("Computing baseline predictions...")
    base = predict_baselines(eval_df, train)

    rows = []
    for model_name, qf in base.items():
        for i in range(len(eval_df)):
            rows.append({
                "init_date": eval_df["init_date"].iloc[i],
                "split": "val" if eval_df["year"].iloc[i] == VAL_YEAR else "test",
                "horizon_days": h,
                "model": model_name,
                "observed": float(eval_df["ground_truth_precip"].iloc[i]),
                "mean": float(qf["mean"][i]),
                "q05": float(qf["q05"][i]),
                "q25": float(qf["q25"][i]),
                "q50": float(qf["q50"][i]),
                "q75": float(qf["q75"][i]),
                "q95": float(qf["q95"][i]),
            })

    # XGBoostLSS (Python) — best-effort, only if package present and h=14
    print("Computing xgboostlss predictions...")
    xgblss = predict_xgboostlss(eval_df, h)
    if xgblss is not None:
        for i in range(len(eval_df)):
            rows.append({
                "init_date": eval_df["init_date"].iloc[i],
                "split": "val" if eval_df["year"].iloc[i] == VAL_YEAR else "test",
                "horizon_days": h,
                "model": "xgboostlss",
                "observed": float(eval_df["ground_truth_precip"].iloc[i]),
                "mean": float(xgblss["mean"][i]),
                "q05": float(xgblss["q05"][i]),
                "q25": float(xgblss["q25"][i]),
                "q50": float(xgblss["q50"][i]),
                "q75": float(xgblss["q75"][i]),
                "q95": float(xgblss["q95"][i]),
            })

    # ML models (R)
    print("Computing R model predictions...")
    r_preds = predict_r_models_via_subprocess(h, REPO / "models")
    for model_name, frame in r_preds.items():
        for _, r in frame.iterrows():
            rows.append({
                "init_date": r["init_date"],
                "split": r["split"],
                "horizon_days": h,
                "model": model_name,
                "observed": float(r["observed"]),
                "mean": float(r["mean"]),
                "q05": float(r["q05"]),
                "q25": float(r["q25"]),
                "q50": float(r["q50"]),
                "q75": float(r["q75"]),
                "q95": float(r["q95"]),
            })

    out = pd.DataFrame(rows)
    # Normalise mixed tz-naive (Python) and tz-aware (R) timestamps to plain dates
    out["init_date"] = pd.to_datetime(out["init_date"], utc=True).dt.tz_localize(None).dt.normalize()
    out_path = REPO / "models" / f"predictions_{h}d.csv"
    out.sort_values(["model", "init_date"]).to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out)} rows)")
    print(out.groupby("model").size())


if __name__ == "__main__":
    main()
