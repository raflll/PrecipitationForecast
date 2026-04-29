"""
State-of-the-art baselines for probabilistic precipitation forecasting.

Three baselines are evaluated against the same temporal split (train through
2023, validate on 2024, test on 2025) used by the XGBoost-family models:

  1. CLIMATOLOGY        -- Gamma distribution fit to historical observations
                           in a +/- 15-day day-of-year window. Uses
                           `clim_gamma_shape` and `clim_gamma_scale` features
                           that are already engineered into the parquet.
                           Represents the no-NWP-information lower bound.

  2. RAW_GEFS           -- Method-of-moments Gamma dressing of the raw GEFS
                           ensemble: shape = (mean/spread)^2, scale = spread^2/mean.
                           Represents what you get directly from the NWP system
                           without any statistical post-processing.

  3. EMOS_GAMMA         -- Ensemble Model Output Statistics (Gneiting et al. 2005)
                           with a Gamma response. Non-homogeneous regression:
                           log(mu)    = a + b * log(1 + cum_apcp_mean)
                           log(sigma) = c + d * log(1 + cum_apcp_spread_rss) + e * log(1 + clim_mean)
                           Fit by minimum-CRPS estimation on the training set,
                           one model per horizon. This is the standard
                           parametric post-processing baseline in the NWP
                           post-processing literature.

Each baseline produces a Gamma (shape, scale) per (init_date, horizon) and is
scored with the same metrics module used by the XGBoost models (CRPS, NLL, MAE,
bias, prediction-interval coverage at 50/60/70/80/90/95%).

Output: `models/baselines_metrics.csv` with one row per (model, horizon, split).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gamma as sp_gamma
from scipy.special import beta as beta_fn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

EPS = 0.1
HORIZONS = [3, 7, 14, 21, 28, 35]
TRAIN_END_YEAR = 2023
VAL_YEAR = 2024
TEST_YEAR = 2025


# ---------------------------------------------------------------------------
# Metrics (mirrors src/evaluation.py to keep this script self-contained)
# ---------------------------------------------------------------------------

def crps_gamma(y: np.ndarray, shape: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Vectorised analytical CRPS for the Gamma distribution."""
    a, s = shape, scale
    F_y = sp_gamma.cdf(y, a=a, scale=s)
    F_y_a1 = sp_gamma.cdf(y, a=a + 1, scale=s)
    B = np.array([beta_fn(0.5, ai) for ai in np.atleast_1d(a)]).reshape(np.shape(a))
    return y * (2 * F_y - 1) - a * s * (2 * F_y_a1 - 1 + 1.0 / (a * B))


def coverage(y, shape, scale, level):
    lo = sp_gamma.ppf((1 - level) / 2, a=shape, scale=scale)
    hi = sp_gamma.ppf(1 - (1 - level) / 2, a=shape, scale=scale)
    return float(np.mean((y >= lo) & (y <= hi)))


def compute_metrics(y, shape, scale, model: str, split: str, horizon: int) -> dict:
    pred_mean = shape * scale
    nll = np.clip(-sp_gamma.logpdf(y, a=shape, scale=scale), -100, 100)
    crps = crps_gamma(y, shape, scale)
    median_pred = sp_gamma.ppf(0.5, a=shape, scale=scale)
    out = {
        "model": model,
        "split": split,
        "horizon_days": horizon,
        "n_samples": int(len(y)),
        "mean_crps": float(np.mean(crps)),
        "mean_nll": float(np.mean(nll)),
        "mae": float(np.mean(np.abs(pred_mean - y))),
        "bias": float(np.mean(pred_mean - y)),
        "median_ae": float(np.median(np.abs(median_pred - y))),
        "mean_observed": float(np.mean(y)),
        "mean_predicted": float(np.mean(pred_mean)),
    }
    for level in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        out[f"cov_{int(level*100)}"] = coverage(y, shape, scale, level)
    return out


# ---------------------------------------------------------------------------
# Method-of-moments Gamma helpers
# ---------------------------------------------------------------------------

def mom_gamma(mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Method of moments: shape = mean^2 / var, scale = var / mean."""
    mean = np.maximum(mean, EPS)
    std = np.maximum(std, EPS)
    var = std ** 2
    shape = mean ** 2 / var
    scale = var / mean
    # Regularise extreme tails to avoid degenerate Gammas
    shape = np.clip(shape, 0.1, 200.0)
    scale = np.clip(scale, 0.05, 1000.0)
    return shape, scale


# ---------------------------------------------------------------------------
# Baseline 1: Climatology
# ---------------------------------------------------------------------------

def baseline_climatology(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Gamma parameters from the doy-window climatology features."""
    shape = df["clim_gamma_shape"].to_numpy()
    scale = df["clim_gamma_scale"].to_numpy()
    shape = np.clip(np.where(np.isfinite(shape), shape, 1.0), 0.1, 200.0)
    scale = np.clip(np.where(np.isfinite(scale), scale, 1.0), 0.05, 1000.0)
    return shape, scale


# ---------------------------------------------------------------------------
# Baseline 2: Raw GEFS ensemble dressing
# ---------------------------------------------------------------------------

def baseline_raw_gefs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean = df["cum_apcp_mean"].to_numpy()
    spread = df["cum_apcp_spread_rss"].to_numpy()
    return mom_gamma(mean, spread)


# ---------------------------------------------------------------------------
# Baseline 3: EMOS-Gamma (non-homogeneous regression)
# ---------------------------------------------------------------------------

def _emos_predict(theta: np.ndarray, X: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    a, b, c, d, e = theta
    log_mean = a + b * X["log_apcp"]
    log_std = c + d * X["log_spread"] + e * X["log_clim"]
    mu = np.exp(log_mean)
    sigma = np.exp(log_std)
    return mom_gamma(mu, sigma)


def _emos_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "log_apcp": np.log1p(df["cum_apcp_mean"].to_numpy()),
        "log_spread": np.log1p(df["cum_apcp_spread_rss"].to_numpy()),
        "log_clim": np.log1p(df["clim_mean_precip"].to_numpy()),
    }


def fit_emos(train_df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Fit EMOS by minimum-CRPS estimation with multiple restarts (5 parameters)."""
    y = train_df["ground_truth_precip"].to_numpy()
    X = _emos_features(train_df)
    log_std_y = float(np.log(np.std(y) + EPS))

    def objective(theta):
        try:
            shape, scale = _emos_predict(theta, X)
            return float(np.mean(crps_gamma(y, shape, scale)))
        except Exception:
            return 1e10

    rng = np.random.default_rng(seed)
    starts = [
        np.array([0.0, 1.0, log_std_y, 0.5, 0.0]),
        np.array([0.5, 0.8, log_std_y - 0.5, 0.7, 0.1]),
        np.array([1.0, 0.5, log_std_y, 0.5, 0.5]),
        np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
    ]
    starts.extend(rng.normal(loc=[0, 1, log_std_y, 0.5, 0], scale=0.5, size=(8, 5)))

    best = None
    for theta0 in starts:
        res = minimize(objective, theta0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 3000})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def baseline_emos(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    theta = fit_emos(train_df)
    return _emos_predict(theta, _emos_features(eval_df)), theta


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parquet = REPO / "data" / "training" / "training_dataset.parquet"
    df = pd.read_parquet(parquet)
    df["init_date"] = pd.to_datetime(df["init_date"])
    df["year"] = df["init_date"].dt.year

    rows = []
    emos_thetas = {}

    for h in HORIZONS:
        sub = df[df["horizon_days"] == h].copy()
        train = sub[sub["year"] <= TRAIN_END_YEAR].reset_index(drop=True)
        val = sub[sub["year"] == VAL_YEAR].reset_index(drop=True)
        test = sub[sub["year"] == TEST_YEAR].reset_index(drop=True)

        print(f"horizon={h:2d}d  train={len(train):5d}  val={len(val):5d}  test={len(test):5d}")

        for split_name, split_df in [("val", val), ("test", test)]:
            if len(split_df) == 0:
                continue
            y = split_df["ground_truth_precip"].to_numpy()

            # 1. Climatology
            sh, sc = baseline_climatology(split_df)
            rows.append(compute_metrics(y, sh, sc, "climatology", split_name, h))

            # 2. Raw GEFS
            sh, sc = baseline_raw_gefs(split_df)
            rows.append(compute_metrics(y, sh, sc, "raw_gefs", split_name, h))

        # 3. EMOS-Gamma -- needs to fit on train
        theta = fit_emos(train)
        emos_thetas[h] = theta.tolist()
        print(f"           emos theta = {np.round(theta, 3).tolist()}")
        for split_name, split_df in [("val", val), ("test", test)]:
            if len(split_df) == 0:
                continue
            y = split_df["ground_truth_precip"].to_numpy()
            shape, scale = _emos_predict(theta, _emos_features(split_df))
            rows.append(compute_metrics(y, shape, scale, "emos_gamma", split_name, h))

    out = pd.DataFrame(rows).sort_values(["model", "horizon_days", "split"]).reset_index(drop=True)
    out_path = REPO / "models" / "baselines_metrics.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out)} rows).")

    # Save EMOS coefficients separately for reproducibility
    import json
    coef_path = REPO / "models" / "emos_gamma_params.json"
    with open(coef_path, "w") as f:
        json.dump(
            {
                "param_order": ["a", "b", "c", "d", "e"],
                "model": "log(mu) = a + b*log(1+apcp_mean); log(sigma) = c + d*log(1+spread_rss) + e*log(1+clim_mean)",
                "objective": "min-CRPS on training set",
                "coefficients_by_horizon": emos_thetas,
            },
            f,
            indent=2,
        )
    print(f"Wrote {coef_path}.")


if __name__ == "__main__":
    main()
