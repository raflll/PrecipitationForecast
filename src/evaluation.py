"""
Shared evaluation metrics and diagnostic plotting for all precipitation models.

Used by train_xgboostlss.py, train_lstm.py, and any future model scripts.
All plots use the unified theme from viz_theme.py.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gamma as sp_gamma
from scipy.special import beta as beta_fn

import sys
sys.path.insert(0, str(Path(__file__).parent))

from viz_theme import (
    COLORS, CHART_SIZES, FONT,
    apply_theme, set_title, save_figure, format_date_axis, feature_color,
)
from calibration import compute_pit, pit_ks_statistic


# ---------------------------------------------------------------------------
# CRPS analytical formula for Gamma distribution
# ---------------------------------------------------------------------------

def crps_gamma_analytical(y: float, concentration: float, scale: float) -> float:
    """Analytical CRPS for Gamma(concentration, scale) evaluated at observation y.

    CRPS = y*(2F(y) - 1) - a*s*(2*F_{a+1}(y) - 1) - a*s / B(0.5, a)
    """
    a, s = concentration, scale
    F_y = sp_gamma.cdf(y, a=a, scale=s)
    F_y_a1 = sp_gamma.cdf(y, a=a + 1, scale=s)
    return y * (2 * F_y - 1) - a * s * (2 * F_y_a1 - 1 + 1.0 / (a * beta_fn(0.5, a)))


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y: np.ndarray,
    concentration: np.ndarray,
    rate: np.ndarray,
    split_name: str,
    scale_factor: float = 1.0,
) -> tuple[dict, pd.DataFrame]:
    """Compute all evaluation metrics from Gamma params.

    Returns (metrics_dict, pred_quantiles_df).
    """
    scale = (1.0 / rate) * scale_factor

    pred_mean = concentration * scale  # = concentration / (rate / scale_factor)
    quantile_levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    pred_q = pd.DataFrame({
        f"quant_{q}": sp_gamma.ppf(q, a=concentration, scale=scale)
        for q in quantile_levels
    })

    # CRPS
    crps_scores = np.array([
        crps_gamma_analytical(y[i], concentration[i], scale[i])
        for i in range(len(y))
    ])

    # NLL
    nll = -sp_gamma.logpdf(y, a=concentration, scale=scale)
    nll = np.clip(nll, -100, 100)

    # Coverage at standard levels
    coverage_90 = float(np.mean(
        (y >= pred_q["quant_0.05"].values) & (y <= pred_q["quant_0.95"].values)
    ))
    coverage_50 = float(np.mean(
        (y >= pred_q["quant_0.25"].values) & (y <= pred_q["quant_0.75"].values)
    ))

    # Reliability at multiple nominal levels
    reliability = {}
    for level in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        lo_q = (1 - level) / 2
        hi_q = 1 - lo_q
        lo = sp_gamma.ppf(lo_q, a=concentration, scale=scale)
        hi = sp_gamma.ppf(hi_q, a=concentration, scale=scale)
        reliability[f"cov_{int(level * 100)}"] = float(np.mean((y >= lo) & (y <= hi)))

    metrics = {
        "split": split_name,
        "n_samples": len(y),
        "mean_crps": float(np.mean(crps_scores)),
        "mean_nll": float(np.mean(nll)),
        "mae": float(np.mean(np.abs(pred_mean - y))),
        "bias": float(np.mean(pred_mean - y)),
        "coverage_90": coverage_90,
        "coverage_50": coverage_50,
        "median_ae": float(np.median(np.abs(pred_q["quant_0.5"].values - y))),
        "mean_observed": float(np.mean(y)),
        "mean_predicted": float(np.mean(pred_mean)),
        **reliability,
    }

    return metrics, pred_q


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_all_diagnostics(
    y: np.ndarray,
    concentration: np.ndarray,
    rate: np.ndarray,
    pred_q: pd.DataFrame,
    dates: np.ndarray | None,
    feature_names: list[str] | None,
    feature_importance: dict | None,
    horizon: int,
    model_name: str,
    vis_dir: Path,
    scale_factor: float = 1.0,
):
    """Generate all diagnostic plots for a trained model.

    Parameters
    ----------
    y : observed precipitation values
    concentration, rate : predicted Gamma parameters
    pred_q : DataFrame with quant_0.05 .. quant_0.95 columns
    dates : init_date array for x-axis (or None to use integer index)
    feature_names : list of feature column names (for importance)
    feature_importance : dict of {feature: gain} (or None to skip)
    horizon : forecast horizon in days
    model_name : e.g. "XGBoostLSS", "LSTM"
    vis_dir : output directory for PNGs
    scale_factor : calibration scale factor applied to rate
    """
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_name.lower().replace(' ', '_')}_{horizon}d"
    pred_mean = concentration / rate  # uncalibrated mean for scatter

    _plot_pred_vs_obs(y, pred_mean, horizon, model_name, vis_dir, prefix)
    _plot_intervals(y, pred_q, dates, horizon, model_name, vis_dir, prefix)
    _plot_pit(y, concentration, rate, scale_factor, horizon, model_name, vis_dir, prefix)
    _plot_reliability(y, concentration, rate, scale_factor, horizon, model_name, vis_dir, prefix)
    _plot_residuals(y, pred_mean, horizon, model_name, vis_dir, prefix)
    _plot_monthly(y, pred_mean, dates, horizon, model_name, vis_dir, prefix)
    _plot_qq_pit(y, concentration, rate, scale_factor, horizon, model_name, vis_dir, prefix)

    if feature_importance:
        _plot_importance(feature_importance, horizon, model_name, vis_dir, prefix)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def _plot_pred_vs_obs(y, pred_mean, horizon, model_name, vis_dir, prefix):
    """Scatter of predicted mean vs observed with LOESS fit."""
    fig, ax = plt.subplots(figsize=CHART_SIZES["pred_vs_obs"])
    apply_theme(ax)

    ax.scatter(y, pred_mean, alpha=0.35, s=15, color=COLORS["primary"], zorder=2)
    max_val = max(y.max(), pred_mean.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "--", color=COLORS["red"],
            linewidth=1.2, label="Perfect prediction", zorder=3)

    # LOESS fit
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sorted_idx = np.argsort(y)
        smoothed = lowess(pred_mean[sorted_idx], y[sorted_idx], frac=0.4)
        ax.plot(smoothed[:, 0], smoothed[:, 1], "-", color=COLORS["secondary"],
                linewidth=2, label="LOESS fit", zorder=4)
    except ImportError:
        pass

    # Annotation box
    corr = np.corrcoef(y, pred_mean)[0, 1]
    mae = np.mean(np.abs(pred_mean - y))
    bias = np.mean(pred_mean - y)
    txt = f"r = {corr:.3f}\nMAE = {mae:.1f} mm\nBias = {bias:+.1f} mm"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=FONT["legend_size"],
            va="top", ha="left", bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

    ax.set_xlabel("Observed Precipitation (mm)")
    ax.set_ylabel("Predicted Mean (mm)")
    set_title(ax, f"{horizon}-Day Forecast: Predicted vs Observed",
              f"{model_name} | Red dashed = perfect | Orange = LOESS fit")
    ax.legend(loc="lower right", fontsize=FONT["legend_size"])
    save_figure(fig, vis_dir / f"{prefix}_pred_vs_obs.png")


def _plot_intervals(y, pred_q, dates, horizon, model_name, vis_dir, prefix):
    """Full-year prediction interval time series."""
    fig, ax = plt.subplots(figsize=CHART_SIZES["intervals"])
    apply_theme(ax)

    n = len(y)
    if dates is not None and len(dates) == n:
        x = pd.to_datetime(dates)
    else:
        x = np.arange(n)

    # 90% PI ribbon
    ax.fill_between(x, pred_q["quant_0.05"].values, pred_q["quant_0.95"].values,
                     alpha=0.25, color=COLORS["primary"], label="90% PI")
    # 50% PI ribbon
    ax.fill_between(x, pred_q["quant_0.25"].values, pred_q["quant_0.75"].values,
                     alpha=0.35, color=COLORS["primary"], label="50% PI")
    # Median line
    ax.plot(x, pred_q["quant_0.5"].values, "-", color=COLORS["primary"],
            linewidth=1.2, label="Median", zorder=3)

    # Observations: grey inside PI, red X outside 90% PI
    inside = (y >= pred_q["quant_0.05"].values) & (y <= pred_q["quant_0.95"].values)
    ax.scatter(x[inside] if isinstance(x, pd.DatetimeIndex) else np.array(range(n))[inside],
               y[inside], s=8, color="#555555", zorder=4, label="Observed (inside 90%)")
    if (~inside).any():
        outside_x = x[~inside] if isinstance(x, pd.DatetimeIndex) else np.array(range(n))[~inside]
        ax.scatter(outside_x, y[~inside], s=25, color=COLORS["red"],
                   marker="x", linewidths=1.2, zorder=5, label="Observed (outside 90%)")

    # Coverage annotation
    cov_90 = np.mean(inside)
    cov_50 = np.mean((y >= pred_q["quant_0.25"].values) & (y <= pred_q["quant_0.75"].values))

    if dates is not None and len(dates) == n:
        format_date_axis(ax)

    ax.set_xlabel("Date" if dates is not None else "Test Sample Index")
    ax.set_ylabel("Precipitation (mm)")
    set_title(ax, f"{horizon}-Day Forecast: Prediction Intervals",
              f"{model_name} | 90% coverage: {cov_90:.1%} | 50% coverage: {cov_50:.1%}")
    ax.legend(loc="upper right", fontsize=FONT["legend_size"], framealpha=0.9)
    save_figure(fig, vis_dir / f"{prefix}_intervals.png")


def _plot_pit(y, concentration, rate, scale_factor, horizon, model_name, vis_dir, prefix):
    """PIT histogram with KS statistic."""
    fig, ax = plt.subplots(figsize=CHART_SIZES["pit"])
    apply_theme(ax)

    pit = compute_pit(y, concentration, rate, scale_factor)
    ks_stat, ks_pval = pit_ks_statistic(pit)

    ax.hist(pit, bins=20, density=True, alpha=0.8, color=COLORS["primary"],
            edgecolor="white", linewidth=0.5, label="PIT density")
    ax.axhline(1.0, color=COLORS["red"], linestyle="--", linewidth=1.2, label="Ideal (uniform)")

    ax.text(0.95, 0.95, f"KS = {ks_stat:.3f}\np = {ks_pval:.3f}",
            transform=ax.transAxes, fontsize=FONT["legend_size"],
            va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel("PIT Value")
    ax.set_ylabel("Density")
    set_title(ax, f"{horizon}-Day Forecast: PIT Histogram",
              f"{model_name} | Uniform (red dashed) = perfect calibration")
    ax.legend(loc="upper left", fontsize=FONT["legend_size"])
    save_figure(fig, vis_dir / f"{prefix}_pit.png")


def _plot_reliability(y, concentration, rate, scale_factor, horizon, model_name, vis_dir, prefix):
    """Reliability diagram with tolerance band and labels."""
    fig, ax = plt.subplots(figsize=CHART_SIZES["reliability"])
    apply_theme(ax)

    scale = (1.0 / rate) * scale_factor
    nominal_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    actual_covs = []
    for level in nominal_levels:
        lo_q = (1 - level) / 2
        hi_q = 1 - lo_q
        lo = sp_gamma.ppf(lo_q, a=concentration, scale=scale)
        hi = sp_gamma.ppf(hi_q, a=concentration, scale=scale)
        actual_covs.append(float(np.mean((y >= lo) & (y <= hi))))

    # Tolerance band (+-10%)
    ax.fill_between([0, 1], [0 - 0.1, 1 - 0.1], [0 + 0.1, 1 + 0.1],
                     alpha=0.15, color="#999999", label="\u00b110% tolerance")
    # Perfect line
    ax.plot([0, 1], [0, 1], "--", color=COLORS["red"], linewidth=1.2, label="Perfect calibration")
    # Model line
    ax.plot(nominal_levels, actual_covs, "o-", color=COLORS["primary"],
            markersize=6, linewidth=1.5, label=model_name, zorder=4)

    # Percentage labels
    for nl, ac in zip(nominal_levels, actual_covs):
        ax.annotate(f"{ac:.0%}", (nl, ac), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#444444")

    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Actual Coverage")
    set_title(ax, f"{horizon}-Day Forecast: Reliability Diagram",
              f"{model_name} | Grey band = \u00b110% tolerance")
    ax.legend(loc="lower right", fontsize=FONT["legend_size"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")
    save_figure(fig, vis_dir / f"{prefix}_reliability.png")


def _plot_residuals(y, pred_mean, horizon, model_name, vis_dir, prefix):
    """Residual distribution histogram."""
    fig, ax = plt.subplots(figsize=CHART_SIZES["residuals"])
    apply_theme(ax)

    residuals = pred_mean - y
    mean_bias = np.mean(residuals)

    ax.hist(residuals, bins=40, alpha=0.8, color=COLORS["primary"],
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=COLORS["red"], linestyle="--", linewidth=1.2, label="Zero")
    ax.axvline(mean_bias, color=COLORS["secondary"], linewidth=1.5, label=f"Mean bias: {mean_bias:+.1f} mm")

    ax.set_xlabel("Residual (Predicted \u2013 Observed) mm")
    ax.set_ylabel("Count")
    set_title(ax, f"{horizon}-Day Forecast: Residual Distribution", model_name)
    ax.legend(loc="upper right", fontsize=FONT["legend_size"])
    save_figure(fig, vis_dir / f"{prefix}_residuals.png")


def _plot_monthly(y, pred_mean, dates, horizon, model_name, vis_dir, prefix):
    """Monthly observed vs predicted bar chart."""
    if dates is None:
        return

    fig, ax = plt.subplots(figsize=CHART_SIZES["monthly"])
    apply_theme(ax)

    df = pd.DataFrame({"observed": y, "predicted": pred_mean, "date": pd.to_datetime(dates)})
    df["month"] = df["date"].dt.month
    monthly = df.groupby("month").agg(
        obs_mean=("observed", "mean"),
        obs_se=("observed", "sem"),
        pred_mean=("predicted", "mean"),
        pred_se=("predicted", "sem"),
    ).reindex(range(1, 13))

    months = monthly.index.values
    width = 0.35
    x = np.arange(len(months))

    ax.bar(x - width / 2, monthly["obs_mean"], width, yerr=monthly["obs_se"],
           color=COLORS["primary"], alpha=0.85, label="Observed", capsize=3)
    ax.bar(x + width / 2, monthly["pred_mean"], width, yerr=monthly["pred_se"],
           color=COLORS["secondary"], alpha=0.85, label="Predicted", capsize=3)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Precipitation (mm)")
    set_title(ax, f"{horizon}-Day Forecast: Monthly Performance", model_name)
    ax.legend(fontsize=FONT["legend_size"])
    save_figure(fig, vis_dir / f"{prefix}_monthly.png")


def _plot_qq_pit(y, concentration, rate, scale_factor, horizon, model_name, vis_dir, prefix):
    """QQ plot of PIT values against Uniform[0,1]."""
    fig, ax = plt.subplots(figsize=CHART_SIZES["qq"])
    apply_theme(ax)

    pit = compute_pit(y, concentration, rate, scale_factor)
    pit_sorted = np.sort(pit)
    n = len(pit_sorted)
    theoretical = np.linspace(0, 1, n)

    ks_stat, _ = pit_ks_statistic(pit)

    ax.scatter(theoretical, pit_sorted, s=8, color=COLORS["primary"], alpha=0.6, zorder=3)
    ax.plot([0, 1], [0, 1], "--", color=COLORS["red"], linewidth=1.2, label="Perfect calibration")

    ax.text(0.05, 0.95, f"KS = {ks_stat:.3f}", transform=ax.transAxes,
            fontsize=FONT["legend_size"], va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel("Theoretical Quantiles (Uniform)")
    ax.set_ylabel("Empirical PIT Quantiles")
    set_title(ax, f"{horizon}-Day Forecast: PIT QQ Plot", model_name)
    ax.legend(loc="lower right", fontsize=FONT["legend_size"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    save_figure(fig, vis_dir / f"{prefix}_qq_pit.png")


def _plot_importance(feature_importance, horizon, model_name, vis_dir, prefix):
    """Top-25 feature importance bar chart, color-coded by category."""
    fig, ax = plt.subplots(figsize=CHART_SIZES["importance"])
    apply_theme(ax)

    sorted_feats = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:25]
    names = [f[0] for f in sorted_feats]
    gains = [f[1] for f in sorted_feats]
    colors = [feature_color(n) for n in names]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, gains, color=colors, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Gain")
    set_title(ax, f"{horizon}-Day Forecast: Feature Importance (Top 25)",
              f"{model_name} | XGBoost gain metric")

    # Category legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["forecast"], label="Forecast"),
        Patch(facecolor=COLORS["climate"], label="Climatology"),
        Patch(facecolor=COLORS["recent"], label="Recent Conditions"),
        Patch(facecolor=COLORS["calendar"], label="Calendar"),
        Patch(facecolor=COLORS["bias"], label="Bias Correction"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=FONT["legend_size"])
    save_figure(fig, vis_dir / f"{prefix}_importance.png")


def plot_calibration_comparison(
    y: np.ndarray,
    concentration: np.ndarray,
    rate: np.ndarray,
    scale_factor_before: float,
    scale_factor_after: float,
    horizon: int,
    model_name: str,
    vis_dir: Path,
):
    """Side-by-side PIT histograms before/after calibration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=CHART_SIZES["calibration_comparison"])

    for ax, sf, label in [(ax1, scale_factor_before, "Before"), (ax2, scale_factor_after, "After")]:
        apply_theme(ax)
        pit = compute_pit(y, concentration, rate, sf)
        ks_stat, _ = pit_ks_statistic(pit)

        ax.hist(pit, bins=20, density=True, alpha=0.8, color=COLORS["primary"],
                edgecolor="white", linewidth=0.5)
        ax.axhline(1.0, color=COLORS["red"], linestyle="--", linewidth=1.2)
        ax.text(0.95, 0.95, f"KS = {ks_stat:.3f}\nScale = {sf:.2f}",
                transform=ax.transAxes, fontsize=FONT["legend_size"],
                va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.set_xlabel("PIT Value")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} Calibration", fontsize=FONT["subtitle_size"], fontweight="bold")

    fig.suptitle(f"{horizon}-Day Forecast: Calibration Comparison ({model_name})",
                 fontsize=FONT["title_size"], fontweight="bold", y=1.02)
    prefix = f"{model_name.lower().replace(' ', '_')}_{horizon}d"
    save_figure(fig, vis_dir / f"{prefix}_calibration_comparison.png")
