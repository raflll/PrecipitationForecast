"""
Unified visualization theme for all Python-based precipitation forecast models.
Mirrors the R theme_precip() and COLORS palette from train_gamlss.R.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DPI = 200

COLORS = {
    # Core palette (matches R COLORS list)
    "primary": "#1B6CA8",
    "secondary": "#E8751A",
    "accent": "#2CA02C",
    "light_blue": "#A8D4F0",
    "red": "#D62728",
    "purple": "#9467BD",
    # Feature-category colors (for importance plots)
    "forecast": "#2196F3",
    "climate": "#FF9800",
    "recent": "#4CAF50",
    "calendar": "#9C27B0",
    "bias": "#FF5722",
}

CHART_SIZES = {
    "pred_vs_obs": (8, 7),
    "intervals": (14, 6),
    "importance": (11, 8),
    "shap_beeswarm": (12, 10),
    "shap_bar": (12, 8),
    "shap_dep": (14, 10),
    "pit": (7, 5),
    "reliability": (7, 7),
    "residuals": (8, 5),
    "monthly": (10, 5),
    "qq": (6, 6),
    "calibration_comparison": (14, 5),
}

FONT = {
    "family": "sans-serif",
    "title_size": 14,
    "subtitle_size": 12,
    "label_size": 12,
    "tick_size": 11,
    "legend_size": 10,
}

# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------


def apply_theme(ax):
    """Apply the unified theme to a matplotlib Axes (mirrors R theme_precip)."""
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")

    # Grid: light grey major, no minor
    ax.grid(True, which="major", color="#EBEBEB", linewidth=0.4)
    ax.grid(False, which="minor")

    # Tick styling
    ax.tick_params(axis="both", labelsize=FONT["tick_size"], colors="#4D4D4D")
    ax.tick_params(axis="both", which="minor", length=0)

    # Spine styling — keep bottom and left, hide top and right
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#CCCCCC")
        ax.spines[spine].set_linewidth(0.5)

    # Label styling
    ax.xaxis.label.set_size(FONT["label_size"])
    ax.xaxis.label.set_color("#333333")
    ax.yaxis.label.set_size(FONT["label_size"])
    ax.yaxis.label.set_color("#333333")


def set_title(ax, title, subtitle=None):
    """Set a bold title and optional grey subtitle."""
    ax.set_title(title, fontsize=FONT["title_size"], fontweight="bold", color="#222222")
    if subtitle:
        ax.text(
            0.5, 1.01, subtitle,
            transform=ax.transAxes,
            fontsize=FONT["subtitle_size"] - 1,
            color="#666666",
            ha="center", va="bottom",
        )


def save_figure(fig, path):
    """Save a figure with the standard DPI and tight layout."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_date_axis(ax, interval_months=2):
    """Format x-axis as dates with month abbreviation + year."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval_months))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def feature_color(name):
    """Return the category color for a feature name."""
    name = name.lower()
    if any(name.startswith(p) for p in [
        "cum_apcp", "mean_6h", "max_6h", "min_6h", "std_6h", "median_6h",
        "forecast_", "frac_dry", "frac_heavy", "longest_", "num_wet",
        "q1_apcp", "q2_apcp", "q3_apcp", "q4_apcp", "front_half",
        "front_back", "max_spread", "mean_spread", "spread_trend",
        "mean_trend", "spread_mean", "top10pct", "spread_to_mean",
        "cum_spread", "forecast_cv", "forecast_skew", "forecast_p",
        "forecast_range", "forecast_iqr",
    ]):
        return COLORS["forecast"]
    if any(name.startswith(p) for p in [
        "clim_", "persistence_",
    ]):
        return COLORS["climate"]
    if any(name.startswith(p) for p in [
        "precip_last", "precip_lag", "rainy_days", "max_precip_last",
        "precip_intensity", "precip_trend", "recent_", "mean_temp",
        "temp_anomaly", "temp_trend", "mean_dtr", "mean_wind",
        "mean_gusts", "mean_sunshine", "sunshine_anomaly",
        "mean_precip_hours", "days_since", "thunder_", "fog_",
        "haze_", "rain_flag",
    ]):
        return COLORS["recent"]
    if any(name.startswith(p) for p in [
        "day_of_year", "month", "is_wet", "is_hurricane", "is_peak",
        "is_reforecast",
    ]):
        return COLORS["calendar"]
    if any(name.startswith(p) for p in [
        "gefs_bias", "gefs_rolling", "gefs_mae", "gefs_ratio",
        "gefs_overpredict", "gefs_direction", "bias_corrected",
        "rolling_bias", "spread_adequacy", "forecast_bias",
        "forecast_vs_gefs",
    ]):
        return COLORS["bias"]
    return COLORS["primary"]
