"""
Head-to-head comparison plot: observed 14-day cumulative precipitation vs each
model's predicted mean across the 2025 test year, plus a monthly aggregate
comparison and a total annual precipitation bar chart.

Inputs:
  models/predictions_14d.csv  (per-sample (init_date, model, observed, mean, q05..q95))

Outputs:
  visualizations/head_to_head_14d.png   -- 3-panel headline figure for the report
  visualizations/head_to_head_14d_scatter.png -- pred vs obs scatter (one panel per model)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from viz_theme import COLORS, DPI  # noqa: E402

# Same colors / order as model_comparison_panel for consistency
MODEL_ORDER = ["climatology", "raw_gefs", "emos_gamma", "gamlss", "xgboost_gamma"]
MODEL_DISPLAY = {
    "climatology":   "Climatology",
    "raw_gefs":      "Raw GEFS",
    "emos_gamma":    "EMOS-Gamma",
    "gamlss":        "GAMLSS 2-stage",
    "xgboost_gamma": "XGBoost-Gamma (best)",
}
MODEL_COLOR = {
    "climatology":   "#888888",
    "raw_gefs":      "#B5651D",
    "emos_gamma":    "#FF9800",
    "gamlss":        COLORS["primary"],
    "xgboost_gamma": COLORS["accent"],
}
OBS_COLOR = "black"
BEST_MODEL = "xgboost_gamma"


def load() -> pd.DataFrame:
    df = pd.read_csv(REPO / "models" / "predictions_14d.csv")
    df["init_date"] = pd.to_datetime(df["init_date"])
    return df


def plot_three_panel(df: pd.DataFrame, out_path: Path):
    test = df[df["split"] == "test"].copy()
    test["month"] = test["init_date"].dt.month

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 1.4], hspace=0.42)
    ax_ts = fig.add_subplot(gs[0])
    ax_mo = fig.add_subplot(gs[1])
    ax_tot = fig.add_subplot(gs[2])

    # ── Panel 1: time series ─────────────────────────────────────────────────
    obs = test[test["model"] == "climatology"][["init_date", "observed"]].sort_values("init_date")
    ax_ts.plot(obs["init_date"], obs["observed"], color=OBS_COLOR, lw=1.2,
               alpha=0.9, label="Observed (truth)", zorder=10)

    # 90% PI shading for the best model only
    best = test[test["model"] == BEST_MODEL].sort_values("init_date")
    ax_ts.fill_between(best["init_date"], best["q05"], best["q95"],
                       color=MODEL_COLOR[BEST_MODEL], alpha=0.18,
                       label=f"{MODEL_DISPLAY[BEST_MODEL]} 90% PI", zorder=2)

    for m in MODEL_ORDER:
        mdf = test[test["model"] == m].sort_values("init_date")
        if mdf.empty:
            continue
        ls = "-" if m in ("xgboost_gamma", "gamlss") else "--"
        lw = 1.6 if m == BEST_MODEL else 1.0
        alpha = 0.95 if m == BEST_MODEL else 0.7
        ax_ts.plot(mdf["init_date"], mdf["mean"], color=MODEL_COLOR[m],
                   ls=ls, lw=lw, alpha=alpha, label=MODEL_DISPLAY[m])

    ax_ts.set_title("14-day cumulative precipitation forecasts vs observations — 2025 test year",
                    fontsize=12)
    ax_ts.set_ylabel("Cumulative precip [mm]")
    ax_ts.grid(alpha=0.3)
    ax_ts.legend(loc="upper left", ncol=3, fontsize=9, frameon=True, framealpha=0.95)
    ax_ts.set_xlim(obs["init_date"].min(), obs["init_date"].max())

    # ── Panel 2: monthly aggregates (mean of 14d windows initialized that month) ─
    months = sorted(test["month"].unique())
    monthly_obs = (
        test[test["model"] == "climatology"]
        .groupby("month")["observed"].mean()
    )

    width = 0.13
    x = np.arange(len(months))
    ax_mo.bar(x, monthly_obs.values, width=width * 1.6,
              color=OBS_COLOR, label="Observed", zorder=3)

    for i, m in enumerate(MODEL_ORDER, start=1):
        mdf = test[test["model"] == m]
        monthly_pred = mdf.groupby("month")["mean"].mean()
        ax_mo.bar(x + i * width, monthly_pred.values, width=width,
                  color=MODEL_COLOR[m], label=MODEL_DISPLAY[m], zorder=3)

    ax_mo.set_xticks(x + 2.5 * width)
    ax_mo.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax_mo.set_ylabel("Avg 14-d cumulative precip [mm]")
    ax_mo.set_title("Monthly mean of 14-day cumulative precipitation forecasts (2025)",
                    fontsize=12)
    ax_mo.grid(alpha=0.3, axis="y")
    ax_mo.legend(loc="upper left", ncol=3, fontsize=9, frameon=True, framealpha=0.95)

    # ── Panel 3: annual totals ───────────────────────────────────────────────
    obs_total = float(test[test["model"] == "climatology"]["observed"].sum())
    rows = [("Observed", obs_total, OBS_COLOR)]
    for m in MODEL_ORDER:
        rows.append((MODEL_DISPLAY[m], float(test[test["model"] == m]["mean"].sum()),
                     MODEL_COLOR[m]))

    labels = [r[0] for r in rows]
    totals = [r[1] for r in rows]
    cols = [r[2] for r in rows]

    bars = ax_tot.barh(labels, totals, color=cols)
    ax_tot.axvline(obs_total, ls="--", color=OBS_COLOR, lw=1, alpha=0.6, zorder=0)
    ax_tot.invert_yaxis()
    ax_tot.set_xlabel("Sum of 14-d cumulative predictions over all 2025 init dates [mm]")
    ax_tot.set_title("Total estimated precipitation across 2025 — model vs observed",
                     fontsize=12)
    ax_tot.grid(alpha=0.3, axis="x")
    for b, v in zip(bars, totals):
        pct = (v - obs_total) / obs_total * 100
        ax_tot.text(v + obs_total * 0.01, b.get_y() + b.get_height() / 2,
                    f"{v:,.0f} mm  ({pct:+.1f}%)",
                    va="center", fontsize=9)

    fig.suptitle("Head-to-head: our model vs the state-of-the-art baselines",
                 fontsize=14, y=0.995)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_scatter_grid(df: pd.DataFrame, out_path: Path):
    test = df[df["split"] == "test"].copy()
    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(3.0 * len(MODEL_ORDER), 3.6),
                             sharey=True)
    max_v = max(test["observed"].max(), test["mean"].max()) * 1.05

    for ax, m in zip(axes, MODEL_ORDER):
        sub = test[test["model"] == m]
        ax.scatter(sub["observed"], sub["mean"], s=8, alpha=0.35,
                   color=MODEL_COLOR[m])
        ax.plot([0, max_v], [0, max_v], color="black", ls="--", lw=0.8, alpha=0.6)
        bias = (sub["mean"] - sub["observed"]).mean()
        mae = (sub["mean"] - sub["observed"]).abs().mean()
        ax.set_title(f"{MODEL_DISPLAY[m]}\nMAE={mae:.1f}, bias={bias:+.1f}",
                     fontsize=10)
        ax.set_xlabel("Observed [mm]")
        ax.set_xlim(0, max_v)
        ax.set_ylim(0, max_v)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Predicted mean [mm]")
    fig.suptitle("Predicted mean vs observed — 14-day horizon, 2025 test year", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    df = load()
    out_dir = REPO / "visualizations"
    plot_three_panel(df, out_dir / "head_to_head_14d.png")
    plot_scatter_grid(df, out_dir / "head_to_head_14d_scatter.png")


if __name__ == "__main__":
    main()
