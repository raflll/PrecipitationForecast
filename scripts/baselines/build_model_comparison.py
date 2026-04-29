"""
Combine baseline + ML model metrics into a single comparison table and a
multi-panel chart used in the final report.

Inputs (already on disk):
  models/baselines_metrics.csv          -- climatology, raw_gefs, emos_gamma
  models/gamlss_all_metrics.csv         -- GAMLSS 2-stage XGBoost (R)
  models/all_metrics_r.csv              -- XGBoost Gamma reg:gamma (R)
  models/all_metrics.csv                -- XGBoostLSS (Python, only 14d)

Outputs:
  models/comparison_metrics.csv               -- long-format combined metrics
  visualizations/model_comparison_panel.png   -- 4-panel figure (CRPS, NLL,
                                                  MAE, 90% coverage vs horizon)
  visualizations/model_comparison_test_table.tex -- LaTeX table fragment
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

MODEL_DISPLAY = {
    "climatology": "Climatology",
    "raw_gefs": "Raw GEFS",
    "emos_gamma": "EMOS-Gamma",
    "xgboost_gamma": "XGBoost-Gamma (R)",
    "gamlss": "GAMLSS 2-stage (R)",
    "xgboostlss": "XGBoostLSS (Py)",
}

# Plotting order (low to high sophistication)
MODEL_ORDER = [
    "climatology",
    "raw_gefs",
    "emos_gamma",
    "xgboost_gamma",
    "gamlss",
    "xgboostlss",
]

MODEL_COLOR = {
    "climatology": "#888888",
    "raw_gefs":    "#B5651D",
    "emos_gamma":  "#FF9800",
    "xgboost_gamma": COLORS["accent"],
    "gamlss":      COLORS["primary"],
    "xgboostlss":  COLORS["purple"],
}


def load_all() -> pd.DataFrame:
    frames = []

    # Baselines
    base = pd.read_csv(REPO / "models" / "baselines_metrics.csv")
    frames.append(base)

    # GAMLSS 2-stage (R)
    gam = pd.read_csv(REPO / "models" / "gamlss_all_metrics.csv")
    gam["model"] = "gamlss"
    frames.append(gam)

    # XGBoost Gamma (R)
    xg = pd.read_csv(REPO / "models" / "all_metrics_r.csv")
    xg["model"] = "xgboost_gamma"
    frames.append(xg)

    # XGBoostLSS (Python, only 14d available)
    if (REPO / "models" / "all_metrics.csv").exists():
        lss = pd.read_csv(REPO / "models" / "all_metrics.csv")
        lss["model"] = "xgboostlss"
        frames.append(lss)

    df = pd.concat(frames, ignore_index=True, sort=False)

    # Harmonise columns
    keep = [
        "model", "split", "horizon_days", "n_samples",
        "mean_crps", "mean_nll", "mae", "bias",
        "cov_50", "cov_60", "cov_70", "cov_80", "cov_90", "cov_95",
        "mean_observed", "mean_predicted", "median_ae",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep].copy()

    return df


def make_panel(df: pd.DataFrame, split: str, out_path: Path):
    sub = df[df["split"] == split].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    panels = [
        ("mean_crps", "CRPS (lower is better)", "CRPS [mm]", axes[0, 0]),
        ("mean_nll", "Negative log-likelihood (lower is better)", "NLL", axes[0, 1]),
        ("mae", "Mean absolute error", "MAE [mm]", axes[1, 0]),
        ("cov_90", "90% prediction-interval coverage", "Empirical coverage", axes[1, 1]),
    ]

    for col, title, ylab, ax in panels:
        for m in MODEL_ORDER:
            ms = sub[sub["model"] == m].sort_values("horizon_days")
            if ms.empty:
                continue
            marker = "o" if m not in ("climatology", "raw_gefs") else "s"
            ls = "-"
            if m in ("climatology", "raw_gefs", "emos_gamma"):
                ls = "--"
            ax.plot(
                ms["horizon_days"], ms[col],
                marker=marker, ls=ls, lw=1.6, ms=6,
                color=MODEL_COLOR[m],
                label=MODEL_DISPLAY[m],
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Forecast horizon (days)")
        ax.set_ylabel(ylab)
        ax.set_xticks([3, 7, 14, 21, 28, 35])
        ax.grid(alpha=0.3)
        if col == "cov_90":
            ax.axhline(0.90, color="black", ls=":", lw=1, alpha=0.6)
            ax.set_ylim(0.0, 1.05)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10)
    fig.suptitle(f"Model comparison vs. forecast horizon — {split.upper()} set",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def make_latex_table(df: pd.DataFrame, split: str, out_path: Path):
    """One row per model, columns are horizons. Bold the best per column."""
    sub = df[df["split"] == split].copy()
    pivot = sub.pivot_table(
        index="model", columns="horizon_days", values="mean_crps", aggfunc="first"
    )
    pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index])
    pivot.index = [MODEL_DISPLAY[m] for m in pivot.index]

    horizons = sorted(pivot.columns.tolist())
    best = {h: pivot[h].idxmin() for h in horizons if pivot[h].notna().any()}

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\begin{tabular}{l" + "r" * len(horizons) + "}"
    )
    lines.append(r"\toprule")
    header = "Model & " + " & ".join(f"{h}d" for h in horizons) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for model_label in pivot.index:
        cells = []
        for h in horizons:
            v = pivot.loc[model_label, h]
            if pd.isna(v):
                cells.append("--")
            else:
                txt = f"{v:.2f}"
                if best.get(h) == model_label:
                    txt = r"\textbf{" + txt + "}"
                cells.append(txt)
        lines.append(model_label + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    caption = (
        f"Mean CRPS by forecast horizon on the {split} set "
        "(2025 held-out for test, 2024 for validation). Lower is better; "
        "the best in each column is in bold. "
        "`--' marks horizons not retrained for this comparison."
    )
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{tab:comparison-crps-" + split + "}")
    lines.append(r"\end{table}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    df = load_all()
    out = REPO / "models" / "comparison_metrics.csv"
    df.sort_values(["model", "horizon_days", "split"]).to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows).")

    vis = REPO / "visualizations"
    make_panel(df, "test", vis / "model_comparison_panel_test.png")
    make_panel(df, "val", vis / "model_comparison_panel_val.png")

    make_latex_table(df, "test", vis / "model_comparison_test_table.tex")
    make_latex_table(df, "val", vis / "model_comparison_val_table.tex")


if __name__ == "__main__":
    main()
