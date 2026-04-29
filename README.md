# Probabilistic Precipitation Forecasting for Miami

**CSC542 — Statistical Learning with Applications · Spring 2026**
Justin Bonner, Arian Djahed, Alex Morrow · Instructor: Vanessa Aguiar
University of Miami

This repository contains everything needed to reproduce our final-project pipeline: a probabilistic precipitation forecasting system for Miami International Airport that predicts a full Gamma distribution for cumulative rainfall at 3, 7, 14, 21, 28, and 35-day horizons.

- **Final report (PDF):** [final.pdf](final.pdf) — at the repo root for easy navigation
- **Presentation deck:** [docs/Weather_Group_Presentation.pptx](docs/Weather_Group_Presentation.pptx)
- **Midterm report:** [docs/midterm.pdf](docs/midterm.pdf) (the final report extends this)
- **LaTeX sources:** [docs/tex/final.tex](docs/tex/final.tex), [docs/tex/midterm.tex](docs/tex/midterm.tex)

## Highlights

- **Models:** three distributional ML models (XGBoost-Gamma, GAMLSS 2-stage, XGBoostLSS) compared against three classical baselines (climatology, raw GEFS dressing, EMOS-Gamma).
- **Best result:** XGBoost-Gamma wins on test-set CRPS at five of six horizons. At the 14-day horizon, CRPS = 21.7 mm, a 10% improvement over climatology and 30% over raw GEFS.
- **Calibration:** all ML models reach 86–98% empirical coverage at the nominal 90% prediction interval after post-hoc variance calibration.
- **Data:** 49,470 (init\_date, horizon) records from 2000–2025, built from NOAA station observations, Open-Meteo archive data, GEFS reforecast (2000–2019), and GEFS operational (2020+).

## Repository layout

```
GroupProject/
├── final.pdf                      # ← THE final report (start here)
├── docs/                          # Submission deliverables
│   ├── tex/
│   │   ├── final.tex              # LaTeX source for final.pdf
│   │   └── midterm.tex            # LaTeX source for midterm.pdf
│   ├── midterm.pdf                # Original midterm report (the final extends this)
│   ├── Weather_Group_Presentation.pptx
│   ├── MODEL_CARD.md              # Full feature catalog and model details
│   └── DATA_SOURCES.md            # Technical reference for data sources
├── src/                           # Main training / evaluation code
│   ├── train_xgboost_gamma.R      # Best model: XGBoost reg:gamma + MLE shape
│   ├── train_gamlss.R             # GAMLSS 2-stage XGBoost
│   ├── train_xgboostlss.py        # XGBoostLSS (Python)
│   ├── evaluation.py              # Shared metrics + diagnostic plotting
│   ├── calibration.py             # Post-hoc variance scaling
│   └── viz_theme.py               # Unified plot theme
├── scripts/                       # Pipeline scripts (data + helpers)
│   ├── data_collection/           # NOAA + Open-Meteo pulls
│   ├── data_processing/           # Cleaning, feature engineering
│   ├── pull_gefs_*.py             # GEFS data downloads (S3 + Herbie)
│   ├── baselines/
│   │   ├── run_sota_baselines.py        # Climatology + Raw-GEFS + EMOS-Gamma
│   │   └── build_model_comparison.py    # Combined metrics CSV + comparison chart
│   └── visualizations/            # Comparison plots, histograms
├── data/                          # See "Getting the data" below
├── models/                        # Trained model artifacts + metric CSVs
├── visualizations/                # All generated plots
├── requirements.txt
└── README.md
```

The repo follows the assignment-required structure: `docs/` for the report and presentation, `data/` (or download instructions, see below), `src/` for source code.

## Getting the data

Raw GEFS data is too large to commit (≈40 GB across all years). The processed training parquet that all models load is small (~80 MB) and **is** included at `data/training/training_dataset.parquet`.

### Option A — use the processed parquet (recommended for graders)

It is already in the repo. Just clone and run training. No external accounts or downloads needed.

### Option B — rebuild the dataset end-to-end

Reproducing from raw sources requires:

1. **NOAA CDO API token.** Free at <https://www.ncdc.noaa.gov/cdo-web/token>. Save in a `.env` file:
   ```
   NOAA_TOKEN=your_token_here
   ```
2. **Run the pipeline** (each step is idempotent and resumable):
   ```bash
   # 1. Daily observations
   jupyter notebook scripts/data_collection/data_collection.ipynb
   python scripts/data_processing/rebuild_miami_clean.py

   # 2. GEFS forecasts (pulls in chunks; will resume on rerun)
   python scripts/pull_gefs_reforecast.py --start-year 2000 --end-year 2019
   python scripts/pull_gefs_operational.py 2024
   python scripts/pull_gefs_operational.py 2025

   # 3. Feature engineering
   python scripts/data_processing/build_training_dataset.py
   ```
   Output: `data/training/training_dataset.parquet`.

The full data-source reference (URLs, station IDs, rate-limit caveats, GRIB2 conventions, etc.) is in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## Quick start

### Install dependencies

```bash
# Python
python -m venv .venv
source .venv/bin/activate    # or: .venv\Scripts\activate on Windows
pip install -r requirements.txt

# R packages (run from R console)
install.packages(c("arrow", "xgboost", "data.table", "ggplot2",
                   "shapviz", "patchwork", "scoringRules", "scales", "jsonlite"))
```

### Train the best model

```bash
# Train XGBoost-Gamma at all six horizons (R)
Rscript src/train_xgboost_gamma.R

# Or just one horizon
Rscript src/train_xgboost_gamma.R --horizon 14

# Quick test run (faster, fewer trials)
Rscript src/train_xgboost_gamma.R --horizon 14 --quick
```

Outputs go to `models/xgboost_gamma_*.json` and `visualizations/xgboost_gamma_*.png`. Training metrics are appended to `models/all_metrics_r.csv`.

### Train the other two ML models

```bash
Rscript src/train_gamlss.R                 # GAMLSS 2-stage (R)
python src/train_xgboostlss.py             # XGBoostLSS (Python)
```

### Run the SOTA baselines

```bash
python scripts/baselines/run_sota_baselines.py
```

This fits and evaluates climatology, raw GEFS dressing, and EMOS-Gamma on the same train/val/test split as the ML models. Output: `models/baselines_metrics.csv`.

### Build the comparison table and chart

```bash
python scripts/baselines/build_model_comparison.py
```

Combines all model metrics into `models/comparison_metrics.csv` and emits the multi-panel comparison chart used in the report:

- `visualizations/model_comparison_panel_test.png`
- `visualizations/model_comparison_panel_val.png`
- `visualizations/model_comparison_test_table.tex` (LaTeX fragment)

## Models in this comparison

| ID | Model | Lang | Approach |
|----|-------|------|----------|
| B1 | Climatology | Python | Gamma fit to historical observations in a ±15-day DOY window |
| B2 | Raw GEFS dressing | Python | Method-of-moments Gamma from ensemble mean / spread |
| B3 | EMOS-Gamma | Python | Non-homogeneous Gamma regression, min-CRPS fit |
| M1 | **XGBoost-Gamma** (best) | R | XGBoost `reg:gamma` + MLE shape from residuals |
| M2 | GAMLSS 2-stage | R | XGBoost mean on log1p(y) + XGBoost spread on |residuals| |
| M3 | XGBoostLSS | Python | Direct Gamma NLL via XGBoostLSS library |

All ML models include post-hoc variance scaling on the validation set to calibrate the 90% prediction interval. Splits: train 2000–2023, validation 2024, test 2025. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the 132-feature catalog and full hyperparameter spec.

## Top-line results (test set, 2025 held-out)

CRPS by horizon (lower is better; best in **bold**):

| Model              |  3d  |  7d  | 14d  | 21d  | 28d  | 35d  |
|--------------------|-----:|-----:|-----:|-----:|-----:|-----:|
| Climatology        | 9.21 | 15.92| 24.21| 30.39| 35.35| 38.78|
| Raw GEFS           | 7.46 | 15.83| 30.92| 53.06| 76.46| 101.02|
| EMOS-Gamma         | 6.60 | 13.81| 24.46| 36.95| 48.46| 58.70 |
| **XGBoost-Gamma**  | **6.48** | **13.34** | 21.72 | **28.46** | **31.71** | **36.16** |
| GAMLSS 2-stage     | 7.79 | 15.30| 23.32| 28.63| 35.77| 36.91 |
| XGBoostLSS (14d only) | — | — | **21.08** | — | — | — |

Full results with NLL, MAE, bias, and 50/60/70/80/90/95% prediction-interval coverage are in `models/comparison_metrics.csv` and Figure 2 of the final report.

## License & contact

Course project, Spring 2026. For questions, open an issue or contact justin@propagenticai.com.
