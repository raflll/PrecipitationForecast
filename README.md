# Miami Probabilistic Precipitation Forecasting

Probabilistic precipitation forecasting for Miami using XGBoostLSS, GAMLSS, and XGBoost Gamma models with GEFS ensemble weather data. Each model predicts full Gamma probability distributions for cumulative rainfall over 3 to 35-day horizons.

## Quick Start: Training Models

### Prerequisites

```bash
# Python dependencies
pip install -r requirements.txt

# R dependencies (install from R console)
install.packages(c("arrow", "xgboost", "data.table", "ggplot2",
                    "shapviz", "patchwork", "scoringRules", "scales", "jsonlite"))
```

The training dataset must exist at `data/training/training_dataset.parquet`. If it doesn't, follow the [full pipeline steps](#reproducing-the-pipeline) below.

### Train XGBoostLSS (Python)

```bash
# Train all horizons (3, 7, 14, 21, 28, 35 days)
python src/train_xgboostlss.py

# Single horizon
python src/train_xgboostlss.py --horizon 14

# Quick test run (fewer Optuna trials, faster)
python src/train_xgboostlss.py --horizon 14 --quick
```

Outputs: `models/xgboostlss_*d.json`, `models/all_metrics.csv`, `visualizations/xgboostlss_*d_*.png`

### Train GAMLSS (R)

```bash
# Train all horizons
Rscript src/train_gamlss.R

# Single horizon
Rscript src/train_gamlss.R --horizon 14

# Quick test run
Rscript src/train_gamlss.R --horizon 14 --quick
```

Outputs: `models/gamlss_*d.rds`, `models/gamlss_all_metrics.csv`, `visualizations/gamlss_*d_*.png`

### Train XGBoost Gamma (R)

```bash
Rscript src/train_xgboost_gamma.R
Rscript src/train_xgboost_gamma.R --horizon 14
Rscript src/train_xgboost_gamma.R --quick
```

Outputs: `models/xgboost_gamma_*d.rds`, `visualizations/xgboost_gamma_*d_*.png`

### Compare Models

After training multiple models, generate comparison charts:

```bash
python scripts/visualizations/model_comparison.py
python scripts/visualizations/coverage_by_horizon.py
```

### What Each Model Produces

All models generate the same set of diagnostic plots per horizon:

| Plot | Description |
|------|-------------|
| `*_pred_vs_obs.png` | Predicted vs observed scatter with LOESS fit |
| `*_intervals.png` | Full-year prediction intervals (90% and 50% PI) |
| `*_pit.png` | PIT histogram (calibration diagnostic) |
| `*_reliability.png` | Reliability diagram with tolerance band |
| `*_importance.png` | Top 25 features by XGBoost gain |
| `*_shap_beeswarm.png` | SHAP beeswarm plot |
| `*_residuals.png` | Residual distribution |
| `*_monthly.png` | Monthly observed vs predicted bars |

---

## Overview

This project builds a machine learning pipeline that combines NOAA station observations (1948-present) with GEFS ensemble forecast data (2000-present) to produce calibrated probabilistic precipitation forecasts for Miami International Airport. Instead of point predictions, each model outputs Gamma distribution parameters, enabling prediction intervals and risk quantification.

## Data Sources

| Source | Coverage | Auth | Description |
|--------|----------|------|-------------|
| [NOAA CDO](https://www.ncdc.noaa.gov/cdo-web/) | 1948-present | API token | Daily precipitation, temperature, weather types from station USW00012839 (Miami Intl Airport) |
| [Open-Meteo](https://open-meteo.com/) | 1940-present | None | Daily temperature, wind, sunshine, precipitation hours |
| [GEFS Reforecast](https://noaa-gefs-retrospective.s3.amazonaws.com/) | 2000-2019 | None | Wednesday-only 35-day ensemble forecasts (11 members) from AWS S3 |
| [GEFS Operational](https://www.nco.ncep.noaa.gov/pmb/products/gens/) | Sept 2020-present | None | Daily 35-day ensemble forecasts (mean + spread) via Herbie |

## Project Structure

```
GroupProject/
├── src/
│   ├── train_xgboostlss.py              # XGBoostLSS training (Python)
│   ├── train_gamlss.R                    # GAMLSS 2-stage training (R)
│   ├── train_xgboost_gamma.R             # XGBoost Gamma regression (R)
│   ├── evaluation.py                     # Shared metrics + plotting
│   ├── calibration.py                    # Post-hoc calibration utilities
│   └── viz_theme.py                      # Unified visualization theme
├── scripts/
│   ├── data_collection/
│   │   └── data_collection.ipynb         # Pull raw NOAA + Open-Meteo data
│   ├── data_processing/
│   │   ├── rebuild_miami_clean.py        # Clean/merge observations → miami_clean.csv
│   │   └── build_training_dataset.py     # Feature engineering → training_dataset.parquet
│   ├── pull_gefs_reforecast.py           # Download GEFS reforecast data (2000-2019)
│   ├── pull_gefs_operational.py          # Download GEFS operational data (2020+)
│   └── visualizations/
│       ├── miami_monthly_rainfall_histogram.py
│       ├── monthly_precipitation_comparison.py
│       ├── model_comparison.py           # Cross-model metric comparison
│       └── coverage_by_horizon.py        # Coverage summary across horizons
├── data/                                 # All data (gitignored)
│   ├── noaa/                             # Raw NOAA station data
│   ├── open-meteo/                       # Raw Open-Meteo data
│   ├── miami_clean.csv                   # Cleaned daily observations
│   ├── GEFS_Reforecast/                  # Reforecast parquets by year
│   ├── GEFS_Operational/                 # Operational parquets by year
│   └── training/
│       └── training_dataset.parquet      # Final ML-ready dataset
├── models/                               # Trained models + metrics (gitignored)
├── visualizations/                       # Generated plots (gitignored)
├── docs/
│   ├── MODEL_CARD.md                     # Feature documentation + model details
│   └── DATA_SOURCES.md                   # Data source technical reference
└── requirements.txt
```

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

A NOAA CDO API token is required for raw data collection. Get one at https://www.ncdc.noaa.gov/cdo-web/token and save it in a `.env` file:

```
NOAA_TOKEN=your_token_here
```

## Reproducing the Pipeline

### Step 1: Collect raw observations

Open `scripts/data_collection/data_collection.ipynb` in Jupyter and run all cells. This pulls daily weather data from NOAA CDO and Open-Meteo into `data/noaa/` and `data/open-meteo/`.

```bash
jupyter notebook scripts/data_collection/data_collection.ipynb
```

### Step 2: Clean and merge observations

```bash
python scripts/data_processing/rebuild_miami_clean.py
```

Merges NOAA and Open-Meteo data, adds weather type flags, and outputs `data/miami_clean.csv`.

### Step 3: Download GEFS forecast data

```bash
# Reforecast data (2000-2019, Wednesday-only, downloads from S3)
python scripts/pull_gefs_reforecast.py --start-year 2000 --end-year 2019

# Operational data (one year at a time, downloads via Herbie)
python scripts/pull_gefs_operational.py 2024
```

Both scripts support resuming from where they left off if interrupted.

### Step 4: Build training dataset

```bash
python scripts/data_processing/build_training_dataset.py
```

Engineers 125+ features from observations and GEFS data across 6 forecast horizons (3, 7, 14, 21, 28, 35 days). Outputs `data/training/training_dataset.parquet`. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the full feature list.

### Step 5: Train models

See [Quick Start](#quick-start-training-models) above for all training commands. All three models support `--horizon N` and `--quick` flags.

## Models

Three distributional regression models, all outputting Gamma distribution parameters for uncertainty quantification:

| Model | Language | Approach | Tuning |
|-------|----------|----------|--------|
| **XGBoostLSS** | Python | Direct Gamma NLL via XGBoostLSS library | Optuna (30 trials, 3-fold CV) |
| **GAMLSS** | R | 2-stage: XGBoost mean on log1p(precip) + XGBoost variance on CV residuals | Random search (40 + 20 trials) |
| **XGBoost Gamma** | R | Native `reg:gamma` objective + MLE shape from residuals | Bayesian optimization (30 trials) |

All models include post-hoc variance scaling calibration on the validation set to improve prediction interval coverage.

- **Horizons**: 3, 7, 14, 21, 28, 35 days
- **Train**: 2019, 2022 | **Validation**: held-out from train | **Test**: 2025
- **Features**: 132 engineered features across forecast, climatology, recent conditions, calendar, bias correction, and cross-feature categories

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for complete feature documentation and model details.
