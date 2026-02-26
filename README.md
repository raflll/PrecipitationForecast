# Miami Probabilistic Precipitation Forecasting

Probabilistic precipitation forecasting for Miami using XGBoostLSS and GEFS ensemble weather data. The model predicts full Gamma probability distributions for cumulative rainfall over 3 to 35-day horizons.

## Overview

This project builds a machine learning pipeline that combines NOAA station observations (1948-present) with GEFS ensemble forecast data (2000-present) to produce calibrated probabilistic precipitation forecasts for Miami International Airport. Instead of point predictions, the model outputs Gamma distribution parameters (concentration and rate), enabling prediction intervals and risk quantification.

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
│   └── train_xgboostlss.py              # Model training + evaluation + diagnostics
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
│       └── gefs_bias_analysis.py
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

```bash
# Train a single horizon
python src/train_xgboostlss.py --horizon 14

# Train all horizons (3, 7, 14, 21, 28, 35 days)
python src/train_xgboostlss.py

# Quick test run (fewer Optuna trials)
python src/train_xgboostlss.py --horizon 14 --quick
```

Outputs trained models to `models/`, evaluation metrics to `models/all_metrics.csv`, and diagnostic visualizations to `visualizations/`.

## Model

XGBoostLSS with a Gamma distribution, tuned via Optuna (30 trials, 3-fold CV). Predicts concentration and rate parameters, giving full probability distributions rather than point forecasts.

- **Horizons**: 3, 7, 14, 21, 28, 35 days
- **Train**: 2000-2023 | **Validation**: 2024 | **Test**: 2025
- **Features**: 125+ engineered features across forecast, climatology, recent conditions, calendar, bias correction, and cross-feature categories

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for complete feature documentation and model details.

### 14-Day Horizon Results

| Metric | Validation | Test |
|--------|-----------|------|
| CRPS | 26.5 | 21.1 |
| MAE (mm) | 38.4 | 30.6 |
| Bias (mm) | +8.9 | +8.2 |
| 90% Coverage | 67.2% | 72.6% |
| 50% Coverage | 31.1% | 32.9% |
| Mean Observed (mm) | 67.5 | 56.7 |
| Mean Predicted (mm) | 76.4 | 64.9 |
