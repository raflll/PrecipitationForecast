# Model Card: XGBoostLSS Precipitation Forecasting

## Overview

| Property | Value |
|----------|-------|
| Model | XGBoostLSS (distributional gradient boosting) |
| Distribution | Gamma (concentration and rate parameters) |
| Stabilization | MAD (Median Absolute Deviation) |
| Response function | Exponential |
| Loss function | Negative log-likelihood |
| Target | Cumulative observed precipitation (mm) over forecast horizon |
| Location | Miami International Airport (USW00012839, 25.79°N 80.32°W) |
| Horizons | 3, 7, 14, 21, 28, 35 days |

The model predicts full Gamma probability distributions rather than point forecasts, outputting concentration (shape) and rate parameters from which means, quantiles, and prediction intervals are derived analytically.

## Target Variable

`ground_truth_precip` — sum of daily NOAA PRCP (mm) from `init_date` through `init_date + horizon - 1`, plus an epsilon of 0.1 mm for Gamma distribution compatibility (Gamma requires strictly positive values).

## Data Split

| Split | Period | Description |
|-------|--------|-------------|
| Train | 2000-2023 | GEFS reforecast (2000-2019, Wednesdays) + operational (Sept 2020-2023, daily) |
| Validation | 2024 | Operational forecasts, used for early stopping and hyperparameter selection |
| Test | 2025 | Operational forecasts, held out for final evaluation |

## Feature Catalog

All features are engineered in `scripts/data_processing/build_training_dataset.py`. Features are organized into 8 groups.

### 1. Forecast Features (from GEFS ensemble data)

Computed from 6-hourly GEFS forecast steps within each horizon window (forecast hours 6 through `horizon × 24`).

#### Primary Accumulations

| Feature | Description |
|---------|-------------|
| `cum_apcp_mean` | Sum of GEFS ensemble mean precipitation across all 6h steps in the horizon |
| `cum_apcp_spread_linear` | Sum of ensemble spreads (linear combination) |
| `cum_apcp_spread_rss` | Root sum of squares of ensemble spreads |
| `cum_apcp_mean_per_day` | `cum_apcp_mean / horizon_days` (daily rate from forecast) |
| `cum_apcp_mean_log` | `log(1 + cum_apcp_mean)` |
| `cum_apcp_mean_sqrt` | `sqrt(cum_apcp_mean)` |
| `cum_apcp_spread_log` | `log(1 + cum_apcp_spread_rss)` |
| `cum_spread_to_mean_ratio` | `cum_apcp_spread_rss / (cum_apcp_mean + 0.1)` |
| `cum_spread_per_day` | `cum_apcp_spread_rss / horizon_days` |
| `spread_mean_product` | `cum_apcp_mean × cum_apcp_spread_rss` (interaction term) |

#### 6-Hourly Rate Statistics

| Feature | Description |
|---------|-------------|
| `mean_6h_rate` | Mean of all 6h GEFS steps |
| `max_6h_rate` | Maximum 6h step value |
| `min_6h_rate` | Minimum 6h step value |
| `std_6h_rate` | Standard deviation of 6h steps |
| `median_6h_rate` | Median 6h step value |
| `forecast_cv` | Coefficient of variation: `std / mean_rate` |
| `forecast_skew` | Skewness of 6h rate distribution |
| `forecast_p90_6h` | 90th percentile of 6h steps |
| `forecast_p10_6h` | 10th percentile of 6h steps |
| `forecast_range_6h` | `p90 - p10` |
| `forecast_iqr_6h` | `p75 - p25` of 6h steps |

#### Dry/Wet Patterns

| Feature | Description |
|---------|-------------|
| `frac_dry_steps` | Fraction of 6h steps with < 0.01 mm |
| `frac_heavy_steps` | Fraction of 6h steps with > 2.0 mm |
| `longest_dry_run` | Max consecutive dry 6h steps |
| `longest_wet_run` | Max consecutive wet 6h steps |
| `num_wet_spells` | Count of distinct wet periods |

#### Temporal Distribution (Quartiles)

The forecast window is divided into 4 equal quartiles by time.

| Feature | Description |
|---------|-------------|
| `q1_apcp_mean` | Sum of GEFS mean precip in 1st quartile of forecast window |
| `q2_apcp_mean` | Sum in 2nd quartile |
| `q3_apcp_mean` | Sum in 3rd quartile |
| `q4_apcp_mean` | Sum in 4th quartile |
| `q1_apcp_spread` | Sum of ensemble spread in 1st quartile |
| `q2_apcp_spread` | Sum of spread in 2nd quartile |
| `q3_apcp_spread` | Sum of spread in 3rd quartile |
| `q4_apcp_spread` | Sum of spread in 4th quartile |
| `front_half_ratio` | `sum(first half) / cum_mean` — is precip front-loaded? |
| `front_back_diff` | `sum(first half) - sum(second half)` |

#### Spread and Trend Features

| Feature | Description |
|---------|-------------|
| `max_spread` | Maximum ensemble spread across all 6h steps |
| `mean_spread_ratio` | Mean of `spread / (mean + 0.01)` across steps |
| `spread_trend` | Linear trend (slope) of spreads over forecast window |
| `mean_trend` | Linear trend (slope) of mean precip over forecast window |
| `top10pct_frac` | Fraction of total precip in the top 10% of steps (peakiness) |

#### Metadata

| Feature | Description |
|---------|-------------|
| `source` | "reforecast" or "operational" (used as metadata, not as a model feature) |

### 2. Climatological Features

Computed from historical NOAA observations within a ±15 day window around the forecast initialization day-of-year. For each (DOY, horizon), aggregates all matching H-day cumulative precipitation windows across all available years.

| Feature | Description |
|---------|-------------|
| `clim_mean_precip` | Mean cumulative H-day precipitation for this DOY window |
| `clim_median_precip` | Median cumulative H-day precipitation |
| `clim_std_precip` | Standard deviation of H-day precipitation |
| `clim_pct_rainy` | Fraction of days with precipitation > 0 in the window |
| `clim_pct_heavy` | Fraction of days with precipitation > 25 mm |
| `clim_mean_thunder_days` | Average count of thunderstorm days (WT03) in the window |
| `clim_mean_fog_days` | Average count of fog days (WT01) |
| `clim_mean_rain_flag_days` | Average count of rain-flagged days (WT16) |
| `clim_p10` | 10th percentile of historical cumulative precipitation |
| `clim_p25` | 25th percentile |
| `clim_p75` | 75th percentile |
| `clim_p90` | 90th percentile |
| `clim_iqr` | `p75 - p25` |
| `clim_pct_zero` | Fraction of windows with zero precipitation |
| `clim_gamma_shape` | Shape parameter (α) of Gamma distribution fit to positive historical values |
| `clim_gamma_scale` | Scale parameter (β) of Gamma distribution fit |

### 3. Recent Conditions Features

Computed from observed weather in the days immediately preceding each `init_date`.

#### Precipitation Recency

| Feature | Description |
|---------|-------------|
| `precip_last_1d` | Cumulative precipitation in past 1 day |
| `precip_last_3d` | Cumulative precipitation in past 3 days |
| `precip_last_7d` | Cumulative precipitation in past 7 days |
| `precip_last_14d` | Cumulative precipitation in past 14 days |
| `precip_last_30d` | Cumulative precipitation in past 30 days |

#### Rainy Day Counts

| Feature | Description |
|---------|-------------|
| `rainy_days_last_7d` | Count of days with precipitation > 0 in past 7 days |
| `rainy_days_last_14d` | Count in past 14 days |
| `rainy_days_last_30d` | Count in past 30 days |

#### Max Precipitation

| Feature | Description |
|---------|-------------|
| `max_precip_last_7d` | Maximum single-day precipitation in past 7 days |
| `max_precip_last_14d` | Maximum in past 14 days |

#### Intensity and Trends

| Feature | Description |
|---------|-------------|
| `precip_intensity_14d` | Average rain per rainy day in past 14 days (0 if no rainy days) |
| `precip_trend_7v14` | `precip_last_7d - (precip_last_14d - precip_last_7d)` — is last 7d wetter than prior 7d? |
| `days_since_rain` | Days since last measurable precipitation (capped at 60) |

#### Lagged Precipitation

| Feature | Description |
|---------|-------------|
| `precip_lag_1d` | Precipitation on day before init_date |
| `precip_lag_2d` | Precipitation 2 days before |
| `precip_lag_3d` | 3 days before |
| `precip_lag_7d` | 7 days before |
| `precip_lag_14d` | 14 days before |

#### Temperature

| Feature | Description |
|---------|-------------|
| `mean_temp_last_7d` | Mean of `(TMAX + TMIN) / 2` over past 7 days |
| `mean_temp_last_14d` | Mean over past 14 days |
| `temp_anomaly_last_7d` | `mean_temp_last_7d - climatological_mean_temp_for_DOY` |
| `temp_trend_7v14` | `mean_temp_last_7d - mean_temp_last_14d` |
| `mean_dtr_last_7d` | Mean diurnal temperature range (`TMAX - TMIN`) over past 7 days |

#### Wind and Sunshine

| Feature | Description |
|---------|-------------|
| `mean_wind_last_7d` | Mean of Open-Meteo `wind_speed_10m_max` over past 7 days |
| `mean_gusts_last_7d` | Mean of Open-Meteo `wind_gusts_10m_max` over past 7 days |
| `mean_sunshine_last_7d` | Mean of Open-Meteo `sunshine_duration` over past 7 days |
| `sunshine_anomaly_7d` | `mean_sunshine_last_7d - climatological_mean_sunshine_for_DOY` |
| `mean_precip_hours_last_7d` | Mean of Open-Meteo `precipitation_hours` over past 7 days |

#### Weather Type Counts

| Feature | Description |
|---------|-------------|
| `thunder_last_7d` | Count of thunderstorm days (NOAA WT03) in past 7 days |
| `thunder_last_14d` | Count in past 14 days |
| `thunder_last_30d` | Count in past 30 days |
| `fog_last_7d` | Count of fog days (WT01) in past 7 days |
| `rain_flag_last_7d` | Count of rain-flagged days (WT16) in past 7 days |
| `haze_last_7d` | Count of haze/smoke days (WT08) in past 7 days |

### 4. Calendar / Seasonal Features

| Feature | Description |
|---------|-------------|
| `day_of_year_sin` | `sin(2π × DOY / 365.25)` — circular encoding of seasonality |
| `day_of_year_cos` | `cos(2π × DOY / 365.25)` |
| `day_of_year_sin2` | `sin(4π × DOY / 365.25)` — higher harmonic for finer patterns |
| `day_of_year_cos2` | `cos(4π × DOY / 365.25)` |
| `month` | Calendar month (1-12) |
| `is_wet_season` | 1 if June-October, 0 otherwise |
| `is_hurricane_season` | 1 if June-November |
| `is_peak_wet` | 1 if August-October |

### 5. Cross Features (Forecast vs Climatology / Recent Conditions)

Computed after merging forecast, climatological, and recent condition features.

#### Forecast vs Climatology

| Feature | Description |
|---------|-------------|
| `forecast_clim_ratio` | `cum_apcp_mean / (clim_mean_precip + 0.1)` |
| `forecast_clim_anomaly` | `cum_apcp_mean - clim_mean_precip` |
| `forecast_vs_clim_median` | `cum_apcp_mean - clim_median_precip` |
| `forecast_clim_percentile` | `(cum_apcp_mean - clim_mean_precip) / clim_std_precip` — z-score |
| `forecast_gamma_cdf` | CDF value of forecast under historical Gamma distribution |
| `spread_vs_clim_std` | `cum_apcp_spread_rss / (clim_std_precip + 0.1)` |

#### Forecast vs Recent Conditions

| Feature | Description |
|---------|-------------|
| `forecast_vs_recent_7d` | `(cum_apcp_mean / horizon) - (precip_last_7d / 7)` — daily rate difference |

#### Persistence Features

Naive baseline: extrapolate recent daily precipitation rate over the full horizon.

| Feature | Description |
|---------|-------------|
| `persistence_7d` | `(precip_last_7d / 7) × horizon_days` |
| `persistence_14d` | `(precip_last_14d / 14) × horizon_days` |
| `persistence_30d` | `(precip_last_30d / 30) × horizon_days` |
| `forecast_vs_persistence_7d` | `cum_apcp_mean - persistence_7d` |
| `forecast_vs_persistence_14d` | `cum_apcp_mean - persistence_14d` |
| `forecast_vs_persistence_30d` | `cum_apcp_mean - persistence_30d` |
| `persistence_ratio_7d` | `cum_apcp_mean / (persistence_7d + 0.1)` |

#### Recent Precipitation Anomaly

| Feature | Description |
|---------|-------------|
| `recent_precip_anomaly_7d` | `(precip_last_7d / 7) / (clim_daily_rate + 0.1)` |
| `recent_precip_anomaly_30d` | `(precip_last_30d / 30) / (clim_daily_rate + 0.1)` |

### 6. GEFS Bias Features (DOY-based)

For each (init_date, horizon), finds the 30 nearest day-of-year matches from **strictly previous years** (year < current year). Computes GEFS accuracy statistics from those neighbors. This avoids within-year temporal contamination.

| Feature | Description |
|---------|-------------|
| `gefs_bias_doy` | Mean error: `GEFS_mean - observed` across 30 nearest DOY neighbors |
| `gefs_bias_doy_std` | Standard deviation of those errors |
| `gefs_mae_doy` | Mean absolute error |
| `gefs_ratio_doy` | Mean of `GEFS / (observed + 0.1)` |
| `gefs_bias_doy_n` | Number of neighbors used (up to 30) |
| `gefs_overpredict_rate_doy` | Fraction of neighbors where GEFS > observed |
| `gefs_direction_accuracy_doy` | Fraction where GEFS correctly predicts above/below median |
| `gefs_bias_high_forecast_doy` | Mean error when GEFS predicted above median (conditional bias) |
| `gefs_bias_low_forecast_doy` | Mean error when GEFS predicted below median |

### 7. Rolling GEFS Accuracy Features

For each (init_date, horizon), uses same-horizon forecasts from the previous N days to measure recent GEFS performance. Requires at least 5 matching forecasts.

| Feature | Description |
|---------|-------------|
| `gefs_rolling_bias_30d` | Mean error over past 30 days |
| `gefs_rolling_mae_30d` | Mean absolute error over past 30 days |
| `gefs_rolling_rmse_30d` | Root mean squared error over past 30 days |
| `gefs_rolling_bias_60d` | Mean error over past 60 days |
| `gefs_rolling_mae_60d` | MAE over past 60 days |
| `gefs_rolling_rmse_60d` | RMSE over past 60 days |
| `gefs_rolling_bias_90d` | Mean error over past 90 days |
| `gefs_rolling_mae_90d` | MAE over past 90 days |
| `gefs_rolling_rmse_90d` | RMSE over past 90 days |

### 8. Bias-Corrected Features

| Feature | Description |
|---------|-------------|
| `forecast_bias_corrected` | `cum_apcp_mean - gefs_bias_doy` (subtract historical DOY bias) |
| `spread_adequacy` | `cum_apcp_spread_rss / (gefs_mae_doy + 0.1)` — is ensemble spread proportionate to actual error? |
| `bias_corrected_vs_clim` | `forecast_bias_corrected / (clim_mean_precip + 0.1)` |
| `bias_corrected_anomaly` | `forecast_bias_corrected - clim_mean_precip` |
| `forecast_vs_gefs_bias_std` | `(cum_apcp_mean - gefs_bias_doy - clim_mean_precip) / gefs_bias_doy_std` |
| `rolling_bias_x_forecast` | `gefs_rolling_bias_30d × cum_apcp_mean` (interaction: recent bias × current forecast) |

### Additional Metadata Columns

| Column | Description |
|--------|-------------|
| `is_reforecast` | 1 if source is reforecast, 0 if operational |

## Hyperparameter Search Space

Optimized via Optuna Bayesian optimization with 3-fold cross-validation.

| Parameter | Type | Range | Scale |
|-----------|------|-------|-------|
| `eta` (learning rate) | float | 0.001 - 0.2 | log |
| `max_depth` | int | 2 - 5 | linear |
| `gamma` (min split loss) | float | 1e-8 - 40.0 | log |
| `subsample` | float | 0.5 - 0.9 | linear |
| `colsample_bytree` | float | 0.3 - 0.7 | linear |
| `min_child_weight` | float | 10.0 - 500 | log |
| `reg_alpha` (L1) | float | 0.01 - 50.0 | log |
| `reg_lambda` (L2) | float | 0.01 - 50.0 | log |
| `booster` | categorical | gbtree | — |

Standard tuning: 30 trials, 15-minute timeout, 500 boosting rounds, early stopping at 20 rounds.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| CRPS | Continuous Ranked Probability Score (analytical for Gamma) — measures calibration and sharpness |
| NLL | Negative log-likelihood of observed values under predicted Gamma |
| MAE | Mean absolute error of predicted mean vs observed |
| Bias | Mean signed error (predicted mean - observed) |
| Median AE | Median absolute error |
| Coverage (50-95%) | Fraction of observations within the X% prediction interval |

## 14-Day Horizon Results

| Metric | Validation (2024) | Test (2025) |
|--------|-------------------|-------------|
| CRPS | 26.5 | 21.1 |
| NLL | 5.39 | 4.73 |
| MAE (mm) | 38.4 | 30.6 |
| Bias (mm) | +8.9 | +8.2 |
| Median AE (mm) | 27.8 | 18.1 |
| 50% Coverage | 31.1% | 32.9% |
| 60% Coverage | 38.0% | 41.4% |
| 70% Coverage | 44.5% | 51.0% |
| 80% Coverage | 53.8% | 60.3% |
| 90% Coverage | 67.2% | 72.6% |
| 95% Coverage | 77.3% | 78.9% |
| Mean Observed (mm) | 67.5 | 56.7 |
| Mean Predicted (mm) | 76.4 | 64.9 |
| n_samples | 366 | 365 |

## Diagnostic Visualizations

For each horizon, `src/train_xgboostlss.py` generates:

| Plot | Filename | Description |
|------|----------|-------------|
| Predicted vs Observed | `xgboostlss_{h}d_pred_vs_obs.png` | Scatter plot of predicted mean vs observed with perfect-prediction diagonal |
| Prediction Intervals | `xgboostlss_{h}d_intervals.png` | First 100 test samples with 90% and 50% prediction intervals |
| Feature Importance | `xgboostlss_{h}d_importance.png` | Top 25 features by gain, color-coded by category |
| SHAP Beeswarm | `xgboostlss_{h}d_shap_beeswarm.png` | SHAP values for concentration parameter, top 25 features |
| SHAP Bar | `xgboostlss_{h}d_shap_bar.png` | Mean |SHAP| for concentration parameter |
| SHAP Dependence | `xgboostlss_{h}d_shap_dependence.png` | 2×2 grid of top 4 features |
| SHAP Rate | `xgboostlss_{h}d_shap_rate.png` | SHAP beeswarm for rate parameter |
| PIT Histogram | `xgboostlss_{h}d_pit.png` | Probability Integral Transform — should be uniform if calibrated |
| Reliability Diagram | `xgboostlss_{h}d_reliability.png` | Nominal vs actual coverage at 10 confidence levels |
