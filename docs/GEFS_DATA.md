# GEFS Data Collection Plan

This document outlines the data collection strategy for pulling GEFS (Global Ensemble Forecast System) reforecast and operational data using Herbie.

## Overview

We are collecting ensemble forecast data from three sources:
1. **GEFS Reforecast - Weekly 35-day** (Wednesdays, 2000-2019)
2. **GEFS Reforecast - Daily 16-day** (Daily 00 UTC, 2000-2019)
3. **GEFS Operational - Daily 35-day** (Daily 00 UTC, Sept 2020-present)

All data is extracted for **Miami International Airport** coordinates: **25.8°N, 80.3°W**

---

## Data Sources

### 1. GEFS Reforecast - Weekly 35-day Forecasts

| Attribute | Value |
|-----------|-------|
| Model | `gefs_reforecast` |
| Time Period | 2000-01-05 to 2019-12-25 (Wednesdays only) |
| Initialization | 00 UTC |
| Ensemble Members | 11 (control + 10 perturbed) |
| Max Forecast Hour | 840 hours (35 days) |
| Approx. Samples | ~1,040 (52 weeks × 20 years) |

### 2. GEFS Reforecast - Daily 16-day Forecasts

| Attribute | Value |
|-----------|-------|
| Model | `gefs_reforecast` |
| Time Period | 2000-01-01 to 2019-12-31 (daily) |
| Initialization | 00 UTC |
| Ensemble Members | 5 (control + 4 perturbed) |
| Max Forecast Hour | 384 hours (16 days) |
| Approx. Samples | ~7,300 (365 days × 20 years) |

### 3. GEFS Operational - Daily 35-day Forecasts

| Attribute | Value |
|-----------|-------|
| Model | `gefs` |
| Product | `atmos.5` (primary) / `atmos.5b` (secondary) |
| Time Period | 2020-09-23 to present |
| Initialization | 00 UTC (only cycle with 35-day forecasts) |
| Ensemble Members | 31 (control + 30 perturbed) |
| Max Forecast Hour | 840 hours (35 days) |
| Approx. Samples | ~1,970+ (growing daily) |

---

## Variables to Pull

For each variable, we pull **all ensemble members** and compute **mean** and **spread** (standard deviation).

### Precipitation & Moisture

| Variable | Search Pattern | Description | Units |
|----------|----------------|-------------|-------|
| `apcp` | `:APCP:surface:` | Accumulated precipitation (0 to forecast hour) | kg/m² |
| `pwat` | `:PWAT:` | Precipitable water (entire atmosphere) | kg/m² |
| `rh_2m` | `:RH:2 m above ground:` | Relative humidity at 2m | % |
| `rh_850` | `:RH:850 mb:` | Relative humidity at 850mb | % |
| `rh_500` | `:RH:500 mb:` | Relative humidity at 500mb | % |

### Convective Parameters

| Variable | Search Pattern | Description | Units |
|----------|----------------|-------------|-------|
| `cape` | `:CAPE:surface:` | Convective Available Potential Energy | J/kg |
| `cin` | `:CIN:surface:` | Convective Inhibition | J/kg |

### Temperature

| Variable | Search Pattern | Description | Units |
|----------|----------------|-------------|-------|
| `tmp_2m` | `:TMP:2 m above ground:` | Temperature at 2m | K |
| `tmp_850` | `:TMP:850 mb:` | Temperature at 850mb | K |
| `tmp_500` | `:TMP:500 mb:` | Temperature at 500mb | K |

### Wind - Surface

| Variable | Search Pattern | Description | Units |
|----------|----------------|-------------|-------|
| `ugrd_10m` | `:UGRD:10 m above ground:` | U-component of wind at 10m | m/s |
| `vgrd_10m` | `:VGRD:10 m above ground:` | V-component of wind at 10m | m/s |
| `gust` | `:GUST:surface:` | Wind gust | m/s |

### Wind - Pressure Levels

| Variable | Search Pattern | Description | Units |
|----------|----------------|-------------|-------|
| `ugrd_850` | `:UGRD:850 mb:` | U-component of wind at 850mb | m/s |
| `vgrd_850` | `:VGRD:850 mb:` | V-component of wind at 850mb | m/s |
| `ugrd_500` | `:UGRD:500 mb:` | U-component of wind at 500mb | m/s |
| `vgrd_500` | `:VGRD:500 mb:` | V-component of wind at 500mb | m/s |

### Clouds & Radiation

| Variable | Search Pattern | Description | Units |
|----------|----------------|-------------|-------|
| `tcdc` | `:TCDC:` | Total cloud cover | % |
| `dswrf` | `:DSWRF:surface:` | Downward shortwave radiation flux | W/m² |
| `dlwrf` | `:DLWRF:surface:` | Downward longwave radiation flux | W/m² |

### Dynamics & Pressure

| Variable | Search Pattern | Description | Units |
|----------|----------------|-------------|-------|
| `pres_sfc` | `:PRES:surface:` | Surface pressure | Pa |
| `pres_msl` | `:PRMSL:` | Mean sea level pressure | Pa |
| `hgt_850` | `:HGT:850 mb:` | Geopotential height at 850mb | gpm |
| `hgt_500` | `:HGT:500 mb:` | Geopotential height at 500mb | gpm |

### Heat Fluxes

| Variable | Search Pattern | Description | Units |
|----------|----------------|-------------|-------|
| `lhtfl` | `:LHTFL:surface:` | Latent heat flux | W/m² |
| `shtfl` | `:SHTFL:surface:` | Sensible heat flux | W/m² |

**Total: 26 variables × 2 (mean + spread) = 52 columns per row**

---

## Forecast Hours to Pull

We pull **cumulative forecasts** at **7 key time horizons**:

| Forecast Hour | Time Horizon | Available In |
|---------------|--------------|--------------|
| 24 | 1 day | All datasets |
| 72 | 3 days | All datasets |
| 168 | 1 week | All datasets |
| 336 | 2 weeks | All datasets |
| 504 | 3 weeks | 35-day only (Wed reforecast + operational) |
| 672 | 4 weeks | 35-day only (Wed reforecast + operational) |
| 840 | 5 weeks | 35-day only (Wed reforecast + operational) |

**Total: 7 forecast hours per initialization date**

---

## Data Aggregation Strategy

### Ensemble Statistics

For each variable and forecast hour, we:
1. Pull all ensemble members (31 for operational, 11 for Wed reforecast, 5 for daily reforecast)
2. Compute **mean** across all members
3. Compute **spread** (standard deviation) across all members

### Point Extraction

Data is extracted for Miami International Airport:
- **Latitude**: 25.8°N
- **Longitude**: 80.3°W (or 279.7°E in 0-360 convention)

Using xarray's `sel()` method with nearest neighbor interpolation.

---

## Storage Format

All data is stored as **Parquet files** for efficient compression and fast reads.

### Directory Structure

```
data/
├── GEFS_Operational/           # Daily 35-day forecasts (2020-present)
│   ├── gefs_operational_2021.parquet
│   ├── gefs_operational_2022.parquet
│   ├── gefs_operational_2023.parquet
│   ├── gefs_operational_2024.parquet
│   └── gefs_operational_2025.parquet
│
├── GEFS_Reforecast_35day/      # Weekly 35-day forecasts (Wednesdays 2000-2019)
│   ├── gefs_reforecast_35day_2000.parquet
│   ├── gefs_reforecast_35day_2001.parquet
│   └── ...
│
└── GEFS_Reforecast_16day/      # Daily 16-day forecasts (2000-2019)
    ├── gefs_reforecast_16day_2000.parquet
    ├── gefs_reforecast_16day_2001.parquet
    └── ...
```

### Parquet Schema

Each row represents one initialization date and forecast hour combination:

| Column | Type | Description |
|--------|------|-------------|
| `init_date` | datetime | Forecast initialization date (00 UTC) |
| `forecast_hour` | int | Hours from initialization (24, 48, ..., 840) |
| `valid_date` | datetime | Target date (init_date + forecast_hour) |
| `apcp_mean` | float | Ensemble mean of accumulated precipitation |
| `apcp_spread` | float | Ensemble spread of accumulated precipitation |
| `cape_mean` | float | Ensemble mean of CAPE |
| `cape_spread` | float | Ensemble spread of CAPE |
| ... | ... | (all other variables with _mean and _spread suffixes) |

---

## Scripts

| Script | Description | Status |
|--------|-------------|--------|
| `scripts/pull_gefs_operational.py` | Pull GEFS operational 35-day data | Ready |
| `scripts/pull_gefs_reforecast_35day.py` | Pull GEFS reforecast weekly 35-day data | TODO |
| `scripts/pull_gefs_reforecast_16day.py` | Pull GEFS reforecast daily 16-day data | TODO |

### Usage

```bash
# Pull 1 year of GEFS Operational data
python scripts/pull_gefs_operational.py --year 2024

# Pull custom date range with 10 parallel workers
python scripts/pull_gefs_operational.py --start-date 2024-01-01 --end-date 2024-06-30 --workers 10
```

---

## Technical Details

### Retry Logic

All scripts use exponential backoff with jitter:
- Max retries: 5
- Base delay: 1 second
- Max delay: 60 seconds
- Jitter: ±10% of delay

### Parallelization

- Default: 10 parallel workers (configurable via `--workers`)
- Each worker processes one initialization date at a time
- Within each date, forecast hours and variables are processed sequentially

---

## Estimated Data Sizes

| Dataset | Rows per Year | Columns | Est. Size |
|---------|---------------|---------|-----------|
| Operational (35-day) | 365 × 7 = 2,555 | 55 | ~0.5-1 MB |
| Reforecast 35-day (Wed) | 52 × 7 = 364 | 55 | ~0.1-0.2 MB |
| Reforecast 16-day (daily) | 365 × 4 = 1,460 | 55 | ~0.3-0.6 MB |

Note: Daily 16-day reforecast only has 4 of the 7 time horizons (1d, 3d, 1w, 2w) since it maxes out at 384 hours.

---

## Notes

- GEFS uses 0-360° longitude convention (Miami = 279.7°E)
- All precipitation values are **cumulative from forecast hour 0** (not daily increments)
- Extended forecasts (>384 hours) only available from:
  - Wednesday reforecast runs (2000-2019)
  - 00 UTC operational runs (Sept 2020-present)
- GEFS Operational started Sept 23, 2020 with GEFSv12
