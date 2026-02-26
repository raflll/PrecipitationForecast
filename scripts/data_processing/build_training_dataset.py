#!/usr/bin/env python3
"""
Build the training dataset for XGBoostLSS precipitation forecasting.

Combines GEFS forecast data (reforecast + operational) with historical features
from miami_clean.csv to create a single training parquet file. Each row represents
one (init_date, horizon) combination with all features and ground truth.

Usage:
    python scripts/data_processing/build_training_dataset.py

Output:
    data/training/training_dataset.parquet
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma as sp_gamma

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "training_dataset.parquet"

HORIZONS = [3, 7, 14, 21, 28, 35]
EPSILON = 0.1  # mm, added to ground truth for gamma distribution compatibility


# ---------------------------------------------------------------------------
# 1. Load all source data
# ---------------------------------------------------------------------------

def load_miami_clean() -> pd.DataFrame:
    """Load and prepare miami_clean.csv."""
    path = DATA_DIR / "miami_clean.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["avg_temp"] = (df["noaa_TMAX"] + df["noaa_TMIN"]) / 2.0
    df["dtr"] = df["noaa_TMAX"] - df["noaa_TMIN"]  # diurnal temperature range
    df["doy"] = df["date"].dt.dayofyear
    return df


def load_gefs_data() -> pd.DataFrame:
    """Load all GEFS parquet files (reforecast + operational), tag with source."""
    frames = []

    reforecast_dir = DATA_DIR / "GEFS_Reforecast"
    for f in sorted(reforecast_dir.glob("gefs_reforecast_*.parquet")):
        df = pd.read_parquet(f)
        df["source"] = "reforecast"
        frames.append(df)

    operational_dir = DATA_DIR / "GEFS_Operational"
    for f in sorted(operational_dir.glob("gefs_operational_*.parquet")):
        if f.suffix != ".parquet":
            continue
        df = pd.read_parquet(f)
        df["source"] = "operational"
        frames.append(df)

    if not frames:
        print("ERROR: No GEFS parquet files found.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined["init_date"] = pd.to_datetime(combined["init_date"])
    combined = combined.sort_values(["init_date", "forecast_hour"]).reset_index(drop=True)
    print(f"  Loaded {len(combined):,} GEFS rows, "
          f"{combined['init_date'].nunique()} unique init dates "
          f"({combined['init_date'].min().date()} to {combined['init_date'].max().date()})")
    return combined


# ---------------------------------------------------------------------------
# 2. Ground truth computation
# ---------------------------------------------------------------------------

def compute_ground_truths(miami: pd.DataFrame, init_dates: pd.Series) -> pd.DataFrame:
    """Compute cumulative observed precipitation for each (init_date, horizon)."""
    miami_indexed = miami.set_index("date")["noaa_PRCP"]

    rows = []
    for init_date in init_dates:
        for h in HORIZONS:
            end_date = init_date + pd.Timedelta(days=h - 1)
            date_range = pd.date_range(init_date, end_date, freq="D")
            if not all(d in miami_indexed.index for d in date_range):
                continue
            total = miami_indexed.loc[date_range].sum() + EPSILON
            rows.append({
                "init_date": init_date,
                "horizon_days": h,
                "ground_truth_precip": total,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Forecast features (significantly expanded)
# ---------------------------------------------------------------------------

def compute_forecast_features(gefs: pd.DataFrame, init_dates: pd.Series) -> pd.DataFrame:
    """Compute cumulative forecast features for each (init_date, horizon)."""
    gefs_grouped = {
        dt: grp.sort_values("forecast_hour")
        for dt, grp in gefs.groupby("init_date")
    }

    rows = []
    for init_date in init_dates:
        if init_date not in gefs_grouped:
            continue
        grp = gefs_grouped[init_date]
        source = grp["source"].iloc[0]

        for h in HORIZONS:
            max_hour = h * 24
            mask = (grp["forecast_hour"] >= 6) & (grp["forecast_hour"] <= max_hour)
            subset = grp.loc[mask]

            if subset.empty:
                continue

            means = subset["apcp_mean"].values
            spreads = subset["apcp_spread"].values
            num_steps = len(means)

            cum_mean = float(np.nansum(means))
            cum_spread_linear = float(np.nansum(spreads))
            cum_spread_rss = float(np.sqrt(np.nansum(spreads ** 2)))

            row = {
                "init_date": init_date,
                "horizon_days": h,
                "source": source,
                # --- Primary accumulations ---
                "cum_apcp_mean": cum_mean,
                "cum_apcp_spread_linear": cum_spread_linear,
                "cum_apcp_spread_rss": cum_spread_rss,
                # --- Rate statistics ---
                "mean_6h_rate": cum_mean / num_steps if num_steps > 0 else 0.0,
                "max_6h_rate": float(np.nanmax(means)) if num_steps > 0 else 0.0,
                "min_6h_rate": float(np.nanmin(means)) if num_steps > 0 else 0.0,
                "std_6h_rate": float(np.nanstd(means)) if num_steps > 0 else 0.0,
                "median_6h_rate": float(np.nanmedian(means)) if num_steps > 0 else 0.0,
                # --- Dry/wet pattern ---
                "frac_dry_steps": float(np.mean(means < 0.01)) if num_steps > 0 else 1.0,
                "frac_heavy_steps": float(np.mean(means > 2.0)) if num_steps > 0 else 0.0,
                "longest_dry_run": _longest_run(means < 0.01),
                "longest_wet_run": _longest_run(means >= 0.01),
                "num_wet_spells": _count_spells(means >= 0.01),
            }

            # --- Temporal distribution (quartiles of the forecast window) ---
            q_size = max(1, num_steps // 4)
            for qi in range(4):
                start_idx = qi * q_size
                end_idx = min((qi + 1) * q_size, num_steps)
                q_means = means[start_idx:end_idx]
                q_spreads = spreads[start_idx:end_idx]
                row[f"q{qi+1}_apcp_mean"] = float(np.nansum(q_means))
                row[f"q{qi+1}_apcp_spread"] = float(np.nansum(q_spreads))

            # Front/back loading
            half = num_steps // 2
            front_sum = float(np.nansum(means[:half]))
            back_sum = float(np.nansum(means[half:]))
            row["front_half_ratio"] = front_sum / cum_mean if cum_mean > 0 else 0.5
            row["front_back_diff"] = front_sum - back_sum

            # --- Spread features ---
            row["max_spread"] = float(np.nanmax(spreads)) if num_steps > 0 else 0.0
            row["mean_spread_ratio"] = float(np.nanmean(spreads / (means + 0.01)))
            row["spread_trend"] = _linear_trend(spreads)
            row["mean_trend"] = _linear_trend(means)

            # --- Forecast variability ---
            if cum_mean > 0:
                row["forecast_cv"] = float(np.nanstd(means)) / (cum_mean / num_steps)
            else:
                row["forecast_cv"] = 0.0

            # Skewness of 6h forecasts
            if num_steps > 2 and np.nanstd(means) > 0:
                row["forecast_skew"] = float(
                    np.nanmean(((means - np.nanmean(means)) / np.nanstd(means)) ** 3)
                )
            else:
                row["forecast_skew"] = 0.0

            # --- More cumulative total features ---
            row["cum_apcp_mean_per_day"] = cum_mean / h  # daily rate from forecast
            row["cum_apcp_mean_log"] = float(np.log1p(cum_mean))
            row["cum_apcp_mean_sqrt"] = float(np.sqrt(cum_mean))
            row["cum_spread_to_mean_ratio"] = cum_spread_rss / (cum_mean + 0.1)
            row["cum_spread_per_day"] = cum_spread_rss / h
            # Percentile rank of each 6h step within the forecast
            if num_steps > 1:
                pct_90 = float(np.nanpercentile(means, 90))
                pct_10 = float(np.nanpercentile(means, 10))
                row["forecast_p90_6h"] = pct_90
                row["forecast_p10_6h"] = pct_10
                row["forecast_range_6h"] = pct_90 - pct_10
                row["forecast_iqr_6h"] = float(
                    np.nanpercentile(means, 75) - np.nanpercentile(means, 25)
                )
            else:
                row["forecast_p90_6h"] = 0.0
                row["forecast_p10_6h"] = 0.0
                row["forecast_range_6h"] = 0.0
                row["forecast_iqr_6h"] = 0.0
            # Total spread features
            row["cum_apcp_spread_log"] = float(np.log1p(cum_spread_rss))
            row["spread_mean_product"] = cum_mean * cum_spread_rss  # interaction
            # Forecast "peakiness" - how concentrated is precip in few steps
            if cum_mean > 0:
                sorted_means = np.sort(means)[::-1]
                top_10_pct = max(1, num_steps // 10)
                row["top10pct_frac"] = float(np.sum(sorted_means[:top_10_pct]) / cum_mean)
            else:
                row["top10pct_frac"] = 0.0

            rows.append(row)

    return pd.DataFrame(rows)


def _longest_run(condition: np.ndarray) -> int:
    """Length of the longest consecutive run of True in a boolean array."""
    if len(condition) == 0:
        return 0
    runs = np.diff(np.where(np.concatenate(([condition[0]], condition[:-1] != condition[1:], [True])))[0])
    if len(runs) == 0:
        return 0
    # runs alternate between True-runs and False-runs; first run matches condition[0]
    if condition[0]:
        return int(np.max(runs[::2])) if len(runs[::2]) > 0 else 0
    else:
        return int(np.max(runs[1::2])) if len(runs[1::2]) > 0 else 0


def _count_spells(condition: np.ndarray) -> int:
    """Count the number of distinct True spells in a boolean array."""
    if len(condition) == 0:
        return 0
    diffs = np.diff(condition.astype(int))
    return int(np.sum(diffs == 1)) + (1 if condition[0] else 0)


def _linear_trend(values: np.ndarray) -> float:
    """Compute the linear trend (slope) of a series."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return 0.0
    slope = np.polyfit(x[valid], values[valid], 1)[0]
    return float(slope)


# ---------------------------------------------------------------------------
# 4. Climatological features (expanded with quantiles and distribution fits)
# ---------------------------------------------------------------------------

def compute_climatological_features(miami: pd.DataFrame, init_dates: pd.Series) -> pd.DataFrame:
    """Compute climatological statistics for the same calendar window across all years."""
    miami = miami.copy()

    precip = miami["noaa_PRCP"].values
    thunder = miami["noaa_WT03"].values
    fog = miami["noaa_WT01"].values
    rain_flag = miami["noaa_WT16"].values
    doy_arr = miami["doy"].values

    # Pre-compute cumulative H-day stats for every date
    precip_sums = {}
    thunder_counts = {}
    fog_counts = {}
    rainy_counts = {}
    heavy_counts = {}
    rain_flag_counts = {}

    for h in HORIZONS:
        n = len(miami)
        p_sum = np.full(n, np.nan)
        t_count = np.full(n, np.nan)
        f_count = np.full(n, np.nan)
        r_count = np.full(n, np.nan)
        hv_count = np.full(n, np.nan)
        rf_count = np.full(n, np.nan)

        for i in range(n - h + 1):
            wp = precip[i: i + h]
            p_sum[i] = np.sum(wp)
            t_count[i] = np.sum(thunder[i: i + h])
            f_count[i] = np.sum(fog[i: i + h])
            r_count[i] = np.sum(wp > 0)
            hv_count[i] = np.sum(wp > 25)
            rf_count[i] = np.sum(rain_flag[i: i + h])

        precip_sums[h] = p_sum
        thunder_counts[h] = t_count
        fog_counts[h] = f_count
        rainy_counts[h] = r_count
        heavy_counts[h] = hv_count
        rain_flag_counts[h] = rf_count

    # Build climatology lookup per (doy, horizon)
    clim_cache = {}
    for h in HORIZONS:
        for target_doy in range(1, 367):
            doy_diff = np.abs(doy_arr - target_doy)
            doy_diff = np.minimum(doy_diff, 365 - doy_diff)
            window_mask = (doy_diff <= 15) & ~np.isnan(precip_sums[h])
            vals = precip_sums[h][window_mask]
            t_vals = thunder_counts[h][window_mask]
            f_vals = fog_counts[h][window_mask]
            r_vals = rainy_counts[h][window_mask]
            hv_vals = heavy_counts[h][window_mask]
            rf_vals = rain_flag_counts[h][window_mask]

            if len(vals) > 5:
                entry = {
                    "clim_mean_precip": float(np.mean(vals)),
                    "clim_median_precip": float(np.median(vals)),
                    "clim_std_precip": float(np.std(vals)),
                    "clim_pct_rainy": float(np.mean(r_vals / h)),
                    "clim_pct_heavy": float(np.mean(hv_vals / h)),
                    "clim_mean_thunder_days": float(np.mean(t_vals)),
                    "clim_mean_fog_days": float(np.mean(f_vals)),
                    "clim_mean_rain_flag_days": float(np.mean(rf_vals)),
                    # Distribution shape of historical precip
                    "clim_p10": float(np.percentile(vals, 10)),
                    "clim_p25": float(np.percentile(vals, 25)),
                    "clim_p75": float(np.percentile(vals, 75)),
                    "clim_p90": float(np.percentile(vals, 90)),
                    "clim_iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                    "clim_pct_zero": float(np.mean(vals < EPSILON)),
                }
                # Fit a gamma to the historical distribution
                vals_pos = vals[vals > 0]
                if len(vals_pos) > 10:
                    try:
                        shape, _, scale = sp_gamma.fit(vals_pos, floc=0)
                        entry["clim_gamma_shape"] = float(shape)
                        entry["clim_gamma_scale"] = float(scale)
                    except Exception:
                        entry["clim_gamma_shape"] = np.nan
                        entry["clim_gamma_scale"] = np.nan
                else:
                    entry["clim_gamma_shape"] = np.nan
                    entry["clim_gamma_scale"] = np.nan
            else:
                entry = {k: np.nan for k in [
                    "clim_mean_precip", "clim_median_precip", "clim_std_precip",
                    "clim_pct_rainy", "clim_pct_heavy", "clim_mean_thunder_days",
                    "clim_mean_fog_days", "clim_mean_rain_flag_days",
                    "clim_p10", "clim_p25", "clim_p75", "clim_p90", "clim_iqr",
                    "clim_pct_zero", "clim_gamma_shape", "clim_gamma_scale",
                ]}

            clim_cache[(target_doy, h)] = entry

    rows = []
    for init_date in init_dates:
        doy = init_date.timetuple().tm_yday
        for h in HORIZONS:
            row = {"init_date": init_date, "horizon_days": h}
            row.update(clim_cache.get((doy, h), {}))
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Recent conditions features (expanded)
# ---------------------------------------------------------------------------

def compute_recent_conditions(miami: pd.DataFrame, init_dates: pd.Series) -> pd.DataFrame:
    """Compute features from weather conditions in the days preceding each init_date."""
    miami_indexed = miami.set_index("date")
    clim_temp = miami.groupby("doy")["avg_temp"].mean().to_dict()
    clim_sunshine = miami.groupby("doy")["openmeteo_sunshine_duration"].mean().to_dict()

    rows = []
    for init_date in init_dates:
        row = {"init_date": init_date}
        doy = init_date.timetuple().tm_yday

        # Helper: get values for a lookback window
        def _get_window(lookback):
            start = init_date - pd.Timedelta(days=lookback)
            end = init_date - pd.Timedelta(days=1)
            dr = pd.date_range(start, end, freq="D")
            return [d for d in dr if d in miami_indexed.index]

        # --- Precipitation recency ---
        for lb in [1, 3, 7, 14, 30]:
            valid = _get_window(lb)
            if valid:
                vals = miami_indexed.loc[valid, "noaa_PRCP"]
                row[f"precip_last_{lb}d"] = float(vals.sum())
            else:
                row[f"precip_last_{lb}d"] = np.nan

        # --- Rainy day counts ---
        for lb in [7, 14, 30]:
            valid = _get_window(lb)
            if valid:
                vals = miami_indexed.loc[valid, "noaa_PRCP"]
                row[f"rainy_days_last_{lb}d"] = int((vals > 0).sum())
            else:
                row[f"rainy_days_last_{lb}d"] = np.nan

        # --- Max daily precip ---
        for lb in [7, 14]:
            valid = _get_window(lb)
            if valid:
                vals = miami_indexed.loc[valid, "noaa_PRCP"]
                row[f"max_precip_last_{lb}d"] = float(vals.max())
            else:
                row[f"max_precip_last_{lb}d"] = np.nan

        # --- Precipitation intensity (avg rain per rainy day) ---
        valid_14 = _get_window(14)
        if valid_14:
            precip_14 = miami_indexed.loc[valid_14, "noaa_PRCP"]
            rainy_14 = precip_14[precip_14 > 0]
            row["precip_intensity_14d"] = float(rainy_14.mean()) if len(rainy_14) > 0 else 0.0
        else:
            row["precip_intensity_14d"] = np.nan

        # --- Precipitation trend ---
        valid_7 = _get_window(7)
        if valid_7 and valid_14:
            p7 = miami_indexed.loc[valid_7, "noaa_PRCP"].sum()
            p14 = miami_indexed.loc[valid_14, "noaa_PRCP"].sum()
            row["precip_trend_7v14"] = float(p7 - (p14 - p7))  # is last 7d wetter than prior 7d?
        else:
            row["precip_trend_7v14"] = np.nan

        # --- Days since rain ---
        days_since = np.nan
        for offset in range(1, 61):
            check_date = init_date - pd.Timedelta(days=offset)
            if check_date in miami_indexed.index:
                if miami_indexed.loc[check_date, "noaa_PRCP"] > 0:
                    days_since = offset - 1
                    break
            else:
                break
        if np.isnan(days_since) and offset == 60:
            days_since = 60
        row["days_since_rain"] = days_since

        # --- Lag precipitation (individual days) ---
        for lag in [1, 2, 3, 7, 14]:
            lag_date = init_date - pd.Timedelta(days=lag)
            if lag_date in miami_indexed.index:
                row[f"precip_lag_{lag}d"] = float(miami_indexed.loc[lag_date, "noaa_PRCP"])
            else:
                row[f"precip_lag_{lag}d"] = np.nan

        # --- Temperature ---
        for lb in [7, 14]:
            valid = _get_window(lb)
            if valid:
                vals = miami_indexed.loc[valid, "avg_temp"]
                row[f"mean_temp_last_{lb}d"] = float(vals.mean())
            else:
                row[f"mean_temp_last_{lb}d"] = np.nan

        # Temperature anomaly
        clim_t = clim_temp.get(doy, np.nan)
        t7 = row.get("mean_temp_last_7d", np.nan)
        if not np.isnan(t7) and not np.isnan(clim_t):
            row["temp_anomaly_last_7d"] = t7 - clim_t
        else:
            row["temp_anomaly_last_7d"] = np.nan

        # Temperature trend
        t14 = row.get("mean_temp_last_14d", np.nan)
        if not np.isnan(t7) and not np.isnan(t14):
            row["temp_trend_7v14"] = t7 - t14
        else:
            row["temp_trend_7v14"] = np.nan

        # Diurnal temperature range (indicator of dry conditions)
        if valid_7:
            row["mean_dtr_last_7d"] = float(miami_indexed.loc[valid_7, "dtr"].mean())
        else:
            row["mean_dtr_last_7d"] = np.nan

        # --- Wind & sunshine ---
        if valid_7:
            row["mean_wind_last_7d"] = float(
                miami_indexed.loc[valid_7, "openmeteo_wind_speed_10m_max"].mean()
            )
            row["mean_gusts_last_7d"] = float(
                miami_indexed.loc[valid_7, "openmeteo_wind_gusts_10m_max"].mean()
            )
            sunshine_7 = float(
                miami_indexed.loc[valid_7, "openmeteo_sunshine_duration"].mean()
            )
            row["mean_sunshine_last_7d"] = sunshine_7
            # Sunshine anomaly
            clim_s = clim_sunshine.get(doy, np.nan)
            row["sunshine_anomaly_7d"] = (
                sunshine_7 - clim_s if not np.isnan(clim_s) else np.nan
            )
            # Precipitation hours
            row["mean_precip_hours_last_7d"] = float(
                miami_indexed.loc[valid_7, "openmeteo_precipitation_hours"].mean()
            )
        else:
            row["mean_wind_last_7d"] = np.nan
            row["mean_gusts_last_7d"] = np.nan
            row["mean_sunshine_last_7d"] = np.nan
            row["sunshine_anomaly_7d"] = np.nan
            row["mean_precip_hours_last_7d"] = np.nan

        # --- Weather type counts ---
        for lb in [7, 14, 30]:
            valid = _get_window(lb)
            if valid:
                row[f"thunder_last_{lb}d"] = int(miami_indexed.loc[valid, "noaa_WT03"].sum())
            else:
                row[f"thunder_last_{lb}d"] = np.nan

        if valid_7:
            row["fog_last_7d"] = int(miami_indexed.loc[valid_7, "noaa_WT01"].sum())
            row["rain_flag_last_7d"] = int(miami_indexed.loc[valid_7, "noaa_WT16"].sum())
            row["haze_last_7d"] = int(miami_indexed.loc[valid_7, "noaa_WT08"].sum())
        else:
            row["fog_last_7d"] = np.nan
            row["rain_flag_last_7d"] = np.nan
            row["haze_last_7d"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Calendar features
# ---------------------------------------------------------------------------

def compute_calendar_features(init_dates: pd.Series) -> pd.DataFrame:
    """Compute calendar/seasonal features for each init_date."""
    rows = []
    for init_date in init_dates:
        doy = init_date.timetuple().tm_yday
        month = init_date.month
        row = {
            "init_date": init_date,
            "day_of_year_sin": np.sin(2 * np.pi * doy / 365.25),
            "day_of_year_cos": np.cos(2 * np.pi * doy / 365.25),
            # Higher harmonics for finer seasonal patterns
            "day_of_year_sin2": np.sin(4 * np.pi * doy / 365.25),
            "day_of_year_cos2": np.cos(4 * np.pi * doy / 365.25),
            "month": month,
            "is_wet_season": int(month in [6, 7, 8, 9, 10]),
            "is_hurricane_season": int(
                (month >= 6) and (month <= 11)
            ),
            "is_peak_wet": int(month in [8, 9, 10]),  # peak of wet season
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. GEFS bias & rolling accuracy (computed post-merge, no lookahead)
# ---------------------------------------------------------------------------

def compute_gefs_bias_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute GEFS forecast bias using the 30 nearest DOY from previous years.

    For each (init_date, horizon), finds the 30 closest day-of-year matches
    from strictly previous years (year < current year) and computes GEFS
    accuracy statistics. This avoids within-year temporal contamination.
    """
    K = 30  # number of nearest DOY neighbors
    nan_row = {
        "gefs_bias_doy": np.nan,
        "gefs_bias_doy_std": np.nan,
        "gefs_mae_doy": np.nan,
        "gefs_ratio_doy": np.nan,
        "gefs_bias_doy_n": 0,
        "gefs_overpredict_rate_doy": np.nan,
        "gefs_direction_accuracy_doy": np.nan,
        "gefs_bias_high_forecast_doy": np.nan,
        "gefs_bias_low_forecast_doy": np.nan,
    }

    rows = []
    for h in HORIZONS:
        hdf = merged[merged["horizon_days"] == h].sort_values("init_date").reset_index(drop=True)

        init_dates = hdf["init_date"].values
        years = np.array([pd.Timestamp(d).year for d in init_dates])
        doys = np.array([pd.Timestamp(d).timetuple().tm_yday for d in init_dates])
        gefs_vals = hdf["cum_apcp_mean"].values
        obs_vals = hdf["ground_truth_precip"].values - EPSILON  # raw observed
        errors = gefs_vals - obs_vals

        for i in range(len(hdf)):
            row = {"init_date": pd.Timestamp(init_dates[i]), "horizon_days": h}
            current_year = years[i]
            current_doy = doys[i]

            # Only use data from STRICTLY previous years
            prev_year_mask = years[:i] < current_year

            if prev_year_mask.sum() < 10:
                row.update(nan_row)
                rows.append(row)
                continue

            # Filter to previous years only
            prev_doys = doys[:i][prev_year_mask]
            prev_errors = errors[:i][prev_year_mask]
            prev_gefs = gefs_vals[:i][prev_year_mask]
            prev_obs = obs_vals[:i][prev_year_mask]

            # DOY proximity (wrapping around year boundary)
            doy_diff = np.abs(prev_doys - current_doy)
            doy_diff = np.minimum(doy_diff, 365 - doy_diff)

            # Take K nearest DOY neighbors
            k = min(K, len(prev_doys))
            if k < 5:
                row.update(nan_row)
                rows.append(row)
                continue

            nearest_idx = np.argpartition(doy_diff, k)[:k]

            we = prev_errors[nearest_idx]
            wg = prev_gefs[nearest_idx]
            wo = prev_obs[nearest_idx]

            row["gefs_bias_doy"] = float(np.mean(we))
            row["gefs_bias_doy_std"] = float(np.std(we))
            row["gefs_mae_doy"] = float(np.mean(np.abs(we)))
            row["gefs_ratio_doy"] = float(np.mean(wg / (wo + 0.1)))
            row["gefs_bias_doy_n"] = k

            # Directional accuracy: does GEFS overpredict or underpredict?
            row["gefs_overpredict_rate_doy"] = float(np.mean(we > 0))

            # Direction skill: does GEFS correctly predict above/below median?
            median_obs = float(np.median(wo))
            above_median_gefs = wg > median_obs
            above_median_obs = wo > median_obs
            row["gefs_direction_accuracy_doy"] = float(
                np.mean(above_median_gefs == above_median_obs)
            )

            # Conditional bias: error when GEFS predicts high vs low
            median_gefs = float(np.median(wg))
            high_mask = wg >= median_gefs
            low_mask = ~high_mask
            row["gefs_bias_high_forecast_doy"] = (
                float(np.mean(we[high_mask])) if high_mask.sum() >= 3 else np.nan
            )
            row["gefs_bias_low_forecast_doy"] = (
                float(np.mean(we[low_mask])) if low_mask.sum() >= 3 else np.nan
            )

            rows.append(row)

        print(f"    {h}d: {len(hdf)} rows processed")

    return pd.DataFrame(rows)


def compute_rolling_gefs_accuracy(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling GEFS accuracy over recent forecasts (no lookahead).

    For each (init_date, horizon), looks at same-horizon forecasts from
    the previous 30, 60, and 90 days to measure recent GEFS performance.
    """
    rows = []
    for h in HORIZONS:
        hdf = merged[merged["horizon_days"] == h].sort_values("init_date").reset_index(drop=True)

        init_dates = hdf["init_date"].values
        gefs_vals = hdf["cum_apcp_mean"].values
        obs_vals = hdf["ground_truth_precip"].values - EPSILON
        errors = gefs_vals - obs_vals

        for i in range(len(hdf)):
            current_date = init_dates[i]
            row = {"init_date": pd.Timestamp(current_date), "horizon_days": h}

            for suffix, days in [("30d", 30), ("60d", 60), ("90d", 90)]:
                cutoff = current_date - np.timedelta64(days, "D")
                mask = init_dates[:i] >= cutoff

                n_match = int(mask.sum())
                if n_match >= 5:
                    we = errors[:i][mask]
                    row[f"gefs_rolling_bias_{suffix}"] = float(np.mean(we))
                    row[f"gefs_rolling_mae_{suffix}"] = float(np.mean(np.abs(we)))
                    row[f"gefs_rolling_rmse_{suffix}"] = float(np.sqrt(np.mean(we ** 2)))
                else:
                    row[f"gefs_rolling_bias_{suffix}"] = np.nan
                    row[f"gefs_rolling_mae_{suffix}"] = np.nan
                    row[f"gefs_rolling_rmse_{suffix}"] = np.nan

            rows.append(row)

        print(f"    {h}d: {len(hdf)} rows processed")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Assemble final dataset
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Building Training Dataset (v3 — bias/accuracy/persistence features)")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading miami_clean.csv...")
    miami = load_miami_clean()
    print(f"  {len(miami):,} rows, {miami['date'].min().date()} to {miami['date'].max().date()}")

    print("\n[2/6] Loading GEFS parquet files...")
    gefs = load_gefs_data()

    init_dates = gefs["init_date"].unique()
    init_dates = pd.Series(sorted(init_dates))
    print(f"  Processing {len(init_dates)} unique init dates")

    print("\n[3/6] Computing ground truths...")
    gt_df = compute_ground_truths(miami, init_dates)
    print(f"  {len(gt_df):,} ground truth rows")

    print("\n[4/6] Computing forecast features...")
    forecast_df = compute_forecast_features(gefs, init_dates)
    print(f"  {len(forecast_df):,} forecast feature rows")

    print("\n[5/6] Computing climatological features...")
    clim_df = compute_climatological_features(miami, init_dates)
    print(f"  {len(clim_df):,} climatological rows")

    print("\n[6/6] Computing recent conditions & calendar features...")
    recent_df = compute_recent_conditions(miami, init_dates)
    calendar_df = compute_calendar_features(init_dates)
    print(f"  {len(recent_df):,} recent condition rows, {len(calendar_df):,} calendar rows")

    # Merge everything together
    print("\nMerging all features...")

    merged = pd.merge(gt_df, forecast_df, on=["init_date", "horizon_days"], how="inner")
    merged = pd.merge(merged, clim_df, on=["init_date", "horizon_days"], how="left")
    merged = pd.merge(merged, recent_df, on="init_date", how="left")
    merged = pd.merge(merged, calendar_df, on="init_date", how="left")

    # --- Cross-features: forecast vs climatology ---
    merged["forecast_clim_ratio"] = merged["cum_apcp_mean"] / (merged["clim_mean_precip"] + 0.1)
    merged["forecast_clim_anomaly"] = merged["cum_apcp_mean"] - merged["clim_mean_precip"]

    # Where does the forecast sit in the historical distribution?
    merged["forecast_vs_clim_median"] = merged["cum_apcp_mean"] - merged["clim_median_precip"]
    merged["forecast_clim_percentile"] = np.where(
        merged["clim_std_precip"] > 0,
        (merged["cum_apcp_mean"] - merged["clim_mean_precip"]) / merged["clim_std_precip"],
        0.0,
    )

    # Forecast vs historical gamma CDF (what percentile is the forecast in?)
    mask_valid_gamma = (
        merged["clim_gamma_shape"].notna() & merged["clim_gamma_scale"].notna()
        & (merged["clim_gamma_shape"] > 0) & (merged["clim_gamma_scale"] > 0)
    )
    merged["forecast_gamma_cdf"] = np.nan
    if mask_valid_gamma.any():
        merged.loc[mask_valid_gamma, "forecast_gamma_cdf"] = sp_gamma.cdf(
            merged.loc[mask_valid_gamma, "cum_apcp_mean"].values,
            a=merged.loc[mask_valid_gamma, "clim_gamma_shape"].values,
            scale=merged.loc[mask_valid_gamma, "clim_gamma_scale"].values,
        )

    # Spread relative to climatological variability
    merged["spread_vs_clim_std"] = merged["cum_apcp_spread_rss"] / (merged["clim_std_precip"] + 0.1)

    # Is forecast consistent with recent conditions?
    merged["forecast_vs_recent_7d"] = (
        merged["cum_apcp_mean"] / merged["horizon_days"]
    ) - (merged["precip_last_7d"] / 7.0)

    # --- Persistence forecast features ---
    # Naive forecast = extrapolate recent daily precip rate over horizon
    for lb in [7, 14, 30]:
        merged[f"persistence_{lb}d"] = (
            merged[f"precip_last_{lb}d"] / lb
        ) * merged["horizon_days"]
        merged[f"forecast_vs_persistence_{lb}d"] = (
            merged["cum_apcp_mean"] - merged[f"persistence_{lb}d"]
        )
    # Persistence ratio: is GEFS wetter or drier than recent conditions?
    merged["persistence_ratio_7d"] = merged["cum_apcp_mean"] / (merged["persistence_7d"] + 0.1)

    # Metadata
    merged["is_reforecast"] = (merged["source"] == "reforecast").astype(int)

    # --- GEFS bias features (post-merge, no lookahead) ---
    print("\nComputing GEFS bias by day-of-year (no lookahead)...")
    gefs_bias_df = compute_gefs_bias_features(merged)
    n_bias_feats = len([c for c in gefs_bias_df.columns if c not in ["init_date", "horizon_days"]])
    merged = pd.merge(merged, gefs_bias_df, on=["init_date", "horizon_days"], how="left")
    print(f"  Added {n_bias_feats} GEFS bias features")

    # --- Rolling GEFS accuracy features ---
    print("\nComputing rolling GEFS accuracy (no lookahead)...")
    rolling_df = compute_rolling_gefs_accuracy(merged)
    n_rolling_feats = len([c for c in rolling_df.columns if c not in ["init_date", "horizon_days"]])
    merged = pd.merge(merged, rolling_df, on=["init_date", "horizon_days"], how="left")
    print(f"  Added {n_rolling_feats} rolling accuracy features")

    # --- Bias-corrected forecast ---
    # Subtract historical DOY bias from GEFS forecast
    merged["forecast_bias_corrected"] = merged["cum_apcp_mean"] - merged["gefs_bias_doy"].fillna(0)

    # --- Spread adequacy: is ensemble spread proportionate to historical error? ---
    # If spread is small but MAE is large, the ensemble is overconfident
    merged["spread_adequacy"] = merged["cum_apcp_spread_rss"] / (
        merged["gefs_mae_doy"].fillna(merged["cum_apcp_spread_rss"]) + 0.1
    )

    # --- Recent precipitation anomaly vs climatology ---
    # How much wetter or drier is the recent weather compared to normal?
    # (normalized by horizon to make comparable across horizons)
    clim_daily_rate = merged["clim_mean_precip"] / merged["horizon_days"]
    merged["recent_precip_anomaly_7d"] = (
        merged["precip_last_7d"] / 7.0
    ) / (clim_daily_rate + 0.1)
    merged["recent_precip_anomaly_30d"] = (
        merged["precip_last_30d"] / 30.0
    ) / (clim_daily_rate + 0.1)

    # --- Bias-corrected forecast cross-features ---
    merged["bias_corrected_vs_clim"] = merged["forecast_bias_corrected"] / (
        merged["clim_mean_precip"] + 0.1
    )
    merged["bias_corrected_anomaly"] = (
        merged["forecast_bias_corrected"] - merged["clim_mean_precip"]
    )

    # --- GEFS forecast percentile among historical GEFS for this DOY ---
    # Where does this forecast sit vs typical GEFS predictions for this time?
    merged["forecast_vs_gefs_bias_std"] = np.where(
        merged["gefs_bias_doy_std"] > 0,
        (merged["cum_apcp_mean"] - merged["gefs_bias_doy"].fillna(0) - merged["clim_mean_precip"])
        / merged["gefs_bias_doy_std"],
        0.0,
    )

    # --- Rolling bias interaction with current forecast ---
    # If GEFS has been overpredicting recently, is the current forecast also high?
    merged["rolling_bias_x_forecast"] = (
        merged["gefs_rolling_bias_30d"].fillna(0) * merged["cum_apcp_mean"]
    )

    merged = merged.sort_values(["init_date", "horizon_days"]).reset_index(drop=True)

    # --- Data integrity checks ---
    print("\n" + "=" * 60)
    print("Data Integrity Checks")
    print("=" * 60)

    print(f"\nFinal dataset shape: {merged.shape}")

    nan_counts = merged.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"\nColumns with NaN values:")
        for col, count in nan_cols.items():
            pct = 100 * count / len(merged)
            print(f"  {col}: {count} ({pct:.1f}%)")

    print("\nGround truth summary by horizon:")
    for h in HORIZONS:
        subset = merged[merged["horizon_days"] == h]
        gt = subset["ground_truth_precip"]
        print(f"  {h:2d}d: n={len(subset):,}, mean={gt.mean():.1f}mm, "
              f"median={gt.median():.1f}mm, max={gt.max():.1f}mm")

    print("\nCumulative forecast mean summary by horizon:")
    for h in HORIZONS:
        subset = merged[merged["horizon_days"] == h]
        cm = subset["cum_apcp_mean"]
        print(f"  {h:2d}d: mean={cm.mean():.1f}mm, median={cm.median():.1f}mm")

    # Save
    merged.to_parquet(OUTPUT_FILE, index=False)
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Total rows: {len(merged):,}")
    print(f"Total columns: {len(merged.columns)}")

    for src in ["reforecast", "operational"]:
        subset = merged[merged["source"] == src]
        if not subset.empty:
            print(f"  {src}: {subset['init_date'].min().date()} to "
                  f"{subset['init_date'].max().date()} "
                  f"({subset['init_date'].nunique()} init dates)")

    print("\nDone!")


if __name__ == "__main__":
    main()
