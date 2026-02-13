"""
Fetch a FULL day of GFS forecast data for Miami.
All 4 model runs (00, 06, 12, 18 UTC), all forecast hours.

This will give us an accurate storage estimate.
"""

import os
from datetime import datetime
import time

try:
    from herbie import Herbie
    import xarray as xr
    import pandas as pd
    import numpy as np
except ImportError as e:
    print("Missing required packages. Install with:")
    print("  pip install herbie-data cfgrib eccodes xarray pandas numpy")
    raise e

# Miami coordinates (center point)
MIAMI_LAT = 25.75
MIAMI_LON = -80.25

# 3x3 grid around Miami
LAT_RANGE = (MIAMI_LAT - 0.5, MIAMI_LAT + 0.5)
LON_RANGE = (MIAMI_LON - 0.5, MIAMI_LON + 0.5)
LON_RANGE_360 = (360 + LON_RANGE[0], 360 + LON_RANGE[1])

# Variables to fetch
VARIABLES = [
    "TMP:2 m above ground",
    "TMAX:2 m above ground",
    "TMIN:2 m above ground",
    "RH:2 m above ground",
    "DPT:2 m above ground",
    "APCP:surface",
]

# Model runs per day
MODEL_RUNS = ["00", "06", "12", "18"]

# Forecast hours - GFS produces forecasts at these intervals:
# 0-120 hours: every 1 hour (some products) or 3 hours
# 120-384 hours: every 3 hours (some) or 12 hours
# Let's get the standard 3-hourly forecasts up to 384 hours
FORECAST_HOURS_3HOURLY = list(range(0, 241, 3))  # 0, 3, 6, ... 240
FORECAST_HOURS_12HOURLY = list(range(252, 385, 12))  # 252, 264, ... 384
FORECAST_HOURS = FORECAST_HOURS_3HOURLY + FORECAST_HOURS_12HOURLY

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../data/gfs_full_day")


def fetch_forecast(date_str: str, run_hour: str, fxx: int) -> dict:
    """Fetch a single forecast."""
    results = {}
    total_bytes = 0

    for var in VARIABLES:
        try:
            H = Herbie(
                f"{date_str} {run_hour}:00",
                model="gfs",
                product="pgrb2.0p25",
                fxx=fxx
            )

            ds = H.xarray(var, remove_grib=True)

            # Find data variable
            data_vars = [v for v in ds.data_vars if v not in ['time', 'step', 'valid_time']]
            if not data_vars:
                continue

            var_name = data_vars[0]
            lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
            lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'

            subset = ds[var_name].sel(
                **{
                    lat_name: slice(LAT_RANGE[1], LAT_RANGE[0]),
                    lon_name: slice(LON_RANGE_360[0], LON_RANGE_360[1])
                }
            )

            results[var] = subset.values
            total_bytes += subset.nbytes

        except Exception as e:
            # Some forecast hours may not have all variables
            pass

    return results, total_bytes


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_date = "2023-06-15"

    print("=" * 70)
    print("FULL DAY GFS DATA FETCH FOR MIAMI")
    print("=" * 70)
    print(f"Date: {test_date}")
    print(f"Model runs: {MODEL_RUNS}")
    print(f"Forecast hours per run: {len(FORECAST_HOURS)}")
    print(f"  3-hourly (f000-f240): {len(FORECAST_HOURS_3HOURLY)} forecasts")
    print(f"  12-hourly (f252-f384): {len(FORECAST_HOURS_12HOURLY)} forecasts")
    print(f"Variables: {len(VARIABLES)}")
    print(f"Total forecasts to fetch: {len(MODEL_RUNS)} runs × {len(FORECAST_HOURS)} hours = {len(MODEL_RUNS) * len(FORECAST_HOURS)}")
    print()

    all_data = {}
    total_bytes = 0
    successful_fetches = 0
    failed_fetches = 0

    start_time = time.time()

    for run in MODEL_RUNS:
        print(f"\n--- Model Run: {run} UTC ---")
        run_data = {}
        run_bytes = 0

        for i, fxx in enumerate(FORECAST_HOURS):
            try:
                print(f"  Fetching f{fxx:03d}... ", end="", flush=True)
                data, nbytes = fetch_forecast(test_date, run, fxx)

                if data:
                    run_data[f"f{fxx:03d}"] = data
                    run_bytes += nbytes
                    successful_fetches += 1
                    print(f"OK ({nbytes} bytes, {len(data)} vars)")
                else:
                    failed_fetches += 1
                    print("No data")

            except Exception as e:
                failed_fetches += 1
                print(f"FAILED: {e}")

            # Progress update every 10 forecasts
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                total_to_fetch = len(MODEL_RUNS) * len(FORECAST_HOURS)
                done = (MODEL_RUNS.index(run) * len(FORECAST_HOURS)) + i + 1
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (total_to_fetch - done) / rate if rate > 0 else 0
                print(f"      Progress: {done}/{total_to_fetch} | "
                      f"Elapsed: {elapsed:.0f}s | "
                      f"ETA: {remaining:.0f}s")

        all_data[run] = run_data
        total_bytes += run_bytes
        print(f"  Run {run} UTC: {run_bytes:,} bytes ({run_bytes/1024:.1f} KB)")

    elapsed_total = time.time() - start_time

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Successful fetches: {successful_fetches}")
    print(f"Failed fetches: {failed_fetches}")
    print(f"Time elapsed: {elapsed_total:.1f} seconds")
    print()
    print(f"Total raw data size: {total_bytes:,} bytes")
    print(f"                     {total_bytes / 1024:.2f} KB")
    print(f"                     {total_bytes / (1024*1024):.4f} MB")

    # Save to compressed file
    output_file = os.path.join(OUTPUT_DIR, f"miami_gfs_full_{test_date}.npz")

    # Flatten for saving
    save_dict = {}
    for run, run_data in all_data.items():
        for fxx, var_data in run_data.items():
            for var, arr in var_data.items():
                key = f"{run}_{fxx}_{var.replace(':', '_').replace(' ', '_')}"
                save_dict[key] = arr

    np.savez_compressed(output_file, **save_dict)
    compressed_size = os.path.getsize(output_file)

    print()
    print(f"Compressed file: {output_file}")
    print(f"Compressed size: {compressed_size:,} bytes ({compressed_size/1024:.1f} KB)")

    # Storage estimates
    print("\n" + "=" * 70)
    print("STORAGE ESTIMATES FOR FULL ARCHIVE (2015-2025)")
    print("=" * 70)

    days = 365 * 10  # ~10 years

    raw_total = total_bytes * days
    compressed_total = compressed_size * days

    print(f"Days in archive: {days}")
    print()
    print("Raw data:")
    print(f"  Per day: {total_bytes/1024:.1f} KB")
    print(f"  Total:   {raw_total/(1024**3):.2f} GB")
    print()
    print("Compressed:")
    print(f"  Per day: {compressed_size/1024:.1f} KB")
    print(f"  Total:   {compressed_total/(1024**3):.2f} GB")


if __name__ == "__main__":
    main()
