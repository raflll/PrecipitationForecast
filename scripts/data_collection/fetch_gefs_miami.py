"""
Fetch GEFS data for Miami area for monthly precipitation prediction.

Data sources:
- GEFS Reforecast (2000-2019): Historical hindcasts, 5 members, daily at 00 UTC
- Operational GEFS (2017-present): Real-time forecasts from AWS

MIA2 CORS Station: 25.77°N, 80.19°W

Uses FastHerbie for parallel downloads.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

import pandas as pd
import numpy as np

try:
    from herbie import Herbie, FastHerbie
    import xarray as xr
except ImportError as e:
    print("Missing required packages. Install with:")
    print("  pip install herbie-data xarray cfgrib eccodes pandas numpy")
    sys.exit(1)

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Configuration
# =============================================================================

# MIA2 CORS Station coordinates
MIA2_LAT = 25.77  # 25°46'12"N
MIA2_LON = -80.19  # 80°11'24"W

# Bounding box around Miami (~50km radius)
LAT_MIN, LAT_MAX = 25.3, 26.3
LON_MIN, LON_MAX = -80.7, -79.7

# Variables relevant for monthly precipitation prediction
# Format for reforecast: "variable_level" (e.g., "apcp_sfc")
REFORECAST_VARIABLES = [
    "apcp_sfc",      # Total precipitation (KEY)
    "acpcp_sfc",     # Convective precipitation
    "pwat_eatm",     # Precipitable water (column moisture)
    "cape_sfc",      # Convective Available Potential Energy
    "cin_sfc",       # Convective Inhibition
    "tmp_2m",        # 2m temperature
    "rh_2m",         # 2m relative humidity (if available)
    "tcdc_eatm",     # Total cloud cover
    "ugrd_10m",      # U-wind at 10m (if available)
    "vgrd_10m",      # V-wind at 10m (if available)
    "dswrf_sfc",     # Downward shortwave radiation
    "pres_msl",      # Mean sea level pressure
]

# Format for operational GEFS: search strings
OPERATIONAL_VARIABLES = [
    "APCP:surface",           # Total precipitation
    "ACPCP:surface",          # Convective precipitation
    "PWAT:entire atmosphere", # Precipitable water
    "CAPE:surface",           # CAPE
    "CIN:surface",            # CIN
    "TMP:2 m above ground",   # 2m temperature
    "RH:2 m above ground",    # 2m relative humidity
    "TCDC:entire atmosphere", # Total cloud cover
    "UGRD:10 m above ground", # U-wind 10m
    "VGRD:10 m above ground", # V-wind 10m
    "DSWRF:surface",          # Downward shortwave
    "PRMSL:mean sea level",   # Mean sea level pressure
]

# Forecast lead times to fetch (hours)
# For monthly prediction, we want multiple lead times to capture forecast evolution
FORECAST_HOURS = [24, 48, 72, 96, 120, 144, 168, 240]  # Days 1-10

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "gefs"

# Number of parallel workers
MAX_WORKERS = 8


# =============================================================================
# Helper Functions
# =============================================================================

def extract_miami_data(ds, var_name=None):
    """
    Extract Miami area data from xarray dataset.
    Returns dict with MIA2 point value and area statistics.
    """
    try:
        # Find coordinate names
        lat_name = None
        lon_name = None
        for coord in ds.coords:
            if coord.lower() in ['latitude', 'lat']:
                lat_name = coord
            if coord.lower() in ['longitude', 'lon']:
                lon_name = coord

        if not lat_name or not lon_name:
            return None

        # Handle longitude convention (some use 0-360, some use -180 to 180)
        lons = ds[lon_name].values
        if lons.max() > 180:
            # Convert our target coords to 0-360
            lon_min_adj = LON_MIN + 360
            lon_max_adj = LON_MAX + 360
            mia2_lon_adj = MIA2_LON + 360
        else:
            lon_min_adj = LON_MIN
            lon_max_adj = LON_MAX
            mia2_lon_adj = MIA2_LON

        # Find data variable
        if var_name and var_name in ds.data_vars:
            data_var = var_name
        else:
            data_vars = [v for v in ds.data_vars if v not in ['time', 'step', 'valid_time', 'gribfile_projection']]
            if not data_vars:
                return None
            data_var = data_vars[0]

        da = ds[data_var]

        # Check if latitudes are descending (common in GRIB)
        lats = ds[lat_name].values
        if lats[0] > lats[-1]:
            lat_slice = slice(LAT_MAX, LAT_MIN)
        else:
            lat_slice = slice(LAT_MIN, LAT_MAX)

        # Extract Miami area
        miami_area = da.sel(**{
            lat_name: lat_slice,
            lon_name: slice(lon_min_adj, lon_max_adj)
        })

        # Extract MIA2 point
        mia2_point = da.sel(**{
            lat_name: MIA2_LAT,
            lon_name: mia2_lon_adj
        }, method="nearest")

        # Compute statistics
        result = {
            "mia2_value": float(mia2_point.values) if mia2_point.values.size == 1 else float(mia2_point.values.flatten()[0]),
            "area_mean": float(miami_area.mean().values),
            "area_max": float(miami_area.max().values),
            "area_min": float(miami_area.min().values),
            "area_std": float(miami_area.std().values),
        }

        return result

    except Exception as e:
        print(f"    Error extracting Miami data: {e}")
        return None


def fetch_reforecast_variable(date_str, variable_level, fxx, member=0):
    """
    Fetch a single reforecast variable for a date.

    Note: GEFS reforecast files are organized by variable, with each GRIB message
    being a different lead time. We use search to get the specific forecast hour.
    """
    try:
        H = Herbie(
            date_str,
            model="gefs_reforecast",
            fxx=fxx,
            member=member,
            variable_level=variable_level,
            verbose=False
        )

        # Search for the specific forecast hour
        search_str = f":{fxx} hour fcst:"
        ds = H.xarray(search_str, remove_grib=True)

        return extract_miami_data(ds)

    except Exception as e:
        # Many variables may not be available - that's OK
        return None


def fetch_operational_variable(date_str, search_string, fxx, member=0):
    """
    Fetch a single operational GEFS variable for a date.
    """
    try:
        H = Herbie(
            date_str,
            model="gefs",
            product="atmos.25",  # Quarter degree resolution
            fxx=fxx,
            member=member,
            verbose=False
        )

        ds = H.xarray(search_string, remove_grib=True)
        return extract_miami_data(ds)

    except Exception as e:
        return None


def process_reforecast_date(date):
    """Process all variables for a single reforecast date."""
    date_str = date.strftime("%Y-%m-%d")
    row = {"date": date_str, "source": "reforecast"}

    for fxx in FORECAST_HOURS:
        for var in REFORECAST_VARIABLES:
            var_short = var.replace("_", "")
            col_prefix = f"{var_short}_f{fxx:03d}"

            data = fetch_reforecast_variable(date_str, var, fxx)

            if data:
                row[f"{col_prefix}_mia2"] = data["mia2_value"]
                row[f"{col_prefix}_mean"] = data["area_mean"]
                row[f"{col_prefix}_max"] = data["area_max"]
                row[f"{col_prefix}_min"] = data["area_min"]
                row[f"{col_prefix}_std"] = data["area_std"]
            else:
                row[f"{col_prefix}_mia2"] = np.nan
                row[f"{col_prefix}_mean"] = np.nan
                row[f"{col_prefix}_max"] = np.nan
                row[f"{col_prefix}_min"] = np.nan
                row[f"{col_prefix}_std"] = np.nan

    return row


def process_operational_date(date):
    """Process all variables for a single operational GEFS date."""
    date_str = date.strftime("%Y-%m-%d")
    row = {"date": date_str, "source": "operational"}

    for fxx in FORECAST_HOURS:
        for var in OPERATIONAL_VARIABLES:
            # Create column name from variable
            var_short = var.split(":")[0].lower()
            col_prefix = f"{var_short}_f{fxx:03d}"

            data = fetch_operational_variable(date_str, var, fxx)

            if data:
                row[f"{col_prefix}_mia2"] = data["mia2_value"]
                row[f"{col_prefix}_mean"] = data["area_mean"]
                row[f"{col_prefix}_max"] = data["area_max"]
                row[f"{col_prefix}_min"] = data["area_min"]
                row[f"{col_prefix}_std"] = data["area_std"]
            else:
                row[f"{col_prefix}_mia2"] = np.nan
                row[f"{col_prefix}_mean"] = np.nan
                row[f"{col_prefix}_max"] = np.nan
                row[f"{col_prefix}_min"] = np.nan
                row[f"{col_prefix}_std"] = np.nan

    return row


# =============================================================================
# Main Functions
# =============================================================================

def fetch_reforecast_data(start_date="2000-01-01", end_date="2019-12-31",
                          sample_interval_days=7):
    """
    Fetch GEFS reforecast data (2000-2019).

    Args:
        start_date: Start date string
        end_date: End date string
        sample_interval_days: Days between samples (7 = weekly to reduce data volume)
    """
    print("=" * 70)
    print("FETCHING GEFS REFORECAST DATA (2000-2019)")
    print("=" * 70)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Sample interval: Every {sample_interval_days} days")
    print(f"Variables: {len(REFORECAST_VARIABLES)}")
    print(f"Forecast hours: {FORECAST_HOURS}")
    print(f"Miami area: {LAT_MIN}-{LAT_MAX}°N, {LON_MIN}-{LON_MAX}°W")
    print(f"MIA2 station: {MIA2_LAT}°N, {MIA2_LON}°W")
    print()

    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq=f"{sample_interval_days}D")
    print(f"Total dates to process: {len(dates)}")

    # Process dates in parallel
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_date = {executor.submit(process_reforecast_date, date): date
                         for date in dates}

        for i, future in enumerate(as_completed(future_to_date)):
            date = future_to_date[future]
            try:
                row = future.result()
                results.append(row)

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(dates)} dates...")

            except Exception as e:
                print(f"  Error processing {date}: {e}")

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df = df.sort_values("date").reset_index(drop=True)

    output_file = OUTPUT_DIR / "gefs_reforecast_miami.csv"
    df.to_csv(output_file, index=False)

    print()
    print(f"Saved {len(df)} rows to {output_file}")
    print(f"Columns: {len(df.columns)}")

    return df


def fetch_operational_data(start_date="2020-01-01", end_date=None,
                           sample_interval_days=1):
    """
    Fetch operational GEFS data (2017-present, but we use 2020+ to avoid overlap).

    Note: Operational data on AWS may only go back ~30 days for some products.
    For longer historical data, you may need to use other archives.

    Args:
        start_date: Start date string
        end_date: End date string (defaults to yesterday)
        sample_interval_days: Days between samples
    """
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print("=" * 70)
    print("FETCHING OPERATIONAL GEFS DATA (2020-present)")
    print("=" * 70)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Sample interval: Every {sample_interval_days} days")
    print(f"Variables: {len(OPERATIONAL_VARIABLES)}")
    print(f"Forecast hours: {FORECAST_HOURS}")
    print()

    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq=f"{sample_interval_days}D")
    print(f"Total dates to process: {len(dates)}")

    # Process dates in parallel
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_date = {executor.submit(process_operational_date, date): date
                         for date in dates}

        for i, future in enumerate(as_completed(future_to_date)):
            date = future_to_date[future]
            try:
                row = future.result()
                results.append(row)

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(dates)} dates...")

            except Exception as e:
                print(f"  Error processing {date}: {e}")

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df = df.sort_values("date").reset_index(drop=True)

    output_file = OUTPUT_DIR / "gefs_operational_miami.csv"
    df.to_csv(output_file, index=False)

    print()
    print(f"Saved {len(df)} rows to {output_file}")
    print(f"Columns: {len(df.columns)}")

    return df


def fetch_all_data():
    """Fetch both reforecast and operational data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch reforecast (2000-2019)
    # Using weekly sampling to reduce volume - adjust as needed
    df_reforecast = fetch_reforecast_data(
        start_date="2000-01-01",
        end_date="2019-12-31",
        sample_interval_days=7  # Weekly samples
    )

    print("\n" + "=" * 70 + "\n")

    # Fetch operational (2020-present)
    # Daily sampling for recent data
    df_operational = fetch_operational_data(
        start_date="2020-01-01",
        end_date=None,  # Up to yesterday
        sample_interval_days=1
    )

    # Combine and save merged file
    print("\n" + "=" * 70)
    print("MERGING DATASETS")
    print("=" * 70)

    # Find common columns
    common_cols = list(set(df_reforecast.columns) & set(df_operational.columns))

    df_combined = pd.concat([
        df_reforecast[common_cols],
        df_operational[common_cols]
    ], ignore_index=True)

    df_combined = df_combined.sort_values("date").reset_index(drop=True)

    output_file = OUTPUT_DIR / "gefs_miami_combined.csv"
    df_combined.to_csv(output_file, index=False)

    print(f"Combined dataset: {len(df_combined)} rows")
    print(f"Saved to {output_file}")

    return df_combined


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch GEFS data for Miami")
    parser.add_argument("--mode", choices=["reforecast", "operational", "all"],
                        default="all", help="Which data to fetch")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=int, default=7,
                        help="Days between samples (default: 7)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Parallel workers (default: {MAX_WORKERS})")

    args = parser.parse_args()
    MAX_WORKERS = args.workers

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "reforecast":
        fetch_reforecast_data(
            start_date=args.start or "2000-01-01",
            end_date=args.end or "2019-12-31",
            sample_interval_days=args.interval
        )
    elif args.mode == "operational":
        fetch_operational_data(
            start_date=args.start or "2020-01-01",
            end_date=args.end,
            sample_interval_days=args.interval
        )
    else:
        fetch_all_data()
