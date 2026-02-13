#!/usr/bin/env python3
"""
Pull GEFS Operational 35-day forecast data using Herbie byte-range downloads.

This script uses Herbie's byte-range download mechanism to fetch only the
specific GRIB messages needed for each variable, rather than downloading
entire GRIB files. This reduces bandwidth by ~85-94% compared to full-file
downloads (~200-500 MB/day vs ~3.5 GB/day).

Downloads ensemble mean and spread for all variables at ALL 6-hour intervals
(6h to 840h = 35 days). Extracts data for Miami International Airport.

Usage:
    python scripts/pull_gefs_grib.py 2025
    python scripts/pull_gefs_grib.py 2025 --workers 10
    python scripts/pull_gefs_grib.py 2025 --start-date 2025-06-01 --end-date 2025-06-30

Output:
    data/GEFS_Operational/gefs_operational_{year}.parquet
"""

import argparse
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from herbie import Herbie
from tqdm import tqdm

# Set default timeout for all requests (Herbie uses requests internally)
DEFAULT_TIMEOUT = 300  # 5 minutes per request
_original_request = requests.Session.request
def _timeout_request(self, *args, **kwargs):
    kwargs.setdefault('timeout', DEFAULT_TIMEOUT)
    return _original_request(self, *args, **kwargs)
requests.Session.request = _timeout_request

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
log_file = LOGS_DIR / f"gefs_grib_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Suppress noisy Herbie/cfgrib/gribapi warnings
logging.getLogger("herbie").setLevel(logging.ERROR)
logging.getLogger("cfgrib").setLevel(logging.ERROR)
logging.getLogger("gribapi").setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore", message=".*Ignoring index file.*")
warnings.filterwarnings("ignore", message=".*older than GRIB file.*")
warnings.filterwarnings("ignore", message=".*PrematureEndOfFile.*")
warnings.filterwarnings("ignore", message=".*End of resource.*")
warnings.filterwarnings("ignore", message=".*This pattern is interpreted as a regular expression.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib.*")
warnings.filterwarnings("ignore", message=".*Will not remove GRIB file.*")

# Suppress Herbie's emoji print statements
import herbie.core
_original_print = print
def _quiet_print(*args, **kwargs):
    if args and isinstance(args[0], str) and ('Created directory' in args[0] or 'Success' in args[0]):
        return
    _original_print(*args, **kwargs)
herbie.core.print = _quiet_print

# Miami International Airport coordinates
MIAMI_LAT = 25.8
MIAMI_LON = -80.3  # Herbie handles conversion; we convert in extraction

# Forecast hours configuration (all 6-hour intervals, 6h to 840h = 35 days)
ALL_6H_FORECAST_HOURS = list(range(6, 846, 6))  # 6, 12, 18, ..., 840

# Ensemble members
ENSEMBLE_MEMBERS = ['mean', 'spr']

# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 1.0
MAX_DELAY = 60.0

# Data source priority (AWS is faster and doesn't throttle)
DATA_SOURCE_PRIORITY = ["aws", "google", "azure", "nomads"]

# Connection pooling
_SESSION = None

def get_optimized_session():
    """Create a requests Session with connection pooling and retry logic."""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            backoff_factor=1,
        )
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=50,
            max_retries=retry_strategy,
            pool_block=False,
        )
        _SESSION.mount('https://', adapter)
        _SESSION.mount('http://', adapter)
        logger.info("Initialized HTTP session with connection pooling")
    return _SESSION


# Variable batch configuration
# Each batch groups variables fetched with a single H.xarray() byte-range call.
# Format: (batch_name, search_pattern, [(var_name, var_identifier, description), ...])
VARIABLE_BATCHES = [
    # --- Surface precipitation ---
    (
        "apcp",
        ":APCP:surface:",
        [("apcp", "APCP", "accumulated precipitation")]
    ),
]

# Total variable count for progress tracking
TOTAL_VARS = sum(len(vl) for _, _, vl in VARIABLE_BATCHES)


def exponential_backoff(attempt: int) -> float:
    """Calculate delay with exponential backoff and jitter."""
    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    jitter = random.uniform(0, delay * 0.1)
    return delay + jitter


def extract_variable_from_dataset(ds, var_identifier: str) -> float:
    """Extract a specific variable from a multi-variable xarray Dataset.

    Args:
        ds: xarray Dataset potentially containing multiple variables
        var_identifier: The variable abbreviation to look for (e.g., 'TMP', 'RH')

    Returns:
        Float value for Miami location, or np.nan if not found
    """
    try:
        lon = MIAMI_LON if MIAMI_LON >= 0 else MIAMI_LON + 360

        matching_vars = [v for v in ds.data_vars if var_identifier in str(v).upper()]
        if not matching_vars:
            logger.debug(f"No variable found matching {var_identifier} in dataset")
            return np.nan

        data_var = matching_vars[0]

        if "latitude" in ds.coords and "longitude" in ds.coords:
            point = ds[data_var].sel(latitude=MIAMI_LAT, longitude=lon, method="nearest")
        elif "lat" in ds.coords and "lon" in ds.coords:
            point = ds[data_var].sel(lat=MIAMI_LAT, lon=lon, method="nearest")
        else:
            if ds[data_var].ndim == 0:
                return float(ds[data_var].values)
            elif ds[data_var].ndim == 2:
                return float(ds[data_var].values.flatten()[0])
            return np.nan

        val = point.values
        if isinstance(val, np.ndarray):
            return float(val.flatten()[0]) if val.size > 0 else np.nan
        return float(val)

    except Exception as e:
        logger.debug(f"Error extracting {var_identifier} from dataset: {e}")
        return np.nan


def create_herbie_with_retry(
    init_date: datetime,
    fxx: int,
    member_type: str,
) -> Herbie | None:
    """Create a Herbie object with exponential backoff retry."""
    session = get_optimized_session()

    for attempt in range(MAX_RETRIES):
        try:
            H = Herbie(
                init_date.strftime("%Y-%m-%d 00:00"),
                model="gefs",
                product="atmos.5",
                member=member_type,
                fxx=fxx,
                priority=DATA_SOURCE_PRIORITY,
                verbose=False,
            )
            H._session = session
            return H
        except Exception as e:
            logger.debug(f"Herbie init attempt {attempt + 1} failed for {member_type} fxx={fxx}: {e}")
            if attempt < MAX_RETRIES - 1:
                delay = exponential_backoff(attempt)
                time.sleep(delay)

    logger.warning(f"FAILED to create Herbie for {member_type} fxx={fxx} after {MAX_RETRIES} retries")
    return None


def fetch_variables_batched(
    H: Herbie,
    fxx: int,
    member_type: str,
    pbar: tqdm = None,
) -> dict[tuple[int, str, str], float]:
    """Fetch all variables from a single Herbie object using batched byte-range downloads.

    Each batch downloads only the specific GRIB messages needed (50-500 KB)
    instead of the full GRIB file (12.7 MB).
    """
    results = {}

    for batch_name, batch_pattern, var_list in VARIABLE_BATCHES:
        batch_values = {}

        for attempt in range(MAX_RETRIES):
            try:
                ds = H.xarray(batch_pattern, verbose=False, overwrite=True)

                # Handle case where Herbie returns a list of datasets (multiple hypercubes)
                if isinstance(ds, list):
                    import xarray as xr
                    try:
                        ds = xr.merge(ds)
                    except Exception:
                        # If merge fails, extract from each dataset individually
                        for var_name, var_identifier, _ in var_list:
                            value = np.nan
                            for sub_ds in ds:
                                val = extract_variable_from_dataset(sub_ds, var_identifier)
                                if not np.isnan(val):
                                    value = val
                                    break
                            batch_values[var_name] = value
                        break

                for var_name, var_identifier, _ in var_list:
                    batch_values[var_name] = extract_variable_from_dataset(ds, var_identifier)

                logger.debug(f"SUCCESS batch {batch_name} {member_type} fxx={fxx} ({len(var_list)} vars)")
                break

            except Exception as e:
                logger.debug(f"Attempt {attempt + 1} failed for batch {batch_name} {member_type} fxx={fxx}: {e}")
                if attempt < MAX_RETRIES - 1:
                    delay = exponential_backoff(attempt)
                    time.sleep(delay)
                else:
                    logger.warning(f"FAILED batch {batch_name} {member_type} fxx={fxx} after {MAX_RETRIES} retries")
                    for var_name, _, _ in var_list:
                        batch_values[var_name] = np.nan

        for var_name, _, _ in var_list:
            results[(fxx, var_name, member_type)] = batch_values.get(var_name, np.nan)

        if pbar is not None:
            pbar.update(len(var_list))

    return results


def fetch_herbie_group(
    init_date: datetime,
    fxx: int,
    member_type: str,
    pbar: tqdm = None,
) -> dict[tuple[int, str, str], float]:
    """Fetch all variables for a single (fxx, member) combination."""
    H = create_herbie_with_retry(init_date, fxx, member_type)

    if H is None:
        results = {}
        for _, _, var_list in VARIABLE_BATCHES:
            for var_name, _, _ in var_list:
                results[(fxx, var_name, member_type)] = np.nan
            if pbar is not None:
                pbar.update(len(var_list))
        return results

    return fetch_variables_batched(H, fxx, member_type, pbar)


def process_date(
    init_date: datetime,
    workers: int = 10,
    pbar: tqdm = None,
) -> list[dict]:
    """Process all forecast hours for a single date using byte-range downloads.

    Returns:
        List of dicts in long format (one row per fxx/member combination)
    """
    logger.info(f"Processing {init_date.strftime('%Y-%m-%d')}")

    tasks = []
    for fxx in ALL_6H_FORECAST_HOURS:
        for member in ENSEMBLE_MEMBERS:
            tasks.append((fxx, member))

    results_dict = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                fetch_herbie_group, init_date, fxx, member, pbar
            ): (fxx, member)
            for fxx, member in tasks
        }

        for future in as_completed(futures, timeout=3600):
            try:
                group_results = future.result(timeout=600)
                results_dict.update(group_results)
            except Exception as e:
                fxx, member = futures[future]
                logger.warning(f"Failed {member} fxx={fxx}: {e}")
                for _, _, var_list in VARIABLE_BATCHES:
                    for var_name, _, _ in var_list:
                        results_dict[(fxx, var_name, member)] = np.nan
                if pbar is not None:
                    pbar.update(TOTAL_VARS)

    # Build result rows in long format (for pivot_results)
    results = []
    for fxx in ALL_6H_FORECAST_HOURS:
        for member in ENSEMBLE_MEMBERS:
            row = {
                'init_date': init_date,
                'forecast_hour': fxx,
                'valid_date': init_date + timedelta(hours=fxx),
                'member': member,
            }
            for _, _, var_list in VARIABLE_BATCHES:
                for var_name, _, _ in var_list:
                    row[var_name] = results_dict.get((fxx, var_name, member), np.nan)
            results.append(row)

    return results


def pivot_results(results: list[dict]) -> pd.DataFrame:
    """Pivot results from long format to wide format with member suffix.

    Returns:
        DataFrame with columns like 'apcp_mean', 'apcp_spread', etc.
    """
    df = pd.DataFrame(results)

    meta_cols = ['init_date', 'forecast_hour', 'valid_date']
    var_cols = [c for c in df.columns if c not in meta_cols + ['member']]

    pivoted_dfs = []
    for member in ENSEMBLE_MEMBERS:
        df_member = df[df['member'] == member].drop(columns=['member']).copy()
        suffix = 'spread' if member == 'spr' else member
        df_member = df_member.rename(columns={c: f"{c}_{suffix}" for c in var_cols})
        pivoted_dfs.append(df_member)

    df_wide = pivoted_dfs[0]
    for df_member in pivoted_dfs[1:]:
        df_wide = pd.merge(
            df_wide,
            df_member.drop(columns=['valid_date'], errors='ignore'),
            on=['init_date', 'forecast_hour'],
            how='outer'
        )

    return df_wide


def main():
    parser = argparse.ArgumentParser(
        description="Pull GEFS data using byte-range downloads"
    )
    parser.add_argument("year", type=int, help="Year to pull")
    parser.add_argument("--workers", type=int, default=20,
                       help="Parallel download workers (default: 20)")
    parser.add_argument("--start-date", type=str, default=None,
                       help="Start date (YYYY-MM-DD), overrides resume")
    parser.add_argument("--end-date", type=str, default=None,
                       help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "GEFS_Operational"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gefs_operational_{args.year}.parquet"

    # Determine date range
    start_date = datetime(args.year, 1, 1)
    end_date = datetime(args.year, 12, 31)

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # Don't go past yesterday
    yesterday = datetime.now() - timedelta(days=1)
    if end_date > yesterday:
        end_date = yesterday

    # Check for existing data and resume
    df_existing = None
    if output_file.exists() and not args.start_date:
        try:
            df_existing = pd.read_parquet(output_file)
            last_date = pd.to_datetime(df_existing['init_date'].max())
            resume_date = last_date + timedelta(days=1)

            if resume_date > start_date:
                print(f"Resuming from {resume_date.strftime('%Y-%m-%d')} ({len(df_existing)} existing rows)")
                start_date = resume_date
        except Exception as e:
            logger.warning(f"Could not read existing file: {e}")

    # Generate date list
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    if not dates:
        print("No dates to process!")
        return

    # Calculate totals
    groups_per_day = len(ALL_6H_FORECAST_HOURS) * len(ENSEMBLE_MEMBERS)
    vars_per_day = groups_per_day * TOTAL_VARS

    print(f"\n{'='*60}")
    print("GEFS Byte-Range Download")
    print(f"{'='*60}")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Total dates: {len(dates)}")
    print(f"Forecast hours: {len(ALL_6H_FORECAST_HOURS)} (every 6h, 6-840h)")
    print(f"Variables: {TOTAL_VARS} in {len(VARIABLE_BATCHES)} batches")
    print(f"Members: {', '.join(ENSEMBLE_MEMBERS)}")
    print(f"API calls/day: ~{groups_per_day * len(VARIABLE_BATCHES):,} byte-range requests")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_file}")
    print(f"Log: {log_file}")
    print(f"{'='*60}\n")

    # Process each date
    overall_start = time.time()
    total_rows = len(df_existing) if df_existing is not None else 0
    total_vars = vars_per_day * len(dates)

    pbar = tqdm(total=total_vars, desc="Downloading", unit="var",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {postfix}]")
    pbar.set_postfix_str("ETA: calculating...")

    for i, date in enumerate(dates):
        date_start = time.time()
        date_results = process_date(
            date,
            workers=args.workers,
            pbar=pbar,
        )
        date_time = time.time() - date_start

        # Stable ETA based on total elapsed time
        elapsed = time.time() - overall_start
        dates_done = i + 1
        dates_left = len(dates) - dates_done
        eta_seconds = (elapsed / dates_done) * dates_left
        if eta_seconds < 60:
            eta_str = f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            eta_str = f"{eta_seconds / 60:.1f}m"
        else:
            eta_str = f"{eta_seconds / 3600:.1f}h"
        pbar.set_postfix_str(f"ETA: {eta_str} | {date_time:.0f}s/day")

        # Pivot to wide format
        df_day = pivot_results(date_results)
        df_day = df_day.sort_values(['init_date', 'forecast_hour'])

        # Append to existing or create new
        if df_existing is not None:
            df_combined = pd.concat([df_existing, df_day], ignore_index=True)
            df_combined.to_parquet(output_file, index=False)
            df_existing = df_combined
        else:
            df_day.to_parquet(output_file, index=False)
            df_existing = df_day

        total_rows = len(df_existing)
        logger.info(f"Saved {date.strftime('%Y-%m-%d')}: {total_rows} rows in {date_time:.1f}s")
        print(f"\nSaved {date.strftime('%Y-%m-%d')} ({total_rows} total rows, {date_time:.1f}s)")

    pbar.close()

    # Final summary
    df_final = pd.read_parquet(output_file)
    overall_time = time.time() - overall_start
    file_size_mb = output_file.stat().st_size / (1024 * 1024)

    print(f"\n{'='*60}")
    print("Complete!")
    print(f"{'='*60}")
    print(f"Output: {output_file}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Total rows: {len(df_final)}")
    print(f"Columns: {len(df_final.columns)}")
    print(f"Date range: {df_final['init_date'].min()} to {df_final['init_date'].max()}")
    print(f"Time: {overall_time/60:.1f} minutes")

    print(f"\nSample columns: {list(df_final.columns)[:10]}...")

    missing_pct = df_final.isna().sum() / len(df_final) * 100
    high_missing = missing_pct[missing_pct > 10]
    if not high_missing.empty:
        print(f"\nWarning: Columns with >10% missing:")
        for col, pct in high_missing.head(10).items():
            print(f"  {col}: {pct:.1f}%")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
