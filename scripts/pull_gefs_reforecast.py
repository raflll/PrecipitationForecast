#!/usr/bin/env python3
"""
Pull GEFSv12 Reforecast data for Miami (25.8°N, 80.3°W).

Downloads WEDNESDAY-ONLY reforecasts (2000-2019) which have the full 35-day
forecast horizon with 11 ensemble members. Uses direct S3 downloads (not Herbie)
because Herbie doesn't support members p05-p10 or the Days:10-35 path.

Each Wednesday date requires 22 file downloads:
    - 11 members × 2 files (Days:1-10 + Days:10-35)
    - Days:1-10: ~24 MB, 80 steps (3h-240h, 3-hourly, 0.25°)
    - Days:10-35: ~10 MB, 100 steps (246h-840h, 6-hourly, 0.50°)

We extract the Miami point from each, compute mean/spread across 11 members.

Usage:
    python scripts/pull_gefs_reforecast.py --start-year 2000 --end-year 2019
    python scripts/pull_gefs_reforecast.py --start-year 2019 --end-year 2019 --workers 5

Output:
    data/GEFS_Reforecast/gefs_reforecast_{year}.parquet
"""

import argparse
import logging
import os
import random
import signal
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from tqdm import tqdm

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
log_file = LOGS_DIR / f"gefs_reforecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

# Suppress cfgrib noise
logging.getLogger("cfgrib").setLevel(logging.ERROR)
logging.getLogger("gribapi").setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore", message=".*Ignoring index file.*")
warnings.filterwarnings("ignore", message=".*older than GRIB file.*")
warnings.filterwarnings("ignore", message=".*PrematureEndOfFile.*")
warnings.filterwarnings("ignore", message=".*End of resource.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib.*")

# Miami International Airport coordinates
MIAMI_LAT = 25.8
MIAMI_LON = -80.3
MIAMI_LON_360 = MIAMI_LON + 360  # S3 data uses 0-360

# S3 bucket and path template
S3_BUCKET = "noaa-gefs-retrospective"
S3_PATH_TEMPLATE = "GEFSv12/reforecast/{year}/{date_str}00/{member}/Days:{day_range}/{var_level}_{date_str}00_{member}.grib2"

# Members: c00 + p01-p10 for Wednesday runs
MEMBER_NAMES = ["c00"] + [f"p{i:02d}" for i in range(1, 11)]

# Variable configuration
# Format: (output_name, s3_var_level_name)
VARIABLES = [
    ("apcp", "apcp_sfc"),
]

# Day ranges to download
DAY_RANGES = ["1-10", "10-35"]

# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 1.0
MAX_DELAY = 60.0

# Shared S3 filesystem
_FS = None

def get_s3fs():
    """Get a shared anonymous S3 filesystem."""
    global _FS
    if _FS is None:
        _FS = s3fs.S3FileSystem(anon=True)
    return _FS


def exponential_backoff(attempt: int) -> float:
    """Calculate delay with exponential backoff and jitter."""
    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    jitter = random.uniform(0, delay * 0.1)
    return delay + jitter


def download_and_extract(
    init_date: datetime,
    member: str,
    day_range: str,
    var_name: str,
    var_level: str,
) -> tuple[str, str, dict[float, float], dict]:
    """Download one GRIB file from S3 and extract Miami point for all steps.

    Returns:
        (member, day_range, {step_hours: value, ...}, timing_dict)
    """
    fs = get_s3fs()
    timing = {"download": 0.0, "decode": 0.0, "extract": 0.0, "retries": 0}

    date_str = init_date.strftime("%Y%m%d")
    year = init_date.strftime("%Y")
    s3_path = f"{S3_BUCKET}/{S3_PATH_TEMPLATE.format(year=year, date_str=date_str, member=member, day_range=day_range, var_level=var_level)}"

    for attempt in range(MAX_RETRIES):
        tmp_path = None
        try:
            # Download to temp file
            t0 = time.time()
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".grib2")
            os.close(tmp_fd)
            fs.get(s3_path, tmp_path)
            timing["download"] = time.time() - t0

            # Decode GRIB
            t1 = time.time()
            # Perturbed members have both cf and pf dataTypes
            if member == "c00":
                ds = xr.open_dataset(tmp_path, engine="cfgrib")
            else:
                try:
                    ds = xr.open_dataset(
                        tmp_path, engine="cfgrib",
                        backend_kwargs={"filter_by_keys": {"dataType": "pf"}},
                    )
                except Exception:
                    ds = xr.open_dataset(tmp_path, engine="cfgrib")
            timing["decode"] = time.time() - t1

            # Extract Miami point across all steps
            t2 = time.time()
            if "latitude" in ds.coords:
                point = ds.sel(latitude=MIAMI_LAT, longitude=MIAMI_LON_360, method="nearest")
            elif "lat" in ds.coords:
                point = ds.sel(lat=MIAMI_LAT, lon=MIAMI_LON_360, method="nearest")
            else:
                timing["extract"] = time.time() - t2
                ds.close()
                return (member, day_range, {}, timing)

            step_values = {}
            for dv in point.data_vars:
                vals = point[dv].values
                if 'step' in point.coords:
                    steps = point.step.values
                    for s, v in zip(steps, np.atleast_1d(vals)):
                        step_h = float(s / np.timedelta64(1, 'h'))
                        step_values[step_h] = float(v)
                else:
                    step_values[0.0] = float(np.atleast_1d(vals)[0])

            timing["extract"] = time.time() - t2
            ds.close()

            total_time = timing["download"] + timing["decode"] + timing["extract"]
            logger.debug(
                f"{member} Days:{day_range}: {len(step_values)} steps in {total_time:.1f}s "
                f"(dl={timing['download']:.1f}s decode={timing['decode']:.1f}s)"
            )
            return (member, day_range, step_values, timing)

        except Exception as e:
            timing["retries"] = attempt + 1
            logger.debug(f"Attempt {attempt + 1} failed for {member} Days:{day_range}: {e}")
            if attempt < MAX_RETRIES - 1:
                delay = exponential_backoff(attempt)
                time.sleep(delay)
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    logger.warning(f"FAILED {member} Days:{day_range} after {MAX_RETRIES} retries")
    return (member, day_range, {}, timing)


def process_date(
    init_date: datetime,
    workers: int = 5,
) -> tuple[list[dict], dict]:
    """Process one Wednesday: download all members × day ranges, compute mean/spread.

    22 downloads total (11 members × 2 day ranges), each file contains all steps.

    Returns:
        (rows, timing_stats)
    """
    logger.info(f"Processing {init_date.strftime('%Y-%m-%d')}")

    all_timings = []
    # member_steps[var_name][member] = {step_h: value, ...}
    member_steps = {var_name: {m: {} for m in MEMBER_NAMES} for var_name, _ in VARIABLES}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for var_name, var_level in VARIABLES:
            for member in MEMBER_NAMES:
                for day_range in DAY_RANGES:
                    future = executor.submit(
                        download_and_extract,
                        init_date, member, day_range, var_name, var_level,
                    )
                    futures[future] = (var_name, member, day_range)

        for future in as_completed(futures, timeout=3600):
            var_name, member, day_range = futures[future]
            try:
                _, _, step_values, timing = future.result(timeout=600)
                member_steps[var_name][member].update(step_values)
                all_timings.append(timing)
            except Exception as e:
                logger.warning(f"Failed {member} Days:{day_range}: {e}")
                all_timings.append({"download": 0, "decode": 0, "extract": 0, "retries": 0})

    # Collect all available 6h steps across all members
    all_step_hours = set()
    for var_name, _ in VARIABLES:
        for member in MEMBER_NAMES:
            all_step_hours.update(member_steps[var_name][member].keys())

    # Filter to 6h intervals only
    all_step_hours = sorted(s for s in all_step_hours if s > 0 and s % 6 == 0)

    # Compute timing stats
    timing_stats = {}
    if all_timings:
        for phase in ["download", "decode", "extract"]:
            vals = [t[phase] for t in all_timings if t[phase] > 0]
            if vals:
                timing_stats[phase] = {
                    "mean": np.mean(vals),
                    "p95": np.percentile(vals, 95),
                    "total": np.sum(vals),
                }
        timing_stats["retries"] = sum(t["retries"] for t in all_timings)
        timing_stats["num_downloads"] = len(all_timings)
        timing_stats["num_steps"] = len(all_step_hours)
        # Count how many members actually returned data
        members_ok = 0
        for var_name, _ in VARIABLES:
            for member in MEMBER_NAMES:
                if member_steps[var_name][member]:
                    members_ok += 1
        timing_stats["members_ok"] = members_ok

    # Build rows: mean and spread across members for each step
    rows = []
    for step_h in all_step_hours:
        row = {
            "init_date": init_date,
            "forecast_hour": int(step_h),
            "valid_date": init_date + timedelta(hours=step_h),
        }
        for var_name, _ in VARIABLES:
            values = []
            for member in MEMBER_NAMES:
                val = member_steps[var_name][member].get(step_h, np.nan)
                if not np.isnan(val):
                    values.append(val)

            if values:
                row[f"{var_name}_mean"] = float(np.mean(values))
                row[f"{var_name}_spread"] = float(np.std(values, ddof=0))
            else:
                row[f"{var_name}_mean"] = np.nan
                row[f"{var_name}_spread"] = np.nan

        rows.append(row)

    return rows, timing_stats


# Graceful shutdown support
_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\nShutdown requested. Saving progress and exiting after current date...")


def main():
    parser = argparse.ArgumentParser(
        description="Pull GEFSv12 Reforecast data — Wednesday 35-day forecasts (2000-2019)"
    )
    parser.add_argument("--start-year", type=int, required=True,
                       help="Start year (e.g., 2000)")
    parser.add_argument("--end-year", type=int, required=True,
                       help="End year (e.g., 2019)")
    parser.add_argument("--workers", type=int, default=5,
                       help="Parallel download workers (default: 5)")
    parser.add_argument("--start-date", type=str, default=None,
                       help="Start date (YYYY-MM-DD), overrides resume")
    parser.add_argument("--end-date", type=str, default=None,
                       help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Setup output directory
    output_dir = PROJECT_ROOT / "data" / "GEFS_Reforecast"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process year by year
    for year in range(args.start_year, args.end_year + 1):
        if _shutdown_requested:
            print("Shutdown requested. Stopping.")
            break

        output_file = output_dir / f"gefs_reforecast_{year}.parquet"

        # Determine date range for this year
        year_start = datetime(year, 1, 1)
        year_end = datetime(year, 12, 31)

        # Apply date overrides if specified
        if args.start_date:
            override_start = datetime.strptime(args.start_date, "%Y-%m-%d")
            if override_start.year == year:
                year_start = override_start
        if args.end_date:
            override_end = datetime.strptime(args.end_date, "%Y-%m-%d")
            if override_end.year == year:
                year_end = override_end

        # Check for existing data and resume
        df_existing = None
        if output_file.exists() and not args.start_date:
            try:
                df_existing = pd.read_parquet(output_file)
                last_date = pd.to_datetime(df_existing['init_date'].max())
                resume_date = last_date + timedelta(days=1)

                if resume_date > year_start:
                    print(
                        f"[{year}] Resuming from {resume_date.strftime('%Y-%m-%d')} "
                        f"({len(df_existing)} existing rows)"
                    )
                    year_start = resume_date

                if year_start > year_end:
                    print(f"[{year}] Already complete. Skipping.")
                    continue

            except Exception as e:
                logger.warning(f"Could not read existing file for {year}: {e}")

        # Generate Wednesday-only date list
        dates = []
        current = year_start
        while current.weekday() != 2 and current <= year_end:  # 2 = Wednesday
            current += timedelta(days=1)
        while current <= year_end:
            dates.append(current)
            current += timedelta(days=7)

        if not dates:
            print(f"[{year}] No Wednesdays to process. Skipping.")
            continue

        num_downloads_per_date = len(MEMBER_NAMES) * len(DAY_RANGES) * len(VARIABLES)

        print(f"\n{'='*60}")
        print(f"GEFS Reforecast — Year {year}")
        print(f"{'='*60}")
        print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
        print(f"Wednesdays: {len(dates)}")
        print(f"Members: {len(MEMBER_NAMES)} (c00 + p01-p10)")
        print(f"Day ranges: Days:1-10 + Days:10-35 (full 35-day forecast)")
        print(f"Downloads/date: {num_downloads_per_date}")
        print(f"Total downloads: {num_downloads_per_date * len(dates):,}")
        print(f"Workers: {args.workers}")
        print(f"Output: {output_file}")
        print(f"Log: {log_file}")
        print(f"{'='*60}\n")

        # Process each date
        overall_start = time.time()

        pbar = tqdm(
            total=len(dates), desc=f"Year {year}", unit="wed",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {postfix}]",
        )
        pbar.set_postfix_str("ETA: calculating...")

        rolling_timings = []

        for i, date in enumerate(dates):
            if _shutdown_requested:
                print(f"\nSaving progress for {year} before exit...")
                break

            date_start = time.time()
            rows, timing_stats = process_date(date, workers=args.workers)
            date_time = time.time() - date_start

            rolling_timings.append(timing_stats)
            if len(rolling_timings) > 10:
                rolling_timings.pop(0)

            pbar.update(1)

            # ETA calculation
            elapsed = time.time() - overall_start
            dates_done = i + 1
            dates_left = len(dates) - dates_done
            eta_seconds = (elapsed / dates_done) * dates_left
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds / 60:.1f}m"
            elif eta_seconds < 86400:
                eta_str = f"{eta_seconds / 3600:.1f}h"
            else:
                eta_str = f"{eta_seconds / 86400:.1f}d"
            pbar.set_postfix_str(f"ETA: {eta_str} | {date_time:.0f}s/date")

            # Save after every date
            df_day = pd.DataFrame(rows)
            df_day = df_day.sort_values(["init_date", "forecast_hour"])

            if df_existing is not None:
                df_existing = pd.concat([df_existing, df_day], ignore_index=True)
            else:
                df_existing = df_day

            df_existing.to_parquet(output_file, index=False)

            total_rows = len(df_existing)
            num_steps = timing_stats.get("num_steps", 0)
            members_ok = timing_stats.get("members_ok", 0)

            # Bottleneck dashboard
            print(
                f"\n  {date.strftime('%Y-%m-%d')} (Wed) | {num_steps} steps | "
                f"{members_ok}/{len(MEMBER_NAMES)} members | {total_rows} rows | {date_time:.0f}s"
            )
            if timing_stats:
                dl_avg = timing_stats.get("download", {}).get("mean", 0)
                dl_p95 = timing_stats.get("download", {}).get("p95", 0)
                dec_avg = timing_stats.get("decode", {}).get("mean", 0)
                ext_avg = timing_stats.get("extract", {}).get("mean", 0)
                retries = timing_stats.get("retries", 0)

                phases = {"Download": dl_avg, "Decode": dec_avg, "Extract": ext_avg}
                bottleneck = max(phases, key=phases.get)

                print(
                    f"    DL: {dl_avg:.2f}s avg / {dl_p95:.2f}s p95 | "
                    f"Decode: {dec_avg:.2f}s | Ext: {ext_avg:.3f}s | "
                    f"Retries: {retries} | Bottleneck: {bottleneck}"
                )

                if len(rolling_timings) >= 3:
                    rolling_dl = np.mean([
                        t.get("download", {}).get("mean", 0) for t in rolling_timings
                    ])
                    rolling_retries = sum(t.get("retries", 0) for t in rolling_timings)
                    print(
                        f"    Rolling ({len(rolling_timings)}w): "
                        f"DL {rolling_dl:.2f}s avg | Retries {rolling_retries}"
                    )

            logger.info(f"Saved {date.strftime('%Y-%m-%d')}: {total_rows} rows in {date_time:.1f}s")

        pbar.close()

        # Year summary
        if df_existing is not None and len(df_existing) > 0:
            overall_time = time.time() - overall_start
            file_size_mb = output_file.stat().st_size / (1024 * 1024)

            print(f"\n{'='*60}")
            print(f"Year {year} {'INTERRUPTED' if _shutdown_requested else 'Complete'}!")
            print(f"{'='*60}")
            print(f"Output: {output_file}")
            print(f"File size: {file_size_mb:.2f} MB")
            print(f"Total rows: {len(df_existing)}")
            print(f"Date range: {df_existing['init_date'].min()} to {df_existing['init_date'].max()}")
            print(f"Time: {overall_time/60:.1f} minutes")
            print(f"{'='*60}\n")

        if _shutdown_requested:
            break

    print("Done! Run again to resume from where you left off.")


if __name__ == "__main__":
    main()
