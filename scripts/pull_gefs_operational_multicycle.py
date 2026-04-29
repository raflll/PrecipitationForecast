#!/usr/bin/env python3
"""
Pull GEFS Operational forecast data for non-00 UTC cycles via direct S3 byte-range downloads.

Uses HTTP byte-range requests against the noaa-gefs-pds S3 bucket to download
only the APCP GRIB messages (~10-50KB each) instead of full GRIB files. This
skips Herbie entirely and is significantly faster.

Non-00 cycles have a maximum forecast horizon of 384 hours (16 days),
compared to 840 hours (35 days) for the 00 UTC cycle.

Usage:
    # All cycles (06Z, 12Z, 18Z) sequentially
    python scripts/pull_gefs_operational_multicycle.py 2024

    # Single cycle
    python scripts/pull_gefs_operational_multicycle.py 2024 --cycle 6

    # With date range
    python scripts/pull_gefs_operational_multicycle.py 2024 --start-date 2024-06-01 --end-date 2024-06-30

Output:
    data/GEFS_Operational/gefs_operational_{cycle}z_{year}.parquet
"""

import argparse
import logging
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xarray as xr
from tqdm import tqdm

# Create logs directory
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
log_file = LOGS_DIR / f"gefs_multicycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib.*")

# Miami International Airport coordinates
MIAMI_LAT = 25.8
MIAMI_LON_360 = -80.3 + 360  # S3 data uses 0-360 convention

# S3 bucket base URL (publicly accessible, no auth needed)
S3_BASE = "https://noaa-gefs-pds.s3.amazonaws.com"

# GRIB file URL template
# gefs.YYYYMMDD/HH/atmos/pgrb2sp25/ge{avg|spr}.tHHz.pgrb2s.0p25.fFFF
GRIB_URL_TEMPLATE = "{base}/gefs.{date}/{cycle:02d}/atmos/pgrb2sp25/{member_file}.t{cycle:02d}z.pgrb2s.0p25.f{fxx:03d}"

# Member type to filename prefix mapping
MEMBER_FILE = {"mean": "geavg", "spr": "gespr"}

# Forecast hours for non-00 cycles (max 240h = 10 days)
FORECAST_HOURS = list(range(6, 246, 6))  # 6, 12, 18, ..., 240

# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 1.0
MAX_DELAY = 60.0

# Connection pooling
_SESSION = None

def get_session():
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
        _SESSION.mount("https://", adapter)
        _SESSION.mount("http://", adapter)
        logger.info("Initialized HTTP session with connection pooling")
    return _SESSION


def exponential_backoff(attempt: int) -> float:
    """Calculate delay with exponential backoff and jitter."""
    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    return delay + random.uniform(0, delay * 0.1)


def parse_idx_for_apcp(idx_text: str) -> tuple[int, int | None]:
    """Parse a GRIB .idx file to find the byte range for APCP:surface.

    Returns:
        (start_byte, end_byte) where end_byte is None if APCP is the last message.
    """
    lines = idx_text.strip().split("\n")
    for i, line in enumerate(lines):
        if ":APCP:surface:" in line:
            start_byte = int(line.split(":")[1])
            end_byte = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else None
            return (start_byte, end_byte)
    return (-1, -1)


def fetch_apcp_single(
    date_str: str,
    cycle: int,
    fxx: int,
    member_type: str,
) -> float:
    """Fetch APCP value for one (fxx, member_type) via byte-range download.

    Steps:
        1. Download the tiny .idx file (~1-5KB)
        2. Parse it to find the APCP byte range
        3. Download only the APCP GRIB message (~10-50KB)
        4. Decode with cfgrib and extract Miami point
    """
    session = get_session()
    member_file = MEMBER_FILE[member_type]
    base_url = GRIB_URL_TEMPLATE.format(
        base=S3_BASE, date=date_str, cycle=cycle,
        member_file=member_file, fxx=fxx,
    )
    idx_url = f"{base_url}.idx"

    for attempt in range(MAX_RETRIES):
        tmp_path = None
        try:
            # 1. Download .idx file
            idx_resp = session.get(idx_url, timeout=30)
            if idx_resp.status_code != 200:
                logger.debug(f"IDX {idx_resp.status_code} for {member_type} fxx={fxx}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(exponential_backoff(attempt))
                continue

            # 2. Find APCP byte range
            start_byte, end_byte = parse_idx_for_apcp(idx_resp.text)
            if start_byte < 0:
                logger.debug(f"APCP not found in idx for {member_type} fxx={fxx}")
                return np.nan

            # 3. Download only the APCP bytes
            if end_byte is not None:
                range_header = f"bytes={start_byte}-{end_byte}"
            else:
                range_header = f"bytes={start_byte}-"

            data_resp = session.get(base_url, headers={"Range": range_header}, timeout=60)
            if data_resp.status_code not in (200, 206):
                logger.debug(f"Data {data_resp.status_code} for {member_type} fxx={fxx}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(exponential_backoff(attempt))
                continue

            # 4. Validate GRIB magic bytes before decoding
            content = data_resp.content
            if len(content) < 4 or not content.startswith(b"GRIB"):
                logger.debug(f"Non-GRIB response for {member_type} fxx={fxx} (len={len(content)})")
                return np.nan

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".grib2")
            os.close(tmp_fd)
            with open(tmp_path, "wb") as f:
                f.write(content)

            ds = xr.open_dataset(tmp_path, engine="cfgrib")

            if "latitude" in ds.coords:
                point = ds.sel(latitude=MIAMI_LAT, longitude=MIAMI_LON_360, method="nearest")
            elif "lat" in ds.coords:
                point = ds.sel(lat=MIAMI_LAT, lon=MIAMI_LON_360, method="nearest")
            else:
                ds.close()
                return np.nan

            for var in point.data_vars:
                val = point[var].values
                ds.close()
                if isinstance(val, np.ndarray):
                    return float(val.flatten()[0]) if val.size > 0 else np.nan
                return float(val)

            ds.close()
            return np.nan

        except Exception as e:
            logger.debug(f"Attempt {attempt + 1} failed for {member_type} fxx={fxx}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(exponential_backoff(attempt))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    logger.warning(f"FAILED {member_type} fxx={fxx} after {MAX_RETRIES} retries")
    return np.nan


def process_date(
    init_date: datetime,
    cycle: int,
    workers: int = 20,
) -> list[dict]:
    """Fetch APCP mean + spread for all forecast hours in parallel."""
    date_str = init_date.strftime("%Y%m%d")
    init_datetime = datetime(init_date.year, init_date.month, init_date.day, cycle)

    results = {}  # (fxx, member_type) -> value

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for fxx in FORECAST_HOURS:
            for member_type in ("mean", "spr"):
                future = executor.submit(fetch_apcp_single, date_str, cycle, fxx, member_type)
                futures[future] = (fxx, member_type)

        for future in as_completed(futures):
            fxx, member_type = futures[future]
            try:
                results[(fxx, member_type)] = future.result()
            except Exception as e:
                logger.warning(f"Failed fxx={fxx} {member_type}: {e}")
                results[(fxx, member_type)] = np.nan

    rows = []
    for fxx in FORECAST_HOURS:
        rows.append({
            "init_date": init_datetime,
            "forecast_hour": fxx,
            "valid_date": init_datetime + timedelta(hours=fxx),
            "apcp_mean": results.get((fxx, "mean"), np.nan),
            "apcp_spread": results.get((fxx, "spr"), np.nan),
        })

    return rows


def generate_dates(start_date: datetime, end_date: datetime) -> list[datetime]:
    """Generate list of dates between start and end (inclusive)."""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def main():
    parser = argparse.ArgumentParser(
        description="Pull GEFS Operational forecast data for non-00 UTC cycles (06/12/18Z)"
    )
    parser.add_argument("year", type=int, help="Year to pull (e.g., 2024)")
    parser.add_argument("--cycle", type=int, default=None, choices=[6, 12, 18],
                       help="UTC cycle (6, 12, or 18). Omit to run all three.")
    parser.add_argument("--workers", type=int, default=20,
                       help="Parallel workers (default: 20)")
    parser.add_argument("--start-date", type=str, default=None,
                       help="Start date (YYYY-MM-DD), overrides resume")
    parser.add_argument("--end-date", type=str, default=None,
                       help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    cycles = [args.cycle] if args.cycle else [6, 12, 18]
    for cycle in cycles:
        run_cycle(args, cycle)


def run_cycle(args, cycle: int):
    """Pull data for a single UTC cycle."""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "GEFS_Operational"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Date range
    start_date = datetime(args.year, 1, 1)
    end_date = datetime(args.year, 12, 31)
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    if end_date > datetime.now():
        end_date = datetime.now() - timedelta(days=1)

    gefs_start = datetime(2020, 9, 23)
    if start_date < gefs_start:
        print(f"Warning: Adjusting start to GEFS operational start: {gefs_start.strftime('%Y-%m-%d')}")
        start_date = gefs_start

    # Resume from existing data
    output_file = output_dir / f"gefs_operational_{cycle:02d}z_{args.year}.parquet"
    df_existing = None
    if output_file.exists() and not args.start_date:
        try:
            df_existing = pd.read_parquet(output_file)
            last_date = pd.to_datetime(df_existing["init_date"].max()).normalize()
            # Check if the last date has a full set of forecast hours
            last_date_rows = len(df_existing[df_existing["init_date"].dt.normalize() == last_date])
            if last_date_rows < len(FORECAST_HOURS):
                # Drop incomplete last date and re-pull it
                df_existing = df_existing[df_existing["init_date"].dt.normalize() < last_date]
                resume_date = last_date
                print(f"Resuming {cycle:02d}Z from {resume_date.strftime('%Y-%m-%d')} (dropped incomplete last date, {len(df_existing)} rows kept)")
            else:
                resume_date = last_date + timedelta(days=1)
                print(f"Resuming {cycle:02d}Z from {resume_date.strftime('%Y-%m-%d')} ({len(df_existing)} existing rows)")
            if resume_date > start_date:
                start_date = resume_date
        except Exception as e:
            logger.warning(f"Could not read existing file: {e}")

    dates = generate_dates(start_date, end_date)
    if not dates:
        print(f"No dates to process for {cycle:02d}Z.")
        return

    # Summary
    print(f"\n{'='*60}")
    print(f"GEFS Operational — {cycle:02d}Z (direct S3 byte-range)")
    print(f"{'='*60}")
    print(f"Dates: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')} ({len(dates)} days)")
    print(f"Forecast hours: {len(FORECAST_HOURS)} (6h-384h)")
    print(f"Downloads/date: {len(FORECAST_HOURS) * 2} (mean + spread)")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    # Process dates
    overall_start = time.time()

    pbar = tqdm(
        total=len(dates), desc=f"{cycle:02d}Z", unit="day",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
    )

    for i, date in enumerate(dates):
        date_start = time.time()
        rows = process_date(date, cycle, workers=args.workers)
        date_time = time.time() - date_start

        # Save
        df_day = pd.DataFrame(rows).sort_values(["init_date", "forecast_hour"])
        if df_existing is not None:
            df_existing = pd.concat([df_existing, df_day], ignore_index=True)
        else:
            df_existing = df_day
        df_existing.to_parquet(output_file, index=False)

        pbar.update(1)
        pbar.set_postfix_str(f"{date_time:.1f}s/day | {len(df_existing)} rows")

        logger.info(f"Saved {date.strftime('%Y-%m-%d')} {cycle:02d}Z: {len(df_existing)} rows in {date_time:.1f}s")

    pbar.close()

    # Final summary
    if output_file.exists():
        df_final = pd.read_parquet(output_file)
        overall_time = time.time() - overall_start
        file_size_mb = output_file.stat().st_size / (1024 * 1024)

        print(f"\n{'='*60}")
        print(f"{cycle:02d}Z Complete!")
        print(f"{'='*60}")
        print(f"Output: {output_file} ({file_size_mb:.2f} MB)")
        print(f"Total rows: {len(df_final)}")
        print(f"Date range: {df_final['init_date'].min()} to {df_final['init_date'].max()}")
        print(f"Time: {overall_time/60:.1f} minutes")

        missing_pct = df_final.isna().sum() / len(df_final) * 100
        high_missing = missing_pct[missing_pct > 10]
        if not high_missing.empty:
            print(f"\nWarning: Columns with >10% missing:")
            for col, pct in high_missing.head(10).items():
                print(f"  {col}: {pct:.1f}%")

        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
