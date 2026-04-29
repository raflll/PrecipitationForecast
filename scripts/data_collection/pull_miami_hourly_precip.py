#!/usr/bin/env python3
"""
Pull hourly precipitation for Miami International Airport from the Open-Meteo
historical archive API. Used to compute correctly-aligned ground truth for
non-00Z GEFS forecast cycles (06/12/18Z), which span UTC windows that don't
align with NOAA daily totals.

Usage:
    python scripts/data_collection/pull_miami_hourly_precip.py
    python scripts/data_collection/pull_miami_hourly_precip.py --start 2020-01-01 --end 2025-12-31

Output:
    data/open-meteo/miami_hourly.csv  (time UTC, precipitation mm)
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "open-meteo"
OUTPUT_FILE = OUTPUT_DIR / "miami_hourly.csv"

MIAMI_LAT = 25.7617
MIAMI_LON = -80.1918

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_chunk(start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": MIAMI_LAT,
        "longitude": MIAMI_LON,
        "start_date": start,
        "end_date": end,
        "hourly": "precipitation",
        "timezone": "UTC",
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=(datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d"))
    parser.add_argument("--chunk-years", type=int, default=5,
                        help="Years per API request (default: 5)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    frames = []
    cur = start_dt
    while cur <= end_dt:
        chunk_end = min(
            datetime(cur.year + args.chunk_years, 1, 1) - timedelta(days=1),
            end_dt,
        )
        print(f"  Fetching {cur.date()} to {chunk_end.date()}...")
        df = fetch_chunk(cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
        print(f"    got {len(df):,} hourly rows")
        frames.append(df)
        cur = chunk_end + timedelta(days=1)

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset="time")
    combined = combined.sort_values("time").reset_index(drop=True)

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(combined):,} hourly rows to {OUTPUT_FILE}")
    print(f"Range: {combined['time'].min()} to {combined['time'].max()} (UTC)")
    missing = combined["precipitation"].isna().sum()
    if missing:
        print(f"Warning: {missing} rows missing precipitation values")


if __name__ == "__main__":
    main()
