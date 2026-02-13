"""
Rebuild miami_clean.csv with additional NOAA columns including weather types (WT).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
NOAA_PATH = PROJECT_ROOT / "data" / "noaa" / "miami_ghcnd.csv"
OPENMETEO_PATH = PROJECT_ROOT / "data" / "open-meteo" / "miami_daily.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "miami_clean.csv"

def main():
    print("Loading NOAA data...")
    df_noaa = pd.read_csv(NOAA_PATH, parse_dates=['date'])
    print(f"  Loaded {len(df_noaa)} rows, {len(df_noaa.columns)} columns")

    print("\nLoading Open-Meteo data...")
    df_openmeteo = pd.read_csv(OPENMETEO_PATH, parse_dates=['time'])
    df_openmeteo.rename(columns={'time': 'date'}, inplace=True)
    print(f"  Loaded {len(df_openmeteo)} rows, {len(df_openmeteo.columns)} columns")

    # Select NOAA columns to keep
    # Only columns with 100% coverage (always tracked)

    # Core variables (100% coverage)
    noaa_core = ['date', 'PRCP', 'TMAX', 'TMIN']

    # Weather types (binary flags) - these have 100% coverage as binary 0/1
    noaa_weather_types = [
        'WT01',  # Fog
        'WT02',  # Heavy fog
        'WT03',  # Thunder
        'WT05',  # Hail
        'WT08',  # Smoke/haze
        'WT10',  # Tornado/waterspout
        'WT11',  # High winds
        'WT13',  # Mist
        'WT14',  # Drizzle
        'WT16',  # Rain
        'WT21',  # Ground fog
    ]

    # Combine all desired NOAA columns
    noaa_cols = noaa_core + noaa_weather_types

    # Filter to columns that actually exist in the data
    noaa_cols_exist = [c for c in noaa_cols if c in df_noaa.columns]
    missing_cols = set(noaa_cols) - set(noaa_cols_exist)
    if missing_cols:
        print(f"\n  Note: These NOAA columns don't exist: {missing_cols}")

    df_noaa_subset = df_noaa[noaa_cols_exist].copy()

    # Add noaa_ prefix
    df_noaa_subset = df_noaa_subset.rename(
        columns={c: f'noaa_{c}' for c in df_noaa_subset.columns if c != 'date'}
    )

    # Select Open-Meteo columns
    openmeteo_cols = [
        'date',
        'temperature_2m_max',
        'temperature_2m_min',
        'precipitation_sum',
        'rain_sum',
        'precipitation_hours',
        'wind_speed_10m_max',
        'wind_gusts_10m_max',
        'sunshine_duration'
    ]

    openmeteo_cols_exist = [c for c in openmeteo_cols if c in df_openmeteo.columns]
    df_openmeteo_subset = df_openmeteo[openmeteo_cols_exist].copy()

    # Add openmeteo_ prefix
    df_openmeteo_subset = df_openmeteo_subset.rename(
        columns={c: f'openmeteo_{c}' for c in df_openmeteo_subset.columns if c != 'date'}
    )

    # Merge on date
    print("\nMerging datasets...")
    df_merged = pd.merge(df_noaa_subset, df_openmeteo_subset, on='date', how='inner')
    df_merged.sort_values('date', inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    # Convert WT columns to binary (1 if occurred, 0 if not)
    wt_cols = [c for c in df_merged.columns if 'WT' in c]
    for col in wt_cols:
        df_merged[col] = df_merged[col].fillna(0).astype(int)

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    df_merged.to_csv(OUTPUT_PATH, index=False)

    print(f"\n=== Summary ===")
    print(f"Rows: {len(df_merged)}")
    print(f"Columns: {len(df_merged.columns)}")
    print(f"Date range: {df_merged['date'].min()} to {df_merged['date'].max()}")
    print(f"\nColumns:")
    for col in df_merged.columns:
        non_null = df_merged[col].notna().sum()
        pct = 100 * non_null / len(df_merged)
        print(f"  {col}: {non_null} values ({pct:.1f}%)")


if __name__ == "__main__":
    main()
