"""
Monthly Precipitation Comparison
Compares monthly precipitation totals between NOAA and Open-Meteo datasets
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# File paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
NOAA_FILE = DATA_DIR / "noaa" / "miami_ghcnd.csv"
OPENMETEO_FILE = DATA_DIR / "open-meteo" / "miami_daily.csv"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs" / "visualizations"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_process_data():
    """Load and process precipitation data from both sources."""

    # Load NOAA data
    print("Loading NOAA data...")
    df_noaa = pd.read_csv(NOAA_FILE, parse_dates=['date'])
    df_noaa['year_month'] = df_noaa['date'].dt.to_period('M')
    # NOAA PRCP appears to already be in mm (not tenths of mm as documented)
    # Testing shows values are already in mm, so no conversion needed
    df_noaa['prcp_mm'] = df_noaa['PRCP']

    # Group by month and sum precipitation
    noaa_monthly = df_noaa.groupby('year_month')['prcp_mm'].sum().reset_index()
    noaa_monthly.columns = ['year_month', 'noaa_prcp_mm']
    noaa_monthly['date'] = noaa_monthly['year_month'].dt.to_timestamp()

    print(f"NOAA: {len(df_noaa)} daily records")
    print(f"NOAA date range: {df_noaa['date'].min()} to {df_noaa['date'].max()}")

    # Load Open-Meteo data
    print("\nLoading Open-Meteo data...")
    df_openmeteo = pd.read_csv(OPENMETEO_FILE, parse_dates=['time'])
    df_openmeteo['year_month'] = df_openmeteo['time'].dt.to_period('M')
    # Open-Meteo precipitation_sum is already in mm

    # Group by month and sum precipitation
    openmeteo_monthly = df_openmeteo.groupby('year_month')['precipitation_sum'].sum().reset_index()
    openmeteo_monthly.columns = ['year_month', 'openmeteo_prcp_mm']
    openmeteo_monthly['date'] = openmeteo_monthly['year_month'].dt.to_timestamp()

    print(f"Open-Meteo: {len(df_openmeteo)} daily records")
    print(f"Open-Meteo date range: {df_openmeteo['time'].min()} to {df_openmeteo['time'].max()}")

    # Merge datasets
    df_monthly = pd.merge(
        noaa_monthly[['date', 'noaa_prcp_mm']],
        openmeteo_monthly[['date', 'openmeteo_prcp_mm']],
        on='date',
        how='outer'
    )
    df_monthly = df_monthly.sort_values('date').reset_index(drop=True)

    print(f"\nMerged dataset: {len(df_monthly)} months")
    print(f"Date range: {df_monthly['date'].min()} to {df_monthly['date'].max()}")

    return df_monthly

def plot_comparison(df_monthly):
    """Create comparison plots of monthly precipitation."""

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Plot 1: Time series comparison
    ax1 = axes[0]
    ax1.plot(df_monthly['date'], df_monthly['noaa_prcp_mm'],
             label='NOAA', alpha=0.7, linewidth=1, color='blue')
    ax1.plot(df_monthly['date'], df_monthly['openmeteo_prcp_mm'],
             label='Open-Meteo', alpha=0.7, linewidth=1, color='orange')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Monthly Precipitation (mm)')
    ax1.set_title('Monthly Precipitation Comparison: NOAA vs Open-Meteo')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter plot (correlation)
    ax2 = axes[1]
    # Only plot where both datasets have data
    mask = df_monthly['noaa_prcp_mm'].notna() & df_monthly['openmeteo_prcp_mm'].notna()
    df_both = df_monthly[mask]

    if len(df_both) > 0:
        ax2.scatter(df_both['noaa_prcp_mm'], df_both['openmeteo_prcp_mm'],
                   alpha=0.5, s=20)

        # Add diagonal line (perfect correlation)
        max_val = max(df_both['noaa_prcp_mm'].max(), df_both['openmeteo_prcp_mm'].max())
        ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect correlation')

        # Calculate correlation
        corr = df_both['noaa_prcp_mm'].corr(df_both['openmeteo_prcp_mm'])
        ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2.set_xlabel('NOAA Monthly Precipitation (mm)')
        ax2.set_ylabel('Open-Meteo Monthly Precipitation (mm)')
        ax2.set_title('Precipitation Correlation Between Datasets')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Difference over time
    ax3 = axes[2]
    df_monthly['difference'] = df_monthly['openmeteo_prcp_mm'] - df_monthly['noaa_prcp_mm']
    ax3.plot(df_monthly['date'], df_monthly['difference'],
             color='green', alpha=0.7, linewidth=1)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Difference (mm)\n(Open-Meteo - NOAA)')
    ax3.set_title('Monthly Precipitation Difference')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = OUTPUT_DIR / "monthly_precipitation_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    plt.show()

def print_statistics(df_monthly):
    """Print summary statistics."""

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    # Filter to overlapping period
    mask = df_monthly['noaa_prcp_mm'].notna() & df_monthly['openmeteo_prcp_mm'].notna()
    df_both = df_monthly[mask]

    if len(df_both) > 0:
        print(f"\nOverlapping period: {len(df_both)} months")
        print(f"From {df_both['date'].min()} to {df_both['date'].max()}")

        print("\nNOAA Statistics:")
        print(f"  Mean monthly precipitation: {df_both['noaa_prcp_mm'].mean():.2f} mm")
        print(f"  Median: {df_both['noaa_prcp_mm'].median():.2f} mm")
        print(f"  Std Dev: {df_both['noaa_prcp_mm'].std():.2f} mm")
        print(f"  Min: {df_both['noaa_prcp_mm'].min():.2f} mm")
        print(f"  Max: {df_both['noaa_prcp_mm'].max():.2f} mm")

        print("\nOpen-Meteo Statistics:")
        print(f"  Mean monthly precipitation: {df_both['openmeteo_prcp_mm'].mean():.2f} mm")
        print(f"  Median: {df_both['openmeteo_prcp_mm'].median():.2f} mm")
        print(f"  Std Dev: {df_both['openmeteo_prcp_mm'].std():.2f} mm")
        print(f"  Min: {df_both['openmeteo_prcp_mm'].min():.2f} mm")
        print(f"  Max: {df_both['openmeteo_prcp_mm'].max():.2f} mm")

        print("\nDifference Statistics:")
        diff = df_both['openmeteo_prcp_mm'] - df_both['noaa_prcp_mm']
        print(f"  Mean difference: {diff.mean():.2f} mm")
        print(f"  Mean absolute difference: {diff.abs().mean():.2f} mm")
        print(f"  RMSE: {np.sqrt((diff**2).mean()):.2f} mm")

        corr = df_both['noaa_prcp_mm'].corr(df_both['openmeteo_prcp_mm'])
        print(f"\nCorrelation coefficient: {corr:.4f}")

def main():
    """Main execution function."""
    print("Monthly Precipitation Comparison")
    print("="*60)

    # Load and process data
    df_monthly = load_and_process_data()

    # Print statistics
    print_statistics(df_monthly)

    # Create plots
    plot_comparison(df_monthly)

    print("\nDone!")

if __name__ == "__main__":
    main()
