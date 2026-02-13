"""
Generate histogram of monthly rainfall in Miami from GHCND data.
Includes fitted gamma distribution overlay for proposal figure.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the Miami GHCND data
df = pd.read_csv('data/noaa/miami_ghcnd.csv')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract year and month for grouping
df['year_month'] = df['date'].dt.to_period('M')

# Aggregate monthly precipitation (PRCP column)
monthly_precip = df.groupby('year_month')['PRCP'].sum()

# Remove any zero or negative values for gamma fit (gamma requires positive values)
monthly_precip_positive = monthly_precip[monthly_precip > 0]

# Fit gamma distribution to the data
shape, loc, scale = stats.gamma.fit(monthly_precip_positive, floc=0)
alpha = shape
beta = 1 / scale  # Convert scale to rate

# Create x values for plotting the fitted distribution
x = np.linspace(0, monthly_precip.max() * 1.1, 200)
gamma_pdf = stats.gamma.pdf(x, shape, loc=0, scale=scale)

# Create histogram with gamma overlay
fig, ax = plt.subplots(figsize=(8, 5))

# Plot histogram (normalized to density for comparison with PDF)
n, bins, patches = ax.hist(monthly_precip, bins=50, density=True,
                           edgecolor='black', alpha=0.7, color='steelblue',
                           label='Observed')

# Plot fitted gamma distribution
ax.plot(x, gamma_pdf, 'r-', linewidth=2,
        label=f'Gamma fit ($\\alpha$={alpha:.2f}, $\\beta$={beta:.4f})')

ax.set_xlabel('Monthly Precipitation (mm)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distribution of Monthly Precipitation in Miami', fontsize=12)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/miami_monthly_rainfall_histogram.pdf', dpi=300, bbox_inches='tight')
plt.savefig('visualizations/miami_monthly_rainfall_histogram.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Histogram saved to visualizations/miami_monthly_rainfall_histogram.pdf")
print(f"\nSummary Statistics:")
print(f"  Total months: {len(monthly_precip)}")
print(f"  Mean monthly precipitation: {monthly_precip.mean():.2f} mm")
print(f"  Median monthly precipitation: {monthly_precip.median():.2f} mm")
print(f"  Std deviation: {monthly_precip.std():.2f} mm")
print(f"  Min monthly precipitation: {monthly_precip.min():.2f} mm")
print(f"  Max monthly precipitation: {monthly_precip.max():.2f} mm")
print(f"\nFitted Gamma Parameters:")
print(f"  Shape (α): {alpha:.4f}")
print(f"  Rate (β): {beta:.6f}")
print(f"  Scale (1/β): {scale:.4f}")
