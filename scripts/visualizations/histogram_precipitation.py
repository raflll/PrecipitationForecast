import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data/miami_clean.csv')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract year and month for grouping
df['year_month'] = df['date'].dt.to_period('M')

# Aggregate monthly precipitation
monthly_precip = df.groupby('year_month')['noaa_PRCP'].sum()

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(monthly_precip, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Monthly Precipitation (mm)')
plt.ylabel('Frequency')
plt.title('Distribution of Monthly Precipitation in Miami (NOAA PRCP)')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig('data/miami_monthly_precipitation_histogram.png', dpi=150)
plt.show()

print(f"Total months: {len(monthly_precip)}")
print(f"Mean monthly precipitation: {monthly_precip.mean():.2f} mm")
print(f"Median monthly precipitation: {monthly_precip.median():.2f} mm")
print(f"Min monthly precipitation: {monthly_precip.min():.2f} mm")
print(f"Max monthly precipitation: {monthly_precip.max():.2f} mm")
