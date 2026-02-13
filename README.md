# Setup Guide

## Quick Start

### 1. Install Python Dependencies

Using pip with virtual environment (recommended):
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Or using pip with `--user` flag:
```bash
pip3 install --user -r requirements.txt
```

Or using conda:
```bash
conda install pandas numpy requests jupyter matplotlib seaborn
pip install meteostat
```

### 2. Verify NOAA Token

Your NOAA token is stored in `.env` file. To verify it works:

```bash
python3 test_noaa_connection.py
```

This will test:
- Token is loaded correctly
- API connection works
- Can retrieve station info for Miami
- Can fetch sample data

### 3. Start Collecting Data

#### Option A: Use Jupyter Notebooks (Recommended)
```bash
jupyter notebook
```

Then open:
- `spatial_features_collection.ipynb` - For multi-station spatial features
- `scripts/data_collection.ipynb` - For basic data collection

#### Option B: Use Python Scripts
You can convert the notebooks to scripts and run them directly.

## Project Structure

```
GroupProject/
├── .env                          # NOAA API token
├── requirements.txt              # Python dependencies
├── SETUP.md                      # This file
├── test_noaa_connection.py       # Test NOAA API connection
├── spatial_features_collection.ipynb  # Main notebook for spatial features
├── scripts/
│   └── data_collection.ipynb     # Basic data collection
├── data/
│   ├── STATIONS.md               # Documentation of all weather stations
│   ├── noaa/                     # NOAA CDO data
│   ├── open-meteo/               # Open-Meteo data
│   ├── meteostat/                # MeteoStat data
│   └── kalshi/                   # Kalshi data (future)
└── docs/
    └── Project.md                # Project documentation
```

## Data Sources

### 1. NOAA Climate Data Online (CDO)
- **Requires**: API token (already in `.env`)
- **Data**: Historical precipitation, temperature, wind
- **Coverage**: 1948-present for Miami
- **Rate Limits**: 5 req/sec, 10,000 req/day

### 2. Open-Meteo (No token required)
- **Data**: Historical weather back to 1940
- **Coverage**: Hourly resolution, 9km spatial resolution
- **Rate Limits**: None for reasonable usage

### 3. MeteoStat (No token required)
- **Data**: Historical weather from NOAA and other sources
- **Coverage**: Varies by station
- **Installation**: `pip install meteostat`

## Troubleshooting

### "Module not found" errors
Install the missing package:
```bash
pip3 install --user package_name
```

### NOAA API rate limit exceeded
Wait a few seconds between requests. The notebooks include rate limiting.

### Token not working
Check that `.env` file contains:
```
NOAA_TOKEN=your_actual_token
```

### Can't access .env in notebooks
Add this cell at the top of your notebook:
```python
import os
from pathlib import Path

# Load .env
env_file = Path('.env')
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
```

## Next Steps

1. Run `test_noaa_connection.py` to verify setup
2. Open `spatial_features_collection.ipynb` in Jupyter
3. Run the notebook cells to collect data
4. Data will be saved to `data/` folder
5. Build features for precipitation prediction model

## Getting Help

- NOAA CDO API Docs: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
- Open-Meteo Docs: https://open-meteo.com/en/docs
- MeteoStat Docs: https://dev.meteostat.net/python/
