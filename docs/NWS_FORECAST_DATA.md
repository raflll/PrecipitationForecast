# NWS Forecast and Reanalysis Data Sources

This document describes forecast and reanalysis data sources for the Miami monthly precipitation prediction project.

## Data Source Summary

### Forecast Data (via Herbie)

| Model | Date Range | Resolution | Update Freq | Members | Best For |
|-------|------------|------------|-------------|---------|----------|
| **GEFS Reforecast** | 2000-2019 | 0.25° | Daily 00 UTC | 5 (11 weekly) | Historical ensemble forecasts |
| **GEFS Operational** | 2017-present | 0.25° | 4x daily | 31 | Recent ensemble forecasts |
| **GFS** | 2015-present | 0.25° | 4x daily | 1 (deterministic) | Single best-guess forecast |
| **HRRR** | 2014-present | 3km | Hourly | 1 | High-res US forecasts |

### Analysis/Reanalysis Data (Ground Truth)

| Dataset | Date Range | Resolution | Temporal | Access Method |
|---------|------------|------------|----------|---------------|
| **RTMA/URMA** | 2011-present | 2.5km | Hourly | Herbie |
| **RAP Analysis** | 2012-present | 13km | Hourly | Herbie (fxx=0) |
| **HRRR Analysis** | 2014-present | 3km | Hourly | Herbie (fxx=0) |
| **GFS Analysis** | 2015-present | 0.25° | 6-hourly | Herbie (fxx=0) |
| **ERA5** | 1940-present | 0.25° | Daily available | CDS API |
| **NARR** | 1979-present | 32km | 3-hourly | NOAA PSL |
| **CFSR/CFSv2** | 1979-present | 0.5° | 6-hourly | NOAA NCEI |

---

## GEFS (Global Ensemble Forecast System)

### GEFS Reforecast (2000-2019) - PRIMARY HISTORICAL SOURCE

Retrospective forecasts run with GEFSv12 model on historical initial conditions. Ideal for ML training because it provides consistent model physics across 20 years.

**Access via Herbie:**
```python
from herbie import Herbie

H = Herbie(
    "2015-06-01",
    model="gefs_reforecast",
    fxx=24,                    # Forecast hour (0-240 for Days 1-10)
    member=0,                  # 0=control, 1-4=perturbed
    variable_level="apcp_sfc"  # Variable_level format
)
ds = H.xarray(":24 hour fcst:")  # Search for specific lead time
```

**Key Details:**
- **Date range:** 2000-01-01 to 2019-12-31
- **Init time:** 00 UTC only (once daily)
- **Members:** 5 daily (c00, p01-p04), 11 weekly
- **Forecast hours:** 6-hourly to Day 10, 6-hourly to Day 16, weekly runs extend to Day 35
- **AWS Source:** `s3://noaa-gefs-retrospective/GEFSv12/reforecast/`

**Available Variables (variable_level format):**
| Variable | Description | Level Options |
|----------|-------------|---------------|
| `apcp` | Total precipitation | `sfc` |
| `acpcp` | Convective precipitation | `sfc` |
| `pwat` | Precipitable water | `eatm` |
| `cape` | Convective Available Potential Energy | `sfc` |
| `cin` | Convective Inhibition | `sfc` |
| `tmp` | Temperature | `2m`, `sfc`, `pres` |
| `tmax` | Maximum temperature | `2m` |
| `tmin` | Minimum temperature | `2m` |
| `rh` | Relative humidity | `2m`, `pres` |
| `tcdc` | Total cloud cover | `eatm` |
| `ugrd` | U-wind component | `10m`, `pres` |
| `vgrd` | V-wind component | `10m`, `pres` |
| `dswrf` | Downward shortwave radiation | `sfc` |
| `pres` | Pressure | `msl`, `sfc` |
| `spfh` | Specific humidity | `2m` |
| `soilw` | Soil moisture | `bgrnd` |
| `gust` | Wind gust | `sfc` |
| `pbl` | Planetary boundary layer height | `sfc` |
| `hlcy` | Helicity | `sfc` |

### GEFS Operational (2017-present)

Real-time ensemble forecasts with 31 members.

**Access via Herbie:**
```python
H = Herbie(
    "2023-06-15 00:00",
    model="gefs",
    product="atmos.25",  # Quarter degree
    fxx=24,
    member=0             # 0=control, 1-30=perturbed, "mean", "spr"
)
ds = H.xarray("APCP:surface")
```

**Key Details:**
- **Date range:** 2017-present on AWS
- **Init times:** 00, 06, 12, 18 UTC
- **Members:** 31 (control + 30 perturbed), plus mean and spread
- **Products:** `atmos.25` (0.25°), `atmos.5` (0.5°)

---

## GFS (Global Forecast System)

Deterministic global model, same as GEFS control but higher temporal resolution.

**Access via Herbie:**
```python
H = Herbie("2023-06-15 12:00", model="gfs", product="pgrb2.0p25", fxx=24)
ds = H.xarray("TMP:2 m above ground")
```

**Key Details:**
- **Date range:** 2015-present on AWS
- **Resolution:** 0.25° (~28km)
- **Init times:** 00, 06, 12, 18 UTC
- **Forecast hours:** Hourly to Day 5, 3-hourly to Day 10, 12-hourly to Day 16

---

## Analysis Products (Herbie-Accessible Ground Truth)

These products blend model backgrounds with observations to create "analyzed" fields. Use these as ground truth for recent years (2011+).

### RTMA/URMA (Real-Time/Unrestricted Mesoscale Analysis)

Best observational analysis for CONUS, blends observations with model background.

```python
# RTMA Analysis - "truth" for verification
H = Herbie("2023-06-15 12:00", model="rtma", product="anl")
ds = H.xarray("TMP:2 m")

# URMA - same but with additional QC
H = Herbie("2023-06-15 12:00", model="urma", product="anl")
```

**Key Details:**
- **Date range:** 2011-present
- **Resolution:** 2.5km CONUS
- **Products:** `anl` (analysis), `err` (error), `pcp` (precipitation)
- **Update:** Hourly (RTMA), with delay (URMA)

### HRRR/RAP Analysis (fxx=0)

Model analysis fields initialized from observations.

```python
# HRRR Analysis - 3km resolution
H = Herbie("2023-06-15 12:00", model="hrrr", fxx=0)

# RAP Analysis - 13km resolution
H = Herbie("2023-06-15 12:00", model="rap", fxx=0)
```

**Key Details:**
- **HRRR:** 2014-present, 3km, hourly
- **RAP:** 2012-present, 13km, hourly

### GFS Analysis (fxx=0)

```python
H = Herbie("2023-06-15 00:00", model="gfs", fxx=0)
```

**Key Details:**
- **Date range:** 2015-present
- **Resolution:** 0.25°
- **Update:** 4x daily (00, 06, 12, 18 UTC)

---

## Reanalysis Data (Back to 2000+)

For ground truth data extending back to 2000 to match GEFS reforecast coverage, use these reanalysis products. **Not available in Herbie** but accessible via other APIs.

### ERA5 (Recommended for 2000+)

ECMWF's state-of-the-art global reanalysis. Same 0.25° resolution as GEFS.

**Key Details:**
- **Date range:** 1940-present
- **Resolution:** 0.25° (~28km)
- **Temporal:** Hourly (can aggregate to daily)
- **Access:** Copernicus CDS API (free account required)

**Daily aggregates available on Google Cloud:**
```python
import xarray as xr

# ERA5 daily on Google Cloud (no API key needed)
ds = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks={"time": 24}
)

# Subset to Miami and aggregate to daily
miami = ds.sel(latitude=25.77, longitude=-80.19, method="nearest")
daily = miami.resample(time="1D").mean()
```

**Via CDS API:**
```python
import cdsapi

c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['total_precipitation', '2m_temperature'],
        'year': [str(y) for y in range(2000, 2020)],
        'month': [f'{m:02d}' for m in range(1, 13)],
        'day': [f'{d:02d}' for d in range(1, 32)],
        'time': '12:00',  # Daily snapshot
        'area': [26.5, -81, 25, -79.5],  # N, W, S, E
        'format': 'netcdf',
    },
    'era5_miami_daily.nc'
)
```

**Variables:**
| ERA5 Variable | Description |
|---------------|-------------|
| `total_precipitation` | Total precipitation (m) |
| `2m_temperature` | 2-meter temperature (K) |
| `2m_dewpoint_temperature` | 2-meter dewpoint (K) |
| `mean_sea_level_pressure` | MSLP (Pa) |
| `10m_u_component_of_wind` | U-wind (m/s) |
| `10m_v_component_of_wind` | V-wind (m/s) |
| `total_cloud_cover` | Cloud cover (0-1) |
| `surface_solar_radiation_downwards` | Shortwave radiation (J/m²) |
| `convective_available_potential_energy` | CAPE (J/kg) |

### NARR (North American Regional Reanalysis)

NOAA regional reanalysis with precipitation assimilation.

**Key Details:**
- **Date range:** 1979-present
- **Resolution:** 32km
- **Temporal:** 3-hourly (aggregate to daily)
- **Access:** NOAA PSL, NCEI

**Access via NOAA PSL (OPeNDAP):**
```python
import xarray as xr

# NARR via OPeNDAP
url = "https://psl.noaa.gov/thredds/dodsC/Datasets/NARR/Dailies/monolevel/apcp.YYYY.nc"
ds = xr.open_dataset(url.replace("YYYY", "2015"))
miami = ds.sel(lat=25.77, lon=-80.19, method="nearest")
```

**Key Variables:**
- `apcp` - Accumulated precipitation
- `air.2m` - 2-meter temperature
- `rhum.2m` - 2-meter relative humidity
- `uwnd.10m`, `vwnd.10m` - 10-meter winds
- `prmsl` - Mean sea level pressure

### CFSR/CFSv2 (Climate Forecast System Reanalysis)

NCEP's global reanalysis, continuous from 1979.

**Key Details:**
- **CFSR:** 1979-2010
- **CFSv2:** 2011-present
- **Resolution:** 0.5° (~55km)
- **Temporal:** 6-hourly
- **Access:** NOAA NCEI, NCAR RDA

---

## Miami Coordinates

**MIA2 CORS Station (primary reference point):**
- Latitude: 25.77°N (25°46'12"N)
- Longitude: 80.19°W (80°11'24"W)
- Longitude (0-360): 279.81°E

**Bounding box (~50km radius):**
- North: 26.3°N
- South: 25.3°N
- West: 80.7°W (279.3°E)
- East: 79.7°W (280.3°E)

---

## Recommended Data Strategy

### For Training (2000-2019):
1. **Forecasts:** GEFS Reforecast (consistent model physics)
2. **Ground Truth:** ERA5 daily (same 0.25° resolution, covers full period)

### For Validation/Testing (2020-present):
1. **Forecasts:** GEFS Operational
2. **Ground Truth:** RTMA/URMA analysis (best observational product for CONUS)

### Residual Regression Approach:
```
Residual = ERA5_observed - GEFS_forecast
Model learns: Residual ~ f(GEFS_variables, location, season, etc.)
Corrected forecast = GEFS_forecast + predicted_residual
```

---

## Data Collection Scripts

| Script | Purpose |
|--------|---------|
| `scripts/data_collection/fetch_gefs_miami.py` | Fetch GEFS reforecast and operational data |
| `scripts/data_collection/fetch_gfs_full_day.py` | Fetch GFS data (single runs) |

---

## References

- GEFS Reforecast: https://registry.opendata.aws/noaa-gefs-reforecast/
- GEFS Operational: https://registry.opendata.aws/noaa-gefs/
- Herbie Documentation: https://herbie.readthedocs.io/
- ERA5: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
- NARR: https://psl.noaa.gov/data/gridded/data.narr.html
- RTMA/URMA: https://www.nco.ncep.noaa.gov/pmb/products/rtma/
