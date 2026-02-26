# Data Sources Reference

Technical reference for all data sources used in this project.

## Observation Data

### NOAA Climate Data Online (CDO)

- **API**: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
- **Auth**: API token (free, from https://www.ncdc.noaa.gov/cdo-web/token)
- **Rate limits**: 5 req/sec, 10,000 req/day
- **Station**: `GHCND:USW00012839` (Miami International Airport, 25.79°N 80.32°W)
- **Record**: 1948-present, daily

**Variables used:**

| Column | Description | Unit |
|--------|-------------|------|
| `PRCP` | Precipitation (target variable) | mm |
| `TMAX` | Maximum temperature | °C |
| `TMIN` | Minimum temperature | °C |
| `WT01` | Fog | binary |
| `WT03` | Thunder | binary |
| `WT08` | Smoke/haze | binary |
| `WT16` | Rain | binary |

### Open-Meteo Archive API

- **API**: https://archive-api.open-meteo.com/v1/archive
- **Auth**: None
- **License**: CC BY 4.0
- **Resolution**: 9km, based on ERA5 reanalysis
- **Record**: 1940-present (5-day delay)

**Variables used:**

| Variable | Description | Unit |
|----------|-------------|------|
| `temperature_2m_max` | Daily max temperature | °C |
| `temperature_2m_min` | Daily min temperature | °C |
| `precipitation_sum` | Daily total precipitation | mm |
| `wind_speed_10m_max` | Max wind speed | km/h |
| `wind_gusts_10m_max` | Max wind gusts | km/h |
| `sunshine_duration` | Daily sunshine duration | seconds |
| `precipitation_hours` | Hours with precipitation | hours |

---

## GEFS Ensemble Forecast Data

### GEFS Reforecast (2000-2019)

Retrospective forecasts run with GEFSv12 model on historical initial conditions. Provides consistent model physics across 20 years.

| Attribute | Value |
|-----------|-------|
| Source | AWS S3: `noaa-gefs-retrospective` |
| Period | 2000-01-05 to 2019-12-25 (Wednesdays only) |
| Members | 11 (control + 10 perturbed) |
| Horizon | 35 days (840 hours) |
| Resolution | Days 1-10: 0.25° / 3-hourly; Days 10-35: 0.50° / 6-hourly |
| Variable | APCP (accumulated precipitation at surface) |
| Samples | ~1,040 (52 weeks x 20 years) |

**Script**: `scripts/pull_gefs_reforecast.py` downloads GRIB2 files from S3, decodes with cfgrib, extracts the nearest grid point to Miami, computes ensemble mean and spread, and saves to yearly parquet files.

### GEFS Operational (Sept 2020-present)

Real-time ensemble forecasts, accessed via the [Herbie](https://herbie.readthedocs.io/) library.

| Attribute | Value |
|-----------|-------|
| Source | NOAA NOMADS/AWS via Herbie |
| Product | `atmos.5` (0.5° resolution) |
| Period | 2020-09-23 to present |
| Members | Mean + spread (pre-computed by NOAA) |
| Horizon | 35 days (840 hours) |
| Forecast steps | Every 6 hours (6, 12, 18, ..., 840) |
| Variable | APCP (accumulated precipitation at surface) |

**Script**: `scripts/pull_gefs_operational.py` uses Herbie with connection pooling (20 workers) and retry logic. Supports resuming from the last complete date.

### Parquet Schema

Both reforecast and operational data share the same output schema:

| Column | Type | Description |
|--------|------|-------------|
| `init_date` | datetime | Forecast initialization date (00 UTC) |
| `forecast_hour` | int | Lead time in hours (6, 12, 18, ..., 840) |
| `valid_date` | datetime | `init_date + forecast_hour` |
| `apcp_mean` | float | Ensemble mean accumulated precipitation (mm) |
| `apcp_spread` | float | Ensemble standard deviation across members (mm) |

### Technical Notes

- GEFS uses 0-360° longitude convention (Miami = 279.7°E)
- All precipitation values are cumulative from forecast hour 0
- Extended forecasts (>384 hours) only available from Wednesday reforecast runs and 00 UTC operational runs
- GEFSv12 operational started September 23, 2020

---

## GEFS Variables Reference

The full set of variables available in GEFS (not all are currently used — only APCP is pulled):

### Precipitation and Moisture

| Variable | Search Pattern | Description | Units |
|----------|----------------|-------------|-------|
| `apcp` | `:APCP:surface:` | Accumulated precipitation | kg/m² |
| `pwat` | `:PWAT:` | Precipitable water | kg/m² |
| `rh_2m` | `:RH:2 m above ground:` | Relative humidity at 2m | % |
| `rh_850` | `:RH:850 mb:` | Relative humidity at 850mb | % |

### Convective Parameters

| Variable | Search Pattern | Description | Units |
|----------|----------------|-------------|-------|
| `cape` | `:CAPE:surface:` | Convective Available Potential Energy | J/kg |
| `cin` | `:CIN:surface:` | Convective Inhibition | J/kg |

### Temperature

| Variable | Search Pattern | Description | Units |
|----------|----------------|-------------|-------|
| `tmp_2m` | `:TMP:2 m above ground:` | Temperature at 2m | K |
| `tmp_850` | `:TMP:850 mb:` | Temperature at 850mb | K |
| `tmp_500` | `:TMP:500 mb:` | Temperature at 500mb | K |

### Wind

| Variable | Search Pattern | Description | Units |
|----------|----------------|-------------|-------|
| `ugrd_10m` | `:UGRD:10 m above ground:` | U-wind at 10m | m/s |
| `vgrd_10m` | `:VGRD:10 m above ground:` | V-wind at 10m | m/s |
| `ugrd_850` | `:UGRD:850 mb:` | U-wind at 850mb | m/s |
| `vgrd_850` | `:VGRD:850 mb:` | V-wind at 850mb | m/s |

### Clouds, Radiation, and Dynamics

| Variable | Search Pattern | Description | Units |
|----------|----------------|-------------|-------|
| `tcdc` | `:TCDC:` | Total cloud cover | % |
| `dswrf` | `:DSWRF:surface:` | Downward shortwave radiation | W/m² |
| `dlwrf` | `:DLWRF:surface:` | Downward longwave radiation | W/m² |
| `pres_msl` | `:PRMSL:` | Mean sea level pressure | Pa |
| `hgt_500` | `:HGT:500 mb:` | Geopotential height at 500mb | gpm |

---

## Reanalysis Sources (for future work)

These can serve as alternative ground truth or additional feature sources.

| Dataset | Period | Resolution | Access |
|---------|--------|------------|--------|
| ERA5 | 1940-present | 0.25° | [Copernicus CDS](https://cds.climate.copernicus.eu/) or Google Cloud |
| RTMA/URMA | 2011-present | 2.5km | Herbie |
| NARR | 1979-present | 32km | [NOAA PSL](https://psl.noaa.gov/data/gridded/data.narr.html) |
| CFSR/CFSv2 | 1979-present | 0.5° | NOAA NCEI |

---

## Station Network

### Primary Target

| Station ID | Name | Coordinates | Record |
|------------|------|-------------|--------|
| `GHCND:USW00012839` | Miami International Airport | 25.79°N, 80.32°W | 1948-present |

### Nearby Stations (for potential spatial features)

```
                    Boca Raton (66 km N)
                          |
                Fort Lauderdale (36 km N)
                          |
                    Hollywood (28 km N)
                          |
West: Tamiami -------- MIAMI INTL -------- East: Miami Beach
      (12 km)           (TARGET)                   (19 km)
                          |
                    Miami 12 SSW (15 km S)
                          |
                    Homestead (34 km S)
                          |
                    Key West (169 km S)
```

| Direction | Station | ID | Distance |
|-----------|---------|----|----------|
| North | Fort Lauderdale Intl AP | `USW00012849` | 33 km |
| North | Opa Locka AP | `USW00012882` | 14 km |
| North | Hollywood North Perry AP | `USW00092809` | 24 km |
| North | Boca Raton | `USC00080845` | 66 km |
| South | Homestead AFB | `USW00012826` | 34 km |
| South | Miami 12 SSW | `USC00085678` | 15 km |
| West | Tamiami Executive AP | `USW00012888` | 14 km |
| West | Tamiami Canal | `USC00088775` | 12 km |
| East | Miami Beach | `USW00092811` | 19 km |
| East | Miami Bayfront Park | `USC00085653` | 14 km |
| Far South | Key West Intl AP | `USW00012836` | 169 km |

These stations are not currently used in the model but are available for spatial feature engineering (lagged precipitation, wind gradients, coastal-inland differences).

---

## References

- NOAA CDO API: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
- Open-Meteo: https://open-meteo.com/en/docs
- GEFS Reforecast: https://registry.opendata.aws/noaa-gefs-reforecast/
- GEFS Operational: https://registry.opendata.aws/noaa-gefs/
- Herbie: https://herbie.readthedocs.io/
- ERA5: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
- GHCND stations: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
