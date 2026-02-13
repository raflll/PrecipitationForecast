The goal of this project is to come up with a distribution for precipitation in Miami on a given day.

We will use historical data on kalshi as well as historical weather data as a baseline so that we can ensure we are coming up with competitive predictions.

## Data Sources

### 1. NWS (National Weather Service) Forecast API

**API Documentation:** https://www.weather.gov/documentation/services-web-api

- **Cost:** Free (public service, no fees)
- **Rate Limits:** Reasonable limits to prevent abuse (no hard numbers published)
- **Auth:** None required
- **Format:** JSON-LD

**Key Endpoints:**
- Grid forecast: `https://api.weather.gov/gridpoints/{office}/{gridX},{gridY}/forecast`
- Miami office code: `MFL`
- Resolution: ~2.5km x 2.5km grid

**Available Data:**
- 12-hour forecast periods
- Quantitative Precipitation Forecasts (QPF)
- Probabilistic precipitation forecasts
- Hourly forecasts

---

### 2. NOAA Climate Data Online (CDO)

**API Documentation:** https://www.ncdc.noaa.gov/cdo-web/webservices/v2

- **Cost:** Free
- **Rate Limits:** 5 requests/second, 10,000 requests/day per token
- **Auth:** API token required (free, request at https://www.ncdc.noaa.gov/cdo-web/token)
- **Format:** JSON

**Base URL:** `https://www.ncei.noaa.gov/cdo-web/api/v2/{endpoint}`

**Relevant Datasets:**
- `GHCND` - Global Historical Climatology Network Daily
- `PRECIP_15` - 15-minute precipitation data
- Hourly Precipitation Data (DSI-3240) - ~5,500 stations, data back to 1900 for some states

**Example Request:**
```
https://www.ncei.noaa.gov/cdo-web/api/v2/data?datasetid=PRECIP_15&stationid=COOP:010008&units=metric&startdate=2010-05-01&enddate=2010-05-31
```

**Download Formats:** CSV, XML, Shapefile, KMZ, NetCDF

---

### 3. MeteoStat

**Documentation:** https://dev.meteostat.net/python

- **Cost:** Free
- **Rate Limits:** None specified
- **Auth:** None required
- **License:** MIT (code), CC BY-NC 4.0 (data - non-commercial redistribution restricted)

**Installation:**
```bash
pip install meteostat
```

**Features:**
- Historical observations and statistics
- Data from NOAA, DWD, and other national weather services
- Point-based queries with lat/lon
- Interpolation for locations without stations
- Climate normals (30-year averages)
- Returns data as Pandas DataFrames

**Python Usage:**
```python
from meteostat import Point, Daily
from datetime import datetime

miami = Point(25.7617, -80.1918)
data = Daily(miami, datetime(2020, 1, 1), datetime(2024, 12, 31))
data = data.fetch()
```

---

### 4. Open-Meteo

**Documentation:** https://open-meteo.com/en/docs/historical-weather-api

- **Cost:** Free for non-commercial use
- **Rate Limits:** None for reasonable usage
- **Auth:** No API key required
- **License:** CC BY 4.0 (free for commercial use with attribution)

**Historical Endpoint:** `/v1/archive`

**Features:**
- Data from 1940 to present (2-day delay)
- Hourly resolution
- 9km spatial resolution
- Based on ERA5 reanalysis data
- Lightning fast (decades of data in <1 second)

**Precipitation Variables:**
- `precipitation` (default unit: mm)
- `precipitation_sum` (daily totals)

**Example Request:**
```
https://archive-api.open-meteo.com/v1/archive?latitude=25.76&longitude=-80.19&start_date=2020-01-01&end_date=2024-12-31&hourly=precipitation
```

---

### 5. Kalshi Historical Data (Future Work)

*To be implemented - will require Kalshi API access or manual data collection for historical prediction market prices and outcomes.*

---

## Florida NOAA GHCND Stations

### Primary Target Station
| Station ID | Name | Lat | Lon | Notes |
|------------|------|-----|-----|-------|
| `GHCND:USW00012839` | Miami International Airport | 25.79 | -80.32 | Primary - Kalshi verification |

### South Florida Stations for Spatial Features

#### Major Airport Stations (WBAN - High Quality)
| Station ID | Name | Lat | Lon | Direction from Miami |
|------------|------|-----|-----|---------------------|
| `GHCND:USW00012849` | Fort Lauderdale Intl Airport | 26.08 | -80.16 | North (33 km) |
| `GHCND:USW00012888` | Tamiami Executive Airport | 25.64 | -80.43 | Southwest (14 km) |
| `GHCND:USW00012882` | Opa Locka Airport | 25.91 | -80.28 | North-Northwest (14 km) |
| `GHCND:USW00092809` | Hollywood North Perry Airport | 26.00 | -80.24 | North (24 km) |
| `GHCND:USW00012826` | Homestead AFB | 25.48 | -80.38 | South (34 km) |
| `GHCND:USW00012885` | Fort Lauderdale Executive | 26.20 | -80.18 | North (47 km) |

#### COOP Stations (Long Historical Record)
| Station ID | Name | Lat | Lon | Direction from Miami |
|------------|------|-----|-----|---------------------|
| `GHCND:USC00083163` | Fort Lauderdale (HCN) | 26.10 | -80.20 | North (36 km) |
| `GHCND:USC00084050` | Hollywood | 26.03 | -80.13 | North (28 km) |
| `GHCND:USC00085678` | Miami 12 SSW | 25.65 | -80.30 | South (15 km) |
| `GHCND:USC00084091` | Homestead Exp Station | 25.50 | -80.50 | South-Southwest (40 km) |
| `GHCND:USC00088775` | Tamiami Canal | 25.77 | -80.45 | West (12 km) |
| `GHCND:USC00080845` | Boca Raton | 26.37 | -80.11 | North (66 km) |
| `GHCND:USC00085653` | Miami Bayfront Park | 25.78 | -80.18 | East (14 km) |
| `GHCND:USC00085660` | Miami Beach | 25.88 | -80.13 | Northeast (19 km) |

#### Key West Area (South Reference)
| Station ID | Name | Lat | Lon | Distance from Miami |
|------------|------|-----|-----|---------------------|
| `GHCND:USW00012836` | Key West Intl Airport | 24.55 | -81.76 | 169 km |
| `GHCND:USC00080747` | Big Pine Key | 24.72 | -81.38 | 131 km) |

### Spatial Feature Engineering Strategy

**Goal:** Capture how precipitation and wind patterns in nearby areas influence Miami precipitation.

**Directional Coverage:**
- **North**: Fort Lauderdale stations - Atlantic fronts moving south
- **South**: Homestead/Keys - tropical systems moving north
- **West**: Tamiami/Everglades - inland convection patterns
- **East**: Coastal stations - ocean moisture influence

**Recommended Features:**
1. **Lagged precipitation** at surrounding stations (1-day, 2-day lag)
2. **Wind direction gradients** between stations
3. **Precipitation gradient** (spatial rate of change)
4. **Upstream precipitation** based on prevailing wind direction
5. **Distance-weighted averages** from all nearby stations
6. **Coastal vs inland difference** in precipitation

**Note:** Florida Climate Center maintains precipitation data back to ~1900 for many stations, with some records as far back as the 1830s.

### Query Florida Stations via API
```
GET https://www.ncei.noaa.gov/cdo-web/api/v2/stations?locationid=FIPS:12&datasetid=GHCND&datatypeid=PRCP
Header: token: YOUR_TOKEN
```

---

## NWS Historical Data Limitations

**Important:** The NWS API (api.weather.gov) does **NOT** provide historical forecast archives.

- Alerts: Only past 7 days
- Forecasts: Current only (no historical archive)
- Climatology: ~5 years at some locations

**For historical data, use NOAA CDO instead.**

---

## MeteoStat Variables

### Daily Data Columns
| Column | Description | Unit |
|--------|-------------|------|
| `tavg` | Average temperature | °C |
| `tmin` | Minimum temperature | °C |
| `tmax` | Maximum temperature | °C |
| `prcp` | Precipitation | mm |
| `snow` | Snowfall | mm |
| `wdir` | Wind direction | ° |
| `wspd` | Average wind speed | km/h |
| `wpgt` | Peak wind gust | km/h |
| `pres` | Sea-level pressure | hPa |
| `tsun` | Sunshine duration | min |

### Hourly Data Columns
| Column | Description | Unit |
|--------|-------------|------|
| `temp` | Air temperature | °C |
| `dwpt` | Dew point | °C |
| `rhum` | Relative humidity | % |
| `prcp` | Precipitation (1-hour) | mm |
| `snow` | Snow depth | mm |
| `wdir` | Wind direction | ° |
| `wspd` | Average wind speed | km/h |
| `wpgt` | Peak wind gust | km/h |
| `pres` | Sea-level pressure | hPa |
| `tsun` | Sunshine (1-hour) | min |
| `coco` | Weather condition code | - |

---

## Open-Meteo Variables

### Hourly Variables
- `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`
- `precipitation`, `rain`, `snowfall`
- `pressure_msl`, `surface_pressure`
- `cloud_cover`, `cloud_cover_low/mid/high`
- `wind_speed_10m`, `wind_direction_10m`, `wind_gusts_10m`
- `soil_temperature_0_to_7cm` (and deeper levels)
- `soil_moisture_0_to_7cm` (and deeper levels)

### Daily Aggregations
- `temperature_2m_max/min`, `precipitation_sum`, `rain_sum`
- `wind_speed_10m_max`, `wind_gusts_10m_max`
- `sunshine_duration`, `precipitation_hours`

---

## Consolidated Dataset Strategy

### Overlapping Time Period
All sources have data from **1948 to present** (using Miami International Airport as anchor):

| Source | Start Date | End Date | Resolution |
|--------|------------|----------|------------|
| NOAA GHCND (Miami) | 1948-01-01 | Present | Daily |
| Open-Meteo | 1940-01-01 | Present - 5 days | Hourly |
| MeteoStat | Varies by station | Present | Daily/Hourly |

### Recommended Approach
1. **Primary target variable:** NOAA GHCND `PRCP` at Miami International Airport (`USW00012839`)
2. **Additional features from:**
   - Open-Meteo: Temperature, humidity, pressure, wind, cloud cover (hourly → aggregate to daily)
   - MeteoStat: Can cross-validate NOAA data and fill gaps
3. **Time range:** 1948-01-01 to present (or 1940 if using Open-Meteo as primary)
4. **Join key:** Date (daily granularity)

### Feature Set for Model
From combined sources, these features are available for the same time periods:
- Temperature (avg, min, max)
- Precipitation (target variable from NOAA)
- Humidity / Dew point
- Wind speed and direction
- Atmospheric pressure
- Cloud cover
- Soil moisture/temperature (Open-Meteo only)

---

## Data Directory

All downloaded data should be stored in the `/data` folder with appropriate subdirectories:
- `/data/noaa/` - NOAA CDO historical data (primary)
- `/data/meteostat/` - MeteoStat exports
- `/data/open-meteo/` - Open-Meteo historical data
- `/data/kalshi/` - Kalshi market data (future)
