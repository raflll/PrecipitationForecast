# South Florida Weather Stations for Spatial Features

## Station Network Map

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

## Primary Target Station

| Station ID | Name | Coordinates | Elevation | Notes |
|------------|------|-------------|-----------|-------|
| `GHCND:USW00012839` | Miami International Airport | 25.79°N, 80.32°W | 1.5m | Primary - Kalshi verification source |

**Data Available:**
- Precipitation (PRCP)
- Temperature (TMAX, TMIN, TAVG)
- Wind speed (AWND, WSF2)
- Wind direction (WDF2)
- Historical record: 1948-present

---

## Northern Stations (Atlantic Fronts Moving South)

### Fort Lauderdale Area

| Station ID | Name | Coordinates | Distance | Elevation |
|------------|------|-------------|----------|-----------|
| `GHCND:USW00012849` | Fort Lauderdale Intl Airport | 26.08°N, 80.16°W | 33 km | 0.9m |
| `GHCND:USW00012885` | Fort Lauderdale Executive AP | 26.20°N, 80.18°W | 47 km | 3.7m |
| `GHCND:USC00083163` | Fort Lauderdale (HCN) | 26.10°N, 80.20°W | 36 km | 4.9m |
| `GHCND:USC00083168` | Fort Lauderdale Beach | 26.14°N, 80.11°W | 40 km | 1.2m |

**Use for:** Detecting northern precipitation systems before they reach Miami

### Hollywood/Opa-Locka Area

| Station ID | Name | Coordinates | Distance | Elevation |
|------------|------|-------------|----------|-----------|
| `GHCND:USW00012882` | Opa Locka Airport | 25.91°N, 80.28°W | 14 km | 1.2m |
| `GHCND:USW00092809` | Hollywood North Perry AP | 26.00°N, 80.24°W | 24 km | 1.8m |
| `GHCND:USC00084050` | Hollywood | 26.03°N, 80.13°W | 28 km | 2.7m |

**Use for:** Local convection patterns, near-field precipitation

### Boca Raton (Far North)

| Station ID | Name | Coordinates | Distance | Elevation |
|------------|------|-------------|----------|-----------|
| `GHCND:USC00080845` | Boca Raton | 26.37°N, 80.11°W | 66 km | 4.3m |

**Use for:** Early warning of large-scale weather systems

---

## Southern Stations (Tropical Systems Moving North)

| Station ID | Name | Coordinates | Distance | Elevation |
|------------|------|-------------|----------|-----------|
| `GHCND:USW00012826` | Homestead AFB | 25.48°N, 80.38°W | 34 km | 4.9m |
| `GHCND:USC00084091` | Homestead Exp Station | 25.50°N, 80.50°W | 40 km | 3.4m |
| `GHCND:USC00084095` | Homestead Gen Aviation AP | 25.50°N, 80.55°W | 44 km | 2.7m |
| `GHCND:USC00085678` | Miami 12 SSW | 25.65°N, 80.30°W | 15 km | 3.0m |

**Use for:**
- Hurricane and tropical storm tracking
- Southerly flow precipitation
- Agricultural/Everglades edge effects

---

## Western Stations (Inland/Everglades Convection)

| Station ID | Name | Coordinates | Distance | Elevation |
|------------|------|-------------|----------|-----------|
| `GHCND:USW00012888` | Tamiami Executive Airport | 25.64°N, 80.43°W | 14 km | 1.8m |
| `GHCND:USC00088775` | Tamiami Canal | 25.77°N, 80.45°W | 12 km | 3.0m |
| `GHCND:USC00088780` | Tamiami Trail 40 Mi Bend | 25.76°N, 80.82°W | 49 km | 4.6m |

**Use for:**
- Afternoon convective initiation
- Sea breeze convergence
- Everglades moisture influence

---

## Eastern Stations (Coastal/Atlantic Influence)

| Station ID | Name | Coordinates | Distance | Elevation |
|------------|------|-------------|----------|-----------|
| `GHCND:USW00092811` | Miami Beach | 25.81°N, 80.13°W | 19 km | 2.4m |
| `GHCND:USC00085660` | Miami Beach 2 | 25.88°N, 80.13°W | 23 km | 3.0m |
| `GHCND:USC00085653` | Miami Bayfront Park | 25.78°N, 80.18°W | 14 km | 3.0m |

**Use for:**
- Ocean moisture transport
- Coastal vs inland precipitation differences
- Sea breeze effects

---

## Other Miami Metro Stations

| Station ID | Name | Coordinates | Distance | Elevation |
|------------|------|-------------|----------|-----------|
| `GHCND:USW00012852` | Miami MCAS | 25.92°N, 80.28°W | 16 km | 7.9m |
| `GHCND:USW00012859` | Miami WSO City | 25.72°N, 80.28°W | 4 km | 4.6m |
| `GHCND:USC00085667` | Miami NWSFO | 25.76°N, 80.38°W | 7 km | 4.3m |
| `GHCND:USC00088396` | South Miami 5 W | 25.70°N, 80.35°W | 4 km | 24.1m |
| `GHCND:USC00086315` | North Miami Beach #2 | 25.95°N, 80.22°W | 19 km | 3.0m |
| `GHCND:USC00081800` | Cooper City | 26.05°N, 80.28°W | 31 km | 3.4m |
| `GHCND:USC00081858` | Coral Springs | 26.27°N, 80.28°W | 54 km | 3.0m |

---

## Florida Keys Stations (Far South Reference)

| Station ID | Name | Coordinates | Distance | Elevation |
|------------|------|-------------|----------|-----------|
| `GHCND:USW00012836` | Key West Intl Airport | 24.55°N, 81.76°W | 169 km | 0.9m |
| `GHCND:USC00080747` | Big Pine Key | 24.72°N, 81.38°W | 131 km | 3.0m |
| `GHCND:USC00082441` | Duck Key | 24.77°N, 80.91°W | 87 km | 1.2m |
| `GHCND:USC00081795` | Conch Key | 24.78°N, 80.88°W | 85 km | 1.8m |

**Use for:** Tropical system early detection

---

## Spatial Feature Engineering Recommendations

### 1. Lagged Precipitation Features
- 1-day and 2-day lagged precipitation from all directional stations
- Captures upstream weather patterns moving toward Miami

### 2. Wind-Based Features
- Wind speed and direction gradients between stations
- Identify flow patterns (e.g., southerly flow from Keys)
- Upstream precipitation weighted by wind direction

### 3. Directional Aggregates
- **North average**: Mean of Fort Lauderdale, Hollywood, Opa-Locka
- **South average**: Mean of Homestead, Miami SSW
- **West average**: Mean of Tamiami stations
- **East average**: Mean of coastal stations (Miami Beach, Bayfront)

### 4. Distance-Weighted Features
- Inverse distance weighting from all nearby stations
- Closer stations have more influence

### 5. Coastal-Inland Gradient
- Difference between coastal (east) and inland (west) precipitation
- Indicates sea breeze convergence or frontal orientation

### 6. Spatial Precipitation Gradients
- North-south gradient: `(north_avg - south_avg) / distance`
- East-west gradient: `(coastal_avg - inland_avg) / distance`
- Captures synoptic-scale patterns

### 7. Temporal-Spatial Features
- Rolling 3-day average at each station
- Maximum precipitation in any direction (indicates active weather)
- Time since last precipitation event at each station

---

## Data Collection Strategy

1. **Start with core network** (10 stations):
   - Target: Miami Intl
   - North: Fort Lauderdale, Opa-Locka
   - South: Homestead
   - West: Tamiami Executive, Tamiami Canal
   - East: Miami Beach, Bayfront Park
   - Additional: Hollywood, Miami SSW

2. **Expand if needed**:
   - Add more northern stations for better frontal tracking
   - Add Keys stations for tropical system detection
   - Add inland Everglades stations for convection patterns

3. **Quality checks**:
   - Verify data completeness for each station
   - Check for systematic biases between stations
   - Handle missing data appropriately

4. **Historical coverage**:
   - Prioritize stations with long records (1948+)
   - Accept shorter records for spatial features if needed
   - Cross-validate with overlapping stations

---

## NOAA CDO Data Types Available

| Code | Description | Unit |
|------|-------------|------|
| `PRCP` | Precipitation | mm (tenths in raw) |
| `TMAX` | Maximum temperature | °C |
| `TMIN` | Minimum temperature | °C |
| `TAVG` | Average temperature | °C |
| `AWND` | Average wind speed | m/s |
| `WSF2` | Fastest 2-minute wind speed | m/s |
| `WDF2` | Direction of fastest 2-min wind | degrees |
| `SNOW` | Snowfall | mm |
| `SNWD` | Snow depth | mm |

For Miami area, focus on: **PRCP**, **AWND**, **WSF2**, **WDF2**

---

## References

- NOAA GHCND Documentation: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- Station inventory: https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt
- CDO Web Services: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
