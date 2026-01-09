## Koeppen Renewable Energy Potential

This repository produces renewable zone classification, inspired by Koeppen climate zones. An example is as below.

- Wind and solar power abundance (atlases)
- Seasonal and weather variability (ERA5)
- Potential demand (GHSL)

### Workflow
1. Abundance: Resample wind and solar atlas ERA5 grid (0.25° resolution)
2. Compute variability metrics using ERA5 data
3. Compute demand potential from ERA5 temperature, human settlement and industrial area data
4. Process global tiles

### Usage
```bash
conda env create -f environment.yaml
conda activate koeppen-renewables
```

Download the necessary data and put them in `/resources/user` 
- PV potential global https://globalsolaratlas.info/download/world  
`World_PVOUT_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF.zip` and unzip. Put the file `PVOUT.tif` in `/resources/user`.
PVOUT - Photovoltaic power potential [kWh/kWp] average daily totals, one data for year, convert to capacity factor by dividing by 24 hours (around 368 MB).

- Wind atlas capacity factor https://globalwindatlas.info/en/download/gis-files  
Use wind capacity factor for IEC I class https://globalwindatlas.info/en/about/dataset   “IEC Class - Fatigue Loads” Class II Fatigue Loads including Wake, Class III Extreme Loads.   
Filename is `cf_iec1_cog_100m.tif`.
Warning: the wind atlas file is heavy, 14.8 GB.

- Global Human Settlement Layer (GHSL) Total built-up surface and the component of non-residential (NRES) built-up surface (m2) GHS-BUILT-S (2023) https://human-settlement.emergency.copernicus.eu/ghs_buS2023.php. Coordinate system WGS84, resolution 30 arcsec, for the year 2023. Filename is `GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0_R8_C29.tif`.

- ERA5 daily data are processed directly from Earth Data Destine using zarr format. Token is required to download from EarthDatahub. You can register and get the token in https://earthdatahub.destine.eu/getting-started. Put this token in `.env`
```.env
ERA5_TOKEN = "your-token-here"
```
To upgrade the access to EarthDatahub, for example for large data use for academic purpose, check ou https://platform.destine.eu/access-policy-upgrade/.

This repository computes by 20×20° tiles:

For a quick test for a tile around the Netherlands.

```bash
pixi run test-NL
```

The default configuration is to run it globally, with tiles 20x20°, bounds longitude -180 to 180, latitude -60 to 80.

```bash
pixi run global
```

to run within provided bounds, also with tiles 20x20 incrementally (bounds: minx miny maxx maxy). 
For example, to check if the code runs correctly around the change of Prime Meridian and the Equator. To make sure pixi understand negative numnbers, add `argument terminator --` before the bound values.


```bash
pixi run python main.py --bounds -- -20 -10 10 20
```
Note that the results can return tiles outside of the given bounds in increments of 20° from the starting lat/lon.
