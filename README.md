## Koeppen Renewable Energy Potential

This repository computes global 20×20° gridded:
- Wind and solar power abundance (atlases)
- Seasonal and weather variability (ERA5)
- Potential demand (GHSL)

### Workflow
1. Resample atlas and settlement data to ERA5 grid
2. Compute variability metrics using CDO
3. Process global tiles

### Usage
```bash
conda env create -f environment.yml
conda activate koeppen-renewables

python scripts/run_resample.py -180 -60 -160 -40 output.nc
bash scripts/run_variability.sh ...
```

This repository produces renewable zone classification, inspired by Koeppen climate zones. An example is as below.


To run this repository:

CDO must be installed to process weather and seasonal variability from ERA5 data.

Create environment by
`conda env create -f environment.yaml`

Download the necessary data and put them in `/resources` capacity factor of wind and solar
- PV potential global https://globalsolaratlas.info/download/world  
`World_PVOUT_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF.zip` and unzip. Put the file `PVOUT.tif` in `/resources/user`.
PVOUT - Photovoltaic power potential [kWh/kWp] average daily totals, one data for year, convert to capacity factor by dividing by 24 hours.
Around 368 MB.

- Wind atlas power density https://globalwindatlas.info/en/download/gis-files  
Use wind capacity factor for IEC I class https://globalwindatlas.info/en/about/dataset   “IEC Class - Fatigue Loads” Class II Fatigue Loads including Wake, Class III Extreme Loads.   
Put the file `cf_iec1_cog_100m.tif` to `/resources/user`.
Warning: the wind atlas file is heavy, 14.8 GB. 

- ERA5 daily data are processed directly from Earth Destine
using zarr


to run

`python koeppen-renewables`
OR
`python koeppen-renewables --global`
default, run globally, with tiles 20x20 

OR
`python koeppen-renewables --bounds -20 0 0 20`
to run within provided bounds, also with tiles 20x20

`python koeppen-renewables --global --bounds `
will return error