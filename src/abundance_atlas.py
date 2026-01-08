"""
Resample atlas data to reference grid
"""


import xarray as xr
import rioxarray as rxr
from geo_processing import clip_and_resample, create_reference_raster


def resample_atlas(bounds, paths, resolution):
    ref = create_reference_raster(bounds, resolution)
    ds = xr.Dataset()

    solar = rxr.open_rasterio(paths["solar_atlas"]).squeeze()
    # Convert from Photovoltaic power potential [kWh/kWp] average daily totals to capacity factor
    ds["solar_CF"] = clip_and_resample(solar, ref, bounds) / 24 

    wind = rxr.open_rasterio(paths["wind_atlas"]).squeeze()
    ds["wind_CF"] = clip_and_resample(wind, ref, bounds)

    return ds
