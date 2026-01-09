"""
Resample atlas data to reference grid
"""

import xarray as xr
import rioxarray as rxr
import numpy as np
from geo_processing import clip_and_resample, create_reference_raster


def resample_atlas(bounds, paths, resolution):
    ref = create_reference_raster(bounds, resolution)
    ds = xr.Dataset()

    solar = rxr.open_rasterio(paths["solar_atlas"]).squeeze()
    # Convert from Photovoltaic power potential [kWh/kWp] average daily totals to capacity factor
    solar_resampled = clip_and_resample(solar, ref, bounds) / 24
    # rename coordinates to ERA5 standard
    ds["solar_CF"] = solar_resampled.rename({"x": "longitude", "y": "latitude"})

    wind = rxr.open_rasterio(paths["wind_atlas"]).squeeze()
    wind_resampled = clip_and_resample(wind, ref, bounds)
    # rename coordinates to ERA5 standard
    ds["wind_CF"] = wind_resampled.rename({"x": "longitude", "y": "latitude"})

    # Ensure the output matches the exact data types and names of the ERA5 tiles
    ds = ds.assign_coords(
        {
            "longitude": ds.longitude.astype(np.float32),
            "latitude": ds.latitude.astype(np.float32),
        }
    )

    return ds
