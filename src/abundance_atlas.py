"""
Resample atlas data to reference grid
"""

import xarray as xr
import rioxarray as rxr
from geo_processing import clip_and_resample, create_tile_template


def resample_atlas(bounds, paths, resolution):
    template = create_tile_template(bounds, resolution)

    ds = xr.Dataset()
    # Solar PVOUT - Photovoltaic power potential [kWh/kWp] average daily totals
    # convert to capacity factor by dividing by 24 hours
    with rxr.open_rasterio(paths["solar_atlas"], chunks=True).squeeze() as solar_atlas:
        ds["solar_CF"] = clip_and_resample(solar_atlas, template) / 24

    # Wind power capacity factor (unitless)
    with rxr.open_rasterio(paths["wind_atlas"], chunks=True).squeeze() as wind_atlas:
        ds["wind_CF"] = clip_and_resample(wind_atlas, template)

    return ds
