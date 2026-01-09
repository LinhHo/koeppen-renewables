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
    ds["solar_CF"] = (
        clip_and_resample(
            rxr.open_rasterio(
                paths["solar_atlas"]
            ).squeeze(),
            template,
        )
        / 24
    )
    ds["wind_CF"] = clip_and_resample(
        rxr.open_rasterio(paths["wind_atlas"]).squeeze(),
        template,
    )

    return ds
