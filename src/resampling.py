"""
Resample atlas data to reference grid
"""


import xarray as xr
import rioxarray as rxr
from geo_processing import clip_and_resample, determine_pixel_areas
from grids import create_reference_raster


def resample_tile(bounds, paths, resolution):
    ref = create_reference_raster(bounds, resolution)
    ds = xr.Dataset()

    solar = rxr.open_rasterio(paths["solar_atlas"]).squeeze()
    ds["solar_CF"] = clip_and_resample(solar, ref, bounds) / 24

    wind = rxr.open_rasterio(paths["wind_atlas"]).squeeze()
    ds["wind_CF"] = clip_and_resample(wind, ref, bounds)

    built = rxr.open_rasterio(paths["ghsl"]).squeeze().astype("float32")
    settlement = clip_and_resample(built, ref, bounds)

    pixel_area = (
        determine_pixel_areas(settlement)
        .expand_dims(x=settlement.x)
        .transpose("y", "x")
    )

    ds["settlement_share"] = settlement / pixel_area

    return ds
