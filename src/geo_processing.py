"""
clip, resample, pixel area calculation functions for geospatial rasters
"""

import math
import numpy as np
import xarray as xr
from rasterio.enums import Resampling
from rioxarray.exceptions import NoDataInBounds


def clip_and_resample(input_raster, reference_raster, bounds,
                      resampling=Resampling.average):
    """Clip raster to bounding box and resample to ERA5 grid."""
    minx, miny, maxx, maxy = bounds
    try:
        clipped = input_raster.rio.clip_box(
            minx=minx, miny=miny, maxx=maxx, maxy=maxy
        )
        return clipped.rio.reproject_match(
            reference_raster,
            resampling=resampling,
            nodata=np.nan,
            masked=True,
        )
    except NoDataInBounds:
        return xr.full_like(reference_raster, np.nan)


def _area_of_pixel(pixel_size, center_lat):
    """Calculate km^2 area of a wgs84 square pixel.

    Adapted from: https://gis.stackexchange.com/a/127327/2397

    Parameters:
        pixel_size (float): length of side of pixel in degrees.
        center_lat (float): latitude of the center of the pixel. Note this
            value +/- half the `pixel-size` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.

    Returns:
        Area of square pixel of side length `pixel_size` centered at
        `center_lat` in km^2.

    """
    a = 6378137  # meters
    b = 6356752.3142 # meters
    e = math.sqrt(1 - (b / a) ** 2)

    area = []
    for f in [center_lat + pixel_size / 2, center_lat - pixel_size / 2]:
        zm = 1 - e * math.sin(math.radians(f))
        zp = 1 + e * math.sin(math.radians(f))
        area.append(
            math.pi * b**2 *
            (math.log(zp / zm) / (2 * e) + math.sin(math.radians(f)) / (zp * zm))
        )

    return pixel_size / 360 * (area[0] - area[1]) / 1e6


def determine_pixel_areas(raster):
    """Determine area of each pixel.

    Returns a raster in which the value corresponds to the area in [m2] of the pixel.
    based on T.Troendle determine_pixel_areas (utils.py and technically_eligible_area.py)
    This assumes the data comprises square pixel in WGS84.

    Parameters:
        crs: the coordinate reference system of the data (must be WGS84)
    """
    # the following is based on https://gis.stackexchange.com/a/288034/77760
    # and assumes the data to be in EPSG:4326
    assert raster.rio.crs.to_epsg() == 4326
    res = raster.rio.resolution()[0]

    vfunc = np.vectorize(lambda lat: _area_of_pixel(res, lat))
    areas = vfunc(raster.y) * 1e6

    return xr.DataArray(areas, coords={"y": raster.y}, dims="y")

    # assert (
    #     raster_input.rio.crs.to_epsg() == 4326
    # ), "raster_input does not have the projection EPSG:4326"
    # resolution = raster_input.rio.resolution()[0]  # resolution in degrees
    # varea_of_pixel = np.vectorize(lambda lat: _area_of_pixel(resolution, lat))
    # pixel_area = varea_of_pixel(raster_input.y) * 1000**2  # convert to m^2

    # pixel_area_da = xr.DataArray(
    #     pixel_area,
    #     coords={"y": raster_input.y},
    #     dims="y",
    # )
    # return pixel_area_da