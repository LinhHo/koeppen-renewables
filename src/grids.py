"""
Create grids of reference rasters
"""

import numpy as np
import xarray as xr


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
    b = 6356752.3142  # meters
    e = math.sqrt(1 - (b / a) ** 2)
    area_list = []
    for f in [center_lat + pixel_size / 2, center_lat - pixel_size / 2]:
        zm = 1 - e * math.sin(math.radians(f))
        zp = 1 + e * math.sin(math.radians(f))
        area_list.append(
            math.pi
            * b**2
            * (math.log(zp / zm) / (2 * e) + math.sin(math.radians(f)) / (zp * zm))
        )
    return pixel_size / 360.0 * (area_list[0] - area_list[1]) / 1e6


def determine_pixel_areas(raster_input):
    """Determine area of each pixel.

    Returns a raster in which the value corresponds to the area in [m2] of the pixel.
    based on T.Troendle determine_pixel_areas (utils.py and technically_eligible_area.py)
    This assumes the data comprises square pixel in WGS84.

    Parameters:
        crs: the coordinate reference system of the data (must be WGS84)
    """
    # the following is based on https://gis.stackexchange.com/a/288034/77760
    # and assumes the data to be in EPSG:4326
    assert (
        raster_input.rio.crs.to_epsg() == 4326
    ), "raster_input does not have the projection EPSG:4326"
    resolution = raster_input.rio.resolution()[0]  # resolution in degrees
    varea_of_pixel = np.vectorize(lambda lat: _area_of_pixel(resolution, lat))
    pixel_area = varea_of_pixel(raster_input.y) * 1000**2  # convert to m^2

    pixel_area_da = xr.DataArray(
        pixel_area,
        coords={"y": raster_input.y},
        dims="y",
    )
    return pixel_area_da



import numpy as np
import xarray as xr


def create_reference_raster(bounds, resolution, crs="EPSG:4326"):
    """
    Create a synthetic reference raster with given bounds and resolution.
    return:
    xr.DataArray
        Empty reference raster with correct grid and CRS
    """
    minx, miny, maxx, maxy = bounds

    lons = np.arange(minx, maxx + resolution, resolution)
    lats = np.arange(miny, maxy + resolution, resolution)

    da = xr.DataArray(
        np.zeros((len(lats), len(lons))),
        coords={"y": lats, "x": lons},
        dims=("y", "x"),
        name="reference",
    )

    return da.rio.write_crs(crs)
