import math
import numpy as np
import xarray as xr
from rasterio.enums import Resampling
from rioxarray.exceptions import NoDataInBounds
from typing import Iterator, Tuple
import time
import logging

logger = logging.getLogger(__name__)
from config import REFERENCE_RESOLUTION

"""
Load ERA5 variables within bounding box and time range
"""


def open_era5_zarr(url, retries=3, delay=3):
    for attempt in range(retries):
        try:
            return xr.open_dataset(url, chunks={}, engine="zarr")
        except Exception as e:
            if attempt == retries - 1:
                raise
            logger.error(f"ERA5 open failed (attempt {attempt+1}), retrying...")
            time.sleep(delay)


def load_era5_variable(url, var, bounds, start_year, end_year):
    """
    Slices and prepares raw ERA5 variables.
    Handles coordinate wrapping (0-360 to -180-180) and descending latitude.
    """
    min_lon, min_lat, max_lon, max_lat = bounds

    ds = open_era5_zarr(url)

    # Ensure latitude is North-to-South
    sel_lat_upper, sel_lat_lower = max(min_lat, max_lat), min(min_lat, max_lat)
    # Select latitude & longitude excluding the upper edge to avoid overlap when stitching tiles
    sel_lats = np.arange(sel_lat_upper, sel_lat_lower, -REFERENCE_RESOLUTION)

    # Wrap input bounds to 0-360 for ERA5 compatibility
    sel_min_lon, sel_max_lon = min_lon % 360, max_lon % 360

    # Spatial selection
    if sel_min_lon > sel_max_lon:  # Crosses Prime Meridian
        sel_lons = (
            np.arange(sel_min_lon, 360.0, REFERENCE_RESOLUTION).tolist()
            + np.arange(0.0, sel_max_lon, REFERENCE_RESOLUTION).tolist()
        )
    else:
        sel_lons = np.arange(sel_min_lon, sel_max_lon, REFERENCE_RESOLUTION)
    selector = {
        "latitude": sel_lats,
        "longitude": sel_lons,
    }

    # Temporal selection
    ds = ds.sel(valid_time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    # Load variable, calculate wind speed if needed
    try:
        if var == "ws100":
            u, v = ds["u100"].sel(**selector), ds["v100"].sel(**selector)
            da = xr.apply_ufunc(
                np.hypot, u, v, dask="parallelized", output_dtypes=[u.dtype]
            )
            da.name = "ws100"
            da.attrs["units"] = "m/s"
            da.attrs["long_name"] = "Wind Speed at 100m"
        else:
            da = ds[var].sel(**selector)

        if da.size == 0:
            return None

        # Normalize back to -180 to 180 and sort for future processing
        da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby(
            "longitude"
        )
        # convert to daily means to reduce data volume
        return da.resample(valid_time="1D").mean()
    except Exception as e:
        print(f"Error selecting {var}: {e}")
        return None


"""
clip, resample
"""


def clip_and_resample(input_raster, template, resampling=Resampling.average):
    """Automatically maps x/y to lon/lat and resamples."""
    # Temporarily rename template to x/y so rioxarray understands it
    target = template.rename({"longitude": "x", "latitude": "y"})
    minx, miny, maxx, maxy = target.rio.bounds()
    try:
        # add some buffer when clipping to ensure edges are captured
        clipped = input_raster.rio.clip_box(
            minx=minx - REFERENCE_RESOLUTION,
            miny=miny - REFERENCE_RESOLUTION,
            maxx=maxx + REFERENCE_RESOLUTION,
            maxy=maxy + REFERENCE_RESOLUTION,
        )
        resampled = clipped.rio.reproject_match(
            target,
            resampling=resampling,
            nodata=np.nan,
        )
        # Rename back to match the template's original names
        return resampled.rename({"x": "longitude", "y": "latitude"})
    except NoDataInBounds:
        return xr.full_like(target, np.nan)


"""
Create grids of reference rasters
"""

Tile = Tuple[float, float, float, float]


def generate_tiles(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    tile_size: float,
) -> Iterator[Tile]:
    """
    Generate non-overlapping tiles covering a rectangular domain.

    Parameters
    ----------
    minx, miny, maxx, maxy : float
        Bounding box of the domain (lon/lat).
    tile_size : float
        Tile size in degrees.

    Yields
    ------
    tile : (minx, miny, maxx, maxy)
        Bounds of one tile.
    """
    lon = minx
    while lon < maxx:
        lat = miny
        while lat < maxy:
            yield lon, lat, lon + tile_size, lat + tile_size
            lat += tile_size
        lon += tile_size


def create_tile_template(bounds, resolution, crs="EPSG:4326"):
    """
    Creates the reference coordinate system for a tile.
    Matches ERA5 standard: Latitude: North-South, BUT Longitude: -180-180 (same as atlases) instead of 0-360 (ERA5).
    """
    minx, miny, maxx, maxy = bounds

    # Exclusive upper bounds to prevent stitching overlaps: linspace is more reliable, np.arange can be weird with float
    # Longitude: Ascending
    # Latitude: Descending, i.e. North to South to match with ERA5 and load_physical_variable logic
    lons = np.arange(minx, maxx, resolution)
    lats = np.arange(maxy, miny, -resolution)
    # num_lons = int(round(abs(maxx - minx) / resolution))
    # num_lats = int(round(abs(maxy - miny) / resolution))

    # lons = np.linspace(minx, maxx, num_lons, endpoint=False)
    # lats = np.linspace(maxy, miny, num_lats, endpoint=False)

    template = xr.Dataset(
        coords={
            "latitude": (["latitude"], lats.astype(np.float32)),
            "longitude": (["longitude"], lons.astype(np.float32)),
        }
    )
    # rioxarray expects x/y for spatial operations, we'll map them during resampling
    return template.rio.write_crs(crs)


# def create_reference_raster(bounds, resolution, crs="EPSG:4326"):
#     """
#     Create a synthetic reference raster with given bounds and resolution.
#     return:
#     xr.DataArray
#         Empty reference raster with correct grid and CRS
#     """
#     minx, miny, maxx, maxy = bounds

#     # excluding upper edge to avoid overlap when stitching tiles
#     lons = np.arange(minx, maxx, resolution)
#     lats = np.arange(miny, maxy, resolution)

#     da = xr.DataArray(
#         np.zeros((len(lats), len(lons))),
#         coords={"y": lats, "x": lons},
#         dims=("y", "x"),
#         name="reference",
#     )

#     return da.rio.write_crs(crs)


# """
# pixel area calculation functions for normalising human settlement data
# """


# def _area_of_pixel(pixel_size, center_lat):
#     """Calculate km^2 area of a wgs84 square pixel.

#     Adapted from: https://gis.stackexchange.com/a/127327/2397

#     Parameters:
#         pixel_size (float): length of side of pixel in degrees.
#         center_lat (float): latitude of the center of the pixel. Note this
#             value +/- half the `pixel-size` must not exceed 90/-90 degrees
#             latitude or an invalid area will be calculated.

#     Returns:
#         Area of square pixel of side length `pixel_size` centered at
#         `center_lat` in km^2.

#     """
#     a = 6378137  # meters
#     b = 6356752.3142  # meters
#     e = math.sqrt(1 - (b / a) ** 2)
#     area_list = []
#     for f in [center_lat + pixel_size / 2, center_lat - pixel_size / 2]:
#         zm = 1 - e * math.sin(math.radians(f))
#         zp = 1 + e * math.sin(math.radians(f))
#         area_list.append(
#             math.pi
#             * b**2
#             * (math.log(zp / zm) / (2 * e) + math.sin(math.radians(f)) / (zp * zm))
#         )
#     return pixel_size / 360.0 * (area_list[0] - area_list[1]) / 1e6


# def determine_pixel_areas(raster_input):
#     """Determine area of each pixel.

#     Returns a raster in which the value corresponds to the area in [m2] of the pixel.
#     based on T.Troendle determine_pixel_areas (utils.py and technically_eligible_area.py)
#     This assumes the data comprises square pixel in WGS84.

#     Parameters:
#         crs: the coordinate reference system of the data (must be WGS84)
#     """
#     # the following is based on https://gis.stackexchange.com/a/288034/77760
#     # and assumes the data to be in EPSG:4326
#     assert (
#         raster_input.rio.crs.to_epsg() == 4326
#     ), "raster_input does not have the projection EPSG:4326"
#     resolution = raster_input.rio.resolution()[0]  # resolution in degrees
#     varea_of_pixel = np.vectorize(lambda lat: _area_of_pixel(resolution, lat))
#     pixel_area = varea_of_pixel(raster_input.y) * 1000**2  # convert to m^2

#     pixel_area_da = xr.DataArray(
#         pixel_area,
#         coords={"y": raster_input.y},
#         dims="y",
#     )
#     return pixel_area_da
