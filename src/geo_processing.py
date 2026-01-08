import math
import numpy as np
import xarray as xr
from rasterio.enums import Resampling
from rioxarray.exceptions import NoDataInBounds
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
    sel_lat = slice(sel_lat_upper - REFERENCE_RESOLUTION, sel_lat_lower)

    # Wrap input bounds to 0-360 for ERA5 compatibility
    sel_min_lon, sel_max_lon = min_lon % 360, max_lon % 360

    # Spatial selection
    if sel_min_lon > sel_max_lon:  # Crosses Prime Meridian
        lon_slice = xr.concat(
            [
                ds.longitude.sel(
                    longitude=slice(sel_min_lon, 360 - REFERENCE_RESOLUTION)
                ),
                ds.longitude.sel(
                    longitude=slice(0, sel_max_lon - REFERENCE_RESOLUTION)
                ),
            ],
            dim="longitude",
        )
        selector = {"longitude": lon_slice, "latitude": sel_lat}
    else:
        selector = {
            "longitude": slice(sel_min_lon, sel_max_lon - REFERENCE_RESOLUTION),
            "latitude": sel_lat,
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


def clip_and_resample(
    input_raster, reference_raster, bounds, resampling=Resampling.average
):
    """Clip raster to bounding box and resample to ERA5 grid."""
    minx, miny, maxx, maxy = bounds
    try:
        clipped = input_raster.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        return clipped.rio.reproject_match(
            reference_raster,
            resampling=resampling,
            nodata=np.nan,
            masked=True,
        )
    except NoDataInBounds:
        return xr.full_like(reference_raster, np.nan)


"""
Create grids of reference rasters
"""


def create_reference_raster(bounds, resolution, crs="EPSG:4326"):
    """
    Create a synthetic reference raster with given bounds and resolution.
    return:
    xr.DataArray
        Empty reference raster with correct grid and CRS
    """
    minx, miny, maxx, maxy = bounds

    # excluding upper edge to avoid overlap when stitching tiles
    lons = np.arange(minx, maxx, resolution)
    lats = np.arange(miny, maxy, resolution)

    da = xr.DataArray(
        np.zeros((len(lats), len(lons))),
        coords={"y": lats, "x": lons},
        dims=("y", "x"),
        name="reference",
    )

    return da.rio.write_crs(crs)


"""
pixel area calculation functions for normalising human settlement data
"""


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
