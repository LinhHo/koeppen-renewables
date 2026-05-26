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
            # Chunk by spatial dimensions but keep time in single chunk for variability calculations
            return xr.open_dataset(
                url,
                chunks={},
                engine="zarr",
            )
        except Exception as e:
            if attempt == retries - 1:
                raise
            logger.error(f"ERA5 open failed (attempt {attempt+1}), retrying...")
            time.sleep(delay)


def convert_coordinate_ERA5(bounds):
    min_lon, min_lat, max_lon, max_lat = bounds

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
    selector = {"latitude": sel_lats, "longitude": sel_lons, "method": "nearest"}
    return selector


def load_era5_variable(url, var, bounds, start_year, end_year, daily_sum=False):
    """
    Slices and prepares raw ERA5 variables.
    Handles coordinate wrapping (0-360 to -180-180) and descending latitude.
    """

    ds = open_era5_zarr(url)
    selector = convert_coordinate_ERA5(bounds)

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
        if daily_sum or var in ["ssrd", "tp"]:
            return da.resample(valid_time="1D").sum()
        else:
            return da.resample(valid_time="1D").mean()
    except Exception as e:
        print(f"Error selecting {var}: {e}")
        return None


def load_era5_daily_land(url, var, bounds, start_year, end_year):
    """
    Load an ERA5-Land variable within a bounding box and time range.

    ERA5-Land data is already at daily resolution — no aggregation is applied.
    Coordinates are normalised from 0–360 to −180–180 and sorted.

    Parameters
    ----------
    url : str
        ERA5-Land zarr URL (ERA5_LAND_ZARR_URL from config).
    var : str
        Variable name in the zarr store (e.g. 't2m', 'asn').
    bounds : tuple
        (minx, miny, maxx, maxy) in degrees.
    start_year, end_year : int
        Inclusive year range.

    Returns
    -------
    xr.DataArray
        Daily values, dims (valid_time, latitude, longitude).

    Raises
    ------
    KeyError
        If *var* is not present in the ERA5-Land store.
    ValueError
        If the selection yields no data.
    """
    ds = open_era5_zarr(url)
    if var not in ds:
        raise KeyError(f"Variable {var!r} not found in ERA5-Land store. Available: {list(ds.data_vars)}")

    selector = convert_coordinate_ERA5(bounds)
    ds = ds.sel(valid_time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
    da = ds[var].sel(**selector)

    if da.size == 0:
        raise ValueError(f"ERA5-Land selection for {var!r} in {bounds} {start_year}-{end_year} returned empty array.")

    return da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")


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
            minx=minx,  # - REFERENCE_RESOLUTION,
            miny=miny,  # - REFERENCE_RESOLUTION,
            maxx=maxx,  # + REFERENCE_RESOLUTION,
            maxy=maxy,  # + REFERENCE_RESOLUTION,
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


import math
import numpy as np
import xarray as xr


def _area_of_pixel(pixel_size, center_lat):
    """Calculate m^2 area of a wgs84 square pixel (WGS84 Ellipsoid)."""
    a = 6378137  # meters (semi-major axis)
    b = 6356752.3142  # meters (semi-minor axis)
    e = math.sqrt(1 - (b / a) ** 2)
    area_list = []

    # Calculate the area from the equator to the latitude of the pixel edges
    for f in [center_lat + pixel_size / 2, center_lat - pixel_size / 2]:
        lat_rad = math.radians(f)
        zm = 1 - e * math.sin(lat_rad)
        zp = 1 + e * math.sin(lat_rad)
        # Formula for the area of a spherical cap on an ellipsoid
        area_list.append(
            math.pi
            * b**2
            * (math.log(zp / zm) / (2 * e) + math.sin(lat_rad) / (zp * zm))
        )

    # Area of the zone = (fraction of the circle) * (difference in surface area)
    # Result is in m^2 (removed the /1e6 from your original to keep it in m2)
    return abs(pixel_size / 360.0 * (area_list[0] - area_list[1]))


def determine_pixel_areas(raster_input):
    """
    Returns a DataArray of pixel areas in [m2] that matches the latitude
    of raster_input for easy division.
    """
    # 1. Verify CRS
    assert raster_input.rio.crs.to_epsg() == 4326, "CRS must be EPSG:4326"

    # 2. Get resolution (assumes square pixels in degrees)
    # Using abs() because resolution is often negative in the y-axis (lat)
    resolution = abs(raster_input.rio.resolution()[0])

    # 3. Vectorize calculation over latitudes only
    varea_of_pixel = np.vectorize(lambda lat: _area_of_pixel(resolution, lat))
    pixel_area_1d = varea_of_pixel(raster_input.latitude.values)

    # 4. Create DataArray aligned with the latitude dimension
    # This allows: raster_input / pixel_area_da to work instantly
    pixel_area_da = xr.DataArray(
        pixel_area_1d, coords={"latitude": raster_input.latitude}, dims="latitude"
    )

    return pixel_area_da


def calculate_grid_cell_areas(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Calculate accurate grid cell areas using WGS84 ellipsoid.

    Returns 2D array in km²
    """
    dlat = np.abs(np.diff(lat).mean()) if len(lat) > 1 else 0.25
    pixel_size = dlat

    areas_1d = np.array([_area_of_pixel(pixel_size, lat_val) for lat_val in lat])
    areas_2d = np.repeat(areas_1d[:, np.newaxis], len(lon), axis=1)
    areas_km2 = areas_2d / 1e6  # Convert m² to km²

    return areas_km2


# def get_tile_cell_areas(tile, res=0.25):
#     """
#     Returns an xarray.DataArray containing the area in m2
#     for each grid cell within the specified tile.
#     """
#     minx, miny, maxx, maxy = tile

#     # Create the coordinate centers (aligning with ERA5 0.25 deg grid)
#     # Note: ERA5 typically uses center-points like 19.875, 19.625...
#     lats = np.arange(maxy - res/2, miny, -res)
#     lons = np.arange(minx + res/2, maxx, res)

#     R = 6371000.0  # Earth's radius in meters
#     d_lat = np.radians(res)
#     d_lon = np.radians(res)

#     # Calculate area for each latitude
#     # Area = R^2 * cos(lat) * d_lat * d_lon
#     # Use np.cos(np.radians(lats)) to get a 1D array of multipliers
#     areas_1d = (R**2) * np.cos(np.radians(lats)) * d_lat * d_lon

#     # Broadcast to 2D (lat, lon)
#     area_2d = np.tile(areas_1d[:, np.newaxis], (1, len(lons)))

#     return xr.DataArray(
#         data=area_2d,
#         coords={"latitude": lats, "longitude": lons},
#         dims=("latitude", "longitude"),
#         name="cell_area_m2"
#     )
