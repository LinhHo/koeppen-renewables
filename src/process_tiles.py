"""
Tile-based orchestration utilities.

This module:
- generates 20×20° tiles over a spatial domain
- runs ERA5 variability calculations for each tile

It does NOT:
- define global configuration (see config.py)
- perform resampling (see resampling.py)
"""

from pathlib import Path
from typing import Iterator, Tuple

from variability import compute_variability_hourly
from config import START_YEAR, END_YEAR, ERA5_ZARR_URL


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


def run_variability_for_tile(
    tile: Tile,
    variable: str,
):
    """
    Compute ERA5 seasonal and weather variability for one tile.

    For wind:
    - u100 and v100 are combined into ws100 internally
    - variability is computed on wind speed magnitude

    For solar:
    - ssrd is used directly

    Results are saved as NetCDF files in output_dir.
    """
    minx, miny, maxx, maxy = tile
    bounds = (minx, miny, maxx, maxy)

    ds = compute_variability_hourly(
        url=ERA5_ZARR_URL,
        variable=variable,
        bounds=bounds,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )
    return ds
