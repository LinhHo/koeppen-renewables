"""
Demand-related computations for the Koeppen renewables pipeline.

This module provides two independent demand components:

1. Climate-driven demand
   - Heating days
   - Cooling days
   derived from ERA5 temperature fields

2. Settlement-driven demand potential
   - Inverse-distance weighted population / built-up intensity
   - Computed using a buffered tile, then cropped back to the original tile

All functions operate on *tiles* and return xarray objects.
"""

from typing import Tuple

import numpy as np
import xarray as xr
import rioxarray as rxr
from scipy.signal import convolve2d

from src.geo_processing import (
    load_era5_variable,
    clip_and_resample,
    create_tile_template,
)

from config import (
    ERA5_ZARR_URL,
    REFERENCE_RESOLUTION,
    DEMAND_WEIGHTING_BUFFER,
    START_YEAR,
    END_YEAR,
)


Tile = Tuple[float, float, float, float]

# ============================================================
# 1. Climate-driven demand (ERA5 temperature)
# ============================================================

# Estimate from Staffel et al. (2024) Fig 1c (no data provided)
T_heat = 14  # °C
T_cool = 22  # °C
alpha_heat = 0.6  # kWh / day / °C
alpha_cool = 0.7  # kWh / day / °C


def compute_temperature_demand_indicator(
    bounds: Tile,
):
    """
    Compute climate-driven electricity demand intensity from temperature.

    Parameters
    ----------
    tas : xr.DataArray
        ERA5 2m air temperature [K], climatological time series.
    T_heat, T_cool : float
        Heating and cooling threshold temperatures [°C].
    alpha_heat, alpha_cool : float
        Linear demand coefficients [relative units per °C per timestep].
    time_dim : str
        Time dimension name.

    Returns
    -------
    xr.DataArray
        Dimensionless climate demand indicator.
    """

    # Load ERA5 temperature
    da = load_era5_variable(
        ERA5_ZARR_URL,
        "t2m",
        bounds=bounds,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    # Chunk spatially; keep full time for climatology
    da = da.chunk({"valid_time": -1, "latitude": 10, "longitude": 10})

    # Convert to Celsius, compute based on climatology
    T = da - 273.15
    T_clim = T.groupby("valid_time.dayofyear").mean("valid_time")

    # Heating and cooling components
    heating = alpha_heat * xr.where(T_clim < T_heat, T_heat - T_clim, 0.0)
    cooling = alpha_cool * xr.where(T_clim > T_cool, T_clim - T_cool, 0.0)

    # Aggregate over time
    demand_temperature_induced = (heating + cooling).sum(dim="dayofyear")

    return demand_temperature_induced.rename("demand_temperature_induced")

    # # Quantile normalisation to get a dimensionless indicator of demand induced by temperature
    # scale = demand_temperature_induced.quantile(0.95)
    # return xr.where(scale > 0, demand_temperature_induced / scale, 0).clip(0, 1)


# ============================================================
# 2. Settlement-driven demand potential (buffered convolution)
# ============================================================


def compute_demand_settlement_proximity(
    tile: Tile,
    paths: dict,
    radius: float = DEMAND_WEIGHTING_BUFFER,  # degrees
) -> xr.DataArray:
    """
    Resample settlement data onto a buffered reference grid.
    Then compute inverse-distance weighted settlement demand potential.

    The buffer ensures that inverse-distance weighting near tile
    boundaries is physically correct.

    NOTE: The unit is m2 from settlement data, not density or population.

    Parameters
    ----------
    tile : (minx, miny, maxx, maxy)
        True tile bounds.
    paths : dict
        Paths configuration (must include 'ghsl').

    Returns
    -------
    xr.DataArray
        Settlement share per pixel on buffered grid.
    """

    minx, miny, maxx, maxy = tile

    buffer_bounds = (
        minx - radius,
        miny - radius,
        maxx + radius,
        maxy + radius,
    )

    # Create buffered reference grid
    ref = create_tile_template(buffer_bounds, REFERENCE_RESOLUTION)

    # Load and resample settlement raster
    with rxr.open_rasterio(paths["ghsl"], chunks=True).squeeze().astype(
        "float32"
    ) as built:
        settlement = clip_and_resample(built, ref)

    # Kernel radius in pixels of buffer zone
    radius = int(radius / REFERENCE_RESOLUTION)

    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    dist = np.sqrt(xx**2 + yy**2)

    # Mask outside circle
    mask = dist <= radius
    dist[radius, radius] = 1.0  # Avoid singularity at center

    # Circular inverse-distance weights
    weights = np.zeros_like(dist, dtype="float32")
    weights[mask] = 1.0 / dist[mask]
    weights /= weights.sum()  # comparable scale across radius sizes

    # Convolution (Dask-compatible)
    weighted_buffered = xr.apply_ufunc(
        lambda x: convolve2d(x, weights, mode="same", boundary="symm"),
        settlement,
        dask="parallelized",
        output_dtypes=[settlement.dtype],
    )

    return weighted_buffered


# Temporarily put it here. Quantile normalisation seems to make clear jumps in demand potential on the map.
quantile_normalised = False


def run_demand_potential_for_tile(
    tile: Tile,
    paths: dict,
) -> xr.Dataset:
    """
    Compute both climate-driven and settlement-driven demand indicators for one tile.

    Results are saved as NetCDF files in output_dir.
    """

    minx, miny, maxx, maxy = tile
    bounds = (minx, miny, maxx, maxy)

    ds = xr.Dataset()

    print("  -> Computing climate-driven demand indicator...")
    ds["demand_temperature_induced"] = compute_temperature_demand_indicator(bounds)

    print("  -> Computing settlement-driven demand potential...")
    ds["demand_settlement_proximity"] = (
        compute_demand_settlement_proximity(tile, paths)
        .sel(longitude=ds.longitude, latitude=ds.latitude)
        .rename("demand_settlement_proximity")
    )

    # Compute demand potential as product of both indicators
    # Note: log1p used to compress large settlement values
    ##  NOTE LOG1P returns 0~10 not normalised values 0-1 <<<<<
    # (1+temperature_indicator) to ensure non-zero even if no temperature-driven demand
    # temperature as an extra stressor for demand
    if quantile_normalised:
        temperature_q95 = ds["demand_temperature_induced"].quantile(0.95)
        demand_temperature_induced = xr.where(
            temperature_q95 > 0, ds["demand_temperature_induced"] / temperature_q95, 0
        ).clip(0, 1)
    else:
        demand_temperature_induced = ds["demand_temperature_induced"] / np.max(
            ds["demand_temperature_induced"]
        )

    ds["demand_potential"] = (np.log1p(ds["demand_settlement_proximity"])) * (
        1 + demand_temperature_induced
    )

    return ds
