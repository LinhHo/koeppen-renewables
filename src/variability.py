import xarray as xr
import numpy as np
import time
from typing import Tuple
from dask.distributed import Client

from src.geo_processing import load_era5_variable
from config import START_YEAR, END_YEAR, ERA5_ZARR_URL


Tile = Tuple[float, float, float, float]


def calculate_maximum_deficit_dask(imbalance_da, time_dim="time"):
    """
    Core math for Energy Deficit (Maximum Drawdown).

    Logic:
    1. Integrates (cumsum) the net imbalance (Generation - Target).
    2. Duplicates the series to 2nd period to handle the 'worst-case' cycle.
    3. Calculates the 'running minimum' to find the deepest point reached.
    4. Drawdown = Current cumulative level - Running minimum.
    5. Normalizes by total duration T to make it dimensionless (0-1).
    """
    # Ensure time is in a single chunk for this specific pixel-block
    da = imbalance_da.chunk({time_dim: -1})

    # 1. Integrate net imbalance
    cum_sum = da.cumsum(dim=time_dim)
    T = len(da[time_dim])

    # 2. Cyclic extension (concat) allows for search over the annual boundary
    cum_extended = xr.concat(
        [cum_sum, cum_sum + cum_sum.isel({time_dim: -1})], dim=time_dim
    )
    cum_extended = cum_extended.chunk({time_dim: -1})

    # 3. Find the minimum level seen 'so far' in a rolling window of size T
    run_min = cum_extended.rolling({time_dim: T}, min_periods=1).min()

    # 4. Drawdown calculation (The deficit that must be bridged by storage)
    deficit = cum_extended - run_min

    # 5. Extract the first period results and normalize by the period length
    return deficit.isel({time_dim: slice(0, T)}).max(dim=time_dim) / T


def compute_variability_hourly(url, variable, bounds, start_year, end_year):
    """
    Main pipeline: Loads data, calculates Seasonal (climatological)
    and Weather (interannual) variability.
    """
    client = Client()
    da = load_era5_variable(url, variable, bounds, start_year, end_year)
    da = da.chunk(
        {"valid_time": -1, "latitude": 10, "longitude": 10}
    )  # Ensure time is in single chunk for deficit calculation
    # Remove Feb 29 for climatology calculations
    da = da.sel(
        valid_time=~((da.valid_time.dt.month == 2) & (da.valid_time.dt.day == 29))
    )

    # --- 1. SEASONAL VARIABILITY ---
    # Captures the deficit caused by the regular annual/diurnal cycle
    clim = da.groupby("valid_time.dayofyear").mean("valid_time")
    clim_norm = clim / clim.mean(dim="dayofyear")

    seasonal_imb = (clim_norm - 1).rename({"dayofyear": "time"})
    seasonal_var = calculate_maximum_deficit_dask(seasonal_imb, time_dim="time")

    # --- 2. WEATHER (INTERANNUAL) VARIABILITY ---
    # Captures the deficit caused by year-to-year weather fluctuations

    # Vectorized normalization: divides each year by its own mean to isolate variance
    yearly_means = da.groupby("valid_time.year").mean("valid_time")
    year_norm = da / yearly_means

    # Align the climatology reference with the actual hours of the full time series
    # (Maps the 365-day cycle to the multi-decade period)
    clim_expect = clim_norm.sel(dayofyear=da.valid_time.dt.dayofyear).drop_vars(
        "dayofyear"
    )
    clim_expect.coords["valid_time"] = da.valid_time

    # Weather Imbalance = (Actual Hourly Energy) - (Expected Seasonal Average)
    weather_imbalance = (year_norm - clim_expect).rename({"valid_time": "time"})

    # Calculate deficit for every year independently, then find the worst one (max)
    weather_var = (
        weather_imbalance.groupby("time.year")
        .map(calculate_maximum_deficit_dask)
        .max("year")
    )

    return xr.Dataset(
        {
            "seasonal_variability": seasonal_var,
            "weather_variability": weather_var,
        }
    )


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
