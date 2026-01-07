import xarray as xr
import numpy as np
import time
from dask.distributed import Client
from config import REFERENCE_RESOLUTION


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


def load_physical_variable(ds, var, bounds, start_year, end_year):
    """
    Slices and prepares raw ERA5 variables.
    Handles coordinate wrapping (0-360 to -180-180) and descending latitude.
    """
    min_lon, min_lat, max_lon, max_lat = bounds

    # Wrap input bounds to 0-360 for ERA5 compatibility
    sel_min_lon, sel_max_lon = min_lon % 360, max_lon % 360
    # Ensure latitude is North-to-South
    sel_lat_up, sel_lat_lo = max(min_lat, max_lat), min(min_lat, max_lat)

    # Spatial selection
    if sel_min_lon > sel_max_lon:  # Crosses Prime Meridian
        lon_slice = xr.concat(
            [
                ds.longitude.sel(longitude=slice(sel_min_lon, 360)),
                ds.longitude.sel(longitude=slice(0, sel_max_lon)),
            ],
            dim="longitude",
        )
        selector = {"longitude": lon_slice, "latitude": slice(sel_lat_up, sel_lat_lo)}
    else:
        selector = {
            "longitude": slice(sel_min_lon, sel_max_lon),
            "latitude": slice(sel_lat_up, sel_lat_lo),
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


def compute_variability_hourly(
    url, variable, bounds, start_year, end_year, start_client=False
):
    """
    Main pipeline: Loads data, calculates Seasonal (climatological)
    and Weather (interannual) variability.
    """
    if start_client:
        Client()

    ds = xr.open_dataset(url, chunks={}, engine="zarr")
    da = load_physical_variable(ds, variable, bounds, start_year, end_year)

    # Optimization: Chunk by spatial dimensions so each 'task' is a pixel-column
    da = da.chunk({"valid_time": -1, "latitude": 10, "longitude": 10})

    # --- 1. SEASONAL VARIABILITY ---
    # Captures the deficit caused by the regular annual/diurnal cycle
    clim = da.groupby("valid_time.dayofyear").mean("valid_time")
    clim_norm = (
        clim / clim.mean()
    ).compute()  # Small enough to keep in memory as reference

    seasonal_imb = (clim_norm - 1).rename({"dayofyear": "time"})
    seasonal_var = calculate_maximum_deficit_dask(seasonal_imb, time_dim="time")

    # --- 2. WEATHER (INTERANNUAL) VARIABILITY ---
    # Captures the deficit caused by year-to-year weather fluctuations

    # Vectorized normalization: divides each year by its own mean to isolate variance
    yearly_means = da.groupby("valid_time.year").mean("valid_time")
    year_norm = da.groupby("valid_time.year") / yearly_means

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
    ).compute()
