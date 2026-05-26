import numpy as np
import xarray as xr
from typing import Tuple, List
from src.geo_processing import load_era5_variable, load_era5_daily_land
from src.config import ERA5_ZARR_URL, ERA5_LAND_ZARR_URL

# ERA5 single-levels: aggregation rules
VAR_CONFIG = {
    "ws100": {"daily_sum": False, "long_name": "Wind Speed 100m"},
    "ssrd": {"daily_sum": True, "long_name": "Surface Solar Radiation"},
    "tp": {"daily_sum": True, "long_name": "Total Precipitation"},
    "t2m": {"daily_sum": False, "long_name": "2m Temperature"},
}


def get_variable_climatology(
    tile: Tuple[float, ...], var: str, start_year: int, end_year: int
):
    """
    Computes daily climatology (365 days) for a specific ERA5 variable.
    """
    config = VAR_CONFIG.get(var, {"daily_sum": False, "long_name": var})

    da = load_era5_variable(
        ERA5_ZARR_URL, var, tile, start_year, end_year, daily_sum=config["daily_sum"]
    )
    if da is None:
        return None

    # Standardize time: remove leap days and group by day of year
    da = da.convert_calendar("noleap", dim="valid_time")
    clim = da.groupby("valid_time.dayofyear").mean("valid_time")

    clim.name = f"{var}_climatology"
    clim.attrs["long_name"] = f"Daily Mean Climatology of {config['long_name']}"
    return clim.compute()


def get_complementarity_index(tile: Tuple[float, ...], start_year: int, end_year: int):
    """
    Calculates Pearson correlation between wind (ws100) and solar (ssrd).
    """
    ds_wind = load_era5_variable(
        ERA5_ZARR_URL, "ws100", tile, start_year, end_year, daily_sum=False
    )
    ds_solar = load_era5_variable(
        ERA5_ZARR_URL, "ssrd", tile, start_year, end_year, daily_sum=True
    )

    if ds_wind is None or ds_solar is None:
        return None

    # Align calendars
    ds_wind = ds_wind.convert_calendar("noleap", dim="valid_time")
    ds_solar = ds_solar.convert_calendar("noleap", dim="valid_time")

    correlation = xr.corr(ds_wind, ds_solar, dim="valid_time")
    correlation.name = "complementarity"
    correlation.attrs["description"] = "Pearson correlation (ws100 vs ssrd)"

    return correlation.compute()
