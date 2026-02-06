from src.geo_processing import generate_tiles, load_era5_variable
import os
import xarray as xr

from config import ERA5_ZARR_URL
from dotenv import load_dotenv
from dask.distributed import Client

# Load environment variables from .env
load_dotenv()
era5_token = os.getenv("ERA5_TOKEN")

# bounds = [ds_zones.longitude.min(), ds_zones.longitude.max(), ds_zones.latitude.min(), ds_zones.latitude.max()]
domain = (-180, -60, 180, 80)
variable = "tp"

tiles = generate_tiles(domain[0], domain[1], domain[2], domain[3], 20)


# ERA5_ZARR_URL_month = (
#     f"https://edh:{era5_token}"
#     "@data.earthdatahub.destine.eu/era5/"
#     "reanalysis-era5-single-levels-monthly-means-v0.zarr"
# )

for tile in tiles:
    # client.restart()
    tile_str = "_".join(map(str, tile))

    # da = load_era5_variable("https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-monthly-means-v0.zarr", variable, bounds, 1995, 2025)
    # da = load_era5_variable(
    #     ERA5_ZARR_URL_month, variable, tile, 1995, 2026, daily_sum=True
    # )
    da = load_era5_variable(ERA5_ZARR_URL, variable, tile, 1995, 2026, daily_sum=True)
    print(da)
    da = da.chunk(
        {"valid_time": -1, "latitude": 10, "longitude": 10}
    )  # Ensure time is in single chunk for deficit calculation

    # clim = da.groupby("valid_time.dayofyear").mean("valid_time")
    # yearmean = da.groupby("valid_time.year").sum("valid_time").mean("year")
    yearmean = da.resample(valid_time="1Y").sum().mean("valid_time")
    yearmean.to_netcdf(
        f"resources/automatic/tp/tmp_era5_precip_year_average_1995_2025_{tile_str}.nc",
        mode="w",
    )

ds = xr.open_mfdataset(
    "resources/automatic/tp/tmp_era5_precip_year_average_1995_2025_*.nc",
    combine="by_coords",
)
ds.to_netcdf("resources/automatic/tp/era5_precip_year_average_1995_2025.nc", mode="w")
