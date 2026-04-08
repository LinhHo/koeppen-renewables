"""
Central configuration for the Koeppen renewables workflow.

This file defines:
- temporal domain (start/end year)
- default global spatial domain
- tiling parameters
- ERA5 Zarr access (via environment variables)

This module should NOT contain any heavy imports.
"""

import os
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()
era5_token = os.getenv("ERA5_TOKEN")

# -----------------------
# Temporal configuration
# -----------------------

START_YEAR = 2023
END_YEAR = 2024

# -----------------------
# Spatial configuration
# -----------------------

GLOBAL_DOMAIN = {
    "minx": -180.0,
    "miny": -60.0,
    "maxx": 180.0,
    "maxy": 80.0,
}

TILE_SIZE = 20.0  # degrees
REFERENCE_RESOLUTION = 0.25  # ERA5 grid (degrees)

DEMAND_WEIGHTING_BUFFER = 2  # degrees

# -----------------------
# ERA5 access
# -----------------------

ERA5_ZARR_URL = (
    f"https://edh:{era5_token}"
    "@data.earthdatahub.destine.eu/era5/"
    "reanalysis-era5-single-levels-v0.zarr"
)

PATHS = {
    "solar_atlas": "resources/user/PVOUT.tif",
    "wind_atlas": "resources/user/cf_iec2_cog_100m.tif",
    "ghsl": "resources/user/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0_R8_C29.tif",
}

# # -----------------------
# # Koeppen renewable zones
# ABUNDANCE_QUANTILES = {
#     "low": 0.33,
#     "high": 0.66,
# }

# VARIABILITY_QUANTILES = {
#     "low": 0.5,  # below median = reliable
# }
