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
    "wind_atlas": "resources/user/cf_iec1_cog_100m.tif",
    "ghsl": "resources/user/ghsl_built_s.tif",
}
