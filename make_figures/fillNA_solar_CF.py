"""
fillNA_solar_CF.py
==================
Fill NaN solar_CF values in the processed dataset using an OLS regression
trained on ERA5 climatological predictors across all valid global land pixels.

Why global training instead of per-tile:
  The Global Solar Atlas only covers land up to ~60°N (70°N in Scandinavia).
  Training on all valid pixels globally produces more robust coefficients
  than fitting within individual 20°×20° tiles.

Predictors:
  - Annual-mean ssrd  (dominant driver: direct solar irradiance)
  - Annual-mean t2m   (proxy for cloud cover / sky clarity)

Land mask: mask_land(wind_CF) from plot_utils (regionmask natural_earth land_110).

Run:
  pixi run python figures/fillNA_solar_CF.py
Output:
  results/automatic/processed_solar_CF_filled.nc
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_utils import mask_land

PROCESSED_PATTERN = str(RESULTS_DIR / "automatic/abundance/abundance_*.nc")
SSRD_PATTERN = str(RESULTS_DIR / "automatic/climatology/ssrd/*.nc")
T2M_PATTERN = str(RESULTS_DIR / "automatic/climatology/t2m/*.nc")
OUTPUT_PATH = RESULTS_DIR / "post_processed_data/"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_PATH / "processed_solar_CF_filled.nc"

# ── Load data ─────────────────────────────────────────────────────────────────

print(f"Loading processed dataset ...")
ds = xr.open_mfdataset(PROCESSED_PATTERN, combine="by_coords", engine="netcdf4")

print("Loading ssrd climatology ...")
ssrd_clim = xr.open_mfdataset(SSRD_PATTERN, combine="by_coords", engine="netcdf4")[
    "ssrd_climatology"
]

print("Loading t2m climatology ...")
t2m_clim = xr.open_mfdataset(T2M_PATTERN, combine="by_coords", engine="netcdf4")[
    "t2m_climatology"
]


# ── Prepare predictors ─────────────────────────────────────────────────────────

solar = ds["solar_CF"].compute()

is_land = mask_land(ds["wind_CF"].compute())

# Annual means from 365-day climatology
ssrd_mean = ssrd_clim.mean("dayofyear").interp(
    latitude=solar.latitude, longitude=solar.longitude
)
t2m_mean = t2m_clim.mean("dayofyear").interp(
    latitude=solar.latitude, longitude=solar.longitude
)

# Flatten to 1-D
s = solar.values.ravel()
p1 = ssrd_mean.values.ravel()
p2 = t2m_mean.values.ravel()
land = is_land.values.ravel()


# ── Fit OLS on valid land pixels ───────────────────────────────────────────────

fit_mask = land & np.isfinite(s) & np.isfinite(p1) & np.isfinite(p2)
if fit_mask.sum() == 0:
    raise ValueError(
        "No valid land pixels to fit — check ssrd/t2m alignment with solar_CF grid."
    )

X_fit = np.column_stack([p1[fit_mask], p2[fit_mask], np.ones(fit_mask.sum())])
coeffs, _, _, _ = np.linalg.lstsq(X_fit, s[fit_mask], rcond=None)
print(f"OLS fit on {fit_mask.sum():,} land pixels")
print(f"  ssrd={coeffs[0]:.4f}  t2m={coeffs[1]:.4f}  intercept={coeffs[2]:.4f}")


# ── Predict for NaN land pixels ────────────────────────────────────────────────

fill_mask = land & ~np.isfinite(s) & np.isfinite(p1) & np.isfinite(p2)
print(f"Filling {fill_mask.sum():,} NaN land pixels")

s_filled = s.copy()
X_pred = np.column_stack([p1[fill_mask], p2[fill_mask], np.ones(fill_mask.sum())])
s_filled[fill_mask] = np.clip(X_pred @ coeffs, 0.0, 1.0)

solar_filled = xr.DataArray(
    s_filled.reshape(solar.shape),
    dims=solar.dims,
    coords=solar.coords,
    attrs={
        **solar.attrs,
        "gap_fill": (
            "NaN above ~60°N filled by OLS on ERA5 annual-mean ssrd and t2m "
            "(land only, regionmask natural_earth land_110). "
            f"Coefficients: ssrd={coeffs[0]:.4f}, t2m={coeffs[1]:.4f}, intercept={coeffs[2]:.4f}."
        ),
    },
)


# ── Save ───────────────────────────────────────────────────────────────────────

ds_out = ds.copy()
ds_out["solar_CF"] = solar_filled
ds_out.attrs["solar_CF_gap_fill"] = (
    "NaN above ~60°N filled by OLS regression on ERA5 annual-mean ssrd and t2m "
    "(land pixels only, regionmask natural_earth land_110)"
)

print(f"Saving to {OUTPUT_FILE} ...")
ds_out.to_netcdf(OUTPUT_FILE, engine="netcdf4")
print("Done.")
