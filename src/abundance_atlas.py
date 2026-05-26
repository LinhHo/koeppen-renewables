"""
Resample atlas data to reference grid
"""

import numpy as np
import xarray as xr
import rioxarray as rxr
import regionmask
from geo_processing import clip_and_resample, create_tile_template


def resample_atlas(bounds, paths, resolution):
    template = create_tile_template(bounds, resolution)

    ds = xr.Dataset()
    # Solar PVOUT - Photovoltaic power potential [kWh/kWp] average daily totals
    # convert to capacity factor by dividing by 24 hours
    with rxr.open_rasterio(paths["solar_atlas"], chunks=True).squeeze() as solar_atlas:
        ds["solar_CF"] = clip_and_resample(solar_atlas, template) / 24

    # Wind power capacity factor (unitless)
    with rxr.open_rasterio(paths["wind_atlas"], chunks=True).squeeze() as wind_atlas:
        ds["wind_CF"] = clip_and_resample(wind_atlas, template)

    return ds


def fill_solar_cf_gap(
    ds: xr.Dataset,
    ssrd_clim: xr.DataArray,
    t2m_clim: xr.DataArray,
) -> xr.Dataset:
    """
    Fill NaN solar_CF values (above ~60°N atlas coverage limit) via OLS regression
    on ERA5 climatological predictors, restricted to land pixels.

    Fit on land pixels where solar_CF is valid; predict only on land NaN pixels.
    Predictors: annual-mean ssrd (direct irradiance) and annual-mean t2m
    (cloud-cover / sky-clarity proxy).  Result is clipped to [0, 1].

    Parameters
    ----------
    ds : xr.Dataset
        Output of resample_atlas. Must contain 'solar_CF' (latitude, longitude).
    ssrd_clim : xr.DataArray
        ssrd climatology (dayofyear, latitude, longitude) or annual mean (lat, lon).
    t2m_clim : xr.DataArray
        t2m climatology, same convention.

    Returns
    -------
    xr.Dataset
        Copy of ds with solar_CF NaN-filled on land.
    """
    solar = ds["solar_CF"].compute()

    # Land mask — True where land
    land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(solar)
    is_land = land_mask.notnull()

    # Annual means — collapse dayofyear if present
    ssrd_mean = ssrd_clim.mean("dayofyear") if "dayofyear" in ssrd_clim.dims else ssrd_clim
    t2m_mean = t2m_clim.mean("dayofyear") if "dayofyear" in t2m_clim.dims else t2m_clim

    # Align ERA5 grids to the atlas grid
    ssrd_mean = ssrd_mean.interp(latitude=solar.latitude, longitude=solar.longitude)
    t2m_mean = t2m_mean.interp(latitude=solar.latitude, longitude=solar.longitude)

    # Flatten to 1-D
    s = solar.values.ravel()
    p1 = ssrd_mean.values.ravel()
    p2 = t2m_mean.values.ravel()
    land = is_land.values.ravel()

    # Fit: land pixels where solar_CF and both predictors are finite
    fit_mask = land & np.isfinite(s) & np.isfinite(p1) & np.isfinite(p2)
    if fit_mask.sum() == 0:
        raise ValueError("No valid land pixels to fit regression — check ssrd/t2m alignment.")

    X_fit = np.column_stack([p1[fit_mask], p2[fit_mask], np.ones(fit_mask.sum())])
    coeffs, _, _, _ = np.linalg.lstsq(X_fit, s[fit_mask], rcond=None)

    # Predict: land NaN pixels where both predictors are finite
    fill_mask = land & ~np.isfinite(s) & np.isfinite(p1) & np.isfinite(p2)
    X_pred = np.column_stack([p1[fill_mask], p2[fill_mask], np.ones(fill_mask.sum())])

    s_filled = s.copy()
    s_filled[fill_mask] = np.clip(X_pred @ coeffs, 0.0, 1.0)

    solar_filled = xr.DataArray(
        s_filled.reshape(solar.shape),
        dims=solar.dims,
        coords=solar.coords,
        attrs={**solar.attrs, "gap_fill": "OLS on ERA5 annual-mean ssrd and t2m (land only)"},
    )

    ds = ds.copy()
    ds["solar_CF"] = solar_filled
    return ds
