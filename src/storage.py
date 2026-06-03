"""
storage.py: Peak Energy Deficit (LDS) Calculation
=================================================
Calculates the maximum drawdown of storage required to bridge energy deficits
under a unit-load assumption, following Antonini et al. (2024, https://www.nature.com/articles/s41597-024-04129-8).

Reproducibility:
- Window: 01 July to 30 June (365 days, leap days removed).
- Normalization: Wind (W) and Solar (S) are normalized by their specific window mean.
- Combined Generation: G(t, α) = α·W_norm + (1−α)·S_norm.
- Metric: Peak Deficit D = max_t [ E(t) − min_{τ≤t} E(τ) ] over a cyclically extended period.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr

from geo_processing import load_era5_variable
from config import ERA5_ZARR_URL

log = logging.getLogger(__name__)

# α values: wind capacity share (0.0 = pure solar, 1.0 = pure wind)
ALPHA_VALUES = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), decimals=1)


def _clim_annual_mean(clim_dir: Path, var: str) -> xr.DataArray:
    """Return the annual-mean climatology for *var*, using a cached file when available.

    Cache location: ``clim_dir/{var}_annual_mean.nc``.
    On first call the mean is computed from ``clim_dir/{var}/*.nc``
    (variable ``{var}_climatology``, averaged over dayofyear) and saved there.
    Subsequent calls load the cache directly, skipping the mfdataset read.
    """
    cache = Path(clim_dir) / f"{var}_annual_mean.nc"
    if cache.exists():
        log.debug("Loading cached %s annual mean from %s", var, cache)
        return xr.open_dataset(str(cache))[var]

    files = sorted((Path(clim_dir) / var).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {Path(clim_dir) / var}")

    log.info("Computing annual mean for %s (%d files) → %s", var, len(files), cache)
    da = (
        xr.open_mfdataset(files, combine="by_coords", engine="netcdf4")[f"{var}"]
        .mean("dayofyear")
        .compute()
    )
    da.name = var
    da.to_netcdf(str(cache))
    return da


def _peak_energy_deficit_cyclic(
    cum_imbalance: xr.DataArray, time_dim: str = "valid_time"
) -> xr.DataArray:
    """
    Peak energy storage deficit via cyclic extension (Antonini et al. 2024).

    Algorithm
    ---------
    drawdown(t) = cum_imbalance(t) − min_{τ≤t} cum_imbalance(τ)
    peak_deficit = max_t drawdown(t)

    The series is doubled so that deficits straddling the Jul–Jun boundary
    are captured. T is inferred from the data, so 365-day and 366-day
    windows are both handled correctly.

    Parameters
    ----------
    cum_imbalance : xr.DataArray
        Cumulative demand-minus-generation imbalance along *time_dim*.
        Any number of extra dimensions (alpha, latitude, longitude …) are allowed.
    time_dim : str
        Name of the time dimension.

    Returns
    -------
    xr.DataArray
        Peak deficit (days), with *time_dim* reduced out.
    """
    da = cum_imbalance.astype("float32")
    T = da.sizes[time_dim]

    # Cyclic extension: offset the second copy by the end-of-period imbalance
    # so the series is continuous at the seam.
    final_val = da.isel({time_dim: -1})
    cum_extended = xr.concat([da, da + final_val], dim=time_dim)

    # Running minimum over a window of length T (min seen so far at each step).
    run_min = cum_extended.rolling({time_dim: T}, min_periods=1).min()

    # Drawdown = current level minus running minimum.
    # Restricted to the first T steps; second copy is only an extension aid.
    drawdown = (cum_extended - run_min).isel({time_dim: slice(0, T)})

    return drawdown.max(dim=time_dim)


def compute_lds_for_tile(
    tile: Tuple[float, ...],
    start_year: int,
    end_year: int,
    clim_dir: str | Path,
    alpha_values: np.ndarray = ALPHA_VALUES,
) -> xr.Dataset:
    """
    Compute Long-Duration Storage (LDS) metric for one spatial tile.

    Windows span 01 Jul Y to 30 Jun Y+1 for Y in [start_year, end_year).
    Leap days are retained; window length is 365 or 366 days.

    Normalization
    -------------
    Wind and solar are divided by their climatological annual mean
    (mean of ws100 / ssrd over all dayofyear values in clim_dir).
    This fixes the reference across years so inter-annual deficit values
    are comparable (Antonini et al. 2024, constant-target variant).

    Alpha vectorisation
    -------------------
    G(t, α) = α·W_norm + (1−α)·S_norm is computed for all α simultaneously
    (no alpha loop), reducing overhead from repeated ERA5 reads.

    Dask / dask_mpi
    ---------------
    When wind_raw / solar_raw are Dask-backed (ERA5 zarr with chunks), all
    operations build a lazy graph.  Call this function inside a dask_mpi
    worker to parallelise across tiles; the graph for each tile is computed
    on the worker that owns it.

    Parameters
    ----------
    tile : tuple of float
        (lat_min, lat_max, lon_min, lon_max).
    start_year, end_year : int
        Year range: windows Jul start_year – Jun (end_year-1)+1.
    clim_dir : str or Path
        Directory containing climatology_*.nc files with variables
        ws100 and ssrd on a (dayofyear, latitude, longitude) grid.
    alpha_values : array-like
        Wind capacity shares to sweep (default 0.0 … 1.0 in 0.1 steps).

    Returns
    -------
    xr.Dataset
        energy_deficit_days(alpha, year, latitude, longitude)  float32
    """
    # ── 1. ERA5 wind and solar ────────────────────────────────────────────────
    wind_raw = load_era5_variable(
        ERA5_ZARR_URL, "ws100", tile, start_year, end_year, daily_sum=False
    )
    solar_raw = load_era5_variable(
        ERA5_ZARR_URL, "ssrd", tile, start_year, end_year, daily_sum=True
    )

    # ── 2. Climatological normalization values ────────────────────────────────
    # Load (or compute+cache) the annual mean for each variable, then align
    # to the ERA5 tile grid and mask polar-night / missing cells.
    clim_mean = xr.Dataset(
        {var: _clim_annual_mean(clim_dir, var) for var in ("ws100", "ssrd")}
    )
    # ssrd daily sum climatology converted from hourly, so it matches the temporal resolution of solar_raw
    clim_mean["ssrd"] = clim_mean["ssrd"] * 24
    clim_mean = clim_mean.sel(
        latitude=wind_raw.latitude, longitude=wind_raw.longitude, method="nearest"
    )
    clim_mean = clim_mean.where(clim_mean > 0)

    # ── 3. Per-year deficit, vectorised over alpha ────────────────────────────
    alpha_da = xr.DataArray(
        alpha_values.astype("float32"),
        dims=["alpha"],
        coords={"alpha": alpha_values.astype("float32")},
    )
    window_years = list(range(start_year, end_year))
    yearly_results = []

    for yr in window_years:
        w_start, w_end = f"{yr}-07-01", f"{yr + 1}-06-30"
        wind_w = wind_raw.sel(valid_time=slice(w_start, w_end))
        solar_w = solar_raw.sel(valid_time=slice(w_start, w_end))

        T = len(wind_w.valid_time)
        if T not in (365, 366):
            raise ValueError(
                f"Tile {tile}: window Jul {yr}–Jun {yr + 1} has {T} days "
                f"(expected 365 or 366). Verify ERA5 data coverage."
            )

        # Normalize by climatological annual mean (fillna(0) → no generation where missing)
        w_norm = (wind_w / clim_mean["ws100"]).fillna(0.0)
        s_norm = (solar_w / clim_mean["ssrd"]).fillna(0.0)

        # G: (alpha, valid_time, latitude, longitude) via xarray broadcasting
        G = alpha_da * w_norm + (1.0 - alpha_da) * s_norm

        # Cumulative demand-minus-generation imbalance
        imb = (1.0 - G).cumsum(dim="valid_time")

        # Peak deficit for all α simultaneously → (alpha, latitude, longitude)
        def_yr = _peak_energy_deficit_cyclic(imb, time_dim="valid_time")
        yearly_results.append(def_yr.assign_coords(year=yr).expand_dims("year"))
        log.debug("Tile %s  year %d  T=%d  done", tile, yr, T)

    if not yearly_results:
        raise RuntimeError(
            f"Tile {tile}: no valid Jul–Jun windows found in "
            f"[{start_year}, {end_year}). Check ERA5 data."
        )

    # Concatenate → (year, alpha, latitude, longitude),
    # then transpose to canonical (alpha, year, latitude, longitude).
    deficit_full = xr.concat(yearly_results, dim="year").transpose("alpha", "year", ...)

    return xr.Dataset(
        {"energy_deficit_days": deficit_full.astype("float32")},
        attrs={
            "description": "Peak energy deficit following Antonini (2024).",
            "alpha_definition": "alpha=1.0 pure wind | alpha=0.0 pure solar",
            "window": "01 Jul to 30 Jun, leap days included",
            "normalization": "climatological annual mean (dayofyear average)",
        },
    )


# ###
# ## HELPER TO PLOT STORAGE — aggregate_lds
# ###


# def aggregate_lds(
#     lds_dir: str | Path,
#     metric: str = "mean",
#     fixed_alpha: float | None = None,
# ) -> xr.Dataset:
#     """
#     Load all LDS tile files, aggregate across years, return (lat, lon) dataset.

#     Parameters
#     ----------
#     lds_dir : str or Path
#         Directory containing lds_*.nc files produced by storage.py.
#     metric : {"mean", "p99", "max"}
#         How to aggregate the per-year energy deficit across years:
#           "mean" — representative planning value
#           "p99"  — 99th-percentile, near-worst-case design value
#           "max"  — absolute worst year
#     fixed_alpha : float or None
#         If given, report the metric at this specific alpha instead of
#         finding the optimal (minimising) alpha. Must be one of the alpha
#         values stored in the files (e.g. 0.0, 0.1, ..., 1.0).

#     Returns
#     -------
#     xr.Dataset with dims (latitude, longitude) and variables:
#         duration_metric  float32  [days]
#             Aggregated energy deficit at the optimal (or fixed) alpha.
#         optimal_alpha    float32  [0–1]
#             Wind capacity share that minimises duration_metric.
#             Equals fixed_alpha everywhere when fixed_alpha is provided.
#     """
#     lds_dir = Path(lds_dir)
#     metric = metric.lower()

#     if metric not in {"mean", "p99", "max"}:
#         raise ValueError(f"metric must be 'mean', 'p99', or 'max'; got {metric!r}")

#     # ── 1. Load tiles ──────────────────────────────────────────────────────
#     files = sorted(lds_dir.glob("lds_*.nc"))
#     if not files:
#         raise FileNotFoundError(f"No lds_*.nc files found in {lds_dir}")

#     print(f"Loading {len(files)} tile(s)...")
#     tile_datasets = [xr.open_dataset(f, decode_times=False) for f in files]
#     ds = (
#         tile_datasets[0]
#         if len(tile_datasets) == 1
#         else xr.combine_by_coords(tile_datasets, combine_attrs="override")
#     )

#     # (alpha, year, latitude, longitude) float32
#     eld = ds["energy_deficit_days"].astype("float32")

#     # ── 2. Aggregate across years ──────────────────────────────────────────
#     # skipna=True so that ocean/missing cells (all-NaN alpha slices) stay NaN
#     # rather than raising an error.
#     print(f"Aggregating across years using metric={metric!r}...")
#     if metric == "mean":
#         agg = eld.mean(dim="year", skipna=True)
#     elif metric == "p99":
#         agg = eld.quantile(0.99, dim="year", skipna=True).drop_vars("quantile")
#     elif metric == "max":
#         agg = eld.max(dim="year", skipna=True)
#     # agg shape: (alpha, latitude, longitude)

#     # ── 3. Select alpha ────────────────────────────────────────────────────
#     if fixed_alpha is not None:
#         alpha_vals = agg.alpha.values
#         if not np.any(np.isclose(alpha_vals, fixed_alpha, atol=1e-6)):
#             raise ValueError(
#                 f"fixed_alpha={fixed_alpha} not in file. "
#                 f"Available: {alpha_vals.tolist()}"
#             )
#         print(f"Using fixed alpha={fixed_alpha}...")
#         agg_sel = agg.sel(alpha=fixed_alpha, method="nearest").drop_vars("alpha")
#         alpha_map = xr.full_like(agg_sel, fill_value=float(fixed_alpha))

#     else:
#         # argmin over alpha; skipna keeps NaN cells as NaN instead of crashing
#         print("Finding optimal alpha (minimises duration_metric)...")
#         alpha_idx = agg.argmin(dim="alpha", skipna=True).compute()  # (lat, lon) int

#         agg_sel = agg.isel(alpha=alpha_idx).drop_vars("alpha", errors="ignore")

#         alpha_vals = agg.alpha.values  # 1-D numpy
#         alpha_map = xr.DataArray(
#             alpha_vals[alpha_idx.values].astype("float32"),
#             dims=agg_sel.dims,
#             coords={k: agg_sel.coords[k] for k in agg_sel.dims},
#         )
#         # Cells that were all-NaN stay NaN in agg_sel; mirror that in alpha_map
#         alpha_map = alpha_map.where(agg_sel.notnull())

#     # ── 4. Build output dataset ────────────────────────────────────────────
#     agg_sel = agg_sel.astype("float32")
#     alpha_map = alpha_map.astype("float32")

#     agg_sel.attrs = {
#         "long_name": f"Energy deficit duration ({metric} across years)",
#         "units": "days",
#         "metric": metric,
#     }
#     alpha_map.attrs = {
#         "long_name": (
#             "Fixed alpha"
#             if fixed_alpha is not None
#             else "Optimal wind capacity share (minimises duration_metric)"
#         ),
#         "units": "dimensionless",
#         "note": "alpha=1.0 pure wind | alpha=0.0 pure solar",
#     }

#     ds_out = xr.Dataset(
#         {"duration_metric": agg_sel, "optimal_alpha": alpha_map},
#         attrs={
#             "description": (
#                 f"LDS metric aggregated from Jul-Jun windows. metric={metric}. "
#                 "optimal_alpha = wind share minimising duration_metric."
#             ),
#             "source_dir": str(lds_dir),
#             "n_tiles": len(files),
#         },
#     )
#     print("Done.")
#     return ds_out
