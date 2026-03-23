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

import numpy as np
import xarray as xr
from typing import Tuple
from src.geo_processing import load_era5_variable
from config import ERA5_ZARR_URL

# α values: wind capacity share (0.0 = pure solar, 1.0 = pure wind)
ALPHA_VALUES = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), decimals=1)


def _peak_energy_deficit_cyclic(cum_imbalance: xr.DataArray, time_dim: str = "day"):
    """
    Core math for Energy Deficit using Cyclic Extension.
    1. Duplicate series (2*T) to handle deficits straddling the window boundary.
    2. Drawdown(t) = Current level - Running minimum.
    3. Return peak deficit and onset day relative to the original window.
    """
    da = cum_imbalance.astype("float32")
    T = 365  # Fixed window length from noleap calendar

    # Cyclic extension: Concat to handle boundary seams
    # Add the total sum of the first period to the second to maintain integration
    final_val = da.isel({time_dim: -1})
    cum_extended = xr.concat([da, da + final_val], dim=time_dim)

    # Find the minimum level seen in a rolling window of size T
    run_min = cum_extended.rolling({time_dim: T}, min_periods=1).min()
    drawdown_ext = cum_extended - run_min

    # Extract results and identify peak within the first period
    drawdown = drawdown_ext.isel({time_dim: slice(0, T)})
    deficit = drawdown.max(dim=time_dim)

    # Onset calculation: find the deepest trough (argmin) preceding the peak drawdown
    t_peak_idx = drawdown.argmax(dim=time_dim)
    t_coord = xr.DataArray(np.arange(T, dtype=np.int16), dims=[time_dim])
    da_masked = da.where(t_coord <= t_peak_idx, other=np.inf)
    onset = da_masked.argmin(dim=time_dim).astype("int16")

    return deficit, onset


def compute_lds_for_tile(
    tile: Tuple[float, ...], start_year: int, end_year: int, alpha_values=ALPHA_VALUES
):
    """
    Processes July-June windows. Year Y label = 01 Jul Y to 30 Jun Y+1.
    Uses vectorized stacking to prevent Dask scheduler metadata conflicts.
    """
    wind_raw = load_era5_variable(
        ERA5_ZARR_URL, "ws100", tile, start_year, end_year, daily_sum=False
    )
    solar_raw = load_era5_variable(
        ERA5_ZARR_URL, "ssrd", tile, start_year, end_year, daily_sum=True
    )

    # Leap days removed to keep all windows exactly 365 days
    wind_raw = wind_raw.convert_calendar("noleap", dim="valid_time")  # .persist()
    solar_raw = solar_raw.convert_calendar("noleap", dim="valid_time")  # .persist()

    window_years = list(range(start_year, end_year))
    results_by_alpha = []

    for alpha in alpha_values:
        # print(f"--- Computing alpha={alpha} for Tile: {'_'.join(map(str, tile))} ---")
        yearly_imbalances = []

        for yr in window_years:
            w_start, w_end = f"{yr}-07-01", f"{yr + 1}-06-30"
            try:
                wind_w = wind_raw.sel(valid_time=slice(w_start, w_end))
                solar_w = solar_raw.sel(valid_time=slice(w_start, w_end))

                if len(wind_w.valid_time) != 365:
                    continue

                # Normalization: Wind and Solar divided by their specific window mean
                w_norm = wind_w / wind_w.mean("valid_time")
                s_norm = solar_w / solar_w.mean("valid_time")

                G = alpha * w_norm.fillna(0) + (1.0 - alpha) * s_norm.fillna(0)
                imb = (1.0 - G).cumsum(dim="valid_time")

                # Prepare for stacking: rename time to a generic 'day' index
                imb = imb.assign_coords(valid_time=np.arange(365)).rename(
                    {"valid_time": "day"}
                )
                yearly_imbalances.append(imb.assign_coords(year=yr).expand_dims("year"))
            except Exception:
                continue

        if not yearly_imbalances:
            continue

        # Vectorized step: Stack all years and compute LDS in one single graph per alpha
        alpha_stack = xr.concat(yearly_imbalances, dim="year").chunk(
            {"year": 1, "day": -1}
        )
        def_alpha, onset_alpha = _peak_energy_deficit_cyclic(
            alpha_stack, time_dim="day"
        )

        # # Execute the computation for the entire alpha slice
        # def_alpha, onset_alpha = xr.compute(def_alpha, onset_alpha)

        results_by_alpha.append(
            (
                def_alpha.assign_coords(alpha=float(alpha)).expand_dims("alpha"),
                onset_alpha.assign_coords(alpha=float(alpha)).expand_dims("alpha"),
            )
        )

    # Assemble final Dataset (alpha, year, lat, lon)
    deficit_full = xr.concat([r[0] for r in results_by_alpha], dim="alpha")
    onset_full = xr.concat([r[1] for r in results_by_alpha], dim="alpha")

    return xr.Dataset(
        {"energy_deficit_days": deficit_full, "deficit_onset_day": onset_full},
        attrs={
            "description": "Peak energy deficit and onset day following Antonini (2024).",
            "alpha_definition": "alpha=1.0 pure wind | alpha=0.0 pure solar",
            "window": "01 Jul to 30 Jun, leap days removed",
        },
    )


# def _peak_energy_deficit_cyclic(cum_imbalance: xr.DataArray, time_dim: str = "time"):
#     """
#     Core math for Energy Deficit using Cyclic Extension.

#     1. Duplicate series (2*T) to handle deficits straddling the window boundary.
#     2. Drawdown(t) = Current level - Running minimum.
#     3. Return peak deficit and onset day relative to the original window.
#     """
#     da = cum_imbalance.astype("float32").chunk({time_dim: -1})
#     T = da.sizes[time_dim]

#     # Cyclic extension: Concat to handle boundary seams (jump between day 1 and 365)
#     # We add the total sum of the first period to the second to maintain integration
#     cum_extended = xr.concat([da, da + da.isel({time_dim: -1})], dim=time_dim)

#     # Find the minimum level seen in a rolling window of size T
#     run_min = cum_extended.rolling({time_dim: T}, min_periods=1).min()
#     drawdown_ext = cum_extended - run_min

#     # Extract results and identify peak within the first period
#     drawdown = drawdown_ext.isel({time_dim: slice(0, T)})
#     deficit = drawdown.max(dim=time_dim)

#     # Onset calculation: argmin of E(t) preceding peak drawdown
#     t_peak_idx = drawdown.argmax(dim=time_dim)
#     t_coord = xr.DataArray(np.arange(T, dtype=np.int16), dims=[time_dim])
#     da_masked = da.where(t_coord <= t_peak_idx, other=np.inf)
#     onset = da_masked.argmin(dim=time_dim).astype("int16")

#     return deficit, onset


# def compute_lds_for_tile(
#     tile: Tuple[float, ...], start_year: int, end_year: int, alpha_values=ALPHA_VALUES
# ):
#     """
#     Processes July-June windows. Year Y label = 01 Jul Y to 30 Jun Y+1.
#     """
#     wind_raw = load_era5_variable(
#         ERA5_ZARR_URL, "ws100", tile, start_year, end_year, daily_sum=False
#     )
#     solar_raw = load_era5_variable(
#         ERA5_ZARR_URL, "ssrd", tile, start_year, end_year, daily_sum=True
#     )

#     # Leap days removed to keep all windows exactly 365 days
#     wind_raw = wind_raw.convert_calendar("noleap", dim="valid_time").persist()
#     solar_raw = solar_raw.convert_calendar("noleap", dim="valid_time").persist()

#     window_years = list(range(start_year, end_year))
#     results_by_alpha = []

#     for alpha in alpha_values:
#         results_by_year = []
#         for yr in window_years:
#             w_start, w_end = f"{yr}-07-01", f"{yr + 1}-06-30"
#             try:
#                 wind_w = wind_raw.sel(valid_time=slice(w_start, w_end)).chunk(
#                     {"valid_time": -1}
#                 )
#                 solar_w = solar_raw.sel(valid_time=slice(w_start, w_end)).chunk(
#                     {"valid_time": -1}
#                 )

#                 # Normalization and Mixture
#                 w_norm = wind_w / wind_w.mean("valid_time")
#                 s_norm = solar_w / solar_w.mean("valid_time")
#                 G = alpha * w_norm.fillna(0) + (1.0 - alpha) * s_norm.fillna(0)

#                 # Peak deficit using cyclic logic
#                 imb = 1.0 - G
#                 cum_imb = imb.cumsum(dim="valid_time")
#                 deficit_yr, onset_yr = _peak_energy_deficit_cyclic(
#                     cum_imb, "valid_time"
#                 )

#                 results_by_year.append((deficit_yr.compute(), onset_yr.compute()))
#             except Exception:
#                 continue

#         if not results_by_year:
#             continue

#         deficit_alpha = (
#             xr.concat([r[0] for r in results_by_year], dim="year")
#             .assign_coords(alpha=float(alpha))
#             .expand_dims("alpha")
#         )
#         onset_alpha = (
#             xr.concat([r[1] for r in results_by_year], dim="year")
#             .assign_coords(alpha=float(alpha))
#             .expand_dims("alpha")
#         )
#         results_by_alpha.append((deficit_alpha, onset_alpha))

#     if not results_by_alpha:
#         raise RuntimeError(f"No valid windows computed for tile {tile}")

#     # ── 4. Assemble (alpha, year, lat, lon) ───────────────────────────────
#     deficit_full = xr.concat([r[0] for r in results_by_alpha], dim="alpha")
#     onset_full = xr.concat([r[1] for r in results_by_alpha], dim="alpha")

#     return xr.Dataset(
#         {
#             "energy_deficit_days": deficit_full,
#             "deficit_onset_day": onset_full,
#         },
#         attrs={
#             "description": (
#                 "Peak energy deficit and onset day for a wind-solar hybrid system. "
#                 "energy_deficit_days[alpha, year, lat, lon] = maximum consecutive "
#                 "energy shortfall in days of mean combined generation (float). "
#                 "deficit_onset_day[alpha, year, lat, lon] = 0-based day-of-window "
#                 "when the worst deficit period begins (day 0 = 01 Jul of window year). "
#                 "year = calendar year of the July window start "
#                 "(year Y covers 01 Jul Y - 30 Jun Y+1)."
#             ),
#             "alpha_definition": "alpha=1.0 pure wind  |  alpha=0.0 pure solar",
#             "window": "01 Jul (year) to 30 Jun (year+1), leap days removed",
#             "units_deficit": "days (float32)",
#             "units_onset": "days since 01 Jul of window year (int16)",
#             "ERA5_period": f"{start_year}-01-01 to {end_year}-12-31",
#             "normalisation": (
#                 "Wind and solar each divided by their own window mean "
#                 "before alpha-weighting; mixture has unit mean by construction."
#             ),
#         },
#     )


###
## HELPER TO PLOT STORAGE
###

"""
aggregate_lds.py
================
Notebook-friendly function to aggregate the per-year LDS metric tiles
into a single (lat, lon) dataset.
"""

from pathlib import Path
import numpy as np
import xarray as xr


def aggregate_lds(
    lds_dir: str | Path,
    metric: str = "mean",
    fixed_alpha: float | None = None,
) -> xr.Dataset:
    """
    Load all LDS tile files, aggregate across years, return (lat, lon) dataset.

    Parameters
    ----------
    lds_dir : str or Path
        Directory containing lds_*.nc files produced by storage.py.
    metric : {"mean", "p99", "max"}
        How to aggregate the per-year energy deficit across years:
          "mean" — representative planning value
          "p99"  — 99th-percentile, near-worst-case design value
          "max"  — absolute worst year
    fixed_alpha : float or None
        If given, report the metric at this specific alpha instead of
        finding the optimal (minimising) alpha. Must be one of the alpha
        values stored in the files (e.g. 0.0, 0.1, ..., 1.0).

    Returns
    -------
    xr.Dataset with dims (latitude, longitude) and variables:
        duration_metric  float32  [days]
            Aggregated energy deficit at the optimal (or fixed) alpha.
        optimal_alpha    float32  [0–1]
            Wind capacity share that minimises duration_metric.
            Equals fixed_alpha everywhere when fixed_alpha is provided.
    """
    lds_dir = Path(lds_dir)
    metric = metric.lower()

    if metric not in {"mean", "p99", "max"}:
        raise ValueError(f"metric must be 'mean', 'p99', or 'max'; got {metric!r}")

    # ── 1. Load tiles ──────────────────────────────────────────────────────
    files = sorted(lds_dir.glob("lds_*.nc"))
    if not files:
        raise FileNotFoundError(f"No lds_*.nc files found in {lds_dir}")

    print(f"Loading {len(files)} tile(s)...")
    tile_datasets = [xr.open_dataset(f, decode_times=False) for f in files]
    ds = (
        tile_datasets[0]
        if len(tile_datasets) == 1
        else xr.combine_by_coords(tile_datasets, combine_attrs="override")
    )

    # (alpha, year, latitude, longitude) float32
    eld = ds["energy_deficit_days"].astype("float32")

    # ── 2. Aggregate across years ──────────────────────────────────────────
    # skipna=True so that ocean/missing cells (all-NaN alpha slices) stay NaN
    # rather than raising an error.
    print(f"Aggregating across years using metric={metric!r}...")
    if metric == "mean":
        agg = eld.mean(dim="year", skipna=True)
    elif metric == "p99":
        agg = eld.quantile(0.99, dim="year", skipna=True).drop_vars("quantile")
    elif metric == "max":
        agg = eld.max(dim="year", skipna=True)
    # agg shape: (alpha, latitude, longitude)

    # ── 3. Select alpha ────────────────────────────────────────────────────
    if fixed_alpha is not None:
        alpha_vals = agg.alpha.values
        if not np.any(np.isclose(alpha_vals, fixed_alpha, atol=1e-6)):
            raise ValueError(
                f"fixed_alpha={fixed_alpha} not in file. "
                f"Available: {alpha_vals.tolist()}"
            )
        print(f"Using fixed alpha={fixed_alpha}...")
        agg_sel = agg.sel(alpha=fixed_alpha, method="nearest").drop_vars("alpha")
        alpha_map = xr.full_like(agg_sel, fill_value=float(fixed_alpha))

    else:
        # argmin over alpha; skipna keeps NaN cells as NaN instead of crashing
        print("Finding optimal alpha (minimises duration_metric)...")
        alpha_idx = agg.argmin(dim="alpha", skipna=True).compute()  # (lat, lon) int

        agg_sel = agg.isel(alpha=alpha_idx).drop_vars("alpha", errors="ignore")

        alpha_vals = agg.alpha.values  # 1-D numpy
        alpha_map = xr.DataArray(
            alpha_vals[alpha_idx.values].astype("float32"),
            dims=agg_sel.dims,
            coords={k: agg_sel.coords[k] for k in agg_sel.dims},
        )
        # Cells that were all-NaN stay NaN in agg_sel; mirror that in alpha_map
        alpha_map = alpha_map.where(agg_sel.notnull())

    # ── 4. Build output dataset ────────────────────────────────────────────
    agg_sel = agg_sel.astype("float32")
    alpha_map = alpha_map.astype("float32")

    agg_sel.attrs = {
        "long_name": f"Energy deficit duration ({metric} across years)",
        "units": "days",
        "metric": metric,
    }
    alpha_map.attrs = {
        "long_name": (
            "Fixed alpha"
            if fixed_alpha is not None
            else "Optimal wind capacity share (minimises duration_metric)"
        ),
        "units": "dimensionless",
        "note": "alpha=1.0 pure wind | alpha=0.0 pure solar",
    }

    ds_out = xr.Dataset(
        {"duration_metric": agg_sel, "optimal_alpha": alpha_map},
        attrs={
            "description": (
                f"LDS metric aggregated from Jul-Jun windows. metric={metric}. "
                "optimal_alpha = wind share minimising duration_metric."
            ),
            "source_dir": str(lds_dir),
            "n_tiles": len(files),
        },
    )
    print("Done.")
    return ds_out
