"""
storage.py
==========
Computes the long-duration energy deficit metric for each grid cell.

For each (lat, lon) cell the metric answers:
    "If this location generates a mix of wind (fraction α) and solar
     (fraction 1−α), how many days-worth of mean generation would a
     perfect storage device need to discharge to cover the worst
     consecutive energy deficit in a given winter-centred year?"

Output
------
4-D NetCDF with dimensions (alpha, year, latitude, longitude) and
data variable ``energy_deficit_days``.

``year`` labels the *July start* of the window:
    year=2024  →  01 Jul 2024 – 30 Jun 2025

Winter-centred window rationale
--------------------------------
Dunkelflaute (dark-doldrums) events — simultaneous wind and solar lulls
— occur in boreal winter (Nov–Feb) and are always near the centre of a
Jul–Jun window.  Centring on winter avoids:
  - the artificial year-end seam that the cyclic-extension trick patches,
  - any deficit that straddles 31 Dec / 1 Jan being split across two
    analysis windows and therefore under-counted.

Algorithm
---------
For a 365-day window, compute the normalised combined generation:

    G(t, α) = α · W̃(t) + (1−α) · S̃(t)

where W̃ and S̃ are each divided by their own window mean (so the
mixture has unit mean by construction, and α is a true capacity-share
weight independent of physical units).

Instantaneous imbalance relative to flat unit demand:

    δ(t) = 1 − G(t, α)          (+ve when generation < demand)

Cumulative energy deficit:

    E(t) = Σ_{τ=0}^{t} δ(τ)     [days of mean combined generation]

Peak energy deficit (= maximum storage discharge needed without
recharging during the window):

    D = max_{t₂ ≥ t₁} [ E(t₂) − E(t₁) ]
      = max_t [ E(t) − running_min_t E(t) ]

Key design decisions
--------------------
- Window:  01 Jul → 30 Jun  (365 days after removing leap days)
- Year label: calendar year of the July start
- Resolution: daily (ERA5 daily mean/sum); sub-daily diurnal cycling is
  short-duration-battery territory
- Normalise each resource by its own window mean before mixing
- α sweep: 0.0 (pure solar) → 1.0 (pure wind), step 0.1
- No cyclic extension needed: winter-centred window is self-contained

References
----------
Antonini et al. (2024), Commun. Earth Environ. 5:103
Dowling et al. (2020), Joule 4(9):1907-1928
"""

import numpy as np
import xarray as xr
from typing import Tuple
from pathlib import Path
import pandas as pd

from src.geo_processing import load_era5_variable
from config import ERA5_ZARR_URL

Tile = Tuple[float, float, float, float]

# α values: wind capacity share (0.0 = pure solar, 1.0 = pure wind)
ALPHA_VALUES = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), decimals=1)


# ---------------------------------------------------------------------------
# Core: peak energy deficit from a cumulative imbalance series
# ---------------------------------------------------------------------------


def _peak_energy_deficit(
    cum_imbalance: xr.DataArray,
    time_dim: str = "time",
) -> xr.DataArray:
    """
    Peak energy deficit (maximum drawdown) of a cumulative imbalance E(t).

    At each time step t, the drawdown is how far E has risen above the
    deepest trough seen so far:

        dd(t) = E(t) − min_{τ ≤ t} E(τ)

    The peak drawdown is max_t dd(t) — the largest storage discharge
    needed without recharging during the window.

    No cyclic extension is applied; the caller provides a window that
    already contains the worst events (winter-centred).

    Parameters
    ----------
    cum_imbalance : DataArray with dim ``time_dim``
        E(t) in days of mean combined generation.
    time_dim : str

    Returns
    -------
    DataArray (spatial dims only) — peak energy deficit in days.
    """
    da = cum_imbalance.chunk({time_dim: -1})
    T = da.sizes[time_dim]

    # Running minimum: the deepest trough seen up to each time step
    run_min = da.rolling({time_dim: T}, min_periods=1).min()

    # Drawdown at each step
    drawdown = da - run_min

    return drawdown.max(dim=time_dim)


# ---------------------------------------------------------------------------
# Single Jul–Jun window × single α
# ---------------------------------------------------------------------------


def _energy_deficit_one_window(
    wind_window: xr.DataArray,
    solar_window: xr.DataArray,
    alpha: float,
    time_dim: str = "valid_time",
) -> xr.DataArray:
    """
    Peak energy deficit for one Jul–Jun window and one α.

    Parameters
    ----------
    wind_window  : daily mean wind speed, shape (T≈365, lat, lon)
    solar_window : daily sum solar radiation, shape (T≈365, lat, lon)
    alpha        : wind capacity share [0, 1]
    time_dim     : name of the time coordinate

    Returns
    -------
    DataArray (lat, lon) — peak energy deficit in days.
    """
    # ── Normalise by window mean ──────────────────────────────────────────
    # xr.where avoids division-by-zero in polar-night or permanently calm cells
    wind_mean = wind_window.mean(dim=time_dim)
    solar_mean = solar_window.mean(dim=time_dim)

    wind_norm = xr.where(wind_mean > 0, wind_window / wind_mean, 0.0)
    solar_norm = xr.where(solar_mean > 0, solar_window / solar_mean, 0.0)

    # ── Combined normalised generation ────────────────────────────────────
    # Mean of G = α·1 + (1−α)·1 = 1 by construction
    G = alpha * wind_norm + (1.0 - alpha) * solar_norm

    # ── Cumulative imbalance ──────────────────────────────────────────────
    imbalance = 1.0 - G  # +ve when below demand
    cum_imbalance = imbalance.cumsum(dim=time_dim)

    # ── Peak deficit ──────────────────────────────────────────────────────
    deficit = _peak_energy_deficit(cum_imbalance, time_dim=time_dim)
    deficit.attrs.update(
        {
            "units": "days",
            "long_name": (
                f"Peak energy deficit — alpha={alpha:.1f} "
                "(days of mean combined generation)"
            ),
        }
    )
    return deficit


# ---------------------------------------------------------------------------
# Tile-level entry point: all winter-years × all α values
# ---------------------------------------------------------------------------


def compute_lds_for_tile(
    tile: Tile,
    start_year: int,
    end_year: int,
    alpha_values: np.ndarray = ALPHA_VALUES,
) -> xr.Dataset:
    """
    Compute peak energy deficit for every (winter-year, α) for one tile.

    Window labelling
    ----------------
    year Y  →  01 Jul Y – 30 Jun (Y+1)

    ERA5 is loaded from {start_year}-01-01 to {end_year}-12-31 (full
    calendar years, as required by load_era5_variable), then sub-sliced
    to Jul–Jun windows.

    The first complete window is labelled start_year (needs data from
    01 Jul start_year); the last is labelled (end_year − 1) (needs data
    through 30 Jun end_year).

    Example:  start_year=1995, end_year=2025
        → windows 1995 … 2024  (30 windows)
        → ERA5 data 1995-01-01 … 2025-12-31

    Parameters
    ----------
    tile        : (minx, miny, maxx, maxy)
    start_year  : first window label (= calendar year of July start)
    end_year    : last calendar year loaded (= last window label + 1)
    alpha_values: 1-D array of α values to sweep

    Returns
    -------
    xr.Dataset
        Variable ``energy_deficit_days``,
        dims (alpha, year, latitude, longitude).
    """
    # ── 1. Load full ERA5 period ──────────────────────────────────────────
    wind_raw = load_era5_variable(
        ERA5_ZARR_URL, "ws100", tile, start_year, end_year, daily_sum=False
    )
    solar_raw = load_era5_variable(
        ERA5_ZARR_URL, "ssrd", tile, start_year, end_year, daily_sum=True
    )

    if wind_raw is None or solar_raw is None:
        raise ValueError(f"ERA5 load returned None for tile {tile}")

    # Drop leap days: every Jul–Jun window will have exactly 365 days
    wind_raw = wind_raw.convert_calendar("noleap", dim="valid_time")
    solar_raw = solar_raw.convert_calendar("noleap", dim="valid_time")

    # Persist in worker memory once; reused across all window/alpha iterations
    wind_raw = wind_raw.persist()
    solar_raw = solar_raw.persist()

    # ── 2. Winter-year labels ─────────────────────────────────────────────
    # year Y needs data through 30 Jun Y+1, so last label is end_year - 1
    window_years = list(range(start_year, end_year))

    # ── 3. Outer loop: α; inner loop: year ───────────────────────────────
    # Computing one (year × α) slice at a time keeps the dask graph small
    # and avoids accumulating a huge lazy graph before the first .compute().
    results_by_alpha = []

    for alpha in alpha_values:
        results_by_year = []

        for yr in window_years:
            w_start = f"{yr}-07-01"
            w_end = f"{yr + 1}-06-30"

            wind_w = wind_raw.sel(valid_time=slice(w_start, w_end)).chunk(
                {"valid_time": -1, "latitude": -1, "longitude": -1}
            )
            solar_w = solar_raw.sel(valid_time=slice(w_start, w_end)).chunk(
                {"valid_time": -1, "latitude": -1, "longitude": -1}
            )

            n_days = wind_w.sizes["valid_time"]
            if n_days < 360:
                print(
                    f"  [WARN] Window {yr} ({w_start}→{w_end}): "
                    f"only {n_days} days — skipping"
                )
                continue

            # Compute immediately to bound memory usage per iteration
            deficit_yr = _energy_deficit_one_window(
                wind_w,
                solar_w,
                alpha=float(alpha),
                time_dim="valid_time",
            ).compute()

            deficit_yr = deficit_yr.assign_coords(year=yr).expand_dims("year")
            results_by_year.append(deficit_yr)

        if not results_by_year:
            continue

        deficit_alpha = (
            xr.concat(results_by_year, dim="year")
            .assign_coords(alpha=float(alpha))
            .expand_dims("alpha")
        )
        results_by_alpha.append(deficit_alpha)

    if not results_by_alpha:
        raise RuntimeError(f"No valid windows computed for tile {tile}")

    # ── 4. Assemble (alpha, year, lat, lon) ───────────────────────────────
    # convert to float of days
    deficit_full = xr.concat(results_by_alpha, dim="alpha") / pd.Timedelta(days=1)

    return xr.Dataset(
        {"energy_deficit_days": deficit_full},
        attrs={
            "description": (
                "Peak energy deficit for a wind-solar hybrid system. "
                "energy_deficit_days[alpha, year, lat, lon] = maximum "
                "consecutive energy shortfall in days of mean combined "
                "generation. year = calendar year of the July window start "
                "(year Y covers 01 Jul Y – 30 Jun Y+1)."
            ),
            "alpha_definition": "alpha=1.0 pure wind  |  alpha=0.0 pure solar",
            "window": "01 Jul (year) → 30 Jun (year+1), leap days removed",
            "units": "days",
            "ERA5_period": f"{start_year}-01-01 to {end_year}-12-31",
            "normalisation": (
                "Wind and solar each divided by their own window mean "
                "before α-weighting; mixture has unit mean by construction."
            ),
        },
    )
