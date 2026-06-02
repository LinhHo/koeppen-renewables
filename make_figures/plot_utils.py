"""
plot_utils.py
=============

Shared infrastructure for all figure scripts in Ho (2026).

Exported:
  - Geographic masks: _has_data, mask_land, mask_offshore
  - Color helpers: _shade, dirty, light
  - LAND_COLORS dict
  - ClassificationSpec dataclass + SPEC_* instances
  - GROUPS_ABUNDANCE, GROUPS_ABUNDANCE_OFFSHORE_WIND, GROUPS_DETAILED, GROUPS_FULL dicts
  - plot_map_continuous
  - log_df_summary, log_ds_summary

Domain-specific functions (zone classification, country clustering, etc.) live in
the respective figure scripts (figures_renewable_zones.py, figures_by_country.py).
"""

from __future__ import annotations

import colorsys
import warnings
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.colorbar as mcolorbar
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patheffects import withStroke

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import regionmask
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr, spearmanr

import logging

log = logging.getLogger(__name__)

# Project paths — resolved lazily so the module can also be imported from
# other working directories.
BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR.parent / "resources"
RESULTS_DIR = BASE_DIR.parent / "results"


# ---------------------------------------------------------------------------
# 1. Masks
# ---------------------------------------------------------------------------


def _has_data(da: xr.DataArray) -> xr.DataArray:
    return da.where(da != 0).notnull()


def mask_land(da: xr.DataArray) -> xr.DataArray:
    """Return a boolean mask for land pixels that carry data in *da*."""
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(
        da.longitude, da.latitude
    )
    return _has_data(da) & land.notnull()


def mask_offshore(da: xr.DataArray) -> xr.DataArray:
    """Return a boolean mask for ocean pixels that carry data in *da*."""
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(
        da.longitude, da.latitude
    )
    return _has_data(da) & land.isnull()


# ---------------------------------------------------------------------------
# 2. ClassificationSpec dataclass + preset instances
# ---------------------------------------------------------------------------


@dataclass
class ClassificationSpec:
    """Declarative description of a zone classification.

    Attributes
    ----------
    name
        Human-readable identifier used in log messages.
    use_solar_land, use_wind_land
        Whether the land label should include a solar/wind abundance character.
    use_storage
        If True, add a reliability character (R/V) derived from storage duration.
    use_demand
        If True, add a demand character (h/l) derived from demand proximity.
    add_offshore
        If True, classify offshore pixels with an ``"offshore_"`` prefix.
    offshore_use_solar
        Whether the offshore label should include a solar abundance character
        (True for the *full* scheme, False for the *detailed* scheme).
    demand_quantile
        Quantile used to threshold the demand proxy (high vs low).
        Demand is the only threshold still derived from a quantile; all
        resource / storage thresholds are supplied via ``threshold_cf``.
    """

    name: str
    use_solar_land: bool = True
    use_wind_land: bool = True
    use_storage: bool = False
    use_demand: bool = False
    add_offshore: bool = False
    offshore_use_solar: bool = False
    demand_quantile: float = 0.5


# Three preset specs that correspond to the three notebook variants
SPEC_ABUNDANCE = ClassificationSpec(
    name="abundance",
    add_offshore=True,
    offshore_use_solar=True,
)

# Same as SPEC_ABUNDANCE but offshore uses wind only (no solar character).
# Pairs with GROUPS_ABUNDANCE_OFFSHORE_WIND.
SPEC_ABUNDANCE_OFFSHORE_WIND = ClassificationSpec(
    name="abundance_offshore_wind",
    add_offshore=True,
    offshore_use_solar=False,
)

SPEC_DETAILED = ClassificationSpec(
    name="detailed",
    use_storage=True,
    use_demand=True,
    add_offshore=True,
    offshore_use_solar=False,  # offshore uses wind only
)

SPEC_FULL = ClassificationSpec(
    name="full",
    use_storage=True,
    use_demand=True,
    add_offshore=True,
    offshore_use_solar=True,
)


# ---------------------------------------------------------------------------
# 3. Color helpers + LAND_COLORS + GROUPS_* dicts
# ---------------------------------------------------------------------------


def _shade(color, sat_mult=0.5, val_mult=0.7):
    r, g, b = mcolors.to_rgb(color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return colorsys.hsv_to_rgb(
        h, max(0, min(1, s * sat_mult)), max(0, min(1, v * val_mult))
    )


def dirty(color):  # kept for backwards compatibility
    return _shade(color, 0.5, 0.7)


def light(color):  # kept for backwards compatibility
    return _shade(color, 0.4, 1.1)


LAND_COLORS = {
    "A": "#3cb44b",  # abundant both
    "W": "#0099FF",  # wind dominant
    "Ws": "#00FFEE",  # wind favourable
    "S": "#ff8819",  # solar dominant
    "Sw": "#fff319",  # solar favourable
    "P": "#BFBFBF",  # poor both
    "O": "#4a53ff",  # offshore
}


# --- abundance (2-char labels: solar + wind) -------------------------------

# Convention: wind(HML) first, solar(HML) second.
# e.g. "HL" = high wind + low solar  (Wind dominant)
#      "LH" = low wind  + high solar  (Solar dominant)
GROUPS_ABUNDANCE: dict = {
    "A": [
        LAND_COLORS["A"],
        "- Abundance both",
        ["HH", "MM", "offshore_HH", "offshore_MM"],
    ],
    "W": [
        LAND_COLORS["W"],
        "- Wind dominant",
        ["HL", "ML", "offshore_HL", "offshore_ML"],
    ],
    "Ws": [LAND_COLORS["Ws"], "- Wind favourable", ["HM", "offshore_HM"]],
    "S": [
        LAND_COLORS["S"],
        "- Solar dominant",
        ["LH", "LM", "offshore_LH", "offshore_LM"],
    ],
    "Sw": [LAND_COLORS["Sw"], "- Solar favourable", ["MH", "offshore_MH"]],
    "P": [LAND_COLORS["P"], "- Poor both", ["LL", "offshore_LL"]],
}

# Abundance land colours + GROUPS_DETAILED offshore colours (wind only, no
# storage/demand).  Offshore labels are just "offshore_H/M/L".
GROUPS_ABUNDANCE_OFFSHORE_WIND: dict = {
    # Land — same as GROUPS_ABUNDANCE (wind+solar 2-char labels)
    "A": [LAND_COLORS["A"], "- Abundance both", ["HH", "MM"]],
    "W": [LAND_COLORS["W"], "- Wind dominant", ["HL", "ML"]],
    "Ws": [LAND_COLORS["Ws"], "- Wind favourable", ["HM"]],
    "S": [LAND_COLORS["S"], "- Solar dominant", ["LH", "LM"]],
    "Sw": [LAND_COLORS["Sw"], "- Solar favourable", ["MH"]],
    "P": [LAND_COLORS["P"], "- Poor both", ["LL"]],
    # Offshore — colours from GROUPS_DETAILED (no solar char, wind only)
    "O": ["#4a53ff", "- Offshore high/mid wind", ["offshore_H", "offshore_M"]],
    "o": ["#CECDCD", "- Offshore low wind", ["offshore_L"]],
}


# --- detailed (offshore wind only) -----------------------------------------
# Land label:    wind(HML) + solar(HML) + storage(RU) + demand(hl)
#   e.g. "HLRh" = high wind, low solar, reliable, high demand  → Wind zone
#        "LHRh" = low wind,  high solar, reliable, high demand  → Solar zone
# Offshore label: wind(HML) + storage(RU) + demand(hl)  [no solar char]
#   e.g. "HRh"  = high wind, reliable, high demand

GROUPS_DETAILED: dict = {
    "A": ["#3cb44b", "- Abundance both", ["HHRh", "MMRh"]],
    "A_L": [light("#3cb44b"), "", ["HHRl", "MMRl"]],
    "A_V": ["#688818", "", ["HHVh", "MMVh"]],
    "A_VL": [light("#688818"), "", ["HHVl", "MMVl"]],
    # Wind dominant: high/mid wind + low solar
    "W": ["#0099FF", "- Wind", ["HLRh", "MLRh"]],
    "W_L": [light("#0099FF"), "", ["HLRl", "MLRl"]],
    "W_V": ["#2e86a8", "", ["HLVh", "MLVh"]],
    "W_VL": [light("#2e86a8"), "", ["HLVl", "MLVl"]],
    # Wind favourable: high wind + medium solar
    "Ws": ["#00FFEE", "- Wind solar", ["HMRh"]],
    "Ws_L": [light("#00FFEE"), "", ["HMRl"]],
    "Ws_V": ["#22B8AE", "", ["HMVh"]],
    "Ws_VL": [light("#22B8AE"), "", ["HMVl"]],
    # Solar dominant: low wind + high/mid solar
    "S": ["#ff8819", "- Solar", ["LHRh", "LMRh"]],
    "S_L": [light("#ff8819"), "", ["LHRl", "LMRl"]],
    "S_V": ["#b25c0c", "", ["LHVh", "LMVh"]],
    "S_VL": [light("#b25c0c"), "", ["LHVl", "LMVl"]],
    # Solar favourable: medium wind + high solar
    "Sw": ["#fff319", "- Solar wind", ["MHRh"]],
    "Sw_L": [light("#fff319"), "", ["MHRl"]],
    "Sw_V": ["#b2aa13", "", ["MHVh"]],
    "Sw_VL": [light("#b2aa13"), "", ["MHVl"]],
    "P": ["#FF0000", "- Poor", ["LLxh"]],
    "P_L": ["#BFBFBF", "", ["LLxl"]],
    # Offshore wind — H and M share colour; H=3, M=2 in resource calc
    # Offshore label: wind(HML) + storage(RV) + demand(hl)  [already wind-first, unchanged]
    "o": ["#EA6565", "", ["offshore_LRh", "offshore_LVh"]],
    "o_L": ["#CECDCD", "", ["offshore_LRl", "offshore_LVl"]],
    "O": ["#4a53ff", "- Offshore", ["offshore_HRh", "offshore_MRh"]],
    "O_L": ["#a6a3fe", "", ["offshore_HRl", "offshore_MRl"]],
    "O_V": ["#8f13fb", "", ["offshore_HVh", "offshore_MVh"]],
    "O_VL": ["#d3a1ff", "", ["offshore_HVl", "offshore_MVl"]],
}


# --- full (offshore uses wind + solar, 4-char labels everywhere) -----------
# Land label:    wind(HML) + solar(HML) + storage(RV) + demand(hl)
# Offshore label: wind(HML) + solar(HML) + storage(RV) + demand(hl)
# e.g. "HLRh" = high wind, low solar, reliable, high demand  → Wind zone
#      "LHRh" = low wind,  high solar, reliable, high demand  → Solar zone

GROUPS_FULL: dict = {
    "A": [
        LAND_COLORS["A"],
        "- Abundance both",
        ["HHRh", "MMRh", "offshore_HHRh", "offshore_MMRh"],
    ],
    "A_L": [
        light(LAND_COLORS["A"]),
        "low demand",
        ["HHRl", "MMRl", "offshore_HHRl", "offshore_MMRl"],
    ],
    "A_V": [
        dirty(LAND_COLORS["A"]),
        "variable",
        ["HHVh", "MMVh", "offshore_HHVh", "offshore_MMVh"],
    ],
    "A_VL": [
        light(dirty(LAND_COLORS["A"])),
        "",
        ["HHVl", "MMVl", "offshore_HHVl", "offshore_MMVl"],
    ],
    # Wind dominant: high/mid wind + low solar
    "W": [
        LAND_COLORS["W"],
        "- Wind",
        ["HLRh", "MLRh", "offshore_HLRh", "offshore_MLRh"],
    ],
    "W_L": [
        light(LAND_COLORS["W"]),
        "",
        ["HLRl", "MLRl", "offshore_HLRl", "offshore_MLRl"],
    ],
    "W_V": [
        dirty(LAND_COLORS["W"]),
        "",
        ["HLVh", "MLVh", "offshore_HLVh", "offshore_MLVh"],
    ],
    "W_VL": [
        light(dirty(LAND_COLORS["W"])),
        "",
        ["HLVl", "MLVl", "offshore_HLVl", "offshore_MLVl"],
    ],
    # Wind favourable: high wind + medium solar
    "Ws": [LAND_COLORS["Ws"], "- Wind solar", ["HMRh", "offshore_HMRh"]],
    "Ws_L": [light(LAND_COLORS["Ws"]), "", ["HMRl", "offshore_HMRl"]],
    "Ws_V": [dirty(LAND_COLORS["Ws"]), "", ["HMVh", "offshore_HMVh"]],
    "Ws_VL": [light(dirty(LAND_COLORS["Ws"])), "", ["HMVl", "offshore_HMVl"]],
    # Solar dominant: low wind + high/mid solar
    "S": [
        LAND_COLORS["S"],
        "- Solar",
        ["LHRh", "LMRh", "offshore_LHRh", "offshore_LMRh"],
    ],
    "S_L": [
        light(LAND_COLORS["S"]),
        "",
        ["LHRl", "LMRl", "offshore_LHRl", "offshore_LMRl"],
    ],
    "S_V": [
        dirty(LAND_COLORS["S"]),
        "",
        ["LHVh", "LMVh", "offshore_LHVh", "offshore_LMVh"],
    ],
    "S_VL": [
        light(dirty(LAND_COLORS["S"])),
        "",
        ["LHVl", "LMVl", "offshore_LHVl", "offshore_LMVl"],
    ],
    # Solar favourable: medium wind + high solar
    "Sw": [LAND_COLORS["Sw"], "- Solar wind", ["MHRh", "offshore_MHRh"]],
    "Sw_L": [light(LAND_COLORS["Sw"]), "", ["MHRl", "offshore_MHRl"]],
    "Sw_V": [dirty(LAND_COLORS["Sw"]), "", ["MHVh", "offshore_MHVh"]],
    "Sw_VL": [light(dirty(LAND_COLORS["Sw"])), "", ["MHVl", "offshore_MHVl"]],
    "P": ["#FF0000", "- Poor", ["LLRh", "offshore_LLRh", "LLVh", "offshore_LLVh"]],
    "P_L": [LAND_COLORS["P"], "", ["LLRl", "offshore_LLRl", "LLVl", "offshore_LLVl"]],
}


# ---------------------------------------------------------------------------
# 4. Continuous-map helper
# ---------------------------------------------------------------------------


def plot_map_continuous(
    plot_data,
    legend_label: str,
    *,
    cmap="viridis",
    title: Optional[str] = None,
    log_norm: bool = False,
    vmin=None,
    vmax=None,
    levels=None,
    extend: str = "neither",
    path_output: Optional[str] = None,
    extent: Sequence[float] = (-180, 180, -60, 80),
    figsize: tuple = (12, 8),
):
    """Plot a continuous field on a Robinson projection."""
    map_proj = ccrs.Robinson()
    data_proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=map_proj)
    ax.set_extent(list(extent), crs=data_proj)

    norm = LogNorm(vmin=vmin, vmax=vmax) if log_norm else None
    im = plot_data.plot(
        ax=ax,
        transform=data_proj,
        norm=norm,
        vmin=None if log_norm else vmin,
        vmax=None if log_norm else vmax,
        cmap=cmap,
        levels=levels,
        add_colorbar=False,
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5)
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=0)

    gl = ax.gridlines(draw_labels=True, alpha=0.2)
    gl.top_labels = gl.right_labels = False

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.08,
        aspect=40,
        shrink=0.7,
        extend=extend,
    )
    cbar.set_label(legend_label, fontsize=11)

    if title:
        ax.set_title(title, fontsize=14, pad=15)

    if path_output:
        fig.savefig(path_output, dpi=300, bbox_inches="tight")
        log.info("Saved: %s", path_output)

    return fig, ax

###
## HELPER TO PLOT STORAGE — aggregate_lds
###


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


# ---------------------------------------------------------------------------
# 5. Logging helpers
# ---------------------------------------------------------------------------


def log_df_summary(
    df: pd.DataFrame,
    name: str,
    *,
    logger: Optional[logging.Logger] = None,
    numeric_cols: Optional[Sequence[str]] = None,
    head: int = 0,
) -> None:
    """Emit a compact log summary of a pandas DataFrame.

    Logs:
        * shape & column names
        * descriptive stats (count / mean / std / min / max) for numeric cols
        * NaN counts per column (only columns with NaNs)
        * optional head(n) preview
    """
    logger = logger or log
    logger.info("DataFrame summary — %s", name)
    logger.info("  shape=%s  columns=%s", df.shape, list(df.columns))

    num = df.select_dtypes(include=[np.number])
    if numeric_cols is not None:
        num = num[[c for c in numeric_cols if c in num.columns]]
    if not num.empty:
        desc = num.describe().loc[["count", "mean", "std", "min", "max"]]
        for col in num.columns:
            s = desc[col]
            logger.info(
                "  %s: count=%d mean=%.4g std=%.4g min=%.4g max=%.4g",
                col,
                int(s["count"]),
                s["mean"],
                s["std"],
                s["min"],
                s["max"],
            )

    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if not nan_counts.empty:
        logger.info("  NaN counts: %s", nan_counts.to_dict())

    if head:
        logger.debug("  head(%d):\n%s", head, df.head(head).to_string())


def log_ds_summary(
    ds,
    name: str,
    *,
    logger: Optional[logging.Logger] = None,
    variables: Optional[Sequence[str]] = None,
) -> None:
    """Emit a compact log summary of an xarray Dataset / DataArray.

    Reports min / mean / max / %-non-NaN for each data variable (or the
    subset requested via *variables*).
    """
    logger = logger or log
    if isinstance(ds, xr.DataArray):
        data_vars = {ds.name or "data": ds}
    else:
        data_vars = dict(ds.data_vars)
    if variables is not None:
        data_vars = {v: data_vars[v] for v in variables if v in data_vars}

    dims = dict(ds.sizes) if hasattr(ds, "sizes") else dict(ds.dims)
    logger.info("Dataset summary — %s", name)
    logger.info("  dims=%s  vars=%s", dims, list(data_vars.keys()))

    for var_name, da in data_vars.items():
        try:
            arr = da.values
        except Exception:
            continue
        if arr.dtype.kind in "OU":
            # string / categorical variable — just report unique count
            flat = arr.ravel()
            flat = flat[flat != None]  # noqa: E711
            uniq = np.unique(flat)
            logger.info("  %s: dtype=%s unique=%d", var_name, arr.dtype, len(uniq))
            continue
        finite = np.isfinite(arr)
        n_finite = int(finite.sum())
        n_total = arr.size
        if n_finite == 0:
            logger.info("  %s: all NaN (n=%d)", var_name, n_total)
            continue
        vals = arr[finite]
        logger.info(
            "  %s: min=%.4g mean=%.4g max=%.4g finite=%d/%d (%.1f%%)",
            var_name,
            float(vals.min()),
            float(vals.mean()),
            float(vals.max()),
            n_finite,
            n_total,
            100 * n_finite / n_total,
        )
