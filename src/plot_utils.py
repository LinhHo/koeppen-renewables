"""
plot_utils.py
=============

Utilities for classifying grid cells into renewable-resource "zones" and for
producing the manuscript figures in Ho (2026).

The module is organised into four layers:

1. **Masks** (`mask_land`, `mask_offshore`) — reusable geographic masks.
2. **Classification** (`classify_zones`) — a single generalised classifier
   driven by a `ClassificationSpec`. Three preset specs reproduce the
   "abundance", "detailed" (offshore-wind only) and "full" (offshore wind +
   solar + storage + demand) variants used in the notebook.
3. **Group colour schemes** (`GROUPS_ABUNDANCE`, `GROUPS_DETAILED`,
   `GROUPS_FULL`) — dictionaries mapping a group code to
   `[colour, legend_label, [zone_patterns]]`. Callers can pass their own.
4. **Plotting / analysis** — `plot_map_continuous`, `plot_zones_map`,
   `plot_stat_group_climate_cramersv`, clustering helpers, etc.

Most functions take a `groups` argument so that the same plotting code works
for any of the three zone schemes.
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
# 2. Classification
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
        If True, add a reliability character (R/U) derived from storage duration.
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


def _tertile_masks(values: xr.DataArray, low: float, high: float):
    """Return (low, mid, high) boolean masks for a tertile split."""
    low_m = (values < low) | values.isnull()
    mid_m = (values >= low) & (values < high)
    high_m = values >= high
    return low_m, mid_m, high_m


def _require(threshold_cf: dict, key: str, subkey: str, spec_name: str) -> float:
    """Read a required threshold value; raise a clear error if missing."""
    try:
        return float(threshold_cf[key][subkey])
    except (KeyError, TypeError):
        raise ValueError(
            f"[{spec_name}] threshold_cf['{key}']['{subkey}'] is required but missing."
        )


def classify_zones(
    ds: xr.Dataset,
    spec: ClassificationSpec,
    *,
    ds_storage: Optional[xr.DataArray] = None,
    ds_demand: Optional[xr.DataArray] = None,
    threshold_cf: Optional[dict] = None,
    land: Optional[xr.DataArray] = None,
    offshore: Optional[xr.DataArray] = None,
    verbose: bool = True,
) -> xr.DataArray:
    """Classify every grid cell into a zone label described by *spec*.

    All resource / storage thresholds must be provided via ``threshold_cf``
    (no quantile fallbacks).  Only the demand split still uses a quantile
    defined in ``spec.demand_quantile``.

    Required ``threshold_cf`` keys
    --------------------------------
    - ``"solar"``         : ``{"low": float, "high": float}``   [CF]
    - ``"wind_onshore"``  : ``{"low": float, "high": float}``   [CF]
    - ``"wind_offshore"`` : ``{"low": float, "high": float}``   [m/s climatology]
      (required when ``spec.add_offshore`` is True)
    - ``"solar_offshore"``: ``{"low": float, "high": float}``   [W m⁻²]
      (required when ``spec.add_offshore`` and ``spec.offshore_use_solar``)
    - ``"storage"``       : ``{"land": float, "offshore": float}``  [days]
      (required when ``spec.use_storage`` is True)

    Label structure (characters appended left-to-right)
    ---------------------------------------------------
    - solar abundance (``H`` / ``M`` / ``L``)     if ``use_solar_land``
    - wind  abundance (``H`` / ``M`` / ``L``)     if ``use_wind_land``
    - reliability     (``R`` / ``U``)             if ``use_storage``
    - demand          (``h`` / ``l``)             if ``use_demand``

    Offshore labels are prefixed with ``"offshore_"``.

    Returns
    -------
    xarray.DataArray
        2-D DataArray of string labels (``""`` for unclassified pixels).
    """
    if spec.use_storage and ds_storage is None:
        raise ValueError(f"spec {spec.name!r} requires ds_storage")
    if spec.use_demand and ds_demand is None:
        raise ValueError(f"spec {spec.name!r} requires ds_demand")

    if threshold_cf is None:
        threshold_cf = {}

    log_fn = log.info if verbose else log.debug

    wind_cf = ds["wind_CF"]
    if land is None:
        land = mask_land(wind_cf)
    if offshore is None and spec.add_offshore:
        offshore = mask_offshore(wind_cf)

    # -- land abundance ------------------------------------------------------
    solar_cf = ds["solar_CF"].where(land).compute()
    onshore_cf = ds["wind_CF"].where(land).compute()

    sol_lo = _require(threshold_cf, "solar", "low", spec.name)
    sol_hi = _require(threshold_cf, "solar", "high", spec.name)
    win_lo = _require(threshold_cf, "wind_onshore", "low", spec.name)
    win_hi = _require(threshold_cf, "wind_onshore", "high", spec.name)
    log_fn(
        f"[{spec.name}] land thresholds — solar ({sol_lo:.2f}/{sol_hi:.2f} CF), "
        f"wind ({win_lo:.2f}/{win_hi:.2f} CF)"
    )

    solar_low, solar_mid, solar_high = _tertile_masks(solar_cf, sol_lo, sol_hi)
    wind_low, wind_mid, wind_high = _tertile_masks(onshore_cf, win_lo, win_hi)

    # Label order: wind first, solar second (then storage, then demand)
    land_axes: list[list[tuple]] = []
    if spec.use_wind_land:
        land_axes.append([(wind_high, "H"), (wind_mid, "M"), (wind_low, "L")])
    if spec.use_solar_land:
        land_axes.append([(solar_high, "H"), (solar_mid, "M"), (solar_low, "L")])

    # -- land reliability (storage) ------------------------------------------
    if spec.use_storage:
        land_storage = ds_storage.where(land).compute()
        th_land = _require(threshold_cf, "storage", "land", spec.name)
        log_fn(f"[{spec.name}] land storage threshold = {th_land:.1f} days")
        land_axes.append(
            [(land_storage < th_land, "R"), (land_storage >= th_land, "U")]
        )

    # -- demand (still quantile-based) ----------------------------------------
    demand_high = demand_low = None
    if spec.use_demand:
        demand_thresh = float(ds_demand.quantile(spec.demand_quantile).values)
        demand_high = ds_demand >= demand_thresh
        demand_low = ~demand_high
        log_fn(
            f"[{spec.name}] demand threshold (q={spec.demand_quantile}) = "
            f"{demand_thresh:.3f}"
        )
        land_axes.append([(demand_high, "h"), (demand_low, "l")])

    # -- allocate output -----------------------------------------------------
    zones = np.full(solar_cf.shape, "", dtype=object)

    def _paint(axes, base_mask, prefix=""):
        for combo in product(*axes):
            masks, chars = zip(*combo)
            label = prefix + "".join(chars)
            m = base_mask
            for sub in masks:
                m = m & sub
            zones[m.values] = label

    _paint(land_axes, land)

    # -- offshore ------------------------------------------------------------
    if spec.add_offshore:
        off_wind = ds["wind_climatology"].where(offshore).compute()
        ow_lo = _require(threshold_cf, "wind_offshore", "low", spec.name)
        ow_hi = _require(threshold_cf, "wind_offshore", "high", spec.name)
        log_fn(f"[{spec.name}] offshore wind threshold = {ow_lo:.1f}/{ow_hi:.1f} m/s")
        ow_low, ow_mid, ow_high = _tertile_masks(off_wind, ow_lo, ow_hi)

        # Offshore label order: wind first, solar second (if applicable)
        off_axes: list[list[tuple]] = []
        off_axes.append([(ow_high, "H"), (ow_mid, "M"), (ow_low, "L")])

        # ssrd in ERA5 is provided in J/m2, convert to W/m2 by dividing by the number of seconds in hour (60*60)
        if spec.offshore_use_solar:
            off_solar = ds["solar_climatology"].where(offshore).compute() / (60 * 60)
            os_lo = _require(threshold_cf, "solar_offshore", "low", spec.name)
            os_hi = _require(threshold_cf, "solar_offshore", "high", spec.name)
            log_fn(
                f"[{spec.name}] offshore solar threshold = {os_lo:.0f}/{os_hi:.0f} W/m²"
            )
            os_low, os_mid, os_high = _tertile_masks(off_solar, os_lo, os_hi)
            off_axes.append([(os_high, "H"), (os_mid, "M"), (os_low, "L")])

        if spec.use_storage:
            off_storage = ds_storage.where(offshore).compute()
            th_off = _require(threshold_cf, "storage", "offshore", spec.name)
            log_fn(f"[{spec.name}] offshore storage threshold = {th_off:.1f} days")
            off_axes.append([(off_storage < th_off, "R"), (off_storage >= th_off, "U")])

        if spec.use_demand:
            off_axes.append([(demand_high, "h"), (demand_low, "l")])

        _paint(off_axes, offshore, prefix="offshore_")

    return xr.DataArray(zones, dims=solar_cf.dims, coords=solar_cf.coords, name="zones")


# ---------------------------------------------------------------------------
# 3. Group colour schemes
# ---------------------------------------------------------------------------

LAND_COLORS = {
    "B": "#3cb44b",  # both abundant
    "W": "#0099FF",  # wind dominant
    "Ws": "#00FFEE",  # wind favourable
    "S": "#ff8819",  # solar dominant
    "Sw": "#fff319",  # solar favourable
    "P": "#BFBFBF",  # poor both
    "O": "#4a53ff",  # offshore
}


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


# --- abundance (2-char labels: solar + wind) -------------------------------

# Convention: wind(HML) first, solar(HML) second.
# e.g. "HL" = high wind + low solar  (Wind dominant)
#      "LH" = low wind  + high solar  (Solar dominant)
GROUPS_ABUNDANCE: dict = {
    "B": [
        LAND_COLORS["B"],
        "- Both abundance",
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
    "B": [LAND_COLORS["B"], "- Both abundance", ["HH", "MM"]],
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
    "B": ["#3cb44b", "- Both", ["HHRh", "MMRh"]],
    "B_l": [light("#3cb44b"), "", ["HHRl", "MMRl"]],
    "B_v": ["#688818", "", ["HHUh", "MMUh"]],
    "B_vl": [light("#688818"), "", ["HHUl", "MMUl"]],
    # Wind dominant: high/mid wind + low solar
    "W": ["#0099FF", "- Wind", ["HLRh", "MLRh"]],
    "W_l": [light("#0099FF"), "", ["HLRl", "MLRl"]],
    "W_v": ["#2e86a8", "", ["HLUh", "MLUh"]],
    "W_vl": [light("#2e86a8"), "", ["HLUl", "MLUl"]],
    # Wind favourable: high wind + medium solar
    "Ws": ["#00FFEE", "- Wind solar", ["HMRh"]],
    "Ws_l": [light("#00FFEE"), "", ["HMRl"]],
    "Ws_v": ["#22B8AE", "", ["HMUh"]],
    "Ws_vl": [light("#22B8AE"), "", ["HMUl"]],
    # Solar dominant: low wind + high/mid solar
    "S": ["#ff8819", "- Solar", ["LHRh", "LMRh"]],
    "S_l": [light("#ff8819"), "", ["LHRl", "LMRl"]],
    "S_v": ["#b25c0c", "", ["LHUh", "LMUh"]],
    "S_vl": [light("#b25c0c"), "", ["LHUl", "LMUl"]],
    # Solar favourable: medium wind + high solar
    "Sw": ["#fff319", "- Solar wind", ["MHRh"]],
    "Sw_l": [light("#fff319"), "", ["MHRl"]],
    "Sw_v": ["#b2aa13", "", ["MHUh"]],
    "Sw_vl": [light("#b2aa13"), "", ["MHUl"]],
    "P": ["#FF0000", "- Poor", ["LLxh"]],
    "P_l": ["#BFBFBF", "", ["LLxl"]],
    # Offshore wind — H and M share colour; H=3, M=2 in resource calc
    # Offshore label: wind(HML) + storage(RU) + demand(hl)  [already wind-first, unchanged]
    "o": ["#EA6565", "", ["offshore_LRh", "offshore_LUh"]],
    "o_l": ["#CECDCD", "", ["offshore_LRl", "offshore_LUl"]],
    "O": ["#4a53ff", "- Offshore", ["offshore_HRh", "offshore_MRh"]],
    "O_l": ["#a6a3fe", "", ["offshore_HRl", "offshore_MRl"]],
    "O_v": ["#8f13fb", "", ["offshore_HUh", "offshore_MUh"]],
    "O_vl": ["#d3a1ff", "", ["offshore_HUl", "offshore_MUl"]],
}


# --- full (offshore uses wind + solar, 4-char labels everywhere) -----------
# Land label:    wind(HML) + solar(HML) + storage(RU) + demand(hl)
# Offshore label: wind(HML) + solar(HML) + storage(RU) + demand(hl)
# e.g. "HLRh" = high wind, low solar, reliable, high demand  → Wind zone
#      "LHRh" = low wind,  high solar, reliable, high demand  → Solar zone

GROUPS_FULL: dict = {
    "B": [
        LAND_COLORS["B"],
        "- Both",
        ["HHRh", "MMRh", "offshore_HHRh", "offshore_MMRh"],
    ],
    "B_l": [
        light(LAND_COLORS["B"]),
        "low demand",
        ["HHRl", "MMRl", "offshore_HHRl", "offshore_MMRl"],
    ],
    "B_v": [
        dirty(LAND_COLORS["B"]),
        "variable",
        ["HHUh", "MMUh", "offshore_HHUh", "offshore_MMUh"],
    ],
    "B_vl": [
        light(dirty(LAND_COLORS["B"])),
        "",
        ["HHUl", "MMUl", "offshore_HHUl", "offshore_MMUl"],
    ],
    # Wind dominant: high/mid wind + low solar
    "W": [
        LAND_COLORS["W"],
        "- Wind",
        ["HLRh", "MLRh", "offshore_HLRh", "offshore_MLRh"],
    ],
    "W_l": [
        light(LAND_COLORS["W"]),
        "",
        ["HLRl", "MLRl", "offshore_HLRl", "offshore_MLRl"],
    ],
    "W_v": [
        dirty(LAND_COLORS["W"]),
        "",
        ["HLUh", "MLUh", "offshore_HLUh", "offshore_MLUh"],
    ],
    "W_vl": [
        light(dirty(LAND_COLORS["W"])),
        "",
        ["HLUl", "MLUl", "offshore_HLUl", "offshore_MLUl"],
    ],
    # Wind favourable: high wind + medium solar
    "Ws": [LAND_COLORS["Ws"], "- Wind solar", ["HMRh", "offshore_HMRh"]],
    "Ws_l": [light(LAND_COLORS["Ws"]), "", ["HMRl", "offshore_HMRl"]],
    "Ws_v": [dirty(LAND_COLORS["Ws"]), "", ["HMUh", "offshore_HMUh"]],
    "Ws_vl": [light(dirty(LAND_COLORS["Ws"])), "", ["HMUl", "offshore_HMUl"]],
    # Solar dominant: low wind + high/mid solar
    "S": [
        LAND_COLORS["S"],
        "- Solar",
        ["LHRh", "LMRh", "offshore_LHRh", "offshore_LMRh"],
    ],
    "S_l": [
        light(LAND_COLORS["S"]),
        "",
        ["LHRl", "LMRl", "offshore_LHRl", "offshore_LMRl"],
    ],
    "S_v": [
        dirty(LAND_COLORS["S"]),
        "",
        ["LHUh", "LMUh", "offshore_LHUh", "offshore_LMUh"],
    ],
    "S_vl": [
        light(dirty(LAND_COLORS["S"])),
        "",
        ["LHUl", "LMUl", "offshore_LHUl", "offshore_LMUl"],
    ],
    # Solar favourable: medium wind + high solar
    "Sw": [LAND_COLORS["Sw"], "- Solar wind", ["MHRh", "offshore_MHRh"]],
    "Sw_l": [light(LAND_COLORS["Sw"]), "", ["MHRl", "offshore_MHRl"]],
    "Sw_v": [dirty(LAND_COLORS["Sw"]), "", ["MHUh", "offshore_HMUh"]],
    "Sw_vl": [light(dirty(LAND_COLORS["Sw"])), "", ["MHUl", "offshore_MHUl"]],
    "P": ["#FF0000", "- Poor", ["LLRh", "offshore_LLRh", "LLUh", "offshore_LLUh"]],
    "P_l": [LAND_COLORS["P"], "", ["LLRl", "offshore_LLRl", "LLUl", "offshore_LLUl"]],
}


def land_only_groups(groups: dict) -> dict:
    """Return subset of *groups* with only land (non-offshore/O) entries."""
    return {
        k: v for k, v in groups.items() if not (k.startswith("o") or k.startswith("O"))
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


# ---------------------------------------------------------------------------
# 5. Zone map
# ---------------------------------------------------------------------------


def _pattern_matches(zone_label: str, pattern: str) -> bool:
    """Return True if *zone_label* matches *pattern* (``x`` is wildcard)."""
    if pattern.startswith("offshore_"):
        if not zone_label.startswith("offshore_"):
            return False
        zp = zone_label[len("offshore_") :]
        pp = pattern[len("offshore_") :]
    else:
        if zone_label.startswith("offshore_"):
            return False
        zp, pp = zone_label, pattern

    if len(zp) != len(pp):
        return False
    return all(pc == "x" or zc == pc for zc, pc in zip(zp, pp))


def plot_zones_map(
    ds: xr.Dataset,
    groups: dict,
    *,
    out_path: Optional[str] = None,
    figsize: tuple = (15, 9),
    title: Optional[str] = None,
    extent: Sequence[float] = (-180, 180, -60, 80),
    plot_legend: bool = True,
    legend_anchor: tuple = (0.5, -0.25),
    legend_ncol: int = 7,
):
    """Render a classified ``zones`` DataArray using *groups* colour rules."""
    default_extent = (-180, 180, -60, 80)
    if tuple(extent) != default_extent:
        lon_min, lon_max, lat_min, lat_max = extent
        zones = (
            ds["zones"]
            .sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
            .values
        )
        projection = ccrs.PlateCarree()
    else:
        zones = ds["zones"].values
        projection = ccrs.Robinson()

    # Build label→colour lookup so we walk the array once.
    label_to_colour: dict[str, tuple] = {}
    legend_items: list[tuple] = []
    for code, (colour, label, patterns) in groups.items():
        rgb = mcolors.to_rgb(colour)
        legend_items.append((rgb, f"{code} {label}".strip()))
        for p in patterns:
            # eager resolve every label already seen in data
            pass  # we match lazily below for speed on small palettes

    rgb = np.full((*zones.shape, 3), np.nan, dtype=float)
    unique_labels = set(np.unique(zones)) - {""}
    matched: set = set()

    # Precompute group membership for each unique label, then fill.
    label_to_colour = {}
    for lbl in unique_labels:
        for code, (colour, _lab, patterns) in groups.items():
            if any(_pattern_matches(lbl, p) for p in patterns):
                label_to_colour[lbl] = mcolors.to_rgb(colour)
                matched.add(lbl)
                break

    for lbl, col in label_to_colour.items():
        m = zones == lbl
        rgb[m] = col

    unmatched = unique_labels - matched
    if unmatched:
        warnings.warn("Unmatched zone labels: " + ", ".join(sorted(unmatched)))

    rgb = np.flipud(rgb)

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)
    ax.set_global()
    ax.set_extent(list(extent), crs=ccrs.PlateCarree())
    ax.imshow(rgb, origin="lower", extent=list(extent), transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)

    if title:
        ax.set_title(title, fontsize=14)

    if plot_legend:
        patches = [mpatches.Patch(color=c, label=l) for c, l in legend_items]
        ax.legend(
            handles=patches,
            loc="lower center",
            bbox_to_anchor=legend_anchor,
            ncol=legend_ncol,
            frameon=False,
        )

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )
    gl.top_labels = gl.right_labels = False
    gl.xlocator = plt.FixedLocator(range(-180, 181, 30))
    gl.ylocator = plt.FixedLocator(range(-90, 91, 20))

    plt.subplots_adjust(bottom=0.25)

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return fig, ax


def plot_abundance_storage_combined(
    ds_abundance: xr.Dataset,
    groups_abundance: dict,
    storage_data,
    *,
    storage_cmap="Spectral_r",
    storage_vmin: float = 0,
    storage_vmax: float = 45,
    storage_label: str = "Storage duration [days]",
    storage_title: str = "Mean storage duration (1995–2025)",
    abundance_title: str = "Abundance zones",
    extent: Sequence[float] = (-180, 180, -60, 80),
    legend_anchor: tuple = (0.5, -0.2),
    legend_ncol: int = 3,
    figsize: tuple = (15, 16),
    out_path: Optional[str] = None,
):
    """Combined two-panel figure: (a) abundance zones map, (b) storage mean map.

    Parameters
    ----------
    ds_abundance : xr.Dataset
        Zones dataset for the abundance classification.
    groups_abundance : dict
        Colour/label/pattern groups for the abundance classification.
    storage_data : xr.DataArray
        Storage duration field to display in panel (b).
    """
    map_proj = ccrs.Robinson()
    data_proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, figure=fig, hspace=0.15)

    # ── panel (a): abundance zones ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0], projection=map_proj)
    ax_a.set_global()
    ax_a.set_extent(list(extent), crs=data_proj)

    zones = ds_abundance["zones"].values
    label_to_colour: dict[str, tuple] = {}
    legend_items: list[tuple] = []
    for code, (colour, label, patterns) in groups_abundance.items():
        rgb = mcolors.to_rgb(colour)
        legend_items.append((rgb, f"{code} {label}".strip()))

    unique_labels = set(np.unique(zones)) - {""}
    matched: set = set()
    for lbl in unique_labels:
        for code, (colour, _lab, patterns) in groups_abundance.items():
            if any(_pattern_matches(lbl, p) for p in patterns):
                label_to_colour[lbl] = mcolors.to_rgb(colour)
                matched.add(lbl)
                break

    rgb_arr = np.full((*zones.shape, 3), np.nan, dtype=float)
    for lbl, col in label_to_colour.items():
        rgb_arr[zones == lbl] = col
    rgb_arr = np.flipud(rgb_arr)

    ax_a.imshow(rgb_arr, origin="lower", extent=list(extent), transform=data_proj)
    ax_a.coastlines()
    ax_a.add_feature(cfeature.BORDERS, linewidth=0.4)

    if abundance_title:
        ax_a.set_title(abundance_title, fontsize=13)

    patches = [mpatches.Patch(color=c, label=l) for c, l in legend_items]
    ax_a.legend(
        handles=patches,
        loc="lower center",
        bbox_to_anchor=legend_anchor,
        ncol=legend_ncol,
        frameon=False,
        fontsize=9,
    )

    gl_a = ax_a.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.6, linestyle="--"
    )
    gl_a.top_labels = gl_a.right_labels = False
    gl_a.xlocator = plt.FixedLocator(range(-180, 181, 30))
    gl_a.ylocator = plt.FixedLocator(range(-90, 91, 20))

    ax_a.text(
        -0.04,
        1.02,
        "a",
        transform=ax_a.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    # ── panel (b): storage duration ─────────────────────────────────────────
    ax_b = fig.add_subplot(gs[1], projection=map_proj)
    ax_b.set_extent(list(extent), crs=data_proj)

    n_colors = 9
    cmap_obj = plt.get_cmap(storage_cmap, n_colors)
    im = storage_data.plot(
        ax=ax_b,
        transform=data_proj,
        vmin=storage_vmin,
        vmax=storage_vmax,
        cmap=cmap_obj,
        add_colorbar=False,
    )

    ax_b.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax_b.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5)
    ax_b.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=0)

    gl_b = ax_b.gridlines(draw_labels=True, alpha=0.2)
    gl_b.top_labels = gl_b.right_labels = False

    cbar = fig.colorbar(
        im,
        ax=ax_b,
        orientation="horizontal",
        pad=0.08,
        aspect=40,
        shrink=0.7,
        extend="max",
    )
    cbar.set_label(storage_label, fontsize=11)

    if storage_title:
        ax_b.set_title(storage_title, fontsize=13)

    ax_b.text(
        -0.04,
        1.02,
        "b",
        transform=ax_b.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info("Saved: %s", out_path)

    return fig, (ax_a, ax_b)


# ---------------------------------------------------------------------------
# 6. Base/sub-group aggregation + Köppen overlay
# ---------------------------------------------------------------------------


def _expand_wildcards(label: str, replacements=("U", "R")) -> list[str]:
    choices = [replacements if ch == "x" else (ch,) for ch in label]
    return ["".join(p) for p in product(*choices)]


def zone_to_group(groups: dict) -> dict:
    """Flatten *groups* into a ``{zone_label: group_code}`` map."""
    mapping = {}
    for code, (_col, _lab, labels) in groups.items():
        for lbl in labels:
            for expanded in _expand_wildcards(lbl):
                mapping[expanded] = code
    return mapping


KOPPEN_CODE_TO_LABEL = {
    0: "Undefined",
    1: "A (Tropical)",
    2: "B (Dry)",
    3: "C (Temperate)",
    4: "D (Continental)",
    5: "E (Polar)",
}
RENEWABLE_ORDER = ["B", "W", "Ws", "S", "Sw", "P"]
KOPPEN_ORDER = [
    "A (Tropical)",
    "B (Dry)",
    "C (Temperate)",
    "D (Continental)",
    "E (Polar)",
    "Undefined",
]
KOPPEN_ORDER_C = KOPPEN_ORDER[:-1]
KOPPEN_SHORT = ["A\nTropical", "B\nDry", "C\nTemperate", "D\nContin.", "E\nPolar"]


def get_base_and_subgroup(
    ds_zones: xr.Dataset,
    groups: dict,
    koppen_dbf: Optional[Path] = None,
    gridcode_csv: Optional[Path] = None,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Add ``zones_base_grouped``, ``zones_subgrouped`` and ``koppen_code``
    variables to a copy of *ds_zones* and return a tidy % DataFrame."""
    import geopandas as gpd
    import rasterio.features
    from rasterio.transform import from_bounds

    label_to_main: dict[str, str] = {}
    label_to_sub: dict[str, str] = {}
    for code, (_col, _lab, labels) in groups.items():
        base = code.split("_")[0]
        for lbl in labels:
            for expanded in _expand_wildcards(lbl):
                label_to_main[expanded] = base
                label_to_sub[expanded] = code

    ds_out = ds_zones[["zones"]].copy()
    zones_np = ds_out["zones"].data
    main = np.full(zones_np.shape, None, dtype=object)
    sub = np.full(zones_np.shape, None, dtype=object)
    for k, v in label_to_main.items():
        main[zones_np == k] = v
    for k, v in label_to_sub.items():
        sub[zones_np == k] = v
    ds_out["zones_base_grouped"] = xr.DataArray(
        main, coords=ds_out["zones"].coords, dims=ds_out["zones"].dims
    )
    ds_out["zones_subgrouped"] = xr.DataArray(
        sub, coords=ds_out["zones"].coords, dims=ds_out["zones"].dims
    )

    counts = (
        ds_out[["zones_base_grouped", "zones_subgrouped"]]
        .to_dataframe()
        .dropna()
        .groupby(["zones_base_grouped", "zones_subgrouped"])
        .size()
    )
    percentage_df = (counts / counts.sum() * 100).reset_index(name="percentage")

    # Köppen overlay (optional — skipped if resources missing)
    if koppen_dbf is None:
        koppen_dbf = (
            RESOURCES_DIR / "user/koeppen-geiger-world-map/c1976_2000_0/c1976_2000.dbf"
        )
    if gridcode_csv is None:
        gridcode_csv = (
            RESOURCES_DIR
            / "user/koeppen-geiger-world-map/c1976_2000_0/koppen-gridcodes.csv"
        )

    if Path(koppen_dbf).exists() and Path(gridcode_csv).exists():
        koppen_gdf = gpd.read_file(str(koppen_dbf))
        gridcode = pd.read_csv(str(gridcode_csv)).set_index("gridcode")
        gridcode["zone"] = gridcode["koppen"].str[0]
        koppen_gdf["group"] = koppen_gdf["GRIDCODE"].map(gridcode["zone"])
        code_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
        koppen_gdf["koppen_code"] = koppen_gdf["group"].map(code_map)

        lon = ds_out["longitude"].values
        lat = ds_out["latitude"].values
        transform = from_bounds(
            lon.min(), lat.min(), lon.max(), lat.max(), len(lon), len(lat)
        )
        raster = rasterio.features.rasterize(
            ((g, c) for g, c in zip(koppen_gdf.geometry, koppen_gdf["koppen_code"])),
            out_shape=ds_out["zones"].shape,
            transform=transform,
            fill=0,
            dtype="uint8",
        )
        ds_out["koppen_code"] = xr.DataArray(
            raster, coords=ds_out["zones"].coords, dims=ds_out["zones"].dims
        )
    else:
        warnings.warn(f"Köppen resources not found at {koppen_dbf}; skipping overlay.")

    return ds_out, percentage_df


# ---------------------------------------------------------------------------
# 7. Stats panels
# ---------------------------------------------------------------------------


def _build_koppen_df(ds_grouped: xr.Dataset) -> pd.DataFrame:
    renewable = ds_grouped["zones_base_grouped"].values.ravel()
    koppen = ds_grouped["koppen_code"].values.ravel()
    df = pd.DataFrame({"renewable": renewable, "koppen": koppen})
    df["koppen"] = df["koppen"].map(KOPPEN_CODE_TO_LABEL)
    df = df.dropna(subset=["renewable", "koppen"])
    df = df[df["koppen"] != "Undefined"]
    df = df[df["renewable"].isin(RENEWABLE_ORDER)]
    return df


def plot_renewable_subgroups(ax, df_percentage, groups_land, title="Subgroups"):
    stacked = df_percentage.pivot(
        index="zones_base_grouped",
        columns="zones_subgrouped",
        values="percentage",
    ).fillna(0)
    ordered = [g for g in groups_land.keys() if g in stacked.columns]
    stacked = stacked.reindex(index=RENEWABLE_ORDER, columns=ordered)
    colours = [groups_land[c][0] for c in stacked.columns]
    stacked.plot(
        kind="bar", stacked=True, color=colours, edgecolor="none", ax=ax, legend=False
    )
    ax.set_xlabel("")
    ax.set_ylabel("Percentage of grid cells")
    ax.set_title(title, loc="left", fontsize=10)
    ax.tick_params(axis="x", rotation=0)
    return ax


def plot_renewable_by_climate(
    ax, ds_grouped, land_colors, title="Renewable zones by Köppen–Geiger climate"
):
    renewable = ds_grouped["zones_base_grouped"].values.ravel()
    koppen = ds_grouped["koppen_code"].values.ravel()
    df = pd.DataFrame({"renewable": renewable, "koppen": koppen}).dropna()
    df["koppen"] = df["koppen"].map(KOPPEN_CODE_TO_LABEL)
    df = df[df["renewable"].isin(RENEWABLE_ORDER)]

    counts = (
        df.groupby(["koppen", "renewable"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=KOPPEN_ORDER, columns=RENEWABLE_ORDER, fill_value=0)
    )
    pct = counts / counts.values.sum() * 100
    colours = [land_colors[k] for k in pct.columns]
    pct.plot(
        kind="bar", stacked=True, color=colours, edgecolor="none", ax=ax, legend=False
    )
    ax.set_xlabel("")
    ax.set_ylabel("Percentage of grid cells")
    ax.set_title(title, loc="left", fontsize=10)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    return ax


def plot_cramers_v(
    ax,
    ds_grouped,
    fig,
    *,
    cbar_pad=0.12,
    cbar_height=0.025,
    title="Climate vs. Renewable Zone",
    row_normalised: bool = False,
):
    df = _build_koppen_df(ds_grouped)
    ct = pd.crosstab(df["renewable"], df["koppen"]).reindex(
        index=RENEWABLE_ORDER, columns=KOPPEN_ORDER_C, fill_value=0
    )
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    v = np.sqrt(chi2 / (n * min(k - 1, r - 1)))

    if row_normalised:
        norm_ct = ct.div(ct.sum(axis=1), axis=0)
    else:
        norm_ct = ct.div(ct.sum(axis=0), axis=1)

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0, vmax=1)
    im = ax.imshow(norm_ct.values, cmap=cmap, norm=norm, aspect=5 / 6)

    for i in range(norm_ct.shape[0]):
        for j in range(norm_ct.shape[1]):
            val = norm_ct.values[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if val > 0.6 else "black",
                fontweight="bold",
            )

    ax.set_xticks(range(len(KOPPEN_SHORT)))
    ax.set_xticklabels(KOPPEN_SHORT, fontsize=9)
    ax.set_yticks(range(len(RENEWABLE_ORDER)))
    ax.set_yticklabels(RENEWABLE_ORDER, fontsize=9, fontweight="bold")
    ax.set_title(
        f"{title}\nCramér's $V$ = {v:.3f}, $V^2$={v**2:.3f}",
        loc="left",
        fontsize=11,
    )

    fig.canvas.draw()
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x0, pos.y0 - cbar_pad, pos.width, cbar_height])
    mcolorbar.ColorbarBase(
        cbar_ax, cmap=cmap, norm=norm, orientation="horizontal"
    ).set_label("Fraction of Zone in Climate Class", fontsize=10)

    print(f"Cramér's V={v:.4f}, chi²={chi2:.2f}, p={p:.2e}")
    return ax


def plot_stat_group_climate_cramersv(
    ds_grouped: xr.Dataset,
    df_percentage: pd.DataFrame,
    groups_land: dict,
    land_colors: dict,
    *,
    title_suffix: str = "Subgroups in main renewable zones",
    out_path: Optional[str] = None,
):
    fig = plt.figure(figsize=(9, 6))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1, 1],
        height_ratios=[1, 1],
        hspace=0.25,
        wspace=0.25,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[:, 1])

    plot_renewable_subgroups(ax_a, df_percentage, groups_land, title=title_suffix)
    plot_renewable_by_climate(ax_b, ds_grouped, land_colors)
    plot_cramers_v(ax_c, ds_grouped, fig)

    for ax, label in zip([ax_a, ax_b, ax_c], ["a", "b", "c"]):
        ax.text(
            -0.08,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
        )

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    return fig


# ---------------------------------------------------------------------------
# 8. Scatter: poor + high demand countries
# ---------------------------------------------------------------------------


def plot_scatter_elevation_precipitation(
    ds_grouped: xr.Dataset,
    named_countries: Sequence[str],
    *,
    out_path: Optional[str] = None,
    elevation_path: Optional[Path] = None,
    precip_path: Optional[Path] = None,
):
    if elevation_path is None:
        elevation_path = RESOURCES_DIR / "automatic/era5_global_geopotential_surface.nc"
    if precip_path is None:
        precip_path = (
            RESOURCES_DIR
            / "automatic/tp/era5_total_precipitation_year_average_1995_2025.nc"
        )

    geopot = xr.open_dataset(str(elevation_path)).isel(valid_time=0)["z"]
    geopot = geopot.assign_coords(longitude=((geopot.longitude + 180) % 360) - 180)
    elevation = geopot / 9.80665
    precip = xr.open_dataset(str(precip_path))["tp"]

    land = mask_land(ds_grouped["zones"])
    ds_grouped = ds_grouped.copy()
    ds_grouped["elevation"] = elevation.where(land).compute()
    ds_grouped["precipitation"] = precip.where(land).compute()

    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    maskc = countries.mask(
        ds_grouped["zones"].rename({"latitude": "lat", "longitude": "lon"})
    ).rename({"lat": "latitude", "lon": "longitude"})
    ds_grouped["country_id"] = maskc

    tmp = ds_grouped.to_dataframe().reset_index(drop=False)
    df_red = tmp[tmp["zones_subgrouped"] == "P"][
        ["longitude", "latitude", "zones", "elevation", "precipitation", "country_id"]
    ].copy()

    df_red_stats = pd.DataFrame(
        {
            "latitude": df_red.groupby("country_id")["latitude"].mean(),
            "elevation": df_red.groupby("country_id")["elevation"].mean(),
            "precipitation": df_red.groupby("country_id")["precipitation"].mean(),
            "P_count": df_red.groupby("country_id").size(),
        }
    )

    country_with_red = df_red["country_id"].unique()
    df_country_total = (
        maskc.where(maskc.isin(country_with_red))
        .to_dataframe(name="country_id")
        .dropna()
        .groupby("country_id")
        .size()
        .rename("n_grid_cells_total")
        .reset_index()
    )
    df_country_total["country_name"] = df_country_total["country_id"].map(
        dict(enumerate(countries.names))
    )

    toplot = pd.concat(
        [df_red_stats, df_country_total.set_index("country_id")],
        axis=1,
        join="inner",
    )
    toplot["percentage"] = toplot["P_count"] / toplot["n_grid_cells_total"] * 100

    elev_grid = float(np.mean(ds_grouped["elevation"]).values)
    precip_grid = float(np.mean(ds_grouped["precipitation"]).values)

    fig, ax = plt.subplots(figsize=(10, 6))
    lat_binned = np.floor(np.abs(toplot["latitude"]) / 6) * 6
    bounds = np.arange(0, 70, 10)
    bnorm = BoundaryNorm(bounds, ncolors=256)

    sc = ax.scatter(
        toplot["precipitation"],
        toplot["elevation"],
        s=toplot["percentage"] * 10,
        c=lat_binned,
        alpha=0.6,
        cmap="viridis",
        norm=bnorm,
    )
    plt.colorbar(sc, ax=ax).set_label("Absolute latitude |°|")
    ax.axhline(elev_grid)
    ax.axvline(precip_grid)
    ax.set_xlabel("Average Precipitation [m/yr]")
    ax.set_ylabel("Average Elevation [m]")
    ax.set_title("Countries with high share of 'Both poor high demand'")

    for _, row in toplot[toplot["country_name"].isin(named_countries)].iterrows():
        ax.text(
            row["precipitation"] + 0.05,
            row["elevation"],
            row["country_name"],
            fontsize=8,
        )
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    return toplot


# ---------------------------------------------------------------------------
# 9. Resource availability from zone labels
# ---------------------------------------------------------------------------


def parse_zone_to_abundance(zone: str) -> dict:
    """Parse a zone label into numeric wind and solar abundance (L=1, M=2, H=3).

    Handles two offshore label structures:
    - DETAILED (wind-only offshore): ``offshore_<Wind><Storage><Demand>`` →
      wind at position 0, solar=NaN.
    - FULL / ABUNDANCE (solar+wind offshore): ``offshore_<Solar><Wind>...`` →
      solar at position 0, wind at position 1.
    Land labels always have solar at position 0, wind at position 1.
    """
    if not isinstance(zone, str) or zone == "":
        return {"wind": np.nan, "solar": np.nan}
    amap = {"L": 1, "M": 2, "H": 3}
    is_offshore = zone.startswith("offshore_")
    clean = zone.replace("offshore_", "")
    try:
        if is_offshore and len(clean) >= 2 and clean[1].upper() in ("R", "U"):
            # DETAILED offshore: wind(HML) + storage(RU) + demand(hl)
            # → wind at position 0, no solar component
            return {
                "wind": amap.get(clean[0].upper(), np.nan),
                "solar": np.nan,
            }
        else:
            # Land or FULL/ABUNDANCE offshore: wind at [0], solar at [1]
            return {
                "wind": amap.get(clean[0].upper(), np.nan),
                "solar": amap.get(clean[1].upper(), np.nan),
            }
    except IndexError:
        return {"wind": np.nan, "solar": np.nan}


def convert_zones_to_resource_availability(
    ds_zones: xr.Dataset,
    ds_storage: xr.Dataset,
    *,
    land: Optional[xr.DataArray] = None,
    offshore: Optional[xr.DataArray] = None,
    limit_optimal_alpha: Optional[tuple] = None,
    method: Literal["optimal_alpha", "max_abundance"] = "optimal_alpha",
) -> xr.Dataset:
    """Convert zone labels into a normalised resource-availability grid."""
    zones = ds_zones["zones"].values
    optimal_alpha = ds_storage["optimal_alpha"].values
    if limit_optimal_alpha is not None:
        optimal_alpha = np.clip(optimal_alpha, *limit_optimal_alpha)

    lat_dim = "lat" if "lat" in ds_zones.coords else "latitude"
    lon_dim = "lon" if "lon" in ds_zones.coords else "longitude"

    if land is None:
        land = mask_land(ds_zones["zones"])
    if offshore is None:
        offshore = mask_offshore(ds_zones["zones"])
    land_vals = land.values if hasattr(land, "values") else land
    offshore_vals = offshore.values if hasattr(offshore, "values") else offshore

    wind = np.full(zones.shape, np.nan)
    solar = np.full(zones.shape, np.nan)
    cache: dict = {}
    for i in range(zones.shape[0]):
        for j in range(zones.shape[1]):
            z = zones[i, j]
            if z not in cache:
                cache[z] = parse_zone_to_abundance(z)
            wind[i, j] = cache[z]["wind"]
            solar[i, j] = cache[z]["solar"]

    # At least one of wind/solar must be present (offshore DETAILED has solar=NaN)
    valid = ~np.isnan(wind) | ~np.isnan(solar)
    raw = np.full(zones.shape, np.nan)
    if method == "optimal_alpha":
        # For offshore DETAILED (solar=NaN): treat missing as 0 so wind dominates
        w = np.where(np.isnan(wind), 0.0, wind)
        s = np.where(np.isnan(solar), 0.0, solar)
        raw[valid] = (
            optimal_alpha[valid] * w[valid] + (1 - optimal_alpha[valid]) * s[valid]
        )
    elif method == "max_abundance":
        # fmax ignores NaN → returns the non-NaN value for offshore (wind only)
        raw[valid] = np.fmax(wind[valid], solar[valid])
    else:
        raise ValueError(f"unknown method {method!r}")

    out = np.full(raw.shape, np.nan)
    for region_mask in (offshore_vals, land_vals):
        if not np.any(region_mask):
            continue
        vals = raw[region_mask]
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        out[region_mask] = (vals - vmin) / (vmax - vmin) if vmax > vmin else 1.0

    ds_out = xr.Dataset(
        {
            "resource_availability": ([lat_dim, lon_dim], out),
            "wind_cat": ([lat_dim, lon_dim], wind),
            "solar_cat": ([lat_dim, lon_dim], solar),
            "optimal_alpha": ([lat_dim, lon_dim], optimal_alpha),
        },
        coords={c: ds_zones[c] for c in (lat_dim, lon_dim)},
    )
    ds_out.attrs["method"] = method
    return ds_out


# ---------------------------------------------------------------------------
# 10. Country-level masking, correlation, clustering
# ---------------------------------------------------------------------------


def add_country_mask(
    ds: xr.Dataset,
    shapes_path: str,
    yaml_path: str,
    *,
    lon_name: str = "longitude",
    lat_name: str = "latitude",
) -> tuple[xr.Dataset, "object"]:
    """Rasterise country shapes into ``ds['country_maritime']``."""
    import geopandas as gpd
    import yaml
    from shapely.geometry import box

    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    country_list = list(config["module_geo_boundaries"]["countries"].keys())

    shapes = gpd.read_parquet(shapes_path)
    shapes = shapes[shapes["country_id"].isin(country_list)].copy()
    # USA mainland only
    shapes.loc[shapes["country_id"] == "USA", "geometry"] = gpd.clip(
        shapes[shapes["country_id"] == "USA"], box(-130, 20, -60, 55)
    ).geometry.values
    shapes = shapes.reset_index(drop=True)
    if shapes.crs != "EPSG:4326":
        shapes = shapes.to_crs("EPSG:4326")

    regions = regionmask.from_geopandas(
        shapes, names="country_id", abbrevs="country_id"
    )
    ds = ds.sortby(lon_name)
    ds["country_maritime"] = regions.mask(ds[lon_name], ds[lat_name], wrap_lon=False)
    return ds, shapes


def calculate_spatial_correlation(vals1, vals2, method: str = "spearman"):
    valid = ~np.isnan(vals1) & ~np.isnan(vals2)
    v1, v2 = vals1[valid], vals2[valid]
    if len(v1) < 50:
        return np.nan, np.nan
    if method == "spearman":
        return spearmanr(v1, v2)
    if method == "pearson":
        return pearsonr(v1, v2)
    raise ValueError(f"unknown method {method!r}")


def analyze_country_spatial_correlation(
    ds: xr.Dataset,
    shapes,
    cnt_list_iso3: Optional[Iterable[str]] = None,
    *,
    min_size_pixel: int = 50,
    method: str = "pearson",
    extra_groups: Optional[dict] = None,
) -> pd.DataFrame:
    """Per-country resource–demand correlation plus simple averages."""
    from tqdm import tqdm

    present_ids = np.unique(ds["country_maritime"].values)
    present_ids = present_ids[~np.isnan(present_ids)].astype(int)

    results = []
    for c_idx in tqdm(present_ids, desc="Countries"):
        iso3 = shapes.iloc[c_idx]["country_id"]
        if cnt_list_iso3 is not None and iso3 not in cnt_list_iso3:
            continue
        mask = ds["country_maritime"] == c_idx
        subset = ds[["resource", "demand", "storage"]].where(mask, drop=True)
        df = subset.to_dataframe().dropna()
        if len(df) < min_size_pixel:
            continue
        corr, p = calculate_spatial_correlation(
            df["resource"].values, df["demand"].values, method=method
        )
        results.append(
            {
                "iso3": iso3,
                "country_name": shapes.iloc[c_idx]["parent_name"],
                "resource_demand_corr": corr,
                "corr_p_value": p,
                "n_pixels": len(df),
                "avg_resource": df["resource"].mean(),
                "avg_demand": df["demand"].mean(),
                "avg_storage": df["storage"].mean(),
            }
        )

    df_results = pd.DataFrame(results)

    if extra_groups:
        for group_id, members in extra_groups.items():
            ids = shapes[shapes["country_id"].isin(members)].index.tolist()
            mask = ds["country_maritime"].isin(ids)
            subset = ds[["resource", "demand", "storage"]].where(mask, drop=True)
            df_g = subset.to_dataframe().dropna()
            if len(df_g) < min_size_pixel:
                continue
            corr, p = calculate_spatial_correlation(
                df_g["resource"].values, df_g["demand"].values, method=method
            )
            df_results = pd.concat(
                [
                    df_results,
                    pd.DataFrame(
                        [
                            {
                                "iso3": group_id,
                                "country_name": group_id,
                                "resource_demand_corr": corr,
                                "corr_p_value": p,
                                "n_pixels": len(df_g),
                                "avg_resource": df_g["resource"].mean(),
                                "avg_demand": df_g["demand"].mean(),
                                "avg_storage": df_g["storage"].mean(),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    return df_results


def prepare_metric_map(
    ds: xr.Dataset,
    df_results: pd.DataFrame,
    shapes,
    *,
    column: str = "resource_demand_corr",
) -> xr.DataArray:
    """Back-project a per-country scalar metric onto the grid."""
    mapper = np.full(len(shapes), np.nan)
    lookup = dict(zip(df_results["iso3"], df_results[column]))
    for idx, row in shapes.iterrows():
        if row["country_id"] in lookup:
            mapper[idx] = lookup[row["country_id"]]

    mask_values = ds["country_maritime"].values
    flat = mask_values.flatten()
    out = np.full(flat.shape, np.nan)
    valid = ~np.isnan(flat)
    out[valid] = mapper[flat[valid].astype(int)]
    return xr.DataArray(
        out.reshape(mask_values.shape),
        coords=ds["country_maritime"].coords,
        dims=ds["country_maritime"].dims,
        name=column,
    )


def cluster_countries(
    df: pd.DataFrame, *, n_clusters: int = 4, method: str = "k-mean"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-means (or agglomerative) clustering of countries by storage, correlation, resource."""
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.preprocessing import StandardScaler

    features = ["avg_storage", "resource_demand_corr", "avg_resource"]
    X = df[features].copy().fillna(df[features].mean())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    df = df.copy()
    if method == "k-mean":
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["cluster"] = km.fit_predict(Xs)
        centroids = scaler.inverse_transform(km.cluster_centers_)
    else:
        ac = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        df["cluster"] = ac.fit_predict(Xs)
        centroids = np.vstack(
            [X[df["cluster"] == c].mean(axis=0).values for c in range(n_clusters)]
        )

    return df, pd.DataFrame(centroids, columns=features)


def prepare_cluster_map(ds, df_clusters, shapes) -> xr.DataArray:
    return prepare_metric_map(ds, df_clusters, shapes, column="cluster")


def create_cluster_summary_table_full_names(
    df: pd.DataFrame,
    shapes_ref,
    centroids: pd.DataFrame,
    *,
    naming_rules: Optional[dict] = None,
    large_threshold: int = 400,
) -> pd.DataFrame:
    """Build a cluster summary table with full country names and descriptive labels.

    Parameters
    ----------
    df
        Per-country dataframe with ``iso3``, ``cluster``, ``n_pixels`` columns.
    shapes_ref
        Shapes GeoDataFrame with ``country_id`` and ``parent_name`` columns.
    centroids
        Cluster centroids dataframe (output of :func:`cluster_countries`).
    naming_rules
        ``{label: (column, operator, threshold)}`` — rules applied to each centroid
        row to derive a descriptive cluster name.  Operators: ``">="`` or ``"<"``.
        Multiple matching rules are joined with ``" / "``.
        Example::

            {
                "high resource": ("avg_resource", ">=", 0.7),
                "high mismatch": ("resource_demand_corr", "<", -0.3),
            }
    large_threshold
        Country is "large" when its ``n_pixels`` exceeds this value.
    """
    name_map = dict(zip(shapes_ref["country_id"], shapes_ref["parent_name"]))

    df = df.copy()
    df["full_name"] = df["iso3"].map(name_map)

    def split_countries(group):
        large = group[group["n_pixels"] > large_threshold]["full_name"]
        small = group[group["n_pixels"] <= large_threshold]["full_name"]
        return pd.Series(
            {
                "large_countries": ", ".join(sorted(large.dropna().astype(str))),
                "small_countries": ", ".join(sorted(small.dropna().astype(str))),
            }
        )

    country_rows = df.groupby("cluster").apply(split_countries)

    # Summary table: metrics as rows, clusters as columns
    summary_table = centroids[["avg_resource", "avg_storage", "resource_demand_corr"]].T
    summary_table.loc["Large countries"] = country_rows["large_countries"]
    summary_table.loc["Small countries"] = country_rows["small_countries"]

    # Generate column names from naming_rules or fall back to "Cluster N"
    def _name_cluster(idx: int) -> str:
        if not naming_rules:
            return f"Cluster {idx}"
        row = centroids.iloc[idx]
        matched = [
            name
            for name, (col, op, thr) in naming_rules.items()
            if col in row.index
            and ((op == ">=" and row[col] >= thr) or (op == "<" and row[col] < thr))
        ]
        return " / ".join(matched) if matched else f"Cluster {idx}"

    col_names = {i: _name_cluster(i) for i in range(len(centroids))}
    summary_table = summary_table.rename(columns=col_names)
    return summary_table


def plot_combined_analysis(
    df_clustered: pd.DataFrame,
    df_centroids: pd.DataFrame,
    ds_map: xr.DataArray,
    cluster_map: dict,
    *,
    list_cnt_to_plot: Optional[Sequence[str]] = None,
    path_output: Optional[str] = None,
):
    """Two-row figure: cluster scatter on top, cluster map on bottom."""
    fig = plt.figure(figsize=(10, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0)
    ax_scatter = fig.add_subplot(gs[0])
    ax_map = fig.add_subplot(gs[1], projection=ccrs.Robinson())

    palette = {lab: col for col, lab in cluster_map.values()}
    name_map = {cid: lab for cid, (_col, lab) in cluster_map.items()}
    df_plot = df_clustered.copy()
    df_plot["cluster name"] = df_plot["cluster"].map(name_map)

    sorted_ids = sorted(cluster_map.keys())
    discrete_cmap = ListedColormap([cluster_map[i][0] for i in sorted_ids])

    # ── resource-level → marker shape ──────────────────────────────────────
    _RES_CATS = [
        (0.33, "low (≤0.33)", "^"),
        (0.67, "medium (0.33–0.67)", "o"),
        (1.01, "high (>0.67)", "s"),
    ]

    def _res_cat(v):
        for thr, lbl, _ in _RES_CATS:
            if v <= thr:
                return lbl
        return _RES_CATS[-1][1]

    df_plot["Resource level"] = df_plot["avg_resource"].apply(_res_cat)
    res_marker = {lbl: mk for _, lbl, mk in _RES_CATS}
    res_order = [lbl for _, lbl, _ in _RES_CATS]

    # ── size from n_pixels (log-scaled for readability) ────────────────────
    n_px = df_plot["n_pixels"].clip(lower=1)
    log_px = np.log1p(n_px)
    size_min, size_max = 25, 500
    sizes_norm = (log_px - log_px.min()) / (log_px.max() - log_px.min() + 1e-9)
    df_plot["_sz"] = size_min + sizes_norm * (size_max - size_min)

    # ── scatter: one call per resource category to honour marker shape ──────
    for _thr, cat_label, mk in _RES_CATS:
        sub = df_plot[df_plot["Resource level"] == cat_label]
        if sub.empty:
            continue
        colours = sub["cluster name"].map(palette)
        ax_scatter.scatter(
            sub["resource_demand_corr"],
            sub["avg_storage"],
            marker=mk,
            s=sub["_sz"],
            c=colours,
            alpha=0.55,
            edgecolors="w",
            linewidths=0.5,
            zorder=3,
        )

    # ── centroid markers ────────────────────────────────────────────────────
    for cid in sorted_ids:
        colour, _lab = cluster_map[cid]
        row = df_centroids.iloc[cid]
        ax_scatter.scatter(
            row["resource_demand_corr"],
            row["avg_storage"],
            marker="X",
            s=220,
            color=colour,
            edgecolors="black",
            linewidth=1.2,
            zorder=5,
        )

    # ── country labels ──────────────────────────────────────────────────────
    to_label = (
        df_plot[df_plot["iso3"].isin(list_cnt_to_plot)]
        if list_cnt_to_plot
        else df_plot.head(10)
    )
    for _, row in to_label.iterrows():
        colour = dirty(cluster_map[row["cluster"]][0])
        ax_scatter.text(
            row["resource_demand_corr"] + 0.01,
            row["avg_storage"] + 0.01,
            row["country_name"],
            fontsize=9,
            color=colour,
            fontweight="bold",
            path_effects=[withStroke(linewidth=2, foreground="white")],
        )

    # ── custom legend ───────────────────────────────────────────────────────
    import matplotlib.lines as mlines

    # Section 1: cluster colours
    legend_handles = [
        mpatches.Patch(color=None, label="Cluster", fill=False, linewidth=0)
    ]
    for cid in sorted_ids:
        col, lab = cluster_map[cid]
        legend_handles.append(
            mpatches.Patch(facecolor=col, edgecolor="grey", label=lab, alpha=0.7)
        )

    # Section 2: resource level shapes
    legend_handles.append(
        mpatches.Patch(color=None, label="Resource level", fill=False, linewidth=0)
    )
    for _, lbl, mk in _RES_CATS:
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                marker=mk,
                color="w",
                markerfacecolor="#666666",
                markersize=9,
                label=lbl,
                linestyle="None",
            )
        )

    # Section 3: country size (representative quantiles)
    legend_handles.append(
        mpatches.Patch(
            color=None, label="Country size (pixels)", fill=False, linewidth=0
        )
    )
    for q, qlabel in [(0.1, "small"), (0.5, "medium"), (0.9, "large")]:
        qval = float(np.quantile(df_plot["_sz"], q))
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor="#999999",
                markersize=np.sqrt(qval) * 0.9,
                label=qlabel,
                linestyle="None",
            )
        )

    leg = ax_scatter.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.25,
        title="Legend",
        title_fontsize=9,
    )

    ax_scatter.set_title("a) Cluster Characteristics", loc="left", fontweight="bold")
    ax_scatter.axvline(0, color="grey", linestyle="--", alpha=0.3)
    ax_scatter.grid(True, alpha=0.2)
    ax_scatter.set_xlim([-0.9, 0.9])
    ax_scatter.set_xlabel("Resource-demand spatial correlation")
    ax_scatter.set_ylabel("Average storage duration [normalised]")

    ds_map.plot(
        ax=ax_map,
        transform=ccrs.PlateCarree(),
        cmap=discrete_cmap,
        vmin=min(sorted_ids) - 0.5,
        vmax=max(sorted_ids) + 0.5,
        add_colorbar=False,
        zorder=1,
    )
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax_map.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.4)
    ax_map.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=0)
    ax_map.set_extent([-180, 180, -60, 80], crs=ccrs.PlateCarree())
    gl = ax_map.gridlines(draw_labels=True, alpha=0.2)
    gl.top_labels = gl.right_labels = False
    # xarray's .plot() sets a centered title like "surface = 0.0 [1]" —
    # clear it explicitly before writing our own left-aligned title.
    ax_map.set_title("")
    ax_map.set_title(
        "b) Spatial Distribution of Strategies", loc="left", fontweight="bold"
    )

    if path_output:
        fig.savefig(path_output, dpi=300, bbox_inches="tight")
        print(f"Saved: {path_output}")
    return fig


# ---------------------------------------------------------------------------
# 11. ISO-3 mapping helpers
# ---------------------------------------------------------------------------

_MANUAL_ISO2_TO_ISO3 = {
    "N": "NOR",
    "F": "FRA",
    "J": "JPN",
    "S": "ZAF",
    "A": "AUS",
    "D": "DEU",
    "L": "THA",
    "B": "BRA",
    "P": "PAK",
    "E": "ESP",
    "I": "ITA",
    "NM": "MKD",
    "KO": "XKX",
    "INDO": "IDN",
    "SLO": "SVN",
    "DRC": "COD",
    "BiH": "BIH",
}


def to_iso3_robust(code: str) -> Optional[str]:
    import pycountry

    if code in _MANUAL_ISO2_TO_ISO3:
        return _MANUAL_ISO2_TO_ISO3[code]
    if len(code) == 3:
        return code
    try:
        return pycountry.countries.get(alpha_2=code).alpha_3
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 12. 3D cluster plot (manuscript Fig 6)
# ---------------------------------------------------------------------------


def plot_country_clusters_3d(
    df: pd.DataFrame,
    df_centroids: pd.DataFrame,
    cluster_map: dict,
    *,
    list_cnt_to_plot: Optional[Sequence[str]] = None,
    fig_size: tuple = (1000, 900),
    out_path: Optional[str] = None,
    camera_eye: tuple = (1.6, 1.6, 1.1),
    show: bool = False,
):
    """Interactive 3D scatter of country clusters.

    Axes
    ----
    * X — Average resource availability
    * Y — Spatial correlation (resource vs demand)
    * Z — Average storage requirement

    Parameters
    ----------
    df
        Per-country dataframe with ``avg_resource``, ``avg_storage``,
        ``resource_demand_corr``, ``cluster``, ``iso3``, ``country_name``.
    df_centroids
        Cluster centroids dataframe (output of :func:`cluster_countries`).
    cluster_map
        ``{cluster_id: [hex_colour, label]}`` — same structure used by
        :func:`plot_combined_analysis`. Drives marker colours and the legend.
    list_cnt_to_plot
        ISO-3 codes whose names should be drawn next to their marker (as in
        Fig 5a). If ``None``, no labels are drawn.
    fig_size
        ``(width, height)`` in pixels.
    out_path
        If given, save a static image using Kaleido. ``.pdf`` / ``.png`` /
        ``.svg`` are all supported. The HTML version is also written next to
        it for interactive exploration.
    camera_eye
        Initial camera position — tweak if the default view crops labels.
    show
        Whether to call ``fig.show()``. Defaults to ``False`` so the script
        can run headless; set ``True`` to display in a notebook.
    """
    import plotly.graph_objects as go

    df = df.copy()
    df_centroids = df_centroids.copy()

    cluster_ids = sorted(cluster_map.keys())

    fig = go.Figure()

    # One trace per cluster — cluster markers first, labels drawn last so
    # they are never occluded by centroid X markers in the depth sort.
    for cid in cluster_ids:
        colour, label = cluster_map[cid]
        sub = df[df["cluster"] == cid]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=sub["avg_resource"],
                y=sub["resource_demand_corr"],
                z=sub["avg_storage"],
                mode="markers",
                marker=dict(
                    size=9,
                    color=colour,
                    opacity=0.80,
                    line=dict(width=0.8, color="white"),
                ),
                name=str(label),
                text=sub["iso3"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Cluster: " + str(label) + "<br>"
                    "Resource: %{x:.3f}<br>"
                    "Correlation: %{y:.3f}<br>"
                    "Storage: %{z:.3f}<extra></extra>"
                ),
            )
        )

    # Centroids: one 'X' marker per cluster, drawn before labels.
    fig.add_trace(
        go.Scatter3d(
            x=df_centroids["avg_resource"],
            y=df_centroids["resource_demand_corr"],
            z=df_centroids["avg_storage"],
            mode="markers",
            marker=dict(
                symbol="x",
                size=12,
                color=[cluster_map[i][0] for i in range(len(df_centroids))],
                line=dict(width=4, color="black"),
            ),
            name="Cluster centroids",
            hoverinfo="skip",
        )
    )

    # Named-country labels — added LAST so they render on top of all markers.
    if list_cnt_to_plot:
        to_label = df[df["iso3"].isin(list_cnt_to_plot)]
        if not to_label.empty:
            names = to_label.get("country_name", to_label["iso3"])
            fig.add_trace(
                go.Scatter3d(
                    x=to_label["avg_resource"],
                    y=to_label["resource_demand_corr"],
                    z=to_label["avg_storage"],
                    # markers+text: tiny invisible anchor forces correct depth
                    # sort so the text is never hidden behind centroid X marks.
                    mode="markers+text",
                    marker=dict(size=1, opacity=0, color="rgba(0,0,0,0)"),
                    text=names,
                    textposition="top center",
                    textfont=dict(
                        size=12, color="black", family="Arial Black, Arial, sans-serif"
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Scene occupies the left ~82 % of the canvas; legend sits tight beside it.
    fig.update_layout(
        title=dict(
            text=(
                "<b>Global Renewable Strategy Clusters (3D view)</b><br>"
                "<sup>Storage, resource–demand correlation, and resource availability</sup>"
            ),
            x=0.04,
            y=0.96,
            yanchor="top",
            font=dict(size=15),
        ),
        width=fig_size[0],
        height=fig_size[1],
        scene=dict(
            domain=dict(x=[0.0, 0.82], y=[0.0, 1.0]),
            xaxis=dict(
                title=dict(text="Avg resource availability", font=dict(size=13)),
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                title=dict(
                    text="Spatial correlation<br>(resource vs demand)",
                    font=dict(size=13),
                ),
                tickfont=dict(size=11),
            ),
            zaxis=dict(
                title=dict(text="Avg storage [norm.]", font=dict(size=13)),
                tickfont=dict(size=11),
            ),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]),
            ),
            aspectmode="cube",
        ),
        template="plotly_white",
        legend=dict(
            yanchor="middle",
            y=0.50,
            xanchor="left",
            x=0.6,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="lightgrey",
            borderwidth=1,
            font=dict(size=12),
        ),
        margin=dict(l=10, r=10, b=10, t=55),
    )

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Static export — requires kaleido.  Check availability before
        # attempting so we emit a clear warning rather than a cryptic error,
        # and always fall back to writing the interactive HTML.
        _kaleido_ok = False
        try:
            import kaleido  # noqa: F401  (existence check only)

            _kaleido_ok = True
        except ImportError:
            log.warning(
                "kaleido is not installed — static 3D figure (%s) cannot be "
                "saved as %s.  Install kaleido (e.g. `pip install kaleido`) "
                "to enable static export.  The interactive HTML will still be "
                "written.",
                out_path.suffix.lstrip(".").upper(),
                out_path,
            )

        if _kaleido_ok:
            try:
                fig.write_image(
                    str(out_path), width=fig_size[0], height=fig_size[1], scale=2
                )
                log.info("Saved 3D cluster figure: %s", out_path)
            except Exception as exc:
                log.warning("Static 3D export failed for %s: %s", out_path, exc)

        html_path = out_path.with_suffix(".html")
        fig.write_html(str(html_path))
        log.info("Saved interactive 3D cluster figure: %s", html_path)

    if show:
        fig.show()

    return fig


# ---------------------------------------------------------------------------
# 13. Offshore-wind shift: resource–demand scatter with A→B arrows
# ---------------------------------------------------------------------------


def plot_offshore_shift_density(
    df_land: pd.DataFrame,
    df_combined: pd.DataFrame,
    cluster_info: pd.DataFrame,
    cluster_config: dict,
    *,
    figsize: tuple = (12, 5),
    out_path: Optional[str] = None,
) -> "plt.Figure":
    """KDE density of per-country metric shifts (B − A) across clusters.

    Three subplots side-by-side:
      1. Δ avg_storage (normalised)
      2. Δ resource_demand_corr
      3. Δ avg_resource

    Arrow/cluster colours match Fig 5 (``cluster_config``).
    A single shared legend is placed outside the rightmost subplot.
    Per-cluster mean shifts are printed to the log.

    Parameters
    ----------
    df_land : pd.DataFrame
        Experiment A metrics.  ``avg_storage`` must already be normalised.
    df_combined : pd.DataFrame
        Experiment B metrics.  ``avg_storage`` must already be normalised
        on the same scale.
    cluster_info : pd.DataFrame
        Must contain ``iso3`` and ``cluster`` (integer IDs).
    cluster_config : dict
        ``{cluster_id: [hex_colour, label]}`` — same as Fig 5.
    figsize : tuple
    out_path : str or Path, optional
    """
    # ── build shift dataframe ────────────────────────────────────────────────
    merged = pd.merge(
        df_land[
            [
                "iso3",
                "country_name",
                "resource_demand_corr",
                "avg_storage",
                "avg_resource",
            ]
        ],
        df_combined[["iso3", "resource_demand_corr", "avg_storage", "avg_resource"]],
        on="iso3",
        suffixes=("_a", "_b"),
    ).dropna()

    merged["storage_shift"] = merged["avg_storage_b"] - merged["avg_storage_a"]
    merged["resource_demand_corr_shift"] = (
        merged["resource_demand_corr_b"] - merged["resource_demand_corr_a"]
    )
    merged["avg_resource_shift"] = merged["avg_resource_b"] - merged["avg_resource_a"]

    merged = merged.merge(cluster_info[["iso3", "cluster"]], on="iso3", how="left")

    # Map integer IDs → human-readable labels (used as hue in seaborn).
    id_to_label = {cid: cfg[1] for cid, cfg in cluster_config.items()}
    label_to_col = {cfg[1]: cfg[0] for cfg in cluster_config.values()}
    merged["Cluster"] = merged["cluster"].map(id_to_label)

    # ── log per-cluster averages ─────────────────────────────────────────────
    shifts_meta = [
        ("storage_shift", "Δavg_storage"),
        ("resource_demand_corr_shift", "Δcorr"),
        ("avg_resource_shift", "Δavg_resource"),
    ]
    for cid in sorted(merged["cluster"].dropna().unique()):
        sub = merged[merged["cluster"] == cid]
        label = id_to_label[int(cid)]
        parts = ", ".join(f"{name}={sub[col].mean():+.3f}" for col, name in shifts_meta)
        log.info("Cluster %d (%s, n=%d): %s", int(cid), label, len(sub), parts)

    # ── plot ─────────────────────────────────────────────────────────────────
    shift_cols = [s for s, _ in shifts_meta]
    titles = [
        "Δ Avg storage [norm.]",
        "Δ Resource–demand correlation",
        "Δ Avg resource quality",
    ]

    # Ordered cluster labels for consistent hue ordering.
    hue_order = [id_to_label[cid] for cid in sorted(cluster_config)]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for i, (col, title) in enumerate(zip(shift_cols, titles)):
        data = merged.dropna(subset=[col, "Cluster"])
        sns.kdeplot(
            data=data,
            x=col,
            hue="Cluster",
            hue_order=hue_order,
            palette=label_to_col,
            ax=axes[i],
            fill=False,
            alpha=0.85,
            linewidth=1.8,
            legend=(i == 0),  # collect handles from first subplot only
        )
        axes[i].set_title(title, fontsize=12)
        axes[i].set_xlabel("Shift value", fontsize=10)
        axes[i].set_ylabel("Density" if i == 0 else "", fontsize=10)
        axes[i].axvline(0, color="grey", linestyle="--", alpha=0.5, lw=1)
        axes[i].grid(True, alpha=0.2)
        # Remove the per-subplot legend seaborn adds to the first panel.
        if axes[i].get_legend():
            axes[i].get_legend().remove()

    # ── single shared legend centred below all panels ───────────────────────
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=label_to_col[lbl], lw=2, label=lbl) for lbl in hue_order
    ]
    fig.tight_layout()
    fig.legend(
        handles=legend_handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=9,
        frameon=False,
    )

    fig.suptitle(
        "Distribution of country-level metric shifts (when adding offshore wind minus only onshore)",
        fontsize=13,
        y=1.02,
    )

    if out_path:
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        log.info("Saved: %s", out_path)

    return fig


# ---------------------------------------------------------------------------
# 15. Threshold-distribution histograms
# ---------------------------------------------------------------------------


def _draw_threshold_panel(
    ax,
    values: np.ndarray,
    *,
    bins: int = 60,
    xlabel: str = "",
    title: str = "",
    color: str = "steelblue",
    thresh_low: Optional[float] = None,
    thresh_high: Optional[float] = None,
) -> None:
    """Single histogram panel with statistical and fixed-threshold vertical lines.

    Dashed black  : mean
    Dashed blue   : p33 and p67
    Solid red     : fixed low threshold (if given)
    Solid darkred : fixed high threshold (if given)
    """
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        return

    ax.hist(valid, bins=bins, color=color, alpha=0.65, edgecolor="none", density=True)

    mean_val = float(np.mean(valid))
    p33 = float(np.percentile(valid, 33))
    p67 = float(np.percentile(valid, 67))

    ax.axvline(
        mean_val, color="black", linestyle="--", lw=1.6, label=f"Mean = {mean_val:.3g}"
    )
    ax.axvline(p33, color="royalblue", linestyle="--", lw=1.3, label=f"p33 = {p33:.3g}")
    ax.axvline(p67, color="royalblue", linestyle="--", lw=1.3, label=f"p67 = {p67:.3g}")

    if thresh_low is not None:
        ax.axvline(
            thresh_low,
            color="red",
            linestyle="-",
            lw=1.8,
            label=f"Low = {thresh_low:.3g}",
        )
    if thresh_high is not None:
        ax.axvline(
            thresh_high,
            color="darkred",
            linestyle="-",
            lw=1.8,
            label=f"High = {thresh_high:.3g}",
        )

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, framealpha=0.35)
    ax.grid(True, alpha=0.2)


def plot_abundance_histograms(
    ds: xr.Dataset,
    land: xr.DataArray,
    offshore: xr.DataArray,
    thresholds: dict,
    *,
    bins: int = 60,
    figsize: tuple = (10, 3.5),
    out_path: Optional[str] = None,
) -> "plt.Figure":
    """Three-panel histogram of abundance variable distributions (a).

    Panels: onshore wind CF | solar CF (onshore) | offshore wind climatology.
    Dashed lines: mean (black), p33/p67 (blue).
    Solid lines: fixed low (red) and high (darkred) classification thresholds.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain ``wind_CF``, ``solar_CF``, ``wind_climatology``.
    land, offshore : xr.DataArray
        Boolean masks on the same grid as *ds*.
    thresholds : dict
        Same structure as ``FIXED_THRESHOLDS`` — needs keys
        ``"wind_onshore"``, ``"solar"``, ``"wind_offshore"``
        each with ``"low"`` and ``"high"`` sub-keys.
    """
    wind_cf = ds["wind_CF"].where(land).values.ravel()
    solar_cf = ds["solar_CF"].where(land).values.ravel()
    off_wind = ds["wind_climatology"].where(offshore).values.ravel()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    _draw_threshold_panel(
        axes[0],
        wind_cf,
        bins=bins,
        xlabel="Capacity Factor",
        title="Onshore Wind CF",
        color="#0099FF",
        thresh_low=thresholds["wind_onshore"]["low"],
        thresh_high=thresholds["wind_onshore"]["high"],
    )
    _draw_threshold_panel(
        axes[1],
        solar_cf,
        bins=bins,
        xlabel="Capacity Factor",
        title="Solar CF (onshore)",
        color="#ff8819",
        thresh_low=thresholds["solar"]["low"],
        thresh_high=thresholds["solar"]["high"],
    )
    _draw_threshold_panel(
        axes[2],
        off_wind,
        bins=bins,
        xlabel="Wind Speed [m/s]",
        title="Offshore Wind Climatology",
        color="#4a53ff",
        thresh_low=thresholds["wind_offshore"]["low"],
        thresh_high=thresholds["wind_offshore"]["high"],
    )

    for ax, label in zip(axes, ["a", "b", "c"]):
        ax.text(
            -0.08,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    fig.suptitle(
        "Distribution of abundance variables with classification thresholds",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info("Saved: %s", out_path)

    return fig


def plot_storage_histograms(
    storage_da: xr.DataArray,
    land: xr.DataArray,
    offshore: xr.DataArray,
    thresholds: dict,
    *,
    bins: int = 60,
    figsize: tuple = (10, 3.5),
    out_path: Optional[str] = None,
) -> "plt.Figure":
    """Three-panel histogram of storage duration distributions (b).

    Panels: land storage | offshore storage | combined (land + offshore).
    Dashed lines: mean (black), p33/p67 (blue).
    Solid red line: fixed classification threshold.

    Parameters
    ----------
    storage_da : xr.DataArray
        Storage duration in days (e.g. ``ds_mean["duration_metric"]``).
    land, offshore : xr.DataArray
        Boolean masks on the same grid as *storage_da*.
    thresholds : dict
        Needs key ``"storage"`` with ``"land"`` and ``"offshore"`` sub-keys.
    """
    land_vals = storage_da.where(land).values.ravel()
    off_vals = storage_da.where(offshore).values.ravel()
    comb_vals = storage_da.where(land | offshore).values.ravel()

    th_land = float(thresholds["storage"]["land"])
    th_off = float(thresholds["storage"]["offshore"])

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    _draw_threshold_panel(
        axes[0],
        land_vals,
        bins=bins,
        xlabel="Storage Duration [days]",
        title="Land Storage Duration",
        color="#3cb44b",
        thresh_low=th_land,
    )
    _draw_threshold_panel(
        axes[1],
        off_vals,
        bins=bins,
        xlabel="Storage Duration [days]",
        title="Offshore Storage Duration",
        color="#4a53ff",
        thresh_low=th_off,
    )
    _draw_threshold_panel(
        axes[2],
        comb_vals,
        bins=bins,
        xlabel="Storage Duration [days]",
        title="Combined Storage Duration",
        color="#7c5cbf",
        thresh_low=th_land,
    )

    for ax, label in zip(axes, ["a", "b", "c"]):
        ax.text(
            -0.08,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    fig.suptitle(
        "Distribution of storage duration with classification threshold",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info("Saved: %s", out_path)

    return fig


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
