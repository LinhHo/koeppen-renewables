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
        If True, add a demand character (h/l).
    add_offshore
        If True, classify offshore pixels with an ``"offshore_"`` prefix.
    offshore_use_solar
        Whether the offshore label should include a solar abundance character
        (True for the *full* scheme, False for the *detailed* scheme).
    abundance_quantiles
        Tuple ``(low, high)`` defining the tertile thresholds when the caller
        does not supply explicit ``threshold_cf`` values.
    storage_quantile
        Quantile used to threshold storage duration (long vs short).
    demand_quantile
        Quantile used to threshold the demand proxy (high vs low).
    """

    name: str
    use_solar_land: bool = True
    use_wind_land: bool = True
    use_storage: bool = False
    use_demand: bool = False
    add_offshore: bool = False
    offshore_use_solar: bool = False
    abundance_quantiles: tuple = (0.33, 0.66)
    storage_quantile: float = 0.5
    demand_quantile: float = 0.5


# Three preset specs that correspond to the three notebook variants
SPEC_ABUNDANCE = ClassificationSpec(
    name="abundance",
    add_offshore=True,
    offshore_use_solar=True,
)

SPEC_DETAILED = ClassificationSpec(
    name="detailed",
    use_storage=True,
    use_demand=True,
    add_offshore=True,
    offshore_use_solar=False,  # offshore uses wind only
    storage_quantile=0.5,
    demand_quantile=0.5,
)

SPEC_FULL = ClassificationSpec(
    name="full",
    use_storage=True,
    use_demand=True,
    add_offshore=True,
    offshore_use_solar=True,
    storage_quantile=0.5,
    demand_quantile=0.5,
)


def _tertile_masks(values: xr.DataArray, low: float, high: float):
    """Return (low, mid, high) boolean masks for a tertile split."""
    low_m = (values < low) | values.isnull()
    mid_m = (values >= low) & (values < high)
    high_m = values >= high
    return low_m, mid_m, high_m


def _ensure_thresholds(
    threshold_cf: Optional[dict],
    solar_cf: xr.DataArray,
    wind_onshore_cf: xr.DataArray,
    quantiles: tuple,
) -> dict:
    low_q, high_q = quantiles
    if threshold_cf is None:
        threshold_cf = {}
    threshold_cf.setdefault(
        "solar",
        {
            "low": float(solar_cf.quantile(low_q).values),
            "high": float(solar_cf.quantile(high_q).values),
        },
    )
    threshold_cf.setdefault(
        "wind_onshore",
        {
            "low": float(wind_onshore_cf.quantile(low_q).values),
            "high": float(wind_onshore_cf.quantile(high_q).values),
        },
    )
    return threshold_cf


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

    Label structure (characters are appended left-to-right):

    - solar abundance (``H`` / ``M`` / ``L``)           if ``use_solar_land``
    - wind  abundance (``H`` / ``M`` / ``L``)           if ``use_wind_land``
    - reliability     (``R`` / ``U``)                   if ``use_storage``
    - demand          (``h`` / ``l``)                   if ``use_demand``

    Offshore labels are prefixed with ``"offshore_"``.  When
    ``spec.offshore_use_solar`` is False, the offshore label omits the solar
    character.

    Returns
    -------
    xarray.DataArray
        2-D DataArray of string labels (``""`` for pixels that match no group).
    """
    if spec.use_storage and ds_storage is None:
        raise ValueError(f"spec {spec.name!r} needs ds_storage")
    if spec.use_demand and ds_demand is None:
        raise ValueError(f"spec {spec.name!r} needs ds_demand")

    log_fn = log.info if verbose else log.debug

    wind_cf = ds["wind_CF"]
    if land is None:
        land = mask_land(wind_cf)
    if offshore is None and spec.add_offshore:
        offshore = mask_offshore(wind_cf)

    # -- land abundance ------------------------------------------------------
    solar_cf = ds["solar_CF"].where(land).compute()
    onshore_cf = ds["wind_CF"].where(land).compute()

    threshold_cf = _ensure_thresholds(
        threshold_cf, solar_cf, onshore_cf, spec.abundance_quantiles
    )
    log_fn(
        f"[{spec.name}] land thresholds — solar "
        f"({threshold_cf['solar']['low']:.2f}, {threshold_cf['solar']['high']:.2f}), "
        f"wind ({threshold_cf['wind_onshore']['low']:.2f}, "
        f"{threshold_cf['wind_onshore']['high']:.2f})"
    )

    solar_low, solar_mid, solar_high = _tertile_masks(
        solar_cf, threshold_cf["solar"]["low"], threshold_cf["solar"]["high"]
    )
    wind_low, wind_mid, wind_high = _tertile_masks(
        onshore_cf,
        threshold_cf["wind_onshore"]["low"],
        threshold_cf["wind_onshore"]["high"],
    )

    land_axes: list[list[tuple]] = []
    if spec.use_solar_land:
        land_axes.append([(solar_high, "H"), (solar_mid, "M"), (solar_low, "L")])
    if spec.use_wind_land:
        land_axes.append([(wind_high, "H"), (wind_mid, "M"), (wind_low, "L")])

    # -- land reliability (storage) ------------------------------------------
    if spec.use_storage:
        land_storage = ds_storage.where(land).compute()
        th_land = float(land_storage.quantile(spec.storage_quantile).values)
        threshold_cf.setdefault("storage", {})["land"] = th_land
        log_fn(
            f"[{spec.name}] land storage threshold (q={spec.storage_quantile}) = "
            f"{th_land:.2f} days"
        )
        land_axes.append(
            [(land_storage < th_land, "R"), (land_storage >= th_land, "U")]
        )

    # -- demand --------------------------------------------------------------
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
        threshold_cf.setdefault(
            "wind_offshore",
            {
                "low": float(off_wind.quantile(spec.abundance_quantiles[0]).values),
                "high": float(off_wind.quantile(spec.abundance_quantiles[1]).values),
            },
        )
        ow_low, ow_mid, ow_high = _tertile_masks(
            off_wind,
            threshold_cf["wind_offshore"]["low"],
            threshold_cf["wind_offshore"]["high"],
        )

        off_axes: list[list[tuple]] = []

        if spec.offshore_use_solar:
            off_solar = ds["solar_climatology"].where(offshore).compute()
            threshold_cf.setdefault(
                "solar_offshore",
                {
                    "low": float(
                        off_solar.quantile(spec.abundance_quantiles[0]).values
                    ),
                    "high": float(
                        off_solar.quantile(spec.abundance_quantiles[1]).values
                    ),
                },
            )
            os_low, os_mid, os_high = _tertile_masks(
                off_solar,
                threshold_cf["solar_offshore"]["low"],
                threshold_cf["solar_offshore"]["high"],
            )
            off_axes.append([(os_high, "H"), (os_mid, "M"), (os_low, "L")])

        off_axes.append([(ow_high, "H"), (ow_mid, "M"), (ow_low, "L")])

        if spec.use_storage:
            off_storage = ds_storage.where(offshore).compute()
            th_off = float(off_storage.quantile(spec.storage_quantile).values)
            threshold_cf["storage"]["offshore"] = th_off
            log_fn(f"[{spec.name}] offshore storage threshold = {th_off:.2f} days")
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

GROUPS_ABUNDANCE: dict = {
    "B": [
        LAND_COLORS["B"],
        "- Both abundance",
        ["HH", "MM", "offshore_HH", "offshore_MM"],
    ],
    "W": [
        LAND_COLORS["W"],
        "- Wind dominant",
        ["LH", "LM", "offshore_LH", "offshore_LM"],
    ],
    "Ws": [LAND_COLORS["Ws"], "- Wind favourable", ["MH", "offshore_MH"]],
    "S": [
        LAND_COLORS["S"],
        "- Solar dominant",
        ["HL", "ML", "offshore_HL", "offshore_ML"],
    ],
    "Sw": [LAND_COLORS["Sw"], "- Solar favourable", ["HM", "offshore_HM"]],
    "P": [LAND_COLORS["P"], "- Poor both", ["LL", "offshore_LL"]],
}


# --- detailed (offshore wind only, 5-char land labels) ---------------------
# Land: solar(HML) + solar-reliability(RU) + wind(HML) + wind-reliability(RU) + demand(hl)
# This matches the historical `groups_colours` dict. Offshore uses
# `offshore_<WindAbund><Reliability><Demand>`.

GROUPS_DETAILED: dict = {
    "B": ["#3cb44b", "- Both", ["HHRh", "MMRh"]],
    "B_l": [light("#3cb44b"), "", ["HHRl", "MMRl"]],
    "B_u": ["#688818", "", ["HHUh", "MMUh"]],
    "B_ul": [
        light("#688818"),
        "",
        ["HHUl", "MMUl"],
    ],
    "W": ["#0099FF", "- Wind", ["LHRh", "LMRh"]],
    "W_l": [light("#0099FF"), "", ["LHRl", "LMRl"]],
    "W_u": ["#2e86a8", "", ["LHUh", "LMUh"]],
    "W_ul": [light("#2e86a8"), "", ["LHUl", "LMUl"]],
    "Ws": ["#00FFEE", "- Wind solar", ["MHRh"]],
    "Ws_l": [light("#00FFEE"), "", ["MHRl"]],
    "Ws_u": ["#22B8AE", "", ["MHUh"]],
    "Ws_ul": [light("#22B8AE"), "", ["MHUl"]],
    "S": ["#ff8819", "- Solar", ["HLRh", "MLRh"]],
    "S_l": [light("#ff8819"), "", ["HLRl", "MLRl"]],
    "S_u": ["#b25c0c", "", ["HLUh", "MLUh"]],
    "S_ul": [light("#b25c0c"), "", ["HLUl", "MLUl"]],
    "Sw": ["#fff319", "- Solar wind", ["HMRh"]],
    "Sw_l": [light("#fff319"), "", ["HMRl"]],
    "Sw_u": ["#b2aa13", "", ["HMUh"]],
    "Sw_ul": [light("#b2aa13"), "", ["HMUl"]],
    "P": ["#FF0000", "- Poor", ["LLxh"]],
    "P_l": ["#BFBFBF", "", ["LLxl"]],
    "o_h": ["#EA6565", "", ["offshore_LRh", "offshore_LUh"]],
    "o_l": ["#CECDCD", "", ["offshore_LRl", "offshore_LUl"]],
    "O": ["#4a53ff", "- Offshore", ["offshore_HRh", "offshore_MRh"]],
    "O_l": ["#a6a3fe", "", ["offshore_HRl", "offshore_MRl"]],
    "O_u": ["#8f13fb", "", ["offshore_HUh", "offshore_MUh"]],
    "O_ul": ["#d3a1ff", "", ["offshore_HUl", "offshore_MUl"]],
}


# --- full (offshore uses solar + wind, 4-char land labels) -----------------
# Land: solar(HML) + wind(HML) + reliability(RU) + demand(hl)
# Offshore: solar(HML) + wind(HML) + reliability(RU) + demand(hl)


def _build_full_groups() -> dict:
    groups = {}

    def add(code, base_col, label, on_suffixes, off_suffixes=None):
        land = [c for c in on_suffixes]
        if off_suffixes is not None:
            land += [f"offshore_{s}" for s in off_suffixes]
        groups[code] = [base_col, label, land]
        groups[f"{code}_l"] = [
            light(base_col),
            "",
            [
                (
                    p.replace("h", "l", 1)
                    if p.endswith("h")
                    else (p[:-2] + "l" if p[-1] == "h" else p)
                )
                for p in land
            ],
        ]
        # We generate low-demand / variable variants explicitly below for
        # clarity rather than programmatically — see GROUPS_FULL literal.

    # The literal dict below is clearer than programmatic generation.
    return groups  # unused


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
    "W": [
        LAND_COLORS["W"],
        "- Wind",
        ["LHRh", "LMRh", "offshore_LHRh", "offshore_LMRh"],
    ],
    "W_l": [
        light(LAND_COLORS["W"]),
        "",
        ["LHRl", "LMRl", "offshore_LHRl", "offshore_LMRl"],
    ],
    "W_v": [
        dirty(LAND_COLORS["W"]),
        "",
        ["LHUh", "LMUh", "offshore_LHUh", "offshore_LMUh"],
    ],
    "W_vl": [
        light(dirty(LAND_COLORS["W"])),
        "",
        ["LHUl", "LMUl", "offshore_LHUl", "offshore_LMUl"],
    ],
    "Ws": [LAND_COLORS["Ws"], "- Wind solar", ["MHRh", "offshore_MHRh"]],
    "Ws_l": [light(LAND_COLORS["Ws"]), "", ["MHRl", "offshore_MHRl"]],
    "Ws_v": [dirty(LAND_COLORS["Ws"]), "", ["MHUh", "offshore_MHUh"]],
    "Ws_vl": [light(dirty(LAND_COLORS["Ws"])), "", ["MHUl", "offshore_MHUl"]],
    "S": [
        LAND_COLORS["S"],
        "- Solar",
        ["HLRh", "MLRh", "offshore_HLRh", "offshore_MLRh"],
    ],
    "S_l": [
        light(LAND_COLORS["S"]),
        "",
        ["HLRl", "MLRl", "offshore_HLRl", "offshore_MLRl"],
    ],
    "S_v": [
        dirty(LAND_COLORS["S"]),
        "",
        ["HLUh", "MLUh", "offshore_HLUh", "offshore_MLUh"],
    ],
    "S_vl": [
        light(dirty(LAND_COLORS["S"])),
        "",
        ["HLUl", "MLUl", "offshore_HLUl", "offshore_MLUl"],
    ],
    "Sw": [LAND_COLORS["Sw"], "- Solar wind", ["HMRh", "offshore_HMRh"]],
    "Sw_l": [light(LAND_COLORS["Sw"]), "", ["HMRl", "offshore_HMRl"]],
    "Sw_v": [dirty(LAND_COLORS["Sw"]), "", ["HMUh", "offshore_HMUh"]],
    "Sw_vl": [light(dirty(LAND_COLORS["Sw"])), "", ["HMUl", "offshore_HMUl"]],
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
    if not isinstance(zone, str) or zone == "":
        return {"wind": np.nan, "solar": np.nan}
    amap = {"L": 1, "M": 2, "H": 3}
    clean = zone.replace("offshore_", "")
    try:
        return {
            "solar": amap.get(clean[0].upper(), np.nan),
            "wind": amap.get(clean[1].upper(), np.nan),
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

    valid = ~np.isnan(wind) & ~np.isnan(solar)
    raw = np.full(zones.shape, np.nan)
    if method == "optimal_alpha":
        raw[valid] = (
            optimal_alpha[valid] * wind[valid]
            + (1 - optimal_alpha[valid]) * solar[valid]
        )
    elif method == "max_abundance":
        raw[valid] = np.maximum(wind[valid], solar[valid])
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

    df_plot = df_plot.rename(columns={"avg_resource": "Average resource"})
    sns.scatterplot(
        data=df_plot,
        x="resource_demand_corr",
        y="avg_storage",
        size="Average resource",
        hue="cluster name",
        palette=palette,
        sizes=(20, 600),
        alpha=0.5,
        edgecolor="w",
        ax=ax_scatter,
    )

    for cid in sorted_ids:
        colour, _lab = cluster_map[cid]
        row = df_centroids.iloc[cid]
        size = 100 + row["avg_resource"] * 800
        ax_scatter.scatter(
            row["resource_demand_corr"],
            row["avg_storage"],
            marker="X",
            s=size,
            color=colour,
            edgecolors="black",
            linewidth=1.2,
            zorder=5,
        )

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

    ax_scatter.set_title("a) Cluster Characteristics", loc="left", fontweight="bold")
    ax_scatter.axvline(0, color="grey", linestyle="--", alpha=0.3)
    ax_scatter.grid(True, alpha=0.2)
    ax_scatter.set_xlim([-0.9, 0.8])
    ax_scatter.set_xlabel("Resource-demand spatial correlation")
    ax_scatter.set_ylabel("Average storage duration [normalised]")
    leg = ax_scatter.legend(title="Strategic Groups", loc="upper right")
    leg.get_frame().set_alpha(0.2)

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


EU28_ISO3 = [
    "AUT",
    "BEL",
    "BGR",
    "HRV",
    "CYP",
    "CZE",
    "DNK",
    "EST",
    "FIN",
    "FRA",
    "DEU",
    "GRC",
    "HUN",
    "IRL",
    "ITA",
    "LVA",
    "LTU",
    "LUX",
    "MLT",
    "NLD",
    "POL",
    "PRT",
    "ROU",
    "SVK",
    "SVN",
    "ESP",
    "SWE",
    "GBR",
]


# ---------------------------------------------------------------------------
# 12. 3D cluster plot (manuscript Fig 6)
# ---------------------------------------------------------------------------


def plot_country_clusters_3d(
    df: pd.DataFrame,
    df_centroids: pd.DataFrame,
    cluster_map: dict,
    *,
    list_cnt_to_plot: Optional[Sequence[str]] = None,
    fig_size: tuple = (1400, 900),
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

    # One trace per cluster, so the legend shows the manuscript labels.
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
                    size=6,
                    color=colour,
                    opacity=0.75,
                    line=dict(width=0.5, color="white"),
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

    # Centroids: one 'X' marker per cluster, in the cluster colour.
    fig.add_trace(
        go.Scatter3d(
            x=df_centroids["avg_resource"],
            y=df_centroids["resource_demand_corr"],
            z=df_centroids["avg_storage"],
            mode="markers",
            marker=dict(
                symbol="x",
                size=9,
                color=[cluster_map[i][0] for i in range(len(df_centroids))],
                line=dict(width=3, color="black"),
            ),
            name="Cluster centroids",
            hoverinfo="skip",
        )
    )

    # Named-country labels (mirrors Fig 5a).
    if list_cnt_to_plot:
        to_label = df[df["iso3"].isin(list_cnt_to_plot)]
        if not to_label.empty:
            fig.add_trace(
                go.Scatter3d(
                    x=to_label["avg_resource"],
                    y=to_label["resource_demand_corr"],
                    z=to_label["avg_storage"],
                    mode="text",
                    text=to_label.get("country_name", to_label["iso3"]),
                    textposition="top center",
                    textfont=dict(size=11, color="black"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title=go.layout.Title(
            text=(
                "<b>Global Renewable Strategy Clusters (3D view)</b><br>"
                "<sup>Storage, resource–demand correlation, and resource availability</sup>"
            ),
            x=0,
        ),
        width=fig_size[0],
        height=fig_size[1],
        scene=dict(
            xaxis=dict(title="Avg resource availability"),
            yaxis=dict(title="Spatial correlation (resource vs demand)"),
            zaxis=dict(title="Avg storage requirement [normalised]"),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]),
            ),
            aspectmode="cube",
        ),
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        margin=dict(l=0, r=0, b=0, t=100),
    )

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Static export — needs `kaleido`. We also write an HTML alongside
        # so the interactive version is always available.
        try:
            fig.write_image(
                str(out_path), width=fig_size[0], height=fig_size[1], scale=2
            )
            log.info("Saved 3D cluster figure: %s", out_path)
        except Exception as exc:
            log.warning(
                "Could not save static 3D figure to %s (install kaleido): %s",
                out_path,
                exc,
            )
        html_path = out_path.with_suffix(".html")
        fig.write_html(str(html_path))
        log.info("Saved interactive 3D cluster figure: %s", html_path)

    if show:
        fig.show()

    return fig


# ---------------------------------------------------------------------------
# 13. Logging helpers — summaries of datasets and dataframes
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
