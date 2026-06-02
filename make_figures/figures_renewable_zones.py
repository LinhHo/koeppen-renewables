"""
figures_renewable_zones.py
==========================

Part 1 — post-process data into zone classifications and create gridded maps.

Reads from:
  results/automatic/abundance/abundance_*.nc   — solar_CF, wind_CF
  results/automatic/demand/demand_*.nc         — demand_proximity_weighted_buffered
  results/automatic/storage/lds_*.nc           — long-duration storage outputs
  results/automatic/climatology/ws100/*.nc     — ws100 (-> wind_climatology)
  results/automatic/climatology/ssrd/*.nc      — ssrd  (-> solar_climatology)

Saves derived data to results/post_processed_data/ for use by figures_by_country.py.

Run:
  python make_figures/figures_renewable_zones.py
  python make_figures/figures_renewable_zones.py --only fig1 fig2c
"""

from __future__ import annotations

import argparse
import colorsys
import logging
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Sequence

import matplotlib.colorbar as mcolorbar
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patheffects import withStroke

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_utils import (
    ClassificationSpec,
    GROUPS_ABUNDANCE,
    GROUPS_ABUNDANCE_OFFSHORE_WIND,
    GROUPS_DETAILED,
    GROUPS_FULL,
    LAND_COLORS,
    SPEC_ABUNDANCE,
    SPEC_ABUNDANCE_OFFSHORE_WIND,
    SPEC_DETAILED,
    SPEC_FULL,
    log_df_summary,
    log_ds_summary,
    mask_land,
    mask_offshore,
    plot_map_continuous,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR.parent / "resources"
RESULTS_DIR = BASE_DIR.parent / "results"
FIG_DIR = RESULTS_DIR / "figures" / "main"
FIG_DIR.mkdir(parents=True, exist_ok=True)

POST_PROCESSED_DIR = RESULTS_DIR / "post_processed_data"
POST_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = FIG_DIR / "logs"


def setup_logging(verbose: bool = False) -> Path:
    """Configure logging to both console and a timestamped file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"figures_zones_{timestamp}.log"

    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(ch)

    return log_path


log = logging.getLogger("figures_renewable_zones")


# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------
FIXED_THRESHOLDS = {
    "solar": {"high": 0.20, "low": 0.15},  # capacity factor
    "wind_onshore": {"high": 0.35, "low": 0.25},  # capacity factor
    "wind_offshore": {"low": 7.0, "high": 8.5},  # m/s
    "solar_offshore": {"low": 150.0, "high": 225.0},  # W/m2
    "storage": {"land": 14, "offshore": 14},  # days
}


# ---------------------------------------------------------------------------
# Zone utility functions (copied verbatim from plot_utils.py)
# ---------------------------------------------------------------------------


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
    - reliability     (``R`` / ``V``)             if ``use_storage``
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
            [(land_storage < th_land, "R"), (land_storage >= th_land, "V")]
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
    zones = np.full(wind_cf.shape, "", dtype=object)

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
            off_axes.append([(off_storage < th_off, "R"), (off_storage >= th_off, "V")])

        if spec.use_demand:
            off_axes.append([(demand_high, "h"), (demand_low, "l")])

        _paint(off_axes, offshore, prefix="offshore_")

    return xr.DataArray(zones, dims=wind_cf.dims, coords=wind_cf.coords, name="zones")


def land_only_groups(groups: dict) -> dict:
    """Return subset of *groups* with only land (non-offshore/O) entries."""
    return {
        k: v for k, v in groups.items() if not (k.startswith("o") or k.startswith("O"))
    }


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
    rgb = np.flipud(rgb)

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


def plot_zones_climatology(
    ds: xr.Dataset,
    groups: dict,
    *,
    land: Optional[xr.DataArray] = None,
    offshore: Optional[xr.DataArray] = None,
    out_path: Optional[str] = None,
    figsize: tuple = (16, 10),
    title: Optional[str] = None,
    legend_anchor: tuple = (0.5, -0.20),
    legend_ncol: int = 4,
    extent: Sequence[float] = (-180, 180, -60, 80),
) -> "tuple[plt.Figure, object]":
    """Classify and plot abundance zones using climatology variables with quantile thresholds.

    Classification uses ``wind_climatology`` and ``solar_climatology`` from *ds*
    instead of capacity-factor fields.  Land-cell thresholds (p33/p67) are
    derived independently for each variable from the unmasked set of valid land
    pixels; offshore thresholds are derived from offshore pixels.

    Land labels  : wind(H/M/L) + solar(H/M/L)  →  2-char, e.g. ``"HL"``
    Offshore labels: ``"offshore_H"``, ``"offshore_M"``, ``"offshore_L"``

    These are compatible with ``GROUPS_ABUNDANCE_OFFSHORE_WIND``.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain ``wind_climatology`` and ``solar_climatology``.
    groups : dict
        Colour/label/pattern dict (use ``GROUPS_ABUNDANCE_OFFSHORE_WIND``).
    land, offshore : xr.DataArray, optional
        Boolean masks.  Derived automatically from ``wind_climatology`` if not
        supplied.
    out_path : str, optional
        File path to save the figure.
    """
    wind_clim = ds["wind_climatology"]
    if land is None:
        land = mask_land(wind_clim)
    if offshore is None:
        offshore = mask_offshore(wind_clim)

    # ── compute quantile thresholds from masked data ────────────────────────
    def _q(da: xr.DataArray, mask: xr.DataArray, q: float) -> float:
        vals = da.where(mask).values.ravel()
        return float(np.nanpercentile(vals[np.isfinite(vals)], q * 100))

    # land thresholds
    wind_lo = _q(wind_clim, land, 1 / 3)
    wind_hi = _q(wind_clim, land, 2 / 3)
    solar_lo = _q(ds["solar_climatology"], land, 1 / 3)
    solar_hi = _q(ds["solar_climatology"], land, 2 / 3)
    # offshore threshold (wind only)
    off_lo = _q(wind_clim, offshore, 1 / 3)
    off_hi = _q(wind_clim, offshore, 2 / 3)

    log.info(
        "Climatology thresholds — land wind: %.2f/%.2f  land solar: %.2f/%.2f  "
        "offshore wind: %.2f/%.2f",
        wind_lo,
        wind_hi,
        solar_lo,
        solar_hi,
        off_lo,
        off_hi,
    )

    # ── classify ────────────────────────────────────────────────────────────
    wind_land = wind_clim.where(land).compute()
    solar_land = ds["solar_climatology"].where(land).compute()
    wind_off = wind_clim.where(offshore).compute()

    def _label(val, lo, hi):
        return np.where(val >= hi, "H", np.where(val >= lo, "M", "L"))

    zones = np.full(wind_land.shape, "", dtype=object)

    # land: wind-first + solar-second
    w_lbl = _label(wind_land.values, wind_lo, wind_hi)
    s_lbl = _label(solar_land.values, solar_lo, solar_hi)
    land_mask = land.values
    zones[land_mask] = np.char.add(w_lbl, s_lbl)[land_mask]

    # offshore: wind only
    off_mask = offshore.values
    o_lbl = _label(wind_off.values, off_lo, off_hi)
    zones[off_mask] = np.char.add("offshore_", o_lbl)[off_mask]

    ds_zones = xr.DataArray(
        zones,
        dims=wind_land.dims,
        coords=wind_land.coords,
        name="zones",
    ).to_dataset()

    # ── derive threshold strings for a default title ─────────────────────────
    if title is None:
        title = (
            f"Abundance zones — climatology (p33/p67 thresholds)\n"
            f"land wind {wind_lo:.1f}/{wind_hi:.1f} m/s, "
            f"land solar {solar_lo:.0f}/{solar_hi:.0f} J/m², "
            f"offshore wind {off_lo:.1f}/{off_hi:.1f} m/s"
        )

    return plot_zones_map(
        ds_zones,
        groups,
        out_path=out_path,
        figsize=figsize,
        title=title,
        legend_anchor=legend_anchor,
        legend_ncol=legend_ncol,
        extent=extent,
    )


def _expand_wildcards(label: str, replacements=("V", "R")) -> list[str]:
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
RENEWABLE_ORDER = ["A", "W", "Ws", "S", "Sw", "P"]
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
        # Flip if the dataset's latitude axis is ascending (south-up)
        if lat[0] < lat[-1]:
            raster = raster[::-1, :]
        ds_out["koppen_code"] = xr.DataArray(
            raster, coords=ds_out["zones"].coords, dims=ds_out["zones"].dims
        )
    else:
        warnings.warn(f"Köppen resources not found at {koppen_dbf}; skipping overlay.")

    return ds_out, percentage_df


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
        if is_offshore and len(clean) >= 2 and clean[1].upper() in ("R", "V"):
            # DETAILED offshore: wind(HML) + storage(RV) + demand(hl)
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


# ---------------------------------------------------------------------------
# DataBundle — lazy cache
# ---------------------------------------------------------------------------


class DataBundle:
    """Lazy holder for all datasets the zone figures share."""

    def __init__(self) -> None:
        self._ds_processed: xr.Dataset | None = None
        self._ds_mean = None
        self._normalized_demand = None
        self._ds_zones_abundance = None
        self._ds_zones_abundance_wind = None
        self._ds_zones_detailed = None
        self._ds_zones_full = None
        self._grouped_detailed_land = None
        self._df_pct_detailed_land = None
        self._grouped_detailed_all = None
        self._df_pct_detailed_all = None
        self._grouped_full_land = None
        self._df_pct_full_land = None
        self._ds_res_avail = None

    # -- primary inputs --------------------------------------------------
    @property
    def ds_processed(self) -> xr.Dataset:
        """Merge abundance, demand, and annual-mean climatology into one dataset.
        If the post-processed abundance with solar CF filled NA dataset is available, use that for abundance;
        """
        if self._ds_processed is None:
            if (
                RESULTS_DIR / "post_processed_data/processed_solar_CF_filled.nc"
            ).exists():
                abundance_pattern = str(
                    RESULTS_DIR / "post_processed_data/processed_solar_CF_filled.nc"
                )
                ds_ab = xr.open_dataset(abundance_pattern)
            else:
                abundance_pattern = str(
                    RESULTS_DIR / "automatic/abundance/abundance_*.nc"
                )
                ds_ab = xr.open_mfdataset(abundance_pattern, combine="by_coords")
            log.info("Loading abundance from %s", abundance_pattern)

            demand_pattern = str(RESULTS_DIR / "automatic/demand/demand_*.nc")
            ws100_pattern = str(RESULTS_DIR / "automatic/climatology/ws100/*.nc")
            ssrd_pattern = str(RESULTS_DIR / "automatic/climatology/ssrd/*.nc")

            log.info("Loading demand from %s", demand_pattern)
            ds_dm = xr.open_mfdataset(demand_pattern, combine="by_coords")[
                ["demand_proximity_weighted_buffered"]
            ]

            log.info("Loading ws100 climatology from %s", ws100_pattern)
            wind_clim = (
                xr.open_mfdataset(ws100_pattern, combine="by_coords")["ws100"]
                .mean("dayofyear")
                .rename("wind_climatology")
            )

            log.info("Loading ssrd climatology from %s", ssrd_pattern)
            solar_clim = (
                xr.open_mfdataset(ssrd_pattern, combine="by_coords")["ssrd_climatology"]
                .mean("dayofyear")
                .rename("solar_climatology")
            )

            self._ds_processed = xr.merge(
                [ds_ab, ds_dm, wind_clim.to_dataset(), solar_clim.to_dataset()]
            )
            log_ds_summary(
                self._ds_processed,
                "ds_processed",
                variables=[
                    v
                    for v in (
                        "solar_CF",
                        "wind_CF",
                        "demand_proximity_weighted_buffered",
                        "wind_climatology",
                        "solar_climatology",
                    )
                    if v in self._ds_processed.data_vars
                ],
            )
        return self._ds_processed

    @property
    def land(self):
        return mask_land(self.ds_processed["wind_CF"])

    @property
    def offshore(self):
        return mask_offshore(self.ds_processed["wind_CF"])

    @property
    def ds_mean(self):
        if self._ds_mean is None:
            # sys.path.insert(0, str(BASE_DIR.parent / "src"))
            from plot_utils import aggregate_lds

            storage_path = str(RESULTS_DIR / "automatic/storage/")
            log.info("Aggregating storage (metric=mean) from %s", storage_path)
            self._ds_mean = aggregate_lds(storage_path, metric="mean")

            dm = self._ds_mean["duration_metric"]
            land_dm = dm.where(self.land)
            off_dm = dm.where(self.offshore)
            log.info(
                "Storage duration (days): land mean=%.2f max=%.2f | "
                "offshore mean=%.2f max=%.2f",
                float(land_dm.mean()),
                float(land_dm.max()),
                float(off_dm.mean()),
                float(off_dm.max()),
            )
            threshold = FIXED_THRESHOLDS["storage"]["land"]
            land_share = (
                float(np.nansum(land_dm >= threshold))
                / max(1, int(land_dm.notnull().sum()))
                * 100
            )
            off_share = (
                float(np.nansum(off_dm >= threshold))
                / max(1, int(off_dm.notnull().sum()))
                * 100
            )
            log.info(
                "Share of pixels with storage >= %d days: land=%.1f%%  offshore=%.1f%%",
                threshold,
                land_share,
                off_share,
            )
            log_ds_summary(
                self._ds_mean,
                "ds_mean",
                variables=["duration_metric", "optimal_alpha"],
            )
        return self._ds_mean

    @property
    def normalized_demand(self):
        if self._normalized_demand is None:
            demand_raw = self.ds_processed["demand_proximity_weighted_buffered"]
            dmax = demand_raw.quantile(0.95).values
            log.info(
                "Demand raw: min=%.4g  p95=%.4g  max=%.4g",
                float(demand_raw.min()),
                float(dmax),
                float(demand_raw.max()),
            )
            nd = ((demand_raw - demand_raw.min()) / (dmax - demand_raw.min())).clip(
                1e-6, 1
            )
            self._normalized_demand = nd.where(nd > 1e-6)
            log.info(
                "Share of grid cells with normalised demand >= 0.5: %.1f%%",
                100
                * float((self._normalized_demand >= 0.5).sum())
                / max(1, int(self._normalized_demand.notnull().sum())),
            )
        return self._normalized_demand

    # -- derived zones ---------------------------------------------------
    @property
    def threshold(self) -> dict:
        return {k: dict(v) for k, v in FIXED_THRESHOLDS.items()}

    @property
    def ds_zones_abundance(self):
        if self._ds_zones_abundance is None:
            log.info("Classifying zones -- ABUNDANCE (offshore uses wind + solar)")
            self._ds_zones_abundance = classify_zones(
                self.ds_processed,
                SPEC_ABUNDANCE,
                threshold_cf=self.threshold,
                land=self.land,
                offshore=self.offshore,
            ).to_dataset()
            self._log_zone_counts(self._ds_zones_abundance, "ds_zones_abundance")
        return self._ds_zones_abundance

    @property
    def ds_zones_abundance_wind(self):
        if self._ds_zones_abundance_wind is None:
            log.info(
                "Classifying zones -- ABUNDANCE_OFFSHORE_WIND (offshore uses wind only)"
            )
            self._ds_zones_abundance_wind = classify_zones(
                self.ds_processed,
                SPEC_ABUNDANCE_OFFSHORE_WIND,
                threshold_cf=self.threshold,
                land=self.land,
                offshore=self.offshore,
            ).to_dataset()
            self._log_zone_counts(
                self._ds_zones_abundance_wind, "ds_zones_abundance_wind"
            )
        return self._ds_zones_abundance_wind

    @property
    def ds_zones_detailed(self):
        if self._ds_zones_detailed is None:
            log.info("Classifying zones -- DETAILED (offshore uses wind only)")
            self._ds_zones_detailed = classify_zones(
                self.ds_processed,
                SPEC_DETAILED,
                ds_storage=self.ds_mean["duration_metric"],
                ds_demand=self.normalized_demand,
                threshold_cf=self.threshold,
                land=self.land,
                offshore=self.offshore,
            ).to_dataset()
            self._log_zone_counts(self._ds_zones_detailed, "ds_zones_detailed")
        return self._ds_zones_detailed

    @property
    def ds_zones_full(self):
        if self._ds_zones_full is None:
            log.info("Classifying zones -- FULL (offshore uses wind + solar)")
            self._ds_zones_full = classify_zones(
                self.ds_processed,
                SPEC_FULL,
                ds_storage=self.ds_mean["duration_metric"],
                ds_demand=self.normalized_demand,
                threshold_cf=self.threshold,
                land=self.land,
                offshore=self.offshore,
            ).to_dataset()
            self._log_zone_counts(self._ds_zones_full, "ds_zones_full")
        return self._ds_zones_full

    @staticmethod
    def _log_zone_counts(ds_zones: xr.Dataset, name: str) -> None:
        labels, counts = np.unique(ds_zones["zones"].values, return_counts=True)
        pairs = [(l, int(c)) for l, c in zip(labels, counts) if l != ""]
        pairs.sort(key=lambda p: p[1], reverse=True)
        total = sum(c for _, c in pairs) or 1
        log.info(
            "%s: %d distinct labels, %d classified pixels",
            name,
            len(pairs),
            total,
        )
        top = pairs[:10]
        log.info(
            "  top-10 labels: %s",
            ", ".join(f"{l}={c} ({100*c/total:.1f}%)" for l, c in top),
        )
        log.debug("  full label counts: %s", dict(pairs))

    # -- stats panels (main text) ---------------------------------------
    @property
    def grouped_detailed_land(self):
        if self._grouped_detailed_land is None:
            log.info("Computing base+subgroup decomposition (DETAILED, land only)")
            self._grouped_detailed_land, self._df_pct_detailed_land = (
                get_base_and_subgroup(
                    self.ds_zones_detailed.where(self.land),
                    GROUPS_DETAILED,
                )
            )
            log_df_summary(self._df_pct_detailed_land, "df_pct_detailed_land")
            by_base = (
                self._df_pct_detailed_land.groupby("zones_base_grouped")["percentage"]
                .sum()
                .sort_values(ascending=False)
            )
            log.info("Main renewable zones, share of land cells (%%):")
            for k, v in by_base.items():
                log.info("  %-4s %6.2f", k, v)
        return self._grouped_detailed_land

    @property
    def df_pct_detailed_land(self):
        _ = self.grouped_detailed_land
        if self._df_pct_detailed_land is not None:
            csv_path = POST_PROCESSED_DIR / "df_zones_detailed_LAND_percentage.csv"
            self._df_pct_detailed_land.to_csv(str(csv_path), index=False)
            log.info("Saved: %s", csv_path)
        return self._df_pct_detailed_land

    @property
    def grouped_detailed_all(self):
        if self._grouped_detailed_all is None:
            log.info("Computing base+subgroup decomposition (DETAILED, all pixels)")
            self._grouped_detailed_all, self._df_pct_detailed_all = (
                get_base_and_subgroup(
                    self.ds_zones_detailed,
                    GROUPS_DETAILED,
                )
            )
            log_df_summary(self._df_pct_detailed_all, "df_pct_detailed_all")
        return self._grouped_detailed_all

    @property
    def df_pct_detailed_all(self):
        _ = self.grouped_detailed_all
        if self._df_pct_detailed_all is not None:
            csv_path = POST_PROCESSED_DIR / "df_zones_detailed_ALL_percentage.csv"
            self._df_pct_detailed_all.to_csv(str(csv_path), index=False)
            log.info("Saved: %s", csv_path)
        return self._df_pct_detailed_all

    # -- stats panels (supplementary) -----------------------------------
    @property
    def grouped_full_land(self):
        if self._grouped_full_land is None:
            log.info("Computing base+subgroup decomposition (FULL, land only)")
            self._grouped_full_land, self._df_pct_full_land = get_base_and_subgroup(
                self.ds_zones_full.where(self.land),
                GROUPS_FULL,
            )
            log_df_summary(self._df_pct_full_land, "df_pct_full_land")
        return self._grouped_full_land

    @property
    def df_pct_full_land(self):
        _ = self.grouped_full_land
        return self._df_pct_full_land

    # -- resource availability ------------------------------------------
    @property
    def ds_res_avail(self):
        if self._ds_res_avail is None:
            log.info("Converting zones to resource availability (method=max_abundance)")
            self._ds_res_avail = convert_zones_to_resource_availability(
                self.ds_zones_detailed,
                self.ds_mean,
                land=self.land,
                offshore=self.offshore,
                method="max_abundance",
            )
            log_ds_summary(
                self._ds_res_avail,
                "ds_res_avail",
                variables=[
                    "resource_availability",
                    "wind_cat",
                    "solar_cat",
                    "optimal_alpha",
                ],
            )
        return self._ds_res_avail

    # -- save derived data for Part 2 -----------------------------------
    def save_post_processed(self) -> None:
        """Save derived datasets to post_processed_data/ for figures_by_country.py."""
        POST_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        paths = {
            "resource_availability.nc": self.ds_res_avail,
            "storage_mean.nc": self.ds_mean,
            "zones_detailed.nc": self.ds_zones_detailed,
            "normalized_demand.nc": self.normalized_demand.to_dataset(
                name="normalized_demand"
            ),
        }
        for fname, ds in paths.items():
            out = POST_PROCESSED_DIR / fname
            log.info("Saving %s ...", out)
            ds.to_netcdf(str(out), engine="netcdf4", mode="w")
        log.info("Post-processed data saved to %s", POST_PROCESSED_DIR)


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------


def _out(name: str, fmt: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return str(FIG_DIR / f"{name}.{fmt}")


def fig_detailed_zones_map(data: DataBundle, fmt: str) -> None:
    """Fig 1 -- Koppen-style renewable zones (main text, detailed scheme)."""
    plot_zones_map(
        data.ds_zones_detailed,
        GROUPS_DETAILED,
        out_path=_out("fig1_renewable_zones_detailed", fmt),
        figsize=(16, 10),
        legend_anchor=(0.5, -0.24),
        legend_ncol=7,
        title=(
            f"Renewable zones -- main classification (offshore: wind only)\n"
            f"solar CF {FIXED_THRESHOLDS['solar']['low']}/{FIXED_THRESHOLDS['solar']['high']}, "
            f"wind CF {FIXED_THRESHOLDS['wind_onshore']['low']}/{FIXED_THRESHOLDS['wind_onshore']['high']}, "
            f"storage {FIXED_THRESHOLDS['storage']['land']} days, "
            f"offshore wind {FIXED_THRESHOLDS['wind_offshore']['low']}/{FIXED_THRESHOLDS['wind_offshore']['high']} m/s"
        ),
    )


# def fig_abundance_map(data: DataBundle, fmt: str) -> None:
#     """Fig 2a -- abundance-only map (offshore: wind only, GROUPS_DETAILED colours)."""
#     plot_zones_map(
#         data.ds_zones_abundance_wind,
#         GROUPS_ABUNDANCE_OFFSHORE_WIND,
#         out_path=_out("fig2a_abundance_zones", fmt),
#         figsize=(15, 12),
#         legend_anchor=(0.5, -0.2),
#         legend_ncol=4,
#         title=(
#             f"Abundance zones (onshore solar CF {FIXED_THRESHOLDS['solar']['low']}/{FIXED_THRESHOLDS['solar']['high']}, "
#             f"wind CF {FIXED_THRESHOLDS['wind_onshore']['low']}/{FIXED_THRESHOLDS['wind_onshore']['high']}, "
#             f"offshore wind {FIXED_THRESHOLDS['wind_offshore']['low']}/{FIXED_THRESHOLDS['wind_offshore']['high']} m/s)"
#         ),
#     )


def fig_demand_map(data: DataBundle, fmt: str) -> None:
    """Fig 2b -- normalised demand proximity."""
    plot_map_continuous(
        data.normalized_demand,
        legend_label="Demand proximity (normalised p95)",
        cmap=plt.get_cmap("magma_r", 10),
        title="Demand proximity normalised with upper bound p95",
        path_output=_out("fig2b_demand_normalised", fmt),
    )


# def fig_storage_map(data: DataBundle, fmt: str) -> None:
#     """Fig 2c -- mean storage duration."""
#     plot_map_continuous(
#         plot_data=data.ds_mean["duration_metric"].where(data.land | data.offshore),
#         cmap=plt.get_cmap("Spectral_r", 9),
#         vmin=0,
#         vmax=45,
#         extend="max",
#         legend_label="Storage duration [days]",
#         title="Mean storage duration (1995-2025)",
#         path_output=_out("fig2c_storage_mean", fmt),
#     )


def fig_abundance_storage_combined(data: DataBundle, fmt: str) -> None:
    """Fig 2 (combined) -- abundance zones (a) + mean storage duration (b)."""
    plot_abundance_storage_combined(
        ds_abundance=data.ds_zones_abundance_wind,
        groups_abundance=GROUPS_ABUNDANCE_OFFSHORE_WIND,
        storage_data=data.ds_mean["duration_metric"].where(data.land | data.offshore),
        storage_title="Mean storage duration (1995-2025)",
        abundance_title=(
            f"Abundance zones (onshore solar CF {FIXED_THRESHOLDS['solar']['low']}/{FIXED_THRESHOLDS['solar']['high']}, "
            f"wind CF {FIXED_THRESHOLDS['wind_onshore']['low']}/{FIXED_THRESHOLDS['wind_onshore']['high']}, "
            f"offshore wind {FIXED_THRESHOLDS['wind_offshore']['low']}/{FIXED_THRESHOLDS['wind_offshore']['high']} m/s)"
        ),
        legend_ncol=4,
        out_path=_out("Fig2_abundance_storage_separate", fmt),
    )


def fig_optimal_alpha(data: DataBundle, fmt: str) -> None:
    """Fig 2d -- optimal wind share."""
    plot_map_continuous(
        data.ds_mean["optimal_alpha"],
        legend_label="Optimal share of wind CF",
        cmap=plt.get_cmap("RdBu", 10),
        title="Optimal share of wind CF (min. annual energy deficit)",
        path_output=_out("fig2d_optimal_alpha", fmt),
    )


def fig_cramers_v_land(data: DataBundle, fmt: str) -> None:
    """Fig 3 -- subgroup % + climate bars + Cramer's V (main text, detailed)."""
    plot_stat_group_climate_cramersv(
        ds_grouped=data.grouped_detailed_land,
        df_percentage=data.df_pct_detailed_land,
        groups_land=land_only_groups(GROUPS_DETAILED),
        land_colors=LAND_COLORS,
        title_suffix="Subgroups in main renewable zones",
        out_path=_out("fig3_cramers_renewable_vs_climate", fmt),
    )


def fig_scatter_poor_high(data: DataBundle, fmt: str) -> None:
    """Fig 4 -- scatter of 'poor both, high demand' countries."""
    named = [
        "Costa Rica",
        "Ecuador",
        "Gabon",
        "Vietnam",
        "Bhutan",
        "Austria",
        "Hungary",
        "Romania",
        "Croatia",
        "Switzerland",
        "Taiwan",
        "Slovenia",
        "Japan",
        "Serbia",
        "Trinidad and Tobago",
        "Peru",
        "Colombia",
        "Congo",
        "China",
        "Cyprus",
    ]
    toplot = plot_scatter_elevation_precipitation(
        data.grouped_detailed_land,
        named,
        out_path=_out("fig4_scatter_poor_high_demand", fmt),
    )

    csv_path = POST_PROCESSED_DIR / "df_stats_poor_both_high_demand.csv"
    (
        toplot[
            [
                "country_name",
                "latitude",
                "elevation",
                "precipitation",
                "P_count",
                "n_grid_cells_total",
                "percentage",
            ]
        ]
        .sort_values("percentage", ascending=False)
        .head(120)
        .reset_index(drop=True)
        .to_csv(str(csv_path), float_format="%.2f", index=False)
    )
    log.info("Saved: %s", csv_path)


def fig_full_zones_map(data: DataBundle, fmt: str) -> None:
    """Fig S1 -- full-classification zone map (offshore uses wind + solar)."""
    plot_zones_map(
        data.ds_zones_full,
        GROUPS_FULL,
        out_path=_out("figS1_renewable_zones_full", fmt),
        figsize=(16, 10),
        legend_anchor=(0.5, -0.25),
        legend_ncol=6,
        title=(
            f"Renewable zones -- full classification (offshore: wind + solar)\n"
            f"solar CF {FIXED_THRESHOLDS['solar']['low']}/{FIXED_THRESHOLDS['solar']['high']}, "
            f"wind CF {FIXED_THRESHOLDS['wind_onshore']['low']}/{FIXED_THRESHOLDS['wind_onshore']['high']}, "
            f"storage {FIXED_THRESHOLDS['storage']['land']} days, "
            f"offshore wind {FIXED_THRESHOLDS['wind_offshore']['low']}/{FIXED_THRESHOLDS['wind_offshore']['high']} m/s, "
            f"offshore solar {FIXED_THRESHOLDS['solar_offshore']['low']:.0f}/{FIXED_THRESHOLDS['solar_offshore']['high']:.0f} W/m2"
        ),
    )


def fig_resource_availability(data: DataBundle, fmt: str) -> None:
    """Fig S -- normalised resource-availability map."""
    plot_map_continuous(
        data.ds_res_avail["resource_availability"],
        vmin=0,
        vmax=1,
        cmap=plt.get_cmap("RdYlGn", 3),
        legend_label="Resource abundance [normalised]",
        title="Resource availability (max abundance)",
        path_output=_out("figS_resource_availability", fmt),
    )


def fig_abundance_histograms(data: DataBundle, fmt: str) -> None:
    """Fig Sa -- histograms of abundance variables with classification thresholds."""
    plot_abundance_histograms(
        ds=data.ds_processed,
        land=data.land,
        offshore=data.offshore,
        thresholds=data.threshold,
        out_path=_out("figSa_abundance_histograms", fmt),
    )


def fig_storage_histograms(data: DataBundle, fmt: str) -> None:
    """Fig Sb -- histograms of storage duration with classification threshold."""
    plot_storage_histograms(
        storage_da=data.ds_mean["duration_metric"],
        land=data.land,
        offshore=data.offshore,
        thresholds=data.threshold,
        out_path=_out("figSb_storage_histograms", fmt),
    )


def fig_zones_climatology(data: DataBundle, fmt: str) -> None:
    """Fig S -- abundance zones classified from wind/solar climatology (p33/p67 thresholds)."""
    plot_zones_climatology(
        ds=data.ds_processed,
        groups=GROUPS_ABUNDANCE_OFFSHORE_WIND,
        land=data.land,
        offshore=data.offshore,
        out_path=_out("figS_zones_climatology", fmt),
        figsize=(16, 10),
        legend_anchor=(0.5, -0.2),
        legend_ncol=4,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FIGURES: dict[str, Callable[[DataBundle, str], None]] = {
    "fig1": fig_detailed_zones_map,
    # "fig2a": fig_abundance_map,
    "fig2b": fig_demand_map,
    # "fig2c": fig_storage_map,
    "fig2d": fig_optimal_alpha,
    "fig2_abundance_storage": fig_abundance_storage_combined,
    "fig3": fig_cramers_v_land,
    "fig4": fig_scatter_poor_high,
    "figS1_full_zones": fig_full_zones_map,
    "figS_resource": fig_resource_availability,
    "figSa_abundance_histograms": fig_abundance_histograms,
    "figSb_storage_histograms": fig_storage_histograms,
    "figS_zones_climatology": fig_zones_climatology,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--format",
        choices=["pdf", "png"],
        default="pdf",
        help="Output image format (default: pdf).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help=f"Only run these figure names. Available: {', '.join(FIGURES)}",
    )
    # parser.add_argument(
    #     "--continue-on-error",
    #     action="store_true",
    #     help="Skip figures that raise instead of aborting the whole run.",
    # )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    targets = args.only or list(FIGURES)
    unknown = [t for t in targets if t not in FIGURES]
    if unknown:
        parser.error(f"Unknown figure(s): {unknown}. Available: {list(FIGURES)}")

    log_path = setup_logging(verbose=args.verbose)
    log.info("=" * 60)
    log.info("figures_renewable_zones.py run started")
    log.info("Log file:         %s", log_path)
    log.info("Output directory: %s", FIG_DIR)
    log.info("Output format:    %s", args.format)
    log.info("Figures to build: %s", targets)
    log.info("=" * 60)

    data = DataBundle()
    failed: list[str] = []
    for name in targets:
        log.info("-- Building %s --", name)
        try:
            FIGURES[name](data, args.format)
            plt.close("all")
            log.info("%s built successfully", name)
        except Exception as exc:
            failed.append(name)
            log.error("%s failed: %s", name, exc)
            log.debug("Traceback:\n%s", traceback.format_exc())
            # if not args.continue_on_error:
            #     log.info("Aborting (use --continue-on-error to skip failures).")
            #     return 1

    data.save_post_processed()

    if failed:
        log.warning("Done, but %d figure(s) failed: %s", len(failed), failed)
        return 2
    log.info("All figures built successfully. Log saved to %s", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
