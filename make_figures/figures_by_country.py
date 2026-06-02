"""
figures_by_country.py
=====================

Part 2 -- country-level analysis figures.

Reads derived data from results/post_processed_data/ (produced by
figures_renewable_zones.py --save-post-processed):
  resource_availability.nc  — ds_res_avail
  normalized_demand.nc      — normalized_demand
  storage_mean.nc           — ds_mean
  zones_detailed.nc         — ds_zones_detailed

Also reads:
  results/automatic/abundance/abundance_*.nc   — wind_CF for land/offshore masks

Run:
  python make_figures/figures_by_country.py
  python make_figures/figures_by_country.py --only fig5
"""

from __future__ import annotations

import argparse
import colorsys
import logging
import sys
import traceback
import warnings
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
    GROUPS_DETAILED,
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

LOG_DIR = FIG_DIR / "logs"


def setup_logging(verbose: bool = False) -> Path:
    """Configure logging to both console and a timestamped file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"figures_country_{timestamp}.log"

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


log = logging.getLogger("figures_by_country")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LIST_CNT_HIGHLIGHT = [
    "CHN",
    "IRN",
    "MAR",
    "DNK",
    "USA",
    "ALG",
    "MLI",
    "ESP",
    "JPN",
    "VNM",
    "IND",
    "IDN",
    "GRC",
    "ITA",
]

CLUSTER_NAMING_RULES = {
    "high resource": ("avg_resource", ">=", 0.7),
    "modest resource": ("avg_resource", "<", 0.3),
    "long storage": ("avg_storage", ">=", 0.5),
    "short storage": ("avg_storage", "<", 0.3),
    "high mismatch": ("resource_demand_corr", "<", -0.2),
    "match": ("resource_demand_corr", ">=", 0.1),
}


# ---------------------------------------------------------------------------
# Country utility functions (copied verbatim from plot_utils.py)
# ---------------------------------------------------------------------------

# Display order for base groups in the by-cluster bar chart.
# Land groups first (B → P), offshore groups last.
ZONE_ORDER_CLUSTER = ["B", "W", "Ws", "S", "Sw", "P", "O", "o"]


def _expand_wildcards(label: str, replacements=("U", "R")) -> list[str]:
    choices = [replacements if ch == "x" else (ch,) for ch in label]
    return ["".join(p) for p in product(*choices)]


def plot_zones_by_cluster_from_grid(
    ds_zones: xr.Dataset,
    cluster_grid: xr.DataArray,
    cluster_config: dict,
    groups: dict,
    *,
    figsize: tuple = (10, 8),
    out_path: Optional[str] = None,
) -> "plt.Figure":
    """2×2 bar plots of renewable-zone composition per cluster.

    Each of the four panels shows what share of that cluster's grid cells falls
    into each base renewable-zone group (land + offshore), normalised to 100%
    within the cluster.  Colours match *groups*.

    Parameters
    ----------
    ds_zones : xr.Dataset
        Zones dataset (must contain ``"zones"`` variable on the same grid as
        *cluster_grid*).
    cluster_grid : xr.DataArray
        Per-cell cluster ID (float, NaN for unassigned) — output of
        ``prepare_cluster_map``.
    cluster_config : dict
        ``{cluster_id: [hex_colour, label]}`` — same as Fig 5.
    groups : dict
        Colour/label/pattern groups (e.g. ``GROUPS_DETAILED``).
    """
    # ── 1. zone label → base group ───────────────────────────────────────────
    label_to_base: dict[str, str] = {}
    for code, (_col, _lab, patterns) in groups.items():
        base = code.split("_")[0]
        for pat in patterns:
            for expanded in _expand_wildcards(pat):
                label_to_base[expanded] = base

    # ── 2. flat arrays aligned on the grid ──────────────────────────────────
    zones_arr = ds_zones["zones"].values.ravel()  # str labels
    cluster_arr = cluster_grid.values.ravel()  # float / NaN

    base_arr = np.array([label_to_base.get(z, None) for z in zones_arr], dtype=object)

    # ── 3. figure / axes ────────────────────────────────────────────────────
    sorted_ids = sorted(cluster_config.keys())
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes_flat = axes.flatten()

    # colour map: base group → colour (use first matching entry in groups)
    base_colour: dict[str, str] = {}
    for code, (col, _lab, _pats) in groups.items():
        base = code.split("_")[0]
        if base not in base_colour:
            base_colour[base] = col

    # display order: land groups then offshore
    all_bases = list(
        dict.fromkeys([g.split("_")[0] for g in groups])  # preserves insertion order
    )
    display_order = [b for b in ZONE_ORDER_CLUSTER if b in all_bases] + [
        b for b in all_bases if b not in ZONE_ORDER_CLUSTER
    ]

    panel_labels = ["a", "b", "c", "d"]

    for ax_idx, cid in enumerate(sorted_ids):
        ax = axes_flat[ax_idx]
        cluster_colour, cluster_label = cluster_config[cid]

        mask = np.isfinite(cluster_arr) & (cluster_arr == cid)
        bases_in_cluster = base_arr[mask]
        total = np.sum(mask)

        if total == 0:
            ax.set_visible(False)
            continue

        # count and normalise
        pct = {}
        for b in display_order:
            pct[b] = float(np.sum(bases_in_cluster == b)) / total * 100

        x_labels = [b for b in display_order if pct.get(b, 0) > 0]
        x_vals = [pct[b] for b in x_labels]
        colours = [base_colour.get(b, "#CCCCCC") for b in x_labels]

        bars = ax.bar(x_labels, x_vals, color=colours, edgecolor="none", width=0.6)

        # cluster-colour spine + title
        for spine in ax.spines.values():
            spine.set_edgecolor(cluster_colour)
            spine.set_linewidth(2.5)

        ax.set_title(cluster_label, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("% of cluster grid cells")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0, labelsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(x_vals) * 1.18)

        # value labels on bars
        for bar, val in zip(bars, x_vals):
            if val >= 1.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )

        # panel letter
        ax.text(
            -0.08,
            1.04,
            panel_labels[ax_idx],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    fig.suptitle(
        "Renewable zone composition by cluster",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info("Saved: %s", out_path)

    return fig


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
    import matplotlib.lines as mlines

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
        from plot_utils import dirty

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
        mpatches.Patch(color=None, label="Resource quality", fill=False, linewidth=0)
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


# def plot_country_clusters_3d(
#     df: pd.DataFrame,
#     df_centroids: pd.DataFrame,
#     cluster_map: dict,
#     *,
#     list_cnt_to_plot: Optional[Sequence[str]] = None,
#     fig_size: tuple = (900, 900),
#     out_path: Optional[str] = None,
#     camera_eye: tuple = (1.6, 1.6, 1.1),
#     show: bool = False,
# ):
#     """Interactive 3D scatter of country clusters.

#     Axes
#     ----
#     * X — Average resource availability
#     * Y — Spatial correlation (resource vs demand)
#     * Z — Average storage requirement

#     Parameters
#     ----------
#     df
#         Per-country dataframe with ``avg_resource``, ``avg_storage``,
#         ``resource_demand_corr``, ``cluster``, ``iso3``, ``country_name``.
#     df_centroids
#         Cluster centroids dataframe (output of :func:`cluster_countries`).
#     cluster_map
#         ``{cluster_id: [hex_colour, label]}`` — same structure used by
#         :func:`plot_combined_analysis`. Drives marker colours and the legend.
#     list_cnt_to_plot
#         ISO-3 codes whose names should be drawn next to their marker (as in
#         Fig 5a). If ``None``, no labels are drawn.
#     fig_size
#         ``(width, height)`` in pixels.
#     out_path
#         If given, save a static image using Kaleido. ``.pdf`` / ``.png`` /
#         ``.svg`` are all supported. The HTML version is also written next to
#         it for interactive exploration.
#     camera_eye
#         Initial camera position — tweak if the default view crops labels.
#     show
#         Whether to call ``fig.show()``. Defaults to ``False`` so the script
#         can run headless; set ``True`` to display in a notebook.
#     """
#     import plotly.graph_objects as go

#     df = df.copy()
#     df_centroids = df_centroids.copy()

#     cluster_ids = sorted(cluster_map.keys())

#     fig = go.Figure()

#     # One trace per cluster — cluster markers first, labels drawn last so
#     # they are never occluded by centroid X markers in the depth sort.
#     for cid in cluster_ids:
#         colour, label = cluster_map[cid]
#         sub = df[df["cluster"] == cid]
#         if sub.empty:
#             continue
#         fig.add_trace(
#             go.Scatter3d(
#                 x=sub["avg_resource"],
#                 y=sub["resource_demand_corr"],
#                 z=sub["avg_storage"],
#                 mode="markers",
#                 marker=dict(
#                     size=9,
#                     color=colour,
#                     opacity=0.80,
#                     line=dict(width=0.8, color="white"),
#                 ),
#                 name=str(label),
#                 text=sub["iso3"],
#                 hovertemplate=(
#                     "<b>%{text}</b><br>"
#                     "Cluster: " + str(label) + "<br>"
#                     "Resource: %{x:.3f}<br>"
#                     "Correlation: %{y:.3f}<br>"
#                     "Storage: %{z:.3f}<extra></extra>"
#                 ),
#             )
#         )

#     # Centroids: one 'X' marker per cluster, drawn before labels.
#     fig.add_trace(
#         go.Scatter3d(
#             x=df_centroids["avg_resource"],
#             y=df_centroids["resource_demand_corr"],
#             z=df_centroids["avg_storage"],
#             mode="markers",
#             marker=dict(
#                 symbol="x",
#                 size=12,
#                 color=[cluster_map[i][0] for i in range(len(df_centroids))],
#                 line=dict(width=4, color="black"),
#             ),
#             name="Cluster centroids",
#             hoverinfo="skip",
#         )
#     )

#     # Named-country labels — added LAST so they render on top of all markers.
#     if list_cnt_to_plot:
#         to_label = df[df["iso3"].isin(list_cnt_to_plot)]
#         if not to_label.empty:
#             names = to_label.get("country_name", to_label["iso3"])
#             fig.add_trace(
#                 go.Scatter3d(
#                     x=to_label["avg_resource"],
#                     y=to_label["resource_demand_corr"],
#                     z=to_label["avg_storage"],
#                     # markers+text: tiny invisible anchor forces correct depth
#                     # sort so the text is never hidden behind centroid X marks.
#                     mode="markers+text",
#                     marker=dict(size=1, opacity=0, color="rgba(0,0,0,0)"),
#                     text=names,
#                     textposition="top center",
#                     textfont=dict(
#                         size=12, color="black", family="Arial Black, Arial, sans-serif"
#                     ),
#                     showlegend=False,
#                     hoverinfo="skip",
#                 )
#             )

#     # Scene occupies the left ~82 % of the canvas; legend sits tight beside it.
#     fig.update_layout(
#         title=dict(
#             text=(
#                 "<b>Global Renewable Strategy Clusters (3D view)</b><br>"
#                 "<sup>Storage, resource–demand correlation, and resource availability</sup>"
#             ),
#             x=0.04,
#             y=0.96,
#             yanchor="top",
#             font=dict(size=15),
#         ),
#         width=fig_size[0],
#         height=fig_size[1],
#         scene=dict(
#             domain=dict(x=[0.0, 0.82], y=[0.0, 1.0]),
#             xaxis=dict(
#                 title=dict(text="Avg resource availability", font=dict(size=13)),
#                 tickfont=dict(size=11),
#             ),
#             yaxis=dict(
#                 title=dict(
#                     text="Spatial correlation<br>(resource vs demand)",
#                     font=dict(size=13),
#                 ),
#                 tickfont=dict(size=11),
#             ),
#             zaxis=dict(
#                 title=dict(text="Avg storage [norm.]", font=dict(size=13)),
#                 tickfont=dict(size=11),
#             ),
#             camera=dict(
#                 up=dict(x=0, y=0, z=1),
#                 center=dict(x=0, y=0, z=0),
#                 eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]),
#             ),
#             aspectmode="cube",
#         ),
#         template="plotly_white",
#         legend=dict(
#             yanchor="top",
#             y=1,
#             xanchor="left",
#             x=0.6,
#             bgcolor="rgba(255,255,255,0.7)",
#             bordercolor="lightgrey",
#             borderwidth=1,
#             font=dict(size=12),
#         ),
#         margin=dict(l=10, r=10, b=10, t=55),
#     )

#     if out_path:
#         out_path = Path(out_path)
#         out_path.parent.mkdir(parents=True, exist_ok=True)

#         # Static export — requires kaleido.  Check availability before
#         # attempting so we emit a clear warning rather than a cryptic error,
#         # and always fall back to writing the interactive HTML.
#         _kaleido_ok = False
#         try:
#             import kaleido  # noqa: F401  (existence check only)

#             _kaleido_ok = True
#         except ImportError:
#             log.warning(
#                 "kaleido is not installed — static 3D figure (%s) cannot be "
#                 "saved as %s.  Install kaleido (e.g. `pip install kaleido`) "
#                 "to enable static export.  The interactive HTML will still be "
#                 "written.",
#                 out_path.suffix.lstrip(".").upper(),
#                 out_path,
#             )

#         if _kaleido_ok:
#             try:
#                 fig.write_image(
#                     str(out_path), width=fig_size[0], height=fig_size[1], scale=2
#                 )
#                 log.info("Saved 3D cluster figure: %s", out_path)
#             except Exception as exc:
#                 log.warning("Static 3D export failed for %s: %s", out_path, exc)

#         html_path = out_path.with_suffix(".html")
#         fig.write_html(str(html_path))
#         log.info("Saved interactive 3D cluster figure: %s", html_path)

#     if show:
#         fig.show()

#     return fig


def plot_offshore_shift_density(
    df_land: pd.DataFrame,
    df_combined: pd.DataFrame,
    cluster_info: pd.DataFrame,
    cluster_config: dict,
    *,
    figsize: tuple = (10, 4),
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
        fontsize=10,
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
# DataBundle — lazy cache
# ---------------------------------------------------------------------------


class DataBundle:
    """Lazy holder for all datasets the country figures share.

    Derived gridded data (resource availability, demand, storage, zones) is
    loaded from post_processed_data/ rather than recomputed.
    """

    def __init__(self) -> None:
        self._wind_cf = None
        self._ds_res_avail = None
        self._normalized_demand = None
        self._ds_mean = None
        self._ds_zones_detailed = None
        self._ds_corr = None
        self._shapes_ref = None
        self._df_corr_results = None
        self._ds_corr_land = None
        self._shapes_land = None
        self._df_corr_land = None

    # -- land / offshore masks (from abundance files) --------------------
    @property
    def _wind_CF(self) -> xr.DataArray:
        if self._wind_cf is None:
            pattern = str(RESULTS_DIR / "automatic/abundance/abundance_*.nc")
            log.info("Loading wind_CF for masks from %s", pattern)
            self._wind_cf = xr.open_mfdataset(pattern, combine="by_coords")["wind_CF"]
        return self._wind_cf

    @property
    def land(self):
        return mask_land(self._wind_CF)

    @property
    def offshore(self):
        return mask_offshore(self._wind_CF)

    # -- post-processed derived data ------------------------------------
    @property
    def ds_res_avail(self) -> xr.Dataset:
        if self._ds_res_avail is None:
            path = POST_PROCESSED_DIR / "resource_availability.nc"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run figures_renewable_zones.py "
                    "--save-post-processed first."
                )
            log.info("Loading resource_availability from %s", path)
            self._ds_res_avail = xr.open_dataset(str(path))
            log_ds_summary(
                self._ds_res_avail,
                "ds_res_avail",
                variables=["resource_availability"],
            )
        return self._ds_res_avail

    @property
    def normalized_demand(self) -> xr.DataArray:
        if self._normalized_demand is None:
            path = POST_PROCESSED_DIR / "normalized_demand.nc"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run figures_renewable_zones.py "
                    "--save-post-processed first."
                )
            log.info("Loading normalized_demand from %s", path)
            self._normalized_demand = xr.open_dataset(str(path))["normalized_demand"]
        return self._normalized_demand

    @property
    def ds_mean(self) -> xr.Dataset:
        if self._ds_mean is None:
            path = POST_PROCESSED_DIR / "storage_mean.nc"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run figures_renewable_zones.py "
                    "--save-post-processed first."
                )
            log.info("Loading storage_mean from %s", path)
            self._ds_mean = xr.open_dataset(str(path))
            log_ds_summary(
                self._ds_mean, "ds_mean", variables=["duration_metric", "optimal_alpha"]
            )
        return self._ds_mean

    @property
    def ds_zones_detailed(self) -> xr.Dataset:
        if self._ds_zones_detailed is None:
            path = POST_PROCESSED_DIR / "zones_detailed.nc"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run figures_renewable_zones.py "
                    "--save-post-processed first."
                )
            log.info("Loading zones_detailed from %s", path)
            self._ds_zones_detailed = xr.open_dataset(str(path))
        return self._ds_zones_detailed

    # -- shared helpers --------------------------------------------------------
    def _valid_iso3(self) -> list[str]:
        import regionmask as rm

        countries = rm.defined_regions.natural_earth_v5_0_0.countries_110
        iso3 = [to_iso3_robust(a) for a in countries.to_geodataframe().abbrevs]
        valid = [x for x in iso3 if x]
        valid += [
            "SWE",
            "BEL",
            "LUX",
            "CHE",
            "KHM",
            "LAO",
            "PRT",
            "AUT",
            "JAM",
            "ISR",
            "MLT",
        ]
        return sorted(set(valid))

    def _build_corr_ds(
        self,
        shapes_parquet: str,
        shape_class: str | None = None,
        label: str = "ds_corr",
    ):
        """Build the resource/demand/storage dataset masked by country shapes."""
        import geopandas as gpd
        import regionmask as rm
        import yaml
        from shapely.geometry import box as shapely_box

        dir_geo = RESOURCES_DIR / "user/module_geo_boundaries/global"
        log.info(
            "Building %s (shapes=%s, shape_class=%s)",
            label,
            shapes_parquet,
            shape_class,
        )

        shapes = gpd.read_parquet(shapes_parquet)
        if shape_class is not None:
            shapes = shapes[shapes["shape_class"] == shape_class].copy()

        with open(str(dir_geo / "global_config.yaml")) as f:
            config = yaml.safe_load(f)
        country_list = list(config["module_geo_boundaries"]["countries"].keys())
        shapes = shapes[shapes["country_id"].isin(country_list)].copy()
        shapes = shapes.reset_index(drop=True)
        if shapes.crs != "EPSG:4326":
            shapes = shapes.to_crs("EPSG:4326")
        shapes.loc[shapes["country_id"] == "USA", "geometry"] = gpd.clip(
            shapes[shapes["country_id"] == "USA"], shapely_box(-130, 20, -60, 55)
        ).geometry.values

        regions = rm.from_geopandas(shapes, names="country_id", abbrevs="country_id")
        ds = xr.Dataset({"resource": self.ds_res_avail["resource_availability"]})
        ds = ds.sortby("longitude")
        ds["country_maritime"] = regions.mask(
            ds["longitude"], ds["latitude"], wrap_lon=False
        )
        ds["demand"] = self.normalized_demand
        ds["storage"] = self.ds_mean["duration_metric"]
        log.info("%s: shapes loaded (%d rows)", label, len(shapes))
        log_ds_summary(ds, label, variables=["resource", "demand", "storage"])
        return ds, shapes

    # -- combined shapes (land + offshore) ------------------------------------
    @property
    def ds_corr(self):
        if self._ds_corr is None:
            dir_geo = RESOURCES_DIR / "user/module_geo_boundaries/global"
            self._ds_corr, self._shapes_ref = self._build_corr_ds(
                shapes_parquet=str(
                    dir_geo / "results/global/results/shapes_combined.parquet"
                ),
                shape_class=None,
                label="ds_corr",
            )
        return self._ds_corr

    @property
    def shapes_ref(self):
        _ = self.ds_corr
        return self._shapes_ref

    @property
    def df_corr_results(self):
        if self._df_corr_results is None:
            valid = self._valid_iso3()
            log.info(
                "Running per-country spatial correlation for %d countries", len(valid)
            )
            self._df_corr_results = analyze_country_spatial_correlation(
                self.ds_corr,
                self.shapes_ref,
                valid,
                method="spearman",
            )
            log_df_summary(
                self._df_corr_results,
                "df_corr_results",
                numeric_cols=[
                    "resource_demand_corr",
                    "avg_resource",
                    "avg_demand",
                    "avg_storage",
                    "n_pixels",
                ],
            )
            pos = int((self._df_corr_results["resource_demand_corr"] > 0).sum())
            neg = int((self._df_corr_results["resource_demand_corr"] < 0).sum())
            log.info("Country correlation sign: positive=%d, negative=%d", pos, neg)
        return self._df_corr_results

    # -- land-only shapes -----------------------------------------------------
    @property
    def ds_corr_land(self):
        if self._ds_corr_land is None:
            dir_geo = RESOURCES_DIR / "user/module_geo_boundaries/global"
            self._ds_corr_land, self._shapes_land = self._build_corr_ds(
                shapes_parquet=str(dir_geo / "results/global/results/shapes.parquet"),
                shape_class="land",
                label="ds_corr_land",
            )
        return self._ds_corr_land

    @property
    def shapes_land(self):
        _ = self.ds_corr_land
        return self._shapes_land

    @property
    def df_corr_land(self):
        if self._df_corr_land is None:
            valid = self._valid_iso3()
            log.info(
                "Running per-country spatial correlation (land only) for %d countries",
                len(valid),
            )
            self._df_corr_land = analyze_country_spatial_correlation(
                self.ds_corr_land,
                self.shapes_land,
                valid,
                method="spearman",
            )
            log_df_summary(self._df_corr_land, "df_corr_land")
        return self._df_corr_land


# ---------------------------------------------------------------------------
# Shared cluster builder
# ---------------------------------------------------------------------------


def _build_clusters(data: DataBundle, n: int = 4):
    """Cluster countries and build the label/config dict."""
    df = data.df_corr_results.copy()
    all_metric_path = POST_PROCESSED_DIR / "country_all_metrics.csv"
    df.to_csv(str(all_metric_path))
    log.info("Country all metrics saved: %s", all_metric_path)

    df["avg_storage"] = (df["avg_storage"] - df["avg_storage"].min()) / (
        df["avg_storage"].max() - df["avg_storage"].min()
    )
    df_clustered, centroids = cluster_countries(df, n_clusters=n)

    order = centroids["resource_demand_corr"].argsort().values
    remap = {old: new for new, old in enumerate(order)}
    df_clustered["cluster"] = df_clustered["cluster"].map(remap)
    centroids = centroids.iloc[order].reset_index(drop=True)

    _pos_to_id = {0: 2, 1: 0, 2: 1, 3: 3}
    df_clustered["cluster"] = df_clustered["cluster"].map(_pos_to_id)
    centroids = centroids.iloc[[1, 2, 0, 3]].reset_index(drop=True)

    cluster_colors = ["#5EC478", "#7182F3", "#FFD454", "#FF5454"]
    labels = [
        "Cluster I -- high resource / short storage / high mismatch",
        "Cluster II -- short storage / moderate resource",
        "Cluster III -- long storage / high mismatch",
        "Cluster IV -- low resource / match",
    ]
    cluster_config = {i: [cluster_colors[i], labels[i]] for i in range(n)}

    log_df_summary(
        df_clustered,
        "df_clustered",
        numeric_cols=["avg_storage", "resource_demand_corr", "avg_resource", "cluster"],
    )
    log_df_summary(centroids, "centroids")

    cluster_info = create_cluster_summary_table_full_names(
        df_clustered,
        data.shapes_ref,
        centroids,
        naming_rules=CLUSTER_NAMING_RULES,
        large_threshold=400,
    )
    csv_path = POST_PROCESSED_DIR / "cluster_summary.csv"
    cluster_info.to_csv(str(csv_path))
    log.info("Cluster summary saved: %s", csv_path)

    return df_clustered, centroids, cluster_config


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------


def _out(name: str, fmt: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return str(FIG_DIR / f"{name}.{fmt}")


def fig_clusters_combined(data: DataBundle, fmt: str) -> None:
    """Fig 5 -- cluster scatter + map."""
    df_clustered, centroids, cluster_config = _build_clusters(data, n=4)
    cluster_grid = prepare_cluster_map(data.ds_corr, df_clustered, data.shapes_ref)
    plot_combined_analysis(
        df_clustered,
        centroids,
        cluster_grid,
        cluster_config,
        list_cnt_to_plot=LIST_CNT_HIGHLIGHT,
        path_output=_out("fig5_clusters_combined", fmt),
    )


# def fig_clusters_3d(data: DataBundle, fmt: str) -> None:
#     """Fig 6 -- 3D scatter of country clusters."""
#     df_clustered, centroids, cluster_config = _build_clusters(data, n=4)
#     plot_country_clusters_3d(
#         df_clustered,
#         centroids,
#         cluster_map=cluster_config,
#         list_cnt_to_plot=LIST_CNT_HIGHLIGHT,
#         fig_size=(1400, 900),
#         out_path=_out("fig6_clusters_3d", fmt),
#     )


def fig_spatial_correlation_map(data: DataBundle, fmt: str) -> None:
    """Fig S -- country-level resource-demand Spearman correlation."""
    plot_corr = prepare_metric_map(
        data.ds_corr,
        data.df_corr_results,
        data.shapes_ref,
        column="resource_demand_corr",
    )
    plot_map_continuous(
        plot_corr,
        legend_label="Spearman rho (Resource vs Demand)",
        cmap="RdBu",
        title="Global spatial correlation: renewables vs demand",
        vmin=-1,
        vmax=1,
        levels=11,
        path_output=_out("figS_spatial_correlation", fmt),
    )


def fig_offshore_wind_shift(data: DataBundle, fmt: str) -> None:
    """Fig S -- KDE density of per-country metric shifts when adding offshore wind."""
    df_land = data.df_corr_land.copy()
    df_combined = data.df_corr_results.copy()

    land_metric_path = POST_PROCESSED_DIR / "country_land_only_metrics.csv"
    df_land.to_csv(str(land_metric_path))
    log.info("Land-only metrics saved: %s", land_metric_path)

    s_min = df_combined["avg_storage"].min()
    s_max = df_combined["avg_storage"].max()
    for df in (df_land, df_combined):
        df["avg_storage"] = (df["avg_storage"] - s_min) / (s_max - s_min)

    df_clustered, _, cluster_config = _build_clusters(data, n=4)

    plot_offshore_shift_density(
        df_land=df_land,
        df_combined=df_combined,
        cluster_info=df_clustered[["iso3", "cluster"]],
        cluster_config=cluster_config,
        out_path=_out("figS_offshore_wind_shift", fmt),
    )


def fig_correlation_histogram(data: DataBundle, fmt: str) -> None:
    """Fig S -- histogram of country correlations."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        data.df_corr_results["resource_demand_corr"].dropna(),
        bins=25,
        density=True,
        color="#4a53ff",
        alpha=0.8,
    )
    ax.set_xlabel("Spearman correlation")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of renewable-demand spatial correlation")
    fig.tight_layout()
    fig.savefig(_out("figS_correlation_histogram", fmt), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_zones_by_cluster(data: DataBundle, fmt: str) -> None:
    """Fig Sup -- 2x2 bar plots of renewable-zone composition per cluster (I-IV)."""
    df_clustered, _, cluster_config = _build_clusters(data, n=4)
    cluster_grid = prepare_cluster_map(data.ds_corr, df_clustered, data.shapes_ref)
    plot_zones_by_cluster_from_grid(
        ds_zones=data.ds_zones_detailed,
        cluster_grid=cluster_grid,
        cluster_config=cluster_config,
        groups=GROUPS_DETAILED,
        out_path=_out("Sup_zones_by_clusters", fmt),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FIGURES: dict[str, Callable[[DataBundle, str], None]] = {
    "fig5": fig_clusters_combined,
    # "fig6": fig_clusters_3d,
    "figS_corr_map": fig_spatial_correlation_map,
    "figS_corr_hist": fig_correlation_histogram,
    "figS_offshore_shift": fig_offshore_wind_shift,
    "Sup_zones_by_clusters": fig_zones_by_cluster,
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
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip figures that raise instead of aborting the whole run.",
    )
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
    log.info("figures_by_country.py run started")
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
            if not args.continue_on_error:
                log.info("Aborting (use --continue-on-error to skip failures).")
                return 1

    if failed:
        log.warning("Done, but %d figure(s) failed: %s", len(failed), failed)
        return 2
    log.info("All figures built successfully. Log saved to %s", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
