"""
make_figures.py
===============

Reproduce every manuscript figure from Ho-Tran & Pfenninger-Lee (2026)
"Köppen-style renewable zones" in one command::

    python src/make_figures.py                      # default: PDF, all figures
    python src/make_figures.py --format png         # PNG instead of PDF
    python src/make_figures.py --only fig1 fig6     # subset

All figures are written to ``RESULTS_DIR / "figures/main"``. A timestamped
log file is always written to ``RESULTS_DIR / "figures/main/logs"``.

Main vs. supplementary classification
-------------------------------------
* **Main text** uses the *detailed* classification (``SPEC_DETAILED`` +
  ``GROUPS_DETAILED``) — offshore regions are classified on wind abundance
  **only**, no solar. This is the scheme used for Fig 1, Fig 3 (Cramér's V),
  Fig 4 (poor-both scatter), Fig 5 (clusters) and Fig 6 (3D clusters).
* **Supplementary** figures reuse the *full* classification
  (``SPEC_FULL`` + ``GROUPS_FULL``) where offshore considers both wind and
  solar, and the *abundance* classification used in the deconstructed maps
  of Fig 2.

Inputs expected
---------------
* ``results/automatic/processed/*.nc``     — ``solar_CF``, ``wind_CF``,
                                              ``wind_climatology``,
                                              ``solar_climatology``
* ``results/automatic/demand/*.nc``        — demand proximity
* ``results/automatic/storage/lds_*.nc``   — storage outputs
* ``resources/user/koeppen-geiger-world-map/...`` — Köppen overlay
* ``resources/user/module_geo_boundaries/...``    — country shapes
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Support running as `python src/make_figures.py` or `python -m src.make_figures`
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_utils as pu
from plot_utils import (
    GROUPS_ABUNDANCE,
    GROUPS_ABUNDANCE_OFFSHORE_WIND,
    GROUPS_DETAILED,
    GROUPS_FULL,
    LAND_COLORS,
    SPEC_ABUNDANCE,
    SPEC_ABUNDANCE_OFFSHORE_WIND,
    SPEC_DETAILED,
    SPEC_FULL,
    add_country_mask,
    analyze_country_spatial_correlation,
    classify_zones,
    cluster_countries,
    convert_zones_to_resource_availability,
    get_base_and_subgroup,
    land_only_groups,
    log_df_summary,
    log_ds_summary,
    mask_land,
    mask_offshore,
    plot_abundance_storage_combined,
    plot_combined_analysis,
    plot_country_clusters_3d,
    plot_map_continuous,
    plot_scatter_elevation_precipitation,
    plot_stat_group_climate_cramersv,
    plot_zones_map,
    prepare_cluster_map,
    prepare_metric_map,
    to_iso3_robust,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR.parent / "resources"
RESULTS_DIR = BASE_DIR.parent / "results"
FIG_DIR = RESULTS_DIR / "figures" / "main"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = FIG_DIR / "logs"


def setup_logging(verbose: bool = False) -> Path:
    """Configure logging to both console and a timestamped file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"make_figures_{timestamp}.log"

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


log = logging.getLogger("make_figures")


# ---------------------------------------------------------------------------
# Threshold constants — edit here to change the classification
# ---------------------------------------------------------------------------
# Fixed capacity-factor thresholds for solar (annual-mean CF) and onshore
# wind (annual-mean CF).  Storage threshold is in days of duration.
#
# NOTE: Offshore wind and solar (when applicable) always use quantile
# thresholds (0.33 / 0.66) derived from their own climatology — those are
# set inside classify_zones() and are NOT controlled here.
FIXED_THRESHOLDS = {
    "solar": {"high": 0.20, "low": 0.15},  # annual-mean CF
    "wind_onshore": {"high": 0.35, "low": 0.25},  # annual-mean CF
    "wind_offshore": {"low": 7.0, "high": 8.5},  # m/s wind-speed climatology
    "solar_offshore": {"low": 150.0, "high": 225.0},  # W/m² solar climatology
    "storage": {"land": 14, "offshore": 14},  # days of storage duration
}

# ---------------------------------------------------------------------------
# Cluster naming rules — applied to centroid values to label each cluster
# ---------------------------------------------------------------------------
# Each entry: label → (metric_column, operator, threshold)
# Multiple matching labels are joined with " / ".
# Operators: ">=" or "<".
CLUSTER_NAMING_RULES = {
    "high resource": ("avg_resource", ">=", 0.7),
    "modest resource": ("avg_resource", "<", 0.3),
    "long storage": ("avg_storage", ">=", 0.5),
    "short storage": ("avg_storage", "<", 0.3),
    "high mismatch": ("resource_demand_corr", "<", -0.2),
    "match": ("resource_demand_corr", ">=", 0.1),
}


# ---------------------------------------------------------------------------
# DataBundle — lazy cache of all shared inputs & derived datasets
# ---------------------------------------------------------------------------


class DataBundle:
    """Lazy holder for all datasets the figures share."""

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
        self._grouped_full_land = None
        self._df_pct_full_land = None
        self._ds_res_avail = None
        self._ds_corr = None
        self._shapes_ref = None
        self._df_corr_results = None

    # -- primary inputs --------------------------------------------------
    @property
    def ds_processed(self) -> xr.Dataset:
        if self._ds_processed is None:
            pattern = str(RESULTS_DIR / "automatic/processed/*.nc")
            log.info("Loading processed inputs from %s", pattern)
            self._ds_processed = xr.open_mfdataset(pattern, combine="by_coords")
            log_ds_summary(
                self._ds_processed,
                "ds_processed",
                variables=[
                    v
                    for v in (
                        "solar_CF",
                        "wind_CF",
                        "wind_climatology",
                        "solar_climatology",
                        "demand_proximity_fraction",
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
            from storage import aggregate_lds  # project-internal

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
                "Share of pixels with storage ≥ %d days: land=%.1f%%  offshore=%.1f%%",
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
            demand_raw = self.ds_processed["demand_proximity_fraction"]

            # demand_pattern = str(RESULTS_DIR / "automatic/demand/*.nc")
            # log.info("Loading demand proximity from %s", demand_pattern)
            # demand_raw = xr.open_mfdataset(demand_pattern, combine="by_coords")[
            #     "demand_proximity_weighted_buffered"
            # ]
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
                "Share of grid cells with normalised demand ≥ 0.5: %.1f%%",
                100
                * float((self._normalized_demand >= 0.5).sum())
                / max(1, int(self._normalized_demand.notnull().sum())),
            )
        return self._normalized_demand

    # -- derived zones ---------------------------------------------------
    @property
    def threshold(self) -> dict:
        """Single threshold dict shared by all three classification specs.

        Each spec only reads the keys it needs (via ``_require`` in
        ``classify_zones``); unused keys are ignored.
        """
        return {k: dict(v) for k, v in FIXED_THRESHOLDS.items()}

    @property
    def ds_zones_abundance(self):
        if self._ds_zones_abundance is None:
            log.info("Classifying zones — ABUNDANCE (offshore uses wind + solar)")
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
        """Abundance classification with offshore wind only (no offshore solar)."""
        if self._ds_zones_abundance_wind is None:
            log.info(
                "Classifying zones — ABUNDANCE_OFFSHORE_WIND (offshore uses wind only)"
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
        """Main-text classification: offshore = wind only (no solar)."""
        if self._ds_zones_detailed is None:
            log.info("Classifying zones — DETAILED (offshore uses wind only)")
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
        """Supplementary classification: offshore = wind + solar."""
        if self._ds_zones_full is None:
            log.info("Classifying zones — FULL (offshore uses wind + solar)")
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
        return self._df_pct_detailed_land

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

    # -- resource availability & country-level correlation --------------
    @property
    def ds_res_avail(self):
        if self._ds_res_avail is None:
            log.info("Converting zones to resource availability (method=max_abundance)")
            self._ds_res_avail = convert_zones_to_resource_availability(
                # self.ds_zones_abundance, # would include offshore solar
                self.ds_zones_detailed,  # only offshore wind
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

    @property
    def ds_corr(self):
        if self._ds_corr is None:
            dir_geo = RESOURCES_DIR / "user/module_geo_boundaries/global"
            log.info("Building country-masked correlation dataset")
            ds = xr.Dataset({"resource": self.ds_res_avail["resource_availability"]})
            ds, shapes = add_country_mask(
                ds,
                shapes_path=str(
                    dir_geo / "results/global/results/shapes_combined.parquet"
                ),
                yaml_path=str(dir_geo / "global_config.yaml"),
            )
            ds["demand"] = self.normalized_demand
            ds["storage"] = self.ds_mean["duration_metric"]
            self._ds_corr = ds
            self._shapes_ref = shapes
            log.info("Country shapes loaded: %d rows", len(shapes))
            log_ds_summary(
                self._ds_corr,
                "ds_corr",
                variables=["resource", "demand", "storage"],
            )
        return self._ds_corr

    @property
    def shapes_ref(self):
        _ = self.ds_corr
        return self._shapes_ref

    @property
    def df_corr_results(self):
        if self._df_corr_results is None:
            import regionmask

            countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
            df_countries = countries.to_geodataframe()
            iso3 = [to_iso3_robust(a) for a in df_countries.abbrevs]
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
            valid = sorted(set(valid))
            log.info(
                "Running per-country spatial correlation for %d countries",
                len(valid),
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
            log.info(
                "Country correlation sign: positive=%d, negative=%d",
                pos,
                neg,
            )
        return self._df_corr_results


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------


def _out(name: str, fmt: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return str(FIG_DIR / f"{name}.{fmt}")


# ── Fig 1 — main zone map (DETAILED classification) ────────────────────────
def fig_detailed_zones_map(data: DataBundle, fmt: str) -> None:
    """Fig 1 — Köppen-style renewable zones (main text, detailed scheme)."""
    plot_zones_map(
        data.ds_zones_detailed,
        GROUPS_DETAILED,
        out_path=_out("fig1_renewable_zones_detailed", fmt),
        figsize=(16, 10),
        legend_anchor=(0.5, -0.24),
        legend_ncol=7,
        title=(
            f"Renewable zones — main classification (offshore: wind only)\n"
            f"solar CF {FIXED_THRESHOLDS['solar']['low']}/{FIXED_THRESHOLDS['solar']['high']}, "
            f"wind CF {FIXED_THRESHOLDS['wind_onshore']['low']}/{FIXED_THRESHOLDS['wind_onshore']['high']}, "
            f"storage {FIXED_THRESHOLDS['storage']['land']} days, "
            f"offshore wind {FIXED_THRESHOLDS['wind_offshore']['low']}/{FIXED_THRESHOLDS['wind_offshore']['high']} m/s"
        ),
    )


# ── Fig 2 — deconstructed maps ────────────────────────────────────────────
def fig_abundance_map(data: DataBundle, fmt: str) -> None:
    """Fig 2a — abundance-only map (offshore: wind only, GROUPS_DETAILED colours)."""
    plot_zones_map(
        data.ds_zones_abundance_wind,
        GROUPS_ABUNDANCE_OFFSHORE_WIND,
        out_path=_out("fig2a_abundance_zones", fmt),
        figsize=(15, 12),
        legend_anchor=(0.5, -0.2),
        legend_ncol=4,
        title=(
            f"Abundance zones (onshore solar CF {FIXED_THRESHOLDS['solar']['low']}/{FIXED_THRESHOLDS['solar']['high']}, "
            f"wind CF {FIXED_THRESHOLDS['wind_onshore']['low']}/{FIXED_THRESHOLDS['wind_onshore']['high']}, "
            f"offshore wind {FIXED_THRESHOLDS['wind_offshore']['low']}/{FIXED_THRESHOLDS['wind_offshore']['high']} m/s)"
        ),
    )


def fig_demand_map(data: DataBundle, fmt: str) -> None:
    """Fig 2b — normalised demand proximity."""
    plot_map_continuous(
        data.normalized_demand,
        legend_label="Demand proximity (normalised p95)",
        cmap=plt.get_cmap("magma_r", 10),
        title="Demand proximity normalised with upper bound p95",
        path_output=_out("fig2b_demand_normalised", fmt),
    )


def fig_storage_map(data: DataBundle, fmt: str) -> None:
    """Fig 2c — mean storage duration."""
    plot_map_continuous(
        plot_data=data.ds_mean["duration_metric"].where(data.land | data.offshore),
        cmap=plt.get_cmap("Spectral_r", 9),
        vmin=0,
        vmax=45,
        extend="max",
        legend_label="Storage duration [days]",
        title="Mean storage duration (1995–2025)",
        path_output=_out("fig2c_storage_mean", fmt),
    )


def fig_abundance_storage_combined(data: DataBundle, fmt: str) -> None:
    """Fig 2 (combined) — abundance zones (a) + mean storage duration (b).

    Offshore in panel (a) uses wind only with GROUPS_DETAILED colours.
    """
    plot_abundance_storage_combined(
        ds_abundance=data.ds_zones_abundance_wind,
        groups_abundance=GROUPS_ABUNDANCE_OFFSHORE_WIND,
        storage_data=data.ds_mean["duration_metric"].where(data.land | data.offshore),
        storage_title="Mean storage duration (1995–2025)",
        abundance_title=(
            f"Abundance zones (onshore solar CF {FIXED_THRESHOLDS['solar']['low']}/{FIXED_THRESHOLDS['solar']['high']}, "
            f"wind CF {FIXED_THRESHOLDS['wind_onshore']['low']}/{FIXED_THRESHOLDS['wind_onshore']['high']}, "
            f"offshore wind {FIXED_THRESHOLDS['wind_offshore']['low']}/{FIXED_THRESHOLDS['wind_offshore']['high']} m/s)"
        ),
        legend_ncol=4,
        out_path=_out("Fig2_abundance_storage_separate", fmt),
    )


def fig_optimal_alpha(data: DataBundle, fmt: str) -> None:
    """Fig 2d — optimal wind share."""
    plot_map_continuous(
        data.ds_mean["optimal_alpha"],
        legend_label="Optimal share of wind CF",
        cmap=plt.get_cmap("RdBu", 10),
        title="Optimal share of wind CF (min. annual energy deficit)",
        path_output=_out("fig2d_optimal_alpha", fmt),
    )


# ── Fig 3 — Cramér's V panel (DETAILED, land) ─────────────────────────────
def fig_cramers_v_land(data: DataBundle, fmt: str) -> None:
    """Fig 3 — subgroup % + climate bars + Cramér's V (main text, detailed)."""
    plot_stat_group_climate_cramersv(
        ds_grouped=data.grouped_detailed_land,
        df_percentage=data.df_pct_detailed_land,
        groups_land=land_only_groups(GROUPS_DETAILED),
        land_colors=LAND_COLORS,
        title_suffix="Subgroups in main renewable zones",
        out_path=_out("fig3_cramers_renewable_vs_climate", fmt),
    )


# ── Fig 4 — poor-both, high-demand scatter (DETAILED) ─────────────────────
def fig_scatter_poor_high(data: DataBundle, fmt: str) -> None:
    """Fig 4 — scatter of 'poor both, high demand' countries."""
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
    plot_scatter_elevation_precipitation(
        data.grouped_detailed_land,
        named,
        out_path=_out("fig4_scatter_poor_high_demand", fmt),
    )


# ── Figs 5 & 6 — clusters ─────────────────────────────────────────────────
def _build_clusters(data: DataBundle, n: int = 4):
    """Shared helper — cluster countries and build the label/config dict."""
    df = data.df_corr_results.copy()
    all_metric_path = FIG_DIR / "country_all_metrics.csv"
    df.to_csv(str(all_metric_path))
    log.info("Country all metrics saved: %s", all_metric_path)

    df["avg_storage"] = (df["avg_storage"] - df["avg_storage"].min()) / (
        df["avg_storage"].max() - df["avg_storage"].min()
    )
    df_clustered, centroids = cluster_countries(df, n_clusters=n)

    # Sort clusters left-to-right by correlation axis for reproducibility
    order = centroids["resource_demand_corr"].argsort().values
    remap = {old: new for new, old in enumerate(order)}
    df_clustered["cluster"] = df_clustered["cluster"].map(remap)
    centroids = centroids.iloc[order].reset_index(drop=True)

    cluster_colors = ["#FF5454", "#7182F3", "#FFD454", "#5EC478", "#5EBCC4"][:n]

    # Derive labels from CLUSTER_NAMING_RULES applied to centroid values
    def _name_cluster(idx: int) -> str:
        row = centroids.iloc[idx]
        matched = [
            name
            for name, (col, op, thr) in CLUSTER_NAMING_RULES.items()
            if col in row.index
            and ((op == ">=" and row[col] >= thr) or (op == "<" and row[col] < thr))
        ]
        return " / ".join(matched) if matched else f"Cluster {idx}"

    labels = [_name_cluster(i) for i in range(n)]
    cluster_config = {i: [cluster_colors[i], labels[i]] for i in range(n)}

    log_df_summary(
        df_clustered,
        "df_clustered",
        numeric_cols=["avg_storage", "resource_demand_corr", "avg_resource", "cluster"],
    )
    log_df_summary(centroids, "centroids")

    # Save cluster summary CSV to figures/main
    cluster_info = pu.create_cluster_summary_table_full_names(
        df_clustered,
        data.shapes_ref,
        centroids,
        naming_rules=CLUSTER_NAMING_RULES,
        large_threshold=400,
    )
    csv_path = FIG_DIR / "cluster_summary.csv"
    cluster_info.to_csv(str(csv_path))
    log.info("Cluster summary saved: %s", csv_path)

    return df_clustered, centroids, cluster_config


LIST_CNT_HIGHLIGHT = [
    "USA",
    "DEU",
    "CHN",
    "VIE",
    "NLD",
    "FRA",
    "EGP",
    "IND",
    "AUS",
    "SYR",
    "ESP",
    "ITA",
    "COD",
    "COL",
    "PER",
    "GNQ",
    "MLI",
    "IRN",
    "PRT",
    "GAB",
    "BIH",
    "FJI",
    "UGA",
    "CHL",
    "BRA",
    "RUS",
    "NAM",
]


def fig_clusters_combined(data: DataBundle, fmt: str) -> None:
    """Fig 5 — cluster scatter + map."""
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


def fig_clusters_3d(data: DataBundle, fmt: str) -> None:
    """Fig 6 — 3D scatter of country clusters (with highlighted labels)."""
    df_clustered, centroids, cluster_config = _build_clusters(data, n=4)
    plot_country_clusters_3d(
        df_clustered,
        centroids,
        cluster_map=cluster_config,
        list_cnt_to_plot=LIST_CNT_HIGHLIGHT,
        fig_size=(1400, 900),
        out_path=_out("fig6_clusters_3d", fmt),
    )


# ── Supplementary figures ─────────────────────────────────────────────────
def fig_full_zones_map(data: DataBundle, fmt: str) -> None:
    """Fig S1 — full-classification zone map (offshore uses wind + solar)."""
    plot_zones_map(
        data.ds_zones_full,
        GROUPS_FULL,
        out_path=_out("figS1_renewable_zones_full", fmt),
        figsize=(16, 10),
        legend_anchor=(0.5, -0.25),
        legend_ncol=6,
        title=(
            f"Renewable zones — full classification (offshore: wind + solar)\n"
            f"solar CF {FIXED_THRESHOLDS['solar']['low']}/{FIXED_THRESHOLDS['solar']['high']}, "
            f"wind CF {FIXED_THRESHOLDS['wind_onshore']['low']}/{FIXED_THRESHOLDS['wind_onshore']['high']}, "
            f"storage {FIXED_THRESHOLDS['storage']['land']} days, "
            f"offshore wind {FIXED_THRESHOLDS['wind_offshore']['low']}/{FIXED_THRESHOLDS['wind_offshore']['high']} m/s, "
            f"offshore solar {FIXED_THRESHOLDS['solar_offshore']['low']:.0f}/{FIXED_THRESHOLDS['solar_offshore']['high']:.0f} W/m²"
        ),
    )


def fig_resource_availability(data: DataBundle, fmt: str) -> None:
    """Fig S — normalised resource-availability map."""
    plot_map_continuous(
        data.ds_res_avail["resource_availability"],
        vmin=0,
        vmax=1,
        cmap=plt.get_cmap("RdYlGn", 3),
        legend_label="Resource abundance [normalised]",
        title="Resource availability (max abundance)",
        path_output=_out("figS_resource_availability", fmt),
    )


def fig_spatial_correlation_map(data: DataBundle, fmt: str) -> None:
    """Fig S — country-level resource–demand Spearman correlation."""
    plot_corr = prepare_metric_map(
        data.ds_corr,
        data.df_corr_results,
        data.shapes_ref,
        column="resource_demand_corr",
    )
    plot_map_continuous(
        plot_corr,
        legend_label="Spearman ρ (Resource vs Demand)",
        cmap="RdBu",
        title="Global spatial correlation: renewables vs demand",
        vmin=-1,
        vmax=1,
        levels=11,
        path_output=_out("figS_spatial_correlation", fmt),
    )


def fig_correlation_histogram(data: DataBundle, fmt: str) -> None:
    """Fig S — histogram of country correlations."""
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
    ax.set_title("Histogram of renewable–demand spatial correlation")
    fig.tight_layout()
    fig.savefig(_out("figS_correlation_histogram", fmt), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FIGURES: dict[str, Callable[[DataBundle, str], None]] = {
    # Main text
    "fig1": fig_detailed_zones_map,
    "fig2a": fig_abundance_map,
    "fig2b": fig_demand_map,
    "fig2c": fig_storage_map,
    "fig2d": fig_optimal_alpha,
    "fig2_abundance_storage": fig_abundance_storage_combined,
    "fig3": fig_cramers_v_land,
    "fig4": fig_scatter_poor_high,
    "fig5": fig_clusters_combined,
    "fig6": fig_clusters_3d,
    # Supplementary
    "figS1_full_zones": fig_full_zones_map,
    "figS_resource": fig_resource_availability,
    "figS_corr_map": fig_spatial_correlation_map,
    "figS_corr_hist": fig_correlation_histogram,
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
    log.info("make_figures.py run started")
    log.info("Log file:         %s", log_path)
    log.info("Output directory: %s", FIG_DIR)
    log.info("Output format:    %s", args.format)
    log.info("Figures to build: %s", targets)
    log.info("=" * 60)

    data = DataBundle()
    failed: list[str] = []
    for name in targets:
        log.info("── Building %s ──", name)
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
