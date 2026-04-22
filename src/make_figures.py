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
    plot_abundance_histograms,
    plot_abundance_storage_combined,
    plot_combined_analysis,
    plot_zones_by_cluster_from_grid,
    plot_zones_climatology,
    plot_country_clusters_3d,
    plot_map_continuous,
    plot_offshore_shift_density,
    plot_scatter_elevation_precipitation,
    plot_stat_group_climate_cramersv,
    plot_storage_histograms,
    plot_zones_map,
    prepare_cluster_map,
    prepare_metric_map,
    to_iso3_robust,
)

LIST_CNT_HIGHLIGHT = [
    "CHN",  # China
    "IRN",  # Iran
    # "PAK",  # Pakistan
    "MAR",  # Morocco
    "DNK",  # Denmark
    "USA",
    "ALG",  # Algeria
    "MLI",  # Mali
    "ESP",  # Spain
    "JPN",  # Japan
    "VNM",  # Vietnam
    "IND",  # India
    "IDN",  # Indonesia
    "GRC",  # Greece
    # "AUS",  # Australia
    # "DEU",
    # "NLD",
    # "FRA",
    # "EGP",
    # "SYR",
    "ITA",  # Italy
    # "COD",
    # "COL",
    # "PER",
    # "PRT",
    # "GAB",
    # "BIH",
    # "FJI",
    # "UGA",
    # "CHL",
    # "BRA",
    # "RUS",
    # "NAM",
]

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
        self._grouped_detailed_all = None
        self._df_pct_detailed_all = None
        self._grouped_full_land = None
        self._df_pct_full_land = None
        self._ds_res_avail = None
        self._ds_corr = None
        self._shapes_ref = None
        self._df_corr_results = None
        self._ds_corr_land = None
        self._shapes_land = None
        self._df_corr_land = None

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
        if self._df_pct_detailed_land is not None:
            csv_path = FIG_DIR / "df_zones_detailed_LAND_percentage.csv"
            self._df_pct_detailed_land.to_csv(str(csv_path), index=False)
            log.info("Saved: %s", csv_path)
        return self._df_pct_detailed_land

    @property
    def grouped_detailed_all(self):
        """Base+subgroup decomposition for all pixels (land + offshore)."""
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
            csv_path = FIG_DIR / "df_zones_detailed_ALL_percentage.csv"
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

    # -- shared helpers --------------------------------------------------------
    def _valid_iso3(self) -> list[str]:
        """Canonical list of ISO-3 country codes used in all correlation runs."""
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
        """Build the resource/demand/storage dataset masked by country shapes.

        Parameters
        ----------
        shapes_parquet : str
            Path to the .parquet file with country shapes.
        shape_class : str or None
            If given, filter the shapes to rows where ``shape_class == value``
            before rasterising (e.g. ``"land"`` to exclude offshore polygons).
            Pass ``None`` (default) to use all rows, which is the behaviour of
            the original ``add_country_mask`` call.
        label : str
            Name used in log messages.
        """
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
        # USA mainland only
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

    # -- land-only shapes (experiment A) --------------------------------------
    @property
    def ds_corr_land(self):
        """Like ds_corr but restricted to land polygons (shape_class='land')."""
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
        """Per-country metrics using land-only mask (experiment A)."""
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
    toplot = plot_scatter_elevation_precipitation(
        data.grouped_detailed_land,
        named,
        out_path=_out("fig4_scatter_poor_high_demand", fmt),
    )

    # Save top-120 'Poor both high demand' countries sorted by % of subgroup.
    csv_path = FIG_DIR / "df_stats_poor_both_high_demand.csv"
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

    # Step 1 — sort by correlation so positions 0..n-1 run left-to-right on
    # the scatter axis (most-negative corr first).
    order = centroids["resource_demand_corr"].argsort().values
    remap = {old: new for new, old in enumerate(order)}
    df_clustered["cluster"] = df_clustered["cluster"].map(remap)
    centroids = centroids.iloc[order].reset_index(drop=True)

    # Step 2 — re-assign corr-sorted positions to fixed Roman-numeral IDs so
    # that sorted(cluster_config) == [0,1,2,3] == [I, II, III, IV] in the
    # legend.  Mapping (valid for n=4):
    #   corr pos 0  most-negative, long storage          → Cluster III  (ID 2)
    #   corr pos 1  high resource, short storage         → Cluster I    (ID 0)
    #   corr pos 2  short storage, moderate resource     → Cluster II   (ID 1)
    #   corr pos 3  most-positive, low resource / match  → Cluster IV   (ID 3)
    _pos_to_id = {0: 2, 1: 0, 2: 1, 3: 3}
    df_clustered["cluster"] = df_clustered["cluster"].map(_pos_to_id)
    # Reorder centroid rows so that iloc[id] == that cluster's centroid.
    # Desired row order by centroid: [I=pos1, II=pos2, III=pos0, IV=pos3]
    centroids = centroids.iloc[[1, 2, 0, 3]].reset_index(drop=True)

    # Fixed colours and labels indexed by cluster ID [0=I, 1=II, 2=III, 3=IV]
    cluster_colors = ["#5EC478", "#7182F3", "#FFD454", "#FF5454"]
    labels = [
        "Cluster I — high resource / short storage / high mismatch",
        "Cluster II — short storage / moderate resource",
        "Cluster III — long storage / high mismatch",
        "Cluster IV — low resource / match",
    ]
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
def fig_zones_by_cluster(data: DataBundle, fmt: str) -> None:
    """Fig Sup — 2×2 bar plots of renewable-zone composition per cluster (I–IV)."""
    df_clustered, centroids, cluster_config = _build_clusters(data, n=4)
    cluster_grid = prepare_cluster_map(data.ds_corr, df_clustered, data.shapes_ref)
    plot_zones_by_cluster_from_grid(
        ds_zones=data.ds_zones_detailed,
        cluster_grid=cluster_grid,
        cluster_config=cluster_config,
        groups=GROUPS_DETAILED,
        out_path=_out("Sup_zones_by_clusters", fmt),
    )


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


def fig_offshore_wind_shift(data: DataBundle, fmt: str) -> None:
    """Fig S — KDE density of per-country metric shifts when adding offshore wind.

    Three panels: Δavg_storage, Δresource_demand_corr, Δavg_resource.
    Curves are coloured by cluster (same colours as Fig 5).
    Per-cluster mean shifts are logged.
    """
    df_land = data.df_corr_land.copy()
    df_combined = data.df_corr_results.copy()

    # Save land-only metrics.
    land_metric_path = FIG_DIR / "country_land_only_metrics.csv"
    df_land.to_csv(str(land_metric_path))
    log.info("Land-only metrics saved: %s", land_metric_path)

    # Normalise avg_storage using combined's global min/max (same scale as
    # _build_clusters, which normalises df_corr_results before clustering).
    s_min = df_combined["avg_storage"].min()
    s_max = df_combined["avg_storage"].max()
    for df in (df_land, df_combined):
        df["avg_storage"] = (df["avg_storage"] - s_min) / (s_max - s_min)

    # Cluster assignments (reproducible via _build_clusters).
    df_clustered, _, cluster_config = _build_clusters(data, n=4)

    plot_offshore_shift_density(
        df_land=df_land,
        df_combined=df_combined,
        cluster_info=df_clustered[["iso3", "cluster"]],
        cluster_config=cluster_config,
        out_path=_out("figS_offshore_wind_shift", fmt),
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


def fig_abundance_histograms(data: DataBundle, fmt: str) -> None:
    """Fig Sa — histograms of abundance variables with classification thresholds."""
    plot_abundance_histograms(
        ds=data.ds_processed,
        land=data.land,
        offshore=data.offshore,
        thresholds=data.threshold,
        out_path=_out("figSa_abundance_histograms", fmt),
    )


def fig_storage_histograms(data: DataBundle, fmt: str) -> None:
    """Fig Sb — histograms of storage duration with classification threshold."""
    plot_storage_histograms(
        storage_da=data.ds_mean["duration_metric"],
        land=data.land,
        offshore=data.offshore,
        thresholds=data.threshold,
        out_path=_out("figSb_storage_histograms", fmt),
    )


def fig_zones_climatology(data: DataBundle, fmt: str) -> None:
    """Fig S — abundance zones classified from wind/solar climatology (p33/p67 thresholds)."""
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
    "figS_offshore_shift": fig_offshore_wind_shift,
    "figSa_abundance_histograms": fig_abundance_histograms,
    "figSb_storage_histograms": fig_storage_histograms,
    "figS_zones_climatology": fig_zones_climatology,
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
