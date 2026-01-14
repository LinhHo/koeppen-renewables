#!/usr/bin/env python3
"""
Plot histograms of processed Koeppen-renewables outputs.

This script:
- loads all processed_*.nc files
- merges them into a single xarray Dataset
- plots distributions of:
    * Capacity factor (solar & wind)
    * Seasonal variability (solar & wind)
    * Weather variability (solar & wind)

Each histogram includes:
- mean (red dashed)
- ±1 standard deviation (blue dotted)
"""

from pathlib import Path
import sys

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import colorsys


RESULTS_DIR = Path("results/automatic")
FIG_DIR = Path("results/figures")

if not FIG_DIR.exists():
    FIG_DIR.mkdir(parents=True)

# ============================================================
# USER CONFIGURATION
# ============================================================

ABUNDANCE_QUANTILES = {
    "low": 0.33,
    "high": 0.66,
}

VARIABILITY_QUANTILES = {
    "low": 0.66,  # below = reliable
}

# ============================================================
# 1. Plot histograms
# ============================================================


def plot_hist_with_lines(
    data: xr.DataArray,
    axis: plt.Axes,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
):
    """
    Plot histogram with mean and ±1 std lines.
    """
    values = data.values.flatten()
    values = values[np.isfinite(values)]

    axis.hist(values, bins=50)
    mean = np.mean(values)
    std = np.std(values)

    axis.axvline(mean, color="red", linestyle="--", label="Mean")
    axis.axvline(mean + std, color="blue", linestyle=":", label="+1σ")
    axis.axvline(mean - std, color="blue", linestyle=":", label="-1σ")

    if title:
        axis.set_title(title)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)


def plot_histograms(ds: xr.Dataset):
    """
    Plot all histograms in a 3x2 panel layout.
    """
    print("Plotting histograms...")
    fig, ax = plt.subplots(3, 2, figsize=(10, 12), sharey=False)

    datasets = [
        # Row 1: Capacity factors
        (ds["solar_CF"], ax[0, 0], "Solar", "Capacity factor", "Count"),
        (ds["wind_CF"], ax[0, 1], "Wind", "Capacity factor", None),
        # Row 2: Seasonal variability
        (
            ds["solar_seasonal_variability"],
            ax[1, 0],
            "Solar seasonal variability",
            "Normalized deficit",
            "Count",
        ),
        (
            ds["wind_seasonal_variability"],
            ax[1, 1],
            "Wind seasonal variability",
            "Normalized deficit",
            None,
        ),
        # Row 3: Weather variability
        (
            ds["solar_weather_variability"],
            ax[2, 0],
            "Solar weather variability",
            "Normalized deficit",
            "Count",
        ),
        (
            ds["wind_weather_variability"],
            ax[2, 1],
            "Wind weather variability",
            "Normalized deficit",
            None,
        ),
    ]

    for data, axis, title, xlabel, ylabel in datasets:
        plot_hist_with_lines(data, axis, title, xlabel, ylabel)

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    # fig.suptitle(
    #     f"Köppen-renewables (lons {bounds[0]}-{bounds[2]}, lats {bounds[1]}-{bounds[3]}), ({start_year}-{end_year})",
    #     fontsize=16,
    # )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "histograms.png", dpi=300)
    print(f"Histograms plot saved to {FIG_DIR / 'histograms.png'}")

    # bounds_str = "_".join(map(str, bounds))
    # fig.savefig(FIG_DIR / f"histograms_{bounds_str}_{start_year}_{end_year}.png", dpi=300)


# ============================================================
# 2. Plot zones land
# ============================================================

LAND_COLORS = {
    "A": "#28BD28",  # very abundant both
    "B": "#BCBD22",  # moderately abundant both
    "C": "#00BBFF",  # dominant wind
    "D": "#FF6F00",  # dominant solar
    "E": "#BFBFBF",  # poor both
}


def classify_ternary(da, q=(0.33, 0.66)):
    q0, q1 = da.quantile(q)
    return xr.where(da >= q1, 2, xr.where(da >= q0, 1, 0))


def is_reliable(da_var, q=0.33):
    return da_var < da_var.quantile(q)


def dirty(hex_color):
    r, g, b = mcolors.to_rgb(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return colorsys.hsv_to_rgb(h, s * 0.5, v * 0.7)


def plot_zones_land(ds):

    print("Plotting zones land...")
    solar_cf = ds["solar_CF"].compute()
    wind_cf = ds["wind_CF"].compute()

    solar_seasonal_var = ds["solar_seasonal_variability"].compute()
    wind_seasonal_var = ds["wind_seasonal_variability"].compute()

    land = solar_cf.notnull()

    q_solar = solar_cf.quantile([0.33, 0.66])
    q_wind = wind_cf.quantile([0.33, 0.66])

    solar_low = solar_cf < q_solar.sel(quantile=0.33)
    solar_mid = (solar_cf >= q_solar.sel(quantile=0.33)) & (
        solar_cf < q_solar.sel(quantile=0.66)
    )
    solar_high = solar_cf >= q_solar.sel(quantile=0.66)

    wind_low = wind_cf < q_wind.sel(quantile=0.33)
    wind_mid = (wind_cf >= q_wind.sel(quantile=0.33)) & (
        wind_cf < q_wind.sel(quantile=0.66)
    )
    wind_high = wind_cf >= q_wind.sel(quantile=0.66)

    solar_reliable = solar_seasonal_var < solar_seasonal_var.quantile(0.33)
    wind_reliable = wind_seasonal_var < wind_seasonal_var.quantile(0.33)

    # Base land mask
    land = solar_cf.notnull()

    # --- Zone definitions ---
    mask_A = land & solar_high & wind_high  # (2,2)
    mask_B = land & solar_mid & wind_mid  # (1,1)

    mask_C = land & (
        (wind_high & solar_mid) | (wind_high & solar_low) | (wind_mid & solar_low)
    )

    mask_D = land & (
        (solar_high & wind_mid) | (solar_high & wind_low) | (solar_mid & wind_low)
    )

    mask_E = land & solar_low & wind_low  # (0,0)

    # Reliability
    reliable_A = mask_A & wind_reliable & solar_reliable
    reliable_B = mask_B & wind_reliable & solar_reliable

    reliable_C = mask_C & wind_reliable
    reliable_D = mask_D & solar_reliable

    unreliable_A = mask_A & ~reliable_A
    unreliable_B = mask_B & ~reliable_B
    unreliable_C = mask_C & ~reliable_C
    unreliable_D = mask_D & ~reliable_D

    # Plotting
    rgb = np.ones((solar_cf.shape[0], solar_cf.shape[1], 3))

    # --- Reliable zones ---
    rgb[reliable_A] = mcolors.to_rgb(LAND_COLORS["A"])
    rgb[reliable_B] = mcolors.to_rgb(LAND_COLORS["B"])
    rgb[reliable_C] = mcolors.to_rgb(LAND_COLORS["C"])
    rgb[reliable_D] = mcolors.to_rgb(LAND_COLORS["D"])

    # --- Unreliable zones (dirty colours) ---
    rgb[unreliable_A] = dirty(LAND_COLORS["A"])
    rgb[unreliable_B] = dirty(LAND_COLORS["B"])
    rgb[unreliable_C] = dirty(LAND_COLORS["C"])
    rgb[unreliable_D] = dirty(LAND_COLORS["D"])

    # --- Poor both ---
    rgb[mask_E] = mcolors.to_rgb(LAND_COLORS["E"])

    legend_items = [
        ("#28BD28", "A – Very abundant both (reliable)"),
        (dirty("#28BD28"), "A – Very abundant both (less reliable)"),
        ("#BCBD22", "B – Moderately abundant both"),
        (dirty("#BCBD22"), "B – Moderately abundant both (less reliable)"),
        ("#00BBFF", "C – Dominant wind"),
        (dirty("#00BBFF"), "C – Dominant wind (less reliable)"),
        ("#FF6F00", "D – Dominant solar"),
        (dirty("#FF6F00"), "D – Dominant solar (less reliable)"),
        ("#BFBFBF", "E – Poor both"),
    ]

    patches = [
        mpatches.Patch(color=color, label=label) for color, label in legend_items
    ]

    # Plotting

    ds = ds.sortby("latitude")
    rgb = np.flipud(rgb)

    lat = ds["latitude"]
    lon = ds["longitude"]

    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.imshow(
        rgb,
        origin="lower",
        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        transform=ccrs.PlateCarree(),
    )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=-1)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=-1)

    ax.set_title("Koeppen renewable zones – Land", fontsize=14)

    ax.legend(
        handles=patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=False,
    )

    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "land_zones.png", dpi=300)


# ============================================================
# 3. Plot zones offshore
# ============================================================
OFFSHORE_COLORS = {
    "no_data": ("white", "No data"),
    "high_reliable": ("#0000FF", "High wind, reliable"),
    "low_reliable": ("#7EC3E6", "Low wind, reliable"),
    "high_unreliable": ("#8751A1", "High wind, unreliable"),
    "low_unreliable": ("#C5ACD3", "Low wind, unreliable"),
}


def classify_offshore(ds):
    wind_cf = ds["wind_CF"]
    wind_var = ds["wind_seasonal_variability"]

    offshore = wind_cf.notnull() & ds["solar_CF"].isnull()

    wind_hi = wind_cf >= wind_cf.quantile(ABUNDANCE_QUANTILES["high"])
    wind_rel = wind_var < wind_var.quantile(VARIABILITY_QUANTILES["low"])

    masks = {
        "high_reliable": offshore & wind_hi & wind_rel,
        "low_reliable": offshore & ~wind_hi & wind_rel,
        "high_unreliable": offshore & wind_hi & ~wind_rel,
        "low_unreliable": offshore & ~wind_hi & ~wind_rel,
    }

    category = np.zeros(wind_cf.shape, dtype=int)
    key_order = list(OFFSHORE_COLORS.keys())

    for i, key in enumerate(key_order[1:], start=1):
        category[masks[key]] = i

    return category, key_order


def plot_zones_offshore(ds):
    print("Plotting zones offshore...")
    category, key_order = classify_offshore(ds)

    lon = ds["longitude"]
    lat = ds["latitude"]

    cmap = mcolors.ListedColormap([OFFSHORE_COLORS[k][0] for k in key_order])

    legend_patches = [
        mpatches.Patch(
            facecolor=OFFSHORE_COLORS[k][0],
            edgecolor="black",
            linewidth=0.5,
            label=OFFSHORE_COLORS[k][1],
        )
        for k in key_order
    ]

    fig = plt.figure(figsize=(14, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.pcolormesh(lon, lat, category, cmap=cmap, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="#E6F2FF")

    ax.set_title("Köppen renewable zones – Offshore")

    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "offshore_zones.png", dpi=300)


# ============================================================
# 4. Plot demand potential
# ============================================================


def plot_demand(ds, ds_processed):
    print("Plotting demand potential...")

    demand = ds["demand_potential"].where(ds_processed["solar_CF"].notnull())

    lon = ds["longitude"]
    lat = ds["latitude"]

    # --------------------------------------------------
    # Define demand categories
    # --------------------------------------------------
    DEMAND_COLORS = {
        "no_data": ("white", "No data"),
        "low": ("#f5bcee", "Low demand"),
        "high": ("#bf60b4", "High demand"),
    }

    key_order = list(DEMAND_COLORS.keys())

    # Threshold (33rd percentile → “high demand” above this)
    q = demand.quantile(0.33)
    demand_high = demand > q

    # --------------------------------------------------
    # Build category array (xarray-safe)
    # --------------------------------------------------
    category = xr.zeros_like(demand, dtype=int)

    # valid demand points
    valid = demand.notnull()

    # threshold
    q = demand.quantile(0.33)

    category = xr.where(valid, 1, category)  # low demand
    category = xr.where(demand > q, 2, category)  # high demand

    # --------------------------------------------------
    # Colormap & legend
    # --------------------------------------------------
    cmap = mcolors.ListedColormap([DEMAND_COLORS[k][0] for k in key_order])

    legend_patches = [
        mpatches.Patch(
            facecolor=DEMAND_COLORS[k][0],
            edgecolor="black",
            linewidth=0.6,
            label=DEMAND_COLORS[k][1],
        )
        for k in key_order
    ]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig = plt.figure(figsize=(14, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.pcolormesh(
        lon,
        lat,
        category,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
    )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="#E6F2FF")

    ax.set_title("Demand potential (temperature × proximity)")

    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
    )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "demand.png", dpi=300)


def plot_all():
    fname = str(RESULTS_DIR / "processed_*.nc")
    ds_processed = xr.open_mfdataset(fname, combine="by_coords")
    plot_histograms(ds_processed)
    plot_zones_land(ds_processed)
    plot_zones_offshore(ds_processed)
    fname = str(RESULTS_DIR / "demand_potential_*.nc")
    ds_demand = xr.open_mfdataset(fname, combine="by_coords")
    plot_demand(ds_demand, ds_processed)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    plot_all()
