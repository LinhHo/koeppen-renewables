import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import regionmask

import seaborn as sns
import xarray as xr
import colorsys

from itertools import product
from pathlib import Path
from src.geo_processing import open_era5_zarr

BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR.parent / "resources"
RESULTS_DIR = BASE_DIR.parent / "results"

LAND_COLORS = {
    "B": "#0EC90E",  # abundant both
    "W": "#009DFF",  # wind dominant
    "Ws": "#00FFF7",  # wind favourable, solar possible
    "S": "#FF6F00",  # solar dominant
    "Sw": "#FFE600",  # solar favourable, wind possible
    "P": "#BFBFBF",  # poor both
}


def dirty(color):
    r, g, b = mcolors.to_rgb(color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return colorsys.hsv_to_rgb(h, s * 0.5, v * 0.7)


def light(color):
    r, g, b = mcolors.to_rgb(color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return colorsys.hsv_to_rgb(h, s * 0.4, min(1, v * 1.1))


groups_colours = {
    # CODE: [color, label, [subgroups]]
    "B": ["#3cb44b", "- Both", ["HRHRh", "MRMRh"]],  # "Both abundant",
    "B_l": [light("#3cb44b"), "", ["HRHRl", "MRMRl"]],  # "Both abundant (low demand)",
    "B_u": [
        "#688818",
        "",  # "Both abundant, unreliable",
        ["HUHUh", "HRHUh", "HUHRh", "MUMUh", "MUMRh", "MRMUh"],
    ],
    "B_ul": [
        light("#688818"),
        "",  # "Both abundant, unreliable (low demand)",
        ["HUHUl", "HRHUl", "HUHRl", "MUMUl", "MUMRl", "MRMUl"],
    ],
    "W": ["#0099FF", "- Wind", ["LxHRh", "LxMRh"]],  # "Wind dominant",
    "W_l": [light("#0099FF"), "", ["LxHRl", "LxMRl"]],  # "Wind dominant (low demand)",
    "W_u": ["#2e86a8", "", ["LxHUh", "LxMUh"]],  # "Wind dominant, unreliable",
    "W_ul": [
        light("#2e86a8"),
        "",  # "Wind dominant, unreliable (low demand)",
        ["LxHUl", "LxMUl"],
    ],
    "Ws": ["#00FFEE", "- Wind solar", ["MxHRh"]],  # "Wind favourable, solar possible",
    "Ws_l": [
        light("#00FFEE"),
        "",  #  "Wind favourable, solar possible (low demand)",
        ["MxHRl"],
    ],
    "Ws_u": ["#22B8AE", "", ["MxHUh"]],  # "Wind favourable unreliable, solar possible",
    "Ws_ul": [
        light("#22B8AE"),
        "",  # "Wind favourable unreliable, solar possible (low demand)",
        ["MxHUl"],
    ],
    "S": ["#ff8819", "- Solar", ["HRLxh", "MRLxh"]],  # "Solar dominant",
    "S_l": [light("#ff8819"), "", ["HRLxl", "MRLxl"]],  # "Solar dominant (low demand)",
    "S_u": ["#b25c0c", "", ["HULxh", "MULxh"]],  # "Solar dominant, unreliable",
    "S_ul": [
        light("#b25c0c"),
        "",  # "Solar dominant, unreliable (low demand)",
        ["HULxl", "MULxl"],
    ],
    "Sw": ["#fff319", "- Solar wind", ["HRMxh"]],  # "Solar favourable, wind possible",
    "Sw_l": [
        light("#fff319"),
        "",  # "Solar favourable, wind possible (low demand)",
        ["HRMxl"],
    ],
    "Sw_u": ["#b2aa13", "", ["HUMxh"]],  # "Solar favourable unreliable, wind possible",
    "Sw_ul": [
        light("#b2aa13"),
        "",  # "Solar favourable unreliable, wind possible (low demand)",
        ["HUMxl"],
    ],
    "P": ["#FF0000", "- Poor", ["LxLxh"]],  # "Both poor (high demand)",
    "P_l": ["#BFBFBF", "", ["LxLxl"]],  # "Both poor",
    "o": [
        "#EA6565",
        "",  # "Offshore wind low abundant (high demand)",
        ["offshore_LRh", "offshore_LUh"],
    ],
    "o_l": [
        "#CECDCD",
        "",  # "Offshore wind low abundant (low demand)",
        ["offshore_LRl", "offshore_LUl"],
    ],
    "O": ["#4a53ff", "- Offshore", ["offshore_HRh"]],  # "Offshore wind high abundant",
    "O_l": [
        "#a6a3fe",
        "",  # "Offshore wind abundant (low demand)",
        ["offshore_HRl"],
    ],
    "O_u": [
        "#8f13fb",
        "",  # "Offshore wind high abundant unreliable",
        ["offshore_HUh"],
    ],
    "O_ul": [
        "#d3a1ff",
        "",  # "Offshore wind high abundant unreliable (low demand)",
        ["offshore_HUl"],
    ],
}


# Get mask of offshore areas only (mask out solar_CF does not work because it cuts off above ~60N)
def mask_land(dataarray):
    # ds = ds["wind_CF"]
    has_data = dataarray.where(dataarray != 0).notnull()
    land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(
        dataarray.longitude, dataarray.latitude
    )
    return has_data & land_mask.notnull()


# Get mask of offshore areas only (mask out solar_CF does not work because it cuts off above ~60N)
def mask_offshore(dataarray):
    # ds = ds["wind_CF"]
    has_data = dataarray.where(dataarray != 0).notnull()
    land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(
        dataarray.longitude, dataarray.latitude
    )
    is_ocean = land_mask.isnull()
    return has_data & is_ocean


def classify_land_zones_detailed(
    ds,
    ds_demand,
    reliable="seasonal",
    add_offshore=False,
    threshold_cf=None,
    threshold_variability_quantiles=0.66,
):
    """
    Classify each pixel into a 4-character label based on four factors:
      - Solar abundance: H (high), M (medium), or L (low) based on tertiles
      - Solar reliability: R (reliable) or U (unreliable)
      - Wind abundance: H (high), M (medium), or L (low) based on tertiles
      - Wind reliability: R (reliable) or U (unreliable)
      - Demand: h (high demand) or l (low demand)

    Example labels:
      - "HRHR" = High solar, Reliable solar, High wind, Reliable wind
      - "LULU" = Low solar, Unreliable solar, Low wind, Unreliable wind
      - "MRLU" = Medium solar, Reliable solar, Low wind, Unreliable wind
      - 'LxLxH' = 'Low both, high demand'

    Returns the input dataset with a new "zones" variable added.

    """
    print(f"Computing variability metrics based on {reliable} variability...")
    solar_variability = ds[f"solar_{reliable}_variability"].compute()
    wind_variability = ds[f"wind_{reliable}_variability"]  # .compute()

    ###
    # Land
    ###==============================

    land = mask_land(ds["wind_CF"])

    ###
    # Abundance (high, medium, low)
    ###

    solar_cf = ds["solar_CF"].compute()
    onshore_cf = ds["wind_CF"].where(land).compute()
    onshore_variability = wind_variability.where(land).compute()

    if threshold_cf is None:
        print("Using quantile-based thresholds for abundance classification.")
        threshold_cf = {
            "solar": {
                "high": solar_cf.quantile(0.66),
                "low": solar_cf.quantile(0.33).values,
            },
            "wind_onshore": {
                "high": onshore_cf.quantile(0.66).values,
                "low": onshore_cf.quantile(0.33).values,
            },
        }
    else:
        print("Using fixed thresholds for abundance classification.")

    threshold_variability = {
        "solar": solar_variability.quantile(
            threshold_variability_quantiles
        ).values,  # 3/12,
        "wind_onshore": onshore_variability.quantile(
            threshold_variability_quantiles
        ).values,  # 1/12,
    }

    print(
        f"Threshold capacity factor for solar are: {threshold_cf['solar']['low']:.2f}, {threshold_cf['solar']['high']:.2f} \n and for wind onshore are {threshold_cf['wind_onshore']['low']:.2f}, {threshold_cf['wind_onshore']['high']:.2f}"
    )

    solar_low = (solar_cf < threshold_cf["solar"]["low"]) | solar_cf.isnull()
    solar_mid = (solar_cf >= threshold_cf["solar"]["low"]) & (
        solar_cf < threshold_cf["solar"]["high"]
    )
    solar_high = solar_cf >= threshold_cf["solar"]["high"]
    wind_low = onshore_cf < threshold_cf["wind_onshore"]["low"]
    wind_mid = (onshore_cf >= threshold_cf["wind_onshore"]["low"]) & (
        onshore_cf < threshold_cf["wind_onshore"]["high"]
    )
    wind_high = onshore_cf >= threshold_cf["wind_onshore"]["high"]

    ###
    # Reliability (reliable, unreliable)
    ###

    solar_reliable = solar_variability < threshold_variability["solar"]
    solar_unreliable = solar_variability >= threshold_variability["solar"]

    wind_reliable = onshore_variability < threshold_variability["wind_onshore"]
    wind_unreliable = onshore_variability >= threshold_variability["wind_onshore"]
    print(
        f"Thresholds quantile {threshold_variability_quantiles} for wind and solar {reliable} variability are: {threshold_variability['solar']*365:.2f} days and {threshold_variability['wind_onshore']*365:.2f} days."
    )

    ###
    # Offshore
    ###==============================
    if add_offshore:
        offshore = mask_offshore(ds["wind_CF"])
        offshore_variability = wind_variability.where(offshore).compute()

        if "wind_offshore" not in threshold_cf.keys():
            print(
                "Using quantile thresholds 0.5 for ws100 offshore wind abundance classification."
            )
            #### !!!!!!!!!! NOTE: remove 2020_2023 and fix sum-mean problem!
            offshore_abundant = xr.open_dataset(
                str(
                    RESOURCES_DIR
                    / "automatic/ws100/era5_ws100_year_average_2020_2023.nc"
                )
            )["ws100"] / (365 * 24)
            threshold_cf["wind_offshore"] = {
                "high": offshore_abundant.quantile(0.5).values,
                # "low": offshore_cf.quantile(0.33).values,
            }
        else:
            print(
                "Using fixed thresholds for capacity factor offshore wind abundance classification."
            )
            offshore_abundant = ds["wind_CF"].where(offshore).compute()

        threshold_variability["wind_offshore"] = offshore_variability.quantile(
            threshold_variability_quantiles
        ).values
        # : {threshold_cf['wind_offshore']['low']:.2f}
        print(
            f"Threshold capacity factor for wind offshore is {threshold_cf['wind_offshore']['high']:.2f}, and variability threshold is {threshold_variability['wind_offshore']*365:.2f} days."
        )

        wind_offshore_high = offshore_abundant >= threshold_cf["wind_offshore"]["high"]
        wind_offshore_low = offshore_abundant < threshold_cf["wind_offshore"]["high"]

        wind_offshore_reliable = (
            offshore_variability < threshold_variability["wind_offshore"]
        )
        wind_offshore_unreliable = (
            offshore_variability >= threshold_variability["wind_offshore"]
        )

    ###
    # Demand
    ###==============================
    demand_thresh = ds_demand.quantile(0.8).values
    demand_low = ds_demand < demand_thresh
    demand_high = ds_demand >= demand_thresh
    print(
        f"Threshold 0.8 quantile for demand is {np.expm1(demand_thresh):.2f} m2 per grid cell"
    )

    ###
    # Build final raster of "HRHR"-style labels
    ###
    zones = np.full(solar_cf.shape, "", dtype=object)

    LABEL_DICT = [
        [(solar_high, "H"), (solar_mid, "M"), (solar_low, "L")],
        [(solar_reliable, "R"), (solar_unreliable, "U")],
        [(wind_high, "H"), (wind_mid, "M"), (wind_low, "L")],
        [(wind_reliable, "R"), (wind_unreliable, "U")],
        [(demand_high, "h"), (demand_low, "l")],
    ]

    for combo in product(*LABEL_DICT):
        masks, chars = zip(*combo)
        label = "".join(chars)
        mask = land
        for m in masks:
            mask = mask & m  # do NOT use &=, it modifies land mask
        zones[mask.values] = label

    if add_offshore:
        LABEL_DICT_OFFSHORE = [
            [
                (wind_offshore_high, "H"),
                # (wind_offshore_mid, "M"),
                (wind_offshore_low, "L"),
            ],
            [(wind_offshore_reliable, "R"), (wind_offshore_unreliable, "U")],
            [(demand_high, "h"), (demand_low, "l")],
        ]
        for combo in product(*LABEL_DICT_OFFSHORE):
            masks, chars = zip(*combo)
            label = "offshore_" + "".join(chars)  # 'O' prefix for offshore
            mask = offshore
            for m in masks:
                mask = mask & m  # do NOT use &=, it modifies land mask
            zones[mask.values] = label

    zones_da = xr.DataArray(
        zones,
        dims=solar_cf.dims,
        coords=solar_cf.coords,
        name="zones",
    )
    return zones_da  # ds.assign(zones=zones_da)


def _pattern_matches(zone_label, pattern):
    """
    Check if a 4-character zone label matches a pattern.
    Pattern can use 'x' as wildcard for any character.
    E.g., "LxHR" matches "LRHR", "LUHR", etc.
    if offshore then the pattern should also start with offshore and the rest should match
    """
    if zone_label.startswith("offshore_"):
        zone_label = zone_label[len("offshore_") :]
        for zc, pc in zip(zone_label, pattern[len("offshore_") :]):
            if pc != "x" and zc != pc:
                return False
    elif len(zone_label) != 5 or len(pattern) != 5:
        return False
    else:
        for zc, pc in zip(zone_label, pattern):
            if pc != "x" and zc != pc:
                return False
    return True


def plot_land_zones_map(
    ds,
    groups,
    out_path=None,
    figsize=(15, 9),
    extent=[-180, 180, -60, 80],
    plot_legend=True,
    legend_anchor=(0.5, -0.25),
    legend_ncol=7,
):
    """
    Plot map from pre-classified per-pixel labels using group definitions.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with a "zones" variable containing 4-character zone labels.
    groups : dict
        Dictionary mapping group codes to [color, label, [patterns]].
        Patterns can use 'x' as wildcard (e.g., "LxHR" matches any solar reliability).
    out_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig, ax : matplotlib figure and axes

    """
    if extent != [-180, 180, -60, 80]:
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

    # rgb = np.ones((zones.shape[0], zones.shape[1], 3), dtype=float)
    rgb = np.full((zones.shape[0], zones.shape[1], 3), fill_value=np.nan, dtype=float)

    # Collect all unique zone labels in the data (excluding empty strings)
    unique_zones = set(zones.flatten())
    unique_zones.discard("")
    matched_zones = set()

    # Build legend items and apply colors
    legend_items = []
    for group_code, (color, label, patterns) in groups.items():
        color_rgb = mcolors.to_rgb(color)
        legend_items.append((color_rgb, f"{group_code} {label}"))

        # Find all zone labels matching any pattern in this group
        for pattern in patterns:
            for i in range(zones.shape[0]):
                for j in range(zones.shape[1]):
                    if _pattern_matches(zones[i, j], pattern):
                        rgb[i, j] = color_rgb
                        matched_zones.add(zones[i, j])

    # Warn about unmatched zone labels
    unmatched = unique_zones - matched_zones
    if unmatched:
        warnings.warn(
            f"The following zone labels are in the data but not matched by any group pattern: {sorted(unmatched)}"
        )

    patches = [
        mpatches.Patch(color=color, label=label) for color, label in legend_items
    ]

    # Preserve original orientation handling
    # ds_sorted = ds.sortby("latitude")
    rgb_plot = np.flipud(rgb)

    # lat = ds_sorted["latitude"]
    # lon = ds_sorted["longitude"]

    # def plot_robinson(colours, patches, title_suffix=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)
    ax.set_global()
    ax.set_extent(
        extent,  # lon_min, lon_max, lat_min, lat_max
        crs=ccrs.PlateCarree(),
    )
    ax.imshow(
        rgb_plot,
        origin="lower",
        extent=extent,
        transform=ccrs.PlateCarree(),
    )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    # ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=-1)
    # ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=-1)

    # ax.set_title(f"Köppen renewable zones – {title_suffix}", fontsize=14)

    if plot_legend:
        ax.legend(
            handles=patches,
            loc="lower center",
            bbox_to_anchor=legend_anchor,
            ncol=legend_ncol,
            frameon=False,
        )
    # Draw gridlines
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )

    # Choose which labels to show
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True

    # Set grid spacing
    gl.xlocator = plt.FixedLocator(range(-180, 181, 30))  # longitude lines every 30°
    gl.ylocator = plt.FixedLocator(range(-90, 91, 20))  # latitude lines every 15°

    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    # fig.savefig(FIG_DIR / "land_zones.png", dpi=300)
    plt.show()

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig, ax


def labeled_color_palette(
    palette="deep",
    n_colors=None,
    labels=None,
    *,
    ax=None,
    figsize=None,
    edgecolor="white",
    linewidth=1,
    text_kwargs=None,
    return_colors=True,
):
    """
    Draw a seaborn palette as a row of color chips, with a label centered on each chip.

    Parameters
    ----------
    palette : str | list | seaborn palette spec
        Anything you can pass to sns.color_palette().
    n_colors : int | None
        Number of colors. If None, seaborn decides.
    labels : list[str] | None
        Labels to put on chips. If None, uses 1..N.
    ax : matplotlib Axes | None
        Draw into an existing axes.
    figsize : tuple | None
        Figure size if creating a new figure. Defaults to (N, 1.2).
    edgecolor, linewidth : rectangle styling
    text_kwargs : dict | None
        Passed to ax.text (e.g., dict(fontsize=10, fontweight="bold")).
    return_colors : bool
        If True, return the list of RGB tuples.

    Returns
    -------
    colors : list[tuple] | None
    ax : matplotlib Axes

    """
    colors = sns.color_palette(palette, n_colors=n_colors)
    n = len(colors)

    if labels is None:
        labels = [str(i + 1) for i in range(n)]
    if len(labels) != n:
        raise ValueError(f"labels has length {len(labels)} but palette has {n} colors")

    if ax is None:
        if figsize is None:
            figsize = (max(2, n), 1.2)
        _, ax = plt.subplots(figsize=figsize)

    text_kwargs = dict(ha="center", va="center", fontsize=10, fontweight="bold") | (
        text_kwargs or {}
    )

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, (rgb, lab) in enumerate(zip(colors, labels)):
        r, g, b = rgb
        ax.add_patch(
            plt.Rectangle(
                (i, 0), 1, 1, facecolor=rgb, edgecolor=edgecolor, linewidth=linewidth
            )
        )

        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        txt_color = "black" if luminance > 0.55 else "white"

        ax.text(i + 0.5, 0.5, lab, color=txt_color, **text_kwargs)

    return (colors if return_colors else None), ax


def groups_cmap(groups):
    cmap = {k: mcolors.to_rgb(v[0]) for k, v in groups.items()}
    palette = sns.color_palette(cmap.values())
    return labeled_color_palette(palette, labels=[k for k in groups.keys()])


#####
## Bar plot stats
####

from itertools import product


def expand_x(label, replacements=("U", "R")):
    """
    Expand a label containing 'x' into all combinations
    where 'x' is replaced by elements of `replacements`.
    """
    choices = [replacements if ch == "x" else (ch,) for ch in label]
    return ["".join(p) for p in product(*choices)]


def get_base_and_subgroup(ds_zones, groups_dict):
    """
    ds_zones: contains labels (e.g., 'HRHRh', 'HUHUh', etc.) for each grid cell
    Given a label, return its base group and subgroup.
    E.g., for 'HRHRh', return ('B', 'B_u')
    """
    # # small detailed groups with _ul (reliable, demand)
    label_to_maingroup = {}
    label_tosubgroup = {}

    for group, (_, _, sublabels) in groups_dict.items():
        base_group = group.split("_")[0]  # B, W, Ws, S, Sw, P

        for lbl in sublabels:
            for expanded in expand_x(lbl):
                label_to_maingroup[expanded] = base_group
                label_tosubgroup[expanded] = group

    ds_grouped = ds_zones[["zones"]].copy()  # xr.DataArray

    zones = ds_grouped["zones"]
    zones_np = zones.data  # safer than .values

    out_base_groups = np.full(zones.shape, None, dtype=object)
    out_subgroups = np.full(zones.shape, None, dtype=object)

    # base groups
    for k, v in label_to_maingroup.items():
        out_base_groups[zones_np == k] = v

    # subgroups
    for k, v in label_tosubgroup.items():
        out_subgroups[zones_np == k] = v

    ds_grouped["zones_base_grouped"] = xr.DataArray(
        out_base_groups,
        coords=ds_grouped["zones"].coords,
        dims=ds_grouped["zones"].dims,
    )

    ds_grouped["zones_subgrouped"] = xr.DataArray(
        out_subgroups,
        coords=ds_grouped["zones"].coords,
        dims=ds_grouped["zones"].dims,
    )

    # Stats count base groups and subgroups
    counts = (
        ds_grouped[["zones_base_grouped", "zones_subgrouped"]]
        .to_dataframe()
        .dropna()
        .groupby(["zones_base_grouped", "zones_subgrouped"])
        .size()
    )

    percentage_df = (counts / counts.sum() * 100).reset_index(name="percentage")

    return ds_grouped, percentage_df


import geopandas as gpd
import pandas as pd
import rasterio.features
from rasterio.transform import from_bounds

groups_land = {
    k: v
    for k, v in groups_colours.items()
    if not (k.startswith("o") or k.startswith("O"))
}


def plot_stat_group_climate(
    ds_groupped,
    df_percentage,
    add_climate_zone=True,
    title_suffix="Subgroups in main renewable zones",
    out_path=None,
):

    # -----------------------------
    # OPTIONAL: climate-zone stats
    # -----------------------------
    if add_climate_zone:
        koppen_gdf = gpd.read_file(
            str(
                RESOURCES_DIR
                / "user/koeppen-geiger-world-map/c1976_2000_0/c1976_2000.dbf"
            )
        )
        gridcode = pd.read_csv(
            str(
                RESOURCES_DIR
                / "user/koeppen-geiger-world-map/c1976_2000_0/koppen-gridcodes.csv"
            )
        )
        gridcode = gridcode.set_index("gridcode")
        gridcode["zone"] = gridcode["koppen"].str[0]
        koppen_gdf["group"] = koppen_gdf["GRIDCODE"].map(gridcode["zone"])

        koppen_code_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
        koppen_gdf["koppen_code"] = koppen_gdf["group"].map(koppen_code_map)

        lon = ds_groupped["longitude"].values
        lat = ds_groupped["latitude"].values

        transform = from_bounds(
            lon.min(), lat.min(), lon.max(), lat.max(), len(lon), len(lat)
        )

        koppen_raster = rasterio.features.rasterize(
            (
                (geom, code)
                for geom, code in zip(koppen_gdf.geometry, koppen_gdf["koppen_code"])
            ),
            out_shape=ds_groupped["zones"].shape,
            transform=transform,
            fill=0,
            dtype="uint8",
        )

        ds_groupped["koppen_code"] = xr.DataArray(
            koppen_raster,
            coords=ds_groupped["zones"].coords,
            dims=ds_groupped["zones"].dims,
        )

        df = (
            xr.Dataset(
                {
                    "renewable": ds_groupped["zones_base_grouped"],
                    "koppen": ds_groupped["koppen_code"],
                }
            )
            .to_dataframe()
            .dropna()
        )

        counts = df.groupby(["koppen", "renewable"]).size().unstack(fill_value=0)

        counts = counts.rename(
            index={
                0: "Undefined",
                1: "A (Tropical)",
                2: "B (Dry)",
                3: "C (Temperate)",
                4: "D (Continental)",
                5: "E (Polar)",
            }
        )

        koppen_order = [
            "A (Tropical)",
            "B (Dry)",
            "C (Temperate)",
            "D (Continental)",
            "E (Polar)",
            "Undefined",
        ]
        renewable_order = ["B", "W", "Ws", "S", "Sw", "P"]

        counts = (
            counts.reindex(index=koppen_order, columns=renewable_order)
            / counts.values.sum()
            * 100
        )

    # -----------------------------
    # Renewable subgroup stats
    # -----------------------------
    counts_stacked = df_percentage.pivot(
        index="zones_base_grouped",
        columns="zones_subgrouped",
        values="percentage",
    ).fillna(0)

    renewable_order = ["B", "W", "Ws", "S", "Sw", "P"]
    ordered_subgroups = [g for g in groups_land.keys() if g in counts_stacked.columns]

    counts_stacked = counts_stacked.reindex(
        index=renewable_order, columns=ordered_subgroups
    )

    colors_sub = [groups_land[col][0] for col in counts_stacked.columns]

    # -----------------------------
    # Figure + axes (conditional)
    # -----------------------------
    if add_climate_zone:
        fig, axes = plt.subplots(nrows=2, figsize=(6, 6))
        ax0, ax1 = axes
    else:
        fig, ax0 = plt.subplots(figsize=(5, 3))
        ax1 = None

    # --- Panel a: renewable subgroups
    counts_stacked.plot(
        kind="bar",
        stacked=True,
        color=colors_sub,
        edgecolor="none",
        ax=ax0,
        legend=False,
    )

    ax0.set_xlabel("")
    ax0.set_ylabel("Percentage of grid cells")
    ax0.set_title(title_suffix, loc="left")

    # --- Panel b: climate zones (optional)
    if add_climate_zone:
        colors_koppen = [LAND_COLORS[k] for k in counts.columns]

        counts.plot(
            kind="bar",
            stacked=True,
            color=colors_koppen,
            edgecolor="none",
            ax=ax1,
            legend=False,
        )

        ax1.set_xlabel("")
        ax1.set_ylabel("Percentage of grid cells")
        ax1.set_title(
            "Renewable zones by Köppen–Geiger climate",
            loc="left",
        )
        ax1.tick_params(axis="x", rotation=20)

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300)

    plt.show()


#####
## Scatter plot poor-both high demand countries: elevation against precipitation
#####


def plot_scatter_elevation_precipitation(ds_groupped, named_countries, out_path=None):
    geopotential_z0 = xr.open_dataset(
        str(RESOURCES_DIR / "automatic/era5_global_geopotential_surface.nc")
    ).isel(valid_time=0)["z"]

    # convert from 0-360 to -180 to 180:
    geopotential_z0 = geopotential_z0.assign_coords(
        longitude=((geopotential_z0.longitude + 180) % 360) - 180
    )

    # Define standard gravity
    g0 = 9.80665

    elevation = geopotential_z0 / g0

    print(
        f"Elevation min {elevation.min().values:.2f} and max {elevation.max().values:.2f}"
    )

    precip = xr.open_dataset(
        str(
            RESOURCES_DIR
            / "automatic/tp/era5_total_precipitation_year_average_1995_2025.nc"
        )
    )["tp"]

    land = mask_land(ds_groupped["zones"])

    ds_groupped["elevation"] = elevation.where(land).compute()
    ds_groupped["precipitation"] = precip.where(land).compute()

    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    maskcountries = countries.mask(
        ds_groupped["zones"].rename({"latitude": "lat", "longitude": "lon"})
    ).rename({"lat": "latitude", "lon": "longitude"})

    ds_groupped["country_id"] = maskcountries

    # Analyse subgroup poor both and high demand (coloured in red)
    tmp = ds_groupped.to_dataframe().reset_index(drop=False)
    df_red = tmp[tmp["zones_subgrouped"] == "P"][
        ["longitude", "latitude", "zones", "elevation", "precipitation", "country_id"]
    ].copy()

    df_red_stats = pd.DataFrame(
        {
            "latitude": df_red[["latitude", "country_id"]]
            .groupby("country_id")
            .mean()["latitude"],
            "elevation": df_red[["elevation", "country_id"]]
            .groupby("country_id")
            .mean()["elevation"],
            "precipitation": df_red[["precipitation", "country_id"]]
            .groupby("country_id")
            .mean()["precipitation"],
            "P_count": df_red.groupby("country_id").size(),
        }
    )

    country_with_red = df_red["country_id"].unique()
    df_country_total = (
        maskcountries.where(maskcountries.isin(country_with_red))
        .to_dataframe(name="country_id")
        .dropna()
        .groupby("country_id")
        .size()
        .rename("n_grid_cells_total")
        .reset_index()
    )

    # ISO code
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110

    df_country_total["country_name"] = df_country_total["country_id"].map(
        dict(enumerate(countries.names))
    )

    toplot = pd.concat(
        [df_red_stats, df_country_total.set_index("country_id")], axis=1, join="inner"
    )  # .reset_index()
    toplot["percentage"] = toplot["P_count"] / toplot["n_grid_cells_total"] * 100

    # Get stats average elevation and precipitation by country worldwide
    ds = ds_groupped.rename({"latitude": "lat", "longitude": "lon"})
    country_mask = countries.mask(ds)

    elev_by_country = (
        ds["elevation"].groupby(country_mask).mean(skipna=True).mean().values
    )

    precip_by_country = (
        ds["precipitation"].groupby(country_mask).mean(skipna=True).mean().values
    )
    print(
        f"Average by country worldwide elevation {elev_by_country:.2f} [m] and precipitation {precip_by_country:.2f} [m/yr]"
    )

    elev_by_grid_cell = np.mean(ds_groupped["elevation"]).values
    precip_by_grid_cell = np.mean(ds_groupped["precipitation"]).values
    print(
        f"Average by grid worldwide elevation {elev_by_grid_cell:.2f} [m] and precipitation {precip_by_grid_cell:.2f} [m/yr]"
    )

    print(
        f"There are {np.sum(toplot['precipitation'] > precip_by_grid_cell)/len(toplot)*100:.2f} red-area countries with higher precipitation than world average BY GRID CELLS."
    )
    print(
        f"There are {np.sum(toplot['elevation'] < elev_by_grid_cell)/len(toplot)*100:.2f} red-area countries with lower elevation than world average BY GRID CELLS."
    )

    # toplot_big['country_name'].to_list()

    from matplotlib.colors import BoundaryNorm

    fig, ax = plt.subplots(figsize=(10, 6))

    lat_abs = np.abs(toplot["latitude"])
    # Bin latitudes into 10-degree intervals
    lat_binned = np.floor(lat_abs / 6) * 6

    # Define boundaries for discrete color bins
    bounds = np.arange(0, 70, 10)  # [0, 10, 20, 30, 40, 50, 60]
    norm = BoundaryNorm(bounds, ncolors=256)

    sc = ax.scatter(
        toplot["precipitation"],
        toplot["elevation"],
        s=toplot["percentage"] * 10,  # Scale circle size for better visibility
        c=lat_binned,
        alpha=0.6,
        cmap="viridis",
        norm=norm,
    )
    # ax.set_xscale("log")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Absolute latitude |°|")

    ax.axhline(elev_by_grid_cell)
    ax.axvline(precip_by_grid_cell)

    ax.set_xlabel("Average Precipitation [m/yr]")
    ax.set_ylabel("Average Elevation [m]")
    ax.set_title(
        "Countries with high share of 'Both poor high demand' renewable subgroup"
    )

    ## Add selected country names
    for idx, row in toplot[toplot["country_name"].isin(named_countries)].iterrows():
        ax.text(
            row["precipitation"] + 0.05,
            row["elevation"],
            row["country_name"],
            fontsize=8,
        )
    plt.tight_layout()
    plt.show()
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return toplot


######
## Wind rose
######


# Define a function to bin angles into 45-degree chunks with an offset of -22.5 degrees
def bin_angle(angle, nbins=8):
    """
    Bin an angle into one of the specified number of bins.

    Parameters:
        angle (float): The angle in degrees to be binned.
        nbins (int): The number of bins to divide the 360 degrees into. Default is 8.

    Returns:
        str: The label of the bin into which the angle falls,
             formatted as 'start to end' degrees.

    Example:
        >>> bin_angle(45, 8)
        '22.5 to 67.5'
    """
    angle = angle % 360
    width = 360 / nbins
    offset = width / 2
    adjusted_angle = (angle + offset) % 360
    bin_index = int(np.floor(adjusted_angle / width))
    bin_labels = [
        "{:.1f} to {:.1f}".format(i * width - offset, i * width + offset)
        for i in range(nbins)
    ]
    return bin_labels[bin_index]


# Define a function to calculate wind magnitude and convert to km/h
def calculate_magnitude(u, v):
    """
    Calculate the magnitude of a vector given its u and v components.

    Parameters:
        u (array-like): The u component of the vector.
        v (array-like): The v component of the vector.

    Returns:
        numpy.ndarray: The magnitude of the vector.
    """
    metres_per_sec_to_km_h = 3.6
    magnitude = np.sqrt(u**2 + v**2) * metres_per_sec_to_km_h
    return magnitude


# Define a function to bin magnitudes into categories
def bin_magnitude(magnitude, cutoffs):
    """
    Categorizes a given magnitude into a bin based on provided cutoff values.

    Parameters:
        magnitude (float): The magnitude value to be categorized.
        cutoffs (list of int): A list of cutoff values that define the bins.
                               The list should have exactly 4 elements.

    Returns:
        str: A string representing the bin in which the magnitude falls.
    """
    if magnitude < cutoffs[1]:
        if cutoffs[0] == 0:
            return "< {:d}".format(cutoffs[1])
        else:
            return "{:d} < {:d}".format(cutoffs[0], cutoffs[1])
    elif magnitude < cutoffs[2]:
        return "{:d} < {:d}".format(cutoffs[1], cutoffs[2])
    elif magnitude < cutoffs[3]:
        return "{:d} < {:d}".format(cutoffs[2], cutoffs[3])
    else:
        return "{:d} <".format(cutoffs[3])


# Make a function to compute the wind climatology
def windMonthlyClimatology(lat, lon, year_start, year_end):
    """
    Calculate the monthly climatology of wind direction and magnitude.

    This function reads wind data from two NetCDF files, calculates the wind direction
    angle and magnitude, bins the data into specified angle and magnitude bins,
    and returns the percentage of occurrences in each bin.

    Returns:
        bin_percentages (pd.DataFrame): A DataFrame containing the percentage of
                                        occurrences in each angle and magnitude bin.
        angle_bins (list): A list of angle bin labels.
        cutoff_labels (list): A list of magnitude bin labels.
    """

    ds = open_era5_zarr()
    data = (
        ds["u100"]
        .sel(valid_time=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
        .sel(latitude=lat, longitude=lon, method="nearest")
    )
    data2 = (
        ds["v100"]
        .sel(valid_time=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
        .sel(latitude=lat, longitude=lon, method="nearest")
    )

    # Load data
    data_u100_pt = data.compute()
    data_v100_pt = data2.compute()

    # Convert xarray DataArrays to pandas Series
    u_series = data_u100_pt.to_series()
    v_series = data_v100_pt.to_series()

    # Calculate wind direction angle
    angle = np.arctan2(-u_series, -v_series) * (180 / np.pi)

    # Create a DataFrame
    df = pd.DataFrame(
        {"u": u_series, "v": v_series, "Time": u_series.index, "Angle": angle}
    )
    df.sort_values("Time", inplace=True)

    # Add wind magnitude to DataFrame
    # df['Magnitude'] = calculate_magnitude(df['u'], df['v']) # km/h
    df["Magnitude"] = np.sqrt(df["u"] ** 2 + df["v"] ** 2)  # m/s

    # Bin angles and magnitudes
    df["Angle Bin"] = df["Angle"].apply(lambda x: bin_angle(x, 16))
    # nearest_even = int(np.round(df['Magnitude'].median() / 2) * 2)
    # cutoffs = [0, int(nearest_even / 2), nearest_even, int(nearest_even * 1.5)]
    # df['Magnitude Bin'] = df['Magnitude'].apply(lambda x: bin_magnitude(x, cutoffs))
    # Fixed wind-speed bins (m/s)
    cutoffs = [0, 5, 10, 20, np.inf]

    df["Magnitude Bin"] = pd.cut(
        df["Magnitude"],
        bins=cutoffs,
        right=False,
        labels=[
            "0–5",
            "5–10",
            "10–20",
            ">20",
        ],
    )

    # Define bins
    angle_bins = [bin_angle(i * 360 / 16, 16) for i in range(16)]
    cutoff_labels = ["0–5", "5–10", "10–20", ">20"]

    # Count occurrences in bins
    bin_counts = df.groupby(["Angle Bin", "Magnitude Bin"]).size().unstack(fill_value=0)
    bin_counts = bin_counts.reindex(
        index=angle_bins, columns=cutoff_labels, fill_value=0
    )

    # Convert to percentages
    bin_percentages = 100 * bin_counts / bin_counts.sum(axis=1).sum(axis=0)

    # Sort and reindex
    bin_percentages.index = pd.CategoricalIndex(
        bin_percentages.index, categories=angle_bins, ordered=True
    )
    bin_percentages.sort_index(inplace=True)
    bin_percentages = bin_percentages.reindex(columns=cutoff_labels)

    # Get the actual lat/lon used
    nearest_lat = data_u100_pt.latitude.values
    nearest_lng = data_u100_pt.longitude.values

    return bin_percentages, angle_bins, cutoff_labels, nearest_lat, nearest_lng


def plot_wind_rose_on_ax(ax, bin_percentages):
    nbins = bin_percentages.shape[0]
    width = 0.9 * 2 * np.pi / nbins
    angles = np.linspace(0, 2 * np.pi, nbins, endpoint=False) - np.pi / 2

    cumul = bin_percentages.sum(axis=1).values
    ylim1 = max(cumul)

    colors = ["#FFE88F", "#FFC900", "#25408F", "#941333"]

    ax.set_frame_on(False)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, ylim1)
    ax.set_xticks([])
    ax.set_yticks(range(0, int(ylim1) + 5, 5))
    ax.set_yticklabels([])
    # ax.tick_params(axis="y", labelsize=8)

    for i, col in enumerate(bin_percentages.columns):
        ax.bar(
            angles,
            bin_percentages[col],
            width=width,
            bottom=bin_percentages.iloc[:, :i].sum(axis=1),
            color=colors[i],
            edgecolor="black",
            linewidth=0.3,
        )


def add_wind_rose_inset(
    ax_map,
    lon,
    lat,
    bin_percentages,
    size=0.12,
    label=None,
):
    """
    size = fraction of map width
    """

    # Transform lon/lat → display → axes fraction
    proj = ccrs.PlateCarree()
    x_disp, y_disp = ax_map.transData.transform(proj.transform_point(lon, lat, proj))
    x_ax, y_ax = ax_map.transAxes.inverted().transform((x_disp, y_disp))

    inset_ax = ax_map.inset_axes(
        [x_ax - size / 2, y_ax - size / 2, size, size],
        projection="polar",
        zorder=10,
    )

    plot_wind_rose_on_ax(inset_ax, bin_percentages)
    # if label is not None:
    #     ax_map.text(
    #         0,  # 0.5,
    #         0,  # -0.12,
    #         label,
    #         transform=ax_map.transAxes,
    #         ha="center",
    #         va="top",
    #         fontsize=12,
    #         fontweight="bold",
    #     )

    return inset_ax


def _pattern_matches(zone_label, pattern):
    """
    Check if a 4-character zone label matches a pattern.
    Pattern can use 'x' as wildcard for any character.
    E.g., "LxHR" matches "LRHR", "LUHR", etc.
    if offshore then the pattern should also start with offshore and the rest should match
    """
    if zone_label.startswith("offshore_"):
        zone_label = zone_label[len("offshore_") :]
        for zc, pc in zip(zone_label, pattern[len("offshore_") :]):
            if pc != "x" and zc != pc:
                return False
    elif len(zone_label) != 5 or len(pattern) != 5:
        return False
    else:
        for zc, pc in zip(zone_label, pattern):
            if pc != "x" and zc != pc:
                return False
    return True


def plot_land_zones_map(
    ds,
    groups,
    out_path=None,
    figsize=(15, 9),
    extent=[-180, 180, -60, 80],
    plot_legend=True,
    legend_anchor=(0.5, -0.25),
    legend_ncol=7,
    wind_rose=None,  # [(name, (lat, lon)), (name, (lat, lon))]
    size_wind_rose=0.4,
):
    """
    Plot map from pre-classified per-pixel labels using group definitions.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with a "zones" variable containing 4-character zone labels.
    groups : dict
        Dictionary mapping group codes to [color, label, [patterns]].
        Patterns can use 'x' as wildcard (e.g., "LxHR" matches any solar reliability).
    out_path : str, optional
        Path to save the figure.

    Returns
    -------
    fig, ax : matplotlib figure and axes

    """
    if extent != [-180, 180, -60, 80]:
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

    # rgb = np.ones((zones.shape[0], zones.shape[1], 3), dtype=float)
    rgb = np.full((zones.shape[0], zones.shape[1], 3), fill_value=np.nan, dtype=float)

    # Collect all unique zone labels in the data (excluding empty strings)
    unique_zones = set(zones.flatten())
    unique_zones.discard("")
    matched_zones = set()

    # Build legend items and apply colors
    legend_items = []
    for group_code, (color, label, patterns) in groups.items():
        color_rgb = mcolors.to_rgb(color)
        legend_items.append((color_rgb, f"{group_code} {label}"))

        # Find all zone labels matching any pattern in this group
        for pattern in patterns:
            for i in range(zones.shape[0]):
                for j in range(zones.shape[1]):
                    if _pattern_matches(zones[i, j], pattern):
                        rgb[i, j] = color_rgb
                        matched_zones.add(zones[i, j])

    # Warn about unmatched zone labels
    unmatched = unique_zones - matched_zones
    if unmatched:
        warnings.warn(
            f"The following zone labels are in the data but not matched by any group pattern: {sorted(unmatched)}"
        )

    patches = [
        mpatches.Patch(color=color, label=label) for color, label in legend_items
    ]

    # Preserve original orientation handling
    # ds_sorted = ds.sortby("latitude")
    rgb_plot = np.flipud(rgb)

    # lat = ds_sorted["latitude"]
    # lon = ds_sorted["longitude"]

    # def plot_robinson(colours, patches, title_suffix=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=projection)
    ax.set_global()
    ax.set_extent(
        extent,  # lon_min, lon_max, lat_min, lat_max
        crs=ccrs.PlateCarree(),
    )
    ax.imshow(
        rgb_plot,
        origin="lower",
        extent=extent,
        transform=ccrs.PlateCarree(),
    )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    # ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=-1)
    # ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=-1)

    # ax.set_title(f"Köppen renewable zones – {title_suffix}", fontsize=14)

    if plot_legend:
        ax.legend(
            handles=patches,
            loc="lower center",
            bbox_to_anchor=legend_anchor,
            ncol=legend_ncol,
            frameon=False,
        )
    # Draw gridlines
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )

    # Choose which labels to show
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True

    # Set grid spacing
    gl.xlocator = plt.FixedLocator(range(-180, 181, 30))  # longitude lines every 30°
    gl.ylocator = plt.FixedLocator(range(-90, 91, 20))  # latitude lines every 15°

    # Add wind rose(s)
    if wind_rose is not None:
        for name, coords in wind_rose:
            lat_wr, lon_wr = coords

            bin_percentages, _, cutoff_labels, _, _ = windMonthlyClimatology(
                lat_wr, lon_wr, 2020, 2020
            )

            add_wind_rose_inset(
                ax,
                lon_wr,
                lat_wr,
                bin_percentages,
                size=size_wind_rose,  # adjust visually
                label=name,
            )

    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    # fig.savefig(FIG_DIR / "land_zones.png", dpi=300)
    plt.show()

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig, ax
