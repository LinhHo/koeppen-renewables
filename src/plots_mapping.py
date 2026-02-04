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

from itertools import product


# Get mask of offshore areas only (mask out solar_CF does not work because it cuts off above ~60N)
def mask_onshore(ds):
    ds = ds["wind_CF"]
    land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(
        ds.longitude, ds.latitude
    )
    return land_mask.notnull()


def classify_land_zones_detailed(ds, ds_demand):
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
    solar_cf = ds["solar_CF"].compute()
    wind_cf = ds["wind_CF"].compute()
    solar_seasonal_var = ds["solar_seasonal_variability"].compute()
    wind_seasonal_var = ds["wind_seasonal_variability"].compute()

    land = mask_onshore(ds)

    ###
    # Abundance (high, medium, low)
    ###
    q_solar = solar_cf.quantile([0.33, 0.66])
    q_wind = wind_cf.quantile([0.33, 0.66])

    print(
        f"Threshold capacity factor for soar are: {q_solar[0]:.2f}, {q_solar[1]:.2f} \n and for wind onshore are {q_wind[0]:.2f}, {q_wind[1]:.2f}"
    )

    solar_low = (solar_cf < q_solar.sel(quantile=0.33)) | solar_cf.isnull()
    solar_mid = (solar_cf >= q_solar.sel(quantile=0.33)) & (
        solar_cf < q_solar.sel(quantile=0.66)
    )
    solar_high = solar_cf >= q_solar.sel(quantile=0.66)

    wind_low = wind_cf < q_wind.sel(quantile=0.33)
    wind_mid = (wind_cf >= q_wind.sel(quantile=0.33)) & (
        wind_cf < q_wind.sel(quantile=0.66)
    )
    wind_high = wind_cf >= q_wind.sel(quantile=0.66)

    ###
    # Reliability (reliable, unreliable)
    ###
    solar_rel_thresh = solar_seasonal_var.where(land).quantile(0.33)
    wind_rel_thresh = wind_seasonal_var.where(land).quantile(0.33)

    solar_reliable = solar_seasonal_var < solar_rel_thresh
    solar_unreliable = solar_seasonal_var >= solar_rel_thresh

    wind_reliable = wind_seasonal_var < wind_rel_thresh
    wind_unreliable = wind_seasonal_var >= wind_rel_thresh

    print(
        f"Thresholds for wind and solar seasonal variability are: {solar_rel_thresh*365:.2f} days and {wind_rel_thresh*365:.2f} days."
    )

    ###
    # Demand
    ###
    demand_thresh = ds_demand.quantile(0.8).values
    demand_low = ds_demand < demand_thresh
    demand_high = ds_demand >= demand_thresh
    print(f"Threshold for demand is {np.expm1(demand_thresh):.2f} m2 per grid cell")

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

    # for s_ab_mask, s_ab_char in [
    #     (solar_high, "H"),
    #     (solar_mid, "M"),
    #     (solar_low, "L"),
    # ]:
    #     for s_rel_mask, s_rel_char in [
    #         (solar_reliable, "R"),
    #         (solar_unreliable, "U"),
    #     ]:
    #         for w_ab_mask, w_ab_char in [
    #             (wind_high, "H"),
    #             (wind_mid, "M"),
    #             (wind_low, "L"),
    #         ]:
    #             for w_rel_mask, w_rel_char in [
    #                 (wind_reliable, "R"),
    #                 (wind_unreliable, "U"),
    #             ]:
    #                 label = f"{s_ab_char}{s_rel_char}{w_ab_char}{w_rel_char}"
    #                 mask = land & s_ab_mask & s_rel_mask & w_ab_mask & w_rel_mask
    #                 zones[mask.values] = label

    zones_da = xr.DataArray(
        zones,
        dims=solar_cf.dims,
        coords=solar_cf.coords,
        name="zones",
    )
    return ds.assign(zones=zones_da)


def _pattern_matches(zone_label, pattern):
    """
    Check if a 4-character zone label matches a pattern.
    Pattern can use 'x' as wildcard for any character.
    E.g., "LxHR" matches "LRHR", "LUHR", etc.
    """
    if len(zone_label) != 5 or len(pattern) != 5:
        return False
    for zc, pc in zip(zone_label, pattern):
        if pc != "x" and zc != pc:
            return False
    return True


def plot_land_zones_map(
    ds, groups, out_path=None, figsize=(15, 9), legend_anchor=(0.5, -0.25)
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
    zones = ds["zones"].values

    rgb = np.ones((zones.shape[0], zones.shape[1], 3), dtype=float)

    # Collect all unique zone labels in the data (excluding empty strings)
    unique_zones = set(zones.flatten())
    unique_zones.discard("")
    matched_zones = set()

    # Build legend items and apply colors
    legend_items = []
    for group_code, (color, label, patterns) in groups.items():
        color_rgb = mcolors.to_rgb(color)
        legend_items.append((color_rgb, f"{group_code} – {label}"))

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
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.set_extent(
        [-180, 180, -60, 80],  # lon_min, lon_max, lat_min, lat_max
        crs=ccrs.PlateCarree(),
    )
    ax.imshow(
        rgb_plot,
        origin="lower",
        extent=[-180, 180, -60, 80],
        transform=ccrs.PlateCarree(),
    )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=-1)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=-1)

    # ax.set_title(f"Köppen renewable zones – {title_suffix}", fontsize=14)

    ax.legend(
        handles=patches,
        loc="lower center",
        bbox_to_anchor=legend_anchor,
        ncol=3,
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

    # fig = plt.figure(figsize=(15, 8))
    # ax = plt.axes(projection=ccrs.PlateCarree())

    # ax.imshow(
    #     rgb_plot,
    #     origin="lower",
    #     extent=[lon.min(), lon.max(), lat.min(), lat.max()],
    #     transform=ccrs.PlateCarree(),
    # )

    # ax.coastlines()
    # ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    # ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=-1)
    # ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=-1)

    # ax.set_title("Koeppen renewable zones – Land", fontsize=14)
    # ax.legend(
    #     handles=patches,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, -0.25),
    #     ncol=3,
    #     frameon=False,
    # )

    # plt.subplots_adjust(bottom=0.25)
    # plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300)
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


def plot_stats_subgroups(percentage_df, groups_dict, title_suffix=""):
    counts_stacked = percentage_df.pivot(
        index="zones_base_grouped",
        columns="zones_subgrouped",
        values="percentage",
    ).fillna(0)

    ordered_subgroups = [g for g in groups_dict.keys() if g in counts_stacked.columns]

    counts_stacked = counts_stacked[ordered_subgroups]

    colors = [groups_dict[col][0] for col in counts_stacked.columns]

    import matplotlib.pyplot as plt

    ax = counts_stacked.plot(
        kind="bar",
        stacked=True,
        color=colors,
        edgecolor="none",
        figsize=(8, 5),
    )

    ax.set_ylabel("Percentage of grid cells")
    ax.set_xlabel("Main renewable zone")
    ax.set_title(f"Renewable zones by group – {title_suffix}")

    ax.legend(
        title=None,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
    )

    plt.tight_layout()
    plt.show()
