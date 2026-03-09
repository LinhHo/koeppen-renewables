#!/usr/bin/env python3
"""
Main executable for the Koeppen renewables processing pipeline.

This script:
- parses CLI arguments
- determines the spatial domain (global or user-specified)
- iterates over 20×20° tiles
- runs:
    1) atlas & settlement resampling
    2) ERA5 variability calculations

This is the ONLY entry point users should run.
"""

#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import argparse
import xarray as xr
import numpy as np
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import gc

REPO_ROOT = Path(__file__).parent
sys.path.extend([str(REPO_ROOT), str(REPO_ROOT / "src")])

from src.abundance_atlas import resample_atlas
from src.geo_processing import generate_tiles, determine_pixel_areas
from src.variability import run_seasonal_variability_for_tile, get_complementarity_index
from src.demand import (
    compute_demand_settlement_proximity,
)  # , run_demand_potential_for_tile
from src.storage import compute_lds_for_tile, ALPHA_VALUES

# from src.OLD_plots import plot_all
from config import (
    GLOBAL_DOMAIN,
    TILE_SIZE,
    REFERENCE_RESOLUTION,
    PATHS,
    START_YEAR,
    END_YEAR,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Koeppen pipeline")
    parser.add_argument("--global", action="store_true", help="Full domain")
    parser.add_argument(
        "--bounds", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY")
    )
    parser.add_argument(
        "--tile-size", type=float, default=TILE_SIZE, help="Tile size in degrees"
    )
    parser.add_argument(
        "--years",
        nargs=2,
        type=int,
        metavar=("START_YEAR", "END_YEAR"),
        default=(START_YEAR, END_YEAR),
        help="Start and end year",
    )
    parser.add_argument("--with-complementarity", action="store_true")
    parser.add_argument("--with-lds", action="store_true")

    return parser.parse_args()


def main():
    client = Client(n_workers=4, threads_per_worker=1, memory_limit="24GB")
    args = parse_args()
    domain = (
        dict(zip(["minx", "miny", "maxx", "maxy"], args.bounds))
        if args.bounds
        else GLOBAL_DOMAIN
    )
    start_year, end_year = args.years

    output_dir = REPO_ROOT / "results" / "automatic"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "complementarity").mkdir(exist_ok=True)
    (output_dir / "storage").mkdir(exist_ok=True)

    tiles = generate_tiles(
        domain["minx"], domain["miny"], domain["maxx"], domain["maxy"], args.tile_size
    )

    tmp_path = output_dir / "*.tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    for tile in tiles:
        client.restart()
        # Simplified naming: minx_miny_maxx_maxy
        tile_str = "_".join(map(str, tile))

        # ── Complementarity (unchanged logic, now guarded) ──────────
        if args.with_complementarity:
            comp_file = (
                output_dir
                / f"complementarity/complementarity_{tile_str}_{start_year}_{end_year}.nc"
            )
            if not comp_file.exists():
                corr_index = get_complementarity_index(tile, start_year, end_year)
                with ProgressBar():
                    result = corr_index.compute()
                result.to_netcdf(comp_file, engine="netcdf4")

        # ── Long-duration storage ────────────────────────────────────
        if args.with_lds:
            lds_file = output_dir / f"storage/lds_{tile_str}_{start_year}_{end_year}.nc"
            if not lds_file.exists():
                tmp_path = str(lds_file) + ".tmp"
                try:
                    print(
                        f"\n--- Computing long-duration storage for Tile: {tile_str} ---"
                    )
                    ds_lds = compute_lds_for_tile(
                        tile, start_year, end_year, ALPHA_VALUES
                    )
                    ds_lds.to_netcdf(tmp_path, engine="netcdf4")
                    os.rename(tmp_path, lds_file)  # atomic on POSIX/HPC
                    del ds_lds
                except Exception as e:
                    print(f"  [ERROR] Long-duration storage {tile_str}: {e}")
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            else:
                print(
                    f"\n--- Long-duration storage already exists, skipping Tile: {tile_str} ---"
                )

        # out_file = output_dir / f"processed_{tile_str}_{start_year}_{end_year}.nc"

        # if not out_file.exists():
        #     print(f"\n--- Processing Tile: {tile_str} ---")
        #     try:
        #         # 1. Atlas
        #         print("  -> Resampling Atlas...")
        #         ds_main = resample_atlas(tile, PATHS, resolution=REFERENCE_RESOLUTION)
        #         climatology = xr.Dataset()

        #         # 2. Variability (Wind & Solar)
        #         for label, var in {"solar": "ssrd", "wind": "ws100"}.items():
        #             print(f"  -> Computing {label} variability...")
        #             ds_var, clim = run_seasonal_variability_for_tile(
        #                 tile, var, start_year, end_year
        #             )
        #             # Merge into the main dataset for this tile
        #             ds_main = xr.merge(
        #                 [
        #                     ds_main,
        #                     ds_var.rename(
        #                         {v: f"{label}_{v}" for v in ds_var.data_vars}
        #                     ),
        #                 ],
        #                 join="override",  # ignore slight coordinate mismatches
        #                 compat="override",
        #             )
        #             climatology[var] = clim  # (365 timesteps, longitude, latitude)

        #         # 3. Demand Potential (m2 and fraction)
        #         ds_main["demand_settlement_proximity_m2"] = (
        #             compute_demand_settlement_proximity(tile, PATHS)
        #             .sel(longitude=ds_main.longitude, latitude=ds_main.latitude)
        #             .rename("demand_settlement_proximity_m2")
        #         )
        #         pixel_area = determine_pixel_areas(ds_main["demand_settlement_proximity_m2"].rio.write_crs("EPSG:4326"))
        #         ds_main["demand_proximity_fraction"] = (
        #             ds_main["demand_settlement_proximity_m2"] / pixel_area
        #         ).clip(0, 1)
        # demand_proximity_fraction = (
        #     ds_main["demand_settlement_proximity_m2"] / pixel_area
        # ).clip(0, 1)
        # ds_main["demand_proximity_fraction_normalised"] = (demand_proximity_fraction - demand_proximity_fraction.min())/ (demand_proximity_fraction.max() - demand_proximity_fraction.min())

        #         # Atomic Save (HPC Safe)
        #         tmp_path = str(out_file) + ".tmp"
        #         ds_main.to_netcdf(tmp_path, engine="netcdf4")

        #         # Save climatology to the "climatology" directory
        #         clim_file = (
        #             output_dir
        #             / f"climatology/climatology_{tile_str}_{start_year}_{end_year}.nc"
        #         )
        #         if not clim_file.exists():
        #             climatology.to_netcdf(clim_file, engine="netcdf4")
        #         os.rename(tmp_path, out_file)
        #         print(f"  [SUCCESS] Saved to {out_file.name}")
        #         ds_main.close()
        #         del ds_main

        #     except Exception as e:
        #         print(f"  [ERROR] Tile {tile_str} failed: {e}")
        #         if "tmp_path" in locals() and os.path.exists(tmp_path):
        #             os.remove(tmp_path)
        # else:
        #     print(f"Processed tile for {tile_str} exists. Skipping.")

        gc.collect()
    # client.close()
    # plot_all()


if __name__ == "__main__":
    main()
