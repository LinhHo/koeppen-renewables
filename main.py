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

REPO_ROOT = Path(__file__).parent
sys.path.extend([str(REPO_ROOT), str(REPO_ROOT / "src")])

from src.abundance_atlas import resample_atlas
from src.geo_processing import generate_tiles
from src.variability import run_variability_for_tile
from src.demand import run_demand_potential_for_tile
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
    return parser.parse_args()


def main():
    client = Client(n_workers=2, threads_per_worker=2, memory_limit="4GB")
    args = parse_args()
    domain = (
        dict(zip(["minx", "miny", "maxx", "maxy"], args.bounds))
        if args.bounds
        else GLOBAL_DOMAIN
    )

    output_dir = REPO_ROOT / "results" / "automatic"
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles = generate_tiles(
        domain["minx"], domain["miny"], domain["maxx"], domain["maxy"], args.tile_size
    )

    for tile in tiles:
        # Simplified naming: minx_miny_maxx_maxy
        tile_str = "_".join(map(str, tile))
        out_file = output_dir / f"processed_{tile_str}_{START_YEAR}_{END_YEAR}.nc"

        if not out_file.exists():
            print(f"\n--- Processing Tile: {tile_str} ---")
            try:
                # 1. Atlas
                print("  -> Resampling Atlas...")
                ds_main = resample_atlas(tile, PATHS, resolution=REFERENCE_RESOLUTION)

                # 2. Variability (Wind & Solar)
                for label, var in {"solar": "ssrd", "wind": "ws100"}.items():
                    print(f"  -> Computing {label} variability...")
                    ds_var = run_variability_for_tile(tile, var)
                    # Merge into the main dataset for this tile
                    ds_main = xr.merge(
                        [
                            ds_main,
                            ds_var.rename(
                                {v: f"{label}_{v}" for v in ds_var.data_vars}
                            ),
                        ],
                        join="override",  # ignore slight coordinate mismatches
                    )
                # 3. Demand Potential
                # incl. demand_potential, demand_temperature_induced, demand_settlement_proximity
                print("  -> Computing demand potential...")
                ds_demand = run_demand_potential_for_tile(tile, PATHS)
                ds_main = xr.merge([ds_main, ds_demand], join="override")

                # Atomic Save (HPC Safe)
                tmp_path = str(out_file) + ".tmp"
                ds_main.to_netcdf(tmp_path)
                os.rename(tmp_path, out_file)
                print(f"  [SUCCESS] Saved to {out_file.name}")
                ds_main.close()
                del ds_main

            except Exception as e:
                print(f"  [ERROR] Tile {tile_str} failed: {e}")
                if "tmp_path" in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            print(f"Processed tile for {tile_str} exists. Skipping.")

        client.restart()
    client.close()


if __name__ == "__main__":
    main()
