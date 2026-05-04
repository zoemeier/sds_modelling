"""
Build the backtracing input file by merging the already-regridded ERA5
wind + derived wind-speed and the CAMS aerosol fields.

Both inputs must already be on the target grid (produced by
preprocess_era5.py and preprocess_cams.py).

Output:
    data/interim/BACKTRACING_INPUT_2015-2024_{ftag}.nc

Usage:
    python prepare_backtracking_data.py                  # daily, 0.25°
    python prepare_backtracking_data.py -r 6h -s 0.75
"""

import os
import argparse
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

from grid_config import RESOLUTION_CONFIG, file_tag


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data"
ERA5_PATH = DATA_PATH / "era5"
CAMS_PATH = DATA_PATH / "cams"
OUT_PATH = DATA_PATH / "interim"
os.makedirs(OUT_PATH, exist_ok=True)


def nc_encoding_compressed(ds):
    return {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}


def run_build_backtracing(resolution="1D", spatial_res=0.25):
    ftag = file_tag(RESOLUTION_CONFIG[resolution]["tag"], spatial_res)

    print(f"\n{'#' * 70}")
    print(f"  Build backtracing input — {resolution}, {spatial_res}° ({ftag})")
    print(f"{'#' * 70}\n")

    out_file = OUT_PATH / f"BACKTRACING_INPUT_2015-2024_{ftag}.nc"
    if out_file.exists():
        print(f"  Already exists: {out_file.name}")
        return

    era5_file = ERA5_PATH / f"ERA5_u10_v10_2015-2024_{ftag}.nc"
    if not era5_file.exists():
        raise FileNotFoundError(
            f"{era5_file.name} not found. "
            f"Run preprocess_era5.py -r {resolution} -s {spatial_res} first."
        )
    era5 = xr.open_dataset(era5_file, engine="h5netcdf", chunks="auto")

    cams_file = CAMS_PATH / f"CAMS_2015-2024_{ftag}.nc"
    if not cams_file.exists():
        raise FileNotFoundError(
            f"{cams_file.name} not found. "
            f"Run preprocess_cams.py -r {resolution} -s {spatial_res} first."
        )
    cams = xr.open_dataset(cams_file, engine="h5netcdf", chunks="auto")

    print("  Merging ERA5 wind + CAMS aerosol (grids already aligned)")
    cams["u10"] = era5["u10"]
    cams["v10"] = era5["v10"]

    ws_file = ERA5_PATH / f"DERIVED_ws_2015-2024_{ftag}.nc"
    if ws_file.exists():
        print(f"  Adding ERA5 ws (max + mean)")
        ws = xr.open_dataset(ws_file, chunks="auto")
        if "ws_max" in ws:
            cams["ws_max"] = ws["ws_max"]
        if "ws_mean" in ws:
            cams["ws_mean"] = ws["ws_mean"]
        ws.close()
    else:
        print(f"  WARNING: {ws_file.name} not found — skipping wind speed")

    print(f"  → {out_file.name}")
    print(f"    Variables: {list(cams.data_vars)}")
    with ProgressBar():
        cams.to_netcdf(out_file, engine="h5netcdf",
                       encoding=nc_encoding_compressed(cams))

    era5.close()
    cams.close()
    print(f"\nDone: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SDS backtracing input.")
    parser.add_argument("-r", "--resolution", choices=list(RESOLUTION_CONFIG.keys()),
                        default="1D", help="Temporal resolution (default: 1D)")
    parser.add_argument("-s", "--spatial", type=float, choices=[0.25, 0.75],
                        default=0.25, help="Spatial resolution in degrees (default: 0.25)")
    args = parser.parse_args()
    run_build_backtracing(args.resolution, args.spatial)
