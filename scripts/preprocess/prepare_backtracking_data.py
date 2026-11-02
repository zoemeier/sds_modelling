"""
Build the backtracking input file by merging the already-regridded ERA5
wind + derived wind-speed and the CAMS aerosol fields.

Both inputs must already be on the target grid (produced by
preprocess_era5.py and preprocess_cams.py).

Output:
    data/interim/BACKTRACKING_INPUT_<years>_{ftag}.nc

Usage:
    python prepare_backtracking_data.py                  # daily, 0.25°
    python prepare_backtracking_data.py -r 6h -s 0.75
"""

import os
import argparse

import xarray as xr
from dask.diagnostics import ProgressBar

from sds.config import (
    SUPPORTED_SPATIAL_RES, RESOLUTION, SPATIAL_RES, RESOLUTION_CONFIG,
    ERA5_PATH, CAMS_PATH, INTERIM_PATH,
)
from sds.paths import file_tag, YEARS_TAG
from sds.log_utils import header, done


OUT_PATH = INTERIM_PATH
os.makedirs(OUT_PATH, exist_ok=True)


def nc_encoding_compressed(ds):
    return {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}


def run_build_backtracking(resolution=RESOLUTION, spatial_res=SPATIAL_RES):
    ftag = file_tag(resolution, spatial_res)

    header("Backtracking input", {"Resolution": f"{resolution}, {spatial_res}°"})

    out_file = OUT_PATH / f"BACKTRACKING_INPUT_{YEARS_TAG}_{ftag}.nc"
    if out_file.exists():
        print(f"  Already exists: {out_file.name}")
        done(out_file)
        return

    era5_file = ERA5_PATH / f"ERA5_u10_v10_{YEARS_TAG}_{ftag}.nc"
    if not era5_file.exists():
        raise FileNotFoundError(
            f"{era5_file.name} not found. "
            f"Run preprocess_era5.py -r {resolution} -s {spatial_res} first."
        )
    era5 = xr.open_dataset(era5_file, engine="h5netcdf", chunks="auto")

    cams_file = CAMS_PATH / f"CAMS_{YEARS_TAG}_{ftag}.nc"
    if not cams_file.exists():
        raise FileNotFoundError(
            f"{cams_file.name} not found. "
            f"Run preprocess_cams.py -r {resolution} -s {spatial_res} first."
        )
    cams = xr.open_dataset(cams_file, engine="h5netcdf", chunks="auto")

    cams["u10"] = era5["u10"]
    cams["v10"] = era5["v10"]

    ws_file = ERA5_PATH / f"DERIVED_ws_{YEARS_TAG}_{ftag}.nc"
    if ws_file.exists():
        ws = xr.open_dataset(ws_file, chunks="auto")
        if "ws_max" in ws:
            cams["ws_max"] = ws["ws_max"]
        if "ws_mean" in ws:
            cams["ws_mean"] = ws["ws_mean"]
        ws.close()
    else:
        print(f"  WARNING: {ws_file.name} not found, saving without wind speed")

    print(f"  Saving: {out_file.name} ({', '.join(cams.data_vars)})")
    with ProgressBar():
        cams.to_netcdf(out_file, engine="h5netcdf",
                       encoding=nc_encoding_compressed(cams))

    era5.close()
    cams.close()
    done(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SDS backtracking input.")
    parser.add_argument("-r", "--resolution", choices=list(RESOLUTION_CONFIG.keys()),
                        default=RESOLUTION, help=f"Temporal resolution (default: {RESOLUTION})")
    parser.add_argument("-s", "--spatial", type=float, choices=SUPPORTED_SPATIAL_RES,
                        default=SPATIAL_RES, help=f"Spatial resolution in degrees (default: {SPATIAL_RES})")
    args = parser.parse_args()
    run_build_backtracking(args.resolution, args.spatial)
