"""
Preprocess CAMS reanalysis data at a configurable temporal/spatial resolution.

Temporal: "1D" (daily), "6h", "3h"
Spatial:   0.25° or 0.75°  (CAMS native is 0.75°)

Pipeline:
  Step 1: per-year 3-hourly files → resampled mean at target temporal res.
  Step 2: per-year resampled files → single 2015-2024 file at target spatial
          grid. At 0.25°, each year is interpolated separately and then
          concatenated to keep memory low; at 0.75° we just clip to domain.

Usage:
    python preprocess_cams.py                      # daily, 0.25°
    python preprocess_cams.py -r 6h -s 0.75
    python preprocess_cams.py --resolution 3h --spatial 0.25
"""

import os
import gc
import argparse
from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar

from grid_config import (
    RESOLUTION_CONFIG, YEARS, make_target_grid, file_tag,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data"
CAMS_PATH = DATA_PATH / "cams"
os.makedirs(CAMS_PATH, exist_ok=True)


def nc_encoding_compressed(ds):
    return {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}


def step1_aggregate(rule, ttag):
    """Resample 3-hourly CAMS files to the target temporal resolution."""
    print("=" * 70)
    print(f"STEP 1: Aggregate CAMS 3-hourly → {ttag} (rule={rule})")
    print("=" * 70)

    for year in YEARS:
        input_file  = CAMS_PATH / f"CAMS_{year}.nc"
        target_file = CAMS_PATH / f"CAMS_{year}_{ttag}.nc"

        if target_file.exists():
            print(f"  Already exists: {target_file.name}")
            continue
        if not input_file.exists():
            print(f"  {input_file.name} not found — skipping")
            continue

        print(f"\n  Processing {year} (rule={rule})")
        ds = xr.open_dataset(input_file, chunks="auto")
        ds_out = ds if rule == "3h" else ds.resample(valid_time=rule).mean()

        print(f"    → {target_file.name}")
        with ProgressBar():
            ds_out.to_netcdf(target_file, engine="h5netcdf",
                             encoding=nc_encoding_compressed(ds_out))
        ds.close()


def step2_combine_and_regrid(ttag, ftag, target_lat, target_lon, spatial_res):
    """Merge per-year files into 2015-2024 and regrid to target spatial grid."""
    print("\n" + "=" * 70)
    print(f"STEP 2: Combine CAMS {ttag} → 2015-2024 and regrid to {spatial_res}°")
    print("=" * 70)

    target_file = CAMS_PATH / f"CAMS_2015-2024_{ftag}.nc"
    if target_file.exists():
        print(f"  Already exists: {target_file.name}")
        return

    files = [CAMS_PATH / f"CAMS_{y}_{ttag}.nc" for y in YEARS]
    missing = [f for f in files if not f.exists()]
    if missing:
        print(f"  Missing: {[m.name for m in missing]}")
        return

    cams_native = 0.75

    if abs(spatial_res - cams_native) < 1e-6:
        print(f"  Regridding: native {cams_native}° — clipping to domain")
        ds = xr.open_mfdataset(files, combine="by_coords", chunks="auto", join="exact")
        ds = ds.sel(
            latitude=slice(target_lat[0], target_lat[-1]),
            longitude=slice(target_lon[0], target_lon[-1]),
        )
        ds = ds.assign_coords(
            latitude=target_lat[:ds.sizes["latitude"]],
            longitude=target_lon[:ds.sizes["longitude"]],
        )
        print(f"  → {target_file.name}")
        with ProgressBar():
            ds.to_netcdf(target_file, engine="h5netcdf",
                         encoding=nc_encoding_compressed(ds))
        ds.close()
        return

    # Interpolate year by year to keep memory bounded
    print(f"  Regridding: {cams_native}° → {spatial_res}° (linear, per year)")
    per_year_regridded = []

    for year, fpath in zip(YEARS, files):
        tmp_file = CAMS_PATH / f"CAMS_{year}_{ftag}.nc"
        if tmp_file.exists():
            print(f"  Already exists: {tmp_file.name}")
            per_year_regridded.append(tmp_file)
            continue

        print(f"  Regridding {year} ...")
        ds_yr = xr.open_dataset(fpath, chunks="auto")
        ds_yr = ds_yr.interp(latitude=target_lat, longitude=target_lon, method="linear")

        with ProgressBar():
            ds_yr.to_netcdf(tmp_file, engine="h5netcdf",
                            encoding=nc_encoding_compressed(ds_yr))
        ds_yr.close()
        per_year_regridded.append(tmp_file)
        gc.collect()

    print(f"\n  Concatenating {len(per_year_regridded)} years → {target_file.name}")
    ds = xr.open_mfdataset(per_year_regridded, combine="by_coords",
                           chunks="auto", join="exact")
    with ProgressBar():
        ds.to_netcdf(target_file, engine="h5netcdf",
                     encoding=nc_encoding_compressed(ds))
    ds.close()

    print(f"  Cleaning up per-year regridded files ...")
    for tmp_file in per_year_regridded:
        tmp_file.unlink(missing_ok=True)


def run_cams_preprocessing(resolution="1D", spatial_res=0.25):
    cfg  = RESOLUTION_CONFIG[resolution]
    rule = cfg["rule"]
    ttag = cfg["tag"]
    ftag = file_tag(ttag, spatial_res)
    target_lat, target_lon = make_target_grid(spatial_res)

    print(f"\n{'#' * 70}")
    print(f"  CAMS preprocessing")
    print(f"  Temporal: {resolution} ({ttag})  |  Spatial: {spatial_res}°")
    print(f"  File tag: {ftag}  |  Grid: {len(target_lat)}×{len(target_lon)}")
    print(f"{'#' * 70}\n")

    step1_aggregate(rule, ttag)
    step2_combine_and_regrid(ttag, ftag, target_lat, target_lon, spatial_res)
    print(f"\nDone: CAMS_2015-2024_{ftag}.nc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CAMS reanalysis data.")
    parser.add_argument("-r", "--resolution", choices=list(RESOLUTION_CONFIG.keys()),
                        default="1D", help="Temporal resolution (default: 1D)")
    parser.add_argument("-s", "--spatial", type=float, choices=[0.25, 0.75],
                        default=0.25, help="Spatial resolution in degrees (default: 0.25)")
    args = parser.parse_args()
    run_cams_preprocessing(args.resolution, args.spatial)
