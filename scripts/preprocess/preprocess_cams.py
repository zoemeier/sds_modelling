"""
Preprocess CAMS reanalysis data at a configurable temporal/spatial resolution.

Temporal: "daily", "6h", "3h"
Spatial:   0.25° or 0.75°  (CAMS native is 0.75°)

Pipeline:
  Step 1: per-year 3-hourly files → resampled mean at target temporal res.
  Step 2: per-year resampled files → one file for all years at target spatial
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

import xarray as xr
from dask.diagnostics import ProgressBar

from sds.config import (
    CAMS_NATIVE_RES, SUPPORTED_SPATIAL_RES, RESOLUTION, SPATIAL_RES,
    RESOLUTION_CONFIG, YEARS, CAMS_PATH,
)
from sds.paths import file_tag, YEARS_TAG
from sds.grid import make_target_grid
from sds.log_utils import header, section, done


os.makedirs(CAMS_PATH, exist_ok=True)


def nc_encoding_compressed(ds):
    return {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}


def step1_aggregate(rule, ttag):
    """Resample 3-hourly CAMS files to the target temporal resolution."""
    section(f"Step 1: Aggregate 3-hourly → {ttag}")

    for year in YEARS:
        input_file  = CAMS_PATH / f"CAMS_{year}.nc"
        target_file = CAMS_PATH / f"CAMS_{year}_{ttag}.nc"

        if target_file.exists():
            print(f"  Already exists: {target_file.name}")
            continue
        if not input_file.exists():
            print(f"  Missing: {input_file.name} (skipped)")
            continue

        ds = xr.open_dataset(input_file, chunks="auto")
        ds_out = ds if rule == "3h" else ds.resample(valid_time=rule).mean()

        print(f"  Saving: {target_file.name}")
        with ProgressBar():
            ds_out.to_netcdf(target_file, engine="h5netcdf",
                             encoding=nc_encoding_compressed(ds_out))
        ds.close()


def step2_combine_and_regrid(ttag, ftag, target_lat, target_lon, spatial_res):
    """Merge the per-year files into one file and regrid to the target spatial grid."""
    section(f"Step 2: Combine years and regrid {CAMS_NATIVE_RES}° → {spatial_res}°")

    target_file = CAMS_PATH / f"CAMS_{YEARS_TAG}_{ftag}.nc"
    if target_file.exists():
        print(f"  Already exists: {target_file.name}")
        return

    files = [CAMS_PATH / f"CAMS_{y}_{ttag}.nc" for y in YEARS]
    missing = [f for f in files if not f.exists()]
    if missing:
        print(f"  Missing: {[m.name for m in missing]}")
        return

    if abs(spatial_res - CAMS_NATIVE_RES) < 1e-6:
        ds = xr.open_mfdataset(files, combine="by_coords", chunks="auto", join="exact")
        ds = ds.sel(
            latitude=slice(target_lat[0], target_lat[-1]),
            longitude=slice(target_lon[0], target_lon[-1]),
        )
        ds = ds.assign_coords(
            latitude=target_lat[:ds.sizes["latitude"]],
            longitude=target_lon[:ds.sizes["longitude"]],
        )
        print(f"  Saving: {target_file.name}")
        with ProgressBar():
            ds.to_netcdf(target_file, engine="h5netcdf",
                         encoding=nc_encoding_compressed(ds))
        ds.close()
        return

    # Interpolate year by year to keep memory bounded
    per_year_regridded = []

    for year, fpath in zip(YEARS, files):
        tmp_file = CAMS_PATH / f"CAMS_{year}_{ftag}.nc"
        if tmp_file.exists():
            print(f"  Already exists: {tmp_file.name}")
            per_year_regridded.append(tmp_file)
            continue

        print(f"  Saving: {tmp_file.name}")
        ds_yr = xr.open_dataset(fpath, chunks="auto")
        ds_yr = ds_yr.interp(latitude=target_lat, longitude=target_lon, method="linear")

        with ProgressBar():
            ds_yr.to_netcdf(tmp_file, engine="h5netcdf",
                            encoding=nc_encoding_compressed(ds_yr))
        ds_yr.close()
        per_year_regridded.append(tmp_file)
        gc.collect()

    print(f"  Saving: {target_file.name}")
    ds = xr.open_mfdataset(per_year_regridded, combine="by_coords",
                           chunks="auto", join="exact")
    with ProgressBar():
        ds.to_netcdf(target_file, engine="h5netcdf",
                     encoding=nc_encoding_compressed(ds))
    ds.close()

    for tmp_file in per_year_regridded:
        tmp_file.unlink(missing_ok=True)


def run_cams_preprocessing(resolution=RESOLUTION, spatial_res=SPATIAL_RES):
    cfg  = RESOLUTION_CONFIG[resolution]
    rule = cfg["rule"]
    ttag = cfg["tag"]
    ftag = file_tag(resolution, spatial_res)
    target_lat, target_lon = make_target_grid(spatial_res)

    header("CAMS preprocessing", {
        "Resolution": f"{resolution}, {spatial_res}°",
        "Grid": f"{len(target_lat)} × {len(target_lon)}",
        "Years": f"{min(YEARS)}-{max(YEARS)}",
    })

    step1_aggregate(rule, ttag)
    step2_combine_and_regrid(ttag, ftag, target_lat, target_lon, spatial_res)
    done(CAMS_PATH / f"CAMS_{YEARS_TAG}_{ftag}.nc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CAMS reanalysis data.")
    parser.add_argument("-r", "--resolution", choices=list(RESOLUTION_CONFIG.keys()),
                        default=RESOLUTION, help=f"Temporal resolution (default: {RESOLUTION})")
    parser.add_argument("-s", "--spatial", type=float, choices=SUPPORTED_SPATIAL_RES,
                        default=SPATIAL_RES, help=f"Spatial resolution in degrees (default: {SPATIAL_RES})")
    args = parser.parse_args()
    run_cams_preprocessing(args.resolution, args.spatial)
