"""
Flatten ERA5 + NDVI + ACLED + soil features into per-year flat NetCDF files.
Static soil features are broadcast to the time dimension automatically.
Land-sea mask is applied before flattening.

Usage:
    python prepare_features.py --resolution daily --spatial 0.25 --years 2015 2016 2017
"""

import argparse
import sys
import logging
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from config import (
    PROJECT_ROOT, DATA_PATH, ERA5_PATH, NDVI_PATH, ACLED_PATH, SOIL_PATH,
    OUT_PATH, LAND_MASK_FILE, LSM_THRESHOLD, TEMPORAL_TAG,
    spatial_tag, file_tag, should_drop_feature,
)


def setup_logging(log_file=None):
    logger = logging.getLogger("prepare_features")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def normalise_time(ds):
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if ds["time"].dtype == object or hasattr(ds["time"].values[0], "datetime"):
        ds["time"] = pd.to_datetime([
            str(t)[:19] if hasattr(t, "datetime") else t
            for t in ds["time"].values
        ])
    return ds


def normalise_coords(ds):
    rename = {}
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds


def load_land_mask(target_lat, target_lon):
    lsm_ds = xr.open_dataset(LAND_MASK_FILE, engine="cfgrib").squeeze()
    lsm_var = next((v for v in ("lsm", "land_sea_mask", "LSM") if v in lsm_ds),
                   list(lsm_ds.data_vars)[0])
    lsm = lsm_ds[lsm_var]
    lsm = lsm.sel(latitude=target_lat, longitude=target_lon, method="nearest")
    mask = (lsm.values > LSM_THRESHOLD).astype(bool)
    lsm_ds.close()
    return mask


def prepare_features(resolution, spatial_res, years):
    """Prepare flat feature files, dropping unwanted variables."""
    ftag = file_tag(resolution, spatial_res)
    stag = spatial_tag(spatial_res)
    ttag = TEMPORAL_TAG[resolution]

    out_dir = OUT_PATH / ftag
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = PROJECT_ROOT / "logs" / f"prepare_features_{ftag}.log"
    logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info(f"PREPARE FEATURES: {ftag}")
    logger.info(f"  Years: {years}")
    logger.info("=" * 60)

    # ── Open source datasets lazily ───────────────────────────────────────
    era5_file = ERA5_PATH / f"era5_features_2015_2024_{ftag}.nc"
    ndvi_file = NDVI_PATH / f"NDVI_2015-2024_{ftag}.nc"
    acled_file = ACLED_PATH / f"ACLED_2014-2024_{ftag}.nc"
    soil_file = SOIL_PATH / f"soil_2015-2024_{stag}.nc"

    for f, name in [(era5_file, "ERA5"), (ndvi_file, "NDVI"),
                    (acled_file, "ACLED"), (soil_file, "Soil")]:
        if not f.exists():
            logger.error(f"  {name} file not found: {f}")
            sys.exit(1)

    ds_era5 = normalise_coords(normalise_time(
        xr.open_dataset(era5_file, chunks={"time": 365})
    ))
    drop_vars = [v for v in ("number", "step", "surface", "expver") if v in ds_era5]
    if drop_vars:
        ds_era5 = ds_era5.drop_vars(drop_vars)

    ds_ndvi = normalise_coords(normalise_time(
        xr.open_dataset(ndvi_file, chunks="auto")
    ))
    ds_acled = normalise_coords(normalise_time(
        xr.open_dataset(acled_file, chunks="auto")
    ))

    ds_soil = normalise_coords(xr.open_dataset(soil_file)).load()

    ref_lat = ds_era5["latitude"].values
    ref_lon = ds_era5["longitude"].values

    ds_soil = ds_soil.assign_coords(
        latitude=ref_lat[:len(ds_soil.latitude)],
        longitude=ref_lon[:len(ds_soil.longitude)],
    )
    soil_vars = list(ds_soil.data_vars)

    era5_vars_raw = list(ds_era5.data_vars)
    overlap_vars = [v for v in soil_vars if v in era5_vars_raw]
    era5_keep_vars = [v for v in era5_vars_raw if v not in overlap_vars]
    ndvi_vars = [v for v in ds_ndvi.data_vars if v not in ("spatial_ref",)]
    acled_vars = list(ds_acled.data_vars)
    all_feature_names = era5_keep_vars + soil_vars + ndvi_vars + acled_vars

    # drop unwanted features
    kept_features = [v for v in all_feature_names if not should_drop_feature(v)]
    dropped = [v for v in all_feature_names if should_drop_feature(v)]
    logger.info(f"  Total features before drop: {len(all_feature_names)}")
    logger.info(f"  Dropped features ({len(dropped)}): {dropped}")
    logger.info(f"  Kept features ({len(kept_features)}): {len(kept_features)}")

    # land mask
    land_mask = load_land_mask(ref_lat, ref_lon)
    n_land = int(land_mask.sum())
    logger.info(f"  Land cells: {n_land:,} / {land_mask.size:,}")

    # process yearly files
    for year in years:
        out_file = out_dir / f"features_{year}_{ftag}.nc"
        if out_file.exists():
            logger.info(f"  {year}: already exists, skipping")
            continue

        logger.info(f"\n  Processing {year} ...")
        yr_start, yr_end = f"{year}-01-01", f"{year}-12-31"

        era5_yr = ds_era5.sel(time=slice(yr_start, yr_end)).load()
        if overlap_vars:
            era5_yr = era5_yr.drop_vars(overlap_vars, errors="ignore")

        ndvi_yr = ds_ndvi.sel(time=slice(yr_start, yr_end)).load()
        acled_yr = ds_acled.sel(time=slice(yr_start, yr_end)).load()

        ndvi_yr = ndvi_yr.assign_coords(
            latitude=ref_lat[:len(ndvi_yr.latitude)],
            longitude=ref_lon[:len(ndvi_yr.longitude)],
        )
        acled_yr = acled_yr.assign_coords(
            latitude=ref_lat[:len(acled_yr.latitude)],
            longitude=ref_lon[:len(acled_yr.longitude)],
        )

        era5_yr, ndvi_yr, acled_yr = xr.align(
            era5_yr, ndvi_yr, acled_yr, join="inner"
        )
        nt = len(era5_yr.time)

        ds_yr = xr.merge([era5_yr, ds_soil, ndvi_yr, acled_yr], compat="override")

        # drop unwanted variables from the merged dataset (specified in config.py)
        drop_from_ds = [v for v in ds_yr.data_vars if should_drop_feature(v)]
        if drop_from_ds:
            ds_yr = ds_yr.drop_vars(drop_from_ds)

        land_mask_da = xr.DataArray(
            land_mask, dims=["latitude", "longitude"],
            coords={"latitude": ref_lat, "longitude": ref_lon},
        )
        ds_yr = ds_yr.where(land_mask_da)

        ds_flat = ds_yr.stack(sample=("time", "latitude", "longitude"))
        land_mask_flat = np.tile(land_mask.ravel(), nt)
        ds_flat = ds_flat.isel(sample=np.where(land_mask_flat)[0])

        sample_idx = ds_flat["sample"].values
        times = np.array([s[0] for s in sample_idx])
        lats = np.array([s[1] for s in sample_idx], dtype="float32")
        lons = np.array([s[2] for s in sample_idx], dtype="float32")

        ds_flat = ds_flat.reset_index("sample", drop=True)
        ds_flat = ds_flat.assign_coords(sample=np.arange(len(times)))
        ds_flat["valid_time"] = ("sample", times)
        ds_flat["lat"] = ("sample", lats)
        ds_flat["lon"] = ("sample", lons)

        encoding = {v: {"zlib": True, "complevel": 4} for v in ds_flat.data_vars}
        ds_flat.to_netcdf(out_file, engine="h5netcdf", encoding=encoding)
        n_valid = len(times)
        size_mb = out_file.stat().st_size / (1024 * 1024)
        logger.info(f"    Done — {size_mb:.1f} MB, {n_valid:,} samples, "
                    f"{len(list(ds_flat.data_vars))} vars")

        del era5_yr, ndvi_yr, acled_yr, ds_yr, ds_flat
        gc.collect()

    ds_era5.close(); ds_ndvi.close(); ds_acled.close()

    manifest = {
        "ftag": ftag,
        "resolution": resolution,
        "spatial_res": spatial_res,
        "feature_names": kept_features,
        "dropped_features": dropped,
        "years": years,
        "files": {y: str(out_dir / f"features_{y}_{ftag}.nc") for y in years},
    }
    manifest_path = out_dir / f"features_manifest_{ftag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"\nManifest: {manifest_path}")
    logger.info("FEATURE PREPARATION COMPLETE")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", "-r", choices=["daily", "6h", "3h"], default="daily")
    parser.add_argument("--spatial", "-s", type=float, choices=[0.25, 0.75], default=0.25)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    args = parser.parse_args()
    prepare_features(args.resolution, args.spatial, args.years)
