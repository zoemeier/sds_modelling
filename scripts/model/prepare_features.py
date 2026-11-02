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

import numpy as np
import pandas as pd
import xarray as xr

from sds.config import (
    RESOLUTION, SPATIAL_RES, SUPPORTED_SPATIAL_RES, YEARS,
    LOGS_PATH, INTERIM_PATH, NDVI_PATH, ACLED_PATH,
    SOIL_PATH, OUT_PATH,
)
from sds.paths import TEMPORAL_TAG, spatial_tag, file_tag, project_relative, YEARS_TAG, ACLED_YEARS_TAG
from sds.log_utils import header, done
from sds.data_utils import should_drop_feature, load_land_mask


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


def prepare_features(resolution, spatial_res, years):
    """Prepare flat feature files, dropping unwanted variables."""
    ftag = file_tag(resolution, spatial_res)
    stag = spatial_tag(spatial_res)

    out_dir = OUT_PATH / ftag
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_PATH / f"prepare_features_{ftag}.log"
    logger = setup_logging(log_file)

    header("Model features", {
        "Resolution": f"{resolution}, {spatial_res}°",
        "Years": f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0]),
        "Output": project_relative(out_dir),
    }, out=logger.info)

    # ── Open source datasets lazily ───────────────────────────────────────
    era5_file = INTERIM_PATH / f"era5_features_{YEARS_TAG.replace('-', '_')}_{ftag}.nc"
    ndvi_file = NDVI_PATH / f"NDVI_{YEARS_TAG}_{ftag}.nc"
    acled_file = ACLED_PATH / f"ACLED_{ACLED_YEARS_TAG}_{ftag}.nc"
    soil_file = SOIL_PATH / f"soil_{YEARS_TAG}_{stag}.nc"

    for f, name in [(era5_file, "ERA5"), (ndvi_file, "NDVI"),
                    (acled_file, "ACLED"), (soil_file, "Soil")]:
        if not f.exists():
            sys.exit(f"ERROR: {name} file not found: {f} (run the preprocessing first)")

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
    logger.info(f"  Features: {len(kept_features)}"
                + (f" ({len(dropped)} dropped: {', '.join(dropped)})" if dropped else ""))

    # land mask
    land_mask = load_land_mask(ref_lat, ref_lon)

    # process yearly files
    for year in years:
        out_file = out_dir / f"features_{year}_{ftag}.nc"
        if out_file.exists():
            logger.info(f"  Already exists: {out_file.name}")
            continue

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
        logger.info(f"  Saved: {out_file.name} ({len(times):,} samples)")

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
        "files": {y: project_relative(out_dir / f"features_{y}_{ftag}.nc") for y in years},
    }
    manifest_path = out_dir / f"features_manifest_{ftag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    done(out_dir, out=logger.info)
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", "-r", choices=list(TEMPORAL_TAG), default=RESOLUTION)
    parser.add_argument("--spatial", "-s", type=float, choices=SUPPORTED_SPATIAL_RES, default=SPATIAL_RES)
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS),
                        help="years to prepare (default: config.YEARS)")
    args = parser.parse_args()
    prepare_features(args.resolution, args.spatial, args.years)
