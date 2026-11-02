"""
Preprocess ERA5 reanalysis data at a configurable temporal/spatial resolution.

Pipeline:
  Step 1: aggregate hourly ERA5 GRIBs to the target temporal resolution,
          per year. Computes ws (max+mean), wd (vector mean), and rh from
          hourly u10/v10/t2m/dt2m before resampling.
  Step 2: combine the per-year files into one file and regrid to target
          spatial grid (block-mean coarsen for 0.25 → 0.75).
  Step 3: build derived feature files (ws max+mean combined, rv, wd, rh).
  Step 4: merge all features, apply land-sea mask, add static fields,
          rolling stats, lags, log-transform precip.

Usage:
    python preprocess_era5.py                     # daily, 0.25°
    python preprocess_era5.py -r 6h -s 0.75
    python preprocess_era5.py --resolution 3h --spatial 0.25
"""

import os
import argparse

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from sds.config import (
    ERA5_VARIABLES,
    LSM_THRESHOLD, SUPPORTED_SPATIAL_RES, RESOLUTION, SPATIAL_RES,
    RESOLUTION_CONFIG, YEARS, ERA5_PATH, INTERIM_PATH,
    ERA5_NATIVE_RES, LAND_MASK_FILE,
)
from sds.paths import file_tag, YEARS_TAG
from sds.grid import make_target_grid, WIND_VARS
from sds.log_utils import header, section, done


OUT_PATH  = INTERIM_PATH
os.makedirs(ERA5_PATH, exist_ok=True)
os.makedirs(OUT_PATH, exist_ok=True)

# Forecast fields that need (time, step) → valid_time flattening
FORECAST_STEP_VARS = {"tp"}


def nc_encoding_compressed(ds):
    return {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}


def make_era5_regridder(spatial_res, target_lat, target_lon):
    """Identity at native 0.25°; block-mean coarsen otherwise."""
    if abs(spatial_res - ERA5_NATIVE_RES) < 1e-6:
        return lambda ds: ds

    coarsen_factor = int(round(spatial_res / ERA5_NATIVE_RES))

    def _regrid(ds):
        ds = ds.coarsen(
            latitude=coarsen_factor, longitude=coarsen_factor, boundary="pad"
        ).mean()
        ds = ds.assign_coords(
            latitude=target_lat[:ds.sizes["latitude"]],
            longitude=target_lon[:ds.sizes["longitude"]],
        )
        return ds

    return _regrid


def step1_aggregate(rule, ttag):
    """Resample hourly ERA5 GRIBs to the target temporal resolution per year."""
    section(f"Step 1: Aggregate hourly → {ttag}")

    for short, agg in ERA5_VARIABLES.items():
        for year in YEARS:
            target = ERA5_PATH / f"ERA5_{short}_{year}_{ttag}.nc"
            if target.exists():
                continue

            input_file = ERA5_PATH / f"ERA5_{short}_{year}.grib"
            if not input_file.exists():
                print(f"  Missing: {input_file.name} (skipped)")
                continue

            ds = xr.open_dataset(
                input_file, engine="cfgrib",
                chunks={"time": "auto", "latitude": "auto", "longitude": "auto"},
            )

            # Forecast fields have (time, step) — flatten to valid_time
            if short in FORECAST_STEP_VARS:
                valid_times = ds["valid_time"].values
                ds = ds.stack(valid_time=("time", "step"))
                ds = ds.drop_vars(["valid_time", "time", "step"], errors="ignore")
                ds = ds.assign_coords(valid_time=("valid_time", valid_times.flatten()))
                ds = ds.transpose("valid_time", "latitude", "longitude")
            else:
                ds = ds.drop_vars(["valid_time", "step"], errors="ignore")
                if "time" in ds.dims:
                    ds = ds.rename({"time": "valid_time"})

            ds = ds.sel(valid_time=slice(f"{year}-01-01", f"{year}-12-31 23:59:59"))

            drop_extra = ["surface", "number", "step", "depthBelowLandLayer"]

            if agg == "sum_mm":
                ds_out = ds.resample(valid_time=rule).sum() * 1000
            else:  # "mean"
                ds_out = ds.resample(valid_time=rule).mean()
            ds_out = ds_out.drop_vars(drop_extra, errors="ignore")
            print(f"  Saving: {target.name}")
            with ProgressBar():
                ds_out.to_netcdf(target, engine="h5netcdf",
                                 encoding=nc_encoding_compressed(ds_out))

            ds.close()

    # ws (max+mean) and wd (vector mean) from hourly u10/v10
    for year in YEARS:
        ws_targets = {
            "max":  ERA5_PATH / f"ERA5_ws_{year}_{ttag}_max.nc",
            "mean": ERA5_PATH / f"ERA5_ws_{year}_{ttag}_mean.nc",
        }
        wd_target = ERA5_PATH / f"ERA5_wd_{year}_{ttag}_mean.nc"

        all_exist = all(t.exists() for t in ws_targets.values()) and wd_target.exists()
        if all_exist:
            continue

        u_file = ERA5_PATH / f"ERA5_u10_{year}.grib"
        v_file = ERA5_PATH / f"ERA5_v10_{year}.grib"
        if not u_file.exists() or not v_file.exists():
            print(f"  Missing: u10 / v10 for {year} (ws, wd skipped)")
            continue

        ds_u = xr.open_dataset(u_file, engine="cfgrib",
                                chunks={"time": "auto", "latitude": "auto", "longitude": "auto"})
        ds_v = xr.open_dataset(v_file, engine="cfgrib",
                                chunks={"time": "auto", "latitude": "auto", "longitude": "auto"})

        ds_u = ds_u.drop_vars(["valid_time", "step"], errors="ignore")
        ds_v = ds_v.drop_vars(["valid_time", "step"], errors="ignore")
        if "time" in ds_u.dims: ds_u = ds_u.rename({"time": "valid_time"})
        if "time" in ds_v.dims: ds_v = ds_v.rename({"time": "valid_time"})

        ds_u = ds_u.sel(valid_time=slice(f"{year}-01-01", f"{year}-12-31 23:59:59"))
        ds_v = ds_v.sel(valid_time=slice(f"{year}-01-01", f"{year}-12-31 23:59:59"))

        u10 = ds_u["u10"]
        v10 = ds_v["v10"]
        drop_extra = ["surface", "number", "step", "depthBelowLandLayer"]

        ws_hourly = np.sqrt(u10**2 + v10**2)
        ws_hourly.name = "ws"
        ws_ds = ws_hourly.to_dataset()

        if not ws_targets["max"].exists():
            ds_max = ws_ds.resample(valid_time=rule).max()
            ds_max = ds_max.drop_vars(drop_extra, errors="ignore")
            print(f"  Saving: {ws_targets['max'].name}")
            with ProgressBar():
                ds_max.to_netcdf(ws_targets["max"], engine="h5netcdf",
                                 encoding=nc_encoding_compressed(ds_max))

        if not ws_targets["mean"].exists():
            ds_mean = ws_ds.resample(valid_time=rule).mean()
            ds_mean = ds_mean.drop_vars(drop_extra, errors="ignore")
            print(f"  Saving: {ws_targets['mean'].name}")
            with ProgressBar():
                ds_mean.to_netcdf(ws_targets["mean"], engine="h5netcdf",
                                  encoding=nc_encoding_compressed(ds_mean))

        # Wind direction: average sin/cos and atan2 back. Meteorological
        # convention (direction wind comes from, clockwise from north).
        if not wd_target.exists():
            wd_hourly_rad = np.arctan2(-u10, -v10)
            sin_wd = np.sin(wd_hourly_rad)
            cos_wd = np.cos(wd_hourly_rad)

            sin_mean = sin_wd.resample(valid_time=rule).mean()
            cos_mean = cos_wd.resample(valid_time=rule).mean()

            wd_mean = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360) % 360
            wd_mean.name = "wd"
            wd_ds = wd_mean.to_dataset()
            wd_ds = wd_ds.drop_vars(drop_extra, errors="ignore")
            print(f"  Saving: {wd_target.name}")
            with ProgressBar():
                wd_ds.to_netcdf(wd_target, engine="h5netcdf",
                                encoding=nc_encoding_compressed(wd_ds))

        ds_u.close()
        ds_v.close()

    # Relative humidity (Magnus) from hourly t2m/dt2m → mean
    for year in YEARS:
        target = ERA5_PATH / f"ERA5_rh_{year}_{ttag}_mean.nc"
        if target.exists():
            continue

        t2m_file  = ERA5_PATH / f"ERA5_t2m_{year}.grib"
        dt2m_file = ERA5_PATH / f"ERA5_dt2m_{year}.grib"
        if not t2m_file.exists() or not dt2m_file.exists():
            print(f"  Missing: t2m / dt2m for {year} (rh skipped)")
            continue

        ds_t = xr.open_dataset(t2m_file, engine="cfgrib",
                                chunks={"time": "auto", "latitude": "auto", "longitude": "auto"})
        ds_d = xr.open_dataset(dt2m_file, engine="cfgrib",
                                chunks={"time": "auto", "latitude": "auto", "longitude": "auto"})

        ds_t = ds_t.drop_vars(["valid_time", "step"], errors="ignore")
        ds_d = ds_d.drop_vars(["valid_time", "step"], errors="ignore")
        if "time" in ds_t.dims: ds_t = ds_t.rename({"time": "valid_time"})
        if "time" in ds_d.dims: ds_d = ds_d.rename({"time": "valid_time"})
        ds_t = ds_t.sel(valid_time=slice(f"{year}-01-01", f"{year}-12-31 23:59:59"))
        ds_d = ds_d.sel(valid_time=slice(f"{year}-01-01", f"{year}-12-31 23:59:59"))

        t2m_C = ds_t["t2m"] - 273.15
        d2m_var = "d2m" if "d2m" in ds_d.data_vars else list(ds_d.data_vars)[0]
        d2m_C = ds_d[d2m_var] - 273.15

        e_s = 6.112 * np.exp(17.67 * t2m_C / (t2m_C + 243.5))
        e_d = 6.112 * np.exp(17.67 * d2m_C / (d2m_C + 243.5))
        rh_hourly = 100.0 * (e_d / e_s)
        rh_hourly.name = "rh"
        rh_ds = rh_hourly.to_dataset()

        drop_extra = ["surface", "number", "step", "depthBelowLandLayer"]
        ds_mean = rh_ds.resample(valid_time=rule).mean()
        ds_mean = ds_mean.drop_vars(drop_extra, errors="ignore")
        print(f"  Saving: {target.name}")
        with ProgressBar():
            ds_mean.to_netcdf(target, engine="h5netcdf",
                              encoding=nc_encoding_compressed(ds_mean))

        ds_t.close()
        ds_d.close()


def step2_combine_and_regrid(ttag, ftag, target_lat, target_lon, spatial_res):
    """Merge the per-year files into one file and regrid to the target grid."""
    section(f"Step 2: Combine years and regrid {ERA5_NATIVE_RES}° → {spatial_res}°")

    _regrid = make_era5_regridder(spatial_res, target_lat, target_lon)

    for short in ERA5_VARIABLES:
        if short in WIND_VARS:
            continue

        out = ERA5_PATH / f"ERA5_{short}_{YEARS_TAG}_{ftag}.nc"
        if out.exists():
            print(f"  Already exists: {out.name}")
            continue
        files = [ERA5_PATH / f"ERA5_{short}_{y}_{ttag}.nc" for y in YEARS]
        missing = [f for f in files if not f.exists()]
        if missing:
            print(f"  Missing: {[m.name for m in missing]}")
            continue
        print(f"  Saving: {out.name}")
        ds = xr.open_mfdataset(files, combine="by_coords", chunks="auto", join="exact")
        ds = _regrid(ds)
        with ProgressBar():
            ds.to_netcdf(out, engine="h5netcdf",
                         encoding=nc_encoding_compressed(ds))
        ds.close()

    # u10 + v10 together
    out = ERA5_PATH / f"ERA5_u10_v10_{YEARS_TAG}_{ftag}.nc"
    if not out.exists():
        print(f"  Saving: {out.name}")
        per_var = []
        for v in WIND_VARS:
            files = [ERA5_PATH / f"ERA5_{v}_{y}_{ttag}.nc" for y in YEARS]
            ds = xr.open_mfdataset(files, combine="by_coords", chunks="auto", join="exact")
            per_var.append(ds)

        ds_wind = xr.merge(per_var)
        ds_wind = ds_wind.sortby("valid_time")
        vt = ds_wind.valid_time.to_index()
        ds_wind = ds_wind.isel(valid_time=~vt.duplicated())
        ds_wind = _regrid(ds_wind)
        with ProgressBar():
            ds_wind.to_netcdf(out, engine="h5netcdf",
                              encoding=nc_encoding_compressed(ds_wind))
        ds_wind.close()
    else:
        print(f"  Already exists: {out.name}")

    # ws max+mean (computed from hourly in Step 1)
    for label in ("max", "mean"):
        out = ERA5_PATH / f"ERA5_ws_{label}_{YEARS_TAG}_{ftag}.nc"
        if out.exists():
            print(f"  Already exists: {out.name}")
            continue
        files = [ERA5_PATH / f"ERA5_ws_{y}_{ttag}_{label}.nc" for y in YEARS]
        missing = [f for f in files if not f.exists()]
        if missing:
            print(f"  Missing: {[m.name for m in missing]}")
            continue
        print(f"  Saving: {out.name}")
        ds = xr.open_mfdataset(files, combine="by_coords", chunks="auto", join="exact")
        ds = _regrid(ds)
        with ProgressBar():
            ds.to_netcdf(out, engine="h5netcdf",
                         encoding=nc_encoding_compressed(ds))
        ds.close()

    # wd mean
    out = ERA5_PATH / f"ERA5_wd_mean_{YEARS_TAG}_{ftag}.nc"
    if out.exists():
        print(f"  Already exists: {out.name}")
    else:
        files = [ERA5_PATH / f"ERA5_wd_{y}_{ttag}_mean.nc" for y in YEARS]
        missing = [f for f in files if not f.exists()]
        if missing:
            print(f"  Missing: {[m.name for m in missing]}")
        else:
            print(f"  Saving: {out.name}")
            ds = xr.open_mfdataset(files, combine="by_coords", chunks="auto", join="exact")
            ds = _regrid(ds)
            with ProgressBar():
                ds.to_netcdf(out, engine="h5netcdf",
                             encoding=nc_encoding_compressed(ds))
            ds.close()

    # rh mean
    out = ERA5_PATH / f"ERA5_rh_mean_{YEARS_TAG}_{ftag}.nc"
    if out.exists():
        print(f"  Already exists: {out.name}")
    else:
        files = [ERA5_PATH / f"ERA5_rh_{y}_{ttag}_mean.nc" for y in YEARS]
        missing = [f for f in files if not f.exists()]
        if missing:
            print(f"  Missing: {[m.name for m in missing]}")
        else:
            print(f"  Saving: {out.name}")
            ds = xr.open_mfdataset(files, combine="by_coords", chunks="auto", join="exact")
            ds = _regrid(ds)
            with ProgressBar():
                ds.to_netcdf(out, engine="h5netcdf",
                             encoding=nc_encoding_compressed(ds))
            ds.close()


def step3_derived_features(ftag):
    """Build derived feature files: ws (max+mean combined), rv, wd, rh."""
    section("Step 3: Derived features (ws, rv, wd, rh)")

    def _open(name):
        return xr.open_dataset(ERA5_PATH / f"ERA5_{name}_{YEARS_TAG}_{ftag}.nc",
                               chunks="auto")

    def _save(ds, name):
        target = ERA5_PATH / f"DERIVED_{name}_{YEARS_TAG}_{ftag}.nc"
        if target.exists():
            print(f"  Already exists: {target.name}")
            return
        print(f"  Saving: {target.name}")
        with ProgressBar():
            ds.to_netcdf(target, engine="h5netcdf",
                         encoding=nc_encoding_compressed(ds))

    # ws (combine max + mean into single file)
    target = ERA5_PATH / f"DERIVED_ws_{YEARS_TAG}_{ftag}.nc"
    if not target.exists():
        ws_max_ds  = xr.open_dataset(ERA5_PATH / f"ERA5_ws_max_{YEARS_TAG}_{ftag}.nc",  chunks="auto")
        ws_mean_ds = xr.open_dataset(ERA5_PATH / f"ERA5_ws_mean_{YEARS_TAG}_{ftag}.nc", chunks="auto")
        ws_max_ds  = ws_max_ds.rename({"ws": "ws_max"})
        ws_mean_ds = ws_mean_ds.rename({"ws": "ws_mean"})
        ws_max_ds, ws_mean_ds = xr.align(ws_max_ds, ws_mean_ds, join="inner")
        ws_ds = xr.merge([ws_max_ds, ws_mean_ds], compat="override")
        ws_ds["ws_max"].attrs  = {"long_name": "Max wind speed (from hourly)",  "units": "m s**-1"}
        ws_ds["ws_mean"].attrs = {"long_name": "Mean wind speed (from hourly)", "units": "m s**-1"}
        _save(ws_ds, "ws")
        ws_max_ds.close(); ws_mean_ds.close(); ws_ds.close()
    else:
        print(f"  Already exists: {target.name}")

    # Relative vorticity from u10/v10.
    # ERA5 lat is descending (90 → -90) so .differentiate("latitude") returns
    # -∂/∂φ; the extra minus sign restores +∂u/∂φ.
    wind_ds = _open("u10_v10")
    R = 6_371_000.0
    u = wind_ds["u10"].chunk({"latitude": -1, "longitude": -1})
    v = wind_ds["v10"].chunk({"latitude": -1, "longitude": -1})

    lat_rad = np.deg2rad(wind_ds.latitude)
    coslat = np.cos(lat_rad)
    dvdlam = v.differentiate("longitude") * (np.pi / 180.0)
    dudphi = -u.differentiate("latitude") * (np.pi / 180.0)
    vorticity = (dvdlam / (R * coslat)) - (dudphi / R) + (u * np.tan(lat_rad) / R)
    _save(xr.Dataset({"relative_vorticity": vorticity}), "rv")
    del vorticity, wind_ds

    # wd
    target = ERA5_PATH / f"DERIVED_wd_{YEARS_TAG}_{ftag}.nc"
    if not target.exists():
        wd_ds = xr.open_dataset(ERA5_PATH / f"ERA5_wd_mean_{YEARS_TAG}_{ftag}.nc", chunks="auto")
        wd_ds["wd"].attrs = {
            "long_name": "Mean wind direction (vector-mean of hourly, meteorological convention)",
            "units": "degrees",
        }
        _save(wd_ds, "wd")
        wd_ds.close()
    else:
        print(f"  Already exists: {target.name}")

    # rh
    target = ERA5_PATH / f"DERIVED_rh_{YEARS_TAG}_{ftag}.nc"
    if not target.exists():
        rh_ds = xr.open_dataset(ERA5_PATH / f"ERA5_rh_mean_{YEARS_TAG}_{ftag}.nc", chunks="auto")
        rh_ds["rh"].attrs = {"long_name": "Mean relative humidity at 2m (from hourly)",
                              "units": "%"}
        _save(rh_ds, "rh")
        rh_ds.close()
    else:
        print(f"  Already exists: {target.name}")


def step4_combine_features(ftag, steps_per_day, resolution, spatial_res,
                            target_lat, target_lon):
    """Merge all features, apply land-sea mask, add static fields, rolling stats, lags."""
    section("Step 4: Combine all features into one dataset")

    def _open_raw(short):
        return xr.open_dataset(ERA5_PATH / f"ERA5_{short}_{YEARS_TAG}_{ftag}.nc",
                               chunks="auto")

    def _open_derived(name):
        return xr.open_dataset(ERA5_PATH / f"DERIVED_{name}_{YEARS_TAG}_{ftag}.nc",
                               chunks="auto")

    tp     = _open_raw("tp")
    t2m    = _open_raw("t2m")
    t2m["t2m"] = t2m["t2m"] - 273.15
    lai_lv = _open_raw("lai_lv")
    lai_hv = _open_raw("lai_hv")

    ws    = _open_derived("ws")
    rv    = _open_derived("rv")
    wd    = _open_derived("wd")
    rh    = _open_derived("rh")

    tp, t2m, ws, rv, wd, rh, lai_lv, lai_hv = xr.align(
        tp, t2m, ws, rv, wd, rh, lai_lv, lai_hv, join="inner",
    )

    ds_features = xr.merge(
        [tp, t2m, ws, rv, wd, rh, lai_lv, lai_hv],
        compat="override",
    )

    if "relative_vorticity" in ds_features:
        ds_features = ds_features.rename({"relative_vorticity": "rv"})

    nlat = ds_features.sizes["latitude"]
    nlon = ds_features.sizes["longitude"]
    ds_features = ds_features.chunk({
        "valid_time": 720, "latitude": nlat, "longitude": nlon
    })

    # Land-sea mask
    _regrid = make_era5_regridder(spatial_res, target_lat, target_lon)
    land_sea_ds = xr.open_dataset(LAND_MASK_FILE, engine="cfgrib")
    land_sea_ds = land_sea_ds.drop_vars(
        ["step", "time", "valid_time", "number", "surface"], errors="ignore"
    ).squeeze()

    lsm_regrid = _regrid(land_sea_ds)
    lsm_mask_2d = (lsm_regrid > LSM_THRESHOLD).squeeze()
    ds_features = ds_features.where(lsm_mask_2d.lsm)
    ds_features = ds_features.transpose("valid_time", "latitude", "longitude")

    # Static fields (orography, vegetation cover)
    static_vars = {
        "sdor": ("sdor",       "Std dev of orography",                    "m"),
        "slor": ("slor",       "Slope of sub-gridscale orography",        ""),
        "anor": ("anor",       "Anisotropy of sub-gridscale orography",   ""),
        "isor": ("isor",       "Angle of sub-gridscale orography",        "radians"),
        "cvl":  ("cvl",        "Low vegetation cover",                    "0-1"),
        "cvh":  ("cvh",        "High vegetation cover",                   "0-1"),
    }

    static_datasets = []
    for short, (out_name, long_name, units) in static_vars.items():
        static_file = ERA5_PATH / f"ERA5_{short}_static.grib"
        if not static_file.exists():
            print(f"  Missing: {static_file.name} (skipped)")
            continue

        ds_static = xr.open_dataset(static_file, engine="cfgrib")
        ds_static = ds_static.drop_vars(
            ["step", "time", "valid_time", "number", "surface", "depthBelowLandLayer"],
            errors="ignore",
        ).squeeze()
        ds_static = _regrid(ds_static)

        var_name = list(ds_static.data_vars)[0]
        if var_name != out_name:
            ds_static = ds_static.rename({var_name: out_name})

        ds_static[out_name].attrs = {"long_name": long_name, "units": units}
        ds_static = ds_static.where(lsm_mask_2d.lsm)
        static_datasets.append(ds_static)

    if static_datasets:
        static_merged = xr.merge(static_datasets, compat="override")
        static_merged = static_merged.sel(
            latitude=ds_features.latitude,
            longitude=ds_features.longitude,
            method="nearest",
        )
        ds_features = xr.merge([ds_features, static_merged], compat="override")

    # Wind-terrain alignment.
    # +1 = wind parallel to valley axis (channeling), -1 = perpendicular.
    # Factor of 2 because isor has 180° symmetry while wd is full 360°.
    if "wd" in ds_features and "isor" in ds_features and "anor" in ds_features:
        wd_rad = np.deg2rad(ds_features["wd"])
        angle_diff = 2.0 * (wd_rad - ds_features["isor"])
        ds_features["wind_terrain_align"] = (
            ds_features["anor"] * np.cos(angle_diff)
        ).transpose("valid_time", "latitude", "longitude")
        ds_features["wind_terrain_align"].attrs = {
            "long_name": "Wind-terrain alignment (anor * cos(2*(wd - isor)))",
            "units": "1",
        }
    else:
        missing = [v for v in ("wd", "isor", "anor") if v not in ds_features]
        print(f"  WARNING: wind-terrain alignment skipped, missing {missing}")

    # Rolling means/sums
    for d in [30, 90, 180]:
        w = d * steps_per_day
        ds_features[f"tp_roll_{d}d"]    = ds_features["tp"].rolling(valid_time=w, min_periods=1).sum(skipna=True)
        ds_features[f"t2m_roll_{d}d"]   = ds_features["t2m"].rolling(valid_time=w, min_periods=1).mean(skipna=True)

    # Sub-daily lags
    subdaily_cfg = {
        "3h": {"labels": ["3h", "6h", "12h"], "steps": [1, 2, 4]},
        "6h": {"labels": ["6h", "12h"],       "steps": [1, 2]},
        "daily": {"labels": [],                "steps": []},
    }
    sd = subdaily_cfg.get(resolution, {"labels": [], "steps": []})
    if sd["labels"]:
        for var in ["ws_max", "ws_mean", "rv"]:
            if var not in ds_features:
                continue
            for label, shift in zip(sd["labels"], sd["steps"]):
                ds_features[f"{var}_lag_{label}"] = ds_features[var].shift(valid_time=shift)

    # Daily lags
    for var in ["tp", "t2m", "ws_mean", "rh"]:
        if var not in ds_features:
            continue
        for lag_d in range(1, 7):
            ds_features[f"{var}_lag_{lag_d}d"] = ds_features[var].shift(
                valid_time=lag_d * steps_per_day
            )

    # log1p precipitation
    tp_vars = [v for v in ds_features.data_vars if "tp" in v]
    for v in tp_vars:
        ds_features[f"log_{v}"] = xr.apply_ufunc(
            np.log1p, ds_features[v],
            dask="parallelized", output_dtypes=[ds_features[v].dtype],
        )
    ds_features = ds_features.drop_vars(tp_vars)
    ds_features = ds_features.reset_coords(drop=True)

    # Drop spin-up
    ds_features = ds_features.isel(valid_time=slice(180 * steps_per_day, None))
    ds_features = ds_features.transpose("valid_time", "latitude", "longitude", ...)

    time_chunk = min(365 * steps_per_day, ds_features.sizes["valid_time"])
    ds_features = ds_features.chunk({
        "valid_time": time_chunk, "latitude": nlat, "longitude": nlon
    })

    static_var_names = {"sdor", "slor", "anor", "isor", "cvl", "cvh"}
    encoding = {}
    for v in ds_features.data_vars:
        if v in static_var_names:
            encoding[v] = {"dtype": "float32", "zlib": True, "complevel": 4}
        else:
            encoding[v] = {
                "dtype": "float32", "zlib": True, "complevel": 4,
                "chunksizes": (time_chunk, nlat, nlon),
            }

    out_file = OUT_PATH / f"era5_features_{YEARS_TAG.replace('-', '_')}_{ftag}.nc"
    print(f"  Saving: {out_file.name} ({len(ds_features.data_vars)} variables)")
    with ProgressBar():
        ds_features.to_netcdf(out_file, engine="h5netcdf", encoding=encoding)


def run_preprocessing(resolution=RESOLUTION, spatial_res=SPATIAL_RES):
    cfg  = RESOLUTION_CONFIG[resolution]
    rule = cfg["rule"]
    spd  = cfg["steps_per_day"]
    ttag = cfg["tag"]
    ftag = file_tag(resolution, spatial_res)
    target_lat, target_lon = make_target_grid(spatial_res)

    header("ERA5 preprocessing", {
        "Resolution": f"{resolution}, {spatial_res}°",
        "Grid": f"{len(target_lat)} × {len(target_lon)}",
        "Years": f"{min(YEARS)}-{max(YEARS)}",
    })

    step1_aggregate(rule, ttag)
    step2_combine_and_regrid(ttag, ftag, target_lat, target_lon, spatial_res)
    step3_derived_features(ftag)
    step4_combine_features(ftag, spd, resolution, spatial_res, target_lat, target_lon)
    done(OUT_PATH / f"era5_features_{YEARS_TAG.replace('-', '_')}_{ftag}.nc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ERA5 data.")
    parser.add_argument("-r", "--resolution", choices=list(RESOLUTION_CONFIG.keys()),
                        default=RESOLUTION, help=f"Temporal resolution (default: {RESOLUTION})")
    parser.add_argument("-s", "--spatial", type=float, choices=SUPPORTED_SPATIAL_RES,
                        default=SPATIAL_RES, help=f"Spatial resolution in degrees (default: {SPATIAL_RES})")
    args = parser.parse_args()
    run_preprocessing(args.resolution, args.spatial)
