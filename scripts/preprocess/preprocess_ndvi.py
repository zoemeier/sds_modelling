"""
Preprocess MODIS NDVI (MOD13A1.061) at a configurable temporal/spatial
resolution.

Pipeline:
  Step 1: quality-filter raw 500m MODIS data and regrid to the target
          spatial grid.
            - 0.25°: clip to domain, mask by VI quality, coarsen 60x60.
            - 0.75°: coarsen the 0.25° per-year files 3x3 (running Step 1
              at 0.25° first if those files don't exist).
          Output: one file per year at 16-day composite cadence on the
          target grid.
  Step 2: concatenate per-year files and linearly interpolate to the
          target temporal frequency (daily / 6h / 3h), with edge-fill on
          short gaps.

Usage:
    python preprocess_ndvi.py                     # daily, 0.25°
    python preprocess_ndvi.py -r 6h -s 0.75
"""

import os
import gc
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import dask
from dask.diagnostics import ProgressBar

warnings.filterwarnings("ignore", category=RuntimeWarning)

from sds.config import (
    ERA5_NATIVE_RES, CAMS_NATIVE_RES, SUPPORTED_SPATIAL_RES, RESOLUTION,
    SPATIAL_RES, RESOLUTION_CONFIG, YEARS, NDVI_PATH,
)
from sds.paths import file_tag, spatial_tag, YEARS_TAG
from sds.grid import make_target_grid
from sds.log_utils import header, section, done


os.makedirs(NDVI_PATH, exist_ok=True)

EDGE_FILL_LIMIT_DAYS = 32
MAX_INTERP_GAP_DAYS = 48


def step1_filter_and_regrid(spatial_res, target_lat, target_lon, years=YEARS):
    """Quality-filter MODIS 500m NDVI and regrid to the target spatial grid."""
    stag = spatial_tag(spatial_res)

    # 0.75°: coarsen from 0.25° per-year files
    if spatial_res == CAMS_NATIVE_RES:
        stag_025 = spatial_tag(ERA5_NATIVE_RES)
        all_025_exist = all(
            (NDVI_PATH / f"NDVI_{y}_{stag_025}.nc").exists() for y in years
        )
        if not all_025_exist:
            lat_025, lon_025 = make_target_grid(ERA5_NATIVE_RES)
            step1_filter_and_regrid(ERA5_NATIVE_RES, lat_025, lon_025, years=years)

        section(f"Step 1: Coarsen {ERA5_NATIVE_RES}° → {spatial_res}° (3 × 3 mean)")
        for year in years:
            output_file = NDVI_PATH / f"NDVI_{year}_{stag}.nc"
            if output_file.exists():
                print(f"  Already exists: {output_file.name}")
                continue

            src_file = NDVI_PATH / f"NDVI_{year}_{stag_025}.nc"
            if not src_file.exists():
                print(f"  Missing: {src_file.name} (skipped)")
                continue

            ds_025 = xr.open_dataset(src_file)
            lat_dim = "latitude" if "latitude" in ds_025.dims else "lat"
            lon_dim = "longitude" if "longitude" in ds_025.dims else "lon"
            nlat = len(ds_025[lat_dim])
            nlon = len(ds_025[lon_dim])

            # Pad so dimensions divide evenly by 3
            pad_lat = (3 - nlat % 3) % 3
            pad_lon = (3 - nlon % 3) % 3
            if pad_lat > 0 or pad_lon > 0:
                ds_025 = ds_025.pad(
                    {lat_dim: (0, pad_lat), lon_dim: (0, pad_lon)},
                    mode="constant", constant_values=np.nan,
                )

            ds_075 = ds_025.coarsen(
                {lat_dim: 3, lon_dim: 3}, boundary="exact"
            ).mean()

            if lat_dim == "lat":
                ds_075 = ds_075.rename({"lat": "latitude", "lon": "longitude"})

            if (len(ds_075.latitude) != len(target_lat)
                    or len(ds_075.longitude) != len(target_lon)):
                raise ValueError(
                    f"Grid mismatch after coarsening {year}: "
                    f"got {len(ds_075.latitude)}×{len(ds_075.longitude)}, "
                    f"expected {len(target_lat)}×{len(target_lon)}."
                )

            ds_075 = ds_075.assign_coords(latitude=target_lat, longitude=target_lon)
            print(f"  Saving: {output_file.name}")
            ds_075.to_netcdf(output_file, engine="h5netcdf")
            ds_025.close()
        return

    # 0.25°: quality-filter and coarsen 60x60 from native MODIS 500m
    section(f"Step 1: Quality-filter MODIS 500 m and regrid → {spatial_res}°")
    dlat = spatial_res / 2
    dlon = spatial_res / 2
    lat_sorted = np.sort(target_lat)
    LAT_MIN = lat_sorted[0]  - dlat
    LAT_MAX = lat_sorted[-1] + dlat
    LON_MIN = target_lon[0]  - dlon
    LON_MAX = target_lon[-1] + dlon

    for year in years:
        output_file = NDVI_PATH / f"NDVI_{year}_{stag}.nc"
        if output_file.exists():
            print(f"  Already exists: {output_file.name}")
            continue

        input_file = NDVI_PATH / f"{year}-ndvi" / "MOD13A1.061_500m_aid0001.nc"
        if not input_file.exists():
            print(f"  Missing: {input_file.relative_to(NDVI_PATH)} (skipped)")
            continue

        print(f"  Processing {year} ...")
        ds = xr.open_dataset(
            input_file, chunks={"lat": "auto", "lon": "auto", "time": 1}
        )

        ndvi = ds["_500m_16_days_NDVI"].astype("float32")
        qa = ds["_500m_16_days_VI_Quality"].astype("uint16")

        # MODIS lat is descending → use LAT_MAX first in slice
        ndvi = ndvi.sel(lat=slice(LAT_MAX, LAT_MIN), lon=slice(LON_MIN, LON_MAX))
        qa   = qa.sel(lat=slice(LAT_MAX, LAT_MIN), lon=slice(LON_MIN, LON_MAX))

        modland_qa    = qa & 0b11
        vi_usefulness = (qa >> 2) & 0b1111
        land          = (qa >> 11) & 0b111
        snow_ice      = (qa >> 14) & 0b1

        mask = (modland_qa <= 1) & (vi_usefulness <= 5) & (land == 1)
        ndvi_valid = ndvi.where(mask)

        ndvi_mean = ndvi_valid.coarsen(lat=60, lon=60).mean(skipna=True)
        ndvi_std  = ndvi_valid.coarsen(lat=60, lon=60).std(skipna=True)
        snow_frac = snow_ice.coarsen(lat=60, lon=60).mean()

        with ProgressBar():
            ndvi_mean_r, ndvi_std_r, snow_frac_r = dask.compute(
                ndvi_mean, ndvi_std, snow_frac
            )

        ds_out = xr.Dataset({
            "NDVI_mean": ndvi_mean_r.astype("float32"),
            "NDVI_std":  ndvi_std_r.astype("float32"),
            "snow_frac": snow_frac_r.astype("float32"),
        })

        ds_out = ds_out.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        if "lat" in ds_out.dims:
            ds_out = ds_out.rename({"lat": "latitude", "lon": "longitude"})

        ds_out = ds_out.assign_coords(
            latitude=target_lat[:len(ds_out.latitude)],
            longitude=target_lon[:len(ds_out.longitude)],
        )

        if (len(ds_out.latitude) != len(target_lat)
                or len(ds_out.longitude) != len(target_lon)):
            print(f"  WARNING: grid mismatch — got "
                  f"{len(ds_out.latitude)}×{len(ds_out.longitude)}, "
                  f"expected {len(target_lat)}×{len(target_lon)}")

        n_total = np.prod(ds_out["NDVI_mean"].shape)
        n_nan = int(np.isnan(ds_out["NDVI_mean"].values).sum())
        print(f"  Saving: {output_file.name} ({100 * n_nan / n_total:.1f}% NaN cells)")
        ds_out.to_netcdf(output_file, engine="h5netcdf")
        ds.close()
        gc.collect()


def step2_combine_and_interpolate(ttag, ftag, spatial_res, years=YEARS):
    """Concatenate per-year files and interpolate to target temporal freq."""
    import netCDF4 as nc4

    stag = spatial_tag(spatial_res)

    section(f"Step 2: Combine years and interpolate 16-day composites → {ttag}")

    output_file = NDVI_PATH / f"NDVI_{YEARS_TAG}_{ftag}.nc"
    if output_file.exists():
        print(f"  Already exists: {output_file.name}")
        return

    ndvi_files = [NDVI_PATH / f"NDVI_{y}_{stag}.nc" for y in years]
    missing = [f for f in ndvi_files if not f.exists()]
    if missing:
        print(f"  Missing: {[m.name for m in missing]}")
        return

    ds_ndvi = xr.open_mfdataset(
        ndvi_files, combine="by_coords", chunks="auto", engine="h5netcdf"
    )

    _, index = np.unique(ds_ndvi.time.values, return_index=True)
    ds_ndvi = ds_ndvi.isel(time=index)

    rename_map = {}
    if "lat" in ds_ndvi.dims and "latitude" not in ds_ndvi.dims:
        rename_map["lat"] = "latitude"
    if "lon" in ds_ndvi.dims and "longitude" not in ds_ndvi.dims:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds_ndvi = ds_ndvi.rename(rename_map)

    LAT = ds_ndvi["latitude"].values
    LON = ds_ndvi["longitude"].values

    # Convert cftime → numpy datetime64 if needed
    composite_times = ds_ndvi.time.values
    try:
        pd.Timestamp(composite_times[0])
    except (TypeError, ValueError):
        composite_times = np.array([np.datetime64(str(t)[:19]) for t in composite_times])
        ds_ndvi["time"] = composite_times

    freq = {c["tag"]: c["rule"] for c in RESOLUTION_CONFIG.values()}[ttag]
    start_dt = pd.Timestamp(f"{years[0]}-01-01")
    end_dt   = pd.Timestamp(f"{years[-1]}-12-31")
    target_index = pd.date_range(start=start_dt, end=end_dt, freq=freq)
    NT = len(target_index)

    times_f64 = nc4.date2num(
        target_index.to_pydatetime(),
        units="hours since 1900-01-01", calendar="standard",
    )
    ncf = nc4.Dataset(output_file, "w")
    ncf.createDimension("time", NT)
    ncf.createDimension("latitude", len(LAT))
    ncf.createDimension("longitude", len(LON))
    tv = ncf.createVariable("time", "f8", ("time",))
    tv.units = "hours since 1900-01-01"
    tv.calendar = "standard"
    tv[:] = times_f64
    ncf.createVariable("latitude", "f4", ("latitude",))[:] = LAT
    ncf.createVariable("longitude", "f4", ("longitude",))[:] = LON

    nc_vars = {}
    for vname in ["NDVI_mean", "NDVI_std", "snow_frac"]:
        v = ncf.createVariable(
            vname, "f4", ("time", "latitude", "longitude"),
            zlib=True, complevel=4,
            chunksizes=(min(365, NT), len(LAT), len(LON)),
            fill_value=np.float32(np.nan),
        )
        nc_vars[vname] = v

    total_nan = {v: 0 for v in nc_vars}
    total_cells = 0

    for yi, year in enumerate(years):
        yr_start = pd.Timestamp(f"{year}-01-01")
        yr_end   = pd.Timestamp(f"{year}-12-31")
        pad = pd.Timedelta(days=MAX_INTERP_GAP_DAYS + 16)
        sel_start = yr_start - pad
        sel_end   = yr_end + pad

        ds_chunk = ds_ndvi.sel(time=slice(sel_start, sel_end)).compute()
        if len(ds_chunk.time) == 0:
            continue

        yr_mask = (target_index >= yr_start) & (target_index <= yr_end)
        yr_target = target_index[yr_mask]
        if len(yr_target) == 0:
            continue

        wide_mask = (target_index >= sel_start) & (target_index <= sel_end)
        wide_target = target_index[wide_mask]

        ds_reindexed = ds_chunk.reindex(time=wide_target)
        ds_interp = ds_reindexed.interpolate_na(
            dim="time", method="linear",
            max_gap=np.timedelta64(MAX_INTERP_GAP_DAYS, "D"),
        )
        ds_filled = ds_interp.ffill(dim="time", limit=EDGE_FILL_LIMIT_DAYS)
        ds_filled = ds_filled.bfill(dim="time", limit=EDGE_FILL_LIMIT_DAYS)
        ds_yr = ds_filled.sel(time=yr_target)

        i0 = int(np.searchsorted(target_index, yr_target[0]))
        n_yr = len(yr_target)
        for vname, ncvar in nc_vars.items():
            arr = ds_yr[vname].values.astype("float32")
            ncvar[i0:i0 + n_yr] = arr
            total_nan[vname] += int(np.isnan(arr).sum())
        total_cells += n_yr * len(LAT) * len(LON)

        print(f"  {year} written ({yi + 1}/{len(years)})")

    ncf.close()
    ds_ndvi.close()

    print("  NaN after interpolation: " + ", ".join(
        f"{v} {100 * total_nan[v] / max(total_cells, 1):.2f}%" for v in nc_vars))


def run_ndvi_preprocessing(resolution=RESOLUTION, spatial_res=SPATIAL_RES):
    cfg  = RESOLUTION_CONFIG[resolution]
    ttag = cfg["tag"]
    ftag = file_tag(resolution, spatial_res)
    target_lat, target_lon = make_target_grid(spatial_res)

    header("NDVI preprocessing", {
        "Resolution": f"{resolution}, {spatial_res}°",
        "Grid": f"{len(target_lat)} × {len(target_lon)}",
        "Years": f"{min(YEARS)}-{max(YEARS)}",
    })

    step1_filter_and_regrid(spatial_res, target_lat, target_lon)
    step2_combine_and_interpolate(ttag, ftag, spatial_res)
    done(NDVI_PATH / f"NDVI_{YEARS_TAG}_{ftag}.nc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MODIS NDVI data.")
    parser.add_argument("-r", "--resolution", choices=list(RESOLUTION_CONFIG.keys()),
                        default=RESOLUTION, help=f"Temporal resolution (default: {RESOLUTION})")
    parser.add_argument("-s", "--spatial", type=float, choices=SUPPORTED_SPATIAL_RES,
                        default=SPATIAL_RES, help=f"Spatial resolution in degrees (default: {SPATIAL_RES})")
    args = parser.parse_args()
    run_ndvi_preprocessing(args.resolution, args.spatial)
