"""
Build gridded ACLED conflict features at a configurable temporal/spatial
resolution from the raw events downloaded by scripts/download/download_acled.py
(data/acled/acled_raw_events.csv).

Events are snapped to the nearest cell of the model grid (read from the native
ERA5 / CAMS files via sds.grid.make_target_grid), counted per day, and turned
into rolling window sums (30/180/365 days) with log1p. For sub-daily
resolutions, daily counts are repeated at each sub-daily timestep — conflict is
treated as a daily-scale signal.

Output variables (all log1p):
    acled_total_count_log1p
    acled_total_count_30d_sum_log1p
    acled_total_count_180d_sum_log1p
    acled_total_count_365d_sum_log1p

Usage:
    python preprocess_acled.py                    # daily, 0.25°
    python preprocess_acled.py -r 6h -s 0.75
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from sds.config import (
    SUPPORTED_SPATIAL_RES, RESOLUTION, SPATIAL_RES, RESOLUTION_CONFIG,
    ACLED_PATH, ACLED_RAW_EVENTS, YEARS,
)
from sds.paths import file_tag, ACLED_YEARS_TAG
from sds.grid import make_target_grid
from sds.log_utils import header, section, done

# config.YEARS plus one earlier year (same span as download_acled.py)
START_DATE = f"{min(YEARS) - 1}-01-01"
END_DATE = f"{max(YEARS)}-12-31"


def load_raw_events():
    if not ACLED_RAW_EVENTS.exists():
        raise FileNotFoundError(
            f"{ACLED_RAW_EVENTS} not found. Run scripts/download/download_acled.py first.")
    df = pd.read_csv(ACLED_RAW_EVENTS)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def step2_grid_and_aggregate(df, ttag, ftag, spatial_res, steps_per_day,
                              target_lat, target_lon):
    section("Grid events and build rolling counts")

    output_file = ACLED_PATH / f"ACLED_{ACLED_YEARS_TAG}_{ftag}.nc"
    if output_file.exists():
        print(f"  Already exists: {output_file.name}")
        return

    lat_grid = np.sort(target_lat)
    lon_grid = np.sort(target_lon)

    def snap(coord, grid):
        idx = np.searchsorted(grid, coord, side="left")
        idx = np.clip(idx, 1, len(grid) - 1)
        left = grid[idx - 1]
        right = grid[idx]
        return np.where(np.abs(coord - left) <= np.abs(coord - right), left, right)

    # keep events inside the grid cells (outer cells extend half a cell beyond
    # the outermost centres); events further out would otherwise be snapped
    # onto the edge cells
    half = spatial_res / 2
    inside = (
        (df["latitude"] >= lat_grid.min() - half) & (df["latitude"] <= lat_grid.max() + half) &
        (df["longitude"] >= lon_grid.min() - half) & (df["longitude"] <= lon_grid.max() + half)
    )
    print(f"  Events: {len(df):,} ({int((~inside).sum()):,} outside the grid, dropped)")
    df = df[inside].copy()
    df["lat_snap"] = snap(df["latitude"].values, lat_grid)
    df["lon_snap"] = snap(df["longitude"].values, lon_grid)

    counts = (
        df.groupby(["event_date", "lat_snap", "lon_snap", "event_type"])
        .size()
        .rename("count")
        .reset_index()
    )

    all_dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    shape = (len(all_dates), len(lat_grid), len(lon_grid))
    battles = np.zeros(shape, dtype="float32")
    explosions = np.zeros(shape, dtype="float32")

    date_idx = {d: i for i, d in enumerate(all_dates)}
    lat_idx = {round(float(l), 4): i for i, l in enumerate(lat_grid)}
    lon_idx = {round(float(l), 4): i for i, l in enumerate(lon_grid)}

    for row in counts.itertuples(index=False):
        t = date_idx.get(pd.Timestamp(row.event_date))
        la = lat_idx.get(round(float(row.lat_snap), 4))
        lo = lon_idx.get(round(float(row.lon_snap), 4))
        if t is not None and la is not None and lo is not None:
            if row.event_type == "Battles":
                battles[t, la, lo] += row.count
            else:
                explosions[t, la, lo] += row.count

    total = battles + explosions
    print(f"  Gridded: {int(battles.sum()):,} battles, {int(explosions.sum()):,} explosions")

    AGG_WINDOWS = [30, 180, 365]

    acled_vars = {"acled_total_count_log1p": np.log1p(total).astype("float32")}
    for w in AGG_WINDOWS:
        rolled = np.zeros_like(total, dtype="float32")
        cs = np.cumsum(total, axis=0)
        rolled[w:] = cs[w:] - cs[:-w]
        rolled[:w] = cs[:w]
        key = f"acled_total_count_{w}d_sum_log1p"
        acled_vars[key] = np.log1p(rolled).astype("float32")

    if steps_per_day == 1:
        time_index = all_dates
    else:
        freq = {c["tag"]: c["rule"] for c in RESOLUTION_CONFIG.values()}[ttag]
        time_index = pd.date_range(start=START_DATE, end=END_DATE, freq=freq)
        time_index = time_index[time_index < pd.Timestamp(END_DATE) + pd.Timedelta(days=1)]
        for k in acled_vars:
            acled_vars[k] = np.repeat(acled_vars[k], steps_per_day, axis=0)[:len(time_index)]

    # Flip lat to descending (N→S) to match ERA5/CAMS convention
    lat_out = lat_grid[::-1]
    for k in acled_vars:
        acled_vars[k] = acled_vars[k][:, ::-1, :]

    data_vars = {k: (["time", "latitude", "longitude"], v) for k, v in acled_vars.items()}
    ds = xr.Dataset(
        data_vars,
        coords={"time": time_index, "latitude": lat_out, "longitude": lon_grid},
    )

    ds["acled_total_count_log1p"].attrs["units"] = "log1p(events per day)"
    for w in AGG_WINDOWS:
        vname = f"acled_total_count_{w}d_sum_log1p"
        ds[vname].attrs["units"] = f"log1p(events in {w}-day window)"
        ds[vname].attrs["aggregation"] = f"{w}-day rolling sum, then log1p"
    for vname in ds.data_vars:
        if steps_per_day > 1:
            ds[vname].attrs["temporal_note"] = "Daily value repeated at each sub-daily timestep"

    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ds.data_vars}
    print(f"  Saving: {output_file.name}")
    ds.to_netcdf(output_file, engine="h5netcdf", encoding=encoding)


def run_acled_preprocessing(resolution=RESOLUTION, spatial_res=SPATIAL_RES):
    cfg  = RESOLUTION_CONFIG[resolution]
    ttag = cfg["tag"]
    spd  = cfg["steps_per_day"]
    ftag = file_tag(resolution, spatial_res)
    target_lat, target_lon = make_target_grid(spatial_res)

    header("ACLED preprocessing", {
        "Resolution": f"{resolution}, {spatial_res}°",
        "Grid": f"{len(target_lat)} × {len(target_lon)}",
        "Period": f"{START_DATE} to {END_DATE}",
    })

    df = load_raw_events()

    step2_grid_and_aggregate(df, ttag, ftag, spatial_res, spd, target_lat, target_lon)
    done(ACLED_PATH / f"ACLED_{ACLED_YEARS_TAG}_{ftag}.nc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ACLED conflict data.")
    parser.add_argument("-r", "--resolution", choices=list(RESOLUTION_CONFIG.keys()),
                        default=RESOLUTION, help=f"Temporal resolution (default: {RESOLUTION})")
    parser.add_argument("-s", "--spatial", type=float, choices=SUPPORTED_SPATIAL_RES,
                        default=SPATIAL_RES, help=f"Spatial resolution in degrees (default: {SPATIAL_RES})")
    args = parser.parse_args()
    run_acled_preprocessing(args.resolution, args.spatial)
