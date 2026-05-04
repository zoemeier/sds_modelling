"""
Build gridded ACLED conflict features at a configurable temporal/spatial
resolution.

Pipeline:
  Step 1: download raw events from the ACLED API (cached as CSV; resolution
          independent).
  Step 2: snap events to the target grid, compute daily counts and rolling
          window sums (30/180/365 days), apply log1p, and broadcast to the
          target temporal resolution if sub-daily.

For sub-daily resolutions, daily counts are repeated at each sub-daily
timestep — conflict is treated as a daily-scale signal.

Output variables (all log1p):
    acled_total_count_log1p
    acled_total_count_30d_sum_log1p
    acled_total_count_180d_sum_log1p
    acled_total_count_365d_sum_log1p

Usage:
    python preprocess_acled.py                    # daily, 0.25°
    python preprocess_acled.py -r 6h -s 0.75
"""

import os
import json
import argparse
import time as _time
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import xarray as xr

from grid_config import RESOLUTION_CONFIG, make_target_grid, file_tag


EVENT_TYPES = ["Battles", "Explosions/Remote violence"]
START_DATE = "2014-01-01"
END_DATE = "2024-12-31"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data"
ACLED_PATH = DATA_PATH / "acled"
os.makedirs(ACLED_PATH, exist_ok=True)

PW_PATH = PROJECT_ROOT / "passwords.json"
RAW_CSV = ACLED_PATH / "acled_raw_events.csv"


def get_access_token():
    with open(PW_PATH) as f:
        pws = json.load(f)
    response = requests.post(
        "https://acleddata.com/oauth/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "username": pws["acled"]["username"],
            "password": pws["acled"]["pw"],
            "grant_type": "password",
            "client_id": "acled",
        },
    )
    if response.status_code != 200:
        raise RuntimeError(f"ACLED auth failed: {response.status_code} {response.text}")
    print("  Authenticated with ACLED API")
    return response.json()["access_token"]


def fetch_acled_events(access_token, bbox, start_date, end_date,
                       event_types, limit=5000, pause=1, max_pages=1000):
    url = "https://acleddata.com/api/acled/read"
    page = 1
    all_events = []

    while page <= max_pages:
        params = {
            "_format": "json",
            "limit":   limit,
            "page":    page,
            "bbox":    ",".join(map(str, bbox)),
            "fields":  "event_id_cnty|event_date|event_type|latitude|longitude",
        }
        if start_date and end_date:
            params["event_date"] = f"{start_date}|{end_date}"
            params["event_date_where"] = "BETWEEN"
        if event_types:
            params["event_type"] = "|".join(event_types)

        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"ACLED error {response.status_code}: {response.text}")

        data = response.json().get("data", [])
        if not data:
            print(f"  No more data at page {page}. Stopping.")
            break

        all_events.extend(data)
        print(f"  Page {page}: {len(data)} events, total: {len(all_events)}")

        if len(data) < limit:
            break
        page += 1
        _time.sleep(pause)

    df = pd.DataFrame(all_events)
    id_col = next((c for c in df.columns if "event_id" in c), None)
    if id_col:
        before = len(df)
        df = df.drop_duplicates(subset=id_col)
        print(f"  Deduplicated: {before} → {len(df)} events")
    return df


def step1_download():
    print("=" * 70)
    print("STEP 1: Download ACLED events")
    print("=" * 70)

    if RAW_CSV.exists():
        print(f"  Already exists: {RAW_CSV.name}")
        return pd.read_csv(RAW_CSV)

    bbox = [-20, -2, 87, 52]
    token = get_access_token()
    df = fetch_acled_events(
        access_token=token,
        bbox=bbox,
        start_date=START_DATE,
        end_date=END_DATE,
        event_types=EVENT_TYPES,
    )
    print(f"  Total events fetched: {len(df)}")

    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["event_date"] = pd.to_datetime(df["event_date"])

    df.to_csv(RAW_CSV, index=False)
    print(f"  Cached → {RAW_CSV.name}")
    return df


def step2_grid_and_aggregate(df, ttag, ftag, spatial_res, steps_per_day,
                              target_lat, target_lon):
    print("\n" + "=" * 70)
    print(f"STEP 2: Grid + aggregate → {ftag}")
    print("=" * 70)

    output_file = ACLED_PATH / f"ACLED_2014-2024_{ftag}.nc"
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

    df = df.copy()
    df["lat_snap"] = snap(df["latitude"].values, lat_grid)
    df["lon_snap"] = snap(df["longitude"].values, lon_grid)
    df = df[
        (df["lat_snap"] >= lat_grid.min()) & (df["lat_snap"] <= lat_grid.max()) &
        (df["lon_snap"] >= lon_grid.min()) & (df["lon_snap"] <= lon_grid.max())
    ]
    print(f"  Events within grid: {len(df):,}")

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
    print(f"  Daily grid built: {shape}")
    print(f"  Total battles:    {int(battles.sum()):,}")
    print(f"  Total explosions: {int(explosions.sum()):,}")

    AGG_WINDOWS = [30, 180, 365]
    print(f"  Computing rolling window sums: {AGG_WINDOWS} days + log1p")

    acled_vars = {"acled_total_count_log1p": np.log1p(total).astype("float32")}
    for w in AGG_WINDOWS:
        rolled = np.zeros_like(total, dtype="float32")
        cs = np.cumsum(total, axis=0)
        rolled[w:] = cs[w:] - cs[:-w]
        rolled[:w] = cs[:w]
        key = f"acled_total_count_{w}d_sum_log1p"
        acled_vars[key] = np.log1p(rolled).astype("float32")
        print(f"    {key}: mean={acled_vars[key].mean():.3f}, "
              f"max={acled_vars[key].max():.2f}")

    if steps_per_day == 1:
        time_index = all_dates
    else:
        freq_map = {4: "6h", 8: "3h"}
        freq = freq_map[steps_per_day]
        time_index = pd.date_range(start=START_DATE, end=END_DATE, freq=freq)
        time_index = time_index[time_index < pd.Timestamp(END_DATE) + pd.Timedelta(days=1)]
        print(f"  Broadcasting to {steps_per_day} sub-daily steps ({freq})")
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
    print(f"\n  Saving {output_file.name} ...")
    ds.to_netcdf(output_file, engine="h5netcdf", encoding=encoding)
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  Done — {output_file.name} ({size_mb:.1f} MB)")


def run_acled_preprocessing(resolution="1D", spatial_res=0.25):
    cfg  = RESOLUTION_CONFIG[resolution]
    ttag = cfg["tag"]
    spd  = cfg["steps_per_day"]
    ftag = file_tag(ttag, spatial_res)
    target_lat, target_lon = make_target_grid(spatial_res)

    print(f"\n{'#' * 70}")
    print(f"  ACLED preprocessing")
    print(f"  Temporal: {resolution} ({ttag})  |  Spatial: {spatial_res}°")
    print(f"  File tag: {ftag}  |  Grid: {len(target_lat)}×{len(target_lon)}")
    print(f"{'#' * 70}\n")

    df = step1_download()
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["event_date"] = pd.to_datetime(df["event_date"])

    step2_grid_and_aggregate(df, ttag, ftag, spatial_res, spd, target_lat, target_lon)
    print(f"\nDone: ACLED_2014-2024_{ftag}.nc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ACLED conflict data.")
    parser.add_argument("-r", "--resolution", choices=list(RESOLUTION_CONFIG.keys()),
                        default="1D", help="Temporal resolution (default: 1D)")
    parser.add_argument("-s", "--spatial", type=float, choices=[0.25, 0.75],
                        default=0.25, help="Spatial resolution in degrees (default: 0.25)")
    args = parser.parse_args()
    run_acled_preprocessing(args.resolution, args.spatial)
