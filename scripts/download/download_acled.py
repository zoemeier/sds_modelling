"""
Fetch ACLED conflict events (2014-2024) and aggregate them into daily
counts on the 0.25° ERA5 grid.

Event types: Battles, Explosions/Remote violence
Spatial domain: 0-50N, 18W-85E (snapped to 0.25° centres)

Output:
    data/acled/ACLED_2014-2024_daily.nc
"""

import os
import time
import json
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import xarray as xr


FINAL_BBOX = [-18, 0, 85, 50]                    # min_lon, min_lat, max_lon, max_lat
EVENT_TYPES = ["Battles", "Explosions/Remote violence"]
START_DATE = "2014-01-01"
END_DATE = "2024-12-31"
MAX_NUM_NEIGHBORS_AGGREGATION = 0


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data"
ACLED_PATH = DATA_PATH / "acled"
os.makedirs(ACLED_PATH, exist_ok=True)

OUT_FILE_NAME = "ACLED_2014-2024_daily.nc"

PW_PATH = PROJECT_ROOT / "passwords.json"
with open(PW_PATH) as f:
    pws = json.load(f)


# Pad bbox so events near the edge still snap to a valid 0.25° centre
min_lon, min_lat, max_lon, max_lat = FINAL_BBOX
padding = (MAX_NUM_NEIGHBORS_AGGREGATION + 1) * 0.25 - 0.125
BBOX = [min_lon - padding, min_lat - padding, max_lon + padding, max_lat + padding]
min_lon, min_lat, max_lon, max_lat = BBOX


# OAuth token
url = "https://acleddata.com/oauth/token"
data = {
    "username":   pws["acled"]["username"],
    "password":   pws["acled"]["pw"],
    "grant_type": "password",
    "client_id":  "acled",
}
headers = {"Content-Type": "application/x-www-form-urlencoded"}

response = requests.post(url, headers=headers, data=data)
if response.status_code == 200:
    token_data = response.json()
    access_token = token_data['access_token']
    refresh_token = token_data['refresh_token']
    print("Access Token:", access_token)
    print("Refresh Token:", refresh_token)
else:
    print("Error:", response.status_code, response.text)


def fetch_acled_events_filtered(access_token, bbox, start_date=None, end_date=None,
                                 event_types=None, fields=None,
                                 limit=5000, pause=1, max_pages=1000):
    """Page through the ACLED read API and return a deduplicated DataFrame."""
    url = "https://acleddata.com/api/acled/read"
    page = 1
    all_events = []

    while page <= max_pages:
        params = {
            "_format": "json",
            "limit":   limit,
            "page":    page,
            "bbox":    ",".join(map(str, bbox)),
        }
        if fields:
            params["fields"] = fields
        if start_date and end_date:
            params["event_date"] = f"{start_date}|{end_date}"
            params["event_date_where"] = "BETWEEN"
        if event_types:
            params["event_type"] = "|".join(event_types)

        request_headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(url, headers=request_headers, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"ACLED error {response.status_code}: {response.text}")

        data = response.json().get("data", [])
        if not data:
            print(f"No more data returned at page {page}. Stopping.")
            break

        all_events.extend(data)
        print(f"Page {page}: fetched {len(data)} events, total so far: {len(all_events)}")

        if len(data) < limit:
            break
        page += 1
        time.sleep(pause)

    df = pd.DataFrame(all_events)

    id_col = next((c for c in df.columns if "event_id" in c), None)
    if id_col:
        before = len(df)
        df = df.drop_duplicates(subset=id_col)
        print(f"Deduplicated: {before} → {len(df)} events")

    return df


df = fetch_acled_events_filtered(
    access_token=access_token,
    bbox=BBOX,
    start_date=START_DATE,
    end_date=END_DATE,
    event_types=EVENT_TYPES,
    fields="event_id_cnty|event_date|event_type|latitude|longitude",
)
print(f"Total ACLED events fetched: {len(df)}")


df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')

df = df[
    (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon) &
    (df['latitude']  >= min_lat) & (df['latitude']  <= max_lat)
]


print("Aggregating daily event counts on 0.25° grid")
lon_grid = np.arange(min_lon + 0.125, max_lon, 0.25)
lat_grid = np.arange(min_lat + 0.125, max_lat, 0.25)


def snap_to_grid(coord, grid_coords):
    idx = np.abs(grid_coords - coord).argmin()
    return grid_coords[idx]


df['lat_snap'] = df['latitude'].apply(lambda x: snap_to_grid(x, lat_grid))
df['lon_snap'] = df['longitude'].apply(lambda x: snap_to_grid(x, lon_grid))

counts = (
    df.groupby(['event_date', 'lat_snap', 'lon_snap', 'event_type'])
      .size()
      .rename("count")
      .reset_index()
)


# Build xarray Dataset by filling the sparse non-zero cells directly
all_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
shape = (len(all_dates), len(lat_grid), len(lon_grid))
battles = np.zeros(shape, dtype='int16')
explosions = np.zeros(shape, dtype='int16')

date_idx = {d: i for i, d in enumerate(all_dates)}
lat_idx = {round(l, 4): i for i, l in enumerate(lat_grid)}
lon_idx = {round(l, 4): i for i, l in enumerate(lon_grid)}

for row in counts.itertuples(index=False):
    t = date_idx.get(pd.Timestamp(row.event_date))
    la = lat_idx.get(round(row.lat_snap, 4))
    lo = lon_idx.get(round(row.lon_snap, 4))
    if t is not None and la is not None and lo is not None:
        if row.event_type == "Battles":
            battles[t, la, lo] += row.count
        else:
            explosions[t, la, lo] += row.count

total = (battles.astype('int32') + explosions.astype('int32')).astype('int16')

ds = xr.Dataset(
    {
        "Battles_count":                    (["time", "lat", "lon"], battles),
        "Explosions_Remote_violence_count": (["time", "lat", "lon"], explosions),
        "total_count":                      (["time", "lat", "lon"], total),
    },
    coords={
        "time": all_dates,
        "lat":  lat_grid,
        "lon":  lon_grid,
    },
)

ds.to_netcdf(ACLED_PATH / OUT_FILE_NAME, engine="netcdf4", format="NETCDF4")
print(f"Saved to {OUT_FILE_NAME}")
