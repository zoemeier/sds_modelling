"""
Shared grid and resolution configuration for the SDS preprocessing pipeline.

The 0.25° grid is read from native ERA5 files; the 0.75° grid from native
CAMS files. This guarantees alignment with the source data. Grids are
clipped to the spatial domain and cached after first read.

Usage:
    from grid_config import make_target_grid, RESOLUTION_CONFIG
"""

from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH    = PROJECT_ROOT / "data"
ERA5_PATH    = DATA_PATH / "era5"
CAMS_PATH    = DATA_PATH / "cams"


SPATIAL_DOMAIN = {
    "north":  50.0,
    "south":   0.0,
    "west":  -18.0,
    "east":   85.0,
}

RESOLUTION_CONFIG = {
    "1D": {"rule": "1D", "steps_per_day": 1, "tag": "daily"},
    "6h": {"rule": "6h", "steps_per_day": 4, "tag": "6hourly"},
    "3h": {"rule": "3h", "steps_per_day": 8, "tag": "3hourly"},
}

SPATIAL_RESOLUTIONS = [0.25, 0.75]
ERA5_NATIVE_RES = 0.25
CAMS_NATIVE_RES = 0.75

YEARS = range(2015, 2025)


# ERA5 variable -> aggregation mode used during temporal resampling
ERA5_VARIABLES = {
    "tp":     "sum_mm",
    "dt2m":   "mean",
    "t2m":    "mean",
    "u10":    "mean",
    "v10":    "mean",
    "i10fg":  "max+mean",
    "blh":    "mean",
    "swvl1":  "mean",
    "cape":   "max",
    "lai_lv": "mean",
    "lai_hv": "mean",
}

WIND_VARS = {"u10", "v10"}


NC_ENCODING = lambda ds: {
    v: {"zlib": False, "dtype": "float32"} for v in ds.data_vars
}


_grid_cache = {}


def _clip_to_domain(lat, lon):
    dom = SPATIAL_DOMAIN
    lat_mask = (lat >= dom["south"] - 1e-9) & (lat <= dom["north"] + 1e-9)
    lon_mask = (lon >= dom["west"]  - 1e-9) & (lon <= dom["east"]  + 1e-9)
    return lat[lat_mask], lon[lon_mask]


def _read_era5_grid():
    candidates = [
        ERA5_PATH / "ERA5_land_sea_mask_2020.grib",
    ] + sorted(ERA5_PATH.glob("ERA5_t2m_2015.grib"))[:1]

    for f in candidates:
        if f.exists():
            ds = xr.open_dataset(f, engine="cfgrib")
            lat = ds["latitude"].values.astype(np.float64)
            lon = ds["longitude"].values.astype(np.float64)
            ds.close()
            if lat[0] < lat[-1]:
                lat = lat[::-1]
            return _clip_to_domain(lat, lon)

    raise FileNotFoundError(
        f"No ERA5 file found in {ERA5_PATH} to read native grid. "
        f"Download at least one ERA5 file first."
    )


def _read_cams_grid():
    candidates = (sorted(CAMS_PATH.glob("CAMS_2015.nc"))[:1]
                  + sorted(CAMS_PATH.glob("CAMS_*.nc"))[:1])

    for f in candidates:
        if f.exists():
            ds = xr.open_dataset(f)
            lat = ds["latitude"].values.astype(np.float64)
            lon = ds["longitude"].values.astype(np.float64)
            ds.close()
            if lat[0] < lat[-1]:
                lat = lat[::-1]
            return _clip_to_domain(lat, lon)

    raise FileNotFoundError(
        f"No CAMS file found in {CAMS_PATH} to read native grid. "
        f"Download at least one CAMS file first."
    )


def make_target_grid(spatial_res):
    """
    Return (lat, lon) for the target spatial resolution.

    lat is descending (N → S), lon ascending (W → E). Cached after the
    first call.
    """
    if spatial_res not in SPATIAL_RESOLUTIONS:
        raise ValueError(
            f"Unsupported spatial resolution {spatial_res}°. "
            f"Choose from: {SPATIAL_RESOLUTIONS}"
        )

    if spatial_res in _grid_cache:
        return _grid_cache[spatial_res]

    lat, lon = _read_era5_grid() if spatial_res == 0.25 else _read_cams_grid()
    _grid_cache[spatial_res] = (lat, lon)
    return lat, lon


def spatial_tag(spatial_res):
    """Filename tag: 0.25 → '025', 0.75 → '075'."""
    return f"{spatial_res * 100:03.0f}"


def file_tag(temporal_tag, spatial_res):
    """Combined tag: 'daily_025', '6hourly_075'."""
    return f"{temporal_tag}_{spatial_tag(spatial_res)}"
