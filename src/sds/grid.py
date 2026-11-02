"""
Native grids for the preprocessing pipeline.

The 0.25° grid is read from native ERA5 files; the 0.75° grid from native
CAMS files. This guarantees alignment with the source data. Grids are
clipped to the spatial domain (config.DOMAIN) and cached after first read.

Usage:
    from sds.grid import make_target_grid
"""

import numpy as np
import xarray as xr

from sds.config import (
    ERA5_NATIVE_RES, CAMS_PATH, DOMAIN, SUPPORTED_SPATIAL_RES,
    LAND_MASK_FILE, YEARS,
)


WIND_VARS = {"u10", "v10"}


_grid_cache = {}


def _clip_to_domain(lat, lon):
    dom = DOMAIN
    lat_mask = (lat >= dom["south"] - 1e-9) & (lat <= dom["north"] + 1e-9)
    lon_mask = (lon >= dom["west"]  - 1e-9) & (lon <= dom["east"]  + 1e-9)
    return lat[lat_mask], lon[lon_mask]


def _read_era5_grid():
    if not LAND_MASK_FILE.exists():
        raise FileNotFoundError(
            f"{LAND_MASK_FILE} not found (needed for the native 0.25° grid). "
            f"Run scripts/download/download_ecmwf.py first."
        )
    ds = xr.open_dataset(LAND_MASK_FILE, engine="cfgrib")
    lat = ds["latitude"].values.astype(np.float64)
    lon = ds["longitude"].values.astype(np.float64)
    ds.close()
    if lat[0] < lat[-1]:
        lat = lat[::-1]
    return _clip_to_domain(lat, lon)


def _read_cams_grid():
    cams_file = CAMS_PATH / f"CAMS_{min(YEARS)}.nc"
    if not cams_file.exists():
        raise FileNotFoundError(
            f"{cams_file} not found (needed for the native 0.75° grid). "
            f"Run scripts/download/download_ecmwf.py first."
        )
    ds = xr.open_dataset(cams_file)
    lat = ds["latitude"].values.astype(np.float64)
    lon = ds["longitude"].values.astype(np.float64)
    ds.close()
    if lat[0] < lat[-1]:
        lat = lat[::-1]
    return _clip_to_domain(lat, lon)


def make_target_grid(spatial_res):
    """
    Return (lat, lon) for the target spatial resolution.

    lat is descending (N → S), lon ascending (W → E). Cached after the
    first call.
    """
    if spatial_res not in SUPPORTED_SPATIAL_RES:
        raise ValueError(
            f"Unsupported spatial resolution {spatial_res}°. "
            f"Choose from: {SUPPORTED_SPATIAL_RES}"
        )

    if spatial_res in _grid_cache:
        return _grid_cache[spatial_res]

    lat, lon = _read_era5_grid() if spatial_res == ERA5_NATIVE_RES else _read_cams_grid()
    _grid_cache[spatial_res] = (lat, lon)
    return lat, lon
