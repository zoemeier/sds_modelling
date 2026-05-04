"""
SDS Intermediate Source Detection — Parameterised Runner
=========================================================
Step 2 of the backtracing pipeline. Reads outputs from Step 1
(sds_windshift_tracker.py) and detects intermediate dust emission
sources along existing plume tracks.

Detection logic (per pixel in each plume at time t):
  1. Volume pre-filter: skip cell if total DUAOD didn't grow vs t-1
  2. Dust signal: compare pixel at t to the plume-wide MEAN at t-1
       - DUAOD_pixel(t) - DUAOD_mean(t-1) > delta_duaod  (fresh dust added)
       - AE_mean(t-1) - AE_pixel(t) > delta_ae            (coarser = dustier)
       Both required.
  3. Surface filters: ws_max >= threshold, ws_mean >= threshold,
                      low NDVI, no snow, over land

Wind speed variables (ws_max, ws_mean) are computed from hourly u10/v10
*before* temporal aggregation, so ws_max is the true daily maximum wind
speed and ws_mean is the true daily mean wind speed.

All arguments accept multiple values for grid search.

Detection methods:
  - plume_ref:  Compare each pixel to plume-wide reference at t-1
  - upwind:     Compare each pixel to its upwind neighbourhood at t-1
  - combined:   Pixel must pass AE check against BOTH upwind neighbourhood
                AND plume-wide reference.  DUAOD check uses upwind ref.

Output structure:
  backtrace_overlap_output/windshift/{resolution}_{grid_res}/
    {step1_params}/{step2_params}/
      sds_intermediate_sources.nc
      sds_sources_raw.nc
      sds_sources_filtered.nc
      sds_intermediate_sources.csv
      sds_all_sources.csv
      sds_intermediate_filter_log.csv
      sds_pixel_filter_log.csv
      run_log.txt

Usage:
  python sds_intermediate_detector.py
  python sds_intermediate_detector.py --ws_max_thr 8.0 10.0 12.0 --dry_run
  python sds_intermediate_detector.py --delta_duaod 0.01 0.02 --delta_ae 0.01 0.02
  python sds_intermediate_detector.py --detection_method combined --upwind_nb 1
"""

from pathlib import Path
from collections import defaultdict
import argparse
import sys
import time as _time
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc4
from scipy import ndimage
from tqdm import tqdm
from itertools import product
import cftime
import os
import gc


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH    = PROJECT_ROOT / "data"
INTERIM_PATH = DATA_PATH / "interim"
ERA5_PATH    = DATA_PATH / "era5"
NDVI_PATH    = DATA_PATH / "ndvi"

def _experiment_summary_path():
    return (DATA_PATH / "backtrace_overlap_output" / "combination_overlap"
            / "intermediate_experiment_summary.csv")

RESOLUTION_MAP = {
    ("daily", 0.25): ("BACKTRACING_INPUT_2015-2024_daily_025.nc",    24),
    ("6h",    0.25): ("BACKTRACING_INPUT_2015-2024_6hourly_025.nc",   6),
    ("3h",    0.25): ("BACKTRACING_INPUT_2015-2024_3hourly_025.nc",   3),
    ("daily", 0.75): ("BACKTRACING_INPUT_2015-2024_daily_075.nc",    24),
    ("6h",    0.75): ("BACKTRACING_INPUT_2015-2024_6hourly_075.nc",   6),
    ("3h",    0.75): ("BACKTRACING_INPUT_2015-2024_3hourly_075.nc",   3),
}

# Map tracker resolution names to file tag components
TEMPORAL_TAG = {"daily": "daily", "6h": "6hourly", "3h": "3hourly"}

def _spatial_tag(grid_res):
    return f"{grid_res:.2f}".replace(".", "")

LAND_MASK_FILE = ERA5_PATH / "ERA5_land_sea_mask_2020.grib"
LSM_THRESHOLD  = 0.2


# =============================================================================
# LOGGING
# =============================================================================

class TeeLogger:
    def __init__(self, logpath: Path):
        self.terminal = sys.stdout
        self.logfile  = open(logpath, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()


# =============================================================================
# FOLDER NAME HELPERS
# =============================================================================

def _dotfree(val):
    s = f"{val:.10f}".rstrip("0").rstrip(".")
    return s.replace(".", "")


def make_step1_folder_name(angstr, duaod, pm10, min_pixels, iou,
                           max_gap, stubs, memory):
    """Reproduce Step 1's folder naming convention exactly."""
    ae_str    = _dotfree(angstr)
    duaod_str = _dotfree(duaod)
    exp = int(np.floor(np.log10(abs(pm10))))
    coeff = pm10 / (10 ** exp)
    while coeff != int(coeff) and abs(coeff) < 1e6:
        coeff *= 10
        exp -= 1
    pm10_str = f"{int(coeff)}en{exp}"
    iou_str = _dotfree(iou)
    return (f"{ae_str}ae_{duaod_str}duaod_{pm10_str}pm10"
            f"_{min_pixels}px_{iou_str}iou_{max_gap}gap_{stubs}stub_{memory}mem")


def make_step2_folder_name(ws_mean_thr, ws_max_thr, ndvi_thr, snow_thr,
                           delta_duaod, delta_ae, volume_growth, volume_abs, min_cluster,
                           ref_stat_ae, ref_stat_duaod, ae_px_thr=0.6,
                           wind_ndvi_dep=False,
                           ws_mean_base=None, ws_max_base=None,
                           detection_method="plume_ref", upwind_nb=1,
                           upwind_plume_k=4,
                           no_duaod_check=False, no_volume_filter=False,
                           plume_ae_precheck=False, plume_ae_percentile=10, plume_ae_delta=0.01):
    """Encode Step 2 parameters in a dot-free folder name."""
    if wind_ndvi_dep and ws_mean_base is not None and ws_max_base is not None:
        wind_tag = (f"ndvidep_{_dotfree(ws_mean_base)}to{_dotfree(ws_mean_thr)}wsmean"
                    f"_{_dotfree(ws_max_base)}to{_dotfree(ws_max_thr)}wsmax")
    else:
        wind_tag = f"{_dotfree(ws_mean_thr)}wsmean_{_dotfree(ws_max_thr)}wsmax"

    if detection_method == "upwind":
        det_tag = f"upwind{upwind_nb}_{ref_stat_duaod}_fb{ref_stat_ae}"
    elif detection_method == "combined":
        det_tag = f"combined{upwind_nb}_{ref_stat_duaod}_fb{ref_stat_ae}"
    elif detection_method == "upwind_plume":
        det_tag = f"upwindplume_k{upwind_plume_k}_{ref_stat_duaod}_fb{ref_stat_ae}"
    else:
        det_tag = f"{ref_stat_duaod}_fb{ref_stat_ae}"

    # Only include delta_duaod in name if the check is active
    if no_duaod_check:
        det_tag += f"_noduaod_{_dotfree(delta_ae)}dae"
    else:
        det_tag += f"_{_dotfree(delta_duaod)}dduaod_{_dotfree(delta_ae)}dae"

    # Volume filter params only if active
    if no_volume_filter:
        vol_tag = "novol"
    else:
        vol_tag = f"{_dotfree(volume_growth)}vol_{_dotfree(volume_abs)}abs"

    toggle_parts = []
    if plume_ae_precheck:
        toggle_parts.append(f"plumeae_p{plume_ae_percentile}_{_dotfree(plume_ae_delta)}")

    base = (f"{wind_tag}"
            f"_{_dotfree(ndvi_thr)}ndvi_{_dotfree(snow_thr)}snow"
            f"_{_dotfree(ae_px_thr)}aepx"
            f"_{det_tag}"
            f"_{vol_tag}_{min_cluster}clust")
    if toggle_parts:
        base += "_" + "_".join(toggle_parts)
    return base


# =============================================================================
# NDVI-DEPENDENT WIND THRESHOLDS
# =============================================================================

def _compute_wind_thr_ramp(ndvi, ws_thr_base, ws_thr_at_ndvi_max, ndvi_thr):
    """
    Per-pixel wind threshold as linear ramp of NDVI.

    ws_thr(ndvi) = ws_thr_base + (ws_thr_at_ndvi_max - ws_thr_base) * clamp(ndvi / ndvi_thr, 0, 1)

    - At NDVI=0 (bare): ws_thr = ws_thr_base (lowest, easiest to trigger)
    - At NDVI=ndvi_thr: ws_thr = ws_thr_at_ndvi_max (highest)
    - NaN NDVI: treated as bare → ws_thr_base

    Returns 2D float32 array same shape as ndvi.
    """
    ndvi_safe = ndvi.copy()
    nan_mask = np.isnan(ndvi_safe)
    ndvi_safe[nan_mask] = 0.0  # bare surface assumption for NaN
    ndvi_frac = np.clip(ndvi_safe / ndvi_thr, 0.0, 1.0)
    thr = ws_thr_base + (ws_thr_at_ndvi_max - ws_thr_base) * ndvi_frac
    return thr.astype(np.float32)


# =============================================================================
# HELPERS
# =============================================================================

def _time_dim(ds):
    for t in ("valid_time", "time"):
        if t in ds.dims or t in ds.coords:
            return t
    raise ValueError("No time coordinate found.")


def _compute_ae(aod469, aod865):
    with np.errstate(divide="ignore", invalid="ignore"):
        ae = -np.log(aod469 / aod865) / np.log(469.0 / 865.0)
    ae[~np.isfinite(ae)] = np.nan
    return ae


def _nearest_idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))


def _load_ndvi_fast(ti, ndvi_data, snow_data, ndvi_ti_map, ndvi_yi, ndvi_xi):
    """Fast NDVI lookup using preloaded arrays and precomputed index mapping."""
    ni = ndvi_ti_map[ti]
    ndvi_v = ndvi_data[ni]
    snow_v = snow_data[ni]
    if ndvi_yi is not None:
        ndvi_v = ndvi_v[np.ix_(ndvi_yi, ndvi_xi)]
        snow_v = snow_v[np.ix_(ndvi_yi, ndvi_xi)]
    return ndvi_v, snow_v


def load_land_mask(target_lat, target_lon):
    lsm_ds = xr.open_dataset(LAND_MASK_FILE, engine="cfgrib").squeeze()
    lsm_var = next((v for v in ("lsm", "land_sea_mask", "LSM") if v in lsm_ds),
                   list(lsm_ds.data_vars)[0])
    lsm_lat  = lsm_ds["latitude"].values
    lsm_lon  = lsm_ds["longitude"].values
    lsm_vals = lsm_ds[lsm_var].values.astype("float32")
    lsm_ds.close()
    if target_lon.min() < 0 and lsm_lon.max() > 180:
        lsm_lon  = np.where(lsm_lon > 180, lsm_lon - 360, lsm_lon)
        si       = np.argsort(lsm_lon)
        lsm_lon  = lsm_lon[si]
        lsm_vals = lsm_vals[:, si]
    lat_idx = np.array([int(np.argmin(np.abs(lsm_lat - la))) for la in target_lat])
    lon_idx = np.array([int(np.argmin(np.abs(lsm_lon - lo))) for lo in target_lon])
    land = lsm_vals[np.ix_(lat_idx, lon_idx)] > LSM_THRESHOLD
    return land.astype(bool)


# =============================================================================
# VOLUME PRE-FILTER (reused from previous version)
# =============================================================================

def build_event_groups(merge_df, split_df):
    split_events = defaultdict(list)
    merge_events = defaultdict(list)
    involved_cells = defaultdict(set)
    if len(split_df):
        for (par, t), grp in split_df.groupby(["parent_cell", "time"]):
            t = pd.Timestamp(t)
            children = [int(par)] + list(grp["child_cell"].astype(int).values)
            split_events[t].append({"parent": int(par), "children": children})
            for c in children:
                involved_cells[t].add(c)
    if len(merge_df):
        for (dom, t), grp in merge_df.groupby(["dominant_cell", "time"]):
            t = pd.Timestamp(t)
            parents = [int(dom)] + list(grp["absorbed_cell"].astype(int).values)
            merge_events[t].append({"parents": parents, "dominant": int(dom)})
            involved_cells[t].add(int(dom))
            for p in parents:
                involved_cells[t].add(p)
    return dict(split_events), dict(merge_events), dict(involved_cells)


def build_predecessor_lookup(merge_df, split_df):
    lookup = defaultdict(set)
    for _, row in merge_df.iterrows():
        t = pd.Timestamp(row["time"])
        dom = int(row["dominant_cell"]); abs_ = int(row["absorbed_cell"])
        lookup[(dom, t)].add(dom); lookup[(dom, t)].add(abs_)
    for _, row in split_df.iterrows():
        t = pd.Timestamp(row["time"])
        lookup[(int(row["child_cell"]), t)].add(int(row["parent_cell"]))
    return dict(lookup)


def dust_volume(cell_ids, seg_frame, duaod_frame):
    total = 0.0
    for cid in cell_ids:
        mask = seg_frame == cid
        if mask.any():
            total += float(np.nansum(duaod_frame[mask]))
    return total


# =============================================================================
# INTERMEDIATE DETECTION
# =============================================================================

def _compute_plume_ref(values, ref_stat):
    """
    Compute a reference statistic from an array of values.

    ref_stat : str
        "mean", "median", "p25", "p50", "p75", etc.
    """
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan
    if ref_stat == "mean":
        return float(np.mean(finite))
    elif ref_stat == "median":
        return float(np.median(finite))
    elif ref_stat.startswith("p"):
        q = float(ref_stat[1:])
        return float(np.percentile(finite, q))
    else:
        raise ValueError(f"Unknown ref_stat: {ref_stat}")


def detect_intermediates_plume_ref(
    plume_t, duaod_t, ae_t,
    plume_ref_duaod_t1, plume_ref_ae_t1,
    delta_duaod_thr, delta_ae_thr,
    no_duaod_check=False,
):
    """
    Detect intermediate source pixels by comparing each pixel at t to a
    plume-wide reference statistic at t-1.

    A pixel is flagged if BOTH:
      - DUAOD_pixel(t) - ref_DUAOD(t-1) > delta_duaod_thr  (more dust)
      - ref_AE(t-1) - AE_pixel(t) > delta_ae_thr           (coarser)

    Returns (candidate_mask, counts_dict).
    """
    counts = dict(
        px_total=0,
        px_nonfinite=0,
        px_fail_duaod=0,
        px_fail_ae=0,
        px_fail_both=0,
        px_passed=0,
    )

    candidate = np.zeros(plume_t.shape, dtype=bool)

    if not plume_t.any():
        return candidate, counts

    if not (np.isfinite(plume_ref_duaod_t1) and np.isfinite(plume_ref_ae_t1)):
        return candidate, counts

    # Vectorised: evaluate all plume pixels at once
    duaod_vals = duaod_t[plume_t]
    ae_vals    = ae_t[plume_t]

    n_total = int(plume_t.sum())
    counts["px_total"] = n_total

    finite = np.isfinite(duaod_vals) & np.isfinite(ae_vals)
    counts["px_nonfinite"] = int((~finite).sum())

    duaod_pass = np.ones_like(duaod_vals, dtype=bool) if no_duaod_check else \
                 (duaod_vals - plume_ref_duaod_t1) > delta_duaod_thr
    ae_pass    = (plume_ref_ae_t1 - ae_vals) > delta_ae_thr

    both_pass = finite & duaod_pass & ae_pass

    # Detailed failure counts (among finite pixels only)
    fin_fail_duaod = finite & ~duaod_pass
    fin_fail_ae    = finite & ~ae_pass
    counts["px_fail_duaod"] = int((fin_fail_duaod & ae_pass).sum())    # only duaod failed
    counts["px_fail_ae"]    = int((fin_fail_ae & duaod_pass).sum())    # only ae failed
    counts["px_fail_both"]  = int((fin_fail_duaod & fin_fail_ae).sum())  # both failed
    counts["px_passed"]     = int(both_pass.sum())

    # Map back to 2D
    ys, xs = np.where(plume_t)
    candidate[ys[both_pass], xs[both_pass]] = True

    return candidate, counts


def detect_intermediates_upwind(
    plume_t, duaod_t, ae_t, duaod_t1, ae_t1,
    u10_t, v10_t, dt_hours, grid_res, lat_arr,
    delta_duaod_thr, delta_ae_thr,
    neighbourhood=1, ref_stat="median",
    plume_ref_duaod=None, plume_ref_ae=None,
    no_duaod_check=False,
):
    """
    Detect intermediate source pixels by comparing each pixel at t to
    its upwind neighbourhood at t-1.

    For each plume pixel at (y, x) at time t:
      1. Use u10/v10 at that pixel to compute where the air came from at t-1
      2. Look at a (2*neighbourhood+1) × (2*neighbourhood+1) window around
         that upwind position in duaod_t1 / ae_t1
      3. Compute the reference statistic (mean/median) of that window
      4. Flag if DUAOD increased and AE decreased vs the upwind reference

    If the upwind position is out-of-bounds or has no valid data, falls back
    to plume-wide reference (plume_ref_duaod, plume_ref_ae) if provided.

    Returns (candidate_mask, counts_dict).
    """
    counts = dict(
        px_total=0,
        px_nonfinite=0,
        px_fail_duaod=0,
        px_fail_ae=0,
        px_fail_both=0,
        px_no_upwind=0,
        px_fallback=0,
        px_passed=0,
    )

    has_fallback = (plume_ref_duaod is not None and plume_ref_ae is not None and
                    np.isfinite(plume_ref_duaod) and np.isfinite(plume_ref_ae))

    NLAT, NLON = plume_t.shape
    candidate = np.zeros((NLAT, NLON), dtype=bool)

    if not plume_t.any():
        return candidate, counts

    ys, xs = np.where(plume_t)
    n_total = len(ys)
    counts["px_total"] = n_total

    dt_sec = dt_hours * 3600.0
    meters_per_deg_lat = 111_320.0
    lat_descending = lat_arr[0] > lat_arr[-1] if len(lat_arr) > 1 else False
    nb = neighbourhood

    for k in range(n_total):
        y, x = ys[k], xs[k]

        duaod_here = duaod_t[y, x]
        ae_here = ae_t[y, x]

        if not (np.isfinite(duaod_here) and np.isfinite(ae_here)):
            counts["px_nonfinite"] += 1
            continue

        # Compute upwind shift in grid cells
        u = u10_t[y, x]
        v = v10_t[y, x]

        ref_duaod = None
        ref_ae = None
        used_fallback = False

        if np.isfinite(u) and np.isfinite(v):
            lat_center = lat_arr[y]
            meters_per_deg_lon = 111_320.0 * np.cos(np.radians(lat_center))

            # Wind displacement in degrees (where the air went TO)
            dx_deg = (u * dt_sec) / meters_per_deg_lon
            dy_deg = (v * dt_sec) / meters_per_deg_lat

            # Upwind = opposite direction (where air came FROM)
            dx_cells = -dx_deg / grid_res
            if lat_descending:
                dy_cells = dy_deg / grid_res  # flip for descending lat
            else:
                dy_cells = -dy_deg / grid_res

            # Upwind position at t-1
            uy = y + int(round(dy_cells))
            ux = x + int(round(dx_cells))

            # Extract neighbourhood window around upwind position
            y0 = max(0, uy - nb)
            y1 = min(NLAT, uy + nb + 1)
            x0 = max(0, ux - nb)
            x1 = min(NLON, ux + nb + 1)

            if y0 < y1 and x0 < x1:
                duaod_patch = duaod_t1[y0:y1, x0:x1]
                ae_patch = ae_t1[y0:y1, x0:x1]

                duaod_finite = duaod_patch[np.isfinite(duaod_patch)]
                ae_finite = ae_patch[np.isfinite(ae_patch)]

                if len(duaod_finite) > 0 and len(ae_finite) > 0:
                    if ref_stat == "median":
                        ref_duaod = float(np.median(duaod_finite))
                        ref_ae = float(np.median(ae_finite))
                    else:
                        ref_duaod = float(np.mean(duaod_finite))
                        ref_ae = float(np.mean(ae_finite))

        # Fallback to plume-wide reference if upwind failed
        if ref_duaod is None or ref_ae is None:
            if has_fallback:
                ref_duaod = plume_ref_duaod
                ref_ae = plume_ref_ae
                used_fallback = True
                counts["px_fallback"] += 1
            else:
                counts["px_no_upwind"] += 1
                continue

        # Detection: DUAOD increased AND AE decreased vs reference
        duaod_pass = True if no_duaod_check else (duaod_here - ref_duaod) > delta_duaod_thr
        ae_pass = (ref_ae - ae_here) > delta_ae_thr

        if duaod_pass and ae_pass:
            candidate[y, x] = True
            counts["px_passed"] += 1
        elif not duaod_pass and not ae_pass:
            counts["px_fail_both"] += 1
        elif not duaod_pass:
            counts["px_fail_duaod"] += 1
        else:
            counts["px_fail_ae"] += 1

    return candidate, counts


def detect_intermediates_upwind_plume(
    plume_t, plume_t1_mask, duaod_t, ae_t, duaod_t1, ae_t1,
    u10_t, v10_t, dt_hours, grid_res, lat_arr, lon_arr,
    delta_duaod_thr, delta_ae_thr,
    k_neighbours=4, ref_stat="median",
    plume_ref_duaod=None, plume_ref_ae=None,
    no_duaod_check=False,
):
    """
    Detect intermediate source pixels by comparing each pixel at t to the
    K closest predecessor-plume pixels near the upwind position at t-1.

    Unlike the 'upwind' method (which uses a fixed spatial window), this
    method restricts the reference to pixels that belong to the predecessor
    plume at t-1.  This ensures the comparison is always against the dust
    that was actually being advected, not against background or other plumes.

    For each plume pixel at (y, x) at time t:
      1. Use u10/v10 at that pixel to compute the upwind position at t-1
      2. Find the K nearest pixels in plume_t1_mask (predecessor plume at t-1)
         to that upwind position, using a prebuilt KD-tree
      3. Compute the reference statistic (mean/median) of DUAOD/AE at those
         K predecessor-plume pixels
      4. Flag if DUAOD increased and AE decreased vs the reference

    If the upwind position yields no valid reference (e.g. predecessor plume
    is empty, or all K neighbours have NaN values), falls back to plume-wide
    reference if provided.

    Parameters
    ----------
    plume_t : 2D bool, plume mask at time t
    plume_t1_mask : 2D bool, predecessor plume mask at t-1
    duaod_t, ae_t : 2D float, aerosol fields at time t
    duaod_t1, ae_t1 : 2D float, aerosol fields at time t-1
    u10_t, v10_t : 2D float, wind components at time t (m/s)
    dt_hours : int, timestep duration in hours
    grid_res : float, grid resolution in degrees
    lat_arr : 1D float, latitude array
    lon_arr : 1D float, longitude array
    delta_duaod_thr, delta_ae_thr : float, detection thresholds
    k_neighbours : int, number of nearest predecessor-plume pixels (default 4)
    ref_stat : str, "mean" or "median" for aggregation over the K neighbours
    plume_ref_duaod : float or None, plume-wide DUAOD fallback
    plume_ref_ae : float or None, plume-wide AE fallback
    no_duaod_check : bool, if True skip DUAOD check

    Returns (candidate_mask, counts_dict).
    """
    counts = dict(
        px_total=0,
        px_nonfinite=0,
        px_fail_duaod=0,
        px_fail_ae=0,
        px_fail_both=0,
        px_no_upwind=0,
        px_fallback=0,
        px_passed=0,
    )

    NLAT, NLON = plume_t.shape
    candidate = np.zeros((NLAT, NLON), dtype=bool)

    if not plume_t.any():
        return candidate, counts

    has_fallback = (plume_ref_duaod is not None and plume_ref_ae is not None and
                    np.isfinite(plume_ref_duaod) and np.isfinite(plume_ref_ae))

    # Build KD-tree of predecessor plume pixels at t-1
    ys_t1, xs_t1 = np.where(plume_t1_mask)
    if len(ys_t1) == 0:
        # No predecessor plume pixels — fall back to plume-wide ref or skip
        if has_fallback:
            return detect_intermediates_plume_ref(
                plume_t, duaod_t, ae_t,
                plume_ref_duaod, plume_ref_ae,
                delta_duaod_thr, delta_ae_thr,
                no_duaod_check=no_duaod_check,
            )
        return candidate, counts

    # KD-tree in (lat, lon) space — scale lon by cos(lat) for distance
    mean_lat = float(lat_arr[ys_t1].mean())
    cos_lat = np.cos(np.radians(mean_lat))
    tree_coords = np.column_stack([
        lat_arr[ys_t1],
        lon_arr[xs_t1] * cos_lat,
    ])
    from scipy.spatial import cKDTree
    tree_t1 = cKDTree(tree_coords)

    # Preextract t-1 values at predecessor plume pixels for fast indexing
    duaod_t1_plume = duaod_t1[ys_t1, xs_t1]
    ae_t1_plume = ae_t1[ys_t1, xs_t1]

    ys, xs = np.where(plume_t)
    n_total = len(ys)
    counts["px_total"] = n_total

    dt_sec = dt_hours * 3600.0
    meters_per_deg_lat = 111_320.0
    lat_descending = lat_arr[0] > lat_arr[-1] if len(lat_arr) > 1 else False

    k_query = min(k_neighbours, len(ys_t1))

    for idx in range(n_total):
        y, x = ys[idx], xs[idx]

        duaod_here = duaod_t[y, x]
        ae_here = ae_t[y, x]

        if not (np.isfinite(duaod_here) and np.isfinite(ae_here)):
            counts["px_nonfinite"] += 1
            continue

        # Compute upwind position
        u = u10_t[y, x]
        v = v10_t[y, x]

        ref_duaod = None
        ref_ae = None

        if np.isfinite(u) and np.isfinite(v):
            lat_center = lat_arr[y]
            meters_per_deg_lon = 111_320.0 * np.cos(np.radians(lat_center))

            dx_deg = (u * dt_sec) / meters_per_deg_lon
            dy_deg = (v * dt_sec) / meters_per_deg_lat

            dx_cells = -dx_deg / grid_res
            if lat_descending:
                dy_cells = dy_deg / grid_res
            else:
                dy_cells = -dy_deg / grid_res

            up_lat = lat_arr[y] + dy_cells * grid_res
            up_lon = lon_arr[x] + dx_cells * grid_res

            # Query K nearest predecessor-plume pixels to upwind position
            query_pt = np.array([up_lat, up_lon * cos_lat])
            dists, idxs = tree_t1.query(query_pt, k=k_query)
            if np.isscalar(idxs):
                idxs = np.array([idxs])

            # Extract values at those K plume pixels
            ref_duaod_vals = duaod_t1_plume[idxs]
            ref_ae_vals = ae_t1_plume[idxs]

            duaod_finite = ref_duaod_vals[np.isfinite(ref_duaod_vals)]
            ae_finite = ref_ae_vals[np.isfinite(ref_ae_vals)]

            if len(duaod_finite) > 0 and len(ae_finite) > 0:
                if ref_stat == "median":
                    ref_duaod = float(np.median(duaod_finite))
                    ref_ae = float(np.median(ae_finite))
                else:
                    ref_duaod = float(np.mean(duaod_finite))
                    ref_ae = float(np.mean(ae_finite))

        # Fallback to plume-wide reference
        if ref_duaod is None or ref_ae is None:
            if has_fallback:
                ref_duaod = plume_ref_duaod
                ref_ae = plume_ref_ae
                counts["px_fallback"] += 1
            else:
                counts["px_no_upwind"] += 1
                continue

        # Detection
        duaod_pass = True if no_duaod_check else (duaod_here - ref_duaod) > delta_duaod_thr
        ae_pass = (ref_ae - ae_here) > delta_ae_thr

        if duaod_pass and ae_pass:
            candidate[y, x] = True
            counts["px_passed"] += 1
        elif not duaod_pass and not ae_pass:
            counts["px_fail_both"] += 1
        elif not duaod_pass:
            counts["px_fail_duaod"] += 1
        else:
            counts["px_fail_ae"] += 1

    return candidate, counts


def detect_intermediates_combined(
    plume_t, duaod_t, ae_t, duaod_t1, ae_t1,
    u10_t, v10_t, dt_hours, grid_res, lat_arr,
    plume_ref_duaod_t1, plume_ref_ae_t1,
    delta_duaod_thr, delta_ae_thr,
    neighbourhood=1, ref_stat="median",
    no_duaod_check=False,
):
    """
    Combined detection: a pixel passes only if it satisfies the AE decrease
    criterion against BOTH its upwind neighbourhood AND the plume-wide
    reference at t-1.

    Logic per pixel at (y, x):
      1. Compute upwind reference AE and DUAOD (same as upwind method)
      2. AE check must pass against BOTH references:
           plume_ref_AE(t-1) - AE_pixel(t) > delta_ae   AND
           upwind_ref_AE(t-1) - AE_pixel(t) > delta_ae
      3. DUAOD check (if active) uses the upwind reference:
           DUAOD_pixel(t) - upwind_ref_DUAOD(t-1) > delta_duaod

    If the upwind position is out-of-bounds or has no valid data, the pixel
    is skipped (no fallback — both references are required).

    Parameters
    ----------
    plume_t : 2D bool, plume mask at time t
    duaod_t, ae_t : 2D float, aerosol fields at time t
    duaod_t1, ae_t1 : 2D float, aerosol fields at time t-1
    u10_t, v10_t : 2D float, wind components at time t (m/s)
    dt_hours : int, timestep duration in hours
    grid_res : float, grid resolution in degrees
    lat_arr : 1D float, latitude array
    plume_ref_duaod_t1 : float, plume-wide DUAOD reference at t-1
    plume_ref_ae_t1 : float, plume-wide AE reference at t-1
    delta_duaod_thr, delta_ae_thr : float, detection thresholds
    neighbourhood : int, half-size of window around upwind position (default 1 = 3x3)
    ref_stat : str, "mean" or "median" for the neighbourhood aggregation
    no_duaod_check : bool, if True skip DUAOD check

    Returns (candidate_mask, counts_dict).
    """
    counts = dict(
        px_total=0,
        px_nonfinite=0,
        px_fail_duaod=0,
        px_fail_ae_upwind=0,
        px_fail_ae_plume=0,
        px_fail_ae_both=0,
        px_fail_ae_and_duaod=0,
        px_no_upwind=0,
        px_passed=0,
    )

    NLAT, NLON = plume_t.shape
    candidate = np.zeros((NLAT, NLON), dtype=bool)

    if not plume_t.any():
        return candidate, counts

    has_plume_ref = (np.isfinite(plume_ref_duaod_t1) and np.isfinite(plume_ref_ae_t1))
    if not has_plume_ref:
        return candidate, counts

    ys, xs = np.where(plume_t)
    n_total = len(ys)
    counts["px_total"] = n_total

    dt_sec = dt_hours * 3600.0
    meters_per_deg_lat = 111_320.0
    lat_descending = lat_arr[0] > lat_arr[-1] if len(lat_arr) > 1 else False
    nb = neighbourhood

    for k in range(n_total):
        y, x = ys[k], xs[k]

        duaod_here = duaod_t[y, x]
        ae_here = ae_t[y, x]

        if not (np.isfinite(duaod_here) and np.isfinite(ae_here)):
            counts["px_nonfinite"] += 1
            continue

        # ── Compute upwind reference ──────────────────────────────
        u = u10_t[y, x]
        v = v10_t[y, x]

        upwind_ref_duaod = None
        upwind_ref_ae = None

        if np.isfinite(u) and np.isfinite(v):
            lat_center = lat_arr[y]
            meters_per_deg_lon = 111_320.0 * np.cos(np.radians(lat_center))

            dx_deg = (u * dt_sec) / meters_per_deg_lon
            dy_deg = (v * dt_sec) / meters_per_deg_lat

            dx_cells = -dx_deg / grid_res
            if lat_descending:
                dy_cells = dy_deg / grid_res
            else:
                dy_cells = -dy_deg / grid_res

            uy = y + int(round(dy_cells))
            ux = x + int(round(dx_cells))

            y0 = max(0, uy - nb)
            y1 = min(NLAT, uy + nb + 1)
            x0 = max(0, ux - nb)
            x1 = min(NLON, ux + nb + 1)

            if y0 < y1 and x0 < x1:
                duaod_patch = duaod_t1[y0:y1, x0:x1]
                ae_patch = ae_t1[y0:y1, x0:x1]

                duaod_finite = duaod_patch[np.isfinite(duaod_patch)]
                ae_finite = ae_patch[np.isfinite(ae_patch)]

                if len(duaod_finite) > 0 and len(ae_finite) > 0:
                    if ref_stat == "median":
                        upwind_ref_duaod = float(np.median(duaod_finite))
                        upwind_ref_ae = float(np.median(ae_finite))
                    else:
                        upwind_ref_duaod = float(np.mean(duaod_finite))
                        upwind_ref_ae = float(np.mean(ae_finite))

        # No fallback in combined mode — both references required
        if upwind_ref_duaod is None or upwind_ref_ae is None:
            counts["px_no_upwind"] += 1
            continue

        # ── DUAOD check: upwind reference ─────────────────────────
        duaod_pass = True if no_duaod_check else \
                     (duaod_here - upwind_ref_duaod) > delta_duaod_thr

        # ── AE check: must pass BOTH references ──────────────────
        ae_pass_upwind = (upwind_ref_ae - ae_here) > delta_ae_thr
        ae_pass_plume  = (plume_ref_ae_t1 - ae_here) > delta_ae_thr
        ae_pass = ae_pass_upwind and ae_pass_plume

        if duaod_pass and ae_pass:
            candidate[y, x] = True
            counts["px_passed"] += 1
        elif not duaod_pass and not ae_pass:
            counts["px_fail_ae_and_duaod"] += 1
        elif not duaod_pass:
            counts["px_fail_duaod"] += 1
        elif not ae_pass_upwind and not ae_pass_plume:
            counts["px_fail_ae_both"] += 1
        elif not ae_pass_upwind:
            counts["px_fail_ae_upwind"] += 1
        else:
            counts["px_fail_ae_plume"] += 1

    return candidate, counts


def apply_emission_filter(candidate_mask, ws_max_t, ws_mean_t, ae_t,
                          ndvi, snow, land_mask,
                          ws_mean_thr, ws_max_thr, ae_px_thr, ndvi_thr, snow_thr,
                          wind_ndvi_dep=False, ws_mean_base=None, ws_max_base=None):
    """
    Surface emission filter.  A pixel passes if ALL of:
      - ws_mean >= threshold,  ws_max >= threshold
      - AE <= ae_px_thr or NaN  (absolute dust-like AE at this pixel)
      - NDVI < ndvi_thr or NaN,  snow < snow_thr or NaN
      - land mask
    """
    if not candidate_mask.any():
        return candidate_mask, dict(
            n_candidates=0, n_kept=0,
            n_failed_ws_mean=0, n_failed_ws_max=0, n_failed_ae_px=0,
            n_failed_ndvi=0, n_failed_snow=0, n_failed_land=0,
            n_failed_ws_mean_only=0, n_failed_ws_max_only=0,
            n_failed_ae_px_only=0,
            n_failed_ndvi_only=0, n_failed_snow_only=0,
            n_failed_land_only=0, n_failed_multi=0,
        )

    if wind_ndvi_dep and ws_mean_base is not None and ws_max_base is not None:
        ws_mean_thr_arr = _compute_wind_thr_ramp(ndvi, ws_mean_base, ws_mean_thr, ndvi_thr)
        ws_max_thr_arr  = _compute_wind_thr_ramp(ndvi, ws_max_base, ws_max_thr, ndvi_thr)
        ws_mean_ok = ws_mean_t >= ws_mean_thr_arr
        ws_max_ok  = ws_max_t >= ws_max_thr_arr
    else:
        ws_mean_ok = ws_mean_t >= ws_mean_thr
        ws_max_ok  = ws_max_t >= ws_max_thr

    ae_px_ok   = np.isnan(ae_t) | (ae_t <= ae_px_thr)
    ndvi_ok    = np.isnan(ndvi) | (ndvi < ndvi_thr)
    snow_ok    = np.isnan(snow) | (snow < snow_thr)
    land_ok    = land_mask

    filtered = candidate_mask & ws_mean_ok & ws_max_ok & ae_px_ok & ndvi_ok & snow_ok & land_ok

    fwm = candidate_mask & ~ws_mean_ok
    fwx = candidate_mask & ~ws_max_ok
    fae = candidate_mask & ~ae_px_ok
    fn  = candidate_mask & ~ndvi_ok
    fs  = candidate_mask & ~snow_ok
    fl  = candidate_mask & ~land_ok

    fwm_only = fwm & ws_max_ok & ae_px_ok & ndvi_ok & snow_ok & land_ok
    fwx_only = fwx & ws_mean_ok & ae_px_ok & ndvi_ok & snow_ok & land_ok
    fae_only = fae & ws_mean_ok & ws_max_ok & ndvi_ok & snow_ok & land_ok
    fn_only  = fn & ws_mean_ok & ws_max_ok & ae_px_ok & snow_ok & land_ok
    fs_only  = fs & ws_mean_ok & ws_max_ok & ae_px_ok & ndvi_ok & land_ok
    fl_only  = fl & ws_mean_ok & ws_max_ok & ae_px_ok & ndvi_ok & snow_ok

    n_fails = ((~ws_mean_ok).astype(np.int8) + (~ws_max_ok).astype(np.int8) +
               (~ae_px_ok).astype(np.int8) +
               (~ndvi_ok).astype(np.int8) + (~snow_ok).astype(np.int8) +
               (~land_ok).astype(np.int8))

    return filtered, dict(
        n_candidates=int(candidate_mask.sum()),
        n_kept=int(filtered.sum()),
        n_failed_ws_mean=int(fwm.sum()),
        n_failed_ws_max=int(fwx.sum()),
        n_failed_ae_px=int(fae.sum()),
        n_failed_ndvi=int(fn.sum()),
        n_failed_snow=int(fs.sum()),
        n_failed_land=int(fl.sum()),
        n_failed_ws_mean_only=int(fwm_only.sum()),
        n_failed_ws_max_only=int(fwx_only.sum()),
        n_failed_ae_px_only=int(fae_only.sum()),
        n_failed_ndvi_only=int(fn_only.sum()),
        n_failed_snow_only=int(fs_only.sum()),
        n_failed_land_only=int(fl_only.sum()),
        n_failed_multi=int((candidate_mask & (n_fails >= 2)).sum()),
    )


# =============================================================================
# MAIN DETECTION FUNCTION
# =============================================================================

def run_intermediate_detection(
    # Step 1 parameters (to locate output folder)
    resolution="daily", grid_res=0.25,
    s1_angstr=0.8, s1_duaod=0.2, s1_pm10=1.5e-7,
    s1_min_pixels=4, s1_iou=0.1, s1_max_gap=1, s1_stubs=1, s1_memory=0,
    # Step 2 parameters
    ws_mean_thr=6.0, ws_max_thr=8.0, ae_px_thr=0.6,
    ndvi_thr=0.15, snow_thr=0.25,
    delta_duaod=0.01, delta_ae=0.02,
    volume_growth=0.01, volume_abs=1.0, min_cluster=4,
    ref_stat_ae="median", ref_stat_duaod="p50",
    # NDVI-dependent wind thresholds
    wind_ndvi_dep=False, ws_mean_base=3.5, ws_max_base=5.0,
    # Detection method
    detection_method="plume_ref", upwind_nb=1, upwind_plume_k=4,
    # Toggle filters
    no_duaod_check=False, no_volume_filter=False,
    plume_ae_precheck=False, plume_ae_percentile=10, plume_ae_delta=0.01,
):
    wall_start = _time.time()

    # ---- Resolve paths ----
    res_key = (resolution, grid_res)
    if res_key not in RESOLUTION_MAP:
        raise ValueError(f"Unknown resolution combo {res_key}")
    in_filename, dt_hours = RESOLUTION_MAP[res_key]

    grid_res_str = f"{grid_res:.2f}".replace(".", "")
    s1_folder = make_step1_folder_name(
        s1_angstr, s1_duaod, s1_pm10, s1_min_pixels,
        s1_iou, s1_max_gap, s1_stubs, s1_memory,
    )
    s1_dir = (DATA_PATH / "backtrace_overlap_output" / "combination_overlap"
              / f"{resolution}_{grid_res_str}" / s1_folder)

    if not s1_dir.exists():
        print(f"  ERROR: Step 1 output not found: {s1_dir}")
        return None

    s2_folder = make_step2_folder_name(
        ws_mean_thr, ws_max_thr, ndvi_thr, snow_thr, delta_duaod,
        delta_ae, volume_growth, volume_abs, min_cluster, ref_stat_ae, ref_stat_duaod,
        ae_px_thr=ae_px_thr,
        wind_ndvi_dep=wind_ndvi_dep,
        ws_mean_base=ws_mean_base, ws_max_base=ws_max_base,
        detection_method=detection_method, upwind_nb=upwind_nb,
        upwind_plume_k=upwind_plume_k,
        no_duaod_check=no_duaod_check, no_volume_filter=no_volume_filter,
        plume_ae_precheck=plume_ae_precheck, plume_ae_percentile=plume_ae_percentile,
        plume_ae_delta=plume_ae_delta,
    )
    out_dir = s1_dir / s2_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    done_marker = out_dir / "_RUN_COMPLETE"
    if done_marker.exists():
        print(f"  SKIPPING (already completed): {out_dir}")
        return None

    # ---- Logging ----
    logger = TeeLogger(out_dir / "run_log.txt")
    sys.stdout = logger

    print("=" * 72)
    print("SDS Intermediate Source Detection (plume-mean comparison)")
    print("=" * 72)
    print(f"  Resolution     : {resolution} (dt = {dt_hours}h)")
    print(f"  Grid res       : {grid_res}°")
    print(f"  Step 1 dir     : {s1_dir}")
    print(f"  Step 2 dir     : {out_dir}")
    print(f"  WS_MEAN_THR    : {ws_mean_thr}")
    print(f"  WS_MAX_THR     : {ws_max_thr}")
    print(f"  WIND_NDVI_DEP  : {wind_ndvi_dep}")
    if wind_ndvi_dep:
        print(f"  WS_MEAN_BASE   : {ws_mean_base}  (at NDVI=0)")
        print(f"  WS_MAX_BASE    : {ws_max_base}  (at NDVI=0)")
        print(f"    Ramp: ws_mean_thr(ndvi) = {ws_mean_base} + "
              f"{(ws_mean_thr - ws_mean_base)/ndvi_thr:.1f} × ndvi  (up to NDVI={ndvi_thr})")
        print(f"    Ramp: ws_max_thr(ndvi)  = {ws_max_base} + "
              f"{(ws_max_thr - ws_max_base)/ndvi_thr:.1f} × ndvi  (up to NDVI={ndvi_thr})")
    print(f"  AE_PX_THR      : {ae_px_thr}  (per-pixel AE <= this to be a source)")
    print(f"  NDVI_THR       : {ndvi_thr}")
    print(f"  SNOW_THR       : {snow_thr}")
    print(f"  DELTA_DUAOD    : {delta_duaod}")
    print(f"  DELTA_AE       : {delta_ae}")
    print(f"  VOLUME_GROWTH  : {volume_growth}")
    print(f"  VOLUME_ABS     : {volume_abs}")
    print(f"  MIN_CLUSTER    : {min_cluster}")
    print(f"  REF_STAT_AE       : {ref_stat_ae}")
    print(f"  REF_STAT_DUAOD    : {ref_stat_duaod}")
    print(f"  DETECTION_METHOD  : {detection_method}")
    if detection_method in ("upwind", "combined"):
        print(f"  UPWIND_NB         : {upwind_nb}  ({2*upwind_nb+1}×{2*upwind_nb+1} window)")
    if detection_method == "upwind_plume":
        print(f"  UPWIND_PLUME_K    : {upwind_plume_k}  (K nearest predecessor-plume pixels)")
    if detection_method == "combined":
        print(f"    AE must pass BOTH upwind AND plume-wide reference")
        print(f"    DUAOD uses upwind reference")
    print(f"  DUAOD_CHECK       : {not no_duaod_check}")
    print(f"  VOLUME_FILTER     : {not no_volume_filter}")
    print(f"  PLUME_AE_PRECHECK : {plume_ae_precheck}")
    if plume_ae_precheck:
        print(f"    Percentile      : p{plume_ae_percentile}")
        print(f"    Delta           : {plume_ae_delta}")
    print()

    # ---- Load data ----
    print("Loading data ...")
    input_file = INTERIM_PATH / in_filename
    ds = xr.open_dataset(input_file)
    if _time_dim(ds) != "time":
        ds = ds.rename({_time_dim(ds): "time"})

    seg_ds  = xr.open_dataset(s1_dir / "sds_segmentation.nc")
    seg_var = list(seg_ds.data_vars)[0]

    tracks = pd.read_hdf(s1_dir / "sds_tracks.h5", "table")
    tracks["time"] = pd.to_datetime(tracks["time"])
    tracks["cell"] = tracks["cell"].astype(int)

    # Merge / split logs
    merge_csv = s1_dir / "sds_merges.csv"
    split_csv = s1_dir / "sds_splits.csv"
    merge_df  = pd.read_csv(merge_csv) if merge_csv.exists() else pd.DataFrame()
    split_df  = pd.read_csv(split_csv) if split_csv.exists() else pd.DataFrame()
    if len(merge_df):
        merge_df["time"] = pd.to_datetime(merge_df["time"])
        merge_df["dominant_cell"] = merge_df["dominant_cell"].astype(int)
        merge_df["absorbed_cell"] = merge_df["absorbed_cell"].astype(int)
    if len(split_df):
        split_df["time"] = pd.to_datetime(split_df["time"])
        split_df["parent_cell"] = split_df["parent_cell"].astype(int)
        split_df["child_cell"]  = split_df["child_cell"].astype(int)

    predecessor_lookup = build_predecessor_lookup(merge_df, split_df)
    split_events, merge_events, involved_cells = build_event_groups(merge_df, split_df)

    # Genesis birth masks (for painting genesis pixels in output)
    src_masks_xr = xr.open_dataset(s1_dir / "sds_source_masks.nc")
    sm_cids  = src_masks_xr["cell"].values.astype(int)
    sm_times = src_masks_xr["source_time"].values

    # NDVI — use resolution-specific file
    ttag = TEMPORAL_TAG[resolution]
    stag = _spatial_tag(grid_res)
    ndvi_file = NDVI_PATH / f"NDVI_2015-2024_{ttag}_{stag}.nc"
    use_ndvi  = ndvi_file.exists()
    ndvi_data = ndvi_ti_map = ndvi_yi = ndvi_xi = snow_data = None
    ndvi_ds = None
    if use_ndvi:
        ndvi_ds = xr.open_dataset(ndvi_file)
        # Ensure NDVI time axis is datetime64
        if hasattr(ndvi_ds.time.values[0], "datetime"):
            ndvi_ds["time"] = pd.to_datetime([
                str(t)[:19] for t in ndvi_ds.time.values
            ])
        # Preload NDVI data arrays for fast isel
        ndvi_data = ndvi_ds["NDVI_mean"].values.astype("float32")
        snow_data = ndvi_ds["snow_frac"].values.astype("float32")
        ndvi_times = ndvi_ds.time.values.astype("datetime64[ns]")
        # Check if regridding is needed
        lat_name = "lat" if "lat" in ndvi_ds.dims else "latitude"
        lon_name = "lon" if "lon" in ndvi_ds.dims else "longitude"
        ndvi_lat = ndvi_ds[lat_name].values
        ndvi_lon = ndvi_ds[lon_name].values
        ndvi_ds.close()
    print(f"  NDVI: {ndvi_file.name} ({'found' if use_ndvi else 'NOT FOUND'})")

    # Check wind speed variables
    has_ws_max  = "ws_max" in ds.data_vars
    has_ws_mean = "ws_mean" in ds.data_vars
    if not has_ws_max:
        print(f"  WARNING: ws_max not found in input — will compute from u10/v10 (instantaneous, less accurate)")
    if not has_ws_mean:
        print(f"  WARNING: ws_mean not found in input — will compute from u10/v10 (instantaneous)")

    LAT   = ds["latitude"].values
    LON   = ds["longitude"].values
    TIMES = pd.to_datetime(ds["time"].values)
    NT    = len(TIMES)

    # Build NDVI time index mapping now that TIMES is available
    if use_ndvi:
        bt_times = TIMES.values.astype("datetime64[ns]")
        ndvi_ti_map = np.searchsorted(ndvi_times, bt_times, side="right") - 1
        ndvi_ti_map = np.clip(ndvi_ti_map, 0, len(ndvi_times) - 1)
        # For each bt timestep, pick the closer NDVI timestep
        for i in range(len(bt_times)):
            idx = ndvi_ti_map[i]
            if idx + 1 < len(ndvi_times):
                if abs(ndvi_times[idx + 1] - bt_times[i]) < abs(bt_times[i] - ndvi_times[idx]):
                    ndvi_ti_map[i] = idx + 1
        # Check if spatial remap is needed
        ndvi_needs_remap = (len(ndvi_lat) != len(LAT) or len(ndvi_lon) != len(LON) or
                           not (np.allclose(ndvi_lat, LAT, atol=0.01) and
                                np.allclose(ndvi_lon, LON, atol=0.01)))
        if ndvi_needs_remap:
            ndvi_yi = np.array([_nearest_idx(ndvi_lat, la) for la in LAT])
            ndvi_xi = np.array([_nearest_idx(ndvi_lon, lo) for lo in LON])
        else:
            ndvi_yi = ndvi_xi = None

    land_mask = load_land_mask(LAT, LON)
    print(f"  Grid: {len(LAT)}x{len(LON)}, {NT} timesteps, "
          f"{tracks['cell'].nunique()} tracks")
    print(f"  Land cells: {land_mask.sum():,}/{land_mask.size:,}")

    # Index genesis birth masks
    print("  Indexing genesis birth masks ...")
    genesis_birth = {}
    for ci, cid in enumerate(sm_cids):
        cid = int(cid)
        t0  = pd.Timestamp(str(sm_times[ci]))
        ti  = int(np.argmin(np.abs(TIMES - t0)))
        genesis_birth[cid] = (ti, src_masks_xr["source_mask"][ci].values.astype(bool))
    src_masks_xr.close()
    print(f"    {len(genesis_birth)} genesis regions")

    # ---- Helper to load wind speed fields ----
    def _load_wind_fields(ti):
        """Return (ws_max_arr, ws_mean_arr) for timestep ti."""
        if has_ws_max:
            wmax = ds["ws_max"].isel(time=ti).values.astype("float32")
        else:
            u = ds["u10"].isel(time=ti).values.astype("float32")
            v = ds["v10"].isel(time=ti).values.astype("float32")
            wmax = np.hypot(u, v)  # fallback: instantaneous speed

        if has_ws_mean:
            wmean = ds["ws_mean"].isel(time=ti).values.astype("float32")
        else:
            u = ds["u10"].isel(time=ti).values.astype("float32")
            v = ds["v10"].isel(time=ti).values.astype("float32")
            wmean = np.hypot(u, v)  # fallback: instantaneous speed

        return wmax, wmean

    # ---- Create output NetCDFs ----
    times_f64 = nc4.date2num(TIMES.to_pydatetime(),
                              units="hours since 1900-01-01", calendar="standard")

    def _init_nc(path, var_name, desc):
        ncf = nc4.Dataset(path, "w")
        ncf.createDimension("time", NT)
        ncf.createDimension("latitude", len(LAT))
        ncf.createDimension("longitude", len(LON))
        tv = ncf.createVariable("time", "f8", ("time",))
        tv.units = "hours since 1900-01-01"; tv.calendar = "standard"; tv[:] = times_f64
        ncf.createVariable("latitude", "f4", ("latitude",))[:] = LAT
        ncf.createVariable("longitude", "f4", ("longitude",))[:] = LON
        v = ncf.createVariable(var_name, "u1", ("time", "latitude", "longitude"),
                               zlib=True, complevel=4,
                               chunksizes=(1, len(LAT), len(LON)))
        v.description = desc
        z = np.zeros((len(LAT), len(LON)), dtype=np.uint8)
        for _ti in range(NT):
            v[_ti] = z
        return ncf, v

    ncf_i, v_i = _init_nc(out_dir / "sds_intermediate_sources.nc",
                           "intermediate_source",
                           "1=intermediate source pixel")
    ncf_r, v_r = _init_nc(out_dir / "sds_sources_raw.nc",
                           "SDS_SOURCE",
                           "1=genesis pixel (unfiltered) OR intermediate")
    ncf_f, v_f = _init_nc(out_dir / "sds_sources_filtered.nc",
                           "SDS_SOURCE",
                           "1=genesis pixel (filtered) OR intermediate")

    # Add source_type variable to filtered file: 0=none, 1=start, 2=intermediate
    v_ft = ncf_f.createVariable("source_type", "u1", ("time", "latitude", "longitude"),
                                zlib=True, complevel=4,
                                chunksizes=(1, len(LAT), len(LON)))
    v_ft.description = "Source type: 0=none, 1=start (plume origin), 2=intermediate"
    z = np.zeros((len(LAT), len(LON)), dtype=np.uint8)
    for _ti in range(NT):
        v_ft[_ti] = z

    # ---- Paint genesis pixels ----
    print("\nPainting genesis pixels ...")
    ti_to_genesis = defaultdict(list)
    for cid, (ti, bmask) in genesis_birth.items():
        ti_to_genesis[ti].append((cid, bmask))

    for ti, entries in tqdm(sorted(ti_to_genesis.items()), desc="  Genesis dates"):
        # Raw: all genesis pixels
        fr = np.array(v_r[ti], dtype=np.uint8)
        for cid, bmask in entries:
            fr[bmask] = 1
        v_r[ti] = fr

        # Filtered: genesis pixels passing surface filters
        ws_max_b, ws_mean_b = _load_wind_fields(ti)

        # Compute AE for ae_px_thr filter on genesis pixels
        ae_b = _compute_ae(
            ds["aod469"].isel(time=ti).values.astype("float32"),
            ds["aod865"].isel(time=ti).values.astype("float32"),
        )

        if use_ndvi:
            ndvi_b, snow_b = _load_ndvi_fast(ti, ndvi_data, snow_data, ndvi_ti_map, ndvi_yi, ndvi_xi)
        else:
            ndvi_b = np.full((len(LAT), len(LON)), np.nan, dtype="float32")
            snow_b = np.full((len(LAT), len(LON)), np.nan, dtype="float32")

        ok_ndvi = np.isnan(ndvi_b) | (ndvi_b < ndvi_thr)
        ok_snow = np.isnan(snow_b) | (snow_b < snow_thr)
        ok_ae   = np.isnan(ae_b) | (ae_b <= ae_px_thr)

        if wind_ndvi_dep:
            ws_mean_thr_arr = _compute_wind_thr_ramp(ndvi_b, ws_mean_base, ws_mean_thr, ndvi_thr)
            ws_max_thr_arr  = _compute_wind_thr_ramp(ndvi_b, ws_max_base, ws_max_thr, ndvi_thr)
            ok = ((ws_mean_b >= ws_mean_thr_arr) &
                  (ws_max_b >= ws_max_thr_arr) &
                  ok_ae & ok_ndvi & ok_snow & land_mask)
        else:
            ok = ((ws_mean_b >= ws_mean_thr) &
                  (ws_max_b >= ws_max_thr) &
                  ok_ae & ok_ndvi & ok_snow & land_mask)

        ff = np.array(v_f[ti], dtype=np.uint8)
        ft = np.array(v_ft[ti], dtype=np.uint8)
        for cid, bmask in entries:
            passed = bmask & ok
            ff[passed] = 1
            ft[passed] = 1  # 1 = start
        v_f[ti] = ff
        v_ft[ti] = ft

    # ---- Intermediate detection loop ----
    source_rows = []
    filter_log  = []
    bufs = {v_i: {"ti": -1, "data": None},
            v_r: {"ti": -1, "data": None},
            v_f: {"ti": -1, "data": None},
            v_ft: {"ti": -1, "data": None}}

    def _flush(var):
        b = bufs[var]
        if b["ti"] >= 0 and b["data"] is not None:
            var[b["ti"]] = b["data"].astype(np.uint8)

    def _mark(var, ti, mask):
        b = bufs[var]
        if b["ti"] != ti:
            _flush(var)
            b["ti"] = ti
            b["data"] = np.array(var[ti], dtype=np.uint8)
        b["data"][mask] = 1

    def _mark_type(ti, mask, value):
        """Mark source_type variable with a specific value (1=start, 2=intermediate)."""
        b = bufs[v_ft]
        if b["ti"] != ti:
            _flush(v_ft)
            b["ti"] = ti
            b["data"] = np.array(v_ft[ti], dtype=np.uint8)
        b["data"][mask] = value

    seg_times    = pd.to_datetime(seg_ds["time"].values)
    cell_times   = tracks[["cell", "time"]].sort_values("time").drop_duplicates()
    unique_times = cell_times["time"].unique()
    print(f"\nProcessing {len(unique_times)} timesteps ...")

    # Pixel funnel accumulators
    stage1 = dict(
        sds_px_considered=0,
        cell_volume_filter=0,
        cell_no_t1_plume=0,
        plume_ae_precheck_skip=0,
        px_nonfinite=0,
        px_fail_duaod_only=0,
        px_fail_ae_only=0,
        px_fail_both=0,
        px_passed_dust_signal=0,
        px_no_upwind=0,
        px_fallback=0,
        # Combined-specific counters
        px_fail_ae_upwind_only=0,
        px_fail_ae_plume_only=0,
        px_fail_ae_both_refs=0,
        px_fail_ae_and_duaod=0,
    )

    for t_now in tqdm(unique_times):
        ti_input = int(np.argmin(np.abs(TIMES - t_now)))
        if ti_input == 0:
            continue

        t_prev   = TIMES[ti_input - 1]
        ti_sn    = int(np.argmin(np.abs(seg_times - t_now)))
        ti_sp    = int(np.argmin(np.abs(seg_times - t_prev)))
        seg_now  = seg_ds[seg_var].isel(time=ti_sn).values
        seg_prev = seg_ds[seg_var].isel(time=ti_sp).values
        duaod_t  = ds["duaod550"].isel(time=ti_input).values.astype("float32")
        duaod_t1 = ds["duaod550"].isel(time=ti_input - 1).values.astype("float32")
        ae_t     = _compute_ae(
            ds["aod469"].isel(time=ti_input).values.astype("float32"),
            ds["aod865"].isel(time=ti_input).values.astype("float32"),
        )
        ae_t1 = _compute_ae(
            ds["aod469"].isel(time=ti_input - 1).values.astype("float32"),
            ds["aod865"].isel(time=ti_input - 1).values.astype("float32"),
        )

        ws_max_t, ws_mean_t = _load_wind_fields(ti_input)

        # Load u10/v10 for upwind / combined / upwind_plume detection
        if detection_method in ("upwind", "combined", "upwind_plume"):
            u10_t = ds["u10"].isel(time=ti_input).values.astype("float32")
            v10_t = ds["v10"].isel(time=ti_input).values.astype("float32")

        if use_ndvi:
            ndvi, snow = _load_ndvi_fast(ti_input, ndvi_data, snow_data, ndvi_ti_map, ndvi_yi, ndvi_xi)
        else:
            ndvi = np.full((len(LAT), len(LON)), np.nan, dtype="float32")
            snow = np.full((len(LAT), len(LON)), np.nan, dtype="float32")

        t_prev_pd = pd.Timestamp(t_prev)
        t_now_pd  = pd.Timestamp(t_now)
        active_cells = cell_times[cell_times["time"] == t_now]["cell"].values

        # ── Volume pre-filter ──────────────────────────────────────────
        volume_skipped = set()
        involved_now = involved_cells.get(t_now_pd, set())

        if not no_volume_filter:
            pred_to_succ = defaultdict(set)
            succ_to_pred = defaultdict(set)

            for evt in split_events.get(t_now_pd, []):
                parent = evt["parent"]
                for child in evt["children"]:
                    pred_to_succ[parent].add(child)
                    succ_to_pred[child].add(parent)

            for evt in merge_events.get(t_now_pd, []):
                dominant = evt["dominant"]
                for parent in evt["parents"]:
                    pred_to_succ[parent].add(dominant)
                    succ_to_pred[dominant].add(parent)

            all_preds = set(pred_to_succ.keys())
            all_succs = set(succ_to_pred.keys())
            visited_preds = set()
            visited_succs = set()
            components = []

            for seed in all_preds | all_succs:
                if seed in visited_preds or seed in visited_succs:
                    continue
                comp_preds = set()
                comp_succs = set()
                queue = [seed]
                while queue:
                    node = queue.pop()
                    if node in all_preds and node not in comp_preds:
                        comp_preds.add(node)
                        visited_preds.add(node)
                        for s in pred_to_succ[node]:
                            if s not in comp_succs:
                                queue.append(s)
                    if node in all_succs and node not in comp_succs:
                        comp_succs.add(node)
                        visited_succs.add(node)
                        for p in succ_to_pred[node]:
                            if p not in comp_preds:
                                queue.append(p)
                if comp_preds and comp_succs:
                    components.append((comp_preds, comp_succs))

            for pred_set, succ_set in components:
                vol_prev = dust_volume(pred_set, seg_prev, duaod_t1)
                vol_now  = dust_volume(succ_set, seg_now, duaod_t)
                if vol_prev > 0 and (vol_now <= vol_prev * (1.0 + volume_growth)) and (vol_now - vol_prev <= volume_abs):
                    volume_skipped.update(succ_set)

            for cell_id in active_cells:
                cid = int(cell_id)
                if cid in involved_now or cid in volume_skipped:
                    continue
                vol_prev = dust_volume({cid}, seg_prev, duaod_t1)
                vol_now  = dust_volume({cid}, seg_now, duaod_t)
                if vol_prev > 0 and (vol_now <= vol_prev * (1.0 + volume_growth)) and (vol_now - vol_prev <= volume_abs):
                    volume_skipped.add(cid)

        # ── Per-cell detection ─────────────────────────────────────────
        for cell_id in active_cells:
            cell_id = int(cell_id)
            plume_t_mask = (seg_now == cell_id)
            if not plume_t_mask.any():
                continue

            n_plume_px = int(plume_t_mask.sum())
            stage1["sds_px_considered"] += n_plume_px

            if cell_id in volume_skipped:
                stage1["cell_volume_filter"] += n_plume_px
                continue

            pred_cells = predecessor_lookup.get((cell_id, t_now_pd), {cell_id})
            plume_t1_mask = np.zeros(seg_prev.shape, dtype=bool)
            for pc in pred_cells:
                plume_t1_mask |= (seg_prev == pc)
            if not plume_t1_mask.any():
                plume_t1_mask = (seg_prev == cell_id)
            if not plume_t1_mask.any():
                stage1["cell_no_t1_plume"] += n_plume_px
                continue

            plume_ref_duaod = _compute_plume_ref(duaod_t1[plume_t1_mask], ref_stat_duaod)
            plume_ref_ae    = _compute_plume_ref(ae_t1[plume_t1_mask], ref_stat_ae)

            # ── Plume-level AE precheck ────────────────────────────────
            if plume_ae_precheck:
                ae_now_vals  = ae_t[plume_t_mask]
                ae_prev_vals = ae_t1[plume_t1_mask]
                ae_now_fin  = ae_now_vals[np.isfinite(ae_now_vals)]
                ae_prev_fin = ae_prev_vals[np.isfinite(ae_prev_vals)]
                if len(ae_now_fin) >= 5 and len(ae_prev_fin) >= 5:
                    ae_pct_now  = float(np.percentile(ae_now_fin, plume_ae_percentile))
                    ae_pct_prev = float(np.percentile(ae_prev_fin, plume_ae_percentile))
                    if not (ae_pct_now < ae_pct_prev - plume_ae_delta):
                        stage1["plume_ae_precheck_skip"] += n_plume_px
                        continue

            # ── Dispatch to detection method ───────────────────────────
            if detection_method == "combined":
                cands, det_counts = detect_intermediates_combined(
                    plume_t_mask, duaod_t, ae_t, duaod_t1, ae_t1,
                    u10_t, v10_t, dt_hours, grid_res, LAT,
                    plume_ref_duaod, plume_ref_ae,
                    delta_duaod, delta_ae,
                    neighbourhood=upwind_nb, ref_stat=ref_stat_duaod,
                    no_duaod_check=no_duaod_check,
                )
            elif detection_method == "upwind_plume":
                cands, det_counts = detect_intermediates_upwind_plume(
                    plume_t_mask, plume_t1_mask, duaod_t, ae_t, duaod_t1, ae_t1,
                    u10_t, v10_t, dt_hours, grid_res, LAT, LON,
                    delta_duaod, delta_ae,
                    k_neighbours=upwind_plume_k, ref_stat=ref_stat_duaod,
                    plume_ref_duaod=plume_ref_duaod, plume_ref_ae=plume_ref_ae,
                    no_duaod_check=no_duaod_check,
                )
            elif detection_method == "upwind":
                cands, det_counts = detect_intermediates_upwind(
                    plume_t_mask, duaod_t, ae_t, duaod_t1, ae_t1,
                    u10_t, v10_t, dt_hours, grid_res, LAT,
                    delta_duaod, delta_ae,
                    neighbourhood=upwind_nb, ref_stat=ref_stat_duaod,
                    plume_ref_duaod=plume_ref_duaod, plume_ref_ae=plume_ref_ae,
                    no_duaod_check=no_duaod_check,
                )
            else:
                cands, det_counts = detect_intermediates_plume_ref(
                    plume_t_mask, duaod_t, ae_t,
                    plume_ref_duaod, plume_ref_ae,
                    delta_duaod, delta_ae,
                    no_duaod_check=no_duaod_check,
                )

            stage1["px_nonfinite"]          += det_counts["px_nonfinite"]
            stage1["px_passed_dust_signal"] += det_counts["px_passed"]
            stage1["px_no_upwind"]          += det_counts.get("px_no_upwind", 0)
            stage1["px_fallback"]           += det_counts.get("px_fallback", 0)

            # Accumulate failure counts — method-dependent keys
            if detection_method == "combined":
                stage1["px_fail_duaod_only"]     += det_counts.get("px_fail_duaod", 0)
                stage1["px_fail_ae_upwind_only"] += det_counts.get("px_fail_ae_upwind", 0)
                stage1["px_fail_ae_plume_only"]  += det_counts.get("px_fail_ae_plume", 0)
                stage1["px_fail_ae_both_refs"]   += det_counts.get("px_fail_ae_both", 0)
                stage1["px_fail_ae_and_duaod"]   += det_counts.get("px_fail_ae_and_duaod", 0)
            else:
                stage1["px_fail_duaod_only"]  += det_counts.get("px_fail_duaod", 0)
                stage1["px_fail_ae_only"]     += det_counts.get("px_fail_ae", 0)
                stage1["px_fail_both"]        += det_counts.get("px_fail_both", 0)

            if not cands.any():
                continue

            # Emission filter (ws, AE pixel, NDVI, snow, land)
            interm, fstats = apply_emission_filter(
                cands, ws_max_t, ws_mean_t, ae_t,
                ndvi, snow, land_mask,
                ws_mean_thr, ws_max_thr, ae_px_thr, ndvi_thr, snow_thr,
                wind_ndvi_dep=wind_ndvi_dep,
                ws_mean_base=ws_mean_base, ws_max_base=ws_max_base,
            )
            filter_log.append({"cell_id": cell_id, "time": t_now, **fstats})
            if not interm.any():
                continue

            for var in (v_i, v_r, v_f):
                _mark(var, ti_input, interm)
            _mark_type(ti_input, interm, 2)  # 2 = intermediate

            # Cluster and record
            struct = ndimage.generate_binary_structure(2, 2)
            labeled, n_cl = ndimage.label(interm, structure=struct)
            for clu in range(1, n_cl + 1):
                cl = labeled == clu
                sz = int(cl.sum())
                if sz < min_cluster:
                    continue
                ys_c, xs_c = np.where(cl)
                source_rows.append({
                    "cell_id":    cell_id,
                    "time":       t_now,
                    "lat":        float(LAT[ys_c].mean()),
                    "lon":        float(LON[xs_c].mean()),
                    "n_pixels":   sz,
                    "mean_ae":    float(np.nanmean(ae_t[cl])),
                    "mean_duaod": float(np.nanmean(duaod_t[cl])),
                    "mean_ws_max":  float(np.nanmean(ws_max_t[cl])),
                    "mean_ws_mean": float(np.nanmean(ws_mean_t[cl])),
                    "plume_ref_duaod_t1": plume_ref_duaod,
                    "plume_ref_ae_t1":    plume_ref_ae,
                    "ref_stat_ae":           ref_stat_ae,
                    "ref_stat_duaod":        ref_stat_duaod,
                })

    # ---- Flush and save ----
    print("\nSaving ...")
    for var in (v_i, v_r, v_f, v_ft):
        _flush(var)

    v_i.ws_mean_thr     = ws_mean_thr
    v_i.ws_max_thr      = ws_max_thr
    v_i.ae_px_thr       = ae_px_thr
    v_i.wind_ndvi_dep   = str(wind_ndvi_dep)
    if wind_ndvi_dep:
        v_i.ws_mean_base = ws_mean_base
        v_i.ws_max_base  = ws_max_base
    v_i.ndvi_thr        = ndvi_thr
    v_i.snow_frac_thr   = snow_thr
    v_i.delta_duaod_thr = delta_duaod
    v_i.delta_ae_thr    = delta_ae
    v_i.volume_growth   = volume_growth
    v_i.dt_hours        = dt_hours
    v_i.detection_method = detection_method
    if detection_method in ("upwind", "combined"):
        v_i.upwind_neighbourhood = upwind_nb
        v_i.upwind_dt_hours = dt_hours
    if detection_method == "upwind_plume":
        v_i.upwind_plume_k = upwind_plume_k
        v_i.upwind_dt_hours = dt_hours
    v_i.ref_stat_ae        = ref_stat_ae
    v_i.ref_stat_duaod     = ref_stat_duaod

    ncf_i.close(); ncf_r.close(); ncf_f.close()

    pd.DataFrame(source_rows).to_csv(
        out_dir / "sds_intermediate_sources.csv", index=False
    )
    pd.DataFrame(filter_log).to_csv(
        out_dir / "sds_intermediate_filter_log.csv", index=False
    )

    # ---- Combined sources with source_type label ----
    # Origin sources from Step 1
    s1_sources_csv = s1_dir / "sds_sources.csv"
    if s1_sources_csv.exists():
        origins = pd.read_csv(s1_sources_csv)
        origins["source_type"] = "origin"
        origins = origins.rename(columns={
            "source_time": "time",
            "source_lat":  "lat",
            "source_lon":  "lon",
            "cell":        "cell_id",
        })
        origins["time"] = pd.to_datetime(origins["time"])
    else:
        print(f"  WARNING: {s1_sources_csv.name} not found — combined CSV will have intermediates only")
        origins = pd.DataFrame()

    # Intermediate sources
    if source_rows:
        intermediates = pd.DataFrame(source_rows)
        intermediates["source_type"] = "intermediate"
    else:
        intermediates = pd.DataFrame()

    # Merge
    all_sources = pd.concat([origins, intermediates], ignore_index=True)
    if len(all_sources):
        all_sources = all_sources.sort_values("time").reset_index(drop=True)
    all_sources.to_csv(out_dir / "sds_all_sources.csv", index=False)

    n_origin = int((all_sources["source_type"] == "origin").sum()) if len(all_sources) else 0
    n_interm = int((all_sources["source_type"] == "intermediate").sum()) if len(all_sources) else 0
    print(f"  Combined sources: {len(all_sources):,} "
          f"({n_origin:,} origin, {n_interm:,} intermediate)")
    print(f"  → sds_all_sources.csv")

    pixel_log_rows = [
        {"filter": k, "n_pixels": v} for k, v in stage1.items()
    ]
    pd.DataFrame(pixel_log_rows).to_csv(
        out_dir / "sds_pixel_filter_log.csv", index=False
    )

    # ---- Print summary ----
    def _pct(a, b):
        return f"{100*a/b:.1f}%" if b > 0 else "n/a"

    tot = stage1["sds_px_considered"]
    print(f"\n── Pixel funnel ──")
    print(f"  SDS+ pixels considered           : {tot:>12,}")
    print(f"  Volume filter (no growth)        : {stage1['cell_volume_filter']:>12,}  ({_pct(stage1['cell_volume_filter'], tot)})")
    print(f"  No t-1 plume                     : {stage1['cell_no_t1_plume']:>12,}  ({_pct(stage1['cell_no_t1_plume'], tot)})")
    if stage1["plume_ae_precheck_skip"] > 0:
        print(f"  Plume AE precheck (no AE drop)   : {stage1['plume_ae_precheck_skip']:>12,}  ({_pct(stage1['plume_ae_precheck_skip'], tot)})")
    print(f"  Non-finite AE/DUAOD              : {stage1['px_nonfinite']:>12,}  ({_pct(stage1['px_nonfinite'], tot)})")

    if detection_method == "combined":
        print(f"  Fail DUAOD only                  : {stage1['px_fail_duaod_only']:>12,}  ({_pct(stage1['px_fail_duaod_only'], tot)})")
        print(f"  Fail AE upwind only              : {stage1['px_fail_ae_upwind_only']:>12,}  ({_pct(stage1['px_fail_ae_upwind_only'], tot)})")
        print(f"  Fail AE plume-ref only           : {stage1['px_fail_ae_plume_only']:>12,}  ({_pct(stage1['px_fail_ae_plume_only'], tot)})")
        print(f"  Fail AE both refs                : {stage1['px_fail_ae_both_refs']:>12,}  ({_pct(stage1['px_fail_ae_both_refs'], tot)})")
        print(f"  Fail AE + DUAOD                  : {stage1['px_fail_ae_and_duaod']:>12,}  ({_pct(stage1['px_fail_ae_and_duaod'], tot)})")
    else:
        print(f"  Fail DUAOD only (AE passed)      : {stage1['px_fail_duaod_only']:>12,}  ({_pct(stage1['px_fail_duaod_only'], tot)})")
        print(f"  Fail AE only (DUAOD passed)      : {stage1['px_fail_ae_only']:>12,}  ({_pct(stage1['px_fail_ae_only'], tot)})")
        print(f"  Fail both DUAOD + AE             : {stage1['px_fail_both']:>12,}  ({_pct(stage1['px_fail_both'], tot)})")

    if stage1["px_no_upwind"] > 0:
        print(f"  Upwind out-of-bounds             : {stage1['px_no_upwind']:>12,}  ({_pct(stage1['px_no_upwind'], tot)})")
    if stage1["px_fallback"] > 0:
        print(f"  Upwind fallback to plume-ref      : {stage1['px_fallback']:>12,}  ({_pct(stage1['px_fallback'], tot)})")
    print(f"  Passed dust signal               : {stage1['px_passed_dust_signal']:>12,}  ({_pct(stage1['px_passed_dust_signal'], tot)})")

    if filter_log:
        df_fl = pd.DataFrame(filter_log)
        tc = int(df_fl["n_candidates"].sum())
        tk = int(df_fl["n_kept"].sum())
        print(f"\n── Emission filter (of {tc:,} dust-signal candidates) ──")
        print(f"  Kept                             : {tk:>12,}  ({_pct(tk, tc)})")
        print(f"  Failed ws_mean (< {ws_mean_thr})       : {int(df_fl['n_failed_ws_mean'].sum()):>12,}  ({_pct(int(df_fl['n_failed_ws_mean'].sum()), tc)})")
        print(f"  Failed ws_max  (< {ws_max_thr})        : {int(df_fl['n_failed_ws_max'].sum()):>12,}  ({_pct(int(df_fl['n_failed_ws_max'].sum()), tc)})")
        print(f"  Failed AE pixel (> {ae_px_thr})       : {int(df_fl['n_failed_ae_px'].sum()):>12,}  ({_pct(int(df_fl['n_failed_ae_px'].sum()), tc)})")
        print(f"  Failed NDVI (>= {ndvi_thr})          : {int(df_fl['n_failed_ndvi'].sum()):>12,}  ({_pct(int(df_fl['n_failed_ndvi'].sum()), tc)})")
        print(f"  Failed snow (>= {snow_thr})          : {int(df_fl['n_failed_snow'].sum()):>12,}  ({_pct(int(df_fl['n_failed_snow'].sum()), tc)})")
        print(f"  Failed land                      : {int(df_fl['n_failed_land'].sum()):>12,}  ({_pct(int(df_fl['n_failed_land'].sum()), tc)})")

    print(f"\n  Intermediate clusters: {len(source_rows):,}")
    print(f"  Intermediate pixels:  {sum(r['n_pixels'] for r in source_rows):,}")

    wall_elapsed = _time.time() - wall_start
    print(f"\nRun complete in {wall_elapsed / 60:.1f} minutes")

    # ---- Experiment summary ----
    emf_candidates = emf_kept = 0
    emf_failed_ws_mean = emf_failed_ws_max = emf_failed_ae_px = 0
    emf_failed_ndvi = emf_failed_snow = emf_failed_land = 0
    if filter_log:
        df_fl = pd.DataFrame(filter_log)
        emf_candidates    = int(df_fl["n_candidates"].sum())
        emf_kept          = int(df_fl["n_kept"].sum())
        emf_failed_ws_mean = int(df_fl["n_failed_ws_mean"].sum())
        emf_failed_ws_max  = int(df_fl["n_failed_ws_max"].sum())
        emf_failed_ae_px   = int(df_fl["n_failed_ae_px"].sum())
        emf_failed_ndvi   = int(df_fl["n_failed_ndvi"].sum())
        emf_failed_snow   = int(df_fl["n_failed_snow"].sum())
        emf_failed_land   = int(df_fl["n_failed_land"].sum())

    experiment_row = {
        "run_timestamp": pd.Timestamp.now().isoformat(),
        "resolution": resolution, "grid_res": grid_res,
        "s1_folder": s1_folder, "s2_folder": s2_folder,
        # Step 2 parameters
        "ws_mean_thr": ws_mean_thr, "ws_max_thr": ws_max_thr,
        "ae_px_thr": ae_px_thr,
        "wind_ndvi_dep": wind_ndvi_dep,
        "ws_mean_base": ws_mean_base if wind_ndvi_dep else np.nan,
        "ws_max_base": ws_max_base if wind_ndvi_dep else np.nan,
        "ndvi_thr": ndvi_thr,
        "snow_thr": snow_thr, "delta_duaod": delta_duaod,
        "delta_ae": delta_ae, "volume_growth": volume_growth, "volume_abs": volume_abs,
        "min_cluster": min_cluster, "ref_stat_ae": ref_stat_ae, "ref_stat_duaod": ref_stat_duaod,
        "detection_method": detection_method, "upwind_nb": upwind_nb if detection_method in ("upwind", "combined") else np.nan,
        "upwind_plume_k": upwind_plume_k if detection_method == "upwind_plume" else np.nan,
        "no_duaod_check": no_duaod_check, "no_volume_filter": no_volume_filter,
        "plume_ae_precheck": plume_ae_precheck,
        "plume_ae_percentile": plume_ae_percentile if plume_ae_precheck else np.nan,
        "plume_ae_delta": plume_ae_delta if plume_ae_precheck else np.nan,
        # Pixel funnel — stage 1 (pre-dust-signal)
        "sds_px_considered":    stage1["sds_px_considered"],
        "px_volume_filter":     stage1["cell_volume_filter"],
        "px_no_t1_plume":       stage1["cell_no_t1_plume"],
        "px_plume_ae_precheck": stage1["plume_ae_precheck_skip"],
        "px_nonfinite":         stage1["px_nonfinite"],
        # Pixel funnel — dust signal check
        "px_fail_duaod_only":   stage1["px_fail_duaod_only"],
        "px_fail_ae_only":      stage1["px_fail_ae_only"],
        "px_fail_both":         stage1["px_fail_both"],
        "px_passed_dust_signal": stage1["px_passed_dust_signal"],
        # Combined-specific counters
        "px_fail_ae_upwind_only": stage1["px_fail_ae_upwind_only"],
        "px_fail_ae_plume_only":  stage1["px_fail_ae_plume_only"],
        "px_fail_ae_both_refs":   stage1["px_fail_ae_both_refs"],
        "px_fail_ae_and_duaod":   stage1["px_fail_ae_and_duaod"],
        # Pixel funnel — emission filter
        "emf_candidates":       emf_candidates,
        "emf_kept":             emf_kept,
        "emf_failed_ws_mean":   emf_failed_ws_mean,
        "emf_failed_ws_max":    emf_failed_ws_max,
        "emf_failed_ae_px":     emf_failed_ae_px,
        "emf_failed_ndvi":      emf_failed_ndvi,
        "emf_failed_snow":      emf_failed_snow,
        "emf_failed_land":      emf_failed_land,
        # Final output
        "n_origin_sources": n_origin,
        "n_intermediate_clusters": len(source_rows),
        "n_intermediate_pixels": sum(r["n_pixels"] for r in source_rows),
        "n_total_sources": n_origin + n_interm,
        "wall_time_min": round(wall_elapsed / 60, 1),
        "output_dir": str(out_dir),
    }

    exp_csv = _experiment_summary_path()
    exp_csv.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([experiment_row])
    if exp_csv.exists():
        existing = pd.read_csv(exp_csv)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(exp_csv, index=False)
    print(f"Experiment summary appended -> {exp_csv}")

    sys.stdout = logger.terminal
    logger.close()
    done_marker.write_text(pd.Timestamp.now().isoformat())

    ds.close(); seg_ds.close()
    if ndvi_ds is not None:
        try:
            ndvi_ds.close()
        except Exception:
            pass

    return experiment_row


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="SDS Intermediate Source Detection — grid search runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python sds_intermediate_detector.py --ws_max_thr 8.0 10.0 12.0 --dry_run
  python sds_intermediate_detector.py --delta_duaod 0.01 0.02 --delta_ae 0.01 0.02
  python sds_intermediate_detector.py --s1_angstr 0.6 --ws_max_thr 8.0 10.0
  python sds_intermediate_detector.py --detection_method combined --upwind_nb 1
""",
    )
    # Step 1 selection
    p.add_argument("--resolution", nargs="+", default=["daily"])
    p.add_argument("--grid_res",   nargs="+", type=float, default=[0.25])
    p.add_argument("--s1_angstr",  nargs="+", type=float, default=[0.4])
    p.add_argument("--s1_duaod",   nargs="+", type=float, default=[0.3])
    p.add_argument("--s1_pm10",    nargs="+", type=float, default=[1.5e-7])
    p.add_argument("--s1_min_px",  nargs="+", type=int,   default=[0])
    p.add_argument("--s1_iou",     nargs="+", type=float, default=[0.25])
    p.add_argument("--s1_max_gap", nargs="+", type=int,   default=[0])
    p.add_argument("--s1_stubs",   nargs="+", type=int,   default=[0])
    p.add_argument("--s1_memory",  nargs="+", type=int,   default=[0])

    # Step 2 parameters (sweepable)
    p.add_argument("--ws_mean_thr",   nargs="+", type=float, default=[5.0],
                   help="Mean wind speed threshold for sustained saltation (m/s)")
    p.add_argument("--ws_max_thr",    nargs="+", type=float, default=[7.0],
                   help="Max wind speed threshold for saltation initiation (m/s)")
    p.add_argument("--ae_px_thr",     nargs="+", type=float, default=[0.4],
                   help="Per-pixel AE threshold: pixel must have AE <= this to be a source")
    p.add_argument("--ndvi_thr",      nargs="+", type=float, default=[0.15])
    p.add_argument("--snow_thr",      nargs="+", type=float, default=[0.25])
    p.add_argument("--delta_duaod",   nargs="+", type=float, default=[0.02])
    p.add_argument("--delta_ae",      nargs="+", type=float, default=[0.02])
    p.add_argument("--volume_growth", nargs="+", type=float, default=[0.01])
    p.add_argument("--volume_abs",    nargs="+", type=float, default=[1.0],
                   help="Absolute volume growth threshold (hybrid with relative)")
    p.add_argument("--min_cluster",   nargs="+", type=int,   default=[0])
    p.add_argument("--ref_stat_ae",      nargs="+", default=["median"],
                   help="Reference statistic for ae: mean, median, p25, p50, p75")
    p.add_argument("--ref_stat_duaod",   nargs="+", default=["median"],
                   help="Reference statistic for duaod: mean, median, p25, p50, p75")

    p.add_argument("--wind_ndvi_dep", action="store_true",
                   help="Use NDVI-dependent wind thresholds (linear ramp from base to thr)")
    p.add_argument("--ws_mean_base",  nargs="+", type=float, default=[3.0],
                   help="ws_mean threshold at NDVI=0 (bare surface, only used with --wind_ndvi_dep)")
    p.add_argument("--ws_max_base",   nargs="+", type=float, default=[5.0],
                   help="ws_max threshold at NDVI=0 (bare surface, only used with --wind_ndvi_dep)")

    p.add_argument("--detection_method", nargs="+", default=["upwind_plume"],
                   choices=["plume_ref", "upwind", "combined", "upwind_plume"],
                   help="Detection method: plume_ref (plume-wide comparison), "
                        "upwind (local upwind spatial window), "
                        "combined (AE must pass BOTH upwind AND plume-wide ref), "
                        "upwind_plume (K nearest predecessor-plume pixels near upwind position)")
    p.add_argument("--upwind_nb",   nargs="+", type=int, default=[4],
                   help="Half-size of upwind neighbourhood window (1=3x3, 2=5x5; used with upwind and combined methods)")
    p.add_argument("--upwind_plume_k", nargs="+", type=int, default=[4],
                   help="Number of nearest predecessor-plume pixels for upwind_plume method (default: 4)")

    p.add_argument("--no_duaod_check", action="store_true",
                   help="Disable DUAOD increase check in dust signal detection (only AE decrease required)")
    p.add_argument("--no_volume_filter", action="store_true",
                   help="Disable volume growth pre-filter")
    p.add_argument("--plume_ae_precheck", action="store_true",
                   help="Enable plume-level AE precheck: skip cell if low-percentile AE didn't decrease")
    p.add_argument("--plume_ae_percentile", nargs="+", type=int, default=[1],
                   help="Percentile for plume AE precheck (default: 1 = p1)")
    p.add_argument("--plume_ae_delta", nargs="+", type=float, default=[0.01],
                   help="Minimum AE decrease for plume precheck (default: 0.01)")

    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def build_grid(args):
    keys = [
        "resolution", "grid_res",
        "s1_angstr", "s1_duaod", "s1_pm10", "s1_min_px",
        "s1_iou", "s1_max_gap", "s1_stubs", "s1_memory",
        "ws_mean_thr", "ws_max_thr", "ae_px_thr", "ndvi_thr", "snow_thr",
        "delta_duaod", "delta_ae", "volume_growth", "volume_abs",
        "min_cluster", "ref_stat_ae", "ref_stat_duaod",
        "wind_ndvi_dep", "ws_mean_base", "ws_max_base",
        "detection_method", "upwind_nb", "upwind_plume_k",
        "no_duaod_check", "no_volume_filter",
        "plume_ae_precheck", "plume_ae_percentile", "plume_ae_delta"
    ]
    lists = [
        args.resolution, args.grid_res,
        args.s1_angstr, args.s1_duaod, args.s1_pm10, args.s1_min_px,
        args.s1_iou, args.s1_max_gap, args.s1_stubs, args.s1_memory,
        args.ws_mean_thr, args.ws_max_thr, args.ae_px_thr, args.ndvi_thr, args.snow_thr,
        args.delta_duaod, args.delta_ae, args.volume_growth, args.volume_abs,
        args.min_cluster, args.ref_stat_ae, args.ref_stat_duaod,
        [args.wind_ndvi_dep], args.ws_mean_base, args.ws_max_base,
        args.detection_method, args.upwind_nb, args.upwind_plume_k,
        [args.no_duaod_check], [args.no_volume_filter],
        [args.plume_ae_precheck], args.plume_ae_percentile, args.plume_ae_delta
    ]
    return [dict(zip(keys, vals)) for vals in product(*lists)]


def main():
    args = parse_args()
    combos = build_grid(args)

    n = len(combos)
    print(f"Parameter grid: {n} combination{'s' if n != 1 else ''}")
    for i, c in enumerate(combos):
        print(f"  [{i+1:3d}] {c}")

    if args.dry_run:
        print("\n--dry_run set, exiting.")
        return

    for idx, combo in enumerate(combos):
        print(f"\n{'#' * 72}")
        print(f"# Experiment {idx + 1} / {n}")
        print(f"{'#' * 72}\n")

        run_intermediate_detection(
            resolution=combo["resolution"],
            grid_res=combo["grid_res"],
            s1_angstr=combo["s1_angstr"],
            s1_duaod=combo["s1_duaod"],
            s1_pm10=combo["s1_pm10"],
            s1_min_pixels=combo["s1_min_px"],
            s1_iou=combo["s1_iou"],
            s1_max_gap=combo["s1_max_gap"],
            s1_stubs=combo["s1_stubs"],
            s1_memory=combo["s1_memory"],
            ws_mean_thr=combo["ws_mean_thr"],
            ws_max_thr=combo["ws_max_thr"],
            ae_px_thr=combo["ae_px_thr"],
            ndvi_thr=combo["ndvi_thr"],
            snow_thr=combo["snow_thr"],
            delta_duaod=combo["delta_duaod"],
            delta_ae=combo["delta_ae"],
            volume_growth=combo["volume_growth"],
            volume_abs=combo["volume_abs"],
            min_cluster=combo["min_cluster"],
            ref_stat_ae=combo["ref_stat_ae"],
            ref_stat_duaod=combo["ref_stat_duaod"],
            wind_ndvi_dep=combo["wind_ndvi_dep"],
            ws_mean_base=combo["ws_mean_base"],
            ws_max_base=combo["ws_max_base"],
            detection_method=combo["detection_method"],
            upwind_nb=combo["upwind_nb"],
            upwind_plume_k=combo["upwind_plume_k"],
            no_duaod_check=combo["no_duaod_check"],
            no_volume_filter=combo["no_volume_filter"],
            plume_ae_precheck=combo["plume_ae_precheck"],
            plume_ae_percentile=combo["plume_ae_percentile"],
            plume_ae_delta=combo["plume_ae_delta"],
        )
        gc.collect()

    print(f"\nAll done. {n} experiments completed.")
    print(f"Summary: {_experiment_summary_path()}")


if __name__ == "__main__":
    main()