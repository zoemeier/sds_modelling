"""
SDS (Sand and Dust Storm) Plume Tracking — Overlap-based + Wind Advection
==========================================================================
Parameterised runner with logging and experiment summary.
All arguments accept multiple values; the script runs every combination
(cartesian product / grid search).

Usage examples:
  # Default single run (daily, 0.25°, duaod=0.2, angstr=0.8, pm10=1.5e-7)
  python sds_windshift_tracker.py

  # Sweep Angstrom threshold (3 runs)
  python sds_windshift_tracker.py --angstr 0.6 0.7 0.8

  # Full grid: 2 resolutions x 2 grid_res x 3 angstr = 12 runs
  python sds_windshift_tracker.py --resolution daily 6h --grid_res 0.25 0.75 --angstr 0.6 0.7 0.8

  # Preview combinations without running
  python sds_windshift_tracker.py --resolution daily 6h --angstr 0.6 0.7 0.8 --dry_run

Each run saves outputs to:
  DATA_PATH / backtrace_overlap_output / windshift / {resolution}_{grid_res} / {params}

A run_log.txt is written to each output folder capturing all terminal output.
An experiment_summary.csv is appended after each run with parameters and key
result metrics for easy comparison across experiments.
"""

from pathlib import Path


import argparse
import sys
import io
import tempfile
import time as _time
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc4
from scipy import ndimage
from scipy.spatial import cKDTree
from tqdm import tqdm
import gc


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH    = PROJECT_ROOT / "data"
INTERIM_PATH = DATA_PATH / "interim"

# (temporal_resolution, grid_resolution) -> (input filename, dt_hours)
RESOLUTION_MAP = {
    ("daily", 0.25): ("BACKTRACING_INPUT_2015-2024_daily_025.nc",  24),
    ("6h",    0.25): ("BACKTRACING_INPUT_2015-2024_6hourly_025.nc",      6),
    ("3h",    0.25): ("BACKTRACING_INPUT_2015-2024_3hourly_025.nc",      3),
    ("daily", 0.75): ("BACKTRACING_INPUT_2015-2024_daily_075.nc",  24),
    ("6h",    0.75): ("BACKTRACING_INPUT_2015-2024_6hourly_075.nc",      6),
    ("3h",    0.75): ("BACKTRACING_INPUT_2015-2024_3hourly_075.nc",      3),
}

EXPERIMENT_SUMMARY_CSV = DATA_PATH / "backtrace_overlap_output" / "combination_overlap" / "experiment_summary.csv"

U_WIND_VAR = "u10"
V_WIND_VAR = "v10"


# =============================================================================
# LOGGING HELPER — tee stdout to both terminal and a log file
# =============================================================================

class TeeLogger:
    """Write to both stdout and a log file simultaneously."""
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
# PARAMETER FOLDER NAME
# =============================================================================

def make_folder_name(angstr, duaod, pm10, min_pixels, iou, max_gap, stubs, memory):
    """
    Encode all tracking parameters in a dot-free folder name.

    Encoding rules:
      angstr, duaod : remove decimal point, keep leading zeros
                      0.8 -> 08, 0.25 -> 025, 0.2 -> 02
      pm10          : express as integer coefficient + 'en' + exponent
                      1.5e-7 -> 15en-8, 2e-7 -> 2en-7, 1e-8 -> 1en-8
      iou           : remove decimal point, keep leading zeros
                      0.1 -> 01, 0.01 -> 001, 0.001 -> 0001
      others        : integer as-is

    Example: '08ae_02duaod_15en-8pm10_4px_01iou_1gap_1stub_0mem'
    """
    # angstr, duaod, iou: replace "0." with "0", trim trailing zeros.
    # 0.8->"08", 0.25->"025", 0.2->"02", 0.01->"001", 0.001->"0001"
    def _dotfree(val):
        s = f"{val:.10f}".rstrip("0").rstrip(".")  # "0.25", "0.001"
        return s.replace(".", "")                    # "025",  "0001"

    ae_str    = _dotfree(angstr)
    duaod_str = _dotfree(duaod)

    # pm10: express as integer coefficient x 10^exp, e.g. 1.5e-7 = 15e-8
    exp = int(np.floor(np.log10(abs(pm10))))
    coeff = pm10 / (10 ** exp)
    # Shift until coefficient is an integer (e.g. 1.5 -> 15, exp -7 -> -8)
    while coeff != int(coeff) and abs(coeff) < 1e6:
        coeff *= 10
        exp -= 1
    pm10_str = f"{int(coeff)}en{exp}"

    # iou: same as angstr/duaod
    iou_str = _dotfree(iou)

    return (f"{ae_str}ae_{duaod_str}duaod_{pm10_str}pm10"
            f"_{min_pixels}px_{iou_str}iou_{max_gap}gap_{stubs}stub_{memory}mem")


# =============================================================================
# MAIN TRACKING FUNCTION
# =============================================================================

def run_tracking(
    resolution: str = "daily",
    grid_res: float = 0.25,
    duaod_thr: float = 0.2,
    angstr_thr: float = 0.8,
    pm10_thr: float = 1.5e-7,
    min_pixels: int = 4,
    iou_thr: float = 0.1,
    max_gap_pixels: int = 1,
    stubs: int = 1,
    memory: int = 0,
):
    """
    Run the full SDS wind-shift tracking pipeline with the given parameters.
    Returns a dict of summary metrics for the experiment summary table.
    """
    wall_start = _time.time()

    # ---- Resolve paths ----
    res_key = (resolution, grid_res)
    if res_key not in RESOLUTION_MAP:
        raise ValueError(f"Unknown resolution combo {res_key}. "
                         f"Choose from {list(RESOLUTION_MAP.keys())}")
    in_filename, dt_hours = RESOLUTION_MAP[res_key]
    in_fp = INTERIM_PATH / in_filename

    # Grid res as string for folder name: 0.25 -> "025", 0.75 -> "075"
    grid_res_str = f"{grid_res:.2f}".replace(".", "")
    folder_name = make_folder_name(angstr_thr, duaod_thr, pm10_thr, min_pixels,
                                    iou_thr, max_gap_pixels, stubs, memory)
    out_dir = (DATA_PATH / "backtrace_overlap_output" / "combination_overlap"
               / f"{resolution}_{grid_res_str}" / folder_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Check if this experiment was already run ----
    done_marker = out_dir / "_RUN_COMPLETE"
    if done_marker.exists():
        print(f"  SKIPPING (already completed): {out_dir}")
        return None

    # ---- Start logging ----
    log_path = out_dir / "run_log.txt"
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print("=" * 72)
    print("SDS Wind-Shift Plume Tracking")
    print("=" * 72)
    print(f"  Resolution   : {resolution} (dt = {dt_hours}h)")
    print(f"  Grid res     : {grid_res}°")
    print(f"  Input file   : {in_fp}")
    print(f"  Output dir   : {out_dir}")
    print(f"  DUAOD_THR    : {duaod_thr}")
    print(f"  ANGSTR_THR   : {angstr_thr}")
    print(f"  PM10_THR     : {pm10_thr:.2e}")
    print(f"  MIN_PIXELS   : {min_pixels}")
    print(f"  IOU_THR      : {iou_thr}")
    print(f"  MAX_GAP_PX   : {max_gap_pixels}")
    print(f"  STUBS        : {stubs}")
    print(f"  MEMORY       : {memory}")
    print()

    # =========================================================================
    # STEP 1 - OPEN DATASET LAZILY
    # =========================================================================
    print("Opening dataset (lazy) ...")
    ds = xr.open_dataset(in_fp)
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
        print("  Renamed 'valid_time' -> 'time'")

    LAT   = ds["latitude"].values
    LON   = ds["longitude"].values
    TIMES = ds["time"].values
    NT    = len(TIMES)
    NLAT  = len(LAT)
    NLON  = len(LON)
    print(f"  Grid: {NLAT} lat x {NLON} lon,  {NT} timesteps")
    print(f"  Approx input size per variable: "
          f"{NT * NLAT * NLON * 4 / 1e9:.1f} GB (float32)")

    LAT_DESCENDING = LAT[0] > LAT[-1]

    # ---- Local helpers (capture parameters via closure) ----

    def load_sds_mask(ti: int) -> np.ndarray:
        aod469 = ds["aod469"].isel(time=ti).values.astype("float32")
        aod865 = ds["aod865"].isel(time=ti).values.astype("float32")
        duaod  = ds["duaod550"].isel(time=ti).values.astype("float32")
        pm10   = ds["pm10"].isel(time=ti).values.astype("float32")
        with np.errstate(divide="ignore", invalid="ignore"):
            ae = -np.log(aod469 / aod865) / np.log(469.0 / 865.0)
        return (
            (duaod >= duaod_thr) &
            (ae    <= angstr_thr) &
            (pm10  >= pm10_thr) &
            np.isfinite(ae)
        )

    def blob_mean_wind(ti, mask_crop, y0, x0):
        y1 = y0 + mask_crop.shape[0]
        x1 = x0 + mask_crop.shape[1]
        u = ds[U_WIND_VAR].isel(time=ti).values[y0:y1, x0:x1].astype("float32")
        v = ds[V_WIND_VAR].isel(time=ti).values[y0:y1, x0:x1].astype("float32")
        return float(np.nanmean(u[mask_crop])), float(np.nanmean(v[mask_crop]))

    def wind_to_grid_offset(u, v, lat_center):
        dt_sec = dt_hours * 3600.0
        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = 111_320.0 * np.cos(np.radians(lat_center))
        dx_deg = (u * dt_sec) / meters_per_deg_lon
        dy_deg = (v * dt_sec) / meters_per_deg_lat
        dx_cells = dx_deg / grid_res
        if LAT_DESCENDING:
            dy_cells = -dy_deg / grid_res
        else:
            dy_cells = dy_deg / grid_res
        return dy_cells, dx_cells

    def iou_func(crop_a, crop_b):
        arr_a, y0a, x0a = crop_a
        arr_b, y0b, x0b = crop_b
        ha, wa = arr_a.shape
        hb, wb = arr_b.shape
        y0 = min(y0a, y0b); x0 = min(x0a, x0b)
        y1 = max(y0a + ha, y0b + hb); x1 = max(x0a + wa, x0b + wb)
        if y0a + ha <= y0b or y0b + hb <= y0a:
            return 0.0
        if x0a + wa <= x0b or x0b + wb <= x0a:
            return 0.0
        canvas_a = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        canvas_b = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        canvas_a[y0a - y0: y0a - y0 + ha, x0a - x0: x0a - x0 + wa] = arr_a
        canvas_b[y0b - y0: y0b - y0 + hb, x0b - x0: x0b - x0 + wb] = arr_b
        inter = int(np.logical_and(canvas_a, canvas_b).sum())
        if inter == 0:
            return 0.0
        return inter / float(np.logical_or(canvas_a, canvas_b).sum())

    def shift_crop(crop, dy_cells, dx_cells):
        arr, y0_c, x0_c = crop
        return (arr, y0_c + int(round(dy_cells)), x0_c + int(round(dx_cells)))

    def min_pixel_distance(crop_a, crop_b):
        arr_a, y0a, x0a = crop_a
        arr_b, y0b, x0b = crop_b
        ya, xa = np.where(arr_a); yb, xb = np.where(arr_b)
        ya += y0a; xa += x0a; yb += y0b; xb += x0b
        if len(ya) == 0 or len(yb) == 0:
            return np.inf
        if len(ya) * len(yb) <= 500_000:
            dy = np.abs(ya[:, None] - yb[None, :])
            dx = np.abs(xa[:, None] - xb[None, :])
            return float(np.maximum(dy, dx).min())
        else:
            tree = cKDTree(np.column_stack([yb, xb]))
            dists, _ = tree.query(np.column_stack([ya, xa]), k=1)
            return float(dists.min())

    # Match method codes for statistics
    MATCH_NONE      = 0
    MATCH_WINDSHIFT = 1
    MATCH_RAW_IOU   = 2
    MATCH_PROXIMITY = 3
    MATCH_COMBINATION = 4

    def link_score(crop_prev, crop_curr, wind_offset=None):
        """
        Return (score, method_code, pixel_distance) for linking.

        Priority order:
          1. Raw (non-shifted) IoU — if match, accept immediately
          2. Wind-shifted IoU — if match AND raw also matched, score as
             combination_overlap (higher score); otherwise wind-shift only
          3. Proximity gap-bridging — last resort
        """
        # 1. Raw overlap (no wind shift)
        v_raw = iou_func(crop_prev, crop_curr)
        raw_match = v_raw >= iou_thr

        # 2. Wind-shifted overlap
        v_shifted = 0.0
        if wind_offset is not None:
            dy, dx = wind_offset
            shifted = shift_crop(crop_curr, -dy, -dx)
            v_shifted = iou_func(crop_prev, shifted)

        wind_match = v_shifted >= iou_thr

        # Both match → combination (highest confidence)
        if raw_match and wind_match:
            return max(v_raw, v_shifted), MATCH_COMBINATION, 0.0

        # Raw only
        if raw_match:
            return v_raw, MATCH_RAW_IOU, 0.0

        # Wind-shifted only
        if wind_match:
            return v_shifted * 0.9, MATCH_WINDSHIFT, 0.0

        # 3. Proximity
        if max_gap_pixels > 0:
            d = min_pixel_distance(crop_prev, crop_curr)
            if d <= max_gap_pixels:
                return iou_thr * (1.0 - d / (max_gap_pixels + 1)), MATCH_PROXIMITY, d

        return 0.0, MATCH_NONE, 0.0

    # =========================================================================
    # STEP 2 - CONNECTED COMPONENT LABELING
    # =========================================================================
    print("\nLabeling connected components (memory-mapped) ...")
    STRUCT = ndimage.generate_binary_structure(2, 2)

    _mmap_path = out_dir / "_blob_labels.mmap"
    blob_labels = np.memmap(
        _mmap_path, dtype=np.int32, mode="w+", shape=(NT, NLAT, NLON)
    )

    blobs_per_t      = []
    all_feature_rows = []
    global_feature_id = 1
    flagged = 0

    for ti in tqdm(range(NT), desc="  Labeling"):
        binary = load_sds_mask(ti)
        if not binary.any():
            blob_labels[ti] = 0
            blobs_per_t.append([])
            continue
        flagged += 1
        labeled, n = ndimage.label(binary, structure=STRUCT)
        blobs = []
        keep  = np.zeros_like(labeled, dtype=np.int32)
        for local_id in range(1, n + 1):
            pixels = labeled == local_id
            size   = int(pixels.sum())
            if size < min_pixels:
                continue
            ys, xs = np.where(pixels)
            blob = {
                "feature"  : global_feature_id,
                "time"     : pd.Timestamp(TIMES[ti]),
                "frame"    : ti,
                "latitude" : float(LAT[ys].mean()),
                "longitude": float(LON[xs].mean()),
                "num"      : size,
                "cell"     : np.nan,
            }
            blobs.append(blob)
            keep[pixels] = global_feature_id
            all_feature_rows.append(blob)
            global_feature_id += 1
        blob_labels[ti] = keep
        blobs_per_t.append(blobs)

    blob_labels.flush()
    print(f"  Storm-flagged timesteps: {flagged} / {NT}")
    print(f"  Total blobs detected:    {sum(len(b) for b in blobs_per_t):,}")

    # =========================================================================
    # STEP 3 - OVERLAP LINKING
    # =========================================================================
    print("\nLinking blobs (shifted-IoU with wind advection) ...")

    next_cell_id    = 1
    active_tracks   = {}
    closed_tracks   = []
    feature_to_cell = {}
    merge_log       = []
    split_log       = []
    proximity_log   = []
    lineage_registry = {}
    fid_to_size = {row["feature"]: row["num"] for row in all_feature_rows}

    # Match method counters
    match_counts = {
        "combination": 0,
        "raw_iou":     0,
        "windshift":   0,
        "proximity":   0,
    }

    for ti in tqdm(range(NT), desc="  Linking"):
        if ti % 200 == 0:
            try:
                import psutil, os
                mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
            except ImportError:
                mem_mb = 0
            max_lineage = max((len(v) for v in lineage_registry.values()), default=0)
            total_lineage = sum(len(v) for v in lineage_registry.values())
            print(f"\n[ti={ti}] date={pd.Timestamp(TIMES[ti]).date()}  "
                  f"RAM={mem_mb:.0f}MB  active={len(active_tracks)}  "
                  f"lineage_entries={total_lineage:,}  max_per_cell={max_lineage}")

        current_blobs = blobs_per_t[ti]
        layer = blob_labels[ti]

        current_masks = {}
        for b in current_blobs:
            ys, xs = np.where(layer == b["feature"])
            y0, x0 = int(ys.min()), int(xs.min())
            y1, x1 = int(ys.max()) + 1, int(xs.max()) + 1
            crop = (layer[y0:y1, x0:x1] == b["feature"])
            current_masks[b["feature"]] = (crop, y0, x0)
        current_sizes = {b["feature"]: b["num"] for b in current_blobs}

        current_wind_offsets = {}
        for b in current_blobs:
            fid = b["feature"]
            crop, y0, x0 = current_masks[fid]
            try:
                u_mean, v_mean = blob_mean_wind(ti, crop, y0, x0)
                if np.isfinite(u_mean) and np.isfinite(v_mean):
                    dy, dx = wind_to_grid_offset(u_mean, v_mean, b["latitude"])
                    current_wind_offsets[fid] = (dy, dx)
            except Exception:
                pass

        track_ids = list(active_tracks.keys())
        blob_fids = [b["feature"] for b in current_blobs]

        matched_tracks = set()
        matched_blobs  = set()
        _pending_split_inheritance = []
        pending_absorption = set()   # cids flagged for absorption (deferred)

        if track_ids and blob_fids:
            iou_matrix    = np.zeros((len(track_ids), len(blob_fids)))
            method_matrix = np.zeros((len(track_ids), len(blob_fids)), dtype=np.int8)
            dist_matrix   = np.zeros((len(track_ids), len(blob_fids)), dtype=np.float32)
            for i, cid in enumerate(track_ids):
                lm = active_tracks[cid]["last_mask"]
                for j, fid in enumerate(blob_fids):
                    score, method, dist = link_score(
                        lm, current_masks[fid],
                        wind_offset=current_wind_offsets.get(fid),
                    )
                    iou_matrix[i, j]    = score
                    method_matrix[i, j] = method
                    dist_matrix[i, j]   = dist

            # Merge detection  (deferred — don't pop absorbed tracks yet)
            for j, fid in enumerate(blob_fids):
                col = iou_matrix[:, j]
                competing_i = np.where(col >= iou_thr)[0]
                if len(competing_i) < 2:
                    continue
                competing_cids = [(track_ids[i],
                                   active_tracks[track_ids[i]]["last_size"])
                                  for i in competing_i]
                competing_cids.sort(key=lambda x: x[0])
                dominant_cid, dominant_size = competing_cids[0]
                for absorbed_cid, absorbed_size in competing_cids[1:]:
                    merge_log.append({
                        "time"          : pd.Timestamp(TIMES[ti]),
                        "dominant_cell" : dominant_cid,
                        "absorbed_cell" : absorbed_cid,
                        "dominant_size" : dominant_size,
                        "absorbed_size" : absorbed_size,
                        "blob_fid"      : fid,
                    })
                    # Transfer lineage immediately
                    absorbed_lineage = lineage_registry.get(absorbed_cid, [])
                    existing_genesis = {e["genesis_cell"]
                                        for e in lineage_registry.get(dominant_cid, [])}
                    for entry in absorbed_lineage:
                        if entry["genesis_cell"] not in existing_genesis:
                            lineage_registry.setdefault(dominant_cid, []).append({
                                "genesis_cell" : entry["genesis_cell"],
                                "source_time"  : entry["source_time"],
                                "acquired_via" : "merge",
                                "acquired_at"  : pd.Timestamp(TIMES[ti]),
                            })
                            existing_genesis.add(entry["genesis_cell"])
                    # Flag for deferred closing — keep row alive for split detection
                    pending_absorption.add(absorbed_cid)

            # Split detection  (runs on ALL tracks, including pending-absorbed)
            for i, cid in enumerate(track_ids):
                row = iou_matrix[i, :]
                competing_j = np.where(row >= iou_thr)[0]
                if len(competing_j) < 2:
                    continue
                competing_fids = [(blob_fids[j], current_sizes[blob_fids[j]])
                                  for j in competing_j]
                competing_fids.sort(key=lambda x: x[1], reverse=True)
                parent_size = active_tracks[cid]["last_size"]
                for child_fid, child_size in competing_fids[1:]:
                    child_j = blob_fids.index(child_fid)
                    iou_matrix[i, child_j] = -1
                    split_log.append({
                        "time"       : pd.Timestamp(TIMES[ti]),
                        "parent_cell": cid,
                        "parent_size": parent_size,
                        "child_fid"  : child_fid,
                        "child_size" : child_size,
                    })
                    _pending_split_inheritance.append(
                        (cid, child_fid, pd.Timestamp(TIMES[ti]))
                    )

            # Now close absorbed tracks (deferred from merge detection).
            # Zero out their rows so they don't participate in greedy matching.
            for absorbed_cid in pending_absorption:
                absorbed_i = track_ids.index(absorbed_cid)
                matched_tracks.add(absorbed_i)
                iou_matrix[absorbed_i, :] = -1
                closed_tracks.append(
                    (absorbed_cid, active_tracks.pop(absorbed_cid))
                )

            # Greedy matching  (only non-absorbed active tracks)
            surviving_i = [i for i, cid in enumerate(track_ids)
                           if cid in active_tracks]
            if surviving_i and blob_fids:
                greedy_matrix  = iou_matrix[surviving_i, :]
                greedy_methods = method_matrix[surviving_i, :]
                greedy_dists   = dist_matrix[surviving_i, :]
                greedy_tids    = [track_ids[i] for i in surviving_i]
                while greedy_matrix.max() >= iou_thr:
                    i, j = np.unravel_index(
                        greedy_matrix.argmax(), greedy_matrix.shape
                    )
                    cid = greedy_tids[i]
                    fid = blob_fids[j]
                    active_tracks[cid]["feature_ids"].append(fid)
                    active_tracks[cid]["last_seen"] = ti
                    active_tracks[cid]["last_mask"] = current_masks[fid]
                    active_tracks[cid]["last_size"] = current_sizes[fid]
                    active_tracks[cid]["ghost"]     = 0
                    feature_to_cell[fid]            = cid
                    matched_tracks.add(surviving_i[i])
                    matched_blobs.add(j)
                    # Tally match method
                    m = greedy_methods[i, j]
                    if m == MATCH_COMBINATION:
                        match_counts["combination"] += 1
                    elif m == MATCH_RAW_IOU:
                        match_counts["raw_iou"] += 1
                    elif m == MATCH_WINDSHIFT:
                        match_counts["windshift"] += 1
                    elif m == MATCH_PROXIMITY:
                        match_counts["proximity"] += 1
                        proximity_log.append({
                            "time"         : pd.Timestamp(TIMES[ti]),
                            "cell"         : cid,
                            "feature"      : fid,
                            "pixel_dist"   : float(greedy_dists[i, j]),
                            "score"        : float(greedy_matrix[i, j]),
                            "blob_lat"     : float(current_blobs[j]["latitude"]),
                            "blob_lon"     : float(current_blobs[j]["longitude"]),
                            "blob_size"    : current_sizes[fid],
                        })
                    greedy_matrix[i, :]  = -1
                    greedy_matrix[:, j]  = -1
                    greedy_methods[i, :] = MATCH_NONE
                    greedy_methods[:, j] = MATCH_NONE

        # Unmatched blobs -> new tracks
        for j, blob in enumerate(current_blobs):
            if j not in matched_blobs:
                cid = next_cell_id
                next_cell_id += 1
                fid = blob["feature"]
                active_tracks[cid] = {
                    "feature_ids": [fid],
                    "last_seen"  : ti,
                    "last_mask"  : current_masks[fid],
                    "last_size"  : blob["num"],
                    "ghost"      : 0,
                }
                feature_to_cell[fid] = cid
                lineage_registry[cid] = [{
                    "genesis_cell" : cid,
                    "source_time"  : pd.Timestamp(TIMES[ti]),
                    "acquired_via" : "birth",
                    "acquired_at"  : pd.Timestamp(TIMES[ti]),
                }]

        # Resolve split lineage
        for parent_cid, child_fid, split_time in _pending_split_inheritance:
            child_cid = feature_to_cell.get(child_fid)
            if child_cid is None:
                continue
            parent_lineage = lineage_registry.get(parent_cid, [])
            lineage_registry[child_cid] = [{
                "genesis_cell" : entry["genesis_cell"],
                "source_time"  : entry["source_time"],
                "acquired_via" : "split",
                "acquired_at"  : split_time,
            } for entry in parent_lineage]

        # Ghost / close
        for cid in list(active_tracks.keys()):
            if active_tracks[cid]["last_seen"] < ti:
                active_tracks[cid]["ghost"] += 1
                if active_tracks[cid]["ghost"] > memory:
                    closed_tracks.append((cid, active_tracks.pop(cid)))

    # Close remaining
    for cid, tdata in list(active_tracks.items()):
        closed_tracks.append((cid, tdata))

    print(f"  Merge events: {len(merge_log)}")
    print(f"  Split events: {len(split_log)}")
    total_matches = sum(match_counts.values())
    print(f"  Match statistics ({total_matches:,} total links):")
    for method, count in match_counts.items():
        pct = 100 * count / total_matches if total_matches else 0
        print(f"    {method:12s}: {count:>8,}  ({pct:5.1f}%)")
    print(f"  Tracks before stub filter: {len(closed_tracks)}")

    closed_tracks = [
        (cid, td) for cid, td in closed_tracks
        if len(td["feature_ids"]) >= stubs
    ]
    print(f"  Tracks after stub filter (>= {stubs} timesteps): {len(closed_tracks)}")

    valid_cell_ids  = {cid for cid, _ in closed_tracks}
    feature_to_cell = {
        fid: cid for fid, cid in feature_to_cell.items()
        if cid in valid_cell_ids
    }

    # Lineage
    lineage_rows = []
    for cid, entries in lineage_registry.items():
        if cid not in valid_cell_ids:
            continue
        for entry in entries:
            if entry["genesis_cell"] not in valid_cell_ids:
                continue
            lineage_rows.append({
                "cell_id"      : cid,
                "genesis_cell" : entry["genesis_cell"],
                "source_time"  : entry["source_time"],
                "acquired_via" : entry["acquired_via"],
                "acquired_at"  : entry["acquired_at"],
            })
    lineage_df = pd.DataFrame(lineage_rows) if lineage_rows else pd.DataFrame(
        columns=["cell_id", "genesis_cell", "source_time", "acquired_via", "acquired_at"]
    )
    lineage_df.to_csv(out_dir / "sds_lineage.csv", index=False)
    print(f"  Lineage saved ({len(lineage_df)} rows)")

    # Merge log
    merge_df = pd.DataFrame(merge_log) if merge_log else pd.DataFrame(
        columns=["time", "dominant_cell", "absorbed_cell",
                 "dominant_size", "absorbed_size", "blob_fid"]
    )
    if len(merge_df):
        merge_df = merge_df[merge_df["dominant_cell"].isin(valid_cell_ids)]
    merge_df.to_csv(out_dir / "sds_merges.csv", index=False)
    print(f"  Merge log saved ({len(merge_df)} events)")

    # Split log
    split_df = pd.DataFrame(split_log) if split_log else pd.DataFrame(
        columns=["time", "parent_cell", "parent_size", "child_fid", "child_size"]
    )
    if len(split_df):
        split_df["child_cell"] = split_df["child_fid"].map(feature_to_cell)
        split_df = split_df.dropna(subset=["child_cell"])
        split_df["child_cell"] = split_df["child_cell"].astype(int)
        split_df = split_df[split_df["parent_cell"].isin(valid_cell_ids)]
    split_df = split_df.drop(columns=["child_fid"], errors="ignore")
    split_df.to_csv(out_dir / "sds_splits.csv", index=False)
    print(f"  Split log saved ({len(split_df)} events)")

    # Save proximity match log
    prox_df = pd.DataFrame(proximity_log) if proximity_log else pd.DataFrame(
        columns=["time", "cell", "feature", "pixel_dist", "score",
                 "blob_lat", "blob_lon", "blob_size"]
    )
    if len(prox_df):
        prox_df = prox_df[prox_df["cell"].isin(valid_cell_ids)]
    prox_df.to_csv(out_dir / "sds_proximity_matches.csv", index=False)
    print(f"  Proximity match log saved ({len(prox_df)} events)")

    # =========================================================================
    # STEP 4 - BUILD OUTPUT DATAFRAMES
    # =========================================================================
    print("\nBuilding output DataFrames ...")

    features_df = pd.DataFrame(all_feature_rows)
    features_df["cell"] = features_df["feature"].map(feature_to_cell)

    cell_start = (
        features_df.dropna(subset=["cell"])
        .groupby("cell")["time"].min()
        .rename("track_start")
    )
    features_df = features_df.join(cell_start, on="cell")
    features_df["time_cell"] = (
        features_df["time"] - features_df["track_start"]
    ).dt.total_seconds() / 3600
    features_df = features_df.drop(columns=["track_start"])

    tracks_df = (
        features_df.dropna(subset=["cell"])
        .copy()
        .rename(columns={"num": "ncells"})
    )
    tracks_df["cell"] = tracks_df["cell"].astype(int)
    tracks_df["cell_parent_track_id"] = np.nan

    n_tracks = int(tracks_df["cell"].nunique())
    print(f"  Final track count: {n_tracks:,}")

    features_df.to_hdf(out_dir / "sds_features.h5", key="table")
    tracks_df.to_hdf(out_dir / "sds_tracks.h5",     key="table")
    print(f"  Saved features -> sds_features.h5")
    print(f"  Saved tracks   -> sds_tracks.h5")

    del all_feature_rows
    gc.collect()

    # =========================================================================
    # STEP 5 - SEGMENTATION MASK
    # =========================================================================
    print("\nWriting segmentation mask (incremental) ...")

    max_fid = int(features_df["feature"].max()) if len(features_df) else 0
    fid_to_cid = np.zeros(max_fid + 1, dtype=np.int32)
    for fid, cid in feature_to_cell.items():
        if fid <= max_fid:
            fid_to_cid[fid] = int(cid)

    seg_path = out_dir / "sds_segmentation.nc"
    _times_f64 = nc4.date2num(
        pd.to_datetime(TIMES).to_pydatetime(),
        units="hours since 1900-01-01",
        calendar="standard",
    )

    with nc4.Dataset(seg_path, "w") as ncf:
        ncf.createDimension("time",      NT)
        ncf.createDimension("latitude",  NLAT)
        ncf.createDimension("longitude", NLON)

        t_var = ncf.createVariable("time", "f8", ("time",))
        t_var.units    = "hours since 1900-01-01"
        t_var.calendar = "standard"
        t_var[:]       = _times_f64

        lat_var = ncf.createVariable("latitude",  "f4", ("latitude",))
        lat_var[:] = LAT
        lon_var = ncf.createVariable("longitude", "f4", ("longitude",))
        lon_var[:] = LON

        seg_var = ncf.createVariable(
            "segmentation_mask", "i4", ("time", "latitude", "longitude"),
            zlib=True, complevel=4,
            chunksizes=(1, NLAT, NLON),
        )
        seg_var.description   = "Per-pixel cell ID from wind-advected overlap-based SDS tracking."
        seg_var.iou_threshold = iou_thr
        seg_var.min_pixels    = min_pixels
        seg_var.stubs         = stubs
        seg_var.memory        = memory
        seg_var.dt_hours      = dt_hours
        seg_var.wind_advected = "True"

        for ti in tqdm(range(NT), desc="  Writing seg"):
            layer = blob_labels[ti]
            valid = layer > 0
            remapped = np.zeros_like(layer)
            remapped[valid] = fid_to_cid[layer[valid]]
            seg_var[ti] = remapped

    print(f"  Segmentation saved -> {seg_path}")
    del blob_labels
    _mmap_path.unlink(missing_ok=True)
    gc.collect()

    # =========================================================================
    # STEP 6 - SOURCE LOCATIONS
    # =========================================================================
    print("\nExtracting source locations ...")

    sources = (
        tracks_df
        .sort_values("time")
        .groupby("cell", as_index=False)
        .first()
        [["cell", "time", "latitude", "longitude", "feature"]]
        .rename(columns={
            "time"     : "source_time",
            "latitude" : "source_lat",
            "longitude": "source_lon",
            "feature"  : "source_feature_id",
        })
        .reset_index(drop=True)
    )
    sources.to_csv(out_dir / "sds_sources.csv", index=False)
    print(f"  Saved {len(sources):,} source records -> sds_sources.csv")

    # =========================================================================
    # STEP 7 - TRACK SUMMARY
    # =========================================================================
    end = (
        tracks_df.sort_values("time")
        .groupby("cell", as_index=False).last()
        [["cell", "time", "latitude", "longitude"]]
        .rename(columns={
            "time": "end_time", "latitude": "end_lat", "longitude": "end_lon"
        })
    )
    summary = sources.merge(end, on="cell")
    summary["lifetime_h"] = (
        (summary["end_time"] - summary["source_time"]).dt.total_seconds() / 3600
    ).round(1)
    dlat  = np.radians(summary["end_lat"] - summary["source_lat"])
    dlon  = np.radians(summary["end_lon"] - summary["source_lon"])
    lat_m = np.radians((summary["end_lat"] + summary["source_lat"]) / 2)
    summary["displacement_km"] = np.sqrt(
        (dlat * 6371) ** 2 + (dlon * 6371 * np.cos(lat_m)) ** 2
    ).round(1)

    summary.to_csv(out_dir / "sds_track_summary.csv", index=False)
    print(f"\nTop 10 tracks by displacement:")
    print(
        summary.nlargest(10, "displacement_km")
               [["cell", "source_time", "source_lat", "source_lon",
                 "lifetime_h", "displacement_km"]]
               .to_string(index=False)
    )

    # Compute longest track (max number of timesteps)
    track_lengths = tracks_df.groupby("cell")["frame"].nunique()
    longest_track_steps = int(track_lengths.max()) if len(track_lengths) else 0
    longest_track_hours = longest_track_steps * dt_hours
    longest_track_cell  = int(track_lengths.idxmax()) if len(track_lengths) else -1

    # Median / mean track lifetime
    mean_lifetime_h  = float(summary["lifetime_h"].mean()) if len(summary) else 0.0
    median_lifetime_h = float(summary["lifetime_h"].median()) if len(summary) else 0.0
    max_displacement  = float(summary["displacement_km"].max()) if len(summary) else 0.0
    mean_displacement = float(summary["displacement_km"].mean()) if len(summary) else 0.0

    # =========================================================================
    # STEP 8 - SOURCE PIXEL MASKS
    # =========================================================================
    print("\nWriting source pixel masks (incremental) ...")

    first_rows = (
        tracks_df.sort_values("time")
        .groupby("cell", as_index=False)
        .first()[["cell", "time"]]
    )
    cell_ids_sorted = first_rows["cell"].astype(int).tolist()
    n_cells         = len(cell_ids_sorted)

    src_mask_path = out_dir / "sds_source_masks.nc"
    src_seg_ds    = xr.open_dataset(seg_path)
    seg_var_name  = list(src_seg_ds.data_vars)[0]

    with nc4.Dataset(src_mask_path, "w") as ncf:
        ncf.createDimension("cell",      n_cells)
        ncf.createDimension("latitude",  NLAT)
        ncf.createDimension("longitude", NLON)

        cell_var = ncf.createVariable("cell", "i4", ("cell",))
        cell_var[:] = np.array(cell_ids_sorted, dtype=np.int32)

        lat_var = ncf.createVariable("latitude",  "f4", ("latitude",))
        lat_var[:] = LAT
        lon_var = ncf.createVariable("longitude", "f4", ("longitude",))
        lon_var[:] = LON

        mask_var = ncf.createVariable(
            "source_mask", "i1", ("cell", "latitude", "longitude"),
            zlib=True, complevel=4,
            chunksizes=(1, NLAT, NLON),
        )
        mask_var.description = "Binary pixel mask of each SDS plume at its birth timestep."
        mask_var.duaod_thr  = duaod_thr
        mask_var.angstr_thr = angstr_thr
        mask_var.pm10_thr   = float(pm10_thr)
        mask_var.iou_thr    = iou_thr

        st_var = ncf.createVariable("source_time", "f8", ("cell",))
        st_var.units    = "hours since 1900-01-01"
        st_var.calendar = "standard"

        for ci, (_, row) in enumerate(
            tqdm(first_rows.iterrows(), total=n_cells, desc="  Writing masks")
        ):
            cell_id = int(row["cell"])
            t0      = row["time"]
            seg_t0  = src_seg_ds[seg_var_name].sel(time=t0).values
            bmask   = (seg_t0 == cell_id).astype(np.int8)
            mask_var[ci] = bmask
            st_var[ci]   = nc4.date2num(
                pd.Timestamp(t0).to_pydatetime(),
                units="hours since 1900-01-01",
                calendar="standard",
            )

    src_seg_ds.close()
    print(f"  Source masks saved -> {src_mask_path}")

    # =========================================================================
    # STEP 9 - GENESIS MASK
    # =========================================================================
    print("\nBuilding genesis mask (all genesis regions) ...")
    from collections import defaultdict

    inp_times_pd = pd.to_datetime(TIMES)
    all_genesis_cells = set(
        lineage_df["genesis_cell"].astype(int).tolist()
    ) if len(lineage_df) else set()

    genesis_by_ti = defaultdict(list)
    _src_nc = nc4.Dataset(src_mask_path, "r")
    _src_cell_ids = _src_nc["cell"][:].astype(int)
    _src_cell_index = {int(cid): ci for ci, cid in enumerate(_src_cell_ids)}
    _src_st_units = _src_nc["source_time"].units
    _src_st_cal   = getattr(_src_nc["source_time"], "calendar", "standard")
    _src_st_vals  = nc4.num2date(_src_nc["source_time"][:],
                                  units=_src_st_units, calendar=_src_st_cal)
    for gc_cell in all_genesis_cells:
        ci = _src_cell_index.get(gc_cell)
        if ci is None:
            continue
        t0 = pd.Timestamp(str(_src_st_vals[ci]))
        ti = int(np.argmin(np.abs(inp_times_pd - t0)))
        bmask = _src_nc["source_mask"][ci].astype(bool)
        genesis_by_ti[ti].append((gc_cell, bmask))
    _src_nc.close()

    genesis_path = out_dir / "sds_ultimate_genesis.nc"

    with nc4.Dataset(genesis_path, "w") as ncf:
        ncf.createDimension("time",      NT)
        ncf.createDimension("latitude",  NLAT)
        ncf.createDimension("longitude", NLON)

        t_var = ncf.createVariable("time", "f8", ("time",))
        t_var.units    = "hours since 1900-01-01"
        t_var.calendar = "standard"
        t_var[:]       = _times_f64

        lat_var = ncf.createVariable("latitude",  "f4", ("latitude",))
        lat_var[:] = LAT
        lon_var = ncf.createVariable("longitude", "f4", ("longitude",))
        lon_var[:] = LON

        gen_var = ncf.createVariable(
            "ultimate_genesis", "i4", ("time", "latitude", "longitude"),
            zlib=True, complevel=4,
            chunksizes=(1, NLAT, NLON),
            fill_value=0,
        )
        gen_var.description = (
            "Per-pixel genesis cell ID at each timestep."
        )
        gen_var.n_genesis_cells = len(all_genesis_cells)

        for ti in tqdm(range(NT), desc="  Writing genesis"):
            if ti in genesis_by_ti:
                entries = sorted(genesis_by_ti[ti], key=lambda x: x[0], reverse=True)
                layer = np.zeros((NLAT, NLON), dtype=np.int32)
                for gc_cell, bmask in entries:
                    layer[bmask] = gc_cell
                gen_var[ti] = layer
            else:
                gen_var[ti] = np.zeros((NLAT, NLON), dtype=np.int32)

    total_genesis_px = sum(
        int(np.array(bmask).sum())
        for entries in genesis_by_ti.values()
        for _, bmask in entries
    )
    print(f"  Genesis mask saved -> {genesis_path}")
    print(f"  Total genesis regions: {len(all_genesis_cells):,}")
    print(f"  Total genesis pixels : {total_genesis_px:,}")

    # =========================================================================
    # TIMING
    # =========================================================================
    wall_elapsed = _time.time() - wall_start
    print(f"\n{'=' * 72}")
    print(f"Run complete in {wall_elapsed / 60:.1f} minutes")
    print(f"{'=' * 72}")

    # Close dataset
    ds.close()

    # =========================================================================
    # BUILD EXPERIMENT SUMMARY ROW
    # =========================================================================
    experiment_row = {
        "run_timestamp"       : pd.Timestamp.now().isoformat(),
        "method"              : "Combination Overlap",
        "grid_res"            : grid_res,
        "resolution"          : resolution,
        "dt_hours"            : dt_hours,
        "duaod_thr"           : duaod_thr,
        "angstr_thr"          : angstr_thr,
        "pm10_thr"            : pm10_thr,
        "min_pixels"          : min_pixels,
        "iou_thr"             : iou_thr,
        "max_gap_pixels"      : max_gap_pixels,
        "stubs"               : stubs,
        "memory"              : memory,
        "n_timesteps"         : NT,
        "storm_flagged_steps" : flagged,
        "total_blobs"         : sum(len(b) for b in blobs_per_t),
        "n_tracks"            : n_tracks,
        "n_sources"           : len(sources),
        "n_merges"            : len(merge_df),
        "n_splits"            : len(split_df),
        "matches_combination" : match_counts["combination"],
        "matches_windshift"   : match_counts["windshift"],
        "matches_raw_iou"     : match_counts["raw_iou"],
        "matches_proximity"   : match_counts["proximity"],
        "n_genesis_regions"   : len(all_genesis_cells),
        "total_genesis_pixels": total_genesis_px,
        "longest_track_steps" : longest_track_steps,
        "longest_track_hours" : longest_track_hours,
        "longest_track_cell"  : longest_track_cell,
        "mean_lifetime_h"     : round(mean_lifetime_h, 1),
        "median_lifetime_h"   : round(median_lifetime_h, 1),
        "max_displacement_km" : round(max_displacement, 1),
        "mean_displacement_km": round(mean_displacement, 1),
        "wall_time_min"       : round(wall_elapsed / 60, 1),
        "output_dir"          : str(out_dir),
    }

    # Append to experiment summary CSV
    EXPERIMENT_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([experiment_row])
    if EXPERIMENT_SUMMARY_CSV.exists():
        existing = pd.read_csv(EXPERIMENT_SUMMARY_CSV)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(EXPERIMENT_SUMMARY_CSV, index=False)
    print(f"\nExperiment summary appended -> {EXPERIMENT_SUMMARY_CSV}")

    # Restore stdout
    sys.stdout = logger.terminal
    logger.close()

    # Mark experiment as complete
    done_marker.write_text(pd.Timestamp.now().isoformat())

    return experiment_row


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="SDS Wind-Shift Plume Tracking — parameterised runner. "
                    "Pass multiple values per argument to run all combinations "
                    "(grid search).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Single run with defaults
  python sds_windshift_tracker.py 

  # Sweep Angstrom threshold
  python sds_windshift_tracker.py --angstr 0.6 0.7 0.8

  # Full grid: 2 resolutions x 2 grid_res x 3 angstr = 12 runs
  python sds_windshift_tracker.py --resolution daily 6h --grid_res 0.25 0.75 --angstr 0.6 0.7 0.8

  # Dry run — just print the combinations without running
  python sds_windshift_tracker.py --angstr 0.6 0.7 0.8 --grid_res 0.25 0.75 --dry_run
""", #TODO: change filename!!!!!!!!
    )
    p.add_argument("--resolution", nargs="+", default=["daily"],
                   help="Temporal resolution(s): daily, 6h, 3h")
    p.add_argument("--grid_res",   nargs="+", type=float, default=[0.25],
                   help="Spatial grid resolution(s) in degrees: 0.25, 0.75")
    p.add_argument("--duaod",      nargs="+", type=float, default=[0.3])
    p.add_argument("--angstr",     nargs="+", type=float, default=[0.4])
    p.add_argument("--pm10",       nargs="+", type=float, default=[1.5e-7])
    p.add_argument("--min_pixels", nargs="+", type=int,   default=[0])
    p.add_argument("--iou",        nargs="+", type=float, default=[0.25])
    p.add_argument("--max_gap",    nargs="+", type=int,   default=[0])
    p.add_argument("--stubs",      nargs="+", type=int,   default=[0])
    p.add_argument("--memory",     nargs="+", type=int,   default=[0])
    p.add_argument("--dry_run",    action="store_true",
                   help="Print all parameter combinations and exit")
    return p.parse_args()


def build_grid(args):
    """Build list of parameter dicts from all combinations of CLI arguments."""
    from itertools import product

    keys = ["resolution", "grid_res", "duaod", "angstr", "pm10",
            "min_pixels", "iou", "max_gap", "stubs", "memory"]
    lists = [
        args.resolution, args.grid_res, args.duaod, args.angstr, args.pm10,
        args.min_pixels, args.iou, args.max_gap, args.stubs, args.memory,
    ]
    combos = []
    for vals in product(*lists):
        combos.append(dict(zip(keys, vals)))
    return combos


def main():
    args = parse_args()
    combos = build_grid(args)

    n = len(combos)
    print(f"Parameter grid: {n} combination{'s' if n != 1 else ''}")
    for i, c in enumerate(combos):
        print(f"  [{i+1:3d}] {c}")

    if args.dry_run:
        print("\n--dry_run set, exiting without running.")
        return

    print()
    for idx, combo in enumerate(combos):
        print(f"\n{'#' * 72}")
        print(f"# Experiment {idx + 1} / {n}")
        print(f"# {combo}")
        print(f"{'#' * 72}\n")

        # Validate resolution combo
        res_key = (combo["resolution"], combo["grid_res"])
        if res_key not in RESOLUTION_MAP:
            print(f"  ERROR: no input file for {res_key}, skipping.")
            continue

        run_tracking(
            resolution     = combo["resolution"],
            grid_res       = combo["grid_res"],
            duaod_thr      = combo["duaod"],
            angstr_thr     = combo["angstr"],
            pm10_thr       = combo["pm10"],
            min_pixels     = combo["min_pixels"],
            iou_thr        = combo["iou"],
            max_gap_pixels = combo["max_gap"],
            stubs          = combo["stubs"],
            memory         = combo["memory"],
        )
        gc.collect()

    print(f"\nAll done. {n} experiments completed.")
    print(f"Summary at: {EXPERIMENT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()