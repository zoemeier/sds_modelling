"""
File and folder names derived from the settings in config.py.

Backtracking runs encode their parameters in dot-free folder names:
  BACKTRACKING_PATH / {resolution}_{grid} / {step 1 folder} / {step 2 folder}
The model's labels (SDS_PATH) are the step 2 output of the configured run.
"""

from pathlib import Path

import numpy as np

from sds.config import (
    PROJECT_ROOT, INTERIM_PATH, BACKTRACKING_PATH, FIGURES_PATH,
    RESOLUTION, SPATIAL_RES, RESOLUTION_CONFIG, SUPPORTED_SPATIAL_RES,
    STEP1_PARAMS, STEP2_PARAMS, YEARS,
)

TEMPORAL_TAG = {k: v["tag"] for k, v in RESOLUTION_CONFIG.items()}

# period covered by the combined data files, e.g. "2015-2024"
YEARS_TAG = f"{min(YEARS)}-{max(YEARS)}"
# ACLED starts one year earlier (complete 365-day rolling sums)
ACLED_YEARS_TAG = f"{min(YEARS) - 1}-{max(YEARS)}"


# ---------------------------------------------------------------------------
# Resolution tags
# ---------------------------------------------------------------------------
def spatial_tag(res: float) -> str:
    """0.25 -> '025', 0.75 -> '075'."""
    return f"{res:.2f}".replace(".", "")


def file_tag(resolution: str, spatial_res: float) -> str:
    """('daily', 0.25) -> 'daily_025', ('6h', 0.75) -> '6hourly_075'."""
    return f"{TEMPORAL_TAG[resolution]}_{spatial_tag(spatial_res)}"


# ---------------------------------------------------------------------------
# Backtracking
# ---------------------------------------------------------------------------
def backtracking_input(resolution, grid_res):
    """(input file, timestep in hours) for a resolution / grid combination."""
    if resolution not in RESOLUTION_CONFIG or grid_res not in SUPPORTED_SPATIAL_RES:
        raise ValueError(f"Unknown resolution combo {(resolution, grid_res)}. "
                         f"Choose from {list(RESOLUTION_CONFIG)} x {SUPPORTED_SPATIAL_RES}")
    dt_hours = 24 // RESOLUTION_CONFIG[resolution]["steps_per_day"]
    fname = f"BACKTRACKING_INPUT_{YEARS_TAG}_{file_tag(resolution, grid_res)}.nc"
    return INTERIM_PATH / fname, dt_hours


def _dotfree(val):
    # 0.8 -> "08", 0.25 -> "025", 0.001 -> "0001", 4.0 -> "4"
    s = f"{val:.10f}".rstrip("0").rstrip(".")
    return s.replace(".", "")


def grid_dir(resolution, grid_res):
    """Folder holding all step 1 runs for one resolution, e.g. .../daily_025."""
    return BACKTRACKING_PATH / f"{resolution}_{spatial_tag(grid_res)}"


def step1_folder_name(angstr, duaod, pm10, min_pixels, iou, max_gap, stubs, memory):
    """
    Step 1 folder name, e.g. '08ae_02duaod_15en-8pm10_4px_01iou_1gap_1stub_0mem'.
      angstr, duaod, iou : decimal point removed (0.8 -> 08, 0.25 -> 025)
      pm10               : integer coefficient + 'en' + exponent (1.5e-7 -> 15en-8)
      others             : integers as-is
    """
    ae_str    = _dotfree(angstr)
    duaod_str = _dotfree(duaod)
    # pm10: express as integer coefficient x 10^exp, e.g. 1.5e-7 = 15e-8
    exp = int(np.floor(np.log10(abs(pm10))))
    coeff = pm10 / (10 ** exp)
    while coeff != int(coeff) and abs(coeff) < 1e6:
        coeff *= 10
        exp -= 1
    pm10_str = f"{int(coeff)}en{exp}"
    iou_str = _dotfree(iou)
    return (f"{ae_str}ae_{duaod_str}duaod_{pm10_str}pm10"
            f"_{min_pixels}px_{iou_str}iou_{max_gap}gap_{stubs}stub_{memory}mem")


def step2_folder_name(ws_mean_thr, ws_max_thr, ws_mean_base, ws_max_base,
                      ndvi_thr, snow_thr, delta_ae, min_cluster, upwind_plume_k, ref_stat):
    """Step 2 folder name (dot-free encoding of the step 2 parameters)."""
    wind_tag = (f"ndvidep_{_dotfree(ws_mean_base)}to{_dotfree(ws_mean_thr)}wsmean"
                f"_{_dotfree(ws_max_base)}to{_dotfree(ws_max_thr)}wsmax")
    det_tag = f"upwindplume_k{upwind_plume_k}_{ref_stat}_{_dotfree(delta_ae)}dae"
    return (f"{wind_tag}_{_dotfree(ndvi_thr)}ndvi_{_dotfree(snow_thr)}snow"
            f"_{det_tag}_{min_cluster}clust")


def sds_path(resolution, spatial_res, step1, step2):
    """Step 2 output (the model's labels) for a resolution + backtracking parameters."""
    s1 = step1_folder_name(step1["angstr"], step1["duaod"], step1["pm10"], step1["min_pixels"],
                           step1["iou"], step1["max_gap"], step1["stubs"], step1["memory"])
    s2 = step2_folder_name(step2["ws_mean_thr"], step2["ws_max_thr"], step2["ws_mean_base"],
                           step2["ws_max_base"], step2["ndvi_thr"], step2["snow_thr"],
                           step2["delta_ae"], step2["min_cluster"], step2["upwind_plume_k"],
                           step2["ref_stat"])
    return grid_dir(resolution, spatial_res) / s1 / s2 / "sds_sources_filtered.nc"


# labels the model is trained on: step 2 output of the configured run
SDS_PATH = sds_path(RESOLUTION, SPATIAL_RES, STEP1_PARAMS, STEP2_PARAMS)

# one row per backtracking run
STEP1_SUMMARY_CSV = BACKTRACKING_PATH / "build_trajectories_runs.csv"
STEP2_SUMMARY_CSV = BACKTRACKING_PATH / "detect_intermediate_sources_runs.csv"


# ---------------------------------------------------------------------------
# Model outputs and CLI paths
# ---------------------------------------------------------------------------
def model_figures_dir(model_dir, subfolder=""):
    """Figure folder of a trained model: figures/model/<model folder name>/<subfolder>."""
    d = FIGURES_PATH / "model" / Path(model_dir).name / subfolder
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_path(p) -> Path:
    """Path given on the CLI or stored in results.json: as-is if absolute or it
    exists, else relative to PROJECT_ROOT."""
    p = Path(p)
    if p.is_absolute() or p.exists():
        return p
    return PROJECT_ROOT / p


def project_relative(p) -> str:
    """Path to store in results files: relative to PROJECT_ROOT if inside it, so
    saved runs stay valid when the repository is moved."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)
