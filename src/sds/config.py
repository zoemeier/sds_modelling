"""
Settings for the whole SDS pipeline, in pipeline order:

  1. General / system      paths and files
  2. Data                  domain, years, resolution, input data handling
  3. Backtracking          plume tracking + intermediate source detection
  4. Model                 splits, features, random forest, neural network
  5. Resampling            negative subsampling, boundary retraining
  6. Feature importance    permutation, dropout, SHAP, event-level evaluation

Only settings live here. File and folder names derived from them (e.g. the
labels the model trains on, SDS_PATH) are built in sds/paths.py. Scripts read
their defaults from this file and most settings can be overridden via CLI.
"""

from pathlib import Path


# ===========================================================================
# 1. General / system
# ===========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH         = PROJECT_ROOT / "data"
ERA5_PATH         = DATA_PATH / "era5"
CAMS_PATH         = DATA_PATH / "cams"
NDVI_PATH         = DATA_PATH / "ndvi"
ACLED_PATH        = DATA_PATH / "acled"
SOIL_PATH         = DATA_PATH / "soil"
INTERIM_PATH      = DATA_PATH / "interim"        # merged inputs (e.g. backtracking input)
OUT_PATH          = DATA_PATH / "processed"      # per-year model feature files
BACKTRACKING_PATH = DATA_PATH / "backtracking"   # backtracking runs

MODELS_PATH  = PROJECT_ROOT / "models"
LOGS_PATH    = PROJECT_ROOT / "logs"
FIGURES_PATH = PROJECT_ROOT / "figures"

PASSWORDS_FILE   = PROJECT_ROOT / "passwords.json"            # API credentials
LAND_MASK_FILE   = ERA5_PATH / "ERA5_land_sea_mask_2020.grib"
ACLED_RAW_EVENTS = ACLED_PATH / "acled_raw_events.csv"       # written by download_acled.py


# ===========================================================================
# 2. Data
# ===========================================================================
DOMAIN = {"north": 50.0, "south": 0.0, "west": -18.0, "east": 85.0}

YEARS = range(2015, 2025)

# resolution used throughout the pipeline
RESOLUTION  = "daily"     # temporal: a key of RESOLUTION_CONFIG
SPATIAL_RES = 0.25        # degrees: one of SUPPORTED_SPATIAL_RES

# supported temporal resolutions -> resampling rule, steps per day, filename tag
RESOLUTION_CONFIG = {
    "daily": {"rule": "1D", "steps_per_day": 1, "tag": "daily"},
    "6h":    {"rule": "6h", "steps_per_day": 4, "tag": "6hourly"},
    "3h":    {"rule": "3h", "steps_per_day": 8, "tag": "3hourly"},
}
SUPPORTED_SPATIAL_RES = [0.25, 0.75]
ERA5_NATIVE_RES = 0.25
CAMS_NATIVE_RES = 0.75

# ERA5 variable -> aggregation used for the temporal resampling
ERA5_VARIABLES = {
    "tp":     "sum_mm",
    "dt2m":   "mean",
    "t2m":    "mean",
    "u10":    "mean",
    "v10":    "mean",
    "lai_lv": "mean",
    "lai_hv": "mean",
}

LSM_THRESHOLD = 0.2       # land-sea mask value above which a cell counts as land


# ===========================================================================
# 3. Backtracking (the run used for the model)
# ===========================================================================
STEP1_PARAMS = dict(          # plume segmentation + trajectory building
    duaod=0.3,                # dust AOD threshold
    angstr=0.4,               # Angstrom exponent upper bound
    pm10=1.5e-7,              # PM10 threshold (kg m-3)
    min_pixels=0,             # minimum plume size (cells)
    iou=0.25,                 # linking threshold (raw or wind-shifted IoU)
    max_gap=0,
    stubs=0,                  # minimum stub length (timesteps)
    memory=0,                 # ghost timesteps
)
# score multiplier for matches found only after the wind shift
# (not encoded in the run folder name)
WINDSHIFT_SCORE_FACTOR = 0.9

STEP2_PARAMS = dict(          # intermediate source detection
    ws_mean_thr=5.0,          # mean wind threshold at NDVI = ndvi_thr (m/s)
    ws_max_thr=7.0,           # max wind threshold at NDVI = ndvi_thr (m/s)
    ws_mean_base=3.0,         # mean wind threshold at NDVI = 0 (m/s)
    ws_max_base=4.0,          # max wind threshold at NDVI = 0 (m/s)
    ndvi_thr=0.15,
    snow_thr=0.25,
    delta_ae=0.02,            # upwind AE drop threshold
    min_cluster=0,
    upwind_plume_k=4,         # nearest predecessor-plume pixels
    ref_stat="median",
)


# ===========================================================================
# 4. Model
# ===========================================================================
TRAIN_YEARS = [2015, 2022]
VAL_YEARS   = [2023, 2023]
TEST_YEARS  = [2024, 2024]

# features to leave out of the model (a variable and all its lags / rolling stats)
FEATURE_DROP_STEMS = []

# validation split: one part fits the Platt calibration, the rest tunes the threshold
VAL_CALIB_FRAC = 0.5
VAL_SPLIT_SEED = 42

# random forest used in the subsampling sweep
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=50,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    class_weight=None,
    n_jobs=-1,
    random_state=99,
    verbose=0,
)

# Bayesian hyperparameter optimisation (Optuna)
HPO_STUDY_NAME = "rf_opt"
HPO_N_TRIALS   = 60
HPO_N_STARTUP  = 15
HPO_SEARCH_SPACE = dict(
    n_estimators=(100, 500, 50),             # (low, high, step)
    max_depth=(10, 80),
    min_samples_split=(2, 50),               # log scale
    min_samples_leaf=(1, 30),                # log scale
    max_features=["sqrt", "log2", 0.1, 0.2, 0.3, 0.5],
    max_samples=(0.3, 1.0),
)

# final random forest (hyperparameters from the HPO)
FINAL_RF_PARAMS = dict(
    n_estimators=400,
    max_depth=25,
    min_samples_split=6,
    min_samples_leaf=3,
    max_features=0.5,
    max_samples=0.7,
    bootstrap=True,
    class_weight=None,
    n_jobs=-1,
    random_state=99,
    verbose=0,
)
FINAL_RF_TAG = "final_rf"

# neural network baseline
NN_PARAMS = dict(
    hidden_sizes=(512, 256, 128),
    dropout=0.2,
    epochs=100,
    batch_size=4096,
    lr=1e-3,
    weight_decay=1e-4,
    patience=15,              # early stopping (epochs without val F1 improvement)
    lr_decay_factor=0.5,      # ReduceLROnPlateau
    lr_decay_patience=5,
)
NN_TAG = "final_nn"


# ===========================================================================
# 5. Resampling
# ===========================================================================
# Negative subsampling of the training set. Positives are kept in full unless capped.
#   neg_ratio            negatives per positive
#   num_train_total_cap  fixed total training budget (0 = use num_train_pos_cap)
#   num_train_pos_cap    max positives (0 = all)
#   val_test_total_cap   cap for val/test, natural class ratio kept (0 = no cap)

# subsampling ratio sweep: fixed total budget, ratio varies per run
SWEEP_RATIOS        = [1, 2, 5, 10, 20, 30, 50, 100]
SWEEP_CLASS_WEIGHTS = [False, True]
SWEEP_SAMPLING = dict(
    neg_ratio=10,                 # overridden per run
    num_train_total_cap=500_000,
    num_train_pos_cap=0,
    val_test_total_cap=250_000,
    seed=99,
)

# HPO: ratio fixed to the winner of the sweep, different data subset (seed)
HPO_SAMPLING = dict(
    neg_ratio=10,
    num_train_total_cap=0,
    num_train_pos_cap=100_000,
    val_test_total_cap=250_000,
    seed=999,
)

# final random forest and neural network: full training pool
FINAL_RF_SAMPLING = dict(
    neg_ratio=20,
    num_train_total_cap=0,
    num_train_pos_cap=0,
    val_test_total_cap=0,
    seed=99,
)
NN_SAMPLING = dict(
    neg_ratio=20,
    num_train_total_cap=0,
    num_train_pos_cap=0,
    val_test_total_cap=0,
    seed=99,
)

# boundary retraining: oversample negatives the base model finds hard
BOUNDARY_SEED              = 999
BOUNDARY_NEG_THRESHOLD     = 0.01            # negatives above this prob get oversampled
BOUNDARY_OVERSAMPLE_FACTOR = 5.0             # how much to oversample hard vs easy negatives
BOUNDARY_NEG_TOTAL         = 20_000_000      # total negatives to sample (positives kept in full)
BOUNDARY_RF_PARAMS         = None            # None -> FINAL_RF_PARAMS
BOUNDARY_TAG               = "boundary_rf"


# ===========================================================================
# 6. Feature importance and evaluation
# ===========================================================================
# permutation importance + partial dependence
PERM_N_SAMPLES    = 200_000
PERM_N_REPEATS    = 10
PERM_SEED         = 9999
PDP_TOP_N         = 6         # partial dependence plots for the top features
PDP_ICE_SUBSAMPLE = 1000      # samples drawn as ICE lines

# dropout importance (retrain without one feature)
DROPOUT_TRAIN_FRAC = 0.2      # fraction of the training data per retrain
DROPOUT_FEATURES = [          # features retrained without (run.py importance); None = all
    "ws_mean", "ws_max", "NDVI_mean", "soc", "t2m",
    "rh", "t2m_lag_1d", "cec", "wd", "bdod",
]

# SHAP
SHAP_N_SAMPLES           = 50_000
SHAP_N_BACKGROUND        = 5000
SHAP_N_TREES             = 400
SHAP_TOP_FEATURES        = 20
SHAP_DEPENDENCE_FEATURES = 40
SHAP_SEED                = 33

# event-level evaluation (connected source regions instead of pixels)
EVENT_CONNECTIVITY = 8        # 4 or 8: neighbours that join pixels into one event
EVENT_OVERLAP_THRESHOLDS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
EVENT_SIZE_BINS = [1, 2, 5, 10, 20, 50, 100, 500, float("inf")]       # event size (pixels)
EVENT_SIZE_OVERLAP_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
