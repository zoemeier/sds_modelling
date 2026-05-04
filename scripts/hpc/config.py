"""
Configuration for the SDS modeling pipeline. All tunable parameters live
here; scripts read defaults from this file and may override via CLI.
"""

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH    = PROJECT_ROOT / "data"
ERA5_PATH    = DATA_PATH / "era5"
NDVI_PATH    = DATA_PATH / "ndvi"
ACLED_PATH   = DATA_PATH / "acled"
SOIL_PATH    = DATA_PATH / "soil"
INTERIM_PATH = DATA_PATH / "interim"
OUT_PATH     = DATA_PATH / "processed"
MODELS_PATH  = PROJECT_ROOT / "models"
LOGS_PATH    = PROJECT_ROOT / "logs"

LAND_MASK_FILE = ERA5_PATH / "ERA5_land_sea_mask_2020.grib"
LSM_THRESHOLD  = 0.2


TEMPORAL_TAG = {"daily": "daily", "6h": "6hourly", "3h": "3hourly"}


def spatial_tag(res: float) -> str:
    return f"{res:.2f}".replace(".", "")


def file_tag(resolution: str, spatial_res: float) -> str:
    return f"{TEMPORAL_TAG[resolution]}_{spatial_tag(spatial_res)}"


# ---------------------------------------------------------------------------
# Features to drop
# ---------------------------------------------------------------------------
FEATURE_DROP_STEMS = [
    "swvl1",
    "blh",
    "cape",
    "i10fb",
    "i10fg",
    "elevation",
]


def should_drop_feature(varname: str) -> bool:
    """True if varname starts with any stem in FEATURE_DROP_STEMS."""
    for stem in FEATURE_DROP_STEMS:
        if varname == stem or varname.startswith(stem + "_"):
            return True
    return False


# ---------------------------------------------------------------------------
# RF default hyperparameters
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Sampling configurations
# ---------------------------------------------------------------------------
@dataclass
class SamplingConfig:
    neg_ratio: int              # neg:pos ratio
    num_train_total_cap: int    # fixed total train budget (0 = use pos_cap)
    num_train_pos_cap: int      # max positive samples (0 = use total_cap)
    val_test_total_cap: int     # total cap for val/test (0 = no cap)
    seed: int


# Subsampling sweep — fixed total budget; ratio varies per trial
SUBSAMPLE_SWEEP = SamplingConfig(
    neg_ratio=10,                 # overridden per trial
    num_train_total_cap=500_000,
    num_train_pos_cap=0,
    val_test_total_cap=250_000,
    seed=99,
)

# Bayesian HPO — neg_ratio fixed (winner from sweep)
BAYESIAN_HPO = SamplingConfig(
    neg_ratio=10,
    num_train_total_cap=0,
    num_train_pos_cap=100_000,
    val_test_total_cap=250_000,
    seed=999,                     # different subset of data used for HPO
)


# ---------------------------------------------------------------------------
# Shared experiment defaults
# ---------------------------------------------------------------------------
RESOLUTION  = "daily"
SPATIAL_RES = 0.25

TRAIN_YEARS = [2015, 2022]
VAL_YEARS   = [2023, 2023]
TEST_YEARS  = [2024, 2024]

SDS_PATH = (
    DATA_PATH / "backtrace_overlap_output" / "combination_overlap"
    / "daily_025"
    / "04ae_03duaod_15en-8pm10_0px_025iou_0gap_0stub_0mem"
    / "ndvidep_3to5wsmean_4to7wsmax_015ndvi_025snow_06aepx_upwindplume_k4_median_fbmedian_noduaod_002dae_novol_0clust"
    / "sds_sources_filtered.nc"
)


# ---------------------------------------------------------------------------
# Subsampling ratio sweep
# ---------------------------------------------------------------------------
SWEEP_RATIOS        = [1, 2, 5, 10, 20, 30, 50, 100]
SWEEP_CLASS_WEIGHTS = [False, True]


# ---------------------------------------------------------------------------
# Bayesian HPO (Optuna)
# ---------------------------------------------------------------------------
HPO_STUDY_NAME = "rf_opt"
HPO_N_TRIALS   = 60
HPO_N_STARTUP  = 15


# ---------------------------------------------------------------------------
# Final RF
# ---------------------------------------------------------------------------
FINAL_RF = SamplingConfig(
    neg_ratio=10,
    num_train_total_cap=0,
    num_train_pos_cap=0,
    val_test_total_cap=0,
    seed=99,
)

FINAL_RF_PARAMS_OLD = dict(
    n_estimators=300,
    max_depth=32,
    min_samples_split=2,
    min_samples_leaf=9,
    max_features=0.5,
    max_samples=1.0,
    bootstrap=True,
    class_weight=None,
    n_jobs=-1,
    random_state=99,
    verbose=0,
)

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

FINAL_RF_TAG = "10sub_400est_25d_noelevation_hpo"


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
NN_SAMPLING = SamplingConfig(
    neg_ratio=10,
    num_train_total_cap=0,
    num_train_pos_cap=0,
    val_test_total_cap=0,
    seed=99,
)

NN_PARAMS = dict(
    hidden_sizes=(512, 256, 128),
    dropout=0.2,
    epochs=100,
    batch_size=8192,
    lr=1e-3,
    weight_decay=1e-4,
    patience=15,
)

NN_TAG = "nn_v4"


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------
SHAP_N_SAMPLES        = 50_000
SHAP_N_INTERACTION    = 500
SHAP_TOP_FEATURES     = 20
SHAP_DEPENDENCE_FEATURES = 40
SHAP_SEED             = 33
SHAP_COMPUTE_INTERACTIONS = False
SHAP_N_TREES          = 400
SHAP_N_BACKGROUND     = 5000


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------
PERM_N_SAMPLES = 200_000
PERM_N_REPEATS = 10
PERM_SEED      = 9999


# ---------------------------------------------------------------------------
# Boundary retraining
# ---------------------------------------------------------------------------
BOUNDARY_SEED              = 999
BOUNDARY_NEG_THRESHOLD     = 0.01            # negatives above this prob get oversampled
BOUNDARY_OVERSAMPLE_FACTOR = 5.0             # how much to oversample hard vs easy negatives
BOUNDARY_NEG_TOTAL         = 20_000_000      # total negatives to sample (positives kept in full)
BOUNDARY_RF_PARAMS         = None            # None → use FINAL_RF_PARAMS
BOUNDARY_TAG               = "boundary_rf_v2"
