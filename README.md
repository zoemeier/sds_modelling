# SDS Source Detection 

Code for the master thesis "Identifying Sand and Dust Storm Source Regions Through Spatiotemporal Backtracking for Machine Learning-Based Modelling". The repo contains all the data preprocessing, backtracking and ML code. 
The full workflow has four stages, run roughly in this order:

```
download  →  preprocess  →  backtrace  →  hpc (train + analyse)
```

The download and preprocessing stages run on a local workstation; training is ideally run on a cluster.

## Repository layout

```
sds_source_detection/
├── data/                        # all input + intermediate + output data
│   ├── era5/                    # ERA5 reanalysis (gribs, regridded NetCDFs)
│   ├── cams/                    # CAMS aerosol reanalysis
│   ├── ndvi/                    # MODIS MOD13A1 NDVI bundles + regridded files
│   ├── soil/                    # SoilGrids GeoTIFFs + NetCDFs
│   ├── acled/                   # ACLED conflict events (raw + gridded)
│   ├── interim/                 # combined backtracing inputs, ERA5 features
│   ├── processed/               # per-year RF feature files
│   ├── backtrace_overlap_output/  # SDS sources from backtracing
│   └── analysis/                # plots and analysis artefacts
├── scripts/
│   ├── download/                # API requests + downloads (slow; run alone)
│   ├── preprocess/              # quality-filter, regrid, aggregate
│   ├── backtrace/               # plume tracking / source attribution
│   └── hpc/                     # RF + NN training, importance, SHAP, IoU
├── environment.yaml             # conda env spec
├── passwords.json               # API keys (not committed)
└── README.md
```

## Stage 1 — download (`scripts/download/`)

These are the slowest steps (the ECMWF download alone takes >24h). They write
straight into `data/era5/`, `data/cams/`, `data/ndvi/`, `data/soil/`,
`data/acled/`. All of them are resumable — re-running only fetches what's
missing.

| Script                  | What it does                                                          |
| ----------------------- | --------------------------------------------------------------------- |
| `download_ecmwf.py`     | Requests + downloads ERA5 (per variable, per year) and CAMS (per year) from CDS/ADS. |
| `request_ndvi.py`       | Submits AppEEARS area requests for MODIS NDVI, one task per year.     |
| `download_ndvi.py`      | Downloads completed AppEEARS bundles (run after the email arrives).   |
| `download_soilgrids.py` | Pulls SoilGrids 2.0 GeoTIFFs from the ISRIC WCS and combines them into one NetCDF. |
| `download_acled.py`     | Pulls ACLED conflict events and saves daily gridded counts.           |

API keys are read from `passwords.json` at the project root, with keys
`era5`, `cams`, `ndvi`, `acled`.

## Stage 2 — preprocess (`scripts/preprocess/`)

Quality-filter, aggregate, regrid each data source onto a common grid (0.25°
ERA5 grid by default; 0.75° also supported) and temporal resolution (`1D`
default; `6h` and `3h` also supported). All of these support `-r` and `-s`
flags, e.g. `python preprocess_era5.py -r 6h -s 0.75`.

| Script                          | What it does                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------- |
| `grid_config.py`                | Shared grid + resolution config; reads native grids from the data files (cached).     |
| `preprocess_era5.py`            | Aggregates hourly ERA5 to target res, computes ws / wd / rh / rv, builds the merged feature dataset. |
| `preprocess_cams.py`            | Resamples 3-hourly CAMS and regrids onto the target grid.                             |
| `preprocess_ndvi.py`            | Quality-filters MODIS 500m, regrids to target grid, interpolates to target temporal freq. |
| `preprocess_acled.py`           | Snaps events to the target grid, builds rolling-window counts (30/180/365d), log1p.    |
| `regrid_soil.py`                | Conservative regridding of soil to the target grid.                                   |
| `prepare_backtracking_data.py`  | Merges ERA5 wind + CAMS aerosol + derived wind speed into the backtracing input file. |

## Stage 3 — backtrace (`scripts/backtrace/`)

Track aerosol plumes back to candidate sources from the merged ERA5+CAMS file.
Both scripts accept parameters and run a cartesian product of all values
passed (grid search). Outputs go to `data/backtrace_overlap_output/`.

| Script                              | What it does                                                                                |
| ----------------------------------- | ------------------------------------------------------------------------------------------- |
| `backtrace_overlap_windshift.py`    | Step 1 — trajectory building & genesis  detection                                  |
| `backtrace_intermediate_regions.py` | Step 2 — intermediate source detection                                           |

## Stage 4 — HPC (`scripts/hpc/`)

Training and analysis of the RF source-detection model. Paths and most
defaults live in `config.py`; every script accepts CLI overrides.

Shared modules:
- `config.py`        — paths, hyperparameters, sampling configs, feature drop list
- `data_utils.py`    — `FeatureCache`, target/index builders, Platt scaling, metrics
- `feature_names.py` — short → display name + units, plot-grouping definitions

Feature assembly (run once per resolution before any training):
- `prepare_features.py` — flatten ERA5 + NDVI + ACLED + soil into per-year flat NetCDFs in `data/processed/<ftag>/`.
  
Training + tuning:
- `subsample_ratio_experiment.py` — sweep neg:pos ratio for the RF
- `optimize_rf.py`                — Bayesian  HPO over RF hyperparameters
- `train_final_rf.py`             — final RF with the chosen HPO hyperparameters (read from config.py)
- `retrain_boundary.py`           — retrain on top of a saved RF, oversampling FP-prone (high base-prob) negatives to sharpen the decision boundary
- `train_nn.py`                   — MLP baseline using the same data pipeline (need a separate venv, the conda environment doesnt contain pytorch)

Analysis:
- `analyze_rf_results.py`           — figures and metrics for a trained RF
- `compute_permutation_importance.py` — permutation importance
- `compute_dropout_importance.py`   — per-feature dropout (re-train without it)
- `aggregate_dropout.py`            — combine dropout runs into one figure
- `compute_shap.py`                 — SHAP values + diagnostic plots
- `compute_feature_correlations.py` — feature correlation analysis
- `event_iou_analysis.py`           — event-level overlap (IoU) analysis
- `compare_backtracing_params.py`   — RF performance across backtracing parameter sweeps

## Setup

```bash
conda env create -f environment.yaml
conda activate <env-name>
```

`passwords.json` at the project root is required for the download scripts:

```json
{
  "era5":  {"key": "..."},
  "cams":  {"key": "..."},
  "ndvi":  {"username": "...", "pw": "..."},
  "acled": {"username": "...", "pw": "..."}
}
```

## Running the pipeline

End-to-end (default daily, 0.25°):

```bash
# Stage 1 — slow, resumable
python scripts/download/download_ecmwf.py
python scripts/download/request_ndvi.py        # then wait for email
python scripts/download/download_ndvi.py
python scripts/download/download_soilgrids.py
python scripts/download/download_acled.py

# Stage 2
python scripts/preprocess/preprocess_era5.py
python scripts/preprocess/preprocess_cams.py
python scripts/preprocess/preprocess_ndvi.py
python scripts/preprocess/regrid_soil.py
python scripts/preprocess/preprocess_acled.py
python scripts/preprocess/prepare_backtracking_data.py

# Stage 3
python scripts/backtrace/backtrace_overlap_windshift.py
python scripts/backtrace/backtrace_intermediate_regions.py

# Stage 4 (on HPC)
python scripts/hpc/prepare_features.py --years 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024
python scripts/hpc/optimize_rf.py                  # optional: HPO sweep
python scripts/hpc/train_final_rf.py
python scripts/hpc/retrain_boundary.py --model_dir models/<final_rf_run>   # optional refinement
python scripts/hpc/analyze_rf_results.py
# (then permutation/dropout/SHAP as needed)
```

Resolutions other than the default work via `-r {1D,6h,3h}` and `-s {0.25,0.75}`
on the preprocess + backtrace scripts.
