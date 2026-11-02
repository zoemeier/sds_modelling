# SDS Source Detection

Identification of sand and dust storm (SDS) source regions by spatiotemporal
backtracking, and a random forest trained on these sources to predict daily SDS
source activity from environmental data.

## Installation

```bash
conda env create -f environment.yaml
conda activate sds
pip install -e .
```

The download scripts need API credentials in `passwords.json` in the repository root:

```json
{
  "era5":  {"key": "..."},
  "cams":  {"key": "..."},
  "ndvi":  {"username": "...", "pw": "..."},
  "acled": {"username": "...", "pw": "..."}
}
```

## Usage

The pipeline runs in stages:

```bash
python run.py download        # ERA5, CAMS, SoilGrids, ACLED; submits the NDVI requests
python run.py download_ndvi   # once the AppEEARS NDVI requests are complete
python run.py preprocess      # regridding, aggregating and preprocessing
python run.py backtracking    # track dust plumes, detect genesis and intermediate sources
python run.py train           # build model features, train the random forest
python run.py analysis        # evaluation metrics and figures
python run.py importance      # SHAP, permutation and dropout importance, feature correlations
```

Each stage stops at the first error. Downloads, preprocessing and backtracking are
resumable and skip existing outputs, except `request_ndvi.py`, which submits new NDVI
requests every time it runs. `analysis` and `importance` use the most recent model in
`models/` unless `--model_dir` is given; `--dry_run` lists the commands of a stage.
Downloads, training and the feature importance analysis are time-intensive and have
high memory requirements.

Every script in `scripts/` can also be run on its own (`python <script> --help`),
including the optional experiments that are not part of a stage: the subsampling ratio
sweep (`subsample_ratio_experiment.py`), hyperparameter optimisation (`optimize_rf.py`),
boundary retraining (`retrain_boundary.py`) and the neural network baseline
(`train_nn.py`, requires PyTorch).

## Configuration

All settings are in `src/sds/config.py`, ordered by pipeline stage: data (domain, years,
resolution), backtracking parameters, model and sampling settings, and feature
importance. Output file and folder names are derived from these settings, so a change in
the config is picked up by every stage.

## Outputs

- `data/`: downloaded, preprocessed and backtracking data
- `models/`: trained models and their analysis results
- `figures/`: all figures
