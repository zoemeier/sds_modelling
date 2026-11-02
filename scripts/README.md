# Running the scripts individually

Run from the repository root, in this order. Settings come from `src/sds/config.py`;
`python <script> --help` lists the command-line options of each script.

```bash
# download
python scripts/download/download_ecmwf.py
python scripts/download/download_soilgrids.py
python scripts/download/download_acled.py
python scripts/download/request_ndvi.py
python scripts/download/download_ndvi.py       # once the AppEEARS NDVI requests are complete

# preprocess
python scripts/preprocess/preprocess_era5.py
python scripts/preprocess/preprocess_cams.py
python scripts/preprocess/preprocess_ndvi.py
python scripts/preprocess/regrid_soil.py
python scripts/preprocess/preprocess_acled.py
python scripts/preprocess/prepare_backtracking_data.py

# backtracking
python scripts/backtracking/build_trajectories.py
python scripts/backtracking/detect_intermediate_sources.py

# train
python scripts/model/prepare_features.py
python scripts/model/train_final_rf.py         # -> models/final_rf_<timestamp>/

# analysis
python scripts/model/analyze_rf_results.py --model_dir models/final_rf_<timestamp>
python scripts/model/event_iou_analysis.py --model_dir models/final_rf_<timestamp>
python scripts/plotting/plot_backtracking.py
python scripts/plotting/plot_segmentation.py

# importance (in this order)
python scripts/model/compute_shap.py --model_dir models/final_rf_<timestamp>
python scripts/model/compute_permutation_importance.py --model_dir models/final_rf_<timestamp>
python scripts/model/compute_dropout_importance.py --main_model_dir models/final_rf_<timestamp>
python scripts/model/compute_dropout_importance.py --main_model_dir models/final_rf_<timestamp> --drop_feature <feature>
python scripts/model/aggregate_dropout.py --main_model_dir models/final_rf_<timestamp>
python scripts/model/compute_feature_correlations.py --main_model_dir models/final_rf_<timestamp>
```

The dropout script runs once without `--drop_feature` (baseline), then once per
feature in `DROPOUT_FEATURES`.

Optional experiments: `model/subsample_ratio_experiment.py`, `model/optimize_rf.py`,
`model/retrain_boundary.py --model_dir models/final_rf_<timestamp>`, `model/train_nn.py`
(requires PyTorch).
