"""
Run the pipeline stage by stage. Each stage runs its scripts in order and stops
at the first error. Downloads, preprocessing and backtracking skip outputs that
already exist; training and the model analyses run again (a new model folder
per training run). All settings come from src/sds/config.py; every script can
also be run on its own with more options (see each script's --help).

Stages:
    download        ERA5 + CAMS, SoilGrids, ACLED, submit the NDVI requests
    download_ndvi   NDVI bundles (once AppEEARS has emailed that they are ready)
    preprocess      ERA5, CAMS, NDVI, soil, ACLED, backtracking input
    backtracking    step 1 (trajectories) + step 2 (intermediate sources)
    train           model features + final random forest
    analysis        model evaluation, event-level overlap, backtracking figures
    importance      SHAP, permutation, dropout (config.DROPOUT_FEATURES), correlations

Usage:
    python run.py preprocess
    python run.py analysis --model_dir models/final_rf_<timestamp>
    python run.py importance --dry_run
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path

from sds.config import PROJECT_ROOT, MODELS_PATH, FINAL_RF_TAG, DROPOUT_FEATURES
from sds.paths import resolve_path, project_relative

STAGES = ["download", "download_ndvi", "preprocess", "backtracking",
          "train", "analysis", "importance"]
MODEL_STAGES = {"analysis", "importance"}


def latest_model():
    runs = sorted(MODELS_PATH.glob(f"{FINAL_RF_TAG}_*"))
    if not runs:
        sys.exit(f"ERROR: no {FINAL_RF_TAG}_* folder in {MODELS_PATH} (run the train stage first)")
    return runs[-1]


def dropout_features(model_dir):
    if DROPOUT_FEATURES is not None:
        return DROPOUT_FEATURES
    return json.loads((model_dir / "results.json").read_text())["feature_vars"]


def commands(stage, model_dir=None):
    """List of (script, args) for one stage."""
    m = ["--model_dir", project_relative(model_dir)] if model_dir else []
    mm = ["--main_model_dir", project_relative(model_dir)] if model_dir else []
    if stage == "download":
        return [("download/download_ecmwf.py", []),
                ("download/download_soilgrids.py", []),
                ("download/download_acled.py", []),
                ("download/request_ndvi.py", [])]
    if stage == "download_ndvi":
        return [("download/download_ndvi.py", [])]
    if stage == "preprocess":
        return [("preprocess/preprocess_era5.py", []),
                ("preprocess/preprocess_cams.py", []),
                ("preprocess/preprocess_ndvi.py", []),
                ("preprocess/regrid_soil.py", []),
                ("preprocess/preprocess_acled.py", []),
                ("preprocess/prepare_backtracking_data.py", [])]
    if stage == "backtracking":
        return [("backtracking/build_trajectories.py", []),
                ("backtracking/detect_intermediate_sources.py", [])]
    if stage == "train":
        return [("model/prepare_features.py", []),
                ("model/train_final_rf.py", [])]
    if stage == "analysis":
        return [("model/analyze_rf_results.py", m),
                ("model/event_iou_analysis.py", m),
                ("plotting/plot_backtracking.py", []),
                ("plotting/plot_segmentation.py", [])]
    if stage == "importance":
        # SHAP first, so the permutation figures include the SHAP comparison;
        # correlations last, so they use the dropout baseline's permutation ranking
        return ([("model/compute_shap.py", m),
                 ("model/compute_permutation_importance.py", m),
                 ("model/compute_dropout_importance.py", mm)]
                + [("model/compute_dropout_importance.py", mm + ["--drop_feature", f])
                   for f in dropout_features(model_dir)]
                + [("model/aggregate_dropout.py", mm),
                   ("model/compute_feature_correlations.py", mm)])
    raise ValueError(stage)


def main():
    p = argparse.ArgumentParser(description="Run one pipeline stage",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("Usage:")[0])
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--model_dir", type=Path, default=None,
                   help=f"model for analysis / importance (default: newest models/{FINAL_RF_TAG}_*)")
    p.add_argument("--dry_run", action="store_true", help="only print the commands")
    args = p.parse_args()

    model_dir = None
    if args.stage in MODEL_STAGES:
        model_dir = resolve_path(args.model_dir) if args.model_dir else latest_model()
        if not model_dir.exists():
            sys.exit(f"ERROR: model_dir not found: {model_dir}")

    cmds = commands(args.stage, model_dir)
    for i, (script, script_args) in enumerate(cmds, 1):
        cmd = [sys.executable, str(Path("scripts") / script), *script_args]
        print(f"\n[{args.stage} {i}/{len(cmds)}] {' '.join(cmd[1:])}", flush=True)
        if args.dry_run:
            continue
        if subprocess.run(cmd, cwd=PROJECT_ROOT).returncode != 0:
            sys.exit(f"ERROR: {script} failed, stage '{args.stage}' stopped")


if __name__ == "__main__":
    main()
