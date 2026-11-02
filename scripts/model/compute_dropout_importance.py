"""
Train an RF with a single feature dropped and measure the F1 drop vs the full model.

Workflow:
  1. run once without --drop_feature -> subsample-matched baseline (dropout/__baseline__/)
  2. run once per feature with --drop_feature <name> (independent, can run in parallel)
  3. run aggregate_dropout.py to build the combined figure + CSV

Each invocation is self-contained: it reloads the cache, drops the specified feature,
retrains, evaluates on val + test, and writes results to a subdirectory.

Use --train_frac to subsample the training data for tractability (default: config.DROPOUT_TRAIN_FRAC).
The full model baseline is read from main_model_dir/results.json and doesn't need to be retrained.

Usage (one feature at a time):
  python compute_dropout_importance.py \
      --main_model_dir models/final_rf_<timestamp> \
      --drop_feature t2m \
      --train_frac 0.2

After all jobs finish, run aggregate_dropout.py to build the combined plot.
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from sds.config import (
    VAL_CALIB_FRAC, DROPOUT_TRAIN_FRAC, VAL_SPLIT_SEED, RESOLUTION,
    SPATIAL_RES, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    FINAL_RF_SAMPLING, FINAL_RF_PARAMS, PERM_N_SAMPLES, PERM_N_REPEATS,
    PERM_SEED,
)
from sds.paths import project_relative, resolve_path, SDS_PATH
from sds.log_utils import header, section, done
from sds.data_utils import (
    FeatureCache, load_sds_source, compute_land_indices,
    collect_indices,
    platt_fit, platt_apply, compute_metrics, find_best_threshold,
    build_train_set, build_eval_set,
)


def _subsample_indices(pos, neg, train_frac, rng):
    # same fraction of positives and negatives, so the pool's ratio is kept
    if train_frac >= 1.0:
        return pos, neg
    n_pos_keep = max(1, int(round(len(pos) * train_frac)))
    n_neg_keep = max(1, int(round(len(neg) * train_frac)))
    pos_sel = rng.choice(len(pos), size=n_pos_keep, replace=False)
    neg_sel = rng.choice(len(neg), size=n_neg_keep, replace=False)
    return [pos[i] for i in pos_sel], [neg[i] for i in neg_sel]


def main():
    parser = argparse.ArgumentParser(
        description="Train an RF with one feature dropped for dropout importance "
                    "(or with no features dropped for a subsample-matched baseline)")
    parser.add_argument("--main_model_dir", type=str, required=True,
        help="path to the full-model directory - used to read feature_vars and baseline metrics")
    parser.add_argument("--drop_feature", type=str, default=None,
        help="name of the feature to remove before training. "
             "Omit (or pass __baseline__) to train a subsample-matched baseline instead.")
    parser.add_argument("--train_frac", type=float, default=DROPOUT_TRAIN_FRAC,
        help=f"fraction of training data to use (default: {DROPOUT_TRAIN_FRAC})")
    parser.add_argument("--out_dir", type=str, default=None,
        help="where to write this run's artefacts (default: main_model_dir/dropout/{feature})")
    parser.add_argument("--seed_offset", type=int, default=0,
        help="added to the FINAL_RF_SAMPLING seed so different features get different seeds")
    args = parser.parse_args()

    # baseline mode: nothing dropped, same subsample fraction. Dropout runs are
    # compared against this, not the full-data model, to separate the effect of the
    # feature from the effect of the smaller training set.
    is_baseline = args.drop_feature is None or args.drop_feature == "__baseline__"

    main_model_dir = resolve_path(args.main_model_dir)
    if not main_model_dir.exists():
        sys.exit(f"ERROR: main_model_dir not found: {main_model_dir}")

    # feature list and data config from the main run
    main_results = json.loads((main_model_dir / "results.json").read_text())
    all_feature_vars = main_results["feature_vars"]
    data_cfg = main_results["data_config"]

    if not is_baseline and args.drop_feature not in all_feature_vars:
        sys.exit(f"ERROR: '{args.drop_feature}' is not one of the main model's features")

    if is_baseline:
        kept_features = list(all_feature_vars)
        drop_idx = None
        run_label = "__baseline__"
    else:
        kept_features = [f for f in all_feature_vars if f != args.drop_feature]
        drop_idx = all_feature_vars.index(args.drop_feature)
        run_label = args.drop_feature

    # one output folder per feature, so parallel runs don't collide
    out_dir = Path(args.out_dir) if args.out_dir else \
        main_model_dir / "dropout" / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "dropout.log"),
        ],
    )
    logger = logging.getLogger("dropout")

    header("Dropout importance", {
        "Main model": project_relative(main_model_dir),
        "Dropped feature": "none (baseline)" if is_baseline else args.drop_feature,
        "Train fraction": args.train_frac,
        "Output": project_relative(out_dir),
    }, out=logger.info)

    # data config from the main run (config.py for anything missing)
    resolution  = data_cfg.get("resolution", RESOLUTION)
    spatial_res = data_cfg.get("spatial_res", SPATIAL_RES)
    train_years = data_cfg.get("train_years", TRAIN_YEARS)
    val_years   = data_cfg.get("val_years", VAL_YEARS)
    test_years  = data_cfg.get("test_years", TEST_YEARS)
    sds_path    = resolve_path(data_cfg.get("sds_path") or SDS_PATH)
    sds_file    = Path(sds_path)

    # sampling as in the main run, with the seed offset
    sampling_cfg = {**FINAL_RF_SAMPLING, 'seed': FINAL_RF_SAMPLING['seed'] + args.seed_offset}

    rf_params = dict(FINAL_RF_PARAMS)

    # feature cache
    cache = FeatureCache(
        resolution=resolution, spatial_res=spatial_res,
        train_years=train_years, val_years=val_years, test_years=test_years,
        logger=logger,
    )

    # the cache's feature order must match the main run, or drop_idx points at the wrong column
    if cache.feature_vars != all_feature_vars:
        sys.exit("ERROR: feature files and main model have a different feature order")

    # SDS labels + indices
    ds_sds, sds_var = load_sds_source(sds_file, logger)
    land_indices = compute_land_indices(
        ds_sds["latitude"].values, ds_sds["longitude"].values)

    section("Samples", out=logger.info)
    train_pos, train_neg = collect_indices(
        cache, cache.train_years, ds_sds, sds_var, land_indices, logger, label="Train")
    val_pos, val_neg = collect_indices(
        cache, cache.val_years, ds_sds, sds_var, land_indices, logger, label="Val")
    test_pos, test_neg = collect_indices(
        cache, cache.test_years, ds_sds, sds_var, land_indices, logger, label="Test")

    ds_sds.close(); del ds_sds; gc.collect()

    # datasets (training pool subsampled by --train_frac)
    rng = np.random.default_rng(sampling_cfg['seed'])
    if args.train_frac < 1.0:
        train_pos, train_neg = _subsample_indices(
            train_pos, train_neg, args.train_frac, rng)
        logger.info(f"  Train pool subsampled to {args.train_frac:.0%}: "
                    f"{len(train_pos):,} pos, {len(train_neg):,} neg")

    X_train, y_train, n_pos = build_train_set(
        cache, train_pos, train_neg, sampling_cfg, logger)
    X_val, y_val = build_eval_set(
        cache, val_pos, val_neg, sampling_cfg, rng, logger, "Val")
    X_test, y_test = build_eval_set(
        cache, test_pos, test_neg, sampling_cfg, rng, logger, "Test")

    del train_pos, train_neg, val_pos, val_neg, test_pos, test_neg
    gc.collect()

    # remove the dropped feature (nothing to remove in baseline mode)
    if not is_baseline:
        X_train = np.delete(X_train, drop_idx, axis=1)
        X_val   = np.delete(X_val,   drop_idx, axis=1)
        X_test  = np.delete(X_test,  drop_idx, axis=1)

    # train
    section("Train and evaluate", out=logger.info)
    t0 = time.time()
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    t_train = time.time() - t0
    logger.info(f"  Trained in {t_train / 60:.1f} min")

    del X_train, y_train; gc.collect()

    # Platt calibration + threshold tuning on val (same as train_final_rf.py)
    X_cal, X_eval, y_cal, y_eval = train_test_split(
        X_val, y_val, test_size=VAL_CALIB_FRAC, stratify=y_val, random_state=VAL_SPLIT_SEED)
    proba_cal = rf.predict_proba(X_cal)[:, 1]
    platt_lr = platt_fit(y_cal, proba_cal)
    proba_eval = platt_apply(platt_lr, rf.predict_proba(X_eval)[:, 1])
    thr = find_best_threshold(y_eval, proba_eval)
    val_metrics = compute_metrics(y_eval, proba_eval, thr)
    del X_cal, X_eval, y_cal, y_eval, proba_cal, proba_eval, X_val, y_val
    gc.collect()

    logger.info(f"  Val:  F1={val_metrics['f1']:.4f}  "
                f"P={val_metrics['precision']:.4f}  R={val_metrics['recall']:.4f}  "
                f"thr={thr:.4f}")

    # evaluate on test
    test_proba = platt_apply(platt_lr, rf.predict_proba(X_test)[:, 1])
    test_metrics = compute_metrics(y_test, test_proba, thr)

    logger.info(f"  Test: F1={test_metrics['f1']:.4f}  "
                f"P={test_metrics['precision']:.4f}  R={test_metrics['recall']:.4f}")

    # permutation importance on the baseline RF (baseline run only), so it is
    # compared with dropout on the same training subsample
    perm_df = None
    if is_baseline:
        section("Permutation importance (same training subsample)", out=logger.info)
        # stratified test subsample, as in compute_permutation_importance.py
        rng_perm = np.random.default_rng(PERM_SEED)
        n_test = len(y_test)
        n_perm = min(PERM_N_SAMPLES, n_test)
        if n_perm < n_test:
            pos_idx_test = np.where(y_test == 1)[0]
            neg_idx_test = np.where(y_test == 0)[0]
            pos_frac = len(pos_idx_test) / n_test
            n_p = max(1, int(round(n_perm * pos_frac)))
            n_n = n_perm - n_p
            n_p = min(n_p, len(pos_idx_test)); n_n = min(n_n, len(neg_idx_test))
            sel = np.concatenate([
                rng_perm.choice(pos_idx_test, size=n_p, replace=False),
                rng_perm.choice(neg_idx_test, size=n_n, replace=False),
            ])
            X_perm, y_perm = X_test[sel], y_test[sel]
        else:
            X_perm, y_perm = X_test, y_test

        # RF -> Platt -> fixed threshold -> F1, as in compute_permutation_importance.py
        def calibrated_f1_scorer(estimator, X_s, y_s):
            proba_raw = estimator.predict_proba(X_s)[:, 1]
            proba_cal = platt_apply(platt_lr, proba_raw)
            y_pred = (proba_cal >= thr).astype(np.int32)
            return f1_score(y_s, y_pred, zero_division=0)

        t_perm = time.time()
        perm_result = permutation_importance(
            rf, X_perm, y_perm,
            scoring=calibrated_f1_scorer,
            n_repeats=PERM_N_REPEATS,
            random_state=PERM_SEED, n_jobs=1,
        )
        logger.info(f"  Computed in {(time.time() - t_perm) / 60:.1f} min")

        perm_df = pd.DataFrame({
            "feature": cache.feature_vars,
            "perm_importance_mean": perm_result.importances_mean,
            "perm_importance_std": perm_result.importances_std,
            "gini_importance": rf.feature_importances_,
        }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)
        perm_df["perm_rank"] = range(1, len(perm_df) + 1)
        del X_perm, y_perm, perm_result
        gc.collect()

    del X_test, y_test, test_proba; gc.collect()

    # save results (delta F1 against the baseline is computed in aggregate_dropout.py)
    full_model_f1 = main_results["test_metrics"]["f1"]

    results = {
        "is_baseline": is_baseline,
        "drop_feature": None if is_baseline else args.drop_feature,
        "drop_idx_original": drop_idx,
        "n_features_kept": len(kept_features),
        "train_frac": args.train_frac,
        "train_time_s": round(t_train, 1),
        "threshold": thr,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "full_model_test_f1": full_model_f1,  # for reference
        "main_model_dir": project_relative(main_model_dir),
        "sampling_config": dict(sampling_cfg),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str))

    # permutation CSV for panel (a) of aggregate_dropout.py
    if perm_df is not None:
        perm_path = out_dir / "permutation_importance_matched.csv"
        perm_df.to_csv(perm_path, index=False)

    done(out_dir, out=logger.info)


if __name__ == "__main__":
    main()