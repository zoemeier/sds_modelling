"""
Train an RF with a single feature dropped and measure the F1 drop vs the full model.

Designed to run as one SLURM job per feature (see submit_dropout_array.slurm).
Each invocation is self-contained: it reloads the cache, drops the specified feature,
retrains, evaluates on val + test, and writes results to a subdirectory.

Use --train_frac to subsample training data for tractability (Option 3 in the analysis plan).
The full model baseline is read from main_model_dir/results.json and doesn't need to be retrained.

Usage (one feature at a time):
  python compute_dropout_importance.py \
      --main_model_dir models/final_rf_v1_20260413_120000 \
      --drop_feature t2m_mean \
      --train_frac 0.2

After all jobs finish, run aggregate_dropout.py to build the combined plot.
"""

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from config import (
    RESOLUTION, SPATIAL_RES, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    SDS_PATH, FINAL_RF, FINAL_RF_PARAMS,
    PERM_N_SAMPLES, PERM_N_REPEATS, PERM_SEED,
)
from data_utils import (
    FeatureCache, load_sds_source, compute_land_indices,
    collect_indices,
    platt_fit, platt_apply, compute_metrics, find_best_threshold,
    build_train_set, build_eval_set,
)


def _subsample_indices(pos, neg, train_frac, rng):
    # take a fraction of both pos and neg lists, preserving the pool's ratio
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
    parser.add_argument("--train_frac", type=float, default=0.2,
        help="fraction of training data to use (default: 0.2 for a ~5x speedup)")
    parser.add_argument("--out_dir", type=str, default=None,
        help="where to write this run's artefacts (default: main_model_dir/dropout/{feature})")
    parser.add_argument("--seed_offset", type=int, default=0,
        help="added to FINAL_RF.seed so different features get different seeds")
    args = parser.parse_args()

    # baseline mode: no feature dropped, same subsample fraction - the aggregator compares
    # real dropout F1s against this instead of the full-model F1 (otherwise the comparison
    # conflates the dropout effect with the data-size difference)
    is_baseline = args.drop_feature is None or args.drop_feature == "__baseline__"

    main_model_dir = Path(args.main_model_dir)
    if not main_model_dir.exists():
        print(f"ERROR: main_model_dir not found: {main_model_dir}")
        sys.exit(1)

    # read feature list and config from the main run so we stay in lockstep with it
    main_results = json.loads((main_model_dir / "results.json").read_text())
    all_feature_vars = main_results["feature_vars"]
    data_cfg = main_results["data_config"]

    if not is_baseline and args.drop_feature not in all_feature_vars:
        print(f"ERROR: '{args.drop_feature}' is not in the main model's feature list")
        print(f"  Available features: {len(all_feature_vars)} total")
        sys.exit(1)

    # build the reduced feature set (this dropout run's "X" columns)
    if is_baseline:
        kept_features = list(all_feature_vars)
        drop_idx = None
        run_label = "__baseline__"
    else:
        kept_features = [f for f in all_feature_vars if f != args.drop_feature]
        drop_idx = all_feature_vars.index(args.drop_feature)
        run_label = args.drop_feature

    # output dir - one per feature so parallel jobs don't collide
    out_dir = Path(args.out_dir) if args.out_dir else \
        main_model_dir / "dropout" / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    # logging
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

    logger.info("=" * 60)
    logger.info("DROPOUT IMPORTANCE — baseline" if is_baseline
                else "DROPOUT IMPORTANCE — single feature")
    logger.info("=" * 60)
    logger.info(f"  Main model:    {main_model_dir}")
    logger.info(f"  Dropped:       {'(none - baseline)' if is_baseline else args.drop_feature}")
    logger.info(f"  Features kept: {len(kept_features)} / {len(all_feature_vars)}")
    logger.info(f"  Train frac:    {args.train_frac}")
    logger.info(f"  Output:        {out_dir}")
    logger.info("=" * 60)

    # pull data config from the main run (with config.py as fallback for anything missing)
    resolution  = data_cfg.get("resolution", RESOLUTION)
    spatial_res = data_cfg.get("spatial_res", SPATIAL_RES)
    train_years = data_cfg.get("train_years", TRAIN_YEARS)
    val_years   = data_cfg.get("val_years", VAL_YEARS)
    test_years  = data_cfg.get("test_years", TEST_YEARS)
    sds_path    = data_cfg.get("sds_path") or SDS_PATH
    sds_file    = Path(sds_path)

    # sampling config - same as the main run, with our seed offset
    sampling_cfg = replace(FINAL_RF, seed=FINAL_RF.seed + args.seed_offset)

    # RF params - same as the main run
    rf_params = dict(FINAL_RF_PARAMS)

    # ── Load feature cache ────────────────────────────────────────────────
    cache = FeatureCache(
        resolution=resolution, spatial_res=spatial_res,
        train_years=train_years, val_years=val_years, test_years=test_years,
        logger=logger,
    )

    # sanity check: the cache's feature order had better match what was trained on,
    # otherwise our drop_idx will point at the wrong column
    if cache.feature_vars != all_feature_vars:
        logger.error("Feature order mismatch between cache and main model — aborting")
        logger.error(f"  Cache:      {cache.feature_vars[:5]} ...")
        logger.error(f"  Main model: {all_feature_vars[:5]} ...")
        sys.exit(1)

    # ── SDS + indices ─────────────────────────────────────────────────────
    ds_sds, sds_var = load_sds_source(sds_file, logger)
    land_indices = compute_land_indices(
        ds_sds["latitude"].values, ds_sds["longitude"].values)

    logger.info("\nCollecting TRAIN indices ...")
    train_pos, train_neg = collect_indices(
        cache, cache.train_years, ds_sds, sds_var, land_indices, logger)
    logger.info("\nCollecting VAL indices ...")
    val_pos, val_neg = collect_indices(
        cache, cache.val_years, ds_sds, sds_var, land_indices, logger)
    logger.info("\nCollecting TEST indices ...")
    test_pos, test_neg = collect_indices(
        cache, cache.test_years, ds_sds, sds_var, land_indices, logger)

    ds_sds.close(); del ds_sds; gc.collect()

    # ── Build datasets (Option 3: subsample the train pool) ───────────────
    rng = np.random.default_rng(sampling_cfg.seed)
    if args.train_frac < 1.0:
        logger.info(f"\nSubsampling train pool to {args.train_frac:.0%} "
                    f"({len(train_pos):,} pos, {len(train_neg):,} neg before)")
        train_pos, train_neg = _subsample_indices(
            train_pos, train_neg, args.train_frac, rng)
        logger.info(f"  After: {len(train_pos):,} pos, {len(train_neg):,} neg")

    logger.info("\nBuilding train/val/test sets ...")

    # train - apply the main sampling config on the (possibly subsampled) pool
    X_train, y_train, n_pos = build_train_set(
        cache, train_pos, train_neg, sampling_cfg, logger)
    X_val, y_val = build_eval_set(
        cache, val_pos, val_neg, sampling_cfg, rng, logger, "Val")
    X_test, y_test = build_eval_set(
        cache, test_pos, test_neg, sampling_cfg, rng, logger, "Test")

    del train_pos, train_neg, val_pos, val_neg, test_pos, test_neg
    gc.collect()

    # drop the target column everywhere - this is the "dropout" step itself.
    # in baseline mode there's nothing to drop, we just keep the full feature set
    if not is_baseline:
        X_train = np.delete(X_train, drop_idx, axis=1)
        X_val   = np.delete(X_val,   drop_idx, axis=1)
        X_test  = np.delete(X_test,  drop_idx, axis=1)

    logger.info(f"\n  Train: {X_train.shape[0]:,} × {X_train.shape[1]} "
                f"({n_pos:,} pos / {len(y_train) - n_pos:,} neg)")
    logger.info(f"  Val:   {X_val.shape[0]:,} × {X_val.shape[1]}")
    logger.info(f"  Test:  {X_test.shape[0]:,} × {X_test.shape[1]}")

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("\nTraining RF (baseline, no drop)..." if is_baseline
                else "\nTraining RF (one feature dropped) ...")
    t0 = time.time()
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    t_train = time.time() - t0
    logger.info(f"  Trained in {t_train:.1f}s")

    del X_train, y_train; gc.collect()

    # ── Platt calibration + threshold tuning on val ───────────────────────
    # same protocol as train_final_rf.py so the F1 numbers are comparable
    X_cal, X_eval, y_cal, y_eval = train_test_split(
        X_val, y_val, test_size=0.5, stratify=y_val, random_state=42)
    proba_cal = rf.predict_proba(X_cal)[:, 1]
    platt_lr = platt_fit(y_cal, proba_cal)
    proba_eval = platt_apply(platt_lr, rf.predict_proba(X_eval)[:, 1])
    thr = find_best_threshold(y_eval, proba_eval)
    val_metrics = compute_metrics(y_eval, proba_eval, thr)
    del X_cal, X_eval, y_cal, y_eval, proba_cal, proba_eval, X_val, y_val
    gc.collect()

    logger.info(f"  VAL:  F1={val_metrics['f1']:.4f}  "
                f"P={val_metrics['precision']:.4f}  R={val_metrics['recall']:.4f}  "
                f"thr={thr:.4f}")

    # ── Evaluate on test ──────────────────────────────────────────────────
    test_proba = platt_apply(platt_lr, rf.predict_proba(X_test)[:, 1])
    test_metrics = compute_metrics(y_test, test_proba, thr)

    logger.info(f"  TEST: F1={test_metrics['f1']:.4f}  "
                f"P={test_metrics['precision']:.4f}  R={test_metrics['recall']:.4f}")

    # ── Matched-subsample permutation importance (baseline run only) ──────
    # reuses the RF we just trained so the permutation panel and the dropout panel of the
    # final figure are measuring importance on equivalent underlying models - otherwise the
    # comparison conflates the method (perm vs dropout) with the data-size difference
    perm_df = None
    if is_baseline:
        logger.info(f"\nComputing matched-subsample permutation importance "
                    f"(n={PERM_N_SAMPLES:,}, repeats={PERM_N_REPEATS}) ...")
        # stratified subsample of the test set, same shape as compute_permutation_importance.py
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

        # custom scorer: route every permutation through the full deployed pipeline
        # (RF -> Platt -> threshold -> F1) rather than rf.score()'s uncalibrated default.
        # Threshold is held at the baseline's tuned value because permutation asks a
        # deployment-time question: "if feature X got corrupted at inference time, how
        # much F1 would I lose at the threshold I already chose?" - at deployment we
        # don't know in advance which feature will fail, so the threshold is fixed.
        # (Dropout is different: dropout models are retrained without the feature, so
        # their val data also lacks it, so per-run retuning is the fair measure there.)
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
        logger.info(f"  Done in {time.time() - t_perm:.1f}s")
        logger.info(f"  Baseline F1 on perm subset (calibrated): "
                    f"{calibrated_f1_scorer(rf, X_perm, y_perm):.4f}")

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

    # ── Save results ──────────────────────────────────────────────────────
    # note: we don't compute delta_f1 here anymore - the aggregator does that by comparing
    # each dropout run against the *matched-subsample baseline* run (same train_frac, no
    # features dropped). comparing against the main model's full-data F1 would conflate the
    # dropout effect with the data-size effect.
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
        "full_model_test_f1": full_model_f1,   # kept for reference only
        "main_model_dir": str(main_model_dir),
        "sampling_config": asdict(sampling_cfg),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str))

    # save the matched-subsample permutation CSV so the aggregator can use it for panel (a)
    if perm_df is not None:
        perm_path = out_dir / "permutation_importance_matched.csv"
        perm_df.to_csv(perm_path, index=False)
        logger.info(f"  Matched-subsample permutation saved: {perm_path}")

    logger.info(f"\n{'='*60}")
    logger.info(f"DONE - {run_label}")
    logger.info(f"  Full-model F1 (ref):   {full_model_f1:.4f}")
    logger.info(f"  This run's test F1:    {test_metrics['f1']:.4f}")
    logger.info(f"  (Δ F1 computed later by aggregate_dropout.py)")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()