"""
Retrain RF with FP-focused boundary sampling.

Strategy:
  - Keep ALL positives (they're rare, the model needs every one)
  - Score all negatives with the base model
  - Oversample negatives that the base model gives high probability to
    (these are the false-positive-prone samples)
  - Retrain on this rebalanced dataset

This teaches the model to better distinguish "looks like dust but isn't"
from actual dust events.

Usage:
  python retrain_boundary.py --model_dir models/final_rf_v1_20260413_120000
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import (
    LOGS_PATH, MODELS_PATH,
    RESOLUTION, SPATIAL_RES, TRAIN_YEARS, VAL_YEARS, TEST_YEARS, SDS_PATH,
    FINAL_RF_PARAMS,
    BOUNDARY_NEG_THRESHOLD, BOUNDARY_OVERSAMPLE_FACTOR, BOUNDARY_NEG_TOTAL,
    BOUNDARY_RF_PARAMS, BOUNDARY_TAG, BOUNDARY_SEED,
)
from data_utils import (
    FeatureCache, load_sds_source, compute_land_indices,
    collect_indices, extract_from_cache,
    platt_fit, platt_apply, compute_metrics, find_best_threshold,
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def main():
    parser = argparse.ArgumentParser(
        description="Retrain RF with FP-focused boundary sampling")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to saved model from train_final_rf.py")
    parser.add_argument("--sds_path", type=str, default=None)
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--spatial", type=float, default=None)
    parser.add_argument("--train_years", nargs=2, type=int, default=None)
    parser.add_argument("--val_years", nargs=2, type=int, default=None)
    parser.add_argument("--test_years", nargs=2, type=int, default=None)
    parser.add_argument("--neg_threshold", type=float, default=None,
                        help=f"Negatives above this prob are 'hard' "
                             f"(default: {BOUNDARY_NEG_THRESHOLD})")
    parser.add_argument("--oversample", type=float, default=None,
                        help=f"Oversample factor for hard negatives "
                             f"(default: {BOUNDARY_OVERSAMPLE_FACTOR})")
    parser.add_argument("--neg_total", type=int, default=None,
                        help=f"Total negatives to sample "
                             f"(default: {BOUNDARY_NEG_TOTAL:,})")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: model_dir not found: {model_dir}")
        sys.exit(1)

    # ── Load base model ───────────────────────────────────────────────────
    base_rf = joblib.load(model_dir / "rf_model.joblib")
    platt_path = model_dir / "platt_scaler.joblib"
    base_platt = joblib.load(platt_path) if platt_path.exists() else None

    base_results = json.loads((model_dir / "results.json").read_text())
    base_thr = base_results["threshold"]
    feature_vars = base_results["feature_vars"]
    data_cfg = base_results["data_config"]

    # ── Resolve parameters ────────────────────────────────────────────────
    resolution = args.resolution or data_cfg.get("resolution", RESOLUTION)
    spatial_res = args.spatial or data_cfg.get("spatial_res", SPATIAL_RES)
    train_years = args.train_years or data_cfg.get("train_years", TRAIN_YEARS)
    val_years = args.val_years or data_cfg.get("val_years", VAL_YEARS)
    test_years = args.test_years or data_cfg.get("test_years", TEST_YEARS)
    sds_path = args.sds_path or data_cfg.get("sds_path") or SDS_PATH
    tag = args.tag or BOUNDARY_TAG
    seed = args.seed or BOUNDARY_SEED

    neg_threshold = args.neg_threshold if args.neg_threshold is not None \
        else BOUNDARY_NEG_THRESHOLD
    oversample_factor = args.oversample if args.oversample is not None \
        else BOUNDARY_OVERSAMPLE_FACTOR
    neg_total = args.neg_total if args.neg_total is not None \
        else BOUNDARY_NEG_TOTAL

    rf_params = dict(BOUNDARY_RF_PARAMS or FINAL_RF_PARAMS)

    if sds_path is None:
        print("ERROR: No SDS path")
        sys.exit(1)
    sds_file = Path(sds_path)
    if not sds_file.exists():
        print(f"ERROR: SDS file not found: {sds_file}")
        sys.exit(1)

    # ── Output directory ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = MODELS_PATH / f"{tag}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────
    log_file = out_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    logger = logging.getLogger("boundary_retrain")

    logger.info("=" * 60)
    logger.info("FP-FOCUSED BOUNDARY RETRAINING")
    logger.info("=" * 60)
    logger.info(f"  Base model: {model_dir}")
    logger.info(f"  Base threshold: {base_thr:.4f}")
    logger.info(f"  Hard negative threshold: prob > {neg_threshold}")
    logger.info(f"  Oversample factor: {oversample_factor}x")
    logger.info(f"  Negative budget: {neg_total:,}")
    logger.info(f"  Positives: ALL (no cap)")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Tag: {tag}")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"\n  RF params:")
    for k, v in sorted(rf_params.items()):
        if k not in ("n_jobs", "verbose"):
            logger.info(f"    {k}: {v}")
    logger.info("=" * 60)

    # ── Load features ─────────────────────────────────────────────────────
    cache = FeatureCache(
        resolution=resolution,
        spatial_res=spatial_res,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        logger=logger,
    )

    # ── Load SDS + collect indices ────────────────────────────────────────
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

    ds_sds.close()
    del ds_sds
    gc.collect()

    n_pos = len(train_pos)
    n_neg = len(train_neg)
    logger.info(f"\n  Positives: {n_pos:,} (keeping ALL)")
    logger.info(f"  Negatives: {n_neg:,} (will sample {neg_total:,})")

    # ---------------------------------------------------------------------------
    # Step 1: Score NEGATIVES only with base model
    # ---------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("STEP 1: Scoring negatives with base model")
    logger.info(f"{'='*60}")

    t0 = time.time()
    batch_size = 1_000_000
    neg_probas = np.empty(n_neg, dtype=np.float32)

    for start in range(0, n_neg, batch_size):
        end = min(start + batch_size, n_neg)
        X_batch = extract_from_cache(cache, train_neg[start:end])
        raw = base_rf.predict_proba(X_batch)[:, 1]
        if base_platt is not None:
            raw = platt_apply(base_platt, raw)
        neg_probas[start:end] = raw
        del X_batch
        gc.collect()

        if (end // batch_size) % 10 == 0 or end == n_neg:
            logger.info(f"    {end:,} / {n_neg:,} scored")

    t_score = time.time() - t0
    logger.info(f"  Scoring done in {t_score:.1f}s")

    # Free base model memory
    del base_rf, base_platt
    gc.collect()

    # ---------------------------------------------------------------------------
    # Step 2: Compute negative sampling weights
    # ---------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("STEP 2: Computing negative sampling weights")
    logger.info(f"{'='*60}")

    hard_mask = neg_probas > neg_threshold
    n_hard = int(hard_mask.sum())
    n_easy = n_neg - n_hard

    logger.info(f"  Hard negatives (prob > {neg_threshold}): "
                f"{n_hard:,} ({100*n_hard/n_neg:.2f}%)")
    logger.info(f"  Easy negatives (prob ≤ {neg_threshold}): "
                f"{n_easy:,} ({100*n_easy/n_neg:.2f}%)")

    # Probability stats for hard negatives
    if n_hard > 0:
        logger.info(f"  Hard neg probability: "
                    f"mean={neg_probas[hard_mask].mean():.4f}, "
                    f"median={np.median(neg_probas[hard_mask]):.4f}, "
                    f"max={neg_probas[hard_mask].max():.4f}")

    # Assign weights
    neg_weights = np.ones(n_neg, dtype=np.float64)
    neg_weights[hard_mask] = oversample_factor
    neg_weights /= neg_weights.sum()

    # ── Plot: negative probability distribution ───────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(0, 1, 201)
    a1.hist(neg_probas, bins=bins, color="#5B9BD5", alpha=0.7, edgecolor="none")
    a1.axvline(neg_threshold, color="#C00000", ls="--", lw=2,
               label=f"Hard threshold = {neg_threshold}")
    a1.axvline(base_thr, color="#222", ls=":", lw=1.5,
               label=f"Classification threshold = {base_thr:.3f}")
    a1.set_xlabel("Predicted probability (base model)")
    a1.set_ylabel("Count")
    a1.set_title("Negative sample probability distribution")
    a1.set_yscale("log")
    a1.legend()

    # Zoom into the boundary region
    zoom_mask = neg_probas > 0.005
    a2.hist(neg_probas[zoom_mask], bins=np.linspace(0.005, 0.5, 100),
            color="#5B9BD5", alpha=0.7, edgecolor="none")
    a2.axvline(neg_threshold, color="#C00000", ls="--", lw=2,
               label=f"Hard threshold = {neg_threshold}")
    a2.axvline(base_thr, color="#222", ls=":", lw=1.5,
               label=f"Classification thr = {base_thr:.3f}")
    a2.set_xlabel("Predicted probability")
    a2.set_ylabel("Count")
    a2.set_title("Zoomed: negatives with prob > 0.005")
    a2.legend()

    fig.suptitle("Negative samples — base model predictions", fontsize=13)
    fig.savefig(out_dir / "negative_probability_distribution.png")
    plt.close(fig)

    # ---------------------------------------------------------------------------
    # Step 3: Sample negatives + keep all positives
    # ---------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("STEP 3: Sampling")
    logger.info(f"{'='*60}")

    rng = np.random.default_rng(seed)
    neg_total_actual = min(neg_total, n_neg)

    # Sample negatives with weights
    sampled_neg_idx = rng.choice(n_neg, size=neg_total_actual, replace=True,
                                 p=neg_weights)

    n_sampled_hard = int(hard_mask[sampled_neg_idx].sum())
    n_sampled_easy = neg_total_actual - n_sampled_hard

    logger.info(f"  Positives: {n_pos:,} (all)")
    logger.info(f"  Negatives sampled: {neg_total_actual:,}")
    logger.info(f"    Hard: {n_sampled_hard:,} ({100*n_sampled_hard/neg_total_actual:.1f}%)")
    logger.info(f"    Easy: {n_sampled_easy:,} ({100*n_sampled_easy/neg_total_actual:.1f}%)")
    logger.info(f"  Effective ratio: 1:{neg_total_actual // max(n_pos, 1)}")

    total_train = n_pos + neg_total_actual
    logger.info(f"  Total training samples: {total_train:,} "
                f"({100*n_pos/total_train:.2f}% pos)")

    # Extract features
    logger.info("  Extracting positive features ...")
    t0 = time.time()
    X_pos = extract_from_cache(cache, train_pos)

    logger.info("  Extracting negative features ...")
    sampled_neg_indices = [train_neg[i] for i in sampled_neg_idx]
    X_neg = extract_from_cache(cache, sampled_neg_indices)

    X_train = np.vstack([X_pos, X_neg])
    y_train = np.concatenate([
        np.ones(n_pos, dtype=np.float32),
        np.zeros(neg_total_actual, dtype=np.float32),
    ])

    t_extract = time.time() - t0
    logger.info(f"  Extraction done in {t_extract:.1f}s")
    logger.info(f"  X_train shape: {X_train.shape}")

    del X_pos, X_neg, neg_probas, neg_weights, sampled_neg_idx
    del sampled_neg_indices, train_pos, train_neg
    gc.collect()

    # ---------------------------------------------------------------------------
    # Step 4: Build val/test sets
    # ---------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("STEP 4: Building val/test sets")
    logger.info(f"{'='*60}")

    logger.info("  Building val set (full) ...")
    val_indices = val_pos + val_neg
    X_val = extract_from_cache(cache, val_indices)
    y_val = np.concatenate([
        np.ones(len(val_pos), dtype=np.float32),
        np.zeros(len(val_neg), dtype=np.float32),
    ])
    logger.info(f"  Val: {len(y_val):,} ({int(y_val.sum()):,} pos)")

    logger.info("  Building test set (full) ...")
    test_indices = test_pos + test_neg
    X_test = extract_from_cache(cache, test_indices)
    y_test = np.concatenate([
        np.ones(len(test_pos), dtype=np.float32),
        np.zeros(len(test_neg), dtype=np.float32),
    ])
    logger.info(f"  Test: {len(y_test):,} ({int(y_test.sum()):,} pos)")

    del val_pos, val_neg, test_pos, test_neg, val_indices, test_indices
    gc.collect()

    # ---------------------------------------------------------------------------
    # Step 5: Train new RF
    # ---------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("STEP 5: Training FP-focused RF")
    logger.info(f"{'='*60}")

    t0 = time.time()
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    t_train = time.time() - t0
    logger.info(f"  RF trained in {t_train:.1f}s")

    # Feature importances
    imp = rf.feature_importances_
    feat_df = pd.DataFrame({
        "feature": feature_vars,
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    feat_df["rank"] = range(1, len(feat_df) + 1)

    logger.info(f"\n  Top 15 features:")
    for _, row in feat_df.head(15).iterrows():
        logger.info(f"    {row['rank']:3d}. {row['feature']:<35s} "
                     f"{row['importance']:.6f}")

    del X_train, y_train
    gc.collect()

    # ---------------------------------------------------------------------------
    # Step 6: Platt calibration + evaluation
    # ---------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("STEP 6: Calibration and evaluation")
    logger.info(f"{'='*60}")

    proba_val_raw = rf.predict_proba(X_val)[:, 1]

    unique_val = np.unique(y_val)
    if len(unique_val) >= 2 and int(y_val.sum()) >= 4:
        val_idx = np.arange(len(y_val))
        cal_idx, eval_idx = train_test_split(
            val_idx, test_size=0.5, stratify=y_val, random_state=42)

        platt_lr = platt_fit(y_val[cal_idx], proba_val_raw[cal_idx])
        proba_eval_cal = platt_apply(platt_lr, proba_val_raw[eval_idx])

        thr = find_best_threshold(y_val[eval_idx], proba_eval_cal)
        val_m = compute_metrics(y_val[eval_idx], proba_eval_cal, thr)

        thr_uncal = find_best_threshold(y_val[eval_idx],
                                         proba_val_raw[eval_idx])
        val_m_uncal = compute_metrics(y_val[eval_idx],
                                       proba_val_raw[eval_idx], thr_uncal)

        logger.info(f"  VAL (calibrated,   thr={thr:.4f}): "
                    f"F1={val_m['f1']:.4f}  P={val_m['precision']:.4f}  "
                    f"R={val_m['recall']:.4f}  AUC={val_m['auc']:.4f}")
        logger.info(f"  VAL (uncalibrated, thr={thr_uncal:.4f}): "
                    f"F1={val_m_uncal['f1']:.4f}  "
                    f"P={val_m_uncal['precision']:.4f}  "
                    f"R={val_m_uncal['recall']:.4f}")
    else:
        platt_lr = None
        thr = find_best_threshold(y_val, proba_val_raw)
        val_m = compute_metrics(y_val, proba_val_raw, thr)

    del X_val, y_val, proba_val_raw
    gc.collect()

    # Test
    proba_test_raw = rf.predict_proba(X_test)[:, 1]
    if platt_lr is not None:
        proba_test = platt_apply(platt_lr, proba_test_raw)
    else:
        proba_test = proba_test_raw

    test_m = compute_metrics(y_test, proba_test, thr)
    logger.info(f"  TEST (thr={thr:.4f}): "
                f"F1={test_m['f1']:.4f}  P={test_m['precision']:.4f}  "
                f"R={test_m['recall']:.4f}  AUC={test_m['auc']:.4f}")

    # Comparison with base model
    base_val = base_results.get("val_metrics", {})
    base_test = base_results.get("test_metrics", {})
    logger.info(f"\n  Comparison with base model:")
    logger.info(f"    Base  val  F1={base_val.get('f1', 0):.4f}  "
                f"P={base_val.get('precision', 0):.4f}  "
                f"R={base_val.get('recall', 0):.4f}")
    logger.info(f"    New   val  F1={val_m['f1']:.4f}  "
                f"P={val_m['precision']:.4f}  "
                f"R={val_m['recall']:.4f}")
    logger.info(f"    Base  test F1={base_test.get('f1', 0):.4f}  "
                f"P={base_test.get('precision', 0):.4f}  "
                f"R={base_test.get('recall', 0):.4f}")
    logger.info(f"    New   test F1={test_m['f1']:.4f}  "
                f"P={test_m['precision']:.4f}  "
                f"R={test_m['recall']:.4f}")

    # Metrics at thresholds
    from sklearn.metrics import precision_recall_curve

    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, proba_test)
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (
        prec_arr[:-1] + rec_arr[:-1] + 1e-8)

    pr_df = pd.DataFrame({
        "threshold": thr_arr,
        "precision": prec_arr[:-1],
        "recall": rec_arr[:-1],
        "f1": f1_arr,
    })

    fixed_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                        0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
    thr_table = []
    for t in fixed_thresholds:
        m = compute_metrics(y_test, proba_test, t)
        thr_table.append(m)
    thr_df = pd.DataFrame(thr_table)

    logger.info(f"\n  Metrics at selected thresholds (test):")
    logger.info(f"  {'Thr':>6s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s} "
                f"{'TP':>8s} {'FP':>8s} {'FN':>8s}")
    logger.info(f"  {'-'*52}")
    for _, r in thr_df.iterrows():
        logger.info(f"  {r['threshold']:6.2f} {r['f1']:7.4f} "
                    f"{r['precision']:7.4f} {r['recall']:7.4f} "
                    f"{r['tp']:8.0f} {r['fp']:8.0f} {r['fn']:8.0f}")

    del X_test, y_test, proba_test, proba_test_raw
    gc.collect()

    # ---------------------------------------------------------------------------
    # Save everything
    # ---------------------------------------------------------------------------
    logger.info(f"\nSaving to {out_dir} ...")

    joblib.dump(rf, out_dir / "rf_model.joblib")
    if platt_lr is not None:
        joblib.dump(platt_lr, out_dir / "platt_scaler.joblib")
    feat_df.to_csv(out_dir / "feature_importances.csv", index=False)
    pr_df.to_csv(out_dir / "pr_curve.csv", index=False)
    thr_df.to_csv(out_dir / "metrics_by_threshold.csv", index=False)

    results = {
        "tag": tag,
        "timestamp": timestamp,
        "base_model_dir": str(model_dir),
        "base_threshold": base_thr,
        "train_time_s": round(t_train, 1),
        "score_time_s": round(t_score, 1),
        "threshold": thr,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "base_val_metrics": base_val,
        "base_test_metrics": base_test,
        "boundary_config": {
            "neg_threshold": neg_threshold,
            "oversample_factor": oversample_factor,
            "neg_total": neg_total_actual,
            "n_pos": n_pos,
            "n_hard_neg": n_hard,
            "n_hard_neg_sampled": n_sampled_hard,
            "n_easy_neg_sampled": n_sampled_easy,
        },
        "n_features": len(feature_vars),
        "feature_vars": feature_vars,
        "rf_params": {k: str(v) for k, v in rf_params.items()},
        "data_config": {
            "resolution": resolution,
            "spatial_res": spatial_res,
            "train_years": train_years,
            "val_years": val_years,
            "test_years": test_years,
            "sds_path": str(sds_path),
        },
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))

    logger.info(f"  All saved.")

    # ── Done ──────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("FP-FOCUSED RETRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"  Val  F1={val_m['f1']:.4f}  P={val_m['precision']:.4f}  "
                f"R={val_m['recall']:.4f}")
    logger.info(f"  Test F1={test_m['f1']:.4f}  P={test_m['precision']:.4f}  "
                f"R={test_m['recall']:.4f}")
    logger.info(f"  Threshold: {thr:.4f}")
    delta_f1 = test_m['f1'] - base_test.get('f1', 0)
    logger.info(f"  ΔF1 vs base: {delta_f1:+.4f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()