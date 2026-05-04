"""
Evaluate RF performance across pre-computed backtracing outputs.

Scans existing S1/S2 folder combinations, builds splits from the
cached features + SDS source files, trains RF, and saves all metrics
to a CSV. Uses the same pipeline as fast_trial.py but skips the
backtracing steps (assumes outputs already exist).

Usage:
  python compare_backtracing_params.py \
    --s1_folders "06ae_03duaod_15en-8pm10_4px_01iou_0gap_0stub_0mem" \
                 "06ae_03duaod_2en-7pm10_4px_01iou_0gap_0stub_0mem" \
    --s2_folders "s2_folder_a" "s2_folder_b"
"""

import argparse
import csv
import gc
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from config import (
    PROJECT_ROOT, DATA_PATH, OUT_PATH, LOGS_PATH,
    RF_PARAMS, SUBSAMPLE_SEED,
    spatial_tag, file_tag,
)
from fast_trial import (
    FeatureCache, load_sds_source, compute_land_indices,
    make_target_for_year, build_split_inmemory,
    _platt_fit, _platt_apply, _compute_metrics, _find_best_threshold,
    MIN_POS_TRAIN, MIN_POS_VAL,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def evaluate_one(cache, sds_file, s1_folder, s2_folder, logger):
    """
    Run the RF evaluation pipeline on one S1/S2 combination.
    Returns a dict of metrics, or None on failure.
    """
    logger.info(f"\n  SDS: {sds_file}")

    # Load SDS + land indices
    ds_sds, sds_var = load_sds_source(sds_file, logger)
    land_indices = compute_land_indices(
        ds_sds["latitude"].values, ds_sds["longitude"].values,
    )
    logger.info(f"  Land cells: {len(land_indices):,}")

    # ── Build splits ──────────────────────────────────────────────────────
    logger.info("  Building train split ...")
    X_train, y_train, n_pos_train, raw_pos_train = build_split_inmemory(
        cache, cache.train_years, ds_sds, sds_var, land_indices,
        "train", logger)

    if X_train is None or n_pos_train < MIN_POS_TRAIN:
        logger.warning(f"  Too few train positives ({n_pos_train}), skipping")
        ds_sds.close()
        return {"status": "too_few_train_pos", "raw_pos_train": raw_pos_train,
                "n_pos_train": n_pos_train}

    logger.info("  Building val split ...")
    X_val, y_val, n_pos_val, raw_pos_val = build_split_inmemory(
        cache, cache.val_years, ds_sds, sds_var, land_indices,
        "val", logger)

    if X_val is None or n_pos_val < MIN_POS_VAL:
        logger.warning(f"  Too few val positives ({n_pos_val}), skipping")
        ds_sds.close()
        return {"status": "too_few_val_pos", "raw_pos_train": raw_pos_train,
                "n_pos_val": n_pos_val}

    logger.info("  Building test split ...")
    X_test, y_test, n_pos_test, raw_pos_test = build_split_inmemory(
        cache, cache.test_years, ds_sds, sds_var, land_indices,
        "test", logger)

    if X_test is None:
        ds_sds.close()
        return {"status": "test_split_empty", "raw_pos_train": raw_pos_train}

    ds_sds.close()
    del ds_sds
    gc.collect()

    # ── Train RF ──────────────────────────────────────────────────────────
    logger.info("  Training RF ...")
    t0 = time.time()
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    t_train = time.time() - t0
    logger.info(f"  RF trained in {t_train:.1f}s")

    # Top features
    imp = rf.feature_importances_
    for rank, i in enumerate(np.argsort(imp)[::-1][:5], 1):
        logger.info(f"    {rank}. {cache.feature_vars[i]:<30s} {imp[i]:.4f}")

    del X_train, y_train
    gc.collect()

    # ── Evaluate on val (Platt calibration) ───────────────────────────────
    unique_val = np.unique(y_val)
    if len(unique_val) < 2 or n_pos_val < 4:
        logger.warning(f"  Val has {n_pos_val} positives — skipping Platt, "
                       f"using raw probabilities")
        proba_val = rf.predict_proba(X_val)[:, 1]
        thr_cal = _find_best_threshold(y_val, proba_val)
        val_m = _compute_metrics(y_val, proba_val, thr_cal)
    else:
        X_cal, X_eval, y_cal, y_eval = train_test_split(
            X_val, y_val, test_size=0.5, stratify=y_val, random_state=42)

        proba_cal = rf.predict_proba(X_cal)[:, 1]
        platt_lr = _platt_fit(y_cal, proba_cal)

        proba_eval_raw = rf.predict_proba(X_eval)[:, 1]
        proba_eval_cal = _platt_apply(platt_lr, proba_eval_raw)
        thr_cal = _find_best_threshold(y_eval, proba_eval_cal)
        val_m = _compute_metrics(y_eval, proba_eval_cal, thr_cal)

        del X_cal, X_eval, y_cal, y_eval, proba_cal
        gc.collect()

    logger.info(f"  VAL: F1={val_m['f1']:.4f} P={val_m['precision']:.4f} "
                f"R={val_m['recall']:.4f} AUC={val_m['auc']:.4f} "
                f"thr={thr_cal:.4f}")

    # ── Evaluate on test ──────────────────────────────────────────────────
    test_proba_raw = rf.predict_proba(X_test)[:, 1]
    if n_pos_val >= 4:
        test_proba_cal = _platt_apply(platt_lr, test_proba_raw)
    else:
        test_proba_cal = test_proba_raw

    unique_test = np.unique(y_test)
    if len(unique_test) > 1:
        test_m = _compute_metrics(y_test, test_proba_cal, thr_cal)
    else:
        test_m = {"f1": 0, "precision": 0, "recall": 0, "auc": 0}

    logger.info(f"  TEST: F1={test_m['f1']:.4f} P={test_m['precision']:.4f} "
                f"R={test_m['recall']:.4f} AUC={test_m['auc']:.4f}")

    del rf
    gc.collect()

    return {
        "status": "ok",
        "raw_pos_train": raw_pos_train,
        "raw_pos_val": raw_pos_val,
        "raw_pos_test": raw_pos_test,
        "n_pos_train": n_pos_train,
        "n_pos_val": n_pos_val,
        "n_pos_test": n_pos_test,
        "train_time_s": round(t_train, 1),
        "threshold": round(thr_cal, 4),
        "val_f1": round(val_m["f1"], 4),
        "val_precision": round(val_m["precision"], 4),
        "val_recall": round(val_m["recall"], 4),
        "val_auc": round(val_m["auc"], 4),
        "val_tp": val_m.get("tp", 0),
        "val_fp": val_m.get("fp", 0),
        "val_fn": val_m.get("fn", 0),
        "val_tn": val_m.get("tn", 0),
        "test_f1": round(test_m["f1"], 4),
        "test_precision": round(test_m["precision"], 4),
        "test_recall": round(test_m["recall"], 4),
        "test_auc": round(test_m["auc"], 4),
        "test_tp": test_m.get("tp", 0),
        "test_fp": test_m.get("fp", 0),
        "test_fn": test_m.get("fn", 0),
        "test_tn": test_m.get("tn", 0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare RF performance across pre-computed backtracing outputs.")
    parser.add_argument("--s1_folders", nargs="+", required=True,
                        help="Step 1 folder names to scan for S2 subfolders")
    parser.add_argument("--s2_folders", nargs="+", default=None,
                        help="Step 2 folder names to evaluate. "
                             "If not specified, all S2 subfolders within each S1 folder are used.")
    parser.add_argument("--resolution", default="daily")
    parser.add_argument("--spatial", type=float, default=0.25)
    parser.add_argument("--train_years", nargs=2, type=int, default=[2015, 2021])
    parser.add_argument("--val_years", nargs=2, type=int, default=[2022, 2022])
    parser.add_argument("--test_years", nargs=2, type=int, default=[2024, 2024])
    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOGS_PATH / "compare"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"compare_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    logger = logging.getLogger("compare")

    out_csv = log_dir / "compare_results.csv"

    # ── Discover S1/S2 combinations ───────────────────────────────────────
    stag = spatial_tag(args.spatial)
    grid_tag = f"{args.resolution}_{stag}"
    base_dir = DATA_PATH / "backtrace_overlap_output" / "combination_overlap" / grid_tag

    s2_filter = set(args.s2_folders) if args.s2_folders else None

    combos = []
    for s1_name in args.s1_folders:
        s1_dir = base_dir / s1_name
        if not s1_dir.exists():
            logger.warning(f"S1 folder not found: {s1_dir}")
            continue

        if s2_filter is not None:
            # Only check the explicitly requested S2 folders
            s2_candidates = sorted(s2_filter)
        else:
            # Original behaviour: iterate all subdirectories
            s2_candidates = sorted(d.name for d in s1_dir.iterdir() if d.is_dir())

        for s2_name in s2_candidates:
            s2_dir = s1_dir / s2_name
            if not s2_dir.is_dir():
                logger.warning(f"  S2 folder not found: {s2_dir}")
                continue
            sds_file = s2_dir / "sds_sources_filtered.nc"
            if sds_file.exists():
                combos.append((s1_name, s2_name, sds_file))
            else:
                logger.warning(f"  No sds_sources_filtered.nc in {s2_dir.name}")

    logger.info("=" * 70)
    logger.info("COMPARE BACKTRACING PARAMETERS")
    logger.info("=" * 70)
    logger.info(f"  S1 folders: {args.s1_folders}")
    if s2_filter:
        logger.info(f"  S2 folders (filter): {args.s2_folders}")
    else:
        logger.info(f"  S2 folders: all")
    logger.info(f"  Found {len(combos)} S1/S2 combinations")
    logger.info(f"  Output CSV: {out_csv}")
    logger.info(f"  Log: {log_file}")

    if not combos:
        logger.error("No valid combinations found. Exiting.")
        sys.exit(1)

    # ── Resume: load existing results ─────────────────────────────────────
    results = []
    done_keys = set()
    if out_csv.exists():
        with open(out_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
                done_keys.add((row["s1_folder"], row["s2_folder"]))
        logger.info(f"  Resuming: {len(results)} combos already done")

    # ── Load features ─────────────────────────────────────────────────────
    cache = FeatureCache(
        resolution=args.resolution,
        spatial_res=args.spatial,
        train_years=args.train_years,
        val_years=args.val_years,
        test_years=args.test_years,
        logger=logger,
    )

    # ── Evaluate each combination ─────────────────────────────────────────
    for i, (s1_name, s2_name, sds_file) in enumerate(combos):
        if (s1_name, s2_name) in done_keys:
            logger.info(f"\n  [{i+1}/{len(combos)}] Skipping {s1_name} / {s2_name} "
                        f"(already done)")
            continue

        logger.info(f"\n{'=' * 70}")
        logger.info(f"[{i+1}/{len(combos)}] S1: {s1_name}")
        logger.info(f"          S2: {s2_name}")
        logger.info(f"{'=' * 70}")

        try:
            metrics = evaluate_one(cache, sds_file, s1_name, s2_name, logger)
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            metrics = {"status": f"error: {e}"}

        row = {
            "s1_folder": s1_name,
            "s2_folder": s2_name,
            **metrics,
        }
        results.append(row)

        # Write incrementally
        _write_csv(out_csv, results)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 70}")
    logger.info("COMPARISON COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Total: {len(results)} combinations")
    logger.info(f"  CSV: {out_csv}")
    logger.info(f"  Log: {log_file}")

    # Print summary table of successful runs
    ok_rows = [r for r in results if r.get("status") == "ok"]
    if ok_rows:
        logger.info(f"\n{'S1':<55s} {'S2':<65s} {'n_pos':>10s} "
                    f"{'Val F1':>8s} {'Val P':>8s} {'Val R':>8s} "
                    f"{'Test F1':>8s}")
        logger.info("-" * 170)
        for r in sorted(ok_rows, key=lambda x: float(x.get("val_f1", 0)),
                        reverse=True):
            logger.info(f"{r['s1_folder']:<55s} {r['s2_folder']:<65s} "
                        f"{r.get('raw_pos_train', '?'):>10s} "
                        f"{r.get('val_f1', '?'):>8s} "
                        f"{r.get('val_precision', '?'):>8s} "
                        f"{r.get('val_recall', '?'):>8s} "
                        f"{r.get('test_f1', '?'):>8s}")


def _write_csv(path, rows):
    if not rows:
        return
    # Collect all keys across all rows
    all_keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()