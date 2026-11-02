"""
Test different neg:pos subsampling ratios to find the best
precision/recall trade-off for the SDS Random Forest.

defaults are read from config.py, but parameters can be overridden via CLI flags.

Usage:
  python subsample_ratio_experiment.py                          # all defaults
  python subsample_ratio_experiment.py --ratios 1 5 20          # custom ratios
"""

import argparse
import csv
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sds.config import (
    VAL_CALIB_FRAC, VAL_SPLIT_SEED, LOGS_PATH, RF_PARAMS,
    SWEEP_SAMPLING, RESOLUTION, SPATIAL_RES, TRAIN_YEARS,
    VAL_YEARS, TEST_YEARS, SWEEP_RATIOS, SWEEP_CLASS_WEIGHTS,
)
from sds.paths import project_relative, resolve_path, SDS_PATH
from sds.log_utils import header, section, done
from sds.data_utils import (
    FeatureCache, load_sds_source, compute_land_indices,
    collect_indices, build_train_set, build_eval_set,
    platt_fit, platt_apply, compute_metrics, find_best_threshold,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Subsampling ratio sweep for SDS Random Forest")

    # Data / paths — defaults from config
    parser.add_argument("--sds_path", type=str, default=None,
                        help="Path to sds_sources_filtered.nc "
                             f"(default: config.SDS_PATH = {SDS_PATH})")
    parser.add_argument("--resolution", default=None,
                        help=f"Temporal resolution (default: {RESOLUTION})")
    parser.add_argument("--spatial", type=float, default=None,
                        help=f"Spatial resolution (default: {SPATIAL_RES})")
    parser.add_argument("--train_years", nargs=2, type=int, default=None,
                        help=f"Train year range (default: {TRAIN_YEARS})")
    parser.add_argument("--val_years", nargs=2, type=int, default=None,
                        help=f"Val year range (default: {VAL_YEARS})")
    parser.add_argument("--test_years", nargs=2, type=int, default=None,
                        help=f"Test year range (default: {TEST_YEARS})")

    # Sampling overrides
    parser.add_argument("--total_cap", type=int, default=None,
                        help=f"Override num_train_total_cap "
                             f"(default: {SWEEP_SAMPLING['num_train_total_cap']})")
    parser.add_argument("--val_test_cap", type=int, default=None,
                        help=f"Override val_test_total_cap "
                             f"(default: {SWEEP_SAMPLING['val_test_total_cap']})")
    parser.add_argument("--seed", type=int, default=None,
                        help=f"Override seed "
                             f"(default: {SWEEP_SAMPLING['seed']})")

    # Sweep settings
    parser.add_argument("--ratios", nargs="+", type=int, default=None,
                        help=f"Neg:pos ratios to test (default: {SWEEP_RATIOS})")
    parser.add_argument("--no_class_weight", action="store_true",
                        help="Only test without class_weight (skip balanced_subsample)")
    parser.add_argument("--class_weight_only", action="store_true",
                        help="Only test with class_weight='balanced_subsample'")

    # Output
    parser.add_argument("--out_csv", type=str, default=None,
                        help="Output CSV path")
    args = parser.parse_args()

    # ── Resolve all parameters from config + CLI overrides ────────────────
    resolution  = args.resolution or RESOLUTION
    spatial_res = args.spatial or SPATIAL_RES
    train_years = args.train_years or TRAIN_YEARS
    val_years   = args.val_years or VAL_YEARS
    test_years  = args.test_years or TEST_YEARS
    sds_path    = resolve_path(args.sds_path or SDS_PATH)
    ratios      = args.ratios or SWEEP_RATIOS

    # Resolve class weight options
    if args.no_class_weight:
        use_class_weights = [False]
    elif args.class_weight_only:
        use_class_weights = [True]
    else:
        use_class_weights = SWEEP_CLASS_WEIGHTS

    # Build sampling config with overrides
    base_cfg = SWEEP_SAMPLING
    overrides = {}
    if args.total_cap is not None:
        overrides["num_train_total_cap"] = args.total_cap
    if args.val_test_cap is not None:
        overrides["val_test_total_cap"] = args.val_test_cap
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        base_cfg = {**base_cfg, **overrides}

    sds_file = Path(sds_path)
    if not sds_file.exists():
        sys.exit(f"ERROR: labels not found: {sds_file} (run the backtracking first)")

    # ── Logging ───────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("ratio_experiment")

    out_csv = Path(args.out_csv) if args.out_csv else \
        LOGS_PATH / "subsample_ratio" / "subsample_ratio_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── Resume: load existing results ─────────────────────────────────────
    results = []
    done_keys = set()
    if out_csv.exists():
        with open(out_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
                done_keys.add((row["ratio"], row["class_weight"]))

    header("Subsampling ratio sweep", {
        "Labels": project_relative(sds_file),
        "Resolution": f"{resolution}, {spatial_res}°",
        "Train / val / test": f"{train_years} / {val_years} / {test_years}",
        "Ratios": ", ".join(f"1:{r}" for r in ratios) + ", natural",
        "Class weights": ", ".join(str(c) for c in use_class_weights),
        "Sampling": f"train total cap {base_cfg['num_train_total_cap']:,}, "
                    f"val/test cap {base_cfg['val_test_total_cap']:,}, seed {base_cfg['seed']}",
        "Output": project_relative(out_csv),
    }, out=logger.info)
    if results:
        logger.info(f"  Resuming: {len(results)} runs already done")

    # ── Load features ─────────────────────────────────────────────────────
    cache = FeatureCache(
        resolution=resolution,
        spatial_res=spatial_res,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        logger=logger,
    )

    # ── Load SDS + land indices ───────────────────────────────────────────
    ds_sds, sds_var = load_sds_source(sds_file, logger)
    land_indices = compute_land_indices(
        ds_sds["latitude"].values, ds_sds["longitude"].values,
    )

    # ── Collect all indices ───────────────────────────────────────────────
    section("Samples", out=logger.info)
    train_pos, train_neg = collect_indices(
        cache, cache.train_years, ds_sds, sds_var, land_indices, logger, label="Train")

    val_pos, val_neg = collect_indices(
        cache, cache.val_years, ds_sds, sds_var, land_indices, logger, label="Val")

    test_pos, test_neg = collect_indices(
        cache, cache.test_years, ds_sds, sds_var, land_indices, logger, label="Test")

    ds_sds.close()
    del ds_sds
    gc.collect()

    # ── Build val/test once ───────────────────────────────────────────────
    rng = np.random.default_rng(base_cfg['seed'])

    X_val, y_val = build_eval_set(
        cache, val_pos, val_neg, base_cfg, rng, logger, "Val")

    X_test, y_test = build_eval_set(
        cache, test_pos, test_neg, base_cfg, rng, logger, "Test")

    # ── Log data availability ─────────────────────────────────────────────
    n_total_pos = len(train_pos)
    n_total_neg = len(train_neg)
    true_ratio = n_total_neg / max(n_total_pos, 1)
    logger.info(f"  Natural train ratio: 1:{true_ratio:.0f}")

    # ── Run experiments ───────────────────────────────────────────────────
    # Add "natural" ratio (true class ratio) as a special case
    natural_ratio = int(round(true_ratio))
    all_ratios = ratios + [("natural", natural_ratio)]

    for ratio_entry in all_ratios:
        if isinstance(ratio_entry, tuple):
            ratio_label_prefix, ratio = ratio_entry
            ratio_label = f"natural(~1:{ratio})"
            ratio_key = "natural"
        else:
            ratio = ratio_entry
            ratio_label = f"1:{ratio}"
            ratio_key = str(ratio)

        for use_cw in use_class_weights:
            cw_label = "balanced_subsample" if use_cw else "none"

            if (ratio_key, cw_label) in done_keys:
                continue

            # Build train set with this ratio (fixed total budget)
            trial_cfg = {**base_cfg, 'neg_ratio': ratio}

            section(f"Ratio {ratio_label}, class_weight={cw_label}", out=logger.info)

            t0 = time.time()
            X_train, y_train, n_pos = build_train_set(
                cache, train_pos, train_neg, trial_cfg, logger)
            n_neg = len(y_train) - n_pos

            # Train RF
            t0 = time.time()
            rf_params = dict(RF_PARAMS)
            if use_cw:
                rf_params["class_weight"] = "balanced_subsample"
            else:
                rf_params.pop("class_weight", None)

            rf = RandomForestClassifier(**rf_params)
            rf.fit(X_train, y_train)
            t_train = time.time() - t0
            logger.info(f"  Trained in {t_train / 60:.1f} min")

            del X_train, y_train
            gc.collect()

            # ── Evaluate on val ───────────────────────────────────────────
            unique_val = np.unique(y_val)
            if len(unique_val) < 2:
                logger.warning("  Val set has only one class (run skipped)")
                continue

            X_cal, X_eval, y_cal, y_eval = train_test_split(
                X_val, y_val, test_size=VAL_CALIB_FRAC, stratify=y_val, random_state=VAL_SPLIT_SEED)

            proba_cal = rf.predict_proba(X_cal)[:, 1]
            platt_lr = platt_fit(y_cal, proba_cal)

            proba_eval_raw = rf.predict_proba(X_eval)[:, 1]
            proba_eval_cal = platt_apply(platt_lr, proba_eval_raw)
            thr_cal = find_best_threshold(y_eval, proba_eval_cal)
            val_m = compute_metrics(y_eval, proba_eval_cal, thr_cal)

            logger.info(f"  Val:  F1={val_m['f1']:.4f} "
                         f"P={val_m['precision']:.4f} "
                         f"R={val_m['recall']:.4f} "
                         f"AUC={val_m['auc']:.4f} "
                         f"thr={thr_cal:.4f}")

            del X_cal, X_eval, y_cal, y_eval, proba_cal
            gc.collect()

            # ── Evaluate on test ──────────────────────────────────────────
            test_proba_raw = rf.predict_proba(X_test)[:, 1]
            test_proba_cal = platt_apply(platt_lr, test_proba_raw)

            unique_test = np.unique(y_test)
            if len(unique_test) > 1:
                test_m = compute_metrics(y_test, test_proba_cal, thr_cal)
            else:
                test_m = {"f1": 0, "precision": 0, "recall": 0, "auc": 0,
                          "tp": 0, "fp": 0, "fn": 0, "tn": 0}

            logger.info(f"  Test: F1={test_m['f1']:.4f} "
                         f"P={test_m['precision']:.4f} "
                         f"R={test_m['recall']:.4f} "
                         f"AUC={test_m['auc']:.4f}")

            del rf, platt_lr
            gc.collect()

            # ── Store results ─────────────────────────────────────────────
            row = {
                "ratio": ratio_key,
                "ratio_label": ratio_label,
                "class_weight": cw_label,
                "n_pos_train": n_pos,
                "n_neg_train": n_neg,
                "n_total_train": n_pos + n_neg,
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
            results.append(row)
            _write_csv(out_csv, results)

    # ── Final summary ─────────────────────────────────────────────────────
    section("Summary", out=logger.info)
    logger.info(f"  {'Ratio':<16s} {'Class weight':<20s} {'Val F1':>8s} {'Test F1':>8s}")
    for r in results:
        logger.info(f"  {r['ratio_label']:<16s} {r['class_weight']:<20s} "
                    f"{float(r['val_f1']):>8.4f} {float(r['test_f1']):>8.4f}")
    done(out_csv, out=logger.info)


def _write_csv(path, rows):
    """Write results list to CSV (overwrites)."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()