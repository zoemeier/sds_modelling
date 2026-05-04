"""
Bayesian optimisation of Random Forest hyperparameters for SDS prediction.
defaults are read from config.py, but every parameter can be overridden via CLI flags.

Usage:
  python optimize_rf.py                                         # all defaults
  python optimize_rf.py --sds_path /path/to/sds --n_trials 100  # overrides
"""

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import (
    LOGS_PATH, BAYESIAN_HPO, SUBSAMPLE_SWEEP,
    RESOLUTION, SPATIAL_RES, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    SDS_PATH, HPO_STUDY_NAME, HPO_N_TRIALS, HPO_N_STARTUP,
)
from data_utils import (
    FeatureCache, load_sds_source, compute_land_indices,
    collect_indices, build_train_set, build_eval_set,
    platt_fit, platt_apply, compute_metrics, find_best_threshold,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian optimisation of RF hyperparameters")

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

    # Optuna settings — defaults from config
    parser.add_argument("--study_name", type=str, default=None,
                        help=f"Optuna study name (default: {HPO_STUDY_NAME})")
    parser.add_argument("--n_trials", type=int, default=None,
                        help=f"Number of Optuna trials (default: {HPO_N_TRIALS})")
    parser.add_argument("--n_startup", type=int, default=None,
                        help=f"Random startup trials (default: {HPO_N_STARTUP})")

    # Sampling config selector + overrides
    parser.add_argument("--config_name", type=str, default="bayesian_hpo",
                        choices=["bayesian_hpo", "subsample_sweep"],
                        help="Which SamplingConfig to use from config.py")
    parser.add_argument("--ratio", type=int, default=None,
                        help=f"Override neg:pos ratio "
                             f"(default: {BAYESIAN_HPO.neg_ratio})")
    parser.add_argument("--pos_cap", type=int, default=None,
                        help=f"Override num_train_pos_cap "
                             f"(default: {BAYESIAN_HPO.num_train_pos_cap})")
    parser.add_argument("--total_cap", type=int, default=None,
                        help=f"Override num_train_total_cap "
                             f"(default: {BAYESIAN_HPO.num_train_total_cap})")
    parser.add_argument("--val_test_cap", type=int, default=None,
                        help=f"Override val_test_total_cap "
                             f"(default: {BAYESIAN_HPO.val_test_total_cap})")
    parser.add_argument("--seed", type=int, default=None,
                        help=f"Override seed "
                             f"(default: {BAYESIAN_HPO.seed})")
    args = parser.parse_args()

    # ── Resolve all parameters from config + CLI overrides ────────────────
    resolution  = args.resolution or RESOLUTION
    spatial_res = args.spatial or SPATIAL_RES
    train_years = args.train_years or TRAIN_YEARS
    val_years   = args.val_years or VAL_YEARS
    test_years  = args.test_years or TEST_YEARS
    sds_path    = args.sds_path or SDS_PATH
    study_name  = args.study_name or HPO_STUDY_NAME
    n_trials    = args.n_trials if args.n_trials is not None else HPO_N_TRIALS
    n_startup   = args.n_startup if args.n_startup is not None else HPO_N_STARTUP

    # Build sampling config with overrides
    if args.config_name == "bayesian_hpo":
        sampling_cfg = BAYESIAN_HPO
    else:
        sampling_cfg = SUBSAMPLE_SWEEP

    overrides = {}
    if args.ratio is not None:
        overrides["neg_ratio"] = args.ratio
    if args.pos_cap is not None:
        overrides["num_train_pos_cap"] = args.pos_cap
    if args.total_cap is not None:
        overrides["num_train_total_cap"] = args.total_cap
    if args.val_test_cap is not None:
        overrides["val_test_total_cap"] = args.val_test_cap
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        sampling_cfg = replace(sampling_cfg, **overrides)

    # Validate SDS path
    if sds_path is None:
        print("ERROR: No SDS path provided. Set SDS_PATH in config.py "
              "or pass --sds_path on the command line.")
        sys.exit(1)
    sds_file = Path(sds_path)
    if not sds_file.exists():
        print(f"ERROR: SDS file not found: {sds_file}")
        sys.exit(1)

    # ── Setup ─────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOGS_PATH / "rf_opt"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"rf_opt_{study_name}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    logger = logging.getLogger("rf_opt")

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

    # ── Build datasets (once) ─────────────────────────────────────────────
    rng = np.random.default_rng(sampling_cfg.seed)

    logger.info("\nBuilding train set ...")
    X_train, y_train, n_pos = build_train_set(
        cache, train_pos, train_neg, sampling_cfg, logger)

    logger.info("Building val set ...")
    X_val, y_val = build_eval_set(
        cache, val_pos, val_neg, sampling_cfg, rng, logger, "Val")

    logger.info("Building test set ...")
    X_test, y_test = build_eval_set(
        cache, test_pos, test_neg, sampling_cfg, rng, logger, "Test")

    del train_pos, train_neg, val_pos, val_neg, test_pos, test_neg
    gc.collect()

    n_features = X_train.shape[1]
    logger.info(f"\n  Features: {n_features}")
    logger.info(f"  Train: {X_train.shape[0]:,}")
    logger.info(f"  Ratio: 1:{sampling_cfg.neg_ratio}")
    logger.info(f"  Seed: {sampling_cfg.seed}")

    # ── Optuna objective ──────────────────────────────────────────────────
    results_log = []
    _csv_loaded = [False]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 10, 80),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30, log=True),
            "max_features": trial.suggest_categorical("max_features",
                                                       ["sqrt", "log2", 0.1, 0.2, 0.3, 0.5]),
            "max_samples": trial.suggest_float("max_samples", 0.3, 1.0),
            "bootstrap": True,
            "class_weight": None,
            "n_jobs": -1,
            "random_state": sampling_cfg.seed,
            "verbose": 0,
        }

        logger.info(f"\n{'#'*60}")
        logger.info(f"TRIAL {trial.number}")
        logger.info(f"{'#'*60}")
        for k, v in sorted(params.items()):
            if k in ("bootstrap", "class_weight", "n_jobs", "random_state", "verbose"):
                continue
            logger.info(f"  {k}: {v}")

        t0 = time.time()
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)
        t_train = time.time() - t0
        logger.info(f"  RF trained in {t_train:.1f}s")

        imp = rf.feature_importances_
        for rank, i in enumerate(np.argsort(imp)[::-1][:3], 1):
            logger.info(f"    {rank}. {cache.feature_vars[i]:<25s} {imp[i]:.4f}")

        # Evaluate on val with Platt calibration
        unique_val = np.unique(y_val)
        if len(unique_val) < 2:
            logger.warning("  Val has only one class")
            return 0.0

        X_cal, X_eval, y_cal, y_eval = train_test_split(
            X_val, y_val, test_size=0.5, stratify=y_val, random_state=42)

        proba_cal = rf.predict_proba(X_cal)[:, 1]
        platt_lr = platt_fit(y_cal, proba_cal)

        proba_eval = platt_apply(platt_lr, rf.predict_proba(X_eval)[:, 1])
        thr = find_best_threshold(y_eval, proba_eval)
        val_m = compute_metrics(y_eval, proba_eval, thr)

        # Test
        test_proba = platt_apply(platt_lr, rf.predict_proba(X_test)[:, 1])
        test_m = compute_metrics(y_test, test_proba, thr)

        logger.info(f"  VAL:  F1={val_m['f1']:.4f} P={val_m['precision']:.4f} "
                    f"R={val_m['recall']:.4f} AUC={val_m['auc']:.4f} thr={thr:.4f}")
        logger.info(f"  TEST: F1={test_m['f1']:.4f} P={test_m['precision']:.4f} "
                    f"R={test_m['recall']:.4f} AUC={test_m['auc']:.4f}")

        del rf, platt_lr
        gc.collect()

        row = {
            "trial": trial.number,
            "val_f1": round(val_m["f1"], 4),
            "val_precision": round(val_m["precision"], 4),
            "val_recall": round(val_m["recall"], 4),
            "val_auc": round(val_m["auc"], 4),
            "test_f1": round(test_m["f1"], 4),
            "test_precision": round(test_m["precision"], 4),
            "test_recall": round(test_m["recall"], 4),
            "test_auc": round(test_m["auc"], 4),
            "threshold": round(thr, 4),
            "train_time_s": round(t_train, 1),
            **{k: v for k, v in params.items()
               if k not in ("bootstrap", "class_weight", "n_jobs",
                             "random_state", "verbose")},
        }
        results_log.append(row)

        csv_path = log_dir / f"rf_opt_results_{study_name}.csv"
        if not _csv_loaded[0]:
            if csv_path.exists():
                existing = pd.read_csv(csv_path)
                results_log[:0] = existing.to_dict("records")
                logger.info(f"  Loaded {len(existing)} existing rows from CSV")
            _csv_loaded[0] = True
        pd.DataFrame(results_log).to_csv(csv_path, index=False)

        return val_m["f1"]

    # ── Run optimisation ──────────────────────────────────────────────────
    db_path = log_dir / f"study_{study_name}.db"
    storage = f"sqlite:///{db_path}"
    sampler = TPESampler(seed=sampling_cfg.seed, n_startup_trials=n_startup)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        logger.info(f"\n  Resuming: {n_existing} existing trials")
        logger.info(f"  Best so far: {study.best_value:.4f}")

    logger.info(f"\n{'='*60}")
    logger.info("RF HYPERPARAMETER OPTIMISATION")
    logger.info(f"{'='*60}")
    logger.info(f"  SDS: {sds_file}")
    logger.info(f"  Resolution: {resolution}, spatial: {spatial_res}")
    logger.info(f"  Train: {train_years}, Val: {val_years}, Test: {test_years}")
    logger.info(f"  Study: {study_name}")
    logger.info(f"  Trials: {n_trials}")
    logger.info(f"  Config: {args.config_name}")
    logger.info(f"  Ratio: 1:{sampling_cfg.neg_ratio}")
    if sampling_cfg.num_train_total_cap > 0:
        logger.info(f"  Train total cap: {sampling_cfg.num_train_total_cap:,}")
    else:
        logger.info(f"  Train pos cap: {sampling_cfg.num_train_pos_cap:,}")
    logger.info(f"  Val/test cap: {sampling_cfg.val_test_total_cap:,}")
    logger.info(f"  Seed: {sampling_cfg.seed}")
    logger.info(f"  Startup: {n_startup}")
    logger.info(f"{'='*60}")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ── Final report ──────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("OPTIMISATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Best F1: {study.best_value:.4f}")
    logger.info(f"  Best params:")
    for k, v in study.best_params.items():
        logger.info(f"    {k}: {v}")

    summary = {
        "study_name": study_name,
        "best_f1": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "sampling_config": {
            "neg_ratio": sampling_cfg.neg_ratio,
            "num_train_total_cap": sampling_cfg.num_train_total_cap,
            "num_train_pos_cap": sampling_cfg.num_train_pos_cap,
            "val_test_total_cap": sampling_cfg.val_test_total_cap,
            "seed": sampling_cfg.seed,
        },
        "sds_path": str(sds_path),
    }
    summary_path = log_dir / f"rf_opt_summary_{study_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    logger.info(f"\n  Top 5:")
    for t in sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:5]:
        logger.info(f"    #{t.number}: F1={t.value:.4f}  {t.params}")

    logger.info(f"\n  CSV: {log_dir / f'rf_opt_results_{study_name}.csv'}")
    logger.info(f"  DB:  {db_path}")
    logger.info(f"  Log: {log_file}")


if __name__ == "__main__":
    main()