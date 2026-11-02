"""
Train the final Random Forest with optimal hyperparameters (from HPO) and
optimal subsampling ratio (from sweep) on the full training pool.

Saves:
  - Trained RF (joblib)
  - Platt scaling model (joblib)
  - Feature importances (CSV)
  - PR curve, ROC curve, metrics at various thresholds (CSV)
  - Full metrics for val + test (JSON)
  - Optimal threshold
  - Config snapshot for reproducibility
  - Val + test predictions (npz): y, proba, lats, lons, times, stype
    (so downstream scripts don't have to reconstruct coordinates from
    feature files + land mask).

Defaults come from config.py and are overridable via CLI.
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
import xarray as xr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sds.config import (
    VAL_CALIB_FRAC, VAL_SPLIT_SEED, MODELS_PATH, RESOLUTION,
    SPATIAL_RES, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    FINAL_RF_SAMPLING, FINAL_RF_PARAMS, FINAL_RF_TAG,
)
from sds.paths import resolve_path, project_relative, SDS_PATH
from sds.log_utils import header, section, done
from sds.data_utils import (
    FeatureCache, load_sds_source, compute_land_indices, make_stype_for_year,
    collect_indices, build_train_set,
    extract_from_cache,
    platt_fit, platt_apply, compute_metrics, find_best_threshold,
)


# ---------------------------------------------------------------------------
# Helpers for saving val/test predictions with coordinates
# ---------------------------------------------------------------------------

def load_year_coords(cache, years, logger):
    """
    Load lat/lon/time per sample from the feature netCDF for each year.
    Returns dict: year -> (lat_array, lon_array, time_array) where each array
    has length cache.n_samples[year] and is indexed by `idx_in_year`.
    """
    year_coords = {}
    for year in years:
        feat_file = cache.feat_dir / f"features_{year}_{cache.ftag}.nc"
        ds = xr.open_dataset(feat_file)

        lat_v = ds["lat"].values
        lon_v = ds["lon"].values
        time_v = ds["valid_time"].values

        n_expected = cache.n_samples[year]
        if len(lat_v) != n_expected or len(lon_v) != n_expected or len(time_v) != n_expected:
            raise ValueError(
                f"Year {year}: feature coord lengths "
                f"(lat={len(lat_v)}, lon={len(lon_v)}, time={len(time_v)}) "
                f"do not match cache.n_samples[{year}]={n_expected}")

        lat_v = lat_v.astype(np.float32)
        lon_v = lon_v.astype(np.float32)
        time_v = np.asarray(time_v, dtype="datetime64[ns]")

        year_coords[year] = (lat_v, lon_v, time_v)
        ds.close()
    return year_coords


def build_eval_set_with_coords(cache, pos_list, neg_list, sampling_cfg, rng,
                               year_coords, stype_by_year, logger, split_name):
    """
    Mirror of data_utils.build_eval_set, but also returns aligned
    lats/lons/times/stype arrays. (so predictions are saved with coordintes)
    """
    cap = sampling_cfg['val_test_total_cap']
    total = len(pos_list) + len(neg_list)

    if cap > 0 and total > cap:
        pos_frac = len(pos_list) / total
        n_p = max(1, int(round(cap * pos_frac)))
        n_n = cap - n_p
        n_p = min(n_p, len(pos_list))
        n_n = min(n_n, len(neg_list))
    else:
        n_p, n_n = len(pos_list), len(neg_list)

    p_keep = pos_list if len(pos_list) <= n_p else \
        [pos_list[i] for i in rng.choice(len(pos_list), n_p, replace=False)]
    n_keep = neg_list if len(neg_list) <= n_n else \
        [neg_list[i] for i in rng.choice(len(neg_list), n_n, replace=False)]

    indices = p_keep + n_keep
    n_total = len(indices)

    X = extract_from_cache(cache, indices)
    y = np.concatenate([np.ones(len(p_keep), np.float32),
                        np.zeros(len(n_keep), np.float32)])

    # build coordinate arrays in the same row order as X/y
    lats = np.empty(n_total, dtype=np.float32)
    lons = np.empty(n_total, dtype=np.float32)
    times = np.empty(n_total, dtype="datetime64[ns]")
    stype = np.empty(n_total, dtype=np.uint8)

    for i, (year, idx) in enumerate(indices):
        lat_v, lon_v, time_v = year_coords[year]
        lats[i] = lat_v[idx]
        lons[i] = lon_v[idx]
        times[i] = time_v[idx]
        stype[i] = stype_by_year[year][idx]

    logger.info(f"  {split_name} set: {len(p_keep):,} pos, {len(n_keep):,} neg "
                f"({len(p_keep)/len(y)*100:.2f}% pos)")
    return X, y, lats, lons, times, stype


def save_predictions_npz(path, y, proba, lats, lons, times, stype):
    np.savez_compressed(
        path,
        y=y.astype(np.float32),
        proba=proba.astype(np.float32),
        lats=lats.astype(np.float32),
        lons=lons.astype(np.float32),
        times=times.astype("datetime64[ns]").astype(np.int64),
        stype=stype.astype(np.uint8),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train final RF with optimal parameters")

    parser.add_argument("--sds_path", type=str, default=None,
                        help="Path to SDS file (default: config.SDS_PATH)")
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
    parser.add_argument("--ratio", type=int, default=None,
                        help=f"Override neg:pos ratio "
                             f"(default: {FINAL_RF_SAMPLING['neg_ratio']})")
    parser.add_argument("--seed", type=int, default=None,
                        help=f"Override seed (default: {FINAL_RF_SAMPLING['seed']})")
    parser.add_argument("--pos_cap", type=int, default=None,
                        help=f"Override num_train_pos_cap "
                             f"(default: {FINAL_RF_SAMPLING['num_train_pos_cap']}, "
                             f"0 = all positives)")
    parser.add_argument("--total_cap", type=int, default=None,
                        help=f"Override num_train_total_cap "
                             f"(default: {FINAL_RF_SAMPLING['num_train_total_cap']}, "
                             f"0 = use pos_cap mode)")
    parser.add_argument("--tag", type=str, default=None,
                        help=f"Output tag (default: {FINAL_RF_TAG})")
    args = parser.parse_args()

    # ── Resolve parameters ────────────────────────────────────────────────
    resolution  = args.resolution or RESOLUTION
    spatial_res = args.spatial or SPATIAL_RES
    train_years = args.train_years or TRAIN_YEARS
    val_years   = args.val_years or VAL_YEARS
    test_years  = args.test_years or TEST_YEARS
    sds_path    = resolve_path(args.sds_path or SDS_PATH)
    tag         = args.tag or FINAL_RF_TAG

    sampling_cfg = FINAL_RF_SAMPLING
    overrides = {}
    if args.ratio is not None:
        overrides["neg_ratio"] = args.ratio
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.pos_cap is not None:
        overrides["num_train_pos_cap"] = args.pos_cap
    if args.total_cap is not None:
        overrides["num_train_total_cap"] = args.total_cap
    if overrides:
        sampling_cfg = {**sampling_cfg, **overrides}

    rf_params = dict(FINAL_RF_PARAMS)

    sds_file = Path(sds_path)
    if not sds_file.exists():
        sys.exit(f"ERROR: labels not found: {sds_file} (run the backtracking first)")

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
    logger = logging.getLogger("final_rf")

    header("Final random forest", {
        "Labels": project_relative(sds_file),
        "Resolution": f"{resolution}, {spatial_res}°",
        "Train / val / test": f"{train_years} / {val_years} / {test_years}",
        "Sampling": f"1:{sampling_cfg['neg_ratio']}, pos cap {sampling_cfg['num_train_pos_cap']}, "
                    f"total cap {sampling_cfg['num_train_total_cap']}, seed {sampling_cfg['seed']}",
        "RF params": ", ".join(f"{k}={v}" for k, v in sorted(rf_params.items())
                               if k not in ("n_jobs", "verbose")),
        "Output": project_relative(out_dir),
    }, out=logger.info)

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

    section("Samples", out=logger.info)
    train_pos, train_neg = collect_indices(
        cache, cache.train_years, ds_sds, sds_var, land_indices, logger, label="Train")

    val_pos, val_neg = collect_indices(
        cache, cache.val_years, ds_sds, sds_var, land_indices, logger, label="Val")

    test_pos, test_neg = collect_indices(
        cache, cache.test_years, ds_sds, sds_var, land_indices, logger, label="Test")

    # ── Build per-year stype arrays for val + test (BEFORE closing ds_sds) ──
    eval_years = sorted(set(cache.val_years + cache.test_years))
    stype_by_year = {}
    for year in eval_years:
        stype_by_year[year] = make_stype_for_year(
            year, ds_sds, cache.unique_times[year], land_indices)

    # close the SDS dataset
    ds_sds.close()
    del ds_sds
    gc.collect()

    # ── Load lat/lon/time arrays for val + test years ─────────────────────
    year_coords = load_year_coords(cache, eval_years, logger)

    # ── Build datasets ────────────────────────────────────────────────────
    rng = np.random.default_rng(sampling_cfg['seed'])

    X_train, y_train, n_pos = build_train_set(
        cache, train_pos, train_neg, sampling_cfg, logger)
    n_neg = len(y_train) - n_pos

    X_val, y_val, val_lats, val_lons, val_times, val_stype = \
        build_eval_set_with_coords(
            cache, val_pos, val_neg, sampling_cfg, rng,
            year_coords, stype_by_year, logger, "Val")

    X_test, y_test, test_lats, test_lons, test_times, test_stype = \
        build_eval_set_with_coords(
            cache, test_pos, test_neg, sampling_cfg, rng,
            year_coords, stype_by_year, logger, "Test")

    del train_pos, train_neg, val_pos, val_neg, test_pos, test_neg
    del year_coords, stype_by_year
    gc.collect()

    # check that features, labels and coordinates are aligned
    assert len(val_lats) == len(y_val), \
        f"val coord/y length mismatch: {len(val_lats)} vs {len(y_val)}"
    assert len(test_lats) == len(y_test), \
        f"test coord/y length mismatch: {len(test_lats)} vs {len(y_test)}"

    # ── Train RF ──────────────────────────────────────────────────────────
    section("Train", out=logger.info)
    t0 = time.time()
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    t_train = time.time() - t0
    logger.info(f"  Trained in {t_train / 60:.1f} min")

    # ── Feature importances ───────────────────────────────────────────────
    imp = rf.feature_importances_
    feat_df = pd.DataFrame({
        "feature": cache.feature_vars,
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    feat_df["rank"] = range(1, len(feat_df) + 1)

    logger.info("  Top 5 features (MDI): " + ", ".join(feat_df["feature"].head(5)))

    del X_train, y_train
    gc.collect()

    # ── Platt calibration on validation ───────────────────────────────────
    section("Calibrate and evaluate", out=logger.info)
    X_cal, X_eval, y_cal, y_eval = train_test_split(
        X_val, y_val, test_size=VAL_CALIB_FRAC, stratify=y_val, random_state=VAL_SPLIT_SEED)

    proba_cal = rf.predict_proba(X_cal)[:, 1]
    platt_lr = platt_fit(y_cal, proba_cal)

    proba_eval = platt_apply(platt_lr, rf.predict_proba(X_eval)[:, 1])
    thr = find_best_threshold(y_eval, proba_eval)
    val_metrics = compute_metrics(y_eval, proba_eval, thr)

    del X_cal, X_eval, y_cal, y_eval, proba_cal, proba_eval
    gc.collect()

    # metrics on the evaluation half of val (the other half fits Platt); full val predictions are saved below
    logger.info(f"  Val:  F1={val_metrics['f1']:.4f} "
                f"P={val_metrics['precision']:.4f} "
                f"R={val_metrics['recall']:.4f} "
                f"AUC={val_metrics['auc']:.4f} "
                f"thr={thr:.4f}")

    # save full val predictions
    full_val_proba = platt_apply(platt_lr, rf.predict_proba(X_val)[:, 1])
    val_pred_path = out_dir / "val_predictions.npz"
    save_predictions_npz(
        val_pred_path,
        y=y_val,
        proba=full_val_proba,
        lats=val_lats,
        lons=val_lons,
        times=val_times,
        stype=val_stype,
    )
    del full_val_proba

    del X_val, y_val, val_lats, val_lons, val_times, val_stype
    gc.collect()

    # ── Evaluate on test ──────────────────────────────────────────────────
    test_proba_raw = rf.predict_proba(X_test)[:, 1]
    if platt_lr is not None:
        test_proba = platt_apply(platt_lr, test_proba_raw)
    else:
        test_proba = test_proba_raw

    test_metrics = compute_metrics(y_test, test_proba, thr)

    logger.info(f"  Test: F1={test_metrics['f1']:.4f} "
                f"P={test_metrics['precision']:.4f} "
                f"R={test_metrics['recall']:.4f} "
                f"AUC={test_metrics['auc']:.4f}")

    # save test predictions
    test_pred_path = out_dir / "test_predictions.npz"
    save_predictions_npz(
        test_pred_path,
        y=y_test,
        proba=test_proba,
        lats=test_lats,
        lons=test_lons,
        times=test_times,
        stype=test_stype,
    )

    # ── PR curve + metrics at various thresholds ──────────────────────────
    from sklearn.metrics import precision_recall_curve, roc_curve

    # PR curve
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, test_proba)
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)

    pr_df = pd.DataFrame({
        "threshold": thr_arr,
        "precision": prec_arr[:-1],
        "recall": rec_arr[:-1],
        "f1": f1_arr,
    })

    # ROC curve
    fpr_arr, tpr_arr, roc_thr_arr = roc_curve(y_test, test_proba)
    roc_df = pd.DataFrame({
        "threshold": roc_thr_arr,
        "fpr": fpr_arr,
        "tpr": tpr_arr,
    })

    # metrics at fixed thresholds
    fixed_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                        0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
    threshold_table = []
    for t in fixed_thresholds:
        m = compute_metrics(y_test, test_proba, t)
        threshold_table.append(m)

    threshold_df = pd.DataFrame(threshold_table)

    del X_test, y_test, test_proba, test_proba_raw
    del test_lats, test_lons, test_times, test_stype
    gc.collect()

    # ── Save everything ───────────────────────────────────────────────────
    # 1. Model
    model_path = out_dir / "rf_model.joblib"
    joblib.dump(rf, model_path)

    # 2. Platt scaler
    if platt_lr is not None:
        platt_path = out_dir / "platt_scaler.joblib"
        joblib.dump(platt_lr, platt_path)

    # 3. Feature importances
    feat_path = out_dir / "feature_importances.csv"
    feat_df.to_csv(feat_path, index=False)

    # 4. PR curve
    pr_path = out_dir / "pr_curve.csv"
    pr_df.to_csv(pr_path, index=False)

    # 5. ROC curve
    roc_path = out_dir / "roc_curve.csv"
    roc_df.to_csv(roc_path, index=False)

    # 6. Metrics at fixed thresholds
    thr_table_path = out_dir / "metrics_by_threshold.csv"
    threshold_df.to_csv(thr_table_path, index=False)

    # 7. Full results JSON
    results = {
        "tag": tag,
        "timestamp": timestamp,
        "train_time_s": round(t_train, 1),
        "threshold": thr,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "n_train": n_pos + n_neg,
        "n_pos_train": n_pos,
        "n_neg_train": n_neg,
        "n_features": len(cache.feature_vars),
        "feature_vars": cache.feature_vars,
        "rf_params": {k: str(v) for k, v in rf_params.items()},
        "sampling_config": dict(sampling_cfg),
        "data_config": {
            "resolution": resolution,
            "spatial_res": spatial_res,
            "train_years": train_years,
            "val_years": val_years,
            "test_years": test_years,
            "sds_path": project_relative(sds_path),
        },
        "val_predictions_path": project_relative(val_pred_path),
        "test_predictions_path": project_relative(test_pred_path),
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    done(out_dir, out=logger.info)


if __name__ == "__main__":
    main()