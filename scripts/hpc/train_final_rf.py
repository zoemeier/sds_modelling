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
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH, LOGS_PATH, MODELS_PATH,
    RESOLUTION, SPATIAL_RES, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    SDS_PATH, FINAL_RF, FINAL_RF_PARAMS, FINAL_RF_TAG,
)
from data_utils import (
    FeatureCache, load_sds_source, compute_land_indices,
    collect_indices, build_train_set, build_eval_set,
    extract_from_cache,
    platt_fit, platt_apply, compute_metrics, find_best_threshold,
)


SDS_PATH = (
    DATA_PATH / "backtrace_overlap_output" / "combination_overlap"
    / "daily_025"
    / "04ae_03duaod_15en-8pm10_0px_01iou_0gap_0stub_0mem"
    / "ndvidep_3to5wsmean_4to7wsmax_015ndvi_025snow_06aepx_upwindplume_k4_median_fbmedian_noduaod_002dae_novol_0clust"
    / "sds_sources_filtered.nc"
)


# ---------------------------------------------------------------------------
# Helpers for saving val/test predictions with coordinates
# ---------------------------------------------------------------------------

def load_year_coords(cache, years, logger):
    """
    Load lat/lon/time per sample from the feature netCDF for each year.
    Returns dict: year -> (lat_array, lon_array, time_array) where each array
    has length cache.n_samples[year] and is indexed by `idx_in_year`.

    Tolerates missing/alternate dim names ("lat" vs "latitude", "valid_time" vs "time").
    """
    year_coords = {}
    for year in years:
        feat_file = cache.feat_dir / f"features_{year}_{cache.ftag}.nc"
        ds = xr.open_dataset(feat_file)

        # tolerate different naming conventions (although final version has standardized names, some older feature files had different names)
        if "lat" in ds:
            lat_v = ds["lat"].values
        elif "latitude" in ds:
            lat_v = ds["latitude"].values
        else:
            raise KeyError(f"No lat/latitude in {feat_file}")

        if "lon" in ds:
            lon_v = ds["lon"].values
        elif "longitude" in ds:
            lon_v = ds["longitude"].values
        else:
            raise KeyError(f"No lon/longitude in {feat_file}")

        if "valid_time" in ds:
            time_v = ds["valid_time"].values
        elif "time" in ds:
            time_v = ds["time"].values
        else:
            raise KeyError(f"No time/valid_time in {feat_file}")

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
        logger.info(f"  Coords for {year}: {len(lat_v):,} rows")
    return year_coords


def make_stype_for_year(year, ds_sds, sds_var, unique_times, land_indices):
    """
    Build flat source-type array for one year, parallel to make_target_for_year
    in data_utils.py. Returns uint8 array of shape (n_times * n_land,)
    where each element is the source_type value (1=genesis, 2=intermediate, 0=neither).
    """
    yr_start, yr_end = f"{year}-01-01", f"{year}-12-31"
    sds_yr = ds_sds.sel(time=slice(yr_start, yr_end))
    sds_yr = sds_yr.reindex(time=unique_times, fill_value=0)

    if "source_type" in sds_yr.data_vars:
        vals = sds_yr["source_type"].values.astype(np.uint8)
    else:
        # fall back: derive a single class from the main variable so we still
        # return the right shape, but every "positive" pixel is class 1
        vals = (sds_yr[sds_var].values > 0).astype(np.uint8)

    n_times = vals.shape[0]
    parts = [vals[ti].ravel()[land_indices] for ti in range(n_times)]
    return np.concatenate(parts)


def build_eval_set_with_coords(cache, pos_list, neg_list, sampling_cfg, rng,
                               year_coords, stype_by_year, logger, split_name):
    """
    Mirror of data_utils.build_eval_set, but also returns aligned
    lats/lons/times/stype arrays. (so predictions are saved with coordintes)
    """
    cap = sampling_cfg.val_test_total_cap
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

    logger.info(f"  {split_name}: {len(p_keep):,} pos / {len(n_keep):,} neg "
                f"({len(p_keep)/len(y)*100:.3f}% prevalence)")
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
                        help=f"Path to SDS file (default: config.SDS_PATH)")
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
                             f"(default: {FINAL_RF.neg_ratio})")
    parser.add_argument("--seed", type=int, default=None,
                        help=f"Override seed (default: {FINAL_RF.seed})")
    parser.add_argument("--pos_cap", type=int, default=None,
                        help=f"Override num_train_pos_cap "
                             f"(default: {FINAL_RF.num_train_pos_cap}, "
                             f"0 = all positives)")
    parser.add_argument("--total_cap", type=int, default=None,
                        help=f"Override num_train_total_cap "
                             f"(default: {FINAL_RF.num_train_total_cap}, "
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
    sds_path    = args.sds_path or SDS_PATH
    tag         = args.tag or FINAL_RF_TAG

    sampling_cfg = FINAL_RF
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
        sampling_cfg = replace(sampling_cfg, **overrides)

    rf_params = dict(FINAL_RF_PARAMS)

    # Validate SDS path
    if sds_path is None:
        print("ERROR: No SDS path provided. Set SDS_PATH in config.py "
              "or pass --sds_path on the command line.")
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
    logger = logging.getLogger("final_rf")

    logger.info("=" * 60)
    logger.info("TRAINING FINAL RF")
    logger.info("=" * 60)
    logger.info(f"  Tag: {tag}")
    logger.info(f"  SDS: {sds_file}")
    logger.info(f"  Resolution: {resolution}, spatial: {spatial_res}")
    logger.info(f"  Train: {train_years}, Val: {val_years}, Test: {test_years}")
    logger.info(f"  Ratio: 1:{sampling_cfg.neg_ratio}")
    logger.info(f"  Pos cap: {sampling_cfg.num_train_pos_cap} "
                f"(0 = all positives)")
    logger.info(f"  Total cap: {sampling_cfg.num_train_total_cap} "
                f"(0 = use pos_cap mode)")
    logger.info(f"  Seed: {sampling_cfg.seed}")
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

    # ── Build per-year stype arrays for val + test (BEFORE closing ds_sds) ──
    logger.info("\nLoading per-year coordinate + stype arrays for val/test ...")
    eval_years = sorted(set(cache.val_years + cache.test_years))
    stype_by_year = {}
    for year in eval_years:
        stype_by_year[year] = make_stype_for_year(
            year, ds_sds, sds_var,
            cache.unique_times[year], land_indices)
        logger.info(f"  Stype for {year}: {len(stype_by_year[year]):,} cells, "
                    f"{int((stype_by_year[year] == 1).sum()):,} genesis, "
                    f"{int((stype_by_year[year] == 2).sum()):,} intermediate")

    # close SDS dataset - we have everything we need
    ds_sds.close()
    del ds_sds
    gc.collect()

    # ── Load lat/lon/time arrays for val + test years ─────────────────────
    year_coords = load_year_coords(cache, eval_years, logger)

    # ── Build datasets ────────────────────────────────────────────────────
    rng = np.random.default_rng(sampling_cfg.seed)

    logger.info("\nBuilding train set ...")
    X_train, y_train, n_pos = build_train_set(
        cache, train_pos, train_neg, sampling_cfg, logger)
    n_neg = len(y_train) - n_pos

    logger.info("Building val set (with coords) ...")
    X_val, y_val, val_lats, val_lons, val_times, val_stype = \
        build_eval_set_with_coords(
            cache, val_pos, val_neg, sampling_cfg, rng,
            year_coords, stype_by_year, logger, "Val")

    logger.info("Building test set (with coords) ...")
    X_test, y_test, test_lats, test_lons, test_times, test_stype = \
        build_eval_set_with_coords(
            cache, test_pos, test_neg, sampling_cfg, rng,
            year_coords, stype_by_year, logger, "Test")

    del train_pos, train_neg, val_pos, val_neg, test_pos, test_neg
    del year_coords, stype_by_year
    gc.collect()

    logger.info(f"\n  Train: {X_train.shape[0]:,} samples "
                f"({n_pos:,} pos / {n_neg:,} neg)")
    logger.info(f"  Val:   {X_val.shape[0]:,} samples "
                f"(coords: {len(val_lats):,}, stype: {len(val_stype):,})")
    logger.info(f"  Test:  {X_test.shape[0]:,} samples "
                f"(coords: {len(test_lats):,}, stype: {len(test_stype):,})")
    logger.info(f"  Features: {X_train.shape[1]}")

    #sanity checks, since i had some issues with alignment in early versions
    assert len(val_lats) == len(y_val), \
        f"val coord/y length mismatch: {len(val_lats)} vs {len(y_val)}"
    assert len(test_lats) == len(y_test), \
        f"test coord/y length mismatch: {len(test_lats)} vs {len(y_test)}"

    # ── Train RF ──────────────────────────────────────────────────────────
    logger.info("\nTraining RF ...")
    t0 = time.time()
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)
    t_train = time.time() - t0
    logger.info(f"  RF trained in {t_train:.1f}s")

    # ── Feature importances ───────────────────────────────────────────────
    imp = rf.feature_importances_
    feat_df = pd.DataFrame({
        "feature": cache.feature_vars,
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    feat_df["rank"] = range(1, len(feat_df) + 1)

    logger.info(f"\n  Top 20 features:")
    for _, row in feat_df.head(20).iterrows():
        logger.info(f"    {row['rank']:3d}. {row['feature']:<35s} "
                     f"{row['importance']:.6f}")

    del X_train, y_train
    gc.collect()

    # ── Platt calibration on validation ───────────────────────────────────
    logger.info("\nCalibrating ...")
    X_cal, X_eval, y_cal, y_eval = train_test_split(
        X_val, y_val, test_size=0.5, stratify=y_val, random_state=42)

    proba_cal = rf.predict_proba(X_cal)[:, 1]
    platt_lr = platt_fit(y_cal, proba_cal)

    proba_eval = platt_apply(platt_lr, rf.predict_proba(X_eval)[:, 1])
    thr = find_best_threshold(y_eval, proba_eval)
    val_metrics = compute_metrics(y_eval, proba_eval, thr)

    del X_cal, X_eval, y_cal, y_eval, proba_cal, proba_eval
    gc.collect()

    #note that these are only on half of the val set since we split the val set for calibration and thresholding/evaluation, the full one is saved to disk below
    logger.info(f"  VAL: F1={val_metrics['f1']:.4f} "
                f"P={val_metrics['precision']:.4f} "
                f"R={val_metrics['recall']:.4f} "
                f"AUC={val_metrics['auc']:.4f} "
                f"thr={thr:.4f}")

    # ── Save full val predictions  ─────────────
    logger.info("\nSaving val predictions ...")
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
    logger.info(f"  Val predictions ({len(y_val):,} samples) "
                f"with coords + stype: {val_pred_path}")
    del full_val_proba

    del X_val, y_val, val_lats, val_lons, val_times, val_stype
    gc.collect()

    # ── Evaluate on test ──────────────────────────────────────────────────
    logger.info("\nEvaluating on test ...")
    test_proba_raw = rf.predict_proba(X_test)[:, 1]
    if platt_lr is not None:
        test_proba = platt_apply(platt_lr, test_proba_raw)
    else:
        test_proba = test_proba_raw

    test_metrics = compute_metrics(y_test, test_proba, thr)

    logger.info(f"  TEST: F1={test_metrics['f1']:.4f} "
                f"P={test_metrics['precision']:.4f} "
                f"R={test_metrics['recall']:.4f} "
                f"AUC={test_metrics['auc']:.4f}")

    # ── Save test predictions  ───────────────────────────────
    logger.info("Saving test predictions ...")
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
    logger.info(f"  Test predictions ({len(y_test):,} samples) "
                f"with coords + stype: {test_pred_path}")

    # ── PR curve + metrics at various thresholds ──────────────────────────
    from sklearn.metrics import precision_recall_curve, roc_curve

    logger.info("\nComputing PR and ROC curves ...")

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

    # Metrics at a range of fixed thresholds for quick reference
    fixed_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                        0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
    threshold_table = []
    for t in fixed_thresholds:
        m = compute_metrics(y_test, test_proba, t)
        threshold_table.append(m)

    threshold_df = pd.DataFrame(threshold_table)

    logger.info(f"\n  Metrics at selected thresholds (test set):")
    logger.info(f"  {'Thr':>6s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s} "
                f"{'TP':>7s} {'FP':>7s} {'FN':>7s} {'TN':>7s}")
    logger.info(f"  {'-'*55}")
    for _, r in threshold_df.iterrows():
        logger.info(f"  {r['threshold']:6.2f} {r['f1']:7.4f} "
                    f"{r['precision']:7.4f} {r['recall']:7.4f} "
                    f"{r['tp']:7.0f} {r['fp']:7.0f} "
                    f"{r['fn']:7.0f} {r['tn']:7.0f}")

    del X_test, y_test, test_proba, test_proba_raw
    del test_lats, test_lons, test_times, test_stype
    gc.collect()

    # ── Save everything ───────────────────────────────────────────────────
    logger.info(f"\nSaving to {out_dir} ...")

    # 1. Model
    model_path = out_dir / "rf_model.joblib"
    joblib.dump(rf, model_path)
    logger.info(f"  Model: {model_path}")

    # 2. Platt scaler
    if platt_lr is not None:
        platt_path = out_dir / "platt_scaler.joblib"
        joblib.dump(platt_lr, platt_path)
        logger.info(f"  Platt scaler: {platt_path}")

    # 3. Feature importances
    feat_path = out_dir / "feature_importances.csv"
    feat_df.to_csv(feat_path, index=False)
    logger.info(f"  Feature importances: {feat_path}")

    # 4. PR curve
    pr_path = out_dir / "pr_curve.csv"
    pr_df.to_csv(pr_path, index=False)
    logger.info(f"  PR curve ({len(pr_df):,} points): {pr_path}")

    # 5. ROC curve
    roc_path = out_dir / "roc_curve.csv"
    roc_df.to_csv(roc_path, index=False)
    logger.info(f"  ROC curve ({len(roc_df):,} points): {roc_path}")

    # 6. Metrics at fixed thresholds
    thr_table_path = out_dir / "metrics_by_threshold.csv"
    threshold_df.to_csv(thr_table_path, index=False)
    logger.info(f"  Threshold table: {thr_table_path}")

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
        "sampling_config": asdict(sampling_cfg),
        "data_config": {
            "resolution": resolution,
            "spatial_res": spatial_res,
            "train_years": train_years,
            "val_years": val_years,
            "test_years": test_years,
            "sds_path": str(sds_path),
        },
        "val_predictions_path": str(val_pred_path),
        "test_predictions_path": str(test_pred_path),
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"  Results: {results_path}")

    # ── Done ──────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("FINAL RF TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Output dir: {out_dir}")
    logger.info(f"  Val  F1={val_metrics['f1']:.4f}  "
                f"P={val_metrics['precision']:.4f}  "
                f"R={val_metrics['recall']:.4f}")
    logger.info(f"  Test F1={test_metrics['f1']:.4f}  "
                f"P={test_metrics['precision']:.4f}  "
                f"R={test_metrics['recall']:.4f}")
    logger.info(f"  Threshold: {thr:.4f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()