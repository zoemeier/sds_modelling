"""
Shared data loading, feature caching, index collection, subsampling, and
evaluation utilities for the SDS modelling pipeline.

Mainly used for efficiency when running multiple models on the same dataset
(e.g. subsample_ratio_experiment).
"""

import gc
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

from config import (
    OUT_PATH,
    file_tag, should_drop_feature,
    LAND_MASK_FILE, LSM_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Feature Cache
# ---------------------------------------------------------------------------

class FeatureCache:
    """
    Preloads all feature files into memory, stores per-year unique times and
    a pre-computed NaN mask. Populated fully at construction time — after
    __init__ returns, self.X, self.unique_times, self.finite_mask etc. are ready.
    """

    def __init__(self, resolution, spatial_res, train_years=None, val_years=None,
                 test_years=None, logger=None):
        self.resolution = resolution
        self.spatial_res = spatial_res
        self.ftag = file_tag(resolution, spatial_res)
        self.feat_dir = OUT_PATH / self.ftag
        self.logger = logger or logging.getLogger("feature_cache")

        self.train_years = list(range(train_years[0], train_years[1] + 1)) if train_years else []
        self.val_years = list(range(val_years[0], val_years[1] + 1)) if val_years else []
        self.test_years = list(range(test_years[0], test_years[1] + 1)) if test_years else []
        self.all_years = sorted(set(
            self.train_years + self.val_years + self.test_years
        ))

        self.X = {}               # year -> (n_samples, n_features) float32
        self.n_samples = {}       # year -> int
        self.unique_times = {}    # year -> np.ndarray datetime64
        self.finite_mask = {}     # year -> np.ndarray bool (True = usable)
        self.n_land = None        # land cells per timestep
        self.feature_vars = None

        self._load_all()

    def _load_all(self):
        self.logger.info("=" * 60)
        self.logger.info("LOADING ALL FEATURES INTO MEMORY")
        self.logger.info("=" * 60)

        total_bytes = 0   # only for tracking memory purposes
        META_VARS = {"lat", "lon", "valid_time",
                     "latitude", "longitude", "SDS_SOURCE"}

        for year in self.all_years:
            feat_file = self.feat_dir / f"features_{year}_{self.ftag}.nc"
            if not feat_file.exists():
                self.logger.error(f"  Feature file missing: {feat_file}")
                raise FileNotFoundError(feat_file)

            self.logger.info(f"  Loading {year} ...")
            ds = xr.open_dataset(feat_file).load()

            # discover feature list from the first year - subsequent years are
            # assumed to share the same feature set
            if self.feature_vars is None:
                self.feature_vars = [
                    v for v in ds.data_vars
                    if v not in META_VARS and not should_drop_feature(v)
                ]
                self.logger.info(f"  Features: {len(self.feature_vars)}")

            n = ds.sizes["sample"]

            # unique times for alignment with SDS targets later
            times = pd.to_datetime(ds["valid_time"].values)
            self.unique_times[year] = np.unique(times)

            # n_land is derived once from the first year and trusted thereafter
            if self.n_land is None:
                self.n_land = n // len(self.unique_times[year])
                self.logger.info(f"  Land cells per timestep: {self.n_land:,}")

            # stack features into a dense float32 array
            X = np.empty((n, len(self.feature_vars)), dtype=np.float32)
            for i, v in enumerate(self.feature_vars):
                X[:, i] = ds[v].values.astype(np.float32)

            # pre-compute the NaN mask once per year so downstream code can skip the check
            mask = np.all(np.isfinite(X), axis=1)
            n_nan = int((~mask).sum())
            self.finite_mask[year] = mask
            if n_nan > 0:
                self.logger.info(f"    {n_nan:,} NaN rows flagged")

            self.X[year] = X
            self.n_samples[year] = n
            total_bytes += X.nbytes

            ds.close()
            del ds
            gc.collect()

        total_gb = total_bytes / (1024**3)
        total_samples = sum(self.n_samples.values())
        self.logger.info(f"\n  Total: {total_samples:,} samples, "
                         f"{len(self.all_years)} years, {total_gb:.1f} GB")
        self.logger.info("=" * 60)


# ---------------------------------------------------------------------------
# SDS / land mask helpers
# ---------------------------------------------------------------------------

def load_sds_source(sds_file, logger):
    """Load SDS source file into memory. Returns (ds, var_name)."""
    logger.info(f"  Loading SDS source: {sds_file.name}")
    ds = xr.open_dataset(sds_file)
    # normalise the variety of dim/coord names that show up in different sources
    for old, new in [("valid_time", "time"), ("lat", "latitude"),
                     ("lon", "longitude")]:
        if old in ds.dims and new not in ds.dims:
            ds = ds.rename({old: new})
        if old in ds.coords and new not in ds.coords:
            ds = ds.rename({old: new})
    ds = ds.load()
    sds_var = "SDS_SOURCE"
    logger.info(f"  SDS var: {sds_var}, times: {len(ds.time)}")
    return ds, sds_var


def compute_land_indices(sds_lat, sds_lon):
    """Compute land cell indices in the ravelled (lat, lon) grid."""
    # try the NetCDF version first (no cfgrib dependency), fall back to GRIB
    lsm_nc = LAND_MASK_FILE.with_suffix(".nc")
    if lsm_nc.exists():
        lsm_ds = xr.open_dataset(lsm_nc).squeeze()
    else:
        lsm_ds = xr.open_dataset(LAND_MASK_FILE, engine="cfgrib").squeeze()
    lsm_var = next((v for v in ("lsm", "land_sea_mask", "LSM") if v in lsm_ds),
                   list(lsm_ds.data_vars)[0])
    # resample the land mask onto the SDS grid by nearest neighbour
    lsm = lsm_ds[lsm_var].sel(
        latitude=sds_lat, longitude=sds_lon, method="nearest",
    )
    mask_2d = (lsm.values > LSM_THRESHOLD).astype(bool)
    lsm_ds.close()
    return np.where(mask_2d.ravel())[0]


def make_target_for_year(year, ds_sds, sds_var, unique_times, land_indices):
    """
    Build flat target array for one year from the in-memory SDS dataset.
    Returns float32 array of shape (n_times * n_land,) - rows laid out as
    [day0_cells ... day1_cells ... dayN_cells], matching cache.X[year]'s ordering.
    """
    yr_start, yr_end = f"{year}-01-01", f"{year}-12-31"
    sds_yr = ds_sds.sel(time=slice(yr_start, yr_end))
    sds_yr = sds_yr.reindex(time=unique_times, fill_value=0)

    vals = (sds_yr[sds_var].values > 0).astype(np.float32)
    n_times = vals.shape[0]

    parts = []
    for ti in range(n_times):
        parts.append(vals[ti].ravel()[land_indices])
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Index collection & extraction
# ---------------------------------------------------------------------------

def collect_indices(cache, years, ds_sds, sds_var, land_indices, logger):
    """
    Scan years, return lists of (year, idx_in_year) for pos and neg samples.
    Pure function - does not modify the cache.
    """
    all_pos, all_neg = [], []
    for year in years:
        if year not in cache.X:
            continue
        y = make_target_for_year(
            year, ds_sds, sds_var, cache.unique_times[year], land_indices)
        n = cache.n_samples[year]
        if len(y) != n:
            logger.warning(f"  {year}: target/feature mismatch, skipping")
            continue
        # only keep samples where features AND target are finite
        valid = cache.finite_mask[year] & np.isfinite(y)
        valid_idx = np.where(valid)[0]
        y_valid = y[valid_idx]
        pos_idx = valid_idx[y_valid == 1]
        neg_idx = valid_idx[y_valid == 0]
        logger.info(f"  {year}: {len(pos_idx):,} pos, {len(neg_idx):,} neg "
                     f"(dropped {int((~valid).sum()):,} NaN)")
        for idx in pos_idx:
            all_pos.append((year, int(idx)))
        for idx in neg_idx:
            all_neg.append((year, int(idx)))
        del y, valid, y_valid
        gc.collect()
    logger.info(f"  Total: {len(all_pos):,} pos, {len(all_neg):,} neg")
    return all_pos, all_neg


def extract_from_cache(cache, indices):
    """
    Given list of (year, idx_in_year), extract feature rows.
    Returns (n, n_features) float32 array in the same order as `indices`.
    """
    n = len(indices)
    n_feat = len(cache.feature_vars)
    X = np.empty((n, n_feat), dtype=np.float32)
    # group by year so we only touch cache.X[year] once per year
    by_year = defaultdict(list)
    for out_row, (year, idx) in enumerate(indices):
        by_year[year].append((idx, out_row))
    for year, rows in by_year.items():
        idxs = np.array([r[0] for r in rows])
        outs = np.array([r[1] for r in rows])
        # sorted reads into cache.X[year] are cache-friendlier than random ones
        order = np.argsort(idxs)
        X[outs[order]] = cache.X[year][idxs[order]]
    return X


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_train_set(cache, train_pos, train_neg, sampling_cfg, logger):
    """
    Build X_train, y_train from pre-collected pos/neg index lists.

    Two modes controlled by SamplingConfig:
      - num_train_total_cap > 0: fixed total budget, derive
            n_pos = total / (1 + ratio), n_neg = total - n_pos
      - num_train_total_cap == 0: cap positives at num_train_pos_cap,
            then n_neg = n_pos * neg_ratio

    Returns (X_train, y_train, n_pos_used).
    """
    rng = np.random.default_rng(sampling_cfg.seed)
    n_total_pos = len(train_pos)
    n_total_neg = len(train_neg)

    if sampling_cfg.num_train_total_cap > 0:
        # ── Fixed total budget mode ───────────────────────────────────
        total = sampling_cfg.num_train_total_cap
        ratio = sampling_cfg.neg_ratio
        n_pos = total // (1 + ratio)
        n_neg = total - n_pos

        # clamp to what's actually available in the pool
        n_pos = min(n_pos, n_total_pos)
        n_neg = min(n_neg, n_total_neg)

        logger.info(f"  Train (total_cap={total:,}): "
                    f"{n_pos:,} pos / {n_neg:,} neg (1:{ratio})")
    else:
        # ── Pos-cap mode (0 = no cap, use all) ───────────────────────
        if sampling_cfg.num_train_pos_cap > 0:
            n_pos = min(n_total_pos, sampling_cfg.num_train_pos_cap)
        else:
            n_pos = n_total_pos
        n_neg = min(n_pos * sampling_cfg.neg_ratio, n_total_neg)

        logger.info(f"  Train (pos_cap={sampling_cfg.num_train_pos_cap}): "
                    f"{n_pos:,} pos / {n_neg:,} neg "
                    f"(1:{sampling_cfg.neg_ratio})")

    # positive subsample (skip the choice call if we're taking everything)
    if n_total_pos > n_pos:
        sel = rng.choice(n_total_pos, size=n_pos, replace=False)
        pos_sample = [train_pos[i] for i in sel]
    else:
        pos_sample = train_pos

    # negative subsample
    if n_total_neg > n_neg:
        sel = rng.choice(n_total_neg, size=n_neg, replace=False)
        neg_sample = [train_neg[i] for i in sel]
    else:
        neg_sample = train_neg

    X_train = extract_from_cache(cache, pos_sample + neg_sample)
    y_train = np.concatenate([
        np.ones(len(pos_sample), dtype=np.float32),
        np.zeros(len(neg_sample), dtype=np.float32),
    ])
    return X_train, y_train, len(pos_sample)


def build_eval_set(cache, pos_list, neg_list, sampling_cfg, rng, logger,
                   split_name="eval"):
    """
    Build X, y for a val or test split with proportional sampling,
    capped at sampling_cfg.val_test_total_cap. Preserves the original pos:neg
    ratio of the pool (unlike build_train_set which lets you impose a ratio).
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

    X = extract_from_cache(cache, p_keep + n_keep)
    y = np.concatenate([np.ones(len(p_keep), np.float32),
                        np.zeros(len(n_keep), np.float32)])

    logger.info(f"  {split_name}: {len(p_keep):,} pos / {len(n_keep):,} neg "
                f"({len(p_keep)/len(y)*100:.3f}% prevalence)")
    return X, y


# ---------------------------------------------------------------------------
# Platt scaling + metrics
# ---------------------------------------------------------------------------

def platt_fit(y, p):
    """Fit Platt scaling (logistic regression on log-odds of RF probabilities)."""
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(z, y)
    return lr


def platt_apply(lr, p):
    """Apply fitted Platt scaling."""
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps)).reshape(-1, 1)
    return lr.predict_proba(z)[:, 1]


def compute_metrics(y_true, y_proba, threshold):
    """Compute classification metrics at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "threshold": float(threshold),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "auc":       float(roc_auc_score(y_true, y_proba))
                      if len(np.unique(y_true)) > 1 else 0.0,
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }


def find_best_threshold(y_true, y_proba):
    """Find threshold that maximises F1 on the precision-recall curve."""
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    # precision_recall_curve returns len(thr) == len(prec) - 1, so trim f1 to match
    return float(thr[f1[:-1].argmax()])