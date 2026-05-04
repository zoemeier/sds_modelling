"""
Produces four figures for the feature-correlation analysis:

  1. Raw matrix (no grouping, no clustering, no numbers) - for diagnostics,
     shows the correlation structure in the original feature order.
  2. Full clustered matrix (all features, clustered, group colour bars with
     saturation-encoded lag level) - paper appendix.
  3. Top-N labelled matrix (top features by permutation importance, with numbers,
     lower-triangular) - main paper.
  4. Feature-vs-target correlation bar chart (point-biserial with the binary
     SDS target) - simple univariate baseline.

Usage:
  python compute_feature_correlations.py \
      --main_model_dir models/final_rf_v1_20260413_120000 \
      --top_n 20
"""

import argparse
import gc
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    RESOLUTION, SPATIAL_RES, TRAIN_YEARS, VAL_YEARS, TEST_YEARS, SDS_PATH,
)
from data_utils import (
    FeatureCache, load_sds_source, compute_land_indices, make_target_for_year,
)
from feature_names import get_feature_name_map


# plot style - matches the other analysis scripts
TITLE_SIZE = 14
SUBTITLE_SIZE = 12
LABEL_SIZE = 11
TICK_SIZE = 9

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": LABEL_SIZE,
    "axes.titlesize": SUBTITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "figure.dpi": 150,
    "savefig.dpi": 250,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# ---------------------------------------------------------------------------
# Feature grouping - explicit, covers all 61 features
# ---------------------------------------------------------------------------
FEATURE_GROUPS = [
    ("Temperature", [
        "t2m",
        "t2m_lag_1d", "t2m_lag_2d", "t2m_lag_3d",
        "t2m_lag_4d", "t2m_lag_5d", "t2m_lag_6d",
        "t2m_roll_30d", "t2m_roll_90d", "t2m_roll_180d",
    ]),
    ("Wind", [
        "ws_mean", "ws_max", "rv", "wd",
        "ws_mean_lag_1d", "ws_mean_lag_2d", "ws_mean_lag_3d",
        "ws_mean_lag_4d", "ws_mean_lag_5d", "ws_mean_lag_6d",
    ]),
    ("Humidity", [
        "rh",
        "rh_lag_1d", "rh_lag_2d", "rh_lag_3d",
        "rh_lag_4d", "rh_lag_5d", "rh_lag_6d",
    ]),
    ("Precipitation", [
        "log_tp",
        "log_tp_lag_1d", "log_tp_lag_2d", "log_tp_lag_3d",
        "log_tp_lag_4d", "log_tp_lag_5d", "log_tp_lag_6d",
        "log_tp_roll_30d", "log_tp_roll_90d", "log_tp_roll_180d",
    ]),
    ("Land surface", [
        "lai_lv", "lai_hv", "cvl", "cvh",
        "NDVI_mean", "NDVI_std", "snow_frac",
    ]),
    ("Orography", [
        "sdor", "slor", "anor", "isor", "wind_terrain_align",
    ]),
    ("Soil", [
        "sand", "silt", "clay", "bdod", "soc", "cec",
        "fine_fraction", "sand_to_fines",
    ]),
    ("Conflict", [
        "acled_total_count_log1p",
        "acled_total_count_30d_sum_log1p",
        "acled_total_count_180d_sum_log1p",
        "acled_total_count_365d_sum_log1p",
    ]),
]

# base hue per group - pulled from matplotlib's qualitative "Paired" colormap.
# Paired indexing: 0=light blue, 1=dark blue, 2=light green, 3=dark green,
# 4=light red, 5=dark red, 6=light orange, 7=dark orange.
# mapping below uses the light/dark shades of each hue to group semantically
# related variables (moisture=blue, vegetation=green, thermal=orange, static=red).
_PAIRED_COLORS = plt.get_cmap("Paired").colors   # 12-entry ListedColormap
_PAIRED_IDX = {
    "Humidity":      0,   # light blue
    "Precipitation": 1,   # dark blue
    "Soil":          2,   # light green
    "Land surface":  3,   # dark green
    "Temperature":   6,   # light orange
    "Wind":          7,   # dark orange
    "Orography":     4,   # light red
    "Conflict":      5,   # dark red
}
GROUP_BASE_COLOR = {
    name: mcolors.to_hex(_PAIRED_COLORS[_PAIRED_IDX[name]])
    for name, _ in FEATURE_GROUPS
}


def _lag_kind(feat):
    # returns one of: "inst", "short_lag", "long_lag", "rolling"
    # used for within-group ordering in the grouped correlation matrix
    if re.search(r"_roll_\d+d$", feat):
        return "rolling"
    # acled rolling sums use a different suffix pattern
    if re.search(r"_(30d|90d|180d|365d)_sum_log1p$", feat):
        return "rolling"
    m = re.search(r"_lag_(\d+)d$", feat)
    if m:
        n = int(m.group(1))
        return "short_lag" if n <= 3 else "long_lag"
    return "inst"


def _group_for(feat):
    # returns (group_name, base_hex) for a given feature, or None if no group matches
    for name, members in FEATURE_GROUPS:
        if feat in members:
            return name, GROUP_BASE_COLOR[name]
    return None


# order within a group: instantaneous, then short lags (1d→3d), then long lags (4d→6d),
# then rolling (30d→90d→180d→365d). features with no numeric suffix sort first in their tier.
_LAG_KIND_RANK = {"inst": 0, "short_lag": 1, "long_lag": 2, "rolling": 3}

def _sort_key(feat):
    # build a sort key that keeps groups in their defined order and orders within a group
    # by (lag kind, numeric suffix). falls back to feat name for stable ordering.
    g = _group_for(feat)
    group_idx = next((i for i, (name, _) in enumerate(FEATURE_GROUPS)
                      if g is not None and name == g[0]), 999)
    kind = _lag_kind(feat)
    # extract the numeric suffix (lag days, rolling window size) for tie-breaking
    m = re.search(r"_(\d+)d", feat)
    num = int(m.group(1)) if m else 0
    return (group_idx, _LAG_KIND_RANK[kind], num, feat)


def sort_features_by_group(feature_vars):
    # reorder feature_vars so groups come in FEATURE_GROUPS order, and within each group
    # features go instantaneous → short_lag (1d→3d) → long_lag (4d→6d) → rolling (small→large)
    return sorted(feature_vars, key=_sort_key)


def _nice(feat, name_map):
    # None-safe: if name_map has no entry or has an explicit None, fall back to the short name
    long = name_map.get(feat)
    return re.sub(r"\s*\[.*?\]", "", long) if long else feat


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------
def compute_feature_correlation_streaming(cache, logger):
    # streaming implementation - accumulates sums, sums-of-squares, and the cross-product
    # matrix year by year without ever holding the full concatenated feature matrix in memory.
    # peak extra memory is just one year's slice + a (n_feat x n_feat) gram matrix (tiny for 61).
    logger.info("Computing feature-feature correlation (streaming) ...")
    n_feat = len(cache.feature_vars)
    n_total = 0
    sum_x = np.zeros(n_feat, dtype=np.float64)
    sum_x2 = np.zeros(n_feat, dtype=np.float64)
    gram = np.zeros((n_feat, n_feat), dtype=np.float64)   # X.T @ X accumulated

    for year in cache.all_years:
        Xy = cache.X[year][cache.finite_mask[year]]
        # cast to float64 for the reduction to avoid precision drift on big sums,
        # but only for the reduction - keeps peak memory low since we don't materialise a copy
        n_total += Xy.shape[0]
        sum_x  += Xy.sum(axis=0, dtype=np.float64)
        sum_x2 += (Xy.astype(np.float64) ** 2).sum(axis=0)
        # np.dot promotes to float64 automatically since gram is float64
        gram += Xy.T.astype(np.float64) @ Xy.astype(np.float64)
        logger.info(f"  {year}: accumulated (running total {n_total:,})")

    # convert accumulators to Pearson correlation matrix
    mean = sum_x / n_total
    # Cov = gram / n - mean*mean.T;  Var = sum_x2 / n - mean^2
    cov = gram / n_total - np.outer(mean, mean)
    var = sum_x2 / n_total - mean ** 2
    std = np.sqrt(np.clip(var, 0, None))
    # outer product of stds, with epsilon to avoid division by zero on constant columns
    denom = np.outer(std, std)
    denom = np.where(denom > 0, denom, np.nan)
    corr = cov / denom
    # clip numerical noise so values stay in [-1, 1]
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    logger.info(f"  Done. n = {n_total:,}")
    return corr


def compute_target_correlation_streaming(cache, ds_sds, sds_var, land_indices, logger):
    # same streaming idea for the feature-target point-biserial correlation.
    # no need to ever hold the full target vector in memory either.
    logger.info("Computing feature-target correlation (streaming) ...")
    n_feat = len(cache.feature_vars)
    n_total = 0
    sum_x = np.zeros(n_feat, dtype=np.float64)
    sum_x2 = np.zeros(n_feat, dtype=np.float64)
    sum_y = 0.0
    sum_y2 = 0.0
    sum_xy = np.zeros(n_feat, dtype=np.float64)

    for year in cache.all_years:
        y_year = make_target_for_year(
            year, ds_sds, sds_var, cache.unique_times[year], land_indices)
        mask = cache.finite_mask[year]
        Xy = cache.X[year][mask]
        yy = y_year[mask].astype(np.float64)

        n_total += Xy.shape[0]
        sum_x  += Xy.sum(axis=0, dtype=np.float64)
        sum_x2 += (Xy.astype(np.float64) ** 2).sum(axis=0)
        sum_y  += yy.sum()
        sum_y2 += (yy ** 2).sum()
        sum_xy += Xy.astype(np.float64).T @ yy
        logger.info(f"  {year}: accumulated (running total {n_total:,})")

    mean_x = sum_x / n_total
    mean_y = sum_y / n_total
    cov_xy = sum_xy / n_total - mean_x * mean_y
    var_x = sum_x2 / n_total - mean_x ** 2
    var_y = sum_y2 / n_total - mean_y ** 2
    denom = np.sqrt(np.clip(var_x, 0, None)) * np.sqrt(max(var_y, 0))
    corr_y = np.where(denom > 0, cov_xy / denom, np.nan)
    corr_y = np.clip(corr_y, -1.0, 1.0)
    logger.info(f"  Done. n = {n_total:,}, positives = {int(sum_y):,}")
    return corr_y


# ---------------------------------------------------------------------------
# Plot 1: raw matrix, no ordering, no grouping, no numbers
# ---------------------------------------------------------------------------
def plot_grouped(corr, feature_vars, name_map, od):
    # full matrix, features ordered by group (within-group: instant → lags → rolling).
    # draws white dividers between groups, labels each group block on the axes,
    # and keeps a clean legend for both groups and lag-level fade.
    print("  [1/3] Grouped correlation matrix (all features) ...")
    sorted_feats = sort_features_by_group(feature_vars)
    order = [feature_vars.index(f) for f in sorted_feats]
    corr_ord = corr[np.ix_(order, order)]
    n = len(sorted_feats)
    display_names = [_nice(f, name_map) for f in sorted_feats]

    # CHANGED: 0.22 → 0.26 to give larger tick labels more room
    fig_size = max(12, 0.26 * n + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        pd.DataFrame(corr_ord, index=display_names, columns=display_names),
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=True, yticklabels=True,
        cbar_kws={"shrink": 0.35, "label": "Pearson correlation", "pad": 0.02},
        square=True, rasterized=True,
        ax=ax,
    )

    # find group boundaries and the centre of each group's block for labelling
    group_ranges = []   # list of (group_name, start_idx, end_idx_exclusive)
    current_group = None
    start = 0
    for i, f in enumerate(sorted_feats):
        g = _group_for(f)
        g_name = g[0] if g else "Other"
        if current_group is None:
            current_group = g_name
        elif g_name != current_group:
            group_ranges.append((current_group, start, i))
            start = i
            current_group = g_name
    group_ranges.append((current_group, start, n))

    # white dividers between groups - 2px wide so they read clearly on the paper
    for _, _, end in group_ranges[:-1]:
        ax.axhline(end, color="white", lw=2.0)
        ax.axvline(end, color="white", lw=2.0)

    # feature labels stay on the inner axis
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title(f"Feature correlations — all {n} features (ordered by group)",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=12)

    # CHANGED: bottom 0.22 → 0.26, left 0.20 → 0.24 for larger rotated tick labels
    fig.subplots_adjust(left=0.24, right=0.90, top=0.94, bottom=0.26)
    fig.canvas.draw()   # make transforms valid before we compute bracket positions

    # ── bracket-style group annotations ──────────────────────────────────
    # placed outside the tick labels. Uses axes-fraction transform for the
    # offset direction, and data coords for the along-axis position.
    # bracket is: horizontal/vertical spanning line with two small tips pointing inward,
    # and the group name as a plain label alongside (not bold, not coloured).
    def _bbox_padding(axis):
        # figure out how far outside the axes the tick labels extend, in axes fraction
        labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
        if not labels:
            return 0.05
        # take the widest tick-label extent in inches, convert to axes fraction
        bboxes = [lab.get_window_extent(fig.canvas.get_renderer()) for lab in labels]
        if axis == "x":
            height_px = max(b.height for b in bboxes)
            ax_height_px = ax.bbox.height
            return height_px / ax_height_px
        else:
            width_px = max(b.width for b in bboxes)
            ax_width_px = ax.bbox.width
            return width_px / ax_width_px

    x_pad = _bbox_padding("x")
    y_pad = _bbox_padding("y")

    # bracket geometry in axes-fraction units.
    # CHANGED: LABEL_OFFSET now scales with the group label font size so that
    # bumping GROUP_FONT (e.g. to 13) keeps the gap visually proportional.
    GROUP_FONT = 11
    scale = GROUP_FONT / 9.0            # 9pt was the original reference size
    BRACKET_OFFSET  = 0.015             # gap between tick labels and bracket line (unchanged)
    BRACKET_TIP_LEN = 0.012             # little inward tip length (unchanged)
    LABEL_OFFSET    = 0.028 * scale     # CHANGED: 0.025 → 0.028 * scale
    BRACKET_INSET   = 0.1               # how far inside each group-edge the bracket ends (data units)

    # x-axis (bottom): brackets below the x tick labels
    bracket_y_x = -x_pad - BRACKET_OFFSET
    label_y_x = bracket_y_x - LABEL_OFFSET

    for name, s, e in group_ranges:
        # span line in data-x, axes-y
        ax.plot([s + BRACKET_INSET, e - BRACKET_INSET], [bracket_y_x, bracket_y_x],
                transform=ax.get_xaxis_transform(),
                color="#222222", lw=1.2, clip_on=False, solid_capstyle="butt")
        # two small tips pointing up (toward the axes)
        ax.plot([s + BRACKET_INSET, s + BRACKET_INSET], [bracket_y_x, bracket_y_x + BRACKET_TIP_LEN],
                transform=ax.get_xaxis_transform(),
                color="#222222", lw=1.2, clip_on=False)
        ax.plot([e - BRACKET_INSET, e - BRACKET_INSET], [bracket_y_x, bracket_y_x + BRACKET_TIP_LEN],
                transform=ax.get_xaxis_transform(),
                color="#222222", lw=1.2, clip_on=False)
        # group name, rotated 90° (reading upward), centred on the bracket
        ax.text((s + e) / 2, label_y_x, name,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", rotation=90,
                fontsize=GROUP_FONT, color="#222222", clip_on=False)

    # y-axis (left): brackets to the left of the y tick labels
    bracket_x_y = -y_pad - BRACKET_OFFSET
    label_x_y = bracket_x_y - LABEL_OFFSET

    for name, s, e in group_ranges:
        ax.plot([bracket_x_y, bracket_x_y], [s + BRACKET_INSET, e - BRACKET_INSET],
                transform=ax.get_yaxis_transform(),
                color="#222222", lw=1.2, clip_on=False, solid_capstyle="butt")
        ax.plot([bracket_x_y, bracket_x_y + BRACKET_TIP_LEN], [s + BRACKET_INSET, s + BRACKET_INSET],
                transform=ax.get_yaxis_transform(),
                color="#222222", lw=1.2, clip_on=False)
        ax.plot([bracket_x_y, bracket_x_y + BRACKET_TIP_LEN], [e - BRACKET_INSET, e - BRACKET_INSET],
                transform=ax.get_yaxis_transform(),
                color="#222222", lw=1.2, clip_on=False)
        ax.text(label_x_y, (s + e) / 2, name,
                transform=ax.get_yaxis_transform(),
                ha="right", va="center", rotation=90,
                fontsize=GROUP_FONT, color="#222222", clip_on=False)

    out = od / "feature_correlations_grouped.png"
    fig.savefig(out); plt.close(fig)
    print(f"    saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: top-N labelled matrix, lower-triangular, with numbers
# ---------------------------------------------------------------------------
def plot_top_n(corr, feature_vars, ranked_feats, name_map, top_n, od,
               ranking_label, filename_suffix):
    # ranked_feats: list of features in importance order (most important first).
    # ranking_label: short string for the plot title, e.g. "permutation importance" or "SHAP importance".
    # filename_suffix: appended before .png, e.g. "" for permutation (back-compat) or "_shap" for SHAP.
    print(f"  [2/3] Top-{top_n} labelled matrix ({ranking_label}) ...")
    top_feats = [f for f in ranked_feats if f in feature_vars][:top_n]
    if len(top_feats) < top_n:
        print(f"    warning: only {len(top_feats)} of the top {top_n} are in the cache")

    top_idx = [feature_vars.index(f) for f in top_feats]
    sub = corr[np.ix_(top_idx, top_idx)]
    display_names = [_nice(f, name_map) for f in top_feats]
    sub_df = pd.DataFrame(sub, index=display_names, columns=display_names)
    mask = np.triu(np.ones_like(sub, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(max(9, 0.45 * len(top_feats) + 2),
                                    max(7, 0.45 * len(top_feats) + 1)))
    sns.heatmap(
        sub_df, mask=mask,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        square=True, linewidths=0.3, linecolor="white",
        cbar_kws={"shrink": 0.6, "label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=TICK_SIZE)
    ax.set_title(f"Correlations among top {len(top_feats)} features (by {ranking_label})",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=12)
    plt.tight_layout()
    out = od / f"feature_correlations_top{top_n}{filename_suffix}.png"
    fig.savefig(out); plt.close(fig)
    print(f"    saved: {out}")

# ---------------------------------------------------------------------------
# Plot 4: feature-vs-target correlation bar chart
# ---------------------------------------------------------------------------
def plot_target_corr(corr_with_target, feature_vars, shap_ranking, name_map, top_n, od):
    # corr_with_target: 1D array of point-biserial correlations, one per feature
    # shap_ranking: dict {feature -> rank} (lower rank = more important). features missing
    # from the SHAP CSV get the worst rank so they sink to the bottom of the ordering.
    print("  [3/3] Feature-vs-target correlation bar chart (SHAP-ordered) ...")

    # assign SHAP rank to each feature (features not in the CSV get a high rank sentinel)
    worst_rank = max(shap_ranking.values(), default=0) + 1 if shap_ranking else 999
    df = pd.DataFrame({
        "feature": feature_vars,
        "corr": corr_with_target,
        "shap_rank": [shap_ranking.get(f, worst_rank) for f in feature_vars],
    })
    # sort by SHAP rank ascending, so rank 1 (most important) ends up at the top of the
    # horizontal bar chart (matplotlib's y-axis draws index 0 at the bottom by default,
    # so we reverse for display).
    df = df.sort_values("shap_rank", ascending=True).reset_index(drop=True)

    df["display"] = df["feature"].map(lambda f: _nice(f, name_map))
    df["group_color"] = df["feature"].map(
        lambda f: GROUP_BASE_COLOR.get(_group_for(f)[0]) if _group_for(f) else "#888888")

    def _plot(df_sub, title, out_path):
        n = len(df_sub)
        fig, ax = plt.subplots(figsize=(10, max(6, 0.22 * n + 1)))
        # reverse y order so the most-important feature (rank 1) is at the top
        y_positions = range(n - 1, -1, -1)
        ax.barh(list(y_positions), df_sub["corr"].values,
                color=df_sub["group_color"].values, edgecolor="none")
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(list(y_positions))
        # font sizing: y tick labels are the largest reading element here, x label
        # matches them, and the legend title matches the legend body (= x label size).
        YTICK_FONT  = 11   # bumped up from 7
        XLABEL_FONT = 9    # matches y tick label size... wait, see below
        # actually: x label == y tick labels, legend title == legend body == x label.
        # so we want one "small" size and one "large" size:
        SMALL_FONT = 9     # legend body, legend title, x axis label
        LARGE_FONT = 11    # y axis tick labels
        ax.set_yticklabels(df_sub["display"].values, fontsize=LARGE_FONT)
        ax.set_xlabel("Pearson correlation with target (SDS = 1)", fontsize=SMALL_FONT)
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # only include groups that actually appear in this panel in the legend
        groups_present = {
            _group_for(f)[0] for f in df_sub["feature"] if _group_for(f)
        }
        group_handles = [plt.Rectangle((0, 0), 1, 1,
                                       facecolor=GROUP_BASE_COLOR[name],
                                       edgecolor="none")
                         for name, _ in FEATURE_GROUPS if name in groups_present]
        group_labels = [name for name, _ in FEATURE_GROUPS if name in groups_present]
        leg = ax.legend(group_handles, group_labels,
                        loc="lower right", frameon=True,
                        fontsize=SMALL_FONT, title="Feature group",
                        title_fontsize=SMALL_FONT)
        plt.tight_layout()
        fig.savefig(out_path); plt.close(fig)
        print(f"    saved: {out_path}")
        
    # full: all 61 features in SHAP order
    _plot(df, "Feature correlation with target (ordered by SHAP importance)",
          od / "feature_target_correlation.png")

    # top-N: just the top N by SHAP
    _plot(df.head(top_n),
          f"Feature correlation with target — top {top_n} by SHAP importance",
          od / f"feature_target_correlation_top{top_n}.png")

    # full: all features ordered by |correlation| descending
    df_abs = df.reindex(df["corr"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    _plot(df_abs, "Feature correlation with target (ordered by |correlation|)",
          od / "feature_target_correlation_abssorted.png")

    # top-N: just the top N by |correlation|
    _plot(df_abs.head(top_n),
          f"Feature correlation with target — top {top_n} by |correlation|",
          od / f"feature_target_correlation_abssorted_top{top_n}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Feature-correlation analysis: four figures + a correlation CSV")
    parser.add_argument("--main_model_dir", type=str, required=True)
    parser.add_argument("--perm_csv", type=str, default=None,
        help="override the permutation CSV. Default: matched-subsample if present, "
             "else full-model.")
    parser.add_argument("--shap_csv", type=str, default=None,
        help="override the SHAP CSV used to order the target-correlation bar chart. "
             "Default: {main_model_dir}/shap/shap_feature_importance.csv. If not found, "
             "falls back to sorting by |corr_with_target|.")
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--sds_path", type=str, default=None)
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--spatial", type=float, default=None)
    parser.add_argument("--train_years", nargs=2, type=int, default=None)
    parser.add_argument("--val_years", nargs=2, type=int, default=None)
    parser.add_argument("--test_years", nargs=2, type=int, default=None)
    args = parser.parse_args()

    main_model_dir = Path(args.main_model_dir)
    if not main_model_dir.exists():
        print(f"ERROR: main_model_dir not found: {main_model_dir}")
        sys.exit(1)

    main_results = json.loads((main_model_dir / "results.json").read_text())
    data_cfg = main_results["data_config"]
    resolution = args.resolution or data_cfg.get("resolution", RESOLUTION)
    spatial_res = args.spatial or data_cfg.get("spatial_res", SPATIAL_RES)
    train_years = args.train_years or data_cfg.get("train_years", TRAIN_YEARS)
    val_years = args.val_years or data_cfg.get("val_years", VAL_YEARS)
    test_years = args.test_years or data_cfg.get("test_years", TEST_YEARS)
    sds_path = args.sds_path or data_cfg.get("sds_path") or SDS_PATH

    # pick permutation CSV - matched-subsample preferred, same logic as aggregate_dropout
    if args.perm_csv:
        perm_csv = Path(args.perm_csv)
    else:
        matched = main_model_dir / "dropout" / "__baseline__" / "permutation_importance_matched.csv"
        fullmodel = main_model_dir / "permutation_importance" / "permutation_importance_test.csv"
        if matched.exists():
            perm_csv = matched
        elif fullmodel.exists():
            perm_csv = fullmodel
        else:
            print(f"ERROR: no permutation CSV found at:")
            print(f"  {matched}")
            print(f"  {fullmodel}")
            sys.exit(1)

    od = Path(args.out_dir) if args.out_dir else main_model_dir / "feature_correlations"
    od.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("feat_corr")

    logger.info("=" * 60)
    logger.info("FEATURE CORRELATION ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"  Model: {main_model_dir}")
    logger.info(f"  Perm CSV: {perm_csv}")
    logger.info(f"  Top-N (labelled figure): {args.top_n}")
    logger.info(f"  Output: {od}")
    logger.info("=" * 60)

    cache = FeatureCache(
        resolution=resolution, spatial_res=spatial_res,
        train_years=train_years, val_years=val_years, test_years=test_years,
        logger=logger,
    )
    feature_vars = cache.feature_vars
    name_map = get_feature_name_map(feature_vars)

    # ── Feature-feature correlation (streaming, memory-safe) ────────────
    corr = compute_feature_correlation_streaming(cache, logger)
    pd.DataFrame(corr, index=feature_vars, columns=feature_vars).to_csv(
        od / "feature_correlations.csv")

    # ── Feature-target correlation (streaming, memory-safe) ─────────────
    sds_file = Path(sds_path)
    ds_sds, sds_var = load_sds_source(sds_file, logger)
    land_indices = compute_land_indices(
        ds_sds["latitude"].values, ds_sds["longitude"].values)
    corr_y = compute_target_correlation_streaming(
        cache, ds_sds, sds_var, land_indices, logger)
    ds_sds.close(); del ds_sds
    gc.collect()

    pd.DataFrame({
        "feature": feature_vars,
        "corr_with_target": corr_y,
    }).to_csv(od / "feature_target_correlation.csv", index=False)

    # ── Plots ───────────────────────────────────────────────────────────
    perm_df = pd.read_csv(perm_csv)

    # load SHAP ranking for the target-correlation bar chart ordering
    shap_csv = Path(args.shap_csv) if args.shap_csv else \
        main_model_dir / "shap" / "shap_feature_importance.csv"
    if shap_csv.exists():
        shap_df = pd.read_csv(shap_csv).sort_values("mean_abs_shap", ascending=False)
        shap_ranking = {f: r for r, f in enumerate(shap_df["feature"].values, start=1)}
        logger.info(f"  SHAP ranking loaded from: {shap_csv}")
    else:
        # fall back to |corr_with_target| ordering - functionally equivalent to the old behaviour
        logger.info(f"  SHAP CSV not found at {shap_csv}; falling back to |corr| ordering")
        abs_order = np.argsort(-np.abs(corr_y))
        shap_ranking = {feature_vars[i]: r for r, i in enumerate(abs_order, start=1)}

    logger.info("\nPlotting ...")
    plot_grouped(corr, feature_vars, name_map, od)
    # permutation-ordered (existing behaviour, existing filename)
    perm_ranked = perm_df["feature"].tolist()
    plot_top_n(corr, feature_vars, perm_ranked, name_map, args.top_n, od,
               ranking_label="permutation importance",
               filename_suffix="")

    # SHAP-ordered (new). shap_ranking maps feature -> rank (1 = most important).
    # If we fell back to |corr| ordering above (no SHAP CSV), skip this plot rather than
    # produce a misleadingly named file.
    if shap_csv.exists():
        shap_ranked = sorted(shap_ranking, key=shap_ranking.get)
        plot_top_n(corr, feature_vars, shap_ranked, name_map, args.top_n, od,
                   ranking_label="SHAP importance",
                   filename_suffix="_shap")
    else:
        logger.info("  Skipping SHAP-ordered top-N matrix (no SHAP CSV available)")
    plot_target_corr(corr_y, feature_vars, shap_ranking, name_map, args.top_n, od)

    logger.info("\nDONE")
    logger.info(f"  All figures + CSVs saved to: {od}")


if __name__ == "__main__":
    main()