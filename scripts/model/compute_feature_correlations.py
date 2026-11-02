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
      --main_model_dir models/final_rf_<timestamp> \
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

from sds.config import RESOLUTION, SPATIAL_RES, TRAIN_YEARS, VAL_YEARS, TEST_YEARS
from sds.paths import project_relative, model_figures_dir, resolve_path, SDS_PATH
from sds.log_utils import header, section, done
from sds.data_utils import (
    FeatureCache, load_sds_source, compute_land_indices, make_target_for_year,
)
from sds.feature_names import get_feature_name_map


# plot style (same as the other analysis scripts)
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
# Feature groups (all 61 features)
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

# base hue per group from matplotlib's "Paired" colormap: moisture blue,
# vegetation green, thermal orange, static red
_PAIRED_COLORS = plt.get_cmap("Paired").colors
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
    # "inst", "short_lag", "long_lag" or "rolling" (ordering within a group)
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
    # (group_name, base_hex) for a feature, or None
    for name, members in FEATURE_GROUPS:
        if feat in members:
            return name, GROUP_BASE_COLOR[name]
    return None


# order within a group: instantaneous, then short lags (1d→3d), then long lags (4d→6d),
# then rolling (30d→90d→180d→365d). features with no numeric suffix sort first in their tier.
_LAG_KIND_RANK = {"inst": 0, "short_lag": 1, "long_lag": 2, "rolling": 3}

def _sort_key(feat):
    # sort key: group order, then lag kind, then numeric suffix, then name
    g = _group_for(feat)
    group_idx = next((i for i, (name, _) in enumerate(FEATURE_GROUPS)
                      if g is not None and name == g[0]), 999)
    kind = _lag_kind(feat)
    m = re.search(r"_(\d+)d", feat)
    num = int(m.group(1)) if m else 0
    return (group_idx, _LAG_KIND_RANK[kind], num, feat)


def sort_features_by_group(feature_vars):
    return sorted(feature_vars, key=_sort_key)


def _nice(feat, name_map):
    long = name_map.get(feat)
    return re.sub(r"\s*\[.*?\]", "", long) if long else feat


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------
def compute_feature_correlation_streaming(cache, logger):
    # streaming: accumulates sums, sums of squares and X.T @ X year by year,
    # so the full feature matrix is never held in memory
    section("Feature-feature correlation", out=logger.info)
    n_feat = len(cache.feature_vars)
    n_total = 0
    sum_x = np.zeros(n_feat, dtype=np.float64)
    sum_x2 = np.zeros(n_feat, dtype=np.float64)
    gram = np.zeros((n_feat, n_feat), dtype=np.float64)   # X.T @ X accumulated

    for year in cache.all_years:
        Xy = cache.X[year][cache.finite_mask[year]]
        # reduce in float64 to avoid precision drift on large sums
        n_total += Xy.shape[0]
        sum_x  += Xy.sum(axis=0, dtype=np.float64)
        sum_x2 += (Xy.astype(np.float64) ** 2).sum(axis=0)
        gram += Xy.T.astype(np.float64) @ Xy.astype(np.float64)

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
    logger.info(f"  Samples: {n_total:,}")
    return corr


def compute_target_correlation_streaming(cache, ds_sds, sds_var, land_indices, logger):
    # same streaming approach for the point-biserial correlation with the target
    section("Feature-target correlation", out=logger.info)
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

    mean_x = sum_x / n_total
    mean_y = sum_y / n_total
    cov_xy = sum_xy / n_total - mean_x * mean_y
    var_x = sum_x2 / n_total - mean_x ** 2
    var_y = sum_y2 / n_total - mean_y ** 2
    denom = np.sqrt(np.clip(var_x, 0, None)) * np.sqrt(max(var_y, 0))
    corr_y = np.where(denom > 0, cov_xy / denom, np.nan)
    corr_y = np.clip(corr_y, -1.0, 1.0)
    logger.info(f"  Samples: {n_total:,} ({int(sum_y):,} pos)")
    return corr_y


# ---------------------------------------------------------------------------
# Plot: full matrix, ordered by group
# ---------------------------------------------------------------------------
def plot_grouped(corr, feature_vars, name_map, od):
    # features ordered by group (instantaneous -> lags -> rolling), white dividers
    # between groups and bracket labels for each group
    print("  [1/3] Grouped correlation matrix (all features)")
    sorted_feats = sort_features_by_group(feature_vars)
    order = [feature_vars.index(f) for f in sorted_feats]
    corr_ord = corr[np.ix_(order, order)]
    n = len(sorted_feats)
    display_names = [_nice(f, name_map) for f in sorted_feats]

    # ~0.26 in per feature so the tick labels have room
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

    # white dividers between groups
    for _, _, end in group_ranges[:-1]:
        ax.axhline(end, color="white", lw=2.0)
        ax.axvline(end, color="white", lw=2.0)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title(f"Feature correlations — all {n} features (ordered by group)",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=12)

    # wide left/bottom margins for the rotated tick labels + group brackets
    fig.subplots_adjust(left=0.24, right=0.90, top=0.94, bottom=0.26)
    fig.canvas.draw()   # make transforms valid before we compute bracket positions

    # group brackets outside the tick labels (axes-fraction offset, data-coord span)
    def _bbox_padding(axis):
        labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
        if not labels:
            return 0.05
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

    # bracket geometry in axes-fraction units. LABEL_OFFSET scales with the
    # group label font size so the gap stays proportional if GROUP_FONT changes.
    GROUP_FONT = 11
    scale = GROUP_FONT / 9.0            # offsets below are tuned for 9pt
    BRACKET_OFFSET  = 0.015             # gap between tick labels and bracket line
    BRACKET_TIP_LEN = 0.012             # little inward tip length
    LABEL_OFFSET    = 0.028 * scale     # gap between bracket and group label
    BRACKET_INSET   = 0.1               # how far inside each group-edge the bracket ends (data units)

    # x-axis (bottom): brackets below the x tick labels
    bracket_y_x = -x_pad - BRACKET_OFFSET
    label_y_x = bracket_y_x - LABEL_OFFSET

    for name, s, e in group_ranges:
        ax.plot([s + BRACKET_INSET, e - BRACKET_INSET], [bracket_y_x, bracket_y_x],
                transform=ax.get_xaxis_transform(),
                color="#222222", lw=1.2, clip_on=False, solid_capstyle="butt")
        ax.plot([s + BRACKET_INSET, s + BRACKET_INSET], [bracket_y_x, bracket_y_x + BRACKET_TIP_LEN],
                transform=ax.get_xaxis_transform(),
                color="#222222", lw=1.2, clip_on=False)
        ax.plot([e - BRACKET_INSET, e - BRACKET_INSET], [bracket_y_x, bracket_y_x + BRACKET_TIP_LEN],
                transform=ax.get_xaxis_transform(),
                color="#222222", lw=1.2, clip_on=False)
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


# ---------------------------------------------------------------------------
# Plot: top-N labelled matrix, lower-triangular, with numbers
# ---------------------------------------------------------------------------
def plot_top_n(corr, feature_vars, ranked_feats, name_map, top_n, od,
               ranking_label, filename_suffix):
    # ranked_feats: features in importance order; filename_suffix e.g. "" or "_shap"
    print(f"  [2/3] Top-{top_n} labelled matrix ({ranking_label})")
    top_feats = [f for f in ranked_feats if f in feature_vars][:top_n]
    if len(top_feats) < top_n:
        print(f"    WARNING: only {len(top_feats)} of the top {top_n} are in the cache")

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

# ---------------------------------------------------------------------------
# Plot: feature-vs-target correlation bar chart
# ---------------------------------------------------------------------------
def plot_target_corr(corr_with_target, feature_vars, shap_ranking, name_map, top_n, od):
    # shap_ranking: {feature: rank}; features without a SHAP value go to the bottom
    print("  [3/3] Feature-vs-target correlation bar chart")

    worst_rank = max(shap_ranking.values(), default=0) + 1 if shap_ranking else 999
    df = pd.DataFrame({
        "feature": feature_vars,
        "corr": corr_with_target,
        "shap_rank": [shap_ranking.get(f, worst_rank) for f in feature_vars],
    })
    # sort by SHAP rank; reversed below so rank 1 is at the top
    df = df.sort_values("shap_rank", ascending=True).reset_index(drop=True)

    df["display"] = df["feature"].map(lambda f: _nice(f, name_map))
    df["group_color"] = df["feature"].map(
        lambda f: GROUP_BASE_COLOR.get(_group_for(f)[0]) if _group_for(f) else "#888888")

    def _plot(df_sub, title, out_path):
        n = len(df_sub)
        fig, ax = plt.subplots(figsize=(10, max(6, 0.22 * n + 1)))
        y_positions = range(n - 1, -1, -1)
        ax.barh(list(y_positions), df_sub["corr"].values,
                color=df_sub["group_color"].values, edgecolor="none")
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(list(y_positions))
        SMALL_FONT = 9     # legend body, legend title, x axis label
        LARGE_FONT = 11    # y axis tick labels
        ax.set_yticklabels(df_sub["display"].values, fontsize=LARGE_FONT)
        ax.set_xlabel("Pearson correlation with target (SDS = 1)", fontsize=SMALL_FONT)
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # legend only for groups present in this panel
        groups_present = {
            _group_for(f)[0] for f in df_sub["feature"] if _group_for(f)
        }
        group_handles = [plt.Rectangle((0, 0), 1, 1,
                                       facecolor=GROUP_BASE_COLOR[name],
                                       edgecolor="none")
                         for name, _ in FEATURE_GROUPS if name in groups_present]
        group_labels = [name for name, _ in FEATURE_GROUPS if name in groups_present]
        ax.legend(group_handles, group_labels,
                        loc="lower right", frameon=True,
                        fontsize=SMALL_FONT, title="Feature group",
                        title_fontsize=SMALL_FONT)
        plt.tight_layout()
        fig.savefig(out_path); plt.close(fig)
        
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
    parser.add_argument("--fig_dir", type=str, default=None,
                        help="figure folder (default: figures/model/<model>/feature_correlations)")
    parser.add_argument("--sds_path", type=str, default=None)
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--spatial", type=float, default=None)
    parser.add_argument("--train_years", nargs=2, type=int, default=None)
    parser.add_argument("--val_years", nargs=2, type=int, default=None)
    parser.add_argument("--test_years", nargs=2, type=int, default=None)
    args = parser.parse_args()

    main_model_dir = resolve_path(args.main_model_dir)
    if not main_model_dir.exists():
        sys.exit(f"ERROR: main_model_dir not found: {main_model_dir}")

    main_results = json.loads((main_model_dir / "results.json").read_text())
    data_cfg = main_results["data_config"]
    resolution = args.resolution or data_cfg.get("resolution", RESOLUTION)
    spatial_res = args.spatial or data_cfg.get("spatial_res", SPATIAL_RES)
    train_years = args.train_years or data_cfg.get("train_years", TRAIN_YEARS)
    val_years = args.val_years or data_cfg.get("val_years", VAL_YEARS)
    test_years = args.test_years or data_cfg.get("test_years", TEST_YEARS)
    sds_path = resolve_path(args.sds_path or data_cfg.get("sds_path") or SDS_PATH)

    # permutation CSV: subsample-matched baseline from the dropout runs if present, else full model
    if args.perm_csv:
        perm_csv = resolve_path(args.perm_csv)
    else:
        matched = main_model_dir / "dropout" / "__baseline__" / "permutation_importance_matched.csv"
        fullmodel = main_model_dir / "permutation_importance" / "permutation_importance_test.csv"
        if matched.exists():
            perm_csv = matched
        elif fullmodel.exists():
            perm_csv = fullmodel
        else:
            sys.exit(f"ERROR: no permutation CSV found at {matched} or {fullmodel} "
                     "(run compute_permutation_importance.py first)")

    od = Path(args.out_dir) if args.out_dir else main_model_dir / "feature_correlations"
    od.mkdir(parents=True, exist_ok=True)
    fd = Path(args.fig_dir) if args.fig_dir else model_figures_dir(main_model_dir, "feature_correlations")
    fd.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("feat_corr")

    header("Feature correlations", {
        "Model": project_relative(main_model_dir),
        "Permutation ranking": project_relative(perm_csv),
        "Output": f"{project_relative(od)}, {project_relative(fd)}",
    }, out=logger.info)

    cache = FeatureCache(
        resolution=resolution, spatial_res=spatial_res,
        train_years=train_years, val_years=val_years, test_years=test_years,
        logger=logger,
    )
    feature_vars = cache.feature_vars
    name_map = get_feature_name_map(feature_vars)

    # feature-feature correlation (streaming)
    corr = compute_feature_correlation_streaming(cache, logger)
    pd.DataFrame(corr, index=feature_vars, columns=feature_vars).to_csv(
        od / "feature_correlations.csv")

    # feature-target correlation (streaming)
    sds_file = Path(sds_path)
    if not sds_file.exists():
        sys.exit(f"ERROR: labels not found: {sds_file} (run the backtracking first)")
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

    # plots
    perm_df = pd.read_csv(perm_csv)

    # load SHAP ranking for the target-correlation bar chart ordering
    shap_csv = resolve_path(args.shap_csv) if args.shap_csv else \
        main_model_dir / "shap" / "shap_feature_importance.csv"
    if shap_csv.exists():
        shap_df = pd.read_csv(shap_csv).sort_values("mean_abs_shap", ascending=False)
        shap_ranking = {f: r for r, f in enumerate(shap_df["feature"].values, start=1)}
    else:
        # no SHAP ranking: order by |corr_with_target|
        logger.info("  SHAP importance not found, ordering by |correlation| (run compute_shap.py first)")
        abs_order = np.argsort(-np.abs(corr_y))
        shap_ranking = {feature_vars[i]: r for r, i in enumerate(abs_order, start=1)}

    section("Figures", out=logger.info)
    plot_grouped(corr, feature_vars, name_map, fd)
    # permutation-ordered
    perm_ranked = perm_df["feature"].tolist()
    plot_top_n(corr, feature_vars, perm_ranked, name_map, args.top_n, fd,
               ranking_label="permutation importance",
               filename_suffix="")

    # SHAP-ordered (skipped without a SHAP ranking)
    if shap_csv.exists():
        shap_ranked = sorted(shap_ranking, key=shap_ranking.get)
        plot_top_n(corr, feature_vars, shap_ranked, name_map, args.top_n, fd,
                   ranking_label="SHAP importance",
                   filename_suffix="_shap")
    else:
        logger.info("    skipped: SHAP-ordered top-N matrix (no SHAP importance)")
    plot_target_corr(corr_y, feature_vars, shap_ranking, name_map, args.top_n, fd)

    done(od, out=logger.info)


if __name__ == "__main__":
    main()