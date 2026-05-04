"""
Compute SHAP values for the final RF model and produce diagnostic plots.

Uses PermutationExplainer throughout.

Outputs:
  01a. Bar plot — mean |SHAP| per feature (top 20)
  01b. Bar plot — mean |SHAP| per feature (top 30)
  01c. Bar plot — mean |SHAP| per feature (all features)
  02. Beeswarm plot — global feature importance with directionality
  03. Dependence plots — top features with interaction colouring
  04. Spatial maps — mean |SHAP| for top features
  05. Class-conditional SHAP — separate beeswarm for positives vs negatives
  07. CSV of raw SHAP values for further analysis
  08. Waterfall plots — individual TP / FP / FN exemplars
  09. Misclassification comparison — mean |SHAP| correct vs incorrect
  10. Temporal SHAP — monthly variation of top feature importances
  11. Spatial SHAP anomaly — where features deviate from global mean
  12. SHAP violin plots — distribution shape per feature
  13. Feature clustering dendrogram — redundancy analysis

All defaults from config.py, reads model directory from train_final_rf.py.

Usage:
  python compute_shap.py --model_dir models/final_rf_v1_20260413_120000
  python compute_shap.py --model_dir models/... --replot   # re-do plots only, reuses saved SHAP values
"""

import argparse
import gc
import json
import logging
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from config import (
    RESOLUTION, SPATIAL_RES, VAL_YEARS, TEST_YEARS, SDS_PATH,
    SHAP_N_SAMPLES, SHAP_N_BACKGROUND, SHAP_N_TREES,
    SHAP_TOP_FEATURES, SHAP_DEPENDENCE_FEATURES, SHAP_SEED,
)
from data_utils import (
    FeatureCache, load_sds_source, compute_land_indices,
    collect_indices, extract_from_cache,
    platt_apply,
)
from feature_names import get_feature_name_map

DOMAIN = {"north": 50.0, "south": 0.0, "west": -18.0, "east": 85.0}

TITLE_SIZE = 14
SUBTITLE_SIZE = 12
LABEL_SIZE = 11
TICK_SIZE = 9
LEGEND_SIZE = 9

# accent colour for SHAP importance bars (plain version) - matched to the SHAP
# accent used in compute_permutation_importance.py so the two scripts share an
# identity across the paper
SHAP_ACCENT_COLOR = "#a6cee3"

# outcome palette + alpha shared with analyze_rf_results.py. Used by the
# misclassification bar plot so its Correct/Incorrect bars match the TP/FN
# colours from every other figure in the paper.
OUTCOME_COLORS = {"TN": "#5B9BD5", "FP": "#ED7D31", "FN": "#C00000", "TP": "#70AD47"}
OUTCOME_ALPHA = 0.7

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": LABEL_SIZE,
    "axes.titlesize": SUBTITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.dpi": 150,
    "savefig.dpi": 250,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

_YEAR_RANGES: dict = {}  # filled in main, same pattern as analyze_rf_results.py


# ---------------------------------------------------------------------------
# Feature groups + Paired colours
# (kept in sync with compute_feature_correlations.py and
#  compute_permutation_importance.py. Any palette change should happen in all
#  three files so figures stay consistent across the paper.)
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

# Paired indexing: 0=light blue, 1=dark blue, 2=light green, 3=dark green,
# 4=light red, 5=dark red, 6=light orange, 7=dark orange. Shades pair
# semantically-related groups (moisture=blue, vegetation=green, thermal=orange,
# static=red).
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
UNGROUPED_COLOR = "#888888"   # fallback for features not in any group


def _group_name_for(feat):
    # returns the group name for a feature, or None if no group matches
    for name, members in FEATURE_GROUPS:
        if feat in members:
            return name
    return None


def _group_color_for(feat):
    # returns the Paired-based group colour for a feature, or a neutral grey fallback
    g = _group_name_for(feat)
    return GROUP_BASE_COLOR[g] if g is not None else UNGROUPED_COLOR


def _group_legend_handles(features_in_plot):
    # build legend handles only for the groups actually present in the panel,
    # keeping them in the canonical FEATURE_GROUPS order so legends across
    # figures sort consistently
    groups_present = {g for g in (_group_name_for(f) for f in features_in_plot)
                      if g is not None}
    handles = [plt.Rectangle((0, 0), 1, 1,
                             facecolor=GROUP_BASE_COLOR[name],
                             edgecolor="none")
               for name, _ in FEATURE_GROUPS if name in groups_present]
    labels = [name for name, _ in FEATURE_GROUPS if name in groups_present]
    # if any features fall outside the defined groups, add a single "Other" entry
    if any(_group_name_for(f) is None for f in features_in_plot):
        handles.append(plt.Rectangle((0, 0), 1, 1,
                                     facecolor=UNGROUPED_COLOR, edgecolor="none"))
        labels.append("Other")
    return handles, labels


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tag_label(tag):
    # e.g. "Test (2024)" or "Validation (2022-2023)"
    yr = _YEAR_RANGES.get(tag)
    if yr is None:
        return tag
    yr_str = str(yr[0]) if yr[0] == yr[1] else f"{yr[0]}–{yr[1]}"
    return f"{'Test' if tag == 'test' else 'Validation'} ({yr_str})"


def _clean_name(name):
    # strip units (anything in square brackets) for cleaner axis labels
    if not name:
        return ""
    return re.sub(r"\s*\[.*?\]", "", name)


def _nice(feat, name_map):
    # look up a display name, falling back to the short feature name if the map
    # is missing an entry or returns None (happens for features without a long name)
    val = name_map.get(feat)
    return _clean_name(val) if val else feat


def _display_names(feature_vars, name_map):
    # list of display names, parallel to feature_vars
    return [_nice(f, name_map) for f in feature_vars]


def _map_ax(fig, pos=111):
    # standard cartopy axes with coastlines/borders/gridlines
    ax = fig.add_subplot(pos, projection=ccrs.PlateCarree())
    ax.set_extent([DOMAIN["west"], DOMAIN["east"],
                   DOMAIN["south"], DOMAIN["north"]],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, lw=0.6, ec="black", zorder=5)
    ax.add_feature(cfeature.BORDERS, lw=0.3, ec="#444", ls="--", zorder=5)
    gl = ax.gridlines(draw_labels=True, lw=0.3, alpha=0.3, ls=":")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": TICK_SIZE}
    return ax


def _mapplot(lons, lats, data, title, cmap, vmin, vmax, fname, od,
             label="", alpha=0.80):
    # generic pcolormesh map saver - shared by spatial SHAP and anomaly plots
    lo2, la2 = np.meshgrid(lons, lats)
    lon_range = DOMAIN["east"] - DOMAIN["west"]
    lat_range = DOMAIN["north"] - DOMAIN["south"]
    fig_w = 13
    fig_h = fig_w * (lat_range / lon_range) + 1.8
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = _map_ax(fig)
    kw = {"transform": ccrs.PlateCarree()}
    im = ax.pcolormesh(lo2, la2, np.ma.masked_invalid(data), cmap=cmap,
                       vmin=vmin, vmax=vmax, shading="auto", alpha=alpha,
                       zorder=1, **kw)
    ax.set_extent([DOMAIN["west"], DOMAIN["east"],
                   DOMAIN["south"], DOMAIN["north"]],
                  crs=ccrs.PlateCarree())
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.07,
                      shrink=0.7, label=label)
    cb.ax.tick_params(labelsize=TICK_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=20)
    fig.savefig(od / fname)
    plt.close(fig)


def outcome_labels(yt, yp):
    # TN / FP / FN / TP per sample (same helper as the other scripts)
    o = np.empty(len(yt), dtype=object)
    o[(yt == 0) & (yp == 0)] = "TN"
    o[(yt == 0) & (yp == 1)] = "FP"
    o[(yt == 1) & (yp == 0)] = "FN"
    o[(yt == 1) & (yp == 1)] = "TP"
    return o


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_01_bar(feat_importance, feature_vars, od, tag, name_map, logger,
                group_colored=False):
    # three bar charts of mean |SHAP|: top-20, top-30, and everything.
    # `group_colored=False` paints every bar with the shared SHAP accent colour
    # so this plot reads as "SHAP importance" at a glance; `group_colored=True`
    # colours each bar by its feature group (Paired palette, same mapping as
    # the correlation + permutation scripts) to foreground which variable
    # families dominate. Grouped variants are saved alongside the plain ones
    # with a `_grouped` suffix.
    #
    # Font sizes are kept consistent with compute_feature_correlations.py:
    #   TICK_SIZE (9pt) for feature-name y-tick labels and x-axis tick labels
    #   GROUP_FONT (11pt) for the group legend, matching the bracket labels
    #   on the correlation matrix.
    GROUP_FONT = 11   # matches GROUP_FONT in plot_grouped (compute_feature_correlations.py)
    suffix_log = " (grouped)" if group_colored else ""
    logger.info(f"  [01] Bar plots{suffix_log} (top-20, top-30, all) ...")
    n_all = len(feature_vars)
    variants = [
        ("01a", min(20, n_all), "Top 20"),
        ("01b", min(30, n_all), "Top 30"),
        ("01c", n_all, "All"),
    ]
    for prefix, n, subtitle in variants:
        top = feat_importance.head(n).iloc[::-1].copy()
        top["display_name"] = top["feature"].map(lambda f: _nice(f, name_map))
        if group_colored:
            bar_colors = [_group_color_for(f) for f in top["feature"].values]
        else:
            bar_colors = SHAP_ACCENT_COLOR

        # CHANGED: fig height 0.30 → 0.34 per row to give taller 9pt labels room
        fig, ax = plt.subplots(figsize=(9, max(5, n * 0.34)))
        ax.barh(range(len(top)), top["mean_abs_shap"].values, color=bar_colors)
        ax.set_yticks(range(len(top)))
        # CHANGED: fontsize 10 → TICK_SIZE (9) to match correlation script tick labels
        ax.set_yticklabels(top["display_name"].values, fontsize=TICK_SIZE)
        ax.set_xlabel("Mean |SHAP value|")
        # CHANGED: labelsize 8 → TICK_SIZE (9) for x-axis tick consistency
        ax.tick_params(axis='x', labelsize=TICK_SIZE)
        title_suffix = " — grouped" if group_colored else ""
        ax.set_title(
            f"SHAP Feature Importance ({subtitle}){title_suffix} — {_tag_label(tag)}",
            fontsize=TITLE_SIZE, fontweight="bold", loc="center", pad=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if group_colored:
            # legend in the bottom-right corner of the axis, where the shortest
            # bars sit (since we sort ascending) - same placement strategy as
            # the grouped comparison plot in compute_permutation_importance.py.
            # CHANGED: fontsize + title_fontsize 10 → GROUP_FONT (11) to match
            # the bracket group labels on the correlation matrix.
            handles, labels = _group_legend_handles(top["feature"].tolist())
            ax.legend(handles, labels,
                      loc="lower right", frameon=True, fontsize=GROUP_FONT,
                      title="Feature group", title_fontsize=GROUP_FONT)

        file_suffix = "_grouped" if group_colored else ""
        fig.savefig(od / f"{prefix}_shap_bar_{tag}{file_suffix}.png")
        plt.close(fig)


def plot_02_beeswarm(sv, X_shap, base_value, feature_vars, top_n, od, tag,
                     name_map, logger):
    # global beeswarm - one dot per sample, coloured by feature value
    import shap
    logger.info(f"  [02] Beeswarm plot ...")
    explanation = shap.Explanation(
        values=sv, base_values=base_value, data=X_shap,
        feature_names=_display_names(feature_vars, name_map),
    )
    fig = plt.figure(figsize=(10, max(6, top_n * 0.35)))
    shap.plots.beeswarm(explanation, max_display=top_n, show=False)
    plt.title(f"SHAP Beeswarm — {_tag_label(tag)}",
              fontsize=TITLE_SIZE, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(od / f"02_shap_beeswarm_{tag}.png")
    plt.close("all")


def plot_03_dependence(sv, X_shap, base_value, feat_importance, feature_vars,
                       od, tag, name_map, logger):
    # dependence plots for the top features, with automatic interaction colouring
    import shap
    logger.info(f"  [03] Dependence plots ...")
    n_dep = min(SHAP_DEPENDENCE_FEATURES, len(feature_vars))
    top_feats = feat_importance.head(n_dep)["feature"].tolist()
    feat_idx_map = {f: i for i, f in enumerate(feature_vars)}
    explanation = shap.Explanation(
        values=sv, base_values=base_value, data=X_shap,
        feature_names=_display_names(feature_vars, name_map),
    )
    for rank, feat in enumerate(top_feats, 1):
        fi = feat_idx_map[feat]
        display = _nice(feat, name_map)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        shap.plots.scatter(explanation[:, fi], color=explanation,
                           ax=ax, show=False)
        ax.set_title(f"SHAP Dependence: {display} — {_tag_label(tag)}",
                     fontsize=SUBTITLE_SIZE, fontweight="bold", pad=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.savefig(od / f"03_{rank:02d}_dependence_{feat}_{tag}.png")
        plt.close(fig)


def plot_04_spatial(sv, lats, lons, feat_importance, feature_vars, od, tag,
                    name_map, logger):
    # per-cell mean |SHAP| maps for the top 6 features
    logger.info(f"  [04] Spatial SHAP maps ...")
    feat_idx_map = {f: i for i, f in enumerate(feature_vars)}
    top_feats = feat_importance.head(6)["feature"].tolist()

    unique_lats = np.sort(np.unique(lats))
    unique_lons = np.sort(np.unique(lons))
    nlat, nlon = len(unique_lats), len(unique_lons)
    lat_idx = np.searchsorted(unique_lats, lats)
    lon_idx = np.searchsorted(unique_lons, lons)

    for rank, feat in enumerate(top_feats, 1):
        fi = feat_idx_map[feat]
        display = _nice(feat, name_map)
        grid_sum = np.zeros((nlat, nlon))
        grid_cnt = np.zeros((nlat, nlon))
        for k in range(len(lats)):
            grid_sum[lat_idx[k], lon_idx[k]] += abs(sv[k, fi])
            grid_cnt[lat_idx[k], lon_idx[k]] += 1
        grid_mean = np.where(grid_cnt > 0, grid_sum / grid_cnt, np.nan)
        _mapplot(unique_lons, unique_lats, grid_mean,
                 f"Spatial SHAP: {display} — {_tag_label(tag)}",
                 "YlOrRd", 0, None,
                 f"04_{rank:02d}_spatial_shap_{feat}_{tag}.png", od,
                 label=f"Mean |SHAP| for {display}")


def plot_05_class_conditional(sv, X_shap, y_shap, base_value, feature_vars,
                              top_n, od, tag, name_map, logger):
    # separate beeswarms for SDS-positive and SDS-negative samples
    import shap
    logger.info(f"  [05] Class-conditional SHAP ...")
    display_names = _display_names(feature_vars, name_map)
    for cls, label in [(1, "positive"), (0, "negative")]:
        mask = y_shap == cls
        if mask.sum() < 10:
            continue
        cls_explanation = shap.Explanation(
            values=sv[mask],
            base_values=base_value,
            data=X_shap[mask],
            feature_names=display_names,
        )
        fig = plt.figure(figsize=(10, max(6, top_n * 0.35)))
        shap.plots.beeswarm(cls_explanation, max_display=top_n, show=False)
        plt.title(f"SHAP Beeswarm ({label} samples) — {_tag_label(tag)}",
                  fontsize=TITLE_SIZE, fontweight="bold", pad=20)
        plt.tight_layout()
        fig.savefig(od / f"05_shap_beeswarm_{label}_{tag}.png")
        plt.close("all")


def plot_08_waterfall(sv, X_shap, y_shap, proba, yp, base_value,
                      feature_vars, od, tag, name_map, logger):
    # waterfall plots for one exemplar each of TP, FP, FN
    # - TP/FP: most confident positive prediction, FN: least confident positive prediction
    import shap
    logger.info(f"  [08] Waterfall plots (exemplars) ...")
    oc = outcome_labels(y_shap, yp)
    cases = {}
    for oc_label in ["TP", "FP", "FN"]:
        mask = oc == oc_label
        if mask.sum() == 0:
            continue
        if oc_label in ("TP", "FP"):
            idx = np.where(mask)[0][np.argmax(proba[mask])]
        else:
            idx = np.where(mask)[0][np.argmin(proba[mask])]
        cases[oc_label] = idx

    display_names = _display_names(feature_vars, name_map)
    for oc_label, idx in cases.items():
        ex = shap.Explanation(
            values=sv[idx],
            base_values=base_value,
            data=X_shap[idx],
            feature_names=display_names,
        )
        fig = plt.figure(figsize=(10, 7))
        shap.plots.waterfall(ex, max_display=15, show=False)
        true_lbl = "SDS+" if y_shap[idx] == 1 else "SDS−"
        plt.title(
            f"SHAP Waterfall — {oc_label} exemplar "
            f"(p={proba[idx]:.4f}, {true_lbl}) — {_tag_label(tag)}",
            fontsize=SUBTITLE_SIZE, fontweight="bold", pad=20)
        plt.tight_layout()
        fig.savefig(od / f"08_waterfall_{oc_label}_{tag}.png")
        plt.close("all")


def plot_09_misclassification(sv, y_shap, yp, feat_importance, feature_vars,
                              od, tag, name_map, logger):
    # mean |SHAP| comparison between correctly and incorrectly classified samples
    logger.info(f"  [09] Misclassification SHAP comparison ...")
    correct = y_shap == yp
    incorrect = ~correct

    if incorrect.sum() < 5:
        logger.info("    Too few misclassifications, skipping")
        return

    top_n = min(20, len(feature_vars))
    top_feats = feat_importance.head(top_n)["feature"].tolist()
    feat_idx_map = {f: i for i, f in enumerate(feature_vars)}
    top_idx = [feat_idx_map[f] for f in top_feats]

    mean_correct = np.abs(sv[correct][:, top_idx]).mean(axis=0)
    mean_incorrect = np.abs(sv[incorrect][:, top_idx]).mean(axis=0)
    display_names = [_nice(f, name_map) for f in top_feats]

    x = np.arange(top_n)
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, max(5, top_n * 0.3)))
    ax.barh(x + width / 2, mean_correct[::-1], width,
            color=OUTCOME_COLORS["TP"], alpha=OUTCOME_ALPHA, label="Correct")
    ax.barh(x - width / 2, mean_incorrect[::-1], width,
            color=OUTCOME_COLORS["FN"], alpha=OUTCOME_ALPHA, label="Incorrect")
    ax.set_yticks(x)
    ax.set_yticklabels(display_names[::-1], fontsize=TICK_SIZE)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(
        f"SHAP: Correct vs Misclassified — {_tag_label(tag)}",
        fontsize=TITLE_SIZE, fontweight="bold", pad=20)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(od / f"09_misclassification_shap_{tag}.png")
    plt.close(fig)


def plot_10_temporal(sv, times, feat_importance, feature_vars, od, tag,
                     name_map, logger):
    # month-of-year mean |SHAP| for the top 8 features - reveals seasonal shifts in importance
    logger.info(f"  [10] Temporal SHAP ...")
    if times is None or len(times) == 0:
        logger.info("    No time data, skipping")
        return

    top_n = min(8, len(feature_vars))
    top_feats = feat_importance.head(top_n)["feature"].tolist()
    feat_idx_map = {f: i for i, f in enumerate(feature_vars)}

    tpd = pd.to_datetime(times)
    months = tpd.month
    unique_months = np.sort(np.unique(months))
    month_names = ["J", "F", "M", "A", "M", "J",
                   "J", "A", "S", "O", "N", "D"]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors_cycle = plt.cm.tab10(np.linspace(0, 1, top_n))

    for rank, feat in enumerate(top_feats):
        fi = feat_idx_map[feat]
        display = _nice(feat, name_map)
        monthly_means = []
        for m in unique_months:
            mask = months == m
            monthly_means.append(
                np.abs(sv[mask, fi]).mean() if mask.sum() > 0 else np.nan)
        ax.plot(unique_months, monthly_means, "o-", lw=1.8, ms=4,
                color=colors_cycle[rank], label=display)

    ax.set_xticks(unique_months)
    ax.set_xticklabels([month_names[m - 1] for m in unique_months],
                       fontsize=TICK_SIZE)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title(f"Seasonal SHAP Importance — {_tag_label(tag)}",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=20)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper right", ncol=2)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(od / f"10_temporal_shap_{tag}.png")
    plt.close(fig)


def plot_11_spatial_anomaly(sv, lats, lons, feat_importance, feature_vars,
                            od, tag, name_map, logger):
    # spatial deviation of |SHAP| from the global mean - red = feature matters more here than average
    logger.info(f"  [11] Spatial SHAP anomaly maps ...")
    feat_idx_map = {f: i for i, f in enumerate(feature_vars)}
    top_feats = feat_importance.head(4)["feature"].tolist()

    unique_lats = np.sort(np.unique(lats))
    unique_lons = np.sort(np.unique(lons))
    nlat, nlon = len(unique_lats), len(unique_lons)
    lat_idx = np.searchsorted(unique_lats, lats)
    lon_idx = np.searchsorted(unique_lons, lons)

    for rank, feat in enumerate(top_feats, 1):
        fi = feat_idx_map[feat]
        display = _nice(feat, name_map)
        global_mean = np.abs(sv[:, fi]).mean()

        grid_sum = np.zeros((nlat, nlon))
        grid_cnt = np.zeros((nlat, nlon))
        for k in range(len(lats)):
            grid_sum[lat_idx[k], lon_idx[k]] += abs(sv[k, fi])
            grid_cnt[lat_idx[k], lon_idx[k]] += 1
        grid_mean = np.where(grid_cnt > 0, grid_sum / grid_cnt, np.nan)
        anomaly = grid_mean - global_mean

        vabs = np.nanpercentile(np.abs(anomaly), 95) if np.any(
            np.isfinite(anomaly)) else 1
        _mapplot(unique_lons, unique_lats, anomaly,
                 f"SHAP Anomaly: {display} — {_tag_label(tag)}",
                 "RdBu_r", -vabs, vabs,
                 f"11_{rank:02d}_spatial_anomaly_{feat}_{tag}.png",
                 od, label="|SHAP| anomaly (local − global mean)")


def plot_12_violin(sv, X_shap, base_value, feature_vars, top_n, od, tag,
                   name_map, logger):
    # violin plot version of the beeswarm - nicer when samples are very dense
    import shap
    logger.info(f"  [12] Violin plots ...")
    explanation = shap.Explanation(
        values=sv, base_values=base_value, data=X_shap,
        feature_names=_display_names(feature_vars, name_map),
    )
    fig = plt.figure(figsize=(10, max(6, top_n * 0.35)))
    shap.plots.violin(explanation, max_display=top_n, show=False)
    plt.title(f"SHAP Violin — {_tag_label(tag)}",
              fontsize=TITLE_SIZE, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(od / f"12_shap_violin_{tag}.png")
    plt.close("all")


def plot_13_clustering(sv, X_shap, base_value, feature_vars, od, tag,
                       name_map, logger):
    # hclust on SHAP-value correlation, rendered as a bar+dendrogram from shap
    # groups features that behave similarly
    import shap
    logger.info(f"  [13] Feature clustering ...")
    explanation = shap.Explanation(
        values=sv, base_values=base_value, data=X_shap,
        feature_names=_display_names(feature_vars, name_map),
    )
    try:
        fig = plt.figure(figsize=(12, 7))
        clust = shap.utils.hclust(explanation.values, explanation.data)
        shap.plots.bar(explanation, clustering=clust, max_display=20,
                       show=False)
        plt.title(f"SHAP Feature Clustering — {_tag_label(tag)}",
                  fontsize=TITLE_SIZE, fontweight="bold", pad=20)
        plt.tight_layout()
        fig.savefig(od / f"13_shap_clustering_{tag}.png")
        plt.close("all")
    except Exception as e:
        logger.warning(f"  Clustering plot failed: {e}")
        plt.close("all")


# ---------------------------------------------------------------------------
# SHAP value I/O (used by both compute and --replot paths)
# ---------------------------------------------------------------------------

def save_shap_data(od, sv, X_shap, y_shap, proba, lats, lons, times,
                   base_value, feature_vars, has_times):
    # npz for fast replot, CSV for downstream analysis, JSON for feature order
    npz_kwargs = dict(
        shap_values=sv, X_shap=X_shap, y_shap=y_shap,
        proba=proba, lats=lats, lons=lons,
        base_value=np.array([base_value]),
    )
    if has_times:
        # store as int64 nanoseconds so npz doesn't choke on datetime64 dtype
        npz_kwargs["times"] = times.astype("datetime64[ns]").astype(np.int64)
    np.savez_compressed(od / "shap_data.npz", **npz_kwargs)
    (od / "shap_feature_vars.json").write_text(json.dumps(feature_vars))

    # row-per-sample CSV (handy for external tools)
    shap_df = pd.DataFrame(sv, columns=feature_vars)
    shap_df["y_true"] = y_shap
    shap_df["proba"] = proba
    shap_df["lat"] = lats
    shap_df["lon"] = lons
    if has_times:
        shap_df["time"] = times
    shap_df.to_csv(od / "shap_values.csv", index=False)


def load_shap_data(od, logger):
    # used by --replot mode. tolerates older runs that didn't save `times` in the npz
    # (falls back to reading `time` column from shap_values.csv)
    npz_file = od / "shap_data.npz"
    json_file = od / "shap_feature_vars.json"
    if not npz_file.exists() or not json_file.exists():
        raise FileNotFoundError(
            f"Cannot replot: expected {npz_file} and {json_file}. "
            f"Run without --replot first to compute SHAP values.")

    logger.info(f"  Loading SHAP values from {npz_file} ...")
    data = np.load(npz_file)
    sv        = data["shap_values"]
    X_shap    = data["X_shap"]
    y_shap    = data["y_shap"]
    proba     = data["proba"]
    lats      = data["lats"]
    lons      = data["lons"]
    base_value = float(data["base_value"][0])

    if "times" in data.files:
        times = data["times"].astype("datetime64[ns]")
        has_times = True
    else:
        # backward-compat: pull times from the CSV if present
        csv_file = od / "shap_values.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file, usecols=["time"]) if "time" in \
                pd.read_csv(csv_file, nrows=0).columns else None
            if df is not None:
                times = pd.to_datetime(df["time"]).values
                has_times = True
                logger.info("  (times loaded from shap_values.csv)")
            else:
                times = None
                has_times = False
        else:
            times = None
            has_times = False

    feature_vars = json.loads(json_file.read_text())
    logger.info(f"  Loaded: {sv.shape[0]:,} samples × {sv.shape[1]} features")
    return sv, X_shap, y_shap, proba, lats, lons, times, base_value, \
           feature_vars, has_times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute SHAP values for the final RF model")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--sds_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--n_samples", type=int, default=None,
                        help=f"Samples for SHAP values (default: {SHAP_N_SAMPLES})")
    parser.add_argument("--n_background", type=int, default=None,
                        help=f"Background dataset size (default: {SHAP_N_BACKGROUND})")
    parser.add_argument("--n_trees", type=int, default=None,
                        help=f"Trees to use, 0=all (default: {SHAP_N_TREES})")
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--spatial", type=float, default=None)
    parser.add_argument("--val_years", nargs=2, type=int, default=None)
    parser.add_argument("--test_years", nargs=2, type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--replot", action="store_true",
                        help="Skip SHAP computation, reuse saved values to "
                             "regenerate the plots only")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: model_dir not found: {model_dir}")
        sys.exit(1)

    # always read results.json - both paths need thr, feature_vars, and the year ranges
    results = json.loads((model_dir / "results.json").read_text())
    thr = results["threshold"]
    feature_vars = results["feature_vars"]
    data_cfg = results["data_config"]

    val_years = args.val_years or data_cfg.get("val_years", VAL_YEARS)
    test_years = args.test_years or data_cfg.get("test_years", TEST_YEARS)

    _YEAR_RANGES["val"]  = tuple(val_years)
    _YEAR_RANGES["test"] = tuple(test_years)

    od = Path(args.out_dir) if args.out_dir else model_dir / "shap"
    od.mkdir(parents=True, exist_ok=True)

    name_map = get_feature_name_map(feature_vars)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(od / "shap.log"),
        ],
    )
    logger = logging.getLogger("shap")

    # ── Replot-only path ──────────────────────────────────────────────────────
    if args.replot:
        logger.info("=" * 60)
        logger.info(f"SHAP REPLOT (reusing saved values in {od})")
        logger.info("=" * 60)
        sv, X_shap, y_shap, proba, lats, lons, times, base_value, \
            saved_feature_vars, has_times = load_shap_data(od, logger)

        # sanity check: feature_vars in results.json vs saved - they must match order
        if saved_feature_vars != feature_vars:
            logger.warning(
                "  Saved feature_vars differ from results.json; "
                "using saved order.")
            feature_vars = saved_feature_vars
            name_map = get_feature_name_map(feature_vars)

        yp = (proba >= thr).astype(int)
        t_shap = 0.0
        _run_all_plots(sv, X_shap, y_shap, proba, yp, lats, lons, times,
                       base_value, feature_vars, name_map, has_times,
                       od, args.split, t_shap, logger)
        return

    # ── Full compute path ─────────────────────────────────────────────────────
    rf = joblib.load(model_dir / "rf_model.joblib")
    platt_path = model_dir / "platt_scaler.joblib"
    platt_lr = joblib.load(platt_path) if platt_path.exists() else None

    # optional tree subsetting - SHAP's runtime scales linearly with n_estimators,
    # so using fewer trees is a quick way to get a coarse SHAP run for debugging
    n_trees = args.n_trees if args.n_trees is not None else SHAP_N_TREES
    if n_trees > 0 and n_trees < rf.n_estimators:
        full_n = rf.n_estimators
        rf.estimators_ = rf.estimators_[:n_trees]
        rf.n_estimators = n_trees
        print(f"  Tree subset: using {n_trees} of {full_n} trees for SHAP")
    else:
        n_trees = rf.n_estimators
        print(f"  Using all {n_trees} trees")

    # CLI overrides → results.json → config defaults
    resolution = args.resolution or data_cfg.get("resolution", RESOLUTION)
    spatial_res = args.spatial or data_cfg.get("spatial_res", SPATIAL_RES)
    sds_path = args.sds_path or data_cfg.get("sds_path") or SDS_PATH
    n_samples = args.n_samples or SHAP_N_SAMPLES
    n_background = args.n_background or SHAP_N_BACKGROUND

    if sds_path is None:
        print("ERROR: No SDS path")
        sys.exit(1)
    sds_file = Path(sds_path)

    split_years = test_years if args.split == "test" else val_years

    logger.info("=" * 60)
    logger.info("SHAP ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"  Model: {model_dir}")
    logger.info(f"  Split: {args.split} ({split_years})")
    logger.info(f"  Samples: {n_samples:,}")
    logger.info(f"  Background: {n_background}")
    logger.info(f"  Trees: {n_trees}")
    logger.info(f"  Features: {len(feature_vars)}")
    logger.info(f"  Threshold: {thr:.4f}")
    logger.info(f"  Output: {od}")
    logger.info("=" * 60)

    cache = FeatureCache(
        resolution=resolution,
        spatial_res=spatial_res,
        val_years=val_years if args.split == "val" else None,
        test_years=test_years if args.split == "test" else None,
        logger=logger,
    )

    ds_sds, sds_var = load_sds_source(sds_file, logger)
    land_indices = compute_land_indices(
        ds_sds["latitude"].values, ds_sds["longitude"].values)

    years = list(range(split_years[0], split_years[1] + 1))
    logger.info(f"\nCollecting indices for {args.split} ...")
    all_pos, all_neg = collect_indices(
        cache, years, ds_sds, sds_var, land_indices, logger)

    ds_sds.close()
    del ds_sds
    gc.collect()

    # stratified subsample - keeps the pos/neg ratio of the full split
    rng = np.random.default_rng(SHAP_SEED)
    n_total = len(all_pos) + len(all_neg)
    pos_frac = len(all_pos) / n_total
    n_pos_sample = max(1, int(round(n_samples * pos_frac)))
    n_neg_sample = n_samples - n_pos_sample
    n_pos_sample = min(n_pos_sample, len(all_pos))
    n_neg_sample = min(n_neg_sample, len(all_neg))

    logger.info(f"  Subsampling: {n_pos_sample:,} pos + {n_neg_sample:,} neg "
                f"= {n_pos_sample + n_neg_sample:,}")

    pos_sel = rng.choice(len(all_pos), size=n_pos_sample, replace=False)
    neg_sel = rng.choice(len(all_neg), size=n_neg_sample, replace=False)
    sample_indices = ([all_pos[i] for i in pos_sel] +
                      [all_neg[i] for i in neg_sel])

    X_shap = extract_from_cache(cache, sample_indices)
    y_shap = np.concatenate([
        np.ones(n_pos_sample, dtype=np.float32),
        np.zeros(n_neg_sample, dtype=np.float32),
    ])

    # pull lat/lon/time from each year's feature netcdf
    import xarray as xr
    year_coords = {}
    for year in set(idx[0] for idx in sample_indices):
        feat_file = cache.feat_dir / f"features_{year}_{cache.ftag}.nc"
        ds = xr.open_dataset(feat_file)
        lat_v = ds["lat"].values if "lat" in ds else ds["latitude"].values
        lon_v = ds["lon"].values if "lon" in ds else ds["longitude"].values
        time_v = ds["valid_time"].values if "valid_time" in ds else None
        ds.close()
        year_coords[year] = (lat_v, lon_v, time_v)

    lats = np.empty(len(sample_indices), dtype=np.float32)
    lons = np.empty(len(sample_indices), dtype=np.float32)
    times = np.empty(len(sample_indices), dtype="datetime64[ns]")
    has_times = True
    for out_i, (year, idx) in enumerate(sample_indices):
        lat_v, lon_v, time_v = year_coords[year]
        lats[out_i] = lat_v[idx]
        lons[out_i] = lon_v[idx]
        if time_v is not None:
            times[out_i] = time_v[idx]
        else:
            has_times = False

    del all_pos, all_neg
    gc.collect()

    proba_raw = rf.predict_proba(X_shap)[:, 1]
    proba = platt_apply(platt_lr, proba_raw) if platt_lr is not None else proba_raw
    yp = (proba >= thr).astype(int)

    logger.info(f"  Predictions: {yp.sum():,} positive, "
                f"{(yp == 0).sum():,} negative")

    # ── Compute SHAP values via PermutationExplainer ──────────────────────
    import shap

    logger.info("\nComputing SHAP values (PermutationExplainer) ...")
    t0 = time.time()

    n_bg = min(n_background, len(X_shap))
    rng_bg = np.random.default_rng(SHAP_SEED + 1)
    bg_idx = rng_bg.choice(len(X_shap), size=n_bg, replace=False)
    background = X_shap[bg_idx]
    logger.info(f"  Background: {n_bg} samples")

    masker = shap.maskers.Independent(background)
    explainer = shap.PermutationExplainer(rf.predict_proba, masker)

    # chunk the calls so we get progress logs and don't blow up memory
    chunk_size = 200
    n_chunks = int(np.ceil(len(X_shap) / chunk_size))
    sv_chunks = []

    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min(start + chunk_size, len(X_shap))
        try:
            chunk_result = explainer(X_shap[start:end], silent=True)
        except TypeError:
            chunk_result = explainer(X_shap[start:end])
        chunk_sv = chunk_result.values
        if chunk_sv.ndim == 3:
            chunk_sv = chunk_sv[:, :, 1]   # pick the positive-class SHAPs
        sv_chunks.append(chunk_sv)

        elapsed = time.time() - t0
        rate = end / elapsed if elapsed > 0 else 0
        eta = (len(X_shap) - end) / rate if rate > 0 else 0
        logger.info(f"    {end:,}/{len(X_shap):,} "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    sv = np.vstack(sv_chunks)
    del sv_chunks, background

    t_shap = time.time() - t0
    logger.info(f"  SHAP values computed in {t_shap:.1f}s")
    logger.info(f"  Shape: {sv.shape}")

    # base value is the same across chunks for Permutation (it's a property of the background),
    # so grabbing it from the last chunk is fine
    bv = chunk_result.base_values
    base_value = float(bv[0, 1] if bv.ndim == 2 else bv[0])

    # save everything needed for --replot plus the CSV for downstream use
    save_shap_data(od, sv, X_shap, y_shap, proba, lats, lons, times,
                   base_value, feature_vars, has_times)
    logger.info(f"  Saved: shap_data.npz, shap_values.csv, "
                f"shap_feature_vars.json")

    _run_all_plots(sv, X_shap, y_shap, proba, yp, lats, lons, times,
                   base_value, feature_vars, name_map, has_times,
                   od, args.split, t_shap, logger)

    del X_shap, sv
    gc.collect()


def _run_all_plots(sv, X_shap, y_shap, proba, yp, lats, lons, times,
                   base_value, feature_vars, name_map, has_times,
                   od, split, t_shap, logger):
    # shared by the compute path and --replot, so both routes produce identical outputs
    mean_abs_shap = np.abs(sv).mean(axis=0)
    feat_importance = pd.DataFrame({
        "feature": feature_vars,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    feat_importance["rank"] = range(1, len(feat_importance) + 1)
    feat_importance.to_csv(od / "shap_feature_importance.csv", index=False)

    top_n = min(SHAP_TOP_FEATURES, len(feature_vars))

    logger.info("\nPlotting ...")

    # plain (solid SHAP accent) + grouped (Paired-per-group) versions of the bar set
    for gc_flag in (False, True):
        plot_01_bar(feat_importance, feature_vars, od, split, name_map, logger,
                    group_colored=gc_flag)

    plot_02_beeswarm(sv, X_shap, base_value, feature_vars, top_n,
                     od, split, name_map, logger)
    plot_03_dependence(sv, X_shap, base_value, feat_importance, feature_vars,
                       od, split, name_map, logger)
    plot_04_spatial(sv, lats, lons, feat_importance, feature_vars,
                    od, split, name_map, logger)
    plot_05_class_conditional(sv, X_shap, y_shap, base_value, feature_vars,
                              top_n, od, split, name_map, logger)
    plot_08_waterfall(sv, X_shap, y_shap, proba, yp, base_value,
                      feature_vars, od, split, name_map, logger)
    plot_09_misclassification(sv, y_shap, yp, feat_importance, feature_vars,
                              od, split, name_map, logger)
    if has_times:
        plot_10_temporal(sv, times, feat_importance, feature_vars,
                         od, split, name_map, logger)
    plot_11_spatial_anomaly(sv, lats, lons, feat_importance, feature_vars,
                            od, split, name_map, logger)
    plot_12_violin(sv, X_shap, base_value, feature_vars, top_n,
                   od, split, name_map, logger)
    plot_13_clustering(sv, X_shap, base_value, feature_vars,
                       od, split, name_map, logger)

    summary = {
        "split": split,
        "n_samples": len(X_shap),
        "n_pos": int(y_shap.sum()),
        "n_neg": int((y_shap == 0).sum()),
        "shap_time_s": round(t_shap, 1),
        "method": "PermutationExplainer",
        "top_features": feat_importance.head(SHAP_TOP_FEATURES).to_dict("records"),
    }
    with open(od / f"shap_summary_{split}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n  Top {top_n} features by mean |SHAP|:")
    for _, row in feat_importance.head(top_n).iterrows():
        display = _nice(row["feature"], name_map)
        logger.info(f"    {row['rank']:3d}. {display:<45s} "
                    f"{row['mean_abs_shap']:.6f}")

    logger.info(f"\n{'=' * 60}")
    logger.info("SHAP ANALYSIS COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Output: {od}")
    if t_shap > 0:
        logger.info(f"  SHAP time: {t_shap:.1f}s")
    logger.info(f"  Plots: 01a-c, 02, 03, 04, 05, 08, 09, "
                f"{'10, ' if has_times else ''}"
                f"11, 12, 13")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()