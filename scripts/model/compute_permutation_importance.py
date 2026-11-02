"""
Compute permutation importance for the final RF model.

Shuffles each feature independently and measures the drop in performance.
Runs on a subsample of the test set for tractability.

Also loads MDI importance (from the model) and optionally SHAP importance
(if already computed) for a three-way comparison plot.

Finishes with partial-dependence plots (PDP + ICE) for the top features
so we can see *how* each important feature shifts the predicted probability,
not just that it matters.

All defaults from config.py, reads model directory from train_final_rf.py.

Usage:
  python compute_permutation_importance.py \
      --model_dir models/final_rf_<timestamp>
  python compute_permutation_importance.py \
      --model_dir models/final_rf_<timestamp> \
      --n_samples 100000 --n_repeats 10
  python compute_permutation_importance.py \
      --model_dir models/... --no_pdp    # skip PDPs to save a few minutes
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
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import f1_score
from scipy.stats import spearmanr

from sds.config import (
    PDP_TOP_N, PDP_ICE_SUBSAMPLE, RESOLUTION, SPATIAL_RES,
    VAL_YEARS, TEST_YEARS, PERM_N_SAMPLES, PERM_N_REPEATS,
    PERM_SEED,
)
from sds.paths import project_relative, model_figures_dir, resolve_path, SDS_PATH
from sds.log_utils import header, section, done

from sds.data_utils import (
    FeatureCache, load_sds_source, compute_land_indices,
    collect_indices, extract_from_cache, platt_apply,
)
from sds.feature_names import get_feature_name_map

# plot style
TITLE_SIZE = 14
SUBTITLE_SIZE = 12
LABEL_SIZE = 11
TICK_SIZE = 9
LEGEND_SIZE = 9
SUBPLOT_TITLE_SIZE = 11


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

_YEAR_RANGES: dict = {}  # filled in main


# ---------------------------------------------------------------------------
# Feature groups + Paired colours (same as compute_feature_correlations.py)
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
UNGROUPED_COLOR = "#888888"  # features not in any group


def _group_name_for(feat):
    for name, members in FEATURE_GROUPS:
        if feat in members:
            return name
    return None


def _group_color_for(feat):
    g = _group_name_for(feat)
    return GROUP_BASE_COLOR[g] if g is not None else UNGROUPED_COLOR


def _group_legend_handles(features_in_plot):
    # legend handles for the groups present, in FEATURE_GROUPS order
    groups_present = {g for g in (_group_name_for(f) for f in features_in_plot)
                      if g is not None}
    handles = [plt.Rectangle((0, 0), 1, 1,
                             facecolor=GROUP_BASE_COLOR[name],
                             edgecolor="none")
               for name, _ in FEATURE_GROUPS if name in groups_present]
    labels = [name for name, _ in FEATURE_GROUPS if name in groups_present]
    if any(_group_name_for(f) is None for f in features_in_plot):
        handles.append(plt.Rectangle((0, 0), 1, 1,
                                     facecolor=UNGROUPED_COLOR, edgecolor="none"))
        labels.append("Other")
    return handles, labels


def _tag_label(tag):
    # e.g. "Test (2024)"
    yr = _YEAR_RANGES.get(tag)
    if yr is None:
        return tag
    yr_str = str(yr[0]) if yr[0] == yr[1] else f"{yr[0]}–{yr[1]}"
    return f"{'Test' if tag == 'test' else 'Validation'} ({yr_str})"


def _clean_name(name):
    # strip units (anything in square brackets)
    return re.sub(r"\s*\[.*?\]", "", name)


def _nice(feat, name_map):
    # `or` because the name map can contain None values
    long_name = name_map.get(feat)
    return _clean_name(long_name) if long_name else feat


# ---------------------------------------------------------------------------
# Plot functions (each bar plot also has a group-coloured `_grouped` version)
# ---------------------------------------------------------------------------

def plot_perm_bar(perm_df, od, split, n_samples, n_repeats, scoring, name_map,
                  group_colored=False):
    # top-25 permutation importance with error bars
    top_n = min(25, len(perm_df))
    top = perm_df.head(top_n).iloc[::-1].copy()
    top["display_name"] = top["feature"].map(
        lambda f: _nice(f, name_map))

    if group_colored:
        bar_colors = [_group_color_for(f) for f in top["feature"].values]
    else:
        bar_colors = "#b2df8a"

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.3)))
    ax.barh(range(len(top)), top["perm_importance_mean"].values,
            xerr=top["perm_importance_std"].values,
            color=bar_colors, capsize=3, ecolor="#999")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["display_name"].values, fontsize=TICK_SIZE)
    ax.set_xlabel(f"Permutation importance (Δ{scoring})")
    title_suffix = " — grouped" if group_colored else ""
    ax.set_title(f"Permutation Importance{title_suffix} — {_tag_label(split)}\n"
                 f"(n={n_samples:,}, {n_repeats} repeats)",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if group_colored:
        handles, labels = _group_legend_handles(top["feature"].tolist())
        ax.legend(handles, labels,
                  loc="lower right", frameon=True, fontsize=8,
                  title="Feature group")

    suffix = "_grouped" if group_colored else ""
    fig.savefig(od / f"perm_importance_bar_{split}{suffix}.png")
    plt.close(fig)


def plot_importance_comparison(perm_df, has_shap, od, split, name_map,
                               group_colored=False):
    # side-by-side normalised bars per method; each panel is sorted by its own metric
    n_compare = min(20, len(perm_df))
    compare_feats = perm_df.head(n_compare)["feature"].tolist()

    if has_shap:
        fig, axes = plt.subplots(1, 3, figsize=(18, max(5, n_compare * 0.3)))
        comparisons = [
            ("Permutation", "perm_importance_mean", "#b2df8a"),
            ("MDI", "gini_importance", "#fdbf6f"),
            ("SHAP", "shap_importance", "#a6cee3"),
        ]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n_compare * 0.3)))
        comparisons = [
            ("Permutation", "perm_importance_mean", "#b2df8a"),
            ("MDI", "gini_importance", "#fdbf6f"),
        ]

    # legend covers every group that appears in any panel
    all_features_plotted = set()

    for ax, (label, col, default_color) in zip(axes, comparisons):
        sub = perm_df[perm_df["feature"].isin(compare_feats)].copy()
        sub = sub.sort_values(col, ascending=True)
        sub["display_name"] = sub["feature"].map(
            lambda f: _nice(f, name_map))
        vals = sub[col].values
        if np.nanmax(vals) > 0:
            vals = vals / np.nanmax(vals)
        if group_colored:
            bar_colors = [_group_color_for(f) for f in sub["feature"].values]
        else:
            bar_colors = default_color
        ax.barh(range(len(sub)), vals, color=bar_colors)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["display_name"].values, fontsize=7)
        ax.set_xlabel("Normalized importance")
        ax.set_title(label, fontsize=SUBTITLE_SIZE, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        all_features_plotted.update(sub["feature"].tolist())

    title_suffix = " (grouped)" if group_colored else ""
    fig.suptitle(f"Feature Importance Comparison{title_suffix} — {_tag_label(split)}",
                 fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout()

    if group_colored:
        # one-row legend below the panels
        fig.subplots_adjust(bottom=0.18)
        handles, labels = _group_legend_handles(sorted(all_features_plotted))
        fig.legend(handles, labels,
                   loc="lower center", ncol=len(handles),
                   frameon=True, fontsize=8,
                   title="Feature group",
                   bbox_to_anchor=(0.5, 0.02))

    suffix = "_grouped" if group_colored else ""
    fig.savefig(od / f"importance_comparison_{split}{suffix}.png")
    plt.close(fig)


def plot_rank_scatter(perm_df, has_shap, rho_gini, rho_shap, n_feats,
                      od, split, name_map, group_colored=False):
    # rank scatter: diagonal = agreement, the 5 largest disagreements are labelled
    fig, axes_scatter = plt.subplots(1, 2 if has_shap else 1,
                                     figsize=(12 if has_shap else 7, 6))
    if not has_shap:
        axes_scatter = [axes_scatter]

    if group_colored:
        point_colors_gini = [_group_color_for(f) for f in perm_df["feature"]]
        point_colors_shap = point_colors_gini
    else:
        point_colors_gini = "#fdbf6f"
        point_colors_shap = "#a6cee3"

    ax = axes_scatter[0]
    ax.scatter(perm_df["gini_rank"], perm_df["perm_rank"],
               s=30, alpha=0.9, color=point_colors_gini,
               edgecolor="white" if group_colored else "none",
               linewidth=0.4 if group_colored else 0)
    ax.plot([1, n_feats], [1, n_feats], "k--", alpha=0.3, lw=1)
    ax.set_xlabel("MDI rank")
    ax.set_ylabel("Permutation rank")
    ax.set_title(f"Permutation vs MDI (ρ={rho_gini:.3f})",
                 fontsize=SUBTITLE_SIZE, fontweight="bold")
    ax.set_xlim(0, n_feats + 1)
    ax.set_ylim(0, n_feats + 1)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(alpha=0.2)

    # label features that disagree most
    perm_df = perm_df.copy()
    perm_df["rank_diff_gini"] = abs(perm_df["perm_rank"] - perm_df["gini_rank"])
    disagree = perm_df.nlargest(5, "rank_diff_gini")
    for _, r in disagree.iterrows():
        display = _nice(r["feature"], name_map)
        ax.annotate(display,
                    (r["gini_rank"], r["perm_rank"]),
                    fontsize=6, alpha=0.8,
                    xytext=(5, 5), textcoords="offset points")

    if has_shap:
        ax2 = axes_scatter[1]
        ax2.scatter(perm_df["shap_rank"].fillna(n_feats),
                    perm_df["perm_rank"],
                    s=30, alpha=0.9, color=point_colors_shap,
                    edgecolor="white" if group_colored else "none",
                    linewidth=0.4 if group_colored else 0)
        ax2.plot([1, n_feats], [1, n_feats], "k--", alpha=0.3, lw=1)
        ax2.set_xlabel("SHAP rank")
        ax2.set_ylabel("Permutation rank")
        ax2.set_title(f"Permutation vs SHAP (ρ={rho_shap:.3f})",
                      fontsize=SUBTITLE_SIZE, fontweight="bold")
        ax2.set_xlim(0, n_feats + 1)
        ax2.set_ylim(0, n_feats + 1)
        ax2.invert_xaxis()
        ax2.invert_yaxis()
        ax2.grid(alpha=0.2)

        perm_df["rank_diff_shap"] = abs(
            perm_df["perm_rank"] - perm_df["shap_rank"].fillna(n_feats))
        disagree_shap = perm_df.nlargest(5, "rank_diff_shap")
        for _, r in disagree_shap.iterrows():
            display = _nice(r["feature"], name_map)
            ax2.annotate(display,
                         (r.get("shap_rank", n_feats), r["perm_rank"]),
                         fontsize=6, alpha=0.8,
                         xytext=(5, 5), textcoords="offset points")

    title_suffix = " (grouped)" if group_colored else ""
    fig.suptitle(f"Rank Agreement{title_suffix} — {_tag_label(split)}",
                 fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout()

    if group_colored:
        # one-row legend below both panels
        fig.subplots_adjust(bottom=0.22)
        handles, labels = _group_legend_handles(perm_df["feature"].tolist())
        fig.legend(handles, labels,
                   loc="lower center", ncol=len(handles),
                   frameon=True, fontsize=8,
                   title="Feature group",
                   bbox_to_anchor=(0.5, 0.02))

    suffix = "_grouped" if group_colored else ""
    fig.savefig(od / f"rank_scatter_{split}{suffix}.png")
    plt.close(fig)


def plot_pdp_top_features(rf, X, perm_df, feature_vars, od, split,
                          name_map, logger):
    # partial-dependence plots for the top features by permutation importance
    # ICE lines + PD average, 2x3 grid for the top 6. Uses the uncalibrated
    # probabilities: Platt scaling is monotonic, so only the y-axis would change.
    top_feats = perm_df.head(PDP_TOP_N)["feature"].tolist()
    top_idx = [feature_vars.index(f) for f in top_feats]

    nrow, ncol = 2, 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.8 * nrow))

    # one call per feature, so each gets its own axis
    for ax, feat, fi in zip(axes.flatten(), top_feats, top_idx):
        display_clean = _nice(feat, name_map)
        display_full = name_map.get(feat) or feat  # keeps the units
        try:
            PartialDependenceDisplay.from_estimator(
                rf, X, features=[fi],
                kind="both",
                subsample=PDP_ICE_SUBSAMPLE,
                random_state=PERM_SEED,
                ax=ax,
                ice_lines_kw={"color": "#4472C4", "alpha": 0.08, "lw": 0.5},
                pd_line_kw={"color": "#C00000", "lw": 2.2},
            )
            ax.set_title(display_clean, fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
            ax.set_xlabel(display_full, fontsize=TICK_SIZE)
            ax.set_ylabel("P(SDS=1)")
            ax.grid(alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        except Exception as e:
            logger.warning(f"  PDP failed for {feat}: {e}")
            ax.set_visible(False)

    fig.suptitle(f"Partial Dependence — top {PDP_TOP_N} features — "
                 f"{_tag_label(split)}",
                 fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(od / f"pdp_top_features_{split}.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compute permutation importance for the final RF")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--n_samples", type=int, default=None,
                        help=f"Subsample size (default: {PERM_N_SAMPLES:,})")
    parser.add_argument("--n_repeats", type=int, default=None,
                        help=f"Permutation repeats (default: {PERM_N_REPEATS})")
    parser.add_argument("--scoring", type=str, default="f1",
                        choices=["f1", "calibrated_f1", "roc_auc", "average_precision"],
                        help="Scoring metric (default: f1). "
                             "'calibrated_f1' routes each permutation through the full "
                             "deployed pipeline (Platt + tuned threshold) - more faithful "
                             "but requires platt_scaler.joblib + threshold from results.json.")
    parser.add_argument("--sds_path", type=str, default=None)
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--spatial", type=float, default=None)
    parser.add_argument("--val_years", nargs=2, type=int, default=None)
    parser.add_argument("--test_years", nargs=2, type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--fig_dir", type=str, default=None,
                        help="figure folder (default: figures/model/<model>/permutation_importance)")
    parser.add_argument("--no_pdp", action="store_true",
                        help="skip PDP plots (they add a few minutes)")
    args = parser.parse_args()

    model_dir = resolve_path(args.model_dir)
    if not model_dir.exists():
        sys.exit(f"ERROR: model_dir not found: {model_dir}")

    n_samples = args.n_samples or PERM_N_SAMPLES
    n_repeats = args.n_repeats or PERM_N_REPEATS

    # with scoring=f1/roc_auc/average_precision sklearn scores the raw RF (no Platt,
    # no tuned threshold); calibrated_f1 uses the full pipeline (see below)
    rf = joblib.load(model_dir / "rf_model.joblib")

    results_file = json.loads((model_dir / "results.json").read_text())
    feature_vars = results_file["feature_vars"]
    data_cfg = results_file["data_config"]
    thr = results_file.get("threshold")

    # Platt scaler (needed for calibrated_f1)
    platt_path = model_dir / "platt_scaler.joblib"
    platt_lr = joblib.load(platt_path) if platt_path.exists() else None
    if args.scoring == "calibrated_f1":
        if platt_lr is None or thr is None:
            sys.exit("ERROR: calibrated_f1 needs platt_scaler.joblib and a threshold in results.json")

    # CLI args > results.json > config defaults
    resolution = args.resolution or data_cfg.get("resolution", RESOLUTION)
    spatial_res = args.spatial or data_cfg.get("spatial_res", SPATIAL_RES)
    val_years = args.val_years or data_cfg.get("val_years", VAL_YEARS)
    test_years = args.test_years or data_cfg.get("test_years", TEST_YEARS)
    sds_path = resolve_path(args.sds_path or data_cfg.get("sds_path") or SDS_PATH)
    split_years = test_years if args.split == "test" else val_years

    _YEAR_RANGES["val"]  = tuple(val_years)
    _YEAR_RANGES["test"] = tuple(test_years)

    sds_file = Path(sds_path)
    if not sds_file.exists():
        sys.exit(f"ERROR: labels not found: {sds_file} (run the backtracking first)")

    od = Path(args.out_dir) if args.out_dir else model_dir / "permutation_importance"
    od.mkdir(parents=True, exist_ok=True)
    fd = Path(args.fig_dir) if args.fig_dir else model_figures_dir(model_dir, "permutation_importance")
    fd.mkdir(parents=True, exist_ok=True)

    name_map = get_feature_name_map(feature_vars)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(od / f"perm_imp_{args.split}.log"),
        ],
    )
    logger = logging.getLogger("perm_imp")

    header("Permutation importance", {
        "Model": project_relative(model_dir),
        "Split": f"{args.split} {split_years}",
        "Samples / repeats": f"{n_samples:,} / {n_repeats}",
        "Scoring": args.scoring,
        "Output": f"{project_relative(od)}, {project_relative(fd)}",
    }, out=logger.info)

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
    section("Samples", out=logger.info)
    all_pos, all_neg = collect_indices(
        cache, years, ds_sds, sds_var, land_indices, logger, label=args.split.capitalize())

    ds_sds.close()
    del ds_sds
    gc.collect()

    # stratified subsample (keeps the pos/neg ratio of the split)
    rng = np.random.default_rng(PERM_SEED)
    n_total = len(all_pos) + len(all_neg)
    pos_frac = len(all_pos) / n_total
    n_pos_sample = max(1, int(round(n_samples * pos_frac)))
    n_neg_sample = n_samples - n_pos_sample
    n_pos_sample = min(n_pos_sample, len(all_pos))
    n_neg_sample = min(n_neg_sample, len(all_neg))

    logger.info(f"  Subsample: {n_pos_sample:,} pos, {n_neg_sample:,} neg")

    pos_sel = rng.choice(len(all_pos), size=n_pos_sample, replace=False)
    neg_sel = rng.choice(len(all_neg), size=n_neg_sample, replace=False)
    sample_indices = ([all_pos[i] for i in pos_sel] +
                      [all_neg[i] for i in neg_sel])

    X = extract_from_cache(cache, sample_indices)
    y = np.concatenate([
        np.ones(n_pos_sample, dtype=np.int32),
        np.zeros(n_neg_sample, dtype=np.int32),
    ])

    del all_pos, all_neg, sample_indices
    gc.collect()

    # permutation importance (slow: n_features x n_repeats refits of the scorer)
    section("Permutation importance", out=logger.info)
    t0 = time.time()

    if args.scoring == "calibrated_f1":
        # RF -> Platt -> fixed threshold -> F1: how much F1 is lost if a feature is
        # corrupted at prediction time, at the threshold already chosen
        def calibrated_f1_scorer(estimator, X_s, y_s):
            proba_raw = estimator.predict_proba(X_s)[:, 1]
            proba_cal = platt_apply(platt_lr, proba_raw)
            y_pred = (proba_cal >= thr).astype(np.int32)
            return f1_score(y_s, y_pred, zero_division=0)
        scoring_arg = calibrated_f1_scorer
    else:
        scoring_arg = args.scoring

    perm_result = permutation_importance(
        rf, X, y,
        scoring=scoring_arg,
        n_repeats=n_repeats,
        random_state=PERM_SEED,
        n_jobs=1,
    )

    t_perm = time.time() - t0
    logger.info(f"  Computed in {t_perm / 60:.1f} min")

    # results + MDI ranks
    perm_df = pd.DataFrame({
        "feature": feature_vars,
        "perm_importance_mean": perm_result.importances_mean,
        "perm_importance_std": perm_result.importances_std,
        "gini_importance": rf.feature_importances_,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)
    perm_df["perm_rank"] = range(1, len(perm_df) + 1)

    gini_sorted = perm_df.sort_values("gini_importance", ascending=False)
    gini_rank_map = {f: r for r, f in enumerate(gini_sorted["feature"], 1)}
    perm_df["gini_rank"] = perm_df["feature"].map(gini_rank_map)

    # SHAP importance, if compute_shap.py has already run
    shap_file = model_dir / "shap" / "shap_feature_importance.csv"
    if shap_file.exists():
        shap_df = pd.read_csv(shap_file)
        shap_map = dict(zip(shap_df["feature"], shap_df["mean_abs_shap"]))
        shap_rank_map = dict(zip(shap_df["feature"], shap_df["rank"]))
        perm_df["shap_importance"] = perm_df["feature"].map(shap_map)
        perm_df["shap_rank"] = perm_df["feature"].map(shap_rank_map)
        has_shap = True
    else:
        has_shap = False
        logger.info("  SHAP importance not found, comparison without SHAP (run compute_shap.py first)")

    perm_df.to_csv(od / f"permutation_importance_{args.split}.csv", index=False)

    logger.info("  Top 5: " + ", ".join(_nice(f, name_map) for f in perm_df["feature"].head(5)))

    # rank correlation across methods
    rho_gini, p_gini = spearmanr(perm_df["perm_rank"], perm_df["gini_rank"])
    logger.info(f"  Rank correlation with MDI: ρ={rho_gini:.3f}")
    rho_shap, p_shap = None, None
    if has_shap:
        rho_shap, p_shap = spearmanr(
            perm_df["perm_rank"].values,
            perm_df["shap_rank"].fillna(len(feature_vars)).values)
        logger.info(f"  Rank correlation with SHAP: ρ={rho_shap:.3f}")

    # plots
    section("Figures", out=logger.info)
    n_feats = len(feature_vars)
    for gc_flag in (False, True):
        plot_perm_bar(perm_df, fd, args.split, n_samples, n_repeats,
                      args.scoring, name_map, group_colored=gc_flag)
        plot_importance_comparison(perm_df, has_shap, fd, args.split, name_map,
                                   group_colored=gc_flag)
        plot_rank_scatter(perm_df, has_shap, rho_gini,
                          rho_shap if rho_shap is not None else 0.0,
                          n_feats, fd, args.split, name_map,
                          group_colored=gc_flag)
    if not args.no_pdp:
        plot_pdp_top_features(rf, X, perm_df, feature_vars, fd, args.split,
                              name_map, logger)

    summary = {
        "split": args.split,
        "scoring": args.scoring,
        "n_samples": n_samples,
        "n_repeats": n_repeats,
        "time_s": round(t_perm, 1),
        "rank_correlation_perm_vs_gini": {
            "spearman_rho": float(rho_gini),
            "p_value": float(p_gini),
        },
        "top_10": perm_df.head(10)[
            ["feature", "perm_importance_mean", "perm_importance_std",
             "perm_rank", "gini_rank"]
        ].to_dict("records"),
    }
    if has_shap:
        summary["rank_correlation_perm_vs_shap"] = {
            "spearman_rho": float(rho_shap),
            "p_value": float(p_shap),
        }

    with open(od / f"perm_importance_summary_{args.split}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    done(od, out=logger.info)


if __name__ == "__main__":
    main()