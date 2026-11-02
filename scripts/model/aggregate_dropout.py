"""
Collects the per-feature dropout runs (from compute_dropout_importance.py) and
produces the combined two-panel figure: (a) permutation importance, (b) dropout importance.

Also writes a combined CSV for the paper.

Usage:
  python aggregate_dropout.py \
      --main_model_dir models/final_rf_<timestamp> \
      --perm_csv models/final_rf_<timestamp>/permutation_importance/permutation_importance_test.csv

The dropout runs are read from main_model_dir/dropout/{feature}/results.json by default.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sds.paths import project_relative, model_figures_dir, resolve_path
from sds.log_utils import header, done
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


def _nice(feat, name_map):
    long = name_map.get(feat)
    return re.sub(r"\s*\[.*?\]", "", long) if long else feat


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate dropout runs into a combined permutation+dropout plot")
    parser.add_argument("--main_model_dir", type=str, required=True)
    parser.add_argument("--perm_csv", type=str, default=None,
        help="override the permutation CSV path. Default: look for the matched-subsample "
             "CSV in {dropout_dir}/__baseline__/permutation_importance_matched.csv first, "
             "fall back to {main_model_dir}/permutation_importance/permutation_importance_test.csv")
    parser.add_argument("--shap_csv", type=str, default=None,
        help="override the SHAP CSV path. Default: {main_model_dir}/shap/shap_feature_importance.csv. "
             "If not found, the SHAP panel is omitted.")
    parser.add_argument("--dropout_dir", type=str, default=None,
        help="directory containing per-feature subdirs (default: main_model_dir/dropout)")
    parser.add_argument("--out_dir", type=str, default=None,
        help="where to write outputs (default: main_model_dir/dropout)")
    parser.add_argument("--fig_dir", type=str, default=None,
                        help="figure folder (default: figures/model/<model>/dropout)")
    parser.add_argument("--top_n", type=int, default=15,
        help="how many features to plot (default: 15)")
    args = parser.parse_args()

    main_model_dir = resolve_path(args.main_model_dir)
    dropout_dir = resolve_path(args.dropout_dir) if args.dropout_dir else \
        main_model_dir / "dropout"
    out_dir = Path(args.out_dir) if args.out_dir else dropout_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fd = Path(args.fig_dir) if args.fig_dir else model_figures_dir(main_model_dir, "dropout")
    fd.mkdir(parents=True, exist_ok=True)

    # permutation CSV: the one from the dropout baseline (same RF as the baseline F1) if present
    if args.perm_csv:
        perm_csv = resolve_path(args.perm_csv)
        perm_source = "given"
    else:
        matched_path = dropout_dir / "__baseline__" / "permutation_importance_matched.csv"
        fullmodel_path = main_model_dir / "permutation_importance" / "permutation_importance_test.csv"
        if matched_path.exists():
            perm_csv = matched_path
            perm_source = "dropout baseline (same training subsample)"
        elif fullmodel_path.exists():
            perm_csv = fullmodel_path
            perm_source = "full model (not the dropout training subsample)"
        else:
            sys.exit(f"ERROR: no permutation CSV found at {matched_path} or {fullmodel_path}")

    if not perm_csv.exists():
        sys.exit(f"ERROR: permutation CSV not found: {perm_csv}")
    if not dropout_dir.exists():
        sys.exit(f"ERROR: dropout directory not found: {dropout_dir} "
                 "(run compute_dropout_importance.py first)")

    header("Dropout importance: aggregate", {
        "Main model": project_relative(main_model_dir),
        "Permutation": perm_source,
        "Output": f"{project_relative(out_dir)}, {project_relative(fd)}",
    })

    # SHAP is optional; without it the third panel is skipped
    shap_csv = resolve_path(args.shap_csv) if args.shap_csv else \
        main_model_dir / "shap" / "shap_feature_importance.csv"
    shap_df = None
    if shap_csv.exists():
        shap_df = pd.read_csv(shap_csv)
    else:
        print("  SHAP importance not found, figure without the SHAP panel (run compute_shap.py first)")

    perm_df = pd.read_csv(perm_csv)

    # baseline + per-feature runs from dropout/
    baseline_entry = None
    rows = []
    for sub in sorted(dropout_dir.iterdir()):
        if not sub.is_dir():
            continue
        rj = sub / "results.json"
        if not rj.exists():
            print(f"  WARNING: {sub.name} has no results.json (skipped)")
            continue
        r = json.loads(rj.read_text())
        if r.get("is_baseline"):
            baseline_entry = r
            continue
        rows.append({
            "feature":           r["drop_feature"],
            "dropout_f1":        r["test_metrics"]["f1"],
            "dropout_prec":      r["test_metrics"]["precision"],
            "dropout_rec":       r["test_metrics"]["recall"],
            "full_model_f1":     r["full_model_test_f1"],
            "train_frac":        r["train_frac"],
            "train_time_s":      r["train_time_s"],
        })

    if not rows:
        sys.exit(f"ERROR: no dropout runs found in {dropout_dir}")
    if baseline_entry is None:
        sys.exit(f"ERROR: no baseline run in {dropout_dir} (run compute_dropout_importance.py "
                 "without --drop_feature)")

    baseline_f1 = baseline_entry["test_metrics"]["f1"]
    baseline_frac = baseline_entry["train_frac"]
    full_model_f1 = baseline_entry["full_model_test_f1"]
    print(f"  Runs: {len(rows)} features + baseline (F1 {baseline_f1:.4f} with "
          f"{baseline_frac:.0%} of the training data; full model {full_model_f1:.4f})")

    drop_df = pd.DataFrame(rows)
    # delta_f1 = baseline - dropout (positive: worse without the feature)
    drop_df["dropout_delta"] = baseline_f1 - drop_df["dropout_f1"]
    # delta from the full model, for reference
    drop_df["dropout_delta_from_full"] = full_model_f1 - drop_df["dropout_f1"]

    # baseline and dropout runs must use the same train_frac
    mismatched = drop_df[drop_df["train_frac"] != baseline_frac]
    if not mismatched.empty:
        print(f"  WARNING: {len(mismatched)} runs use a different train_frac than the baseline "
              f"({baseline_frac:.0%}): {', '.join(mismatched['feature'])}")

    merged = perm_df.merge(drop_df, on="feature", how="outer")
    # left join: features without a SHAP value are kept (NaN)
    if shap_df is not None:
        merged = merged.merge(
            shap_df[["feature", "mean_abs_shap"]],
            on="feature", how="left",
        )
    merged.to_csv(out_dir / "combined_importance.csv", index=False)
    print("  Saved: combined_importance.csv")

    # plot
    # same row order in all panels: by permutation importance
    both = merged.dropna(subset=["perm_importance_mean", "dropout_delta"]).copy()
    both = both.sort_values("perm_importance_mean", ascending=True)  # ascending for barh
    top = both.tail(args.top_n)

    name_map = get_feature_name_map(top["feature"].tolist())
    top["display"] = top["feature"].map(lambda f: _nice(f, name_map))

    n_panels = 3 if shap_df is not None else 2
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(7 * n_panels, max(6, args.top_n * 0.35)),
                             sharey=True)
    if n_panels == 2:
        a1, a2 = axes
    else:
        a1, a2, a3 = axes

    # light Paired colours, as in compute_permutation_importance.py
    COLOR_PERM = "#b2df8a"
    COLOR_DROP = "#fb9a99"
    COLOR_SHAP = "#a6cee3"

    # (a) permutation importance (can be negative; error bars from repeats)
    perm_vals = top["perm_importance_mean"].values
    perm_errs = top["perm_importance_std"].values
    a1.barh(range(len(top)), perm_vals,
            xerr=perm_errs, color=COLOR_PERM, capsize=2, ecolor="#999")
    a1.axvline(0, color="black", lw=0.8)
    a1.set_xlabel("Mean F1 decrease (permutation)")
    a1.set_title("(a) Permutation importance", fontsize=SUBTITLE_SIZE)
    a1.set_yticks(range(len(top)))
    a1.set_yticklabels(top["display"].values, fontsize=TICK_SIZE)
    a1.spines["top"].set_visible(False)
    a1.spines["right"].set_visible(False)

    # (b) dropout importance (F1 drop when retrained without the feature)
    drop_vals = top["dropout_delta"].values
    a2.barh(range(len(top)), drop_vals, color=COLOR_DROP)
    a2.axvline(0, color="black", lw=0.8)
    a2.set_xlabel("F1 decrease (feature dropout)")
    a2.set_title("(b) Dropout importance", fontsize=SUBTITLE_SIZE)
    a2.spines["top"].set_visible(False)
    a2.spines["right"].set_visible(False)

    # (c) mean |SHAP| if available (zero for features without a SHAP value)
    if shap_df is not None:
        shap_vals = top["mean_abs_shap"].fillna(0).values
        a3.barh(range(len(top)), shap_vals, color=COLOR_SHAP)
        a3.axvline(0, color="black", lw=0.8)
        a3.set_xlabel("Mean |SHAP value|")
        a3.set_title("(c) SHAP importance", fontsize=SUBTITLE_SIZE)
        a3.spines["top"].set_visible(False)
        a3.spines["right"].set_visible(False)

    # caption note: training fraction + SHAP caveat
    frac_note = (f"perm + dropout: {baseline_frac:.0%}-subsample RFs "
                 f"(full-model F1 = {full_model_f1:.3f})")
    if shap_df is not None:
        frac_note += "; SHAP: full-data model (not matched to subsample baseline)"
    fig.suptitle(
        f"Feature importance — permutation vs dropout"
        f"{' vs SHAP' if shap_df is not None else ''}\n{frac_note}",
        fontsize=TITLE_SIZE)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_filename = ("perm_vs_dropout_vs_shap.png"
                    if shap_df is not None else "perm_vs_dropout.png")
    out_png = fd / out_filename
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved: {out_png.name}")

    # second figure: dropout vs SHAP only
    if shap_df is not None:
        fig2, (b1, b2) = plt.subplots(1, 2, figsize=(14, max(6, args.top_n * 0.35)),
                                       sharey=True)
        b1.barh(range(len(top)), top["dropout_delta"].values, color=COLOR_DROP)
        b1.axvline(0, color="black", lw=0.8)
        b1.set_xlabel("F1 decrease (feature dropout)")
        b1.set_title("(a) Dropout importance", fontsize=SUBTITLE_SIZE)
        b1.set_yticks(range(len(top)))
        b1.set_yticklabels(top["display"].values, fontsize=TICK_SIZE)
        b1.spines["top"].set_visible(False)
        b1.spines["right"].set_visible(False)

        b2.barh(range(len(top)), top["mean_abs_shap"].fillna(0).values,
                color=COLOR_SHAP)
        b2.axvline(0, color="black", lw=0.8)
        b2.set_xlabel("Mean |SHAP value|")
        b2.set_title("(b) SHAP importance", fontsize=SUBTITLE_SIZE)
        b2.spines["top"].set_visible(False)
        b2.spines["right"].set_visible(False)

        fig2.suptitle(
            f"Feature importance — dropout vs SHAP\n"
            f"dropout: {baseline_frac:.0%}-subsample RF "
            f"(full-model F1 = {full_model_f1:.3f}); "
            f"SHAP: full-data model",
            fontsize=TITLE_SIZE)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out2 = fd / "dropout_vs_shap.png"
        fig2.savefig(out2)
        plt.close(fig2)
        print(f"  Saved: {out2.name}")

    done(fd)


if __name__ == "__main__":
    main()