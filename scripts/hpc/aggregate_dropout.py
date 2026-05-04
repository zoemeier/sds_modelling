"""
Collects the per-feature dropout runs (from compute_dropout_importance.py) and
produces the combined two-panel figure: (a) permutation importance, (b) dropout importance.

Also writes a combined CSV for the paper.

Usage:
  python aggregate_dropout.py \
      --main_model_dir models/final_rf_v1_20260413_120000 \
      --perm_csv models/final_rf_v1_20260413_120000/permutation_importance/permutation_importance_test.csv

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


def _nice(feat, name_map):
    # same None-safe helper as the other scripts
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
    parser.add_argument("--top_n", type=int, default=15,
        help="how many features to plot (default: 15)")
    args = parser.parse_args()

    main_model_dir = Path(args.main_model_dir)
    dropout_dir = Path(args.dropout_dir) if args.dropout_dir else \
        main_model_dir / "dropout"
    out_dir = Path(args.out_dir) if args.out_dir else dropout_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # decide which permutation CSV to use - prefer the matched-subsample one if present
    # since it was computed on the same RF that our baseline F1 comes from
    if args.perm_csv:
        perm_csv = Path(args.perm_csv)
        perm_source = "user-specified"
    else:
        matched_path = dropout_dir / "__baseline__" / "permutation_importance_matched.csv"
        fullmodel_path = main_model_dir / "permutation_importance" / "permutation_importance_test.csv"
        if matched_path.exists():
            perm_csv = matched_path
            perm_source = "matched-subsample"
        elif fullmodel_path.exists():
            perm_csv = fullmodel_path
            perm_source = "full-model (WARNING: not matched to dropout subsample)"
        else:
            print(f"ERROR: no permutation CSV found at either:")
            print(f"  {matched_path}")
            print(f"  {fullmodel_path}")
            sys.exit(1)

    if not perm_csv.exists():
        print(f"ERROR: permutation CSV not found: {perm_csv}")
        sys.exit(1)
    if not dropout_dir.exists():
        print(f"ERROR: dropout directory not found: {dropout_dir}")
        sys.exit(1)

    # SHAP is optional - if the CSV is missing we just skip the third panel
    shap_csv = Path(args.shap_csv) if args.shap_csv else \
        main_model_dir / "shap" / "shap_feature_importance.csv"
    shap_df = None
    if shap_csv.exists():
        shap_df = pd.read_csv(shap_csv)
        print(f"SHAP CSV: {shap_csv}")
        print(f"  note: SHAP was computed on the full-data model, not the subsample baseline; "
              f"see figure caption for caveat")
    else:
        print(f"SHAP CSV not found at {shap_csv}; figure will have only perm + dropout panels")

    print(f"Permutation CSV: {perm_csv}")
    print(f"  source: {perm_source}")

    perm_df = pd.read_csv(perm_csv)

    # walk every subdir of dropout/ and pick out the baseline run + per-feature runs
    baseline_entry = None
    rows = []
    for sub in sorted(dropout_dir.iterdir()):
        if not sub.is_dir():
            continue
        rj = sub / "results.json"
        if not rj.exists():
            print(f"  skipping {sub.name}: no results.json")
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
        print(f"ERROR: no dropout runs found in {dropout_dir}")
        sys.exit(1)
    if baseline_entry is None:
        print(f"ERROR: no __baseline__ run found in {dropout_dir}.")
        print(f"  Submit one with:")
        print(f"    python compute_dropout_importance.py --main_model_dir {main_model_dir.name} \\")
        print(f"        --drop_feature __baseline__ --train_frac <same as dropouts>")
        sys.exit(1)

    # the matched-subsample baseline is what we compare dropouts against
    baseline_f1 = baseline_entry["test_metrics"]["f1"]
    baseline_frac = baseline_entry["train_frac"]
    full_model_f1 = baseline_entry["full_model_test_f1"]
    print(f"Baseline F1 (train_frac={baseline_frac:.0%}):  {baseline_f1:.4f}")
    print(f"Full-model F1 (reference):                    {full_model_f1:.4f}")
    print(f"  gap from subsampling alone: {full_model_f1 - baseline_f1:+.4f}")

    drop_df = pd.DataFrame(rows)
    # delta_f1 = baseline - dropout, so positive means "model got worse without the feature"
    drop_df["dropout_delta"] = baseline_f1 - drop_df["dropout_f1"]
    # also the naive delta from the full model, for diagnostic purposes
    drop_df["dropout_delta_from_full"] = full_model_f1 - drop_df["dropout_f1"]

    # flag any train_frac mismatch between the baseline and the dropout runs - comparing
    # a 10%-baseline against 20%-dropouts is not valid
    mismatched = drop_df[drop_df["train_frac"] != baseline_frac]
    if not mismatched.empty:
        print(f"\nWARNING: {len(mismatched)} dropout runs use a different train_frac "
              f"than the baseline ({baseline_frac:.0%})")
        for _, r in mismatched.iterrows():
            print(f"  {r['feature']}: train_frac={r['train_frac']:.0%}")
        print("  Δ F1 values for those features may be biased by the data-size difference.")

    print(f"\nRead {len(drop_df)} dropout runs")

    # merge with permutation results on feature name
    merged = perm_df.merge(drop_df, on="feature", how="outer")
    # also bring in the SHAP rankings if we have them - left join so features without SHAP
    # data still show up (NaN means SHAP didn't rank that feature, e.g. because it was
    # below the SHAP top-N cutoff at compute time)
    if shap_df is not None:
        merged = merged.merge(
            shap_df[["feature", "mean_abs_shap"]],
            on="feature", how="left",
        )
    merged.to_csv(out_dir / "combined_importance.csv", index=False)
    print(f"Wrote combined CSV: {out_dir / 'combined_importance.csv'}")

    # ── Plot ──────────────────────────────────────────────────────────────
    # keep the order consistent between panels - sort by permutation importance desc,
    # so the top-N same features show in the same row order in all panels
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

    # colours match compute_permutation_importance.py's comparison panel
    # (Paired palette light variants - same intensity across methods)
    COLOR_PERM = "#b2df8a"   # light green
    COLOR_DROP = "#fb9a99"   # light red/pink - same intensity as the others
    COLOR_SHAP = "#a6cee3"   # light blue

    # panel (a): permutation importance (can be negative; error bars from repeats)
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

    # panel (b): dropout importance (F1 drop when retrained without the feature)
    drop_vals = top["dropout_delta"].values
    a2.barh(range(len(top)), drop_vals, color=COLOR_DROP)
    a2.axvline(0, color="black", lw=0.8)
    a2.set_xlabel("F1 decrease (feature dropout)")
    a2.set_title("(b) Dropout importance", fontsize=SUBTITLE_SIZE)
    a2.spines["top"].set_visible(False)
    a2.spines["right"].set_visible(False)

    # panel (c): SHAP mean |value| if available - features without SHAP data show as zero
    if shap_df is not None:
        shap_vals = top["mean_abs_shap"].fillna(0).values
        a3.barh(range(len(top)), shap_vals, color=COLOR_SHAP)
        a3.axvline(0, color="black", lw=0.8)
        a3.set_xlabel("Mean |SHAP value|")
        a3.set_title("(c) SHAP importance", fontsize=SUBTITLE_SIZE)
        a3.spines["top"].set_visible(False)
        a3.spines["right"].set_visible(False)

    # note about the training fraction (honest caveat for the paper) plus the SHAP caveat
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
    out_png = out_dir / out_filename
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Wrote plot: {out_png}")

    # ── Second figure: dropout vs SHAP only ───────────────────────────────
    # this is the cleanest pair because both are answering "if this feature were absent,
    # how would the model's behaviour change?" - dropout via retraining, SHAP via attribution.
    # permutation is a different question (deployment-time corruption) and is left out here.
    if shap_df is not None:
        fig2, (b1, b2) = plt.subplots(1, 2, figsize=(14, max(6, args.top_n * 0.35)),
                                       sharey=True)
        # same row order as the three-panel figure: by perm importance descending
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
        out2 = out_dir / "dropout_vs_shap.png"
        fig2.savefig(out2)
        plt.close(fig2)
        print(f"Wrote plot: {out2}")

    # brief text summary to stdout
    print("\nTop 10 by dropout importance:")
    summary = drop_df.sort_values("dropout_delta", ascending=False).head(10)
    for _, r in summary.iterrows():
        disp = _nice(r["feature"],
                     get_feature_name_map([r["feature"]]))
        print(f"  {disp:<40s}  Δ F1 = {r['dropout_delta']:+.4f}  "
              f"(dropout F1 = {r['dropout_f1']:.4f})")


if __name__ == "__main__":
    main()