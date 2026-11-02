"""
Event-level overlap analysis for SDS predictions.

Two independent directions (no event matching):
  A. Recall direction: For each ground-truth event (connected component),
     what fraction of its pixels were predicted positive?

  B. Precision direction: For each predicted event (connected component),
     what fraction of its pixels are actual positives?

Reports both event-level rates and pixel-weighted rates at multiple
overlap thresholds.

Usage:
  python event_iou_analysis.py --model_dir models/final_rf_<timestamp>
  python event_iou_analysis.py --model_dir models/final_rf_<timestamp> --split val
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

from sds.config import (
    EVENT_CONNECTIVITY, EVENT_OVERLAP_THRESHOLDS, EVENT_SIZE_BINS, EVENT_SIZE_OVERLAP_THRESHOLDS,
    RESOLUTION, SPATIAL_RES, VAL_YEARS, TEST_YEARS,
)
from sds.paths import project_relative, model_figures_dir, resolve_path, SDS_PATH
from sds.log_utils import header, section, done
from sds.data_utils import (
    FeatureCache, load_sds_source, compute_land_indices,
    platt_apply, make_target_for_year,
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Event analysis
# ---------------------------------------------------------------------------
def analyze_events_for_day(truth_grid, pred_grid, struct):
    # labels connected components in both grids, then measures per-event overlap
    # in each direction (recall = truth events, precision = pred events)
    recall_events = []
    precision_events = []

    # ── Recall direction: ground-truth events ─────────────────────────
    if truth_grid.any():
        labeled_truth, n_truth = ndimage.label(truth_grid, structure=struct)
        for ev in range(1, n_truth + 1):
            ev_mask = labeled_truth == ev
            ev_size = int(ev_mask.sum())
            overlap = int((ev_mask & pred_grid).sum())
            recall_events.append({
                "size": ev_size,
                "overlap": overlap,
                "overlap_frac": overlap / ev_size,
            })

    # ── Precision direction: predicted events ─────────────────────────
    if pred_grid.any():
        labeled_pred, n_pred = ndimage.label(pred_grid, structure=struct)
        for ev in range(1, n_pred + 1):
            ev_mask = labeled_pred == ev
            ev_size = int(ev_mask.sum())
            overlap = int((ev_mask & truth_grid).sum())
            precision_events.append({
                "size": ev_size,
                "overlap": overlap,
                "overlap_frac": overlap / ev_size,
            })

    return recall_events, precision_events


def compute_rates(events_df, thresholds):
    # event-level rate = fraction of events with overlap >= thr
    # pixel-weighted rate = same but weighted by event size (big events count more)
    results = []
    for thr in thresholds:
        found = events_df["overlap_frac"] >= thr
        n_found = int(found.sum())
        n_total = len(events_df)
        event_rate = n_found / max(n_total, 1)

        total_pixels = events_df["size"].sum()
        found_pixels = events_df.loc[found, "size"].sum()
        pixel_rate = found_pixels / max(total_pixels, 1)

        results.append({
            "threshold": thr,
            "n_found": n_found,
            "n_total": n_total,
            "event_rate": float(event_rate),
            "pixel_weighted_rate": float(pixel_rate),
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_rates(recall_rates, precision_rates, od, tag):
    # two-panel: recall on the left, precision on the right, each showing both weighting schemes
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))

    a1.plot(recall_rates["threshold"], recall_rates["event_rate"],
            "o-", color="#70AD47", lw=2, ms=6, label="Event-level")
    a1.plot(recall_rates["threshold"], recall_rates["pixel_weighted_rate"],
            "s--", color="#70AD47", lw=1.5, ms=5, alpha=0.7,
            label="Pixel-weighted")
    a1.set_xlabel("Overlap Threshold")
    a1.set_ylabel("Recall Rate")
    a1.set_title("Recall: fraction of ground-truth events found", fontsize=12)
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1.05)
    a1.legend()
    a1.grid(alpha=0.3)

    a2.plot(precision_rates["threshold"], precision_rates["event_rate"],
            "o-", color="#4472C4", lw=2, ms=6, label="Event-level")
    a2.plot(precision_rates["threshold"], precision_rates["pixel_weighted_rate"],
            "s--", color="#4472C4", lw=1.5, ms=5, alpha=0.7,
            label="Pixel-weighted")
    a2.set_xlabel("Overlap Threshold")
    a2.set_ylabel("Precision Rate")
    a2.set_title("Precision: fraction of predicted events that are real", fontsize=12)
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 1.05)
    a2.legend()
    a2.grid(alpha=0.3)

    fig.suptitle(f"Event-Level Overlap Analysis — {tag}", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(od / f"event_iou_rates_{tag}.png")
    plt.close(fig)


def plot_rates_pixel_only(recall_rates, precision_rates, od, tag):
    # same layout as plot_rates but pixel-weighted only, so no legend needed
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))

    a1.plot(recall_rates["threshold"], recall_rates["pixel_weighted_rate"],
            "s-", color="#70AD47", lw=2, ms=6)
    a1.set_xlabel("Overlap Threshold")
    a1.set_ylabel("Recall Rate (pixel-weighted)")
    a1.set_title("Recall: fraction of ground-truth events found", fontsize=12)
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1.05)
    a1.grid(alpha=0.3)

    a2.plot(precision_rates["threshold"], precision_rates["pixel_weighted_rate"],
            "s-", color="#4472C4", lw=2, ms=6)
    a2.set_xlabel("Overlap Threshold")
    a2.set_ylabel("Precision Rate (pixel-weighted)")
    a2.set_title("Precision: fraction of predicted events that are real", fontsize=12)
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 1.05)
    a2.grid(alpha=0.3)

    fig.suptitle(f"Event-Level Overlap Analysis (pixel-weighted) — {tag}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(od / f"event_iou_rates_pixel_{tag}.png")
    plt.close(fig)


def plot_overlap_histograms(recall_df, precision_df, od, tag):
    # per-event overlap-fraction distribution, with median line
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))

    bins = np.linspace(0, 1, 51)

    a1.hist(recall_df["overlap_frac"], bins=bins, color="#70AD47", alpha=0.7,
            edgecolor="white")
    a1.set_xlabel("Overlap fraction (predicted ∩ event / event)")
    a1.set_ylabel("Number of ground-truth events")
    a1.set_title(f"Recall: per-event overlap\n"
                 f"(n={len(recall_df):,} events, "
                 f"median={recall_df['overlap_frac'].median():.2f})")
    a1.axvline(recall_df["overlap_frac"].median(), color="#222", ls="--",
               lw=1.5, label=f"Median = {recall_df['overlap_frac'].median():.2f}")
    a1.legend()
    a1.grid(alpha=0.2)

    a2.hist(precision_df["overlap_frac"], bins=bins, color="#4472C4", alpha=0.7,
            edgecolor="white")
    a2.set_xlabel("Overlap fraction (truth ∩ pred_event / pred_event)")
    a2.set_ylabel("Number of predicted events")
    a2.set_title(f"Precision: per-event overlap\n"
                 f"(n={len(precision_df):,} events, "
                 f"median={precision_df['overlap_frac'].median():.2f})")
    a2.axvline(precision_df["overlap_frac"].median(), color="#222", ls="--",
               lw=1.5, label=f"Median = {precision_df['overlap_frac'].median():.2f}")
    a2.legend()
    a2.grid(alpha=0.2)

    fig.suptitle(f"Per-Event Overlap Distribution — {tag}", fontsize=14)
    fig.savefig(od / f"event_iou_histograms_{tag}.png")
    plt.close(fig)


def plot_by_event_size(recall_df, precision_df, od, tag):
    # breaks recall/precision rates down by event size so we can see whether small events are the ones being missed
    size_bins = EVENT_SIZE_BINS
    overlap_thresholds = EVENT_SIZE_OVERLAP_THRESHOLDS

    for overlap_thr in overlap_thresholds:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))

        for ax, df, color, direction in [
            (a1, recall_df, "#70AD47", "Recall"),
            (a2, precision_df, "#4472C4", "Precision"),
        ]:
            labels, rates, counts = [], [], []
            for i in range(len(size_bins) - 1):
                lo, hi = size_bins[i], size_bins[i + 1]
                mask = (df["size"] >= lo) & (df["size"] < hi)
                if mask.sum() > 0:
                    label = f"{lo}+" if hi == np.inf else f"{lo}–{hi-1}"
                    labels.append(label)
                    rates.append(
                        (df.loc[mask, "overlap_frac"] >= overlap_thr).mean())
                    counts.append(int(mask.sum()))

            x = range(len(labels))
            bars = ax.bar(x, rates, color=color, alpha=0.8, edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30)
            ax.set_xlabel("Event Size (grid cells)")
            ax.set_ylabel(f"{direction} rate")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{direction} by Event Size", fontsize=12)
            ax.grid(alpha=0.2, axis="y")

            for bar, cnt in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"n={cnt}", ha="center", va="bottom", fontsize=7)

        fig.suptitle(
            f"Event Rates by Size (overlap ≥ {overlap_thr:.0%}) — {tag}",
            fontsize=14, fontweight="bold")
        fig.tight_layout()
        thr_str = f"{int(overlap_thr * 100):02d}"
        fig.savefig(od / f"event_iou_by_size_thr{thr_str}_{tag}.png")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Event-level overlap analysis (no matching)")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--sds_path", type=str, default=None)
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--spatial", type=float, default=None)
    parser.add_argument("--val_years", nargs=2, type=int, default=None)
    parser.add_argument("--test_years", nargs=2, type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--fig_dir", type=str, default=None,
                        help="figure folder (default: figures/model/<model>/event_analysis)")
    args = parser.parse_args()

    model_dir = resolve_path(args.model_dir)
    if not model_dir.exists():
        sys.exit(f"ERROR: model_dir not found: {model_dir}")

    # load model
    rf = joblib.load(model_dir / "rf_model.joblib")
    platt_path = model_dir / "platt_scaler.joblib"
    platt_lr = joblib.load(platt_path) if platt_path.exists() else None

    results = json.loads((model_dir / "results.json").read_text())
    thr = results["threshold"]
    data_cfg = results["data_config"]

    # CLI overrides → results.json → config defaults
    resolution = args.resolution or data_cfg.get("resolution", RESOLUTION)
    spatial_res = args.spatial or data_cfg.get("spatial_res", SPATIAL_RES)
    val_years = args.val_years or data_cfg.get("val_years", VAL_YEARS)
    test_years = args.test_years or data_cfg.get("test_years", TEST_YEARS)
    sds_path = resolve_path(args.sds_path or data_cfg.get("sds_path") or SDS_PATH)
    split_years = test_years if args.split == "test" else val_years

    sds_file = Path(sds_path)
    if not sds_file.exists():
        sys.exit(f"ERROR: labels not found: {sds_file} (run the backtracking first)")

    od = Path(args.out_dir) if args.out_dir else model_dir / "event_analysis"
    od.mkdir(parents=True, exist_ok=True)
    fd = Path(args.fig_dir) if args.fig_dir else model_figures_dir(model_dir, "event_analysis")
    fd.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(od / f"event_iou_{args.split}.log"),
        ],
    )
    logger = logging.getLogger("event_iou")

    header("Event-level overlap analysis", {
        "Model": project_relative(model_dir),
        "Split": f"{args.split} {split_years}",
        "Threshold": f"{thr:.4f}",
        "Output": f"{project_relative(od)}, {project_relative(fd)}",
    }, out=logger.info)

    # load data
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

    unique_lats = ds_sds["latitude"].values
    unique_lons = ds_sds["longitude"].values
    nlat, nlon = len(unique_lats), len(unique_lons)
    n_land = len(land_indices)

    section("Events per day", out=logger.info)

    # connected-component structure: 8 = diagonal neighbours join an event, 4 = only edges
    struct = np.ones((3, 3)) if EVENT_CONNECTIVITY == 8 else ndimage.generate_binary_structure(2, 1)
    all_recall_events = []
    all_precision_events = []

    t0 = time.time()
    n_days_processed = 0

    for year in years:
        if year not in cache.X:
            continue

        unique_times = cache.unique_times[year]
        n_times = len(unique_times)

        # ground-truth target for this year (1 row per land cell per day)
        y_year = make_target_for_year(
            year, ds_sds, sds_var, unique_times, land_indices)

        X_year = cache.X[year]
        finite_mask = cache.finite_mask[year]

        # RF inference for all finite samples this year, in batches
        proba_year = np.zeros(cache.n_samples[year], dtype=np.float32)
        finite_idx = np.where(finite_mask)[0]

        if len(finite_idx) > 0:
            batch_size = 500_000
            for start in range(0, len(finite_idx), batch_size):
                end = min(start + batch_size, len(finite_idx))
                idx_batch = finite_idx[start:end]
                raw = rf.predict_proba(X_year[idx_batch])[:, 1]
                if platt_lr is not None:
                    raw = platt_apply(platt_lr, raw)
                proba_year[idx_batch] = raw

        # go through each day, build truth/pred grids, count events
        for ti in range(n_times):
            start_idx = ti * n_land
            end_idx = start_idx + n_land

            y_day = y_year[start_idx:end_idx]
            p_day = proba_year[start_idx:end_idx]
            fin_day = finite_mask[start_idx:end_idx]

            truth_grid = np.zeros((nlat, nlon), dtype=bool)
            pred_grid = np.zeros((nlat, nlon), dtype=bool)

            # scatter the 1D land-cell values back into the 2D lat/lon grid
            for k in range(n_land):
                li_2d = land_indices[k] // nlon
                lo_2d = land_indices[k] % nlon
                if fin_day[k]:
                    if y_day[k] == 1:
                        truth_grid[li_2d, lo_2d] = True
                    if p_day[k] >= thr:
                        pred_grid[li_2d, lo_2d] = True

            # skip days with no activity
            if not truth_grid.any() and not pred_grid.any():
                continue

            recall_ev, precision_ev = analyze_events_for_day(
                truth_grid, pred_grid, struct)

            day_str = str(unique_times[ti])[:10]
            for ev in recall_ev:
                ev["day"] = day_str
            for ev in precision_ev:
                ev["day"] = day_str

            all_recall_events.extend(recall_ev)
            all_precision_events.extend(precision_ev)
            n_days_processed += 1

        logger.info(f"  {year}: {n_times} days")

        del y_year, proba_year
        gc.collect()

    ds_sds.close()
    t_total = time.time() - t0
    recall_df = pd.DataFrame(all_recall_events)
    precision_df = pd.DataFrame(all_precision_events)

    logger.info(f"  Events: {len(recall_df):,} true, {len(precision_df):,} predicted "
                f"({t_total / 60:.1f} min)")

    if len(recall_df) == 0 or len(precision_df) == 0:
        logger.warning("  No events found, analysis skipped")
        return

    # compute rates at multiple overlap thresholds
    thresholds = EVENT_OVERLAP_THRESHOLDS

    recall_rates = compute_rates(recall_df, thresholds)
    precision_rates = compute_rates(precision_df, thresholds)

    # plots
    tag = args.split

    plot_rates(recall_rates, precision_rates, fd, tag)
    plot_rates_pixel_only(recall_rates, precision_rates, fd, tag)
    plot_overlap_histograms(recall_df, precision_df, fd, tag)
    plot_by_event_size(recall_df, precision_df, fd, tag)

    # save raw data alongside the plots
    recall_df.to_csv(od / f"recall_events_{tag}.csv", index=False)
    precision_df.to_csv(od / f"precision_events_{tag}.csv", index=False)
    recall_rates.to_csv(od / f"recall_rates_{tag}.csv", index=False)
    precision_rates.to_csv(od / f"precision_rates_{tag}.csv", index=False)

    summary = {
        "split": tag,
        "threshold": thr,
        "n_days": n_days_processed,
        "n_truth_events": len(recall_df),
        "n_pred_events": len(precision_df),
        "recall_overlap_mean": float(recall_df["overlap_frac"].mean()),
        "recall_overlap_median": float(recall_df["overlap_frac"].median()),
        "precision_overlap_mean": float(precision_df["overlap_frac"].mean()),
        "precision_overlap_median": float(precision_df["overlap_frac"].median()),
        # event rate per overlap threshold (config.EVENT_OVERLAP_THRESHOLDS)
        "recall": {f"{t:g}": float(r) for t, r in
                   zip(recall_rates["threshold"], recall_rates["event_rate"])},
        "precision": {f"{t:g}": float(r) for t, r in
                      zip(precision_rates["threshold"], precision_rates["event_rate"])},
    }
    with open(od / f"event_iou_summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    section("Results", out=logger.info)
    shown = [k for k in ("0.1", "0.3", "0.5") if k in summary["recall"]] or list(summary["recall"])
    for k in shown:
        logger.info(f"  {float(k):.0%} overlap: {summary['recall'][k]:.1%} of true events found, "
                    f"{summary['precision'][k]:.1%} of predicted events real")
    done(od, out=logger.info)


if __name__ == "__main__":
    main()