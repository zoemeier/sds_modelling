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
  python event_iou_analysis.py --model_dir models/final_rf_v1_20260413_120000
  python event_iou_analysis.py --model_dir models/final_rf_v1_20260413_120000 --split val
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

from config import (
    RESOLUTION, SPATIAL_RES, VAL_YEARS, TEST_YEARS, SDS_PATH,
)
from data_utils import (
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


def compute_event_f1(recall_rates, precision_rates):
    # harmonic mean of recall and precision at each overlap threshold, both versions
    merged = recall_rates.merge(precision_rates, on="threshold",
                                 suffixes=("_recall", "_precision"))
    r = merged["event_rate_recall"]
    p = merged["event_rate_precision"]
    merged["event_f1"] = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)

    rw = merged["pixel_weighted_rate_recall"]
    pw = merged["pixel_weighted_rate_precision"]
    merged["pixel_f1"] = np.where((pw + rw) > 0, 2 * pw * rw / (pw + rw), 0.0)

    return merged


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_event_f1(f1_df, od, tag):
    # two plots: one pixel-weighted, one event-level - each shows F1/precision/recall vs overlap thr
    lw = 2.0
    ms = 5

    # ── Pixel-weighted plot ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(f1_df["threshold"], f1_df["pixel_f1"],
            "o-", color="#222", lw=lw, ms=ms, label="F1")
    ax.plot(f1_df["threshold"], f1_df["pixel_weighted_rate_recall"],
            "^-", color="#70AD47", lw=lw, ms=ms, label="Recall")
    ax.plot(f1_df["threshold"], f1_df["pixel_weighted_rate_precision"],
            "s-", color="#4472C4", lw=lw, ms=ms, label="Precision")
    ax.set_xlabel("Overlap Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    ax.set_title(f"Pixel-Weighted Event Metrics vs Overlap Threshold — {tag}",
                 fontsize=14, fontweight="bold")
    fig.savefig(od / f"event_iou_f1_pixel_{tag}.png")
    plt.close(fig)

    # ── Event-level plot ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(f1_df["threshold"], f1_df["event_f1"],
            "o-", color="#222", lw=lw, ms=ms, label="F1")
    ax.plot(f1_df["threshold"], f1_df["event_rate_recall"],
            "^-", color="#70AD47", lw=lw, ms=ms, label="Recall")
    ax.plot(f1_df["threshold"], f1_df["event_rate_precision"],
            "s-", color="#4472C4", lw=lw, ms=ms, label="Precision")
    ax.set_xlabel("Overlap Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    ax.set_title(f"Event-Level Metrics vs Overlap Threshold — {tag}",
                 fontsize=14, fontweight="bold")
    fig.savefig(od / f"event_iou_f1_event_{tag}.png")
    plt.close(fig)


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
    size_bins = [1, 2, 5, 10, 20, 50, 100, 500, np.inf]
    overlap_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: model_dir not found: {model_dir}")
        sys.exit(1)

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
    sds_path = args.sds_path or data_cfg.get("sds_path") or SDS_PATH
    split_years = test_years if args.split == "test" else val_years

    if sds_path is None:
        print("ERROR: No SDS path")
        sys.exit(1)
    sds_file = Path(sds_path)

    od = Path(args.out_dir) if args.out_dir else model_dir / "event_analysis"
    od.mkdir(parents=True, exist_ok=True)

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

    logger.info("=" * 60)
    logger.info("EVENT-LEVEL OVERLAP ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"  Model: {model_dir}")
    logger.info(f"  Split: {args.split} ({split_years})")
    logger.info(f"  Threshold: {thr:.4f}")
    logger.info(f"  Output: {od}")
    logger.info("=" * 60)

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

    logger.info(f"  Grid: {nlat} × {nlon}, land cells: {n_land:,}")
    logger.info(f"\nProcessing days ...")

    struct = np.ones((3, 3))   # 8-connectivity for connected components
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

        logger.info(f"  {year}: {n_times} days processed "
                    f"({len(all_recall_events):,} truth events, "
                    f"{len(all_precision_events):,} pred events so far)")

        del y_year, proba_year
        gc.collect()

    ds_sds.close()
    t_total = time.time() - t0
    logger.info(f"\n  Total: {n_days_processed} days in {t_total:.1f}s")

    recall_df = pd.DataFrame(all_recall_events)
    precision_df = pd.DataFrame(all_precision_events)

    logger.info(f"  Ground-truth events: {len(recall_df):,}")
    logger.info(f"  Predicted events: {len(precision_df):,}")

    if len(recall_df) == 0 or len(precision_df) == 0:
        logger.warning("  No events found — skipping analysis")
        return

    # compute rates at multiple overlap thresholds
    thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    recall_rates = compute_rates(recall_df, thresholds)
    precision_rates = compute_rates(precision_df, thresholds)

    logger.info(f"\n  Recall direction (ground-truth events found):")
    logger.info(f"  {'Overlap':>10s} {'Event rate':>12s} {'Pixel-weighted':>16s} "
                f"{'Found':>8s} {'Total':>8s}")
    logger.info(f"  {'-'*58}")
    for _, r in recall_rates.iterrows():
        logger.info(f"  {r['threshold']:>10.2f} {r['event_rate']:>12.1%} "
                    f"{r['pixel_weighted_rate']:>16.1%} "
                    f"{r['n_found']:>8,} {r['n_total']:>8,}")

    logger.info(f"\n  Precision direction (predicted events that are real):")
    logger.info(f"  {'Overlap':>10s} {'Event rate':>12s} {'Pixel-weighted':>16s} "
                f"{'Found':>8s} {'Total':>8s}")
    logger.info(f"  {'-'*58}")
    for _, r in precision_rates.iterrows():
        logger.info(f"  {r['threshold']:>10.2f} {r['event_rate']:>12.1%} "
                    f"{r['pixel_weighted_rate']:>16.1%} "
                    f"{r['n_found']:>8,} {r['n_total']:>8,}")

    logger.info(f"\n  Recall event overlap: "
                f"mean={recall_df['overlap_frac'].mean():.3f}, "
                f"median={recall_df['overlap_frac'].median():.3f}")
    logger.info(f"  Precision event overlap: "
                f"mean={precision_df['overlap_frac'].mean():.3f}, "
                f"median={precision_df['overlap_frac'].median():.3f}")
    logger.info(f"  Recall event sizes: "
                f"mean={recall_df['size'].mean():.1f}, "
                f"median={recall_df['size'].median():.0f}")
    logger.info(f"  Precision event sizes: "
                f"mean={precision_df['size'].mean():.1f}, "
                f"median={precision_df['size'].median():.0f}")

    # plots
    logger.info(f"\nPlotting ...")
    tag = args.split

    plot_rates(recall_rates, precision_rates, od, tag)
    plot_rates_pixel_only(recall_rates, precision_rates, od, tag)
    plot_overlap_histograms(recall_df, precision_df, od, tag)
    plot_by_event_size(recall_df, precision_df, od, tag)

    f1_df = compute_event_f1(recall_rates, precision_rates)
    plot_event_f1(f1_df, od, tag)

    logger.info(f"\n  Event-level F1 (pixel-weighted):")
    logger.info(f"  {'Overlap':>10s} {'Recall':>10s} {'Precision':>10s} {'F1':>10s}")
    logger.info(f"  {'-'*44}")
    for _, r in f1_df.iterrows():
        logger.info(f"  {r['threshold']:>10.2f} "
                    f"{r['pixel_weighted_rate_recall']:>10.3f} "
                    f"{r['pixel_weighted_rate_precision']:>10.3f} "
                    f"{r['pixel_f1']:>10.3f}")

    best_idx = f1_df["pixel_f1"].idxmax()
    best_thr = f1_df.loc[best_idx, "threshold"]
    best_f1 = f1_df.loc[best_idx, "pixel_f1"]
    logger.info(f"\n  Best pixel-weighted F1: {best_f1:.3f} "
                f"(at overlap ≥ {best_thr:.2f})")

    # save raw data alongside the plots
    recall_df.to_csv(od / f"recall_events_{tag}.csv", index=False)
    precision_df.to_csv(od / f"precision_events_{tag}.csv", index=False)
    recall_rates.to_csv(od / f"recall_rates_{tag}.csv", index=False)
    precision_rates.to_csv(od / f"precision_rates_{tag}.csv", index=False)
    f1_df.to_csv(od / f"event_f1_{tag}.csv", index=False)

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
        "recall_at_0.1": float(recall_rates.loc[
            recall_rates["threshold"] == 0.1, "event_rate"].iloc[0]),
        "recall_at_0.3": float(recall_rates.loc[
            recall_rates["threshold"] == 0.3, "event_rate"].iloc[0]),
        "recall_at_0.5": float(recall_rates.loc[
            recall_rates["threshold"] == 0.5, "event_rate"].iloc[0]),
        "precision_at_0.1": float(precision_rates.loc[
            precision_rates["threshold"] == 0.1, "event_rate"].iloc[0]),
        "precision_at_0.3": float(precision_rates.loc[
            precision_rates["threshold"] == 0.3, "event_rate"].iloc[0]),
        "precision_at_0.5": float(precision_rates.loc[
            precision_rates["threshold"] == 0.5, "event_rate"].iloc[0]),
        "best_pixel_f1": float(best_f1),
        "best_pixel_f1_threshold": float(best_thr),
    }
    with open(od / f"event_iou_summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("EVENT OVERLAP ANALYSIS COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Output: {od}")
    logger.info(f"  Key results:")
    logger.info(f"    Recall  @10% overlap: "
                f"{summary['recall_at_0.1']:.1%} of storms found")
    logger.info(f"    Recall  @30% overlap: "
                f"{summary['recall_at_0.3']:.1%} of storms found")
    logger.info(f"    Recall  @50% overlap: "
                f"{summary['recall_at_0.5']:.1%} of storms found")
    logger.info(f"    Precision @10% overlap: "
                f"{summary['precision_at_0.1']:.1%} of predictions real")
    logger.info(f"    Precision @30% overlap: "
                f"{summary['precision_at_0.3']:.1%} of predictions real")
    logger.info(f"    Precision @50% overlap: "
                f"{summary['precision_at_0.5']:.1%} of predictions real")
    logger.info(f"    Best pixel-weighted F1: {best_f1:.3f} "
                f"(overlap ≥ {best_thr:.2f})")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()