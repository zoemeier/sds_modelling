"""
Visualisation of RF results for SDS detection
parameters read from config, only need to specify which model to run it for: 

Usage:
  python analyze_rf_results.py --model_dir models/... 
  (set "--no_animation" to skip animations, theyre quite slow)
"""

import argparse, gc, json, logging, sys, warnings
from pathlib import Path

import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np, pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score, accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from scipy.stats import gaussian_kde
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# custom imports 
from config import RESOLUTION, SPATIAL_RES, VAL_YEARS, TEST_YEARS, SDS_PATH
from data_utils import (
    FeatureCache, load_sds_source, compute_land_indices,
    collect_indices, extract_from_cache, platt_apply,
)
from feature_names import get_feature_name_map, get_active_groups

warnings.filterwarnings("ignore")


# Plotting parameters
# ---------------------------------------------------------------------------
# Styling rule for this module (kept consistent across every plot function):
#
#   - Main titles (suptitle, or a single set_title on solo-axis figures):
#         fontsize=TITLE_SIZE,  fontweight="bold"
#   - Subplot/panel titles inside multi-axis figures:
#         fontsize=SUBTITLE_SIZE or SUBPLOT_TITLE_SIZE, non-bold
#         (SUBTITLE_SIZE for a few big subplots like ROC+PR;
#          SUBPLOT_TITLE_SIZE for small-grid subplots like KDE groups / composite)
#   - Axis labels:   LABEL_SIZE (rcParams default)
#   - Tick labels:   TICK_SIZE  (rcParams default)
#   - Legends:       LEGEND_SIZE (rcParams default)
#
# In-axis text that is *data* rather than *chrome* (e.g. confusion-matrix cell
# values, bar-chart count annotations) uses LABEL_SIZE ± a small offset and
# may be bold if it's the plot's primary readout.
#
# Don't hardcode raw point sizes - derive from these constants so a single
# edit here retunes the whole module.
TITLE_SIZE = 14
SUBTITLE_SIZE = 12
LABEL_SIZE = 11
TICK_SIZE = 9
LEGEND_SIZE = 9
SUBPLOT_TITLE_SIZE = 11

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": LABEL_SIZE,
    "axes.titlesize": SUBTITLE_SIZE, "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE, "figure.dpi": 150,
    "savefig.dpi": 250, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})

OUTCOME_COLORS = {"TN": "#5B9BD5", "FP": "#ED7D31", "FN": "#C00000", "TP": "#70AD47"}
OUTCOME_ALPHA = 0.7  # shared alpha for outcome-coloured elements 
LABEL_COLORS = {0: "#5B9BD5", 1: "#C00000"}

F1_CMAP = LinearSegmentedColormap.from_list(
    "f1", ["#D04040", "#E8944A", "#F0D060", "#B8D8A0", "#6E9E48"])

# colors for genesis (source origin) vs intermediate (downwind) positives
STYPE_COLORS = {1: "#7E4E91", 2: "#FFB347"}   # purple = genesis, orange = intermediate
STYPE_LABELS = {1: "Genesis", 2: "Intermediate"}

VAR_SUBPLOT_W = 4.0
VAR_SUBPLOT_H = 3.0

DOMAIN = {"north": 50.0, "south": 0.0, "west": -18.0, "east": 85.0}
_YEAR_RANGES: dict = {} #filled in main with years from config


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def safe_div(a, b): return np.where(b > 0, a / b, np.nan)  # for precision/recall in regions with no positives

def _tag_label(tag):
    # for nice labels of splits using years from config, e.g. Test (2024)
    yr = _YEAR_RANGES.get(tag)
    yr_str = str(yr[0]) if yr[0] == yr[1] else f"{yr[0]}–{yr[1]}"
    return f"{'Test' if tag == 'test' else 'Validation'} ({yr_str})"
    
def outcome_labels(yt, yp):
    # TN / FP / FN / TP per sample
    o = np.empty(len(yt), dtype=object)
    o[(yt==0)&(yp==0)]="TN"; o[(yt==0)&(yp==1)]="FP"
    o[(yt==1)&(yp==0)]="FN"; o[(yt==1)&(yp==1)]="TP"
    return o

def _map_ax(fig, pos=111):
    # standard cartopy axes with coastlines/borders/gridlines - used by all map plots
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

def _mapplot(lons, lats, data, title, cmap, vmin, vmax, fname, od, label="",
             alpha=0.80):
    finite = data[np.isfinite(data)]
    if vmin is None:
        vmin = float(finite.min()) if len(finite) > 0 else 0.0
    if vmax is None:
        vmax = float(finite.max()) if len(finite) > 0 else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    lo2, la2 = np.meshgrid(lons, lats)
    lon_range = DOMAIN["east"] - DOMAIN["west"]
    lat_range = DOMAIN["north"] - DOMAIN["south"]
    fig_w = 13
    fig_h = fig_w * (lat_range / lon_range) + 1.8
    fig = plt.figure(figsize=(fig_w, fig_h))

    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([DOMAIN["west"], DOMAIN["east"],
                   DOMAIN["south"], DOMAIN["north"]],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, lw=0.6, ec="black", zorder=5)
    ax.add_feature(cfeature.BORDERS,   lw=0.3, ec="#444",  ls="--", zorder=5)
    gl = ax.gridlines(draw_labels=True, lw=0.3, alpha=0.3, ls=":")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": TICK_SIZE}

    im = ax.pcolormesh(lo2, la2, np.ma.masked_invalid(data), cmap=cmap,
                       vmin=vmin, vmax=vmax, shading="auto", alpha=alpha,
                       zorder=1, transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)

    # ---- colorbar aligned with the actual visible map frame ----
    # Force a draw so the cartopy axes reports its true (post-letterbox) position,
    # then carve the colorbar axes out by hand using that exact y-range.
    fig.canvas.draw()
    pos = ax.get_position()  # bbox of the *visible* map frame in figure coords
    cb_w   = 0.018           # colorbar width (figure fraction)
    cb_pad = 0.012           # gap between map and colorbar
    cax = fig.add_axes([pos.x1 + cb_pad, pos.y0, cb_w, pos.height])

    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_label(label, fontsize=LABEL_SIZE)
    cb.ax.tick_params(labelsize=TICK_SIZE)

    print(f"    _mapplot: vmin={vmin:.4f} vmax={vmax:.4f} "
          f"finite={len(finite)} nan={np.isnan(data).sum()} fname={fname}")

    
    fig.savefig(od / fname)
    plt.close(fig)

def _build_grids(lat, lon, y, proba, oc):
    ul, ulo = np.sort(np.unique(lat)), np.sort(np.unique(lon))
    nl, no = len(ul), len(ulo)
    li, loi = np.searchsorted(ul, lat), np.searchsorted(ulo, lon)

    print(f"    _build_grids: {len(y)} samples, {nl} lats, {no} lons, "
          f"li range [{li.min()},{li.max()}], loi range [{loi.min()},{loi.max()}]")

    TP = np.zeros((nl, no)); FP = np.zeros((nl, no))
    FN = np.zeros((nl, no)); TN = np.zeros((nl, no))
    ps = np.zeros((nl, no)); pc = np.zeros((nl, no))
    has_data = np.zeros((nl, no), dtype=bool)

    # ---- accumulate per-cell counts (this is what was missing) ----
    # vectorised binning - much faster than a python loop over samples
    np.add.at(TP, (li, loi), (oc == "TP").astype(np.float64))
    np.add.at(FP, (li, loi), (oc == "FP").astype(np.float64))
    np.add.at(FN, (li, loi), (oc == "FN").astype(np.float64))
    np.add.at(TN, (li, loi), (oc == "TN").astype(np.float64))
    np.add.at(ps, (li, loi), proba.astype(np.float64))
    np.add.at(pc, (li, loi), 1.0)
    has_data[li, loi] = True

    tot = TP + FP + FN + TN
    tp_fn = TP + FN

    # F1 / precision / recall, with the NaN conventions described in the docstring
    f1   = np.full((nl, no), np.nan)
    prec = np.full((nl, no), np.nan)
    rec  = np.full((nl, no), np.nan)

    for i in range(nl):
        for j in range(no):
            if not has_data[i, j]:
                continue
            if tp_fn[i, j] == 0 and FP[i, j] == 0:
                # nothing happened here — all metrics undefined
                continue
            elif tp_fn[i, j] == 0 and FP[i, j] > 0:
                f1[i, j] = 0.0
                prec[i, j] = 0.0
                # recall stays NaN
            else:
                denom = 2 * TP[i, j] + FP[i, j] + FN[i, j]
                f1[i, j] = 2 * TP[i, j] / denom if denom > 0 else 0.0
                prec[i, j] = TP[i, j] / (TP[i, j] + FP[i, j]) if (TP[i, j] + FP[i, j]) > 0 else np.nan
                rec[i, j]  = TP[i, j] / tp_fn[i, j] if tp_fn[i, j] > 0 else np.nan

    return dict(lats=ul, lons=ulo, TP=TP, FP=FP, FN=FN, TN=TN,
                total=tot, total_pos=tp_fn, f1=f1, precision=prec, recall=rec,
                mean_prob=safe_div(ps, pc), has_data=has_data)


def make_stype_for_year(year, ds_sds, unique_times, land_indices):
    # parallel to make_target_for_year in data_utils - returns uint8 array of the same shape
    # and ordering, so stype[k] aligns with y[k] and cache.X[year][k]
    yr_start, yr_end = f"{year}-01-01", f"{year}-12-31"
    sds_yr = ds_sds.sel(time=slice(yr_start, yr_end))
    sds_yr = sds_yr.reindex(time=unique_times, fill_value=0)
    vals = sds_yr["source_type"].values.astype(np.uint8)
    parts = [vals[ti].ravel()[land_indices] for ti in range(vals.shape[0])]
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Basic plotting functions
# ---------------------------------------------------------------------------
def plot_feat_imp(feat_df, od, tag, name_map):
    # top 30 features by MDI, horizontal bar chart
    print(f"  [01] Feature importance (MDI) ...")
    top = feat_df.head(30).iloc[::-1].copy()
    import re
    top["full_name"] = top["feature"].map(
        lambda f: re.sub(r"\s*\[.*?\]", "", name_map.get(f, f)))
    fig, ax = plt.subplots(figsize=(9, max(6, len(top)*0.32)))
    ax.barh(range(len(top)), top["importance"].values, color="#4472C4")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top["full_name"].values, fontsize=TICK_SIZE)
    ax.set_xlabel("Importance (Mean Decrease in Impurity)")
    ax.set_title(f"Feature Importance — {_tag_label(tag)}", fontsize=TITLE_SIZE, fontweight="bold", loc="center")
    fig.savefig(od/f"01_feature_importance_{tag}.png"); plt.close(fig)

def plot_roc_pr(y, proba, thr, od, tag):
    # ROC curve + precision-recall curve side by side
    print("  [02] ROC + PR ...")
    fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5.5))
    fpr,tpr,_=roc_curve(y,proba); auc=roc_auc_score(y,proba)
    a1.plot(fpr,tpr,color="#4472C4",lw=2,label=f"AUC = {auc:.4f}")
    a1.plot([0,1],[0,1],"k--",alpha=.4,lw=1); a1.set_xlabel("False Positive Rate")
    a1.set_ylabel("True Positive Rate"); a1.set_title("ROC Curve",fontsize=SUBTITLE_SIZE)
    a1.legend(); a1.grid(alpha=.2)
    pr,rc,_=precision_recall_curve(y,proba); f1v=2*pr*rc/(pr+rc+1e-8)
    a2.plot(rc,pr,color="#4472C4",lw=2,label=f"Best F1 = {f1v.max():.4f}")
    a2.axhline(y.mean(),color="#999",ls="--",lw=1,label=f"Baseline = {y.mean():.4f}")
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision")
    a2.set_title("Precision–Recall Curve",fontsize=SUBTITLE_SIZE); a2.legend(); a2.grid(alpha=.2)
    fig.suptitle(f"Classification Curves — {_tag_label(tag)}",fontsize=TITLE_SIZE, fontweight="bold")
    fig.savefig(od/f"02_roc_pr_{tag}.png"); plt.close(fig)

def plot_confusion(y, yp, od, tag):
    print("  [03] Confusion matrix ...")
    cm=confusion_matrix(y,yp); fig,ax=plt.subplots(figsize=(6.5,5.5))
    im=ax.imshow(cm,cmap="Blues",aspect="auto")
    for i in range(2):
        for j in range(2):
            c="white" if cm[i,j]>cm.max()/2 else "black"
            ax.text(j,i,f"{cm[i,j]:,}",ha="center",va="center",fontsize=LABEL_SIZE + 3,fontweight="bold",color=c)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Predicted Negative","Predicted Positive"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["Actual Negative","Actual Positive"])
    ax.set_title(f"Confusion Matrix — {_tag_label(tag)}",fontsize=TITLE_SIZE, fontweight="bold")
    plt.colorbar(im,ax=ax,shrink=.8); fig.savefig(od/f"03_confusion_{tag}.png"); plt.close(fig)

def plot_prob_hist(proba, y, thr, od, tag):
    # overlaid histograms of predicted prob for pos/neg classes (linear + log)
    print("  [04] Probability histogram ...")
    bins=np.linspace(0,1,101)

    fig,ax=plt.subplots(figsize=(10,5))
    ax.hist(proba[y==0],bins=bins,alpha=.6,color=LABEL_COLORS[0],label="SDS Negative",density=True)
    ax.hist(proba[y==1],bins=bins,alpha=.6,color=LABEL_COLORS[1],label="SDS Positive",density=True)
    ax.axvline(thr,color="#222",lw=2,ls="--",label=f"Threshold = {thr:.3f}")
    ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Density")
    ax.set_title(f"Probability Distribution — {_tag_label(tag)}",fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend()
    fig.savefig(od/f"04_prob_hist_{tag}.png"); plt.close(fig)

    # log density version
    fig,ax=plt.subplots(figsize=(10,5))
    ax.hist(proba[y==0],bins=bins,alpha=.6,color=LABEL_COLORS[0],label="SDS Negative",density=True)
    ax.hist(proba[y==1],bins=bins,alpha=.6,color=LABEL_COLORS[1],label="SDS Positive",density=True)
    ax.axvline(thr,color="#222",lw=2,ls="--",label=f"Threshold = {thr:.3f}")
    ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Density (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"Probability Distribution (log) — {_tag_label(tag)}",fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend()
    fig.savefig(od/f"04_prob_hist_log_{tag}.png"); plt.close(fig)

def plot_outcome_prob_hist(proba, oc, thr, od, tag):
    # one subplot per outcome with summary stats - also dumps a CSV of the stats
    print("  [04b] Outcome probability distributions ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    pairs = [("TN",axes[0,0],"FP",axes[0,1]),("FN",axes[1,0],"TP",axes[1,1])]
    for oc1,ax1,oc2,ax2 in pairs:
        bins=np.linspace(0,1,201)
        counts1,_=np.histogram(proba[oc==oc1],bins=bins) if (oc==oc1).sum()>0 else (np.zeros(200),None)
        counts2,_=np.histogram(proba[oc==oc2],bins=bins) if (oc==oc2).sum()>0 else (np.zeros(200),None)
        ymx=max(counts1.max(),counts2.max(),1)*2
        for oc_l,ax in [(oc1,ax1),(oc2,ax2)]:
            mask=oc==oc_l; n=int(mask.sum())
            if n==0: ax.set_title(f"{oc_l} (n=0)",fontsize=SUBPLOT_TITLE_SIZE); continue
            vals=proba[mask]
            ax.hist(vals,bins=bins,color=OUTCOME_COLORS[oc_l],alpha=OUTCOME_ALPHA,edgecolor="none")
            mn,mx,mean,med=vals.min(),vals.max(),vals.mean(),np.median(vals)
            p25,p75=np.percentile(vals,25),np.percentile(vals,75)
            ax.axvline(mean,color="#222",ls="-",lw=1.5,label=f"Mean = {mean:.4f}")
            ax.axvline(med,color="#222",ls="--",lw=1.5,label=f"Median = {med:.4f}")
            ax.axvline(thr,color="#999",ls=":",lw=1.5,label=f"Threshold = {thr:.3f}")
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Count (log scale)")
            ax.set_title(f"{oc_l} (n={n:,})  [{mn:.4f} – {mx:.4f}]  IQR=[{p25:.4f}, {p75:.4f}]",
                        fontsize=SUBPLOT_TITLE_SIZE)
            ax.legend(fontsize=LEGEND_SIZE); ax.set_yscale("log"); ax.set_ylim(0.8,ymx)
    fig.suptitle(f"Probability Distribution by Outcome — {_tag_label(tag)}",fontsize=TITLE_SIZE, fontweight="bold", y=1.01)
    plt.tight_layout(); fig.savefig(od/f"04b_outcome_prob_dist_{tag}.png"); plt.close(fig)
  
    rows=[]
    for oc_l in ["TN","FP","FN","TP"]:
        m=oc==oc_l
        if m.sum()==0: continue
        v=proba[m]
        rows.append({"outcome":oc_l,"count":int(m.sum()),"min":float(v.min()),
                      "p05":float(np.percentile(v,5)),"p25":float(np.percentile(v,25)),
                      "median":float(np.median(v)),"mean":float(v.mean()),
                      "p75":float(np.percentile(v,75)),"p95":float(np.percentile(v,95)),
                      "max":float(v.max())})
    pd.DataFrame(rows).to_csv(od/f"04b_outcome_prob_stats_{tag}.csv",index=False)

def plot_calibration(y, proba, od, tag):
    # reliability diagram + prediction histogram - checks if probabilities are well-calibrated
    print("  [05] Calibration ...")
    fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5.5))
    fp,mp=calibration_curve(y,proba,n_bins=20,strategy="uniform")
    a1.plot(mp,fp,"o-",color="#4472C4",lw=2,label="Model")
    a1.plot([0,1],[0,1],"k--",alpha=.4,lw=1,label="Perfect")
    a1.set_xlabel("Mean Predicted Probability"); a1.set_ylabel("Observed Frequency")
    a1.set_title("Reliability Diagram",fontsize=SUBTITLE_SIZE); a1.legend(); a1.grid(alpha=.2)
    a2.hist(proba,bins=20,range=(0,1),color="#4472C4",alpha=.7,edgecolor="white")
    a2.set_xlabel("Predicted Probability"); a2.set_ylabel("Count")
    a2.set_title("Prediction Distribution",fontsize=SUBTITLE_SIZE); a2.grid(alpha=.2)
    fig.suptitle(f"Probability Calibration — {_tag_label(tag)}",fontsize=TITLE_SIZE, fontweight="bold")
    fig.savefig(od/f"05_calibration_{tag}.png"); plt.close(fig)

def plot_threshold_tradeoff(y, proba, thr, od, tag):
    # precision, recall, accuracy & F1 at different thresholds 
    print("  [06] Threshold tradeoff ...")
    thresholds=np.linspace(0.01,0.99,200)
    f1s,precs,recs,accs=[],[],[],[]
    for t in thresholds:
        yp=(proba>=t).astype(int)
        f1s.append(f1_score(y,yp,zero_division=0)); precs.append(precision_score(y,yp,zero_division=0))
        recs.append(recall_score(y,yp,zero_division=0)); accs.append(accuracy_score(y,yp))
    fig,ax=plt.subplots(figsize=(10,5.5))
    ax.plot(thresholds,f1s,color="#222",lw=2.5,label="F1")
    ax.plot(thresholds,precs,color=OUTCOME_COLORS["FP"],lw=1.8,ls="--",label="Precision")
    ax.plot(thresholds,recs,color=OUTCOME_COLORS["FN"],lw=1.8,ls="--",label="Recall")
    ax.plot(thresholds,accs,color="#4472C4",lw=1.8,ls="-.",label="Accuracy")
    ax.axvline(thr,color="#999",lw=1.5,ls=":",label=f"Selected = {thr:.3f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score"); ax.set_ylim(0,1)
    ax.set_title(f"Threshold vs Metrics — {_tag_label(tag)}",fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(od/f"06_threshold_{tag}.png"); plt.close(fig)


# ---------------------------------------------------------------------------
# Feature distribution (KDE) analysis — overview grids + grouped per-variable plots
# ---------------------------------------------------------------------------
def _kde_grid(X, groups, fnames, colors, labels, title, fname, od, name_map, max_per=100_000):
    # overview grid: one small KDE subplot per feature, overlaid per group
    rng=np.random.default_rng(99)
    sampled={k:rng.choice(v,min(max_per,len(v)),replace=False) for k,v in groups.items() if len(v)>0}
    nf=len(fnames); nc=5; nr=int(np.ceil(nf/nc))
    fig,axes=plt.subplots(nr,nc,figsize=(nc*VAR_SUBPLOT_W,nr*VAR_SUBPLOT_H))
    axes=axes.flatten()
    for i,fn in enumerate(fnames):
        ax=axes[i]
        for lbl,idxs in sampled.items():
            vals=X[idxs,i]; vals=vals[np.isfinite(vals)]
            if len(vals)>1:
                try:
                    kde=gaussian_kde(vals,bw_method="scott")
                    xs=np.linspace(np.percentile(vals,1),np.percentile(vals,99),200)
                    ax.plot(xs,kde(xs),color=colors[lbl],lw=1.5)
                    ax.fill_between(xs,kde(xs),alpha=.12,color=colors[lbl])
                except: pass
        ax.set_title(name_map.get(fn,fn),fontsize=SUBPLOT_TITLE_SIZE - 4)
        ax.set_yticks([]); ax.set_ylabel("Density",fontsize=TICK_SIZE - 2); ax.tick_params(labelsize=TICK_SIZE - 2)
    for j in range(i+1,len(axes)): axes[j].set_visible(False)
    handles=[Line2D([0],[0],color=colors[k],lw=2,label=labels[k]) for k in colors]
    # Compact title+legend band at the top. tight_layout(rect=...) reserves the
    # top 6% for chrome, then we place legend and suptitle inside that band in
    # figure-fraction coords so they sit close together without dead space in
    # between. The exact y values depend on how many rows the grid has - more
    # rows means 6% of fig_h is a larger absolute gap, so we tighten proportionally.
    # Rough rule: reserve ~0.7" for the chrome regardless of grid size.
    fig_h = fig.get_size_inches()[1]
    chrome_frac = min(0.08, 0.8 / fig_h)   # ~0.8" for title+legend, capped at 8% of fig
    plt.tight_layout(rect=[0, 0, 1, 1 - chrome_frac])
    # Legend sits just above the top row; suptitle sits just above the legend.
    # Both anchored in figure-fraction so they scale with fig_h.
    legend_y = 1 - chrome_frac * 0.55
    title_y  = 1 - chrome_frac * 0.15
    fig.legend(handles=handles, loc="center", ncol=min(len(colors), 6),
               fontsize=LEGEND_SIZE, frameon=True,
               bbox_to_anchor=(0.5, legend_y))
    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold", y=title_y)
    fig.savefig(od/fname); plt.close(fig)

def _best_grid(n):
    # for nice arrangement of variable group sizes (no empty spots)
    layout = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3)}
    return layout.get(n, (int(np.ceil(n / 3)), 3))

def _kde_group(X, groups, fnames_in_group, colors, labels, group_title, tag, od, name_map, prefix, feature_vars, max_per=100_000):
    # helper to run kde for subgroups (to avoid the one huge plot)
    rng=np.random.default_rng(99)
    sampled={k:rng.choice(v,min(max_per,len(v)),replace=False) for k,v in groups.items() if len(v)>0}
    nf=len(fnames_in_group)
    nr, nc = _best_grid(nf)
    fig,axes=plt.subplots(nr,nc,figsize=(nc*VAR_SUBPLOT_W, nr*VAR_SUBPLOT_H))
    if nf==1: axes=np.array([axes])
    axes=axes.flatten()
    for i,fn in enumerate(fnames_in_group):
        ax=axes[i]
        if fn not in feature_vars:
            ax.set_visible(False); continue
        fi=feature_vars.index(fn)
        for lbl,idxs in sampled.items():
            vals=X[idxs,fi]; vals=vals[np.isfinite(vals)]
            if len(vals)>1:
                try:
                    kde=gaussian_kde(vals,bw_method="scott")
                    xs=np.linspace(np.percentile(vals,1),np.percentile(vals,99),200)
                    ax.plot(xs,kde(xs),color=colors[lbl],lw=1.8)
                    ax.fill_between(xs,kde(xs),alpha=.15,color=colors[lbl])
                except: pass
        ax.set_title(name_map.get(fn,fn),fontsize=SUBPLOT_TITLE_SIZE)
        ax.set_yticks([]); ax.set_ylabel("Density",fontsize=TICK_SIZE); ax.tick_params(labelsize=TICK_SIZE)
    for j in range(i+1,len(axes)): axes[j].set_visible(False)
    handles=[Line2D([0],[0],color=colors[k],lw=2,label=labels[k]) for k in colors]
    safe_title=group_title.replace("/","_").replace(" ","_").lower()
    # Legend goes at the bottom here (different from the overview grid): these
    # per-group plots have only a handful of subplots, so a bottom legend reads
    # naturally and keeps the top of the figure clear for the suptitle + panel
    # titles (which would otherwise get cramped by a top legend).
    # Reserve space for suptitle (top) and legend (bottom), then anchor both.
    fig_h = fig.get_size_inches()[1]
    top_chrome    = min(0.10, 0.55 / fig_h)   # ~0.55" for suptitle
    bottom_chrome = min(0.10, 0.45 / fig_h)   # ~0.45" for legend
    plt.tight_layout(rect=[0, bottom_chrome, 1, 1 - top_chrome])
    fig.legend(handles=handles, loc="center", ncol=min(len(colors), 6),
               fontsize=LEGEND_SIZE, frameon=True,
               bbox_to_anchor=(0.5, bottom_chrome * 0.5))
    fig.suptitle(f"{group_title} — {_tag_label(tag)}",
                 fontsize=TITLE_SIZE, fontweight="bold",
                 y=1 - top_chrome * 0.25)
    fig.savefig(od/f"{prefix}_{safe_title}_{tag}.png"); plt.close(fig)

def plot_kde_by_class(X, y, od, tag, name_map, feature_vars):
    # splits by true label (0 vs 1) - shows which features separate the classes
    print(f"  [07] KDE by class (overview + groups) ...")
    groups={0:np.where(y==0)[0], 1:np.where(y==1)[0]}
    colors={0:LABEL_COLORS[0], 1:LABEL_COLORS[1]}
    labels={0:"SDS Negative", 1:"SDS Positive"}
    _kde_grid(X,groups,feature_vars,colors,labels,
              f"Features by Class — {_tag_label(tag)}",
              f"07_kde_class_{tag}.png",od,name_map)
    for grp_title, members in get_active_groups(feature_vars):
        existing=[f for f in members if f in feature_vars]
        if existing:
            _kde_group(X,groups,existing,colors,labels,grp_title,tag,od,
                      name_map,"07g",feature_vars)

def plot_kde_by_outcome(X, oc, od, tag, name_map, feature_vars):
    # splits by confusion-matrix outcome - useful for seeing what distinguishes FPs from TNs etc.
    print(f"  [08] KDE by outcome (overview + groups) ...")
    groups={o:np.where(oc==o)[0] for o in ["TN","FP","FN","TP"]}
    colors=OUTCOME_COLORS
    labels={"TN": "TN", "FP": "FP", "FN": "FN", "TP": "TP"}
    _kde_grid(X,groups,feature_vars,colors,labels,
              f"Features by Outcome — {_tag_label(tag)}",
              f"08_kde_outcome_{tag}.png",od,name_map)
    for grp_title, members in get_active_groups(feature_vars):
        existing=[f for f in members if f in feature_vars]
        if existing:
            _kde_group(X,groups,existing,colors,labels,grp_title,tag,od,
                      name_map,"08g",feature_vars)


# ---------------------------------------------------------------------------
# Spatial plots
# ---------------------------------------------------------------------------
def plot_spatial(G, od, tag):
    # per-cell F1/precision/recall on the map
    print("  [09] Spatial metrics ...")
    la,lo=G["lats"],G["lons"]
    for key,label in [("f1","F1"),("precision","Precision"),("recall","Recall")]:
        _mapplot(lo,la,G[key],f"{label} — {_tag_label(tag)}",F1_CMAP,0,1,
                 f"09_{key}_{tag}.png",od,label)

def plot_count(G, od, tag):
    print("  [10] Count maps ...")
    la, lo = G["lats"], G["lons"]

    cnt = G["total"].copy().astype(float); cnt[cnt == 0] = np.nan
    _mapplot(lo, la, cnt, f"Sample Count — {_tag_label(tag)}", "viridis",
             1, float(np.nanmax(cnt)), f"10a_count_{tag}.png", od, "Count")

    pos = G["total_pos"].copy().astype(float); pos[pos == 0] = np.nan
    _mapplot(lo, la, pos, f"Positive Count — {_tag_label(tag)}", "YlOrRd",
             1, float(np.nanmax(pos)), f"10b_pos_{tag}.png", od, "Count")

    mp = G["mean_prob"].copy()
    _mapplot(lo, la, mp, f"Mean P(SDS=1) — {_tag_label(tag)}", "YlOrRd",
             0, float(np.nanpercentile(mp[np.isfinite(mp)], 99)),
             f"10c_prob_{tag}.png", od, "P(SDS=1)")

def plot_outcome_count_maps(G, od, tag):
    # per-cell TP / FP / FN counts on the map, using a perceptually uniform
    # log-friendly colorscale so sparse and dense regions both read clearly.
    print("  [10d] Outcome count maps (TP / FP / FN) ...")
    la, lo = G["lats"], G["lons"]

    # Each metric gets its own vmax so the color scale is not dominated by
    # whichever outcome has the single highest cell count. Using the 98th
    # percentile (ignoring zeros) keeps outlier cells from washing out the rest.
    def _vmax(arr):
        pos = arr[arr > 0]
        return float(np.percentile(pos, 98)) if len(pos) > 0 else 1.0

    # Colormaps chosen to match the outcome color palette used elsewhere
    # (OUTCOME_COLORS) while still giving a full luminance range:
    #   TP  → greens  (correct detections, "good")
    #   FP  → oranges (false alarms, "warning")
    #   FN  → reds    (misses, "bad")
    # Sequential perceptually-uniform cmaps so count differences are readable.
    specs = [
        ("TP", G["TP"], "YlGn",     "TP count"),
        ("FP", G["FP"], "YlOrBr",   "FP count"),
        ("FN", G["FN"], "OrRd",     "FN count"),
    ]

    for name, raw, cmap, label in specs:
        grid = raw.copy().astype(float)
        grid[grid == 0] = np.nan          # zeros → NaN so they render as background
        vm = _vmax(raw)
        _mapplot(lo, la, grid,
                 f"{name} Count — {_tag_label(tag)}",
                 cmap, 0, vm,
                 f"10d_{name.lower()}_count_{tag}.png", od, label)


# ---------------------------------------------------------------------------
# Temporal plots
# ---------------------------------------------------------------------------
def plot_temporal(time, oc, od, tag):
    # daily TP/FP/FN counts + monthly F1/precision/recall
    print("  [11] Temporal ...")
    df=pd.DataFrame({"time":pd.to_datetime(time),"outcome":oc})
    daily=df.groupby(["time","outcome"]).size().unstack(fill_value=0)
    for o in ["TP","FP","FN","TN"]:
        if o not in daily.columns: daily[o]=0
    fig,axes=plt.subplots(2,1,figsize=(16,9),sharex=True)
    for o in ["TP","FP","FN"]:
        axes[0].plot(daily.index,daily[o],color=OUTCOME_COLORS[o],label=o,lw=.9)
    axes[0].set_ylabel("Count"); axes[0].set_title(f"Daily TP / FP / FN",fontsize=SUBTITLE_SIZE)
    axes[0].legend(); axes[0].grid(alpha=.2)
    df["month"]=df["time"].dt.to_period("M")
    mo=df.groupby(["month","outcome"]).size().unstack(fill_value=0)
    for o in ["TP","FP","FN"]:
        if o not in mo.columns: mo[o]=0
    tp,fp,fn=mo["TP"].values,mo["FP"].values,mo["FN"].values
    f1m=safe_div(2*tp,2*tp+fp+fn); pm=safe_div(tp,tp+fp); rm=safe_div(tp,tp+fn)
    mc=mo.index.to_timestamp()
    axes[1].plot(mc,f1m,"k-o",ms=4,lw=2,label="F1")
    axes[1].plot(mc,pm,"--s",ms=3,lw=1.5,color=OUTCOME_COLORS["FP"],label="Precision")
    axes[1].plot(mc,rm,"--^",ms=3,lw=1.5,color=OUTCOME_COLORS["FN"],label="Recall")
    axes[1].set_ylim(0,1); axes[1].set_ylabel("Score")
    axes[1].set_title("Monthly Metrics",fontsize=SUBTITLE_SIZE); axes[1].legend(); axes[1].grid(alpha=.2)
    fig.suptitle(f"Temporal Analysis — {_tag_label(tag)}",fontsize=TITLE_SIZE, fontweight="bold")
    fig.savefig(od/f"11_temporal_{tag}.png"); plt.close(fig)

def plot_seasonal(time, oc, od, tag):
    # month-of-year breakdown (aggregated across all years)
    print("  [12] Seasonal ...")
    df=pd.DataFrame({"time":pd.to_datetime(time),"outcome":oc}); df["month"]=df["time"].dt.month
    bm=df.groupby(["month","outcome"]).size().unstack(fill_value=0)
    for o in ["TP","FP","FN"]:
        if o not in bm.columns: bm[o]=0
    fig,a1=plt.subplots(figsize=(12,5.5))
    bm[["TP","FP","FN"]].plot(kind="bar",ax=a1,color=[OUTCOME_COLORS[o] for o in ["TP","FP","FN"]],width=.7,alpha=OUTCOME_ALPHA)
    a1.set_xlabel("Month"); a1.set_ylabel("Count")
    a1.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"],rotation=0)
    a2=a1.twinx()
    ts,fs,fns=bm["TP"].values,bm["FP"].values,bm["FN"].values
    f1s=safe_div(2*ts,2*ts+fs+fns)
    a2.plot(range(len(f1s)),f1s,"k-o",ms=5,lw=2,label="F1",zorder=5)
    a2.set_ylim(0,1); a2.set_ylabel("F1")
    l1,lb1=a1.get_legend_handles_labels(); l2,lb2=a2.get_legend_handles_labels()
    a1.legend(l1+l2,lb1+lb2,loc="upper right")
    fig.suptitle(f"Seasonal Breakdown — {_tag_label(tag)}",fontsize=TITLE_SIZE, fontweight="bold")
    fig.savefig(od/f"12_seasonal_{tag}.png"); plt.close(fig)


# ---------------------------------------------------------------------------
# Source-type analysis: genesis (=1) vs intermediate (=2) pixels
# Only runs if the SDS source file has a source_type variable
# ---------------------------------------------------------------------------
def plot_stype_breakdown(y, yp, stype, od, tag):
    # counts of genesis/intermediate in the split + recall on each
    print("  [13] Source-type breakdown ...")
    pos_mask = y == 1
    is_gen   = pos_mask & (stype == 1)
    is_int   = pos_mask & (stype == 2)
    n_gen, n_int = int(is_gen.sum()), int(is_int.sum())
    # recall per stype = TP_stype / (TP_stype + FN_stype)
    rec_gen = float((yp[is_gen] == 1).mean()) if n_gen > 0 else np.nan
    rec_int = float((yp[is_int] == 1).mean()) if n_int > 0 else np.nan

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    # left: counts
    a1.bar([0, 1], [n_gen, n_int],
           color=[STYPE_COLORS[1], STYPE_COLORS[2]],
           edgecolor="white", alpha=0.85)
    a1.set_xticks([0, 1]); a1.set_xticklabels(["Genesis", "Intermediate"])
    a1.set_ylabel("Number of positive pixels")
    a1.set_title("Positive composition", fontsize=SUBTITLE_SIZE)
    for i, n in enumerate([n_gen, n_int]):
        a1.text(i, n, f"{n:,}", ha="center", va="bottom", fontsize=LABEL_SIZE - 1)

    # right: recall per stype
    a2.bar([0, 1], [rec_gen if not np.isnan(rec_gen) else 0,
                    rec_int if not np.isnan(rec_int) else 0],
           color=[STYPE_COLORS[1], STYPE_COLORS[2]],
           edgecolor="white", alpha=0.85)
    a2.set_xticks([0, 1]); a2.set_xticklabels(["Genesis", "Intermediate"])
    a2.set_ylabel("Recall"); a2.set_ylim(0, 1.05)
    a2.set_title("Recall by source type", fontsize=SUBTITLE_SIZE)
    for i, r in enumerate([rec_gen, rec_int]):
        a2.text(i, r if not np.isnan(r) else 0,
                f"{r:.3f}" if not np.isnan(r) else "n/a",
                ha="center", va="bottom", fontsize=LABEL_SIZE - 1)

    fig.suptitle(f"Source-Type Breakdown — {_tag_label(tag)}",
                 fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout()
    fig.savefig(od / f"13_stype_breakdown_{tag}.png"); plt.close(fig)

def plot_stype_fraction_map(lats, lons, y, stype, od, tag):
    # per-cell fraction of positives that are genesis (vs intermediate)
    # useful for seeing whether source composition varies regionally
    print("  [14] Source-type fraction map ...")
    ul, ulo = np.sort(np.unique(lats)), np.sort(np.unique(lons))
    nl, no = len(ul), len(ulo)
    li  = np.searchsorted(ul,  lats)
    loi = np.searchsorted(ulo, lons)

    gen_ct = np.zeros((nl, no))
    int_ct = np.zeros((nl, no))
    pos_mask = y == 1
    for k in np.where(pos_mask)[0]:
        if stype[k] == 1:
            gen_ct[li[k], loi[k]] += 1
        elif stype[k] == 2:
            int_ct[li[k], loi[k]] += 1

    total = gen_ct + int_ct
    frac_gen = np.where(total > 0, gen_ct / total, np.nan)
    _mapplot(ulo, ul, frac_gen,
             f"Fraction Genesis (of positives) — {_tag_label(tag)}",
             "PuOr_r", 0, 1,
             f"14_stype_fraction_{tag}.png", od,
             label="Genesis / (Genesis + Intermediate)")

def plot_kde_by_stype(X, y, stype, od, tag, name_map, feature_vars):
    # restricted to positives - contrasts meteorology of genesis vs intermediate pixels
    print("  [15] KDE by source type (overview + groups) ...")
    pos_mask = y == 1
    gen_idx = np.where(pos_mask & (stype == 1))[0]
    int_idx = np.where(pos_mask & (stype == 2))[0]
    if len(gen_idx) < 10 or len(int_idx) < 10:
        print("    too few genesis or intermediate samples, skipping KDE")
        return
    groups = {1: gen_idx, 2: int_idx}
    _kde_grid(X, groups, feature_vars, STYPE_COLORS, STYPE_LABELS,
              f"Features by Source Type — {_tag_label(tag)}",
              f"15_kde_stype_{tag}.png", od, name_map)
    for grp_title, members in get_active_groups(feature_vars):
        existing = [f for f in members if f in feature_vars]
        if existing:
            _kde_group(X, groups, existing, STYPE_COLORS, STYPE_LABELS,
                       grp_title, tag, od, name_map, "15g", feature_vars)

def _stype_count_grid(lats, lons, mask):
    # bins a boolean sample mask into a per-cell count grid on the full lat/lon axes
    ul, ulo = np.sort(np.unique(lats)), np.sort(np.unique(lons))
    nl, no = len(ul), len(ulo)
    grid = np.zeros((nl, no))
    if mask.any():
        li  = np.searchsorted(ul,  lats[mask])
        loi = np.searchsorted(ulo, lons[mask])
        np.add.at(grid, (li, loi), 1)
    return ul, ulo, grid

def plot_spatial_by_stype(lats, lons, y, yp, stype, od, tag):
    # per-stype spatial breakdown - recall, TP count, FN count for each source type
    # plus a single shared FP map (FPs don't have a source type, they're truth=0)
    print("  [16] Spatial maps split by source type ...")
    pos_mask = y == 1
    for st_val, st_label in STYPE_LABELS.items():
        tp_mask = pos_mask & (stype == st_val) & (yp == 1)
        fn_mask = pos_mask & (stype == st_val) & (yp == 0)

        ul, ulo, tp_grid = _stype_count_grid(lats, lons, tp_mask)
        _, _,   fn_grid  = _stype_count_grid(lats, lons, fn_mask)

        total = tp_grid + fn_grid
        recall = np.where(total > 0, tp_grid / total, np.nan)

        tp_plot = tp_grid.copy(); tp_plot[tp_plot == 0] = np.nan
        fn_plot = fn_grid.copy(); fn_plot[fn_plot == 0] = np.nan

        safe = st_label.lower()
        _mapplot(ulo, ul, recall,
                 f"Recall — {st_label} — {_tag_label(tag)}",
                 F1_CMAP, 0, 1,
                 f"16a_recall_{safe}_{tag}.png", od, "Recall")
        _mapplot(ulo, ul, tp_plot,
                 f"TP count — {st_label} — {_tag_label(tag)}",
                 "YlGn", 1, None,
                 f"16b_tp_{safe}_{tag}.png", od, "Count")
        _mapplot(ulo, ul, fn_plot,
                 f"FN count — {st_label} — {_tag_label(tag)}",
                 "YlOrRd", 1, None,
                 f"16c_fn_{safe}_{tag}.png", od, "Count")

    # shared FP map - FPs don't have a source type so this one is not split
    fp_mask = (y == 0) & (yp == 1)
    ul, ulo, fp_grid = _stype_count_grid(lats, lons, fp_mask)
    fp_plot = fp_grid.copy(); fp_plot[fp_plot == 0] = np.nan
    _mapplot(ulo, ul, fp_plot,
             f"FP count (shared, truth=0) — {_tag_label(tag)}",
             "YlOrRd", 1, None,
             f"16d_fp_shared_{tag}.png", od, "Count")

def plot_roc_recall_by_stype(y, proba, yp, stype, thr, od, tag):
    # two panels: (a) ROC curve per stype with shared negatives, (b) recall vs threshold per stype
    # shared negatives mean the FPR axis is comparable across stypes
    print("  [17] ROC + recall-vs-threshold by source type ...")
    neg_mask = (y == 0)
    has_gen = ((y == 1) & (stype == 1)).sum() > 0
    has_int = ((y == 1) & (stype == 2)).sum() > 0
    if not (has_gen and has_int):
        print("    need both genesis and intermediate positives, skipping")
        return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) ROC per stype: positives restricted to one stype, negatives shared
    for st_val, st_label in STYPE_LABELS.items():
        sub = neg_mask | ((y == 1) & (stype == st_val))
        y_sub = y[sub]
        p_sub = proba[sub]
        fpr, tpr, _ = roc_curve(y_sub, p_sub)
        auc = roc_auc_score(y_sub, p_sub)
        a1.plot(fpr, tpr, lw=2,
                color=STYPE_COLORS[st_val],
                label=f"{st_label} (AUC = {auc:.4f})")
    a1.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
    a1.set_xlabel("False Positive Rate (shared negatives)")
    a1.set_ylabel("True Positive Rate")
    a1.set_title("ROC by source type", fontsize=SUBTITLE_SIZE)
    a1.legend(); a1.grid(alpha=.2)

    # (b) recall vs threshold per stype - "how does sensitivity differ?"
    thresholds = np.linspace(0.01, 0.99, 200)
    for st_val, st_label in STYPE_LABELS.items():
        mask = (y == 1) & (stype == st_val)
        p_pos = proba[mask]
        recs = [(p_pos >= t).mean() for t in thresholds]
        a2.plot(thresholds, recs, lw=2,
                color=STYPE_COLORS[st_val], label=st_label)
    a2.axvline(thr, color="#999", lw=1.3, ls=":",
               label=f"Selected = {thr:.3f}")
    a2.set_xlabel("Threshold"); a2.set_ylabel("Recall"); a2.set_ylim(0, 1.05)
    a2.set_title("Recall vs threshold by source type", fontsize=SUBTITLE_SIZE)
    a2.legend(); a2.grid(alpha=.2)

    fig.suptitle(f"Source-Type Sensitivity Analysis — {_tag_label(tag)}",
                 fontsize=TITLE_SIZE, fontweight="bold")
    plt.tight_layout()
    fig.savefig(od / f"17_stype_roc_recall_{tag}.png"); plt.close(fig)


# ---------------------------------------------------------------------------
# Animations & daily frames
# ---------------------------------------------------------------------------

# shared grid helpers
def _anim_grid_prep(lat, lon, times):
    # builds the lat/lon/day lookups that the animation and per-day PNGs both need
    ulat, ulon = np.sort(np.unique(lat)), np.sort(np.unique(lon))
    nlat, nlon = len(ulat), len(ulon)
    li = np.searchsorted(ulat, lat)
    loi = np.searchsorted(ulon, lon)
    tpd = pd.to_datetime(times)
    udays = np.sort(np.unique(tpd.date))
    lo2, la2 = np.meshgrid(ulon, ulat)
    return dict(ulat=ulat, ulon=ulon, nlat=nlat, nlon=nlon,
                li=li, loi=loi, tpd=tpd, udays=udays, lo2=lo2, la2=la2)

def _anim_map_fig(title_text, legend_handles):
    # figure + cartopy axes with a side legend - used for both the animation and the daily frame PNGs
    lon_range = DOMAIN["east"] - DOMAIN["west"]
    lat_range = DOMAIN["north"] - DOMAIN["south"]
    fig_w = 14; fig_h = fig_w * (lat_range / lon_range) + 1.2
    fig = plt.figure(figsize=(fig_w, fig_h))

    ax = fig.add_axes(
        [0.04, 0.06, 0.80, 0.84],
        projection=ccrs.PlateCarree()
    )
    ax.set_extent([DOMAIN["west"], DOMAIN["east"],
                   DOMAIN["south"], DOMAIN["north"]],
                  crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND,  facecolor="white", zorder=0)
    ax.add_feature(cfeature.COASTLINE, lw=0.6, ec="black", zorder=5)
    ax.add_feature(cfeature.BORDERS, lw=0.3, ec="#444", ls="--", zorder=5)

    gl = ax.gridlines(draw_labels=True, lw=0.4, alpha=0.4, ls=":",
                      color="grey", crs=ccrs.PlateCarree())
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": TICK_SIZE}
    gl.ylabel_style = {"size": TICK_SIZE}

    # single combined title: base text + date, all on one line - date gets filled in per frame
    date_text = ax.set_title(title_text, fontsize=TITLE_SIZE,
                             fontweight="bold", pad=8)

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.855, 0.50),
        ncol=1,
        fontsize=LEGEND_SIZE,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="#ccc",
        borderpad=0.8,
        handletextpad=0.6,
        labelspacing=0.8,
    )

    return fig, ax, {"transform": ccrs.PlateCarree()}, date_text

def _save_animation(fig, update_fn, n_frames, od, basename, fps=6):
    ani = animation.FuncAnimation(fig, update_fn, frames=n_frames,
                                  interval=1000 // fps, blit=False)
    p = od / f"{basename}.mp4"
    ani.save(str(p), writer=animation.FFMpegWriter(fps=fps, bitrate=2500))
    plt.close(fig)

def _oc_grid_for_day(day, gp, oc_map):
    # builds one outcome grid (FP/FN/TP) for a single day
    # TNs (oc_map == -1) are filtered out so they stay NaN and render as background -
    # otherwise BoundaryNorm would clip -1 to the first colour (FP) and the whole map goes orange
    m = gp["tpd"].date == day
    grid = np.full((gp["nlat"], gp["nlon"]), np.nan)
    idx = np.where(m)[0]
    pos = idx[oc_map[idx] > 0]
    grid[gp["li"][pos], gp["loi"][pos]] = oc_map[pos]
    return grid

def _oc_cmap_norm_handles():
    # discrete colormap for FP/FN/TP + matching legend handles
    # uses the shared OUTCOME_COLORS (same palette as 04b plot). alpha is applied
    # at the pcolormesh level so the map colours match the histogram fill colours
    cmap = mcolors.ListedColormap([
        OUTCOME_COLORS["FP"],
        OUTCOME_COLORS["FN"],
        OUTCOME_COLORS["TP"],
    ])
    norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    # legend swatches mirror the on-map appearance (colour + alpha)
    handles = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=c, alpha=OUTCOME_ALPHA,
               markersize=12, label=l)
        for l, c in [("FP", OUTCOME_COLORS["FP"]),
                     ("FN", OUTCOME_COLORS["FN"]),
                     ("TP", OUTCOME_COLORS["TP"])]
    ]
    return cmap, norm, handles

def make_animation(lat, lon, y, oc, times, od, tag, fps=6):
    # mp4/gif animation cycling through each day
    print(f"  [Anim] Building outcome animation for {_tag_label(tag)} ...")
    gp = _anim_grid_prep(lat, lon, times)
    oc_int = {"FP": 1, "FN": 2, "TP": 3}
    oc_map = np.array([oc_int.get(o, -1) for o in oc])
    cmap, norm, handles = _oc_cmap_norm_handles()
    base_title = f"Daily SDS Predictions — {_tag_label(tag)}"
    fig, ax, kw, date_text = _anim_map_fig(base_title, handles)
    mesh = ax.pcolormesh(gp["lo2"], gp["la2"],
                         np.full((gp["nlat"], gp["nlon"]), np.nan),
                         cmap=cmap, norm=norm, shading="auto",
                         alpha=OUTCOME_ALPHA, zorder=1, **kw)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    def update(fi):
        grid = _oc_grid_for_day(gp["udays"][fi], gp, oc_map)
        mesh.set_array(grid.ravel())
        date_text.set_text(f"{base_title}  —  {gp['udays'][fi]}")
        return mesh, date_text

    _save_animation(fig, update, len(gp["udays"]), od, f"animation_outcome_{tag}", fps)

def save_daily_frames(lat, lon, oc, times, od, tag):
    # one PNG per day - for picking out specific dates to build combination plot 
    gp = _anim_grid_prep(lat, lon, times)
    oc_int = {"FP": 1, "FN": 2, "TP": 3}
    oc_map = np.array([oc_int.get(o, -1) for o in oc])
    cmap, norm, handles = _oc_cmap_norm_handles()
    oc_dir = od / f"frames_outcome_{tag}"
    oc_dir.mkdir(parents=True, exist_ok=True)
    n_days = len(gp["udays"])
    base_title = f"Daily SDS Predictions — {_tag_label(tag)}"
    print(f"  [Frames] Saving {n_days} daily frames for {_tag_label(tag)} ...")
    for fi, day in enumerate(gp["udays"]):
        day_str = str(day)
        grid = _oc_grid_for_day(day, gp, oc_map)
        fig, ax, kw, dt = _anim_map_fig(base_title, handles)
        ax.pcolormesh(gp["lo2"], gp["la2"], grid, cmap=cmap,
                      norm=norm, shading="auto",
                      alpha=OUTCOME_ALPHA, zorder=1, **kw)
        dt.set_text(f"{base_title}  —  {day_str}")
        metrics = _day_metrics(day, times, oc)
        if np.isnan(metrics["f1"]):
            mline = f"F1=n/a  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}"
        else:
            mline = (f"F1={metrics['f1']:.3f}  "
                     f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        # place metrics AFTER tight_layout so figure coords are stable.
        # y=0.935 sits in the reserved top strip, just below the bold title (~0.955)
        # and just above the top edge of the map axes (~0.92).
        fig.text(0.5, 0.935, mline,
                 ha="center", va="top",
                 fontsize=SUBTITLE_SIZE, fontweight="normal", color="#333333")
        fig.savefig(oc_dir / f"{day_str}.png"); plt.close(fig)
        if (fi + 1) % 50 == 0 or fi == n_days - 1:
            print(f"    {fi + 1}/{n_days} frames done")
    print(f"    → {oc_dir}  ({n_days} files)")


def _day_metrics(day, times, oc):
    # F1/precision/recall + raw counts for a single day, using the same per-sample
    # outcome labels that the map rendering uses. Returns a dict with floats (NaN
    # where the metric is undefined) and ints for the counts. The caller decides
    # how to display NaN (typically "n/a").
    #
    # conventions (match _build_grids):
    #   - TP+FN == 0 and FP == 0  → F1/P/R all NaN (nothing happened, metric undefined)
    #   - TP+FN == 0 and FP >  0  → F1=0, P=0, R=NaN (model fired, no ground truth)
    #   - otherwise               → standard formulae
    tpd = pd.to_datetime(times)
    m = tpd.date == day
    if not m.any():
        return dict(f1=np.nan, precision=np.nan, recall=np.nan,
                    tp=0, fp=0, fn=0, tn=0)
    sub = oc[m]
    tp = int((sub == "TP").sum())
    fp = int((sub == "FP").sum())
    fn = int((sub == "FN").sum())
    tn = int((sub == "TN").sum())

    if tp + fn == 0 and fp == 0:
        f1 = precision = recall = np.nan
    elif tp + fn == 0 and fp > 0:
        precision = 0.0
        recall = np.nan
        f1 = 0.0
    else:
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return dict(f1=f1, precision=precision, recall=recall,
                tp=tp, fp=fp, fn=fn, tn=tn)


def plot_composite_frames(lat, lon, oc, times, dates, od, tag,
                          composite_title="SDS Prediction examples"):
    # 2×3 grid of daily frames for hand-picked dates (passed as YYYY-MM-DD strings).
    # Shares the per-day rendering logic with save_daily_frames/make_animation so the
    # colours and outcome mapping stay identical; the difference is only layout and
    # chrome (one figure title, one shared bottom legend, per-panel date subtitles).
    #
    # The function tolerates dates that aren't present in the split - those panels
    # get a grey "no data" message instead of failing the whole plot, which makes it
    # safe to run from both run_split calls (val + test) even when the dates only
    # line up with one of them.
    if not dates:
        return
    print(f"  [Composite] Building composite frame plot for {_tag_label(tag)} ...")

    # parse user-supplied strings once. pd.to_datetime will raise on malformed input
    # before any plotting happens, which is what we want (fail loudly at parse time).
    try:
        parsed = [pd.to_datetime(d).date() for d in dates]
    except Exception as e:
        print(f"    ERROR: could not parse dates {dates}: {e}")
        return

    # filter to the dates that actually exist in this split. a date absent from the
    # split isn't necessarily a bug - it might belong to the other split - so we warn
    # and continue with an empty panel rather than aborting.
    gp = _anim_grid_prep(lat, lon, times)
    available = set(gp["udays"].tolist())
    matched = [d for d in parsed if d in available]
    missing = [d for d in parsed if d not in available]
    if not matched:
        print(f"    no requested dates fall in {_tag_label(tag)} "
              f"(requested: {', '.join(str(d) for d in parsed)}) - skipping")
        return
    if missing:
        print(f"    note: {len(missing)} date(s) not in this split, leaving those "
              f"panels blank: {', '.join(str(d) for d in missing)}")

    oc_int = {"FP": 1, "FN": 2, "TP": 3}
    oc_map = np.array([oc_int.get(o, -1) for o in oc])
    cmap, norm, handles = _oc_cmap_norm_handles()

    # pin the layout to exactly 2×3 regardless of how many dates were supplied - if
    # the caller gives fewer than 6, the remaining slots render blank with a note;
    # more than 6 get truncated (with a warning) so the figure shape stays predictable.
    nr, nc = 2, 3
    n_slots = nr * nc
    if len(parsed) > n_slots:
        print(f"    warning: {len(parsed)} dates supplied, only first {n_slots} will be shown")
        parsed = parsed[:n_slots]

    # figure geometry: each panel keeps the single-frame aspect ratio (lat/lon of
    # DOMAIN). We size based on lon_range so the overall figure stays readable, then
    # reserve headroom for suptitle and footer for the legend.
    lon_range = DOMAIN["east"] - DOMAIN["west"]
    lat_range = DOMAIN["north"] - DOMAIN["south"]
    panel_w = 5.0                              # inches per panel (width)
    panel_h = panel_w * (lat_range / lon_range)
    fig_w = panel_w * nc
    fig_h = panel_h * nr + 1.9                  # +1.9" padding for suptitle + two-line subtitles + legend

    fig = plt.figure(figsize=(fig_w, fig_h))

    # padding in figure fraction. bottom_pad leaves room for the single-row legend;
    # top_pad gives the suptitle breathing room without cramping the top row of panels.
    left_pad   = 0.035
    right_pad  = 0.035
    bottom_pad = 0.09
    top_pad    = 0.88  # top of panel area in fig-fraction (below suptitle)

    grid_w = 1.0 - left_pad - right_pad
    grid_h = top_pad - bottom_pad
    cell_w = grid_w / nc
    cell_h = grid_h / nr
    # shrink each panel slightly so neighbours don't touch; the gap also gives the
    # gridline tick labels + the two-line subtitle (date + F1 metrics) somewhere to
    # live without overlapping the next panel.
    inner_w = cell_w * 0.92
    inner_h = cell_h * 0.78
    dx = (cell_w - inner_w) / 2
    dy = (cell_h - inner_h) / 2

    for i in range(n_slots):
        row, col = divmod(i, nc)
        # row 0 sits at the TOP of the figure, so its y-origin is the higher value
        x0 = left_pad + col * cell_w + dx
        y0 = bottom_pad + (nr - 1 - row) * cell_h + dy

        ax = fig.add_axes([x0, y0, inner_w, inner_h],
                          projection=ccrs.PlateCarree())
        ax.set_extent([DOMAIN["west"], DOMAIN["east"],
                       DOMAIN["south"], DOMAIN["north"]],
                      crs=ccrs.PlateCarree())
        ax.set_facecolor("white")
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        ax.add_feature(cfeature.LAND,  facecolor="white", zorder=0)
        ax.add_feature(cfeature.COASTLINE, lw=0.5, ec="black", zorder=5)
        ax.add_feature(cfeature.BORDERS, lw=0.25, ec="#444", ls="--", zorder=5)

        gl = ax.gridlines(draw_labels=True, lw=0.3, alpha=0.35, ls=":",
                          color="grey", crs=ccrs.PlateCarree())
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {"size": TICK_SIZE - 1}
        gl.ylabel_style = {"size": TICK_SIZE - 1}

        if i >= len(parsed):
            # unused slot - mark it visually so it's clear the panel wasn't forgotten
            ax.text(0.5, 0.5, "(no date given)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=TICK_SIZE, color="#888888")
            ax.set_title(" ", fontsize=SUBPLOT_TITLE_SIZE)  # keeps row heights aligned
            continue

        day = parsed[i]
        if day not in available:
            ax.text(0.5, 0.5, "no data for this date",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=TICK_SIZE, color="#888888")
            ax.set_title(str(day), fontsize=SUBPLOT_TITLE_SIZE, pad=4)
        else:
            grid = _oc_grid_for_day(day, gp, oc_map)
            ax.pcolormesh(gp["lo2"], gp["la2"], grid, cmap=cmap, norm=norm,
                          shading="auto", alpha=OUTCOME_ALPHA, zorder=1,
                          transform=ccrs.PlateCarree())
            # two-line title: date + per-day metrics. Using a single title with
            # an embedded \n keeps the panel's plotting area untouched - matplotlib
            # grows the title box upwards rather than eating into the axes. "n/a"
            # fallback mirrors _day_metrics' NaN convention for days with no
            # positives and no FPs (F1 is undefined there).
            metrics = _day_metrics(day, times, oc)
            if np.isnan(metrics["f1"]):
                mline = f"F1=n/a  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}"
            else:
                mline = (f"F1={metrics['f1']:.3f}  "
                         f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
            ax.set_title(f"{day}\n{mline}",
                         fontsize=SUBPLOT_TITLE_SIZE, pad=4)

    # suptitle + single-row legend. the legend sits just above the bottom edge in
    # figure coords, and the bottom_pad above gives it the vertical space it needs.
    fig.suptitle(f"{composite_title} — {_tag_label(tag)}",
                 fontsize=TITLE_SIZE, fontweight="bold",
                 y=0.965)
    fig.legend(handles=handles,
               loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.015),
               frameon=True, fancybox=True, framealpha=0.9,
               edgecolor="#ccc", fontsize=LEGEND_SIZE,
               handletextpad=0.6, columnspacing=2.0)

    out = od / f"composite_frames_{tag}.png"
    fig.savefig(out); plt.close(fig)
    print(f"    → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_split(tag, cache, split_years, ds_sds, sds_var, land_indices,
              rf, platt_lr, thr, feat_df, feature_vars, od, spatial_res,
              model_dir, no_animation=False, frames_only=False,
              composite_only=False, composite_dates=None):
    # runs the full plotting pipeline for one split (val or test)
    print(f"\n{'='*60}\nAnalysing: {_tag_label(tag)}\n{'='*60}")
    years = list(range(split_years[0], split_years[1] + 1))

    # if available loads saved predictions
    pred_file = model_dir / f"{tag}_predictions.npz"
    predictions_loaded = False

    if pred_file.exists():
        print(f"  Loading saved predictions from {pred_file} ...")
        saved = np.load(pred_file)
        y     = saved["y"].astype(np.float32)
        proba = saved["proba"].astype(np.float32)
        predictions_loaded = True
        print(f"  Loaded {len(y):,} samples ({int(y.sum()):,} pos)")

    # reconstruct coordinates (needed for spatial/temporal plots)
    all_pos, all_neg = collect_indices(
        cache, years, ds_sds, sds_var, land_indices,
        logging.getLogger("analyze"))
    all_indices = all_pos + all_neg

    # pull lat/lon/time from each year's feature netcdf
    year_coords = {}
    for year in sorted(set(idx[0] for idx in all_indices)):
        ds = xr.open_dataset(cache.feat_dir / f"features_{year}_{cache.ftag}.nc")
        lat_v  = ds["lat"].values  if "lat"  in ds else ds["latitude"].values
        lon_v  = ds["lon"].values  if "lon"  in ds else ds["longitude"].values
        time_v = ds["valid_time"].values
        ds.close()
        year_coords[year] = (lat_v, lon_v, time_v)

    lats  = np.empty(len(all_indices), dtype=np.float32)
    lons  = np.empty(len(all_indices), dtype=np.float32)
    times = np.empty(len(all_indices), dtype="datetime64[ns]")
    for i, (year, idx) in enumerate(all_indices):
        lv, lov, tv = year_coords[year]
        lats[i] = lv[idx]; lons[i] = lov[idx]; times[i] = tv[idx]

    # redo predictions if no saved ones found
    if not predictions_loaded:
        print("  No saved predictions found — running inference ...")
        X_inf = extract_from_cache(cache, all_indices)
        y = np.concatenate([
            np.ones(len(all_pos),  dtype=np.float32),
            np.zeros(len(all_neg), dtype=np.float32),
        ])
        proba_raw = rf.predict_proba(X_inf)[:, 1]
        proba = platt_apply(platt_lr, proba_raw) if platt_lr is not None \
                else proba_raw
        del X_inf, proba_raw
        gc.collect()

    print(f"  Samples: {len(y):,} ({int(y.sum()):,} pos)")
    yp = (proba >= thr).astype(np.int32)
    oc = outcome_labels(y, yp)
    cm = confusion_matrix(y, yp); tn, fp, fn, tp = cm.ravel()
    print(f"  TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}")
    print(f"  F1={f1_score(y,yp):.4f} P={precision_score(y,yp):.4f} R={recall_score(y,yp):.4f}")

    # source-type lookup (1=genesis, 2=intermediate, 0=neither)
    # Skipped in composite-only mode since the composite plot doesn't use stype,
    # and re-reading the SDS source dataset year-by-year is slow.
    if not composite_only:
        stype_by_year = {
            year: make_stype_for_year(year, ds_sds,
                                      cache.unique_times[year], land_indices)
            for year in sorted(set(idx[0] for idx in all_indices))
        }
        stype = np.empty(len(all_indices), dtype=np.uint8)
        for i, (year, idx) in enumerate(all_indices):
            stype[i] = stype_by_year[year][idx]
        del stype_by_year
        n_gen = int(((y == 1) & (stype == 1)).sum())
        n_int = int(((y == 1) & (stype == 2)).sum())
        print(f"  Positives: {n_gen:,} genesis + {n_int:,} intermediate")
    else:
        stype = None  # downstream del handles None safely via the guard below

    # get long (nice) name from short name
    name_map = get_feature_name_map(feature_vars)

    if not frames_only and not composite_only:
        # load full data for feature-based plots (skipped in frames-only mode since it's slow)
        X = extract_from_cache(cache, all_indices)

        plot_roc_pr(y,proba,thr,od,tag)
        plot_confusion(y,yp,od,tag)
        plot_prob_hist(proba,y,thr,od,tag)
        plot_outcome_prob_hist(proba,oc,thr,od,tag)
        plot_calibration(y,proba,od,tag)
        plot_threshold_tradeoff(y,proba,thr,od,tag)
        plot_feat_imp(feat_df,od,tag,name_map)
        plot_kde_by_class(X,y,od,tag,name_map,feature_vars)
        plot_kde_by_outcome(X,oc,od,tag,name_map,feature_vars)
        plot_kde_by_stype(X, y, stype, od, tag, name_map, feature_vars)
        del X; gc.collect()

        G = _build_grids(lats,lons,y,proba,oc)
        print(f"  G sanity: total sum={G['total'].sum():.0f}, "
            f"TP sum={G['TP'].sum():.0f}, FP sum={G['FP'].sum():.0f}, "
            f"FN sum={G['FN'].sum():.0f}, cells with data={G['has_data'].sum()}")
        plot_spatial(G,od,tag)
        plot_count(G,od,tag)
        plot_outcome_count_maps(G, od, tag) 
        plot_temporal(times,oc,od,tag)
        plot_seasonal(times,oc,od,tag)

        plot_stype_breakdown(y, yp, stype, od, tag)
        plot_stype_fraction_map(lats, lons, y, stype, od, tag)
        plot_spatial_by_stype(lats, lons, y, yp, stype, od, tag)
        plot_roc_recall_by_stype(y, proba, yp, stype, thr, od, tag)

    if not no_animation and not composite_only:
        make_animation(lats, lons, y, oc, times, od, tag)
        save_daily_frames(lats, lons, oc, times, od, tag)

    # composite plot is cheap and useful even in frames_only / no_animation modes,
    # so it runs independently of the animation/frames toggles
    if composite_dates:
        plot_composite_frames(lats, lons, oc, times, composite_dates, od, tag)

    del y, yp, proba, oc, lats, lons, times
    if stype is not None:
        del stype
    gc.collect()

def main():
    parser=argparse.ArgumentParser(description="RF analysis")
    parser.add_argument("--model_dir",type=str,required=True)
    parser.add_argument("--sds_path",type=str,default=None)
    parser.add_argument("--resolution",default=None)
    parser.add_argument("--spatial",type=float,default=None)
    parser.add_argument("--val_years",nargs=2,type=int,default=None)
    parser.add_argument("--test_years",nargs=2,type=int,default=None)
    parser.add_argument("--no_animation",action="store_true") #can be skipped if not needed, they take quite long
    parser.add_argument("--frames_only",action="store_true",
        help="skip every analysis plot and only produce the animation + daily frames")
    parser.add_argument("--composite_only", action="store_true",
        help="skip everything except the composite frame plot. Requires --composite_dates.")
    parser.add_argument("--composite_dates", nargs="+", default=None,
        help="up to 6 dates (YYYY-MM-DD) to render as a 2x3 composite frame plot. "
             "Dates not in the current split are skipped with a warning, so it's "
             "fine to pass dates that belong to just val or just test.")
    parser.add_argument("--out_dir",type=str,default=None)
    args=parser.parse_args()
    # --composite_only is mutually exclusive with the other "only"/"no" flags, and
    # needs dates to be meaningful (otherwise the whole run produces nothing).
    only_flags = sum([args.frames_only, args.no_animation, args.composite_only])
    if only_flags > 1:
        print("ERROR: --frames_only, --no_animation, and --composite_only are mutually exclusive")
        sys.exit(1)
    if args.composite_only and not args.composite_dates:
        print("ERROR: --composite_only requires --composite_dates")
        sys.exit(1)
    model_dir=Path(args.model_dir)

    if not model_dir.exists(): print(f"ERROR: {model_dir}"); sys.exit(1)
    # read model
    rf=joblib.load(model_dir/"rf_model.joblib")
    pp=model_dir/"platt_scaler.joblib"
    platt_lr=joblib.load(pp) if pp.exists() else None
    results=json.loads((model_dir/"results.json").read_text())
    feat_df=pd.read_csv(model_dir/"feature_importances.csv")
    thr=results["threshold"]; feature_vars=results["feature_vars"]; dc=results["data_config"]

    # CLI args override, otherwise fall back to the values saved in results.json, otherwise to the config defaults
    resolution=args.resolution or dc.get("resolution",RESOLUTION)
    spatial_res=args.spatial or dc.get("spatial_res",SPATIAL_RES)
    val_years=args.val_years or dc.get("val_years",VAL_YEARS)
    test_years=args.test_years or dc.get("test_years",TEST_YEARS)
    sds_path=args.sds_path or dc.get("sds_path") or SDS_PATH

    _YEAR_RANGES["val"]  = tuple(val_years)
    _YEAR_RANGES["test"] = tuple(test_years)

    if sds_path is None: print("ERROR: No SDS path"); sys.exit(1)

    sds_file=Path(sds_path)
    od=Path(args.out_dir) if args.out_dir else model_dir/"analysis"
    od.mkdir(parents=True,exist_ok=True)

    logging.basicConfig(level=logging.INFO,format="%(message)s")

    print("="*60); print(f"RF ANALYSIS — {model_dir}"); print(f"  Threshold: {thr:.4f}")
    print(f"  Output: {od}"); print("="*60)

    cache=FeatureCache(resolution=resolution,spatial_res=spatial_res,
                       val_years=val_years,test_years=test_years,logger=logging.getLogger("analyze"))

    ds_sds,sds_var=load_sds_source(sds_file,logging.getLogger("analyze"))
    land_indices=compute_land_indices(ds_sds["latitude"].values,ds_sds["longitude"].values)

    for tag,yrs in [("val",val_years),("test",test_years)]:
        run_split(tag,cache,yrs,ds_sds,sds_var,land_indices,rf,platt_lr,thr,feat_df,feature_vars,od,spatial_res,
                  model_dir=model_dir,no_animation=args.no_animation,
                  frames_only=args.frames_only,
                  composite_only=args.composite_only,
                  composite_dates=args.composite_dates)
    ds_sds.close()

    print(f"\n{'='*60}\nANALYSIS COMPLETE — {od}\n{'='*60}")

if __name__=="__main__": main()