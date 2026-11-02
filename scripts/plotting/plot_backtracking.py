"""
Overview figures of the backtracking output (step 1 + step 2).

Outputs (in figures/backtracking/ by default):
  source_density_genesis.png        genesis pixels that passed the emission filter
  source_density_intermediate.png   intermediate source pixels
  source_density_combined.png       genesis + intermediate
  source_density_side_by_side.png   genesis | intermediate
  source_meteorology.png            conditions at genesis vs intermediate source regions
  source_meteorology_regions.csv    per-region values behind source_meteorology.png
  trajectories_all.png              all plume trajectories, coloured by duration
  source_counts_summary.txt         genesis / intermediate sources per trajectory

A source region is the set of pixels of one source type that belong to one
plume at one timestep. Genesis and intermediate regions are both defined from
sds_sources_filtered.nc (source_type 1 / 2) and the step 1 segmentation, so the
two distributions in source_meteorology.png are directly comparable.

Defaults to the backtracking run the model was trained on (SDS_PATH in config.py).

Usage:
  python plot_backtracking.py
  python plot_backtracking.py --only maps counts
  python plot_backtracking.py --only meteorology --replot   # reuse the saved CSV
  python plot_backtracking.py --s2_dir <step 1 dir>/<step 2 folder> --input <BACKTRACKING_INPUT nc>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from scipy.stats import gaussian_kde
from tqdm import tqdm

from sds.config import FIGURES_PATH
from sds.paths import project_relative
from sds.log_utils import header, section, done
from sds.plot_utils import (
    S2_DIR, BACKTRACKING_INPUT, PROJ,
    TITLE_SIZE, SUBTITLE_SIZE, SUBPLOT_TITLE_SIZE, TICK_SIZE, LEGEND_SIZE,
    VAR_SUBPLOT_W, VAR_SUBPLOT_H, STYPE_COLORS, STYPE_LABELS,
    DENSITY_CMAP, MAP_ALPHA,
    map_figsize, map_axes, add_colorbar, save_figure,
)

PLOTS = ["maps", "meteorology", "trajectories", "counts"]

# (column, panel title, log x-axis)
MET_VARS = [
    ("duaod",    "Dust AOD (550 nm)",               False),
    ("ae",       "Ångström exponent (469/865 nm)",  False),
    ("pm10",     "PM10 (kg m⁻³)",                   True),
    ("ws_mean",  "Mean wind speed (m s⁻¹)",         False),
    ("ws_max",   "Max wind speed (m s⁻¹)",          False),
    ("n_pixels", "Region size (pixels)",            True),
]


def _time_dim(ds):
    return "valid_time" if "valid_time" in ds.dims else "time"


# ---------------------------------------------------------------------------
# Source density maps
# ---------------------------------------------------------------------------
def _density_mesh(ax, lon, lat, counts):
    data = np.where(counts > 0, counts, np.nan)
    return ax.pcolormesh(lon, lat, data, cmap=DENSITY_CMAP, shading="nearest",
                         alpha=MAP_ALPHA, zorder=1, transform=PROJ)


def plot_source_density(source_type, lat, lon, od):
    section("Source pixel density")
    gen = (source_type == 1).sum(axis=0)
    itm = (source_type == 2).sum(axis=0)

    for counts, title, name in [
        (gen,       "Genesis Source Pixels",      "source_density_genesis"),
        (itm,       "Intermediate Source Pixels", "source_density_intermediate"),
        (gen + itm, "All Source Pixels",          "source_density_combined"),
    ]:
        fig = plt.figure(figsize=map_figsize())
        ax = map_axes(fig)
        im = _density_mesh(ax, lon, lat, counts)
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=10)
        add_colorbar(fig, ax, im, "Pixel count")
        save_figure(fig, od, name)

    fig = plt.figure(figsize=map_figsize(ncols=2, width=2 * 9))
    axes = [map_axes(fig, 121), map_axes(fig, 122)]
    fig.subplots_adjust(wspace=0.25)
    meshes = []
    for ax, counts, title in [(axes[0], gen, "(a) Genesis"),
                              (axes[1], itm, "(b) Intermediate")]:
        meshes.append(_density_mesh(ax, lon, lat, counts))
        ax.set_title(title, fontsize=SUBTITLE_SIZE)
    for ax, im in zip(axes, meshes):
        add_colorbar(fig, ax, im, "Pixel count")
    # anchor the suptitle to the (letterboxed) map frames, not the figure top
    fig.suptitle("Source Pixel Density", fontsize=TITLE_SIZE, fontweight="bold",
                 y=axes[0].get_position().y1 + 0.12)
    save_figure(fig, od, "source_density_side_by_side")


# ---------------------------------------------------------------------------
# Meteorology at source regions
# ---------------------------------------------------------------------------
def source_region_stats(source_type, src_times, seg_path, input_path):
    """Mean conditions per source region (one row per plume, timestep, source type)."""
    seg_ds = xr.open_dataset(seg_path)
    seg = seg_ds[list(seg_ds.data_vars)[0]]
    inp = xr.open_dataset(input_path)
    tdim = _time_dim(inp)

    seg_idx = pd.Index(pd.to_datetime(seg_ds["time"].values)).get_indexer(src_times)
    inp_idx = pd.Index(pd.to_datetime(inp[tdim].values)).get_indexer(src_times)
    if (seg_idx < 0).any() or (inp_idx < 0).any():
        raise ValueError("Source file times are not all present in the segmentation / input file")

    active = np.where((source_type > 0).any(axis=(1, 2)))[0]
    frames = []
    n_unassigned = 0
    for ti in tqdm(active, desc="  Source regions"):
        st = source_type[ti]
        seg_t = seg.isel(time=seg_idx[ti]).values
        sl = {tdim: inp_idx[ti]}
        aod469 = inp["aod469"].isel(sl).values
        aod865 = inp["aod865"].isel(sl).values
        with np.errstate(divide="ignore", invalid="ignore"):
            ae = -np.log(aod469 / aod865) / np.log(469.0 / 865.0)
        fields = {
            "duaod":   inp["duaod550"].isel(sl).values,
            "ae":      ae,
            "pm10":    inp["pm10"].isel(sl).values,
            "ws_mean": inp["ws_mean"].isel(sl).values,
            "ws_max":  inp["ws_max"].isel(sl).values,
        }
        for stype in (1, 2):
            m = (st == stype) & (seg_t > 0)
            n_unassigned += int(((st == stype) & (seg_t <= 0)).sum())
            if not m.any():
                continue
            cells, inv = np.unique(seg_t[m], return_inverse=True)
            df = pd.DataFrame({
                "source_type": stype, "cell": cells, "time": src_times[ti],
                "n_pixels": np.bincount(inv, minlength=len(cells)),
            })
            for name, field in fields.items():
                vals = field[m]
                ok = np.isfinite(vals)
                s = np.bincount(inv[ok], weights=vals[ok], minlength=len(cells))
                c = np.bincount(inv[ok], minlength=len(cells))
                with np.errstate(invalid="ignore"):
                    df[name] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
            frames.append(df)

    seg_ds.close()
    inp.close()
    if n_unassigned:
        print(f"  WARNING: {n_unassigned:,} source pixels outside any plume were skipped")
    return pd.concat(frames, ignore_index=True)


def plot_source_meteorology(regions, od):
    fig, axes = plt.subplots(2, 3, figsize=(3 * VAR_SUBPLOT_W, 2 * VAR_SUBPLOT_H))
    for ax, (col, title, log) in zip(axes.flat, MET_VARS):
        per_type = {}
        for stype in (1, 2):
            v = regions.loc[regions["source_type"] == stype, col].to_numpy(float)
            v = v[np.isfinite(v) & (v > 0)] if log else v[np.isfinite(v)]
            per_type[stype] = np.log10(v) if log else v
        both = np.concatenate(list(per_type.values()))
        xs = np.linspace(*np.percentile(both, [1, 99]), 300)
        for stype, v in per_type.items():
            if len(v) < 2:
                continue
            dens = gaussian_kde(v, bw_method="scott")(xs)
            x_plot = 10 ** xs if log else xs
            ax.plot(x_plot, dens, color=STYPE_COLORS[stype], lw=1.8)
            ax.fill_between(x_plot, dens, alpha=0.15, color=STYPE_COLORS[stype])
            med = np.median(v)
            ax.axvline(10 ** med if log else med, color=STYPE_COLORS[stype],
                       lw=1.2, ls="--")
        if log:
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, fontsize=SUBPLOT_TITLE_SIZE)
        ax.set_yticks([])
        ax.set_ylabel("Density", fontsize=TICK_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(alpha=0.2)

    n = regions["source_type"].value_counts()
    handles = [Line2D([0], [0], color=STYPE_COLORS[s], lw=2,
                      label=f"{STYPE_LABELS[s]} (n = {n.get(s, 0):,} regions)")
               for s in (1, 2)]
    handles.append(Line2D([0], [0], color="#555", lw=1.2, ls="--", label="Median"))

    fig_h = fig.get_size_inches()[1]
    top_chrome = min(0.10, 0.55 / fig_h)
    bottom_chrome = min(0.10, 0.45 / fig_h)
    plt.tight_layout(rect=[0, bottom_chrome, 1, 1 - top_chrome])
    fig.legend(handles=handles, loc="center", ncol=3, fontsize=LEGEND_SIZE,
               frameon=True, bbox_to_anchor=(0.5, bottom_chrome * 0.5))
    fig.suptitle("Conditions at Genesis vs Intermediate Source Regions",
                 fontsize=TITLE_SIZE, fontweight="bold", y=1 - top_chrome * 0.25)
    save_figure(fig, od, "source_meteorology")


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------
def plot_all_trajectories(features, od):
    section("All trajectories")
    cent = (features.groupby(["cell", "time"])[["longitude", "latitude"]]
            .mean().reset_index().sort_values(["cell", "time"]))
    span = features.groupby("cell")["time"].agg(["min", "max"])
    duration = (span["max"] - span["min"]).dt.total_seconds() / 3600
    norm = mcolors.LogNorm(vmin=max(duration.min(), 1), vmax=max(duration.max(), 1))
    cmap = plt.cm.viridis
    features = features.assign(dur=features["cell"].map(duration).clip(lower=1))

    # draw short tracks first so long ones stay visible on top
    segs, durs = [], []
    for cell, g in cent.groupby("cell", sort=False):
        if len(g) > 1:
            segs.append(g[["longitude", "latitude"]].to_numpy())
            durs.append(max(duration[cell], 1))
    order = np.argsort(durs)
    segs = [segs[i] for i in order]
    durs = np.asarray(durs)[order]

    fig = plt.figure(figsize=map_figsize())
    ax = map_axes(fig)
    f = features.sort_values("dur")
    ax.scatter(f["longitude"], f["latitude"], c=f["dur"], cmap=cmap, norm=norm,
               s=1, alpha=0.2, edgecolors="none", transform=PROJ, zorder=2)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=0.4, alpha=0.4,
                        transform=PROJ, zorder=3)
    lc.set_array(durs)
    ax.add_collection(lc)
    ax.set_title(f"Plume Trajectories (n = {features['cell'].nunique():,})",
                 fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    add_colorbar(fig, ax, sm, "Track duration (hours)")
    save_figure(fig, od, "trajectories_all")


# ---------------------------------------------------------------------------
# Source counts per trajectory
# ---------------------------------------------------------------------------
def write_source_counts(summary_df, interm_df, lineage_df, od):
    section("Source counts per trajectory")
    res = {}
    if len(lineage_df):
        g = lineage_df.groupby("cell_id")["genesis_cell"].nunique()
        res.update(avg_genesis_per_traj=g.mean(), median_genesis_per_traj=g.median(),
                   max_genesis_per_traj=g.max())
    else:
        res.update(avg_genesis_per_traj=1.0, median_genesis_per_traj=1.0,
                   max_genesis_per_traj=1.0)

    all_trajs = summary_df["cell"].astype(int).unique()
    per_traj = (interm_df.groupby("cell_id").size()
                .reindex(all_trajs, fill_value=0))
    res.update(avg_intermediate_per_traj=per_traj.mean(),
               median_intermediate_per_traj=per_traj.median(),
               max_intermediate_per_traj=per_traj.max(),
               frac_traj_with_intermediate=(per_traj > 0).mean())

    od.mkdir(parents=True, exist_ok=True)
    out = od / "source_counts_summary.txt"
    with open(out, "w") as f:
        f.write("Source Counts Per Trajectory Summary\n")
        f.write("=" * 40 + "\n\n")
        for key, value in res.items():
            f.write(f"{key}: {value:.2%}\n" if "frac" in key else f"{key}: {value:.2f}\n")
    print(f"  Saved: {project_relative(out)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Backtracking overview figures")
    p.add_argument("--s2_dir", type=Path, default=S2_DIR,
                   help="Step 2 output folder (its parent is the step 1 folder)")
    p.add_argument("--input", type=Path, default=BACKTRACKING_INPUT,
                   help="BACKTRACKING_INPUT file the backtracking was run on")
    p.add_argument("--out_dir", type=Path, default=FIGURES_PATH / "backtracking")
    p.add_argument("--only", nargs="+", choices=PLOTS, default=PLOTS)
    p.add_argument("--replot", action="store_true",
                   help="Reuse source_meteorology_regions.csv instead of recomputing")
    args = p.parse_args()

    s2_dir, s1_dir, od = args.s2_dir, args.s2_dir.parent, args.out_dir
    header("Backtracking figures", {
        "Step 2 run": project_relative(s2_dir),
        "Output": project_relative(od),
    })

    if {"maps", "meteorology"} & set(args.only):
        region_csv = od / "source_meteorology_regions.csv"
        need_regions = "meteorology" in args.only and not (args.replot and region_csv.exists())
        if "maps" in args.only or need_regions:
            with xr.open_dataset(s2_dir / "sds_sources_filtered.nc") as ds:
                source_type = ds["source_type"].values
                lat, lon = ds["latitude"].values, ds["longitude"].values
                src_times = pd.to_datetime(ds["time"].values)
        if "maps" in args.only:
            plot_source_density(source_type, lat, lon, od)
        if "meteorology" in args.only:
            section("Conditions at source regions")
            if need_regions:
                regions = source_region_stats(source_type, src_times,
                                              s1_dir / "sds_segmentation.nc", args.input)
                od.mkdir(parents=True, exist_ok=True)
                regions.to_csv(region_csv, index=False)
                print(f"  Saved: {project_relative(region_csv)}")
            else:
                regions = pd.read_csv(region_csv, parse_dates=["time"])
            plot_source_meteorology(regions, od)

    if "trajectories" in args.only:
        features = pd.read_hdf(s1_dir / "sds_features.h5", key="table")
        features = features.dropna(subset=["cell"]).astype({"cell": int})
        features["time"] = pd.to_datetime(features["time"])
        plot_all_trajectories(features, od)

    if "counts" in args.only:
        summary_df = pd.read_csv(s1_dir / "sds_track_summary.csv")
        interm_df = pd.read_csv(s2_dir / "sds_intermediate_sources.csv")
        lineage_path = s1_dir / "sds_lineage.csv"
        lineage_df = pd.read_csv(lineage_path) if lineage_path.exists() else pd.DataFrame()
        write_source_counts(summary_df, interm_df, lineage_df, od)

    done(od)


if __name__ == "__main__":
    main()
