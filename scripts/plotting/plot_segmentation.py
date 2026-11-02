"""
Example days of the step 1 plume segmentation.

Each figure shows one timestep as a 2x2 panel:
  PM10 | Dust AOD
  AE   | segmented plumes (one colour per plume)
The detection thresholds are drawn as black contours; they are read from the
step 1 output (sds_source_masks.nc), so they always match the run plotted.

Days are either given with --dates or drawn at random from timesteps with at
least --min_plumes plumes (weighted towards days with more plumes).

Usage:
  python plot_segmentation.py                               # 20 random days
  python plot_segmentation.py --dates 2018-03-22 2022-06-14
  python plot_segmentation.py --s1_dir <step 1 dir> --input <BACKTRACKING_INPUT nc>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sds.config import FIGURES_PATH
from sds.paths import project_relative
from sds.log_utils import header, done
from sds.plot_utils import (
    S1_DIR, BACKTRACKING_INPUT, PROJ,
    TITLE_SIZE, SUBTITLE_SIZE, MAP_WIDTH,
    map_axes, add_colorbar, save_figure,
)


def _time_dim(ds):
    return "valid_time" if "valid_time" in ds.dims else "time"


def read_thresholds(s1_dir):
    with xr.open_dataset(s1_dir / "sds_source_masks.nc") as ds:
        a = ds["source_mask"].attrs
    return {"duaod": float(a["duaod_thr"]), "ae": float(a["angstr_thr"]),
            "pm10": float(a["pm10_thr"])}


def pick_timesteps(seg, n_plots, min_plumes, seed):
    """Random timesteps with >= min_plumes plumes, weighted by sqrt(n_plumes)."""
    n_plumes = np.array([len(np.unique(seg.isel(time=ti).values)) - 1
                         for ti in range(seg.sizes["time"])])
    candidates = np.where(n_plumes >= min_plumes)[0]
    if len(candidates) == 0:
        raise SystemExit("No timesteps with enough plumes; lower --min_plumes")
    w = np.sqrt(n_plumes[candidates])
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=min(n_plots, len(candidates)),
                        replace=False, p=w / w.sum())
    return sorted(chosen)


def _plume_cmap(n):
    base = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    cmap = mcolors.ListedColormap([base[i % len(base)] for i in range(max(n, 1))])
    return cmap, mcolors.BoundaryNorm(np.arange(0.5, max(n, 1) + 1.5), cmap.N)


def plot_timestep(seg_ds, seg_var, inp, ti, thr, od):
    tdim = _time_dim(inp)
    t = pd.Timestamp(seg_ds["time"].values[ti])
    s_lat, s_lon = seg_ds["latitude"].values, seg_ds["longitude"].values
    extent = [s_lon.min(), s_lon.max(), s_lat.min(), s_lat.max()]

    x = inp.sel({tdim: t}, method="nearest")
    i_lat, i_lon = inp["latitude"].values, inp["longitude"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        ae = -np.log(x["aod469"].values / x["aod865"].values) / np.log(469.0 / 865.0)

    seg = seg_ds[seg_var].isel(time=ti).values
    ids, inv = np.unique(seg, return_inverse=True)
    inv = inv.reshape(seg.shape).astype(float)
    if ids[0] == 0:
        inv[seg == 0] = np.nan     # background transparent, plumes -> 1..n
    else:
        inv += 1
    n_plumes = int((ids > 0).sum())

    # (data, title, cmap, threshold, which side of the threshold is "dusty")
    panels = [
        (x["pm10"].values,     f"PM10 (kg m⁻³), contour ≥ {thr['pm10']:.1e}", "YlOrRd",   thr["pm10"], "ge"),
        (x["duaod550"].values, f"Dust AOD 550 nm, contour ≥ {thr['duaod']:g}", "YlOrRd",   thr["duaod"], "ge"),
        (ae,                   f"Ångström exponent, contour ≤ {thr['ae']:g}",  "YlOrRd_r", thr["ae"], "le"),
    ]

    fig = plt.figure(figsize=(MAP_WIDTH, 6.6))
    axes = [map_axes(fig, 221 + k, extent=extent) for k in range(4)]
    fig.subplots_adjust(wspace=0.28, hspace=0.22)

    meshes = []
    for ax, (data, title, cmap, level, side) in zip(axes, panels):
        finite = data[np.isfinite(data)]
        vmin, vmax = np.percentile(finite, [0.1, 99.9])
        meshes.append(ax.pcolormesh(i_lon, i_lat, data, cmap=cmap, vmin=vmin, vmax=vmax,
                                    shading="nearest", transform=PROJ, zorder=1))
        # fill NaNs on the non-dusty side so they don't create spurious contours
        fill = finite.min() if side == "ge" else finite.max()
        ax.contour(i_lon, i_lat, np.where(np.isfinite(data), data, fill), levels=[level],
                   colors="black", linewidths=1.0, transform=PROJ, zorder=4)
        ax.set_title(title, fontsize=SUBTITLE_SIZE)

    cmap, norm = _plume_cmap(n_plumes)
    axes[3].pcolormesh(s_lon, s_lat, inv, cmap=cmap, norm=norm,
                       shading="nearest", transform=PROJ, zorder=1)
    axes[3].set_title(f"Segmented plumes (n = {n_plumes})", fontsize=SUBTITLE_SIZE)

    for ax, im in zip(axes[:3], meshes):
        add_colorbar(fig, ax, im)
    fig.suptitle(t.strftime("%Y-%m-%d"), fontsize=TITLE_SIZE, fontweight="bold",
                 y=axes[0].get_position().y1 + 0.07)
    save_figure(fig, od, f"segmentation_{t:%Y-%m-%d}")


def main():
    p = argparse.ArgumentParser(description="Plot example days of the plume segmentation")
    p.add_argument("--s1_dir", type=Path, default=S1_DIR, help="Step 1 output folder")
    p.add_argument("--input", type=Path, default=BACKTRACKING_INPUT,
                   help="BACKTRACKING_INPUT file the backtracking was run on")
    p.add_argument("--out_dir", type=Path, default=FIGURES_PATH / "segmentation")
    p.add_argument("--dates", nargs="+", default=None,
                   help="Plot these dates (YYYY-MM-DD) instead of random ones")
    p.add_argument("--n_plots", type=int, default=20)
    p.add_argument("--min_plumes", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    thr = read_thresholds(args.s1_dir)
    header("Segmentation figures", {
        "Step 1 run": project_relative(args.s1_dir),
        "Output": project_relative(args.out_dir),
    })

    seg_ds = xr.open_dataset(args.s1_dir / "sds_segmentation.nc")
    seg_var = list(seg_ds.data_vars)[0]
    inp = xr.open_dataset(args.input)

    if args.dates:
        seg_times = pd.Index(pd.to_datetime(seg_ds["time"].values))
        tis = seg_times.get_indexer(pd.to_datetime(args.dates))
        if (tis < 0).any():
            missing = [d for d, i in zip(args.dates, tis) if i < 0]
            raise SystemExit(f"Dates not in segmentation: {missing}")
    else:
        tis = pick_timesteps(seg_ds[seg_var], args.n_plots, args.min_plumes, args.seed)

    for ti in tis:
        plot_timestep(seg_ds, seg_var, inp, ti, thr, args.out_dir)

    seg_ds.close()
    inp.close()
    done(args.out_dir)


if __name__ == "__main__":
    main()
