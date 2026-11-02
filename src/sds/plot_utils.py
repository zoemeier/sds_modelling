"""
Shared style, paths and map helpers for the plotting scripts.

Font sizes, colours and the map layout match the model analysis scripts
(analyze_rf_results.py, compute_shap.py, ...) so all figures read as one set.
Every figure is written through save_figure(), which decides the output
folder and file format(s) in one place.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from sds.config import DOMAIN, RESOLUTION, SPATIAL_RES
from sds.paths import SDS_PATH
from sds.paths import backtracking_input

# raised by shapely for every cartopy coastline/border draw, harmless
warnings.filterwarnings("ignore", message="invalid value encountered in create_collection")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIG_FORMATS  = ("png",)          # add "pdf" for vector output

# default backtracking run = the one the RF was trained on (SDS_PATH)
S2_DIR = SDS_PATH.parent         # step 2: intermediate sources
S1_DIR = S2_DIR.parent           # step 1: tracks, segmentation, genesis
BACKTRACKING_INPUT = backtracking_input(RESOLUTION, SPATIAL_RES)[0]


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
TITLE_SIZE = 14           # suptitle / single-axis title, bold
SUBTITLE_SIZE = 12        # panel titles in large multi-panel figures
SUBPLOT_TITLE_SIZE = 11   # panel titles in small-multiple grids
LABEL_SIZE = 11
TICK_SIZE = 9
LEGEND_SIZE = 9

VAR_SUBPLOT_W = 4.0       # small-multiple panel size (inches)
VAR_SUBPLOT_H = 3.0
MAP_WIDTH = 13            # full-domain map width (inches)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": LABEL_SIZE,
    "axes.titlesize": SUBTITLE_SIZE, "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE, "figure.dpi": 150,
    "savefig.dpi": 250, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
})

# genesis (source origin) vs intermediate (downwind) sources
STYPE_COLORS = {1: "#7E4E91", 2: "#FFB347"}
STYPE_LABELS = {1: "Genesis", 2: "Intermediate"}

DENSITY_CMAP = "YlOrRd"
MAP_ALPHA = 0.80

DOMAIN_EXTENT = [DOMAIN["west"], DOMAIN["east"], DOMAIN["south"], DOMAIN["north"]]
PROJ = ccrs.PlateCarree()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def map_figsize(nrows=1, ncols=1, extent=DOMAIN_EXTENT, width=MAP_WIDTH, extra_h=1.8):
    """Figure size that keeps the map aspect ratio, plus room for titles."""
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    return width, width / ncols * aspect * nrows + extra_h


def map_axes(fig, pos=111, extent=DOMAIN_EXTENT, tick_size=TICK_SIZE, max_ticks=None):
    """Cartopy axes with the standard coastline / border / gridline style.

    max_ticks: optional (n_lon, n_lat) upper bound on the gridline labels.
    """
    ax = fig.add_subplot(pos, projection=PROJ)
    ax.set_extent(extent, crs=PROJ)
    ax.add_feature(cfeature.COASTLINE, lw=0.6, ec="black", zorder=5)
    ax.add_feature(cfeature.BORDERS, lw=0.3, ec="#444", ls="--", zorder=5)
    gl = ax.gridlines(draw_labels=True, lw=0.3, alpha=0.3, ls=":")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": tick_size}
    if max_ticks:
        gl.xlocator = mticker.MaxNLocator(nbins=max_ticks[0])
        gl.ylocator = mticker.MaxNLocator(nbins=max_ticks[1])
    return ax


def add_colorbar(fig, ax, mappable, label="", **kwargs):
    """Vertical colorbar exactly as tall as the visible map frame."""
    # cartopy letterboxes the axes, so draw first to get the real frame position
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.018, pos.height])
    cb = fig.colorbar(mappable, cax=cax, orientation="vertical", **kwargs)
    cb.set_label(label, fontsize=LABEL_SIZE)
    cb.ax.tick_params(labelsize=TICK_SIZE)
    return cb


def save_figure(fig, out_dir, name):
    """Save fig as out_dir/name.<fmt> for every format in FIG_FORMATS."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in FIG_FORMATS:
        fig.savefig(out_dir / f"{name}.{fmt}")
    plt.close(fig)
    print(f"  Saved: {name}.{'/'.join(FIG_FORMATS)}")
