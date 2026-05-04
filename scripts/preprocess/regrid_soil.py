"""
Regrid SoilGrids data to the target spatial grid (0.25° from ERA5 or
0.75° from CAMS) using xesmf conservative regridding.

Conservative regridding is used because the primary soil variables are
intensive (fractions, densities, concentrations). Derived variables
(fine_fraction, sand_to_fines) are recomputed from the regridded
primaries, since mean(a/b) ≠ mean(a)/mean(b).

Usage:
    python regrid_soil.py             # 0.25° (default)
    python regrid_soil.py -s 0.75
"""

import os
import argparse
from pathlib import Path

import xarray as xr
import xesmf as xe
from dask.diagnostics import ProgressBar

from grid_config import make_target_grid, spatial_tag


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data"
SOIL_PATH = DATA_PATH / "soil"
OUT_PATH = SOIL_PATH
os.makedirs(OUT_PATH, exist_ok=True)

NETCDF_INTERMEDIATE = SOIL_PATH / "soilgrids.nc"
PRIMARY_VARS = ["sand", "silt", "clay", "bdod", "soc", "cec"]


def regrid_soil(spatial_res=0.25):
    stag = spatial_tag(spatial_res)
    output_file = OUT_PATH / f"soil_2015-2024_{stag}.nc"

    print("=" * 60)
    print(f"Regrid SoilGrids → {spatial_res}° (conservative)")
    print("=" * 60)

    if output_file.exists():
        print(f"  Already exists: {output_file.name}")
        return
    if not NETCDF_INTERMEDIATE.exists():
        print(f"  ERROR: {NETCDF_INTERMEDIATE} not found.")
        print(f"  Run download_soilgrids.py first.")
        return

    ds_soil = xr.open_dataset(NETCDF_INTERMEDIATE)
    ds_soil = ds_soil.drop_vars("spatial_ref", errors="ignore")

    if "depth" in ds_soil.sizes and ds_soil.sizes["depth"] == 1:
        ds_soil = ds_soil.squeeze("depth", drop=True)

    target_lat, target_lon = make_target_grid(spatial_res)
    ds_target = xr.Dataset({
        "lat": (["lat"], target_lat),
        "lon": (["lon"], target_lon),
    })

    print(f"  Source grid:  {len(ds_soil.lat)}×{len(ds_soil.lon)}")
    print(f"  Target grid:  {len(target_lat)}×{len(target_lon)}")

    ds_primary = ds_soil[PRIMARY_VARS]

    print("  Building regridder ...")
    regridder = xe.Regridder(ds_primary, ds_target, method="conservative")
    print(f"  {regridder}")

    ds_regrid = regridder(ds_primary, skipna=True, na_thres=1.0)

    ds_regrid["fine_fraction"] = ds_regrid["silt"] + ds_regrid["clay"]
    ds_regrid["fine_fraction"].attrs = {
        "units": "%", "long_name": "fine fraction (silt + clay)"
    }

    ds_regrid["sand_to_fines"] = ds_regrid["sand"] / ds_regrid["fine_fraction"].where(
        ds_regrid["fine_fraction"] > 0
    )
    ds_regrid["sand_to_fines"].attrs = {"units": "1", "long_name": "sand-to-fines ratio"}

    ds_regrid = ds_regrid.rename({"lat": "latitude", "lon": "longitude"})

    print("\n  Sanity checks:")
    texture_sum = ds_regrid["sand"] + ds_regrid["silt"] + ds_regrid["clay"]
    print(f"    sand+silt+clay: min={float(texture_sum.min(skipna=True)):.2f}  "
          f"max={float(texture_sum.max(skipna=True)):.2f}  "
          f"mean={float(texture_sum.mean(skipna=True)):.2f}  (expect ~100)")
    for v in PRIMARY_VARS + ["fine_fraction", "sand_to_fines"]:
        vmin = float(ds_regrid[v].min(skipna=True))
        vmax = float(ds_regrid[v].max(skipna=True))
        vmean = float(ds_regrid[v].mean(skipna=True))
        print(f"    {v:16s}  min={vmin:8.3f}  max={vmax:8.3f}  mean={vmean:8.3f}")

    print(f"\n  Saving {output_file} ...")
    encoding = {v: {"zlib": False, "dtype": "float32"} for v in ds_regrid.data_vars}
    with ProgressBar():
        ds_regrid.to_netcdf(output_file, engine="h5netcdf", encoding=encoding)

    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  Done — {output_file} ({size_mb:.1f} MB)")
    ds_soil.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regrid SoilGrids data to target spatial resolution.")
    parser.add_argument("-s", "--spatial", type=float, choices=[0.25, 0.75],
                        default=0.25, help="Spatial resolution in degrees (default: 0.25)")
    args = parser.parse_args()
    regrid_soil(args.spatial)
