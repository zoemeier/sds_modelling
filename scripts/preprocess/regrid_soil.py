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

import xarray as xr
import xesmf as xe
from dask.diagnostics import ProgressBar

from sds.config import SUPPORTED_SPATIAL_RES, SPATIAL_RES, SOIL_PATH
from sds.paths import spatial_tag, YEARS_TAG
from sds.grid import make_target_grid
from sds.log_utils import header, done


OUT_PATH = SOIL_PATH
os.makedirs(OUT_PATH, exist_ok=True)

NETCDF_INTERMEDIATE = SOIL_PATH / "soilgrids.nc"
PRIMARY_VARS = ["sand", "silt", "clay", "bdod", "soc", "cec"]


def regrid_soil(spatial_res=SPATIAL_RES):
    stag = spatial_tag(spatial_res)
    output_file = OUT_PATH / f"soil_{YEARS_TAG}_{stag}.nc"

    header("Soil regridding", {"Resolution": f"{spatial_res}° (conservative)"})

    if output_file.exists():
        print(f"  Already exists: {output_file.name}")
        done(output_file)
        return
    if not NETCDF_INTERMEDIATE.exists():
        raise FileNotFoundError(
            f"{NETCDF_INTERMEDIATE.name} not found. Run download_soilgrids.py first.")

    ds_soil = xr.open_dataset(NETCDF_INTERMEDIATE)
    ds_soil = ds_soil.drop_vars("spatial_ref", errors="ignore")

    if "depth" in ds_soil.sizes and ds_soil.sizes["depth"] == 1:
        ds_soil = ds_soil.squeeze("depth", drop=True)

    target_lat, target_lon = make_target_grid(spatial_res)
    ds_target = xr.Dataset({
        "lat": (["lat"], target_lat),
        "lon": (["lon"], target_lon),
    })

    print(f"  Grid: {len(ds_soil.lat)} × {len(ds_soil.lon)} → {len(target_lat)} × {len(target_lon)}")

    ds_primary = ds_soil[PRIMARY_VARS]
    regridder = xe.Regridder(ds_primary, ds_target, method="conservative")

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

    texture_sum = ds_regrid["sand"] + ds_regrid["silt"] + ds_regrid["clay"]
    print(f"  Check: sand + silt + clay = {float(texture_sum.mean(skipna=True)):.1f}% on average (expect ~100)")

    print(f"  Saving: {output_file.name}")
    encoding = {v: {"zlib": False, "dtype": "float32"} for v in ds_regrid.data_vars}
    with ProgressBar():
        ds_regrid.to_netcdf(output_file, engine="h5netcdf", encoding=encoding)

    ds_soil.close()
    done(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regrid SoilGrids data to target spatial resolution.")
    parser.add_argument("-s", "--spatial", type=float, choices=SUPPORTED_SPATIAL_RES,
                        default=SPATIAL_RES, help=f"Spatial resolution in degrees (default: {SPATIAL_RES})")
    args = parser.parse_args()
    regrid_soil(args.spatial)
