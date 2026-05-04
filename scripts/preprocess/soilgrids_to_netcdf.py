"""
Convert downloaded SoilGrids GeoTIFFs (sand, silt, clay) into a single
NetCDF with derived variables.

Reads the GeoTIFFs produced by download_soilgrids.py, converts the raw
integer values to weight-percent, and adds:
  - fine_fraction = silt + clay  (%)
  - sand_to_fines = sand / fine_fraction  (dimensionless)

Output:
    data/soil/soil_0_5_mean_features.nc
    Variables: sand, silt, clay, fine_fraction, sand_to_fines
    Dimensions: depth, lat, lon
"""

import os
from pathlib import Path

import xarray as xr
import rioxarray  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "soil"
OUTPUT_FILE = INPUT_DIR / "soil_0_5_mean_features.nc"

PROPERTIES = ["sand", "silt", "clay"]
DEPTHS = ["0-5cm"]
QUANTILE = "mean"
SCALE = 0.1                 # SoilGrids stores g/kg as int → divide by 10 for wt-%
NODATA_VALUE = -32768


def read_tiff(prop, depth):
    fname = os.path.join(INPUT_DIR, f"{prop}_{depth}_{QUANTILE}.tif")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing: {fname}")

    da = rioxarray.open_rasterio(fname).squeeze("band", drop=True)
    da = da.where(da != NODATA_VALUE)

    if da.y[0] > da.y[-1]:
        da = da.isel(y=slice(None, None, -1))

    da = da.rename({"x": "lon", "y": "lat"})
    return da


def main():
    print("Reading GeoTIFFs ...")

    data = {prop: [] for prop in PROPERTIES}
    for depth in DEPTHS:
        for prop in PROPERTIES:
            print(f"  {prop}_{depth}_{QUANTILE}.tif")
            data[prop].append(read_tiff(prop, depth))

    arrays = {}
    for prop in PROPERTIES:
        stacked = xr.concat(data[prop], dim="depth")
        stacked["depth"] = DEPTHS
        stacked = stacked * SCALE
        stacked.attrs["units"] = "%"
        stacked.attrs["long_name"] = f"{prop} content"
        arrays[prop] = stacked

    fine_fraction = arrays["silt"] + arrays["clay"]
    fine_fraction.attrs = {"units": "%", "long_name": "fine fraction (silt + clay)"}

    sand_to_fines = arrays["sand"] / fine_fraction.where(fine_fraction > 0)
    sand_to_fines.attrs = {"units": "1",
                            "long_name": "sand-to-fines ratio (sand / (silt + clay))"}

    ds = xr.Dataset({
        "sand": arrays["sand"],
        "silt": arrays["silt"],
        "clay": arrays["clay"],
        "fine_fraction": fine_fraction,
        "sand_to_fines": sand_to_fines,
    })

    ds.attrs["title"] = "SoilGrids 2.0 — sand, silt, clay texture fractions"
    ds.attrs["source"] = "https://soilgrids.org (ISRIC — World Soil Information)"
    ds.attrs["license"] = "CC-BY 4.0"
    ds.attrs["quantile"] = QUANTILE
    ds.attrs["units_note"] = ("sand, silt, clay, fine_fraction in weight-%; "
                               "sand_to_fines is dimensionless.")

    ds["lon"].attrs = {"units": "degrees_east",  "long_name": "longitude"}
    ds["lat"].attrs = {"units": "degrees_north", "long_name": "latitude"}
    ds["depth"].attrs = {"long_name": "depth interval"}

    encoding = {
        var: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for var in ["sand", "silt", "clay", "fine_fraction", "sand_to_fines"]
    }

    print(f"\nWriting {OUTPUT_FILE} ...")
    ds.to_netcdf(OUTPUT_FILE, encoding=encoding)
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"Done — {OUTPUT_FILE} ({size_mb:.1f} MB)")

    print("\nDataset overview:")
    print(ds)
    print("\nSample statistics (first depth):")
    for var in ["sand", "silt", "clay", "fine_fraction", "sand_to_fines"]:
        v = ds[var].isel(depth=0)
        print(f"  {var:16s}  min={float(v.min()):7.2f}  "
              f"max={float(v.max()):7.2f}  mean={float(v.mean()):7.2f}")


if __name__ == "__main__":
    main()
