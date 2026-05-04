"""
Download SoilGrids 2.0 soil property GeoTIFFs via the ISRIC WCS service,
then convert them to a single NetCDF with unit conversions and derived
variables (fine_fraction, sand_to_fines).

Output:
    data/soil/<prop>_<depth>_<quantile>.tif    (per-layer GeoTIFFs)
    data/soil/soilgrids.nc                     (combined NetCDF)
"""

import os
import time
from pathlib import Path

import xarray as xr
import rioxarray  # noqa: F401
from owslib.wcs import WebCoverageService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data"
SOIL_PATH = DATA_PATH / "soil"
os.makedirs(SOIL_PATH, exist_ok=True)

BBOX = (-18, 0, 85, 50)              # lon_min, lat_min, lon_max, lat_max in EPSG:4326
PROPERTIES = ["sand", "silt", "clay", "bdod", "soc", "cec"]
DEPTHS = ["0-5cm"]
QUANTILE = "mean"
CRS = "urn:ogc:def:crs:EPSG::4326"
RESOLUTION = 0.05                    # degrees, ~5 km
WCS_VERSION = "1.0.0"
PAUSE = 5                            # seconds between WCS requests
NODATA_VALUE = -32768

# raw integer / factor -> physical units
CONVERSIONS = {
    "sand": {"factor": 10,  "units": "%",          "long_name": "sand content"},
    "silt": {"factor": 10,  "units": "%",          "long_name": "silt content"},
    "clay": {"factor": 10,  "units": "%",          "long_name": "clay content"},
    "bdod": {"factor": 100, "units": "kg/dm3",     "long_name": "bulk density of fine earth"},
    "soc":  {"factor": 10,  "units": "g/kg",       "long_name": "soil organic carbon content"},
    "cec":  {"factor": 10,  "units": "cmol(c)/kg", "long_name": "cation exchange capacity"},
}

NETCDF_INTERMEDIATE = SOIL_PATH / "soilgrids.nc"


def download_layer(prop, depth, quantile):
    """Download one SoilGrids coverage (prop_depth_quantile) via WCS."""
    coverage_id = f"{prop}_{depth}_{quantile}"
    output_file = SOIL_PATH / f"{coverage_id}.tif"

    if output_file.exists():
        print(f"  Already exists, skipping: {output_file}")
        return output_file

    url = f"https://maps.isric.org/mapserv?map=/map/{prop}.map"

    print(f"  Connecting to WCS: {prop} ...")
    wcs = WebCoverageService(url, version=WCS_VERSION)

    available = list(wcs.contents)
    if coverage_id not in available:
        print(f"  WARNING: '{coverage_id}' not found. Available: {available[:6]}...")
        return None

    print(f"  Requesting: {coverage_id}  bbox={BBOX}  res={RESOLUTION}° ...")
    response = wcs.getCoverage(
        identifier=coverage_id,
        crs=CRS,
        bbox=BBOX,
        resx=RESOLUTION,
        resy=RESOLUTION,
        format="GEOTIFF_INT16",
    )

    with open(output_file, "wb") as f:
        f.write(response.read())

    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_file} ({size_mb:.1f} MB)")
    return output_file


def download_all():
    print("=" * 60)
    print("Step 1: Download SoilGrids data via WCS")
    print(f"  Properties : {PROPERTIES}")
    print(f"  Depths     : {DEPTHS}")
    print(f"  Statistic  : {QUANTILE}")
    print(f"  BBox       : lon [{BBOX[0]}, {BBOX[2]}], lat [{BBOX[1]}, {BBOX[3]}]")
    print(f"  Resolution : {RESOLUTION}° (~{RESOLUTION * 111:.0f} km)")
    print("=" * 60)

    total = len(PROPERTIES) * len(DEPTHS)
    count = 0
    for prop in PROPERTIES:
        for depth in DEPTHS:
            count += 1
            print(f"\n[{count}/{total}] {prop} — {depth}")
            try:
                download_layer(prop, depth, QUANTILE)
            except Exception as e:
                print(f"  ERROR: {e}")
            if count < total:
                time.sleep(PAUSE)

    print(f"\nDownload complete. Files in: {SOIL_PATH}")


def read_tiff(prop, depth):
    """Read a GeoTIFF, mask nodata, ensure ascending lat, return (lat, lon) DataArray."""
    fname = SOIL_PATH / f"{prop}_{depth}_{QUANTILE}.tif"
    if not fname.exists():
        raise FileNotFoundError(f"Missing: {fname}")

    da = rioxarray.open_rasterio(fname).squeeze("band", drop=True)
    da = da.where(da != NODATA_VALUE)

    if da.y[0] > da.y[-1]:
        da = da.isel(y=slice(None, None, -1))

    da = da.rename({"x": "lon", "y": "lat"})
    return da


def convert_to_netcdf():
    """Combine downloaded GeoTIFFs into one NetCDF with unit conversions."""
    print("\n" + "=" * 60)
    print("Step 2: Convert GeoTIFFs → NetCDF with unit conversions")
    print("=" * 60)

    if NETCDF_INTERMEDIATE.exists():
        print(f"  Already exists: {NETCDF_INTERMEDIATE.name}")
        return xr.open_dataset(NETCDF_INTERMEDIATE)

    data = {prop: [] for prop in PROPERTIES}
    for depth in DEPTHS:
        for prop in PROPERTIES:
            print(f"  Reading {prop}_{depth}_{QUANTILE}.tif")
            data[prop].append(read_tiff(prop, depth))

    arrays = {}
    for prop in PROPERTIES:
        stacked = xr.concat(data[prop], dim="depth")
        stacked["depth"] = DEPTHS
        conv = CONVERSIONS[prop]
        stacked = stacked / conv["factor"]
        stacked.attrs["units"] = conv["units"]
        stacked.attrs["long_name"] = conv["long_name"]
        arrays[prop] = stacked
        print(f"  {prop}: ÷{conv['factor']} → {conv['units']}")

    fine_fraction = arrays["silt"] + arrays["clay"]
    fine_fraction.attrs = {"units": "%", "long_name": "fine fraction (silt + clay)"}

    sand_to_fines = arrays["sand"] / fine_fraction.where(fine_fraction > 0)
    sand_to_fines.attrs = {"units": "1", "long_name": "sand-to-fines ratio"}

    ds = xr.Dataset({
        "sand": arrays["sand"],
        "silt": arrays["silt"],
        "clay": arrays["clay"],
        "bdod": arrays["bdod"],
        "soc":  arrays["soc"],
        "cec":  arrays["cec"],
        "fine_fraction": fine_fraction,
        "sand_to_fines": sand_to_fines,
    })

    ds.attrs["title"] = "SoilGrids 2.0 — soil texture, bulk density, SOC, CEC"
    ds.attrs["source"] = "https://soilgrids.org (ISRIC — World Soil Information)"
    ds.attrs["license"] = "CC-BY 4.0"
    ds.attrs["quantile"] = QUANTILE
    ds.attrs["units_note"] = (
        "sand/silt/clay/fine_fraction in wt-%; "
        "bdod in kg/dm3; soc in g/kg; cec in cmol(c)/kg; "
        "sand_to_fines dimensionless."
    )

    print("\nSanity checks:")
    texture_sum = ds["sand"] + ds["silt"] + ds["clay"]
    print(f"  sand+silt+clay: min={float(texture_sum.min(skipna=True)):.1f}  "
          f"max={float(texture_sum.max(skipna=True)):.1f}  "
          f"mean={float(texture_sum.mean(skipna=True)):.1f}  (expect ~100)")
    for v in PROPERTIES:
        vmin = float(ds[v].min(skipna=True))
        vmax = float(ds[v].max(skipna=True))
        vmean = float(ds[v].mean(skipna=True))
        print(f"  {v:6s}: min={vmin:8.2f}  max={vmax:8.2f}  mean={vmean:8.2f}  [{CONVERSIONS[v]['units']}]")

    encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds.data_vars}
    print(f"\nWriting {NETCDF_INTERMEDIATE} ...")
    ds.to_netcdf(NETCDF_INTERMEDIATE, encoding=encoding)
    size_mb = NETCDF_INTERMEDIATE.stat().st_size / (1024 * 1024)
    print(f"Done — {NETCDF_INTERMEDIATE} ({size_mb:.1f} MB)")
    return ds


def run_soil_download():
    download_all()
    convert_to_netcdf()
    print(f"\nDone. Intermediate file: {NETCDF_INTERMEDIATE}")


if __name__ == "__main__":
    run_soil_download()
