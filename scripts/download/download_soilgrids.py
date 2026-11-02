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

import xarray as xr
import rioxarray  # noqa: F401
from owslib.wcs import WebCoverageService


from sds.config import DOMAIN, SOIL_PATH
from sds.log_utils import header, section, done
os.makedirs(SOIL_PATH, exist_ok=True)

BBOX = (DOMAIN["west"], DOMAIN["south"], DOMAIN["east"], DOMAIN["north"])   # lon/lat min/max, EPSG:4326
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
        print(f"  Already exists: {output_file.name}")
        return output_file

    url = f"https://maps.isric.org/mapserv?map=/map/{prop}.map"

    wcs = WebCoverageService(url, version=WCS_VERSION)

    available = list(wcs.contents)
    if coverage_id not in available:
        print(f"  WARNING: '{coverage_id}' not offered by the WCS (available: {available[:6]} ...)")
        return None

    print(f"  Requesting: {coverage_id}")
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

    print(f"  Saved: {output_file.name}")
    return output_file


def download_all():
    section("Step 1: Download GeoTIFFs (WCS)")

    total = len(PROPERTIES) * len(DEPTHS)
    count = 0
    for prop in PROPERTIES:
        for depth in DEPTHS:
            count += 1
            try:
                download_layer(prop, depth, QUANTILE)
            except Exception as e:
                print(f"  ERROR: {e}")
            if count < total:
                time.sleep(PAUSE)



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
    section("Step 2: Combine GeoTIFFs into one NetCDF")

    if NETCDF_INTERMEDIATE.exists():
        print(f"  Already exists: {NETCDF_INTERMEDIATE.name}")
        return xr.open_dataset(NETCDF_INTERMEDIATE)

    data = {prop: [] for prop in PROPERTIES}
    for depth in DEPTHS:
        for prop in PROPERTIES:
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

    texture_sum = ds["sand"] + ds["silt"] + ds["clay"]
    print(f"  Check: sand + silt + clay = {float(texture_sum.mean(skipna=True)):.1f}% on average (expect ~100)")

    encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds.data_vars}
    print(f"  Saving: {NETCDF_INTERMEDIATE.name}")
    ds.to_netcdf(NETCDF_INTERMEDIATE, encoding=encoding)
    return ds


def run_soil_download():
    header("SoilGrids download", {
        "Properties": ", ".join(PROPERTIES),
        "Depth": ", ".join(DEPTHS),
        "Resolution": f"{RESOLUTION}°",
    })
    download_all()
    convert_to_netcdf()
    done(NETCDF_INTERMEDIATE)


if __name__ == "__main__":
    run_soil_download()
