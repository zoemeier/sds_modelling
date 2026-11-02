"""
Request ERA5 and CAMS reanalysis data from the Copernicus CDS/ADS APIs,
year by year (config.YEARS).

ERA5 dynamic variables (per year):
    tp, dt2m, t2m, u10, v10, lai_lv, lai_hv
ERA5 static variables (single time step):
    sdor, slor, anor, isor, cvl, cvh, land_sea_mask
CAMS variables (per year):
    duaod550, pm10, aod469, aod865

Output:
    ERA5: data/era5/ERA5_<short>_<year>.grib   (hourly)
    CAMS: data/cams/CAMS_<year>.nc             (3-hourly)

Spatial domain: 50N, 18W, 0S, 85E
                (CAMS uses 50.25N to fully cover the 0.75 grid)

Notes:
    Requires API keys in passwords.json at the project root.
    A full run takes >24h. The script is resumable — files that already
    exist on disk are skipped.
"""

import os
import json

import cdsapi


from sds.config import (
    DOMAIN, ERA5_PATH, CAMS_PATH, LAND_MASK_FILE,
    YEARS, PASSWORDS_FILE as PW_PATH,
)
from sds.log_utils import header, section, done
os.makedirs(ERA5_PATH, exist_ok=True)
os.makedirs(CAMS_PATH, exist_ok=True)

with open(PW_PATH) as f:
    pws = json.load(f)

cds_client = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key=pws["era5"]["key"],
)
ads_client = cdsapi.Client(
    url="https://ads.atmosphere.copernicus.eu/api",
    key=pws["cams"]["key"],
)


# CDS variable name -> short name used in filenames
era5_variables = {
    "total_precipitation":                       "tp",
    "2m_dewpoint_temperature":                   "dt2m",
    "2m_temperature":                            "t2m",
    "10m_u_component_of_wind":                   "u10",
    "10m_v_component_of_wind":                   "v10",
    "leaf_area_index_low_vegetation":            "lai_lv",
    "leaf_area_index_high_vegetation":           "lai_hv",
}

era5_static_variables = {
    "standard_deviation_of_orography":           "sdor",
    "slope_of_sub_gridscale_orography":          "slor",
    "anisotropy_of_sub_gridscale_orography":     "anor",
    "angle_of_sub_gridscale_orography":          "isor",
    "low_vegetation_cover":                      "cvl",
    "high_vegetation_cover":                     "cvh",
}

# CDS/ADS area = [north, west, south, east]; CAMS gets 0.25° extra in the north
# so the 0.75° grid is fully covered
ERA5_AREA = [DOMAIN["north"], DOMAIN["west"], DOMAIN["south"], DOMAIN["east"]]
CAMS_AREA = [DOMAIN["north"] + 0.25, DOMAIN["west"], DOMAIN["south"], DOMAIN["east"]]


header("ERA5 + CAMS download", {
    "Years": f"{min(YEARS)}-{max(YEARS)}",
    "Domain": f"{DOMAIN['south']}-{DOMAIN['north']}°N, {DOMAIN['west']}-{DOMAIN['east']}°E",
})

# ERA5 dynamic variables, year by year
section("ERA5 hourly")
for var, short in era5_variables.items():
    for year in YEARS:
        filename = ERA5_PATH / f"ERA5_{short}_{year}.grib"
        if filename.exists():
            print(f"  Already exists: {filename.name}")
            continue

        print(f"  Requesting: {filename.name}")
        cds_client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [var],
                "year": [str(year)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": ERA5_AREA,
                "format": "grib",
            },
            target=str(filename),
        )
        print(f"  Saved: {filename.name}")


# ERA5 static variables (single time step)
section("ERA5 static")
for var, short in era5_static_variables.items():
    filename = ERA5_PATH / f"ERA5_{short}_static.grib"
    if filename.exists():
        print(f"  Already exists: {filename.name}")
        continue

    print(f"  Requesting: {filename.name}")
    cds_client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [var],
            "year": ["2020"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
            "area": ERA5_AREA,
            "format": "grib",
            "download_format": "unarchived",
        },
        target=str(filename),
    )
    print(f"  Saved: {filename.name}")


# CAMS, year by year
cams_variables = [
    "dust_aerosol_optical_depth_550nm",
    "particulate_matter_10um",
    "total_aerosol_optical_depth_469nm",
    "total_aerosol_optical_depth_865nm",
]
section("CAMS 3-hourly")
for year in YEARS:
    filename = CAMS_PATH / f"CAMS_{year}.nc"
    if filename.exists():
        print(f"  Already exists: {filename.name}")
        continue

    print(f"  Requesting: {filename.name}")
    ads_client.retrieve(
        "cams-global-reanalysis-eac4",
        {
            "date": [f"{year}-01-01/{year}-12-31"],
            "time": [
                "00:00", "03:00", "06:00",
                "09:00", "12:00", "15:00",
                "18:00", "21:00",
            ],
            "format": "netcdf",
            "variable": cams_variables,
            "area": CAMS_AREA,
        }
    ).download(filename)
    print(f"  Saved: {filename.name}")


# Land-sea mask (single time step)
section("ERA5 land-sea mask")
filename = LAND_MASK_FILE
if filename.exists():
    print(f"  Already exists: {filename.name}")
else:
    print(f"  Requesting: {filename.name}")
    cds_client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["land_sea_mask"],
            "year": ["2020"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
            "area": ERA5_AREA,
            "format": "grib",
            "download_format": "unarchived",
        },
        target=str(filename),
    )
    print(f"  Saved: {filename.name}")

done()
