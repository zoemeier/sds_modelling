"""
Request ERA5 and CAMS reanalysis data from the Copernicus CDS/ADS APIs,
year by year (2015-2024).

ERA5 dynamic variables (per year):
    tp, dt2m, t2m, u10, v10, soil_type, i10fg, blh, swvl1, cape, lai_lv, lai_hv
ERA5 static variables (single time step):
    z, sdor, slor, anor, isor, cvl, cvh, land_sea_mask
CAMS variables (per year):
    u10, v10, daod550, pm10, aod469, aod550, aod865

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

from pathlib import Path
import os
import json

import cdsapi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data"
ERA5_PATH = DATA_PATH / "era5"
CAMS_PATH = DATA_PATH / "cams"
os.makedirs(ERA5_PATH, exist_ok=True)
os.makedirs(CAMS_PATH, exist_ok=True)

PW_PATH = PROJECT_ROOT / "passwords.json"
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
    "soil_type":                                 "soil",
    "instantaneous_10m_wind_gust":               "i10fg",
    "boundary_layer_height":                     "blh",
    "volumetric_soil_water_layer_1":             "swvl1",
    "convective_available_potential_energy":     "cape",
    "leaf_area_index_low_vegetation":            "lai_lv",
    "leaf_area_index_high_vegetation":           "lai_hv",
}

era5_static_variables = {
    "geopotential":                              "z",
    "standard_deviation_of_orography":           "sdor",
    "slope_of_sub_gridscale_orography":          "slor",
    "anisotropy_of_sub_gridscale_orography":     "anor",
    "angle_of_sub_gridscale_orography":          "isor",
    "low_vegetation_cover":                      "cvl",
    "high_vegetation_cover":                     "cvh",
}

ERA5_AREA = [50, -18, 0, 85]
CAMS_AREA = [50.25, -18, 0, 85]


# ERA5 dynamic variables, year by year
for var, short in era5_variables.items():
    for year in range(2015, 2025):
        filename = ERA5_PATH / f"ERA5_{short}_{year}.grib"
        if filename.exists():
            print(f"  Already exists: {filename.name}")
            continue

        print(f"Submitting ERA5 request: {var} {year}")
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
        print(f"Downloaded ERA5 {var} {year}")


# ERA5 static variables (single time step)
for var, short in era5_static_variables.items():
    filename = ERA5_PATH / f"ERA5_{short}_static.grib"
    if filename.exists():
        print(f"  Already exists: {filename.name}")
        continue

    print(f"Submitting ERA5 static request: {var}")
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
    print(f"Downloaded ERA5 static {var}")


# CAMS, year by year
cams_variables = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "dust_aerosol_optical_depth_550nm",
    "particulate_matter_10um",
    "total_aerosol_optical_depth_469nm",
    "total_aerosol_optical_depth_550nm",
    "total_aerosol_optical_depth_865nm",
]
cams_periods = {
    "2015-2019": range(2015, 2020),
    "2020-2024": range(2020, 2025),
}

for period, years in cams_periods.items():
    for year in years:
        filename = CAMS_PATH / f"CAMS_{year}.nc"
        if filename.exists():
            print(f"  Already exists: {filename.name}")
            continue

        print(f"Submitting CAMS request for {year} -> {filename}")
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
        print(f"Downloaded CAMS {year}")


# Land-sea mask (single time step)
filename = ERA5_PATH / "ERA5_land_sea_mask_2020.grib"
if not filename.exists():
    print("Submitting ERA5 request: Land Sea Mask 2020")
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
    print("Downloaded ERA5 Land Sea Mask 2020")
