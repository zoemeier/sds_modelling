"""
Submit NDVI (MODIS MOD13A1.061) area requests to the NASA AppEEARS API,
one task per year (config.YEARS).

This script only submits requests. Once they are processed (you'll get an
email), run download_ndvi.py to fetch the resulting bundles.

Reference:
    https://github.com/nasa/AppEEARS-Data-Resources/blob/main/Python/tutorials/AppEEARS_API_Point.ipynb
"""

import requests as r
import json




from sds.config import YEARS, DOMAIN, ERA5_NATIVE_RES, PASSWORDS_FILE as PW_PATH
from sds.log_utils import header, done

# domain extended by half a 0.25° cell, so the outer grid cells are fully covered
PAD = ERA5_NATIVE_RES / 2
W, E = DOMAIN["west"] - PAD, DOMAIN["east"] + PAD
S, N = DOMAIN["south"] - PAD, DOMAIN["north"] + PAD
with open(PW_PATH) as f:
    pws = json.load(f)


api = 'https://appeears.earthdatacloud.nasa.gov/api/'
user = pws["ndvi"]["username"]
password = pws["ndvi"]["pw"]
token_response = r.post(f'{api}login', auth=(user, password)).json()
token = token_response['token']
head = {'Authorization': f'Bearer {token}'}


prods = ['MOD13A1.061']
layers = [(prods[0], '_500m_16_days_NDVI')]
prodLayer = [{"layer": l[1], "product": l[0]} for l in layers]

bounding_box = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [W, S], [W, N], [E, N], [E, S], [W, S],
            ]],
        },
    }],
}


header("NDVI request (AppEEARS)", {"Years": f"{min(YEARS)}-{max(YEARS)}"})

for year in YEARS:
    task = {
        'task_type': 'area',
        'task_name': f"{year}-ndvi",
        'params': {
            'dates': [{
                'startDate': f"01-01-{year}",
                'endDate':   f"12-31-{year}",
            }],
            'layers': prodLayer,
            'output': {'format': {'type': 'netcdf4'}, 'projection': 'geographic'},
            'geo': bounding_box,
        },
    }

    task_response = r.post(f'{api}task', json=task, headers=head).json()
    print(f"  Submitted: {year}-ndvi ({task_response})")

done("run download_ndvi.py once AppEEARS has emailed that the tasks are done")
