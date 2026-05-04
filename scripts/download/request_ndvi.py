"""
Submit NDVI (MODIS MOD13A1.061) area requests to the NASA AppEEARS API,
one task per year (2015-2024).

This script only submits requests. Once they are processed (you'll get an
email), run download_ndvi.py to fetch the resulting bundles.

Reference:
    https://github.com/nasa/AppEEARS-Data-Resources/blob/main/Python/tutorials/AppEEARS_API_Point.ipynb
"""

import requests as r
from pathlib import Path
import json


START_YEAR = 2015
END_YEAR = 2024


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PW_PATH = PROJECT_ROOT / "passwords.json"
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
                [-18.125, -0.125],
                [-18.125, 50.125],
                [85.125, 50.125],
                [85.125, -0.125],
                [-18.125, -0.125],
            ]],
        },
    }],
}


for year in range(START_YEAR, END_YEAR + 1):
    task = {
        'task_type': 'area',
        'task_name': f"{year}-ndvi-big",
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
    print(f"Year {year} task submitted: {task_response}")
