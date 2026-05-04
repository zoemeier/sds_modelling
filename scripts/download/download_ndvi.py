"""
Download NDVI bundles from AppEEARS for tasks submitted by request_ndvi.py.

Tasks must already be in 'done' state on the AppEEARS server. Downloads
support resume — partially downloaded files (.part) are continued rather
than restarted.

Output:
    data/ndvi/<task_name>/...
"""

import os
from pathlib import Path
import json

import requests as r
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "data" / "ndvi"
PW_PATH = PROJECT_ROOT / "passwords.json"
with open(PW_PATH) as f:
    pws = json.load(f)


api = 'https://appeears.earthdatacloud.nasa.gov/api/'
user = pws["ndvi"]["username"]
password = pws["ndvi"]["pw"]

token_response = r.post(f'{api}login', auth=(user, password)).json()
token = token_response['token']
headers = {'Authorization': f'Bearer {token}'}


def download_task_bundle(task_id, destDir, head):
    """Download all files in a task bundle, resuming any .part files."""
    os.makedirs(destDir, exist_ok=True)
    bundle = r.get(f"{api}bundle/{task_id}", headers=head).json()
    files = {f['file_id']: f['file_name'] for f in bundle['files']}

    for f_key, fname in files.items():
        url = f"{api}bundle/{task_id}/{f_key}"
        filepath = os.path.join(destDir, os.path.basename(fname))
        temp_path = filepath + ".part"
        resume_byte_pos = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

        request_headers = head.copy()
        if resume_byte_pos > 0:
            request_headers["Range"] = f"bytes={resume_byte_pos}-"

        with r.get(url, headers=request_headers, stream=True) as resp:
            resp.raise_for_status()
            total_size = int(resp.headers.get('content-length', 0)) + resume_byte_pos
            mode = 'ab' if resume_byte_pos > 0 else 'wb'
            with open(temp_path, mode) as f_out, tqdm(
                total=total_size, initial=resume_byte_pos,
                unit='B', unit_scale=True, desc=fname
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f_out.write(chunk)
                        pbar.update(len(chunk))
        os.rename(temp_path, filepath)
        print(f"Downloaded {filepath}")


resp = r.get(f'{api}task', headers=headers)
resp.raise_for_status()
tasks = resp.json()

years = range(2015, 2025)
target_task_names = {f"{y}-ndvi-big" for y in years}

for t in tasks:
    print(t['task_name'], t['status'])

tasks_to_download = [
    t for t in tasks
    if t['task_name'] in target_task_names and t['status'] == 'done'
]


for task in tasks_to_download:
    task_id = task['task_id']
    task_name = task['task_name']
    file_path = OUT_PATH / task_name
    print(f"Downloading {task_name} ...")
    download_task_bundle(task_id, file_path, headers)
