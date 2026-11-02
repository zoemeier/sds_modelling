"""
Download NDVI bundles from AppEEARS for tasks submitted by request_ndvi.py.

Tasks must already be in 'done' state on the AppEEARS server. Downloads
support resume — partially downloaded files (.part) are continued rather
than restarted.

Output:
    data/ndvi/<task_name>/...
"""

import os
import json

import requests as r
from tqdm import tqdm


from sds.config import YEARS, NDVI_PATH as OUT_PATH, PASSWORDS_FILE as PW_PATH
from sds.log_utils import header, done
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
        if os.path.exists(filepath):
            print(f"  Already exists: {fname}")
            continue
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
        print(f"  Saved: {fname}")


header("NDVI download (AppEEARS)", {"Years": f"{min(YEARS)}-{max(YEARS)}"})

resp = r.get(f'{api}task', headers=headers)
resp.raise_for_status()
tasks = resp.json()

target_task_names = {f"{y}-ndvi" for y in YEARS}

status = {t['task_name']: t['status'] for t in tasks if t['task_name'] in target_task_names}
not_ready = sorted(n for n in target_task_names if status.get(n) != 'done')
if not_ready:
    print("  WARNING: not ready (skipped): "
          + ", ".join(f"{n} ({status.get(n, 'not submitted')})" for n in not_ready))

tasks_to_download = [
    t for t in tasks
    if t['task_name'] in target_task_names and t['status'] == 'done'
]


for task in tasks_to_download:
    task_id = task['task_id']
    task_name = task['task_name']
    file_path = OUT_PATH / task_name
    print(f"\n{task_name}")
    download_task_bundle(task_id, file_path, headers)

done(OUT_PATH)
