"""
Download ACLED conflict events from the ACLED API and save the raw events as
data/acled/acled_raw_events.csv. The period is config.YEARS plus one earlier
year, so the 365-day rolling sums are complete from the first year on.

Event types: Battles, Explosions/Remote violence. Events are downloaded for all
locations; preprocess_acled.py keeps those on the model grid.

Uses the API's cursor-based pagination. Every page is appended to a partial
file and the next cursor is saved, so an interrupted run resumes where it
stopped. Authorisation / rate-limit errors are retried after a pause with a new
token.

Gridding onto the model grid happens in scripts/preprocess/preprocess_acled.py.
Requires ACLED credentials in passwords.json ("acled": username + pw).
Skipped if the CSV already exists.
"""

import json
import time

import requests
import pandas as pd

from sds.config import ACLED_PATH, ACLED_RAW_EVENTS, YEARS, PASSWORDS_FILE
from sds.log_utils import header, done

EVENT_TYPES = ["Battles", "Explosions/Remote violence"]
FIELDS = "event_id_cnty|event_date|event_type|latitude|longitude"
# config.YEARS plus one earlier year, so the 365-day rolling sums are complete
START_DATE = f"{min(YEARS) - 1}-01-01"
END_DATE = f"{max(YEARS)}-12-31"

TOKEN_URL = "https://acleddata.com/oauth/token"
READ_URL = "https://acleddata.com/api/acled/read"
PAGE_SIZE = 5000
PAUSE_S = 1                        # between pages
RETRY_WAITS_S = [60, 120, 240, 480, 960]
RETRY_STATUS = {401, 429, 500, 502, 503, 504}

PARTIAL_CSV = ACLED_PATH / "acled_raw_events.partial.csv"
STATE_FILE = ACLED_PATH / "acled_download_state.json"


def get_access_token():
    with open(PASSWORDS_FILE) as f:
        pws = json.load(f)
    response = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "username": pws["acled"]["username"],
            "password": pws["acled"]["pw"],
            "grant_type": "password",
            "client_id": "acled",
        },
    )
    if response.status_code != 200:
        raise RuntimeError(f"ACLED auth failed: {response.status_code} {response.text}")
    return response.json()["access_token"]


def query_params():
    return {
        "_format": "json",
        "limit": PAGE_SIZE,
        "fields": FIELDS,
        "event_date": f"{START_DATE}|{END_DATE}",
        "event_date_where": "BETWEEN",
        "event_type": "|".join(EVENT_TYPES),
    }


def fetch_page(token, cursor):
    """One page from the given cursor. Returns (events, next_cursor, token)."""
    for attempt in range(len(RETRY_WAITS_S) + 1):
        response = requests.get(READ_URL, headers={"Authorization": f"Bearer {token}"},
                                params={**query_params(), "cursor": cursor})
        if response.status_code == 200:
            body = response.json()
            if "next_cursor" not in body:
                raise RuntimeError("ACLED response has no 'next_cursor' - cursor pagination "
                                   "not supported for this request?")
            return body.get("data", []), body["next_cursor"], token
        if response.status_code not in RETRY_STATUS or attempt == len(RETRY_WAITS_S):
            raise RuntimeError(
                f"ACLED error {response.status_code}: {response.text}\n"
                f"Progress is saved; rerun the script later to continue.")
        wait = RETRY_WAITS_S[attempt]
        print(f"\n  WARNING: ACLED error {response.status_code}, retrying in {wait // 60} min "
              f"({attempt + 1}/{len(RETRY_WAITS_S)})")
        time.sleep(wait)
        token = get_access_token()


def load_state():
    """Cursor and event count of an interrupted run of the same query, if any."""
    if not STATE_FILE.exists():
        return 0, 0
    state = json.loads(STATE_FILE.read_text())
    if state["query"] != query_params():
        raise RuntimeError(
            f"{STATE_FILE.name} belongs to a different query (e.g. changed YEARS). "
            f"Delete it and {PARTIAL_CSV.name} to start over.")
    print(f"  Resuming at cursor {state['cursor']} ({state['n_events']:,} events so far)")
    return state["cursor"], state["n_events"]


def save_state(cursor, n_events):
    STATE_FILE.write_text(json.dumps(
        {"query": query_params(), "cursor": cursor, "n_events": n_events}, indent=2))


def main():
    header("ACLED download", {"Period": f"{START_DATE} to {END_DATE}",
                              "Event types": ", ".join(EVENT_TYPES)})
    if ACLED_RAW_EVENTS.exists():
        print(f"  Already exists: {ACLED_RAW_EVENTS.name}")
        return
    ACLED_PATH.mkdir(parents=True, exist_ok=True)

    cursor, n_events = load_state()
    if cursor == 0 and PARTIAL_CSV.exists():
        PARTIAL_CSV.unlink()           # partial file without state: start over
    token = get_access_token()

    while cursor is not None:
        events, next_cursor, token = fetch_page(token, cursor)
        if events:
            pd.DataFrame(events).to_csv(PARTIAL_CSV, mode="a", index=False,
                                        header=not PARTIAL_CSV.exists())
            n_events += len(events)
        print(f"  Events so far: {n_events:,}", end="\r", flush=True)
        cursor = next_cursor
        save_state(cursor, n_events)
        if cursor is not None:
            time.sleep(PAUSE_S)

    print()
    df = pd.read_csv(PARTIAL_CSV)
    df = df.drop_duplicates(subset="event_id_cnty")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["event_date"] = pd.to_datetime(df["event_date"])
    df.to_csv(ACLED_RAW_EVENTS, index=False)
    PARTIAL_CSV.unlink()
    STATE_FILE.unlink()
    print(f"  Saved: {ACLED_RAW_EVENTS.name} ({len(df):,} events)")
    done(ACLED_RAW_EVENTS)


if __name__ == "__main__":
    main()
