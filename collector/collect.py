"""
Divvy live-data collector.

Runs every 15 minutes on GitHub Actions (see .github/workflows/collect.yml).
Each run saves one snapshot of every Divvy station (bikes, e-bikes, open docks)
plus the dockless e-bikes parked near Loyola's Lake Shore Campus.

Why: Divvy's trip files record rides, not how many bikes sit at a station.
These snapshots are the only record of station levels over time, and they
become the training and test data for the "empty in 30 minutes?" model.

Uses only the Python standard library, so the job starts fast and has
nothing to install.
"""
import csv
import gzip
import io
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

FEED = os.environ.get("GBFS_URL", "https://gbfs.lyft.com/gbfs/1.1/chi/en")
DATA_DIR = os.environ.get("DATA_DIR", "data")

# Box around Loyola's Lake Shore Campus (41.99922, -87.65796), about 3 km each way.
# Dockless e-bikes are only saved inside it, to keep files small.
BOX = {"lat_min": 41.97, "lat_max": 42.03, "lng_min": -87.69, "lng_max": -87.63}

STATION_FIELDS = [
    "snapshot_utc", "station_id", "num_bikes_available", "num_ebikes_available",
    "num_docks_available", "num_bikes_disabled", "num_docks_disabled",
    "is_installed", "is_renting", "is_returning", "last_reported",
]
FREE_BIKE_FIELDS = ["snapshot_utc", "lat", "lon", "is_reserved", "is_disabled"]
INFO_FIELDS = ["station_id", "name", "lat", "lon", "capacity"]


def fetch(name, tries=3):
    """Download one GBFS feed as JSON. Retries because a single failed
    request should not cost us a whole 15-minute snapshot."""
    url = f"{FEED}/{name}.json"
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "divvy-loyola-class-project"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:  # network hiccup: wait and try again
            print(f"{name}: attempt {attempt} failed: {e}", file=sys.stderr)
            time.sleep(5 * attempt)
    raise RuntimeError(f"could not download {name}")


def append_rows(path, fields, rows):
    """Append rows to a gzipped daily CSV, writing the header only for a new file.
    One file per day keeps the repo tidy and easy to load later."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with gzip.open(path, "at", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerows(rows)


def in_box(lat, lon):
    return BOX["lat_min"] <= lat <= BOX["lat_max"] and BOX["lng_min"] <= lon <= BOX["lng_max"]


def main():
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    day = now.strftime("%Y-%m-%d")

    # 1. Every station's current bikes and docks.
    status = fetch("station_status")
    stations = status["data"]["stations"]
    rows = [{**s, "snapshot_utc": stamp} for s in stations]
    append_rows(f"{DATA_DIR}/station_status/{day}.csv.gz", STATION_FIELDS, rows)

    # 2. Dockless e-bikes near Loyola (the Check-in 2 surprise: ~1 in 5 trips skip stations).
    free = fetch("free_bike_status")["data"]["bikes"]
    near = [{**b, "snapshot_utc": stamp} for b in free
            if in_box(float(b["lat"]), float(b["lon"]))]
    append_rows(f"{DATA_DIR}/free_bikes/{day}.csv.gz", FREE_BIKE_FIELDS, near)

    # 3. Station names, locations and capacity, refreshed once a day (they rarely change).
    info_path = f"{DATA_DIR}/station_information.csv"
    if not os.path.exists(info_path) or now.hour == 0 and now.minute < 15:
        info = fetch("station_information")["data"]["stations"]
        with open(info_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=INFO_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(info)

    empty = sum(1 for s in stations if s.get("is_renting") and s.get("num_bikes_available", 0) == 0)
    print(f"{stamp}: saved {len(stations)} stations ({empty} empty), {len(near)} dockless e-bikes near Loyola")


if __name__ == "__main__":
    main()
