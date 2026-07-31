"""Archive 1991-2020 daily records for the six-station catalog from the
Open-Meteo historical archive (free, no auth, no API key) as one JSON file
keyed by station slug. One HTTP call per station, six calls per run.

Source quirks pinned at discovery (2026-07-31, archive-api.open-meteo.com):

* The archive is *reanalysis*, not station observations: coordinates are
  snapped to the nearest ERA5/ERA5-Land grid cell, and the response echoes the
  cell it actually used (``latitude``/``longitude``/``elevation``). Those echo
  fields are archived verbatim rather than dropped, because they are the only
  record of how far a request drifted from the station it names -- Saiq's
  requested 23.067/57.633 resolves to a 1919 m cell, which is what makes the
  mountain station read as a mountain station.
* ``daily`` is a *columnar* block: one ``time`` array plus one parallel array
  per requested variable, not a list of per-day objects. A request that names
  an unknown variable is rejected with HTTP 400 and a JSON ``reason``, so a
  200 with a ``daily`` block containing every requested key is a real answer.
* ``timezone=Asia/Muscat`` matters for a *daily* aggregation: without it the
  API buckets days on UTC, which cuts Omani days four hours early and shifts
  each daily max/min into the wrong local day.
* 30 years of daily data is ~340 KB per station, so the combined snapshot is
  ~2 MB. That is the price of a reproducible normal; the request is made once
  and the dataset is static (cadence: static, never refreshed by cron).
* The free tier's quota is *weighted*, not one-unit-per-request: a 30-year
  three-variable window is worth many nominal calls, and six of them back to
  back trip the minutely limit -- observed on the fourth station, HTTP 429 with
  ``{"reason": "Minutely API request limit exceeded. Please try again in one
  minute."}``. So the loop paces itself between stations and retries a 429
  after the minute the API asks for. Six stations therefore take minutes, not
  seconds. A 429 that survives ``_MAX_429_RETRIES`` is raised, because a
  partial snapshot would silently publish normals for a subset of the catalog.

If Open-Meteo changes the response shape, every one of these checks raises at
the boundary and the runner leaves the last-good published data in place.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import requests

API = "https://archive-api.open-meteo.com/v1/archive"
STATIONS_CSV = Path(__file__).parent / "stations.csv"
_UA = "oman-data/0.1 (+https://github.com/NasserAlbusaidi/oman-data)"

START_DATE = "1991-01-01"
END_DATE = "2020-12-31"
DAILY_VARS = ("temperature_2m_max", "temperature_2m_min", "precipitation_sum")

# the API's own remedy for a 429 is "try again in one minute"; the pause
# between stations keeps the common case from getting there at all
_RETRY_AFTER_S = 65
_PACE_S = 20
_MAX_429_RETRIES = 6


def _get(params: dict, station: str) -> requests.Response:
    """GET with the free tier's weighted rate limit respected (see module docstring)."""
    for attempt in range(_MAX_429_RETRIES + 1):
        r = requests.get(API, params=params, headers={"User-Agent": _UA}, timeout=180)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        if attempt == _MAX_429_RETRIES:
            raise RuntimeError(
                f"open-meteo still rate-limiting {station} after "
                f"{_MAX_429_RETRIES} retries: {r.text[:200]}")
        print(f"[climate_normals] {station}: rate limited, waiting "
              f"{_RETRY_AFTER_S}s ({attempt + 1}/{_MAX_429_RETRIES})")
        time.sleep(_RETRY_AFTER_S)
    raise AssertionError("unreachable")


def fetch(raw_dir: Path) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    combined: dict[str, dict] = {}
    with STATIONS_CSV.open(encoding="utf-8", newline="") as f:
        stations = list(csv.DictReader(f))
    if not stations:
        raise ValueError(f"{STATIONS_CSV.name} has no station rows")
    slugs = [s["station"] for s in stations]
    if len(set(slugs)) != len(slugs):
        raise ValueError(f"{STATIONS_CSV.name} has duplicate station slugs: {slugs}")
    for i, row in enumerate(stations):
        if i:
            time.sleep(_PACE_S)
        r = _get({
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "start_date": START_DATE,
            "end_date": END_DATE,
            "daily": ",".join(DAILY_VARS),
            "timezone": "Asia/Muscat",
        }, row["station"])
        payload = r.json()
        daily = payload.get("daily")
        if not isinstance(daily, dict):
            raise ValueError(f"open-meteo response for {row['station']} has no "
                             f"daily block: {str(payload)[:400]}")
        missing = [v for v in ("time", *DAILY_VARS) if v not in daily]
        if missing:
            raise ValueError(f"open-meteo daily block for {row['station']} is "
                             f"missing {missing} — variable names renamed?")
        n = len(daily["time"])
        bad = {v: len(daily[v]) for v in DAILY_VARS if len(daily[v]) != n}
        if bad:
            raise ValueError(f"open-meteo daily arrays for {row['station']} are "
                             f"ragged: time={n}, {bad}")
        combined[row["station"]] = payload
    out = raw_dir / "openmeteo_1991_2020.json"
    out.write_text(json.dumps(combined, ensure_ascii=False), encoding="utf-8")
    return out
