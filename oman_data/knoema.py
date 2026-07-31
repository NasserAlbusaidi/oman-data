"""Thin client for data.gov.om's Knoema REST API.

The portal's frontend calls /api/1.0/* with a public client_id embedded in
the homepage HTML; without it every endpoint returns 403. The key is not a
credential (it ships in public HTML) but it can rotate — on 403 we re-scrape
it once and retry. Responses can be paged via continuationToken; none of our
filtered queries page today, so paging is refused loudly rather than half-read.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.gov.om"
CLIENT_ID = "EZj54KGFo3rzIvnLczrElvAitEyU28DGw9R73tif"
_UA = "oman-data/0.1 (+https://github.com/NasserAlbusaidi/oman-data)"
_CLIENT_ID_RE = re.compile(r"client_id=([A-Za-z0-9]{20,})")


class KnoemaError(Exception):
    pass


def resolve_client_id(session: requests.Session | None = None) -> str:
    s = session or requests.Session()
    r = s.get(f"{BASE}/", headers={"User-Agent": _UA}, timeout=60)
    r.raise_for_status()
    m = _CLIENT_ID_RE.search(r.text)
    if not m:
        raise KnoemaError("could not scrape client_id from portal homepage")
    return m.group(1)


def _post_raw(session, dataset: str, filters: list[dict],
              frequencies: list[str], client_id: str):
    return session.post(
        f"{BASE}/api/1.0/data/raw",
        params={"client_id": client_id},
        json={"Dataset": dataset, "Filter": filters, "Frequencies": frequencies},
        headers={"User-Agent": _UA},
        timeout=120,
    )


def fetch_raw(dataset: str, filters: list[dict], frequencies: list[str],
              raw_dir: Path, filename: str,
              session: requests.Session | None = None) -> Path:
    s = session or requests.Session()
    r = _post_raw(s, dataset, filters, frequencies, CLIENT_ID)
    if r.status_code == 403:
        r = _post_raw(s, dataset, filters, frequencies, resolve_client_id(s))
    r.raise_for_status()
    payload = r.json()
    if payload.get("continuationToken"):
        raise KnoemaError(f"{dataset}: response is paged — narrow the filter")
    if not payload.get("data"):
        raise KnoemaError(f"{dataset}: empty data payload")
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / filename
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out


def dimension_members(dataset: str, dimension: str,
                      session: requests.Session | None = None) -> dict[str, int]:
    s = session or requests.Session()
    url = f"{BASE}/api/1.0/meta/dataset/{dataset}/dimension/{dimension}"
    r = s.get(url, params={"client_id": CLIENT_ID},
              headers={"User-Agent": _UA}, timeout=60)
    if r.status_code == 403:
        r = s.get(url, params={"client_id": resolve_client_id(s)},
                  headers={"User-Agent": _UA}, timeout=60)
    r.raise_for_status()
    return {i["name"]: i["key"] for i in r.json().get("items", [])}


def iter_series(payload: dict) -> list[dict]:
    return payload["data"]


def dim_name(row: dict, dimension_id: str) -> str:
    member = row.get(dimension_id)
    if isinstance(member, str):
        return member
    if isinstance(member, dict) and "name" in member:
        return member["name"]
    raise KnoemaError(f"row has no readable member for dimension {dimension_id!r}")


def monthly_periods(start_date: str, n: int) -> list[str]:
    start = pd.Period(start_date[:7], freq="M")
    return [str(start + i) for i in range(n)]


def annual_periods(start_date: str, n: int) -> list[int]:
    start = int(start_date[:4])
    return [start + i for i in range(n)]
