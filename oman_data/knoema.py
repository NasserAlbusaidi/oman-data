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


class KnoemaError(ValueError):
    """A payload that cannot be trusted.

    Subclasses ``ValueError`` because every use is "this data is not what it
    claims to be", and the pipelines' callers and tests treat parse failures
    as ``ValueError`` — a separate hierarchy would let a guard failure slip
    past an ``except ValueError`` that was written to catch exactly this.
    """


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


def norm_name(name: str) -> str:
    """Member names as compared: lowercased, whitespace collapsed.

    The portal is inconsistent about casing and padding ("Total", "total ",
    "TOTAL"), so every name comparison in the pipelines goes through this.
    """
    return " ".join(str(name).lower().split())


def dimension_ids(payload: dict) -> set[str]:
    """The dimension ids the payload itself declares.

    ``dimensionFields`` maps each dimension id to the list of member-field
    descriptors for that dimension (``id``, ``iso``, the Arabic ``اسم``, ...),
    so its keys are the authoritative dimension set — including dimensions
    added upstream after a pipeline was written. Driving ``check_totals`` from
    this, rather than from whatever keys happen to be on a row, is what makes
    a *missing* dimension detectable at all.
    """
    fields = payload.get("dimensionFields")
    if not fields:
        raise KnoemaError(
            "payload declares no dimensionFields — the expected dimension set "
            "is unknown, so no series can be verified as a national total"
        )
    if not isinstance(fields, dict):
        raise KnoemaError(
            f"dimensionFields is {type(fields).__name__}, expected an object "
            f"keyed by dimension id — the payload layout changed"
        )
    return set(fields)


def check_totals(row: dict, indicator_dim: str, expected_dims: set[str],
                 total_names: frozenset[str] | set[str]) -> None:
    """Every dimension but the indicator must be present and on its total.

    The fetch filters pin the dimensions that actually fan out; this catches
    the rest, so a governorate, a power station or an accident-cause slice can
    never be published as the national total. Two holes this deliberately
    closes, both present in the private copies this replaced:

    * **String members.** ``dim_name`` accepts a bare-string member, but the
      old guards skipped anything that was not a ``dict`` — so a payload
      encoding ``"regions": "Muscat"`` bypassed the check entirely.
    * **Missing dimensions.** The old guards asked "is any *present* member a
      non-total", which a row that simply dropped a dimension passed. Checking
      against ``expected_dims`` (from ``dimension_ids``) makes absence loud.
    """
    for dim in sorted(expected_dims):
        if dim == indicator_dim:
            continue
        if dim not in row:
            raise KnoemaError(
                f"dimension {dim!r} is missing from the series row — the "
                f"payload declares it, so this series is an unknown slice, "
                f"not a verified total"
            )
        name = norm_name(dim_name(row, dim))
        if name not in total_names:
            raise KnoemaError(
                f"dimension {dim!r} is on member {name!r}, not a total — "
                f"the fetch filter drifted and this is a breakdown, not Oman"
            )


def periods_for(row: dict, frequency: str,
                label: str = "") -> list[str] | list[int]:
    """Label a series' observations, checked against the row's own declarations.

    Observations arrive as a bare array with no per-point dates, so labels are
    walked forward from ``startDate``. A series that switched frequency or lost
    leading observations would be relabelled silently and published under
    confidently wrong dates — so both of the row's own declarations,
    ``frequency`` and ``endDate``, must agree with the walk.

    ``label`` is only for error context (the caller knows the indicator name;
    this helper does not).
    """
    what = f"series {label!r}" if label else "series"
    declared_freq = row.get("frequency")
    if declared_freq != frequency:
        raise KnoemaError(
            f"{what} declares frequency {declared_freq!r}, expected "
            f"{frequency!r} — the period labels would be wrong"
        )
    if frequency not in ("M", "Q", "A"):
        raise KnoemaError(
            f"{what} asks for frequency {frequency!r}; only 'M', 'Q' and 'A' "
            f"periods are labelled here"
        )
    for key in ("values", "startDate", "endDate"):
        if key not in row:
            raise KnoemaError(f"{what} has no {key!r} — the payload layout changed")
    values = row["values"]
    if not values:
        raise KnoemaError(f"{what} carries no observations")
    if frequency == "M":
        periods: list = monthly_periods(str(row["startDate"]), len(values))
        declared_end = str(row["endDate"])[:7]
        walked_end = periods[-1]
    elif frequency == "Q":
        periods = quarterly_periods(str(row["startDate"]), len(values))
        # Monthly and annual labels are prefixes of the declared endDate, so
        # they compare by slicing. A quarterly label is not: the portal declares
        # the *date* the last quarter starts ("2026-01-01" for 2026Q1), and
        # slicing that gives "2026-01" or "2026", neither of which is "2026Q1".
        # Converting the declared date through the same pd.Period the walk uses
        # is what makes the two comparable — and it is robust to the portal
        # switching to the quarter's last day, which lands in the same quarter.
        declared_end = str(pd.Period(str(row["endDate"])[:10], freq="Q"))
        walked_end = periods[-1]
    else:
        periods = annual_periods(str(row["startDate"]), len(values))
        declared_end = str(row["endDate"])[:4]
        walked_end = str(periods[-1])
    if walked_end != declared_end:
        raise KnoemaError(
            f"{what} spans {periods[0]}..{periods[-1]} from {len(values)} "
            f"values but declares endDate {declared_end} — the series is "
            f"truncated or misaligned"
        )
    return periods


def monthly_periods(start_date: str, n: int) -> list[str]:
    start = pd.Period(start_date[:7], freq="M")
    return [str(start + i) for i in range(n)]


def quarterly_periods(start_date: str, n: int) -> list[str]:
    """Quarter labels as pandas spells them: "2018Q1", "2018Q2", ...

    The label format is ``str(pd.Period(..., freq="Q"))`` rather than anything
    hand-rolled, so the labels the walk produces and the label the declared
    endDate is converted to in ``periods_for`` come from one implementation and
    cannot disagree about, say, a fiscal-year quarter convention.
    """
    start = pd.Period(start_date[:10], freq="Q")
    return [str(start + i) for i in range(n)]


def annual_periods(start_date: str, n: int) -> list[int]:
    start = int(start_date[:4])
    return [start + i for i in range(n)]
