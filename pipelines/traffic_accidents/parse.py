"""Tidy annual national traffic-accident counts (ROP data) from data.gov.om.

Long format (year, metric, count) because the three series end in different
years — accidents ran to 2024, deaths to 2025, injuries stopped in 2021 —
and the validation gate rightly refuses the nulls a wide layout would need.
The injuries series going stale upstream is preserved as-is: honest data
beats an invented tail.

Source quirks pinned at discovery (2026-07-31, dataset gehye):

* The dimension is ``indicators``, plural — unlike Electricity's singular
  ``indicator``.
* ``Deaths`` is the series to publish, not ``Fatalities``. They are different
  things: ``Deaths`` counts *people* killed (371 in 2020), ``Fatalities``
  counts *accidents* that killed someone (320 in 2020 — exactly 23.9% of that
  year's 1,341 accidents, which is what the source's own ``% Accidents
  Fatalities`` series reports). Publishing ``Fatalities`` under the name
  "deaths" would understate road deaths by roughly 14%.
* ``gehye`` is a nine-dimension cube. Asking for the three indicators alone
  returns 555 series — every governorate and wilayat, every accident cause,
  and nationality/gender/road-user splits crossed in — so ``fetch.py`` pins
  ``regions``, ``nationality``, ``gender``, ``accidents-by-causes`` and
  ``deceased-and-injuries`` to their totals, which brings it back to exactly
  three. One series row per metric is therefore expected here, and a second
  row for the same metric raises instead of silently blending two breakdowns.
  Every non-indicator dimension on the row is checked for a total-shaped
  member anyway, so a drift on a dimension nobody pinned (the fire and
  ambulance dimensions, or one added upstream later) fails loudly rather
  than publishing a slice as the national total.
* The national member of ``regions`` is named ``Oman``; the dimension's
  separate ``Total`` member carries no data for these indicators. Same
  quirk as the Electricity dataset.
* Values are plain counts (``unit`` "Number", ``scale`` 1), so there is no
  scale factor. Rows arriving with another unit or scale raise, because the
  ``count`` column would then be a different quantity entirely.
* Observations are a bare array with no per-point dates, so the year labels
  are walked forward from ``startDate``. Both of the row's own declarations —
  ``frequency`` and ``endDate`` — are checked against that walk, so a series
  that changes frequency or loses leading points fails loudly instead of
  publishing every value under a shifted year.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from oman_data import knoema

FREQUENCY_ANNUAL = "A"
EXPECTED_UNIT = "number"  # normalized; values are plain counts, no scale factor
EXPECTED_SCALE = 1

# normalized member name -> metric slug; exact names pinned at discovery
_METRICS = {
    "accidents": "accidents",
    "deaths": "deaths",
    "injuries": "injuries",
}

# "oman" is this dataset's national member; "total" is how every other
# dimension names its own aggregate
_TOTAL_OK = {"total", "oman", "sultanate of oman"}

_INDICATOR_DIM = "indicators"


def _norm(name: str) -> str:
    return " ".join(name.lower().split())


def _check_totals(row: dict) -> None:
    """Every dimension other than the indicator must sit on its total member.

    ``fetch.py`` pins the five dimensions that actually fan out; this catches
    the rest — including a dimension added upstream after this was written —
    so a governorate, an accident cause or a gender slice can never be
    published as the national total.
    """
    for dim, member in row.items():
        if dim == _INDICATOR_DIM or not isinstance(member, dict):
            continue
        name = _norm(knoema.dim_name(row, dim))
        if name not in _TOTAL_OK:
            raise ValueError(
                f"dimension {dim!r} is on member {name!r}, not a total — "
                f"the fetch filter drifted and this is a breakdown, not Oman"
            )


def _years_for(name: str, row: dict) -> list[int]:
    """Label a series' observations, checked against the row's own declarations.

    ``annual_periods`` just walks forward from ``startDate``, so a series that
    switched frequency or lost leading observations would be relabelled
    silently and published under confidently wrong years. The payload states
    both ``frequency`` and ``endDate``; both must agree with the walk.
    """
    if row.get("frequency") != FREQUENCY_ANNUAL:
        raise ValueError(
            f"metric {name!r} declares frequency {row.get('frequency')!r}, "
            f"expected {FREQUENCY_ANNUAL!r} — year labels would be wrong"
        )
    values = row["values"]
    if not values:
        raise ValueError(f"metric {name!r} carries no observations")
    years = knoema.annual_periods(row["startDate"], len(values))
    declared_end = int(str(row["endDate"])[:4])
    if years[-1] != declared_end:
        raise ValueError(
            f"metric {name!r} spans {years[0]}..{years[-1]} from "
            f"{len(values)} values but declares endDate {declared_end} — "
            f"the series is truncated or misaligned"
        )
    return years


def parse(raw_path: Path) -> tuple[pd.DataFrame, str]:
    raw_path = Path(raw_path)
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    records: list[tuple[int, str, int]] = []
    seen: set[str] = set()
    for row in knoema.iter_series(payload):
        name = _norm(knoema.dim_name(row, _INDICATOR_DIM))
        if name not in _METRICS:
            raise ValueError(f"unexpected indicator {name!r} in accidents payload")
        metric = _METRICS[name]
        if metric in seen:
            raise ValueError(
                f"metric {name!r} arrived twice — the payload carries "
                f"breakdowns, not totals; fix the filter in fetch.py"
            )
        seen.add(metric)
        _check_totals(row)
        if _norm(str(row.get("unit"))) != EXPECTED_UNIT or row.get("scale") != EXPECTED_SCALE:
            raise ValueError(
                f"metric {name!r} publishes unit {row.get('unit')!r} at scale "
                f"{row.get('scale')!r}, expected plain counts at scale "
                f"{EXPECTED_SCALE} — the 'count' column would be wrong"
            )
        for year, value in zip(_years_for(name, row), row["values"]):
            if value is not None:
                records.append((year, metric, int(round(float(value)))))
    if seen != set(_METRICS.values()):
        raise ValueError(
            f"missing metric series in {raw_path.name}: got {sorted(seen)}"
        )
    df = pd.DataFrame(records, columns=["year", "metric", "count"])
    if df.empty:
        raise ValueError(f"unexpected layout in {raw_path.name}")
    df = (df.astype({"year": int, "metric": str, "count": int})
            .sort_values(["year", "metric"], ignore_index=True))
    return df, str(df["year"].max())
