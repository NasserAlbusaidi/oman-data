"""Tidy monthly hotel-sector indicators from the data.gov.om Tourism dataset.

Three series (guests, occupancy %, revenues) pivoted wide to one row per
month; months missing any of the three are dropped rather than published
with nulls (the validation gate forbids nulls, deliberately). Occupancy is
already a percentage; revenue scale pinned at discovery.

Source quirks pinned at discovery (2026-07-31, dataset dedblxg):

* The dataset is a twelve-dimension cube, so ``fetch.py`` pins ``regions``,
  ``nationality`` and ``classification-of-hotels`` to their total members —
  unfiltered, the same three indicators come back 828 times (per governorate,
  per guest nationality, per hotel class). One series row per indicator is
  therefore expected here, and a second row for the same indicator raises
  instead of silently blending two breakdowns into one column.
* The occupancy series is ``Room Occupancy Ratio`` (Oman/Total, monthly from
  2012-01, unrounded). The dataset's other percentage member, ``Occupancy
  Rate (%)``, is the same figure rounded to one decimal, exists only under the
  "HOTELS CLASSIFIED (3 - 5 ) STARS" hotel class, and starts two years later —
  so it is not used.
* ``Hotels Revenues`` carries an empty ``unit`` and ``scale: 1``; the values
  are plain riyals (~1.4e7/month), hence ``REVENUE_SCALE``. Rows arriving with
  any other ``scale`` raise, because that pinned constant would be wrong.
* Observations are a bare array with no per-point dates, so the month labels
  are walked forward from ``startDate``. Both of the row's own declarations —
  ``frequency`` and ``endDate`` — are checked against that walk, so a series
  that changes frequency or loses leading points fails loudly instead of
  publishing every value under a shifted month.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from oman_data import knoema

REVENUE_SCALE = 1e-6  # riyals -> millions OMR; pinned at discovery
FREQUENCY_MONTHLY = "M"

# normalized member name -> output column; exact names pinned at discovery
_INDICATORS = {
    "number of guests": "guests",
    "room occupancy ratio": "occupancy_pct",
    "hotels revenues": "revenue_omr_mn",
}


def _norm(name: str) -> str:
    return " ".join(name.lower().split())


def _months_for(name: str, row: dict) -> list[str]:
    """Label a series' observations, checked against the row's own declarations.

    ``monthly_periods`` just walks forward from ``startDate``, so a series that
    switched frequency or lost leading observations would be relabelled silently
    and published under confidently wrong dates. The payload states both
    ``frequency`` and ``endDate``; both must agree with the walk.
    """
    if row.get("frequency") != FREQUENCY_MONTHLY:
        raise ValueError(
            f"indicator {name!r} declares frequency {row.get('frequency')!r}, "
            f"expected {FREQUENCY_MONTHLY!r} — month labels would be wrong"
        )
    values = row["values"]
    if not values:
        raise ValueError(f"indicator {name!r} carries no observations")
    months = knoema.monthly_periods(row["startDate"], len(values))
    declared_end = str(row["endDate"])[:7]
    if months[-1] != declared_end:
        raise ValueError(
            f"indicator {name!r} spans {months[0]}..{months[-1]} from "
            f"{len(values)} values but declares endDate {declared_end} — "
            f"the series is truncated or misaligned"
        )
    return months


def parse(raw_path: Path):
    raw_path = Path(raw_path)
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    frames: dict[str, dict[str, float]] = {}
    for row in knoema.iter_series(payload):
        name = _norm(knoema.dim_name(row, "indicators"))
        if name not in _INDICATORS:
            raise ValueError(f"unexpected indicator {name!r} in tourism payload")
        col = _INDICATORS[name]
        if col in frames:
            raise ValueError(
                f"indicator {name!r} arrived twice — the payload carries "
                f"breakdowns, not totals; fix the filter in fetch.py"
            )
        if row.get("scale") != 1:
            raise ValueError(
                f"indicator {name!r} has scale {row.get('scale')!r}, expected 1 "
                f"— the pinned unit scaling no longer holds"
            )
        months = _months_for(name, row)
        series: dict[str, float] = {}
        frames[col] = series
        for month, value in zip(months, row["values"]):
            if value is not None:
                series[month] = float(value)
    if set(frames) != set(_INDICATORS.values()):
        raise ValueError(f"missing indicator series in {raw_path.name}: "
                         f"got {sorted(frames)}")
    df = pd.DataFrame(frames).rename_axis("month").reset_index()
    df = df.dropna()  # keep only months where all three series exist
    df["revenue_omr_mn"] = df["revenue_omr_mn"] * REVENUE_SCALE
    df["guests"] = df["guests"].round().astype(int)
    df = (df[["month", "guests", "occupancy_pct", "revenue_omr_mn"]]
          .astype({"month": str, "occupancy_pct": float, "revenue_omr_mn": float})
          .sort_values("month", ignore_index=True))
    if df.empty:
        raise ValueError(f"unexpected layout in {raw_path.name}")
    return df, df["month"].max()
