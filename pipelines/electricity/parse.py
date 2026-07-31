"""Tidy annual electricity production/consumption from data.gov.om.

Two annual series pivoted wide to one row per year; years missing either
series are dropped rather than published with nulls (the validation gate
forbids nulls, deliberately) — which is what trims the output to 2004
onwards even though consumption reaches back to 2002.

Source quirks pinned at discovery (2026-07-31, dataset OMELCT2016):

* The dimension is ``indicator``, singular, and the member names are the
  source's own — including the unbalanced parenthesis in
  ``(Total production of electricity (GW/H)``. Do not "fix" that typo: it is
  the string the API returns, and normalising it here would stop matching.
* Production is the *gross* series (all generation before the plants' own
  consumption is deducted). The dataset also carries ``Net production of
  electricity (GW/H)``; it is a different, smaller number and is not used.
* ``OMELCT2016`` is a five-dimension cube. Asking for the two indicators
  alone returns 129 series — every governorate and every one of ~105 power
  stations crossed in — so ``fetch.py`` pins ``region`` and ``station`` to
  their total members. One series row per indicator is therefore expected
  here, and a second row for the same indicator raises instead of silently
  blending two breakdowns into one column. Every non-indicator dimension on
  the row is checked for a total-shaped member, so a filter drift on a
  dimension nobody pinned (``networks``, ``opwp-purchases``, or one added
  upstream later) fails loudly rather than publishing a slice as the total.
* Values are gigawatt-hours as published (``unit`` "Gw/H", ``scale`` 1), so
  there is no scale factor. Rows arriving with another unit or scale raise,
  because the ``_gwh`` column names would then be lies.
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
EXPECTED_UNIT = "gw/h"  # normalized; values are GWh already, no scale factor
EXPECTED_SCALE = 1

# normalized member name -> output column; exact names pinned at discovery
_INDICATORS = {
    "(total production of electricity (gw/h)": "production_gwh",
    "consm. of electric power (gw/h)": "consumption_gwh",
}

_TOTAL_OK = {"total", "oman", "sultanate of oman"}

# the portal names this dimension "indicator", singular — unlike Tourism's
_INDICATOR_DIM = "indicator"


def _norm(name: str) -> str:
    return " ".join(name.lower().split())


def _check_totals(row: dict) -> None:
    """Every dimension other than the indicator must sit on its total member.

    ``fetch.py`` pins the two dimensions that actually fan out; this catches
    the rest — including a dimension added upstream after this was written —
    so a governorate or power-station slice can never be published as the
    national total.
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
            f"indicator {name!r} declares frequency {row.get('frequency')!r}, "
            f"expected {FREQUENCY_ANNUAL!r} — year labels would be wrong"
        )
    values = row["values"]
    if not values:
        raise ValueError(f"indicator {name!r} carries no observations")
    years = knoema.annual_periods(row["startDate"], len(values))
    declared_end = int(str(row["endDate"])[:4])
    if years[-1] != declared_end:
        raise ValueError(
            f"indicator {name!r} spans {years[0]}..{years[-1]} from "
            f"{len(values)} values but declares endDate {declared_end} — "
            f"the series is truncated or misaligned"
        )
    return years


def parse(raw_path: Path) -> tuple[pd.DataFrame, str]:
    raw_path = Path(raw_path)
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    frames: dict[str, dict[int, float]] = {}
    for row in knoema.iter_series(payload):
        name = _norm(knoema.dim_name(row, _INDICATOR_DIM))
        if name not in _INDICATORS:
            raise ValueError(f"unexpected indicator {name!r} in electricity payload")
        col = _INDICATORS[name]
        if col in frames:
            raise ValueError(
                f"indicator {name!r} arrived twice — the payload carries "
                f"breakdowns, not totals; fix the filter in fetch.py"
            )
        _check_totals(row)
        if _norm(str(row.get("unit"))) != EXPECTED_UNIT or row.get("scale") != EXPECTED_SCALE:
            raise ValueError(
                f"indicator {name!r} publishes unit {row.get('unit')!r} at scale "
                f"{row.get('scale')!r}, expected gigawatt-hours at scale "
                f"{EXPECTED_SCALE} — the '_gwh' columns would be wrong"
            )
        years = _years_for(name, row)
        series: dict[int, float] = {}
        frames[col] = series
        for year, value in zip(years, row["values"]):
            if value is not None:
                series[year] = float(value)
    if set(frames) != set(_INDICATORS.values()):
        raise ValueError(f"missing series in {raw_path.name}: got {sorted(frames)}")
    df = pd.DataFrame(frames).rename_axis("year").reset_index()
    df = df.dropna()  # keep only years where both series exist
    df = (df[["year", "production_gwh", "consumption_gwh"]]
          .astype({"year": int, "production_gwh": float, "consumption_gwh": float})
          .sort_values("year", ignore_index=True))
    if df.empty:
        raise ValueError(f"unexpected layout in {raw_path.name}")
    return df, str(df["year"].max())
