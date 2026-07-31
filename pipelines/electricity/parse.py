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
  blending two breakdowns into one column. ``knoema.check_totals`` then
  requires every dimension the payload declares to be present *and* on its
  total member, so a filter drift on a dimension nobody pinned (``networks``,
  ``opwp-purchases``, or one added upstream later) fails loudly rather than
  publishing a slice as the total.
* Values are gigawatt-hours as published (``unit`` "Gw/H", ``scale`` 1), so
  there is no scale factor. Rows arriving with another unit or scale raise,
  because the ``_gwh`` column names would then be lies.
* Observations are a bare array with no per-point dates, so the year labels
  are walked forward from ``startDate``. Both of the row's own declarations —
  ``frequency`` and ``endDate`` — are checked against that walk by
  ``knoema.periods_for``, so a series that changes frequency or loses leading
  points fails loudly instead of publishing every value under a shifted year.

The guards themselves live in ``oman_data.knoema`` and are shared with the
Tourism and Traffic-accidents pipelines; see ``check_totals`` for the two
holes the private copies used to share (string members skipping the check,
and a dimension going missing unnoticed).
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

_TOTAL_OK = frozenset({"total", "oman", "sultanate of oman"})

# the portal names this dimension "indicator", singular — unlike Tourism's
_INDICATOR_DIM = "indicator"


def parse(raw_path: Path) -> tuple[pd.DataFrame, str]:
    raw_path = Path(raw_path)
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    expected_dims = knoema.dimension_ids(payload)
    frames: dict[str, dict[int, float]] = {}
    for row in knoema.iter_series(payload):
        name = knoema.norm_name(knoema.dim_name(row, _INDICATOR_DIM))
        if name not in _INDICATORS:
            raise ValueError(f"unexpected indicator {name!r} in electricity payload")
        col = _INDICATORS[name]
        if col in frames:
            raise ValueError(
                f"indicator {name!r} arrived twice — the payload carries "
                f"breakdowns, not totals; fix the filter in fetch.py"
            )
        knoema.check_totals(row, _INDICATOR_DIM, expected_dims, _TOTAL_OK)
        if (knoema.norm_name(row.get("unit")) != EXPECTED_UNIT
                or row.get("scale") != EXPECTED_SCALE):
            raise ValueError(
                f"indicator {name!r} publishes unit {row.get('unit')!r} at scale "
                f"{row.get('scale')!r}, expected gigawatt-hours at scale "
                f"{EXPECTED_SCALE} — the '_gwh' columns would be wrong"
            )
        years = knoema.periods_for(row, FREQUENCY_ANNUAL, label=name)
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
