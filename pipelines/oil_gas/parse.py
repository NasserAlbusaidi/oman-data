"""Tidy Oman's annual oil and gas headline series into one row per year.

Wide format, like ``electricity``: the four series are genuinely different
quantities in three different units — barrels a day, dollars a barrel, cubic
feet a year — so they are four columns, not four rows of one measure. Years
missing any of the four are dropped rather than published with nulls (the
validation gate forbids nulls, deliberately), which is what trims the table to
2002-2023 even though natural gas reaches 2025.

Source quirks pinned at discovery (2026-08-05, dataset OMOLGS2016):

* **The units differ per indicator**, which is why this pipeline carries a map
  of expected unit and scale keyed by indicator instead of Electricity's single
  ``EXPECTED_UNIT``. Getting one of them wrong would not corrupt the numbers so
  much as make a column *name* a lie, so a row whose declaration does not match
  its indicator's entry raises.
* Member names are the source's own and are the matching key, so the trailing
  period in ``Average Daily Exports of Crude Oil.`` and the double spaces in
  ``Total Production of  Natural  Gas`` are preserved rather than "fixed" —
  ``knoema.norm_name`` collapses the whitespace, so the keys below are the
  collapsed forms.
* **The gas figure is a yearly total, not a daily rate**, unlike the two oil
  volume series either side of it. Three things say so, none of them a guess:
  the source names those two "Average daily ..." and this one "Total Production
  of ..."; 1,908,026.7 million standard cubic feet in a single *day* would be a
  physically absurd national output, whereas divided across 2023 it is ~5,230 a
  day; and ~5,230 a day is exactly the scale the monthly series for this same
  indicator now runs at — its 2026-05 value is 5,405.5, against an annual 2025
  figure of 2,018,799.2, i.e. 5,531 a day (monthly figures read live
  2026-08-05; they are not in the committed annual fixture). Hence
  ``gas_production_mn_scf`` (per year) against ``crude_production_kbbl_day``
  (per day).
* The cube's indicator-level metadata is not trustworthy and is not used. It
  contradicts itself on the gas unit — English ``"unit": "cubic meter"``,
  Arabic ``"الوحدة": "قدم . مكعب"`` (cubic *foot*) — against the series-level
  ``MNSCF`` that is actually checked here; and the English ``description`` of
  the crude-production indicator is the crude-*price* indicator's description,
  copy-pasted upstream. Both are visible in the committed fixture.
* ``region`` must be "Oman" on every row. The cube has a second aggregate
  region named "Total", which ``check_totals`` would happily accept as a total
  — so the national member is additionally checked by name, because "Total"
  publishes nothing for these indicators today and whatever it published
  tomorrow would not be verifiable as Oman.
* One series row per indicator is expected; a second row for the same indicator
  raises instead of silently overwriting a column with a breakdown.
* Observations are a bare array with no per-point dates, so the year labels are
  walked forward from ``startDate``. Both of the row's own declarations —
  ``frequency`` and ``endDate`` — are checked against that walk by
  ``knoema.periods_for``, so a series that changes frequency or loses leading
  points fails loudly instead of publishing every value under a shifted year.

**Monthly is not a fallback for this cube** (verified live 2026-08-05, so it is
not visible in the committed annual fixture). Monthly crude production runs
2014-01 to 2026-06 under one declared unit, ``BBL(000)``, and switches regime
mid-series with no declaration: it opens at 958,700.0 and closes at 1,145.734.
Monthly gas does the same far more violently under ``MNCM``: 107,139,700,000.0
in 2014-01 against 5,405.5 in 2026-05. Monthly exports only begin in 2023-01.
Only the monthly price series is clean. Annual is the honest cadence here.

The guards themselves live in ``oman_data.knoema`` and are shared with the
Electricity, GDP, Tourism and Traffic-accidents pipelines.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from oman_data import knoema

FREQUENCY_ANNUAL = "A"


class Series(NamedTuple):
    """What one indicator must publish for its column name to be true."""

    column: str
    unit: str  # normalized, as ``knoema.norm_name`` returns it
    scale: int


# normalized indicator name -> output column and its required declaration.
# One map rather than two so a column can never drift away from its unit.
_INDICATORS: dict[str, Series] = {
    "average daily production of crude oil":
        Series("crude_production_kbbl_day", "(000) bbl", 1),
    "average daily exports of crude oil.":
        Series("crude_exports_kbbl_day", "(000) bbl", 1),
    "average price of crude oil":
        Series("crude_price_usd_bbl", "us$/bbl", 1),
    "total production of natural gas":
        Series("gas_production_mn_scf", "mnscf", 1),
}

# This cube names its aggregates four different ways: plain "Total" (country,
# minerals, producing companies, petroleum products), "Oman" (region), and two
# spelled-out totals of its own.
_TOTAL_OK = frozenset({
    "total",
    "oman",
    "total refinery & petroleum industriesco",
    "total gas uses",
})

# the portal names this dimension "indicator", singular — as Electricity does
_INDICATOR_DIM = "indicator"
_REGION_DIM = "region"
_REGION_OMAN = "oman"

_COLUMNS = ["year"] + [s.column for s in _INDICATORS.values()]


def parse(raw_path: Path) -> tuple[pd.DataFrame, str]:
    raw_path = Path(raw_path)
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    expected_dims = knoema.dimension_ids(payload)
    frames: dict[str, dict[int, float]] = {}
    for row in knoema.iter_series(payload):
        name = knoema.norm_name(knoema.dim_name(row, _INDICATOR_DIM))
        spec = _INDICATORS.get(name)
        if spec is None:
            raise ValueError(
                f"unexpected indicator {name!r} in oil_gas payload — this cube "
                f"carries 43 of them and only four are published here"
            )
        if spec.column in frames:
            raise ValueError(
                f"indicator {name!r} arrived twice — the payload carries "
                f"breakdowns, not totals; fix the filter in fetch.py"
            )
        knoema.check_totals(row, _INDICATOR_DIM, expected_dims, _TOTAL_OK)
        region = knoema.norm_name(knoema.dim_name(row, _REGION_DIM))
        if region != _REGION_OMAN:
            raise ValueError(
                f"indicator {name!r} carries region {region!r}, expected "
                f"{_REGION_OMAN!r} — this cube has a second aggregate region "
                f"named 'Total' that publishes nothing for these indicators, "
                f"and a total-of-something-else is not the national figure"
            )
        if (knoema.norm_name(row.get("unit")) != spec.unit
                or row.get("scale") != spec.scale):
            raise ValueError(
                f"indicator {name!r} publishes unit {row.get('unit')!r} at "
                f"scale {row.get('scale')!r}, expected {spec.unit!r} at scale "
                f"{spec.scale} — the {spec.column!r} column would be a lie"
            )
        years = knoema.periods_for(row, FREQUENCY_ANNUAL, label=name)
        series: dict[int, float] = {}
        frames[spec.column] = series
        for year, value in zip(years, row["values"]):
            if value is not None:
                series[year] = float(value)
    missing = sorted({s.column for s in _INDICATORS.values()} - set(frames))
    if missing:
        raise ValueError(f"missing series in {raw_path.name}: {missing}")
    df = pd.DataFrame(frames).rename_axis("year").reset_index()
    df = df.dropna()  # keep only years where all four series exist
    df = (df[_COLUMNS]
          .astype({"year": int} | {s.column: float for s in _INDICATORS.values()})
          .sort_values("year", ignore_index=True))
    if df.empty:
        raise ValueError(f"unexpected layout in {raw_path.name}")
    return df, str(df["year"].max())
