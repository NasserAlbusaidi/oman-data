"""Tidy Oman's annual GDP at market prices into year x price-basis rows.

Long format, like ``cpi`` and ``trade``: one measure (GDP in millions of Omani
Riyals) fanning out over one dimension (the price basis it is valued at), so a
``price_basis`` column carries ``current`` or ``constant`` rather than two
columns carrying the same measure twice. That choice is what makes the ragged
end of the series publishable: NCSI has published 2024 at current prices but
not yet at constant prices, and in long format that year simply has one row
instead of two. A wide table would have to either drop 2024 — the headline
number — or publish it with a null, which the validation gate forbids.

Source quirks pinned at discovery (2026-08-05, dataset zoangf):

* ``price-type`` has three members, not two. "Current Prices" and "Constant
  Prices" carry the series; "Total" is a structural artefact that returns
  nothing for this indicator. A row arriving on it would be a filter or
  upstream change, not a third basis to publish, so it raises.
* Constant prices are at the base year 2018. The cube's own description says
  so ("The new system adopted 2018 as a new base year for (SNA) in Oman
  instead of 2010"), and the data agrees: 2018 is the single year where the
  two bases are equal to the last decimal (35184.0 both ways). The older
  base-2010 national accounts live in a separate dataset (``OMNTAC2016``) and
  are not mixed in here.
* ``check_totals`` is called with ``price-type`` as the exempt dimension,
  because here it is ``price-type`` that fans out — the reverse of the
  Electricity and Tourism pipelines, where the indicator dimension does.
  ``indicators`` is pinned by ``fetch.py`` to a single non-total member, so it
  is removed from the set handed to ``check_totals`` (which would otherwise,
  correctly, reject "GDP at Market Prices" for not being a total) and checked
  by name on every row instead. That pin is the whole dataset: one of the 39
  members of this dimension is GDP at market prices and the rest are other
  aggregates in the same unit, so a filter drift would publish, say, Gross
  National Income under a ``gdp_`` column with nothing looking wrong.
* Values are millions of Omani Riyals as published (``unit`` "Mn.R.O",
  ``scale`` 1), so there is no scale factor. Rows arriving with another unit
  or scale raise, because the ``gdp_mn_omr`` column name would then be a lie.
* Observations are a bare array with no per-point dates, so the year labels
  are walked forward from ``startDate``. Both of the row's own declarations —
  ``frequency`` and ``endDate`` — are checked against that walk by
  ``knoema.periods_for``, so a series that changes frequency or loses leading
  points fails loudly instead of publishing every value under a shifted year.

The guards themselves live in ``oman_data.knoema`` and are shared with the
Electricity, Tourism and Traffic-accidents pipelines.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from oman_data import knoema

FREQUENCY_ANNUAL = "A"
EXPECTED_UNIT = "mn.r.o"  # normalized; values are millions of OMR already
EXPECTED_SCALE = 1

# the fan-out dimension: normalized member name -> price_basis value.
# The cube's third member, "Total", is deliberately absent — see the docstring.
_PRICE_DIM = "price-type"
_PRICE_TYPES = {
    "current prices": "current",
    "constant prices": "constant",
}

# pinned to one member by fetch.py, so it cannot go through check_totals
_INDICATOR_DIM = "indicators"
_INDICATOR = "gdp at market prices"

# "oman" is this cube's national member of ``regions`` — unlike the Electricity
# cube it has no separate "Total" region; the other two dimensions name their
# aggregate "Total".
_TOTAL_OK = frozenset({"total", "oman"})

_COLUMNS = ["year", "price_basis", "gdp_mn_omr"]


def parse(raw_path: Path) -> tuple[pd.DataFrame, str]:
    raw_path = Path(raw_path)
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    expected_dims = knoema.dimension_ids(payload)
    if _INDICATOR_DIM not in expected_dims:
        raise ValueError(
            f"payload declares no {_INDICATOR_DIM!r} dimension — the fetch pins "
            f"this dataset to one of its 39 members, so without it these rows "
            f"are an unknown national-accounts aggregate"
        )
    records: list[dict] = []
    seen: set[str] = set()
    for row in knoema.iter_series(payload):
        member = knoema.norm_name(knoema.dim_name(row, _PRICE_DIM))
        if member not in _PRICE_TYPES:
            raise ValueError(
                f"unexpected price type {member!r} in gdp payload — the cube "
                f"also carries a 'Total' price type that publishes nothing for "
                f"this indicator; a row on it is a filter or upstream change"
            )
        basis = _PRICE_TYPES[member]
        if basis in seen:
            raise ValueError(
                f"price basis {basis!r} arrived twice — the payload carries "
                f"breakdowns, not totals; fix the filter in fetch.py"
            )
        seen.add(basis)
        indicator = knoema.norm_name(knoema.dim_name(row, _INDICATOR_DIM))
        if indicator != _INDICATOR:
            raise ValueError(
                f"series carries indicator {indicator!r}, expected "
                f"{_INDICATOR!r} — the fetch filter drifted onto a different "
                f"national-accounts aggregate published in the same unit"
            )
        knoema.check_totals(row, _PRICE_DIM, expected_dims - {_INDICATOR_DIM},
                            _TOTAL_OK)
        if (knoema.norm_name(row.get("unit")) != EXPECTED_UNIT
                or row.get("scale") != EXPECTED_SCALE):
            raise ValueError(
                f"price basis {basis!r} publishes unit {row.get('unit')!r} at "
                f"scale {row.get('scale')!r}, expected millions of Omani Riyals "
                f"at scale {EXPECTED_SCALE} — the 'gdp_mn_omr' column would be "
                f"a lie"
            )
        years = knoema.periods_for(row, FREQUENCY_ANNUAL, label=member)
        for year, value in zip(years, row["values"]):
            if value is not None:
                records.append({"year": int(year), "price_basis": basis,
                                "gdp_mn_omr": float(value)})
    missing = sorted(set(_PRICE_TYPES.values()) - seen)
    if missing:
        raise ValueError(f"missing price basis in {raw_path.name}: {missing}")
    df = pd.DataFrame.from_records(records, columns=_COLUMNS)
    if df.empty:
        raise ValueError(f"unexpected layout in {raw_path.name}")
    df = (df.astype({"year": int, "price_basis": str, "gdp_mn_omr": float})
            .sort_values(["year", "price_basis"], ignore_index=True))
    return df, str(df["year"].max())
