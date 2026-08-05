"""Tidy Oman's quarterly non-oil Producer Price Index into quarter x group rows.

Long format, like ``cpi`` and ``trade``: one measure (an index number) fanning
out over one dimension (the commodity section it prices), so a ``group`` column
carries the five series rather than five columns carrying the same measure five
times. All five series are complete over the same 33 quarters today, so this is
not the ragged-data argument ``gdp`` makes — it is that adding the sixth
section NCSI might publish tomorrow should add rows, not migrate a schema.

Source quirks pinned at discovery (2026-08-05, dataset jhmydsg):

* **The declared unit is ``%`` and these are not percentages.** Every row
  arrives with ``"unit": "%"``, ``"scale": 1``, and the indicator's own Arabic
  metadata says ``"الوحدة": "نسبة مئوية"`` ("percentage") — all visible in the
  committed fixture. They are index points on a 2018 base: the values run from
  67.17 to 139.04, and the four quarters of 2018 average to within 1.14 points
  of 100 in every group, which is what an index does and not what a percentage
  change does (the cube publishes those separately, as "Inflation (%)"). The
  wrong declaration is pinned exactly as it arrives rather than corrected,
  because it is the only declaration there is: a *change* to it means the
  source changed something, which is worth failing over even when the change
  looks like a fix.
* **Base year 2018, checked arithmetically at parse time** — see
  ``base_year_complaints``. The claim lives in the published titles ("2018 =
  100"), the tests only ever see the pinned fixture, and the refresh workflow
  runs no tests at all, so without a runtime guard a silently rebased NCSI
  series would publish under a title that had become a lie. The cube's own
  metadata corroborates the base year independently: its "Weight" series
  declares ``"unit": "Base Year 2018"`` (verified live 2026-08-05; the Weight
  indicator is not in the committed fixture, which pins "Index Value" alone).
* The five commodity member names are the source's own and are the matching
  key, so ``General Price Index_ Non_Oil`` keeps its underscores and the
  sections keep the ordinal prefixes they are numbered with. ``norm_name``
  lowercases and collapses whitespace, so the keys below are those forms.
* ``check_totals`` is called with ``commodities`` as the exempt dimension,
  because here it is ``commodities`` that fans out. ``indicators`` is pinned by
  ``fetch.py`` to a single non-total member and so is removed from the set
  handed to ``check_totals`` (which would otherwise, correctly, reject "Index
  Value" for not being a total) and checked by name on every row instead. That
  leaves ``regions`` as the one dimension ``check_totals`` actually examines
  here — which is the point, because this cube genuinely publishes governorate
  indices (Muscat and Dhofar each return a full 33-quarter series, read live
  2026-08-05), so a filter drift onto one would yield a complete, plausible,
  wrong national series that no range or completeness check could detect. A
  dimension appearing upstream that this pipeline has never seen is caught by
  the same call, via ``dimension_ids``.
* Observations are a bare array with no per-point dates, so the quarter labels
  are walked forward from ``startDate``. Both of the row's own declarations —
  ``frequency`` and ``endDate`` — are checked against that walk by
  ``knoema.periods_for``, so a series that changes frequency or loses leading
  points fails loudly instead of publishing every value under a shifted
  quarter.

Quarter labels are ``str(pd.Period(..., freq="Q"))``, i.e. ``2018Q1``. That
format sorts lexicographically into chronological order for every year this
dataset can reach, which is what lets ``as_of`` be a plain ``max()``.

The guards themselves live in ``oman_data.knoema`` and are shared with the
Electricity, GDP, Oil-and-gas, Tourism and Traffic-accidents pipelines.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from oman_data import knoema

FREQUENCY_QUARTERLY = "Q"

# Normalized, and deliberately "%": see the docstring. These are index points,
# not percentages, but the declaration is the source's and a change to it must
# raise rather than be quietly tolerated.
EXPECTED_UNIT = "%"
EXPECTED_SCALE = 1

# The base year the published titles name. An index is 100 in its base period
# by construction, so the claim is arithmetic and checkable on every refresh.
BASE_YEAR = 2018
BASE_INDEX = 100.0
QUARTERS_PER_YEAR = 4
# Tolerance in index points on the base year's four-quarter mean. The four
# quarterly indices do not average to exactly 100 — the widest miss in the data
# as published on 2026-08-05 is Water at 98.86, i.e. 1.14 points — so this is a
# band, not an equality. 3.0 sits between that 1.14 and the *narrowest* miss a
# rebase would produce: revaluing every series so that some later year averaged
# 100 instead moves at least one group's base-year mean by 6.33 points (2021,
# the closest candidate; 2019 moves it 11.00, 2020 33.19, 2022 21.49, 2023
# 12.82, 2024 15.04, 2025 14.72). So ~2.6x headroom above today's noise and
# ~2.1x below the nearest thing this is meant to catch.
BASE_YEAR_TOL = 3.0

# the fan-out dimension: normalized member name -> group label.
_COMMODITY_DIM = "commodities"
_GROUPS = {
    "general price index_ non_oil": "general_nonoil",
    "1- mining and quarrying": "mining_quarrying",
    "2- manufacturing": "manufacturing",
    "3- electrical energy": "electrical_energy",
    "4- water": "water",
}

# pinned to one member by fetch.py, so it cannot go through check_totals
_INDICATOR_DIM = "indicators"
_INDICATOR = "index value"

# ``regions`` is the only dimension ``check_totals`` examines here, and "Oman"
# is the only member that can be the national figure — this cube's 77 regions
# include no "Total" at all (verified live 2026-08-05). "total" is deliberately
# *not* accepted: the Oil-and-gas cube taught that a region named "Total" is not
# verifiable as Oman, and nothing else in this payload needs the allowance.
_NATIONAL_OK = frozenset({"oman"})

_COLUMNS = ["quarter", "group", "index"]


def base_year_complaints(df: pd.DataFrame) -> list[str]:
    """Why the base year named in the titles is not evidenced by the data.

    One complaint per offending group, empty when the "2018 = 100" claim holds
    for every group published. A group must carry all four quarters of the base
    year — an index whose base period is absent cannot be checked against its
    own base at all — and their mean must sit within ``BASE_YEAR_TOL`` of 100.
    """
    complaints: list[str] = []
    in_base_year = df["quarter"].str.startswith(f"{BASE_YEAR}Q")
    for group in sorted(set(df["group"])):
        base = df.loc[in_base_year & (df["group"] == group), "index"]
        if len(base) != QUARTERS_PER_YEAR:
            complaints.append(
                f"{group} carries {len(base)} of the {QUARTERS_PER_YEAR} "
                f"{BASE_YEAR} quarters"
            )
            continue
        mean = float(base.mean())
        if abs(mean - BASE_INDEX) > BASE_YEAR_TOL:
            complaints.append(
                f"{group} averages {mean:.2f} over {BASE_YEAR}, not ~{BASE_INDEX:.0f}"
            )
    return complaints


def parse(raw_path: Path) -> tuple[pd.DataFrame, str]:
    raw_path = Path(raw_path)
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    expected_dims = knoema.dimension_ids(payload)
    if _INDICATOR_DIM not in expected_dims:
        raise ValueError(
            f"payload declares no {_INDICATOR_DIM!r} dimension — the fetch pins "
            f"this dataset to one of its four members, so without it these rows "
            f"could be basket weights or year-on-year inflation rather than the "
            f"index itself"
        )
    if _COMMODITY_DIM not in expected_dims:
        raise ValueError(
            f"payload declares no {_COMMODITY_DIM!r} dimension — that is the "
            f"dimension the 'group' column is built from, and it is exempted "
            f"from check_totals, so its absence would otherwise go unnoticed"
        )
    records: list[dict] = []
    seen: set[str] = set()
    for row in knoema.iter_series(payload):
        member = knoema.norm_name(knoema.dim_name(row, _COMMODITY_DIM))
        if member not in _GROUPS:
            raise ValueError(
                f"unexpected commodity {member!r} in ppi payload — this cube "
                f"carries 30 of them and only the general index and its four "
                f"sections are published here; the other 25 are subgroups "
                f"nested inside those sections"
            )
        group = _GROUPS[member]
        if group in seen:
            raise ValueError(
                f"commodity group {group!r} arrived twice — the payload carries "
                f"breakdowns, not one series per section; fix the filter in "
                f"fetch.py"
            )
        seen.add(group)
        indicator = knoema.norm_name(knoema.dim_name(row, _INDICATOR_DIM))
        if indicator != _INDICATOR:
            raise ValueError(
                f"series carries indicator {indicator!r}, expected "
                f"{_INDICATOR!r} — this cube's other indicators are basket "
                f"weights and year-on-year inflation, neither of which is an "
                f"index level"
            )
        knoema.check_totals(row, _COMMODITY_DIM, expected_dims - {_INDICATOR_DIM},
                            _NATIONAL_OK)
        if (knoema.norm_name(row.get("unit")) != EXPECTED_UNIT
                or row.get("scale") != EXPECTED_SCALE):
            raise ValueError(
                f"group {group!r} publishes unit {row.get('unit')!r} at scale "
                f"{row.get('scale')!r}, expected {EXPECTED_UNIT!r} at scale "
                f"{EXPECTED_SCALE} — that declaration is wrong (these are index "
                f"points, not percentages) but it is the source's own, so a "
                f"change to it means something upstream moved"
            )
        quarters = knoema.periods_for(row, FREQUENCY_QUARTERLY, label=member)
        for quarter, value in zip(quarters, row["values"]):
            if value is not None:
                records.append({"quarter": str(quarter), "group": group,
                                "index": float(value)})
    missing = sorted(set(_GROUPS.values()) - seen)
    if missing:
        raise ValueError(f"missing commodity group in {raw_path.name}: {missing}")
    df = pd.DataFrame.from_records(records, columns=_COLUMNS)
    if df.empty:
        raise ValueError(f"unexpected layout in {raw_path.name}")
    df = (df.astype({"quarter": str, "group": str, "index": float})
            .sort_values(["quarter", "group"], ignore_index=True))
    complaints = base_year_complaints(df)
    if complaints:
        raise ValueError(
            f"the {BASE_YEAR} base year is not evidenced by the data: "
            f"{'; '.join(complaints)}. Nothing is published: this dataset's "
            f"titles say '{BASE_YEAR} = 100' and would now be a lie. Check the "
            f"cube's Weight series, whose unit names the base year, then update "
            f"BASE_YEAR here and the titles in dataset.yaml"
        )
    return df, str(df["quarter"].max())
