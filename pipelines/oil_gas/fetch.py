"""Fetch Oman's annual oil and gas headline series from the data.gov.om Oil and Gas dataset.

``OMOLGS2016`` is an eight-dimension cube (region, indicator,
oil-refineries-in-oman, type-of-petroleum-products, oil-producing-companies,
minerals-type, gas-uses-type, country) carrying 43 indicators. Four of them are
the national headline series published here: average daily crude production
(1000010), average daily crude exports (1000060), average crude price (1000080)
and total natural gas production (1000220).

Two dimensions are pinned and six are left unfiltered — the reverse of the
Electricity pipeline's shape, so both halves need saying:

* ``region`` is pinned to "Oman" (1000000), and *not* because that is what
  keeps the row count down: asking for these four indicators with no region
  filter at all returns the same four rows, every one of them already on
  "Oman" (verified live 2026-08-05). It is pinned because the dimension has 83
  members — every governorate and wilayat among them — so an upstream that
  starts publishing these indicators by region would otherwise blend
  breakdowns into the national rows. The member to pin is "Oman"; the cube
  carries a *second* aggregate named "Total" (1000010) which returns zero rows
  for these four indicators (also verified live 2026-08-05), so asking for
  both would add nothing today and an unverifiable second aggregate later.
* The other six dimensions are left unfiltered because for these indicators
  they already arrive on their aggregate member, and those members are not
  uniformly named: the refineries dimension calls its total "Total Refinery &
  Petroleum IndustriesCo" and the gas-uses dimension calls its own "Total Gas
  Uses". ``parse.py`` checks every non-indicator dimension by name on every row
  through ``knoema.check_totals``, so a breakdown leaking in raises rather than
  being published as the national figure.

Annual only, deliberately. The monthly series exist and are a trap — two of
them change unit regime mid-series without declaring it, and a third starts in
2023; see the docstring in ``parse.py`` for the figures.
"""

from __future__ import annotations

from pathlib import Path

from oman_data import knoema

DATASET = "OMOLGS2016"

# Average daily production of crude oil, Average Daily Exports of Crude Oil.,
# Average Price of Crude Oil, Total Production of  Natural  Gas
INDICATOR_MEMBERS: list[int] = [1000010, 1000060, 1000080, 1000220]
# "Oman" — note this cube's region dimension has a separate "Total" (1000010)
# that publishes nothing for these indicators; see the docstring.
REGION_OMAN = 1000000


def fetch(raw_dir: Path) -> Path:
    return knoema.fetch_raw(
        DATASET,
        [
            {"DimensionId": "indicator", "Members": INDICATOR_MEMBERS},
            {"DimensionId": "region", "Members": [REGION_OMAN]},
        ],
        ["A"],
        raw_dir,
        "oil_gas.json",
    )
