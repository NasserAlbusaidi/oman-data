"""Fetch Oman's annual GDP at market prices from the data.gov.om National Accounts dataset.

``zoangf`` is a five-dimension cube (regions, indicators, price-type,
economic-activity-of-non-petrol, institutional-sector) carrying 39 national-
accounts indicators. The query pins four of the five: the indicator to "GDP at
Market Prices" (1000010), the region to "Oman" (1000000 — this cube's regions
dimension has no separate "Total" member), and both the economic-activity and
institutional-sector dimensions to their "Total" members, which is what keeps
52 activities and 8 sectors from crossing in.

``price-type`` is deliberately left unfiltered: it is this dataset's fan-out
dimension, and its two published members — "Current Prices" and "Constant
Prices" — become the ``price_basis`` column. Its third member, "Total", is a
structural artefact of the cube and carries no series for this indicator; if it
ever arrives, ``parse.py`` raises rather than publishing a meaningless third
basis.

Annual only. Asking this slice for Q or M returns zero rows — NCSI publishes
quarterly national accounts elsewhere in the cube, not for this indicator at
this level of aggregation.
"""

from __future__ import annotations

from pathlib import Path

from oman_data import knoema

DATASET = "zoangf"

INDICATOR_GDP_MARKET_PRICES = 1000010
REGION_OMAN = 1000000  # this cube's regions dimension has no separate "Total"
ACTIVITY_TOTAL = 1000000
SECTOR_TOTAL = 1000000


def fetch(raw_dir: Path) -> Path:
    return knoema.fetch_raw(
        DATASET,
        [
            {"DimensionId": "indicators", "Members": [INDICATOR_GDP_MARKET_PRICES]},
            {"DimensionId": "regions", "Members": [REGION_OMAN]},
            {"DimensionId": "economic-activity-of-non-petrol",
             "Members": [ACTIVITY_TOTAL]},
            {"DimensionId": "institutional-sector", "Members": [SECTOR_TOTAL]},
        ],
        ["A"],
        raw_dir,
        "gdp.json",
    )
