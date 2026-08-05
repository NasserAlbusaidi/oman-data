"""Fetch Oman's annual GDP at market prices from the data.gov.om National Accounts dataset.

``zoangf`` is a five-dimension cube (regions, indicators, price-type,
economic-activity-of-non-petrol, institutional-sector) carrying 39 national-
accounts indicators. The query pins four of the five: the indicator to "GDP at
Market Prices" (1000010), the region to "Oman" (1000000), and both the
economic-activity and institutional-sector dimensions to their "Total" members,
which is what keeps 53 activities and 8 sectors from crossing in.

Member counts here are whole-dimension counts as the portal's dimension
endpoint returns them, verified live 2026-08-05: regions 78, indicators 39,
price-type 3, economic-activity-of-non-petrol 53, institutional-sector 8. Two
things to know before recounting them. The regions dimension has no "Total"
member at all — "Oman" is the national one, which is why it, not a total, is
what this query pins. And the activity dimension publishes the name
"Telecommunications and Information service activities" twice, under keys
1000300 and 1000310, so anything that keys members by name (including
``knoema.dimension_members``) sees 52 rather than 53.

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
REGION_OMAN = 1000000  # no "Total" region exists in this cube; see the docstring
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
