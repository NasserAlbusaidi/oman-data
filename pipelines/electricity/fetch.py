"""Fetch annual national electricity totals from the data.gov.om Electricity dataset.

``OMELCT2016`` is a five-dimension cube (region, indicator, station, networks,
opwp-purchases). Asking for the two indicators alone returns 129 series — every
governorate and every one of ~105 named power stations crossed in — so the query
also pins ``region`` to "Oman" and ``station`` to "Total", which brings it back
to exactly two. ``networks`` and ``opwp-purchases`` carry only their total member
for these two indicators and are left unfiltered; ``parse.py`` checks every
non-indicator dimension on each row anyway, so a breakdown leaking in raises.

Production is the source's gross series, ``(Total production of electricity
(GW/H)`` — the unbalanced parenthesis is the portal's, not a typo here. Its
sibling ``Net production of electricity (GW/H)`` deducts the plants' own
consumption and is not used.

Annual only. The source does publish a monthly production series (2014-01 to
2024-12), but there is no monthly consumption at all and the monthly production
lags the annual one by a full year, so a monthly cadence could not carry this
dataset's two columns.
"""

from __future__ import annotations

from pathlib import Path

from oman_data import knoema

DATASET = "OMELCT2016"

# indicators: (Total production of electricity (GW/H), Consm. of Electric Power (GW/H)
INDICATOR_MEMBERS: list[int] = [1000010, 1000080]
REGION_OMAN = 1000000  # "Oman" — note the region dimension's separate "Total" is 1000010
STATION_TOTAL = 1000000


def fetch(raw_dir: Path) -> Path:
    assert len(INDICATOR_MEMBERS) == 2, "pin the two indicator keys first"
    return knoema.fetch_raw(
        DATASET,
        [
            {"DimensionId": "indicator", "Members": INDICATOR_MEMBERS},
            {"DimensionId": "region", "Members": [REGION_OMAN]},
            {"DimensionId": "station", "Members": [STATION_TOTAL]},
        ],
        ["A"],
        raw_dir,
        "electricity.json",
    )
