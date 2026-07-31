"""Fetch monthly hotel-sector totals from the data.gov.om Tourism dataset.

``dedblxg`` is a twelve-dimension cube (region, nationality, hotel class,
gender, age group, point of entry, ...). Asking for the three indicators alone
returns 828 series — every governorate, guest nationality and hotel class
crossed in — so the query also pins ``regions``, ``nationality`` and
``classification-of-hotels`` to their total members. The other eight dimensions
carry a single "Total" member for these indicators and are left unfiltered.

Occupancy comes from ``Room Occupancy Ratio`` rather than the dataset's other
percentage member, ``Occupancy Rate (%)``: same figure, but rounded, starting
2014-01 instead of 2012-01, and published only under the "HOTELS CLASSIFIED
(3 - 5 ) STARS" hotel class, which the total-only filter above excludes.
"""

from __future__ import annotations

from pathlib import Path

from oman_data import knoema

DATASET = "dedblxg"

# indicators: Number of Guests, Room Occupancy Ratio, Hotels Revenues
INDICATOR_MEMBERS: list[int] = [1000210, 1000310, 1000430]
TOTAL_MEMBER = 1000000  # "Oman" for regions; "Total" for the other two


def fetch(raw_dir: Path) -> Path:
    assert len(INDICATOR_MEMBERS) == 3, "pin the three indicator keys first"
    return knoema.fetch_raw(
        DATASET,
        [
            {"DimensionId": "indicators", "Members": INDICATOR_MEMBERS},
            {"DimensionId": "regions", "Members": [TOTAL_MEMBER]},
            {"DimensionId": "nationality", "Members": [TOTAL_MEMBER]},
            {"DimensionId": "classification-of-hotels", "Members": [TOTAL_MEMBER]},
        ],
        ["M"],
        raw_dir,
        "tourism.json",
    )
