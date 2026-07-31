"""Fetch annual national traffic-accident totals from the data.gov.om Security & Safety dataset.

``gehye`` is a nine-dimension cube (regions, indicators, nationality, gender,
type-of-fire-accidents, fire-by-cause, ambulance-by-cause, accidents-by-causes,
deceased-and-injuries). Asking for the three indicators alone returns 555
series — every governorate and wilayat, all thirteen accident causes, and the
Omani/expatriate, male/female and driver/passenger/pedestrian splits crossed in
— so the query also pins ``regions`` to "Oman" and ``nationality``, ``gender``,
``accidents-by-causes`` and ``deceased-and-injuries`` to their totals, which
brings it back to exactly three. The three fire/ambulance dimensions carry only
their total member for these indicators and are left unfiltered; ``parse.py``
checks every non-indicator dimension on each row anyway, so a breakdown leaking
in raises.

Deaths is the source's ``Deaths`` member — people killed — not its ``Fatalities``
member, which counts accidents that killed someone and is a materially smaller
number (320 vs 371 in 2020). See ``parse.py`` for the arithmetic that pins the
distinction.

Note the dimension is ``indicators``, plural, unlike Electricity's singular
``indicator``; and this cube's national member is ``regions``/"Oman" (key
1000000) while the dimension's separate "Total" member (1000010) carries no
data for these indicators.
"""

from __future__ import annotations

from pathlib import Path

from oman_data import knoema

DATASET = "gehye"

# indicators: Accidents, Injuries, Deaths (NOT Fatalities, 1000100)
INDICATOR_MEMBERS: list[int] = [1000030, 1000120, 1000130]

REGION_OMAN = 1000000  # "Oman" — the region dimension's separate "Total" is 1000010
TOTAL = 1000000  # every other dimension names its aggregate member "Total"

# dimensions that fan out and must be pinned to their total member
_PINNED_TOTALS = ("nationality", "gender", "accidents-by-causes",
                  "deceased-and-injuries")


def fetch(raw_dir: Path) -> Path:
    assert len(INDICATOR_MEMBERS) == 3, "pin the three indicator keys first"
    filters = [
        {"DimensionId": "indicators", "Members": INDICATOR_MEMBERS},
        {"DimensionId": "regions", "Members": [REGION_OMAN]},
    ]
    filters += [{"DimensionId": dim, "Members": [TOTAL]} for dim in _PINNED_TOTALS]
    return knoema.fetch_raw(
        DATASET,
        filters,
        ["A"],
        raw_dir,
        "traffic_accidents.json",
    )
