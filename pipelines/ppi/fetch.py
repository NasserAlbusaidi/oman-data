"""Fetch Oman's quarterly non-oil Producer Price Index from data.gov.om.

``jhmydsg`` is a three-dimension cube (regions, indicators, commodities). The
query pins two of the three and fans out over the third:

* ``regions`` -> "Oman" (1000000). 77 members, verified live 2026-08-05, and
  none of them is called "Total" — the governorates and wilayats make up the
  rest, so "Oman" is the national member this pins. **The governorates really
  do publish**: Muscat and Dhofar each return a full 33-quarter general index
  (96.95 and 97.55 in 2018Q1, read live 2026-08-05), and the cube's own
  description advertises them. That is what makes this pin load-bearing rather
  than defensive — a drift onto Muscat would publish a complete, plausible,
  wrong series that no range or completeness check could tell from the national
  one. The wilayats below them return nothing.
* ``indicators`` -> "Index Value" (1000010). Four members. "Producer price
  indices" (1000000) is a structural parent that returns zero rows (verified
  live 2026-08-05). "Weight" (1000020) is the basket weight, constant through
  the series, not a price. "Inflation (%)" (1000030) is the year-on-year change
  of exactly the series published here — derivable from it, four quarters
  shorter (2019Q1 rather than 2018Q1), and deliberately left upstream.
* ``commodities`` is the fan-out dimension, restricted to five of its 30
  members: the general index and the four sections beneath it. The other 25 are
  subgroups nested inside those sections (divisions of manufacturing, and so
  on) and are out of scope; publishing them alongside their parents would make
  the ``group`` column a mix of levels that cannot be summed or compared.

Quarterly only. Asking this exact slice for A or M returns zero rows (verified
live 2026-08-05) even though the indicator's own metadata advertises
``"frequencies": "Quartarly, Annual"`` — the typo is the source's.
"""

from __future__ import annotations

from pathlib import Path

from oman_data import knoema

DATASET = "jhmydsg"

REGION_OMAN = 1000000  # no "Total" region exists in this cube; see the docstring
INDICATOR_INDEX_VALUE = 1000010

# The general index and its four sections. Keys are the portal's; the names
# they carry are pinned in parse.py, which is where they become group labels.
COMMODITY_SECTIONS = [
    1000000,  # General Price Index_ Non_Oil
    1000010,  # 1- Mining and Quarrying
    1000050,  # 2- Manufacturing
    1000280,  # 3- Electrical Energy
    1000290,  # 4- Water
]


def fetch(raw_dir: Path) -> Path:
    return knoema.fetch_raw(
        DATASET,
        [
            {"DimensionId": "regions", "Members": [REGION_OMAN]},
            {"DimensionId": "indicators", "Members": [INDICATOR_INDEX_VALUE]},
            {"DimensionId": "commodities", "Members": COMMODITY_SECTIONS},
        ],
        ["Q"],
        raw_dir,
        "ppi.json",
    )
