"""Archive the National Subsidy System homepage, which carries the month's fuel prices.

The prices are in the server-rendered HTML -- no JavaScript execution needed --
inside the ``fuelpricesubsidyvalue`` panel that ``parse.py`` reads. That class
name is the sentinel checked here rather than a looser word like "Fuel", which
also appears in the site's boilerplate and in unrelated advisory tickers: if
the panel is gone the page is worthless to this pipeline and the run should
stop at the boundary, not three layers in.

The curated history lives beside this file in ``prices.csv``; NSS publishes
only the current month.
"""
from __future__ import annotations

from pathlib import Path

import requests

SOURCE_URL = "https://nss.gov.om/site/home?ln=en"
_UA = "oman-data/0.1 (+https://github.com/NasserAlbusaidi/oman-data)"
_PANEL_SENTINEL = b"fuelpricesubsidyvalue"


def fetch(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(SOURCE_URL, headers={"User-Agent": _UA}, timeout=120)
    r.raise_for_status()
    if _PANEL_SENTINEL not in r.content:
        raise ValueError(
            f"{SOURCE_URL} response has no fuel-price panel "
            f"({len(r.content)} bytes) — site redesigned or blocking us")
    out = raw_dir / "nss_home.html"
    out.write_bytes(r.content)
    return out
