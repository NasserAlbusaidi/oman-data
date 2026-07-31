"""Fetch the NCSI NSDP Merchandise Trade workbook.

NCSI's National Summary Data Page (the IMF e-GDDS national page, linked from
https://www.ncsi.gov.om/NationalStatistics) republishes each series at a fixed,
unversioned filename, so the same URL always serves the newest month.

Chosen over the national data portal's Knoema "Foreign Trade" dataset
(data.gov.om/tkjlhlb), which at recon carried no monthly total-exports series at
all: monthly there covers only imports, non-oil exports and re-exports by H.S.
section from 2014-01, with total merchandise exports published annually only.
NSDP carries all three national totals monthly back to 2006-07.

The response is checked for the ZIP magic bytes that open an .xlsx, so a captive
portal or error page is never archived under an .xlsx name and parsed later.
"""

from __future__ import annotations

from pathlib import Path

import requests

SOURCE_URL = "https://nsdp.ncsi.gov.om/MerchandiseTrade.xlsx"

_UA = {"User-Agent": "oman-data/0.1 (+https://github.com/NasserAlbusaidi/oman-data)"}
_XLSX_MAGIC = b"PK\x03\x04"


def fetch(raw_dir: Path) -> Path:
    assert SOURCE_URL, "SOURCE_URL must be set from source discovery"
    raw_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(SOURCE_URL, timeout=120, headers=_UA)
    r.raise_for_status()
    if not r.content.startswith(_XLSX_MAGIC):
        raise ValueError(f"{SOURCE_URL} did not return an xlsx workbook ({len(r.content)} bytes)")
    out = raw_dir / SOURCE_URL.rstrip("/").split("/")[-1].split("?")[0]
    out.write_bytes(r.content)
    return out
