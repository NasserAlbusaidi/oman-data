"""Fetch the NCSI NSDP Consumer Price Index workbook.

NCSI's National Summary Data Page (the IMF e-GDDS national page, linked from
https://www.ncsi.gov.om/NationalStatistics) republishes each series at a fixed,
unversioned filename, so unlike the Monthly Statistical Bulletin there is no
timestamped URL to resolve — the same URL always serves the newest month.

The response is checked for the ZIP magic bytes that open an .xlsx, so a captive
portal or error page is never archived under an .xlsx name and parsed later.
"""

from pathlib import Path

import requests

SOURCE_URL = "https://nsdp.ncsi.gov.om/ConsumerPriceIndex.xlsx"

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
