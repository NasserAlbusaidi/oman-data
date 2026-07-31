"""Fetch the NCSI Monthly Statistical Bulletin workbook (table 1 = population).

NCSI republishes the bulletin every month under a new timestamped filename, so the
publications page is scraped for the current English workbook URL. SOURCE_URL is the
issue that was verified by hand (July 2026) and is used as the fallback when the
listing page changes shape — it stays reachable, so a scrape failure degrades to a
stale-but-honest fetch rather than a crash.
"""

from pathlib import Path
import re

import requests

LISTING_URL = "https://www.ncsi.gov.om/publications"
SOURCE_URL = (
    "https://api.ncsi.gov.om/uploads/xls/"
    "monthly_statistical_bulletin___july_2026_1784800254.xlsx"
)

_XLSX = re.compile(
    r"https://api\.ncsi\.gov\.om/uploads/xls/monthly_statistical_bulletin[A-Za-z0-9_.\-]*\.xlsx"
)
_UA = {"User-Agent": "oman-data/0.1 (+https://github.com/NasserAlbusaidi/oman-data)"}


def latest_workbook_url() -> str:
    """Newest English bulletin workbook on the publications page, else SOURCE_URL."""
    try:
        r = requests.get(LISTING_URL, timeout=60, headers=_UA)
        r.raise_for_status()
    except requests.RequestException:
        return SOURCE_URL
    urls = [u for u in _XLSX.findall(r.text) if "_ar_" not in u]
    if not urls:
        return SOURCE_URL
    # The trailing number is an upload timestamp; the largest is the newest issue.
    return max(urls, key=lambda u: int(re.search(r"_(\d+)\.xlsx$", u).group(1)))


def fetch(raw_dir: Path) -> Path:
    assert SOURCE_URL, "SOURCE_URL must be set from source discovery"
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = latest_workbook_url()
    r = requests.get(url, timeout=120, headers=_UA)
    r.raise_for_status()
    out = raw_dir / url.rstrip("/").split("/")[-1].split("?")[0]
    out.write_bytes(r.content)
    return out
