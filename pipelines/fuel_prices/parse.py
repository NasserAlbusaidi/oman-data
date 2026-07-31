"""Merge the curated fuel-price history with the current month from nss.gov.om.

There is no official machine-readable archive of Oman fuel prices; provenance
is disclosed per row in the ``source`` column (see dataset.yaml notes). The
National Subsidy System homepage is the official current-month figure, and it
publishes only that one month -- ``prices.csv`` carries everything before it.

Source quirks pinned at discovery (2026-07-31, nss.gov.om/site/home?ln=en):

* The page states the prices twice. The **price panel** is the structural one:
  a heading ``Fuel Price - July'26 <span>Baisa/Ltr</span>`` followed by two
  sibling blocks, ``fuelpricesubsidytitle`` holding the labels (M91, M95,
  Diesel) and ``fuelpricesubsidyvalue`` holding the bare numbers. Labels and
  values are *separate elements*, so they are paired by position and the label
  block is the authority on which fuel each number belongs to -- a column
  reorder upstream moves the values with it instead of silently relabelling
  them.
* The **advisory marquee** ("July 2026 Fuel Prices M95 239Bz, M91 229Bz and
  Diesel 258Bz") restates the same figures in a different order. It is a
  ticker, not a data element -- it has carried scheduled-maintenance notices
  instead -- so it is used as a free cross-check when present and ignored when
  absent. When it is present and disagrees with the panel, the page is
  mid-update or broken and nothing is published.
* A few hundred bytes above the panel sits the *subsidy* table: ``180
  Baisa/Ltr``, ``400 Ltrs/Month``, ``M91``. That 180 is what a registered
  beneficiary pays, not the retail price, and it sits next to the string
  "M91" -- which is why the panel is located by its container classes rather
  than by proximity to a fuel name.
* Prices are inclusive of VAT, per the page's own footnote.

If NSS redesigns, every one of these lookups raises rather than guessing, and
the runner leaves the last-good published data in place.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PRICES_CSV = Path(__file__).parent / "prices.csv"

COLUMNS = ["month", "fuel_type", "price_baisa", "source"]
FUELS = ("m91", "m95", "diesel")
NSS_SOURCE = "nss.gov.om"

# "Fuel Price - July'26"
_MONTH_RE = re.compile(r"Fuel\s*Price\s*-\s*([A-Za-z]+)\s*'\s*(\d{2})")
# container classes of the panel's label block, value block, and the VAT-note
# row that closes it
_PANEL_LABELS = "fuelpricesubsidytitle"
_PANEL_VALUES = "fuelpricesubsidyvalue"
_PANEL_END = "fuelpricebox"
_P_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")

# marquee: "July 2026 Fuel Prices M95 239Bz, M91 229Bz and Diesel 258Bz"
_MARQUEE_MONTH_RE = re.compile(r"([A-Za-z]+)\s+(\d{4})\s+Fuel\s+Prices", re.IGNORECASE)
_MARQUEE_PRICE_RE = re.compile(r"\b(M\s*9\s*1|M\s*9\s*5|Diesel)\s*(\d{2,4})\s*Bz",
                               re.IGNORECASE)

_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub("", fragment)).strip()


def _fuel_key(label: str) -> str:
    """'M91' / ' M 95' / 'Diesel' -> 'm91' / 'm95' / 'diesel'."""
    return re.sub(r"[^a-z0-9]", "", label.lower())


def _month_from_heading(html: str) -> str:
    m = _MONTH_RE.search(html)
    if not m:
        raise ValueError("NSS page: fuel-price month header not found — markup changed?")
    name, yy = m.group(1).capitalize(), int(m.group(2))
    if name not in _MONTHS:
        raise ValueError(f"NSS page: unknown month name {m.group(1)!r} in price heading")
    return f"20{yy:02d}-{_MONTHS[name]:02d}"


def _panel_prices(html: str) -> dict[str, int]:
    """Read the price panel, pairing the label block with the value block."""
    i_labels = html.find(_PANEL_LABELS)
    i_values = html.find(_PANEL_VALUES)
    if i_labels < 0 or i_values < 0 or i_labels >= i_values:
        raise ValueError("NSS page: fuel-price panel not found — markup changed?")
    i_end = html.find(_PANEL_END, i_values)
    if i_end < 0:
        i_end = len(html)
    labels = [_text(x) for x in _P_RE.findall(html[i_labels:i_values])]
    values = [_text(x) for x in _P_RE.findall(html[i_values:i_end])]
    if not labels or len(labels) != len(values):
        raise ValueError(
            f"NSS page: fuel-price panel has {len(labels)} labels but "
            f"{len(values)} values — markup changed?")
    prices: dict[str, int] = {}
    for label, value in zip(labels, values):
        fuel = _fuel_key(label)
        if not value.isdigit():
            raise ValueError(
                f"NSS page: {fuel!r} price {value!r} is not a number — markup changed?")
        prices[fuel] = int(value)
    if set(prices) != set(FUELS):
        raise ValueError(
            f"NSS page: fuel-price panel lists {sorted(prices)}, expected "
            f"{sorted(FUELS)} — markup changed?")
    return prices


def _cross_check_marquee(html: str, month: str, prices: dict[str, int]) -> None:
    """Compare the advisory ticker with the panel, when the ticker carries prices."""
    found = {_fuel_key(fuel): int(value)
             for fuel, value in _MARQUEE_PRICE_RE.findall(html)}
    if found and found != prices:
        raise ValueError(
            f"NSS page: marquee prices {found} disagree with the price panel "
            f"{prices} — the page looks mid-update")
    m = _MARQUEE_MONTH_RE.search(html)
    if found and m:
        name, year = m.group(1).capitalize(), int(m.group(2))
        if name not in _MONTHS:
            raise ValueError(f"NSS page: unknown month name {m.group(1)!r} in marquee")
        marquee_month = f"{year:04d}-{_MONTHS[name]:02d}"
        if marquee_month != month:
            raise ValueError(
                f"NSS page: marquee month {marquee_month} disagrees with the "
                f"price panel month {month} — the page looks mid-update")


def scrape_nss(html: str) -> tuple[str, dict[str, int]]:
    """Return (YYYY-MM, {fuel: baisa}) for the month NSS is currently showing."""
    month = _month_from_heading(html)
    prices = _panel_prices(html)
    _cross_check_marquee(html, month, prices)
    return month, prices


def _check_continuous(df: pd.DataFrame) -> None:
    months = sorted(set(df["month"]))
    expected = pd.period_range(months[0], months[-1], freq="M").astype(str).tolist()
    if months != expected:
        missing = sorted(set(expected) - set(months))
        raise ValueError(
            f"month gap in the merged series: {missing[:6]}"
            f"{'...' if len(missing) > 6 else ''} — prices.csv is stale, "
            f"backfill it up to the month before {months[-1]}")


def parse(raw_path: Path) -> tuple[pd.DataFrame, str]:
    raw_path = Path(raw_path)
    month, prices = scrape_nss(raw_path.read_text(encoding="utf-8"))
    df = pd.read_csv(PRICES_CSV, encoding="utf-8")
    if list(df.columns) != COLUMNS or df.empty:
        raise ValueError(f"unexpected layout in {PRICES_CSV.name}")

    if month in set(df["month"]):
        # curation already covers this month — cross-check, don't duplicate
        curated = df[df["month"] == month].set_index("fuel_type")["price_baisa"]
        for fuel, price in prices.items():
            if fuel not in curated.index:
                raise ValueError(f"prices.csv has no {fuel!r} row for {month}")
            if int(curated[fuel]) != price:
                raise ValueError(
                    f"NSS {month} {fuel}={price} disagrees with curated "
                    f"{int(curated[fuel])} — fix prices.csv")
    else:
        new = pd.DataFrame(
            [(month, fuel, price, NSS_SOURCE) for fuel, price in prices.items()],
            columns=COLUMNS)
        df = pd.concat([df, new], ignore_index=True)

    df = (df.astype({"month": str, "fuel_type": str,
                     "price_baisa": int, "source": str})
            .sort_values(["month", "fuel_type"], ignore_index=True))
    if list(df.columns) != COLUMNS or df.empty:
        raise ValueError(f"unexpected layout in {raw_path.name}")
    _check_continuous(df)
    return df, df["month"].max()
