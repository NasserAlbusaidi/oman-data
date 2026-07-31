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

Because NSS drops a month the moment it rolls over, the months learnt here have
to be written back into ``prices.csv`` or they are lost. That write is
``persist``, the runner's post-validation hook -- not part of ``parse``, which
stays pure. See ``persist`` for why the ordering matters.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PRICES_CSV = Path(__file__).parent / "prices.csv"

COLUMNS = ["month", "fuel_type", "price_baisa", "source"]
FUELS = ("m91", "m95", "diesel")
NSS_SOURCE = "nss.gov.om"

# The subsidy price cap, and the only definition of it in the repo: Sultan
# Haitham bin Tariq's decision of 9 Nov 2021 pegged the three grades to their
# October 2021 average with effect from 1 Dec 2021, and it has held since (see
# dataset.yaml notes). The curated tail from 2023-02 is generated from these
# three numbers, the auto-extend below reuses them, and the tests import them
# from here rather than restating them.
SUBSIDY_CAP = {"m91": 229, "m95": 239, "diesel": 258}
CAP_SOURCE = "subsidy-cap-freeze"

# How many skipped months the auto-extend is willing to invent from the cap
# before it demands a human. One is normal operation (a refresh that ran a day
# late); a whole quarter of silence means CI has been broken long enough that
# the interval deserves checking against archived price boards rather than
# being papered over with constants.
MAX_AUTO_EXTEND_MONTHS = 3

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
    """Last line of defence: a monthly series must have no holes.

    A hole at the *end* -- prices.csv trailing the month NSS is showing -- is
    handled upstream by ``_auto_extend``, which either fills it from the cap or
    refuses loudly. Anything reaching here is a hole in the middle of the
    curated file, i.e. someone deleted or mistyped a month.
    """
    months = sorted(set(df["month"]))
    expected = pd.period_range(months[0], months[-1], freq="M").astype(str).tolist()
    if months != expected:
        missing = sorted(set(expected) - set(months))
        raise ValueError(
            f"month gap in the merged series: {missing[:6]}"
            f"{'...' if len(missing) > 6 else ''} — prices.csv has a hole in it, "
            f"backfill those months")


def _months_between(last: str, scraped: str) -> list[str]:
    """Months strictly between ``last`` and ``scraped`` (empty if adjacent)."""
    if scraped <= last:
        return []
    return pd.period_range(last, scraped, freq="M").astype(str).tolist()[1:-1]


def _auto_extend(df: pd.DataFrame, month: str, prices: dict[str, int]) -> pd.DataFrame:
    """Fill months a missed refresh skipped -- but only while the cap holds.

    NSS publishes one month and forgets it, so a refresh that does not run in
    August leaves a hole no source can fill afterwards. For the capped era that
    hole is nonetheless *knowable*: if the price was at the cap before the gap
    and is still at the cap after it, the months in between were at the cap too,
    which is exactly the reasoning behind the hand-curated ``subsidy-cap-freeze``
    tail. So fill them, and label them with that same provenance -- never with
    ``nss.gov.om``, because nobody observed them.

    The inference dies the moment either endpoint is off the cap:

    * scraped prices != the cap -> the cap has been lifted (or restated), and
      the missing months could be anywhere between the old and new level. Refuse
      to guess. A lifted cap is a human decision, not a silent backfill.
    * the CSV's last month is off the cap -> the freeze narrative no longer
      describes the tail, so cap constants are not the right filler either.
    * more than ``MAX_AUTO_EXTEND_MONTHS`` missing -> see that constant.
    """
    last = df["month"].max()
    missing = _months_between(last, month)
    if not missing:
        return df

    if prices != SUBSIDY_CAP:
        raise ValueError(
            f"NSS {month} prices {prices} differ from the subsidy cap "
            f"{SUBSIDY_CAP} and prices.csv is missing {missing} — the cap looks "
            f"lifted, so the missing months cannot be inferred from it. Backfill "
            f"prices.csv by hand from the monthly announcements, and re-check the "
            f"cap notes in dataset.yaml.")

    tail = {fuel: int(price) for fuel, price in
            df[df["month"] == last].set_index("fuel_type")["price_baisa"].items()}
    if tail != SUBSIDY_CAP:
        raise ValueError(
            f"prices.csv ends at {last} with {tail}, which is off the subsidy cap "
            f"{SUBSIDY_CAP} — refusing to auto-extend {missing} from cap "
            f"constants. Backfill prices.csv by hand.")

    if len(missing) > MAX_AUTO_EXTEND_MONTHS:
        raise ValueError(
            f"NSS is showing {month} but prices.csv ends at {last}: {len(missing)} "
            f"months missing ({missing[0]}..{missing[-1]}), more than the "
            f"{MAX_AUTO_EXTEND_MONTHS} the cap-freeze auto-extend will invent. "
            f"This gap needs a human — verify each month against an archived "
            f"price board and backfill prices.csv.")

    filled = pd.DataFrame(
        [(m, fuel, price, CAP_SOURCE)
         for m in missing for fuel, price in SUBSIDY_CAP.items()],
        columns=COLUMNS)
    return pd.concat([df, filled], ignore_index=True)


def persist(df: pd.DataFrame, prices_csv: Path | None = None) -> list[str]:
    """Write months the curated CSV does not yet carry back into it.

    The runner's optional post-validation hook (see oman_data.run.run_dataset),
    and deliberately *not* something ``parse`` does. NSS drops a month as soon as
    it rolls over, so a month held only in the returned frame is gone by the next
    run — but a month written into prices.csv is permanent, and prices.csv is the
    only record of it that will ever exist. So the write happens strictly after
    validation has accepted the frame: a scrape defect that slips past the panel
    and marquee cross-checks but trips the validator must not become the curated
    truth that every later run compares against.

    Returns the months added, for the runner to log. Rewrites the whole file
    rather than appending so the sort order stays canonical, and refuses to
    write a frame that is not a superset of what is already curated — this file
    is irreplaceable and a truncating write would be unrecoverable.
    """
    prices_csv = Path(prices_csv) if prices_csv is not None else PRICES_CSV
    curated = pd.read_csv(prices_csv, encoding="utf-8")
    added = sorted(set(df["month"]) - set(curated["month"]))
    if not added:
        return []

    have = set(map(tuple, curated[["month", "fuel_type"]].to_numpy().tolist()))
    keep = set(map(tuple, df[["month", "fuel_type"]].to_numpy().tolist()))
    if not have <= keep:
        raise ValueError(
            f"refusing to write {prices_csv.name}: the frame is missing "
            f"{sorted(have - keep)[:6]} that the curated file already carries — "
            f"this would delete history no source can return")

    out = (df.astype({"month": str, "fuel_type": str,
                      "price_baisa": int, "source": str})
             .sort_values(["month", "fuel_type"], ignore_index=True))
    out.to_csv(prices_csv, index=False, encoding="utf-8", lineterminator="\n")
    return added


def parse(raw_path: Path, prices_csv: Path | None = None) -> tuple[pd.DataFrame, str]:
    """Merge the curated history with the month NSS is showing.

    Pure: it reads prices.csv and never writes it. Committing a newly learnt
    month to the curated file is ``persist``'s job, which the runner calls only
    once the frame has passed validation.

    ``prices_csv`` overrides the curated file so tests can exercise the append
    and auto-extend paths against a copy.
    """
    raw_path = Path(raw_path)
    prices_csv = Path(prices_csv) if prices_csv is not None else PRICES_CSV
    month, prices = scrape_nss(raw_path.read_text(encoding="utf-8"))
    df = pd.read_csv(prices_csv, encoding="utf-8")
    if list(df.columns) != COLUMNS or df.empty:
        raise ValueError(f"unexpected layout in {prices_csv.name}")
    known = set(df["month"])

    if month in known:
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
        df = _auto_extend(df, month, prices)
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
