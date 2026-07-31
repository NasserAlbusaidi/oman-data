import re
from pathlib import Path

import pandas as pd
import pytest

from oman_data.run import _load_callable

FIXTURE = Path("tests/fixtures/fuel_prices/nss_home.html")
PRICES = Path("pipelines/fuel_prices/prices.csv")
MONTH = re.compile(r"^\d{4}-\d{2}$")

CAP = {"m91": 229, "m95": 239, "diesel": 258}

# Exact markup anchors in the saved fixture; doctoring tests edit these.
MARQUEE = "<li>July 2026 Fuel Prices M95 239Bz, M91 229Bz and Diesel 258Bz</li>"
WIDGET_HEADING = "<h3> Fuel Price - July'26 <span class=\"ltrb\">Baisa/Ltr</span></h3>"


def load(name: str):
    return _load_callable(Path("pipelines/fuel_prices/parse.py"), name)


def doctored(tmp_path: Path, *replacements: tuple[str, str]) -> Path:
    """The real fixture with exact substrings swapped, written to a temp file."""
    html = FIXTURE.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in html, f"anchor vanished from fixture: {old!r}"
        html = html.replace(old, new)
    out = tmp_path / "doctored.html"
    out.write_text(html, encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# contract tests (per the task brief)
# --------------------------------------------------------------------------

def test_curated_csv_is_complete_and_sane():
    df = pd.read_csv(PRICES, encoding="utf-8")
    assert list(df.columns) == ["month", "fuel_type", "price_baisa", "source"]
    assert set(df["fuel_type"]) == {"m91", "m95", "diesel"}
    assert df["price_baisa"].between(100, 1000).all()
    assert not df.duplicated(["month", "fuel_type"]).any()
    counts = df.groupby("month")["fuel_type"].nunique()
    assert (counts == 3).all(), "every month needs all three fuels"
    assert df["month"].min() == "2015-12"
    # continuous months, no gaps
    months = sorted(df["month"].unique())
    expected = pd.period_range(months[0], months[-1], freq="M").astype(str)
    assert months == list(expected)


def test_parse_merges_curated_and_scraped():
    parse = _load_callable(Path("pipelines/fuel_prices/parse.py"), "parse")
    df, as_of = parse(FIXTURE)
    assert list(df.columns) == ["month", "fuel_type", "price_baisa", "source"]
    assert df["month"].map(lambda m: bool(MONTH.match(m))).all()
    assert not df.duplicated(["month", "fuel_type"]).any()
    assert df["price_baisa"].dtype.kind == "i"
    assert as_of == df["month"].max()
    # the fixture's month (2026-07) must be present with NSS-scraped values
    assert "2026-07" in set(df["month"])


def test_scrape_extracts_known_fixture_prices():
    scrape = _load_callable(Path("pipelines/fuel_prices/parse.py"), "scrape_nss")
    month, prices = scrape(FIXTURE.read_text(encoding="utf-8"))
    assert month == "2026-07"
    assert prices == {"m91": 229, "m95": 239, "diesel": 258}


def test_mangled_layout_fails_loudly(tmp_path):
    parse = _load_callable(Path("pipelines/fuel_prices/parse.py"), "parse")
    bad = tmp_path / "mangled.html"
    bad.write_text("<html>no prices here</html>", encoding="utf-8")
    with pytest.raises(Exception):
        parse(bad)


# --------------------------------------------------------------------------
# the curated history: provenance and the curation decisions inside it
# --------------------------------------------------------------------------

def test_every_row_declares_a_known_provenance():
    """``source`` is a closed vocabulary, and the archive ends where the
    freeze-fill begins (2023-01 / 2023-02) with no overlap.

    The current month is never in the CSV — it comes from NSS at parse time.
    """
    df = pd.read_csv(PRICES, encoding="utf-8")
    assert set(df["source"]) <= {
        "archive-corroborated", "archive-single-source",
        "archive-news-resolved", "subsidy-cap-freeze",
    }
    archive = df[df["source"] != "subsidy-cap-freeze"]
    freeze = df[df["source"] == "subsidy-cap-freeze"]
    assert archive["month"].min() == "2015-12"
    assert archive["month"].max() == "2023-01"
    assert freeze["month"].min() == "2023-02"
    assert freeze["month"].max() == df["month"].max()


def test_disputed_months_keep_their_adjudicated_values():
    """Four months where the two independent compilations disagreed.

    opendata.om carries January 2019's prices in December 2018, and copies of
    the autumn-2020 prices in October and November 2019; thefuelprice.com is
    wrong for November 2018. Each was settled against a dated primary record
    (the Times of Oman announcement of 1 Dec 2018, and Oman Oil Marketing's own
    archived price boards) — see dataset.yaml. Re-deriving the CSV from either
    compilation alone must fail here.
    """
    df = pd.read_csv(PRICES, encoding="utf-8")
    wide = df.pivot(index="month", columns="fuel_type", values="price_baisa")
    resolved = {
        "2018-11": {"m91": 222, "m95": 233, "diesel": 261},
        "2018-12": {"m91": 211, "m95": 223, "diesel": 251},
        "2019-10": {"m91": 207, "m95": 217, "diesel": 245},
        "2019-11": {"m91": 203, "m95": 216, "diesel": 240},
    }
    for month, prices in resolved.items():
        assert wide.loc[month].to_dict() == prices, month
        assert set(df[df["month"] == month]["source"]) == {"archive-news-resolved"}


def test_months_checked_against_the_press_record_still_match():
    """Eleven months read off dated primary announcements during curation.

    These are the evidence that the two compilations behind the corroborated
    rows are trustworthy in between; if a re-derivation moves any of them, the
    reconciliation drifted and the whole archive span needs re-checking. URLs
    and verbatim quotes are in dataset.yaml and the task report.
    """
    df = pd.read_csv(PRICES, encoding="utf-8")
    wide = df.pivot(index="month", columns="fuel_type", values="price_baisa")
    confirmed = {
        "2015-12": {"m91": 114, "m95": 120, "diesel": 146},
        "2016-01": {"m91": 140, "m95": 160, "diesel": 160},
        "2016-07": {"m91": 170, "m95": 180, "diesel": 188},
        "2016-08": {"m91": 156, "m95": 166, "diesel": 178},
        "2016-09": {"m91": 161, "m95": 170, "diesel": 176},
        "2016-10": {"m91": 169, "m95": 179, "diesel": 185},
        "2018-10": {"m91": 222, "m95": 233, "diesel": 258},
        "2019-01": {"m91": 198, "m95": 209, "diesel": 238},
        "2019-09": {"m91": 201, "m95": 211, "diesel": 241},
        "2019-12": {"m91": 211, "m95": 222, "diesel": 240},
        "2023-01": {"m91": 229, "m95": 239, "diesel": 258},
    }
    for month, prices in confirmed.items():
        assert wide.loc[month].to_dict() == prices, month
        # a primary record is a second witness, so none of these is single-source
        assert set(df[df["month"] == month]["source"]) != {"archive-single-source"}, month


def test_only_the_known_months_rest_on_a_single_record():
    """Six months have one record and no press confirmation. Keep the list
    short and explicit: it is the weakest part of the dataset, and it should
    shrink, never grow.
    """
    df = pd.read_csv(PRICES, encoding="utf-8")
    single = sorted(df[df["source"] == "archive-single-source"]["month"].unique())
    assert single == ["2016-02", "2016-03", "2016-04", "2016-05", "2016-06", "2022-12"]


def test_m91_series_splices_the_withdrawn_m90_grade():
    """Before Nov 2016 the regular grade was M-90, not M-91.

    The source table carries them as two strictly complementary columns; this
    dataset publishes the union as one continuous regular-grade series under
    ``m91`` (see dataset.yaml). Pin the join so nobody silently re-cuts it:
    the level is continuous across the switch (169 -> 173 baisa), which is why
    splicing is defensible in the first place.
    """
    df = pd.read_csv(PRICES, encoding="utf-8")
    m91 = df[df["fuel_type"] == "m91"].set_index("month")["price_baisa"]
    assert m91["2015-12"] == 114   # M-90 era, the first month of the series
    assert m91["2016-10"] == 169   # last M-90 month
    assert m91["2016-11"] == 173   # first M-91 month


def test_curated_tail_is_flat_at_the_capped_prices():
    """The cap has held since Dec 2021; Nov 2021 was the last month above it.

    Spot-verified against news archives for 2023-06, 2024-11 and 2025-08
    before this file was committed (URLs in dataset.yaml notes / task report).
    """
    df = pd.read_csv(PRICES, encoding="utf-8")
    tail = df[df["month"] >= "2021-12"]
    assert not tail.empty
    for fuel, price in CAP.items():
        assert (tail[tail["fuel_type"] == fuel]["price_baisa"] == price).all(), fuel
    nov = df[df["month"] == "2021-11"].set_index("fuel_type")["price_baisa"]
    assert nov["m91"] == 233 and nov["m95"] == 242 and nov["diesel"] == 275


def test_curated_prices_never_move_more_than_a_third_in_a_month():
    """A transcription slip (dropped digit, shifted column) shows up as a jump.

    The largest real move in the whole record is M95 rising 120 -> 160 baisa in
    Jan 2016 when the subsidy was removed, exactly 33.3%; the next largest is
    15.3%. The bound is therefore deliberately tight — a hand-maintained CSV
    should trip it long before a wrong number reaches the API.
    """
    df = pd.read_csv(PRICES, encoding="utf-8")
    wide = df.pivot(index="month", columns="fuel_type", values="price_baisa").sort_index()
    moves = wide.pct_change().abs()
    worst = moves.max().max()
    assert worst < 0.34, (
        f"implausible month-on-month move of {worst:.0%} at "
        f"{moves.max(axis=1).idxmax()} — check prices.csv against the source")


# --------------------------------------------------------------------------
# the NSS scrape: it must read the price panel, not the subsidy-rate table
# --------------------------------------------------------------------------

def test_scrape_ignores_the_subsidised_rate_table(tmp_path):
    """The page also shows the *subsidy* rate -- 180 baisa/ltr, 400 ltrs/month,
    fuel type M91 -- a few hundred bytes above the price panel. That 180 is
    what a beneficiary pays, not the retail price, and scraping it would
    understate M91 by 21%. Strip the marquee so the subsidy table is the only
    other M91-adjacent number on the page, and the panel must still win.
    """
    scrape = load("scrape_nss")
    bad = doctored(tmp_path, (MARQUEE, ""))
    month, prices = scrape(bad.read_text(encoding="utf-8"))
    assert month == "2026-07"
    assert prices == CAP, "scraped the subsidised rate instead of the retail price"


def test_scrape_pairs_labels_with_values_positionally(tmp_path):
    """The panel is two sibling blocks -- labels, then values -- so the pairing
    is by position. If NSS reorders the columns the values must follow.
    """
    scrape = load("scrape_nss")
    swapped = doctored(
        tmp_path,
        ("<p> M91</p>", "<p> @@FUEL@@</p>"),
        ("<p> Diesel</p>", "<p> M91</p>"),
        ("<p> @@FUEL@@</p>", "<p> Diesel</p>"),
        (MARQUEE, ""),
    )
    _, prices = scrape(swapped.read_text(encoding="utf-8"))
    # labels now read Diesel, M95, M91 against values 229, 239, 258
    assert prices == {"diesel": 229, "m95": 239, "m91": 258}


def test_missing_price_panel_fails_loudly(tmp_path):
    """If NSS redesigns, publish nothing rather than a guess."""
    scrape = load("scrape_nss")
    bad = doctored(tmp_path, ("fuelpricesubsidyvalue", "somethingelse"))
    with pytest.raises(ValueError, match="price"):
        scrape(bad.read_text(encoding="utf-8"))


def test_missing_month_heading_fails_loudly(tmp_path):
    scrape = load("scrape_nss")
    bad = doctored(tmp_path, ("Fuel Price - July'26", "Fuel Price"))
    with pytest.raises(ValueError, match="month"):
        scrape(bad.read_text(encoding="utf-8"))


def test_unknown_month_name_fails_loudly(tmp_path):
    scrape = load("scrape_nss")
    bad = doctored(tmp_path, ("Fuel Price - July'26", "Fuel Price - Jjuly'26"))
    with pytest.raises(ValueError, match="month"):
        scrape(bad.read_text(encoding="utf-8"))


def test_marquee_disagreeing_with_the_panel_fails_loudly(tmp_path):
    """The ticker and the panel are two independent statements of the same
    numbers. If they diverge, the page is mid-update or broken -- publishing
    either one would be a coin flip.
    """
    scrape = load("scrape_nss")
    bad = doctored(tmp_path, ("M91 229Bz", "M91 329Bz"))
    with pytest.raises(ValueError, match="marquee"):
        scrape(bad.read_text(encoding="utf-8"))


def test_marquee_month_disagreeing_with_the_panel_fails_loudly(tmp_path):
    scrape = load("scrape_nss")
    bad = doctored(tmp_path, ("July 2026 Fuel Prices", "June 2026 Fuel Prices"))
    with pytest.raises(ValueError, match="marquee"):
        scrape(bad.read_text(encoding="utf-8"))


def test_absent_marquee_is_tolerated(tmp_path):
    """The marquee is an advisory ticker -- it has carried maintenance notices
    instead of prices -- so it is a cross-check when present, never a
    requirement.
    """
    scrape = load("scrape_nss")
    ok = doctored(tmp_path, (MARQUEE, "<li>Scheduled maintenance notice</li>"))
    month, prices = scrape(ok.read_text(encoding="utf-8"))
    assert (month, prices) == ("2026-07", CAP)


def test_non_numeric_price_fails_loudly(tmp_path):
    scrape = load("scrape_nss")
    bad = doctored(tmp_path, ("<p>229</p>", "<p>N/A</p>"), (MARQUEE, ""))
    with pytest.raises(ValueError, match="price"):
        scrape(bad.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# the merge: cross-check, append, and the gap guard
# --------------------------------------------------------------------------

def test_scraped_month_already_curated_is_cross_checked_not_duplicated(tmp_path):
    """Point the scrape at a month the CSV already covers: one row per fuel."""
    parse = load("parse")
    june = doctored(
        tmp_path,
        ("Fuel Price - July'26", "Fuel Price - June'26"),
        ("July 2026 Fuel Prices", "June 2026 Fuel Prices"),
    )
    df, as_of = parse(june)
    assert not df.duplicated(["month", "fuel_type"]).any()
    row = df[df["month"] == "2026-06"].set_index("fuel_type")
    assert row["price_baisa"].to_dict() == CAP
    # curated provenance is kept -- the scrape confirmed it, it did not supply it
    assert set(row["source"]) == {"subsidy-cap-freeze"}
    assert as_of == "2026-06"


def test_nss_disagreeing_with_curated_history_fails_loudly(tmp_path):
    """If NSS restates a month the CSV already has, publish neither number."""
    parse = load("parse")
    bad = doctored(
        tmp_path,
        ("Fuel Price - July'26", "Fuel Price - June'26"),
        ("July 2026 Fuel Prices", "June 2026 Fuel Prices"),
        ("<p>229</p>", "<p>239</p>"),
        ("M91 229Bz", "M91 239Bz"),
    )
    with pytest.raises(ValueError, match="disagrees"):
        parse(bad)


def test_scraped_current_month_is_appended_with_nss_provenance():
    parse = load("parse")
    df, as_of = parse(FIXTURE)
    curated = pd.read_csv(PRICES, encoding="utf-8")
    assert as_of == "2026-07"
    new = df[df["month"] == "2026-07"]
    assert set(new["source"]) == {"nss.gov.om"}
    assert new.set_index("fuel_type")["price_baisa"].to_dict() == CAP
    assert len(df) == len(curated) + 3


def test_stale_prices_csv_fails_loudly_instead_of_publishing_a_gap(tmp_path):
    """The CSV stops at the last settled month and NSS only ever publishes the
    current one, so an unmaintained CSV would silently punch a hole in a
    monthly series. Nothing downstream validates continuity -- this does.
    """
    parse = load("parse")
    ahead = doctored(
        tmp_path,
        ("Fuel Price - July'26", "Fuel Price - December'26"),
        (MARQUEE, ""),
    )
    with pytest.raises(ValueError, match="gap"):
        parse(ahead)


def test_parse_output_is_sorted_and_typed():
    parse = load("parse")
    df, _ = parse(FIXTURE)
    assert df["price_baisa"].dtype.kind == "i"
    assert list(df.index) == list(range(len(df)))
    assert df[["month", "fuel_type"]].apply(tuple, axis=1).is_monotonic_increasing
    assert not df.isna().any().any()
    assert df["price_baisa"].between(100, 1000).all()
