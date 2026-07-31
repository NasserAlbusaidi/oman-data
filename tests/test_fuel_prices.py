import re
from pathlib import Path

import pandas as pd
import pytest

from oman_data.run import _load_callable

FIXTURE = Path("tests/fixtures/fuel_prices/nss_home.html")
PRICES = Path("pipelines/fuel_prices/prices.csv")
MONTH = re.compile(r"^\d{4}-\d{2}$")

# Exact markup anchor in the saved fixture; doctoring tests edit it.
MARQUEE = "<li>July 2026 Fuel Prices M95 239Bz, M91 229Bz and Diesel 258Bz</li>"

_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def load(name: str):
    return _load_callable(Path("pipelines/fuel_prices/parse.py"), name)


# The cap lives in the pipeline, not here — restating 229/239/258 in the tests
# would let the two drift apart and still go green.
CAP = load("SUBSIDY_CAP")


def prices_copy(tmp_path: Path) -> Path:
    """A writable copy of the curated CSV.

    ``parse`` persists newly learnt months (see parse._persist), so every test
    that exercises the append or auto-extend path must point it at a copy or it
    would edit the repo's prices.csv as a side effect of running the suite.
    """
    out = tmp_path / "prices.csv"
    out.write_bytes(PRICES.read_bytes())
    return out


def doctored(tmp_path: Path, *replacements: tuple[str, str]) -> Path:
    """The real fixture with exact substrings swapped, written to a temp file.

    ``str.replace`` hits *every* occurrence, so an anchor that appears twice is
    doctored in both places — which is why the fixture's own comments must not
    restate the markup (see test_fixture_comments_are_not_load_bearing).
    """
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

def test_curated_csv_round_trips_byte_for_byte():
    """pandas must reproduce the committed file exactly.

    ``parse`` rewrites the whole CSV when it learns a month, so any formatting
    difference between what pandas writes and what is committed would show up as
    a 385-line diff on the first CI refresh, burying the one real change.
    """
    df = pd.read_csv(PRICES, encoding="utf-8")
    rewritten = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    assert rewritten == PRICES.read_bytes()


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

    Past that boundary the two live labels interleave rather than succeed one
    another: months the pipeline observed on NSS carry nss.gov.om and are
    retained here (NSS drops a month once it rolls over), while months a missed
    refresh skipped are filled from the cap and carry subsidy-cap-freeze. So the
    tail is asserted as a set membership, not as an ordering.
    """
    df = pd.read_csv(PRICES, encoding="utf-8")
    assert set(df["source"]) <= {
        "archive-corroborated", "archive-single-source",
        "archive-news-resolved", "subsidy-cap-freeze", "nss.gov.om",
    }
    spans = df.groupby("source")["month"].agg(["min", "max"])
    archive = df[df["source"].str.startswith("archive-")]
    assert archive["month"].min() == "2015-12"
    assert archive["month"].max() == "2023-01"
    assert spans.loc["subsidy-cap-freeze", "min"] == "2023-02"
    assert set(df[df["month"] >= "2023-02"]["source"]) <= {
        "subsidy-cap-freeze", "nss.gov.om"}
    # the series must end on an observed month, never on an invented one
    assert set(df[df["month"] == df["month"].max()]["source"]) == {"nss.gov.om"}


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


def test_only_the_known_rows_rest_on_a_single_record():
    """Eight rows have one record and no press confirmation.

    Provenance is per (month, fuel), not per month, and that matters here: for
    Feb-Jun 2016 the second compilation does carry M95 and diesel — and agrees —
    but has no row for the regular grade, because the grade of the day was M-90
    and it does not list it. So only the m91 column of those months is
    uncorroborated. Dec 2022 falls after the second compilation stops and is
    uncorroborated across all three. This list is the weakest part of the
    dataset; it should shrink, never grow.
    """
    df = pd.read_csv(PRICES, encoding="utf-8")
    single = sorted(
        df[df["source"] == "archive-single-source"][["month", "fuel_type"]]
        .itertuples(index=False, name=None))
    assert single == [
        ("2016-02", "m91"), ("2016-03", "m91"), ("2016-04", "m91"),
        ("2016-05", "m91"), ("2016-06", "m91"),
        ("2022-12", "diesel"), ("2022-12", "m91"), ("2022-12", "m95"),
    ]


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

def test_fixture_comments_are_not_load_bearing():
    """No parser anchor may appear inside an HTML comment in the fixture.

    The fixture carries a hand-written provenance header. An earlier wording of
    it repeated the panel heading verbatim ("Fuel Price - July'26"), and since
    ``re.search`` returns the *first* match, every fixture-driven test read that
    annotation instead of the NSS markup 940 lines below. The negative tests
    could not catch it either, because ``doctored`` rewrites every occurrence,
    annotation included.

    Note the weak version of this test — strip the comments and check the answer
    is unchanged — would NOT have caught it, because the annotation quoted the
    correct month. So assert inertness directly: the parser's own regexes and
    container markers must find nothing in the commented-out text.
    """
    html = FIXTURE.read_text(encoding="utf-8")
    comments = _COMMENT_RE.findall(html)
    assert comments, "fixture lost its provenance header"
    blob = "\n".join(comments)
    for name in ("_MONTH_RE", "_MARQUEE_MONTH_RE", "_MARQUEE_PRICE_RE"):
        assert not load(name).search(blob), f"{name} matches inside an HTML comment"
    for marker in ("fuelpricesubsidytitle", "fuelpricesubsidyvalue", "fuelpricebox"):
        assert marker not in blob, f"{marker!r} appears inside an HTML comment"
    # belt and braces: removing the comments must not move the answer either
    scrape = load("scrape_nss")
    assert scrape(_COMMENT_RE.sub("", html)) == scrape(html) == ("2026-07", CAP)


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
    curated = pd.read_csv(PRICES, encoding="utf-8")
    assert not df.duplicated(["month", "fuel_type"]).any()
    assert len(df) == len(curated), "cross-check must not add rows"
    row = df[df["month"] == "2026-06"].set_index("fuel_type")
    assert row["price_baisa"].to_dict() == CAP
    # curated provenance is kept -- the scrape confirmed it, it did not supply it
    assert set(row["source"]) == {"subsidy-cap-freeze"}
    # as_of tracks the series, not the scraped month: the CSV already runs past it
    assert as_of == curated["month"].max()


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


def test_fixture_month_is_carried_by_the_csv_and_merely_confirmed():
    """The fixture's own month has since been written into prices.csv.

    NSS drops a month as soon as it rolls over, so a month observed on the live
    page has to be retained in the CSV or the next run finds a hole. The scrape
    therefore confirms it rather than supplying it, and the row count does not
    move.
    """
    parse = load("parse")
    df, as_of = parse(FIXTURE)
    curated = pd.read_csv(PRICES, encoding="utf-8")
    assert as_of == "2026-07"
    new = df[df["month"] == "2026-07"]
    assert set(new["source"]) == {"nss.gov.om"}
    assert new.set_index("fuel_type")["price_baisa"].to_dict() == CAP
    assert len(df) == len(curated)


def test_next_months_scrape_appends_with_nss_provenance(tmp_path):
    """The append path: the month after the CSV's last is added, not rejected.

    This is what actually happens on the first run of a new month, and it is
    the path that must keep working when prices.csv trails the calendar by one
    month.
    """
    parse = load("parse")
    csv = prices_copy(tmp_path)
    curated = pd.read_csv(PRICES, encoding="utf-8")
    assert curated["month"].max() == "2026-07", "update this test's next month"
    august = doctored(
        tmp_path,
        ("Fuel Price - July'26", "Fuel Price - August'26"),
        ("July 2026 Fuel Prices", "August 2026 Fuel Prices"),
    )
    df, as_of = parse(august, prices_csv=csv)
    assert as_of == "2026-08"
    assert len(df) == len(curated) + 3
    new = df[df["month"] == "2026-08"]
    assert set(new["source"]) == {"nss.gov.om"}
    assert new.set_index("fuel_type")["price_baisa"].to_dict() == CAP
    assert not df.duplicated(["month", "fuel_type"]).any()
    months = sorted(df["month"].unique())
    assert months == list(pd.period_range(months[0], months[-1], freq="M").astype(str))


# --------------------------------------------------------------------------
# persistence and the cap-freeze auto-extend: what keeps the monthly refresh
# running without a human editing prices.csv every month
# --------------------------------------------------------------------------

def test_scraped_month_is_written_back_to_the_csv(tmp_path):
    """NSS forgets a month the instant it rolls over.

    A month held only in the returned frame is gone by the next run, which
    would then find a hole. So the merge is persisted, and the CSV's own
    tail — not the returned frame — is what proves it.
    """
    parse = load("parse")
    csv = prices_copy(tmp_path)
    before = pd.read_csv(csv, encoding="utf-8")
    august = doctored(
        tmp_path,
        ("Fuel Price - July'26", "Fuel Price - August'26"),
        ("July 2026 Fuel Prices", "August 2026 Fuel Prices"),
    )
    parse(august, prices_csv=csv)
    after = pd.read_csv(csv, encoding="utf-8")
    assert after["month"].max() == "2026-08"
    assert len(after) == len(before) + 3
    assert set(after[after["month"] == "2026-08"]["source"]) == {"nss.gov.om"}
    # untouched rows must be untouched, byte for byte
    assert csv.read_bytes().startswith(PRICES.read_bytes())


def test_persisting_is_idempotent(tmp_path):
    """Re-running the same month must cross-check, not append a second time."""
    parse = load("parse")
    csv = prices_copy(tmp_path)
    august = doctored(
        tmp_path,
        ("Fuel Price - July'26", "Fuel Price - August'26"),
        ("July 2026 Fuel Prices", "August 2026 Fuel Prices"),
    )
    df1, _ = parse(august, prices_csv=csv)
    first = csv.read_bytes()
    df2, _ = parse(august, prices_csv=csv)
    assert csv.read_bytes() == first
    assert df2.equals(df1)


def test_confirming_an_already_curated_month_does_not_rewrite_the_csv(tmp_path):
    """No new months, no write — a no-op run leaves the file's mtime story clean."""
    parse = load("parse")
    csv = prices_copy(tmp_path)
    before = csv.read_bytes()
    parse(FIXTURE, prices_csv=csv)   # 2026-07, already curated
    assert csv.read_bytes() == before


def test_one_skipped_month_is_auto_extended_from_the_cap(tmp_path):
    """The case that used to break the pipeline outright.

    prices.csv ends 2026-07; if the September refresh is the next one to run
    (August's never did, or ran before the announcement), 2026-08 exists in no
    source anywhere. It is still knowable: the cap held on both sides of it. So
    it is filled from the cap constants and labelled subsidy-cap-freeze — the
    same provenance the hand-curated 2023-02+ tail carries — never nss.gov.om,
    because nobody saw it.
    """
    parse = load("parse")
    csv = prices_copy(tmp_path)
    september = doctored(
        tmp_path,
        ("Fuel Price - July'26", "Fuel Price - September'26"),
        ("July 2026 Fuel Prices", "September 2026 Fuel Prices"),
    )
    df, as_of = parse(september, prices_csv=csv)
    assert as_of == "2026-09"
    filled = df[df["month"] == "2026-08"]
    assert set(filled["source"]) == {"subsidy-cap-freeze"}
    assert filled.set_index("fuel_type")["price_baisa"].to_dict() == CAP
    assert set(df[df["month"] == "2026-09"]["source"]) == {"nss.gov.om"}
    months = sorted(df["month"].unique())
    assert months == list(pd.period_range(months[0], months[-1], freq="M").astype(str))
    # and it is persisted, so the next run starts from a continuous CSV
    assert pd.read_csv(csv, encoding="utf-8")["month"].max() == "2026-09"


def test_auto_extend_stops_at_its_declared_bound(tmp_path):
    """Three invented months is the ceiling; the fourth demands a human.

    A quarter of missed refreshes is a broken pipeline, not a late one, and the
    cap inference gets weaker the wider the interval it spans — verify those
    months against archived price boards instead.
    """
    parse = load("parse")
    assert load("MAX_AUTO_EXTEND_MONTHS") == 3, "update this test's months"

    # Nov'26: 2026-08, -09, -10 missing — exactly three, allowed
    november = doctored(
        tmp_path,
        ("Fuel Price - July'26", "Fuel Price - November'26"),
        ("July 2026 Fuel Prices", "November 2026 Fuel Prices"),
    )
    df, as_of = parse(november, prices_csv=prices_copy(tmp_path))
    assert as_of == "2026-11"
    invented = df[df["month"].between("2026-08", "2026-10")]
    assert set(invented["source"]) == {"subsidy-cap-freeze"}
    assert len(invented) == 9

    # Dec'26: four missing — refuse
    december = doctored(
        tmp_path,
        ("Fuel Price - July'26", "Fuel Price - December'26"),
        ("July 2026 Fuel Prices", "December 2026 Fuel Prices"),
    )
    csv = prices_copy(tmp_path)
    before = csv.read_bytes()
    with pytest.raises(ValueError, match="gap needs a human"):
        parse(december, prices_csv=csv)
    assert csv.read_bytes() == before, "a refused run must not touch the CSV"


def test_a_lifted_cap_blocks_the_backfill_instead_of_inventing_months(tmp_path):
    """The one failure this whole mechanism exists to avoid.

    If NSS shows a price that is not the cap and months are missing, those
    months could be anywhere between the old level and the new one — the cap
    constants are no longer evidence about them. Inventing them would fabricate
    a flat run through a price change and bury the most newsworthy event this
    dataset can record. Refuse, and name the cap in the message.
    """
    parse = load("parse")
    csv = prices_copy(tmp_path)
    lifted = doctored(
        tmp_path,
        ("Fuel Price - July'26", "Fuel Price - September'26"),
        (MARQUEE, ""),
        ("<p>229</p>", "<p>259</p>"),
    )
    with pytest.raises(ValueError, match="cap looks lifted"):
        parse(lifted, prices_csv=csv)
    assert csv.read_bytes() == PRICES.read_bytes(), "a refused run must not backfill"


def test_a_lifted_cap_with_no_gap_is_recorded_not_refused(tmp_path):
    """A real price change in a month that needs no inference is just data.

    The auto-extend refuses because it cannot infer *missing* months; it has no
    opinion on the observed one. NSS is the official source and the row is
    appended with nss.gov.om provenance. The change still surfaces to a human on
    the next test run, because test_curated_tail_is_flat_at_the_capped_prices
    goes red the moment an off-cap month lands in the committed CSV — that is
    the alarm, not a silent parse failure.
    """
    parse = load("parse")
    csv = prices_copy(tmp_path)
    lifted = doctored(
        tmp_path,
        ("Fuel Price - July'26", "Fuel Price - August'26"),
        (MARQUEE, ""),
        ("<p>229</p>", "<p>259</p>"),
    )
    df, as_of = parse(lifted, prices_csv=csv)
    assert as_of == "2026-08"
    row = df[df["month"] == "2026-08"].set_index("fuel_type")
    assert row["price_baisa"]["m91"] == 259
    assert set(row["source"]) == {"nss.gov.om"}


def test_auto_extend_refuses_when_the_csv_tail_is_off_cap(tmp_path):
    """The other endpoint of the inference.

    If the CSV already ends on an off-cap month, the freeze narrative no longer
    describes the tail and cap constants are not the right filler either, even
    though today's page happens to match them.
    """
    parse = load("parse")
    csv = prices_copy(tmp_path)
    df = pd.read_csv(csv, encoding="utf-8")
    df.loc[(df["month"] == "2026-07") & (df["fuel_type"] == "m91"), "price_baisa"] = 249
    df.to_csv(csv, index=False, encoding="utf-8", lineterminator="\n")
    september = doctored(
        tmp_path,
        ("Fuel Price - July'26", "Fuel Price - September'26"),
        ("July 2026 Fuel Prices", "September 2026 Fuel Prices"),
    )
    with pytest.raises(ValueError, match="off the subsidy cap"):
        parse(september, prices_csv=csv)


def test_a_hole_inside_the_curated_csv_still_fails_loudly(tmp_path):
    """The auto-extend only closes gaps at the *end* of the series.

    A month deleted from the middle of prices.csv is a curation accident, not a
    missed refresh, and no constant can be trusted to fill it.
    """
    parse = load("parse")
    csv = prices_copy(tmp_path)
    df = pd.read_csv(csv, encoding="utf-8")
    df[df["month"] != "2020-05"].to_csv(
        csv, index=False, encoding="utf-8", lineterminator="\n")
    with pytest.raises(ValueError, match="hole in it"):
        parse(FIXTURE, prices_csv=csv)


def test_parse_output_is_sorted_and_typed():
    parse = load("parse")
    df, _ = parse(FIXTURE)
    assert df["price_baisa"].dtype.kind == "i"
    assert list(df.index) == list(range(len(df)))
    assert df[["month", "fuel_type"]].apply(tuple, axis=1).is_monotonic_increasing
    assert not df.isna().any().any()
    assert df["price_baisa"].between(100, 1000).all()
