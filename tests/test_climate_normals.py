"""Tests for the 1991-2020 climate normals dataset.

The fixture is the real Open-Meteo archive response for all six stations
(6 x 10958 days), so the physical-sanity block below is a genuine assertion
about Omani climate, not about a hand-written stub. The adversarial block
doctors that same real fixture -- structurally, since it is JSON -- so every
negative test starts from a payload the parser is known to accept.
"""
import csv
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from oman_data.run import _load_callable

FIXTURE_DIR = Path("tests/fixtures/climate_normals")
STATIONS_CSV = Path("pipelines/climate_normals/stations.csv")
PARSE_PY = Path("pipelines/climate_normals/parse.py")

STATIONS = ("muscat", "salalah", "sohar", "sur", "nizwa", "saiq")


def fixture_path() -> Path:
    files = [p for p in FIXTURE_DIR.iterdir() if p.is_file()]
    assert len(files) == 1, "exactly one golden fixture expected"
    return files[0]


def load(name: str):
    return _load_callable(PARSE_PY, name)


_CACHE: dict[str, tuple[pd.DataFrame, str]] = {}


def parsed() -> tuple[pd.DataFrame, str]:
    """Parse the 2 MB fixture once and hand out copies — it is used by most tests."""
    if "df" not in _CACHE:
        df, as_of = load("parse")(fixture_path())
        _CACHE["df"] = (df, as_of)
    df, as_of = _CACHE["df"]
    return df.copy(), as_of


def raw_payload() -> dict:
    return json.loads(fixture_path().read_text(encoding="utf-8"))


def doctored(tmp_path: Path, mutate) -> Path:
    """The real fixture with ``mutate`` applied to the decoded payload."""
    payload = raw_payload()
    mutate(payload)
    out = tmp_path / "doctored.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# contract tests (per the task brief)
# --------------------------------------------------------------------------

def test_parse_produces_exact_normals_table():
    parse = _load_callable(Path("pipelines/climate_normals/parse.py"), "parse")
    df, as_of = parse(fixture_path())
    assert list(df.columns) == [
        "station", "station_ar", "station_en", "month", "variable", "value",
    ]
    assert len(df) == 6 * 12 * 3
    assert set(df["variable"]) == {"tmax_c", "tmin_c", "rain_mm"}
    assert df["month"].between(1, 12).all()
    assert not df.isna().any().any()
    assert as_of == "1991-2020"


def test_bilingual_station_names():
    import re
    parse = _load_callable(Path("pipelines/climate_normals/parse.py"), "parse")
    df, _ = parse(fixture_path())
    arabic = re.compile(r"[\u0600-\u06FF]")
    assert df["station_ar"].map(lambda s: bool(arabic.search(s))).all()
    assert df["station"].str.match(r"^[a-z]+$").all()


def test_climatology_is_physically_sane():
    parse = _load_callable(Path("pipelines/climate_normals/parse.py"), "parse")
    df, _ = parse(fixture_path())
    d = df.set_index(["station", "variable", "month"])["value"]
    # Muscat summers are hot
    assert d[("muscat", "tmax_c", 6)] > 35
    # tmax always exceeds tmin
    tmax = df[df["variable"] == "tmax_c"].set_index(["station", "month"])["value"]
    tmin = df[df["variable"] == "tmin_c"].set_index(["station", "month"])["value"]
    assert (tmax > tmin).all()
    # Salalah khareef: July rain in Salalah beats July rain in Muscat
    assert d[("salalah", "rain_mm", 7)] > d[("muscat", "rain_mm", 7)]
    # Saiq (2000 m mountain) is cooler than Muscat year-round
    saiq = df[(df["station"] == "saiq") & (df["variable"] == "tmax_c")]["value"]
    muscat = df[(df["station"] == "muscat") & (df["variable"] == "tmax_c")]["value"]
    assert saiq.mean() < muscat.mean()


def test_mangled_layout_fails_loudly(tmp_path):
    parse = _load_callable(Path("pipelines/climate_normals/parse.py"), "parse")
    bad = tmp_path / "mangled.json"
    bad.write_text('{"muscat": {"daily": {}}}', encoding="utf-8")
    with pytest.raises(Exception):
        parse(bad)


# --------------------------------------------------------------------------
# the curated station catalog
# --------------------------------------------------------------------------

def test_stations_csv_is_the_six_station_catalog():
    with STATIONS_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["station"] for r in rows] == list(STATIONS)
    arabic = re.compile(r"^[\u0600-\u06FF\s()]+$")
    for r in rows:
        assert re.fullmatch(r"[a-z]+", r["station"]), r
        assert r["station_en"].strip(), r
        assert arabic.fullmatch(r["station_ar"]), r
        # every station is inside Oman's bounding box
        assert 16.5 <= float(r["latitude"]) <= 26.5, r
        assert 52.0 <= float(r["longitude"]) <= 60.0, r


def test_fixture_grid_cells_match_the_requested_coordinates():
    """The archive snaps each request to a reanalysis grid cell and echoes it.

    A transposed lat/lon or a mis-keyed station would still return a valid
    200 with plausible-looking weather — for somewhere else. Pin the echo
    against stations.csv so the fixture is provably the catalog's coordinates,
    within one ERA5-Land cell (0.1 deg).
    """
    with STATIONS_CSV.open(encoding="utf-8") as f:
        wanted = {r["station"]: (float(r["latitude"]), float(r["longitude"]))
                  for r in csv.DictReader(f)}
    payload = raw_payload()
    assert set(payload) == set(wanted)
    for station, (lat, lon) in wanted.items():
        got = payload[station]
        assert abs(got["latitude"] - lat) <= 0.1, station
        assert abs(got["longitude"] - lon) <= 0.1, station


def test_fixture_covers_the_full_normal_period():
    """A normal is only a normal if all 30 years are there: 10958 days."""
    for station, payload in raw_payload().items():
        time = payload["daily"]["time"]
        assert len(time) == 10958, station
        assert time[0] == "1991-01-01" and time[-1] == "2020-12-31", station


# --------------------------------------------------------------------------
# the aggregation: the arithmetic that turns 10958 days into 36 numbers
# --------------------------------------------------------------------------

def test_rain_normal_is_a_mean_yearly_total_not_a_30_year_sum():
    """The classic climatology bug is summing 30 years of rain into "the normal".

    ``rain_mm`` is the WMO-style monthly normal: total the month within each
    year, then average across years. Recompute it here from the raw fixture by
    an independent route and pin the magnitude, so a missing
    ``.groupby("month").mean()`` shows up as a 30x error rather than a
    plausible-looking large number.
    """
    df, _ = parsed()
    payload = raw_payload()["salalah"]["daily"]
    raw = pd.DataFrame({"date": pd.to_datetime(payload["time"]),
                        "rain": payload["precipitation_sum"]})
    raw["year"] = raw["date"].dt.year
    raw["month"] = raw["date"].dt.month
    expected = raw.groupby(["year", "month"])["rain"].sum().groupby("month").mean()
    got = (df[(df["station"] == "salalah") & (df["variable"] == "rain_mm")]
           .set_index("month")["value"])
    for month in range(1, 13):
        assert got[month] == pytest.approx(round(expected[month], 1)), month
    # and the annual total is a plausible Salalah year, not a 30-year pile
    assert 40 < got.sum() < 400, got.sum()


def test_temperature_normals_match_an_independent_recomputation():
    df, _ = parsed()
    payload = raw_payload()["nizwa"]["daily"]
    raw = pd.DataFrame({"date": pd.to_datetime(payload["time"]),
                        "tmax": payload["temperature_2m_max"],
                        "tmin": payload["temperature_2m_min"]})
    raw["month"] = raw["date"].dt.month
    got = df[df["station"] == "nizwa"].set_index(["variable", "month"])["value"]
    for month in range(1, 13):
        for var, col in (("tmax_c", "tmax"), ("tmin_c", "tmin")):
            expected = round(raw[raw["month"] == month][col].mean(), 1)
            assert got[(var, month)] == pytest.approx(expected), (var, month)


def test_every_station_month_variable_appears_exactly_once():
    df, _ = parsed()
    assert not df.duplicated(["station", "month", "variable"]).any()
    assert set(df["station"]) == set(STATIONS)
    counts = df.groupby(["station", "variable"])["month"].nunique()
    assert (counts == 12).all()


def test_parse_output_is_sorted_and_typed():
    df, _ = parsed()
    assert list(df.index) == list(range(len(df)))
    assert df["month"].dtype.kind == "i"
    assert df["value"].dtype.kind == "f"
    assert df[["station", "month", "variable"]].apply(tuple, axis=1).is_monotonic_increasing
    # values are rounded to 0.1, so the API carries no fake precision
    assert (df["value"].round(1) == df["value"]).all()


def test_station_names_are_joined_by_key_not_by_position(tmp_path):
    """The fixture is a dict keyed by slug, and dict order is the fetch order.

    If the parser ever paired names with payloads positionally, reordering the
    JSON would silently relabel every station — Salalah's khareef would show up
    under Muscat. Reverse the fixture's key order; the table must not move.
    """
    parse = load("parse")
    expected, _ = parsed()
    out = doctored(tmp_path, lambda p: [p.__setitem__(k, p.pop(k))
                                        for k in list(p)[::-1]])
    assert list(json.loads(out.read_text(encoding="utf-8"))) != list(STATIONS)
    got, _ = parse(out)
    pd.testing.assert_frame_equal(got, expected)


# --------------------------------------------------------------------------
# the values themselves: pinned, including the surprising ones
# --------------------------------------------------------------------------

def test_headline_normals_are_pinned():
    """The dataset is static, so its numbers are a fixed answer, not a moving one.

    These are spot-checks against the published record for the two stations
    anyone will look up first. ERA5 runs a touch cool against gauge data —
    Seeb's official June mean max is ~38.6 C against 37.5 here — which is the
    reanalysis caveat in dataset.yaml, not a parsing error. Muscat's ~98 mm/yr
    is within a couple of mm of the published Seeb annual normal.
    """
    df, _ = parsed()
    d = df.set_index(["station", "variable", "month"])["value"]
    assert d[("muscat", "tmax_c", 6)] == pytest.approx(37.5)
    assert d[("muscat", "tmin_c", 1)] == pytest.approx(16.6)
    assert d[("salalah", "tmax_c", 7)] == pytest.approx(29.3)  # khareef overcast
    assert d[("saiq", "tmin_c", 1)] == pytest.approx(8.2)
    annual = df[df["variable"] == "rain_mm"].groupby("station")["value"].sum()
    assert annual["muscat"] == pytest.approx(98.5, abs=0.05)
    assert annual["sohar"] == pytest.approx(69.8, abs=0.05)


def test_salalahs_may_rain_normal_is_cyclone_dominated_and_that_is_correct():
    """Salalah's wettest normal month is May, not the khareef — deliberately.

    This looks wrong and is not. A WMO normal is a *mean*, and three of the
    thirty Mays carry a landfalling Arabian Sea system: 2018 (Cyclone Mekunu,
    214 mm on 25 May alone), 2020 (a late-May system, 29-31 May), and 2002.
    The other 27 Mays are near-dry — the median May total is under 2 mm. So
    the 39 mm May normal is three storms divided by thirty, and "fixing" it to
    a median or trimmed mean would stop being a WMO normal.

    The khareef itself is under-represented for a different reason (ERA5
    resolves cyclone rain far better than the monsoon's orographic fog-drizzle;
    see dataset.yaml). Both effects are pinned here so a future re-derivation
    that quietly changes either one is visible.
    """
    df, _ = parsed()
    salalah = (df[(df["station"] == "salalah") & (df["variable"] == "rain_mm")]
               .set_index("month")["value"])
    assert salalah.idxmax() == 5, "May is the wettest normal month at Salalah"
    assert salalah[5] > salalah[[6, 7, 8, 9]].sum()

    payload = raw_payload()["salalah"]["daily"]
    raw = pd.DataFrame({"rain": payload["precipitation_sum"]},
                       index=pd.to_datetime(payload["time"]))
    may = raw[raw.index.month == 5].groupby(raw[raw.index.month == 5].index.year)["rain"].sum()
    assert len(may) == 30
    assert may.median() < 2.0, "27 of 30 Mays are near-dry"
    wet = sorted(may[may > 100].index)
    assert wet == [2002, 2018, 2020], wet
    # those three years supply 1050.7 of the 1172.1 mm of May rain in the record
    assert may.loc[wet].sum() / may.sum() > 0.85, "the mean is three storms"


# --------------------------------------------------------------------------
# adversarial: the real fixture, doctored, must fail loudly
# --------------------------------------------------------------------------

def test_missing_station_fails_loudly(tmp_path):
    """A partial fetch must not publish normals for five of six stations."""
    parse = load("parse")
    bad = doctored(tmp_path, lambda p: p.pop("saiq"))
    with pytest.raises(ValueError, match="stations"):
        parse(bad)


def test_unknown_extra_station_fails_loudly(tmp_path):
    """stations.csv is the catalog; the snapshot may not quietly grow past it."""
    parse = load("parse")
    bad = doctored(tmp_path, lambda p: p.__setitem__("dubai", p["muscat"]))
    with pytest.raises(ValueError, match="stations"):
        parse(bad)


def test_truncated_history_fails_loudly(tmp_path):
    """One year is not a 1991-2020 normal, and it would still average cleanly."""
    parse = load("parse")

    def keep_one_year(p):
        d = p["sur"]["daily"]
        for key in ("time", "temperature_2m_max", "temperature_2m_min",
                    "precipitation_sum"):
            d[key] = d[key][:365]

    bad = doctored(tmp_path, keep_one_year)
    with pytest.raises(ValueError, match="daily rows"):
        parse(bad)


def test_missing_month_fails_loudly(tmp_path):
    """Drop every June from a station: 12 months must still be 12 months.

    Without a per-month completeness check this would raise a KeyError deep in
    the row loop or, worse, quietly emit a short table.
    """
    parse = load("parse")

    def drop_june(p):
        d = p["sohar"]["daily"]
        keep = [i for i, t in enumerate(d["time"]) if t[5:7] != "06"]
        for key in ("time", "temperature_2m_max", "temperature_2m_min",
                    "precipitation_sum"):
            d[key] = [d[key][i] for i in keep]

    bad = doctored(tmp_path, drop_june)
    # 900 days short of 10958 still clears MIN_DAYS, so this really does
    # exercise the month-completeness check and not the row-count one
    with pytest.raises(ValueError, match="month"):
        parse(bad)


def test_crlf_stations_csv_still_parses(tmp_path):
    """``.gitattributes`` pins ``*.csv text eol=lf``, and the parser still must
    not depend on it.

    That line normalises checkouts made after it landed; it does nothing for a
    working tree cloned before it, for a stations.csv pulled out of a zip or a
    GitHub archive, or for one round-tripped through a Windows editor. Any of
    those hands the parser CRLF while CI on Linux sees LF, and the last field
    of every row would pick up a trailing ``\\r`` if the file were opened
    carelessly. Asserting the committed bytes are LF would just be testing git;
    this asserts the thing that matters — the same normals come out either way.
    """
    parse = load("parse")
    expected, _ = parsed()
    crlf = tmp_path / "stations.csv"
    crlf.write_bytes(STATIONS_CSV.read_bytes().replace(b"\r\n", b"\n")
                     .replace(b"\n", b"\r\n"))
    assert b"\r\n" in crlf.read_bytes()
    parse.__globals__["STATIONS_CSV"] = crlf
    got, _ = parse(fixture_path())
    pd.testing.assert_frame_equal(got, expected)


def test_duplicate_station_slug_fails_loudly(tmp_path):
    """A repeated slug would collapse the catalog on *both* sides at once.

    stations.csv and the snapshot are both keyed by slug, so a duplicate
    shrinks each by the same station and every internal count still agrees —
    a five-station table published as six. Only an explicit check catches it.
    """
    parse = load("parse")
    rows = STATIONS_CSV.read_text(encoding="utf-8").splitlines()
    dupe = tmp_path / "stations.csv"
    dupe.write_text("\n".join(rows + [rows[1]]) + "\n", encoding="utf-8")
    parse.__globals__["STATIONS_CSV"] = dupe
    try:
        with pytest.raises(ValueError, match="duplicate"):
            parse(fixture_path())
    finally:
        parse.__globals__["STATIONS_CSV"] = STATIONS_CSV


def test_renamed_variable_fails_loudly(tmp_path):
    """If Open-Meteo renames a daily variable, stop rather than publish a hole."""
    parse = load("parse")

    def rename(p):
        d = p["muscat"]["daily"]
        d["temperature_2m_maximum"] = d.pop("temperature_2m_max")

    bad = doctored(tmp_path, rename)
    with pytest.raises(Exception):
        parse(bad)


def test_nulls_are_dropped_per_variable_not_per_day(tmp_path):
    """ERA5 gaps arrive as JSON nulls, and they are one variable's gap, not the day's.

    Two things must hold. Nulls must never reach the output as NaN. And a
    missing tmax must not take that day's *rainfall* with it: a frame-wide
    ``dropna()`` would drop the whole row, and because rain_mm is a monthly
    **total**, every dropped day biases it low — 3% of days missing is 3% less
    rain, silently. So tmax may move here; rain may not.
    """
    parse = load("parse")
    expected, _ = parsed()

    def punch_holes(p):
        d = p["muscat"]["daily"]
        for i in range(0, len(d["time"]), 30):  # ~365 days, still >= MIN_DAYS
            d["temperature_2m_max"][i] = None

    bad = doctored(tmp_path, punch_holes)
    df, _ = parse(bad)
    assert not df.isna().any().any()
    assert len(df) == 6 * 12 * 3
    key = ["station", "month", "variable"]
    rain = df[df["variable"] == "rain_mm"].set_index(key)["value"]
    rain_before = expected[expected["variable"] == "rain_mm"].set_index(key)["value"]
    pd.testing.assert_series_equal(rain, rain_before)
    # the untouched stations are untouched too
    other = df[df["station"] != "muscat"].set_index(key)["value"]
    other_before = expected[expected["station"] != "muscat"].set_index(key)["value"]
    pd.testing.assert_series_equal(other, other_before)


def test_empty_daily_block_fails_loudly(tmp_path):
    parse = load("parse")

    def empty(p):
        p["nizwa"]["daily"] = {"time": [], "temperature_2m_max": [],
                               "temperature_2m_min": [], "precipitation_sum": []}

    bad = doctored(tmp_path, empty)
    with pytest.raises(ValueError, match="daily rows"):
        parse(bad)
