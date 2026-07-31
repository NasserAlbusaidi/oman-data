import json
from pathlib import Path

import pytest

from oman_data.run import _load_callable

FIXTURE_DIR = Path("tests/fixtures/traffic_accidents")


def fixture_path() -> Path:
    files = [p for p in FIXTURE_DIR.iterdir() if p.is_file()]
    assert len(files) == 1, "exactly one golden fixture expected"
    return files[0]


def test_parse_produces_tidy_accidents_table():
    parse = _load_callable(Path("pipelines/traffic_accidents/parse.py"), "parse")
    df, as_of = parse(fixture_path())
    assert list(df.columns) == ["year", "metric", "count"]
    assert set(df["metric"]) == {"accidents", "deaths", "injuries"}
    assert df["year"].between(2000, 2035).all()
    assert df["count"].ge(0).all()
    assert df["year"].dtype.kind == "i"
    assert df["count"].dtype.kind == "i"
    assert not df.isna().any().any()
    assert not df.duplicated(["year", "metric"]).any()
    assert as_of == str(df["year"].max())


def test_known_series_shapes():
    parse = _load_callable(Path("pipelines/traffic_accidents/parse.py"), "parse")
    df, _ = parse(fixture_path())
    # ragged endings are real: deaths runs past accidents, injuries stops 2021
    assert df.loc[df["metric"] == "deaths", "year"].max() >= 2024
    assert df.loc[df["metric"] == "accidents", "year"].max() >= 2023
    for metric in ("accidents", "deaths", "injuries"):
        assert len(df[df["metric"] == metric]) >= 15, metric
    # sanity: deaths never exceed injuries+accidents scale absurdly
    assert df.loc[df["metric"] == "deaths", "count"].max() < 5_000


def test_mangled_layout_fails_loudly(tmp_path):
    parse = _load_callable(Path("pipelines/traffic_accidents/parse.py"), "parse")
    bad = tmp_path / "mangled.json"
    bad.write_text('{"data": [{"nope": 1}]}', encoding="utf-8")
    with pytest.raises(Exception):
        parse(bad)


def test_ragged_endings_are_preserved_verbatim():
    """The three series really do stop in different years — never pad or truncate.

    Pinned at discovery (2026-07-31): accidents 2002-2024, deaths 2002-2025,
    injuries 2002-2021. A wide layout would need nulls the gate forbids, so a
    regression that "tidies" this into a rectangle must fail here.
    """
    parse = _load_callable(Path("pipelines/traffic_accidents/parse.py"), "parse")
    df, as_of = parse(fixture_path())
    spans = {
        m: (int(g["year"].min()), int(g["year"].max()), len(g))
        for m, g in df.groupby("metric")
    }
    assert spans["accidents"] == (2002, 2024, 23)
    assert spans["injuries"] == (2002, 2021, 20)
    assert spans["deaths"] == (2002, 2025, 24)
    assert as_of == "2025"


def test_headline_figures_match_the_source():
    """Spot-checks against data.gov.om's own published figures."""
    parse = _load_callable(Path("pipelines/traffic_accidents/parse.py"), "parse")
    df, _ = parse(fixture_path())
    at = df.set_index(["metric", "year"])["count"]
    assert at[("accidents", 2024)] == 1_854
    assert at[("deaths", 2025)] == 595
    assert at[("injuries", 2021)] == 1_621
    # the long decline from the 2000s peak is the story this dataset tells
    assert at[("accidents", 2003)] == 10_197


def test_deaths_series_is_people_not_fatal_accidents():
    """``Deaths`` (people killed) must outnumber the source's ``Fatalities``.

    ``gehye`` carries both: ``Fatalities`` is the count of *accidents* that
    killed someone (320 in 2020, = 23.9% of that year's 1,341 accidents, which
    is exactly the source's ``% Accidents Fatalities`` series), while ``Deaths``
    is the count of *people* (371 in 2020). Publishing the wrong one would
    understate road deaths by ~14%, so pin the relationship.
    """
    parse = _load_callable(Path("pipelines/traffic_accidents/parse.py"), "parse")
    df, _ = parse(fixture_path())
    at = df.set_index(["metric", "year"])["count"]
    assert at[("deaths", 2020)] == 371  # Fatalities 2020 is 320 — not this series
    assert at[("accidents", 2020)] == 1_341


def doctored(tmp_path: Path, mutate, index: int = 0) -> Path:
    """The golden fixture with one series row edited, written to a temp file."""
    payload = json.loads(fixture_path().read_text(encoding="utf-8"))
    mutate(payload["data"][index])
    out = tmp_path / "doctored.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return out


def test_non_annual_frequency_fails_loudly(tmp_path):
    parse = _load_callable(Path("pipelines/traffic_accidents/parse.py"), "parse")

    def to_monthly(row):
        row["frequency"] = "M"

    with pytest.raises(ValueError, match="frequency"):
        parse(doctored(tmp_path, to_monthly))


def test_truncated_series_fails_loudly(tmp_path):
    """Dropping leading observations shifts every year label; the row's own
    endDate no longer matches the walk and must catch it."""
    parse = _load_callable(Path("pipelines/traffic_accidents/parse.py"), "parse")

    def drop_leading(row):
        row["values"] = row["values"][3:]

    with pytest.raises(ValueError, match="endDate"):
        parse(doctored(tmp_path, drop_leading))


def test_non_total_region_fails_loudly(tmp_path):
    """A filter drift letting governorate breakdowns through would publish a
    slice of the country as the national total."""
    parse = _load_callable(Path("pipelines/traffic_accidents/parse.py"), "parse")

    def to_governorate(row):
        row["regions"] = {"key": 1000020, "name": "Muscat"}

    with pytest.raises(ValueError, match="regions"):
        parse(doctored(tmp_path, to_governorate))


def test_non_total_cause_breakdown_fails_loudly(tmp_path):
    """``accidents-by-causes`` fans out to 13 members; only Total is national."""
    parse = _load_callable(Path("pipelines/traffic_accidents/parse.py"), "parse")

    def to_speed(row):
        row["accidents-by-causes"] = {"key": 1000010, "name": "Speed"}

    with pytest.raises(ValueError, match="accidents-by-causes"):
        parse(doctored(tmp_path, to_speed))


def test_unpinned_dimension_breakdown_fails_loudly(tmp_path):
    """``gender`` is pinned in fetch.py, but the parser must catch drift on any
    dimension — including one added upstream after this was written."""
    parse = _load_callable(Path("pipelines/traffic_accidents/parse.py"), "parse")

    def to_male(row):
        row["gender"] = {"key": 1000010, "name": "Male"}

    with pytest.raises(ValueError, match="gender"):
        parse(doctored(tmp_path, to_male))


def test_unexpected_indicator_fails_loudly(tmp_path):
    """``Fatalities`` is a different series; it must never be silently absorbed."""
    parse = _load_callable(Path("pipelines/traffic_accidents/parse.py"), "parse")

    def to_fatalities(row):
        row["indicators"] = {"key": 1000100, "name": "Fatalities"}

    with pytest.raises(ValueError, match="unexpected indicator"):
        parse(doctored(tmp_path, to_fatalities))


def test_duplicate_metric_fails_loudly(tmp_path):
    parse = _load_callable(Path("pipelines/traffic_accidents/parse.py"), "parse")
    payload = json.loads(fixture_path().read_text(encoding="utf-8"))
    payload["data"].append(json.loads(json.dumps(payload["data"][0])))
    out = tmp_path / "dupe.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        parse(out)


def test_missing_metric_fails_loudly(tmp_path):
    """Two of three series is not this dataset; publishing it would silently
    drop a whole metric from the API."""
    parse = _load_callable(Path("pipelines/traffic_accidents/parse.py"), "parse")
    payload = json.loads(fixture_path().read_text(encoding="utf-8"))
    payload["data"] = payload["data"][:2]
    out = tmp_path / "short.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        parse(out)


def test_rescaled_values_fail_loudly(tmp_path):
    """Counts are published at scale 1; a scale change would silently multiply
    every number in the table."""
    parse = _load_callable(Path("pipelines/traffic_accidents/parse.py"), "parse")

    def to_thousands(row):
        row["scale"] = 1000

    with pytest.raises(ValueError, match="scale"):
        parse(doctored(tmp_path, to_thousands))
