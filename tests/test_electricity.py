import json
from pathlib import Path

import pytest

from oman_data.run import _load_callable

FIXTURE_DIR = Path("tests/fixtures/electricity")


def fixture_path() -> Path:
    files = [p for p in FIXTURE_DIR.iterdir() if p.is_file()]
    assert len(files) == 1, "exactly one golden fixture expected"
    return files[0]


def test_parse_produces_tidy_electricity_table():
    parse = _load_callable(Path("pipelines/electricity/parse.py"), "parse")
    df, as_of = parse(fixture_path())
    assert list(df.columns) == ["year", "production_gwh", "consumption_gwh"]
    assert df["year"].between(2000, 2035).all()
    assert df["year"].is_unique
    assert df["year"].dtype.kind == "i"
    assert not df.isna().any().any()
    assert as_of == str(df["year"].max())


def test_values_in_plausible_bands():
    parse = _load_callable(Path("pipelines/electricity/parse.py"), "parse")
    df, _ = parse(fixture_path())
    assert df["production_gwh"].between(5_000, 200_000).all()
    assert df["consumption_gwh"].between(1_000, 200_000).all()
    # consumption can't exceed production plus imports; a unit mixup breaks this
    assert (df["consumption_gwh"] <= df["production_gwh"] * 1.2).all()
    assert len(df) >= 15, "expected two decades of annual history"


def test_latest_production_matches_the_source():
    """Spot-check against data.gov.om's own 2025 figure (pinned at discovery)."""
    parse = _load_callable(Path("pipelines/electricity/parse.py"), "parse")
    df, _ = parse(fixture_path())
    latest = df.loc[df["year"].idxmax()]
    assert latest["year"] == 2025
    assert latest["production_gwh"] == pytest.approx(50_973.5)
    assert latest["consumption_gwh"] == pytest.approx(45_033.1)


def doctored(tmp_path: Path, mutate) -> Path:
    """The golden fixture with one series row edited, written to a temp file."""
    payload = json.loads(fixture_path().read_text(encoding="utf-8"))
    mutate(payload["data"][0])
    out = tmp_path / "doctored.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return out


def test_non_annual_frequency_fails_loudly(tmp_path):
    parse = _load_callable(Path("pipelines/electricity/parse.py"), "parse")

    def to_monthly(row):
        row["frequency"] = "M"

    with pytest.raises(ValueError, match="frequency"):
        parse(doctored(tmp_path, to_monthly))


def test_truncated_series_fails_loudly(tmp_path):
    """Dropping leading observations shifts every year label; the row's own
    endDate no longer matches the walk and must catch it."""
    parse = _load_callable(Path("pipelines/electricity/parse.py"), "parse")

    def drop_leading(row):
        row["values"] = row["values"][3:]

    with pytest.raises(ValueError, match="endDate"):
        parse(doctored(tmp_path, drop_leading))


def test_non_total_breakdown_fails_loudly(tmp_path):
    """A filter drift that lets governorate/station breakdowns through would
    blend several series into one column instead of publishing totals."""
    parse = _load_callable(Path("pipelines/electricity/parse.py"), "parse")

    def to_governorate(row):
        row["region"] = {"key": 1000020, "name": "Muscat"}

    with pytest.raises(ValueError, match="region"):
        parse(doctored(tmp_path, to_governorate))


def test_duplicate_indicator_fails_loudly(tmp_path):
    parse = _load_callable(Path("pipelines/electricity/parse.py"), "parse")
    payload = json.loads(fixture_path().read_text(encoding="utf-8"))
    payload["data"].append(json.loads(json.dumps(payload["data"][0])))
    out = tmp_path / "dupe.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        parse(out)


def test_series_that_never_overlap_fail_loudly(tmp_path):
    """Both series present and internally valid, but sharing no year: dropping
    the years that lack either one then empties the frame, and the last guard
    in ``parse`` has to catch it. Without it a zero-row table reaches
    ``validate_table``, where "table is empty" is an error — but by then the
    message no longer points at the source, and ``parse`` would have claimed a
    successful read. (Found while binding the identical guard in the oil_gas
    pipeline, which inherited it from this file.)"""
    parse = _load_callable(Path("pipelines/electricity/parse.py"), "parse")

    def move_production_to_the_2030s(row):
        row["startDate"] = "2030-01-01T00:00:00"
        row["endDate"] = "2051-01-01T00:00:00"  # 22 values, so the walk agrees

    with pytest.raises(ValueError, match="unexpected layout"):
        parse(doctored(tmp_path, move_production_to_the_2030s))


def test_mangled_layout_fails_loudly(tmp_path):
    parse = _load_callable(Path("pipelines/electricity/parse.py"), "parse")
    bad = tmp_path / "mangled.json"
    bad.write_text('{"data": [{"nope": 1}]}', encoding="utf-8")
    with pytest.raises(Exception):
        parse(bad)
