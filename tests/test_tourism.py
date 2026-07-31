import json
import re
from pathlib import Path

import pytest

from oman_data.run import _load_callable

FIXTURE_DIR = Path("tests/fixtures/tourism")
MONTH = re.compile(r"^\d{4}-\d{2}$")


def fixture_path() -> Path:
    files = [p for p in FIXTURE_DIR.iterdir() if p.is_file()]
    assert len(files) == 1, "exactly one golden fixture expected"
    return files[0]


def test_parse_produces_tidy_tourism_table():
    parse = _load_callable(Path("pipelines/tourism/parse.py"), "parse")
    df, as_of = parse(fixture_path())
    assert list(df.columns) == ["month", "guests", "occupancy_pct", "revenue_omr_mn"]
    assert df["month"].map(lambda m: bool(MONTH.match(m))).all()
    assert df["month"].is_unique
    assert not df.isna().any().any()
    assert df["guests"].dtype.kind == "i"
    assert as_of == df["month"].max()


def test_values_in_plausible_bands():
    parse = _load_callable(Path("pipelines/tourism/parse.py"), "parse")
    df, _ = parse(fixture_path())
    assert df["occupancy_pct"].between(0, 100).all()
    assert df["guests"].between(0, 2_000_000).all()
    recent = df[df["month"] >= "2019-01"]
    assert 1 <= recent["revenue_omr_mn"].median() <= 200, "revenue scale wrong?"
    assert len(df) >= 60, "expected years of monthly history"


def doctored(tmp_path: Path, mutate) -> Path:
    """The golden fixture with one series row edited, written to a temp file."""
    payload = json.loads(fixture_path().read_text(encoding="utf-8"))
    mutate(payload["data"][0])
    out = tmp_path / "doctored.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return out


def test_non_monthly_frequency_fails_loudly(tmp_path):
    parse = _load_callable(Path("pipelines/tourism/parse.py"), "parse")

    def to_quarterly(row):
        row["frequency"] = "Q"

    with pytest.raises(ValueError, match="frequency"):
        parse(doctored(tmp_path, to_quarterly))


def test_truncated_series_fails_loudly(tmp_path):
    """Dropping leading observations shifts every month label by that many
    months; the row's own endDate no longer matches and must catch it."""
    parse = _load_callable(Path("pipelines/tourism/parse.py"), "parse")

    def drop_leading(row):
        row["values"] = row["values"][6:]

    with pytest.raises(ValueError, match="endDate"):
        parse(doctored(tmp_path, drop_leading))


def test_mangled_layout_fails_loudly(tmp_path):
    parse = _load_callable(Path("pipelines/tourism/parse.py"), "parse")
    bad = tmp_path / "mangled.json"
    bad.write_text('{"data": [{"nope": 1}]}', encoding="utf-8")
    with pytest.raises(Exception):
        parse(bad)
