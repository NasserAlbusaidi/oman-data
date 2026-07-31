import re
from pathlib import Path

import pytest

from oman_data.run import _load_callable

FIXTURE_DIR = Path("tests/fixtures/cpi")
MONTH = re.compile(r"^\d{4}-\d{2}$")


def fixture_path() -> Path:
    files = [p for p in FIXTURE_DIR.iterdir() if p.is_file()]
    assert len(files) == 1
    return files[0]


def test_parse_produces_tidy_cpi_table():
    parse = _load_callable(Path("pipelines/cpi/parse.py"), "parse")
    df, as_of = parse(fixture_path())
    assert list(df.columns) == ["month", "group", "index"]
    assert df["month"].map(lambda m: bool(MONTH.match(m))).all()
    assert "general" in set(df["group"])
    assert df["index"].between(30, 300).all()
    assert not df.isna().any().any()
    assert as_of == df["month"].max()


def test_general_group_is_monthly_continuous():
    parse = _load_callable(Path("pipelines/cpi/parse.py"), "parse")
    df, _ = parse(fixture_path())
    months = sorted(df.loc[df["group"] == "general", "month"])
    assert len(months) == len(set(months))  # no duplicate months
    assert len(months) >= 6  # at least half a year of history


def test_mangled_layout_fails_loudly(tmp_path):
    parse = _load_callable(Path("pipelines/cpi/parse.py"), "parse")
    bad = tmp_path / "mangled.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(Exception):
        parse(bad)
