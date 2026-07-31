from pathlib import Path

import pandas as pd
import pytest

from oman_data.run import _load_callable

FIXTURE_DIR = Path("tests/fixtures/population")


def fixture_path() -> Path:
    files = [p for p in FIXTURE_DIR.iterdir() if p.is_file()]
    assert len(files) == 1, "exactly one golden fixture expected"
    return files[0]


def test_parse_produces_tidy_population_table():
    parse = _load_callable(Path("pipelines/population/parse.py"), "parse")
    df, as_of = parse(fixture_path())
    assert list(df.columns) == ["year", "governorate_code", "population"]
    assert df["year"].between(2010, 2035).all()
    assert df["population"].gt(0).all()
    assert df["population"].lt(5_000_000).all()
    assert not df.isna().any().any()
    assert df["year"].dtype.kind == "i"
    assert df["population"].dtype.kind == "i"
    assert as_of == str(df["year"].max())


def test_governorate_codes_join_admin_geography():
    parse = _load_callable(Path("pipelines/population/parse.py"), "parse")
    df, _ = parse(fixture_path())
    names = pd.read_csv("pipelines/admin_geography/names.csv", encoding="utf-8")
    assert set(df["governorate_code"]) <= set(names["governorate_code"])
    assert df["governorate_code"].nunique() == 11


def test_mangled_layout_fails_loudly(tmp_path):
    parse = _load_callable(Path("pipelines/population/parse.py"), "parse")
    bad = tmp_path / "mangled.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(Exception):
        parse(bad)
