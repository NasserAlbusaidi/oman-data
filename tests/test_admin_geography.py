import re
from pathlib import Path

import pandas as pd

NAMES = Path("pipelines/admin_geography/names.csv")
ARABIC = re.compile(r"[\u0600-\u06FF]")
SLUG = re.compile(r"^[a-z0-9-]+$")


def test_names_csv_shape_and_counts():
    df = pd.read_csv(NAMES, encoding="utf-8")
    assert list(df.columns) == [
        "governorate_code", "governorate_en", "governorate_ar",
        "wilayat_code", "wilayat_en", "wilayat_ar",
    ]
    assert df["governorate_code"].nunique() == 11
    assert 55 <= len(df) <= 70  # wilayat count sanity band
    assert df["wilayat_code"].is_unique


def test_no_empty_cells_and_scripts_correct():
    df = pd.read_csv(NAMES, encoding="utf-8")
    assert not df.isna().any().any()
    assert df["governorate_ar"].map(lambda s: bool(ARABIC.search(s))).all()
    assert df["wilayat_ar"].map(lambda s: bool(ARABIC.search(s))).all()
    assert df["governorate_code"].str.match(r"^OM-[A-Z]{2}$").all()
    assert df["wilayat_code"].map(lambda s: bool(SLUG.match(s))).all()


def test_fetch_archives_the_curated_table(tmp_path):
    from oman_data.run import _load_callable
    fetch = _load_callable(Path("pipelines/admin_geography/fetch.py").resolve(), "fetch")
    raw = fetch(tmp_path / "2026-07-31")
    assert raw.name == "names.csv"
    assert raw.read_text(encoding="utf-8") == NAMES.read_text(encoding="utf-8")


def test_parse_returns_names_table(tmp_path):
    from oman_data.run import _load_callable
    parse = _load_callable(Path("pipelines/admin_geography/parse.py").resolve(), "parse")
    raw = tmp_path / "names.csv"
    raw.write_bytes(NAMES.read_bytes())
    df, as_of = parse(raw)
    assert len(df) >= 55
    assert list(df.columns) == [
        "governorate_code", "governorate_en", "governorate_ar",
        "wilayat_code", "wilayat_en", "wilayat_ar",
    ]
    assert as_of == "2026-07"
