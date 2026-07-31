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


def test_parse_returns_names_table(tmp_path, monkeypatch):
    from oman_data.run import _load_callable
    # resolve BEFORE chdir; parse writes its boundaries artifact to a
    # cwd-relative data/ path, so isolate cwd or the test would overwrite
    # the real published boundaries file
    parse = _load_callable(Path("pipelines/admin_geography/parse.py").resolve(), "parse")
    monkeypatch.chdir(tmp_path)
    import json
    gj = {"type": "FeatureCollection", "features": []}
    raw = tmp_path / "omn_adm2.geojson"
    raw.write_text(json.dumps(gj), encoding="utf-8")
    df, as_of = parse(raw)
    assert len(df) >= 55
    assert as_of == "2026-07"
    assert (tmp_path / "data" / "admin_geography" / "boundaries.geojson").exists()
