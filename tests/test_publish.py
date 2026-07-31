import json
import pandas as pd
from oman_data.schema import ColumnSpec, DatasetConfig
from oman_data.publish import publish_dataset, load_published

CFG = DatasetConfig(
    id="cpi", title_ar="الرقم القياسي", title_en="CPI", source_name="NCSI",
    source_url="https://data.gov.om/", license="OGL-Oman", cadence="monthly",
    columns=(ColumnSpec("month", "str"), ColumnSpec("index", "float")),
)

DF = pd.DataFrame({"month": ["2026-06"], "index": [104.2]})

def test_publish_writes_all_artifacts(tmp_path):
    out = publish_dataset(DF, CFG, tmp_path, as_of="2026-06", fetched_at="2026-07-31T12:00:00Z")
    assert (out / "cpi.csv").exists()
    assert (out / "cpi.parquet").exists()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta == {
        "id": "cpi", "as_of": "2026-06", "fetched_at": "2026-07-31T12:00:00Z",
        "source_name": "NCSI", "source_url": "https://data.gov.om/",
        "license": "OGL-Oman", "rows": 1,
    }
    changelog = (out / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "2026-06" in changelog

def test_publish_appends_changelog(tmp_path):
    publish_dataset(DF, CFG, tmp_path, as_of="2026-05", fetched_at="a")
    publish_dataset(DF, CFG, tmp_path, as_of="2026-06", fetched_at="b")
    lines = (tmp_path / "cpi" / "CHANGELOG.md").read_text(encoding="utf-8").strip().splitlines()
    assert len([l for l in lines if l.startswith("- ")]) == 2

def test_load_published_round_trip(tmp_path):
    publish_dataset(DF, CFG, tmp_path, as_of="2026-06", fetched_at="x")
    df = load_published(tmp_path, "cpi")
    assert list(df.columns) == ["month", "index"]
    assert len(df) == 1

def test_load_published_missing_returns_none(tmp_path):
    assert load_published(tmp_path, "nope") is None
