import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from oman_data.api_build import build_api, is_stale
from oman_data.publish import publish_dataset
from oman_data.schema import ColumnSpec, DatasetConfig

NOW = datetime(2026, 7, 31, tzinfo=timezone.utc)

CFG_YAML = """\
id: cpi
title_ar: "الرقم القياسي"
title_en: "CPI"
source_name: "NCSI"
source_url: "https://data.gov.om/"
license: "OGL-Oman"
cadence: monthly
columns:
  - {name: month, dtype: str}
  - {name: index, dtype: float}
"""


def make_repo(tmp_path: Path, fetched_at: str) -> Path:
    (tmp_path / "pipelines" / "cpi").mkdir(parents=True)
    (tmp_path / "pipelines" / "cpi" / "dataset.yaml").write_text(CFG_YAML, encoding="utf-8")
    cfg = DatasetConfig(
        id="cpi", title_ar="الرقم القياسي", title_en="CPI", source_name="NCSI",
        source_url="https://data.gov.om/", license="OGL-Oman", cadence="monthly",
        columns=(ColumnSpec("month", "str"), ColumnSpec("index", "float")),
    )
    df = pd.DataFrame({"month": ["2026-06"], "index": [104.2]})
    publish_dataset(df, cfg, tmp_path / "data", as_of="2026-06", fetched_at=fetched_at)
    return tmp_path


def test_builds_catalog_and_latest(tmp_path):
    repo = make_repo(tmp_path, "2026-07-15T00:00:00Z")
    written = build_api(repo, repo / "api", NOW)
    catalog = json.loads((repo / "api" / "v1" / "datasets.json").read_text(encoding="utf-8"))
    assert catalog["datasets"][0]["id"] == "cpi"
    assert catalog["datasets"][0]["title_ar"] == "الرقم القياسي"
    assert catalog["datasets"][0]["stale"] is False
    latest = json.loads((repo / "api" / "v1" / "cpi" / "latest.json").read_text(encoding="utf-8"))
    assert latest["meta"]["stale"] is False
    assert latest["data"] == [{"month": "2026-06", "index": 104.2}]
    assert len(written) == 2


def test_arabic_not_escaped_in_output(tmp_path):
    repo = make_repo(tmp_path, "2026-07-15T00:00:00Z")
    build_api(repo, repo / "api", NOW)
    raw = (repo / "api" / "v1" / "datasets.json").read_text(encoding="utf-8")
    assert "الرقم" in raw  # ensure_ascii=False


def test_stale_flag_set_for_old_monthly(tmp_path):
    repo = make_repo(tmp_path, "2026-01-01T00:00:00Z")
    build_api(repo, repo / "api", NOW)
    latest = json.loads((repo / "api" / "v1" / "cpi" / "latest.json").read_text(encoding="utf-8"))
    assert latest["meta"]["stale"] is True


def test_unpublished_pipeline_skipped(tmp_path):
    repo = make_repo(tmp_path, "2026-07-15T00:00:00Z")
    (repo / "pipelines" / "ghost").mkdir()
    (repo / "pipelines" / "ghost" / "dataset.yaml").write_text(
        CFG_YAML.replace("id: cpi", "id: ghost"), encoding="utf-8")
    build_api(repo, repo / "api", NOW)
    catalog = json.loads((repo / "api" / "v1" / "datasets.json").read_text(encoding="utf-8"))
    assert [d["id"] for d in catalog["datasets"]] == ["cpi"]


@pytest.mark.parametrize("cadence,fetched,expect", [
    ("monthly", "2026-07-01T00:00:00Z", False),
    ("monthly", "2026-05-01T00:00:00Z", True),
    ("annual", "2025-09-01T00:00:00Z", False),
    ("annual", "2025-01-01T00:00:00Z", True),
    ("static", "2020-01-01T00:00:00Z", False),
])
def test_is_stale(cadence, fetched, expect):
    assert is_stale(cadence, fetched, NOW) is expect


def test_output_matches_contract_schemas(tmp_path):
    import jsonschema
    repo = make_repo(tmp_path, "2026-07-15T00:00:00Z")
    build_api(repo, repo / "api", NOW)
    root = Path(__file__).resolve().parents[1]
    catalog_schema = json.loads((root / "schemas" / "datasets.schema.json").read_text(encoding="utf-8"))
    latest_schema = json.loads((root / "schemas" / "latest.schema.json").read_text(encoding="utf-8"))
    catalog = json.loads((repo / "api" / "v1" / "datasets.json").read_text(encoding="utf-8"))
    jsonschema.validate(catalog, catalog_schema)
    for latest_path in (repo / "api" / "v1").glob("*/latest.json"):
        jsonschema.validate(json.loads(latest_path.read_text(encoding="utf-8")), latest_schema)
