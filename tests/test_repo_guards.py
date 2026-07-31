import json
from pathlib import Path

import jsonschema

from oman_data.schema import load_dataset_config

ROOT = Path(__file__).resolve().parents[1]


def all_configs():
    return sorted(ROOT.glob("pipelines/*/dataset.yaml"))


def test_at_least_three_pipelines_exist():
    assert len(all_configs()) >= 3


def test_every_config_loads_and_is_bilingual():
    for cfg_path in all_configs():
        cfg = load_dataset_config(cfg_path)  # loader enforces title_ar/title_en
        assert cfg.title_ar.strip() and cfg.title_en.strip()


def test_every_pipeline_has_fetch_and_parse():
    for cfg_path in all_configs():
        assert (cfg_path.parent / "fetch.py").exists(), cfg_path.parent
        assert (cfg_path.parent / "parse.py").exists(), cfg_path.parent


def test_committed_api_tree_matches_contracts():
    catalog_path = ROOT / "api" / "v1" / "datasets.json"
    assert catalog_path.exists(), "run the pipelines before this test"
    catalog_schema = json.loads((ROOT / "schemas" / "datasets.schema.json").read_text(encoding="utf-8"))
    latest_schema = json.loads((ROOT / "schemas" / "latest.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(json.loads(catalog_path.read_text(encoding="utf-8")), catalog_schema)
    for latest in (ROOT / "api" / "v1").glob("*/latest.json"):
        jsonschema.validate(json.loads(latest.read_text(encoding="utf-8")), latest_schema)
