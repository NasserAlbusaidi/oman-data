import json
import re
from pathlib import Path

import jsonschema

from oman_data.schema import load_dataset_config

ROOT = Path(__file__).resolve().parents[1]

WORKFLOWS = ROOT / ".github" / "workflows"
# each refresh workflow runs one dataset per step: `python -m oman_data.run cpi`
_RUN_RE = re.compile(r"python -m oman_data\.run\s+(\w+)")
# which workflow owns which cadence
REFRESH_WORKFLOWS = {"monthly": "refresh-monthly.yml", "annual": "refresh-annual.yml"}


def all_configs():
    return sorted(ROOT.glob("pipelines/*/dataset.yaml"))


def scheduled_ids(workflow: str) -> set[str]:
    return set(_RUN_RE.findall((WORKFLOWS / workflow).read_text(encoding="utf-8")))


def test_at_least_nine_pipelines_exist():
    assert len(all_configs()) >= 9


def test_every_config_loads_and_is_bilingual():
    for cfg_path in all_configs():
        cfg = load_dataset_config(cfg_path)  # loader enforces title_ar/title_en
        assert cfg.title_ar.strip() and cfg.title_en.strip()


def test_every_pipeline_has_fetch_and_parse():
    for cfg_path in all_configs():
        assert (cfg_path.parent / "fetch.py").exists(), cfg_path.parent
        assert (cfg_path.parent / "parse.py").exists(), cfg_path.parent


def test_no_dataset_is_silently_unscheduled():
    """Every dataset is refreshed by the workflow its cadence names, or is static.

    ``population`` shipped on no schedule at all and nobody noticed for six
    datasets, which is the whole reason this guard exists: adding a pipeline
    without wiring it into a refresh workflow must fail here, not go quietly
    stale in production.
    """
    by_cadence: dict[str, set[str]] = {}
    for cfg_path in all_configs():
        cfg = load_dataset_config(cfg_path)
        by_cadence.setdefault(cfg.cadence, set()).add(cfg.id)

    for cadence, workflow in REFRESH_WORKFLOWS.items():
        assert scheduled_ids(workflow) == by_cadence.get(cadence, set()), workflow

    # static datasets are refreshed by nothing, on purpose — no source to poll
    all_scheduled = set().union(*(scheduled_ids(w) for w in REFRESH_WORKFLOWS.values()))
    assert not (by_cadence.get("static", set()) & all_scheduled)


def test_refresh_workflows_isolate_each_dataset():
    """One failing source must not skip the others, nor the commit of successes.

    The as-built workflow ran every dataset in one step; the first failure took
    the rest of the month's data with it.
    """
    for workflow in REFRESH_WORKFLOWS.values():
        text = (WORKFLOWS / workflow).read_text(encoding="utf-8")
        ids = scheduled_ids(workflow)
        assert text.count("continue-on-error: true") == len(ids), workflow
        for dataset_id in ids:
            assert f"id: {dataset_id}" in text, f"{workflow}: {dataset_id} has no step id"
            assert f"steps.{dataset_id}.outcome" in text, \
                f"{workflow}: {dataset_id}'s outcome is never checked"


def test_committed_api_tree_matches_contracts():
    catalog_path = ROOT / "api" / "v1" / "datasets.json"
    assert catalog_path.exists(), "run the pipelines before this test"
    catalog_schema = json.loads((ROOT / "schemas" / "datasets.schema.json").read_text(encoding="utf-8"))
    latest_schema = json.loads((ROOT / "schemas" / "latest.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(json.loads(catalog_path.read_text(encoding="utf-8")), catalog_schema)
    for latest in (ROOT / "api" / "v1").glob("*/latest.json"):
        jsonschema.validate(json.loads(latest.read_text(encoding="utf-8")), latest_schema)
