import ast
import json
import re
from pathlib import Path

import jsonschema

from oman_data.schema import load_dataset_config

ROOT = Path(__file__).resolve().parents[1]

WORKFLOWS = ROOT / ".github" / "workflows"
# each refresh workflow runs one dataset per step: `python -m oman_data.run cpi`
_RUN_RE = re.compile(r"python -m oman_data\.run\s+(\w+)")
# every path a workflow hands to `git add`, one token per staged path
_GIT_ADD_RE = re.compile(r"git add ([^\n]*)")
# which workflow owns which cadence
REFRESH_WORKFLOWS = {"monthly": "refresh-monthly.yml",
                     "quarterly": "refresh-quarterly.yml",
                     "annual": "refresh-annual.yml"}


def all_configs():
    return sorted(ROOT.glob("pipelines/*/dataset.yaml"))


def workflow_text(workflow: str) -> str:
    return (WORKFLOWS / workflow).read_text(encoding="utf-8")


def scheduled_ids(workflow: str) -> set[str]:
    return set(_RUN_RE.findall(workflow_text(workflow)))


def staged_paths(workflow: str) -> list[str]:
    """Every path token the workflow stages, e.g. ``raw``, ``pipelines/x/y.csv``."""
    return [token
            for args in _GIT_ADD_RE.findall(workflow_text(workflow))
            for token in args.split()]


def defines_persist(parse_py: Path) -> bool:
    """Does this parse.py define a module-level ``persist``?

    Read with ``ast`` rather than imported: this guard must hold for a pipeline
    whose imports are broken or whose module body reaches for the network, and
    it is the *declaration* that matters, not a working call.
    """
    module = ast.parse(parse_py.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "persist":
            return True
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "persist" for t in node.targets):
            return True
    return False


def test_at_least_twelve_pipelines_exist():
    assert len(all_configs()) >= 12


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

    # The loop below iterates REFRESH_WORKFLOWS, so a cadence missing from that
    # map is checked by nothing — which is the original bug wearing a new hat.
    # Adding a third cadence made that reachable by omission, so the map is
    # itself pinned to the cadences actually in use.
    assert set(by_cadence) - {"static"} <= set(REFRESH_WORKFLOWS), sorted(by_cadence)

    for cadence, workflow in REFRESH_WORKFLOWS.items():
        assert scheduled_ids(workflow) == by_cadence.get(cadence, set()), workflow

    # static datasets are refreshed by nothing, on purpose — no source to poll
    all_scheduled = set().union(*(scheduled_ids(w) for w in REFRESH_WORKFLOWS.values()))
    assert not (by_cadence.get("static", set()) & all_scheduled)


def test_every_persisting_pipeline_is_staged_by_its_refresh_workflow():
    """A pipeline that curates its own input file must have that file committed.

    ``oman_data.run`` calls an optional ``persist`` hook so a pipeline can write
    back a history its upstream does not keep — fuel_prices does, because NSS
    drops a month the instant it rolls over. That write lands on the CI runner's
    disk, and the refresh workflows stage curated files *by hand* (a blanket
    ``git add pipelines`` would commit whatever a half-failed run left behind).
    So a new persisting pipeline that nobody wires into its workflow would
    quietly recompute, write, and throw away the only record of a month that no
    source can return: a green run that loses data. Only this guard catches it,
    because nothing in the runner can see what CI commits.
    """
    for cfg_path in all_configs():
        if not defines_persist(cfg_path.parent / "parse.py"):
            continue
        cfg = load_dataset_config(cfg_path)
        workflow = REFRESH_WORKFLOWS.get(cfg.cadence)
        assert workflow, (
            f"{cfg.id}'s parse.py defines persist but its cadence {cfg.cadence!r} "
            f"is on no refresh workflow — nothing would ever commit what it writes")
        prefix = f"pipelines/{cfg.id}"
        assert any(path.startswith(prefix) for path in staged_paths(workflow)), (
            f"{workflow} stages {staged_paths(workflow)} and nothing under "
            f"{prefix}/ — {cfg.id}'s persist hook writes a curated file on the "
            f"runner that no commit would carry, so the months it records are lost")


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
