from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

from oman_data.api_build import build_api
from oman_data.publish import load_published, publish_dataset
from oman_data.schema import load_dataset_config
from oman_data.validate import validate_table


def _load_callable(py_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"pipeline_{py_path.parent.name}_{py_path.stem}", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def run_dataset(dataset_id: str, repo_root: Path, now: datetime) -> bool:
    pdir = repo_root / "pipelines" / dataset_id
    config = load_dataset_config(pdir / "dataset.yaml")
    fetch = _load_callable(pdir / "fetch.py", "fetch")
    parse = _load_callable(pdir / "parse.py", "parse")

    raw_dir = repo_root / "raw" / dataset_id / now.strftime("%Y-%m-%d")
    raw_path = fetch(raw_dir)
    df, as_of = parse(raw_path)

    previous = load_published(repo_root / "data", dataset_id)
    result = validate_table(df, config, previous)
    for f in result.findings:
        print(f"[{dataset_id}] {f.level}: {f.message}")
    if not result.ok:
        print(f"[{dataset_id}] validation FAILED — nothing published, last-good preserved")
        return False

    fetched_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    publish_dataset(df, config, repo_root / "data", as_of, fetched_at)
    build_api(repo_root, repo_root / "api", now)
    print(f"[{dataset_id}] published as_of={as_of}, rows={len(df)}")
    return True


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m oman_data.run <dataset_id>")
        raise SystemExit(2)
    ok = run_dataset(sys.argv[1], Path.cwd(), datetime.now(timezone.utc))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
