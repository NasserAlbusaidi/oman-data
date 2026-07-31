from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

from oman_data.api_build import build_api
from oman_data.publish import load_published, publish_dataset
from oman_data.schema import load_dataset_config
from oman_data.validate import validate_table


def _load_module(py_path: Path):
    spec = importlib.util.spec_from_file_location(f"pipeline_{py_path.parent.name}_{py_path.stem}", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_callable(py_path: Path, name: str):
    return getattr(_load_module(py_path), name)


def run_dataset(dataset_id: str, repo_root: Path, now: datetime) -> bool:
    pdir = repo_root / "pipelines" / dataset_id
    config = load_dataset_config(pdir / "dataset.yaml")
    fetch = _load_callable(pdir / "fetch.py", "fetch")
    parse_module = _load_module(pdir / "parse.py")
    parse = getattr(parse_module, "parse")
    # Optional hook. A pipeline that curates an input file of its own — a
    # history the upstream source does not keep, which is the whole reason the
    # file exists — defines `persist(df)` next to `parse` to write that file
    # back. Most pipelines have nothing to persist and simply omit it.
    persist = getattr(parse_module, "persist", None)

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

    # Strictly after the gate above: a frame validation rejected must never
    # reach a curated file, because that file is the only record of months the
    # source has already dropped. Persisting a bad month there would make it the
    # truth every later run cross-checks against — and the correct reading, when
    # it arrived, would be the thing that got rejected.
    if persist is not None:
        for month in persist(df) or ():
            print(f"[{dataset_id}] curated {month} into the pipeline's own history")

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
