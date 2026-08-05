from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from oman_data.schema import load_dataset_config

# How long since the last *successful refresh* before a dataset is flagged
# stale on the site. Every run rewrites `fetched_at` whether or not the source
# published anything new, so these windows detect a dead workflow or a dead
# upstream, not a slow-publishing statistician.
#
# quarterly=140: a quarter is ~91 days, so this is ~1.5 publication cycles —
# the same shape as monthly=45 against a ~30-day cycle. The refresh workflow
# polls monthly, so in normal operation `fetched_at` is never more than ~31
# days old and this never fires; 140 days is ~4 consecutive failed monthly runs,
# which is long enough that a transient portal outage does not brand current
# data as stale, and short enough that at most one quarter can go missing
# unnoticed. (annual=420 is a year plus a two-month grace period — 1.15 cycles,
# not 1.5 — because the annual workflow only polls four times a year and a
# tighter window would flag on a couple of missed runs.)
STALE_AFTER_DAYS = {"monthly": 45, "quarterly": 140, "annual": 420}


def is_stale(cadence: str, fetched_at_iso: str, now: datetime) -> bool:
    limit = STALE_AFTER_DAYS.get(cadence)
    if limit is None:  # static
        return False
    fetched = datetime.fromisoformat(fetched_at_iso.replace("Z", "+00:00"))
    return (now - fetched).days > limit


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_api(repo_root: Path, out_root: Path, now: datetime) -> list[Path]:
    written: list[Path] = []
    catalog: list[dict] = []
    for cfg_path in sorted(repo_root.glob("pipelines/*/dataset.yaml")):
        config = load_dataset_config(cfg_path)
        ddir = repo_root / "data" / config.id
        meta_path = ddir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stale = is_stale(config.cadence, meta["fetched_at"], now)
        full_meta = {
            **meta,
            "title_ar": config.title_ar,
            "title_en": config.title_en,
            "cadence": config.cadence,
            "stale": stale,
            "columns": [
                {"name": c.name, "dtype": c.dtype, "min": c.min, "max": c.max}
                for c in config.columns
            ],
            "notes": config.notes,
        }
        df = pd.read_csv(ddir / f"{config.id}.csv", encoding="utf-8")
        written.append(_write_json(
            out_root / "v1" / config.id / "latest.json",
            {"meta": full_meta, "data": df.to_dict(orient="records")},
        ))
        catalog.append({
            "id": config.id,
            "title_ar": config.title_ar,
            "title_en": config.title_en,
            "source_name": config.source_name,
            "cadence": config.cadence,
            "as_of": meta["as_of"],
            "fetched_at": meta["fetched_at"],
            "stale": stale,
            "rows": meta["rows"],
        })
    written.append(_write_json(
        out_root / "v1" / "datasets.json",
        {"generated_at": now.isoformat(), "datasets": catalog},
    ))
    return written
