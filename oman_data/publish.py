from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from oman_data.schema import DatasetConfig


def publish_dataset(
    df: pd.DataFrame,
    config: DatasetConfig,
    data_root: Path,
    as_of: str,
    fetched_at: str,
) -> Path:
    out = data_root / config.id
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{config.id}.csv", index=False, encoding="utf-8")
    df.to_parquet(out / f"{config.id}.parquet", index=False)
    meta = {
        "id": config.id,
        "as_of": as_of,
        "fetched_at": fetched_at,
        "source_name": config.source_name,
        "source_url": config.source_url,
        "license": config.license,
        "rows": len(df),
    }
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    changelog = out / "CHANGELOG.md"
    header = "" if changelog.exists() else f"# {config.id} changelog\n\n"
    with changelog.open("a", encoding="utf-8") as f:
        f.write(f"{header}- {fetched_at}: as_of={as_of}, rows={len(df)}\n")
    return out


def load_published(data_root: Path, dataset_id: str) -> pd.DataFrame | None:
    csv = data_root / dataset_id / f"{dataset_id}.csv"
    if not csv.exists():
        return None
    return pd.read_csv(csv, encoding="utf-8")
