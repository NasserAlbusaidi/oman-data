from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(Exception):
    pass


VALID_CADENCES = {"static", "monthly", "annual"}
VALID_DTYPES = {"int", "float", "str"}

_REQUIRED = [
    "id", "title_ar", "title_en", "source_name", "source_url",
    "license", "cadence", "columns",
]


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True)
class DatasetConfig:
    id: str
    title_ar: str
    title_en: str
    source_name: str
    source_url: str
    license: str
    cadence: str
    columns: tuple[ColumnSpec, ...]
    notes: str = ""


def load_dataset_config(path: Path) -> DatasetConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: not a mapping")
    missing = [f for f in _REQUIRED if not raw.get(f)]
    if missing:
        raise ConfigError(f"{path}: missing/empty fields: {missing}")
    if raw["cadence"] not in VALID_CADENCES:
        raise ConfigError(f"{path}: cadence {raw['cadence']!r} not in {sorted(VALID_CADENCES)}")
    columns = []
    for c in raw["columns"]:
        if c.get("dtype") not in VALID_DTYPES:
            raise ConfigError(f"{path}: column {c.get('name')!r} dtype {c.get('dtype')!r} not in {sorted(VALID_DTYPES)}")
        columns.append(ColumnSpec(
            name=c["name"], dtype=c["dtype"],
            min=c.get("min"), max=c.get("max"),
        ))
    return DatasetConfig(
        id=raw["id"], title_ar=raw["title_ar"], title_en=raw["title_en"],
        source_name=raw["source_name"], source_url=raw["source_url"],
        license=raw["license"], cadence=raw["cadence"],
        columns=tuple(columns), notes=raw.get("notes", ""),
    )
