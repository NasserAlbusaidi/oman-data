from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from oman_data.schema import DatasetConfig


@dataclass(frozen=True)
class Finding:
    level: str  # "warning" | "error"
    message: str


@dataclass
class ValidationResult:
    findings: list[Finding]

    @property
    def ok(self) -> bool:
        return not any(f.level == "error" for f in self.findings)


def _dtype_ok(series: pd.Series, dtype: str) -> bool:
    if dtype == "int":
        return pd.api.types.is_integer_dtype(series)
    if dtype == "float":
        return pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series)
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def validate_table(
    df: pd.DataFrame,
    config: DatasetConfig,
    previous: pd.DataFrame | None = None,
) -> ValidationResult:
    findings: list[Finding] = []
    expected = [c.name for c in config.columns]
    if list(df.columns) != expected:
        findings.append(Finding("error", f"columns {list(df.columns)} != expected {expected}"))
        return ValidationResult(findings)
    if len(df) == 0:
        findings.append(Finding("error", "table is empty"))
        return ValidationResult(findings)
    if df.isna().any().any():
        null_cols = df.columns[df.isna().any()].tolist()
        findings.append(Finding("error", f"null values in columns: {null_cols}"))
    for col in config.columns:
        s = df[col.name]
        if not _dtype_ok(s, col.dtype):
            findings.append(Finding("error", f"{col.name}: dtype {s.dtype} incompatible with {col.dtype}"))
            continue
        if col.dtype in ("int", "float") and s.notna().any():
            if col.min is not None and (s.dropna() < col.min).any():
                findings.append(Finding("error", f"{col.name}: value below min {col.min}"))
            if col.max is not None and (s.dropna() > col.max).any():
                findings.append(Finding("error", f"{col.name}: value above max {col.max}"))
    if previous is not None and len(previous) > 0:
        ratio = len(df) / len(previous)
        if ratio < 0.5:
            findings.append(Finding("error", f"row count collapsed: {len(previous)} -> {len(df)}"))
        elif abs(ratio - 1.0) > 0.2:
            findings.append(Finding("warning", f"row count changed >20%: {len(previous)} -> {len(df)}"))
    return ValidationResult(findings)
