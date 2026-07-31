"""Parse the NCSI NSDP Consumer Price Index workbook into tidy monthly rows.

Source layout (single sheet "Dataset", IMF e-GDDS / SDMX "ECOFIN_DSD" export):
a metadata block, then a header row whose cells are the observation periods
("2013-01", "2013-02", ...) running left to right, then one row per series. Each
series row carries an SDMX indicator code — ``PCPI_IX`` for the all-items index,
``PCPI_CP_nn_IX`` for COICOP division nn, and ``PCPI*_WT`` for the basket weights
— plus an English descriptor.

Only the ``_IX`` rows are kept; the ``_WT`` rows are basket weights, not indices.
Group labels are slugified from the descriptor so a source rename surfaces as a
changed label rather than a silently mismapped one; the all-items series is
relabelled ``general``. Selectors are found by content, not by fixed row/column
numbers, because NSDP republishes the file monthly with one more month column.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")
MIN_MONTHS = 6

_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_INDEX_CODE = re.compile(r"^PCPI(?:_CP_(\d{2}))?_IX$")
_ANY_CODE = re.compile(r"^PCPI(?:_CP_\d{2})?_(?:IX|WT)$")
_NON_SLUG = re.compile(r"[^a-z0-9]+")

GENERAL = "general"


def parse(raw_path: Path) -> tuple[pd.DataFrame, str]:
    raw_path = Path(raw_path)
    if raw_path.suffix.lower() not in EXCEL_SUFFIXES:
        raise ValueError(f"unexpected file type for CPI source: {raw_path.name}")
    wide = pd.read_excel(raw_path, sheet_name=0, header=None)
    df = _tidy(wide)
    if list(df.columns) != ["month", "group", "index"] or df.empty:
        raise ValueError(f"unexpected layout in {raw_path.name}")
    return df, df["month"].max()


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _slug(label: str) -> str:
    slug = _NON_SLUG.sub("_", label.casefold()).strip("_")
    if not slug:
        raise ValueError(f"series descriptor {label!r} does not yield a group label")
    return slug


def _month_columns(text: pd.DataFrame) -> tuple[int, dict[int, str]]:
    """Locate the period header row and its column -> "YYYY-MM" mapping."""
    best_row, best_months = None, {}
    for row in text.index:
        months = {c: text.iat[row, c] for c in text.columns if _MONTH.match(text.iat[row, c])}
        if len(months) > len(best_months):
            best_row, best_months = row, months
    if best_row is None or len(best_months) < MIN_MONTHS:
        raise ValueError(
            f"no header row carrying at least {MIN_MONTHS} 'YYYY-MM' periods "
            f"(best row matched {len(best_months)})"
        )
    return best_row, best_months


def _code_column(text: pd.DataFrame, header_row: int) -> int:
    best_col, best_hits = None, 0
    for col in text.columns:
        hits = sum(
            1 for row in text.index if row > header_row and _ANY_CODE.match(text.iat[row, col])
        )
        if hits > best_hits:
            best_col, best_hits = col, hits
    if best_col is None:
        raise ValueError("no column holds SDMX CPI indicator codes (PCPI_IX / PCPI_CP_nn_IX)")
    return best_col


def _descriptor_column(text: pd.DataFrame, header_row: int, code_col: int) -> int:
    for col in text.columns:
        if text.iat[header_row, col].casefold() == "descriptor":
            return col
    fallback = code_col + 1
    if fallback not in text.columns:
        raise ValueError("no 'Descriptor' column beside the indicator codes")
    return fallback


def _as_index(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return None if pd.isna(value) else float(value)
    text = str(value).replace(",", "").strip()
    if not re.fullmatch(r"-?\d+(\.\d+)?", text):
        return None
    return float(text)


def _trim(wide: pd.DataFrame) -> pd.DataFrame:
    """Drop the workbook's trailing empty columns (its declared sheet dimension is
    ~16k columns wide against ~165 used ones) and relabel so labels are positions."""
    trimmed = wide.dropna(axis=1, how="all").copy()
    trimmed.columns = range(trimmed.shape[1])
    return trimmed


def _tidy(wide: pd.DataFrame) -> pd.DataFrame:
    wide = _trim(wide)
    text = wide.map(_text)
    header_row, month_by_col = _month_columns(text)
    code_col = _code_column(text, header_row)
    desc_col = _descriptor_column(text, header_row, code_col)

    records = []
    seen_groups = set()
    for row in text.index:
        if row <= header_row:
            continue
        if not _INDEX_CODE.match(text.iat[row, code_col]):
            continue
        code = text.iat[row, code_col]
        group = GENERAL if code == "PCPI_IX" else _slug(text.iat[row, desc_col])
        if group in seen_groups:
            raise ValueError(f"duplicate CPI group series {group!r} at row {row + 1}")
        seen_groups.add(group)
        for col, month in month_by_col.items():
            index = _as_index(wide.iat[row, col])
            if index is None:
                raise ValueError(
                    f"non-numeric CPI value for {group} {month} at row {row + 1}"
                )
            records.append({"month": month, "group": group, "index": index})

    if GENERAL not in seen_groups:
        raise ValueError("no all-items series (PCPI_IX) in the workbook")

    df = pd.DataFrame.from_records(records, columns=["month", "group", "index"])
    df = df.astype({"month": "str", "group": "str", "index": "float64"})
    return df.sort_values(["month", "group"], ignore_index=True)
