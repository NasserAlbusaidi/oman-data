"""Tidy Oman's monthly merchandise trade into month x flow rows.

Source layout (single sheet, IMF e-GDDS / SDMX "ECOFIN_DSD" export, same shape
as the NSDP CPI workbook): a metadata block declaring ``UNIT_MULT`` and ``FREQ``,
then a header row whose cells are the observation periods ("2006-07", "2006-08",
...) running left to right, then one row per series carrying an SDMX indicator
code plus an English descriptor.

Only the three national totals are kept — ``TMG_CIF_XDC`` (recorded merchandise
imports, cif), ``TXG_FOB_XDC`` (total merchandise exports, fob) and
``TRX_FOB_XDC`` (re-exports, fob); every other row is a component breakdown that
would double-count. Note the source's own hierarchy: total exports already
contains re-exports, so the two flows must not be summed.

The workbook declares its scale in the metadata block (``UNIT_MULT`` 6 = millions
of domestic currency) and the published values are already millions of OMR, so
VALUE_SCALE is 1.0. A change of declared scale or frequency raises rather than
silently shifting every number by six orders of magnitude;
test_values_are_plausible_millions pins the resulting band.

Selectors are found by content, not by fixed row/column numbers, because NSDP
republishes the file monthly with one more month column.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")
MIN_MONTHS = 6

VALUE_SCALE = 1.0  # workbook ships millions of OMR (UNIT_MULT 6), asserted below
UNIT_MULT_MILLIONS = "6"
FREQ_MONTHLY = "M"

# SDMX indicator code -> flow slug; pinned at discovery against the NSDP sheet
_FLOW_CODES = {
    "TMG_CIF_XDC": "imports",
    "TXG_FOB_XDC": "exports",
    "TRX_FOB_XDC": "re_exports",
}

_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_NUMBER = re.compile(r"-?\d+(\.\d+)?")


def parse(raw_path: Path) -> tuple[pd.DataFrame, str]:
    raw_path = Path(raw_path)
    if raw_path.suffix.lower() not in EXCEL_SUFFIXES:
        raise ValueError(f"unexpected file type for trade source: {raw_path.name}")
    wide = pd.read_excel(raw_path, sheet_name=0, header=None)
    df = _tidy(wide)
    if list(df.columns) != ["month", "flow", "value_omr_mn"] or df.empty:
        raise ValueError(f"unexpected layout in {raw_path.name}")
    return df, df["month"].max()


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _trim(wide: pd.DataFrame) -> pd.DataFrame:
    """Drop the workbook's trailing empty columns and relabel so labels are
    positions (its declared sheet dimension is far wider than its used range)."""
    trimmed = wide.dropna(axis=1, how="all").copy()
    trimmed.columns = range(trimmed.shape[1])
    return trimmed


def _declared(text: pd.DataFrame, field: str) -> str:
    """Read a value out of the sheet's leading SDMX metadata block."""
    for row in text.index:
        for col in text.columns:
            if text.iat[row, col] == field:
                for right in [c for c in text.columns if c > col]:
                    if text.iat[row, right]:
                        return text.iat[row, right]
    raise ValueError(f"no {field} declaration in the trade workbook metadata block")


def _check_declared_units(text: pd.DataFrame) -> None:
    unit_mult = _declared(text, "UNIT_MULT")
    if unit_mult != UNIT_MULT_MILLIONS:
        raise ValueError(
            f"workbook declares UNIT_MULT {unit_mult!r}, expected "
            f"{UNIT_MULT_MILLIONS!r} (millions) — VALUE_SCALE would be wrong"
        )
    freq = _declared(text, "FREQ")
    if freq != FREQ_MONTHLY:
        raise ValueError(f"workbook declares FREQ {freq!r}, expected monthly")


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
            1 for row in text.index if row > header_row and text.iat[row, col] in _FLOW_CODES
        )
        if hits > best_hits:
            best_col, best_hits = col, hits
    if best_col is None:
        raise ValueError(f"no column holds the SDMX trade codes {sorted(_FLOW_CODES)}")
    return best_col


def _as_value(value: object) -> float | None:
    """Millions of OMR, or None for a month the source has not published."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        raise ValueError(f"non-numeric trade value {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    if not _NUMBER.fullmatch(text):
        raise ValueError(f"non-numeric trade value {value!r}")
    return float(text)


def _tidy(wide: pd.DataFrame) -> pd.DataFrame:
    wide = _trim(wide)
    text = wide.map(_text)
    _check_declared_units(text)
    header_row, month_by_col = _month_columns(text)
    code_col = _code_column(text, header_row)

    records = []
    seen_flows = set()
    for row in text.index:
        if row <= header_row:
            continue
        flow = _FLOW_CODES.get(text.iat[row, code_col])
        if flow is None:
            continue
        if flow in seen_flows:
            raise ValueError(f"duplicate trade series {flow!r} at row {row + 1}")
        seen_flows.add(flow)
        for col, month in month_by_col.items():
            value = _as_value(wide.iat[row, col])
            if value is None:  # month not published yet, or a source gap
                continue
            records.append({"month": month, "flow": flow, "value_omr_mn": value * VALUE_SCALE})

    missing = set(_FLOW_CODES.values()) - seen_flows
    if missing:
        raise ValueError(f"trade workbook is missing the total series for {sorted(missing)}")

    df = pd.DataFrame.from_records(records, columns=["month", "flow", "value_omr_mn"])
    df = df.astype({"month": "str", "flow": "str", "value_omr_mn": "float64"})
    return df.sort_values(["month", "flow"], ignore_index=True)
