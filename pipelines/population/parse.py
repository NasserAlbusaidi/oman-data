"""Parse NCSI Monthly Statistical Bulletin table 1 into tidy population rows.

Source layout (bulletin sheet 1, "Total Population Registered By Governorates"):
a small wide block with one row per governorate, English labels in one column and
Arabic labels in another, and several snapshot columns. Above the data rows sit a
year row (e.g. 2025 | 2026 | 2026) and a breakdown row (Total | Expatriate | Omani).
We keep only the "Total" columns, one per year, leftmost wins (the leftmost column
for a year is the most recent month's snapshot in this layout).

Selectors are found by content, not by fixed row/column numbers, because the
bulletin is republished monthly and the block shifts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

MIN_YEAR = 2010
MAX_YEAR = 2035

# Normalised source spelling -> governorate_code (ISO 3166-2:OM).
# Covers both the bulletin's own spellings and the official spellings in
# pipelines/admin_geography/names.csv, so either column can carry the labels.
GOVERNORATE_MAP = {
    "muscat": "OM-MA",
    "مسقط": "OM-MA",
    "dhofar": "OM-ZU",
    "ظفار": "OM-ZU",
    "musandam": "OM-MU",
    "مسندم": "OM-MU",
    "alburaimi": "OM-BU",
    "alburaymi": "OM-BU",
    "البريمي": "OM-BU",
    "addakhiliyah": "OM-DA",
    "addakhliyah": "OM-DA",
    "الداخلية": "OM-DA",
    "northalbatinah": "OM-BS",
    "albatinahnorth": "OM-BS",
    "شمالالباطنة": "OM-BS",
    "southalbatinah": "OM-BJ",
    "albatinahsouth": "OM-BJ",
    "جنوبالباطنة": "OM-BJ",
    "southashsharqiyah": "OM-SJ",
    "ashsharqiahsouth": "OM-SJ",
    "ashsharqiyahsouth": "OM-SJ",
    "جنوبالشرقية": "OM-SJ",
    "northashsharqiyah": "OM-SS",
    "ashsharqiahnorth": "OM-SS",
    "ashsharqiyahnorth": "OM-SS",
    "شمالالشرقية": "OM-SS",
    "adhdhahirah": "OM-ZA",
    "الظاهرة": "OM-ZA",
    "alwusta": "OM-WU",
    "الوسطى": "OM-WU",
}

EXPECTED_CODES = frozenset(GOVERNORATE_MAP.values())

_KEEP = re.compile(r"[^0-9A-Za-z؀-ۿ]+")


def parse(raw_path: Path) -> tuple[pd.DataFrame, str]:
    raw_path = Path(raw_path)
    if raw_path.suffix.lower() not in (".xlsx", ".xlsm", ".xls"):
        raise ValueError(f"unexpected file type for population source: {raw_path.name}")
    wide = pd.read_excel(raw_path, sheet_name=0, header=None)
    df = _tidy(wide)
    if list(df.columns) != ["year", "governorate_code", "population"] or df.empty:
        raise ValueError(f"unexpected layout in {raw_path.name}")
    return df, str(df["year"].max())


def _norm(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return _KEEP.sub("", str(value)).casefold()


def _as_year(value: object) -> int | None:
    text = _norm(value)
    if not re.fullmatch(r"\d{4}", text):
        return None
    year = int(text)
    return year if MIN_YEAR <= year <= MAX_YEAR else None


def _as_count(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if pd.isna(value) or value <= 0:
            return None
        return int(round(float(value)))
    text = str(value).replace(",", "").strip()
    if not re.fullmatch(r"\d+(\.\d+)?", text):
        return None
    number = float(text)
    return int(round(number)) if number > 0 else None


def _label_column(norm: pd.DataFrame) -> int:
    best_col, best_hits = None, 0
    for col in norm.columns:
        hits = norm[col].map(GOVERNORATE_MAP.get).dropna().nunique()
        if hits > best_hits:
            best_col, best_hits = col, hits
    if best_col is None or best_hits < len(EXPECTED_CODES):
        raise ValueError(
            f"no column holds all {len(EXPECTED_CODES)} governorate labels "
            f"(best column matched {best_hits})"
        )
    return best_col


def _year_columns(wide: pd.DataFrame, norm: pd.DataFrame, first_data_row: int) -> dict[int, int]:
    """Map spreadsheet column -> year, keeping only 'Total' columns, leftmost per year."""
    year_row = None
    years: dict[int, int] = {}
    for row in range(first_data_row):
        found = {c: y for c in wide.columns if (y := _as_year(wide.iat[row, c])) is not None}
        if len(found) >= 2:
            year_row, years = row, found
    if year_row is None:
        raise ValueError("no header row carrying two or more years above the data block")

    total_cols = set()
    for row in range(year_row + 1, first_data_row):
        candidates = {c for c in wide.columns if "total" in norm.iat[row, c]}
        if len(candidates) >= 2:
            total_cols = candidates
    if not total_cols:
        raise ValueError("no 'Total' breakdown row found between the year row and the data")

    picked: dict[int, int] = {}
    for col in sorted(years):
        year = years[col]
        if col in total_cols and year not in picked.values():
            picked[col] = year
    if not picked:
        raise ValueError("no column is both a year column and a 'Total' column")
    return picked


def _tidy(wide: pd.DataFrame) -> pd.DataFrame:
    norm = wide.map(_norm)
    label_col = _label_column(norm)
    data_rows = [r for r in wide.index if norm.iat[r, label_col] in GOVERNORATE_MAP]
    first_data_row = min(data_rows)
    year_by_col = _year_columns(wide, norm, first_data_row)

    records = []
    for row in data_rows:
        code = GOVERNORATE_MAP[norm.iat[row, label_col]]
        for col, year in year_by_col.items():
            count = _as_count(wide.iat[row, col])
            if count is None:
                raise ValueError(
                    f"non-numeric population for {code} year {year} "
                    f"at row {row + 1} column {col + 1}"
                )
            records.append({"year": year, "governorate_code": code, "population": count})

    df = pd.DataFrame.from_records(records, columns=["year", "governorate_code", "population"])
    df = df.drop_duplicates(subset=["year", "governorate_code"], keep="first")
    codes = set(df["governorate_code"])
    if codes != EXPECTED_CODES:
        raise ValueError(f"governorate coverage mismatch: missing {sorted(EXPECTED_CODES - codes)}")
    if len(df) != len(EXPECTED_CODES) * len(year_by_col):
        raise ValueError(f"expected {len(EXPECTED_CODES) * len(year_by_col)} rows, got {len(df)}")
    df = df.astype({"year": "int64", "governorate_code": "str", "population": "int64"})
    return df.sort_values(["year", "governorate_code"], ignore_index=True)
