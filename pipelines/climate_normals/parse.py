"""Compute 1991-2020 monthly climate normals from the archived Open-Meteo daily data.

``tmax_c``/``tmin_c`` are the mean of the daily maximum/minimum over all days
of a calendar month across all 30 years. ``rain_mm`` is the WMO-style
precipitation normal: total the month *within each year first*, then average
those 30 totals. Averaging the daily values instead would give a millimetre-
per-day figure roughly thirty times too small, and summing the whole record
would give a thirty-year pile roughly thirty times too large — both are
plausible-looking numbers, which is why the arithmetic is pinned by an
independent recomputation in the tests.

Two things worth knowing about the inputs:

* ``daily`` is a columnar block — one ``time`` array and one parallel array per
  variable — so the frame is assembled column-wise, not row-wise.
* Missing values arrive as JSON ``null`` and are dropped **per variable**, not
  per day. A day whose tmax is missing is still a real rain observation, and
  discarding it would quietly bias that month's rainfall *total* low; the two
  are independent measurements and are treated as such. (The 2026-07-31
  snapshot has no gaps at all in any of the six stations, so this is
  future-proofing rather than a correction being applied today.)

Everything the parser depends on is checked before it publishes: the station
set must equal the curated catalog, each variable must have close to 30 years
of days, and all twelve months must be present. A short or lopsided record
still averages perfectly cleanly, so silence here would mean shipping a
"normal" computed from the wrong period.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

AS_OF = "1991-2020"
STATIONS_CSV = Path(__file__).parent / "stations.csv"

COLUMNS = ["station", "station_ar", "station_en", "month", "variable", "value"]
# 30 years is 10958 days; allow a little slack for ERA5 gaps, but not a lot --
# a materially shorter record is not a 1991-2020 normal
MIN_DAYS = 10_000


def _station_names() -> dict[str, tuple[str, str]]:
    """The curated catalog, keyed by slug.

    The duplicate check is not paranoia: a repeated slug would collapse two
    rows into one dict entry, and since the snapshot is keyed by slug too, both
    sides would shrink by the same station and every downstream count would
    still agree with itself. The result would be a quietly five-station
    "six-station" dataset.
    """
    with STATIONS_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{STATIONS_CSV.name} has no station rows")
    names = {r["station"]: (r["station_ar"], r["station_en"]) for r in rows}
    if len(names) != len(rows):
        raise ValueError(f"{STATIONS_CSV.name} has duplicate station slugs: "
                         f"{[r['station'] for r in rows]}")
    return names


def _monthly(station: str, variable: str, s: pd.Series, how: str) -> pd.Series:
    """Monthly normal of one variable's non-null daily series.

    ``how="mean"`` averages the daily values (temperatures); ``how="total"``
    totals each year-month and then averages those totals (precipitation).
    """
    s = s.dropna()
    if len(s) < MIN_DAYS:
        raise ValueError(f"{station}/{variable}: only {len(s)} daily rows "
                         f"(need >= {MIN_DAYS}) — fetch broken or period truncated")
    by_month = s.index.month
    if how == "mean":
        normal = s.groupby(by_month).mean()
    else:
        normal = (s.groupby([s.index.year, by_month]).sum()
                   .groupby(level=1).mean())
    missing = [m for m in range(1, 13) if m not in normal.index]
    if missing:
        raise ValueError(f"{station}/{variable}: no data for month(s) {missing} "
                         f"— the record does not cover a full year")
    return normal


def parse(raw_path: Path) -> tuple[pd.DataFrame, str]:
    raw_path = Path(raw_path)
    combined = json.loads(raw_path.read_text(encoding="utf-8"))
    names = _station_names()
    if set(combined) != set(names):
        raise ValueError(f"stations in {raw_path.name} != stations.csv: "
                         f"{sorted(combined)} vs {sorted(names)}")

    records: list[tuple] = []
    for station in names:
        daily = combined[station]["daily"]
        frame = pd.DataFrame({
            "tmax": daily["temperature_2m_max"],
            "tmin": daily["temperature_2m_min"],
            "rain": daily["precipitation_sum"],
        }, index=pd.to_datetime(daily["time"]))
        tmax = _monthly(station, "tmax_c", frame["tmax"], "mean")
        tmin = _monthly(station, "tmin_c", frame["tmin"], "mean")
        rain = _monthly(station, "rain_mm", frame["rain"], "total")
        station_ar, station_en = names[station]
        for month in range(1, 13):
            records.append((station, station_ar, station_en, month,
                            "tmax_c", round(float(tmax[month]), 1)))
            records.append((station, station_ar, station_en, month,
                            "tmin_c", round(float(tmin[month]), 1)))
            records.append((station, station_ar, station_en, month,
                            "rain_mm", round(float(rain[month]), 1)))

    df = pd.DataFrame(records, columns=COLUMNS)
    df = (df.astype({"station": str, "station_ar": str, "station_en": str,
                     "month": int, "variable": str, "value": float})
            .sort_values(["station", "month", "variable"], ignore_index=True))
    expected_rows = len(names) * 12 * 3
    if len(df) != expected_rows:
        raise ValueError(f"expected {expected_rows} normal rows, got {len(df)}")
    if df.isna().any().any():
        raise ValueError(f"null values in the normals table from {raw_path.name}")
    return df, AS_OF
