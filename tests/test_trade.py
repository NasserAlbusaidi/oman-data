import re
from pathlib import Path

import pandas as pd
import pytest

from oman_data.run import _load_callable

FIXTURE_DIR = Path("tests/fixtures/trade")
MONTH = re.compile(r"^\d{4}-\d{2}$")


def fixture_path() -> Path:
    files = [p for p in FIXTURE_DIR.iterdir() if p.is_file()]
    assert len(files) == 1, "exactly one golden fixture expected"
    return files[0]


def test_parse_produces_tidy_trade_table():
    parse = _load_callable(Path("pipelines/trade/parse.py"), "parse")
    df, as_of = parse(fixture_path())
    assert list(df.columns) == ["month", "flow", "value_omr_mn"]
    assert df["month"].map(lambda m: bool(MONTH.match(m))).all()
    assert set(df["flow"]) == {"imports", "exports", "re_exports"}
    assert not df.isna().any().any()
    assert as_of == df["month"].max()


def test_values_are_plausible_millions():
    parse = _load_callable(Path("pipelines/trade/parse.py"), "parse")
    df, _ = parse(fixture_path())
    # Oman's monthly merchandise imports/exports run in the hundreds to low
    # thousands of millions OMR. A scale error (riyals vs millions) misses
    # this band by six orders of magnitude, which is the point of the test.
    recent = df[df["month"] >= "2020-01"]
    for flow in ("imports", "exports"):
        med = recent.loc[recent["flow"] == flow, "value_omr_mn"].median()
        assert 100 <= med <= 10_000, f"{flow} median {med} — scale wrong?"
    assert (df["value_omr_mn"] >= 0).all()


def test_each_flow_is_monthly_unique_with_history():
    parse = _load_callable(Path("pipelines/trade/parse.py"), "parse")
    df, _ = parse(fixture_path())
    for flow in ("imports", "exports", "re_exports"):
        months = df.loc[df["flow"] == flow, "month"]
        assert months.is_unique, flow
        assert len(months) >= 24, f"{flow}: expected years of history"


def test_mangled_layout_fails_loudly(tmp_path):
    parse = _load_callable(Path("pipelines/trade/parse.py"), "parse")
    bad = tmp_path / "mangled.json"
    bad.write_text('{"data": [{"nope": 1}]}', encoding="utf-8")
    with pytest.raises(Exception):
        parse(bad)


def test_mangled_workbook_fails_loudly(tmp_path):
    """A .json only trips the suffix guard, one line into parse.

    NSDP serves a workbook, so the failure that will actually happen is a
    *well-formed* .xlsx whose sheet no longer looks like the export we pinned —
    a republish under a new layout, not a wrong file type. This one gets past
    the suffix check and read_excel and has to be caught by the content
    selectors and the UNIT_MULT/FREQ guards, which is the path the suffix test
    never reaches.
    """
    parse = _load_callable(Path("pipelines/trade/parse.py"), "parse")
    bad = tmp_path / "mangled.xlsx"
    pd.DataFrame({"a": [1], "b": [2]}).to_excel(bad, index=False)
    with pytest.raises(Exception):
        parse(bad)
