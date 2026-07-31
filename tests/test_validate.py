import pandas as pd
import pytest
from oman_data.schema import ColumnSpec, DatasetConfig
from oman_data.validate import validate_table

CFG = DatasetConfig(
    id="t", title_ar="ت", title_en="t", source_name="s", source_url="u",
    license="l", cadence="monthly",
    columns=(
        ColumnSpec("month", "str"),
        ColumnSpec("index", "float", min=30, max=300),
    ),
)


def good(n=10):
    return pd.DataFrame({
        "month": [f"2026-{i+1:02d}" for i in range(n)],
        "index": [100.0 + i for i in range(n)],
    })


CASES = [
    # (label, df, previous, expect_ok, expect_warning)
    ("clean", good(), None, True, False),
    ("clean_vs_same_previous", good(), good(), True, False),
    ("wrong_columns", good().rename(columns={"index": "value"}), None, False, False),
    ("extra_column", good().assign(extra=1), None, False, False),
    ("empty", good(0), None, False, False),
    ("null_value", good().assign(index=[None] + [100.0] * 9), None, False, False),
    ("bad_dtype", good().assign(index=["x"] * 10), None, False, False),
    ("below_min", good().assign(index=[5.0] + [100.0] * 9), None, False, False),
    ("above_max", good().assign(index=[500.0] + [100.0] * 9), None, False, False),
    ("row_collapse_error", good(4), good(10), False, False),
    ("row_change_warning", good(7), good(10), True, True),
]


@pytest.mark.parametrize("label,df,previous,expect_ok,expect_warning", CASES)
def test_validation(label, df, previous, expect_ok, expect_warning):
    result = validate_table(df, CFG, previous)
    assert result.ok is expect_ok, f"{label}: {[f.message for f in result.findings]}"
    has_warning = any(f.level == "warning" for f in result.findings)
    assert has_warning is expect_warning, label


def test_int_column_accepts_int_rejects_float():
    cfg = DatasetConfig(
        id="t", title_ar="ت", title_en="t", source_name="s", source_url="u",
        license="l", cadence="annual",
        columns=(ColumnSpec("year", "int"), ColumnSpec("population", "int", min=0)),
    )
    ok_df = pd.DataFrame({"year": [2025], "population": [5000000]})
    assert validate_table(ok_df, cfg).ok
    bad_df = pd.DataFrame({"year": [2025.5], "population": [5000000]})
    assert not validate_table(bad_df, cfg).ok
