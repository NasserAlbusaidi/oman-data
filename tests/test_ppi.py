import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from oman_data.run import _load_callable, _load_module
from oman_data.schema import load_dataset_config
from oman_data.validate import validate_table

FIXTURE_DIR = Path("tests/fixtures/ppi")
PARSE_PY = Path("pipelines/ppi/parse.py")
CONFIG_PATH = Path("pipelines/ppi/dataset.yaml")

GROUPS = {"general_nonoil", "mining_quarrying", "manufacturing",
          "electrical_energy", "water"}


def fixture_path() -> Path:
    files = [p for p in FIXTURE_DIR.iterdir() if p.is_file()]
    assert len(files) == 1, "exactly one golden fixture expected"
    return files[0]


def parser():
    return _load_callable(PARSE_PY, "parse")


def payload() -> dict:
    return json.loads(fixture_path().read_text(encoding="utf-8"))


def test_parse_produces_tidy_long_ppi_table():
    df, as_of = parser()(fixture_path())
    assert list(df.columns) == ["quarter", "group", "index"]
    assert set(df["group"]) == GROUPS
    assert pd.api.types.is_string_dtype(df["quarter"])
    assert df["index"].dtype.kind == "f"
    assert not df.duplicated(["quarter", "group"]).any()
    assert not df.isna().any().any()
    assert as_of == df["quarter"].max()


def test_the_published_frame_satisfies_its_own_config():
    """Columns, dtypes and the declared bounds, checked by the same validator
    the refresh runs — a yaml bound that the real data violates would block a
    refresh in CI rather than here."""
    df, _ = parser()(fixture_path())
    result = validate_table(df, load_dataset_config(CONFIG_PATH))
    assert result.ok, [f.message for f in result.findings]
    assert [f.message for f in result.findings] == []


def test_all_five_groups_span_the_same_thirty_three_quarters():
    """Every section is complete over 2018Q1..2026Q1 — no raggedness to handle,
    unlike gdp. 33 quarters x 5 groups is the whole table."""
    df, as_of = parser()(fixture_path())
    quarters = sorted(set(df["quarter"]))
    assert quarters[0] == "2018Q1"
    assert quarters[-1] == "2026Q1"
    assert len(quarters) == 33
    for group in GROUPS:
        assert sorted(df[df["group"] == group]["quarter"]) == quarters, group
    assert len(df) == 165
    assert as_of == "2026Q1"


def test_quarter_labels_sort_chronologically_as_strings():
    """as_of is a plain max() over strings; that is only correct because the
    "YYYYQn" format happens to sort chronologically. 2018Q9 would not exist,
    but 2026Q1 vs 2025Q4 is the case that has to hold."""
    df, _ = parser()(fixture_path())
    quarters = sorted(set(df["quarter"]))
    assert quarters[-2:] == ["2025Q4", "2026Q1"]
    assert max(quarters) == "2026Q1"


def test_values_in_plausible_bands():
    df, _ = parser()(fixture_path())
    config = load_dataset_config(CONFIG_PATH)
    index_col = next(c for c in config.columns if c.name == "index")
    assert df["index"].between(index_col.min, index_col.max).all()
    # tighter than the declared bounds: an index on a 2018 base that had drifted
    # to 40 or 250 in eight years would be a story, not a routine refresh
    assert df["index"].between(60, 150).all()


def test_headline_figures_match_the_source():
    """Spot-checks against data.gov.om's own jhmydsg values (pinned 2026-08-05)."""
    df, _ = parser()(fixture_path())
    by_key = {(r.quarter, r.group): r.index for r in df.itertuples()}
    assert by_key[("2018Q1", "general_nonoil")] == pytest.approx(97.451387988434)
    assert by_key[("2026Q1", "general_nonoil")] == pytest.approx(108.820388005394)
    assert by_key[("2026Q1", "manufacturing")] == pytest.approx(113.85989932529)
    assert by_key[("2026Q1", "mining_quarrying")] == pytest.approx(109.362580912664)
    assert by_key[("2026Q1", "electrical_energy")] == pytest.approx(104.310724459607)
    assert by_key[("2026Q1", "water")] == pytest.approx(97.1557669322018)


def test_base_year_2018_averages_to_one_hundred_in_every_group():
    """The titles claim "2018 = 100"; this is the arithmetic that makes the
    claim checkable rather than asserted.

    The sole-base-year property is asserted through the parser's own
    ``base_year_complaints``, which is the same code path the runtime guard
    uses, so the two cannot drift apart.
    """
    module = _load_module(PARSE_PY)
    df, _ = module.parse(fixture_path())
    means = {
        group: df.loc[df["quarter"].str.startswith("2018Q")
                      & (df["group"] == group), "index"].mean()
        for group in sorted(GROUPS)
    }
    assert set(means) == GROUPS
    for group, mean in means.items():
        assert abs(mean - 100.0) <= 1.14, (group, mean)
    assert module.BASE_YEAR == 2018
    assert module.base_year_complaints(df) == []


def test_the_titles_name_the_base_year_the_parser_enforces():
    """Whoever bumps BASE_YEAR after a rebase must fix both titles too — the
    guard exists to protect a claim that lives in the published metadata."""
    module = _load_module(PARSE_PY)
    config = load_dataset_config(CONFIG_PATH)
    assert str(module.BASE_YEAR) in config.title_en
    arabic = str(module.BASE_YEAR).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
    assert arabic in config.title_ar


def test_the_config_declares_the_new_quarterly_cadence():
    assert load_dataset_config(CONFIG_PATH).cadence == "quarterly"


def test_the_percent_unit_is_pinned_exactly_as_the_source_declares_it():
    """These are index points; the source calls the unit "%". The wrong
    declaration is what the parser requires, so a change to it still raises."""
    module = _load_module(PARSE_PY)
    assert module.EXPECTED_UNIT == "%"
    assert module.EXPECTED_SCALE == 1
    assert {row["unit"] for row in payload()["data"]} == {"%"}
    assert {row["scale"] for row in payload()["data"]} == {1}


def doctored(tmp_path: Path, mutate) -> Path:
    """The golden fixture with one edit applied, written to a temp file."""
    data = payload()
    mutate(data)
    out = tmp_path / "doctored.json"
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return out


def test_unknown_commodity_fails_loudly(tmp_path):
    """25 of the cube's 30 commodities are subgroups nested inside the four
    sections; publishing one beside its parent would mix levels in one column."""
    def to_subgroup(data):
        data["data"][0]["commodities"]["name"] = "Refined petroleum products"

    with pytest.raises(ValueError, match="unexpected commodity"):
        parser()(doctored(tmp_path, to_subgroup))


def test_duplicate_group_fails_loudly(tmp_path):
    def clone_first(data):
        data["data"].append(copy.deepcopy(data["data"][0]))

    with pytest.raises(ValueError, match="arrived twice"):
        parser()(doctored(tmp_path, clone_first))


def test_wrong_indicator_fails_loudly(tmp_path):
    """The cube's other indicators are basket weights and year-on-year
    inflation; neither is an index level, and both share these dimensions."""
    def to_inflation(data):
        data["data"][0]["indicators"]["name"] = "Inflation (%)"

    with pytest.raises(ValueError, match="expected 'index value'"):
        parser()(doctored(tmp_path, to_inflation))


def test_missing_indicator_dimension_fails_loudly(tmp_path):
    """``check_totals`` cannot vouch for the pinned indicator dimension, so the
    parser checks the payload still declares it at all."""
    def drop_declaration(data):
        del data["dimensionFields"]["indicators"]

    with pytest.raises(ValueError, match="declares no 'indicators' dimension"):
        parser()(doctored(tmp_path, drop_declaration))


def test_missing_commodity_dimension_fails_loudly(tmp_path):
    """The fan-out dimension is exempted from ``check_totals``, so its absence
    is invisible to the shared guard and needs its own declaration check."""
    def drop_declaration(data):
        del data["dimensionFields"]["commodities"]

    with pytest.raises(ValueError, match="declares no 'commodities' dimension"):
        parser()(doctored(tmp_path, drop_declaration))


def test_unit_change_fails_loudly(tmp_path):
    def to_index_points(data):
        data["data"][0]["unit"] = "Index"

    with pytest.raises(ValueError, match=r"expected '%' at scale 1"):
        parser()(doctored(tmp_path, to_index_points))


def test_scale_change_fails_loudly(tmp_path):
    def rescale(data):
        data["data"][0]["scale"] = 100

    with pytest.raises(ValueError, match=r"expected '%' at scale 1"):
        parser()(doctored(tmp_path, rescale))


def test_governorate_region_fails_loudly(tmp_path):
    """This is the failure mode that actually threatens this dataset: the cube
    publishes a full, plausible 33-quarter index for Muscat and Dhofar, so a
    filter drift onto one is invisible to every range and completeness check."""
    def to_muscat(data):
        data["data"][0]["regions"] = {"key": 1000010, "name": "Muscat"}

    with pytest.raises(ValueError, match="'regions' is on member 'muscat'"):
        parser()(doctored(tmp_path, to_muscat))


def test_missing_group_fails_loudly(tmp_path):
    """Losing one section must fail rather than quietly publish four fifths of
    the dataset — the row-count gate alone would let it through."""
    def drop_water(data):
        data["data"] = [row for row in data["data"]
                        if row["commodities"]["name"] != "4- Water"]

    with pytest.raises(ValueError, match="missing commodity group"):
        parser()(doctored(tmp_path, drop_water))


def test_non_quarterly_frequency_fails_loudly(tmp_path):
    def to_annual(data):
        data["data"][0]["frequency"] = "A"

    with pytest.raises(ValueError, match="expected 'Q'"):
        parser()(doctored(tmp_path, to_annual))


def test_truncated_series_fails_loudly(tmp_path):
    """Dropping leading observations shifts every quarter label; the row's own
    endDate no longer matches the walk and must catch it."""
    def drop_leading(data):
        data["data"][0]["values"] = data["data"][0]["values"][4:]

    with pytest.raises(ValueError, match="truncated or misaligned"):
        parser()(doctored(tmp_path, drop_leading))


def test_enddate_in_a_different_quarter_of_the_same_year_fails_loudly(tmp_path):
    """The quarterly-specific trap. 2026-04-01 is 2026Q2 while the walk reaches
    2026Q1 — same year, so nothing but a quarter-level comparison catches it."""
    def shift_end(data):
        data["data"][0]["endDate"] = "2026-04-01T00:00:00"

    with pytest.raises(ValueError, match="truncated or misaligned"):
        parser()(doctored(tmp_path, shift_end))


def test_a_rebased_series_fails_loudly_at_parse_time(tmp_path):
    """The base-year claim has to hold at refresh time, not just against the
    pinned fixture: CI runs ``oman_data.run ppi`` and no tests at all, so
    without a runtime guard a rebased NCSI series would publish under the
    "2018 = 100" title with every test still green.

    2021 is deliberately the hardest case to catch of the seven candidate
    rebase years: it is the one that leaves the base-year means closest to 100
    (Water lands on 100.02, Manufacturing on 97.02). Mining still moves 6.33
    points, twice the tolerance, so the guard fires.
    """
    def rebase_to_2021(data):
        for row in data["data"]:
            # 2018Q1 is index 0, so 2021Q1..2021Q4 are indices 12..15
            mean_2021 = sum(row["values"][12:16]) / 4
            factor = 100.0 / mean_2021
            row["values"] = [v * factor for v in row["values"]]

    with pytest.raises(ValueError, match="base year is not evidenced"):
        parser()(doctored(tmp_path, rebase_to_2021))


def test_a_series_starting_after_the_base_year_fails_loudly(tmp_path):
    """The other way the claim dies: NCSI drops the base year's own quarters,
    exactly as its "Inflation (%)" series already does by starting at 2019Q1.
    An index whose base period is absent cannot be checked against its base."""
    def start_at_2019(data):
        for row in data["data"]:
            row["values"] = row["values"][4:]
            row["startDate"] = "2019-01-01T00:00:00"

    with pytest.raises(ValueError, match="carries 0 of the 4 2018 quarters"):
        parser()(doctored(tmp_path, start_at_2019))


def test_an_all_null_payload_fails_loudly(tmp_path):
    """Every group present and correctly labelled, every observation missing:
    the group and layout guards above all pass, and only the empty-frame check
    stands between this and publishing nothing under a green run."""
    def blank_every_value(data):
        for row in data["data"]:
            row["values"] = [None] * len(row["values"])

    with pytest.raises(ValueError, match="unexpected layout"):
        parser()(doctored(tmp_path, blank_every_value))


def test_mangled_layout_fails_loudly(tmp_path):
    bad = tmp_path / "mangled.json"
    bad.write_text('{"data": [{"nope": 1}]}', encoding="utf-8")
    with pytest.raises(Exception):
        parser()(bad)
