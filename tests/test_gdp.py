import copy
import json
from pathlib import Path

import pytest

from oman_data.run import _load_callable, _load_module
from oman_data.schema import load_dataset_config

FIXTURE_DIR = Path("tests/fixtures/gdp")
PARSE_PY = Path("pipelines/gdp/parse.py")
CONFIG_PATH = Path("pipelines/gdp/dataset.yaml")


def fixture_path() -> Path:
    files = [p for p in FIXTURE_DIR.iterdir() if p.is_file()]
    assert len(files) == 1, "exactly one golden fixture expected"
    return files[0]


def parser():
    return _load_callable(PARSE_PY, "parse")


def test_parse_produces_tidy_long_gdp_table():
    df, as_of = parser()(fixture_path())
    assert list(df.columns) == ["year", "price_basis", "gdp_mn_omr"]
    assert df["year"].between(2000, 2035).all()
    assert df["year"].dtype.kind == "i"
    assert set(df["price_basis"]) == {"current", "constant"}
    assert not df.duplicated(["year", "price_basis"]).any()
    assert not df.isna().any().any()
    assert as_of == str(df["year"].max())


def test_the_final_year_is_ragged_and_still_published():
    """2024 exists at current prices only; long format keeps it rather than
    dropping the headline year or publishing it with a null."""
    df, as_of = parser()(fixture_path())
    current = df[df["price_basis"] == "current"]
    constant = df[df["price_basis"] == "constant"]
    assert list(current["year"]) == list(range(2010, 2025))
    assert list(constant["year"]) == list(range(2010, 2024))
    assert list(df[df["year"] == 2024]["price_basis"]) == ["current"]
    assert as_of == "2024"
    assert len(df) == 29


def test_values_in_plausible_bands():
    df, _ = parser()(fixture_path())
    assert df["gdp_mn_omr"].between(1_000, 200_000).all()
    # a riyals-vs-millions mixup, or a swap onto a per-capita indicator, breaks
    # this band hard: Oman's GDP has sat in the 24-43 bn OMR range since 2010
    assert df["gdp_mn_omr"].between(20_000, 60_000).all()


def test_headline_figures_match_the_source():
    """Spot-checks against data.gov.om's own zoangf values (pinned 2026-08-05)."""
    df, _ = parser()(fixture_path())
    by_key = {(int(r.year), r.price_basis): r.gdp_mn_omr for r in df.itertuples()}
    assert by_key[(2024, "current")] == pytest.approx(41_194.3)
    assert by_key[(2023, "current")] == pytest.approx(40_824.2)
    assert by_key[(2023, "constant")] == pytest.approx(37_674.5)
    assert by_key[(2010, "current")] == pytest.approx(24_990.0193717838)
    assert by_key[(2010, "constant")] == pytest.approx(26_293.7156309943)
    assert (2024, "constant") not in by_key


def test_base_year_2018_is_the_year_the_two_bases_agree():
    """The title claims "constant 2018 prices"; this is the arithmetic that
    makes that claim checkable — a rebased series would move this year.

    The independent evidence is the pair of raw figures below; the sole-year
    property is asserted through the parser's own ``agreeing_years``, which is
    the same code path the runtime guard uses, so the two cannot drift apart.
    """
    module = _load_module(PARSE_PY)
    df, _ = module.parse(fixture_path())
    by_key = {(int(r.year), r.price_basis): r.gdp_mn_omr for r in df.itertuples()}
    assert by_key[(2018, "current")] == pytest.approx(35_184.0)
    assert by_key[(2018, "constant")] == pytest.approx(35_184.0)
    assert module.BASE_YEAR == 2018
    assert module.agreeing_years(df) == [2018]


def test_the_titles_name_the_base_year_the_parser_enforces():
    """Whoever bumps BASE_YEAR after a rebase must fix both titles too — the
    guard exists to protect a claim that lives in the published metadata."""
    module = _load_module(PARSE_PY)
    config = load_dataset_config(CONFIG_PATH)
    assert str(module.BASE_YEAR) in config.title_en
    arabic = str(module.BASE_YEAR).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
    assert arabic in config.title_ar


def doctored(tmp_path: Path, mutate) -> Path:
    """The golden fixture with one edit applied, written to a temp file."""
    payload = json.loads(fixture_path().read_text(encoding="utf-8"))
    mutate(payload)
    out = tmp_path / "doctored.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return out


def test_unknown_price_type_fails_loudly(tmp_path):
    """The cube's third price-type member ("Total") publishes nothing for this
    indicator; a row on it means the filter or the cube changed."""
    def to_total(payload):
        payload["data"][0]["price-type"] = {"key": 1000000, "name": "Total"}

    with pytest.raises(ValueError, match="unexpected price type"):
        parser()(doctored(tmp_path, to_total))


def test_duplicate_price_basis_fails_loudly(tmp_path):
    def clone_first(payload):
        payload["data"].append(copy.deepcopy(payload["data"][0]))

    with pytest.raises(ValueError, match="twice"):
        parser()(doctored(tmp_path, clone_first))


def test_wrong_indicator_fails_loudly(tmp_path):
    """39 indicators share this cube and several share the unit; drifting onto
    one of them would publish a different aggregate under a gdp_ column."""
    def to_gni(payload):
        payload["data"][0]["indicators"]["name"] = "Gross National Income (GNI)"

    with pytest.raises(ValueError, match="expected 'gdp at market prices'"):
        parser()(doctored(tmp_path, to_gni))


def test_missing_indicator_dimension_fails_loudly(tmp_path):
    """``check_totals`` cannot vouch for the pinned indicator dimension, so the
    parser checks the payload still declares it at all."""
    def drop_declaration(payload):
        del payload["dimensionFields"]["indicators"]

    with pytest.raises(ValueError, match="declares no 'indicators' dimension"):
        parser()(doctored(tmp_path, drop_declaration))


def test_unit_change_fails_loudly(tmp_path):
    def to_plain_riyals(payload):
        payload["data"][0]["unit"] = "R.O"

    with pytest.raises(ValueError, match="gdp_mn_omr"):
        parser()(doctored(tmp_path, to_plain_riyals))


def test_scale_change_fails_loudly(tmp_path):
    def rescale(payload):
        payload["data"][0]["scale"] = 1000

    with pytest.raises(ValueError, match="gdp_mn_omr"):
        parser()(doctored(tmp_path, rescale))


def test_non_total_breakdown_fails_loudly(tmp_path):
    """A filter drift letting the institutional-sector or activity breakdowns
    through would publish a slice of GDP as the national figure."""
    def to_oil_and_gas(payload):
        payload["data"][0]["institutional-sector"] = {"key": 1000020,
                                                      "name": "Oil and Gas"}

    with pytest.raises(ValueError, match="institutional-sector"):
        parser()(doctored(tmp_path, to_oil_and_gas))


def test_non_oman_region_fails_loudly(tmp_path):
    """This cube's regions dimension carries 78 members and no "Total" member
    at all (verified live 2026-08-05); only "Oman" is the national figure."""
    def to_muscat(payload):
        payload["data"][0]["regions"] = {"key": 1000020, "name": "Muscat"}

    with pytest.raises(ValueError, match="regions"):
        parser()(doctored(tmp_path, to_muscat))


def test_a_rebased_constant_series_fails_loudly_at_parse_time(tmp_path):
    """The base-year claim has to hold at refresh time, not just against the
    pinned fixture: CI runs ``oman_data.run gdp`` and no tests at all, so
    without a runtime guard a rebased NCSI series would publish under the
    "constant 2018 prices" title with every test still green."""
    def rebase_to_2020(payload):
        row = next(r for r in payload["data"]
                   if r["price-type"]["name"] == "Constant Prices")
        # 2020 current / 2020 constant — revalues the series so that 2020, not
        # 2018, becomes the year the two bases agree
        factor = 29187.2 / 33611.2
        row["values"] = [value * factor for value in row["values"]]

    with pytest.raises(ValueError, match="rebased"):
        parser()(doctored(tmp_path, rebase_to_2020))


def test_a_one_sided_base_year_revision_fails_loudly(tmp_path):
    """The other way the claim dies: NCSI restates 2018 on one basis only, so
    no year agrees at all and the title's base year is no longer evidenced."""
    def revise_2018_current(payload):
        row = next(r for r in payload["data"]
                   if r["price-type"]["name"] == "Current Prices")
        row["values"][2018 - 2010] = 35_500.0

    with pytest.raises(ValueError, match=r"agree in years \[\]"):
        parser()(doctored(tmp_path, revise_2018_current))


def test_non_annual_frequency_fails_loudly(tmp_path):
    def to_quarterly(payload):
        payload["data"][0]["frequency"] = "Q"

    with pytest.raises(ValueError, match="frequency"):
        parser()(doctored(tmp_path, to_quarterly))


def test_truncated_series_fails_loudly(tmp_path):
    """Dropping leading observations shifts every year label; the row's own
    endDate no longer matches the walk and must catch it."""
    def drop_leading(payload):
        payload["data"][0]["values"] = payload["data"][0]["values"][3:]

    with pytest.raises(ValueError, match="endDate"):
        parser()(doctored(tmp_path, drop_leading))


def test_missing_price_basis_fails_loudly(tmp_path):
    """Losing the constant-price series entirely must fail, not quietly publish
    half the dataset — the row-count gate alone would let it through."""
    def drop_constant(payload):
        payload["data"] = [row for row in payload["data"]
                           if row["price-type"]["name"] != "Constant Prices"]

    with pytest.raises(ValueError, match="missing price basis"):
        parser()(doctored(tmp_path, drop_constant))


def test_mangled_layout_fails_loudly(tmp_path):
    bad = tmp_path / "mangled.json"
    bad.write_text('{"data": [{"nope": 1}]}', encoding="utf-8")
    with pytest.raises(Exception):
        parser()(bad)
