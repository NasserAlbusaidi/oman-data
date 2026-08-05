import copy
import json
from pathlib import Path

import pytest

from oman_data.run import _load_callable, _load_module
from oman_data.schema import load_dataset_config

FIXTURE_DIR = Path("tests/fixtures/oil_gas")
PARSE_PY = Path("pipelines/oil_gas/parse.py")
CONFIG_PATH = Path("pipelines/oil_gas/dataset.yaml")

VALUE_COLUMNS = ["crude_production_kbbl_day", "crude_exports_kbbl_day",
                 "crude_price_usd_bbl", "gas_production_mn_scf"]


def fixture_path() -> Path:
    files = [p for p in FIXTURE_DIR.iterdir() if p.is_file()]
    assert len(files) == 1, "exactly one golden fixture expected"
    return files[0]


def parser():
    return _load_callable(PARSE_PY, "parse")


def payload() -> dict:
    return json.loads(fixture_path().read_text(encoding="utf-8"))


def row_named(data: list[dict], fragment: str) -> dict:
    """The one series row whose indicator name contains ``fragment``."""
    hits = [r for r in data if fragment in r["indicator"]["name"]]
    assert len(hits) == 1, f"{fragment!r} matched {len(hits)} rows"
    return hits[0]


def test_parse_produces_tidy_wide_oil_gas_table():
    df, as_of = parser()(fixture_path())
    assert list(df.columns) == ["year"] + VALUE_COLUMNS
    assert df["year"].between(2000, 2035).all()
    assert df["year"].is_unique
    assert df["year"].dtype.kind == "i"
    assert not df.isna().any().any()
    assert as_of == str(df["year"].max())


def test_the_table_is_the_years_all_four_series_share():
    """Wide format cannot publish a ragged tail: gas reaches 2025 in the very
    payload committed here, the three oil series stop at 2023, and the years
    only gas covers are dropped rather than published with nulls."""
    data = payload()["data"]
    gas = row_named(data, "Natural  Gas")
    assert gas["endDate"][:4] == "2025" and len(gas["values"]) == 24
    for fragment in ("production of crude oil", "Exports of Crude Oil",
                     "Price of Crude Oil"):
        row = row_named(data, fragment)
        assert row["endDate"][:4] == "2023" and len(row["values"]) == 22

    df, as_of = parser()(fixture_path())
    assert list(df["year"]) == list(range(2002, 2024))
    assert as_of == "2023"
    assert len(df) == 22


def test_headline_figures_match_the_source():
    """Spot-checks against data.gov.om's own OMOLGS2016 values (pinned
    2026-08-05); each is readable in the committed fixture."""
    df, _ = parser()(fixture_path())
    by_year = {int(r.year): r for r in df.itertuples()}
    assert by_year[2023].crude_production_kbbl_day == pytest.approx(1048.7)
    assert by_year[2023].crude_exports_kbbl_day == pytest.approx(850.2)
    assert by_year[2023].crude_price_usd_bbl == pytest.approx(82.3)
    assert by_year[2023].gas_production_mn_scf == pytest.approx(1_908_026.67888404)
    assert by_year[2002].crude_production_kbbl_day == pytest.approx(897.0)
    assert by_year[2002].crude_exports_kbbl_day == pytest.approx(838.0)
    assert by_year[2002].crude_price_usd_bbl == pytest.approx(24.0)
    assert by_year[2002].gas_production_mn_scf == pytest.approx(789_361.0)


def test_values_in_plausible_bands():
    df, _ = parser()(fixture_path())
    # thousand barrels a day: Oman has run 0.7-1.1 mb/d across this window
    assert df["crude_production_kbbl_day"].between(500, 2_000).all()
    assert df["crude_exports_kbbl_day"].between(500, 2_000).all()
    # exports are drawn from production; a unit or column mixup breaks this
    assert (df["crude_exports_kbbl_day"] < df["crude_production_kbbl_day"]).all()
    assert df["crude_price_usd_bbl"].between(10, 200).all()
    # a yearly total in MNSCF, not a daily rate — a daily rate would be ~1/365th
    assert df["gas_production_mn_scf"].between(500_000, 5_000_000).all()
    assert len(df) >= 20, "expected two decades of annual history"


def test_declared_bounds_cover_the_data_and_catch_a_thousandfold_error():
    """The yaml bounds are the last line of defence if a unit declaration is
    ever changed *and* this pipeline is updated to accept it: every published
    value sits inside them, and the same value scaled by a thousand either way
    sits outside. The bounds are enforced by ``validate_table`` at publish
    time, which is what makes them worth pinning here."""
    df, _ = parser()(fixture_path())
    config = load_dataset_config(CONFIG_PATH)
    assert [c.name for c in config.columns] == list(df.columns)
    for column in config.columns:
        if column.name == "year":
            continue
        values = df[column.name]
        assert values.between(column.min, column.max).all(), column.name
        assert (values * 1000 > column.max).all(), column.name
        assert (values / 1000 < column.min).all(), column.name


def doctored(tmp_path: Path, mutate) -> Path:
    """The golden fixture with one edit applied, written to a temp file."""
    edited = payload()
    mutate(edited)
    out = tmp_path / "doctored.json"
    out.write_text(json.dumps(edited, ensure_ascii=False), encoding="utf-8")
    return out


def test_a_unit_regime_change_fails_loudly(tmp_path):
    """The drift this dataset is most exposed to, and it is not hypothetical:
    the *monthly* series for this same gas indicator is published under "MNCM"
    (verified live 2026-08-05), three orders of magnitude away from the annual
    "MNSCF". If the annual series ever moved onto that unit, the numbers would
    still look like numbers and ``gas_production_mn_scf`` would be a lie. The
    guard runs inside ``parse``, which is what the refresh workflow calls on
    live data, so this fails the refresh rather than publishing."""
    def to_cubic_metres(edited):
        row_named(edited["data"], "Natural  Gas")["unit"] = "MNCM"

    with pytest.raises(ValueError,
                       match=r"'gas_production_mn_scf' column would be a lie"):
        parser()(doctored(tmp_path, to_cubic_metres))


def test_a_units_valid_for_another_indicator_still_fails(tmp_path):
    """The expected unit is keyed per indicator, not a shared allowlist: the
    barrels unit that the two volume series legitimately declare must still be
    rejected on the price series."""
    def price_in_barrels(edited):
        row_named(edited["data"], "Price of Crude Oil")["unit"] = "(000) BBL"

    with pytest.raises(ValueError,
                       match=r"'crude_price_usd_bbl' column would be a lie"):
        parser()(doctored(tmp_path, price_in_barrels))


def test_scale_change_fails_loudly(tmp_path):
    def rescale(edited):
        row_named(edited["data"], "production of crude oil")["scale"] = 1000

    with pytest.raises(ValueError,
                       match=r"'crude_production_kbbl_day' column would be a lie"):
        parser()(doctored(tmp_path, rescale))


def test_unexpected_indicator_fails_loudly(tmp_path):
    """43 indicators share this cube; anything but the four pinned ones means
    the filter drifted onto a series with no column to land in."""
    def to_refined_products(edited):
        edited["data"][0]["indicator"]["name"] = "Total Production of Refined Products"

    with pytest.raises(ValueError, match="unexpected indicator"):
        parser()(doctored(tmp_path, to_refined_products))


def test_duplicate_indicator_fails_loudly(tmp_path):
    """Two rows for one indicator means a breakdown crossed in; one of them
    would otherwise overwrite the other's column silently."""
    def clone_first(edited):
        edited["data"].append(copy.deepcopy(edited["data"][0]))

    with pytest.raises(ValueError, match="arrived twice"):
        parser()(doctored(tmp_path, clone_first))


def test_the_second_aggregate_region_fails_loudly(tmp_path):
    """This cube carries two aggregate regions: "Oman" (1000000), which has the
    data, and "Total" (1000010), which returns nothing for these indicators.
    ``check_totals`` accepts anything named "Total", so only the explicit
    national-member check stops a row from that second aggregate."""
    def to_region_total(edited):
        edited["data"][0]["region"] = {"key": 1000010, "name": "Total"}

    with pytest.raises(ValueError, match="carries region"):
        parser()(doctored(tmp_path, to_region_total))


def test_governorate_region_fails_loudly(tmp_path):
    """The other 81 region members are governorates and wilayats."""
    def to_muscat(edited):
        edited["data"][0]["region"] = {"key": 1000020, "name": "Muscat"}

    with pytest.raises(ValueError, match=r"'region' is on member 'muscat'"):
        parser()(doctored(tmp_path, to_muscat))


def test_non_total_breakdown_on_an_unfiltered_dimension_fails_loudly(tmp_path):
    """Six of the eight dimensions are left unfiltered because they already
    arrive on their aggregate member; ``check_totals`` is what makes that an
    assertion rather than a hope."""
    def to_one_gas_use(edited):
        edited["data"][0]["gas-uses-type"] = {"key": 1000010,
                                              "name": "Industrial Uses"}

    with pytest.raises(ValueError,
                       match=r"'gas-uses-type' is on member 'industrial uses'"):
        parser()(doctored(tmp_path, to_one_gas_use))


def test_the_spelled_out_totals_are_recognised():
    """Two of this cube's aggregates are not named "Total" — "Total Refinery &
    Petroleum IndustriesCo" and "Total Gas Uses". If either fell out of the
    accepted set, every row would be rejected as a breakdown, so this pins the
    exact strings that keep the pipeline running at all."""
    module = _load_module(PARSE_PY)
    assert {"total refinery & petroleum industriesco", "total gas uses",
            "total", "oman"} == set(module._TOTAL_OK)
    for row in payload()["data"]:
        assert row["oil-refineries-in-oman"]["name"] == \
            "Total Refinery & Petroleum IndustriesCo"
        assert row["gas-uses-type"]["name"] == "Total Gas Uses"


def test_missing_dimension_fails_loudly(tmp_path):
    """A dimension the payload declares but a row omits is an unknown slice."""
    def drop_country(edited):
        del edited["data"][0]["country"]

    with pytest.raises(ValueError, match="is missing from the series row"):
        parser()(doctored(tmp_path, drop_country))


def test_missing_series_fails_loudly(tmp_path):
    """Losing one of the four upstream must fail, not publish a narrower table:
    the row-count gate would not notice, because the years all still line up."""
    def drop_exports(edited):
        edited["data"] = [r for r in edited["data"]
                          if "Exports" not in r["indicator"]["name"]]

    with pytest.raises(ValueError, match="missing series"):
        parser()(doctored(tmp_path, drop_exports))


def test_non_annual_frequency_fails_loudly(tmp_path):
    def to_monthly(edited):
        edited["data"][0]["frequency"] = "M"

    with pytest.raises(ValueError, match="declares frequency"):
        parser()(doctored(tmp_path, to_monthly))


def test_truncated_series_fails_loudly(tmp_path):
    """Dropping leading observations shifts every year label; the row's own
    endDate no longer matches the walk and must catch it."""
    def drop_leading(edited):
        edited["data"][0]["values"] = edited["data"][0]["values"][3:]

    with pytest.raises(ValueError, match="truncated or misaligned"):
        parser()(doctored(tmp_path, drop_leading))


def test_series_that_never_overlap_fail_loudly(tmp_path):
    """All four series present, each internally valid, but sharing no year:
    ``dropna`` then empties the frame and the last guard in ``parse`` has to
    catch it. Without that guard a zero-row table reaches ``validate_table``,
    where "table is empty" is an error — but by then the message no longer
    points at the cube, and ``parse`` would have claimed a successful read.
    Reachable for real if NCSI ever re-bases this cube's oil series onto a
    later start than the gas series' end."""
    def move_gas_to_the_2030s(edited):
        gas = row_named(edited["data"], "Natural  Gas")
        gas["startDate"] = "2030-01-01T00:00:00"
        gas["endDate"] = "2053-01-01T00:00:00"  # 24 values, so the walk agrees

    with pytest.raises(ValueError, match="unexpected layout"):
        parser()(doctored(tmp_path, move_gas_to_the_2030s))


def test_mangled_layout_fails_loudly(tmp_path):
    bad = tmp_path / "mangled.json"
    bad.write_text('{"data": [{"nope": 1}]}', encoding="utf-8")
    with pytest.raises(Exception):
        parser()(bad)
