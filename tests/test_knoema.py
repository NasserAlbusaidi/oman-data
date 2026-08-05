import json
from pathlib import Path

import pytest

from oman_data import knoema


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Returns queued responses; records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.responses.pop(0)

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.responses.pop(0)


GOOD = {"data": [{"startDate": "2002-01-01T00:00:00", "values": [1.0]}],
        "continuationToken": None}


def test_fetch_raw_archives_payload(tmp_path):
    s = FakeSession([FakeResponse(200, GOOD)])
    out = knoema.fetch_raw("tkjlhlb", [], ["M"], tmp_path, "trade.json", session=s)
    assert out == tmp_path / "trade.json"
    assert json.loads(out.read_text(encoding="utf-8")) == GOOD
    assert s.calls[0][1].endswith("/api/1.0/data/raw")
    assert s.calls[0][2]["params"]["client_id"] == knoema.CLIENT_ID


def test_fetch_raw_rescrapes_client_id_on_403(tmp_path):
    homepage = FakeResponse(200, text='...client_id=NEWKEY00000000000000NEWKEY...')
    s = FakeSession([FakeResponse(403), homepage, FakeResponse(200, GOOD)])
    knoema.fetch_raw("tkjlhlb", [], ["M"], tmp_path, "t.json", session=s)
    assert s.calls[-1][2]["params"]["client_id"] == "NEWKEY00000000000000NEWKEY"


def test_fetch_raw_refuses_paged_response(tmp_path):
    paged = {"data": [{"values": [1]}], "continuationToken": "abc"}
    s = FakeSession([FakeResponse(200, paged)])
    with pytest.raises(knoema.KnoemaError, match="paged"):
        knoema.fetch_raw("d", [], ["M"], tmp_path, "d.json", session=s)


def test_fetch_raw_refuses_empty_data(tmp_path):
    s = FakeSession([FakeResponse(200, {"data": [], "continuationToken": None})])
    with pytest.raises(knoema.KnoemaError, match="empty"):
        knoema.fetch_raw("d", [], ["M"], tmp_path, "d.json", session=s)


def test_resolve_client_id_scrape_failure_is_loud():
    s = FakeSession([FakeResponse(200, text="<html>no key here</html>")])
    with pytest.raises(knoema.KnoemaError, match="client_id"):
        knoema.resolve_client_id(session=s)


def test_dimension_members():
    items = {"items": [{"key": 1000000, "name": "Accidents"},
                       {"key": 1000010, "name": "Deaths"}]}
    s = FakeSession([FakeResponse(200, items)])
    assert knoema.dimension_members("gehye", "indicators", session=s) == {
        "Accidents": 1000000, "Deaths": 1000010,
    }


def test_dim_name_handles_both_member_shapes():
    assert knoema.dim_name({"indicators": "Deaths"}, "indicators") == "Deaths"
    assert knoema.dim_name({"indicators": {"name": "Deaths"}}, "indicators") == "Deaths"
    with pytest.raises(knoema.KnoemaError):
        knoema.dim_name({"other": 1}, "indicators")


@pytest.mark.parametrize("fn,start,n,expect", [
    (knoema.monthly_periods, "2002-01-01T00:00:00", 3, ["2002-01", "2002-02", "2002-03"]),
    (knoema.monthly_periods, "2025-11-01T00:00:00", 3, ["2025-11", "2025-12", "2026-01"]),
    (knoema.annual_periods, "2002-01-01T00:00:00", 3, [2002, 2003, 2004]),
    (knoema.quarterly_periods, "2018-01-01T00:00:00", 5,
     ["2018Q1", "2018Q2", "2018Q3", "2018Q4", "2019Q1"]),
    (knoema.quarterly_periods, "2025-10-01T00:00:00", 3,
     ["2025Q4", "2026Q1", "2026Q2"]),
])
def test_periods(fn, start, n, expect):
    assert fn(start, n) == expect


def test_quarterly_periods_span_the_whole_ppi_series():
    """33 quarters from 2018Q1 is exactly 2018Q1..2026Q1 — the PPI series as
    published on 2026-08-05. Off-by-one here would relabel every observation."""
    labels = knoema.quarterly_periods("2018-01-01T00:00:00", 33)
    assert labels[0] == "2018Q1"
    assert labels[-1] == "2026Q1"
    assert labels[4] == "2019Q1"


# --- shared guards -------------------------------------------------------
#
# The three Knoema pipelines used to carry private copies of these; both of
# the holes the copies shared are pinned here.

TOTALS = frozenset({"total", "oman", "sultanate of oman"})

# shape copied from the real fixtures: dimension id -> member-field descriptors
SYNTH_PAYLOAD = {
    "continuationToken": None,
    "dimensionFields": {
        "regions": [{"key": 1, "name": "id", "displayName": "Id"}],
        "indicators": [{"key": 2, "name": "id", "displayName": "Id"}],
        "gender": [{"key": 3, "name": "id", "displayName": "Id"}],
    },
    "data": [{
        "regions": {"key": 1000000, "name": "Oman"},
        "indicators": {"key": 1000030, "name": "Accidents"},
        "gender": {"key": 1000000, "name": "Total"},
        "startDate": "2002-01-01T00:00:00",
        "endDate": "2004-01-01T00:00:00",
        "frequency": "A",
        "values": [1.0, 2.0, 3.0],
    }],
}


def synth_row(**overrides) -> dict:
    row = json.loads(json.dumps(SYNTH_PAYLOAD["data"][0]))
    row.update(overrides)
    return row


def test_norm_name_lowercases_and_collapses_whitespace():
    assert knoema.norm_name("  Sultanate   OF\tOman\n") == "sultanate of oman"


def test_dimension_ids_reads_the_payloads_own_declaration():
    assert knoema.dimension_ids(SYNTH_PAYLOAD) == {"regions", "indicators", "gender"}


@pytest.mark.parametrize("payload", [{}, {"dimensionFields": {}},
                                     {"dimensionFields": None},
                                     {"dimensionFields": []}])
def test_dimension_ids_without_declaration_is_loud(payload):
    with pytest.raises(knoema.KnoemaError, match="dimensionFields"):
        knoema.dimension_ids(payload)


def test_check_totals_passes_when_every_dimension_is_on_its_total():
    knoema.check_totals(synth_row(), "indicators",
                        {"regions", "indicators", "gender"}, TOTALS)


def test_check_totals_ignores_the_indicator_dimension():
    """The indicator is the series' identity, never a total."""
    knoema.check_totals(synth_row(), "indicators", {"indicators"}, TOTALS)


def test_check_totals_rejects_a_dict_breakdown_member():
    row = synth_row(gender={"key": 1000010, "name": "Male"})
    with pytest.raises(knoema.KnoemaError, match="gender"):
        knoema.check_totals(row, "indicators",
                            {"regions", "indicators", "gender"}, TOTALS)


def test_check_totals_rejects_a_string_breakdown_member():
    """Hole 1: bare-string members used to skip the guard entirely.

    ``dim_name`` accepts a plain string member, so a payload encoding
    ``"regions": "Muscat"`` would have published a governorate as the national
    total under the old ``isinstance(member, dict)`` skip.
    """
    row = synth_row(regions="Muscat")
    with pytest.raises(knoema.KnoemaError, match="Muscat|muscat"):
        knoema.check_totals(row, "indicators",
                            {"regions", "indicators", "gender"}, TOTALS)


def test_check_totals_accepts_a_string_total_member():
    """String members are checked, not rejected — a string total still passes."""
    knoema.check_totals(synth_row(regions="Oman"), "indicators",
                        {"regions", "indicators", "gender"}, TOTALS)


def test_check_totals_rejects_a_missing_dimension():
    """Hole 2: a dimension absent from the row used to go unnoticed.

    The old guard asked "is any present member a non-total", so a row that
    simply dropped ``gender`` passed — even though the series is then an
    unknown slice.
    """
    row = synth_row()
    del row["gender"]
    with pytest.raises(knoema.KnoemaError, match="gender"):
        knoema.check_totals(row, "indicators",
                            {"regions", "indicators", "gender"}, TOTALS)


def test_periods_for_annual_happy_path():
    assert knoema.periods_for(synth_row(), "A", label="accidents") == [2002, 2003, 2004]


def test_periods_for_monthly_happy_path():
    row = synth_row(frequency="M", endDate="2002-03-01T00:00:00")
    assert knoema.periods_for(row, "M") == ["2002-01", "2002-02", "2002-03"]


def test_periods_for_quarterly_happy_path():
    row = synth_row(frequency="Q", endDate="2002-07-01T00:00:00")
    assert knoema.periods_for(row, "Q") == ["2002Q1", "2002Q2", "2002Q3"]


def test_periods_for_quarterly_accepts_an_enddate_on_the_quarters_last_day():
    """The portal declares the *first* day of the last quarter today. Comparing
    through pd.Period rather than by slicing means a switch to the last day —
    the same quarter either way — is not read as a misalignment."""
    row = synth_row(frequency="Q", endDate="2002-09-30T00:00:00")
    assert knoema.periods_for(row, "Q") == ["2002Q1", "2002Q2", "2002Q3"]


def test_periods_for_quarterly_rejects_an_enddate_in_a_different_quarter():
    """The exact trap a naive endDate check falls into.

    The declared end "2002-10-01" is 2002Q4; the walk reaches 2002Q3. Both are
    the same *year*, so a year-slice comparison would pass this, and neither
    "2002-07" nor "2002-10" ever equals a "2002Qn" label, so a month-slice
    comparison would reject the happy path above instead. Only converting the
    declared date to a quarter label distinguishes the two cases.
    """
    row = synth_row(frequency="Q", endDate="2002-10-01T00:00:00")
    with pytest.raises(knoema.KnoemaError, match="truncated or misaligned"):
        knoema.periods_for(row, "Q", label="ppi general")


def test_periods_for_quarterly_rejects_a_truncated_series():
    """Losing leading observations shifts every quarter label; the row's own
    endDate no longer matches the walk and must catch it."""
    row = synth_row(frequency="Q", endDate="2002-07-01T00:00:00",
                    values=[2.0, 3.0])
    with pytest.raises(knoema.KnoemaError, match="truncated or misaligned"):
        knoema.periods_for(row, "Q")


def test_periods_for_rejects_a_frequency_mismatch():
    with pytest.raises(knoema.KnoemaError, match="frequency"):
        knoema.periods_for(synth_row(frequency="M"), "A", label="accidents")


def test_periods_for_rejects_an_enddate_mismatch():
    """Losing leading observations shifts every label; endDate catches it."""
    with pytest.raises(knoema.KnoemaError, match="endDate"):
        knoema.periods_for(synth_row(values=[2.0, 3.0]), "A")


def test_periods_for_names_the_series_in_errors():
    with pytest.raises(knoema.KnoemaError, match="injuries"):
        knoema.periods_for(synth_row(values=[]), "A", label="injuries")


@pytest.mark.parametrize("key", ["values", "startDate", "endDate"])
def test_periods_for_missing_key_is_a_knoema_error(key):
    row = synth_row()
    del row[key]
    with pytest.raises(knoema.KnoemaError, match=key):
        knoema.periods_for(row, "A", label="accidents")


def test_periods_for_rejects_an_unsupported_frequency():
    """M, Q and A are labelled; the portal's other frequencies are not.

    This used to be asserted with "Q", which the PPI dataset made a supported
    frequency — "W" keeps the guard bound rather than letting it quietly become
    a test of nothing.
    """
    with pytest.raises(knoema.KnoemaError, match="only 'M', 'Q' and 'A'"):
        knoema.periods_for(synth_row(frequency="W"), "W")


def test_knoema_error_is_a_value_error():
    """Pipelines surface guard failures as ValueError; keep that contract."""
    assert issubclass(knoema.KnoemaError, ValueError)
