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
])
def test_periods(fn, start, n, expect):
    assert fn(start, n) == expect
