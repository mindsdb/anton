import pytest
from anton_state import StateSchema, Attr, ConditionalCheckFailed, StateThrottled, StateUnavailable
from anton_state.http_driver import HTTPDriver, _WireError

_S = StateSchema(pk=Attr(name="pk"), sk=Attr(name="sk"), ttl_attribute="exp")


def _drv(monkeypatch, responder):
    d = HTTPDriver(url="https://broker/_state", token="t.sig", schema=_S)
    monkeypatch.setattr(d, "_post", responder)
    return d


def test_get_sends_op_and_returns_item(monkeypatch):
    seen = {}

    def responder(op, payload):
        seen["op"], seen["payload"] = op, payload
        return {"item": {"pk": "p", "sk": "s", "n": 1}}

    d = _drv(monkeypatch, responder)
    item = d.get("p", "s", consistent=True)
    assert item == {"pk": "p", "sk": "s", "n": 1}
    assert seen["op"] == "get"
    assert seen["payload"]["pk"] == "p" and seen["payload"]["sk"] == "s"
    assert "namespace" not in seen["payload"] and "artifact_id" not in seen["payload"]


def test_put_extracts_ttl_and_cond(monkeypatch):
    seen = {}

    def responder(op, payload):
        seen.update(payload)
        return {"ok": True}

    d = _drv(monkeypatch, responder)
    d.put({"pk": "p", "sk": "s", "exp": 1234, "n": 1}, if_not_exists=True, if_version=None)
    assert seen["ttl"] == 1234
    assert seen["cond"] == {"if_not_exists": True}


def test_increment_returns_value(monkeypatch):
    d = _drv(monkeypatch, lambda op, p: {"value": 42})
    assert d.increment("c", "g", field="n", by=2) == 42


def test_conditional_error_mapped(monkeypatch):
    def responder(op, payload):
        raise _WireError(409, "conditional", "nope")

    d = _drv(monkeypatch, responder)
    with pytest.raises(ConditionalCheckFailed):
        d.put({"pk": "p", "sk": "s"}, if_not_exists=True, if_version=None)


def test_mutation_not_retried_on_unavailable(monkeypatch):
    calls = {"n": 0}

    def responder(op, payload):
        calls["n"] += 1
        raise _WireError(503, "unavailable", "down")

    d = _drv(monkeypatch, responder)
    with pytest.raises(StateUnavailable):
        d.put({"pk": "p", "sk": "s"}, if_not_exists=False, if_version=None)
    assert calls["n"] == 1  # no retry on mutation


def test_read_retried_on_unavailable(monkeypatch):
    calls = {"n": 0}

    def responder(op, payload):
        calls["n"] += 1
        if calls["n"] < 2:
            raise _WireError(503, "unavailable", "down")
        return {"item": None}

    d = _drv(monkeypatch, responder)
    assert d.get("p", "s", consistent=True) is None
    assert calls["n"] == 2  # retried once


def test_unauthorized_not_treated_as_unavailable(monkeypatch):
    def responder(op, payload):
        raise _WireError(401, "unauthorized", "bad token")

    d = _drv(monkeypatch, responder)
    with pytest.raises(Exception) as ei:
        d.get("p", "s", consistent=True)
    assert not isinstance(ei.value, StateUnavailable)  # definitely-not-applied, clearer error
