import sys
import types

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def server_app(monkeypatch: pytest.MonkeyPatch):
    fake_langfuse = types.ModuleType("langfuse")
    fake_decorators = types.ModuleType("langfuse.decorators")

    def _observe_stub(f=None, **_):
        if f is None:
            return lambda x: x
        return f

    from types import SimpleNamespace as _SN

    fake_decorators.observe = _observe_stub
    fake_decorators.langfuse_context = _SN(
        update_current_observation=lambda **kwargs: None,
        get_current_trace_id=lambda: None,
    )

    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse)
    monkeypatch.setitem(sys.modules, "langfuse.decorators", fake_decorators)

    # Avoid real DB session creation
    from types import SimpleNamespace as _SSN

    monkeypatch.setattr(
        "minds.db.pg_session.get_session",
        lambda: _SSN(commit=lambda: None, rollback=lambda: None, close=lambda: None),
        raising=False,
    )

    # Import server after stubs so it binds the fakes
    import minds.server as server

    # 5) Stub the heavy handlers used by endpoints
    async def _fake_cc_handler(**_):
        from starlette.responses import JSONResponse

        return JSONResponse({"ok": True})

    monkeypatch.setattr(server, "chat_completions_request_handler", _fake_cc_handler)

    return server.app


@pytest.fixture()
def headers():
    return {"Authorization": "Bearer 1234567890"}


def test_healthz(server_app):
    client = TestClient(server_app)
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_chat_completions(server_app, headers):
    client = TestClient(server_app)
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "q"}],
        "metadata": {"doc_id": "1"},
    }
    r = client.post("/chat/completions", json=payload, headers=headers)
    assert r.status_code == 200 and r.json() == {"ok": True}


def test_chat_completions_v1(server_app, headers):
    client = TestClient(server_app)
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "q"}],
        "metadata": {"doc_id": "1"},
    }
    r = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert r.status_code == 200 and r.json() == {"ok": True}
