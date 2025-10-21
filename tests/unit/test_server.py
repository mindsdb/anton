import sys
import types

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def server_app(monkeypatch: pytest.MonkeyPatch):
    fake_langfuse = types.ModuleType("langfuse")

    def _observe_stub(f=None, **_):
        if f is None:
            return lambda x: x
        return f

    fake_langfuse.observe = _observe_stub

    # Mock Langfuse client for v3
    class MockTrace:
        def __init__(self):
            self.id = "mock-trace-id"

    class MockLangfuseClient:
        def trace(self, **kwargs):
            return MockTrace()

        def update_current_trace(self, **kwargs):
            pass

        def get_current_trace_id(self):
            return "mock-trace-id"

    def _get_client_stub():
        return MockLangfuseClient()

    fake_langfuse.get_client = _get_client_stub
    fake_langfuse.Langfuse = MockLangfuseClient

    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse)

    # Avoid real DB session creation
    from types import SimpleNamespace as _SSN

    monkeypatch.setattr(
        "minds.db.pg_session.get_session",
        lambda: _SSN(commit=lambda: None, rollback=lambda: None, close=lambda: None),
        raising=False,
    )

    # Mock context extraction
    from minds.requests.context import Context

    def _fake_extract_context(request):
        return Context(user_id="test-user-123", user_email="test@example.com")

    monkeypatch.setattr(
        "minds.requests.context.extract_context_from_request",
        _fake_extract_context,
        raising=False,
    )

    # Mock MindsDB client creation
    class MockMindsDBClient:
        pass

    def _fake_create_mindsdb_client(request, **kwargs):
        return MockMindsDBClient()

    monkeypatch.setattr(
        "minds.client.mindsdb.create_mindsdb_client_from_request",
        _fake_create_mindsdb_client,
        raising=False,
    )

    # 5) Mock the chat completions endpoint dependencies BEFORE importing
    async def _fake_chat_completions_handler(request_id, session, context, mindsdb_client, chat_completions_request):
        from starlette.responses import JSONResponse

        from minds.schemas.chat import ChatCompletion, Choice, Message, Usage

        response = ChatCompletion(
            id="test-completion",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[Choice(index=0, message=Message(role="assistant", content="Test response"), finish_reason="stop")],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        return JSONResponse(content=response.model_dump())

    monkeypatch.setattr("minds.api.v1.endpoints.chat.chat_completions_request_handler", _fake_chat_completions_handler)

    # Import server after stubs so it binds the fakes
    import minds.server as server

    return server.app


@pytest.fixture()
def headers():
    return {"Authorization": "Bearer 1234567890", "x-user-id": "1", "x-company-id": "2"}


def test_healthz(server_app):
    client = TestClient(server_app)
    r = client.get("/api/v1/health/")
    assert r.status_code == 200 and r.json() == {"status": "ok", "version": "v1"}


def test_chat_completions(server_app, headers):
    client = TestClient(server_app)
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "q"}],
        "metadata": {"doc_id": "1"},
    }
    r = client.post("/api/v1/chat/completions", json=payload, headers=headers)
    expected_response = {
        "id": "test-completion",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Test response"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }
    assert r.status_code == 200 and r.json() == expected_response


def test_chat_completions_v1(server_app, headers):
    client = TestClient(server_app)
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "q"}],
        "metadata": {"doc_id": "1"},
    }
    r = client.post("/api/v1/chat/completions", json=payload, headers=headers)
    expected_response = {
        "id": "test-completion",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Test response"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "system_fingerprint": None,
    }
    assert r.status_code == 200 and r.json() == expected_response
