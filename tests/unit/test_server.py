import sys
import types
from unittest.mock import MagicMock

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
        # Context.user_id is a UUID in the application schema
        return Context(user_id="00000000-0000-0000-0000-000000000001", user_email="test@example.com")

    monkeypatch.setattr(
        "minds.requests.context.extract_context_from_request",
        _fake_extract_context,
        raising=False,
    )

    # Mock MindsDB client creation
    class MockMindsDBClient:
        pass

    def _fake_create_mindsdb_client(request, context=None, **kwargs):
        return MockMindsDBClient()

    monkeypatch.setattr(
        "minds.client.mindsdb.create_mindsdb_client_from_request",
        _fake_create_mindsdb_client,
        raising=False,
    )

    # 5) Mock the chat completions endpoint dependencies BEFORE importing
    async def _fake_chat_completions_handler(
        session, context, mindsdb_client, chat_completions_request, instrument=True, limits_service=None
    ):
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

    # Mock Statsig init/shutdown so no real SDK is created
    mock_statsig_instance = MagicMock()

    # Import server so we can patch on it directly (handles cached modules)
    import minds.server as server

    monkeypatch.setattr(server, "init_statsig", lambda settings=None: mock_statsig_instance)
    monkeypatch.setattr(server, "shutdown_statsig", lambda: None)

    # Patch is_langfuse_enabled where it's actually used (bound reference in chat module)
    import minds.api.v1.endpoints.chat as chat_mod

    monkeypatch.setattr(chat_mod, "is_langfuse_enabled", lambda context, settings=None: True)

    # Mock usage guard so it doesn't try to hit the real database
    async def _async_noop(*args, **kwargs):
        pass

    monkeypatch.setattr(chat_mod, "require_usage_available", _async_noop)

    return server.app


@pytest.fixture()
def headers():
    return {
        "Authorization": "Bearer 1234567890",
        "X-User-Id": "00000000-0000-0000-0000-000000000001",
        "X-Organization-Id": "00000000-0000-0000-0000-000000000002",
    }


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


class TestStatsigLifecycle:
    """Test that Statsig is initialized on startup and shut down on shutdown."""

    def test_startup_initializes_statsig(self, monkeypatch):
        """Verify that Statsig is initialized during app startup."""
        import minds.server as server

        init_called = {"value": False}
        mock_statsig = MagicMock()

        def _mock_init(settings=None):
            init_called["value"] = True
            return mock_statsig

        # Patch directly on the server module (where the bound reference lives)
        monkeypatch.setattr(server, "init_statsig", _mock_init)
        monkeypatch.setattr(server, "shutdown_statsig", lambda: None)

        app = server.create_app()

        with TestClient(app):
            assert init_called["value"] is True

    def test_shutdown_calls_shutdown_statsig(self, monkeypatch):
        """Verify that Statsig is shut down during app shutdown."""
        import minds.server as server

        shutdown_called = {"value": False}
        mock_statsig = MagicMock()

        def _mock_shutdown():
            shutdown_called["value"] = True

        # Patch directly on the server module (where the bound reference lives)
        monkeypatch.setattr(server, "init_statsig", lambda settings=None: mock_statsig)
        monkeypatch.setattr(server, "shutdown_statsig", _mock_shutdown)

        app = server.create_app()

        with TestClient(app):
            pass  # startup + shutdown happen

        assert shutdown_called["value"] is True
