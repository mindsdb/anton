import os
import sys
import types
from uuid import UUID

import pytest


@pytest.fixture(scope="session", autouse=True)
def disable_opentelemetry_and_langfuse_env() -> None:
    """
    Make OpenTelemetry and Langfuse no-op during tests.

    - Disables the OpenTelemetry SDK entirely so no exporters run.
    - Sets exporters to "none" for extra safety.
    - Hints Langfuse to be disabled if it honors the env var.
    """
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
    os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
    os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")
    os.environ.setdefault("LANGFUSE_DISABLED", "true")


@pytest.fixture(scope="session", autouse=True)
def stub_langfuse_module() -> None:
    """
    Provide a minimal stub for the `langfuse` package so importing it in code under test
    does not initialize real telemetry/exporters.

    This exposes:
    - langfuse.observe: a no-op decorator
    - langfuse.decorators.langfuse_context: object with the methods used in code

    Individual tests can still monkeypatch these as needed.
    """

    # If tests or environment already provided a stub, do nothing
    if "langfuse" in sys.modules:
        return

    fake_langfuse = types.ModuleType("langfuse")

    def _observe_stub(f=None, **_):
        if f is None:
            return lambda x: x
        return f

    fake_langfuse.observe = _observe_stub

    # Mock client API for v3
    class _MockTrace:
        def __init__(self):
            self.id = "mock-trace-id"

    class _MockObservation:
        def end(self, **kwargs):  # noqa: ARG002
            return self

        def update(self, **kwargs):  # noqa: ARG002
            return self

    class _MockLangfuseClient:
        def trace(self, **kwargs):  # noqa: ARG002
            return _MockTrace()

        def update_current_trace(self, **kwargs):  # noqa: ARG002
            return None

        def update_current_generation(self, **kwargs):  # noqa: ARG002
            return None

        def get_current_trace_id(self):
            return "mock-trace-id"

        def get_current_observation_id(self):
            return "mock-observation-id"

        def start_observation(self, **kwargs):  # noqa: ARG002
            return _MockObservation()

    def _get_client_stub():
        return _MockLangfuseClient()

    fake_langfuse.get_client = _get_client_stub
    fake_langfuse.Langfuse = _MockLangfuseClient

    sys.modules["langfuse"] = fake_langfuse


@pytest.fixture
def test_uuid():
    """Shared UUID for testing datasource responses."""
    return UUID("c58da050-a5b2-4707-ae80-ceeb0e701271")
