"""Real-Langfuse capture fixture for observability tests.

These tests exercise the actual Langfuse v3 SDK end-to-end and inspect the
real OpenTelemetry spans it produces — no mocks of update_current_generation,
start_observation, or any of our own helpers. The only patch is the OTLP HTTP
exporter (replaced with a no-op so spans are not also POSTed to a non-existent
``localhost:3000``).

Use the ``langfuse_capture`` fixture in any test that wants to verify token
usage actually lands on a Langfuse generation span:

    async def test_usage(langfuse_capture):
        # ... exercise production code ...
        spans = langfuse_capture.get_spans()
        assert any(
            s.attributes.get("langfuse.observation.usage_details")
            for s in spans
        )

Why a session-scoped Langfuse client?
    Langfuse v3 caches a per-public-key ``LangfuseResourceManager`` and refuses
    to trace once it sees multiple instances in the same process ("cross-project
    leakage safeguard"). Re-initializing per test would either drop spans or
    have them silently skipped. Instead we instantiate Langfuse ONCE per session
    with a single in-memory exporter, and per-test we just clear the exporter.
"""

from __future__ import annotations

import contextlib
import json
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@dataclass
class LangfuseCapture:
    """Handle exposed by the ``langfuse_capture`` fixture.

    ``get_spans()`` returns the OTel spans Langfuse has emitted during the
    current test. Each span carries its real attributes — e.g.
    ``langfuse.observation.type``, ``langfuse.observation.usage_details``,
    ``langfuse.observation.model.name`` — so tests can assert directly on
    what reaches the dashboard.
    """

    exporter: InMemorySpanExporter
    client: Any  # langfuse.Langfuse instance

    def get_spans(self) -> list:
        """Flush any queued spans and return everything captured this test."""
        with contextlib.suppress(Exception):
            self.client.flush()
        return list(self.exporter.get_finished_spans())

    def clear(self) -> None:
        self.exporter.clear()

    @staticmethod
    def usage_details(span) -> dict | None:
        """Decode the JSON-encoded ``usage_details`` attribute, if present."""
        raw = span.attributes.get("langfuse.observation.usage_details")
        if raw is None:
            return None
        return json.loads(raw)

    @staticmethod
    def observation_type(span) -> str | None:
        return span.attributes.get("langfuse.observation.type")

    @staticmethod
    def model_name(span) -> str | None:
        return span.attributes.get("langfuse.observation.model.name")

    @staticmethod
    def metadata(span) -> dict | None:
        """Decode the JSON-encoded ``metadata`` attribute, if present.

        Langfuse stores metadata under either a single ``langfuse.observation.metadata``
        key (whole dict, JSON-encoded) or per-field keys
        ``langfuse.observation.metadata.<k>`` depending on SDK version. We
        normalize both shapes back to a Python dict.
        """
        raw = span.attributes.get("langfuse.observation.metadata")
        if raw is not None:
            try:
                return json.loads(raw)
            except (TypeError, ValueError):
                return None
        # Per-field encoding: collect every attribute prefixed with the metadata namespace.
        prefix = "langfuse.observation.metadata."
        collected: dict = {}
        for k, v in span.attributes.items():
            if not k.startswith(prefix):
                continue
            inner_key = k[len(prefix):]
            # Values may be plain strings or JSON-encoded; try JSON first.
            try:
                collected[inner_key] = json.loads(v) if isinstance(v, str) else v
            except (TypeError, ValueError):
                collected[inner_key] = v
        return collected or None


@pytest.fixture(scope="session")
def _real_langfuse_session():
    """Boot ONE real Langfuse client for the whole observability test session.

    Yields ``(client, exporter)``. Uses a session-shared
    ``InMemorySpanExporter`` that the function-scoped fixture clears between
    tests.
    """
    # 1. Restore the real langfuse module if some other conftest (notably
    #    ``tests/unit/agents/candidate_sql_agent/conftest.py``) stuffed a fake
    #    one into ``sys.modules`` at collection time. Pop the fake plus any
    #    transitive submodules so the next ``import langfuse`` reaches the
    #    real package on disk.
    fake_lf = sys.modules.get("langfuse")
    if isinstance(fake_lf, types.ModuleType) and getattr(fake_lf, "__file__", None) is None:
        for mod_name in list(sys.modules):
            if mod_name == "langfuse" or mod_name.startswith("langfuse."):
                sys.modules.pop(mod_name, None)

    # 1b. Re-import the real langfuse and re-bind it on every ``minds.*`` module
    #     that had already done ``from langfuse import get_client/observe`` while
    #     the fake was in place. Without this, those modules' module-level
    #     ``get_client`` symbol still points at the fake — which means
    #     ``capture_langfuse_generation_context`` returns Mocks and
    #     ``update_current_generation`` writes to a Mock client, no real span.
    import importlib

    real_langfuse = importlib.import_module("langfuse")
    real_observe = real_langfuse.observe
    real_get_client = real_langfuse.get_client

    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("minds.") and mod_name != "minds":
            continue
        if not isinstance(mod, types.ModuleType):
            continue
        # Re-bind only the symbols modules typically import from langfuse.
        if hasattr(mod, "get_client"):
            mod.get_client = real_get_client
        if hasattr(mod, "observe"):
            mod.observe = real_observe

    # 2. Re-enable OTel + Langfuse for the duration of this session-scoped
    #    fixture. We can't use monkeypatch here (function-scoped only), so we
    #    set the env vars directly and restore them on teardown.
    import os

    original_env = {
        "OTEL_SDK_DISABLED": os.environ.get("OTEL_SDK_DISABLED"),
        "LANGFUSE_DISABLED": os.environ.get("LANGFUSE_DISABLED"),
        "LANGFUSE_PUBLIC_KEY": os.environ.get("LANGFUSE_PUBLIC_KEY"),
        "LANGFUSE_SECRET_KEY": os.environ.get("LANGFUSE_SECRET_KEY"),
        "LANGFUSE_HOST": os.environ.get("LANGFUSE_HOST"),
    }
    os.environ.pop("OTEL_SDK_DISABLED", None)
    os.environ.pop("LANGFUSE_DISABLED", None)
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-observability-test"
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-observability-test"
    os.environ["LANGFUSE_HOST"] = "http://localhost-test"

    # 3. Replace Langfuse's network OTLP exporter with a no-op so spans don't
    #    leak to the network. Patch BEFORE importing Langfuse so the import
    #    binds to our class.
    import langfuse._client.span_processor as lf_sp
    import opentelemetry.exporter.otlp.proto.http.trace_exporter as otlp_mod

    class _NoOpOTLPExporter:
        def __init__(self, *a, **k):
            pass

        def export(self, *a, **k):
            return 0  # SUCCESS

        def shutdown(self, *a, **k):
            return None

        def force_flush(self, *a, **k):
            return True

    original_otlp = otlp_mod.OTLPSpanExporter
    original_lf_otlp = lf_sp.OTLPSpanExporter
    otlp_mod.OTLPSpanExporter = _NoOpOTLPExporter
    lf_sp.OTLPSpanExporter = _NoOpOTLPExporter

    # 4. Reset Langfuse's per-public-key singleton cache. Earlier tests in the
    #    suite may have already booted Langfuse (transitively via
    #    ``from minds.handlers...``) under ``OTEL_SDK_DISABLED=true``, baking
    #    a NoOp tracer into a cached resource manager. Without clearing
    #    ``_instances`` our subsequent ``Langfuse(...)`` call would short-
    #    circuit to that cached, no-op resource and our tracer_provider
    #    would be ignored.
    from langfuse._client.resource_manager import LangfuseResourceManager

    for inst in list(LangfuseResourceManager._instances.values()):
        with contextlib.suppress(Exception):
            inst.shutdown()
    LangfuseResourceManager._instances.clear()

    # 5. Wire OTel + Langfuse.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    from langfuse import Langfuse

    client = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ["LANGFUSE_HOST"],
        tracer_provider=provider,
        flush_at=1,
        flush_interval=0.01,
    )

    try:
        yield client, exporter
    finally:
        try:
            client.flush()
        finally:
            otlp_mod.OTLPSpanExporter = original_otlp
            lf_sp.OTLPSpanExporter = original_lf_otlp
            for k, v in original_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


@pytest.fixture
def langfuse_capture(_real_langfuse_session) -> LangfuseCapture:
    """Per-test handle into the session-shared Langfuse client + exporter.

    Clears the in-memory exporter before AND after the test so each test sees
    only its own spans.
    """
    client, exporter = _real_langfuse_session
    exporter.clear()
    capture = LangfuseCapture(exporter=exporter, client=client)
    yield capture
    exporter.clear()
