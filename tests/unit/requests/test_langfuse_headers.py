"""Tests for the Langfuse-proxy convention header wiring.

Exercises three pieces:

1. ``extract_context_from_request`` reads the three ``Langfuse-*`` headers
   into the request ``Context`` (with malformed JSON ignored, not raised).
2. ``create_langfuse_context`` merges client tags into the identity tag set,
   threads ``session_id`` through, and derives a ``harness:turn-N`` trace
   name when both pieces are present in the client metadata.
3. ``setup_langfuse_observation`` passes ``session_id`` and ``name`` to the
   Langfuse client's ``update_current_trace`` and falls back gracefully if
   the SDK doesn't recognize the extended kwargs.
"""

from __future__ import annotations

from unittest.mock import Mock, patch
from uuid import UUID

from fastapi import Request

from minds.common.constants import (
    HEADER_LANGFUSE_METADATA,
    HEADER_LANGFUSE_SESSION_ID,
    HEADER_LANGFUSE_TAGS,
    HEADER_ORGANIZATION_ID,
    HEADER_USER_EMAIL,
    HEADER_USER_ID,
    HEADER_USER_ROLES,
)
from minds.requests.context import (
    Context,
    create_langfuse_context,
    extract_context_from_request,
)
from minds.requests.langfuse_tracing import setup_langfuse_observation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _request_with_headers(extra: dict[str, str]) -> Request:
    """Build a ``Mock`` request with the minimum identity headers + ``extra``."""
    mock_request = Mock(spec=Request)
    mock_request.headers = {
        HEADER_USER_ID: "00000000-0000-0000-0000-000000000001",
        HEADER_ORGANIZATION_ID: "00000000-0000-0000-0000-000000000002",
        HEADER_USER_EMAIL: "user@example.com",
        HEADER_USER_ROLES: "",
        **extra,
    }
    return mock_request


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


class TestExtractLangfuseHeaders:
    def test_all_three_headers_populated(self):
        request = _request_with_headers(
            {
                HEADER_LANGFUSE_SESSION_ID: "20260515_120000_abc123",
                HEADER_LANGFUSE_TAGS: "cowork, agent",
                HEADER_LANGFUSE_METADATA: '{"turn_id": 3, "harness": "cowork"}',
            }
        )

        ctx = extract_context_from_request(request)

        assert ctx.langfuse_session_id == "20260515_120000_abc123"
        assert ctx.langfuse_tags == ["cowork", "agent"]
        assert ctx.langfuse_metadata == {"turn_id": 3, "harness": "cowork"}

    def test_absent_headers_default_to_empty(self):
        request = _request_with_headers({})

        ctx = extract_context_from_request(request)

        assert ctx.langfuse_session_id is None
        assert ctx.langfuse_tags == []
        assert ctx.langfuse_metadata == {}

    def test_malformed_metadata_json_is_warned_and_ignored(self):
        request = _request_with_headers({HEADER_LANGFUSE_METADATA: "{not json"})

        ctx = extract_context_from_request(request)

        # Malformed JSON should not break extraction — just be dropped.
        assert ctx.langfuse_metadata == {}

    def test_metadata_must_be_a_json_object(self):
        # A bare array or string is valid JSON but not the contract Langfuse-
        # Metadata advertises; reject it cleanly rather than letting downstream
        # code crash on .get().
        request = _request_with_headers({HEADER_LANGFUSE_METADATA: '["a","b"]'})

        ctx = extract_context_from_request(request)

        assert ctx.langfuse_metadata == {}

    def test_tags_strip_whitespace_and_drop_empties(self):
        request = _request_with_headers({HEADER_LANGFUSE_TAGS: "  cowork ,, agent,  "})

        ctx = extract_context_from_request(request)

        assert ctx.langfuse_tags == ["cowork", "agent"]

    def test_session_id_whitespace_only_is_none(self):
        request = _request_with_headers({HEADER_LANGFUSE_SESSION_ID: "   "})

        ctx = extract_context_from_request(request)

        assert ctx.langfuse_session_id is None


# ---------------------------------------------------------------------------
# LangfuseContext construction
# ---------------------------------------------------------------------------


class TestCreateLangfuseContext:
    def _ctx(self, **overrides) -> Context:
        return Context(
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            organization_id=UUID("00000000-0000-0000-0000-000000000002"),
            request_id=UUID("00000000-0000-0000-0000-000000000099"),
            user_email="user@example.com",
            **overrides,
        )

    def test_session_id_propagates(self):
        ctx = self._ctx(langfuse_session_id="sess-1")

        lc = create_langfuse_context(ctx)

        assert lc.session_id == "sess-1"

    def test_trace_name_derived_from_harness_plus_turn_id(self):
        ctx = self._ctx(langfuse_metadata={"harness": "cowork", "turn_id": 4})

        lc = create_langfuse_context(ctx)

        assert lc.trace_name == "cowork:turn-4"

    def test_trace_name_unset_when_only_harness_present(self):
        ctx = self._ctx(langfuse_metadata={"harness": "cowork"})

        lc = create_langfuse_context(ctx)

        assert lc.trace_name is None

    def test_client_tags_appended_after_identity_tags_deduped(self):
        ctx = self._ctx(langfuse_tags=["cowork", "agent", "cowork"])

        lc = create_langfuse_context(ctx)

        # Identity tags come first; client tags follow; duplicates collapsed.
        assert "cowork" in lc.tags
        assert "agent" in lc.tags
        assert lc.tags.count("cowork") == 1
        # Identity tags still present.
        assert any(t.startswith("user_id:") for t in lc.tags)

    def test_extra_metadata_preserved_for_trace_merge(self):
        ctx = self._ctx(langfuse_metadata={"turn_id": 0, "harness": "cowork", "exp": "A"})

        lc = create_langfuse_context(ctx)

        assert lc.extra_metadata == {"turn_id": 0, "harness": "cowork", "exp": "A"}


# ---------------------------------------------------------------------------
# Trace setup
# ---------------------------------------------------------------------------


class TestSetupLangfuseObservation:
    @patch("minds.requests.langfuse_tracing.get_client")
    def test_passes_session_id_and_trace_name_when_present(self, mock_get_client):
        captured: dict = {}

        class _Client:
            def update_current_trace(self, **kwargs):
                captured.update(kwargs)

            def get_current_trace_id(self):
                return "trace-1"

        mock_get_client.return_value = _Client()

        ctx = Context(
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            organization_id=UUID("00000000-0000-0000-0000-000000000002"),
            langfuse_session_id="sess-1",
            langfuse_metadata={"harness": "cowork", "turn_id": 2},
            langfuse_tags=["cowork"],
        )

        setup_langfuse_observation(ctx)

        assert captured["session_id"] == "sess-1"
        assert captured["name"] == "cowork:turn-2"
        # Client tag merged into identity tags without duplication.
        assert "cowork" in captured["tags"]

    @patch("minds.requests.langfuse_tracing.get_client")
    def test_falls_back_to_legacy_kwargs_if_sdk_rejects_extended(self, mock_get_client):
        attempts: list[dict] = []

        class _LegacyClient:
            def update_current_trace(self, **kwargs):
                attempts.append(kwargs)
                # Older SDKs don't accept ``session_id`` / ``name``.
                if "session_id" in kwargs or "name" in kwargs:
                    raise TypeError("unexpected kwarg")

            def get_current_trace_id(self):
                return "trace-2"

        mock_get_client.return_value = _LegacyClient()

        ctx = Context(
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            organization_id=UUID("00000000-0000-0000-0000-000000000002"),
            langfuse_session_id="sess-x",
        )

        # Should not raise — second attempt drops the extended kwargs.
        setup_langfuse_observation(ctx)

        assert len(attempts) == 2
        assert "session_id" in attempts[0]
        assert "session_id" not in attempts[1]


# ---------------------------------------------------------------------------
# update_generation_usage now accepts input/output — make sure it flows.
# ---------------------------------------------------------------------------


class TestUpdateGenerationUsageInputOutput:
    @patch("minds.requests.langfuse_tracing.get_client")
    def test_in_scope_passes_input_and_output(self, mock_get_client):
        captured: dict = {}

        class _Client:
            def update_current_generation(self, **kwargs):
                captured.update(kwargs)

        mock_get_client.return_value = _Client()

        from minds.requests.langfuse_tracing import update_generation_usage

        update_generation_usage(
            usage=(10, 5),
            model="claude-sonnet-4-6",
            input={"messages": [{"role": "user", "content": "hi"}]},
            output={"role": "assistant", "content": "hello"},
            metadata={"alias": "sonnet"},
        )

        assert captured["model"] == "claude-sonnet-4-6"
        assert captured["input"]["messages"][0]["content"] == "hi"
        assert captured["output"]["content"] == "hello"
        assert captured["metadata"] == {"alias": "sonnet"}
