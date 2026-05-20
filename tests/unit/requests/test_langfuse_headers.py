"""Unit tests for the Langfuse-proxy convention header parser.

End-to-end proof that the parsed values land on real Langfuse traces
(``session.id``, ``langfuse.trace.tags``, ``langfuse.trace.name``,
trace-level metadata) lives in
``tests/unit/observability/test_passthrough_traces.py`` — those run the
real Langfuse SDK and assert on actual OpenTelemetry spans.

What's left here are the pure-function pieces — header string → Python
value conversion — that have no Langfuse dependency. They cover the
edge cases (malformed JSON, whitespace, non-dict JSON) that are hard to
exercise from the end-to-end harness without contriving the input.
"""

from __future__ import annotations

from unittest.mock import Mock
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


def _request_with_headers(extra: dict[str, str]) -> Request:
    """Build a ``Mock`` request with the minimum identity headers + ``extra``.

    The identity headers (user_id / org_id / email / roles) are required
    by ``extract_context_from_request`` and aren't under test here, so
    every case starts from a valid baseline.
    """
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
# Header → Context parsing
# ---------------------------------------------------------------------------


class TestExtractLangfuseHeaders:
    def test_all_three_headers_populated(self):
        ctx = extract_context_from_request(
            _request_with_headers(
                {
                    HEADER_LANGFUSE_SESSION_ID: "20260518_abc123",
                    HEADER_LANGFUSE_TAGS: "cowork, agent",
                    HEADER_LANGFUSE_METADATA: '{"turn_id": 3, "harness": "cowork"}',
                }
            )
        )
        assert ctx.langfuse_session_id == "20260518_abc123"
        assert ctx.langfuse_tags == ["cowork", "agent"]
        assert ctx.langfuse_metadata == {"turn_id": 3, "harness": "cowork"}

    def test_absent_headers_default_to_empty(self):
        ctx = extract_context_from_request(_request_with_headers({}))
        assert ctx.langfuse_session_id is None
        assert ctx.langfuse_tags == []
        assert ctx.langfuse_metadata == {}

    def test_malformed_metadata_json_is_ignored(self):
        # A client emitting bad JSON should not break the request.
        ctx = extract_context_from_request(_request_with_headers({HEADER_LANGFUSE_METADATA: "{not json"}))
        assert ctx.langfuse_metadata == {}

    def test_metadata_must_be_a_json_object(self):
        # A bare array is valid JSON but not the Langfuse-Metadata contract;
        # downstream code calls ``.get()`` so we reject it cleanly.
        ctx = extract_context_from_request(_request_with_headers({HEADER_LANGFUSE_METADATA: '["a","b"]'}))
        assert ctx.langfuse_metadata == {}

    def test_tags_strip_whitespace_and_drop_empties(self):
        ctx = extract_context_from_request(_request_with_headers({HEADER_LANGFUSE_TAGS: "  cowork ,, agent,  "}))
        assert ctx.langfuse_tags == ["cowork", "agent"]

    def test_session_id_whitespace_only_is_none(self):
        ctx = extract_context_from_request(_request_with_headers({HEADER_LANGFUSE_SESSION_ID: "   "}))
        assert ctx.langfuse_session_id is None


# ---------------------------------------------------------------------------
# create_langfuse_context: the only branches worth pinpointing at unit
# level are the tag-dedupe and the trace_name derivation, since those are
# easier to debug here than from the end-to-end harness. Happy-path
# session_id/tags/metadata flow is proven against real spans in
# tests/unit/observability/test_passthrough_traces.py.
# ---------------------------------------------------------------------------


def _ctx(**overrides) -> Context:
    return Context(
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        organization_id=UUID("00000000-0000-0000-0000-000000000002"),
        request_id=UUID("00000000-0000-0000-0000-000000000099"),
        user_email="user@example.com",
        **overrides,
    )


class TestCreateLangfuseContext:
    def test_client_tags_appended_and_deduped(self):
        """A client retransmitting an identity tag should not double it on the trace."""
        lc = create_langfuse_context(_ctx(langfuse_tags=["cowork", "agent", "cowork"]))
        assert lc.tags.count("cowork") == 1
        assert "agent" in lc.tags
        # Identity tags still present (set-membership only — order not under test).
        assert any(t.startswith("user_id:") for t in lc.tags)

    def test_trace_name_requires_both_harness_and_turn_id(self):
        # Only harness present: no derived name (turn_id is the second half).
        assert create_langfuse_context(_ctx(langfuse_metadata={"harness": "cowork"})).trace_name is None
        # Only turn_id present: no derived name (harness is the first half).
        assert create_langfuse_context(_ctx(langfuse_metadata={"turn_id": 3})).trace_name is None
        # Both: derived.
        lc = create_langfuse_context(_ctx(langfuse_metadata={"harness": "cowork", "turn_id": 4}))
        assert lc.trace_name == "cowork:turn-4"
