"""Unit tests for OpenAIProvider._build_trace_headers.

Locks in the load-bearing behavior of the langfuse trace headers that the
MindsHub router reads: caller-supplied tags are appended after the harness
identity and sanitized, and built-in identity metadata (turn_id / harness)
always wins over caller-supplied metadata on key collision.
"""

import json

from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.tracing import (
    TraceContext,
    reset_trace_context,
    set_trace_context,
)


def _provider() -> OpenAIProvider:
    # A MindsHub base URL turns on trace-header emission; no network at init.
    return OpenAIProvider(api_key="test", base_url="https://api.mindshub.ai/v1")


def _headers_for(ctx: TraceContext) -> dict[str, str] | None:
    token = set_trace_context(ctx)
    try:
        return _provider()._build_trace_headers()
    finally:
        reset_trace_context(token)


def test_caller_tags_appended_after_harness():
    headers = _headers_for(
        TraceContext(session_id="s1", turn_id=3, harness="anton",
                     tags=("eval", "eval_run:r1"))
    )
    assert headers["Langfuse-Session-Id"] == "s1"
    assert headers["Langfuse-Tags"] == "anton,eval,eval_run:r1"


def test_builtin_metadata_wins_over_caller():
    headers = _headers_for(
        TraceContext(turn_id=7, harness="anton",
                     metadata={"harness": "spoof", "turn_id": "spoof", "eval_run_id": "r1"})
    )
    meta = json.loads(headers["Langfuse-Metadata"])
    assert meta["harness"] == "anton"   # identity wins
    assert meta["turn_id"] == 7         # identity wins
    assert meta["eval_run_id"] == "r1"  # caller-only key preserved


def test_tags_are_sanitized():
    headers = _headers_for(
        TraceContext(harness="anton",
                     tags=("good", "ba,d", "wi\nth-nl", "   ", "  spaced  "))
    )
    # comma + newline stripped (not split into new tags), blank dropped,
    # surrounding whitespace trimmed.
    assert headers["Langfuse-Tags"].split(",") == [
        "anton", "good", "bad", "with-nl", "spaced",
    ]


def test_no_trace_context_returns_none():
    assert _provider()._build_trace_headers() is None
