"""Completion-verifier fail-safe (ENG-1079).

Before this fix, an exception from the verifier's ``generate_object_code``
call (provider hiccup, malformed structured output, etc.) was silently
treated as a COMPLETE verdict — the task loop just broke with no message,
making the agent look like it had died on the first hurdle. It should
instead behave like the STUCK path: append an honest SYSTEM note and let the
model generate a real message summarizing progress and asking the user how
to proceed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.conftest import make_mock_llm

from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTaskProgress,
    ToolCall,
    Usage,
)


@pytest.fixture()
def workspace():
    # Keep scratchpad venvs inside the repo workspace (pytest runs sandboxed and
    # can't write to the real home directory).
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


def _scratchpad_response(text: str, action: str, name: str, code: str = "") -> LLMResponse:
    tc_input: dict = {"action": action, "name": name}
    if code:
        tc_input["code"] = code
    return LLMResponse(
        content=text,
        tool_calls=[ToolCall(id="tc_1", name="scratchpad", input=tc_input)],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
    )


class _FakeAsyncIter:
    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


async def test_verifier_exception_yields_real_message_not_silent_stop(workspace):
    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = AsyncMock(side_effect=RuntimeError("provider hiccup"))

    call_count = 0

    def fake_plan_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Tool-call round.
            return _FakeAsyncIter([
                StreamComplete(response=_scratchpad_response("Running.", "exec", "main", "print(1)"))
            ])
        if call_count == 2:
            # Final round of the tool loop — this text is what gets sent to
            # (the now-failing) verifier.
            return _FakeAsyncIter([
                StreamComplete(response=_text_response("Done running the script."))
            ])
        # Third call is the honest-diagnosis re-prompt (plan_stream_with_recovery),
        # only reached via the verifier-exception fail-safe.
        return _FakeAsyncIter([
            StreamComplete(
                response=_text_response(
                    "Here's what I've done so far — ran the script. An internal "
                    "check failed, so let me know how you'd like to proceed."
                )
            )
        ])

    mock_llm.plan_stream = fake_plan_stream

    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    try:
        events = [e async for e in session.turn_stream("run my script")]

        # Never silently stops: a progress event surfaces the check-in, and
        # exactly one more model call happens (the honest diagnosis) — not a
        # forced "Continue working" continuation, and not silence.
        progress_msgs = [e.message for e in events if isinstance(e, StreamTaskProgress)]
        assert "Something went wrong — checking in with you..." in progress_msgs
        assert call_count == 3

        history_texts = [
            m["content"] for m in session.history
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ]
        assert any("task-completion check failed to run" in t for t in history_texts)
        assert not any("Continue working on the original request" in t for t in history_texts)
        # The model must be asked for a solvability self-assessment, not just
        # a status dump — otherwise "let me know how to proceed" is a vague
        # ask instead of an actual recommendation (ENG-1079 follow-up).
        assert any("whether you believe this is still solvable on your own" in t for t in history_texts)

        final_texts = [
            m["content"] for m in session.history
            if m.get("role") == "assistant" and isinstance(m.get("content"), str)
        ]
        assert any("how you'd like to proceed" in t for t in final_texts)
    finally:
        await session.close()


async def test_truncation_exhausting_every_budget_also_gets_the_diagnosis(workspace):
    """The ENG-1081 ladder can run dry: a verdict truncated at 2048 AND at the
    4096 retry (narrating model, huge session) used to fall into the same
    silent fake-COMPLETE this file exists to remove. Exhausted budgets are not
    a verdict either — the honest-diagnosis path must fire for them too.
    """
    from anton.core.llm.provider import StructuredOutputError
    from anton.core.session import _VERIFIER_TOKEN_BUDGETS

    budgets_seen: list[int] = []

    async def always_truncated(_schema, *, system, messages, max_tokens):
        budgets_seen.append(max_tokens)
        raise StructuredOutputError(
            "no tool call", truncated=True, output_tokens=max_tokens,
            max_tokens=max_tokens, stop_reason="stop",
        )

    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = AsyncMock(side_effect=always_truncated)

    call_count = 0

    def fake_plan_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _FakeAsyncIter([
                StreamComplete(response=_scratchpad_response("Running.", "exec", "main", "print(1)"))
            ])
        if call_count == 2:
            return _FakeAsyncIter([
                StreamComplete(response=_text_response("Done running the script."))
            ])
        return _FakeAsyncIter([
            StreamComplete(
                response=_text_response(
                    "I ran the script; an internal check failed — how would "
                    "you like to proceed?"
                )
            )
        ])

    mock_llm.plan_stream = fake_plan_stream

    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    try:
        events = [e async for e in session.turn_stream("run my script")]

        assert budgets_seen == list(_VERIFIER_TOKEN_BUDGETS), (
            "the ladder must exhaust every budget before giving up"
        )
        progress_msgs = [e.message for e in events if isinstance(e, StreamTaskProgress)]
        assert "Something went wrong — checking in with you..." in progress_msgs, (
            "exhausted budgets must fail toward the diagnosis, not a silent COMPLETE"
        )
        assert call_count == 3
    finally:
        await session.close()


async def test_empty_diagnosis_is_logged_not_circular(workspace, caplog):
    """Review follow-up: an empty diagnosis must not silently recreate the
    out-of-sync history this path fixes — it logs, and the post-loop fallback
    keeps history consistent (stale-but-true beats empty)."""
    import logging

    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = AsyncMock(side_effect=RuntimeError("hiccup"))

    call_count = 0

    def fake_plan_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _FakeAsyncIter([
                StreamComplete(response=_scratchpad_response("Running.", "exec", "main", "print(1)"))
            ])
        if call_count == 2:
            return _FakeAsyncIter([
                StreamComplete(response=_text_response("Done running the script."))
            ])
        # Diagnosis comes back empty (refusal / provider hiccup).
        return _FakeAsyncIter([StreamComplete(response=_text_response(""))])

    mock_llm.plan_stream = fake_plan_stream

    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    try:
        with caplog.at_level(logging.WARNING):
            async for _ in session.turn_stream("run my script"):
                pass

        assert any(
            "diagnosis returned no content" in r.message for r in caplog.records
        ), "an empty diagnosis must be visible in logs, not silent"
        # History must not gain an empty assistant entry; the post-loop
        # fallback re-appends the (real) pre-verification reply instead.
        assert all(
            m.get("content") not in ("",)
            for m in session.history
            if m.get("role") == "assistant"
        )
    finally:
        await session.close()
