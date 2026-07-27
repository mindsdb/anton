"""Completion-verifier truncation handling (ENG-1081).

The verdict call is a forced tool call. Models that narrate before acting
(MindsHub's Fireworks aliases — `mindshub_air`/`kimi`, `deepseek`) spend the
output budget on plain prose and never reach the call, so a tight `max_tokens`
fails them deterministically: 98.6% of `mindshub_air` verdict calls in prod
returned no tool call, and the fail-safe turned each one into a silent
"task complete" with no message to the user.

Two behaviours are covered here:

1. `_generate_object_with` reports *why* there was no tool call, distinguishing a
   blown budget (retryable) from a genuine failure — by token count, because the
   MindsHub gateway reports `finish_reason: "stop"` at the cap (ENG-1082).
2. The verifier retries a truncated verdict once with a bigger budget, and does
   NOT spend a retry on a failure a bigger budget can't fix.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.conftest import make_mock_llm

from anton.core.llm.client import LLMClient
from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StructuredOutputError,
    ToolCall,
    Usage,
)
from anton.core.session import (
    _VERIFIER_TOKEN_BUDGETS,
    ChatSession,
    ChatSessionConfig,
    _VerifierVerdict,
)


@pytest.fixture()
def workspace():
    # Keep scratchpad venvs inside the repo workspace (pytest runs sandboxed and
    # can't write to the real home directory).
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _text_response(text: str, output_tokens: int = 20, stop_reason: str = "end_turn") -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=output_tokens),
        stop_reason=stop_reason,
    )


def _scratchpad_response(text: str, code: str = "print(1)") -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[ToolCall(
            id="tc_1", name="scratchpad",
            input={"action": "exec", "name": "main", "code": code},
        )],
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


# --------------------------------------------------------------------------
# 1. The client reports *why* the tool call is missing.
# --------------------------------------------------------------------------


def _client_with_response(response: LLMResponse) -> LLMClient:
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=response)
    return LLMClient(
        planning_provider=provider,
        planning_model="planner",
        coding_provider=provider,
        coding_model="coder",
    )


async def test_no_tool_call_at_the_cap_is_reported_as_truncated():
    """Prose that spends the whole budget == truncation, even though the
    gateway calls it `finish_reason: "stop"` (ENG-1082)."""
    llm = _client_with_response(
        _text_response("Let me analyze this conversation carefully. The user...",
                       output_tokens=256, stop_reason="stop")
    )

    with pytest.raises(StructuredOutputError) as exc_info:
        await llm.generate_object_code(
            _VerifierVerdict, system="s", messages=[{"role": "user", "content": "m"}],
            max_tokens=256,
        )

    exc = exc_info.value
    assert exc.truncated is True
    assert exc.output_tokens == 256
    assert exc.max_tokens == 256
    # Callers that only know the documented ValueError still catch it.
    assert isinstance(exc, ValueError)


@pytest.mark.parametrize("stop_reason", ["length", "max_tokens"])
async def test_both_provider_dialects_for_truncation(stop_reason):
    """The gateway/OpenAI dialect says "length"; AnthropicProvider passes
    Anthropic's own "max_tokens" through raw. Both mean truncated."""
    llm = _client_with_response(_text_response("narrating…", output_tokens=100,
                                               stop_reason=stop_reason))

    with pytest.raises(StructuredOutputError) as exc_info:
        await llm.generate_object_code(
            _VerifierVerdict, system="s", messages=[{"role": "user", "content": "m"}],
            max_tokens=2048,
        )

    assert exc_info.value.truncated is True


async def test_stop_reason_length_is_honoured_below_the_cap():
    """Gemini reports truncation honestly and can return almost nothing —
    trust `stop_reason` too, not only the token count."""
    llm = _client_with_response(_text_response("", output_tokens=9, stop_reason="length"))

    with pytest.raises(StructuredOutputError) as exc_info:
        await llm.generate_object_code(
            _VerifierVerdict, system="s", messages=[{"role": "user", "content": "m"}],
            max_tokens=256,
        )

    assert exc_info.value.truncated is True


async def test_short_empty_response_is_not_truncated():
    """A provider that returns nothing well inside the budget is a genuine
    failure — a bigger budget won't fix it, so it must not be retried."""
    llm = _client_with_response(_text_response("", output_tokens=5, stop_reason="stop"))

    with pytest.raises(StructuredOutputError) as exc_info:
        await llm.generate_object_code(
            _VerifierVerdict, system="s", messages=[{"role": "user", "content": "m"}],
            max_tokens=256,
        )

    assert exc_info.value.truncated is False


# --------------------------------------------------------------------------
# 2. The verifier retries a truncated verdict, once, with more room.
# --------------------------------------------------------------------------


def _session_that_uses_a_tool(mock_llm, workspace) -> ChatSession:
    """Session whose every turn uses one tool, so the completion verifier runs.

    Each turn makes exactly two `plan_stream` calls — the tool round, then the
    final text — so alternating keeps the pattern correct across several turns.
    """
    call_count = 0

    def fake_plan_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:
            return _FakeAsyncIter([StreamComplete(response=_scratchpad_response("Running."))])
        return _FakeAsyncIter([StreamComplete(response=_text_response("Done."))])

    mock_llm.plan_stream = fake_plan_stream
    return ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))


async def test_truncated_verdict_is_retried_with_a_bigger_budget(workspace):
    """The narrating-model case: first budget truncates, the retry succeeds, and
    the retried verdict is the one that counts — no silent 'COMPLETE'."""
    budgets: list[int] = []

    async def fake_verdict(_schema, *, system, messages, max_tokens):
        budgets.append(max_tokens)
        if len(budgets) == 1:
            raise StructuredOutputError(
                "no tool call", truncated=True, output_tokens=max_tokens,
                max_tokens=max_tokens, stop_reason="stop",
            )
        return _VerifierVerdict(status="WAITING", reason="asked the user a question")

    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = AsyncMock(side_effect=fake_verdict)

    session = _session_that_uses_a_tool(mock_llm, workspace)
    try:
        async for _ in session.turn_stream("build me a dashboard"):
            pass
    finally:
        await session.close()

    assert budgets == list(_VERIFIER_TOKEN_BUDGETS[:2]), (
        "a truncated verdict must be retried once, with the larger budget"
    )
    # The verdict came from the retry (WAITING → a valid stop), so no
    # "Continue working" continuation was injected.
    assert not any(
        "SYSTEM: Task verification determined this task is not yet complete"
        in str(m.get("content", ""))
        for m in session.history
    )


async def test_non_truncated_failure_is_not_retried(workspace):
    """A failure a bigger budget can't fix costs exactly one call, then falls
    through to the fail-safe."""
    calls: list[int] = []

    async def fake_verdict(_schema, *, system, messages, max_tokens):
        calls.append(max_tokens)
        raise StructuredOutputError(
            "no tool call", truncated=False, output_tokens=3,
            max_tokens=max_tokens, stop_reason="stop",
        )

    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = AsyncMock(side_effect=fake_verdict)

    session = _session_that_uses_a_tool(mock_llm, workspace)
    try:
        async for _ in session.turn_stream("build me a dashboard"):
            pass
    finally:
        await session.close()

    assert calls == [_VERIFIER_TOKEN_BUDGETS[0]], "must not pay for a hopeless retry"


async def test_retry_is_not_bought_twice_in_one_session(workspace):
    """If the retry is truncated too, the model can't fit a verdict at any budget
    we'll pay for — later turns in the session must stop paying for the retry."""
    budgets: list[int] = []

    async def fake_verdict(_schema, *, system, messages, max_tokens):
        budgets.append(max_tokens)
        raise StructuredOutputError(
            "no tool call", truncated=True, output_tokens=max_tokens,
            max_tokens=max_tokens, stop_reason="stop",
        )

    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = AsyncMock(side_effect=fake_verdict)

    session = _session_that_uses_a_tool(mock_llm, workspace)
    try:
        async for _ in session.turn_stream("first turn"):
            pass
        assert budgets == list(_VERIFIER_TOKEN_BUDGETS), "turn 1 tries both budgets"

        # Same session, second turn: only the first budget should be attempted.
        budgets.clear()
        async for _ in session.turn_stream("second turn"):
            pass
    finally:
        await session.close()

    assert budgets == [_VERIFIER_TOKEN_BUDGETS[0]], (
        "the retry must not be re-bought on every later turn"
    )


async def test_verifier_prompt_forbids_preamble(workspace):
    """The no-preamble instruction reaches the model. It is not sufficient on its
    own (0/3 at 256 with it), but it shortens the preamble enough to matter."""
    seen: dict = {}

    async def fake_verdict(_schema, *, system, messages, max_tokens):
        seen["system"] = system
        return _VerifierVerdict(status="COMPLETE", reason="done")

    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = AsyncMock(side_effect=fake_verdict)

    session = _session_that_uses_a_tool(mock_llm, workspace)
    try:
        async for _ in session.turn_stream("build me a dashboard"):
            pass
    finally:
        await session.close()

    assert "immediately as your first action" in seen["system"]
    assert "Do not think out loud" in seen["system"]
