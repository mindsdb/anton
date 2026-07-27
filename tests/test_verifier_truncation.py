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


def _damaged_tool_call_response(output_tokens: int) -> LLMResponse:
    """A forced tool call that ran out of budget *inside* its JSON arguments.

    `safe_parse_tool_input` repairs what it can and sets `parse_error`, so the
    call arrives non-empty but incomplete — here missing the required `reason`.
    """
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc_v", name="_VerifierVerdict",
                             input={"status": "WAITING"},
                             parse_error="Unterminated string starting at: line 1")],
        usage=Usage(input_tokens=100, output_tokens=output_tokens),
        stop_reason="stop",
    )


async def test_tool_call_truncated_mid_arguments_is_retryable():
    """The budget can run out *inside* the tool call, not only before it.

    That arrives as a non-empty but incomplete tool call, so the missing-call
    check doesn't see it and validation fails instead — which without this
    would read as a schema bug, skip the retry, and land right back on the
    silent-stop fail-safe this whole change exists to remove.
    """
    llm = _client_with_response(_damaged_tool_call_response(output_tokens=2048))

    with pytest.raises(StructuredOutputError) as exc_info:
        await llm.generate_object_code(
            _VerifierVerdict, system="s", messages=[{"role": "user", "content": "m"}],
            max_tokens=2048,
        )

    assert exc_info.value.truncated is True, "must be retryable, not a schema error"
    assert "unusable tool call" in str(exc_info.value)


async def test_schema_mismatch_under_budget_is_not_disguised_as_truncation():
    """The mirror case: a malformed tool call with budget to spare is a real
    schema failure. It must keep propagating as a validation error rather than
    being relabelled truncation and buying a retry that cannot help."""
    llm = _client_with_response(_damaged_tool_call_response(output_tokens=60))

    with pytest.raises(ValueError) as exc_info:
        await llm.generate_object_code(
            _VerifierVerdict, system="s", messages=[{"role": "user", "content": "m"}],
            max_tokens=2048,
        )

    assert not isinstance(exc_info.value, StructuredOutputError), (
        "a genuine schema mismatch must stay a validation error"
    )


def test_shared_classifier_is_used_by_both_structured_paths():
    """The async client and the scratchpad's sync twin must classify identically.

    `raise_unusable_tool_call` is the single implementation both call — the whole
    point of `structured.py`. Guarded here because the previous version of this
    fix put the logic in `client.py`, which silently left the sync path
    (exposed to model-written scratchpad code, with a caller-chosen budget)
    raising a blind ValueError.
    """
    import ast

    from anton.core.llm import structured

    assert hasattr(structured, "raise_unusable_tool_call")
    # Parse the source rather than importing it — `scratchpad_boot` is a
    # subprocess bootstrap and reads stdin at import time. AST rather than a
    # substring search, so a passing mention in a comment or a stale import
    # can't satisfy it: the call must be inside `generate_object` itself.
    boot_src = (
        Path(__file__).resolve().parents[1]
        / "anton" / "core" / "backends" / "scratchpad_boot.py"
    ).read_text()
    tree = ast.parse(boot_src)
    generate_object = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "generate_object"),
        None,
    )
    assert generate_object is not None, "sync generate_object not found"
    called = {
        n.func.id for n in ast.walk(generate_object)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "raise_unusable_tool_call" in called, (
        "the sync scratchpad path must call the shared classifier, not raise its own"
    )
    assert "LLM did not return structured output." not in boot_src, (
        "the old blind ValueError should be gone from the sync path"
    )


def test_classifier_tolerates_a_response_without_usage():
    """Defensive: a provider response missing `usage` must not blow up the
    classifier — it just can't prove truncation, which is the safe direction
    (no retry bought without evidence)."""
    from anton.core.llm.structured import raise_unusable_tool_call

    class _Bare:
        content = "some prose"

    with pytest.raises(StructuredOutputError) as exc_info:
        raise_unusable_tool_call(_Bare(), tool_name="_VerifierVerdict", budget=2048)

    assert exc_info.value.truncated is False
    assert exc_info.value.output_tokens == 0


# --------------------------------------------------------------------------
# 2. The verifier retries a truncated verdict, once, with more room.
# --------------------------------------------------------------------------


class _ToolThenText:
    """`plan_stream` fake: one tool round per turn, then plain text.

    Keyed off "have I already used the tool this turn?" rather than a call
    counter — a counter with `% 2` would silently desync if a turn ever made a
    third `plan_stream` call (an extra tool round, an internal recovery retry),
    and the test would then be asserting something other than it claims.
    Call `next_turn()` between turns.
    """

    def __init__(self):
        self.tool_used = False

    def next_turn(self) -> None:
        self.tool_used = False

    def __call__(self, **kwargs):
        if not self.tool_used:
            self.tool_used = True
            return _FakeAsyncIter([StreamComplete(response=_scratchpad_response("Running."))])
        return _FakeAsyncIter([StreamComplete(response=_text_response("Done."))])


def _session_that_uses_a_tool(mock_llm, workspace) -> ChatSession:
    """Session whose turn uses one tool, so the completion verifier runs."""
    mock_llm.plan_stream = _ToolThenText()
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


async def test_attempts_are_bounded_and_repeat_each_turn(workspace):
    """Truncation retries are bounded by the budget list — and are re-tried on a
    later turn rather than latched off.

    Output length varies ~6.7x per call for an identical request, so one
    truncation is a tail sample, not proof the model can never fit a verdict.
    Latching the retry off on that evidence would bring silent stops back.
    """
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
        assert budgets == list(_VERIFIER_TOKEN_BUDGETS), "bounded by the budget list"

        budgets.clear()
        mock_llm.plan_stream.next_turn()
        async for _ in session.turn_stream("second turn"):
            pass
    finally:
        await session.close()

    assert budgets == list(_VERIFIER_TOKEN_BUDGETS), (
        "a later turn gets a fresh chance — the retry is not latched off"
    )


@pytest.mark.parametrize("first_failure", ["no_tool_call", "damaged_tool_call"])
async def test_truncation_retry_through_the_real_client(workspace, first_failure):
    """End-to-end: both shapes of a truncated verdict must flow through the real
    `LLMClient` detection into a session-level retry — prose-at-the-cap with no
    call at all, and a call cut off inside its own arguments.

    The other session tests raise `StructuredOutputError` by hand, so they would
    pass even if the detection in `raise_unusable_tool_call` were wrong. This one
    wires a fake *provider* through the real client, so provider → client →
    session is the actual code path (ENG-747: don't hand-build error fixtures).
    """
    calls: list[int] = []

    async def fake_complete(*, model, system, messages, tools, tool_choice, max_tokens):
        calls.append(max_tokens)
        if len(calls) == 1:
            if first_failure == "damaged_tool_call":
                # Budget ran out *inside* the call's JSON arguments.
                return _damaged_tool_call_response(output_tokens=max_tokens)
            # Narrated right up to the ceiling, never reached the tool call —
            # and the gateway calls that `finish_reason: "stop"` (ENG-1082).
            return _text_response("Let me analyze this conversation carefully...",
                                  output_tokens=max_tokens, stop_reason="stop")
        return LLMResponse(
            content="",
            tool_calls=[ToolCall(id="tc_v", name="_VerifierVerdict",
                                 input={"status": "WAITING", "reason": "asked the user"})],
            usage=Usage(input_tokens=100, output_tokens=60),
            stop_reason="tool_calls",
        )

    provider = MagicMock()
    provider.complete = AsyncMock(side_effect=fake_complete)
    real_client = LLMClient(
        planning_provider=provider, planning_model="planner",
        coding_provider=provider, coding_model="coder",
    )

    # Only the verdict call goes through the real client; the turn itself still
    # uses the mock so the test stays a unit test.
    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = real_client.generate_object_code

    session = _session_that_uses_a_tool(mock_llm, workspace)
    try:
        async for _ in session.turn_stream("build me a dashboard"):
            pass
    finally:
        await session.close()

    assert calls == list(_VERIFIER_TOKEN_BUDGETS), (
        "the real client must classify prose-at-the-cap as truncated and the "
        "session must retry it with the larger budget"
    )
    # The retried verdict (WAITING) stands: no forced continuation.
    assert not any(
        "SYSTEM: Task verification determined this task is not yet complete"
        in str(m.get("content", ""))
        for m in session.history
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


# --------------------------------------------------------------------------
# 3. Verifier logs must not carry conversation content (review finding).
# --------------------------------------------------------------------------


def test_error_detail_never_leaks_the_rejected_value():
    """`_safe_error_detail` must identify the failure without quoting content.

    A pydantic ValidationError's message embeds the rejected `input_value` — for
    the verifier that's model-generated text derived from the user's
    conversation — so `str(exc)` cannot go into ordinary application logs.
    """
    import pydantic

    from anton.core.session import _safe_error_detail

    secret = "the user's bank balance is 12345"
    try:
        _VerifierVerdict.model_validate({"status": secret, "reason": secret})
    except pydantic.ValidationError as exc:
        detail = _safe_error_detail(exc)
    else:
        raise AssertionError("expected a ValidationError")

    assert secret not in detail, f"rejected value leaked into the log detail: {detail!r}"
    assert "12345" not in detail
    # Still useful: names the exception type and which field failed.
    assert "ValidationError" in detail
    assert "status" in detail


def test_error_detail_of_a_provider_error_is_type_and_status_only():
    from anton.core.session import _safe_error_detail

    class _FakeAPIError(Exception):
        status_code = 503

    detail = _safe_error_detail(_FakeAPIError("upstream said: <full response body>"))
    assert detail == "_FakeAPIError(status=503)"
    assert "response body" not in detail


def test_error_detail_falls_back_to_the_type_name():
    from anton.core.session import _safe_error_detail

    detail = _safe_error_detail(RuntimeError("provider hiccup with conversation text"))
    assert detail == "RuntimeError"
