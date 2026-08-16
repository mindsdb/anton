"""ENG-1310 — a persistent provider-auth failure must propagate, not flatten.

A `ConnectionError` (anton's "Invalid API key — …" copy for a 401 from the
LLM gateway, see `openai.py`/`anthropic.py`) used to fall into a generic
`except Exception` branch and get dumped into the chat as "An unexpected
error occurred: Invalid API key … Please try again or rephrase your
request." instead of reaching cowork-server's `turn_errors.is_auth_error()`,
which already renders the correct "Reconnect MindsHub" / BYOK-key card.

Two sites in `turn_stream` needed the same auth-shaped check, mirroring how
ENG-1139 treats `EndpointConfigurationError` (also deterministic — retrying
can't fix it):

1. The immediate re-raise at the top of the retry loop — an invalid key
   fails on the FIRST attempt instead of burning the count-based retry
   budget on doomed retries.
2. The retry-exhaustion fallback's own wrap-up call — belt-and-suspenders
   for the case where retries were legitimately spent on a DIFFERENT
   failure and the key only turns out to be bad on the final summary call.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.conftest import make_mock_llm

from anton.core.llm.provider import EndpointConfigurationError
from anton.core.session import ChatSession, ChatSessionConfig, _is_provider_auth_error

_AUTH_ERROR_MESSAGE = "Invalid API key — check your OpenAI API key configuration."


@pytest.fixture()
def workspace():
    # Keep scratchpad venvs inside the repo workspace (pytest runs sandboxed
    # and can't write to the real home directory).
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


class _AlwaysRaisingPlanStream:
    """`plan_stream` fake that raises the same exception on every call —
    every retry attempt AND the final wrap-up call see the same failure,
    the way a genuinely invalid key does."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        raise self._exc


class _ScriptedExceptionPlanStream:
    """`plan_stream` fake that raises a scripted sequence of exceptions, one
    per call, holding on the last entry once the script runs out — so a
    fixed prefix (e.g. retries that legitimately exhaust the count-based
    budget) can be followed by a different failure on the final call."""

    def __init__(self, excs: list[Exception]):
        self._excs = list(excs)
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        idx = min(self.calls - 1, len(self._excs) - 1)
        raise self._excs[idx]


async def _run_turn(session: ChatSession, prompt: str = "what's in my inbox?"):
    events = []
    try:
        async for event in session.turn_stream(prompt):
            events.append(event)
    finally:
        await session.close()
    return events


async def test_persistent_auth_failure_fails_immediately_without_wasting_retries(workspace):
    """An invalid key can't be fixed by retrying — it must fail on the first
    attempt, the same way EndpointConfigurationError (ENG-1139) does, not
    after burning the count-based retry budget on doomed re-attempts."""
    mock_llm = make_mock_llm()
    script = _AlwaysRaisingPlanStream(ConnectionError(_AUTH_ERROR_MESSAGE))
    mock_llm.plan_stream = script
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))

    with pytest.raises(ConnectionError, match="Invalid API key"):
        await _run_turn(session)

    assert script.calls == 1, "an auth failure must not be retried"


async def test_auth_failure_on_the_final_wrapup_call_still_reraises(workspace):
    """Retries legitimately exhaust on a DIFFERENT, retryable failure — the
    key only turns out to be bad on the retry-exhaustion fallback's own
    wrap-up call. That must still propagate instead of flattening into chat
    text, even though the auth error never triggered the fast-fail path
    above."""
    mock_llm = make_mock_llm()
    script = _ScriptedExceptionPlanStream(
        [RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom"),
         ConnectionError(_AUTH_ERROR_MESSAGE)]
    )
    mock_llm.plan_stream = script
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))

    with pytest.raises(ConnectionError, match="Invalid API key"):
        await _run_turn(session)

    # 3 retry attempts (max_auto_retries=2) on the unrelated RuntimeError,
    # then the final direct wrap-up call hits the auth error.
    assert script.calls == 4


def test_is_provider_auth_error_matches_only_the_invalid_key_copy():
    """The predicate both re-raise sites share — pinned directly so the two
    call sites can't drift from each other (review feedback on ENG-1310)."""
    assert _is_provider_auth_error(ConnectionError(_AUTH_ERROR_MESSAGE))
    assert _is_provider_auth_error(ConnectionError("INVALID API KEY — case insensitive"))
    assert not _is_provider_auth_error(ConnectionError("temporarily unavailable"))
    assert not _is_provider_auth_error(RuntimeError(_AUTH_ERROR_MESSAGE))


async def test_endpoint_configuration_error_on_the_final_wrapup_call_still_reraises(workspace):
    """The wrap-up call's except block must treat EndpointConfigurationError
    (ENG-1139 — also deterministic, also must default to 'setup' not
    'retry') the same way the immediate re-raise site already does, instead
    of flattening it into chat text (review feedback on ENG-1310)."""
    mock_llm = make_mock_llm()
    script = _ScriptedExceptionPlanStream(
        [RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom"),
         EndpointConfigurationError("The model endpoint returned 404.")]
    )
    mock_llm.plan_stream = script
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))

    with pytest.raises(EndpointConfigurationError):
        await _run_turn(session)

    assert script.calls == 4


async def test_generic_connection_error_still_falls_back_to_chat_text(workspace):
    """Only the auth-shaped message re-raises — an unrelated ConnectionError
    (e.g. the generic 'temporarily unavailable' case) keeps the existing
    fallback-text behavior instead of failing the turn."""
    mock_llm = make_mock_llm()
    script = _AlwaysRaisingPlanStream(ConnectionError("temporarily unavailable"))
    mock_llm.plan_stream = script
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))

    events = await _run_turn(session)

    from anton.core.llm.provider import StreamTextDelta

    fallback_text = "".join(
        e.text for e in events if isinstance(e, StreamTextDelta)
    )
    assert "temporarily unavailable" in fallback_text
    assert "unexpected error occurred" in fallback_text
