"""ENG-1310 — a persistent provider-auth failure must propagate, not flatten.

`session.py`'s retry-exhaustion fallback only re-raised `TokenLimitExceeded`/
`ModelUnavailableError` so cowork-server could map them to actionable error
cards; a `ConnectionError` (anton's "Invalid API key — …" copy for a 401 from
the LLM gateway, see `openai.py`/`anthropic.py`) fell into the generic
`except Exception` branch and got dumped into the chat as
"An unexpected error occurred: Invalid API key … Please try again or
rephrase your request." instead of reaching cowork-server's
`turn_errors.is_auth_error()`, which already renders the correct "Reconnect
MindsHub" / BYOK-key card — it just never got the chance.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.conftest import make_mock_llm

from anton.core.session import ChatSession, ChatSessionConfig

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


async def _run_turn(session: ChatSession, prompt: str = "what's in my inbox?"):
    events = []
    try:
        async for event in session.turn_stream(prompt):
            events.append(event)
    finally:
        await session.close()
    return events


async def test_persistent_auth_failure_reraises_after_retries_exhausted(workspace):
    mock_llm = make_mock_llm()
    script = _AlwaysRaisingPlanStream(ConnectionError(_AUTH_ERROR_MESSAGE))
    mock_llm.plan_stream = script
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))

    with pytest.raises(ConnectionError, match="Invalid API key"):
        await _run_turn(session)

    # 3 retry attempts (max_auto_retries=2) + the final direct wrap-up call.
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
