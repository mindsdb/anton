"""A dropped stream in the artifact pipeline is retried once at half the budget."""
from __future__ import annotations

import httpx2
import pytest

from anton.core.llm.provider import LLMResponse, StreamComplete, TransientProviderError, Usage
from anton.core.tools.generate_artifact.engine import _call_with_stream_retry


def _scripted(*outcomes):
    calls: list[dict] = []
    pending = list(outcomes)

    def llm_call(**kw):
        calls.append(kw)
        outcome = pending.pop(0)

        async def gen():
            if isinstance(outcome, BaseException):
                raise outcome
            yield StreamComplete(response=outcome)

        return gen()

    return llm_call, calls


OK = LLMResponse(content="ok", usage=Usage(output_tokens=3))


@pytest.mark.parametrize(
    "drop",
    [
        httpx2.RemoteProtocolError("peer closed connection"),
        TransientProviderError("lost", code="connection_error", session_backoff=True),
    ],
)
async def test_drop_is_retried_at_half_budget(drop):
    llm_call, calls = _scripted(drop, OK)
    response, used = await _call_with_stream_retry(
        llm_call, system="s", messages=[], tools=None, max_tokens=16384, default_cap=16384,
    )
    assert response is OK
    assert [c["max_tokens"] for c in calls] == [16384, 8192]
    assert used == 8192


async def test_other_transient_errors_propagate():
    llm_call, calls = _scripted(TransientProviderError("boom", code="stream_error", session_backoff=True))
    with pytest.raises(TransientProviderError):
        await _call_with_stream_retry(
            llm_call, system="s", messages=[], tools=None, max_tokens=16384, default_cap=16384,
        )
    assert len(calls) == 1
