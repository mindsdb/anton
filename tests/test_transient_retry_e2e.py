"""ENG-673 e2e — deterministic mock-provider harness.

Drives the **real** anthropic / openai SDK stream parsers against an
``httpx.MockTransport`` so we exercise the genuine wire path, not a hand-built
exception. Proves the core fix on BOTH providers: a provider failure arriving
*mid-stream inside an HTTP-200 stream* is classified as a retryable
``TransientProviderError`` (``session_backoff=True``), never the misleading
"Server returned 200".

Complements ``test_transient_retry.py`` (unit-level classifier + session
backoff/budget). The turn-level cadence/budget/idempotency behavior lives there
against a faked stream; here we prove the classification the real SDK produces
is the input those session tests assume.

No network, no sleeping, no waiting for a real incident — the transport returns
canned SSE bytes.
"""

from __future__ import annotations

import anthropic
import httpx
import openai
import pytest

from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import (
    StreamComplete,
    StreamTextDelta,
    TransientProviderError,
)


# --------------------------------------------------------------------------- #
# wire payloads
# --------------------------------------------------------------------------- #

def _sse(*lines: str) -> bytes:
    return ("".join(lines)).encode()


# Anthropic Messages API: a 200 stream that emits message_start then an SSE
# `error` event carrying overloaded_error — the exact BUG-CM-001 shape.
_ANTH_MIDSTREAM_OVERLOAD = _sse(
    'event: message_start\n',
    'data: {"type":"message_start","message":{"id":"msg_1","type":"message",'
    '"role":"assistant","model":"claude","content":[],"stop_reason":null,'
    '"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}\n\n',
    'event: error\n',
    'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n',
)

# Anthropic: a clean, complete stream (sanity — the harness itself is honest).
_ANTH_GOOD = _sse(
    'event: message_start\n',
    'data: {"type":"message_start","message":{"id":"msg_2","type":"message",'
    '"role":"assistant","model":"claude","content":[],"stop_reason":null,'
    '"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}\n\n',
    'event: content_block_start\n',
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
    'event: content_block_delta\n',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello world"}}\n\n',
    'event: content_block_stop\n',
    'data: {"type":"content_block_stop","index":0}\n\n',
    'event: message_delta\n',
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},'
    '"usage":{"output_tokens":3}}\n\n',
    'event: message_stop\n',
    'data: {"type":"message_stop"}\n\n',
)

# OpenAI chat.completions: a 200 stream that emits one good chunk then an SSE
# `error` — the SDK surfaces this as a BARE openai.APIError (status_code=None).
_OAI_MIDSTREAM_ERROR = _sse(
    'data: {"id":"1","object":"chat.completion.chunk","choices":'
    '[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}\n\n',
    'data: {"error":{"message":"The server is overloaded","type":"server_error","code":null}}\n\n',
)


# --------------------------------------------------------------------------- #
# provider factories wired to a MockTransport
# --------------------------------------------------------------------------- #

def _anthropic_provider(handler) -> AnthropicProvider:
    prov = AnthropicProvider(api_key="test")
    prov._client = anthropic.AsyncAnthropic(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return prov


def _openai_provider(handler) -> OpenAIProvider:
    # Default flavor is openai-compatible-generic → the chat.completions path
    # (the MindsHub-passthrough dialect), which is what MindsHub routing uses.
    prov = OpenAIProvider(api_key="test", base_url="http://mock/v1")
    prov._client = openai.AsyncOpenAI(
        api_key="test", base_url="http://mock/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return prov


def _static(body: bytes):
    def handler(_req):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )
    return handler


async def _drain(prov):
    return [
        ev
        async for ev in prov.stream(
            model="latest:sonnet", system="s",
            messages=[{"role": "user", "content": "hi"}],
        )
    ]


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #

async def test_e2e_anthropic_midstream_overload_is_transient_not_confusing_200():
    prov = _anthropic_provider(_static(_ANTH_MIDSTREAM_OVERLOAD))
    with pytest.raises(TransientProviderError) as ei:
        await _drain(prov)
    exc = ei.value
    assert exc.code == "overloaded_error"
    # The whole point of ENG-673: the 200 must NOT leak into the message, and
    # the session must be allowed to back off (mid-stream had no SDK retry).
    assert "200" not in str(exc)
    assert exc.session_backoff is True


async def test_e2e_anthropic_clean_stream_completes():
    prov = _anthropic_provider(_static(_ANTH_GOOD))
    events = await _drain(prov)
    text = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    completes = [e for e in events if isinstance(e, StreamComplete)]
    assert text == "Hello world"
    assert len(completes) == 1
    assert completes[0].response.stop_reason == "end_turn"


async def test_e2e_anthropic_recovers_after_n_failures():
    """Attempt log verifies the fail-N-then-succeed shape at the SDK layer:
    the same request resent after transient failures eventually completes."""
    state = {"n": 0}

    def handler(_req):
        state["n"] += 1
        body = _ANTH_MIDSTREAM_OVERLOAD if state["n"] <= 2 else _ANTH_GOOD
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )

    prov = _anthropic_provider(handler)
    # First two calls fail transiently...
    for _ in range(2):
        with pytest.raises(TransientProviderError):
            await _drain(prov)
    # ...the third (identical) request succeeds — exactly what the session's
    # backoff loop replays.
    events = await _drain(prov)
    assert "".join(e.text for e in events if isinstance(e, StreamTextDelta)) == "Hello world"
    assert state["n"] == 3


# --------------------------------------------------------------------------- #
# OpenAI / MindsHub (chat.completions dialect)
# --------------------------------------------------------------------------- #

async def test_e2e_openai_midstream_error_is_transient():
    """The OpenAI-shaped gap: a mid-stream SSE error is a bare openai.APIError
    (no status_code), which must still classify as transient + backoff-able —
    'the fix is not tested if only Anthropic is mocked' (Sam's review)."""
    prov = _openai_provider(_static(_OAI_MIDSTREAM_ERROR))
    with pytest.raises(TransientProviderError) as ei:
        await _drain(prov)
    exc = ei.value
    assert "200" not in str(exc)
    assert exc.session_backoff is True
    # server_error is in the transient set → classified by type, not a fallback.
    assert exc.code in ("server_error", "stream_error")
