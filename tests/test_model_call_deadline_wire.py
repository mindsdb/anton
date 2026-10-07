"""The model-call deadline against a real OpenAI SDK client on a real socket.

The OpenAI SDK drops SSE comment lines, so a gateway's `: keepalive` every few
seconds holds the connection open while the client sees nothing. The deadline
must still fire on such a call, and must not fire on one that keeps writing.
The threaded stub serves both shapes.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from anton.core.llm.client import LLMClient
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import StreamComplete, StreamTextDelta
from tests.e2e.stub_server import StubServer

DEADLINE_S = 0.5


@pytest.fixture
def stub():
    with StubServer() as server:
        yield server


def _client(stub: StubServer) -> LLMClient:
    provider = OpenAIProvider(
        api_key="stub-key",
        base_url=stub.base_url,
        flavor=OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC,
    )
    client = LLMClient(
        planning_provider=provider, planning_model="stub-model",
        coding_provider=provider, coding_model="stub-model",
    )
    client.model_call_idle_timeout_s = DEADLINE_S
    return client


async def test_a_hold_with_keepalives_ends_at_the_deadline(stub):
    stub.script(streamed=["hold:3:done"], keepalive_s=0.05)
    client = _client(stub)
    text: list[str] = []
    started = time.monotonic()

    try:
        async def _consume():
            async for event in client.plan_stream(
                system="s", messages=[{"role": "user", "content": "hi"}]
            ):
                if isinstance(event, StreamTextDelta):
                    text.append(event.text)

        try:
            await asyncio.wait_for(_consume(), timeout=5)
        except Exception as exc:  # noqa: BLE001 - the type is the assertion
            raised = exc
        else:
            raised = None
        elapsed = time.monotonic() - started
    finally:
        await client.aclose()

    assert raised is not None, f"the held call answered after {elapsed:.2f}s: {''.join(text)!r}"
    assert type(raised).__name__ == "ModelCallTimeoutError", repr(raised)
    assert elapsed < 0.8, f"took {elapsed:.2f}s against a {DEADLINE_S}s deadline"
    assert "done" not in "".join(text)
    assert stub.request_count == 1, "the deadline must not trigger an SDK retry"


async def test_a_trickling_stream_is_not_cut(stub):
    stub.script(streamed=["trickle:1.2:0.1"], keepalive_s=0)
    client = _client(stub)
    events: list = []
    try:
        async for event in client.plan_stream(
            system="s", messages=[{"role": "user", "content": "hi"}]
        ):
            events.append(event)
    finally:
        await client.aclose()

    deltas = [e for e in events if isinstance(e, StreamTextDelta)]
    assert len(deltas) >= 8, "the trickle must outlast the deadline for this to mean anything"
    assert isinstance(events[-1], StreamComplete)
