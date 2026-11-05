"""Remote cells carry turn identity as HTTP metadata, separately from source."""

import asyncio
import json
from pathlib import Path

import httpx2 as httpx

from anton.core.backends.remote import RemoteScratchpadRuntime
from anton.core.llm.tracing import (
    TraceContext,
    get_trace_context,
    reset_trace_context,
    set_trace_context,
)


def _cell_response(code: str) -> httpx.Response:
    event = {"type": "cell", "cell": {"code": code, "stdout": "ok", "error": None}}
    return httpx.Response(
        200,
        headers={"Content-Type": "text/event-stream"},
        content=f"data: {json.dumps(event)}\n\n",
    )


def _use_transport(monkeypatch, handler):
    client_class = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "anton.core.backends.remote.httpx.AsyncClient",
        lambda **kwargs: client_class(transport=transport, **kwargs),
    )


async def test_each_remote_cell_sends_its_current_trace_separately_from_code(monkeypatch):
    requests = []

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/scratchpad/execute-stream"
        body = json.loads(request.content)
        requests.append(body)
        return _cell_response(body["code"])

    _use_transport(monkeypatch, handler)
    pad = RemoteScratchpadRuntime("main", endpoint_url="https://pad.test", api_key="test-key")
    code = 'answer = get_llm().complete(system="Be brief.", messages=[])'
    traces = [
        TraceContext(
            session_id="conversation",
            turn_id=turn,
            harness="anton",
            surface="desktop",
            tags=("eval",),
            # Hosts can attach path-like annotations. Match the local wire
            # serializer's string fallback instead of failing the cell.
            metadata={"question_id": f"q{turn}", "artifact": Path("output.csv")},
        )
        for turn in (1, 2)
    ] + [None]

    for trace in traces:
        token = set_trace_context(trace)
        try:
            cell = await pad.execute(code, description="Ask", estimated_seconds=30)
            assert get_trace_context() is trace
        finally:
            reset_trace_context(token)
        assert cell.error is None
        assert cell.code == code

    assert [request["trace_context"] for request in requests] == [
        {
            "session_id": "conversation",
            "turn_id": turn,
            "harness": "anton",
            "surface": "desktop",
            "tags": ["eval"],
            "metadata": {"question_id": f"q{turn}", "artifact": "output.csv"},
        }
        for turn in (1, 2)
    ] + [None]
    assert all(request["code"] == code for request in requests)
    assert all(request["name"] == "main" for request in requests)
    assert all(request["description"] == "Ask" for request in requests)
    assert all(request["estimated_seconds"] == 30 for request in requests)
    assert [cell.code for cell in pad.cells] == [code] * 3


async def test_concurrent_remote_requests_keep_their_own_trace(monkeypatch):
    requests = []
    both_started = asyncio.Event()

    async def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=5)
        return _cell_response(body["code"])

    _use_transport(monkeypatch, handler)

    async def execute(question):
        pad = RemoteScratchpadRuntime(question, endpoint_url="https://pad.test", api_key="test-key")
        token = set_trace_context(
            TraceContext(session_id=question, metadata={"question_id": question})
        )
        try:
            return await pad.execute("print('ok')")
        finally:
            reset_trace_context(token)

    cells = await asyncio.gather(execute("q1"), execute("q2"))

    assert all(cell.error is None for cell in cells)
    assert {
        (request["name"], request["trace_context"]["session_id"],
         request["trace_context"]["metadata"]["question_id"])
        for request in requests
    } == {("q1", "q1", "q1"), ("q2", "q2", "q2")}
