"""A pad's model helpers keep answering after their first call.

Every sync helper runs its model call under ``asyncio.run``, which closes its
event loop on return. These tests run the real boot script against a local
HTTP/1.1 server that keeps connections alive. A client kept across calls would
then hand the next call a connection pooled on the earlier call's closed loop.
That call either raises "Event loop is closed" or, on openai SDKs that retry
any exception, sends its request again with a nonzero
``x-stainless-retry-count``. Either way it shows here as a cell error, an
extra request, or a retry count above 0. A call's client also has to be closed
on that call's loop; one left open shows as an error the ``asyncio`` logger
records. And a call closes only its own client, so calls running beside it
still get their answers.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter

from anton.core.backends.wire import CELL_DELIM, RESULT_START


REPO_ROOT = Path(__file__).resolve().parents[1]
BOOT_SCRIPT = REPO_ROOT / "anton" / "core" / "backends" / "scratchpad_boot.py"


@dataclass(frozen=True)
class _SeenRequest:
    path: str
    retry_count: str | None  # x-stainless-retry-count: "0" on a call's first attempt
    connection: int  # the client's source port, one per TCP connection
    prompt: str


class _InputItem(BaseModel):
    type: str = "message"
    role: str | None = None
    content: str | list[Any] | None = None


class _Tool(BaseModel):
    type: str
    name: str | None = None


class _ResponsesRequest(BaseModel):
    """The fields of a Responses API request that the stub reads."""

    model: str
    input: list[_InputItem]
    tools: list[_Tool] = []

    def prompt(self) -> str:
        """The first user message, which names the helper call that sent it."""
        return next(
            item.content
            for item in self.input
            if item.role == "user" and isinstance(item.content, str)
        )


class _CellResult(BaseModel):
    """One cell's result block, as the boot script writes it."""

    stdout: str
    stderr: str
    logs: str
    error: str | None


class _LoggedRecord(BaseModel):
    """A log record the pad kept, as the last cell prints it."""

    name: str
    levelno: int
    message: str


def _responses_object(request: _ResponsesRequest) -> dict[str, Any]:
    """A completed, non-streamed Responses API object.

    The first round of a request that offers a function tool gets a call to
    it, so agentic_loop takes a second round; everything else gets text.
    """
    offers_function = any(tool.type == "function" for tool in request.tools)
    answered_tool = any(item.type == "function_call_output" for item in request.input)
    if offers_function and not answered_tool:
        output = [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": request.tools[0].name,
                "arguments": "{}",
                "status": "completed",
            }
        ]
    else:
        output = [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": f"answer to: {request.prompt()}",
                        "annotations": [],
                    }
                ],
            }
        ]
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": request.model,
        "output": output,
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


# Seconds the stub waits before it answers a held prompt. The call answered at
# once finishes and closes its client well inside this, while the held calls
# beside it still wait for their answers.
_HELD_S = 0.5


class _StubServer(ThreadingHTTPServer):
    # A handler thread can sit on an idle kept-alive socket; neither process
    # exit nor server_close() waits for it.
    daemon_threads = True
    block_on_close = False

    def __init__(self, *, idle_timeout: float | None, held: frozenset[str]) -> None:
        super().__init__(("127.0.0.1", 0), _ResponsesHandler)
        self.idle_timeout = idle_timeout
        self.held = held  # prompts answered _HELD_S late
        self.seen: list[_SeenRequest] = []


class _ResponsesHandler(BaseHTTPRequestHandler):
    """Answers every POST with one Responses API object and records it. A held
    prompt gets its answer ``_HELD_S`` late."""

    # HTTP/1.1 with Content-Length and no "Connection: close": the client keeps
    # the connection in its pool for the next request.
    protocol_version = "HTTP/1.1"
    server: _StubServer

    def setup(self) -> None:
        # None keeps an idle connection open until the client closes it. A
        # number drops it after that many idle seconds, as a load balancer does.
        self.timeout = self.server.idle_timeout
        super().setup()

    def log_message(self, format: str, *args: object) -> None:
        """Silent: the test reads ``server.seen`` instead."""

    def do_POST(self) -> None:
        request = _ResponsesRequest.model_validate_json(
            self.rfile.read(int(self.headers["Content-Length"]))
        )
        self.server.seen.append(
            _SeenRequest(
                path=self.path,
                retry_count=self.headers.get("x-stainless-retry-count"),
                connection=self.client_address[1],
                prompt=request.prompt(),
            )
        )
        if request.prompt() in self.server.held:
            time.sleep(_HELD_S)
        payload = json.dumps(_responses_object(request)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextmanager
def _serving(
    *, idle_timeout: float | None = None, held: frozenset[str] = frozenset()
) -> Iterator[_StubServer]:
    server = _StubServer(idle_timeout=idle_timeout, held=held)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _run_cells(cells: list[str], tmp_path: Path, base_url: str) -> list[_CellResult]:
    """Run ``cells`` in order in one pad process: the real boot script."""
    env = os.environ.copy()
    env.update(
        {
            "ANTON_SCRATCHPAD_MODEL": "stub-model",
            "ANTON_SCRATCHPAD_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": base_url,
            "ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL": "0",
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    completed = subprocess.run(
        [sys.executable, str(BOOT_SCRIPT)],
        input="".join(f"{cell}\n{CELL_DELIM}\n" for cell in cells),
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    results = [
        _CellResult.model_validate_json(lines[index + 1])
        for index, line in enumerate(lines)
        if line == RESULT_START
    ]
    assert len(results) == len(cells), completed.stdout
    return results


# Keeps the pad's log records from here to the end of the session, so the test
# can filter them by logger name and level. A cell's `logs` field is formatted
# text with no level in it.
_KEEP_LOG_RECORDS_CELL = """
import logging

class KeptRecords(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)

kept = KeptRecords()
logging.getLogger().addHandler(kept)
"""

_PRINT_LOG_RECORDS_CELL = """
import json

print(json.dumps([{"name": r.name, "levelno": r.levelno, "message": r.getMessage()} for r in kept.records]))
"""

# Two web_search() calls, then two get_llm().complete() calls, in one cell, the
# way a research question uses them. `pause()` waits between calls; PAUSE_S is
# set per case.
_FOUR_CALLS_CELL = """
import json
import time

def pause():
    time.sleep(PAUSE_S)

answers = [web_search("web one")]
pause()
answers.append(web_search("web two"))
pause()
answers.append(get_llm().complete(system="Be brief.", messages=[{"role": "user", "content": "complete one"}]).content)
pause()
answers.append(get_llm().complete(system="Be brief.", messages=[{"role": "user", "content": "complete two"}]).content)
print(json.dumps(answers))
"""

# A later cell of the same pad: agentic_loop (two rounds, one tool call), then
# complete_async() twice, each inside an asyncio.run() of the cell's own.
_LATER_CELL = """
import asyncio

tools_run = []

def handle_tool(name, tool_input):
    tools_run.append(name)
    return "sunny"

weather = {"name": "weather", "description": "Today's weather.", "input_schema": {"type": "object", "properties": {}}}
pause()
looped = agentic_loop(system="Be brief.", user_message="agentic loop", tools=[weather], handle_tool=handle_tool)

async def ask_async(prompt):
    response = await get_llm().complete_async(system="Be brief.", messages=[{"role": "user", "content": prompt}])
    return response.content

pause()
async_one = asyncio.run(ask_async("complete async one"))
pause()
async_two = asyncio.run(ask_async("complete async two"))
print(json.dumps({"agentic_loop": looped, "tools_run": tools_run, "complete_async": [async_one, async_two]}))
"""


@pytest.mark.slow
@pytest.mark.parametrize(
    ("idle_timeout", "pause"),
    [(None, 0.0), (0.2, 0.5)],
    ids=["back-to-back-on-a-kept-alive-connection", "after-the-server-drops-the-idle-connection"],
)
def test_every_helper_call_gets_its_answer_from_one_request(tmp_path, idle_timeout, pause):
    """Back to back, a client kept across calls would reuse the earlier call's
    connection. After the server drops an idle connection, as real endpoints
    do, that client would have to close it first: the stale connection fails
    before anything is sent, so only the retry count shows the hidden retry."""
    with _serving(idle_timeout=idle_timeout) as server:
        _, first, later, logged = _run_cells(
            [
                _KEEP_LOG_RECORDS_CELL,
                f"PAUSE_S = {pause}\n{_FOUR_CALLS_CELL}",
                _LATER_CELL,
                _PRINT_LOG_RECORDS_CELL,
            ],
            tmp_path,
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
        )

    assert first.error is None, first.error
    assert later.error is None, later.error
    assert json.loads(first.stdout) == [
        "answer to: web one",
        "answer to: web two",
        "answer to: complete one",
        "answer to: complete two",
    ]
    assert json.loads(later.stdout) == {
        "agentic_loop": "answer to: agentic loop",
        "tools_run": ["weather"],
        "complete_async": ["answer to: complete async one", "answer to: complete async two"],
    }
    seen = "\n".join(map(repr, server.seen)) + "\n" + first.logs + later.logs
    # One request per model call: a retried call repeats its prompt.
    assert [request.prompt for request in server.seen] == [
        "web one",
        "web two",
        "complete one",
        "complete two",
        "agentic loop",
        "agentic loop",
        "complete async one",
        "complete async two",
    ], seen
    assert [request.retry_count for request in server.seen] == ["0"] * 8, seen
    assert {request.path for request in server.seen} == {"/v1/responses"}, seen
    # A client left open is closed later by the SDK's finalizer, on whichever
    # loop runs then. Closing its connection there fails on the connection's
    # own closed loop, and asyncio logs the unretrieved task exception.
    records = TypeAdapter(list[_LoggedRecord]).validate_json(logged.stdout)
    assert [
        record.message
        for record in records
        if record.name == "asyncio" and record.levelno >= logging.ERROR
    ] == []


# Three complete_async() calls under one asyncio.gather, the fan-out the docs
# describe for it. gather starts all three before any answer comes back. The
# first is answered at once and closes its client while the other two wait.
_GATHER_CELL = """
import asyncio
import json

async def ask_async(prompt):
    response = await get_llm().complete_async(system="Be brief.", messages=[{"role": "user", "content": prompt}])
    return response.content

async def ask_side_by_side():
    return await asyncio.gather(ask_async("gather fast"), ask_async("gather held one"), ask_async("gather held two"))

print(json.dumps(asyncio.run(ask_side_by_side())))
"""

# Two web_search() calls on two threads, each under its own asyncio.run. The
# held call goes first, and it still waits for its answer when the fast call
# finishes and closes its client.
_THREADS_CELL = """
import json
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(2) as pool:
    answers = list(pool.map(web_search, ["thread held", "thread fast"]))
print(json.dumps(answers))
"""


@pytest.mark.slow
def test_side_by_side_calls_all_get_their_answers(tmp_path):
    """Each call closes only its own client, with ``aclose``. A call that ran
    ``close_live_providers`` instead would, as the fast call finishes, close
    the held calls' clients too, and they would fail without an answer."""
    held = frozenset({"gather held one", "gather held two", "thread held"})
    with _serving(held=held) as server:
        gathered, threaded = _run_cells(
            [_GATHER_CELL, _THREADS_CELL],
            tmp_path,
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
        )

    assert gathered.error is None, gathered.error
    assert threaded.error is None, threaded.error
    assert json.loads(gathered.stdout) == [
        "answer to: gather fast",
        "answer to: gather held one",
        "answer to: gather held two",
    ]
    assert json.loads(threaded.stdout) == [
        "answer to: thread held",
        "answer to: thread fast",
    ]
    seen = "\n".join(map(repr, server.seen)) + "\n" + gathered.logs + threaded.logs
    # Calls side by side arrive in any order, so the prompts are compared sorted.
    assert sorted(request.prompt for request in server.seen) == [
        "gather fast",
        "gather held one",
        "gather held two",
        "thread fast",
        "thread held",
    ], seen
    assert [request.retry_count for request in server.seen] == ["0"] * 5, seen
