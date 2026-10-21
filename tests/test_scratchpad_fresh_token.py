"""A pad's model helpers send the host's current credential, not the one it booted with.

A signed-in desktop user's MindsHub access token lives ten minutes, and the
desktop replaces it well before then. The main-process provider reads the
replacement through its key supplier on every request. The pad is a separate
process, so the supplier cannot reach it; these tests run a real pad against
a stub gateway that stops accepting the boot-time token and check the bearer
each later request carries, between cells and during one long cell.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from anton.core.backends.local import LocalScratchpadRuntime

pytestmark = pytest.mark.slow


class _Gateway(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.accepted = "token-a"
        self.bearers: list[str | None] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/v1"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Gateway

    def log_message(self, format: str, *args: object) -> None:
        """Silent: the test reads ``server.bearers`` instead."""

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        auth = self.headers.get("Authorization") or ""
        bearer = auth.removeprefix("Bearer ") if auth.startswith("Bearer ") else None
        self.server.bearers.append(bearer)
        if bearer != self.server.accepted:
            self._send(
                401,
                {
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "Invalid or missing credentials.",
                    },
                },
            )
            return
        # The provider picks the Responses API or chat completions by host;
        # the test only cares which bearer arrived, so both get an answer.
        items = body.get("messages") or body.get("input") or []
        prompt = next(
            (i["content"] for i in items if i.get("role") == "user" and isinstance(i.get("content"), str)),
            "",
        )
        text = f"answer to: {prompt}"
        usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        if self.path.endswith("/chat/completions"):
            answer: dict[str, Any] = {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        else:
            answer = {
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": body["model"],
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
                "usage": usage,
            }
        self._send(200, answer)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@contextmanager
def _serving() -> Iterator[_Gateway]:
    server = _Gateway()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


class _Credential:
    """What the host's key supplier returns: the test swaps it like a desktop refresh."""

    def __init__(self) -> None:
        self.value = "token-a"

    async def supply(self) -> str:
        return self.value


# Binds ask(prompt) and ask_async(prompt) in the pad's namespace.
_HELPERS_CELL = """
llm = get_llm()
def ask(prompt):
    return llm.complete(system="Be brief.", messages=[{"role": "user", "content": prompt}]).content
async def ask_async(prompt):
    return (await llm.complete_async(system="Be brief.", messages=[{"role": "user", "content": prompt}])).content
"""


def _pad(gateway: _Gateway, credential: _Credential) -> LocalScratchpadRuntime:
    return LocalScratchpadRuntime(
        name="fresh-token",
        coding_provider="openai",
        coding_model="stub-model",
        coding_api_key=credential.value,
        coding_base_url=gateway.url,
        coding_api_key_supplier=credential.supply,
    )


async def test_next_cell_uses_the_renewed_credential():
    """A cell started after the host's credential changed sends the new one."""
    credential = _Credential()
    with _serving() as gateway:
        pad = _pad(gateway, credential)
        await pad.start()
        try:
            first = await pad.execute(_HELPERS_CELL + "print(ask('one'))\n")
            assert first.error is None, first.error
            assert gateway.bearers == ["token-a"]

            credential.value = gateway.accepted = "token-b"
            second = await pad.execute(
                "import asyncio\n"
                "print(ask('two'))\n"
                "async def four():\n"
                "    return await asyncio.gather(*(ask_async(f'n{i}') for i in range(4)))\n"
                "print(len(asyncio.run(four())))\n"
            )
            assert second.error is None, second.error
            assert second.stdout.splitlines()[-1] == "4"
            assert gateway.bearers[1:] == ["token-b"] * 5
        finally:
            await pad.close()


async def test_running_cell_picks_up_the_renewed_credential(monkeypatch):
    """A credential renewed while a cell runs reaches that cell's later calls."""
    # The pad inherits the host's environment; a fast liveness heartbeat is
    # the signal the host refreshes the credential on.
    monkeypatch.setenv("ANTON_SCRATCHPAD_HEARTBEAT_INTERVAL", "0.1")
    credential = _Credential()
    with _serving() as gateway:
        pad = _pad(gateway, credential)
        await pad.start()
        try:
            running = asyncio.ensure_future(
                pad.execute(
                    _HELPERS_CELL + "import time\n"
                    "print(ask('before'))\n"
                    "time.sleep(1.5)\n"
                    "print(ask('after'))\n"
                )
            )
            await asyncio.sleep(0.5)
            assert gateway.bearers == ["token-a"]
            credential.value = gateway.accepted = "token-b"
            cell = await running
            assert cell.error is None, cell.error
            assert gateway.bearers == ["token-a", "token-b"]
        finally:
            await pad.close()
