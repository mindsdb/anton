"""
Minimal OpenAI-compatible stub LLM server for E2E scenario testing.

Speaks the OpenAI chat completions API (streaming SSE + non-streaming JSON).
Queue scripted responses before running a scenario; the stub pops them in order.

Usage:
    with StubServer() as stub:
        stub.queue_text("Hello!")
        stub.queue_verification_ok()
        # ... run anton subprocess against stub.base_url ...
        assert stub.request_count == 2
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Empty, Queue
from typing import Any

@dataclass
class _Response:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    # None = honour request's `stream` flag; True/False = override
    force_streaming: bool | None = None


class StubServer:
    """Thread-safe OpenAI-compatible stub server.

    Start with ``StubServer()`` as a context manager or call ``.start()``
    manually and ``.stop()`` when done.

    Response queue contract:
    - One response is consumed per incoming request.
    - Responses are returned in FIFO order.
    - If the queue is empty when a request arrives, a fallback empty-text
      response is returned so the subprocess does not hang.
    """

    def __init__(self) -> None:
        self._queue: Queue[_Response] = Queue()
        self._log: list[dict] = []
        self._lock = threading.Lock()
        self._httpd: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._port: int = 0


    def queue_text(self, text: str) -> "StubServer":
        """Queue a streaming text-only response (main turn)."""
        self._queue.put(_Response(content=text))
        return self

    def queue_tool_call(self, name: str, arguments: dict) -> "StubServer":
        """Queue a streaming response that calls one tool."""
        self._queue.put(_Response(tool_calls=[{
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "name": name,
            "arguments": arguments,
        }]))
        return self

    def _queue_verdict(self, status: str, reason: str) -> "StubServer":
        """Queue a non-streaming structured verifier verdict.

        The completion verifier now runs via ``generate_object_code`` with a
        forced tool_choice on the ``_VerifierVerdict`` schema, so the stub must
        answer with a tool call (status + reason), not free text (ENG-716).
        """
        self._queue.put(_Response(
            tool_calls=[{
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "name": "_VerifierVerdict",
                "arguments": {"status": status, "reason": reason},
            }],
            force_streaming=False,
        ))
        return self

    def queue_verification_ok(self) -> "StubServer":
        """Queue a COMPLETE verifier verdict."""
        return self._queue_verdict("COMPLETE", "task is done.")

    def queue_verification_incomplete(self, reason: str = "still more to do") -> "StubServer":
        return self._queue_verdict("INCOMPLETE", reason)

    def queue_verification_waiting(self, reason: str = "waiting on the user") -> "StubServer":
        return self._queue_verdict("WAITING", reason)

    def queue_verification_stuck(self, reason: str = "blocked") -> "StubServer":
        return self._queue_verdict("STUCK", reason)

    def queue_summary(self, text: str = "Summary of earlier turns.") -> "StubServer":
        """Queue a response for _summarize_history's coding model call."""
        self._queue.put(_Response(content=text, force_streaming=False))
        return self

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}/v1"

    @property
    def requests(self) -> list[dict]:
        with self._lock:
            return list(self._log)

    @property
    def request_count(self) -> int:
        with self._lock:
            return len(self._log)

    def system_prompts(self) -> list[str]:
        """Extract system prompts from logged requests (first 'system' role message)."""
        out = []
        for req in self.requests:
            for msg in req.get("messages", []):
                if msg.get("role") == "system":
                    out.append(msg.get("content", ""))
                    break
        return out


    def start(self) -> "StubServer":
        httpd = HTTPServer(("127.0.0.1", 0), self._make_handler())
        self._httpd = httpd
        self._port = httpd.server_address[1]
        self._thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()

    def __enter__(self) -> "StubServer":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    def _make_handler(self) -> type:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(length))
                except Exception:
                    body = {}

                with stub._lock:
                    stub._log.append(body)

                try:
                    resp = stub._queue.get(timeout=5)
                except Empty:
                    resp = _Response(content="[STUB: no response queued — returning empty]")

                streaming = resp.force_streaming
                if streaming is None:
                    streaming = bool(body.get("stream", False))

                if streaming:
                    _send_sse(self, resp)
                else:
                    _send_json(self, resp)

            def log_message(self, *_: Any) -> None:
                pass  # suppress server access logs

        return Handler


def _send_sse(handler: BaseHTTPRequestHandler, resp: _Response) -> None:
    rid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    chunks: list[dict] = []

    if resp.tool_calls:
        tc = resp.tool_calls[0]
        args_str = json.dumps(tc["arguments"])
        # Chunk 1: announce tool call with id + name (must be together for the provider to emit
        # StreamToolUseStart)
        chunks.append(_chunk(rid, {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "index": 0,
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": ""},
            }],
        }, finish_reason=None))
        # Chunk 2: full arguments in one delta (provider accumulates with "".join)
        chunks.append(_chunk(rid, {
            "tool_calls": [{"index": 0, "function": {"arguments": args_str}}],
        }, finish_reason=None))
        # Final
        chunks.append(_chunk(rid, {}, finish_reason="tool_calls"))
    else:
        if resp.content:
            chunks.append(_chunk(rid, {"role": "assistant", "content": resp.content},
                                 finish_reason=None))
        chunks.append(_chunk(rid, {}, finish_reason="stop"))

    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    body += "data: [DONE]\n\n"
    body_bytes = body.encode()

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Content-Length", str(len(body_bytes)))
    handler.end_headers()
    handler.wfile.write(body_bytes)


def _send_json(handler: BaseHTTPRequestHandler, resp: _Response) -> None:
    message: dict = {"role": "assistant", "content": resp.content or None}
    finish_reason = "stop"
    if resp.tool_calls:
        tc = resp.tool_calls[0]
        message["tool_calls"] = [{
            "id": tc["id"],
            "type": "function",
            "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
        }]
        finish_reason = "tool_calls"
    data = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gpt-test",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }
    body = json.dumps(data).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _chunk(rid: str, delta: dict, *, finish_reason: str | None) -> dict:
    return {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "gpt-test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
