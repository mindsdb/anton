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

Timed script, for slow or silent model calls:
    with StubServer() as stub:
        stub.script(streamed=["hold:3:done"], keepalive_s=0.5)
        # every streamed call: a role chunk, `: keepalive` comments every
        # 0.5 s for 3 s, then "done"

While a script is set it answers every request in place of the queue. The
streamed steps are used in order and the last one repeats:

    text:<s>                           answer <s> at once
    hold:<secs>[:<s>]                  role chunk, keepalives for <secs>, then <s>
                                       (default "done")
    slow_args:<secs>[:<tool>[:<json>]] a tool call whose argument JSON trickles
                                       out over <secs>
    silent_args:<secs>[:<tool>[:<json>]]
                                       the tool call's start, keepalives for
                                       <secs>, then every argument in one burst
    length_cut                         no content, finish_reason "length", and
                                       completion_tokens equal to the budget
    trickle:<secs>:<every>             one text delta every <every> s for <secs>

The default tool is `scratchpad` with `{"action":"view","name":"stub"}` when the
request offers it, else the first tool offered with its required fields filled.

Non-streamed requests answer at once. A forced `tool_choice` gets one call of
that tool (`_VerifierVerdict` answers COMPLETE); anything else gets the text
"ok", or the text of a `text:<s>` step in `nonstream`.

Run it on its own (from an anton checkout):

    uv run --group dev python -m tests.e2e.stub_server --port 8765 \\
        --streamed 'hold:360:done' --keepalive-s 15

Routes: `GET /health`, `GET /v1/models` (and `/v1/models/`),
`POST /v1/chat/completions`, `POST /_stub/script` with
`{"streamed": [...], "nonstream": [...], "keepalive_s": 15}`, and
`GET /_stub/requests`.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from queue import Empty, Queue
from typing import Any

STUB_MODEL_ID = "stub-model"

# A scratchpad call every required field of which is valid, so a scripted
# tool call runs without the model's help.
_DEFAULT_TOOL = "scratchpad"
_DEFAULT_TOOL_ARGS = {"action": "view", "name": "stub"}


@dataclass
class _Response:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    # None = honour request's `stream` flag; True/False = override
    force_streaming: bool | None = None
    # Reported as `usage.completion_tokens`. Set it equal to the request's
    # `max_tokens` to emulate a truncated response.
    #
    # This stub reports `finish_reason: "stop"` at the cap. That is NO LONGER
    # what the gateway does — ENG-1082 was fixed 2026-08-03 and it now returns
    # `"length"` on every alias. The stub keeps the old behaviour ON PURPOSE:
    # it is the harder case, and a recovery that survives a gateway which lies
    # also survives one that is honest. Do not "correct" this to `"length"` —
    # that deletes the only coverage of the case ENG-1042 was built for.
    output_tokens: int = 10


@dataclass
class _Script:
    """The timed answers that replace the queue while set."""

    streamed: list[str]
    nonstream: list[str]
    keepalive_s: float
    streamed_used: int = 0
    nonstream_used: int = 0

    def next_streamed(self) -> str:
        return _next_step(steps=self.streamed, used=self.streamed_used, default="text:ok")

    def next_nonstream(self) -> str:
        return _next_step(steps=self.nonstream, used=self.nonstream_used, default="auto")


def _next_step(*, steps: list[str], used: int, default: str) -> str:
    if not steps:
        return default
    return steps[min(used, len(steps) - 1)]


@dataclass(frozen=True)
class RequestRecord:
    """One chat request the stub served. `GET /_stub/requests` reports it as JSON."""

    path: str
    stream: bool
    budget: int | None
    tool_choice: Any
    at: float

    def to_json(self) -> dict:
        return {
            "path": self.path,
            "stream": self.stream,
            "budget": self.budget,
            "tool_choice": self.tool_choice,
            "at": self.at,
        }


@dataclass
class _Answer:
    """How one chat request is answered: a queued response, or a script step."""

    response: _Response | None = None
    step: str | None = None
    keepalive_s: float = 15.0


@dataclass
class _ScriptedCall:
    name: str
    args_json: str


class StubServer:
    """Thread-safe OpenAI-compatible stub server.

    Start with ``StubServer()`` as a context manager or call ``.start()``
    manually and ``.stop()`` when done.

    Response queue contract:
    - One response is consumed per incoming request.
    - Responses are returned in FIFO order.
    - If the queue is empty when a request arrives, a fallback empty-text
      response is returned so the subprocess does not hang.

    Requests are served on their own threads, so a held stream does not
    block the next request. When a response is already queued, the log append
    and the queue pop happen under one lock, so the Nth logged request gets the
    Nth queued response. Requests that arrive before their response is queued
    wait outside the lock and may be answered out of order.
    """

    def __init__(self, *, port: int = 0, bind: str = "127.0.0.1") -> None:
        self._queue: Queue[_Response] = Queue()
        self._log: list[dict] = []
        self._records: list[RequestRecord] = []
        self._script: _Script | None = None
        self._lock = threading.Lock()
        self._httpd: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._bind = bind
        self._requested_port = port
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

    def queue_verification_truncated(self, output_tokens: int = 2048) -> "StubServer":
        """Queue a verdict call that narrated instead of calling the tool and ran
        out of budget doing it (ENG-1081).

        Text, no tool call, and `completion_tokens` equal to the verifier's first
        budget — exactly what `mindshub_air`/`kimi`/`deepseek` return. The
        session should retry with the larger budget rather than treating this as
        a verdict.
        """
        self._queue.put(_Response(
            content="Let me analyze this conversation carefully. The user asked for",
            force_streaming=False,
            output_tokens=output_tokens,
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

    def script(
        self,
        *,
        streamed: list[str] | None = None,
        nonstream: list[str] | None = None,
        keepalive_s: float = 15.0,
    ) -> "StubServer":
        """Answer every request from these timed steps instead of the queue.

        See the module docstring for the step grammar. Raises ValueError on a
        step it cannot parse, so a typo fails at setup rather than mid-turn.
        """
        streamed = list(streamed or [])
        nonstream = list(nonstream or [])
        for step in streamed:
            _parse_streamed_step(step=step)
        for step in nonstream:
            _parse_nonstream_step(step=step)
        with self._lock:
            self._script = _Script(
                streamed=streamed, nonstream=nonstream, keepalive_s=float(keepalive_s)
            )
        return self

    def clear_script(self) -> "StubServer":
        """Go back to answering from the queue."""
        with self._lock:
            self._script = None
        return self

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self._bind in ("", "0.0.0.0") else self._bind
        return f"http://{host}:{self._port}/v1"

    @property
    def port(self) -> int:
        return self._port

    @property
    def requests(self) -> list[dict]:
        with self._lock:
            return list(self._log)

    @property
    def request_records(self) -> list[RequestRecord]:
        """Path, stream flag, budget, tool_choice and time of each chat request."""
        with self._lock:
            return list(self._records)

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
        httpd = ThreadingHTTPServer((self._bind, self._requested_port), self._make_handler())
        httpd.daemon_threads = True
        self._httpd = httpd
        self._port = httpd.server_address[1]
        self._thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()

    def __enter__(self) -> "StubServer":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    def _take(self, *, path: str, body: dict) -> _Answer:
        """Log one chat request and pick its answer, atomically."""
        streaming = bool(body.get("stream", False))
        with self._lock:
            self._log.append(body)
            self._records.append(RequestRecord(
                path=path,
                stream=streaming,
                budget=_budget(body=body),
                tool_choice=body.get("tool_choice"),
                at=time.time(),
            ))
            script = self._script
            if script is not None:
                if streaming:
                    step = script.next_streamed()
                    script.streamed_used += 1
                else:
                    step = script.next_nonstream()
                    script.nonstream_used += 1
                return _Answer(step=step, keepalive_s=script.keepalive_s)
            try:
                return _Answer(response=self._queue.get_nowait())
            except Empty:
                pass
        # Nothing queued yet: wait outside the lock, so the other routes and
        # other requests are not blocked while this one waits.
        try:
            return _Answer(response=self._queue.get(timeout=5))
        except Empty:
            return _Answer(response=_Response(
                content="[STUB: no response queued — returning empty]"
            ))

    def _make_handler(self) -> type:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            # HTTP/1.1 for chunked streams; queued answers carry Content-Length.
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/health":
                    _send_json_body(self, {"status": "ok"})
                elif path in ("/v1/models", "/v1/models/", "/models", "/models/"):
                    _send_json_body(self, {
                        "object": "list",
                        "data": [{"id": STUB_MODEL_ID, "object": "model",
                                  "created": 0, "owned_by": "stub"}],
                    })
                elif path == "/_stub/requests":
                    records = [r.to_json() for r in stub.request_records]
                    _send_json_body(self, {"requests": records})
                else:
                    _send_json_body(self, {"error": {"message": "not found"}}, status=404)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(length))
                except Exception:
                    body = {}
                if not isinstance(body, dict):
                    body = {}
                path = self.path.split("?", 1)[0]

                if path == "/_stub/script":
                    try:
                        stub.script(
                            streamed=body.get("streamed"),
                            nonstream=body.get("nonstream"),
                            keepalive_s=float(body.get("keepalive_s", 15.0)),
                        )
                    except (TypeError, ValueError) as exc:
                        _send_json_body(self, {"error": {"message": str(exc)}}, status=400)
                        return
                    _send_json_body(self, {"ok": True})
                    return

                answer = stub._take(path=path, body=body)
                resp = answer.response
                try:
                    if resp is not None:
                        streaming = resp.force_streaming
                        if streaming is None:
                            streaming = bool(body.get("stream", False))
                        if streaming:
                            _send_sse(self, resp)
                        else:
                            _send_json(self, resp)
                    elif body.get("stream"):
                        _send_scripted_stream(
                            self, step=answer.step, body=body, keepalive_s=answer.keepalive_s
                        )
                    else:
                        _send_scripted_json(self, step=answer.step, body=body)
                except (BrokenPipeError, ConnectionResetError):
                    # The client gave up on the call (a deadline, a cancel).
                    self.close_connection = True

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
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": resp.output_tokens,
            "total_tokens": 10 + resp.output_tokens,
        },
    }
    _send_json_body(handler, data)


def _send_json_body(handler: BaseHTTPRequestHandler, data: dict, *, status: int = 200) -> None:
    body = json.dumps(data).encode()
    handler.send_response(status)
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


# --------------------------------------------------------------------------- #
# Scripted answers
# --------------------------------------------------------------------------- #

@dataclass
class _StreamedStep:
    kind: str
    secs: float = 0.0
    every: float = 0.0
    text: str = ""
    tool: str | None = None
    args_json: str | None = None


def _parse_streamed_step(*, step: str) -> _StreamedStep:
    kind = step.split(":", 1)[0]
    if kind == "text":
        _, text = (step.split(":", 1) + [""])[:2]
        return _StreamedStep(kind=kind, text=text)
    if kind == "hold":
        parts = step.split(":", 2)
        if len(parts) < 2:
            raise ValueError(f"hold needs seconds: {step!r}")
        return _StreamedStep(
            kind=kind, secs=float(parts[1]), text=parts[2] if len(parts) > 2 else "done"
        )
    if kind in ("slow_args", "silent_args"):
        parts = step.split(":", 3)
        if len(parts) < 2:
            raise ValueError(f"{kind} needs seconds: {step!r}")
        args_json = parts[3] if len(parts) > 3 else None
        if args_json is not None:
            json.loads(args_json)
        return _StreamedStep(
            kind=kind,
            secs=float(parts[1]),
            tool=parts[2] if len(parts) > 2 and parts[2] else None,
            args_json=args_json,
        )
    if kind == "length_cut":
        return _StreamedStep(kind=kind)
    if kind == "trickle":
        parts = step.split(":")
        if len(parts) != 3:
            raise ValueError(f"trickle needs seconds and an interval: {step!r}")
        every = float(parts[2])
        if every <= 0:
            raise ValueError(f"trickle interval must be positive: {step!r}")
        return _StreamedStep(kind=kind, secs=float(parts[1]), every=every)
    raise ValueError(f"unknown streamed step: {step!r}")


def _parse_nonstream_step(*, step: str) -> str:
    if step == "auto" or step.startswith("text:"):
        return step
    raise ValueError(f"unknown non-streamed step: {step!r}")


def _budget(*, body: dict) -> int | None:
    budget = body.get("max_completion_tokens", body.get("max_tokens"))
    return budget if isinstance(budget, int) else None


def _offered_tools(*, body: dict) -> list[dict]:
    out = []
    for tool in body.get("tools") or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            out.append(fn)
    return out


def _default_for(*, schema: Any) -> Any:
    if not isinstance(schema, dict):
        return "stub"
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    for key in ("anyOf", "oneOf"):
        if schema.get(key):
            return _default_for(schema=schema[key][0])
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    if kind == "integer" or kind == "number":
        return 0
    if kind == "boolean":
        return False
    if kind == "array":
        return []
    if kind == "object":
        return _required_args(schema=schema)
    return "stub"


def _required_args(*, schema: dict) -> dict:
    props = schema.get("properties") or {}
    return {name: _default_for(schema=props.get(name)) for name in schema.get("required") or []}


def _tool_call_for(*, body: dict, name: str | None, args_json: str | None) -> _ScriptedCall:
    """The tool call a scripted step makes."""
    offered = _offered_tools(body=body)
    if name is None:
        if any(t["name"] == _DEFAULT_TOOL for t in offered) or not offered:
            name = _DEFAULT_TOOL
        else:
            name = offered[0]["name"]
    if args_json is not None:
        return _ScriptedCall(name=name, args_json=args_json)
    if name == _DEFAULT_TOOL:
        return _ScriptedCall(name=name, args_json=json.dumps(_DEFAULT_TOOL_ARGS))
    if name == "_VerifierVerdict":
        return _ScriptedCall(
            name=name, args_json=json.dumps({"status": "COMPLETE", "reason": "stub"})
        )
    fn = next((t for t in offered if t["name"] == name), None)
    return _ScriptedCall(
        name=name,
        args_json=json.dumps(_required_args(schema=fn.get("parameters") or {}) if fn else {}),
    )


def _send_scripted_json(handler: BaseHTTPRequestHandler, *, step: str, body: dict) -> None:
    choice = body.get("tool_choice")
    forced = None
    if isinstance(choice, dict):
        forced = (choice.get("function") or {}).get("name") or choice.get("name")
    message: dict = {"role": "assistant", "content": None}
    finish_reason = "stop"
    if forced:
        call = _tool_call_for(body=body, name=forced, args_json=None)
        message["tool_calls"] = [{
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": call.name, "arguments": call.args_json},
        }]
        finish_reason = "tool_calls"
    else:
        message["content"] = step.split(":", 1)[1] if step.startswith("text:") else "ok"
    _send_json_body(handler, {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": STUB_MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    })


class _ChunkedStream:
    """Writes SSE frames with chunked transfer, flushing after each one."""

    def __init__(self, handler: BaseHTTPRequestHandler, *, model: str) -> None:
        self._handler = handler
        self.rid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        self.model = model
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()

    def _write(self, text: str) -> None:
        data = text.encode()
        self._handler.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self._handler.wfile.flush()

    def data(self, payload: dict | str) -> None:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        self._write(f"data: {text}\n\n")

    def delta(self, delta: dict, *, finish_reason: str | None = None) -> None:
        chunk = _chunk(self.rid, delta, finish_reason=finish_reason)
        chunk["model"] = self.model
        self.data(chunk)

    def keepalive(self) -> None:
        self._write(": keepalive\n\n")

    def hold(self, *, secs: float, keepalive_s: float) -> None:
        """Send nothing but keepalive comments for ``secs``."""
        deadline = time.monotonic() + secs
        next_keepalive = time.monotonic() + keepalive_s if keepalive_s > 0 else None
        while True:
            now = time.monotonic()
            if now >= deadline:
                return
            wake = deadline if next_keepalive is None else min(deadline, next_keepalive)
            time.sleep(max(0.0, wake - now))
            if next_keepalive is not None and time.monotonic() >= next_keepalive:
                self.keepalive()
                next_keepalive += keepalive_s

    def finish(self, *, body: dict, completion_tokens: int) -> None:
        options = body.get("stream_options") or {}
        if options.get("include_usage"):
            self.data({
                "id": self.rid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model,
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": completion_tokens,
                    "total_tokens": 10 + completion_tokens,
                },
            })
        self.data("[DONE]")
        self._handler.wfile.write(b"0\r\n\r\n")
        self._handler.wfile.flush()


def _send_scripted_stream(
    handler: BaseHTTPRequestHandler, *, step: str, body: dict, keepalive_s: float
) -> None:
    parsed = _parse_streamed_step(step=step)
    out = _ChunkedStream(handler, model=STUB_MODEL_ID)
    budget = _budget(body=body) or 0

    if parsed.kind in ("text", "hold"):
        out.delta({"role": "assistant", "content": ""})
        if parsed.kind == "hold":
            out.hold(secs=parsed.secs, keepalive_s=keepalive_s)
        if parsed.text:
            out.delta({"content": parsed.text})
        out.delta({}, finish_reason="stop")
        out.finish(body=body, completion_tokens=10)
        return

    if parsed.kind == "trickle":
        out.delta({"role": "assistant", "content": ""})
        sent = 0
        deadline = time.monotonic() + parsed.secs
        while time.monotonic() < deadline:
            time.sleep(parsed.every)
            out.delta({"content": f"tick {sent} "})
            sent += 1
        out.delta({}, finish_reason="stop")
        out.finish(body=body, completion_tokens=max(1, sent))
        return

    if parsed.kind == "length_cut":
        out.delta({"role": "assistant", "content": ""})
        out.delta({}, finish_reason="length")
        out.finish(body=body, completion_tokens=budget)
        return

    # slow_args / silent_args: one tool call.
    call = _tool_call_for(body=body, name=parsed.tool, args_json=parsed.args_json)
    args = call.args_json
    out.delta({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "index": 0,
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": call.name, "arguments": ""},
        }],
    })
    if parsed.kind == "silent_args":
        out.hold(secs=parsed.secs, keepalive_s=keepalive_s)
        out.delta({"tool_calls": [{"index": 0, "function": {"arguments": args}}]})
    else:
        # Up to one piece per 50 ms, at most one per character, spread evenly
        # over the window.
        size = -(-len(args) // max(1, min(len(args), int(parsed.secs / 0.05) or 1)))
        pieces = [args[i:i + size] for i in range(0, len(args), size)] or [args]
        gap = parsed.secs / len(pieces)
        for piece in pieces:
            time.sleep(gap)
            out.delta({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]})
    out.delta({}, finish_reason="tool_calls")
    out.finish(body=body, completion_tokens=10)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.e2e.stub_server",
        description="OpenAI-compatible stub endpoint with timed, scripted answers.",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument(
        "--streamed", action="append", default=[],
        help="a streamed step; repeat for a sequence, the last one repeats",
    )
    parser.add_argument(
        "--nonstream", action="append", default=[],
        help="a non-streamed step (auto or text:<s>); repeat for a sequence",
    )
    parser.add_argument("--keepalive-s", type=float, default=15.0)
    args = parser.parse_args(argv)

    stub = StubServer(port=args.port, bind=args.bind)
    stub.script(streamed=args.streamed, nonstream=args.nonstream, keepalive_s=args.keepalive_s)
    stub.start()
    print(f"stub listening on {stub.base_url} (model {STUB_MODEL_ID})", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        stub.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
