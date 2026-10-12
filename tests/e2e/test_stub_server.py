"""The stub endpoint's contract: routes, timed steps and the queue.

Tests and the manual How to test steps both script it, so a change to a route
or a step's shape must fail here rather than in a scenario that happens to
use it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tests.e2e.stub_server import STUB_MODEL_ID, StubServer

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def stub():
    with StubServer() as server:
        yield server


def _origin(stub: StubServer) -> str:
    return stub.base_url[: -len("/v1")]


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def _post(url: str, body: dict) -> urllib.request.addinfourl:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=10)


def _chat(stub: StubServer, **body) -> dict:
    body.setdefault("model", STUB_MODEL_ID)
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    with _post(f"{stub.base_url}/chat/completions", body) as resp:
        return json.loads(resp.read())


class _Frames:
    """A streamed answer, split into data frames and keepalive comments."""

    def __init__(self, raw: str, elapsed: float) -> None:
        self.elapsed = elapsed
        self.keepalives = raw.count(": keepalive")
        self.data = [
            line[len("data: "):]
            for line in raw.splitlines()
            if line.startswith("data: ")
        ]

    @property
    def chunks(self) -> list[dict]:
        return [json.loads(d) for d in self.data if d != "[DONE]"]

    def content(self) -> str:
        return "".join(
            (c["choices"][0]["delta"].get("content") or "")
            for c in self.chunks if c["choices"]
        )

    def args(self) -> str:
        out = ""
        for c in self.chunks:
            for tc in (c["choices"][0]["delta"].get("tool_calls") or []) if c["choices"] else []:
                out += tc.get("function", {}).get("arguments") or ""
        return out

    def finish_reason(self) -> str | None:
        reasons = [c["choices"][0]["finish_reason"] for c in self.chunks if c["choices"]]
        return next((r for r in reversed(reasons) if r), None)

    def usage(self) -> dict | None:
        return next((c["usage"] for c in self.chunks if c.get("usage")), None)


def _stream(stub: StubServer, **body) -> _Frames:
    body.update(stream=True, stream_options={"include_usage": True})
    body.setdefault("model", STUB_MODEL_ID)
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    started = time.monotonic()
    with _post(f"{stub.base_url}/chat/completions", body) as resp:
        assert resp.headers.get("Transfer-Encoding") == "chunked"
        raw = resp.read().decode()
    return _Frames(raw, time.monotonic() - started)


_SCRATCHPAD = {"type": "function", "function": {
    "name": "scratchpad",
    "parameters": {"type": "object", "properties": {"action": {"type": "string"}}},
}}


def test_health_and_both_model_routes(stub):
    assert _get(f"{_origin(stub)}/health") == {"status": "ok"}
    for path in ("/v1/models", "/v1/models/"):
        listing = _get(f"{_origin(stub)}{path}")
        assert listing["object"] == "list"
        assert listing["data"][0]["id"] == STUB_MODEL_ID


def test_the_queue_still_answers_in_order(stub):
    stub.queue_summary("first").queue_summary("second")

    assert _chat(stub)["choices"][0]["message"]["content"] == "first"
    assert _chat(stub)["choices"][0]["message"]["content"] == "second"
    assert stub.request_count == 2


def test_a_queued_cut_still_reports_stop(stub):
    """The queue keeps reporting `stop` at the cap, the harder case."""
    stub.queue_verification_truncated(output_tokens=2048)

    answer = _chat(stub)

    assert answer["choices"][0]["finish_reason"] == "stop"
    assert answer["usage"]["completion_tokens"] == 2048


def test_text_answers_at_once_with_usage_and_done(stub):
    stub.script(streamed=["text:hello"])

    frames = _stream(stub)

    assert frames.content() == "hello"
    assert frames.finish_reason() == "stop"
    assert frames.usage()["completion_tokens"] > 0
    assert frames.data[-1] == "[DONE]"


def test_hold_sends_only_keepalives_then_answers(stub):
    stub.script(streamed=["hold:0.5:later"], keepalive_s=0.1)

    frames = _stream(stub)

    assert frames.elapsed >= 0.5
    assert frames.keepalives >= 3
    assert frames.content() == "later"


def test_slow_args_trickle_the_default_scratchpad_call(stub):
    stub.script(streamed=["slow_args:0.4"])

    frames = _stream(stub, tools=[_SCRATCHPAD])

    first = frames.chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert first["function"]["name"] == "scratchpad"
    assert first["id"]
    assert json.loads(frames.args()) == {"action": "view", "name": "stub"}
    arg_chunks = [
        c for c in frames.chunks[1:]
        if c["choices"] and c["choices"][0]["delta"].get("tool_calls")
    ]
    assert len(arg_chunks) > 1, "slow_args must trickle, not burst"
    assert frames.finish_reason() == "tool_calls"
    assert frames.elapsed >= 0.4


def test_silent_args_burst_after_keepalives(stub):
    stub.script(streamed=['silent_args:0.4:lookup:{"q": "x"}'], keepalive_s=0.1)

    frames = _stream(stub)

    assert frames.keepalives >= 2
    arg_chunks = [
        c for c in frames.chunks[1:]
        if c["choices"] and c["choices"][0]["delta"].get("tool_calls")
    ]
    assert len(arg_chunks) == 1, "silent_args sends every argument in one burst"
    assert json.loads(frames.args()) == {"q": "x"}


def test_the_first_offered_tool_is_used_when_scratchpad_is_absent(stub):
    stub.script(streamed=["slow_args:0.1"])
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {
        "type": "object",
        "properties": {"q": {"type": "string"}, "n": {"type": "integer"}},
        "required": ["q", "n"],
    }}}

    frames = _stream(stub, tools=[tool])

    assert frames.chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "lookup"
    assert json.loads(frames.args()) == {"q": "stub", "n": 0}


def test_length_cut_echoes_the_budget(stub):
    stub.script(streamed=["length_cut", "text:retried"])

    cut = _stream(stub, max_completion_tokens=8192)
    retry = _stream(stub, max_completion_tokens=16384)

    assert cut.finish_reason() == "length"
    assert cut.content() == ""
    assert cut.usage()["completion_tokens"] == 8192
    assert retry.content() == "retried"
    budgets = [r.budget for r in stub.request_records]
    assert budgets == [8192, 16384]


def test_the_last_step_repeats(stub):
    stub.script(streamed=["text:one", "text:two"])

    assert [_stream(stub).content() for _ in range(3)] == ["one", "two", "two"]


def test_trickle_sends_a_delta_per_interval(stub):
    stub.script(streamed=["trickle:0.5:0.1"])

    frames = _stream(stub)

    deltas = [c for c in frames.chunks if c["choices"] and c["choices"][0]["delta"].get("content")]
    assert len(deltas) >= 4
    assert frames.finish_reason() == "stop"


def test_a_forced_verdict_answers_complete(stub):
    stub.script(streamed=["hold:100:never"])
    verdict_tool = {"type": "function", "function": {"name": "_VerifierVerdict", "parameters": {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["COMPLETE", "INCOMPLETE"]},
                       "reason": {"type": "string"}},
        "required": ["status", "reason"],
    }}}
    started = time.monotonic()

    answer = _chat(
        stub, tools=[verdict_tool],
        tool_choice={"type": "function", "function": {"name": "_VerifierVerdict"}},
    )

    assert time.monotonic() - started < 2, "non-streamed requests answer at once"
    call = answer["choices"][0]["message"]["tool_calls"][0]["function"]
    assert call["name"] == "_VerifierVerdict"
    assert json.loads(call["arguments"]) == {"status": "COMPLETE", "reason": "stub"}
    assert answer["usage"]["total_tokens"] > 0


def test_other_forced_tools_get_their_required_fields(stub):
    stub.script()
    tool = {"type": "function", "function": {"name": "_Summary", "parameters": {
        "type": "object",
        "properties": {"facts": {"type": "array"}, "ok": {"type": "boolean"},
                       "kind": {"enum": ["a", "b"]}},
        "required": ["facts", "ok", "kind"],
    }}}

    answer = _chat(stub, tools=[tool], tool_choice={"type": "function", "function": {"name": "_Summary"}})

    args = json.loads(answer["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {"facts": [], "ok": False, "kind": "a"}


def test_a_request_without_tools_gets_ok_or_its_scripted_text(stub):
    stub.script(nonstream=["text:scripted"])
    assert _chat(stub)["choices"][0]["message"]["content"] == "scripted"

    stub.script()
    assert _chat(stub)["choices"][0]["message"]["content"] == "ok"


def test_the_script_route_and_the_request_log(stub):
    with _post(f"{_origin(stub)}/_stub/script", {"streamed": ["text:via-http"], "keepalive_s": 1}) as resp:
        assert json.loads(resp.read()) == {"ok": True}

    assert _stream(stub, max_tokens=99).content() == "via-http"

    log = _get(f"{_origin(stub)}/_stub/requests")["requests"]
    assert len(log) == 1
    assert log[0]["path"] == "/v1/chat/completions"
    assert log[0]["stream"] is True
    assert log[0]["budget"] == 99
    assert log[0]["tool_choice"] is None
    assert isinstance(log[0]["at"], float)


def test_a_bad_step_is_refused_at_setup(stub):
    with pytest.raises(ValueError):
        stub.script(streamed=["sleep:5"])
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(f"{_origin(stub)}/_stub/script", {"streamed": ["hold"]})
    assert err.value.code == 400


def test_the_cli_serves_its_script():
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.e2e.stub_server", "--port", "0",
         "--streamed", "text:from-cli", "--keepalive-s", "1"],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        line = proc.stdout.readline()
        assert "stub listening on http://127.0.0.1:" in line, line
        base = line.split()[3]
        assert _get(f"{base[: -len('/v1')]}/health") == {"status": "ok"}
        body = {"model": STUB_MODEL_ID, "messages": [], "stream": True}
        with _post(f"{base}/chat/completions", body) as resp:
            assert "from-cli" in resp.read().decode()
    finally:
        proc.terminate()
        proc.wait(timeout=5)
