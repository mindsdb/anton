"""A structured call whose forced tool choice the model refuses goes again without it.

Claude Opus 5.5, Sonnet 5.5 and Fable 5.1 answer a forced ``tool_choice`` with a
400, and an older Claude refuses one while thinking is on. These tests drive
the real providers' SDK clients through ``httpx.MockTransport``, as
``test_bad_request_reaches_the_classifier.py`` does, because the retry keys on
the SDK's own error and the request each attempt sends: a mock that raised a
hand-built error, or returned a canned response, would model neither.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anthropic
import httpx2 as httpx
import openai
import pytest
from pydantic import BaseModel

from anton.core.llm import structured
from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.client import LLMClient
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import RequestRefusedError, StructuredOutputError
from anton.core.llm.structured import no_preamble_instruction
from tests.conftest import make_mock_llm
from tests.test_scratchpad_llm_loop import _run_cells


class Answer(BaseModel):
    answer: str


# Anthropic's 400 for a forced choice on Claude Opus 5.5, Sonnet 5.5, Fable 5.1
# and Mythos 5.1 (platform.claude.com, "Forced tool use not supported").
CLAUDE_5_5_REFUSAL = 'tool_choice: type "tool" and "any" are not supported for this model.'
# An older Claude's 400 for a forced choice with thinking on.
THINKING_REFUSAL = "Thinking may not be enabled when tool_choice forces tool use."
REFUSALS = [CLAUDE_5_5_REFUSAL, THINKING_REFUSAL]

CHAT = "chat.completions"
RESPONSES = "responses"
MESSAGES = "messages"
APIS = [CHAT, RESPONSES, MESSAGES]

BASE_URL = "https://models.example.test/v1"
USER = [{"role": "user", "content": "Which parts run short next week?"}]
CALL_FIRST = no_preamble_instruction(Answer)


@pytest.fixture(autouse=True)
def _no_remembered_refusals(monkeypatch):
    # Process-wide by design, so each test starts with none.
    monkeypatch.setattr(structured, "_FORCED_CHOICE_REFUSED", set())


@dataclass(frozen=True)
class Sent:
    """One request as the endpoint received it."""

    tool_choice: object
    system: str
    effort: str | None


@dataclass
class Endpoint:
    """Answers each request with ``answer(request_number, body)``: a status and a body."""

    api: str
    answer: Callable[[int, dict], tuple[int, dict]]
    sent: list[Sent] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.sent.append(Sent(tool_choice=body.get("tool_choice"), system=_system_of(self.api, body),
                              effort=_effort_of(self.api, body)))
        status, payload = self.answer(len(self.sent), body)
        return httpx.Response(status, json=payload)


def _system_of(api: str, body: dict) -> str:
    if api == CHAT:
        return body["messages"][0]["content"]
    if api == RESPONSES:
        return body["instructions"]
    return body["system"]


def _effort_of(api: str, body: dict) -> str | None:
    if api == CHAT:
        return body.get("reasoning_effort")
    if api == RESPONSES:
        return (body.get("reasoning") or {}).get("effort")
    return (body.get("output_config") or {}).get("effort")


def _provider(endpoint: Endpoint, *, effort: str | None = "medium"):
    """A real provider for ``endpoint.api`` whose SDK client sends to ``endpoint``."""
    transport = httpx.AsyncClient(transport=httpx.MockTransport(endpoint.handle))
    if endpoint.api == MESSAGES:
        provider = AnthropicProvider(api_key="test-key", reasoning_effort=effort)
        provider._client = anthropic.AsyncAnthropic(api_key="test-key", max_retries=0, http_client=transport)
        return provider
    flavor = OpenAIProvider.FLAVOR_OPENAI if endpoint.api == RESPONSES else OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC
    provider = OpenAIProvider(api_key="test-key", base_url=BASE_URL, flavor=flavor, reasoning_effort=effort)
    provider._client = openai.AsyncOpenAI(api_key="test-key", base_url=BASE_URL, max_retries=0, http_client=transport)
    return provider


def _client(provider, model: str = "claude-opus-5-5") -> LLMClient:
    return LLMClient(planning_provider=provider, planning_model=model, coding_provider=provider, coding_model=model)


def _refusal(api: str, message: str) -> tuple[int, dict]:
    error = {"type": "invalid_request_error", "message": message}
    return 400, ({"type": "error", "error": error} if api == MESSAGES else {"error": error})


def _tool_call(api: str, name: str, arguments: dict) -> tuple[int, dict]:
    if api == CHAT:
        message = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        ]}
        return 200, _chat(message, finish="tool_calls")
    if api == RESPONSES:
        item = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": name,
                "arguments": json.dumps(arguments), "status": "completed"}
        return 200, _response([item])
    return 200, _message([{"type": "tool_use", "id": "toolu_1", "name": name, "input": arguments}], stop="tool_use")


def _text(api: str, text: str, *, cut_off: bool = False) -> tuple[int, dict]:
    if api == CHAT:
        return 200, _chat({"role": "assistant", "content": text}, finish="length" if cut_off else "stop")
    if api == RESPONSES:
        item = {"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}]}
        return 200, _response([item], incomplete=cut_off)
    return 200, _message([{"type": "text", "text": text}], stop="max_tokens" if cut_off else "end_turn")


def _chat(message: dict, *, finish: str) -> dict:
    return {"id": "c1", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _response(output: list[dict], *, incomplete: bool = False) -> dict:
    return {"id": "resp_1", "object": "response", "created_at": 1, "model": "m",
            "status": "incomplete" if incomplete else "completed", "error": None,
            "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None, "output": output,
            "usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 1,
                      "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2}}


def _message(content: list[dict], *, stop: str) -> dict:
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "m", "content": content,
            "stop_reason": stop, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}


def _refuse_forced(api: str, message: str = CLAUDE_5_5_REFUSAL):
    """Refuses every forced call with ``message``, and answers every other with the tool."""
    def answer(_n: int, body: dict) -> tuple[int, dict]:
        if body.get("tool_choice"):
            return _refusal(api, message)
        return _tool_call(api, "Answer", {"answer": "42"})
    return answer


@pytest.mark.parametrize("message", REFUSALS)
@pytest.mark.parametrize("api", APIS)
async def test_a_refused_forced_choice_is_sent_again_without_it(api, message):
    endpoint = Endpoint(api, _refuse_forced(api, message))
    client = _client(_provider(endpoint))
    result = await client.generate_object(Answer, system="Decide.", messages=USER)

    assert result == Answer(answer="42")
    assert [sent.tool_choice is not None for sent in endpoint.sent] == [True, False]
    # No choice at all: a proxy that checks requests can refuse "auto" on chat completions.
    assert endpoint.sent[1].tool_choice is None
    assert endpoint.sent[1].system == "Decide." + CALL_FIRST
    assert [sent.effort for sent in endpoint.sent] == ["medium", "medium"]


@pytest.mark.parametrize("api", APIS)
async def test_an_unrelated_400_raises_as_today(api):
    endpoint = Endpoint(api, lambda _n, _body: _refusal(api, "max_tokens: must be greater than 0"))
    client = _client(_provider(endpoint))
    sdk_error = anthropic.BadRequestError if api == MESSAGES else openai.BadRequestError
    with pytest.raises(sdk_error):
        await client.generate_object(Answer, system="Decide.", messages=USER)
    assert len(endpoint.sent) == 1


async def test_the_reasoning_effort_refusal_still_ends_the_call():
    """The 400 OpenAI sends for function tools with an effort on chat completions
    names `reasoning_effort`, not `tool_choice`, and keeps its own error."""
    body = {"error": {"type": "invalid_request_error", "param": "reasoning_effort", "message": (
        "Function tools with reasoning_effort are not supported for gpt-5.6-luna in /v1/chat/completions. "
        "To use function tools, use /v1/responses or set reasoning_effort to 'none'."
    )}}
    endpoint = Endpoint(CHAT, lambda _n, _body: (400, body))
    with pytest.raises(RequestRefusedError):
        await _client(_provider(endpoint)).generate_object(Answer, system="Decide.", messages=USER)
    assert len(endpoint.sent) == 1


@pytest.mark.parametrize("api", APIS)
async def test_later_calls_to_that_endpoint_and_model_skip_the_forced_choice(api):
    endpoint = Endpoint(api, _refuse_forced(api))
    # A provider per call, as Cowork builds one per turn.
    await _client(_provider(endpoint)).generate_object(Answer, system="Decide.", messages=USER)
    await _client(_provider(endpoint)).generate_object(Answer, system="Decide.", messages=USER)
    assert [sent.tool_choice is not None for sent in endpoint.sent] == [True, False, False]

    # Another model on the same endpoint is still forced first.
    await _client(_provider(endpoint), model="claude-opus-5").generate_object(Answer, system="Decide.", messages=USER)
    assert endpoint.sent[3].tool_choice is not None


@pytest.mark.parametrize("api", APIS)
async def test_a_retry_with_no_tool_call_gets_one_more_try_then_raises(api):
    def answer(_n: int, body: dict) -> tuple[int, dict]:
        return _refusal(api, CLAUDE_5_5_REFUSAL) if body.get("tool_choice") else _text(api, "I would say 42.")

    endpoint = Endpoint(api, answer)
    with pytest.raises(StructuredOutputError) as err:
        await _client(_provider(endpoint)).generate_object(Answer, system="Decide.", messages=USER)
    assert not err.value.truncated
    assert len(endpoint.sent) == 3


@pytest.mark.parametrize("api", APIS)
async def test_a_retry_cut_off_by_its_budget_is_not_tried_again(api):
    """A bigger budget is the cure there, which the caller's ladder buys."""
    def answer(_n: int, body: dict) -> tuple[int, dict]:
        return _refusal(api, CLAUDE_5_5_REFUSAL) if body.get("tool_choice") else _text(api, "Let me think", cut_off=True)

    endpoint = Endpoint(api, answer)
    with pytest.raises(StructuredOutputError) as err:
        await _client(_provider(endpoint)).generate_object(Answer, system="Decide.", messages=USER)
    assert err.value.truncated
    assert len(endpoint.sent) == 2


async def test_a_second_try_that_calls_the_tool_returns_its_object():
    def answer(n: int, body: dict) -> tuple[int, dict]:
        if body.get("tool_choice"):
            return _refusal(CHAT, CLAUDE_5_5_REFUSAL)
        return _text(CHAT, "I would say 42.") if n == 2 else _tool_call(CHAT, "Answer", {"answer": "42"})

    endpoint = Endpoint(CHAT, answer)
    result = await _client(_provider(endpoint)).generate_object(Answer, system="Decide.", messages=USER)
    assert result == Answer(answer="42")
    assert len(endpoint.sent) == 3


async def test_the_retry_keeps_the_configured_effort_above_high():
    endpoint = Endpoint(RESPONSES, _refuse_forced(RESPONSES))
    client = _client(_provider(endpoint, effort="xhigh"))
    await client.generate_object(Answer, system="Decide.", messages=USER)
    assert [sent.effort for sent in endpoint.sent] == ["xhigh", "xhigh"]


async def test_a_prompt_that_already_asks_for_the_tool_first_is_sent_as_is():
    endpoint = Endpoint(CHAT, _refuse_forced(CHAT))
    await _client(_provider(endpoint)).generate_object(Answer, system="Decide." + CALL_FIRST, messages=USER)
    assert endpoint.sent[1].system == "Decide." + CALL_FIRST


async def test_every_reply_reports_its_usage_under_the_calls_role():
    def answer(n: int, body: dict) -> tuple[int, dict]:
        if body.get("tool_choice"):
            return _refusal(CHAT, CLAUDE_5_5_REFUSAL)
        return _text(CHAT, "I would say 42.") if n == 2 else _tool_call(CHAT, "Answer", {"answer": "42"})

    client = _client(_provider(Endpoint(CHAT, answer)))
    reported: list[tuple[str, str]] = []
    client.usage_listener = lambda role, model, _usage: reported.append((role, model))
    await client.generate_object_code(Answer, system="Decide.", messages=USER)
    # The refusal carries no usage. Both replies spent tokens, the empty one too.
    assert reported == [("coding", "claude-opus-5-5"), ("coding", "claude-opus-5-5")]


# ── the completion verifier ───────────────────────────────────────────────────


async def test_the_verifier_gets_its_verdict_and_never_counts_a_failure(monkeypatch):
    """The latch tests mock `generate_object_code` to raise, so they can't see a
    retry inside it. Here the real client's call meets a real 400."""
    from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
    from anton.core.session import ChatSession, ChatSessionConfig
    from unittest.mock import MagicMock

    monkeypatch.setenv("ANTON_VERIFY_MIN_TOOL_ROUNDS", "1")
    verdict = {"status": "COMPLETE", "reason": "The answer is complete.", "close_to_done": False}

    def answer(_n: int, body: dict) -> tuple[int, dict]:
        if body.get("tool_choice"):
            return _refusal(RESPONSES, CLAUDE_5_5_REFUSAL)
        return _tool_call(RESPONSES, body["tools"][0]["name"], verdict)

    endpoint = Endpoint(RESPONSES, answer)
    real = _client(_provider(endpoint))
    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = real.generate_object_code

    class _Replay:
        def __init__(self, items):
            self._items = items

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

    def turn_plan():
        calls = {"n": 0}

        def plan(**_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                cell = ToolCall(id="tc_1", name="scratchpad", input={"action": "exec", "name": "main", "code": "print(1)"})
                response = LLMResponse(content="Running.", tool_calls=[cell], usage=Usage(input_tokens=1, output_tokens=1),
                                       stop_reason="tool_use")
            else:
                response = LLMResponse(content="REPLY", tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1),
                                       stop_reason="end_turn")
            return _Replay([StreamComplete(response=response)])

        return plan, calls

    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=MagicMock(base=base)))
    try:
        for turn in range(2):
            plan, calls = turn_plan()
            mock_llm.plan_stream = plan
            async for _ in session.turn_stream(f"do step {turn}"):
                pass
            # A third planning call would be the no-verdict diagnosis.
            assert calls["n"] == 2
        assert session._verifier_latch.no_verdict_failures == 0
        assert session._verifier_latch.latched is False
        assert [sent.tool_choice is not None for sent in endpoint.sent] == [True, False, False]
    finally:
        await session.close()


# ── the scratchpad's in-cell twin ─────────────────────────────────────────────


class _PadEndpoint(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _PadHandler)
        self.forced: list[bool] = []


class _PadHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _PadEndpoint

    def log_message(self, format: str, *args: object) -> None:
        """Silent: the test reads ``server.forced``."""

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.forced.append(bool(body.get("tool_choice")))
        status, payload = _refuse_forced(RESPONSES)(len(self.server.forced), body)
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@contextmanager
def _pad_endpoint() -> Iterator[_PadEndpoint]:
    server = _PadEndpoint()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


_OBJECT_CELL = """
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str

print(get_llm().generate_object(Answer, system="Decide.", messages=[{"role": "user", "content": "Which parts?"}]).answer)
"""


def test_the_scratchpad_remembers_the_refusal_across_its_calls(tmp_path):
    """Each in-cell call builds a fresh provider, so a flag kept on one would
    leave the second cell refused again."""
    with _pad_endpoint() as server:
        results = _run_cells([_OBJECT_CELL, _OBJECT_CELL], tmp_path, f"http://127.0.0.1:{server.server_port}/v1")
    assert [result.error for result in results] == [None, None]
    assert [result.stdout.strip() for result in results] == ["42", "42"]
    assert server.forced == [True, False, False]
