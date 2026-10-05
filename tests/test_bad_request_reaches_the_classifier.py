"""ENG-2689 — a 400 must reach the content classifier on the REAL call path.

The gap this file exists to close. `test_status_error_mapper.py` calls
`_raise_for_status_error(exc, ...)` directly, and
`test_content_rejection_terminal.py` injects an already-built
`ContentTooLargeError` into the session. Both halves were correct and tested,
and the join between them was broken: **no 400 ever reaches
`_raise_for_status_error`.**

`BadRequestError` subclasses `APIStatusError`, so the `except
openai.BadRequestError` clause at each call site matches a 400 first, and its
bare `raise` re-raises out of the whole `try` — a sibling `except` cannot catch
an exception re-raised from its own handler. The classifier was therefore
unreachable on exactly the path the incident came from, and ENG-1992's branch
had been unreachable the same way since it shipped (that fix only ever worked
because cowork-server independently matched the provider's message text).

So these tests drive the real provider methods against a real SDK client whose
transport returns the incident's real body. Nothing is called directly, nothing
is injected: if the except-chain ordering regresses, these fail and the
mapper-level tests do not.

Both flavors are exercised because they take different transports — the
Responses API and chat.completions have their own `except` chains, which is
four call sites in total, and the bug was present in every one.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import anthropic
import httpx2 as httpx
import openai
import pytest

from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import (
    ContentTooLargeError,
    ContentValidationError,
    RequestRefusedError,
)

# The live ENG-2689 body: a user's ~8.3M-pixel PNG, comfortably under every
# byte limit anton enforces, refused by the provider on pixel count.
_PATCHES_400 = {"error": {
    "message": (
        "The image you provided requires 32400 patches after processing, "
        "exceeding the limit of 30000. Please resize the image and try again."
    ),
    "type": "invalid_request_error",
    "param": "input",
    "code": "invalid_value",
}}

# ENG-1992's family, which travels the same broken path.
_SHAPE_400 = {"error": {
    "message": (
        "Invalid value: 'image'. Supported values are: 'input_text', "
        "'input_image', 'input_audio', 'output_text', 'refusal', 'input_file', "
        "'computer_screenshot', 'summary_text', and 'encrypted_content'."
    ),
    "type": "invalid_request_error",
    "param": "input[70].content[0].type",
    "code": "invalid_value",
}}

_MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "what is this?"}]}]


def _provider(body: dict, flavor: str) -> OpenAIProvider:
    """A real provider whose real SDK client fails with `body`.

    The client is the pinned SDK driven through `httpx.MockTransport`, not a
    mock: the whole defect is about which SDK exception CLASS is raised and how
    Python's `except` ordering treats it, and an `AsyncMock` raising a
    hand-built error would model none of that.
    """
    provider = OpenAIProvider(
        api_key="test-key", base_url="https://api.mindshub.ai/v1", flavor=flavor,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=body)

    provider._client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="https://api.mindshub.ai/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return provider


# Both flavors, because they route to different transports and therefore
# different `except` chains: minds-passthrough uses chat.completions, openai
# uses the Responses API.
_FLAVORS = [OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH, OpenAIProvider.FLAVOR_OPENAI]


@pytest.mark.parametrize("flavor", _FLAVORS)
async def test_complete_maps_an_oversized_image_400(flavor):
    provider = _provider(_PATCHES_400, flavor)
    try:
        with pytest.raises(ContentTooLargeError) as err:
            await provider.complete(model="m", system="s", messages=_MESSAGES)
        assert "Please resize the image and try again." in str(err.value)
    finally:
        await provider.aclose()


@pytest.mark.parametrize("flavor", _FLAVORS)
async def test_stream_maps_an_oversized_image_400(flavor):
    provider = _provider(_PATCHES_400, flavor)
    try:
        with pytest.raises(ContentTooLargeError):
            async for _ in provider.stream(model="m", system="s", messages=_MESSAGES):
                pass
    finally:
        await provider.aclose()


@pytest.mark.parametrize("flavor", _FLAVORS)
async def test_complete_maps_a_content_shape_400(flavor):
    """ENG-1992's family reaches the classifier now too. It never did before:
    its branch sat in the same unreachable function, and the repair users
    actually got came from cowork-server matching the message text instead."""
    provider = _provider(_SHAPE_400, flavor)
    try:
        with pytest.raises(ContentValidationError) as err:
            await provider.complete(model="m", system="s", messages=_MESSAGES)
        assert not isinstance(err.value, ContentTooLargeError)
    finally:
        await provider.aclose()


@pytest.mark.parametrize("flavor", _FLAVORS)
async def test_an_unrelated_400_still_propagates_as_the_sdk_error(flavor):
    """The blast-radius guard. The fix names only what it can name and leaves
    every other 400 byte-identical to today — deliberately NOT routing 400s
    through the full status ladder, which would hand a 400 carrying
    `type: api_error` to `classify_transient` and silently make terminal
    failures retryable."""
    provider = _provider(
        {"error": {"message": "'model' is a required property",
                   "type": "invalid_request_error", "param": "model"}},
        flavor,
    )
    try:
        with pytest.raises(openai.BadRequestError) as err:
            await provider.complete(model="m", system="s", messages=_MESSAGES)
        assert not isinstance(err.value, ContentValidationError)
    finally:
        await provider.aclose()


@pytest.mark.parametrize("flavor", _FLAVORS)
async def test_a_context_overflow_400_keeps_its_own_mapping(flavor):
    """The one 400 this clause already handled. Routing the rest through a
    helper must not displace it — it is checked first, inside the helper."""
    from anton.core.llm.provider import ContextOverflowError

    provider = _provider(
        {"error": {"message": "This model's maximum context length is 128000 tokens.",
                   "type": "invalid_request_error", "code": "context_length_exceeded"}},
        flavor,
    )
    try:
        with pytest.raises(ContextOverflowError):
            await provider.complete(model="m", system="s", messages=_MESSAGES)
    finally:
        await provider.aclose()


# ── the Anthropic provider has the identical trap ──────────────────────────
# `anthropic.BadRequestError` subclasses `anthropic.APIStatusError` too, and
# this provider's two call sites were shaped the same way. Worth its own
# coverage rather than assumed-by-symmetry: the two providers' `except` chains
# are separate code, and only one of them had ENG-1992's branch at all.


def _anthropic_provider(body: dict) -> AnthropicProvider:
    provider = AnthropicProvider(api_key="test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=body)

    provider._client = anthropic.AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return provider


_ANTHROPIC_SIZE_400 = {"type": "error", "error": {
    "type": "invalid_request_error",
    "message": (
        "messages.0.content.0.image.source.base64.data: At least one of the "
        "image dimensions exceed max allowed size for many-image requests: "
        "2000 pixels"
    ),
}}


async def test_anthropic_complete_maps_an_oversized_image_400():
    provider = _anthropic_provider(_ANTHROPIC_SIZE_400)
    with pytest.raises(ContentTooLargeError):
        await provider.complete(model="claude-sonnet", system="s", messages=_MESSAGES)


async def test_anthropic_stream_maps_an_oversized_image_400():
    provider = _anthropic_provider(_ANTHROPIC_SIZE_400)
    with pytest.raises(ContentTooLargeError):
        async for _ in provider.stream(
            model="claude-sonnet", system="s", messages=_MESSAGES
        ):
            pass


async def test_anthropic_leaves_an_unrelated_400_as_the_sdk_error():
    provider = _anthropic_provider({"type": "error", "error": {
        "type": "invalid_request_error", "message": "max_tokens: must be greater than 0",
    }})
    with pytest.raises(anthropic.BadRequestError) as err:
        await provider.complete(model="claude-sonnet", system="s", messages=_MESSAGES)
    assert not isinstance(err.value, ContentValidationError)


# ── the whole chain, end to end ────────────────────────────────────────────
# The two halves above each prove their own seam. This proves the product
# behaviour the ticket is actually about: a real SDK 400 carrying the
# incident's body, through the real provider, the real LLMClient and a real
# ChatSession, produces ONE attempt and an exception the host can map to a
# card — not four attempts and "try again in a moment".
#
# Worth its own test because "the provider raises X" and "the session doesn't
# retry X" were both true and separately tested while the product was broken.


async def test_an_oversized_image_400_ends_the_turn_in_one_attempt():
    from unittest.mock import patch

    from anton.chat import ChatSession
    from anton.core.llm.client import LLMClient
    from anton.core.session import ChatSessionConfig

    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json=_PATCHES_400)

    provider = OpenAIProvider(
        api_key="test-key",
        base_url="https://api.mindshub.ai/v1",
        flavor=OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH,
    )
    provider._client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="https://api.mindshub.ai/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    client = LLMClient(
        planning_provider=provider, planning_model="m",
        coding_provider=provider, coding_model="m",
    )
    session = ChatSession(ChatSessionConfig(llm_client=client, session_id="conv-2689-e2e"))

    try:
        with patch("anton.analytics.send_event") as send:
            with pytest.raises(ContentTooLargeError) as err:
                _ = [e async for e in session.turn_stream("what does this image say?")]
    finally:
        await provider.aclose()

    # The headline number from the incident: four requests, 33.7s, one dead task.
    assert attempts == 1, f"expected a single provider request, got {attempts}"
    assert "Please resize the image and try again." in str(err.value)
    assert send.call_args.kwargs["retry_terminal_reason"] == "content_rejected"


# ── the destructive false positive, on the real provider path (#484 review) ─
# Classifying a 400 as a content rejection makes the host strip EVERY image
# from the conversation's stored history and report it fixed. A bad enum value
# on an unrelated parameter must therefore never qualify — otherwise a
# `reasoning_effort` typo costs the user their images AND leaves the real
# configuration error unfixed and unexplained.
#
# Verified against the real provider path, not the classifier in isolation:
# that is where the old tests were blind, and this is the direction where being
# wrong destroys data rather than merely showing worse copy.

_UNRELATED_ENUM_400S = {
    "reasoning_effort": {"error": {
        "message": "Invalid value: 'ultra'. Supported values are: 'low', 'medium', 'high'.",
        "type": "invalid_request_error", "param": "reasoning_effort",
        "code": "invalid_value"}},
    "tool_choice": {"error": {
        "message": "Invalid value: 'always'. Supported values are: 'none', 'auto', 'required'.",
        "type": "invalid_request_error", "param": "tool_choice",
        "code": "invalid_value"}},
    "service_tier": {"error": {
        "message": "Invalid value: 'turbo'. Supported values are: 'auto', 'default', 'flex'.",
        "type": "invalid_request_error", "param": "service_tier",
        "code": "invalid_value"}},
    # The nastiest of the family, and the one the first version of the guard
    # still let through: `modalities` legitimately ACCEPTS the value 'image',
    # so its enum message names a content-block token while having nothing to
    # do with content. Found by adversarially reviewing the fix, not by the
    # review that prompted it.
    "modalities": {"error": {
        "message": "Invalid value: 'text'. Supported values are: 'image', 'audio'.",
        "type": "invalid_request_error", "param": "modalities",
        "code": "invalid_value"}},
}


@pytest.mark.parametrize("param", sorted(_UNRELATED_ENUM_400S))
@pytest.mark.parametrize("flavor", _FLAVORS)
async def test_an_unrelated_enum_400_never_triggers_image_repair(flavor, param):
    provider = _provider(_UNRELATED_ENUM_400S[param], flavor)
    # A refused `reasoning_effort` is the refusal rung's to name (see the
    # section below); every other enum stays the SDK error.
    expected = RequestRefusedError if param == "reasoning_effort" else openai.BadRequestError
    try:
        with pytest.raises(expected) as err:
            await provider.complete(model="m", system="s", messages=_MESSAGES)
        assert not isinstance(err.value, ContentValidationError), (
            f"a bad {param} enum was classified as a content rejection — the host "
            "would delete every image in the conversation over a config typo"
        )
    finally:
        await provider.aclose()


@pytest.mark.parametrize("param", ["reasoning_effort", "tool_choice"])
async def test_an_unrelated_enum_400_does_not_strip_history_images(param):
    """The consequence, end to end: the session must leave the user's images
    alone when the 400 had nothing to do with content."""
    from anton.chat import ChatSession
    from anton.core.llm.client import LLMClient
    from anton.core.session import ChatSessionConfig

    provider = _provider(_UNRELATED_ENUM_400S[param],
                         OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH)
    client = LLMClient(planning_provider=provider, planning_model="m",
                       coding_provider=provider, coding_model="m")
    session = ChatSession(ChatSessionConfig(llm_client=client, session_id="conv-enum"))
    session._append_history({
        "role": "user",
        "content": [{"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/png", "data": "AAAA"}}],
    })
    try:
        # A refused reasoning_effort fails the turn with RequestRefusedError;
        # any other unrelated 400 is still retried and ends as prose. Either
        # way, what matters here is that the images are untouched.
        try:
            _ = [e async for e in session.turn_stream("hello")]
        except BaseException:
            pass
    finally:
        await provider.aclose()

    survived = [
        b for m in session._history
        if isinstance(m, dict) and isinstance(m.get("content"), list)
        for b in m["content"] if isinstance(b, dict) and b.get("type") == "image"
    ]
    assert survived, "an unrelated config error destroyed the conversation's images"


# ── a request the provider refuses outright fails the turn on the first attempt
# OpenAI's 400 when chat.completions refuses function tools for a model that
# reasons by default, even when the request sends no effort. Left on the retry
# path, the session re-sends it twice, then asks the model without tools to
# explain the error, and that explanation streams out as the answer: four
# requests, and nothing raised for the host to show.

_TOOLS_WITH_EFFORT_400 = {"error": {
    "type": "invalid_request_error",
    "param": "reasoning_effort",
    "message": (
        "Function tools with reasoning_effort are not supported for gpt-6-luna "
        "in /v1/chat/completions. To use function tools, use /v1/responses or "
        "set reasoning_effort to 'none'."
    ),
}}

# Azure documents the same refusal by its sentence alone, with no `param`.
_TOOLS_WITH_EFFORT_400_NO_PARAM = {"error": {
    "type": "invalid_request_error",
    "message": (
        "Function tools with reasoning_effort are not supported for gpt-5.6-sol "
        "in /v1/chat/completions. To use function tools, use /v1/responses or "
        "set reasoning_effort to 'none'."
    ),
}}

# A prompt the provider's policy blocks. Azure's prompt filter, as Microsoft
# documents it for chat completions: no `type`, so a rule keyed on
# invalid_request_error never sees it.
_AZURE_PROMPT_FILTER_400 = {"error": {
    "message": (
        "The response was filtered due to the prompt triggering Azure OpenAI's "
        "content management policy. Please modify your prompt and retry. To learn "
        "more about our content filtering policies please read our documentation: "
        "https://go.microsoft.com/fwlink/?linkid=2198766"
    ),
    "type": None,
    "param": "prompt",
    "code": "content_filter",
    "status": 400,
    "innererror": {
        "code": "ResponsibleAIPolicyViolation",
        "content_filter_result": {
            "hate": {"filtered": False, "severity": "safe"},
            "jailbreak": {"filtered": False, "detected": False},
            "self_harm": {"filtered": False, "severity": "safe"},
            "sexual": {"filtered": False, "severity": "safe"},
            "violence": {"filtered": True, "severity": "medium"},
        },
    },
}}

# OpenAI's flagged prompt.
_OPENAI_INVALID_PROMPT_400 = {"error": {
    "message": (
        "Invalid prompt: your prompt was flagged as potentially violating our "
        "usage policy. Please try again with a different prompt: "
        "https://platform.openai.com/docs/guides/reasoning#advice-on-prompting"
    ),
    "type": "invalid_request_error",
    "param": None,
    "code": "invalid_prompt",
}}

# A history-shape 400: an assistant tool call with no tool result after it.
# Today it stays the SDK error, and the session re-sends after sealing the call.
_ORPHAN_TOOL_CALL_400 = {"error": {
    "type": "invalid_request_error",
    "param": "messages.[3].role",
    "message": (
        "An assistant message with 'tool_calls' must be followed by tool "
        "messages responding to each 'tool_call_id'. The following "
        "tool_call_ids did not have response messages: call_7Qx"
    ),
}}

# A param inside a message's content. The content rung runs before the refusal
# rung and claims it, as it does today.
_CONTENT_PARAM_400 = {"error": {
    "type": "invalid_request_error",
    "param": "messages[3].content",
    "code": "string_above_max_length",
    "message": (
        "Invalid 'messages[3].content': string too long. Expected a string with "
        "maximum length 10485760, but got a string with length 10485761 instead."
    ),
}}

_TOOLS = [{
    "name": "scratchpad",
    "description": "Run Python in a persistent session.",
    "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}},
}]

# The two transports a refusal comes back on: the generic flavor sends
# chat.completions, the openai flavor sends the Responses API.
_REFUSAL_FLAVORS = [
    OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC,
    OpenAIProvider.FLAVOR_OPENAI,
]


# What the model answers when a request reaches it, such as the session's
# tool-less call asking it to explain the error.
_EXPLANATION = "The model settings do not allow tools with this reasoning effort."


def _streamed_answer(flavor: str, text: str) -> bytes:
    """`text` as a 200 streamed answer, in the wire format `flavor` reads."""
    if flavor == OpenAIProvider.FLAVOR_OPENAI:
        response = {"id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-6-luna",
                    "output": [], "parallel_tool_calls": True, "tool_choice": "auto", "tools": []}
        events = [
            {"type": "response.created", "response": {**response, "status": "in_progress"}},
            {"type": "response.output_item.added", "output_index": 0, "item": {
                "type": "message", "id": "msg_1", "role": "assistant",
                "status": "in_progress", "content": []}},
            {"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0,
             "content_index": 0, "delta": text, "logprobs": []},
            {"type": "response.completed", "response": {**response, "status": "completed"}},
        ]
        return "".join(
            f"event: {e['type']}\ndata: {json.dumps({'sequence_number': i, **e})}\n\n"
            for i, e in enumerate(events)
        ).encode()
    chunk = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0,
             "model": "gpt-6-luna"}
    chunks = [
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                               "finish_reason": None}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


def _recording_provider(
    body: dict, flavor: str, requests: list[dict],
    *, refuses: Callable[[dict], bool] = lambda request: True,
) -> OpenAIProvider:
    """A real provider on OpenAI's host that appends each request body to
    `requests`. A request that `refuses` picks gets a 400 carrying `body`, and
    any other gets `_EXPLANATION` as a streamed answer."""
    base_url = "https://api.openai.com/v1"
    provider = OpenAIProvider(api_key="test-key", base_url=base_url, flavor=flavor)

    def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        requests.append(sent)
        if refuses(sent):
            return httpx.Response(400, json=body)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=_streamed_answer(flavor, _EXPLANATION),
        )

    provider._client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url=base_url,
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return provider


async def _call(provider: OpenAIProvider, method: str) -> None:
    """One provider call with a function tool attached, as the agent loop makes it."""
    if method == "complete":
        await provider.complete(model="gpt-6-luna", system="s", messages=_MESSAGES, tools=_TOOLS)
        return
    async for _ in provider.stream(model="gpt-6-luna", system="s", messages=_MESSAGES, tools=_TOOLS):
        pass


@pytest.mark.parametrize("method", ["complete", "stream"])
@pytest.mark.parametrize("flavor", _REFUSAL_FLAVORS)
@pytest.mark.parametrize(
    "body", [_TOOLS_WITH_EFFORT_400, _TOOLS_WITH_EFFORT_400_NO_PARAM],
    ids=["param", "message-only"],
)
async def test_a_refused_reasoning_effort_400_raises_the_refusal(body, flavor, method):
    requests: list[dict] = []
    provider = _recording_provider(body, flavor, requests)
    try:
        with pytest.raises(RequestRefusedError) as err:
            await _call(provider, method)
    finally:
        await provider.aclose()

    assert err.value.code == "parameter_refused"
    assert err.value.status_code == 400
    assert "Function tools with reasoning_effort are not supported" in str(err.value)
    assert isinstance(err.value.__cause__, openai.BadRequestError)
    assert len(requests) == 1


@pytest.mark.parametrize("method", ["complete", "stream"])
@pytest.mark.parametrize("flavor", _REFUSAL_FLAVORS)
@pytest.mark.parametrize(
    "body,code",
    [(_AZURE_PROMPT_FILTER_400, "content_filter"), (_OPENAI_INVALID_PROMPT_400, "invalid_prompt")],
    ids=["azure-content_filter", "openai-invalid_prompt"],
)
async def test_a_refused_prompt_400_raises_the_refusal(body, code, flavor, method):
    """A blocked prompt is in the history every re-send and explain call
    carries, so each of them would be refused the same way."""
    requests: list[dict] = []
    provider = _recording_provider(body, flavor, requests)
    try:
        with pytest.raises(RequestRefusedError) as err:
            await _call(provider, method)
    finally:
        await provider.aclose()

    assert (err.value.code, err.value.status_code) == (code, 400)
    assert body["error"]["message"][:60] in str(err.value)
    assert "ResponsibleAIPolicyViolation" not in str(err.value)
    assert isinstance(err.value.__cause__, openai.BadRequestError)
    assert len(requests) == 1


async def test_the_refusal_quotes_the_provider_scrubbed_and_capped():
    """The message is ours, with a short quote of the provider's words: never
    the raw body, and never a key the provider echoed back."""
    leaked = "sk-proj-" + "A" * 40
    body = {"error": {
        "type": "invalid_request_error",
        "param": "reasoning_effort",
        "message": f"Unsupported value for key {leaked}. " + "x" * 2000,
    }}
    provider = _recording_provider(body, OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC, [])
    try:
        with pytest.raises(RequestRefusedError) as err:
            await _call(provider, "stream")
    finally:
        await provider.aclose()

    text = str(err.value)
    assert leaked not in text
    assert "[REDACTED_API_KEY]" in text
    assert "x" * 400 not in text
    assert "Error code" not in text and "invalid_request_error" not in text


@pytest.mark.parametrize(
    "body,today",
    [(_ORPHAN_TOOL_CALL_400, openai.BadRequestError), (_CONTENT_PARAM_400, ContentValidationError)],
    ids=["orphan-tool-call", "content-param"],
)
@pytest.mark.parametrize("flavor", _REFUSAL_FLAVORS)
async def test_a_history_shape_400_keeps_its_mapping(flavor, body, today):
    provider = _recording_provider(body, flavor, [])
    try:
        with pytest.raises(today) as err:
            await _call(provider, "stream")
    finally:
        await provider.aclose()

    assert not isinstance(err.value, RequestRefusedError)


# Each refusal as its provider applies it. Function tools with a reasoning
# effort are refused only on a request that carries tools, so the session's
# tool-less explain call is answered, and its prose would end the turn as the
# answer. A blocked prompt is refused on every request in the conversation.
_REFUSED_TURNS = {
    "tools-with-effort": (_TOOLS_WITH_EFFORT_400, lambda request: bool(request.get("tools"))),
    "azure-prompt-filter": (_AZURE_PROMPT_FILTER_400, lambda request: True),
    "openai-invalid-prompt": (_OPENAI_INVALID_PROMPT_400, lambda request: True),
}


@pytest.mark.parametrize("refusal", sorted(_REFUSED_TURNS))
@pytest.mark.parametrize("flavor", _REFUSAL_FLAVORS)
async def test_a_refused_request_ends_the_turn_in_one_request(flavor, refusal):
    """The product behaviour: one refused request, then an error the host can
    show. No re-send, and no tool-less call asking the model to explain it."""
    from unittest.mock import patch

    from anton.chat import ChatSession
    from anton.core.llm.client import LLMClient
    from anton.core.llm.provider import StreamTextDelta
    from anton.core.session import ChatSessionConfig

    body, refuses = _REFUSED_TURNS[refusal]
    requests: list[dict] = []
    provider = _recording_provider(body, flavor, requests, refuses=refuses)
    client = LLMClient(planning_provider=provider, planning_model="gpt-6-luna",
                       coding_provider=provider, coding_model="gpt-6-luna")
    session = ChatSession(ChatSessionConfig(llm_client=client, session_id="conv-refusal"))

    events: list = []
    try:
        with patch("anton.analytics.send_event") as send:
            with pytest.raises(RequestRefusedError):
                async for event in session.turn_stream(
                        "Run a Python cell that prints the first 20 prime numbers."):
                    events.append(event)
    finally:
        await provider.aclose()

    assert len(requests) == 1, f"expected one provider request, got {len(requests)}"
    assert requests[0].get("tools"), "the refused request carried no tools"
    assert not [e for e in events if isinstance(e, StreamTextDelta)], (
        "prose reached the user instead of the error"
    )
    assert "The task has failed" not in json.dumps(session._history)
    assert send.call_args.kwargs["retry_terminal_reason"] == "request_refused"
    assert send.call_args.kwargs["provider_http_status"] == "400"


@pytest.mark.parametrize("flavor", _REFUSAL_FLAVORS)
async def test_a_history_shape_400_still_retries_then_explains(flavor):
    """The narrow rule leaves this 400 on today's path: three attempts with
    tools, then one tool-less call asking the model to explain, which fails
    too here and ends as prose. An orphan tool call can heal on the re-send
    once the session seals it, so it must not end the turn on the first one."""
    from anton.chat import ChatSession
    from anton.core.llm.client import LLMClient
    from anton.core.llm.provider import StreamTextDelta
    from anton.core.session import ChatSessionConfig

    requests: list[dict] = []
    provider = _recording_provider(_ORPHAN_TOOL_CALL_400, flavor, requests)
    client = LLMClient(planning_provider=provider, planning_model="gpt-6-luna",
                       coding_provider=provider, coding_model="gpt-6-luna")
    session = ChatSession(ChatSessionConfig(llm_client=client, session_id="conv-orphan"))

    try:
        events = [e async for e in session.turn_stream("hello")]
    finally:
        await provider.aclose()

    assert len(requests) == 4, f"expected three attempts and one explain call, got {len(requests)}"
    assert all(r.get("tools") for r in requests[:3])
    assert "tools" not in requests[3]
    text = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    assert "An unexpected error occurred" in text
