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

import anthropic
import httpx2 as httpx
import openai
import pytest

from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import ContentTooLargeError, ContentValidationError

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
}


@pytest.mark.parametrize("param", sorted(_UNRELATED_ENUM_400S))
@pytest.mark.parametrize("flavor", _FLAVORS)
async def test_an_unrelated_enum_400_never_triggers_image_repair(flavor, param):
    provider = _provider(_UNRELATED_ENUM_400S[param], flavor)
    try:
        with pytest.raises(openai.BadRequestError) as err:
            await provider.complete(model="m", system="s", messages=_MESSAGES)
        assert not isinstance(err.value, ContentValidationError), (
            f"a bad {param} enum was classified as a content rejection — the host "
            "would delete every image in the conversation over a config typo"
        )
    finally:
        await provider.aclose()


async def test_an_unrelated_enum_400_does_not_strip_history_images():
    """The consequence, end to end: the session must leave the user's images
    alone when the 400 had nothing to do with content."""
    from anton.chat import ChatSession
    from anton.core.llm.client import LLMClient
    from anton.core.session import ChatSessionConfig

    provider = _provider(_UNRELATED_ENUM_400S["reasoning_effort"],
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
        # An unrelated 400 keeps its existing behaviour: retried, then ending as
        # prose rather than a typed error. Not this PR's to change (ENG-1283
        # owns it) — what matters here is that the images are untouched.
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
