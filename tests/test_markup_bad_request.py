"""A 400 whose body is HTML or XML must not become a chat reply.

Both SDKs keep a body that is not JSON as the raw string, and use it as the
exception's message too. Unclassified, that error re-raised from the 400 path,
was retried, and ended as the turn's prose reply, past the host's redaction.
The provider now classifies a markup body as a misrouted request, and the
session never streams an exception as prose.

These drive the pinned SDKs through ``httpx.MockTransport`` rather than mocks,
for the reason ``test_bad_request_reaches_the_classifier.py`` gives: which SDK
exception class is raised, and what it carries, is the defect.
"""

from __future__ import annotations

import json

import anthropic
import httpx2 as httpx
import openai
import pytest

from anton.core.llm.anthropic import AnthropicProvider
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import (
    EndpointConfigurationError,
    StreamTextDelta,
    classify_markup_bad_request,
)

# An XML error body of the kind a storage service sends.
_XML = (
    "<?xml version='1.0' encoding='UTF-8'?><Error><Code>InvalidArgument</Code>"
    "<Message>Invalid argument.</Message><Details>POST object expects "
    "Content-Type multipart/form-data</Details></Error>"
)
_HTML = "<html><head><title>400 Bad Request</title></head><body>nginx</body></html>"

_MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

# Both transports: minds-passthrough uses chat.completions, openai the
# Responses API, and each has its own `except` chain.
_FLAVORS = [OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH, OpenAIProvider.FLAVOR_OPENAI]


def _markup_response(body: str) -> httpx.Response:
    return httpx.Response(400, headers={"content-type": "application/xml"}, content=body.encode())


def _openai_provider(body: str, flavor: str, requests: list | None = None) -> OpenAIProvider:
    base_url = "https://api.mindshub.ai/v1"
    provider = OpenAIProvider(api_key="test-key", base_url=base_url, flavor=flavor)

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(json.loads(request.content))
        return _markup_response(body)

    provider._client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url=base_url,
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return provider


def _anthropic_provider(body: str) -> AnthropicProvider:
    provider = AnthropicProvider(api_key="test-key")
    provider._client = anthropic.AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: _markup_response(body))
        ),
    )
    return provider


def _assert_no_markup(text: str) -> None:
    assert "<" not in text
    assert "InvalidArgument" not in text
    assert "multipart" not in text


# ── the classifier ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("body", [_XML, _HTML, "  \n" + _HTML], ids=["xml", "html", "leading-space"])
def test_a_markup_body_is_a_misrouted_request(body):
    err = classify_markup_bad_request(body)
    assert isinstance(err, EndpointConfigurationError)
    _assert_no_markup(str(err))


@pytest.mark.parametrize(
    "body",
    [{"error": {"message": "<b> is not a valid role"}}, "plain text error", "", None, ["<x>"]],
    ids=["json-dict", "plain-text", "empty", "none", "list"],
)
def test_any_other_body_is_left_to_the_existing_rungs(body):
    assert classify_markup_bad_request(body) is None


# ── each provider's 400 path ───────────────────────────────────────────────


@pytest.mark.parametrize("flavor", _FLAVORS)
async def test_openai_complete_maps_a_markup_400(flavor):
    provider = _openai_provider(_XML, flavor)
    with pytest.raises(EndpointConfigurationError) as err:
        await provider.complete(model="gpt-6-luna", system="s", messages=_MESSAGES)
    _assert_no_markup(str(err.value))


@pytest.mark.parametrize("flavor", _FLAVORS)
async def test_openai_stream_maps_a_markup_400(flavor):
    provider = _openai_provider(_XML, flavor)
    with pytest.raises(EndpointConfigurationError) as err:
        async for _ in provider.stream(model="gpt-6-luna", system="s", messages=_MESSAGES):
            pass
    _assert_no_markup(str(err.value))


async def test_anthropic_complete_maps_a_markup_400():
    provider = _anthropic_provider(_HTML)
    with pytest.raises(EndpointConfigurationError):
        await provider.complete(model="claude-sonnet", system="s", messages=_MESSAGES)


async def test_anthropic_stream_maps_a_markup_400():
    provider = _anthropic_provider(_HTML)
    with pytest.raises(EndpointConfigurationError):
        async for _ in provider.stream(model="claude-sonnet", system="s", messages=_MESSAGES):
            pass


# ── the whole turn ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("flavor", _FLAVORS)
async def test_a_markup_400_fails_the_turn_without_xml_in_text_or_history(flavor):
    """End to end: one request, a raised error the host maps to its own copy,
    and no raw body in the streamed text or the stored history."""
    from anton.chat import ChatSession
    from anton.core.llm.client import LLMClient
    from anton.core.session import ChatSessionConfig

    requests: list[dict] = []
    provider = _openai_provider(_XML, flavor, requests)
    client = LLMClient(planning_provider=provider, planning_model="gpt-6-luna",
                       coding_provider=provider, coding_model="gpt-6-luna")
    session = ChatSession(ChatSessionConfig(llm_client=client, session_id="conv-markup-400"))

    events: list = []
    try:
        with pytest.raises(EndpointConfigurationError):
            async for event in session.turn_stream("hi"):
                events.append(event)
    finally:
        await provider.aclose()

    assert len(requests) == 1, f"expected one provider request, got {len(requests)}"
    assert not [e for e in events if isinstance(e, StreamTextDelta)]
    _assert_no_markup(json.dumps(session._history))
