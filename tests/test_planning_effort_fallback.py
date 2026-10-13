"""A reasoning effort picked for the turn, and refused by the provider, falls
back to the configured effort instead of failing the turn.

Driven through `LLMClient.from_settings` and the real OpenAI SDK on a mock
transport, so the refusal comes from the provider's own 400 classification.
"""

from __future__ import annotations

import json

import httpx2 as httpx
import openai
import pytest

from anton.config.settings import AntonSettings
from anton.core.llm.client import LLMClient
from anton.core.llm.openai import OpenAIProvider
from anton.core.llm.provider import RequestRefusedError, StreamTextDelta
from tests.test_bad_request_reaches_the_classifier import (
    _MESSAGES,
    _TOOLS,
    _TOOLS_WITH_EFFORT_400,
    _streamed_answer,
)

_BASE_URL = "https://api.openai.com/v1"
_FLAVORS = [OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC, OpenAIProvider.FLAVOR_OPENAI]
_PROMPT_BLOCKED_400 = {"error": {
    "type": "invalid_request_error", "code": "content_filter", "param": None,
    "message": "The prompt was blocked.",
}}


def _sent_effort(request: dict) -> str | None:
    """The effort a request body carries, in either wire format."""
    return request.get("reasoning_effort") or (request.get("reasoning") or {}).get("effort")


def _client(monkeypatch, flavor, *, configured, override, refused_effort, body=_TOOLS_WITH_EFFORT_400):
    """A real client whose transport refuses any request carrying `refused_effort`."""
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        requests.append(sent)
        if _sent_effort(sent) == refused_effort:
            return httpx.Response(400, json=body)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=_streamed_answer(flavor, "answered"),
        )

    real_async_openai = openai.AsyncOpenAI

    def mock_async_openai(**kwargs):
        kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return real_async_openai(**{**kwargs, "max_retries": 0})

    monkeypatch.setattr(openai, "AsyncOpenAI", mock_async_openai)
    provider = "openai" if flavor == OpenAIProvider.FLAVOR_OPENAI else "openai-compatible"
    settings = AntonSettings(
        _env_file=None,
        planning_provider=provider, coding_provider=provider,
        planning_model="gpt-6-luna", coding_model="gpt-6-luna",
        openai_api_key="test-key", openai_base_url=_BASE_URL,
        planning_reasoning_effort=configured,
    )
    if flavor != OpenAIProvider.FLAVOR_OPENAI:
        monkeypatch.setattr("anton.core.llm.client._resolve_openai_compatible_flavor", lambda s: flavor)
    return LLMClient.from_settings(settings, planning_effort_override=override), requests


async def _answer(client: LLMClient) -> str:
    text = ""
    async for event in client.plan_stream(system="s", messages=_MESSAGES, tools=_TOOLS):
        if isinstance(event, StreamTextDelta):
            text += event.text
    return text


@pytest.mark.asyncio
@pytest.mark.parametrize("flavor", _FLAVORS)
@pytest.mark.parametrize("configured", [None, "low"])
async def test_a_refused_selected_effort_is_resent_once_at_the_configured_effort(
    monkeypatch, flavor, configured,
):
    client, requests = _client(
        monkeypatch, flavor, configured=configured, override="max", refused_effort="max",
    )
    try:
        assert await _answer(client) == "answered"
        assert [_sent_effort(r) for r in requests] == ["max", configured]
        assert await _answer(client) == "answered"
        assert len(requests) == 3, "later turns go straight to the configured effort"
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("flavor", _FLAVORS)
async def test_a_refused_configured_effort_still_fails_as_before(monkeypatch, flavor):
    client, requests = _client(
        monkeypatch, flavor, configured="max", override=None, refused_effort="max",
    )
    try:
        with pytest.raises(RequestRefusedError) as err:
            await _answer(client)
        assert err.value.code == "parameter_refused"
        assert len(requests) == 1
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("flavor", _FLAVORS)
async def test_a_blocked_prompt_is_not_resent(monkeypatch, flavor):
    client, requests = _client(
        monkeypatch, flavor, configured=None, override="max", refused_effort="max",
        body=_PROMPT_BLOCKED_400,
    )
    try:
        with pytest.raises(RequestRefusedError) as err:
            await _answer(client)
        assert err.value.code == "content_filter"
        assert len(requests) == 1
    finally:
        await client.aclose()
