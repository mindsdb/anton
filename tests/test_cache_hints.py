"""MindsHub prompt-cache hints: `X-MindsHub-Cache-TTL` and `prompt_cache_key`.

Only the MindsHub passthrough sends either. The lifetime header goes on the
main loop's planning calls alone, when `LLMClient.main_cache_ttl` is set; the
cache key goes on every call whose conversation (Langfuse session) and role
are known.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from anton.core.llm.client import LLMClient, ModelRoute, model_route
from anton.core.llm.openai import CACHE_TTL_HEADER, OpenAIProvider
from anton.core.llm.tracing import TraceContext, call_role, reset_trace_context, set_trace_context


def _reply():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop")],
        usage=None,
        model="m",
    )


async def _chunks():
    yield SimpleNamespace(
        model="m", usage=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop")],
    )


def _provider(flavor: str = OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH, *, stream: bool = False) -> OpenAIProvider:
    provider = OpenAIProvider(api_key="k", base_url="https://api.mindshub.ai/v1", flavor=flavor)
    provider._client.chat.completions.create = AsyncMock(
        side_effect=lambda **kw: _chunks() if stream else _reply()
    )
    return provider


def _sent(provider: OpenAIProvider) -> dict:
    return provider._client.chat.completions.create.call_args.kwargs


def _client(planning, coding, **kw) -> LLMClient:
    return LLMClient(planning_provider=planning, planning_model="opus", coding_provider=coding,
                     coding_model="haiku", **kw)


@pytest.fixture()
def conversation():
    token = set_trace_context(TraceContext(session_id="conv-1", turn_id=1, harness="anton"))
    yield
    reset_trace_context(token)


async def test_planning_calls_ask_for_the_main_cache_ttl(conversation):
    planning, coding = _provider(), _provider()
    client = _client(planning, coding, main_cache_ttl="1h")

    await client.plan(system="s", messages=[{"role": "user", "content": "hi"}])

    sent = _sent(planning)
    assert sent["extra_headers"][CACHE_TTL_HEADER] == "1h"
    assert sent["prompt_cache_key"] == "conv-1:planning"
    # The trace headers are still there beside it.
    assert sent["extra_headers"]["Langfuse-Session-Id"] == "conv-1"


async def test_streamed_planning_calls_ask_for_it_too(conversation):
    planning = _provider(stream=True)
    client = _client(planning, _provider(), main_cache_ttl="5m")

    async for _ in client.plan_stream(system="s", messages=[{"role": "user", "content": "hi"}]):
        pass

    sent = _sent(planning)
    assert sent["extra_headers"][CACHE_TTL_HEADER] == "5m"
    assert sent["prompt_cache_key"] == "conv-1:planning"


async def test_other_roles_send_a_cache_key_but_no_lifetime(conversation):
    planning, coding, sidekick = _provider(), _provider(), _provider(stream=True)
    client = _client(planning, coding, main_cache_ttl="1h", sidekick_provider=sidekick, sidekick_model="gpt-sol")

    await client.code(system="s", messages=[{"role": "user", "content": "hi"}])
    async for _ in client.sidekick_stream(system="s", messages=[{"role": "user", "content": "hi"}]):
        pass
    await client.summarize(system="s", messages=[{"role": "user", "content": "hi"}])

    for provider, role in ((coding, "router"), (sidekick, "sidekick")):
        sent = _sent(provider)
        assert CACHE_TTL_HEADER not in sent.get("extra_headers", {})
        assert sent["prompt_cache_key"] == f"conv-1:{role}"
    first_coding_call = coding._client.chat.completions.create.call_args_list[0].kwargs
    assert first_coding_call["prompt_cache_key"] == "conv-1:coding"
    assert CACHE_TTL_HEADER not in first_coding_call["extra_headers"]


async def test_no_lifetime_is_asked_for_when_none_is_set(conversation):
    planning = _provider()
    await _client(planning, _provider()).plan(system="s", messages=[{"role": "user", "content": "hi"}])
    assert CACHE_TTL_HEADER not in _sent(planning)["extra_headers"]


async def test_a_routed_planning_call_asks_for_no_lifetime(conversation):
    # generate_artifact's builder on a model of its own has no long prefix to keep.
    planning = _provider()
    client = _client(planning, _provider(), main_cache_ttl="1h")
    with model_route(ModelRoute(model="builder")):
        await client.plan(system="s", messages=[{"role": "user", "content": "hi"}])
    assert CACHE_TTL_HEADER not in _sent(planning)["extra_headers"]
    assert _sent(planning)["prompt_cache_key"] == "conv-1:planning"


@pytest.mark.parametrize(
    "flavor", [OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC],
)
async def test_other_flavors_send_neither(conversation, flavor):
    planning = _provider(flavor)
    await _client(planning, _provider(flavor), main_cache_ttl="1h").plan(
        system="s", messages=[{"role": "user", "content": "hi"}]
    )
    sent = _sent(planning)
    assert CACHE_TTL_HEADER not in (sent.get("extra_headers") or {})
    assert "prompt_cache_key" not in sent


async def test_without_a_conversation_there_is_no_cache_key():
    planning = _provider()
    await _client(planning, _provider(), main_cache_ttl="1h").plan(
        system="s", messages=[{"role": "user", "content": "hi"}]
    )
    sent = _sent(planning)
    assert "prompt_cache_key" not in sent
    # The lifetime does not depend on a trace.
    assert sent["extra_headers"] == {CACHE_TTL_HEADER: "1h"}


def test_a_direct_provider_call_keys_on_its_default_role(conversation):
    # A host calling the provider outside the client (cowork-server's route
    # gate) is keyed on the role the client stamped on it, with no lifetime.
    provider = _provider()
    provider.trace_role = "router"
    kwargs: dict = {}
    provider._apply_cache_hints(kwargs)
    assert kwargs == {"prompt_cache_key": "conv-1:router"}
    with call_role("planning", cache_ttl="1h"):
        provider._apply_cache_hints(kwargs)
    assert kwargs == {"prompt_cache_key": "conv-1:planning", "extra_headers": {CACHE_TTL_HEADER: "1h"}}
