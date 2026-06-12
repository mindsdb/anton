"""Unit tests for the Fireworks external-search proxy + adapter.

Fireworks-shaped configs run web search server-side: the model emits a
``web_search`` / ``fetch_url`` tool_use, we execute it via an injected
``SearchProvider``, feed a tool_result back, and loop until the model stops.
All offline: the Anthropic SDK client and the SearchProvider are fakes.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from minds.common.search.base import FetchedContent, SearchResult
from minds.inference.providers import fireworks as fireworks_mod
from minds.inference.providers.fireworks import build_search_provider_for_request, proxy_fireworks
from minds.inference.types import ApiKind, PassthroughModelConfig, UsageBox, WebSearchMode


def _external_config() -> PassthroughModelConfig:
    return PassthroughModelConfig(
        api_kind=ApiKind.FIREWORKS,
        model_name="accounts/fireworks/models/kimi-k2p6",
        api_key="test-key",
        base_url="https://api.fireworks.ai/inference",
        web_search_mode=WebSearchMode.EXTERNAL_TOOL,
        label="fireworks",
    )


def _tool_use_response(*, tool_id="tu_1", name="web_search", inp=None, in_tok=10, out_tok=5):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id=tool_id, name=name, input=inp or {"query": "weather"})],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
    )


def _final_response(*, text="final answer", in_tok=7, out_tok=3):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
    )


def _fake_provider(*, search_results=None, fetch_content=None, search_error=None):
    prov = SimpleNamespace()
    if search_error is not None:
        prov.search = AsyncMock(side_effect=search_error)
    else:
        prov.search = AsyncMock(
            return_value=search_results or [SearchResult(title="T", url="https://u", snippet="snip")]
        )
    prov.fetch = AsyncMock(
        return_value=fetch_content or FetchedContent(url="https://u", title="T", text="body", truncated=False)
    )
    return prov


def _fake_client(create_mock):
    return SimpleNamespace(messages=SimpleNamespace(create=create_mock))


async def _run_proxy(*, create_mock, provider, tools, stream=False, messages=None):
    """Drive proxy_fireworks with fakes; return (response, usage_box)."""
    usage_box = UsageBox()
    response = await proxy_fireworks(
        client=_fake_client(create_mock),
        config=_external_config(),
        usage_box=usage_box,
        messages=messages or [{"role": "user", "content": "hi"}],
        stream=stream,
        request_id="req",
        tools=tools,
        search_provider=provider,
    )
    return response, usage_box


async def test_loop_executes_search_then_returns_final():
    create_mock = AsyncMock(side_effect=[_tool_use_response(), _final_response()])
    provider = _fake_provider()

    resp, usage_box = await _run_proxy(create_mock=create_mock, provider=provider, tools=[{"type": "web_search"}])

    body = json.loads(bytes(resp.body))
    assert body["choices"][0]["message"]["content"] == "final answer"
    provider.search.assert_awaited_once()
    assert create_mock.await_count == 2
    # Usage is summed across both turns, not just the final one.
    assert usage_box.value == (17, 8)


async def test_loop_feeds_tool_result_keyed_by_id():
    create_mock = AsyncMock(side_effect=[_tool_use_response(tool_id="tu_42"), _final_response()])

    await _run_proxy(create_mock=create_mock, provider=_fake_provider(), tools=[{"type": "web_search"}])

    # Second call's messages must include the assistant tool_use turn followed
    # by a user tool_result keyed to the same id.
    second_messages = create_mock.await_args_list[1].kwargs["messages"]
    tool_results = [
        blk
        for msg in second_messages
        if isinstance(msg.get("content"), list)
        for blk in msg["content"]
        if isinstance(blk, dict) and blk.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_use_id"] == "tu_42"


async def test_loop_hits_iteration_cap_then_forces_final(monkeypatch):
    monkeypatch.setattr(fireworks_mod._settings.search, "max_iterations", 2)
    # Model never stops requesting tools.
    create_mock = AsyncMock(return_value=_tool_use_response())

    resp, _ = await _run_proxy(create_mock=create_mock, provider=_fake_provider(), tools=[{"type": "web_search"}])

    # max_iterations (2) loop calls + 1 forced tool-less final = 3.
    assert create_mock.await_count == 3
    # The forced final call must omit tools so the model commits to an answer.
    assert "tools" not in create_mock.await_args_list[-1].kwargs
    assert getattr(resp, "status_code", 200) == 200


async def test_loop_provider_error_is_recoverable():
    create_mock = AsyncMock(side_effect=[_tool_use_response(), _final_response()])
    provider = _fake_provider(search_error=RuntimeError("exa down"))

    await _run_proxy(create_mock=create_mock, provider=provider, tools=[{"type": "web_search"}])

    # Error fed back as a tool_result string; loop still reached the final turn.
    second_messages = create_mock.await_args_list[1].kwargs["messages"]
    tool_result = next(
        blk
        for msg in second_messages
        if isinstance(msg.get("content"), list)
        for blk in msg["content"]
        if isinstance(blk, dict) and blk.get("type") == "tool_result"
    )
    assert "Search failed" in tool_result["content"]
    assert create_mock.await_count == 2


async def test_loop_records_server_artifacts():
    create_mock = AsyncMock(side_effect=[_tool_use_response(), _final_response()])
    provider = _fake_provider(search_results=[SearchResult(title="T1", url="https://one", snippet="s")])

    _, usage_box = await _run_proxy(create_mock=create_mock, provider=provider, tools=[{"type": "web_search"}])

    artifacts = usage_box.server_artifacts
    assert any(a["type"] == "external_search" and a["tool"] == "web_search" for a in artifacts)
    search_artifact = next(a for a in artifacts if a["type"] == "external_search")
    assert search_artifact["results"] == [{"title": "T1", "url": "https://one"}]


async def test_streaming_replay_populates_usagebox():
    create_mock = AsyncMock(side_effect=[_tool_use_response(), _final_response(text="streamed answer")])

    resp, usage_box = await _run_proxy(
        create_mock=create_mock, provider=_fake_provider(), tools=[{"type": "web_search"}], stream=True
    )

    chunks = [chunk async for chunk in resp.body_iterator]
    joined = "".join(chunks)
    assert "streamed answer" in joined
    assert joined.rstrip().endswith("data: [DONE]")
    # UsageBox is populated (the handler reads it after draining the stream).
    assert usage_box.value == (17, 8)
    assert usage_box.output_payload["content"] == "streamed answer"


async def test_fetch_tool_dispatches_to_provider_fetch():
    create_mock = AsyncMock(
        side_effect=[_tool_use_response(name="fetch_url", inp={"url": "https://x"}), _final_response()]
    )
    provider = _fake_provider()

    await _run_proxy(create_mock=create_mock, provider=provider, tools=[{"type": "fetch"}])

    provider.fetch.assert_awaited_once()
    provider.search.assert_not_awaited()


async def test_no_provider_drops_web_tools_single_shot():
    # No search provider (e.g. exa key unset): the request degrades to a
    # single-shot call with the web tool dropped — never injects a function
    # tool the model could call but nobody would execute.
    create_mock = AsyncMock(return_value=_final_response(text="no search"))

    resp, usage_box = await _run_proxy(create_mock=create_mock, provider=None, tools=[{"type": "web_search"}])

    assert create_mock.await_count == 1
    assert "tools" not in create_mock.await_args_list[0].kwargs
    body = json.loads(bytes(resp.body))
    assert body["choices"][0]["message"]["content"] == "no search"
    assert usage_box.value == (7, 3)


def test_build_provider_none_without_web_tool(monkeypatch):
    # A Fireworks request with no web tool must NOT build a provider, so it
    # never requires an Exa key (regression: plain Fireworks chat).
    sentinel = object()
    monkeypatch.setattr("minds.common.search.get_search_provider", lambda _s: sentinel)

    assert build_search_provider_for_request(_external_config(), None) is None
    function_tool = [{"type": "function", "function": {"name": "f"}}]
    assert build_search_provider_for_request(_external_config(), function_tool) is None


def test_build_provider_built_when_web_tool_present(monkeypatch):
    sentinel = object()
    monkeypatch.setattr("minds.common.search.get_search_provider", lambda _s: sentinel)

    assert build_search_provider_for_request(_external_config(), [{"type": "web_search"}]) is sentinel


def test_build_provider_degrades_on_construction_error(monkeypatch):
    # A misconfigured provider (e.g. missing exa key) must not 5xx the request:
    # build_search_provider_for_request swallows the error and returns None.
    def _boom(_s):
        raise ValueError("SEARCH__EXA_API_KEY is not set")

    monkeypatch.setattr("minds.common.search.get_search_provider", _boom)

    assert build_search_provider_for_request(_external_config(), [{"type": "web_search"}]) is None
