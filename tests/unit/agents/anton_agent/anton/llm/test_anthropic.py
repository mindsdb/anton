from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_anthropic_provider_complete_parses_text_and_tool_use(monkeypatch):
    from minds.agents.anton_agent.anton.llm.anthropic import AnthropicProvider

    class _BlockText:
        type = "text"

        def __init__(self, text):
            self.text = text

    class _BlockTool:
        type = "tool_use"

        def __init__(self, _id, name, input_):
            self.id = _id
            self.name = name
            self.input = input_

    class _Usage:
        input_tokens = 3
        output_tokens = 4

    class _Resp:
        content = [_BlockText("hi"), _BlockTool("t1", "recall", {"query": "x"})]
        usage = _Usage()
        stop_reason = "end_turn"

    class _Msgs:
        async def create(self, **_kwargs):
            return _Resp()

    class _AsyncAnthropic:
        def __init__(self, **_kwargs):
            self.messages = _Msgs()

    import minds.agents.anton_agent.anton.llm.anthropic as ant_mod

    monkeypatch.setattr(ant_mod.anthropic, "AsyncAnthropic", _AsyncAnthropic)

    p = AnthropicProvider(api_key="k")
    resp = await p.complete(model="claude-3", system="s", messages=[{"role": "user", "content": "hi"}])
    assert resp.content == "hi"
    assert resp.tool_calls[0].id == "t1"
    assert resp.tool_calls[0].name == "recall"
    assert resp.tool_calls[0].input == {"query": "x"}


@pytest.mark.asyncio
async def test_anthropic_provider_stream_emits_tool_events_and_complete(monkeypatch):
    from minds.agents.anton_agent.anton.llm.anthropic import AnthropicProvider
    from minds.agents.anton_agent.anton.llm.provider import (
        StreamComplete,
        StreamTextDelta,
        StreamToolUseDelta,
        StreamToolUseEnd,
        StreamToolUseStart,
    )

    class _Event:
        def __init__(self, type_, **kwargs):
            self.type = type_
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _StreamCM:
        async def __aenter__(self):
            async def _iter():
                yield _Event(
                    "message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=3, output_tokens=0))
                )
                yield _Event(
                    "content_block_start",
                    index=0,
                    content_block=SimpleNamespace(type="tool_use", id="t1", name="recall"),
                )
                yield _Event(
                    "content_block_delta",
                    index=0,
                    delta=SimpleNamespace(type="input_json_delta", partial_json='{"query":"x"}'),
                )
                yield _Event("content_block_stop", index=0)
                yield _Event("content_block_start", index=1, content_block=SimpleNamespace(type="text"))
                yield _Event(
                    "content_block_delta",
                    index=1,
                    delta=SimpleNamespace(type="text_delta", text="hi"),
                )
                yield _Event(
                    "message_delta", delta=SimpleNamespace(stop_reason="stop"), usage=SimpleNamespace(output_tokens=2)
                )

            return _iter()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _Msgs:
        def stream(self, **_kwargs):
            return _StreamCM()

    class _AsyncAnthropic:
        def __init__(self, **_kwargs):
            self.messages = _Msgs()

    import minds.agents.anton_agent.anton.llm.anthropic as ant_mod

    monkeypatch.setattr(ant_mod.anthropic, "AsyncAnthropic", _AsyncAnthropic)

    p = AnthropicProvider(api_key="k")
    events = [e async for e in p.stream(model="claude-3", system="", messages=[{"role": "user", "content": "hi"}])]
    assert any(isinstance(e, StreamToolUseStart) and e.id == "t1" for e in events)
    assert any(isinstance(e, StreamToolUseDelta) and e.id == "t1" for e in events)
    assert any(isinstance(e, StreamToolUseEnd) and e.id == "t1" for e in events)
    assert any(isinstance(e, StreamTextDelta) and e.text == "hi" for e in events)
    assert any(isinstance(e, StreamComplete) for e in events)


@pytest.mark.asyncio
async def test_anthropic_provider_complete_raises_context_overflow(monkeypatch):
    import minds.agents.anton_agent.anton.llm.anthropic as ant_mod
    from minds.agents.anton_agent.anton.llm.anthropic import AnthropicProvider
    from minds.agents.anton_agent.anton.llm.provider import ContextOverflowError

    class BadRequestError(Exception):
        pass

    class _Msgs:
        async def create(self, **_kwargs):
            raise BadRequestError("prompt is too long")

    class _AsyncAnthropic:
        def __init__(self, **_kwargs):
            self.messages = _Msgs()

    monkeypatch.setattr(ant_mod.anthropic, "AsyncAnthropic", _AsyncAnthropic)
    monkeypatch.setattr(ant_mod.anthropic, "BadRequestError", BadRequestError)

    p = AnthropicProvider(api_key="k")
    with pytest.raises(ContextOverflowError):
        await p.complete(model="claude-3", system="", messages=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_anthropic_provider_stream_raises_context_overflow(monkeypatch):
    import minds.agents.anton_agent.anton.llm.anthropic as ant_mod
    from minds.agents.anton_agent.anton.llm.anthropic import AnthropicProvider
    from minds.agents.anton_agent.anton.llm.provider import ContextOverflowError

    class BadRequestError(Exception):
        pass

    class _StreamCM:
        async def __aenter__(self):
            raise BadRequestError("context limit")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _Msgs:
        def stream(self, **_kwargs):
            return _StreamCM()

    class _AsyncAnthropic:
        def __init__(self, **_kwargs):
            self.messages = _Msgs()

    monkeypatch.setattr(ant_mod.anthropic, "AsyncAnthropic", _AsyncAnthropic)
    monkeypatch.setattr(ant_mod.anthropic, "BadRequestError", BadRequestError)

    p = AnthropicProvider(api_key="k")
    with pytest.raises(ContextOverflowError):
        async for _ in p.stream(model="claude-3", system="", messages=[{"role": "user", "content": "hi"}]):
            pass
