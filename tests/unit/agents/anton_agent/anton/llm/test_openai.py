from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_openai_translation_helpers_cover_blocks_and_tool_choice():
    from minds.agents.anton_agent.anton.llm.openai import _translate_messages, _translate_tool_choice, _translate_tools

    tools = [
        {
            "name": "scratchpad",
            "description": "Run",
            "input_schema": {"type": "object", "properties": {"action": {"type": "string"}}},
        }
    ]
    assert _translate_tools(tools)[0]["function"]["name"] == "scratchpad"
    assert _translate_tool_choice({"type": "tool", "name": "scratchpad"}) == {
        "type": "function",
        "function": {"name": "scratchpad"},
    }
    assert _translate_tool_choice({"type": "any"}) == "required"
    assert _translate_tool_choice({"type": "auto"}) == "auto"

    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "id": "t1", "name": "scratchpad", "input": {"action": "view"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "ok"},
                {"type": "tool_result", "tool_use_id": "t1", "content": "done"},
            ],
        },
    ]
    out = _translate_messages("SYS", messages)
    assert out[0]["role"] == "system"
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in out)
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "t1" for m in out)


@pytest.mark.asyncio
async def test_openai_provider_complete_parses_tool_calls_without_network(monkeypatch):
    from minds.agents.anton_agent.anton.llm.openai import OpenAIProvider

    class _Fn:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class _ToolCall:
        def __init__(self, _id, fn):
            self.id = _id
            self.function = fn

    class _Msg:
        def __init__(self):
            self.content = "ok"
            self.tool_calls = [_ToolCall("tc1", _Fn("scratchpad", '{"action":"view"}'))]

    class _Choice:
        def __init__(self):
            self.message = _Msg()
            self.finish_reason = "stop"

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 2

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    async def _create(**_kwargs):
        return _Resp()

    class _Chat:
        class completions:
            create = staticmethod(_create)

    class _AsyncOpenAI:
        def __init__(self, **_kwargs):
            self.chat = _Chat()

    import minds.agents.anton_agent.anton.llm.openai as oai_mod

    monkeypatch.setattr(oai_mod.openai, "AsyncOpenAI", _AsyncOpenAI)

    p = OpenAIProvider(api_key="k")
    resp = await p.complete(model="gpt-4o", system="", messages=[{"role": "user", "content": "hi"}])
    assert resp.content == "ok"
    assert resp.tool_calls[0].name == "scratchpad"
    assert resp.tool_calls[0].input == {"action": "view"}
    assert resp.usage.input_tokens == 10


@pytest.mark.asyncio
async def test_openai_provider_stream_emits_tool_events_and_complete(monkeypatch):
    from minds.agents.anton_agent.anton.llm.openai import OpenAIProvider
    from minds.agents.anton_agent.anton.llm.provider import (
        StreamComplete,
        StreamToolUseDelta,
        StreamToolUseEnd,
        StreamToolUseStart,
    )

    class _TCDelta:
        def __init__(self, index, _id=None, name=None, args=None):
            self.index = index
            self.id = _id
            self.function = SimpleNamespace(name=name) if name is not None else None
            if self.function is not None and args is not None:
                self.function.arguments = args

    class _Delta:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class _Choice:
        def __init__(self, delta, finish_reason=None):
            self.delta = delta
            self.finish_reason = finish_reason

    class _Chunk:
        def __init__(self, delta=None, finish_reason=None, usage=None):
            self.choices = [_Choice(delta, finish_reason)] if delta is not None else []
            self.usage = usage

    class _Usage:
        def __init__(self, p, c):
            self.prompt_tokens = p
            self.completion_tokens = c

    async def _stream_iter():
        yield _Chunk(
            delta=_Delta(
                tool_calls=[_TCDelta(0, _id="t1", name="scratchpad", args='{"action":"exec","code":"print(1)"}')]
            )
        )
        yield _Chunk(delta=_Delta(content="x"), finish_reason="stop", usage=_Usage(5, 2))

    async def _create(**_kwargs):
        return _stream_iter()

    class _Chat:
        class completions:
            create = staticmethod(_create)

    class _AsyncOpenAI:
        def __init__(self, **_kwargs):
            self.chat = _Chat()

    import minds.agents.anton_agent.anton.llm.openai as oai_mod

    monkeypatch.setattr(oai_mod.openai, "AsyncOpenAI", _AsyncOpenAI)

    p = OpenAIProvider(api_key="k")
    events = [e async for e in p.stream(model="gpt-4o", system="", messages=[{"role": "user", "content": "hi"}])]
    assert any(isinstance(e, StreamToolUseStart) and e.id == "t1" for e in events)
    assert any(isinstance(e, StreamToolUseDelta) and e.id == "t1" for e in events)
    assert any(isinstance(e, StreamToolUseEnd) and e.id == "t1" for e in events)
    assert any(isinstance(e, StreamComplete) for e in events)


@pytest.mark.asyncio
async def test_openai_provider_complete_raises_context_overflow(monkeypatch):
    import minds.agents.anton_agent.anton.llm.openai as oai_mod
    from minds.agents.anton_agent.anton.llm.openai import OpenAIProvider
    from minds.agents.anton_agent.anton.llm.provider import ContextOverflowError

    class BadRequestError(Exception):
        pass

    class _Chat:
        class completions:
            @staticmethod
            async def create(**_kwargs):
                raise BadRequestError("maximum context length exceeded")

    class _AsyncOpenAI:
        def __init__(self, **_kwargs):
            self.chat = _Chat()

    monkeypatch.setattr(oai_mod.openai, "AsyncOpenAI", _AsyncOpenAI)
    monkeypatch.setattr(oai_mod.openai, "BadRequestError", BadRequestError)

    p = OpenAIProvider(api_key="k")
    with pytest.raises(ContextOverflowError):
        await p.complete(model="gpt-4o", system="", messages=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_openai_translate_user_blocks_image_branch():
    from minds.agents.anton_agent.anton.llm.openai import _translate_user_blocks

    blocks = [
        {"type": "text", "text": "hi"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "AAA"},
        },
    ]
    msgs = _translate_user_blocks(blocks)
    assert msgs[0]["role"] == "user"
    assert isinstance(msgs[0]["content"], list)
    assert any(p.get("type") == "image_url" for p in msgs[0]["content"])


@pytest.mark.asyncio
async def test_openai_stream_tool_call_id_name_late(monkeypatch):
    from minds.agents.anton_agent.anton.llm.openai import OpenAIProvider

    class _TCDelta:
        def __init__(self, index, _id=None, name=None, args=None):
            self.index = index
            self.id = _id
            self.function = (
                SimpleNamespace(name=name, arguments=args) if (name is not None or args is not None) else None
            )

    class _Delta:
        def __init__(self, tool_calls=None):
            self.tool_calls = tool_calls
            self.content = None

    class _Choice:
        def __init__(self, delta, finish_reason=None):
            self.delta = delta
            self.finish_reason = finish_reason

    class _Chunk:
        def __init__(self, delta=None, finish_reason=None, usage=None):
            self.choices = [_Choice(delta, finish_reason)] if delta is not None else []
            self.usage = usage

    class _Usage:
        def __init__(self, p, c):
            self.prompt_tokens = p
            self.completion_tokens = c

    async def _stream_iter():
        yield _Chunk(delta=_Delta(tool_calls=[_TCDelta(0, args='{"a":')]))
        yield _Chunk(delta=_Delta(tool_calls=[_TCDelta(0, _id="t1", name="recall", args='"b"}')]))
        yield _Chunk(delta=SimpleNamespace(content="x", tool_calls=None), finish_reason="stop", usage=_Usage(1, 1))

    async def _create(**_kwargs):
        return _stream_iter()

    class _Chat:
        class completions:
            create = staticmethod(_create)

    class _AsyncOpenAI:
        def __init__(self, **_kwargs):
            self.chat = _Chat()

    import minds.agents.anton_agent.anton.llm.openai as oai_mod

    monkeypatch.setattr(oai_mod.openai, "AsyncOpenAI", _AsyncOpenAI)

    p = OpenAIProvider(api_key="k")
    events = [e async for e in p.stream(model="gpt-4o", system="", messages=[{"role": "user", "content": "hi"}])]
    assert events
