from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_llm_provider_default_stream_yields_delta_and_complete():
    from minds.agents.anton_agent.anton.llm.provider import LLMProvider, StreamComplete, StreamTextDelta

    class DummyProvider(LLMProvider):
        async def complete(self, *, model, system, messages, tools=None, tool_choice=None, max_tokens=4096):
            from minds.agents.anton_agent.anton.llm.provider import LLMResponse

            return LLMResponse(content="hello")

    provider = DummyProvider()
    events = [e async for e in provider.stream(model="x", system="", messages=[])]
    assert isinstance(events[0], StreamTextDelta)
    assert events[0].text == "hello"
    assert isinstance(events[1], StreamComplete)


def test_compute_context_pressure_uses_model_prefix():
    from minds.agents.anton_agent.anton.llm.provider import compute_context_pressure

    assert compute_context_pressure("gpt-4o-mini", 64_000) == pytest.approx(0.5)
    assert compute_context_pressure("unknown-model", 999_999) == 1.0
