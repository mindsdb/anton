from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_llm_client_plan_and_code_and_plan_stream():
    from minds.agents.anton_agent.anton.llm.client import LLMClient
    from minds.agents.anton_agent.anton.llm.provider import LLMResponse, StreamComplete, StreamTextDelta, Usage

    class P:
        async def complete(self, *, model, system, messages, tools=None, tool_choice=None, max_tokens=4096):
            return LLMResponse(
                content=f"{model}:{system}:{len(messages)}", usage=Usage(input_tokens=1, output_tokens=2)
            )

        async def stream(self, *, model, system, messages, tools=None, max_tokens=4096):
            yield StreamTextDelta(text="x")
            yield StreamComplete(response=LLMResponse(content="x"))

    llm = LLMClient(
        planning_provider=P(),
        planning_model="plan",
        coding_provider=P(),
        coding_model="code",
        max_tokens=10,
    )
    r1 = await llm.plan(system="s", messages=[{"role": "user", "content": "hi"}])
    assert r1.content.startswith("plan:")
    r2 = await llm.code(system="s", messages=[])
    assert r2.content.startswith("code:")
    events = [e async for e in llm.plan_stream(system="s", messages=[])]
    assert isinstance(events[-1], StreamComplete)
