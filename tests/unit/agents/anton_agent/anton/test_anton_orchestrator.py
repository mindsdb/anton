from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_anton_orchestrator_wires_session_and_repairs_history(monkeypatch, tmp_path):
    from minds.agents.anton_agent.anton.anton import Anton

    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.scratchpad_manager.ScratchpadManager.probe_packages",
        staticmethod(lambda: []),
    )

    history = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "recall", "input": {"query": "x"}}],
        }
    ]
    a = Anton(
        workspace_dir=str(tmp_path),
        runtime_context="rc",
        history=history,
        backend="docker",
        planning_provider="openai",
        planning_model="gpt-4o",
        planning_api_key="k",
        coding_provider="openai",
        coding_model="gpt-4o",
        coding_api_key="k",
        extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"},
    )

    assert a.session.history
    assert a.session.history[-1]["role"] == "user"
    await a.close()


@pytest.mark.asyncio
async def test_anton_orchestrator_anthropic_branch(tmp_path, monkeypatch):
    from minds.agents.anton_agent.anton.anton import Anton

    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.scratchpad_manager.ScratchpadManager.probe_packages",
        staticmethod(lambda: []),
    )
    a = Anton(
        workspace_dir=str(tmp_path),
        runtime_context="rc",
        history=None,
        backend="docker",
        planning_provider="anthropic",
        planning_model="claude-opus-4-6",
        planning_api_key="k",
        coding_provider="anthropic",
        coding_model="claude-haiku-4-5-20251001",
        coding_api_key="k",
        extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"},
    )
    await a.close()
