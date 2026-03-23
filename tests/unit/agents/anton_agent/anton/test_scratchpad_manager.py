from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_scratchpad_manager_get_or_create_remove_and_close(monkeypatch, tmp_path):
    from minds.agents.anton_agent.anton.scratchpad_manager import ScratchpadManager

    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.scratchpad_manager.ScratchpadManager.probe_packages",
        staticmethod(lambda: ["pytest"]),
    )

    class _Pad:
        def __init__(self):
            self.started = False
            self.closed = False
            self.cancelled = False

        async def start(self):
            self.started = True

        async def close(self, cleanup=False):
            self.closed = True

        async def cancel(self):
            self.cancelled = True

    pad = _Pad()

    class _Factory:
        def create(self, **_kwargs):
            return pad

    monkeypatch.setattr(
        "minds.agents.anton_agent.anton.scratchpad_manager.ScratchpadRuntimeFactory", lambda: _Factory()
    )

    mgr = ScratchpadManager(
        backend="docker",
        coding_provider="anthropic",
        coding_model="m",
        coding_api_key="k",
        workspace_path=tmp_path,
        extra_env={"ANTON_MINDS_CONVERSATION_ID": "c1"},
    )
    got = await mgr.get_or_create()
    assert got is pad and pad.started
    got2 = await mgr.get_or_create()
    assert got2 is pad

    await mgr.cancel_all_running()
    assert pad.cancelled
    assert await mgr.remove() == "Scratchpad removed."
    await mgr.close_all()
    assert mgr.get() is None
