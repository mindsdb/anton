from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_tools_dispatch_recall_memorize_and_scratchpad_actions(tmp_path):
    from minds.agents.anton_agent.anton.tools import dispatch_tool, format_cell_result, prepare_scratchpad_exec

    session = SimpleNamespace()
    session._episodic = SimpleNamespace(enabled=True, recall_formatted=lambda q, **_: f"found:{q}")
    session._cortex = SimpleNamespace(
        mode="autopilot", encode=AsyncMock(return_value=["ok"]), encoding_gate=lambda _e: False
    )

    class _Pad:
        async def install_packages(self, packages):
            return f"Packages {', '.join(packages)} installed."

        async def execute(self, code, *, description="", estimated_time="", estimated_seconds=0):
            return SimpleNamespace(
                code=code, stdout="out", stderr="", error=None, description=description, estimated_time=estimated_time
            )

        def view(self):
            return "view"

        async def reset(self):
            return None

        def render_notebook(self):
            return "dump"

    pad = _Pad()
    session._scratchpads = SimpleNamespace(
        get_or_create=AsyncMock(return_value=pad),
        get=lambda: pad,
        remove=AsyncMock(return_value="Scratchpad removed."),
    )

    assert await dispatch_tool(session, "recall", {"query": "x"}) == "found:x"
    out = await dispatch_tool(
        session, "memorize", {"entries": [{"text": "Use progress()", "kind": "always", "scope": "project"}]}
    )
    assert out.startswith("Memory updated:")

    exec_out = await dispatch_tool(
        session,
        "scratchpad",
        {"action": "exec", "code": "print(1)", "one_line_description": "do", "estimated_execution_time_seconds": 3},
    )
    assert "[output]" in exec_out

    assert await dispatch_tool(session, "scratchpad", {"action": "view"}) == "view"
    assert await dispatch_tool(session, "scratchpad", {"action": "dump"}) == "dump"
    assert await dispatch_tool(session, "scratchpad", {"action": "install", "packages": ["x"]})
    assert await dispatch_tool(session, "scratchpad", {"action": "remove"}) == "Scratchpad removed."
    assert await dispatch_tool(session, "scratchpad", {"action": "reset"}) == "Scratchpad reset. All state cleared."

    assert await prepare_scratchpad_exec(session, {"code": ""}) == "No code provided."
    assert (
        format_cell_result(SimpleNamespace(stdout="", stderr="", error=None))
        == "Code executed successfully (no output)."
    )


@pytest.mark.asyncio
async def test_tools_more_branches():
    from minds.agents.anton_agent.anton.tools import dispatch_tool

    sess = SimpleNamespace(_episodic=SimpleNamespace(enabled=False), _cortex=None, _scratchpads=None)
    assert await dispatch_tool(sess, "recall", {"query": "x"}) == "Episodic memory is not available."
    assert (
        await dispatch_tool(sess, "memorize", {"entries": [{"text": "x", "kind": "always", "scope": "project"}]})
        == "Memory system not available."
    )

    sess2 = SimpleNamespace(
        _scratchpads=SimpleNamespace(get=lambda: None, get_or_create=AsyncMock(), remove=AsyncMock()),
        _episodic=None,
        _cortex=None,
    )
    assert await dispatch_tool(sess2, "scratchpad", {"action": "view"}) == "No scratchpad available for this session."
    assert await dispatch_tool(sess2, "scratchpad", {"action": "reset"}) == "No scratchpad available for this session."
    assert await dispatch_tool(sess2, "scratchpad", {"action": "dump"}) == "No scratchpad available for this session."
    assert await dispatch_tool(sess2, "scratchpad", {"action": "install"}) == "No packages specified."
    assert "Unknown scratchpad action" in await dispatch_tool(sess2, "scratchpad", {"action": "wat"})
    assert "Unknown tool" in await dispatch_tool(sess2, "nope", {})


@pytest.mark.asyncio
async def test_tools_prepare_scratchpad_exec_install_fail():
    from minds.agents.anton_agent.anton.tools import prepare_scratchpad_exec

    class Pad:
        async def install_packages(self, packages):
            return "Install failed: nope"

    sess = SimpleNamespace(_scratchpads=SimpleNamespace(get_or_create=AsyncMock(return_value=Pad())))
    out = await prepare_scratchpad_exec(sess, {"code": "x", "packages": ["bad"]})
    assert out == "Install failed: nope"


@pytest.mark.asyncio
async def test_tools_edge_cases_cover_more():
    from minds.agents.anton_agent.anton.tools.tool_handlers import handle_memorize, handle_recall

    sess = SimpleNamespace(_episodic=SimpleNamespace(enabled=True, recall_formatted=lambda q, **_: q), _cortex=None)
    assert await handle_recall(sess, {"query": ""}) == "No query provided."
    assert await handle_memorize(sess, {"entries": []}) == "Memory system not available."
