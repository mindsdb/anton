"""Minimal analog of the `generate_artifact` FSM (see
`anton/core/tools/generate_artifact/`): a tool that internally runs several
steps, each in its own isolated context, rather than doing everything in one
LLM turn. This one is a stub for exercising that structure — three steps,
each just sleeps and reports its own name.

Self-contained on purpose: delete this file and the two lines that reference
`TEST_TOOL`/`test_tool` in `anton/core/session.py` (`_build_core_tools`) to
remove it entirely.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from anton.core.tools.tool_defs import ToolDef

if TYPE_CHECKING:
    from anton.core.session import ChatSession


STEP_NAMES = ["step_1", "step_2", "step_3"]


async def _run_step(name: str) -> dict:
    """One isolated step. No state is shared with the other steps."""
    await asyncio.sleep(1)
    return {"step": name}


async def handle_test_tool(session: "ChatSession", tc_input: dict) -> str:
    steps = [await _run_step(name) for name in STEP_NAMES]
    return json.dumps({"steps": steps}, indent=2)


TEST_TOOL = ToolDef(
    name="test_tool",
    description=(
        "Diagnostic no-op tool: runs three internal steps, each in its own "
        "isolated context, and returns their names. Only call this tool if the "
        "user explicitly asks for it by name in their message (e.g. \"call "
        "test_tool\", \"run the test tool\") — never as part of normal task "
        "handling."
    ),
    input_schema={"type": "object", "properties": {}},
    handler=handle_test_tool,
)
