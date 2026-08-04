"""Minimal analog of the `generate_artifact` FSM (see
`anton/core/tools/generate_artifact/`): a tool whose handler streams progress
via the ToolProgress protocol (see `anton/core/tools/progress.py`) instead of
doing everything in one shot. This one is a stub for exercising that
protocol — three sequential steps, each reporting its own name before and
after a one-second delay.

Self-contained on purpose: delete this file and the two lines that reference
`TEST_TOOL`/`test_tool` in `anton/core/session.py` (`_build_core_tools`) to
remove it entirely.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from anton.core.tools.progress import ToolProgress
from anton.core.tools.tool_defs import ToolDef

if TYPE_CHECKING:
    from anton.core.session import ChatSession


STEP_NAMES = ["step_1", "step_2", "step_3"]


async def handle_test_tool(session: "ChatSession", tc_input: dict):
    steps = []
    for name in STEP_NAMES:
        yield ToolProgress(f"{name} executing")
        await asyncio.sleep(1)
        steps.append({"step": name})
        yield ToolProgress(f"{name} done")
    yield json.dumps({"steps": steps}, indent=2)


TEST_TOOL = ToolDef(
    name="test_tool",
    description=(
        "Diagnostic no-op tool: runs three internal steps, each reporting "
        "progress before and after a one-second delay, and returns their "
        "names. Only call this tool if the user explicitly asks for it by "
        "name in their message (e.g. \"call test_tool\", \"run the test "
        "tool\") — never as part of normal task handling."
    ),
    input_schema={"type": "object", "properties": {}},
    handler=handle_test_tool,
)
