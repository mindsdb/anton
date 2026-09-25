"""Progress marker yielded by a streaming tool handler.

Lives in its own leaf module rather than tool_defs.py: tool_defs.py imports
tool_handlers.py at module scope, which pulls in anton.core.backends.base,
anton.core.utils.scratchpad and other heavy dependencies. registry.py imports
ToolDef only under TYPE_CHECKING (a deliberate boundary so the low-level
dispatcher doesn't drag in that chain) but needs ToolProgress at runtime for
isinstance checks — importing it from here keeps that boundary intact.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolProgress:
    """`kind` says how the text is shown: a `"step"` line is printed once and
    stays; a `"peek"` is the live tail of what the tool is producing right
    now and replaces the previous peek in the transient footer (relayed as
    the `tool_peek` phase — see `ToolRegistry.dispatch_tool`)."""

    text: str
    kind: str = "step"
