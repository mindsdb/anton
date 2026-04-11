from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.core.session import ChatSession
    from anton.core.tools.tool_defs import ToolDef


class ToolRegistry:
    """
    Registry of tools available to the LLM.
    """

    def __init__(self) -> None:
        self._tools: list[ToolDef] = []

    def __bool__(self) -> bool:
        return bool(self._tools)

    def register_tool(self, tool_def: "ToolDef") -> None:
        """Register a tool. Skips duplicates by name."""
        if any(t.name == tool_def.name for t in self._tools):
            return
        self._tools.append(tool_def)

    def get_tool_defs(self) -> list["ToolDef"]:
        """Return registered ToolDef objects (for prompt injection, etc.)."""
        return list(self._tools)

    async def dispatch_tool(
        self, session: "ChatSession", tool_name: str, tc_input: dict
    ) -> str:
        """Dispatch a tool call by name. Returns result text."""
        tool_def = next((t for t in self._tools if t.name == tool_name), None)
        if tool_def is None:
            raise ValueError(f"Tool {tool_name} not found")
        return await tool_def.handler(session, tc_input)

    def dump(self) -> list[dict]:
        """
        Dump the registry as a list of LLM-facing tool schemas.
        Excludes handler and prompt — those are internal only.
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools
        ]
