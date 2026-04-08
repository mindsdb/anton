from __future__ import annotations
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.core.session import ChatSession
    from anton.core.tools.tool_defs import ToolDef


class ToolRegistry:
    """
    Registry of tools available to the LLM.
    """
    def __init__(self) -> None:
        # Register core tools.
        self._tools = []

    def __bool__(self) -> bool:
        """
        Return True if there are any tools registered.
        """
        return bool(self._tools)

    def register_tool(self, tool_def: ToolDef) -> None:
        """
        Register a new (extra to core) tool.
        """
        self._tools.append(tool_def)

    async def dispatch_tool(
        self, session: "ChatSession", tool_name: str, tc_input: dict
    ) -> str:
        """
        Dispatch a tool call by name. Returns result text.
        """
        tool_def = next((tool for tool in self._tools if tool.name == tool_name), None)
        if tool_def is None:
            raise ValueError(f"Tool {tool_name} not found")
        return await tool_def.handler(session, tc_input)

    def dump(self) -> list[dict]:
        """
        Dump the registry as a list of tool definitions.
        This is used to build the tools list for the LLM. As a result, the handler is not needed.
        """
        tool_defs = []
        for tool_def in self._tools:
            # Remove the handler from the tool definition.
            tool_def = asdict(tool_def)
            tool_def.pop("handler")
            tool_defs.append(tool_def)
        return tool_defs
