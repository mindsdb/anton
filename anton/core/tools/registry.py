from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from anton.core.tools.progress import ToolProgress

if TYPE_CHECKING:
    from anton.core.session import ChatSession
    from anton.core.tools.tool_defs import ToolDef


class ToolRegistry:
    """
    Registry of tools available to the LLM.
    """

    def __init__(self) -> None:
        self._tools: list["ToolDef"] = []

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

    def _find_tool(self, tool_name: str) -> "ToolDef":
        tool_def = next((t for t in self._tools if t.name == tool_name), None)
        if tool_def is None:
            raise ValueError(f"Tool {tool_name} not found")
        return tool_def

    async def dispatch_tool_stream(
        self, session: "ChatSession", tool_name: str, tc_input: dict
    ):
        """Dispatch a tool call, forwarding ToolProgress markers as they arrive.

        For a plain-coroutine handler, yields exactly one item — its result.
        For an async-generator handler, forwards every ToolProgress it yields,
        then its final (non-ToolProgress) item, in order.

        Raises RuntimeError if a generator handler completes without ever
        yielding a non-ToolProgress item — that is a bug in the tool, and
        silently returning None here would crash scrub_credentials() further
        up the call chain in session.py instead of surfacing a clean error.

        The type check is on the CALLED handler's return value
        (inspect.isasyncgen(obj)), not on the handler function itself
        (inspect.isasyncgenfunction would misclassify a functools.partial,
        lambda, or decorator wrapping a generator — ToolDef.handler is typed
        as a plain Callable, so such wrappers are already permitted).
        """
        tool_def = self._find_tool(tool_name)
        obj = tool_def.handler(session, tc_input)
        if inspect.isasyncgen(obj):
            seen_result = False
            async for item in obj:
                seen_result = seen_result or not isinstance(item, ToolProgress)
                yield item
            if not seen_result:
                raise RuntimeError(f"Tool '{tool_name}' produced no result")
        else:
            yield await obj

    async def dispatch_tool(
        self, session: "ChatSession", tool_name: str, tc_input: dict
    ) -> "str | list[dict] | None":
        """Dispatch a tool call by name. Returns result text or multimodal blocks.

        Returns None only if a plain (non-generator) handler explicitly
        returns None itself — pre-existing behavior, unrelated to streaming
        support. A generator handler that never yields a non-ToolProgress
        item raises RuntimeError instead (see dispatch_tool_stream above); it
        never reaches this method's `return` with nothing set, so no
        "missing vs. None" sentinel is needed here.

        ToolProgress markers from a streaming handler are discarded here —
        callers that want them should consume dispatch_tool_stream directly
        (see ChatSession.turn_stream's generic tool branch).
        """
        result = None
        async for item in self.dispatch_tool_stream(session, tool_name, tc_input):
            if not isinstance(item, ToolProgress):
                result = item
        return result

    def unregister_tool(self, name: str) -> None:
        """Remove a tool by name. No-op if not found."""
        self._tools = [t for t in self._tools if t.name != name]

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
