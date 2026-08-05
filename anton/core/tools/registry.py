from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.core.session import ChatSession
    from anton.core.tools.tool_defs import ToolDef


@dataclass
class ToolOutcome:
    """A tool result plus the handler's own verdict on whether it failed.

    The failure signal drives the per-tool error streak (resilience nudge at
    2, circuit breaker at 5). It used to be re-derived by substring-matching
    the result text, which misclassified in both directions: a success whose
    output contained the word "failed" incremented the streak, and a genuine
    failure whose text lacked all five marker phrases RESET it — the
    mechanism behind the ENG-836 runaway, where interleaved false
    "successes" kept the breaker asleep for ~50 minutes (ENG-1276).

    ``ok``:
      - True/False — the handler's explicit verdict; classification uses it
        directly.
      - None — the handler hasn't been migrated; the dispatcher falls back
        to the legacy substring match and logs when that fallback classifies
        an error, so remaining call sites are discoverable rather than
        assumed.

    ``reason`` is a short, machine-comparable cause (exception type, missing
    library name). Unused by the streak itself today; it is the input the
    ENG-1286 root-cause thrash breaker keys on.
    """

    content: str | list[dict]
    ok: bool | None = None
    reason: str = field(default="")


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
    ) -> ToolOutcome:
        """Dispatch a tool call by name.

        Always returns a ``ToolOutcome``. Handlers that return a plain string
        or multimodal block list are wrapped with ``ok=None`` (not yet
        migrated → legacy substring classification applies); handlers that
        return a ``ToolOutcome`` declare their own verdict (ENG-1276).
        """
        tool_def = next((t for t in self._tools if t.name == tool_name), None)
        if tool_def is None:
            raise ValueError(f"Tool {tool_name} not found")
        result = await tool_def.handler(session, tc_input)
        if isinstance(result, ToolOutcome):
            return result
        return ToolOutcome(content=result)

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
