from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from anton.core.llm.provider import StreamTaskProgress
from anton.core.tools.progress import ToolProgress

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
        self, session: "ChatSession", tool_name: str, tc_input: dict,
        *, tool_call_id: str | None = None,
    ) -> ToolOutcome:
        """Dispatch a tool call by name. Always returns a ``ToolOutcome``.

        Handlers that return a plain string or multimodal block list are
        wrapped with ``ok=None`` (not yet migrated → legacy substring
        classification applies); handlers that return a ``ToolOutcome``
        declare their own verdict, returned as-is — same object, not a copy
        (ENG-1276).

        Supports streaming (async-generator) handlers via
        ``dispatch_tool_stream`` — but ``dispatch_tool`` itself commonly runs
        inside a background task now (``ChatSession._dispatch_draining``,
        for ask_user-mid-tool-call support), which has no direct access to
        the caller's yield point. So a ``ToolProgress`` marker is relayed
        through ``session.emitter`` instead of yielded here — the same
        out-of-band path ``ask_user`` already uses for exactly this reason
        (ENG-763 + ENG-1276 integration). ``tool_call_id`` lets the relayed
        marker carry the originating tool_use id for frontend correlation
        (see ``StreamTaskProgress``); pass it wherever the caller has one.
        Silently skipped when ``session`` has no emitter (the non-streaming
        ``turn()`` path, or a bare test double) — there's no drain loop there
        to receive it anyway.
        """
        emitter = getattr(session, "emitter", None)
        result = None
        async for item in self.dispatch_tool_stream(session, tool_name, tc_input):
            if isinstance(item, ToolProgress):
                if emitter is not None:
                    await emitter.emit(
                        StreamTaskProgress(
                            phase="tool_progress", message=item.text, id=tool_call_id,
                        )
                    )
            else:
                result = item
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
