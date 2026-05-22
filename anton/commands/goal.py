"""Handler for the /goal autonomous execution command."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from anton.core.llm.provider import (
    StreamContextCompacted,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolResult,
    StreamToolUseEnd,
    StreamToolUseDelta,
    StreamToolUseStart,
)
from anton.core.tools.tool_defs import ToolDef
from anton.prompts import GOAL_CONTINUATION_PROMPT

if TYPE_CHECKING:
    from rich.console import Console
    from anton.chat_ui import StreamDisplay
    from anton.core.session import ChatSession


def parse_goal_args(raw: str, default_turns: int = 50) -> tuple[str, int]:
    """Parse '/goal' argument string into (objective, max_turns).

    Normalises embedded newlines (terminal line-wrap artefacts) before
    extracting the optional --turns flag, so inputs like
    ``"my goal" --tur\\nns 20`` are handled correctly.
    """
    arg = raw.replace("\r\n", "").replace("\r", "").replace("\n", "").strip()
    turns_match = re.search(r"--turns\s+(\d+)", arg)
    max_turns = int(turns_match.group(1)) if turns_match else default_turns
    objective = re.sub(r"--turns\s+\d+", "", arg).strip().strip('"').strip("'").strip()
    return objective, max_turns


async def run_goal_loop(
    console: "Console",
    session: "ChatSession",
    display: "StreamDisplay",
    objective: str,
    max_turns: int,
) -> None:
    """Run autonomous goal-directed turns until complete, exhausted, or interrupted."""
    from anton.chat_ui import EscapeWatcher

    @dataclass
    class _GoalState:
        completed: bool = False
        completion_reason: str = ""

    goal_state = _GoalState()

    async def _handle_mark_goal_complete(_session, tc_input: dict) -> str:
        reason = tc_input.get("reason", "Goal completed.")
        goal_state.completed = True
        goal_state.completion_reason = reason
        return f"Goal marked as complete: {reason}"

    mark_goal_complete_tool = ToolDef(
        name="mark_goal_complete",
        description=(
            "Signal that the goal has been fully achieved. Call this ONLY when you have "
            "concrete proof that every requirement implied by the goal is satisfied. "
            "Do not call this speculatively — treat uncertain evidence as 'not yet done'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "One-sentence summary of what was accomplished and why the goal is complete.",
                },
            },
            "required": ["reason"],
        },
        handler=_handle_mark_goal_complete,
    )

    # Ensure the core tools are built, then register the goal tool on top.
    session._build_tools()
    session.tool_registry.register_tool(mark_goal_complete_tool)

    console.print()
    console.print(f"[anton.cyan]Goal:[/] {objective}")
    console.print(f"[anton.muted]Running up to {max_turns} turns autonomously. Ctrl+C to stop.[/]")
    console.print()

    consecutive_failures = 0
    completed_turn = 0
    try:
        for turn in range(1, max_turns + 1):
            if goal_state.completed:
                break
            completed_turn = turn

            continuation_msg = GOAL_CONTINUATION_PROMPT.format(
                objective=objective,
                turn=turn,
                max_turns=max_turns,
            )

            console.print(f"[anton.muted][goal {turn}/{max_turns}] working...[/]")
            display.start()
            session._cancel_event.clear()

            try:
                async with EscapeWatcher(on_cancel=display.show_cancelling) as esc:
                    session._escape_watcher = esc
                    async for event in session.turn_stream(continuation_msg):
                        if esc.cancelled.is_set():
                            session._cancel_event.set()
                            raise KeyboardInterrupt
                        if isinstance(event, StreamTextDelta):
                            display.append_text(event.text)
                        elif isinstance(event, StreamToolResult):
                            if event.name == "scratchpad" and event.action == "dump":
                                display.show_tool_result(event.content)
                        elif isinstance(event, StreamToolUseStart):
                            display.on_tool_use_start(event.id, event.name)
                        elif isinstance(event, StreamToolUseDelta):
                            display.on_tool_use_delta(event.id, event.json_delta)
                        elif isinstance(event, StreamToolUseEnd):
                            display.on_tool_use_end(event.id)
                        elif isinstance(event, StreamTaskProgress):
                            display.update_progress(event.phase, event.message, event.eta_seconds)
                        elif isinstance(event, StreamContextCompacted):
                            display.show_context_compacted(event.message)

                display.finish()
                consecutive_failures = 0
                if goal_state.completed:
                    break

            except KeyboardInterrupt:
                display.abort()
                raise
            except Exception as exc:
                display.abort()
                consecutive_failures += 1
                console.print(f"\n[anton.warning][goal {turn}/{max_turns}] Turn failed: {exc}[/]")
                if consecutive_failures >= 3:
                    console.print("[anton.error]3 consecutive failures. Aborting goal.[/]")
                    break
                session.repair_history()
                continue

        console.print()
        if goal_state.completed:
            console.print(f"[anton.cyan]Goal complete[/] after {completed_turn} turn(s): {goal_state.completion_reason}")
        else:
            console.print(f"[anton.warning]Goal not completed after {completed_turn} turn(s).[/]")
        console.print()

    except KeyboardInterrupt:
        session.repair_history()
        console.print()
        console.print(f"[anton.muted]Goal interrupted after {completed_turn} turn(s).[/]")
        console.print()

    finally:
        session.tool_registry.unregister_tool("mark_goal_complete")
        # Anchor the model back to normal chat mode. Two synthetic turns:
        #
        # 1. If repair_history() ran, history ends with user:[tool_results].
        #    Merging a text block into that is a malformed mixed-type
        #    message Anthropic rejects — close it with an assistant ack first.
        # 2. A SYSTEM user message declaring goal end.
        # 3. A synthetic assistant acknowledgment so the user's NEXT message
        #    arrives after an explicit context break. Without (3), ambiguous
        #    replies like "ok" / "oj" are interpreted as task continuations.
        if session._history and session._history[-1].get("role") == "user":
            session._append_history({
                "role": "assistant",
                "content": "[Goal session interrupted.]",
            })
        session._append_history({
            "role": "user",
            "content": (
                "SYSTEM: The autonomous goal session has ended. "
                "Do NOT continue any prior task unless the user explicitly asks."
            ),
        })
        session._append_history({
            "role": "assistant",
            "content": "Understood — the goal session has ended. What would you like to do?",
        })
