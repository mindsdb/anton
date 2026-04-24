"""The `erase_scratchpad_history` tool — drop intermediate scratchpad code from
context once the artifact it produced has been written to disk.

When the agent builds an HTML dashboard (or any file artifact) across many
scratchpad cells, every `exec` call and its code string lives on in the message
history. Once the final HTML is on disk, that intermediate code is redundant —
the file is the source of truth. This tool blanks the code of cells that the
agent marked with `# DELETABLE: <description>` as the first line, both in the
live pad and in the assistant message history, so follow-up turns don't pay
the context tax.

Scope is deliberately narrow: only `input.code` of matching scratchpad exec
tool calls and `cell.code` of matching cells. Outputs, descriptions, and other
tool types are untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from anton.core.tools.tool_defs import ToolDef

if TYPE_CHECKING:
    from anton.core.session import ChatSession


_MARKER = "# DELETABLE:"


_DESCRIPTION = (
    "Erase the Python code of scratchpad cells you previously marked with "
    "`# DELETABLE: <short description>` as their first line. Call this after "
    "a file artifact (HTML dashboard, presentation, etc.) has been written to "
    "disk — the file is the source of truth, so the intermediate generation "
    "code is no longer useful in the conversation context.\n\n"
    "The tool scans both the live scratchpad cells and the assistant message "
    "history, and replaces every matching `code` value with "
    "`# DELETED: <description>` (the description from the marker is preserved "
    "so you can still see what each cell did). Outputs, descriptions on the "
    "tool call, and anything without the marker are left untouched. Safe to "
    "call repeatedly — cells already cleared are skipped."
)


_INPUT_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}


def _extract_deletable_description(code: str) -> str | None:
    """If the first non-blank line of `code` is `# DELETABLE: <desc>`, return
    <desc>. Otherwise None.

    Leading blank lines (common when the model emits multi-line code blocks)
    are ignored so the marker still matches.
    """
    if not isinstance(code, str):
        return None
    stripped = code.lstrip()
    first_line = stripped.split("\n", 1)[0].strip()
    if not first_line.startswith(_MARKER):
        return None
    return first_line[len(_MARKER):].strip()


def _replacement(description: str) -> str:
    if description:
        return f"# DELETED: {description}"
    return "# DELETED"


async def handle_erase_scratchpad_history(
    session: "ChatSession", tc_input: dict
) -> str:
    """Clear code of DELETABLE-marked cells in live pads and history."""
    cells_cleared = 0
    history_blocks_cleared = 0

    # 1. Live scratchpad cells
    for pad in session._scratchpads.pads.values():
        for cell in pad.cells:
            desc = _extract_deletable_description(cell.code)
            if desc is None:
                continue
            cell.code = _replacement(desc)
            cells_cleared += 1

    # 2. Message history — only scratchpad exec tool_use blocks.
    # Skip the last assistant message (it contains this tool call itself).
    history = session._history
    last_idx = len(history) - 1
    for idx, msg in enumerate(history):
        if idx == last_idx and msg.get("role") == "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            if block.get("name") != "scratchpad":
                continue
            tc_in = block.get("input")
            if not isinstance(tc_in, dict):
                continue
            if tc_in.get("action") != "exec":
                continue
            code = tc_in.get("code")
            desc = _extract_deletable_description(code)
            if desc is None:
                continue
            tc_in["code"] = _replacement(desc)
            history_blocks_cleared += 1

    session._persist_history()

    if cells_cleared == 0 and history_blocks_cleared == 0:
        return (
            "No DELETABLE cells found. Either none were marked with "
            "`# DELETABLE: <description>` on their first line, or they have "
            "already been cleared."
        )

    return (
        f"Cleared {cells_cleared} live scratchpad cells and "
        f"{history_blocks_cleared} tool-call entries in history."
    )


ERASE_SCRATCHPAD_HISTORY_TOOL = ToolDef(
    name="erase_scratchpad_history",
    description=_DESCRIPTION,
    input_schema=_INPUT_SCHEMA,
    handler=handle_erase_scratchpad_history,
)


__all__ = [
    "ERASE_SCRATCHPAD_HISTORY_TOOL",
    "handle_erase_scratchpad_history",
]
