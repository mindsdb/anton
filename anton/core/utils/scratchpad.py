from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.core.session import ChatSession


async def prepare_scratchpad_exec(session: ChatSession, tc_input: dict):
    """Validate and prepare a scratchpad exec call.

    Returns (pad, code, description, estimated_time, estimated_seconds) or
    a str error message if validation fails.
    """
    name = tc_input.get("name", "")
    code = tc_input.get("code", "")
    if not code or not code.strip():
        return "No code provided."

    pad = await session._scratchpads.get_or_create(name)

    # Auto-install packages before running the cell
    packages = tc_input.get("packages", [])
    if packages:
        install_result = await pad.install_packages(packages)
        if "Install failed" in install_result or "timed out" in install_result:
            return install_result

    description = tc_input.get("one_line_description", "")
    estimated_seconds = tc_input.get("estimated_execution_time_seconds", 0)
    if isinstance(estimated_seconds, str):
        try:
            estimated_seconds = int(estimated_seconds)
        except ValueError:
            estimated_seconds = 0

    estimated_time = f"{estimated_seconds}s" if estimated_seconds > 0 else ""
    return pad, code, description, estimated_time, estimated_seconds


def format_cell_result(cell) -> str:
    """Format a Cell into a tool result string.

    Every section is labeled so the LLM can tell what came from where:
    [output] — print() / stdout from the cell code
    [logs]   — library logging (httpx, urllib3, etc.) captured at INFO+
    [stderr] — warnings and stderr writes
    [error]  — Python traceback if the cell raised an exception
    """
    parts: list[str] = []
    if cell.stdout:
        stdout = cell.stdout
        if len(stdout) > 10_000:
            stdout = stdout[:10_000] + f"\n\n... (truncated, {len(stdout)} chars total)"
        parts.append(f"[output]\n{stdout}")
    if cell.logs if hasattr(cell, "logs") else False:
        logs = cell.logs.strip()
        if len(logs) > 3_000:
            logs = logs[:3_000] + "\n... (logs truncated)"
        parts.append(f"[logs]\n{logs}")
    if cell.stderr:
        parts.append(f"[stderr]\n{cell.stderr}")
    if cell.error:
        parts.append(f"[error]\n{cell.error}")
    if not parts:
        return "Code executed successfully (no output)."
    return "\n".join(parts)
