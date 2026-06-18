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
        # An empty `code` on an exec call is almost never the model meaning
        # to run nothing — it's the large-payload drop: an oversized `code`
        # argument gets truncated to "" in transit. Returning a bare "no
        # code" here used to read as a no-op, so the model would retry the
        # same oversized cell. Make the failure self-correcting and ensure
        # it reads as an error (note the word "failed") so the per-tool
        # error streak in _apply_error_tracking counts it toward the
        # circuit breaker instead of silently resetting.
        return (
            "Scratchpad exec failed: the `code` argument was empty. This usually "
            "means the code payload was too large and got truncated in transit. "
            "Do NOT retry the same large cell — instead write the output to disk in "
            "small append steps (open(path, 'a'), keep each cell's string under ~5KB), "
            "or generate the content inside the cell rather than passing a big literal."
        )

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
