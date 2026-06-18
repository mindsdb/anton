from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anton.core.session import ChatSession


def _acc_observe(session, kind: str, detail: dict, *, severity: int = 1) -> None:
    """Safe ACC emit — no-op if the session has no observer wired."""
    fn = getattr(session, "_acc_observe", None)
    if fn is not None:
        fn(kind, detail, severity=severity)


def observe_scratchpad_cell(session, name: str, cell) -> None:
    """Emit the post-execute ACC event for a finished cell.

    Distinguishes a kill (timeout/cancel/OOM) from a plain runtime error so
    detect_kill_loop sees `scratchpad_killed`. Shared by both exec paths —
    `handle_scratchpad` (CLI `turn()`) and the inline streaming exec in
    `ChatSession.turn_stream` — so the ACC instrumentation is identical
    regardless of which path ran the cell.
    """
    if cell is None:
        return
    err = (cell.error or "").strip()
    if err.startswith(("Cancelled", "Cell timed out", "Cell killed")):
        _acc_observe(session, "scratchpad_killed", {"name": name, "reason": err[:120]}, severity=6)
    else:
        success = not err and not (cell.stderr or "").strip()
        _acc_observe(
            session,
            "scratchpad_result",
            {
                "name": name,
                "success": success,
                "stdout_len": len(cell.stdout or ""),
                "error": err[:300] if err else "",
            },
            severity=5 if not success else 1,
        )


async def prepare_scratchpad_exec(session: ChatSession, tc_input: dict):
    """Validate and prepare a scratchpad exec call.

    Returns (pad, code, description, estimated_time, estimated_seconds) or
    a str message if the call should not run (empty code, a single-scratchpad
    challenge, or a failed package install).

    This is the SHARED entry point for both exec paths — `handle_scratchpad`
    (CLI) and the inline streaming exec in `ChatSession.turn_stream` (cowork)
    both call it — so the single-scratchpad guard and the pre-execute ACC
    events live here, not in `handle_scratchpad` (which the streaming path
    bypasses).
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
        _acc_observe(session, "scratchpad_empty_code", {"name": name}, severity=7)
        return (
            "Scratchpad exec failed: the `code` argument was empty. This usually "
            "means the code payload was too large and got truncated in transit. "
            "Do NOT retry the same large cell — instead write the output to disk in "
            "small append steps (open(path, 'a'), keep each cell's string under ~5KB), "
            "or generate the content inside the cell rather than passing a big literal."
        )

    # Single-scratchpad guard: the agent should reuse ONE scratchpad per task.
    # A new name spins up a separate, empty process — state from the existing
    # pad isn't visible there — a common source of wasted rounds (re-import,
    # re-fetch, shuffling state across pads). Challenge a new name when the
    # agent already has a working scratchpad this session, unless it confirms
    # it needs isolation. Tracked names are ones the agent has exec'd here —
    # NOT session._scratchpads.pads, which also holds system-created pads
    # (e.g. the artifact backend launcher's slug pad), which must never count
    # against the agent. Challenge AT MOST ONCE per session: the challenge is
    # not an error (it resets no streak), so re-challenging every new name
    # could loop to the round cap with nothing to stop it; one firm nudge is
    # the enforcement, then respect the model's choice. `is True` (not
    # truthiness) so a MagicMock attr in tests doesn't read as "challenged".
    seen = getattr(session, "_agent_scratchpad_names", None)
    if not isinstance(seen, set):
        seen = set()
        session._agent_scratchpad_names = seen
    confirm_new = bool(tc_input.get("confirm_new_scratchpad", False))
    challenged_before = getattr(session, "_scratchpad_challenged", False) is True
    if name not in seen and seen and not confirm_new and not challenged_before:
        session._scratchpad_challenged = True
        existing = "', '".join(sorted(seen))
        return (
            f"You already have an active scratchpad ('{existing}') with live state "
            f"(imports, variables, fetched data). Starting a new one named '{name}' "
            "creates a SEPARATE, empty environment — nothing from the existing "
            "scratchpad is available there, so you'd re-import and re-fetch. Reuse the "
            "existing scratchpad for this task; it is stateful across cells. If you "
            "genuinely need an isolated environment, call scratchpad exec again with "
            "confirm_new_scratchpad=true."
        )
    seen.add(name)

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
    _acc_observe(
        session,
        "scratchpad_call",
        {
            "name": name,
            "code_len": len(code or ""),
            "one_line_description": description or "",
        },
    )
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
