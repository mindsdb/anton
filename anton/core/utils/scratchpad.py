from __future__ import annotations

import os
from typing import TYPE_CHECKING

from anton.core.tools.registry import ToolOutcome

if TYPE_CHECKING:
    from anton.core.session import ChatSession


def _acc_observe(session, kind: str, detail: dict, *, severity: int = 1) -> None:
    """Safe ACC emit — no-op if the session has no observer wired."""
    fn = getattr(session, "_acc_observe", None)
    if fn is not None:
        fn(kind, detail, severity=severity)


_DISCOVERY_MAX_PADS = 10
_DISCOVERY_MAX_ROOT_ENTRIES = 30


def _age_label(mtime: float | None) -> str:
    if mtime is None:
        return ""
    import time

    mins = max(0, int((time.time() - mtime) / 60))
    if mins < 60:
        return f" (snapshot {mins}m old)"
    if mins < 60 * 48:
        return f" (snapshot {mins // 60}h old)"
    return f" (snapshot {mins // (60 * 24)}d old)"


def build_workspace_discovery_context(manager) -> str:
    """Compact turn-start block: known pads + project-root names (ENG-578).

    Cold-start blindness made the agent invent pad names and rebuild state it
    already had on disk. Source of truth is agent_pads() — live pads only add
    the "active" label, so system-created pads never leak in. Best-effort by
    construction: any failure degrades to omitting that part or the whole
    block, never to breaking the turn.
    """
    pads_line = ""
    try:
        known = sorted(manager.agent_pads())
        if known:
            shown = known[:_DISCOVERY_MAX_PADS]
            live = set(manager.pads)
            parts = []
            for name in shown:
                label = (
                    " (active)"
                    if name in live
                    else _age_label(manager.pad_snapshot_mtime(name))
                )
                parts.append(f"{name}{label}")
            more = (
                f" … and {len(known) - len(shown)} more"
                if len(known) > len(shown)
                else ""
            )
            pads_line = (
                "\nScratchpads for this conversation: "
                + ", ".join(parts)
                + more
                + " — reuse via scratchpad exec with the same name; each name is a "
                "separate environment."
            )
    except Exception:
        pads_line = ""

    root_line = ""
    try:
        root = manager.workspace_path
        if root is not None:
            # Hidden entries are excluded deliberately: this block ships to
            # the model, and filenames hinting at secret locations (.env,
            # .aws) don't belong in LLM-visible context — even names-only.
            # scandir rather than iterdir+is_dir: dir-ness comes from the
            # dirent for free instead of a stat per entry, and this runs
            # every turn. Full iteration is inherent to a sorted first-30.
            with os.scandir(root) as it:
                entries = sorted(
                    entry.name + ("/" if entry.is_dir() else "")
                    for entry in it
                    if not entry.name.startswith(".")
                )
            if entries:
                shown_entries = entries[:_DISCOVERY_MAX_ROOT_ENTRIES]
                more = (
                    f" … and {len(entries) - len(shown_entries)} more"
                    if len(entries) > len(shown_entries)
                    else ""
                )
                root_line = "\nProject root: " + ", ".join(shown_entries) + more
    except Exception:
        root_line = ""

    if not pads_line and not root_line:
        return ""
    return "\n\nWorkspace state:" + pads_line + root_line


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
    a ToolOutcome whose content is the message when the call should not run
    (empty code, a single-scratchpad challenge, or a failed package install).
    The outcome carries the explicit failure verdict, so the error streak no
    longer depends on the message's wording (ENG-1276).

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
        # same oversized cell. The explicit ok=False is what counts it toward
        # the circuit breaker — the message's wording no longer matters to
        # classification (it used to need the word "failed" for the substring
        # matcher, ENG-1276).
        _acc_observe(session, "scratchpad_empty_code", {"name": name}, severity=7)
        return ToolOutcome(
            content=(
                "Scratchpad exec failed: the `code` argument was empty. This usually "
                "means the code payload was too large and got truncated in transit. "
                "Do NOT retry the same large cell — instead write the output to disk in "
                "small append steps (open(path, 'a'), keep each cell's string under ~5KB), "
                "or generate the content inside the cell rather than passing a big literal."
            ),
            ok=False,
            reason="scratchpad_empty_code",
        )

    # Single-scratchpad guard: the agent should reuse ONE scratchpad per task.
    # A new name spins up a separate, empty process — state from the existing
    # pad isn't visible there — a common source of wasted rounds (re-import,
    # re-fetch, shuffling state across pads). Challenge a new name when the
    # agent already has a working scratchpad, unless it confirms it needs
    # isolation.
    #
    # `seen` must include pads used in EARLIER TURNS (ENG-1124 Fix 5). This guard
    # originally consulted only the in-memory set below — but cowork-server builds a
    # fresh `ChatSession` per user message, and the agent switches pad names precisely
    # *at* turn boundaries (measured: 37 of 39 turns used exactly one name). So the set
    # was always empty exactly when the guard needed it, and it fired 0 times across
    # 676 cells of a real session that accumulated 22 pad names. The manager keeps a
    # small per-conversation record on disk so a later turn can still see them.
    #
    # Still NOT `session._scratchpads.pads`, and not the snapshot files either: both
    # include system-created pads (the artifact backend launcher's slug pad), which must
    # never count against the agent. `agent_pads()` returns only names this guard
    # explicitly recorded.
    #
    # Challenge AT MOST ONCE PER TURN (the flag lives on the per-turn session): the
    # challenge is not an error (it resets no streak), so re-challenging every new name
    # could loop to the round cap with nothing to stop it. One firm nudge, then respect
    # the model's choice — and a later turn gets a fresh chance to nudge again. `is
    # True` (not truthiness) so a MagicMock attr in tests doesn't read as "challenged".
    seen = getattr(session, "_agent_scratchpad_names", None)
    if not isinstance(seen, set):
        seen = set()
        session._agent_scratchpad_names = seen
    manager = getattr(session, "_scratchpads", None)
    known = set(seen)
    if manager is not None and hasattr(manager, "agent_pads"):
        try:
            persisted = manager.agent_pads()
            if isinstance(persisted, set):
                known |= persisted
        except Exception:
            pass
    confirm_new = bool(tc_input.get("confirm_new_scratchpad", False))
    challenged_before = getattr(session, "_scratchpad_challenged", False) is True
    if name not in known and known and not confirm_new and not challenged_before:
        session._scratchpad_challenged = True
        existing = "', '".join(sorted(known))
        # ok=True: the challenge is guidance, not a tool failure — it must
        # not count toward the error streak (it previously relied on its
        # wording happening to avoid the substring markers, ENG-1276).
        return ToolOutcome(
            content=(
                f"You already have an active scratchpad ('{existing}') with live state "
                f"(imports, variables, fetched data). Starting a new one named '{name}' "
                "creates a SEPARATE, empty environment — nothing from the existing "
                "scratchpad is available there, so you'd re-import and re-fetch. Reuse the "
                "existing scratchpad for this task; it is stateful across cells. If you "
                "genuinely need an isolated environment, call scratchpad exec again with "
                "confirm_new_scratchpad=true."
            ),
            ok=True,
        )
    seen.add(name)
    if manager is not None and hasattr(manager, "record_agent_pad"):
        try:
            manager.record_agent_pad(name)
        except Exception:
            pass

    pad = await session._scratchpads.get_or_create(name)

    # Auto-install packages before running the cell
    packages = tc_input.get("packages", [])
    if packages:
        install_result = await pad.install_packages(packages)
        # The substring check against install_packages' message is this
        # handler's own protocol with the runtime (its return shape predates
        # ToolOutcome); the verdict it produces is explicit from here on out.
        if "Install failed" in install_result or "timed out" in install_result:
            return ToolOutcome(
                content=install_result, ok=False, reason="package_install_failed"
            )

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
