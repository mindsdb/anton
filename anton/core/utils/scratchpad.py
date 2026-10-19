from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from anton.core.backends.sandbox import cell_sandbox_of, validate_pad_name
from anton.core.tools.registry import ToolOutcome
from anton.core.utils.write_check import (
    CellWrites,
    FileChange,
    ProjectSnapshot,
    find_cell_writes,
    snapshot_project_files,
)

if TYPE_CHECKING:
    from anton.core.backends.manager import ScratchpadManager
    from anton.core.session import ChatSession

logger = logging.getLogger(__name__)

# Reasons a scratchpad exec result carries when its cell never ran, so a
# caller that records executed cells can tell them from a cell that did
# (`discovery.engine`, `generate_artifact.engine._run_loop` and the turn's
# hand-back report).
CELL_NOT_RUN_REASONS = frozenset({"new_scratchpad_challenged", "write_not_approved"})

RefusalCause = Literal["cancelled", "timeout", "unavailable", "error", "unanswered_earlier"]
# What a logged decision says came of a held cell: "stopped" when the turn
# ended while its card waited, so no refusal went back to the model.
DecisionStatus = RefusalCause | Literal["approved", "not_held", "stopped"]

_WHY = {
    "cancelled": "they declined it",
    "timeout": "no one answered in time",
    "unavailable": "there is no way to ask them here",
    "error": "the approval request failed",
    "unanswered_earlier": "an earlier approval request in this turn went unanswered",
}
# Paths the refusal names for one file change; the card names every one.
_REFUSAL_PATHS = 20
_TILDE_RUN = re.compile("~{3,}")
_BACKTICK_RUN = re.compile("`+")


@dataclass
class WriteGate:
    """Per-session state for holding cells that change a database or existing files.

    Present on a session only when the host turns the setting on. `begin_turn`
    records what the project and shared folders held when the turn began.
    """

    snapshots: tuple[ProjectSnapshot, ...] = ()
    # After one card goes unanswered, later flagged cells in the same turn are
    # refused without a new card, so an unattended turn doesn't wait out the
    # timeout once per cell.
    unanswered: bool = False

    async def begin_turn(self, *, manager: ScratchpadManager) -> None:
        self.unanswered = False
        taken_at = time.time()
        roots: list[tuple[Literal["project", "shared"], Path]] = []
        if manager.workspace_path is not None:
            roots.append(("project", manager.workspace_path))
        policy = cell_sandbox_of(manager)
        if policy is not None and policy.shared_dir is not None:
            roots.append(("shared", policy.shared_dir))
        self.snapshots = tuple([
            await asyncio.to_thread(
                snapshot_project_files, root=root, label=label, taken_at=taken_at
            )
            for label, root in roots
        ])


class WriteApprovalDecision(BaseModel):
    """One logged decision: what a held cell would change and what came of it.
    Never the code or the statement itself."""

    model_config = ConfigDict(frozen=True)

    kinds: tuple[str, ...]
    count: int
    status: DecisionStatus


def _inert(text: str) -> str:
    """`text` with each HTML comment mark split by a zero-width space. Cowork
    drops `<!-- ... -->` from a card before it reads the Markdown, code
    included, and drops everything after an unclosed `<!--`."""
    return text.replace("<!--", "<\u200b!--").replace("-->", "--\u200b>")


def _fence(text: str) -> str:
    """`text` in a code fence nothing inside it can end, so the card shows it
    as written. Cowork's math pass rewrites a `$` or `\\(` outside code, and
    it reads a fence as code from one run of three backticks, or of three
    tildes, to the next. So a text with three backticks in a row goes in
    tildes, and a run of three tildes inside it is split with a zero-width
    space."""
    text = _inert(text)
    if "```" not in text:
        return "```\n" + text + "\n```"
    return "~~~\n" + _TILDE_RUN.sub(lambda m: "\u200b".join(m.group()), text) + "\n~~~"


def _code_span(text: str) -> str:
    """`text` in a code span longer than any run of backticks inside it."""
    ticks = max((len(run) for run in _BACKTICK_RUN.findall(text)), default=0) + 1
    pad = " " if text[:1] in ("`", " ") or text[-1:] in ("`", " ") else ""
    return "`" * ticks + pad + text + pad + "`" * ticks


def _shown_path(path: str) -> str:
    """`path` as the card and the refusal name it: as written, or as Python
    writes it, in quotes, when it holds a character that doesn't show as
    itself (a line break, a tab, a text direction override, a zero-width
    space) or a backslash, or starts with a quote."""
    if path.isprintable() and "\\" not in path and not path.startswith(("'", '"')):
        return path
    return repr(path)


def _wholes(change: FileChange) -> str:
    """The root folders `change` takes whole, as the card names them; "" for none."""
    wholes = []
    if change.whole_project:
        wholes.append("the whole project folder")
    if change.whole_shared:
        wholes.append("the whole shared folder")
    return " and ".join(wholes)


def _describe_files(writes: CellWrites) -> list[str]:
    """Each held file change for the refusal: the first paths it names, or
    the code that picks the path when it runs. The card names every path."""
    out = []
    for change in writes.held_files:
        wholes = _wholes(change)
        if wholes:
            out.append(f"{change.action} {wholes}")
        if change.paths:
            named = ", ".join(_code_span(_shown_path(p)) for p in change.paths[:_REFUSAL_PATHS])
            more = len(change.paths) - _REFUSAL_PATHS
            out.append(f"{change.action} {named}" + (f" and {more} more" if more > 0 else ""))
        if not (wholes or change.paths):
            out.append(f"{change.action} a file or folder it picks when it runs, in {_code_span(change.call)}")
    return out


def _summary(writes: CellWrites) -> str:
    parts = []
    if writes.sql:
        parts.append("change data in a connected source")
    parts += _describe_files(writes)
    if writes.unchecked:
        parts.append("run code the check couldn't read")
    return "; ".join(parts)


def _file_lines(change: FileChange, *, lead: str) -> list[str]:
    """One file change on the card: the root folders it takes whole, every
    path it names, or the code that picks the path when it runs."""
    lines: list[str] = []
    wholes = _wholes(change)
    if wholes:
        lines += [f"{lead} {wholes}.", ""]
    if change.paths:
        lines += [f"{lead}:", "", _fence("\n".join(_shown_path(p) for p in change.paths)), ""]
    if not (wholes or change.paths):
        lines += [f"{lead} a file or folder it picks when it runs, in this code:", "", _fence(change.call), ""]
    return lines


def _card_prompt(writes: CellWrites) -> str:
    """The card: every statement and every file change the step would make,
    whole, since one Approve runs them all. Statements, paths and code go in
    fences, so Cowork's Markdown shows them as written."""
    lines = ["**Approve this change?**", ""]
    if writes.sql:
        lines += ["This step would change data in a connected source:", ""]
        for write in writes.sql:
            text = write.statement + (f"\nwith values {write.values}" if write.values else "")
            lines += [_fence(text), ""]
        if any(write.run_time_parts for write in writes.sql):
            lines += ["A part in braces, `{...}`, is the code that fills in that value when the step runs.", ""]
    for change in writes.held_files:
        lines += _file_lines(change, lead=f"This step would {change.action}")
    # An overwrite raises no card on its own, but the Approve runs it too.
    for change in writes.overwrites:
        lines += _file_lines(change, lead="This step would also overwrite")
    if any(_shown_path(p) != p for change in writes.files for p in change.paths):
        lines += ["A name in quotes is written with escapes, as Python writes it: `\\n` is a line break "
                  "and `\\\\` a backslash.", ""]
    if writes.unchecked:
        lines += ["This step can't be checked before it runs.", ""]
    lines.append("It runs only if you approve. Cancel, Skip or no answer stops it.")
    return "\n".join(lines)


def _refusal(writes: CellWrites, cause: RefusalCause) -> ToolOutcome:
    # ok=True on purpose: a refusal is the user's decision, not a tool
    # failure, so it must not feed the error streak, whose nudge tells the
    # model to try another library or source.
    return ToolOutcome(
        content=(
            f"Not run: this step would {_summary(writes)}. That needs the user's approval, "
            f"and {_WHY[cause]}. Nothing in this cell ran. The user didn't approve this "
            "change, so it must not be made another way: don't rewrite the cell, build the "
            "statement or path at run time, or use another tool for it. Tell the user what "
            "the step would change and that it needs their approval."
        ),
        ok=True,
        reason="write_not_approved",
    )


async def _ask_approval(session: ChatSession, writes: CellWrites) -> RefusalCause | None:
    """Ask Approve or Cancel. None when the user picks Approve, else why not."""
    from anton.core.interaction.elicit import AskOption, AskRequest, elicit

    request = AskRequest(
        prompt=_card_prompt(writes),
        options=(
            AskOption(value="approve", label="Approve", detail="Run this step once"),
            AskOption(value="cancel", label="Cancel", detail="Leave the data and files as they are"),
        ),
        # No typed answer and no default: only a picked Approve runs the cell.
        allow_custom=False,
        timeout_s=getattr(session.elicitor, "timeout_s", None),
    )
    try:
        # Not counted against the turn's questions: the card is the gate's,
        # not the model's, so it spends none of the model's own.
        answer = await elicit(session, f"write:{uuid.uuid4().hex}", request, counted=False)
    except Exception as exc:
        # The type only: a traceback's frames would print the card and the
        # statement under a host that logs local variables.
        logger.warning("write approval: the question failed (%s)", type(exc).__name__)
        return "error"
    if answer.status == "answered" and answer.values == ("approve",):
        return None
    if answer.status in ("answered", "cancelled"):
        return "cancelled"
    if answer.status in ("timeout", "unavailable"):
        return answer.status
    return "error"


def _log_decision(*, writes: CellWrites, status: DecisionStatus) -> None:
    decision = WriteApprovalDecision(
        kinds=writes.kinds(), count=len(writes.sql) + len(writes.files), status=status,
    )
    # WARNING: hosts log at WARNING by default, and this line is the record of
    # a held write.
    logger.warning("write approval: %s", decision.model_dump_json())


async def hold_unapproved_write(session: ChatSession, *, pad, code: str) -> ToolOutcome | None:
    """Hold a cell that changes a database or existing files until the user
    approves it. None when the cell may run; otherwise the refusal to return
    in its place, before any of it runs.

    Only sessions with a `WriteGate` (the `write_approval` setting) check
    anything. A check that fails asks too, since it can't clear the cell.
    """
    gate = getattr(session, "_write_gate", None)
    if not isinstance(gate, WriteGate):
        return None
    # The cells compaction folded away come first: the pad's process still
    # holds what they imported and bound.
    earlier = (
        *getattr(pad, "compacted_code", ()),
        *(cell.code for cell in getattr(pad, "cells", ()) or ()),
    )
    try:
        writes = await asyncio.to_thread(
            find_cell_writes,
            code=code, earlier_code=earlier, snapshots=gate.snapshots,
        )
    except Exception as exc:
        # The type only, as for a failed question.
        logger.warning("write approval: the check failed (%s)", type(exc).__name__)
        writes = CellWrites(unchecked=True)
    if not writes.needs_approval():
        if writes.overwrites:
            _log_decision(writes=writes, status="not_held")
        return None
    try:
        cause = "unanswered_earlier" if gate.unanswered else await _ask_approval(session, writes)
    except asyncio.CancelledError:
        # Stop, or the host ending the turn, while the card waited.
        _log_decision(writes=writes, status="stopped")
        raise
    _log_decision(writes=writes, status=cause or "approved")
    if cause is None:
        return None
    if cause == "timeout":
        gate.unanswered = True
    return _refusal(writes, cause)


def _acc_observe(session, kind: str, detail: dict, *, severity: int = 1) -> None:
    """Safe ACC emit — no-op if the session has no observer wired."""
    fn = getattr(session, "_acc_observe", None)
    if fn is not None:
        fn(kind, detail, severity=severity)


# install_packages' own return shape — no structured success/failure value,
# so both call sites (exec+packages here, and the standalone install action
# in tool_handlers.py) key off these same literals. Anchored to the message
# HEAD: failures are reported as a first-line prefix, while a SUCCESSFUL
# pip/uv run may mention "timed out" mid-output (a retried network read
# inside an install that then succeeded) without having failed.
_INSTALL_FAILURE_PREFIXES = ("Install failed", "Install timed out", "Install refused")
_INSTALL_NOOP_RESULTS = frozenset(
    {"No packages specified.", "All packages already installed."}
)


def refuse_pad_name(name: str) -> ToolOutcome | None:
    """A refusal for a scratchpad name that is not one plain folder name, else None.

    The name becomes a folder the host deletes and recreates, so anything but
    one plain folder name (see `validate_pad_name`) is refused before any pad
    is built or recorded.
    """
    try:
        validate_pad_name(name)
    except ValueError as exc:
        return ToolOutcome(content=str(exc), ok=False, reason="scratchpad_invalid_name")
    return None


def install_call_failed(result: str) -> bool:
    return result.startswith(_INSTALL_FAILURE_PREFIXES)


def invalid_package_spec(entry: str) -> bool:
    """True if `entry` is not a plain PEP 508 requirement (ENG-1635).

    Entries land in `pip install` argv unquoted, so a flag-shaped string
    ("--index-url=…") would redirect resolution for every package in the
    call — dependency confusion through the one install path this ticket
    leaves open. Direct-URL requirements ("name @ https://…") are refused
    for the same reason: installs come from the configured index or not
    at all.
    """
    from packaging.requirements import InvalidRequirement, Requirement

    try:
        return Requirement(str(entry)).url is not None
    except InvalidRequirement:
        return True


def reject_invalid_packages(packages: list[str]) -> str | None:
    """The shared "Install refused" message, or None when every entry parses.

    Called before install_packages at the model-facing call sites (exec's
    'packages' array and the install action), and again inside the local
    runtime as the belt for every other caller (requirements.txt launches).
    """
    bad = [str(p) for p in packages if invalid_package_spec(p)]
    if not bad:
        return None
    return (
        "Install refused: not a plain PyPI package specifier: "
        + ", ".join(repr(p) for p in bad)
        + ". Use requirement syntax like 'requests' or 'requests==2.32.0' — "
        "flags, URLs, and local paths are never passed to pip."
    )


# Heads of the runtime's own kill messages (`backends/local.py`). These lead
# with the cause and append recovery notes after it, the reverse of a
# traceback — and the ACC kill events, nudge routing and kill-loop detector
# all key on the head.
KILL_ERROR_PREFIXES = ("Cancelled", "Cell timed out", "Cell killed")


def cell_error_headline(error: str | None) -> str:
    """The one line of a cell error that names the cause.

    A traceback ends with its cause, so that is the LAST line — never a hint
    prepended above it. A runtime kill message starts with its cause, so
    there it is the FIRST line; its last line is a recovery note that says
    nothing about why the cell died.
    """
    error = (error or "").strip()
    if not error:
        return ""
    lines = error.splitlines()
    return lines[0] if error.startswith(KILL_ERROR_PREFIXES) else lines[-1]


def cell_failure_reason(error: str | None) -> str:
    """The machine-comparable failure key: the cell error's headline, capped.
    Tests call this rather than restating the extraction."""
    return cell_error_headline(error)[:160]


def install_call_installed_something(result: str) -> bool:
    """True if install_packages ran pip and it succeeded.

    Package-level, not request-level: a request naming one cached and one new
    package still counts as "installed" for every name in it (install_packages
    doesn't report which of the requested names were actually new).
    """
    return result not in _INSTALL_NOOP_RESULTS and not install_call_failed(result)


def send_package_install_event(session: ChatSession, packages: list[str]) -> None:
    """Fire one telemetry event per package named in the call — name only, never code."""
    try:
        settings = getattr(session, "_settings", None)
        if settings is None or not hasattr(settings, "analytics_enabled"):
            from anton.config.settings import AntonSettings

            settings = AntonSettings()
        from anton.analytics import send_event

        for package in packages:
            send_event(settings, "scratchpad_package_installed", package=package)
    except Exception:
        pass


_DISCOVERY_MAX_PADS = 10
_DISCOVERY_MAX_ROOT_ENTRIES = 30


def build_workspace_discovery_context(manager) -> str:
    """Compact turn-start block: known pads + project-root names (ENG-578).

    Cold-start blindness made the agent invent pad names and rebuild state it
    already had on disk. Source of truth is agent_pads(), so system-created
    pads never leak in. Best-effort by construction: any failure degrades to
    omitting that part or the whole block, never to breaking the turn.

    This block sits inside the cached system prompt, ahead of the whole
    conversation. Its bytes must depend only on the set of pad and file names:
    anything that varies while that set is unchanged re-writes the conversation
    cache on the next turn.
    """
    pads_line = ""
    try:
        known = sorted(manager.agent_pads())
        if known:
            shown = known[:_DISCOVERY_MAX_PADS]
            # Names only: no snapshot age, no live/active flag. Both changed
            # between turns without the workspace changing.
            more = (
                f" … and {len(known) - len(shown)} more"
                if len(known) > len(shown)
                else ""
            )
            pads_line = (
                "\nScratchpads for this conversation: "
                + ", ".join(shown)
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
    if err.startswith(KILL_ERROR_PREFIXES):
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
    (empty code, a single-scratchpad challenge, a change the user didn't
    approve, or a failed package install).
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
    refused_name = refuse_pad_name(name)
    if refused_name is not None:
        return refused_name
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
                # "Not run: " opens every result for a cell that never ran, so a
                # host can mark the step as not run (cowork-server's
                # classify_cell_status).
                f"Not run: you already have an active scratchpad ('{existing}') with live state "
                f"(imports, variables, fetched data). Starting a new one named '{name}' "
                "creates a SEPARATE, empty environment — nothing from the existing "
                "scratchpad is available there, so you'd re-import and re-fetch. Reuse the "
                "existing scratchpad for this task; it is stateful across cells. If you "
                "genuinely need an isolated environment, call scratchpad exec again with "
                "confirm_new_scratchpad=true."
            ),
            ok=True,
            # Named so a caller that records executed cells can tell this
            # guidance from a cell that ran (`discovery.engine`).
            reason="new_scratchpad_challenged",
        )
    seen.add(name)
    if manager is not None and hasattr(manager, "record_agent_pad"):
        try:
            manager.record_agent_pad(name)
        except Exception:
            pass

    pad = await session._scratchpads.get_or_create(name)

    # Before the installs, so a refused cell installs nothing either.
    held = await hold_unapproved_write(session, pad=pad, code=code)
    if held is not None:
        return held

    # Install explicitly declared packages before running the cell
    packages = tc_input.get("packages", [])
    if packages:
        refused = reject_invalid_packages(packages)
        if refused:
            return ToolOutcome(
                content=refused, ok=False, reason="package_install_rejected"
            )
        install_result = await pad.install_packages(packages)
        # The substring check against install_packages' message is this
        # handler's own protocol with the runtime (its return shape predates
        # ToolOutcome); the verdict it produces is explicit from here on out.
        if install_call_failed(install_result):
            return ToolOutcome(
                content=install_result, ok=False, reason="package_install_failed"
            )
        if install_call_installed_something(install_result):
            send_package_install_event(session, packages)

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

    cowork-server's figure sources read these labels, the truncation note and
    the no-output sentence (cowork/harnesses/anton_harness/stream_formatter.py),
    so a change here needs the same change there.
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
