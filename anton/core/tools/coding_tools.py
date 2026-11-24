"""Native general-purpose coding and cowork tools: bash, write_file, edit_file,
grep_search, and find_files.

Designed to work alongside Anton's persistent Python `scratchpad` and `read_text_file`:
- Respects `CellSandbox` (Landlock) isolation and `WriteGate` (write-approval) policies
  fail-closed so multi-tenant sandboxes and approval gates can never be bypassed.
- Blocks Anton/Cowork/user credential files (`.anton/.env*`, `.anton/data_vault`,
  `.anton/.published.json`, `.anton/secrets.json`, `.cowork/.env*`, `.ssh`, `.aws`,
  `.gnupg`) across every file/search tool.
- Strips provider/gateway/Cowork secrets from `bash` subprocess environments and runs
  `scrub_output()` before head+tail truncation so secrets can never leak across cuts.
- Preserves Cowork artifact tracking and linting (`capture_turn_pages`,
  `snapshot_cell_artifacts`, `check_cell_artifacts`) whenever an artifact folder exists
  and is modified.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import fnmatch
import os
import re
import selectors
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from anton.core.backends.sandbox import cell_sandbox_of
from anton.core.tools.registry import ToolOutcome
from anton.core.tools.tool_defs import ToolDef
from anton.utils.datasources import _PROVIDER_SECRET_VARS, scrub_output

if TYPE_CHECKING:
    from anton.core.session import ChatSession

# Output caps tuned to keep high signal (especially error tails) without bloating context.
MAX_BASH_OUTPUT_CHARS = 16_000
BASH_HEAD_CHARS = 4_000
BASH_TAIL_CHARS = 12_000
MAX_BASH_CAPTURE_HEAD_BYTES = 256 * 1024
MAX_BASH_CAPTURE_TAIL_BYTES = 256 * 1024
BASH_POST_EXIT_DRAIN_S = 0.3
DEFAULT_BASH_TIMEOUT_S = 120
MAX_BASH_TIMEOUT_S = 1800

MAX_GREP_OUTPUT_CHARS = 12_000
MAX_GREP_LINE_CHARS = 400
DEFAULT_GREP_RESULTS = 80

# Cap edit_file diff echo so large edits don't redundantly double context size
# (the model already has its exact old_string/new_string in the preceding tool_use).
MAX_EDIT_DIFF_LINES = 24
MAX_EDIT_DIFF_CHARS = 1_200

DEFAULT_FIND_RESULTS = 200
SKIP_WALK_DIRS = frozenset({
    ".git",
    ".anton",
    ".cowork",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
})

# Environment variables stripped from `bash` subprocesses (Anton/Cowork/provider secrets).
_STRIPPED_ENV_EXACT = frozenset(
    set(_PROVIDER_SECRET_VARS)
    | {
        "MINDS_API_KEY",
        "MINDSHUB_API_KEY",
        "DATABASE_URI",
        "EXA_API_KEY",
        "BRAVE_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTON_POSTHOG_KEY",
    }
)
_STRIPPED_ENV_PREFIXES = ("ANTON_", "COWORK_")

# Pure, async, non-mutating, non-eliciting read-only tools safe to execute
# concurrently when a single LLM response batches calls exclusively from this set.
READ_ONLY_TOOLS = frozenset({
    "read_text_file",
    "read_image",
    "grep_search",
    "find_files",
})


_CONFIG_SECRET_SUBDIRS = frozenset({"gh", "gcloud", "stripe", "hub", "helm", "git", "pip", "op"})


def is_protected_secret_path(resolved: Path) -> bool:
    """Return True if ``resolved`` points to an Anton/Cowork/user credential file."""
    parts = resolved.parts
    lower_parts = tuple(p.lower() for p in parts)

    # Block ~/.ssh, ~/.aws, ~/.gnupg, cloud/CLI credential dirs, and credential rc files
    for sensitive_dir in (".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure", ".gcloud"):
        if sensitive_dir in lower_parts:
            return True

    # Block known credential subdirectories under .config (and $HOME/.config itself),
    # while preserving legitimate workspace project configs like <repo>/.config/nextest.toml.
    for i, part in enumerate(lower_parts[:-1]):
        if part == ".config" and lower_parts[i + 1] in _CONFIG_SECRET_SUBDIRS:
            return True
    try:
        home_cfg = (Path.home() / ".config").resolve()
        if resolved == home_cfg or home_cfg in resolved.parents:
            return True
    except (OSError, RuntimeError):
        pass

    name = resolved.name.lower()
    if name in (
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".git-credentials",
        ".pgpass",
        ".envrc",
        ".bash_history",
        ".zsh_history",
        "credentials.json",
        "secrets.json",
        ".published.json",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "id_dsa",
    ):
        return True

    # Block Anton/Cowork secret files and vaults (including .cowork-* variants)
    for i, part in enumerate(lower_parts):
        if part == ".anton" or part == ".cowork" or part.startswith(".cowork-"):
            rest = lower_parts[i + 1 :]
            if not rest:
                return True
            first = rest[0]
            if (
                first == ".env"
                or first.startswith(".env.")
                or first == "data_vault"
                or first in (".published.json", "secrets.json")
            ):
                return True

    # Block project-level .env secrets (while allowing harmless .env.example / .env.sample / .env.template)
    if name == ".env" or (
        name.startswith(".env.")
        and not name.endswith((".example", ".sample", ".template", ".dist"))
    ):
        return True

    return False


def _workspace_root(session: "ChatSession") -> Path:
    ws = getattr(session, "_workspace", None)
    if ws is not None:
        base = getattr(ws, "base", None) or getattr(ws, "root", None) or getattr(ws, "path", None)
        if isinstance(base, (str, Path)):
            return Path(base).resolve()
    settings = getattr(session, "_settings", None)
    if settings is not None:
        wp = getattr(settings, "workspace_path", None)
        if isinstance(wp, (str, Path)):
            return Path(wp).resolve()
    return Path.cwd().resolve()


def _resolve_path(session: "ChatSession", raw: str) -> Path:
    policy = cell_sandbox_of(getattr(session, "_scratchpads", None))
    if policy is not None and (raw == "~" or raw.startswith("~/")):
        p = policy.home_dir / raw[1:].lstrip("/")
    else:
        p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = _workspace_root(session) / p
    return p.resolve()


def _truncate_head_tail(text: str, *, max_chars: int, head_chars: int, tail_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - head_chars - tail_chars
    omitted_lines = text[head_chars : len(text) - tail_chars].count("\n")
    return (
        f"{text[:head_chars]}\n"
        f"... [{omitted:,} chars / ~{omitted_lines:,} lines omitted] ...\n"
        f"{text[-tail_chars:]}"
    )


# Linear-time ANSI CSI / OSC escape strippers for terminal output hygiene.
_ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ANSI_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


_REPEAT_LINE_COLLAPSE_THRESHOLD = 5


def _clean_terminal_output(text: str) -> str:
    """Normalize terminal progress-bar overwrites, ANSI escapes, and log spam.

    Strips ANSI CSI/OSC sequences, collapses carriage-return (`\\r`) progress
    updates (keeping the final non-empty segment per line while preserving
    CRLF-separated lines by normalizing `\\r\\n` to `\\n`), and collapses runs
    of 5+ identical consecutive non-empty lines so repetitive warning/progress
    floods do not waste context tokens or push real errors out of truncation bounds.
    """
    if "\x1b" in text:
        text = _ANSI_OSC_RE.sub("", text)
        text = _ANSI_CSI_RE.sub("", text)
    if "\r" in text:
        text = text.replace("\r\n", "\n")
        if "\r" in text:
            cleaned_lines: list[str] = []
            for line in text.split("\n"):
                if "\r" in line:
                    parts = [p for p in line.split("\r") if p]
                    cleaned_lines.append(parts[-1] if parts else "")
                else:
                    cleaned_lines.append(line)
            text = "\n".join(cleaned_lines)
    if "\n" in text:
        lines = text.split("\n")
        if len(lines) >= _REPEAT_LINE_COLLAPSE_THRESHOLD:
            out_lines: list[str] = []
            i = 0
            n = len(lines)
            while i < n:
                line = lines[i]
                j = i + 1
                if line.strip():
                    while j < n and lines[j] == line:
                        j += 1
                run_len = j - i
                if run_len >= _REPEAT_LINE_COLLAPSE_THRESHOLD:
                    out_lines.append(line)
                    out_lines.append(f"... (repeated {run_len - 1} more times)")
                else:
                    out_lines.extend(lines[i:j])
                i = j
            if len(out_lines) < n:
                text = "\n".join(out_lines)
    return text


def _scrub_and_clean_terminal(raw: bytes) -> str:
    """Decode bytes, scrub secrets before and after terminal normalization."""
    text = raw.decode("utf-8", errors="replace")
    scrubbed = scrub_output(text)
    cleaned = _clean_terminal_output(scrubbed)
    if cleaned != scrubbed:
        cleaned = scrub_output(cleaned)
    return cleaned


def _sanitized_shell_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for k, v in os.environ.items():
        ku = k.upper()
        if ku in _STRIPPED_ENV_EXACT or ku.startswith(_STRIPPED_ENV_PREFIXES):
            continue
        env[k] = v
    env.setdefault("DEBIAN_FRONTEND", "noninteractive")
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("PAGER", "cat")
    return env


def _is_under_dir(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except (ValueError, OSError):
        return False


async def _maybe_snapshot_artifacts(session: "ChatSession", target: Path | None = None):
    """Snapshot artifacts only when an artifact store exists and is potentially affected."""
    from anton.core.tools.tool_handlers import (
        capture_turn_pages,
        resolve_artifact_store,
        snapshot_cell_artifacts,
    )

    store = resolve_artifact_store(session)
    if store is None:
        return None, None
    store_root = getattr(store, "root", None)
    if target is not None and store_root is not None and not _is_under_dir(target, Path(store_root)):
        return None, None
    if target is None and store_root is not None and not Path(store_root).exists():
        return None, None
    await capture_turn_pages(session, store=store)
    before = await snapshot_cell_artifacts(session=session, store=store)
    return store, before


async def _maybe_check_artifacts(
    session: "ChatSession", store, before, content: str, *, ok: bool
) -> str | list[dict]:
    if store is None or before is None:
        return content
    from anton.core.tools.tool_handlers import cell_content_with_lint, check_cell_artifacts

    lint = await check_cell_artifacts(session, store=store, before=before)
    return cell_content_with_lint(content, lint, cell_ok=ok)


def _write_tools_blocked_reason(session: "ChatSession") -> str | None:
    """Fail-closed check: refuse unsandboxed coding write/exec tools when Landlock
    CellSandbox or WriteGate approval mode is active.
    """
    if cell_sandbox_of(getattr(session, "_scratchpads", None)) is not None:
        return "This tool is disabled under CellSandbox isolation; use `scratchpad` instead."
    if getattr(session, "_write_gate", None) is not None:
        return "Write approval (`write_approval`) is active; use `scratchpad` so changes route through user approval."
    return None


async def handle_bash(session: "ChatSession", tc_input: dict) -> ToolOutcome:
    """Run a shell command with secret hygiene, output scrubbing, and clean timeout."""
    blocked = _write_tools_blocked_reason(session)
    if blocked:
        return ToolOutcome(content=f"Error: {blocked}", ok=False, reason="policy_blocked")

    command = tc_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return ToolOutcome(content="Error: `command` must be a non-empty string.", ok=False, reason="missing_command")

    raw_timeout = tc_input.get("timeout_seconds", DEFAULT_BASH_TIMEOUT_S)
    try:
        timeout_s = max(1, min(int(raw_timeout), MAX_BASH_TIMEOUT_S))
    except (TypeError, ValueError):
        timeout_s = DEFAULT_BASH_TIMEOUT_S

    base = _workspace_root(session)
    raw_wd = tc_input.get("workdir")
    if isinstance(raw_wd, str) and raw_wd.strip():
        cwd = _resolve_path(session, raw_wd.strip())
        if not cwd.is_dir():
            return ToolOutcome(content=f"Error: workdir does not exist: {raw_wd}", ok=False, reason="invalid_workdir")
    else:
        cwd = base

    store, before_artifacts = await _maybe_snapshot_artifacts(session, target=None)
    env = _sanitized_shell_env()

    is_windows = os.name == "nt"
    if is_windows:
        bash_bin = shutil.which("bash")
        argv = [bash_bin, "-c", command] if bash_bin else ["cmd", "/c", command]
        spawn_kwargs: dict = {}
    else:
        shell_bin = "/bin/bash" if Path("/bin/bash").exists() else "/bin/sh"
        # Run the user command inside a child shell whose $PPID is the disposable
        # outer session leader (not Anton's host Python PID). Using a non-empty
        # `trap ":" INT TERM` (NOT `""`/SIG_IGN, so execve resets both to SIG_DFL
        # in the inner shell) keeps the outer shell waiting for the inner command
        # on both `kill -INT $PPID` and graceful `SIGTERM` timeouts across bash
        # and dash, while the trailing `exit "$rc"` defeats tail-call `exec`.
        argv = [
            shell_bin,
            "-c",
            'trap ":" INT TERM; rc=0; "$1" -c -- "$2" || rc=$?; exit "$rc"',
            "_",
            shell_bin,
            command,
        ]
        spawn_kwargs = {"start_new_session": True}

    t0 = time.monotonic()
    timed_out = False
    try:
        # Use subprocess.Popen + real OS waitpid() rather than asyncio.subprocess:
        # CPython's BaseSubprocessTransport.wait() blocks until ALL pipe FDs hit EOF,
        # which hangs for timeout_s whenever a command launches a background daemon
        # (e.g. `cd /app && nohup server &`) that inherits stdout/stderr FDs.
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            **spawn_kwargs,
        )
    except OSError as exc:
        return ToolOutcome(content=f"Error starting shell: {exc}", ok=False, reason="spawn_failed")

    def _kill_proc(*, force: bool) -> None:
        if is_windows:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                proc.kill() if force else proc.terminate()
        else:
            # With start_new_session=True, proc.pid IS the session/pgrp leader ID;
            # signalling proc.pid directly works even after the shell leader exits
            # while children remain in the process group (unlike os.getpgid(proc.pid)).
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)

    buf_lock = threading.Lock()
    head_buf = bytearray()
    tail_buf = bytearray()
    dropped_bytes = 0
    drain_deadline: list[float | None] = [None]

    def _append_chunk(chunk: bytes) -> None:
        nonlocal dropped_bytes
        with buf_lock:
            if len(head_buf) < MAX_BASH_CAPTURE_HEAD_BYTES:
                take = min(len(chunk), MAX_BASH_CAPTURE_HEAD_BYTES - len(head_buf))
                head_buf.extend(chunk[:take])
                rem = chunk[take:]
            else:
                rem = chunk
            if rem:
                tail_buf.extend(rem)
                if len(tail_buf) > MAX_BASH_CAPTURE_TAIL_BYTES:
                    overflow = len(tail_buf) - MAX_BASH_CAPTURE_TAIL_BYTES
                    del tail_buf[:overflow]
                    dropped_bytes += overflow

    def _reader_thread() -> None:
        # Reader exclusively owns `stream` and closes it in its finally block so
        # the main thread never closes an FD while os.read() is in flight.
        stream = proc.stdout
        if stream is None:
            return
        try:
            if is_windows:
                fd = stream.fileno()
                while True:
                    deadline = drain_deadline[0]
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    try:
                        chunk = os.read(fd, 8192)
                    except (OSError, ValueError):
                        break
                    if not chunk:
                        break
                    _append_chunk(chunk)
                return

            fd = stream.fileno()
            sel = selectors.DefaultSelector()
            try:
                sel.register(fd, selectors.EVENT_READ)
                while True:
                    deadline = drain_deadline[0]
                    if deadline is not None:
                        rem_s = deadline - time.monotonic()
                        if rem_s <= 0:
                            break
                        poll_s = min(0.05, rem_s)
                    else:
                        poll_s = 0.05
                    events = sel.select(timeout=poll_s)
                    if not events:
                        if drain_deadline[0] is not None and time.monotonic() >= drain_deadline[0]:
                            break
                        continue
                    try:
                        chunk = os.read(fd, 8192)
                    except (OSError, ValueError):
                        break
                    if not chunk:
                        break
                    _append_chunk(chunk)
            finally:
                sel.close()
        finally:
            with contextlib.suppress(OSError, ValueError):
                stream.close()

    loop = asyncio.get_running_loop()
    exited = asyncio.Event()

    def _waiter_thread() -> None:
        try:
            proc.wait()
        finally:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(exited.set)

    reader = threading.Thread(target=_reader_thread, daemon=True)
    waiter = threading.Thread(target=_waiter_thread, daemon=True)
    reader.start()
    waiter.start()

    try:
        try:
            await asyncio.wait_for(exited.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            timed_out = True
            _kill_proc(force=False)
            try:
                await asyncio.wait_for(exited.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                _kill_proc(force=True)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(exited.wait(), timeout=2.0)
            else:
                # Reap any SIGTERM-ignoring grandchildren left in the timed-out
                # process group even if the outer/inner shell exited cleanly.
                _kill_proc(force=True)

        # Give buffered output from the exited shell a brief window to flush,
        # then let the reader exit and close its pipe cleanly if a background
        # daemon still holds the write end open.
        drain_deadline[0] = time.monotonic() + BASH_POST_EXIT_DRAIN_S
        await asyncio.to_thread(reader.join, BASH_POST_EXIT_DRAIN_S + 0.5)
    except asyncio.CancelledError:
        drain_deadline[0] = 0.0
        if proc.poll() is None:
            _kill_proc(force=True)
        raise

    elapsed = time.monotonic() - t0
    rc = proc.returncode if proc.returncode is not None else -1
    # Snapshot captured buffers under lock; scrub secrets BEFORE truncation so
    # tokens can never split across the cut boundary.
    with buf_lock:
        head_snap = bytes(head_buf)
        tail_snap = bytes(tail_buf)
        dropped_snap = dropped_bytes
    if dropped_snap > 0:
        head_scrubbed = _scrub_and_clean_terminal(head_snap)
        tail_scrubbed = _scrub_and_clean_terminal(tail_snap)
        scrubbed = (
            f"{head_scrubbed}\n"
            f"... [omitted {dropped_snap:,} middle bytes of command output] ...\n"
            f"{tail_scrubbed}"
        )
    else:
        scrubbed = _scrub_and_clean_terminal(head_snap + tail_snap)
    output = _truncate_head_tail(
        scrubbed,
        max_chars=MAX_BASH_OUTPUT_CHARS,
        head_chars=BASH_HEAD_CHARS,
        tail_chars=BASH_TAIL_CHARS,
    )

    if timed_out:
        header = f"Command timed out after {timeout_s}s (exit code {rc})."
        body = f"{header}\n{output}" if output.strip() else header
        final = await _maybe_check_artifacts(session, store, before_artifacts, body, ok=False)
        return ToolOutcome(content=final, ok=False, reason="timeout")

    if rc == 0:
        body = (
            f"(exit 0, {elapsed:.1f}s, no output)"
            if not output.strip()
            else f"(exit 0, {elapsed:.1f}s)\n{output}"
        )
        final = await _maybe_check_artifacts(session, store, before_artifacts, body, ok=True)
        return ToolOutcome(content=final, ok=True)

    body = f"(exit {rc}, {elapsed:.1f}s)\n{output}" if output.strip() else f"(exit {rc}, {elapsed:.1f}s, no output)"
    final = await _maybe_check_artifacts(session, store, before_artifacts, body, ok=False)
    return ToolOutcome(content=final, ok=False, reason=f"exit_{rc}")


async def handle_write_file(session: "ChatSession", tc_input: dict) -> ToolOutcome:
    """Create or overwrite a UTF-8 text file cleanly."""
    blocked = _write_tools_blocked_reason(session)
    if blocked:
        return ToolOutcome(content=f"Error: {blocked}", ok=False, reason="policy_blocked")

    raw_path = tc_input.get("path")
    content = tc_input.get("content")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return ToolOutcome(content="Error: `path` is required.", ok=False, reason="missing_path")
    if not isinstance(content, str):
        return ToolOutcome(content="Error: `content` must be a string.", ok=False, reason="invalid_content")

    try:
        target = _resolve_path(session, raw_path.strip())
    except (ValueError, OSError) as exc:
        return ToolOutcome(content=f"Error: invalid path {raw_path!r}: {exc}", ok=False, reason="invalid_path")
    if is_protected_secret_path(target):
        return ToolOutcome(content=f"Error: access denied for protected path {raw_path}.", ok=False, reason="access_denied")

    store, before_artifacts = await _maybe_snapshot_artifacts(session, target=target)

    def _do_write() -> ToolOutcome:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            data = content.encode("utf-8")
            target.write_bytes(data)
        except OSError as exc:
            return ToolOutcome(content=f"Error writing {raw_path}: {exc}", ok=False, reason="write_failed")
        n_lines = content.count("\n") + (0 if (not content or content.endswith("\n")) else 1)
        return ToolOutcome(
            content=f"Wrote {raw_path} ({n_lines} lines, {len(data):,} bytes).",
            ok=True,
        )

    outcome = await asyncio.to_thread(_do_write)
    if outcome.ok:
        final = await _maybe_check_artifacts(session, store, before_artifacts, str(outcome.content), ok=True)
        outcome = ToolOutcome(content=final, ok=True)
    return outcome


def _build_edit_mismatch_hint(working: str, old_string: str) -> str:
    """Return a scrubbed diagnostic hint when `edit_file`'s `old_string` is not found.

    Scrubs full `working` *before* searching or slicing so neither raw multi-line
    secrets nor single-line secrets can ever leak or be probed via mismatch hints.
    """
    o_lines = old_string.splitlines()
    if not o_lines:
        return ""
    scrubbed_lines = scrub_output(working).splitlines()

    def _format_window(lo: int, hi: int) -> str:
        lo = max(0, lo)
        hi = min(len(scrubbed_lines), hi)
        snippet = "\n".join(
            f"  {ln_no + 1}: {scrubbed_lines[ln_no][:200]}" for ln_no in range(lo, hi)
        )
        if len(snippet) > 900:
            snippet = snippet[:900] + "\n  ..."
        return snippet

    # 1. Detect whitespace/indentation-only mismatches with boundary-safe checks.
    norm_o_lines = [" ".join(ln.split()) for ln in o_lines]
    norm_target_tokens = old_string.split()
    norm_target_str = " ".join(norm_target_tokens)
    anchor_ws_idx: int | None = None
    anchor_ws_len = 0
    for idx, s in enumerate(norm_o_lines):
        if len(s) > anchor_ws_len:
            anchor_ws_len = len(s)
            anchor_ws_idx = idx

    if (
        len(norm_target_str) >= 8
        and anchor_ws_idx is not None
        and len(o_lines) <= 200
        and len(scrubbed_lines) <= 20_000
    ):
        anchor_ws = norm_o_lines[anchor_ws_idx]
        win = len(o_lines)
        has_trailing_newline = old_string.endswith(("\n", "\r"))
        checks = 0
        max_ws_checks = 500
        for i, line in enumerate(scrubbed_lines):
            if anchor_ws not in " ".join(line.split()):
                continue
            start = i - anchor_ws_idx
            if start < 0 or start + win > len(scrubbed_lines):
                continue
            checks += 1
            if checks > max_ws_checks:
                break
            if win >= 2:
                matched = all(
                    " ".join(scrubbed_lines[start + k].split()) == norm_o_lines[k]
                    for k in range(win)
                )
            elif has_trailing_newline:
                matched = " ".join(scrubbed_lines[start].split()) == norm_o_lines[0]
            else:
                line_tokens = scrubbed_lines[start].split()
                n_tok = len(norm_target_tokens)
                matched = n_tok > 0 and any(
                    line_tokens[pos : pos + n_tok] == norm_target_tokens
                    for pos in range(len(line_tokens) - n_tok + 1)
                )
            if matched:
                lo = max(0, start - 2)
                hi = min(len(scrubbed_lines), start + min(win, 10) + 2)
                snippet = _format_window(lo, hi)
                return (
                    f" Text matches near line {start + 1} with different whitespace/indentation. "
                    f"Context around line {start + 1}:\n{snippet}"
                )

    # 2. Try unique non-empty stripped lines in descending distinctiveness order
    # (longest lines with len >= 4 first, then shorter fallbacks), capped to avoid
    # quadratic scans on large non-matching files, stopping after 5 line hits.
    seen_candidates: set[str] = set()
    long_candidates: list[tuple[int, int, str]] = []
    short_candidates: list[tuple[int, int, str]] = []
    for idx, ln in enumerate(o_lines):
        stripped = ln.strip()
        if not stripped or stripped in seen_candidates:
            continue
        seen_candidates.add(stripped)
        if len(stripped) >= 4:
            long_candidates.append((-len(stripped), idx, stripped))
        else:
            short_candidates.append((-len(stripped), idx, stripped))
    long_candidates.sort()
    short_candidates.sort()

    hits: list[int] = []
    anchor_offset = 0
    for _, idx, candidate in (*long_candidates, *short_candidates)[:32]:
        cand_hits: list[int] = []
        for i, line in enumerate(scrubbed_lines):
            if candidate in line:
                cand_hits.append(i + 1)
                if len(cand_hits) >= 5:
                    break
        if cand_hits:
            hits = cand_hits
            anchor_offset = idx
            break
    if not hits:
        return ""
    center = hits[0] - 1
    est_start = max(0, center - anchor_offset)
    lo = max(0, min(center - 3, est_start - 2))
    hi = min(len(scrubbed_lines), max(center + 4, est_start + min(len(o_lines), 8) + 2))
    if hi - lo > 12:
        lo = max(0, center - 4)
        hi = min(len(scrubbed_lines), center + 8)
    snippet = _format_window(lo, hi)
    return f" Closest match appears near lines {hits}. Context around line {hits[0]}:\n{snippet}"


async def handle_edit_file(session: "ChatSession", tc_input: dict) -> ToolOutcome:
    """Surgical exact-match string replacement with unified diff feedback."""
    blocked = _write_tools_blocked_reason(session)
    if blocked:
        return ToolOutcome(content=f"Error: {blocked}", ok=False, reason="policy_blocked")

    raw_path = tc_input.get("path")
    raw_edits = tc_input.get("edits")
    old_string = tc_input.get("old_string")
    new_string = tc_input.get("new_string")
    replace_all = bool(tc_input.get("replace_all", False))

    if not isinstance(raw_path, str) or not raw_path.strip():
        return ToolOutcome(content="Error: `path` is required.", ok=False, reason="missing_path")

    edit_specs: list[tuple[str, str, bool]] = []
    from_edits_array = raw_edits is not None
    if from_edits_array:
        if old_string is not None or new_string is not None or "replace_all" in tc_input:
            return ToolOutcome(
                content="Error: pass either `edits` OR (`old_string`, `new_string`, optional `replace_all`), not both.",
                ok=False,
                reason="invalid_args",
            )
        if not isinstance(raw_edits, list) or not raw_edits:
            return ToolOutcome(
                content="Error: `edits` must be a non-empty list of {old_string, new_string, replace_all?} objects.",
                ok=False,
                reason="invalid_edits",
            )
        for idx, item in enumerate(raw_edits):
            if not isinstance(item, dict):
                return ToolOutcome(
                    content=f"Error: edits[{idx}] must be an object.",
                    ok=False,
                    reason="invalid_edits",
                )
            o_s = item.get("old_string")
            n_s = item.get("new_string")
            r_all = bool(item.get("replace_all", False))
            if not isinstance(o_s, str) or not o_s:
                return ToolOutcome(
                    content=f"Error: edits[{idx}].old_string must be a non-empty string.",
                    ok=False,
                    reason="missing_old_string",
                )
            if not isinstance(n_s, str):
                return ToolOutcome(
                    content=f"Error: edits[{idx}].new_string must be a string.",
                    ok=False,
                    reason="missing_new_string",
                )
            if o_s == n_s:
                return ToolOutcome(
                    content=f"Error: edits[{idx}].new_string is identical to old_string.",
                    ok=False,
                    reason="noop_edit",
                )
            edit_specs.append((o_s, n_s, r_all))
    else:
        if not isinstance(old_string, str) or not old_string:
            return ToolOutcome(content="Error: `old_string` must be a non-empty string.", ok=False, reason="missing_old_string")
        if not isinstance(new_string, str):
            return ToolOutcome(content="Error: `new_string` must be a string.", ok=False, reason="missing_new_string")
        if old_string == new_string:
            return ToolOutcome(content="Error: `new_string` is identical to `old_string`.", ok=False, reason="noop_edit")
        edit_specs.append((old_string, new_string, replace_all))

    try:
        target = _resolve_path(session, raw_path.strip())
    except (ValueError, OSError) as exc:
        return ToolOutcome(content=f"Error: invalid path {raw_path!r}: {exc}", ok=False, reason="invalid_path")
    if is_protected_secret_path(target):
        return ToolOutcome(content=f"Error: access denied for protected path {raw_path}.", ok=False, reason="access_denied")

    store, before_artifacts = await _maybe_snapshot_artifacts(session, target=target)

    def _do_edit() -> ToolOutcome:
        if not target.is_file():
            return ToolOutcome(content=f"Error: file not found: {raw_path}", ok=False, reason="path_not_found")
        try:
            original = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return ToolOutcome(content=f"Error reading {raw_path}: {exc}", ok=False, reason="read_failed")

        working = original
        total_replaced = 0
        multi = len(edit_specs) > 1
        for idx, (o_s, n_s, r_all) in enumerate(edit_specs):
            label = f"edits[{idx}].old_string" if from_edits_array else "`old_string`"
            count = working.count(o_s)
            if count == 0:
                hint = _build_edit_mismatch_hint(working, o_s)
                rollback_note = " No changes were written." if multi else ""
                return ToolOutcome(
                    content=f"Error: {label} not found in {raw_path}.{hint}{rollback_note}",
                    ok=False,
                    reason="old_string_not_found",
                )

            if count > 1 and not r_all:
                positions: list[int] = []
                pos_idx = 0
                while len(positions) < 10:
                    found = working.find(o_s, pos_idx)
                    if found == -1:
                        break
                    positions.append(working.count("\n", 0, found) + 1)
                    pos_idx = found + len(o_s)
                rollback_note = " No changes were written." if multi else ""
                return ToolOutcome(
                    content=(
                        f"Error: {label} matched {count} times in {raw_path} (lines {positions}). "
                        f"Provide more surrounding context to make it unique, or pass `replace_all=true`.{rollback_note}"
                    ),
                    ok=False,
                    reason="ambiguous_edit",
                )

            working = working.replace(o_s, n_s) if r_all else working.replace(o_s, n_s, 1)
            total_replaced += count if r_all else 1

        try:
            target.write_text(working, encoding="utf-8")
        except OSError as exc:
            return ToolOutcome(content=f"Error writing {raw_path}: {exc}", ok=False, reason="write_failed")

        orig_lines = original.splitlines()
        new_lines = working.splitlines()
        raw_diff = list(
            difflib.unified_diff(
                orig_lines,
                new_lines,
                fromfile=f"a/{raw_path}",
                tofile=f"b/{raw_path}",
                n=1,
                lineterm="",
            )
        )
        # File headers (`--- a/...`, `+++ b/...`) are strictly indices 0 and 1;
        # counting from index 2 onward keeps lines starting with `--` or `++`
        # (e.g. SQL comments, YAML `---`, C++ `++i`) counted accurately.
        body_diff = raw_diff[2:] if len(raw_diff) >= 2 else raw_diff
        added = sum(1 for ln in body_diff if ln.startswith("+"))
        removed = sum(1 for ln in body_diff if ln.startswith("-"))

        # Scrub raw file contents BEFORE building the preview diff so multi-line
        # secrets (e.g. PEM private keys or `Bearer\n<token>`) are not split by
        # unified-diff `+`/`-`/` ` line prefixes, then scrub the rendered diff
        # for header/path safety BEFORE capping lines and characters.
        scrubbed_orig = scrub_output(original)
        scrubbed_new = scrub_output(working)
        if scrubbed_orig != original or scrubbed_new != working:
            preview_diff = list(
                difflib.unified_diff(
                    scrubbed_orig.splitlines(),
                    scrubbed_new.splitlines(),
                    fromfile=f"a/{raw_path}",
                    tofile=f"b/{raw_path}",
                    n=1,
                    lineterm="",
                )
            )
        else:
            preview_diff = raw_diff
        scrubbed_lines = scrub_output("\n".join(preview_diff)).splitlines()
        diff_text = "\n".join(scrubbed_lines[:MAX_EDIT_DIFF_LINES])
        truncated = len(scrubbed_lines) > MAX_EDIT_DIFF_LINES
        if len(diff_text) > MAX_EDIT_DIFF_CHARS:
            diff_text = diff_text[:MAX_EDIT_DIFF_CHARS].rstrip()
            truncated = True
        if truncated:
            diff_text += f"\n... (diff preview capped; +{added}/-{removed} lines applied)"
        return ToolOutcome(
            content=(
                f"Edited {raw_path} ({total_replaced} replacement{'s' if total_replaced != 1 else ''}, "
                f"+{added}/-{removed} lines; total {len(new_lines)} lines).\n{diff_text}"
            ),
            ok=True,
        )

    outcome = await asyncio.to_thread(_do_edit)
    if outcome.ok:
        final = await _maybe_check_artifacts(session, store, before_artifacts, str(outcome.content), ok=True)
        outcome = ToolOutcome(content=final, ok=True)
    return outcome


async def handle_grep_search(session: "ChatSession", tc_input: dict) -> ToolOutcome:
    """Fast regex/literal content search with CellSandbox and secret protection."""
    pattern = tc_input.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return ToolOutcome(content="Error: `pattern` is required.", ok=False, reason="missing_pattern")

    raw_path = tc_input.get("path") or "."
    glob_filter = tc_input.get("glob")
    case_insensitive = bool(tc_input.get("case_insensitive", False))
    try:
        max_results = max(1, min(int(tc_input.get("max_results", DEFAULT_GREP_RESULTS)), 500))
    except (TypeError, ValueError):
        max_results = DEFAULT_GREP_RESULTS
    try:
        context_lines = max(0, min(int(tc_input.get("context_lines", 0)), 5))
    except (TypeError, ValueError):
        context_lines = 0

    base = _workspace_root(session)
    policy = cell_sandbox_of(getattr(session, "_scratchpads", None))
    try:
        target = _resolve_path(session, str(raw_path).strip())
    except (ValueError, OSError) as exc:
        return ToolOutcome(content=f"Error: invalid path {raw_path!r}: {exc}", ok=False, reason="invalid_path")

    if is_protected_secret_path(target):
        return ToolOutcome(content=f"Error: access denied for protected path {raw_path}.", ok=False, reason="access_denied")
    if policy is not None and not policy.allows_read(target):
        return ToolOutcome(content=f"Error: access denied for {raw_path} under sandbox policy.", ok=False, reason="access_denied")
    if not target.exists():
        return ToolOutcome(content=f"Error: path does not exist: {raw_path}", ok=False, reason="path_not_found")

    # Under CellSandbox, always use the policy-checked Python walker so every file is verified.
    rg_bin = None if policy is not None else shutil.which("rg")
    if rg_bin:
        args = [
            rg_bin,
            "--null",
            "--with-filename",
            "--no-heading",
            "--line-number",
            "--color=never",
            "--max-columns",
            str(MAX_GREP_LINE_CHARS),
            "--max-columns-preview",
        ]
        if context_lines > 0:
            args.extend(["-C", str(context_lines)])
        if case_insensitive:
            args.append("-i")
        # User glob goes BEFORE security negative globs (ripgrep evaluates last-match-wins).
        if isinstance(glob_filter, str) and glob_filter.strip():
            args.extend(["--glob", glob_filter.strip()])
        for neg_glob in (
            "!.anton/**",
            "!.cowork/**",
            "!.cowork-*/**",
            "!.git/**",
            "!.ssh/**",
            "!.aws/**",
            "!.gnupg/**",
            "!.kube/**",
            "!.docker/**",
            "!.azure/**",
            "!.gcloud/**",
            "!.config/gh/**",
            "!.config/gcloud/**",
            "!.config/stripe/**",
            "!.config/hub/**",
            "!.config/helm/**",
            "!.config/git/**",
            "!.config/pip/**",
            "!.config/op/**",
            "!.env",
            "!.env.*",
            "!.envrc",
            "!credentials.json",
            "!secrets.json",
            "!.netrc",
            "!.npmrc",
            "!.pypirc",
            "!.git-credentials",
            "!.pgpass",
            "!.bash_history",
            "!.zsh_history",
            "!.published.json",
            "!id_rsa",
            "!id_ed25519",
            "!id_ecdsa",
            "!id_dsa",
            "!*.pem",
            "!*.key",
            "!*.p12",
            "!*.pfx",
            "!*.kdbx",
        ):
            args.extend(["--glob", neg_glob])
        args.extend(["--", pattern, str(target)])
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(base),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        except asyncio.CancelledError:
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            raise
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            return ToolOutcome(content="Error: grep_search timed out after 30s; narrow `path`.", ok=False, reason="timeout")

        if proc.returncode in (0, 1):
            raw_lines = stdout_b.decode("utf-8", errors="replace").splitlines()
            prefix = str(base) + "/"
            # Defense-in-depth: parse NUL-delimited path unambiguously and drop protected paths.
            # Count matching lines (not context lines) against max_results.
            safe_out: list[str] = []
            total_matches = 0
            shown_matches = 0
            pending_sep = False
            trailing_budget = 0
            last_shown_fp = ""
            for ln in raw_lines:
                if ln == "--":
                    trailing_budget = 0
                    if safe_out and shown_matches < max_results:
                        pending_sep = True
                    continue
                if "\x00" not in ln:
                    continue
                fp, rest = ln.split("\x00", 1)
                p = Path(fp)
                candidate = (p if p.is_absolute() else base / p).resolve()
                if is_protected_secret_path(candidate):
                    continue
                # Distinguish match line (`123:text`) vs context line (`123-text`)
                is_match = False
                sep = ":"
                m_num = re.match(r"^(\d+)([:-])", rest)
                if m_num:
                    sep = m_num.group(2)
                    is_match = sep == ":"
                if is_match:
                    total_matches += 1
                    if shown_matches >= max_results:
                        trailing_budget = 0
                        continue
                    shown_matches += 1
                    trailing_budget = context_lines
                    last_shown_fp = fp
                else:
                    if shown_matches >= max_results:
                        if pending_sep or fp != last_shown_fp or trailing_budget <= 0:
                            continue
                        trailing_budget -= 1
                if pending_sep:
                    safe_out.append("--")
                    pending_sep = False
                display_fp = fp[len(prefix) :] if fp.startswith(prefix) else fp
                safe_out.append(f"{display_fp}{sep}{rest}")
            if total_matches == 0:
                return ToolOutcome(content=f"No matches for {pattern!r} in {raw_path}.", ok=True)
            body = scrub_output("\n".join(safe_out))
            if len(body) > MAX_GREP_OUTPUT_CHARS:
                body = body[:MAX_GREP_OUTPUT_CHARS] + "\n... [truncated]"
            if total_matches > max_results:
                body += f"\n... ({total_matches - max_results} more matches; narrow path/glob or raise max_results)"
            return ToolOutcome(content=body, ok=True)
        err = stderr_b.decode("utf-8", errors="replace").strip()
        return ToolOutcome(content=f"Error running grep: {err}", ok=False, reason="grep_error")

    # Pure Python policy-checked walker
    def _py_grep() -> ToolOutcome:
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            rx = re.compile(pattern, flags)
        except re.error as exc:
            return ToolOutcome(content=f"Error: invalid regex {pattern!r}: {exc}", ok=False, reason="invalid_regex")

        output_lines: list[str] = []
        total_found = 0
        shown_matches = 0
        files = [target] if target.is_file() else None

        def _iter_files():
            if files is not None:
                yield from files
                return
            for root, dirs, fnames in os.walk(target):
                dirs[:] = sorted(
                    d
                    for d in dirs
                    if d not in SKIP_WALK_DIRS
                    and not is_protected_secret_path((Path(root) / d).resolve())
                    and (policy is None or policy.allows_read((Path(root) / d).resolve()))
                )
                for fn in sorted(fnames):
                    if glob_filter and not fnmatch.fnmatch(fn, glob_filter):
                        continue
                    yield Path(root) / fn

        for fpath in _iter_files():
            try:
                resolved_f = fpath.resolve()
            except OSError:
                continue
            if is_protected_secret_path(resolved_f):
                continue
            if policy is not None and not policy.allows_read(resolved_f):
                continue
            try:
                if resolved_f.stat().st_size > 4 * 1024 * 1024:
                    continue
                raw = resolved_f.read_bytes()
                if b"\x00" in raw[:4096]:
                    continue
                text = raw.decode("utf-8", errors="replace")
            except OSError:
                continue
            try:
                rel = str(resolved_f.relative_to(base))
            except ValueError:
                rel = str(resolved_f)
            file_lines = text.splitlines()
            hit_indices = [i for i, line in enumerate(file_lines) if rx.search(line)]
            if not hit_indices:
                continue
            total_found += len(hit_indices)
            if shown_matches >= max_results:
                continue
            take_hits = hit_indices[: max_results - shown_matches]
            shown_matches += len(take_hits)
            hit_set = set(take_hits)
            if context_lines == 0:
                for i in take_hits:
                    line = scrub_output(file_lines[i])
                    clipped = line[:MAX_GREP_LINE_CHARS] + ("..." if len(line) > MAX_GREP_LINE_CHARS else "")
                    output_lines.append(f"{rel}:{i + 1}:{clipped}")
            else:
                # Merge overlapping context windows within the file
                ranges: list[tuple[int, int]] = []
                n_lines = len(file_lines)
                for i in take_hits:
                    lo = max(0, i - context_lines)
                    hi = min(n_lines - 1, i + context_lines)
                    if ranges and lo <= ranges[-1][1] + 1:
                        ranges[-1] = (ranges[-1][0], max(ranges[-1][1], hi))
                    else:
                        ranges.append((lo, hi))
                for r_idx, (lo, hi) in enumerate(ranges):
                    if output_lines and (r_idx > 0 or context_lines > 0):
                        output_lines.append("--")
                    for i in range(lo, hi + 1):
                        line = scrub_output(file_lines[i])
                        clipped = line[:MAX_GREP_LINE_CHARS] + ("..." if len(line) > MAX_GREP_LINE_CHARS else "")
                        sep = ":" if i in hit_set else "-"
                        output_lines.append(f"{rel}{sep}{i + 1}{sep}{clipped}")

        if total_found == 0:
            return ToolOutcome(content=f"No matches for {pattern!r} in {raw_path}.", ok=True)
        body = scrub_output("\n".join(output_lines))
        if len(body) > MAX_GREP_OUTPUT_CHARS:
            body = body[:MAX_GREP_OUTPUT_CHARS] + "\n... [truncated]"
        if total_found > max_results:
            body += f"\n... ({total_found - max_results} more matches)"
        return ToolOutcome(content=body, ok=True)

    return await asyncio.to_thread(_py_grep)


async def handle_find_files(session: "ChatSession", tc_input: dict) -> ToolOutcome:
    """List files matching a glob pattern under a directory with CellSandbox & secret protection."""
    pattern = tc_input.get("pattern") or "*"
    raw_path = tc_input.get("path") or "."
    include_hidden = bool(tc_input.get("include_hidden", False))
    try:
        max_results = max(1, min(int(tc_input.get("max_results", DEFAULT_FIND_RESULTS)), 1000))
    except (TypeError, ValueError):
        max_results = DEFAULT_FIND_RESULTS

    base = _workspace_root(session)
    policy = cell_sandbox_of(getattr(session, "_scratchpads", None))
    try:
        target = _resolve_path(session, str(raw_path).strip())
    except (ValueError, OSError) as exc:
        return ToolOutcome(content=f"Error: invalid path {raw_path!r}: {exc}", ok=False, reason="invalid_path")

    if is_protected_secret_path(target):
        return ToolOutcome(content=f"Error: access denied for protected path {raw_path}.", ok=False, reason="access_denied")
    if policy is not None and not policy.allows_read(target):
        return ToolOutcome(content=f"Error: access denied for {raw_path} under sandbox policy.", ok=False, reason="access_denied")
    if not target.exists():
        return ToolOutcome(content=f"Error: path does not exist: {raw_path}", ok=False, reason="path_not_found")

    def _walk() -> ToolOutcome:
        found: list[str] = []
        total = 0
        has_slash = "/" in pattern
        for root, dirs, fnames in os.walk(target):
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in SKIP_WALK_DIRS
                and (include_hidden or not d.startswith("."))
                and not is_protected_secret_path((Path(root) / d).resolve())
                and (policy is None or policy.allows_read((Path(root) / d).resolve()))
            )
            for fn in sorted(fnames):
                if not include_hidden and fn.startswith("."):
                    continue
                try:
                    resolved_f = (Path(root) / fn).resolve()
                except OSError:
                    continue
                if is_protected_secret_path(resolved_f):
                    continue
                if policy is not None and not policy.allows_read(resolved_f):
                    continue
                try:
                    rel = str(resolved_f.relative_to(base))
                except ValueError:
                    rel = str(resolved_f)
                match_str = rel if has_slash else fn
                if fnmatch.fnmatch(match_str, pattern) or fnmatch.fnmatch(rel, pattern):
                    total += 1
                    if len(found) < max_results:
                        found.append(rel)
        if not found:
            return ToolOutcome(content=f"No files matching {pattern!r} under {raw_path}.", ok=True)
        body = "\n".join(found)
        if total > max_results:
            body += f"\n... ({total - max_results} more files; narrow `path` or `pattern`)"
        return ToolOutcome(content=body, ok=True)

    return await asyncio.to_thread(_walk)


BASH_TOOL = ToolDef(
    name="bash",
    description=(
        "Execute a shell command directly in the workspace (or optional `workdir`). "
        "Prefer this for CLI tools, git, compilers, builds, tests, package installation, "
        "system inspection, and service management. Non-interactive (stdin is /dev/null). "
        "Combine related inspection or verification checks into one compound command rather "
        "than separate round-trips, and pass non-interactive flags for installs/builds. "
        "Output uses head+tail truncation up to 16,000 characters so errors at the end are preserved."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "workdir": {
                "type": "string",
                "description": "Optional working directory (relative to workspace or absolute).",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Timeout in seconds (default 120, max 1800).",
            },
        },
        "required": ["command"],
    },
    handler=handle_bash,
)

WRITE_FILE_TOOL = ToolDef(
    name="write_file",
    description=(
        "Create a new UTF-8 text file or overwrite an existing file completely. "
        "Creates parent directories automatically. For targeted edits to existing files, "
        "prefer `edit_file` to save tokens and avoid unintended changes."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to file (relative to workspace or absolute).",
            },
            "content": {
                "type": "string",
                "description": "Full text content to write.",
            },
        },
        "required": ["path", "content"],
    },
    handler=handle_write_file,
)

EDIT_FILE_TOOL = ToolDef(
    name="edit_file",
    description=(
        "Make surgical exact-match replacements in an existing text file. "
        "Pass either (`old_string`, `new_string`, optional `replace_all`) for a single edit, "
        "or `edits` (an array of `{old_string, new_string, replace_all?}` objects) to apply "
        "multiple edits atomically in one call (all succeed or none are written). "
        "Include enough surrounding lines in `old_string` to match uniquely. "
        "Returns a unified diff of the applied changes."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to existing file (relative to workspace or absolute).",
            },
            "old_string": {
                "type": "string",
                "description": "Exact existing text to replace (for a single edit).",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text (for a single edit).",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of requiring a unique match (default false).",
            },
            "edits": {
                "type": "array",
                "description": "List of sequential atomic edits to apply to the file in one call (mutually exclusive with old_string/new_string).",
                "items": {
                    "type": "object",
                    "properties": {
                        "old_string": {"type": "string", "description": "Exact existing text to replace."},
                        "new_string": {"type": "string", "description": "Replacement text."},
                        "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)."},
                    },
                    "required": ["old_string", "new_string"],
                },
            },
        },
        "required": ["path"],
    },
    handler=handle_edit_file,
)

GREP_SEARCH_TOOL = ToolDef(
    name="grep_search",
    description=(
        "Search file contents for a regex pattern across the workspace or a subdirectory/file. "
        "Returns `path:line:text` matches. Safe to call in parallel with other read-only tools."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for.",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search (default '.').",
            },
            "glob": {
                "type": "string",
                "description": "Optional filename glob filter, e.g. '*.py' or '*.ts'.",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Case-insensitive search (default false).",
            },
            "context_lines": {
                "type": "integer",
                "description": "Number of surrounding context lines before and after each match (default 0, max 5).",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum matches to return (default 80, max 500).",
            },
        },
        "required": ["pattern"],
    },
    handler=handle_grep_search,
)

FIND_FILES_TOOL = ToolDef(
    name="find_files",
    description=(
        "Find files matching a glob pattern under a directory (skips .git, node_modules, __pycache__, .venv). "
        "Safe to call in parallel with other read-only tools."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Filename or path glob pattern (default '*').",
            },
            "path": {
                "type": "string",
                "description": "Root directory to search (default '.').",
            },
            "include_hidden": {
                "type": "boolean",
                "description": "Include dotfiles and dot-directories (default false).",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum paths to return (default 200, max 1000).",
            },
        },
        "required": [],
    },
    handler=handle_find_files,
)

CODING_WRITE_TOOLS = (
    BASH_TOOL,
    WRITE_FILE_TOOL,
    EDIT_FILE_TOOL,
)

CODING_READ_TOOLS = (
    GREP_SEARCH_TOOL,
    FIND_FILES_TOOL,
)

CODING_TOOLS = CODING_WRITE_TOOLS + CODING_READ_TOOLS
CODING_TOOL_NAMES = frozenset(t.name for t in CODING_TOOLS)
