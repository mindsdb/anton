"""What an inline page reported about itself, as the text `inspect_artifact` returns.

Two files in the artifact folder, both written by the Cowork preview shim:
`state.json` (what the page saved through `anton.setState`) and
`snapshot.json` (its form controls and visible text, captured even when the
page never calls setState). Either may be missing. Both are user content read
as data: a link is never followed, a file over the size cap is not read, and
bad JSON is reported rather than raised.
"""

from __future__ import annotations

import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anton.core.artifacts.internal_files import INLINE_SNAPSHOT_FILENAME, INLINE_STATE_FILENAME
from anton.core.artifacts.listing import _escape, _one_line
from anton.core.artifacts.models import Artifact

# Larger than anything the shim writes (setState is capped at 32 KB); a file
# past it is not ours to load whole.
PAGE_FILE_MAX_BYTES = 256 * 1024
STATE_SHOWN_MAX_CHARS = 4000
TEXT_SHOWN_MAX_CHARS = 1500
CONTROLS_SHOWN_MAX = 50
CONTROL_VALUE_MAX_CHARS = 200

NOTHING_REPORTED = (
    "The page hasn't reported anything yet. It reports its form controls once "
    "the user changes one, and custom page state only if the page calls "
    "anton.setState."
)


@dataclass
class PageFile:
    """One page file: `data` when it parsed, else `problem` says why not."""

    mtime: float
    size: int
    data: Any = None
    problem: str | None = None


def read_page_file(folder: Path, name: str, max_bytes: int = PAGE_FILE_MAX_BYTES) -> PageFile | None:
    """`folder/name` parsed as JSON, or None when it is absent, a link or not a regular file.

    Opened with O_NOFOLLOW and checked on the open descriptor, so a link
    swapped in between a check and the read is still refused.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(folder / name, flags)
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return None
        page = PageFile(mtime=info.st_mtime, size=info.st_size)
        if info.st_size > max_bytes:
            page.problem = f"{info.st_size} bytes, over the {max_bytes // 1024} KB cap; not read"
            return page
        with os.fdopen(fd, "rb", closefd=False) as fh:
            raw = fh.read(max_bytes + 1)
    except OSError:
        return None
    finally:
        os.close(fd)
    if len(raw) > max_bytes:
        page.problem = f"over the {max_bytes // 1024} KB cap; not read"
        return page
    try:
        page.data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        page.problem = f"{page.size} bytes, not valid JSON"
    return page


def snapshot_controls(snapshot: PageFile | None) -> list[dict]:
    """The well-formed control entries of a parsed snapshot."""
    if snapshot is None or not isinstance(snapshot.data, dict):
        return []
    controls = snapshot.data.get("controls")
    if not isinstance(controls, list):
        return []
    return [c for c in controls if isinstance(c, dict)]


def _cut(text: str, limit: int) -> tuple[str, bool]:
    return (text, False) if len(text) <= limit else (text[:limit], True)


def _value_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _control_line(control: dict) -> str:
    label = control.get("label") or control.get("name") or "(unlabelled)"
    kind = control.get("type")
    value, cut = _cut(_one_line(_value_text(control.get("value", ""))), CONTROL_VALUE_MAX_CHARS)
    head = f"{_one_line(str(label))} ({_one_line(str(kind))})" if kind else _one_line(str(label))
    return f"  {head}: {value}{' …' if cut else ''}"


def relative_time(mtime: float, now: float | None = None) -> str:
    seconds = max(0, int((time.time() if now is None else now) - mtime))
    if seconds < 45:
        return "just now"
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if seconds >= size or unit == "minute":
            count = max(1, round(seconds / size))
            return f"{count} {unit}{'' if count == 1 else 's'} ago"
    return "just now"  # unreachable


def render_inspection(artifact: Artifact, folder: Path, now: float | None = None) -> str:
    """The compact text `inspect_artifact` returns for an html-app."""
    lines = [
        f"## {_escape(artifact.slug)}",
        f"title: {_one_line(artifact.name)}",
        f"display: {artifact.display or '- (a full page, not shown in the chat)'}",
    ]
    state = read_page_file(folder, INLINE_STATE_FILENAME)
    snapshot = read_page_file(folder, INLINE_SNAPSHOT_FILENAME)
    if state is None and snapshot is None:
        lines.append(NOTHING_REPORTED)
        return "\n".join(lines)

    lines.append(f"last saved: {relative_time(max(f.mtime for f in (state, snapshot) if f), now)}")

    if state is None:
        lines.append("page state: none (the page has not called anton.setState)")
    elif state.problem:
        lines.append(f"page state: {INLINE_STATE_FILENAME} {state.problem}")
    else:
        compact = json.dumps(state.data, ensure_ascii=False, separators=(",", ":"))
        shown, cut = _cut(compact, STATE_SHOWN_MAX_CHARS)
        note = f" … (cut: {len(compact)} characters in all)" if cut else ""
        lines.append(f"page state ({INLINE_STATE_FILENAME}, from anton.setState): {_escape(shown)}{note}")

    if snapshot is None:
        lines.append("controls: none reported yet (the user has not changed a form control)")
        return "\n".join(lines)
    if snapshot.problem:
        lines.append(f"controls: {INLINE_SNAPSHOT_FILENAME} {snapshot.problem}")
        return "\n".join(lines)
    controls = snapshot_controls(snapshot)
    if controls:
        lines.append(f"controls ({len(controls)}):")
        lines += [_control_line(c) for c in controls[:CONTROLS_SHOWN_MAX]]
        if len(controls) > CONTROLS_SHOWN_MAX:
            lines.append(f"  … {len(controls) - CONTROLS_SHOWN_MAX} more")
    else:
        lines.append("controls: none")
    text = snapshot.data.get("text") if isinstance(snapshot.data, dict) else None
    if isinstance(text, str) and text.strip():
        shown, cut = _cut(_one_line(text), TEXT_SHOWN_MAX_CHARS)
        lines.append(f"visible text: {shown}{' … (cut)' if cut else ''}")
    return "\n".join(lines)
