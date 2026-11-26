"""The text `list_artifacts` returns.

Plain text rather than JSON: only the model reads it, and repeated keys,
quotes and escapes cost tokens without telling it anything.
"""

from __future__ import annotations

import json
import stat
import unicodedata
from collections.abc import Sequence
from pathlib import Path

from anton.core.artifacts.internal_files import (
    INLINE_SNAPSHOT_FILENAME,
    INLINE_STATE_FILENAME,
    NON_CONTENT_NAMES,
)
from anton.core.artifacts.models import Artifact

FIELDS = ("id", "name", "type", "updatedAt", "description", "primary", "display", "state", "controls", "files", "service_files")
SUMMARY_FIELDS = ("name", "type", "updatedAt", "description", "primary", "display", "state", "controls")

LEGEND = "Artifact folder = <root>/<slug>; primary and file paths are relative to it."
SIZE_LEGEND = "The number before a file path is its size in bytes."
NO_ARTIFACTS = "No artifacts yet."
NO_MATCHES = "No artifacts matched."

_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _escape(value: str) -> str:
    """One line whatever the value holds, so no value can fake a `## <slug>` heading."""
    out = []
    for ch in value:
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif unicodedata.category(ch) in ("Cc", "Zl", "Zp"):
            code = ord(ch)
            out.append(f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}")
        else:
            out.append(ch)
    return "".join(out)


def _one_line(value: str) -> str:
    return _escape(" ".join(value.split()))


def _hidden(rel_path: str) -> bool:
    return any(part.startswith(".") for part in rel_path.split("/"))


def service_files(folder: Path) -> list[tuple[str, int]]:
    """Housekeeping and generation files present in `folder`, dot-files excluded."""
    found: list[tuple[str, int]] = []
    for name in sorted(NON_CONTENT_NAMES):
        if name.startswith("."):
            continue
        try:
            info = (folder / name).lstat()
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode):
            found.append((name, info.st_size))
    return found


# A saved state up to this size is shown in the listing itself, as compact
# JSON; a larger one is only flagged, and the agent reads the file.
STATE_INLINE_MAX_BYTES = 400


def state_line(artifact: Artifact, folder: Path) -> str | None:
    """The `state:` line for an inline artifact or widget whose page saved a state.

    None for every other artifact, so they cost no tokens. A small state is
    inlined (it is the user's own edits, shown as data); a larger one is
    flagged with its size so the agent reads `state.json` when it needs it.
    """
    if not artifact.display:  # every display is shown in the reply with a page state
        return None
    path = folder / INLINE_STATE_FILENAME
    try:
        info = path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    flag = f"{INLINE_STATE_FILENAME}, {info.st_size} bytes: the user's latest edits; read it for the current selections"
    if info.st_size > STATE_INLINE_MAX_BYTES:
        return f"state: {flag}"
    try:
        compact = json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, separators=(",", ":"))
    except (OSError, UnicodeDecodeError, ValueError):
        return f"state: {flag}"
    return f"state: {_escape(compact)} ({INLINE_STATE_FILENAME}, the user's latest edits)"


def controls_line(artifact: Artifact, folder: Path) -> str | None:
    """`controls: N` when the shim captured an inline page's form controls.

    Only a count: the values are what `inspect_artifact` is for.
    """
    if not artifact.display:
        return None
    from anton.core.artifacts.page_state import read_page_file, snapshot_controls

    snapshot = read_page_file(folder, INLINE_SNAPSHOT_FILENAME)
    if snapshot is None:
        return None
    if snapshot.problem:
        return f"controls: {INLINE_SNAPSHOT_FILENAME} {snapshot.problem}"
    count = len(snapshot_controls(snapshot))
    return f"controls: {count} ({INLINE_SNAPSHOT_FILENAME}; inspect_artifact shows their values)"


def _file_lines(label: str, entries: Sequence[tuple[str, int]]) -> list[str]:
    if not entries:
        return [f"{label}: none"]
    return [f"{label}:"] + [f"  {size}  {_escape(path)}" for path, size in entries]


def _entry(artifact: Artifact, folder: Path, fields: Sequence[str]) -> list[str]:
    lines = [f"## {_escape(artifact.slug)}"]
    for field in FIELDS:
        if field not in fields:
            continue
        if field in ("name", "description"):
            lines.append(f"{field}: {_one_line(getattr(artifact, field))}")
        elif field == "primary":
            lines.append(f"primary: {_escape(artifact.primary) if artifact.primary else '-'}")
        elif field == "display":
            # Only an artifact shown inline or as a widget says so: the usual card is the
            # default, and a `display: -` line on every entry costs tokens.
            if artifact.display:
                lines.append(f"display: {artifact.display}")
        elif field == "state":
            line = state_line(artifact, folder)
            if line:
                lines.append(line)
        elif field == "controls":
            line = controls_line(artifact, folder)
            if line:
                lines.append(line)
        elif field == "files":
            entries = [(f.path, f.bytes) for f in artifact.files if not _hidden(f.path)]
            lines += _file_lines("files", entries)
        elif field == "service_files":
            lines += _file_lines("service_files", service_files(folder))
        else:
            lines.append(f"{field}: {_escape(getattr(artifact, field))}")
    return lines


def render_listing(
    groups: Sequence[tuple[Path, Sequence[Artifact]]],
    fields: Sequence[str],
    unmatched: Sequence[str] = (),
) -> str:
    """`groups` holds (artifacts root, its artifacts newest first) per root."""
    unmatched_line = f"unmatched: {', '.join(_escape(key) for key in unmatched)}"
    if not any(artifacts for _, artifacts in groups):
        return f"{NO_MATCHES}\n{unmatched_line}" if unmatched else NO_ARTIFACTS
    lines = [LEGEND]
    if "files" in fields or "service_files" in fields:
        lines.append(SIZE_LEGEND)
    for root, artifacts in groups:
        if not artifacts:
            continue
        lines += ["", f"Artifacts root: {_escape(str(root))}"]
        for artifact in artifacts:
            lines.append("")
            lines += _entry(artifact, root / artifact.slug, fields)
    if unmatched:
        lines += ["", unmatched_line]
    return "\n".join(lines)
