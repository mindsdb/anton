"""The text `list_artifacts` returns.

Plain text rather than JSON: only the model reads it, and repeated keys,
quotes and escapes cost tokens without telling it anything.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from pathlib import Path

from anton.core.artifacts.internal_files import NON_CONTENT_NAMES
from anton.core.artifacts.models import Artifact

FIELDS = ("id", "name", "type", "updatedAt", "description", "primary", "files", "service_files")
SUMMARY_FIELDS = ("name", "type", "updatedAt", "description", "primary")

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
        path = folder / name
        try:
            if path.is_file() and not path.is_symlink():
                found.append((name, path.stat().st_size))
        except OSError:
            continue
    return found


def _file_lines(label: str, entries: Sequence[tuple[str, int]]) -> list[str]:
    if not entries:
        return [f"{label}: none"]
    return [f"{label}:"] + [f"  {size}  {_escape(path)}" for path, size in entries]


def _entry(artifact: Artifact, folder: Path, fields: Sequence[str]) -> list[str]:
    lines = [f"## {_escape(artifact.slug)}"]
    for field in FIELDS:
        if field not in fields:
            continue
        if field == "id":
            lines.append(f"id: {artifact.id}")
        elif field in ("name", "description"):
            lines.append(f"{field}: {_one_line(getattr(artifact, field))}")
        elif field == "type":
            lines.append(f"type: {artifact.type}")
        elif field == "updatedAt":
            lines.append(f"updatedAt: {_escape(artifact.updatedAt)}")
        elif field == "primary":
            lines.append(f"primary: {_escape(artifact.primary) if artifact.primary else '-'}")
        elif field == "files":
            entries = [(f.path, f.bytes) for f in artifact.files if not _hidden(f.path)]
            lines += _file_lines("files", entries)
        elif field == "service_files":
            lines += _file_lines("service_files", service_files(folder))
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
