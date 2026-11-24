"""Keep a file anton writes inside ``.anton`` out of git."""

from __future__ import annotations

from pathlib import Path


def ensure_gitignored(*, folder: Path, pattern: str) -> None:
    """List ``pattern`` in ``folder/.gitignore``, adding the file when it is missing.

    Appends when a ``.gitignore`` already exists rather than replacing it, so
    the lines a project keeps there survive. Raises OSError when the file
    can't be read or written.
    """
    gitignore = folder / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.is_file() else ""
    if pattern in existing.split():
        return
    prefix = existing if not existing or existing.endswith("\n") else existing + "\n"
    gitignore.write_text(f"{prefix}{pattern}\n", encoding="utf-8")
