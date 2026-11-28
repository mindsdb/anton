"""Keep a file anton writes inside ``.anton`` out of git."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path


def ensure_gitignored(*, folder: Path, pattern: str) -> None:
    """Append a literal pattern to a regular, single-link .gitignore without recoding it."""
    path = folder / ".gitignore"
    encoded = pattern.encode("utf-8")
    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags | getattr(os, "O_BINARY", 0), 0o666)
    with os.fdopen(fd, "r+b") as handle:
        opened = os.fstat(fd)
        entry = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or getattr(entry, "st_reparse_tag", 0)
            or not os.path.samestat(opened, entry)
        ):
            raise OSError(errno.EINVAL, "Cannot update a linked or non-regular .gitignore", str(path))
        existing = handle.read()
        if b"\x00" in existing:
            raise OSError(errno.EINVAL, "Cannot append a Git pattern to a binary .gitignore", str(path))
        if encoded in existing.splitlines():
            return
        prefix = b"\n" if existing and not existing.endswith(b"\n") else b""
        handle.write(prefix + encoded + b"\n")
