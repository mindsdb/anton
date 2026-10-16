"""Host-side writes inside a workspace that refuse a symlink below its root.

Agent code can write anywhere in its project folder, and agent code that runs
unlocked can plant symlinks there that stay on the volume. The host runs
unlocked, so a bookkeeping write that followed such a link would land in
someone else's files. Each helper
opens the trusted ``root`` normally and every component below it with
``O_NOFOLLOW`` relative to its parent's descriptor, so a link at any level
raises ``OSError`` instead of being traversed, and nothing can be swapped
between the check and the write.

Windows has no ``dir_fd`` or ``O_NOFOLLOW``. There the helpers act on the
joined path; only the single local user could have planted a link.
"""

from __future__ import annotations

import errno
import os
import shutil
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_IS_WINDOWS = sys.platform == "win32"
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
# Read once, from the real rmtree: whether it can delete relative to a dir_fd.
_RMTREE_TAKES_DIR_FD = bool(getattr(shutil.rmtree, "avoids_symlink_attacks", False))


def relative_parts(root: Path | str, path: Path | str) -> tuple[str, ...]:
    """The components of ``path`` below ``root``; ValueError when it is not below it."""
    rel = Path(os.path.normpath(path)).relative_to(os.path.normpath(root))
    if not rel.parts or ".." in rel.parts:
        raise ValueError(f"{path} is not inside {root}")
    return rel.parts


def _is_link_at(name: str, dir_fd: int) -> bool:
    try:
        return stat.S_ISLNK(os.lstat(name, dir_fd=dir_fd).st_mode)
    except OSError:
        return False


def _open_at(name: str, flags: int, mode: int, *, dir_fd: int, shown: Path) -> int:
    """``os.open(name, flags, mode, dir_fd=dir_fd)``, naming ``shown`` when a symlink refused it.

    The kernel reports a refused link as ENOTDIR or ELOOP with only the bare
    component name, which tells nobody what to remove.
    """
    try:
        return os.open(name, flags, mode, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in (errno.ENOTDIR, errno.ELOOP) and _is_link_at(name, dir_fd):
            raise OSError(errno.ELOOP, "refused to follow a symlink", str(shown)) from exc
        raise


@contextmanager
def opened_dir_nofollow(
    root: Path | str, parts: tuple[str, ...], *, create: bool = False
) -> Iterator[int | None]:
    """A descriptor for ``root/<parts...>``, each part opened with ``O_NOFOLLOW``.

    With ``create``, each missing part is made first; a link already at that
    name is still refused. Yields None on Windows, where callers use paths.
    """
    if _IS_WINDOWS:
        if create:
            Path(root, *parts).mkdir(parents=True, exist_ok=True)
        yield None
        return
    fd = os.open(root, os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC)
    try:
        for i, name in enumerate(parts):
            if create:
                try:
                    os.mkdir(name, dir_fd=fd)
                except FileExistsError:
                    pass
            child = _open_at(
                name,
                os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
                0o777,
                dir_fd=fd,
                shown=Path(root, *parts[: i + 1]),
            )
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def make_dirs_nofollow(root: Path | str, path: Path | str) -> None:
    """``mkdir -p`` for ``path`` below ``root``, refusing a symlink at any level."""
    with opened_dir_nofollow(root, relative_parts(root, path), create=True):
        pass


def open_nofollow(
    root: Path | str, path: Path | str, flags: int, mode: int = 0o666, *, create_parents: bool = False
) -> int:
    """``os.open(path, flags)`` refusing a symlink at ``path`` or any folder above it below ``root``."""
    parts = relative_parts(root, path)
    if _IS_WINDOWS:
        if create_parents:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        return os.open(path, flags | getattr(os, "O_BINARY", 0), mode)
    with opened_dir_nofollow(root, parts[:-1], create=create_parents) as dir_fd:
        return _open_at(parts[-1], flags | _O_NOFOLLOW | _O_CLOEXEC, mode, dir_fd=dir_fd, shown=Path(path))


def write_text_nofollow(
    root: Path | str, path: Path | str, text: str, *, create_parents: bool = False
) -> None:
    """Create or truncate ``path`` and write ``text`` as UTF-8, refusing symlinks."""
    fd = open_nofollow(root, path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, create_parents=create_parents)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def create_text_nofollow(root: Path | str, path: Path | str, text: str, *, mode: int = 0o666) -> bool:
    """Write ``text`` to ``path`` only when nothing (file, folder or link) holds the name.

    Returns whether it wrote. ``O_EXCL`` never follows a link at the name.
    """
    try:
        fd = open_nofollow(root, path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    return True


def replace_text_nofollow(
    root: Path | str, path: Path | str, text: str, *, create_parents: bool = False
) -> None:
    """Write ``text`` to ``<path>.tmp`` and rename it over ``path`` in the pinned parent."""
    parts = relative_parts(root, path)
    tmp_name = parts[-1] + ".tmp"
    if _IS_WINDOWS:
        target = Path(path)
        if create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(tmp_name)
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(target)
        return
    with opened_dir_nofollow(root, parts[:-1], create=create_parents) as dir_fd:
        fd = _open_at(
            tmp_name,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _O_NOFOLLOW | _O_CLOEXEC,
            0o666,
            dir_fd=dir_fd,
            shown=Path(path).with_name(tmp_name),
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, parts[-1], src_dir_fd=dir_fd, dst_dir_fd=dir_fd)


def rmtree_nofollow(root: Path | str, path: Path | str) -> None:
    """Delete the folder ``path``, refusing a symlink at it or at any folder above it below ``root``."""
    parts = relative_parts(root, path)
    if _IS_WINDOWS or not _RMTREE_TAKES_DIR_FD:
        with opened_dir_nofollow(root, parts):
            pass
        shutil.rmtree(path)
        return
    with opened_dir_nofollow(root, parts[:-1]) as dir_fd:
        shutil.rmtree(parts[-1], dir_fd=dir_fd)
