"""Folders and files that only the current user can open.

anton keeps keys, memory and conversation records in ``~/.anton`` and in each
project's ``.anton``. On macOS and Linux the process umask leaves a new folder
at 0755, so another local account can list it and read every file in it.
`make_private_dir` makes such a folder 0700, and the files inside are then
private because their folder is. It changes a folder only when the current
user owns it, and it never follows a symbolic link at the folder's name: the
link's target belongs to whoever made the link.

On Windows a new folder takes its parent's access list, and a repository
outside the profile, such as ``C:\\src\\app``, lets every local account read
it. `make_private_dir` gives the folder an access list that names only SYSTEM,
Administrators and the folder's owner, with inheritance removed. That runs
once per folder: a marker file inside records that the list is in place.
"""

from __future__ import annotations

import errno
import logging
import ntpath
import os
import stat
import subprocess
import sys
from pathlib import Path

from anton.utils.nofollow import opened_dir_nofollow

_log = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
# Opening a FIFO planted at the name returns at once instead of waiting.
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)

PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600

#: Written inside a folder once its owner-only access list is in place, so
#: the list is applied once per folder rather than at every call.
WINDOWS_ACL_MARKER = ".owner-only-acl"

#: SYSTEM, Administrators and OWNER RIGHTS, each with full control that every
#: file and folder below inherits. CPython's os.mkdir(mode=0o700) applies the
#: same list on Windows since 3.12.4; icacls also covers older Pythons and
#: folders that already exist. cowork gives ``<repo>/.claude-mindshub`` the
#: same list, and tests/test_workspace_private.py pins it.
OWNER_ONLY_GRANTS = (
    "*S-1-5-18:(OI)(CI)F",
    "*S-1-5-32-544:(OI)(CI)F",
    "*S-1-3-4:(OI)(CI)F",
)


def anton_home() -> Path:
    """``~/.anton``, created or made private to the current user."""
    home = Path.home() / ".anton"
    make_private_dir(home)
    return home


def make_private_dir(path: Path) -> None:
    """Create ``path`` so that only the current user can open it, or make an existing folder so.

    Missing parents are created with the default mode, because they belong to
    the user's home or project, not to anton. On macOS and Linux the folder
    ends 0700. On Windows it gets the owner-only access list once. A symbolic
    link (or a Windows junction) at the name, and on macOS and Linux a folder
    another account owns, are left as they are. Raises FileExistsError when
    something other than a folder holds the name, as
    ``Path.mkdir(exist_ok=True)`` does.
    """
    try:
        os.mkdir(path, PRIVATE_DIR_MODE)
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(path, PRIVATE_DIR_MODE)
        except FileExistsError:
            pass
    except FileExistsError:
        pass

    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or _is_junction(path):
        _log.info(
            "Left a folder's permissions as they are",
            extra={"event": "private_folder_left", "reason": "symlink"},
        )
        return
    if not stat.S_ISDIR(st.st_mode):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(path))
    if _IS_WINDOWS:
        _apply_owner_only_acl(path)
        return
    if st.st_uid != os.getuid():
        _log.info(
            "Left a folder's permissions as they are",
            extra={"event": "private_folder_left", "reason": "other_owner"},
        )
        return
    if stat.S_IMODE(st.st_mode) == PRIVATE_DIR_MODE:
        return
    # Opened without following a link, so a link swapped in after the lstat
    # raises instead of passing the chmod to its target.
    with opened_dir_nofollow(path.parent, (path.name,)) as fd:
        if os.fstat(fd).st_uid == os.getuid():
            os.fchmod(fd, PRIVATE_DIR_MODE)


def tighten_private_file(path: Path) -> None:
    """Make an existing file at ``path`` readable and writable only by its owner (0600).

    Does nothing when nothing holds the name, and nothing on Windows, where
    the folder's access list decides who can read the file. A symbolic link,
    anything but a regular file, and a file another account owns are left as
    they are.
    """
    if _IS_WINDOWS:
        return
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(st.st_mode):
        _log.info(
            "Left a file's permissions as they are",
            extra={"event": "private_file_left", "reason": "not_a_regular_file"},
        )
        return
    if st.st_uid != os.getuid():
        _log.info(
            "Left a file's permissions as they are",
            extra={"event": "private_file_left", "reason": "other_owner"},
        )
        return
    if stat.S_IMODE(st.st_mode) == PRIVATE_FILE_MODE:
        return
    fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC)
    try:
        # The same file the lstat saw, not one swapped in after it.
        if os.fstat(fd).st_ino == st.st_ino:
            os.fchmod(fd, PRIVATE_FILE_MODE)
    finally:
        os.close(fd)


def icacls_command(path: Path) -> list[str]:
    """The icacls call that gives ``path`` the owner-only access list.

    Named by its full path in System32: a bare ``icacls`` would let Windows
    search the current folder first, which can be a repository someone else
    filled.
    """
    system_root = os.environ.get("SystemRoot") or r"C:\Windows"
    icacls = ntpath.join(system_root, "System32", "icacls.exe")
    return [icacls, str(path), "/inheritance:r", "/grant:r", *OWNER_ONLY_GRANTS]


def _apply_owner_only_acl(path: Path) -> None:
    """Give ``path`` the owner-only access list, unless its marker says it has it.

    A failure logs a warning and leaves no marker, so the next call tries again.
    """
    marker = path / WINDOWS_ACL_MARKER
    if marker.exists():
        return
    try:
        result = subprocess.run(
            icacls_command(path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log.warning(
            "Could not make a folder's access list owner-only",
            extra={"event": "private_folder_acl_failed", "reason": "icacls_not_run", "error_type": type(exc).__name__},
        )
        return
    if result.returncode != 0:
        _log.warning(
            "Could not make a folder's access list owner-only",
            extra={"event": "private_folder_acl_failed", "reason": "icacls_refused"},
        )
        return
    try:
        marker.write_bytes(b"")
    except OSError as exc:
        _log.warning(
            "Could not record a folder's owner-only access list",
            extra={"event": "private_folder_acl_failed", "reason": "marker_not_written", "error_type": type(exc).__name__},
        )


def _is_junction(path: Path) -> bool:
    """Whether ``path`` is a Windows junction; always False elsewhere and before Python 3.12."""
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))
