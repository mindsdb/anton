"""Folders and files that only the current user can open.

anton keeps keys, memory and conversation records in ``~/.anton`` and in each
project's ``.anton``. On macOS and Linux the process umask leaves a new folder
at 0755, so another local account can list it and read every file in it.
`make_private_dir` makes such a folder 0700, and the files inside are then
private because their folder is. It changes a folder only when the current
user owns it, and it never follows a symbolic link at the folder's name: the
link's target belongs to whoever made the link.

On Windows a new folder takes its parent's access list. Inside the user's
profile that list already shuts out other standard accounts, so a folder there,
such as ``~/.anton``, is only created. A repository outside the profile, such as
``C:\\src\\app``, lets every local account read it, so for a folder that can
sit anywhere, such as a project's ``.anton``, `make_private_dir` can give the
folder an access list that names only SYSTEM, Administrators and the folder's
owner, with inheritance removed. That runs once per folder: a marker file inside
records which folder got the list, so a copy or a clone that carries the marker
gets the list again.
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

from anton.utils.gitignore import ensure_gitignored
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
#: the list is applied once per folder rather than at every call. It holds the
#: folder's volume serial number and file index, and the folder's .gitignore
#: lists it, so it is never committed.
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
    """``~/.anton``, created or made private to the current user.

    On Windows it is only created: it sits in the user's profile, and an
    owner-only list there would drop any grant an organization added to the
    profile.
    """
    home = Path.home() / ".anton"
    make_private_dir(path=home)
    return home


def make_private_dir(*, path: Path, windows_acl: bool = False) -> None:
    """Create ``path`` so that only the current user can open it, or make an existing folder so.

    Missing parents are created with the default mode, because they belong to
    the user's home or project, not to anton. On macOS and Linux the folder
    ends 0700. A symbolic link at the name and a folder another account owns
    are left as they are. So is a folder whose mode can't change, such as one
    on a read-only mount or a FAT drive: that logs a warning and returns,
    because the folder still works. Raises FileExistsError when something
    other than a folder holds the name, as ``Path.mkdir(exist_ok=True)`` does.

    On Windows the folder is only created, with its parent's access list,
    unless ``windows_acl`` is set. With it, the folder gets the owner-only
    access list once, and a link or junction at the name is left as it is.
    Set it for a folder that can sit outside the user's profile.
    """
    if _IS_WINDOWS and not windows_acl:
        path.mkdir(parents=True, exist_ok=True)
        return
    try:
        _mkdir_unless_taken(path=path)
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        _mkdir_unless_taken(path=path)

    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or _is_junction(path=path):
        _log.info(
            "Left a folder's permissions as they are",
            extra={"event": "private_folder_left", "reason": "symlink"},
        )
        return
    if not stat.S_ISDIR(st.st_mode):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(path))
    if _IS_WINDOWS:
        _apply_owner_only_acl(path=path, st=st)
        return
    if st.st_uid != os.getuid():
        _log.info(
            "Left a folder's permissions as they are",
            extra={"event": "private_folder_left", "reason": "other_owner"},
        )
        return
    if stat.S_IMODE(st.st_mode) == PRIVATE_DIR_MODE:
        return
    try:
        # Opened without following a link, so a link swapped in after the
        # lstat raises instead of passing the chmod to its target.
        with opened_dir_nofollow(path.parent, (path.name,)) as fd:
            if os.fstat(fd).st_uid == os.getuid():
                os.fchmod(fd, PRIVATE_DIR_MODE)
    except OSError as exc:
        _log.warning(
            "Could not make a folder private",
            extra={
                "event": "private_folder_chmod_failed",
                "reason": errno.errorcode.get(exc.errno, "unknown"),
                "error_type": type(exc).__name__,
            },
        )


def _mkdir_unless_taken(*, path: Path) -> None:
    """``os.mkdir(path, 0o700)``, passing over a name something already holds.

    Some systems report EACCES or EROFS rather than EEXIST for a name that
    exists, as ``Path.mkdir`` notes, so any error counts as taken when
    something is at the name. FileNotFoundError for a missing parent raises.
    """
    try:
        os.mkdir(path, PRIVATE_DIR_MODE)
    except OSError:
        if not os.path.lexists(path):
            raise


def tighten_private_file(*, path: Path) -> None:
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


def icacls_command(*, path: Path) -> list[str]:
    """The icacls call that gives ``path`` the owner-only access list.

    Named by its full path in System32: a bare ``icacls`` would let Windows
    search the current folder first, which can be a repository someone else
    filled.
    """
    system_root = os.environ.get("SystemRoot") or r"C:\Windows"
    icacls = ntpath.join(system_root, "System32", "icacls.exe")
    return [icacls, str(path), "/inheritance:r", "/grant:r", *OWNER_ONLY_GRANTS]


def _apply_owner_only_acl(*, path: Path, st: os.stat_result) -> None:
    """Give ``path`` the owner-only access list, unless its marker says this folder has it.

    ``st`` is the folder's own lstat. The marker names the folder by its volume
    serial number and file index, which a copy or a clone does not share, so a
    marker that came along with the folder's contents gets the list applied
    again. A marker that can't be read counts as missing, because applying the
    list again is safe. A failure logs a warning and leaves the marker as it
    was, so the next call tries again.
    """
    marker = path / WINDOWS_ACL_MARKER
    identity = f"{st.st_dev}:{st.st_ino}"
    try:
        if marker.read_text(encoding="utf-8").strip() == identity:
            return
    except (OSError, UnicodeDecodeError):
        pass
    try:
        result = subprocess.run(
            icacls_command(path=path),
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
        # Listed first, so the marker never shows up as a file to commit.
        ensure_gitignored(folder=path, pattern=WINDOWS_ACL_MARKER)
        marker.write_text(identity + "\n", encoding="utf-8")
    except OSError as exc:
        _log.warning(
            "Could not record a folder's owner-only access list",
            extra={"event": "private_folder_acl_failed", "reason": "marker_not_written", "error_type": type(exc).__name__},
        )


def _is_junction(*, path: Path) -> bool:
    """Whether ``path`` is a Windows junction; always False elsewhere and before Python 3.12."""
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))
