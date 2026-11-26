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
owner, with inherited access removed. Project callers reapply the list before
writing so a copied or restored directory cannot retain broad inherited access.
"""

from __future__ import annotations

import errno
import logging
import ntpath
import os
import stat
import subprocess
import sys
import tempfile
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
    require_private_dir(path=home, follow_link=True)
    return home


def require_private_dir(*, path: Path, windows_acl: bool = False, follow_link: bool = False) -> None:
    """Refuse private persistence when the folder cannot be protected."""
    if not make_private_dir(path=path, windows_acl=windows_acl, follow_link=follow_link):
        raise PermissionError(errno.EACCES, "Could not establish a private directory", str(path))


def make_private_dir(*, path: Path, windows_acl: bool = False, follow_link: bool = False) -> bool:
    """Create or tighten a directory and report whether it is private.

    A foreign-owned directory is never changed, and neither is a link unless
    ``follow_link`` is set. The user's own ``~/.anton`` sets it, since a
    dotfiles setup or a move to another disk links it; the folder the link
    resolves to is then tightened when this user owns it. A project's
    ``.anton`` never sets it: a repository can commit a link that points
    outside itself. Missing ancestors belong to the project and retain their
    normal creation modes. Windows profile directories keep their profile ACL.
    Elsewhere pass windows_acl=True to remove inherited grants before writing.

    A filesystem that keeps no modes, such as WSL's ``/mnt/c``, exFAT or an
    SMB share, leaves the folder as it is, with a warning: no mode can protect
    it, and refusing would stop the user without stopping any reader.
    """
    if follow_link and os.path.islink(path):
        target = Path(os.path.realpath(path))
        if not target.is_dir():
            return False
        return make_private_dir(path=target, windows_acl=windows_acl)
    if _IS_WINDOWS and not windows_acl:
        path.mkdir(parents=True, exist_ok=True)
    else:
        try:
            _mkdir_unless_taken(path=path)
        except FileNotFoundError:
            path.parent.mkdir(parents=True, exist_ok=True)
            _mkdir_unless_taken(path=path)

    st = os.lstat(path)
    # Before Python 3.12 os.path.isjunction is absent, and lstat represents
    # a directory junction as a directory with a mount-point reparse tag.
    if stat.S_ISLNK(st.st_mode) or getattr(st, "st_reparse_tag", 0):
        return False
    if not stat.S_ISDIR(st.st_mode):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(path))
    if _IS_WINDOWS:
        if windows_acl:
            _apply_owner_only_acl(path=path)
        return True
    if st.st_uid != os.getuid():
        return False
    if stat.S_IMODE(st.st_mode) == PRIVATE_DIR_MODE:
        return True
    try:
        with opened_dir_nofollow(path.parent, (path.name,)) as fd:
            if not os.path.samestat(os.fstat(fd), st):
                return False
            try:
                os.fchmod(fd, PRIVATE_DIR_MODE)
            except OSError as exc:
                _warn_modes_not_kept(path=path, cause=errno.errorcode.get(exc.errno, "unknown"))
                return True
            if stat.S_IMODE(os.fstat(fd).st_mode) != PRIVATE_DIR_MODE:
                _warn_modes_not_kept(path=path, cause="mode_not_kept")
            return True
    except OSError as exc:
        _log.warning(
            "Could not make a folder private",
            extra={"event": "private_folder_chmod_failed", "reason": errno.errorcode.get(exc.errno, "unknown")},
        )
        return False


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


_WARNED_MODES_NOT_KEPT: set[str] = set()


def _warn_modes_not_kept(*, path: Path, cause: str) -> None:
    """Warn once per folder that its filesystem keeps no private modes."""
    key = os.path.normcase(os.path.abspath(path))
    if key in _WARNED_MODES_NOT_KEPT:
        return
    _WARNED_MODES_NOT_KEPT.add(key)
    _log.warning(
        "The filesystem keeps no private permissions for a folder; using it as it is",
        extra={"event": "private_folder_modes_not_kept", "reason": cause},
    )


def write_private_text(path: Path, text: str) -> None:
    """Publish a private UTF-8 file, preserving the target of any previous link."""
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


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
        if os.path.samestat(os.fstat(fd), st):
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
    return [icacls, str(path), "/L", "/inheritance:r", "/grant:r", *OWNER_ONLY_GRANTS]


#: GetVolumeInformationW's flag for a volume that stores access lists. NTFS
#: and ReFS set it; FAT, FAT32 and exFAT don't.
_FILE_PERSISTENT_ACLS = 0x00000008


def _volume_keeps_acls(*, path: Path) -> bool:
    """Whether the volume holding ``path`` stores access lists.

    True when Windows can't say, so an unknown volume still gets the list and
    still refuses when icacls fails.
    """
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        root = ctypes.create_unicode_buffer(32768)
        if not kernel32.GetVolumePathNameW(str(path), root, len(root)):
            return True
        flags = wintypes.DWORD()
        if not kernel32.GetVolumeInformationW(root, None, 0, None, None, ctypes.byref(flags), None, 0):
            return True
    except (AttributeError, OSError):
        return True
    return bool(flags.value & _FILE_PERSISTENT_ACLS)


def _apply_owner_only_acl(*, path: Path) -> None:
    """Remove inherited access on each use, preserving explicit administrator grants.

    A volume with no access lists, such as a FAT or exFAT drive, is used as
    it is, with a warning, for the reason `make_private_dir` gives for modes.
    """
    if not _volume_keeps_acls(path=path):
        _warn_modes_not_kept(path=path, cause="volume_keeps_no_acls")
        return
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
        raise PermissionError(errno.EACCES, "Could not run icacls", str(path)) from exc
    if result.returncode:
        raise PermissionError(errno.EACCES, f"icacls exited with status {result.returncode}", str(path))
