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
import functools
import logging
import ntpath
import os
import re
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
#: same list, and tests/test_workspace_private.py pins it. The current user's
#: SID is added beside these when Windows names it (`icacls_command`).
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
    if follow_link and _is_link_or_junction(path=path):
        target = Path(os.path.realpath(path))
        if not target.is_dir():
            return False
        # The profile's list protects a home inside the profile. A link to
        # another disk lands where that list doesn't reach, so the folder
        # there gets the owner-only list, as POSIX tightens it to 0700.
        outside_profile = _IS_WINDOWS and not _is_within(path=target, root=Path.home())
        return make_private_dir(path=target, windows_acl=windows_acl or outside_profile)
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


#: The reparse tag of a Windows directory junction, as `mklink /J` makes.
_IO_REPARSE_TAG_MOUNT_POINT = 0xA0000003


def _is_link_or_junction(*, path: Path) -> bool:
    """Whether ``path`` is a symbolic link or, on Windows, a directory junction.

    A folder moved to another disk is usually linked back with a junction, and
    os.path.islink is False for one. Before Python 3.12 os.path.isjunction is
    absent, so the mount-point reparse tag from lstat stands in for it.
    """
    if os.path.islink(path):
        return True
    isjunction = getattr(os.path, "isjunction", None)
    if isjunction is not None:
        return bool(isjunction(path))
    try:
        return getattr(os.lstat(path), "st_reparse_tag", 0) == _IO_REPARSE_TAG_MOUNT_POINT
    except OSError:
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


def _is_within(*, path: Path, root: Path) -> bool:
    """Whether ``path`` sits in ``root`` once both are resolved, ignoring case on Windows."""
    resolved = os.path.normcase(os.path.realpath(path))
    base = os.path.normcase(os.path.realpath(root))
    try:
        return os.path.commonpath([resolved, base]) == base
    except ValueError:
        # Different drives.
        return False


def _system_tool(name: str) -> str:
    """A System32 tool by its full path: a bare name would let Windows search
    the current folder first, which can be a repository someone else filled."""
    system_root = os.environ.get("SystemRoot") or r"C:\Windows"
    return ntpath.join(system_root, "System32", name)


def _run_system_tool(argv: list[str]) -> subprocess.CompletedProcess:
    """Run a System32 tool with no window and no shell, capturing its output."""
    return subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


@functools.lru_cache(maxsize=1)
def _current_user_sid() -> str | None:
    """The SID of the account this process runs as, or None when Windows can't say.

    The owner-only list names it beside OWNER RIGHTS. A folder made from an
    elevated terminal is owned by Administrators, so OWNER RIGHTS alone would
    shut the same user out of it from a normal terminal.
    """
    try:
        result = _run_system_tool([_system_tool("whoami.exe"), "/user", "/fo", "csv", "/nh"])
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r'"(S-1-[0-9-]+)"', result.stdout.decode("utf-8", "replace")) if result.returncode == 0 else None
    return match.group(1) if match else None


def icacls_command(*, path: Path, user_sid: str | None = None) -> list[str]:
    """The icacls call that gives ``path`` the owner-only access list, naming
    ``user_sid`` too when it is known."""
    grants = [*OWNER_ONLY_GRANTS, *([f"*{user_sid}:(OI)(CI)F"] if user_sid else [])]
    return [_system_tool("icacls.exe"), str(path), "/L", "/inheritance:r", "/grant:r", *grants]


#: One ACE in SDDL: type, flags, rights, object, inherited object, SID.
_SDDL_ACE = re.compile(r"\(([A-Z]+);([A-Z]*);([A-Za-z0-9]+);[^;)]*;[^;)]*;([A-Za-z0-9-]+)\)")


def _sddl_is_owner_only(*, sddl: str, user_sid: str | None) -> bool:
    """Whether a folder's DACL, as SDDL, already is the owner-only list.

    That is: inheritance is off (``D:P``), no entry is inherited (``ID``), and
    SYSTEM, Administrators, OWNER RIGHTS and the user each have full control
    that every entry below inherits. An administrator's other explicit grant
    may stay, as icacls keeps it too.
    """
    match = re.search(r"D:([A-Z]*)((?:\([^)]*\))*)", sddl)
    if not match or "P" not in match.group(1):
        return False
    granted = set()
    for ace in re.findall(r"\([^)]*\)", match.group(2)):
        parts = _SDDL_ACE.fullmatch(ace)
        if not parts:
            return False
        kind, flags, rights, sid = parts.groups()
        if "ID" in flags:
            return False
        if kind == "A" and rights == "FA" and "OI" in flags and "CI" in flags and "IO" not in flags:
            granted.add(sid)
    return {"SY", "BA", "OW", *([user_sid] if user_sid else [])} <= granted


def _acl_is_owner_only(*, path: Path, user_sid: str | None) -> bool:
    """Read the folder's own list with ``icacls /save``, which walks nothing.

    False whenever the list can't be read or parsed, so the caller applies it.
    """
    saved = Path(tempfile.mkdtemp(prefix="anton-acl-")) / "acl"
    try:
        result = _run_system_tool([_system_tool("icacls.exe"), str(path), "/save", str(saved), "/L", "/Q"])
        if result.returncode != 0 or not saved.is_file():
            return False
        return _sddl_is_owner_only(sddl=saved.read_bytes().decode("utf-16", "replace"), user_sid=user_sid)
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        saved.unlink(missing_ok=True)
        saved.parent.rmdir()


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

    A folder whose list already is owner-only is left alone: applying it again
    makes Windows rewrite every entry below, scratchpad venvs included, at each
    command start. A restored inheritance shows in the list and gets the full
    apply. A volume with no access lists, such as a FAT or exFAT drive, is used
    as it is, with a warning, for the reason `make_private_dir` gives for modes.
    """
    if not _volume_keeps_acls(path=path):
        _warn_modes_not_kept(path=path, cause="volume_keeps_no_acls")
        return
    user_sid = _current_user_sid()
    if _acl_is_owner_only(path=path, user_sid=user_sid):
        return
    try:
        result = _run_system_tool(icacls_command(path=path, user_sid=user_sid))
    except (OSError, subprocess.SubprocessError) as exc:
        raise PermissionError(errno.EACCES, "Could not run icacls", str(path)) from exc
    if result.returncode:
        raise PermissionError(errno.EACCES, f"icacls exited with status {result.returncode}", str(path))
