"""A project's ``.anton`` opens only for the user who runs anton.

It holds the project's memory, conversation records and ``.env``. On macOS and
Linux the folder is 0700, new or existing. On Windows it gets an access list
that names only SYSTEM, Administrators and its owner, once, recorded by a
marker inside that names the folder; ``~/.anton`` sits in the profile and keeps
the profile's list. The last tests check that list on a real Windows runner.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.workspace import Workspace

_posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX owners and modes")

#: Kept before any test replaces it, for the tests that run git while icacls is faked.
_real_run = subprocess.run

#: The helper module and its marker are named as text, not imported, so each
#: test reports its own result against code that predates the helper.
_HELPER = "anton.utils.private_dir"

#: The marker's name is on-disk state an existing folder already carries, so
#: it is pinned here rather than read from the helper.
_MARKER = ".owner-only-acl"

_GRANTS = ["/inheritance:r", "/grant:r", "*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F", "*S-1-3-4:(OI)(CI)F"]

#: Users, Authenticated Users and Everyone: a rule for any of them lets
#: another local account in.
_BROAD_SIDS = {"S-1-5-32-545", "S-1-5-11", "S-1-1-0"}


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


@pytest.fixture
def umask_022():
    old = os.umask(0o022)
    yield
    os.umask(old)


# ── macOS and Linux ─────────────────────────────────────────────────────────


@_posix_only
def test_a_new_anton_folder_is_private_and_its_project_keeps_its_mode(tmp_path, umask_022):
    project = tmp_path / "project"

    Workspace(project).initialize()

    assert _mode(project / ".anton") == 0o700
    # The project is the user's folder, not anton's: created with the default mode.
    assert _mode(project) == 0o755


@_posix_only
def test_an_existing_anton_folder_is_made_private(tmp_path, umask_022):
    anton_dir = tmp_path / ".anton"
    anton_dir.mkdir(mode=0o755)
    os.chmod(anton_dir, 0o755)
    note = anton_dir / "note.md"
    note.write_text("kept")
    os.chmod(note, 0o644)

    Workspace(tmp_path).initialize()

    assert _mode(anton_dir) == 0o700
    # Files are private because their folder is; their own modes stay.
    assert _mode(note) == 0o644
    assert note.read_text() == "kept"


@_posix_only
def test_the_link_refusing_initialize_makes_anton_private_too(tmp_path, umask_022):
    Workspace(tmp_path).initialize(refuse_links=True)

    assert _mode(tmp_path / ".anton") == 0o700


@_posix_only
def test_a_saved_secret_creates_a_private_anton_folder(tmp_path, umask_022):
    Workspace(tmp_path).set_secret("ANTON_TEST_PRIVATE_KEY", "value")

    assert _mode(tmp_path / ".anton") == 0o700
    assert _mode(tmp_path / ".anton" / ".env") == 0o600
    os.environ.pop("ANTON_TEST_PRIVATE_KEY", None)


@_posix_only
def test_a_symlinked_anton_folder_is_used_but_its_target_keeps_its_mode(tmp_path, umask_022):
    target = tmp_path / "elsewhere"
    target.mkdir()
    os.chmod(target, 0o755)
    project = tmp_path / "project"
    project.mkdir()
    (project / ".anton").symlink_to(target, target_is_directory=True)

    Workspace(project).initialize()

    assert _mode(target) == 0o755
    assert (target / "memory").is_dir()


@_posix_only
def test_an_anton_folder_another_account_owns_keeps_its_mode(tmp_path, monkeypatch, umask_022):
    anton_dir = tmp_path / ".anton"
    anton_dir.mkdir()
    os.chmod(anton_dir, 0o755)
    other_uid = os.getuid() + 1
    monkeypatch.setattr(os, "getuid", lambda: other_uid)

    Workspace(tmp_path).initialize()

    assert _mode(anton_dir) == 0o755


@_posix_only
def test_initialize_carries_on_when_the_folder_mode_cannot_change(tmp_path, monkeypatch, caplog, umask_022):
    """A FAT drive or a read-only mount refuses the chmod, and the folder still works."""
    anton_dir = tmp_path / ".anton"
    anton_dir.mkdir()
    os.chmod(anton_dir, 0o755)

    def refuse(fd, mode):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "fchmod", refuse)

    with caplog.at_level(logging.WARNING, logger=_HELPER):
        actions = Workspace(tmp_path).initialize()

    assert f"Created {anton_dir}" in actions
    assert (anton_dir / ".env").is_file()
    assert _mode(anton_dir) == 0o755
    assert [
        (r.levelno, getattr(r, "event", None), getattr(r, "reason", None))
        for r in caplog.records if r.name == _HELPER
    ] == [(logging.WARNING, "private_folder_chmod_failed", "EPERM")]


@_posix_only
def test_initialize_takes_an_existing_folder_whose_mkdir_reports_another_error(tmp_path, monkeypatch):
    """Some systems answer EROFS or EACCES, not EEXIST, for a folder that exists on a read-only mount."""
    Workspace(tmp_path).initialize()

    def read_only(path, mode=0o777, **kwargs):
        raise OSError(errno.EROFS, "Read-only file system", str(path))

    monkeypatch.setattr(os, "mkdir", read_only)

    actions = Workspace(tmp_path).initialize()

    assert actions == [f"Created {tmp_path / '.anton'}"]


def test_a_file_at_the_anton_name_still_fails_initialize(tmp_path):
    (tmp_path / ".anton").write_text("not a folder")

    with pytest.raises(FileExistsError):
        Workspace(tmp_path).initialize()


@_posix_only
def test_posix_never_runs_icacls(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append(argv))

    Workspace(tmp_path).initialize()
    Workspace(tmp_path).set_secret("ANTON_TEST_PRIVATE_KEY", "value")
    os.environ.pop("ANTON_TEST_PRIVATE_KEY", None)

    assert calls == []
    assert not (tmp_path / ".anton" / _MARKER).exists()


# ── Windows, with icacls replaced ───────────────────────────────────────────


@pytest.fixture
def icacls_calls(monkeypatch):
    """Run the Windows branch, recording each icacls call instead of making it."""
    monkeypatch.setattr(f"{_HELPER}._IS_WINDOWS", True)
    calls: list[tuple[list[str], dict]] = []
    result = {"returncode": 0}

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if isinstance(result["returncode"], BaseException):
            raise result["returncode"]
        return subprocess.CompletedProcess(argv, result["returncode"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls, result


def _assert_owner_only_call(call: tuple[list[str], dict], folder: Path) -> None:
    argv, kwargs = call
    assert argv[0].lower().endswith("\\system32\\icacls.exe")
    assert argv[1:] == [str(folder), *_GRANTS]
    assert not kwargs.get("shell")


def test_windows_gives_a_new_anton_folder_the_owner_only_list_once(tmp_path, icacls_calls):
    calls, _ = icacls_calls

    Workspace(tmp_path).initialize()
    Workspace(tmp_path).initialize()
    Workspace(tmp_path).set_secret("ANTON_TEST_PRIVATE_KEY", "value")
    os.environ.pop("ANTON_TEST_PRIVATE_KEY", None)

    assert len(calls) == 1
    _assert_owner_only_call(calls[0], tmp_path / ".anton")
    assert (tmp_path / ".anton" / _MARKER).is_file()


def test_windows_gives_an_existing_anton_folder_the_list_once(tmp_path, icacls_calls):
    calls, _ = icacls_calls
    (tmp_path / ".anton").mkdir()
    (tmp_path / ".anton" / "anton.md").write_text("# mine\n")

    Workspace(tmp_path).initialize()
    Workspace(tmp_path).initialize()

    assert len(calls) == 1
    _assert_owner_only_call(calls[0], tmp_path / ".anton")
    assert (tmp_path / ".anton" / "anton.md").read_text() == "# mine\n"


def test_windows_gives_a_copy_of_a_marked_anton_folder_the_list_again(tmp_path, icacls_calls):
    """A clone of a repository that committed the marker, or a copy in Explorer, carries it."""
    calls, _ = icacls_calls
    original = tmp_path / "original"
    original.mkdir()
    Workspace(original).initialize()
    copy = tmp_path / "copy"
    shutil.copytree(original, copy)
    assert (copy / ".anton" / _MARKER).is_file()

    Workspace(copy).initialize()
    Workspace(copy).initialize()
    Workspace(original).initialize()

    assert len(calls) == 2
    _assert_owner_only_call(calls[1], copy / ".anton")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_windows_keeps_the_marker_out_of_git(tmp_path, icacls_calls):
    _real_run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".anton").mkdir()
    (tmp_path / ".anton" / ".gitignore").write_text(".env*")

    Workspace(tmp_path).initialize()

    assert (tmp_path / ".anton" / _MARKER).is_file()
    # The project's own lines stay, and the marker joins them.
    assert (tmp_path / ".anton" / ".gitignore").read_text().split() == [".env*", _MARKER]
    status = _real_run(
        ["git", "-C", str(tmp_path), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert ".anton/anton.md" in status
    assert _MARKER not in status


def test_windows_leaves_the_home_anton_folder_to_the_profiles_list(tmp_path, monkeypatch, icacls_calls):
    """~/.anton sits in the profile, whose list already shuts out other standard accounts."""
    from anton.chat import _persist_first_run_done

    calls, _ = icacls_calls
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    Workspace(home).initialize()
    Workspace(home).set_secret("ANTON_TEST_PRIVATE_KEY", "value")
    os.environ.pop("ANTON_TEST_PRIVATE_KEY", None)
    _persist_first_run_done(SimpleNamespace(first_run_done=False))

    assert calls == []
    assert (home / ".anton" / "anton.md").is_file()
    assert not (home / ".anton" / _MARKER).exists()


def _acl_warnings(caplog) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records
        if r.name == _HELPER and r.levelno == logging.WARNING
        and getattr(r, "event", None) == "private_folder_acl_failed"
    ]


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [(5, "icacls_refused"), (FileNotFoundError(2, "icacls not found"), "icacls_not_run")],
)
def test_windows_logs_a_failed_icacls_call_and_tries_again_next_time(
    tmp_path, icacls_calls, caplog, outcome, reason
):
    calls, result = icacls_calls
    result["returncode"] = outcome

    with caplog.at_level(logging.WARNING, logger=_HELPER):
        actions = Workspace(tmp_path).initialize()

    assert f"Created {tmp_path / '.anton'}" in actions
    assert (tmp_path / ".anton" / ".env").is_file()
    assert not (tmp_path / ".anton" / _MARKER).exists()
    assert [getattr(r, "reason", None) for r in _acl_warnings(caplog)] == [reason]

    result["returncode"] = 0
    Workspace(tmp_path).initialize()

    assert len(calls) == 2
    assert (tmp_path / ".anton" / _MARKER).is_file()


# ── Windows, on a real runner ───────────────────────────────────────────────


def _acl(path: Path) -> tuple[bool, set[str]]:
    """Whether ``path``'s access list ignores its parent's, and the SIDs its rules name."""
    script = (
        "$acl = Get-Acl -LiteralPath $env:ANTON_ACL_PATH; "
        "$sids = @($acl.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier]) "
        "| ForEach-Object { $_.IdentityReference.Value }); "
        "@{ protected = $acl.AreAccessRulesProtected; sids = $sids } | ConvertTo-Json -Compress"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        env={**os.environ, "ANTON_ACL_PATH": str(path)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    data = json.loads(out)
    sids = data["sids"]
    return bool(data["protected"]), {sids} if isinstance(sids, str) else set(sids)


_windows_only = pytest.mark.skipif(sys.platform != "win32", reason="reads a real Windows access list")


@pytest.fixture
def users_readable_project(tmp_path) -> Path:
    """A project folder that grants Users read, as a folder under ``C:\\`` does."""
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["icacls", str(project), "/grant", "*S-1-5-32-545:(OI)(CI)RX"], check=True, capture_output=True)
    return project


@_windows_only
def test_windows_anton_folder_shuts_out_a_users_grant_on_its_project(users_readable_project):
    project = users_readable_project
    # A plain folder in the same project inherits that grant, so the check below can fire.
    plain = project / "plain"
    plain.mkdir()
    plain_protected, plain_sids = _acl(plain)
    assert not plain_protected
    assert "S-1-5-32-545" in plain_sids

    Workspace(project).initialize()

    protected, sids = _acl(project / ".anton")
    assert protected
    assert not sids & _BROAD_SIDS, sids
    assert (project / ".anton" / _MARKER).is_file()
    # The files inside take the folder's list.
    _, env_sids = _acl(project / ".anton" / ".env")
    assert not env_sids & _BROAD_SIDS, env_sids


@_windows_only
def test_windows_existing_anton_folder_and_its_files_lose_a_users_grant(users_readable_project):
    """An .anton an earlier version made inherits the project's grant, and so does each file in it.

    Python's os.mkdir(mode=0o700) protects only a folder it creates, so this
    case passes only when icacls replaces the existing folder's list and the
    change reaches the files already inside.
    """
    anton_dir = users_readable_project / ".anton"
    (anton_dir / "memory").mkdir(parents=True)
    (anton_dir / ".env").write_text("ANTON_TEST_PRIVATE_KEY=value\n")
    (anton_dir / "memory" / "x.md").write_text("a note\n")
    existing = (anton_dir, anton_dir / ".env", anton_dir / "memory" / "x.md")
    for path in existing:
        assert "S-1-5-32-545" in _acl(path)[1], path

    Workspace(users_readable_project).initialize()

    assert _acl(anton_dir)[0]
    for path in existing:
        _, sids = _acl(path)
        assert not sids & _BROAD_SIDS, (path, sids)
    assert (anton_dir / ".env").read_text() == "ANTON_TEST_PRIVATE_KEY=value\n"
