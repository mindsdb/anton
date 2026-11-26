"""A project's ``.anton`` opens only for the user who runs anton.

It holds the project's memory, conversation records and ``.env``. On macOS and
Linux the folder is 0700, new or existing. On Windows it gets an access list
that removes inherited access before private writes. The global ``~/.anton``
keeps the profile's list. The last tests check that list on a real Windows runner.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.workspace import Workspace

_posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX owners and modes")

#: Named as text for mocking the platform branch.
_HELPER = "anton.utils.private_dir"

#: Older development builds left this marker. It is repository data now.
_MARKER = ".owner-only-acl"

_GRANTS = ["/L", "/inheritance:r", "/grant:r", "*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F", "*S-1-3-4:(OI)(CI)F"]

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

    Workspace(project).initialize(private=True)

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

    Workspace(tmp_path).initialize(private=True)

    assert _mode(anton_dir) == 0o700
    # Files are private because their folder is; their own modes stay.
    assert _mode(note) == 0o644
    assert note.read_text() == "kept"


@_posix_only
def test_the_link_refusing_initialize_makes_anton_private_too(tmp_path, umask_022):
    Workspace(tmp_path).initialize(refuse_links=True, private=True)

    assert _mode(tmp_path / ".anton") == 0o700


@_posix_only
def test_a_saved_secret_creates_a_private_anton_folder(tmp_path, umask_022):
    Workspace(tmp_path).set_secret("ANTON_TEST_PRIVATE_KEY", "value")

    assert _mode(tmp_path / ".anton") == 0o700
    assert _mode(tmp_path / ".anton" / ".env") == 0o600
    os.environ.pop("ANTON_TEST_PRIVATE_KEY", None)


@_posix_only
def test_a_linked_private_project_is_refused_without_changing_its_target(tmp_path, umask_022):
    target = tmp_path / "elsewhere"
    target.mkdir()
    os.chmod(target, 0o755)
    project = tmp_path / "project"
    project.mkdir()
    (project / ".anton").symlink_to(target, target_is_directory=True)

    with pytest.raises(PermissionError):
        Workspace(project).initialize(private=True)

    assert _mode(target) == 0o755
    assert not (target / "memory").exists()


@_posix_only
def test_an_anton_folder_another_account_owns_keeps_its_mode(tmp_path, monkeypatch, umask_022):
    anton_dir = tmp_path / ".anton"
    anton_dir.mkdir()
    os.chmod(anton_dir, 0o755)
    other_uid = os.getuid() + 1
    monkeypatch.setattr(os, "getuid", lambda: other_uid)

    with pytest.raises(PermissionError):
        Workspace(tmp_path).initialize(private=True)

    assert not (anton_dir / ".env").exists()
    assert _mode(anton_dir) == 0o755


@_posix_only
def test_initialize_refuses_private_writes_when_the_folder_mode_cannot_change(tmp_path, monkeypatch, caplog, umask_022):
    """A FAT drive or a read-only mount refuses the chmod, so secrets must not be written there."""
    anton_dir = tmp_path / ".anton"
    anton_dir.mkdir()
    os.chmod(anton_dir, 0o755)

    def refuse(fd, mode):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "fchmod", refuse)

    with caplog.at_level(logging.WARNING, logger=_HELPER):
        with pytest.raises(PermissionError):
            Workspace(tmp_path).initialize(private=True)

    assert not (anton_dir / ".env").exists()
    assert _mode(anton_dir) == 0o755
    assert [
        (r.levelno, getattr(r, "event", None), getattr(r, "reason", None))
        for r in caplog.records if r.name == _HELPER
    ] == [(logging.WARNING, "private_folder_chmod_failed", "EPERM")]


@_posix_only
def test_initialize_takes_an_existing_folder_whose_mkdir_reports_another_error(tmp_path, monkeypatch):
    """Some systems answer EROFS or EACCES, not EEXIST, for a folder that exists on a read-only mount."""
    Workspace(tmp_path).initialize(private=True)

    def read_only(path, mode=0o777, **kwargs):
        raise OSError(errno.EROFS, "Read-only file system", str(path))

    monkeypatch.setattr(os, "mkdir", read_only)

    actions = Workspace(tmp_path).initialize(private=True)

    assert actions == [f"Created {tmp_path / '.anton'}"]


def test_a_file_at_the_anton_name_still_fails_initialize(tmp_path):
    (tmp_path / ".anton").write_text("not a folder")

    with pytest.raises(FileExistsError):
        Workspace(tmp_path).initialize(private=True)


@_posix_only
def test_posix_never_runs_icacls(tmp_path, monkeypatch):
    calls: list[Path] = []
    monkeypatch.setattr(f"{_HELPER}._apply_owner_only_acl", lambda *, path: calls.append(path))

    Workspace(tmp_path).initialize(private=True)
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

    # Libraries imported by the CLI use the real subprocess module as well.
    acl_subprocess = SimpleNamespace(**vars(subprocess))
    acl_subprocess.run = fake_run
    monkeypatch.setattr(f"{_HELPER}.subprocess", acl_subprocess)
    return calls, result


def _assert_owner_only_call(call: tuple[list[str], dict], folder: Path) -> None:
    argv, kwargs = call
    assert argv[0].lower().endswith("\\system32\\icacls.exe")
    assert argv[1:] == [str(folder), *_GRANTS]
    assert not kwargs.get("shell")


def test_windows_reapplies_the_acl_before_each_private_write(tmp_path, icacls_calls):
    calls, _ = icacls_calls
    workspace = Workspace(tmp_path)
    workspace.initialize(private=True)
    workspace.initialize(private=True)
    workspace.set_secret("ANTON_TEST_PRIVATE_KEY", "value")
    os.environ.pop("ANTON_TEST_PRIVATE_KEY", None)

    assert len(calls) == 3
    for call in calls:
        _assert_owner_only_call(call, tmp_path / ".anton")
    assert not (tmp_path / ".anton" / _MARKER).exists()


def test_windows_acl_setup_preserves_repository_metadata_bytes(tmp_path, icacls_calls):
    calls, _ = icacls_calls
    anton_dir = tmp_path / ".anton"
    anton_dir.mkdir()
    ignore = anton_dir / ".gitignore"
    original = "# café\n.env*\n".encode("cp1252")
    ignore.write_bytes(original)
    old_marker = anton_dir / _MARKER
    old_marker.write_bytes(b"previous release")

    Workspace(tmp_path).initialize(private=True)

    assert len(calls) == 1
    assert ignore.read_bytes() == original
    assert old_marker.read_bytes() == b"previous release"


def test_windows_leaves_the_home_anton_folder_to_the_profiles_list(tmp_path, monkeypatch, icacls_calls):
    """~/.anton sits in the profile, whose list already shuts out other standard accounts."""
    from anton.chat import _persist_first_run_done

    calls, _ = icacls_calls
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    Workspace(home).initialize(private=True)
    Workspace(home).set_secret("ANTON_TEST_PRIVATE_KEY", "value")
    os.environ.pop("ANTON_TEST_PRIVATE_KEY", None)
    _persist_first_run_done(SimpleNamespace(first_run_done=False))

    assert calls == []
    assert (home / ".anton" / "anton.md").is_file()
    assert not (home / ".anton" / _MARKER).exists()


@pytest.mark.parametrize("outcome", [5, FileNotFoundError(2, "icacls not found")])
def test_windows_refuses_private_writes_until_icacls_succeeds(tmp_path, icacls_calls, outcome):
    calls, result = icacls_calls
    result["returncode"] = outcome
    workspace = Workspace(tmp_path)

    with pytest.raises(PermissionError):
        workspace.initialize(private=True)
    assert not (tmp_path / ".anton" / ".env").exists()
    with pytest.raises(PermissionError):
        workspace.set_secret("TEST_KEY", "value")
    assert not (tmp_path / ".anton" / ".env").exists()

    result["returncode"] = 0
    workspace.initialize(private=True)

    assert len(calls) == 3
    assert (tmp_path / ".anton" / ".env").is_file()


@_posix_only
@pytest.mark.parametrize("refuse_links", [True, False])
def test_managed_initialization_preserves_operator_permissions(tmp_path, refuse_links):
    directory = tmp_path / ".anton"
    directory.mkdir()
    directory.chmod(0o770)

    workspace = Workspace(tmp_path)
    workspace.initialize(create_anton_md=False, refuse_links=refuse_links)
    workspace.set_secret("TEST_SHARED_PROJECT_SETTING", "setting")
    workspace.remove_secret("TEST_SHARED_PROJECT_SETTING")

    assert _mode(directory) == 0o770
    assert (directory / "memory").is_dir()


# ── Windows, on a real runner ───────────────────────────────────────────────


def _acl(path: Path) -> tuple[bool, set[str]]:
    """Whether ``path``'s access list ignores its parent's, and the SIDs its rules name.

    Windows PowerShell 5.1 runs with the inherited PSModulePath removed: a
    PowerShell 7 parent, such as a CI step's shell, puts its own module folders
    there, and Get-Acl then fails to load. The errors are made terminating, so
    a failed read fails the test instead of reading as an empty list.
    """
    script = (
        "$ErrorActionPreference = 'Stop'; "
        "$acl = Get-Acl -LiteralPath $env:ANTON_ACL_PATH; "
        "$sids = @($acl.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier]) "
        "| ForEach-Object { $_.IdentityReference.Value }); "
        # Joined, because 5.1 can serialize an array inside a hashtable as an object.
        "@{ protected = $acl.AreAccessRulesProtected; sids = ($sids -join ',') } | ConvertTo-Json -Compress"
    )
    done = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        env={
            **{key: value for key, value in os.environ.items() if key.upper() != "PSMODULEPATH"},
            "ANTON_ACL_PATH": str(path),
        },
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    data = json.loads(done.stdout)
    return bool(data["protected"]), {sid for sid in data["sids"].split(",") if sid}


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

    Workspace(project).initialize(private=True)

    protected, sids = _acl(project / ".anton")
    assert protected
    assert not sids & _BROAD_SIDS, sids
    assert not (project / ".anton" / _MARKER).exists()
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

    Workspace(users_readable_project).initialize(private=True)

    assert _acl(anton_dir)[0]
    for path in existing:
        _, sids = _acl(path)
        assert not sids & _BROAD_SIDS, (path, sids)
    assert (anton_dir / ".env").read_text() == "ANTON_TEST_PRIVATE_KEY=value\n"


@_windows_only
def test_windows_revisiting_project_repairs_reenabled_inheritance(users_readable_project):
    workspace = Workspace(users_readable_project)
    directory = users_readable_project / ".anton"
    workspace.initialize(private=True)
    # An explicit administrator grant survives the repair of inherited access.
    service_sid = "S-1-5-19"
    subprocess.run(["icacls", str(directory), "/grant", f"*{service_sid}:RX"], check=True, capture_output=True)
    subprocess.run(["icacls", str(directory), "/inheritance:e"], check=True, capture_output=True)
    assert "S-1-5-32-545" in _acl(directory)[1]

    workspace.initialize(private=True)

    protected, sids = _acl(directory)
    assert protected
    assert service_sid in sids
    assert not sids & _BROAD_SIDS


@_windows_only
def test_windows_refuses_directory_junction_before_changing_acl(tmp_path):
    from _winapi import CreateJunction

    target = tmp_path / "original"
    target.mkdir()
    directory = tmp_path / ".anton"
    CreateJunction(str(target), str(directory))
    before = _acl(target)

    with pytest.raises(PermissionError):
        Workspace(tmp_path).make_anton_dir_private()

    assert _acl(target) == before
