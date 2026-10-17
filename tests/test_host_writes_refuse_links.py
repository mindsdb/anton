"""The host's own writes inside a workspace and links planted there.

Agent code could plant links in its project before the sandbox, and the host
runs unlocked, so a write that followed one would land in the link's target:
another person's file. Writes inside a venv anton builds, and the agent-pads
record, refuse a symlink and write no file with another hard link on Linux
and macOS; Windows follows links. Workspace-level folders (a linked `.anton`
or conversation folder) and the reads of the workspace's `.env`, `anton.md`
and agent-pads record are refused only under a sandbox policy; without one,
the desktop, the CLI and the cloud pod follow them as they always have.
"""

from __future__ import annotations

import errno
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import anton.core.backends.local as local
from anton.core.artifacts import backend_launcher
from anton.core.backends.local import LocalScratchpadRuntime, snapshot_file
from anton.core.backends.manager import ScratchpadManager
from anton.core.backends.sandbox import CellSandbox
from anton.utils import nofollow
from anton.utils.nofollow import make_dirs_nofollow, write_text_nofollow
from anton.workspace import Workspace

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")


@pytest.fixture
def victim(tmp_path: Path) -> Path:
    folder = tmp_path / "alice"
    folder.mkdir()
    (folder / "private.txt").write_text("alice's")
    return folder


def _unchanged(victim: Path) -> bool:
    return sorted(p.name for p in victim.iterdir()) == ["private.txt"] and (
        (victim / "private.txt").read_text() == "alice's"
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    folder = tmp_path / "project"
    folder.mkdir()
    return folder


@pytest.fixture
def policy(tmp_path: Path, project: Path) -> CellSandbox:
    for name in ("home", "tmp"):
        (tmp_path / name).mkdir()
    return CellSandbox(project_dir=project, home_dir=tmp_path / "home", tmp_dir=tmp_path / "tmp", user_name="bob")


@pytest.fixture
def linked_anton(tmp_path: Path, project: Path) -> Path:
    """The person's own `.anton`, linked to another folder of theirs."""
    target = tmp_path / "elsewhere"
    target.mkdir()
    (project / ".anton").symlink_to(target, target_is_directory=True)
    return target


def _pad(project: Path, policy: CellSandbox | None = None, *, session_id: str | None = None) -> LocalScratchpadRuntime:
    return LocalScratchpadRuntime(
        "main", coding_provider="anthropic", coding_model="", coding_api_key="", coding_base_url="",
        workspace_path=project, cell_sandbox=policy, session_id=session_id,
    )


def _manager(project: Path, policy: CellSandbox | None = None) -> ScratchpadManager:
    return ScratchpadManager(
        runtime_factory=local.local_scratchpad_runtime_factory,
        coding_provider="anthropic", coding_model="", coding_api_key="", coding_base_url="",
        workspace_path=project, session_id="conv-1", cell_sandbox=policy,
    )


def _venv(project: Path) -> Path:
    venv = project / ".anton" / "scratchpad-venvs" / "main"
    (venv / "lib" / "python3.12").mkdir(parents=True)
    return venv


# ── Inside a venv: refused in every mode ────────────────────────────────────


def test_requirements_and_version_files_refuse_a_link(project, victim):
    venv = _venv(project)
    (venv / "requirements.txt").symlink_to(victim / "private.txt")
    (venv / ".python_version").symlink_to(victim / "private.txt")
    pad = _pad(project)
    pad._venv_dir = str(venv)
    pad._installed_packages.add("pandas")
    pad._save_requirements()
    pad._save_python_version()
    assert _unchanged(victim)


def test_the_parent_packages_file_refuses_a_linked_site_packages(project, victim, monkeypatch):
    import site

    monkeypatch.setattr(local.sys, "prefix", "/venv")
    monkeypatch.setattr(local.sys, "base_prefix", "/usr")
    monkeypatch.setattr(site, "getsitepackages", lambda: ["/parent/site-packages"])
    venv = _venv(project)
    (venv / "lib" / "python3.12" / "site-packages").symlink_to(victim, target_is_directory=True)
    pad = _pad(project)
    pad._venv_dir = str(venv)
    with pytest.raises(OSError):
        pad._setup_parent_site_packages()
    assert _unchanged(victim)


def test_building_a_venv_never_empties_another_pads_venv_a_link_points_at(project, monkeypatch):
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    base = project / ".anton" / "scratchpad-venvs"
    (base / "other").mkdir(parents=True)
    (base / "other" / "alice-data.csv").write_text("alice")
    (base / "main").symlink_to(base / "other", target_is_directory=True)
    pad = _pad(project)
    with pytest.raises(OSError, match="refused to follow a symlink"):
        pad._create_venv()
    assert (base / "other" / "alice-data.csv").read_text() == "alice"


def test_requirements_and_version_files_refuse_a_hard_link(project, victim):
    venv = _venv(project)
    os.link(victim / "private.txt", venv / "requirements.txt")
    os.link(victim / "private.txt", venv / ".python_version")
    pad = _pad(project)
    pad._venv_dir = str(venv)
    pad._installed_packages.add("pandas")
    pad._save_requirements()
    pad._save_python_version()
    assert _unchanged(victim)


def test_a_host_write_refuses_a_fifo_without_waiting_for_a_reader(project):
    venv = _venv(project)
    os.mkfifo(venv / "requirements.txt")
    with pytest.raises(OSError):
        write_text_nofollow(project, venv / "requirements.txt", "pandas\n")


@pytest.mark.parametrize("kind", ["symlink", "hard link"])
@pytest.mark.parametrize("name", ["_agent_pads.json.tmp", "_agent_pads.json"])
def test_the_agent_pads_record_never_writes_through_a_link(project, victim, kind, name):
    manager = _manager(project)
    record = project / ".anton" / "scratchpad-sessions" / "conv-1"
    record.mkdir(parents=True)
    if kind == "symlink":
        (record / name).symlink_to(victim / "private.txt")
    else:
        os.link(victim / "private.txt", record / name)
    manager.record_agent_pad("main")
    assert _unchanged(victim)
    assert json.loads((record / "_agent_pads.json").read_text()) == ["main"]
    assert not (record / "_agent_pads.json.tmp").exists()


def test_a_refused_link_is_named_with_its_full_path(project, victim):
    (project / ".anton").symlink_to(victim, target_is_directory=True)
    with pytest.raises(OSError) as folder_refusal:
        make_dirs_nofollow(project, project / ".anton" / "memory")
    assert folder_refusal.value.errno == errno.ELOOP
    assert folder_refusal.value.strerror == "refused to follow a symlink"
    assert folder_refusal.value.filename == str(project / ".anton")

    (project / "notes.txt").symlink_to(victim / "private.txt")
    with pytest.raises(OSError) as file_refusal:
        write_text_nofollow(project, project / "notes.txt", "x")
    assert file_refusal.value.errno == errno.ELOOP
    assert file_refusal.value.filename == str(project / "notes.txt")
    assert _unchanged(victim)


# ── Workspace-level links: refused only under a sandbox policy ──────────────


def test_building_a_venv_under_the_sandbox_never_empties_a_folder_a_link_points_at(
    project, victim, policy, monkeypatch
):
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    (project / ".anton").symlink_to(victim, target_is_directory=True)
    (victim / "scratchpad-venvs" / "main").mkdir(parents=True)
    (victim / "scratchpad-venvs" / "main" / "alice-data.csv").write_text("alice")
    pad = _pad(project, policy)
    with pytest.raises(OSError, match="refused to follow a symlink"):
        pad._create_venv()
    assert (victim / "scratchpad-venvs" / "main" / "alice-data.csv").read_text() == "alice"


def test_deleting_a_venv_under_the_sandbox_never_follows_a_link_above_it(project, victim, policy):
    (project / ".anton").symlink_to(victim, target_is_directory=True)
    (victim / "scratchpad-venvs" / "main").mkdir(parents=True)
    (victim / "scratchpad-venvs" / "main" / "alice-data.csv").write_text("alice")
    pad = _pad(project, policy)
    pad._venv_dir = str(project / ".anton" / "scratchpad-venvs" / "main")
    pad._nuke_venv()
    assert (victim / "scratchpad-venvs" / "main" / "alice-data.csv").read_text() == "alice"


def test_the_agent_pads_record_under_the_sandbox_refuses_a_linked_anton_folder(project, victim, policy):
    (project / ".anton").symlink_to(victim, target_is_directory=True)
    manager = _manager(project, policy)
    manager.record_agent_pad("main")
    assert manager.agent_pads() == set()
    assert _unchanged(victim)


def _alice_pads(victim: Path, sessions: Path, how: str) -> None:
    """Alice's agent-pads record, reached from Bob's conversation `conv-1` through a link."""
    (victim / "_agent_pads.json").write_text('["alice-pad"]')
    sessions.mkdir(parents=True)
    if how == "linked folder":
        (sessions / "conv-1").symlink_to(victim, target_is_directory=True)
        return
    (sessions / "conv-1").mkdir()
    if how == "linked file":
        (sessions / "conv-1" / "_agent_pads.json").symlink_to(victim / "_agent_pads.json")
    else:
        os.link(victim / "_agent_pads.json", sessions / "conv-1" / "_agent_pads.json")


@pytest.mark.parametrize("how", ["linked folder", "linked file", "hard link"])
def test_the_agent_pads_record_under_the_sandbox_never_reads_through_a_link(project, victim, policy, how):
    _alice_pads(victim, project / ".anton" / "scratchpad-sessions", how)
    assert _manager(project, policy).agent_pads() == set()


def test_under_the_sandbox_the_agent_pads_record_still_reads_its_own(project, policy):
    manager = _manager(project, policy)
    manager.record_agent_pad("main")
    manager.record_agent_pad("other")
    assert manager.agent_pads() == {"main", "other"}


def test_without_the_sandbox_the_agent_pads_record_follows_a_linked_folder(project, victim):
    _alice_pads(victim, project / ".anton" / "scratchpad-sessions", "linked folder")
    assert _manager(project).agent_pads() == {"alice-pad"}


def test_without_the_sandbox_a_linked_anton_folder_still_builds_and_deletes_venvs(
    project, linked_anton, monkeypatch
):
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    pad = _pad(project)
    pad._create_venv()
    assert (linked_anton / "scratchpad-venvs" / "main" / "pyvenv.cfg").is_file()
    pad._nuke_venv()
    assert not (linked_anton / "scratchpad-venvs" / "main").exists()


def test_without_the_sandbox_a_linked_anton_folder_still_records_agent_pads(project, linked_anton):
    manager = _manager(project)
    manager.record_agent_pad("main")
    assert manager.agent_pads() == {"main"}
    assert (linked_anton / "scratchpad-sessions" / "conv-1" / "_agent_pads.json").is_file()


def test_workspace_setup_never_creates_a_file_through_a_dangling_link(project, victim):
    anton_dir = project / ".anton"
    anton_dir.mkdir()
    (anton_dir / ".env").symlink_to(victim / "planted.env")
    (anton_dir / "anton.md").symlink_to(victim / "planted.md")
    Workspace(project).initialize(refuse_links=True)
    assert _unchanged(victim)


def test_workspace_setup_refuses_a_linked_anton_folder(project, victim):
    (project / ".anton").symlink_to(victim, target_is_directory=True)
    with pytest.raises(OSError, match="refused to follow a symlink"):
        Workspace(project).initialize(refuse_links=True)
    assert _unchanged(victim)


def test_workspace_setup_without_the_sandbox_follows_a_linked_anton_folder(project, linked_anton):
    actions = Workspace(project).initialize()
    assert f"Created {project / '.anton' / 'anton.md'}" in actions
    assert (linked_anton / "anton.md").read_text().startswith("# Anton Workspace")
    assert (linked_anton / ".env").read_text() == "# Anton environment variables\n"
    assert (linked_anton / ".env").stat().st_mode & 0o777 == 0o600
    assert (linked_anton / "memory").is_dir()


@pytest.mark.parametrize("refuse_links", [False, True])
def test_workspace_setup_still_creates_its_files_once(project, refuse_links):
    anton_dir = project / ".anton"
    first = Workspace(project).initialize(refuse_links=refuse_links)
    assert first[:3] == [
        f"Created {anton_dir}", f"Created {anton_dir / 'anton.md'}", f"Created {anton_dir / '.env'}",
    ]
    assert (anton_dir / "anton.md").read_text().startswith("# Anton Workspace")
    assert (anton_dir / ".env").read_text() == "# Anton environment variables\n"
    assert (anton_dir / ".env").stat().st_mode & 0o777 == 0o600
    assert (anton_dir / "memory").is_dir()
    assert Workspace(project).initialize(refuse_links=refuse_links) == [f"Created {anton_dir}"]


@pytest.mark.parametrize("name", [".env", "anton.md"])
@pytest.mark.parametrize("kind", ["symlink", "hard link"])
def test_under_the_sandbox_workspace_reads_refuse_a_linked_env_or_anton_md(project, victim, name, kind):
    (victim / "private.txt").write_text("ALICE_KEY=alice's\n")
    anton_dir = project / ".anton"
    anton_dir.mkdir()
    if kind == "symlink":
        (anton_dir / name).symlink_to(victim / "private.txt")
    else:
        os.link(victim / "private.txt", anton_dir / name)
    workspace = Workspace(project)
    workspace.initialize(refuse_links=True)
    read = workspace.load_env if name == ".env" else workspace.build_anton_md_context
    with pytest.raises(OSError, match="symlink|hard link"):
        read()


def test_under_the_sandbox_workspace_reads_refuse_a_linked_anton_folder(project, victim):
    (victim / ".env").write_text("ALICE_KEY=alice's\n")
    (victim / "anton.md").write_text("alice's notes")
    workspace = Workspace(project)
    workspace.initialize(refuse_links=True)
    (project / ".anton" / ".env").unlink()
    (project / ".anton" / "anton.md").unlink()
    (project / ".anton").rename(project / ".anton-was")
    (project / ".anton").symlink_to(victim, target_is_directory=True)
    with pytest.raises(OSError, match="refused to follow a symlink"):
        workspace.load_env()
    with pytest.raises(OSError, match="refused to follow a symlink"):
        workspace.read_anton_md()


def test_under_the_sandbox_workspace_reads_still_read_their_own_files(project):
    workspace = Workspace(project)
    workspace.initialize(refuse_links=True)
    (project / ".anton" / ".env").write_text("MY_KEY=mine\n")
    (project / ".anton" / "anton.md").write_text("my notes")
    assert workspace.load_env() == {"MY_KEY": "mine"}
    assert workspace.read_anton_md() == "my notes"
    (project / ".anton" / "anton.md").unlink()
    assert workspace.read_anton_md() is None


def test_without_the_sandbox_workspace_reads_follow_a_link(project, victim):
    (victim / "private.txt").write_text("ALICE_KEY=alice's\n")
    anton_dir = project / ".anton"
    anton_dir.mkdir()
    (anton_dir / ".env").symlink_to(victim / "private.txt")
    workspace = Workspace(project)
    workspace.initialize()
    assert workspace.load_env() == {"ALICE_KEY": "alice's"}


def _alice_snapshot(victim: Path) -> tuple[Path, bytes]:
    """Alice's saved state for a pad named `main`, under the name Bob's pad would use."""
    name = snapshot_file(Path("/x/.anton/scratchpad-venvs"), "conv-1", "main").name
    saved = victim / name
    saved.write_bytes(b"alice's namespace")
    (victim / f"{name}.123.tmp").write_bytes(b"alice's partial save")
    return saved, saved.read_bytes()


def test_under_the_sandbox_a_reset_never_deletes_through_a_linked_conversation_folder(project, victim, policy):
    saved, before = _alice_snapshot(victim)
    sessions = project / ".anton" / "scratchpad-sessions"
    sessions.mkdir(parents=True)
    (sessions / "conv-1").symlink_to(victim, target_is_directory=True)
    pad = _pad(project, policy, session_id="conv-1")
    pad._discard_session_snapshot()
    assert saved.read_bytes() == before
    assert (victim / f"{saved.name}.123.tmp").read_bytes() == b"alice's partial save"
    # Nor does the next start hand the cell the linked folder.
    assert pad._session_snapshot_path(create=True) is None


def test_under_the_sandbox_a_reset_still_deletes_its_own_snapshot(project, policy):
    pad = _pad(project, policy, session_id="conv-1")
    own = pad._session_snapshot_path(create=True)
    assert own == project / ".anton" / "scratchpad-sessions" / "conv-1" / own.name
    own.write_bytes(b"my namespace")
    (own.parent / f"{own.name}.42.tmp").write_bytes(b"partial")
    (own.parent / "_agent_pads.json").write_text('["main"]')
    pad._discard_session_snapshot()
    assert sorted(p.name for p in own.parent.iterdir()) == ["_agent_pads.json"]


def test_under_the_sandbox_a_folder_at_the_snapshot_name_still_lets_a_reset_delete_the_rest(project, policy):
    pad = _pad(project, policy, session_id="conv-1")
    own = pad._session_snapshot_path(create=True)
    own.mkdir()
    (own.parent / f"{own.name}.42.tmp").write_bytes(b"partial")
    pad._discard_session_snapshot()
    assert sorted(p.name for p in own.parent.iterdir()) == [own.name]


def test_without_the_sandbox_a_linked_conversation_folder_is_followed_as_before(project, victim):
    saved, _ = _alice_snapshot(victim)
    sessions = project / ".anton" / "scratchpad-sessions"
    sessions.mkdir(parents=True)
    (sessions / "conv-1").symlink_to(victim, target_is_directory=True)
    _pad(project, session_id="conv-1")._discard_session_snapshot()
    assert not saved.exists()


# ── The Windows branches, driven on this host ───────────────────────────────


def test_the_windows_branches_write_where_they_always_did(project, monkeypatch):
    """Windows has no dir_fd, so each helper acts on the joined path there."""
    import site

    monkeypatch.setattr(nofollow, "_IS_WINDOWS", True)
    monkeypatch.setattr(LocalScratchpadRuntime, "_find_uv", staticmethod(lambda: None))
    created: list[str] = []
    monkeypatch.setattr(local.venv, "create", lambda path, **kwargs: created.append(path))
    pad = _pad(project)
    pad._create_venv()
    venv = project / ".anton" / "scratchpad-venvs" / "main"
    assert created == [str(venv)] and venv.is_dir()

    (venv / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    monkeypatch.setattr(local.sys, "prefix", "/venv")
    monkeypatch.setattr(local.sys, "base_prefix", "/usr")
    monkeypatch.setattr(site, "getsitepackages", lambda: ["/parent/a", "/parent/b"])
    pad._setup_parent_site_packages()
    pth = venv / "lib" / "python3.12" / "site-packages" / "_parent_venv.pth"
    assert pth.read_text() == "/parent/a\n/parent/b\n"

    pad._installed_packages.update({"pandas", "numpy"})
    pad._save_requirements()
    assert (venv / "requirements.txt").read_text() == "numpy\npandas\n"

    manager = _manager(project)
    manager.record_agent_pad("main")
    assert json.loads((project / ".anton" / "scratchpad-sessions" / "conv-1" / "_agent_pads.json").read_text()) == [
        "main"
    ]

    pad._nuke_venv()
    assert not venv.exists()
    assert (project / ".anton" / "scratchpad-venvs").is_dir()


class _Pool:
    async def venv_python(self, name):
        return sys.executable

    async def get_or_create(self, name):
        return SimpleNamespace()


async def test_a_linked_artifact_folder_still_launches_its_backend(tmp_path):
    """App backends run only without a sandbox policy, so backend.log follows links as before."""
    real = tmp_path / "real-app"
    real.mkdir()
    (real / "backend.py").write_text("print('backend output', flush=True)\n")
    folder = tmp_path / "artifacts" / "app"
    folder.parent.mkdir()
    folder.symlink_to(real, target_is_directory=True)
    result = await backend_launcher.launch_artifact_backend(
        slug="app", artifact_folder=folder, scratchpad_pool=_Pool(), tracked_backends={},
        health_timeout=10,
    )
    assert isinstance(result, str) and "exited early" in result and "backend output" in result
    assert "backend output" in (real / "backend.log").read_text()
