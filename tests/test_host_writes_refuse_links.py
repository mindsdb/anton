"""The host's own writes inside a workspace and links planted there.

Agent code could plant links in its project before the sandbox, and the host
runs unlocked, so a write that followed one would land in the link's target:
another person's file. Writes inside a venv anton builds, and the agent-pads
record, refuse a link in every mode. Workspace-level folders (a linked
`.anton`) are refused only under a sandbox policy; without one, the desktop,
the CLI and the cloud pod follow them as they always have.
"""

from __future__ import annotations

import errno
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import anton.core.backends.local as local
from anton.core.artifacts import backend_launcher
from anton.core.backends.local import LocalScratchpadRuntime
from anton.core.backends.manager import ScratchpadManager
from anton.core.backends.sandbox import CellSandbox
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


def _pad(project: Path, policy: CellSandbox | None = None) -> LocalScratchpadRuntime:
    return LocalScratchpadRuntime(
        "main", coding_provider="anthropic", coding_model="", coding_api_key="", coding_base_url="",
        workspace_path=project, cell_sandbox=policy,
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


def test_deleting_a_venv_never_follows_a_link_at_it(project, victim):
    base = project / ".anton" / "scratchpad-venvs"
    base.mkdir(parents=True)
    (base / "main").symlink_to(victim, target_is_directory=True)
    pad = _pad(project)
    pad._venv_dir = str(base / "main")
    pad._nuke_venv()
    assert _unchanged(victim)


def test_the_agent_pads_record_refuses_a_link(project, victim):
    manager = _manager(project)
    record = project / ".anton" / "scratchpad-sessions" / "conv-1"
    record.mkdir(parents=True)
    (record / "_agent_pads.json.tmp").symlink_to(victim / "private.txt")
    manager.record_agent_pad("main")
    assert _unchanged(victim)

    (record / "_agent_pads.json.tmp").unlink()
    manager.record_agent_pad("main")
    assert json.loads((record / "_agent_pads.json").read_text()) == ["main"]


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
