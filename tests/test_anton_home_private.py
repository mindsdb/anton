"""``~/.anton`` opens only for its user, and its ``.env`` reads only for them.

The folder holds API keys, global memory and saved connections. Every step that
can create it creates it 0700, the terms and first-run steps create ``.env``
0600, and the CLI tightens an older folder and ``.env`` when it starts, along
with the project's ``.anton``.
"""

from __future__ import annotations

import io
import os
import stat
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from rich.console import Console

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX owners and modes")


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fresh home with no ``.anton``, under the umask most systems use."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    old = os.umask(0o022)
    yield h
    os.umask(old)


def test_the_terms_step_creates_a_private_home_and_env(home, monkeypatch):
    import anton.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_render_terms_screen", lambda console: None)
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **kw: "y")
    monkeypatch.setattr(cli_mod.os, "system", lambda command: 0)
    settings = SimpleNamespace(terms_consent=False)

    cli_mod._ensure_terms_consent(Console(file=io.StringIO()), settings)

    assert _mode(home / ".anton") == 0o700
    assert _mode(home / ".anton" / ".env") == 0o600
    assert (home / ".anton" / ".env").read_text() == "ANTON_TERMS_CONSENT=true\n"
    assert settings.terms_consent is True


def test_the_first_run_step_creates_a_private_home_and_env(home):
    from anton.chat import _persist_first_run_done

    settings = SimpleNamespace(first_run_done=False)

    _persist_first_run_done(settings)

    assert _mode(home / ".anton") == 0o700
    assert _mode(home / ".anton" / ".env") == 0o600
    assert (home / ".anton" / ".env").read_text() == "ANTON_FIRST_RUN_DONE=true\n"


def test_the_first_run_step_tightens_the_env_it_appends_to(home):
    from anton.chat import _persist_first_run_done

    (home / ".anton").mkdir(mode=0o700)
    env = home / ".anton" / ".env"
    env.write_text("ANTON_MINDS_API_KEY=abc")
    os.chmod(env, 0o644)

    _persist_first_run_done(SimpleNamespace(first_run_done=False))

    assert _mode(env) == 0o600
    assert env.read_text() == "ANTON_MINDS_API_KEY=abc\nANTON_FIRST_RUN_DONE=true\n"


def test_a_saved_key_creates_a_private_home(home):
    from anton.workspace import Workspace

    Workspace(Path.home()).set_secret("ANTON_TEST_HOME_KEY", "value")
    os.environ.pop("ANTON_TEST_HOME_KEY", None)

    assert _mode(home / ".anton") == 0o700
    assert _mode(home / ".anton" / ".env") == 0o600


def test_a_custom_datasource_save_creates_a_private_home(home):
    """Cowork's agent reaches this step through anton's connect handler."""
    from anton.core.datasources.datasource_registry import DatasourceField, DatasourceRegistry
    from anton.utils.datasources import persist_custom_engine

    persist_custom_engine(
        DatasourceRegistry(), "Private Engine", [DatasourceField(name="token", secret=True)]
    )

    assert _mode(home / ".anton") == 0o700
    assert "engine: private_engine" in (home / ".anton" / "datasources.md").read_text()


def test_the_default_data_vault_creates_a_private_home(home):
    from anton.core.datasources.data_vault import LocalDataVault

    LocalDataVault().save("postgres", "private", {"host": "db.internal"})

    assert _mode(home / ".anton") == 0o700
    assert _mode(home / ".anton" / "data_vault") == 0o700


def test_a_linked_data_vault_with_a_group_writable_target_is_tightened(home, tmp_path):
    """Ubuntu's umask 0002 leaves a new folder 0775; the user's own target is tightened, not refused."""
    from anton.core.datasources.data_vault import LocalDataVault

    target = tmp_path / "vault-target"
    target.mkdir()
    os.chmod(target, 0o775)
    (home / ".anton").mkdir(mode=0o700)
    (home / ".anton" / "data_vault").symlink_to(target, target_is_directory=True)

    record = LocalDataVault().save("postgres", "private", {"host": "db.internal"})

    assert _mode(target) == 0o700
    assert _mode(record) == 0o600


def test_a_symlinked_data_vault_keeps_its_targets_mode(home, tmp_path):
    """The link's target belongs to whoever made the link; each record is still 0600."""
    from anton.core.datasources.data_vault import LocalDataVault

    target = tmp_path / "vault-target"
    target.mkdir()
    os.chmod(target, 0o755)
    (home / ".anton").mkdir(mode=0o700)
    (home / ".anton" / "data_vault").symlink_to(target, target_is_directory=True)

    record = LocalDataVault().save("postgres", "private", {"host": "db.internal"})

    assert _mode(target) == 0o755
    assert record.resolve().parent == target.resolve()
    assert _mode(record) == 0o600


def test_a_data_vault_elsewhere_leaves_the_home_alone(home, tmp_path):
    from anton.core.datasources.data_vault import LocalDataVault

    LocalDataVault(tmp_path / "vault").save("postgres", "private", {"host": "db.internal"})

    assert not (home / ".anton").exists()


def test_the_installation_id_creates_a_private_home(home, monkeypatch):
    import anton.analytics as analytics

    monkeypatch.setattr(analytics, "_cached_aid", None)
    # The multicast bit marks a node id Python made up: no real MAC address.
    monkeypatch.setattr(uuid, "getnode", lambda: 1 << 40)

    aid = analytics.get_installation_id()

    assert aid != "unknown"
    assert _mode(home / ".anton") == 0o700


async def test_the_cli_chat_creates_a_private_home(home, tmp_path):
    """The chat keeps its global memory and the skills it saves under ~/.anton."""
    import anton.chat as chat
    import anton.chat_session as chat_session
    from anton.config.settings import AntonSettings
    from tests.conftest import make_mock_llm

    class Built(Exception):
        """Stops _chat_loop once it has built its session."""

    def fake_session(config):
        raise Built

    project = tmp_path / "project"
    project.mkdir()
    settings = AntonSettings(episodic_memory=False, _env_file=None)
    settings.resolve_workspace(str(project))
    vault = mock.MagicMock()
    vault.list_connections.return_value = []
    with (
        mock.patch("anton.core.llm.client.LLMClient.from_settings", return_value=make_mock_llm()),
        mock.patch.object(chat, "LocalDataVault", return_value=vault),
        mock.patch.object(chat, "ChatSession", fake_session),
        mock.patch.object(chat, "build_runtime_context", lambda _s: ""),
        mock.patch.object(chat_session, "get_runtime_factory", lambda *_a, **_k: mock.MagicMock()),
        pytest.raises(Built),
    ):
        await chat._chat_loop(mock.MagicMock(), settings)

    assert _mode(home / ".anton") == 0o700


def _boot_cli(args: list[str], *, run_chat: mock.Mock | None = None):
    """Run the real `anton` entry point up to the subcommand, with the network off.

    With no subcommand the entry point opens the project and starts a chat;
    ``run_chat`` stands in for the chat, so the project steps run for real.
    """
    from typer.testing import CliRunner

    import anton.cli as cli_mod

    with (
        mock.patch.object(cli_mod, "_reconcile_publish_identity", return_value=False),
        mock.patch.object(cli_mod, "_ensure_terms_consent"),
        # Real boot logging stops anton.usage propagating, which hides usage lines from later tests.
        mock.patch("anton.diagnostics.configure_logging"),
        mock.patch("anton.updater.check_and_update", return_value=False),
        mock.patch("anton.analytics.send_event"),
        mock.patch.object(cli_mod, "_has_api_key", return_value=True),
        mock.patch("anton.channel.branding.render_banner"),
        mock.patch("anton.chat.run_chat", run_chat or mock.Mock()),
    ):
        return CliRunner().invoke(cli_mod.app, args)


def test_the_cli_start_tightens_an_existing_home_and_env(home):
    anton_dir = home / ".anton"
    anton_dir.mkdir()
    os.chmod(anton_dir, 0o755)
    env = anton_dir / ".env"
    env.write_text("ANTON_ANTHROPIC_API_KEY=secret\n")
    os.chmod(env, 0o644)

    result = _boot_cli(["--no-update", "version"])

    assert result.exit_code == 0, result.output
    assert _mode(anton_dir) == 0o700
    assert _mode(env) == 0o600
    assert env.read_text() == "ANTON_ANTHROPIC_API_KEY=secret\n"


def test_the_cli_start_tightens_a_linked_home_and_its_env(home, tmp_path):
    """A dotfiles setup links ~/.anton; the folder it resolves to is the user's own."""
    target = tmp_path / "dotfiles-anton"
    target.mkdir()
    os.chmod(target, 0o755)
    env = target / ".env"
    env.write_text("ANTON_ANTHROPIC_API_KEY=secret\n")
    os.chmod(env, 0o644)
    (home / ".anton").symlink_to(target, target_is_directory=True)

    result = _boot_cli(["--no-update", "version"])

    assert result.exit_code == 0, result.output
    assert (home / ".anton").is_symlink()
    assert _mode(target) == 0o700
    assert _mode(env) == 0o600


def test_a_linked_home_serves_the_chat_paths(home, tmp_path):
    from anton.utils.private_dir import anton_home

    target = tmp_path / "moved-anton"
    target.mkdir()
    os.chmod(target, 0o755)
    (home / ".anton").symlink_to(target, target_is_directory=True)

    assert anton_home() == home / ".anton"
    assert _mode(target) == 0o700


def test_a_junctioned_home_is_followed_like_a_symlinked_one(home, tmp_path, monkeypatch):
    """On Windows os.path.islink is False for the junction `mklink /J` makes."""
    from anton.utils import private_dir

    target = tmp_path / "moved-anton"
    target.mkdir()
    os.chmod(target, 0o755)
    junction = home / ".anton"
    junction.mkdir()
    real_realpath = os.path.realpath
    monkeypatch.setattr(os.path, "isjunction", lambda p: str(p) == str(junction), raising=False)
    monkeypatch.setattr(
        os.path, "realpath", lambda p, *a, **k: str(target) if str(p) == str(junction) else real_realpath(p, *a, **k)
    )

    assert private_dir._is_link_or_junction(path=junction)
    assert private_dir.make_private_dir(path=junction, follow_link=True)
    assert _mode(target) == 0o700


def test_a_junction_is_found_by_its_reparse_tag_before_python_3_12(tmp_path, monkeypatch):
    from anton.utils import private_dir

    folder = tmp_path / "folder"
    folder.mkdir()
    monkeypatch.delattr(os.path, "isjunction", raising=False)
    assert not private_dir._is_link_or_junction(path=folder)

    real_lstat = os.lstat
    tagged = SimpleNamespace(st_mode=stat.S_IFDIR, st_reparse_tag=private_dir._IO_REPARSE_TAG_MOUNT_POINT)
    monkeypatch.setattr(os, "lstat", lambda p, *a, **k: tagged if str(p) == str(folder) else real_lstat(p, *a, **k))
    assert private_dir._is_link_or_junction(path=folder)


def test_a_linked_home_another_account_owns_is_refused(home, tmp_path, monkeypatch):
    from anton.utils.private_dir import anton_home

    target = tmp_path / "their-anton"
    target.mkdir()
    os.chmod(target, 0o755)
    (home / ".anton").symlink_to(target, target_is_directory=True)
    other_uid = os.getuid() + 1
    monkeypatch.setattr(os, "getuid", lambda: other_uid)

    with pytest.raises(PermissionError):
        anton_home()
    assert _mode(target) == 0o755


def test_the_cli_start_carries_on_when_the_home_cannot_be_made_private(home, monkeypatch):
    (home / ".anton").mkdir()
    (home / ".anton" / ".env").write_text("ANTON_ANTHROPIC_API_KEY=secret\n")

    def refuse(*, path):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr("anton.utils.private_dir.tighten_private_file", refuse)

    result = _boot_cli(["--no-update", "version"])

    assert result.exit_code == 0, result.output
    assert "Could not make ~/.anton private" in result.output


def test_the_cli_start_creates_no_home(home):
    """A container whose home can't be written must not print an error on every command."""
    os.chmod(home, 0o555)
    try:
        result = _boot_cli(["--no-update", "version"])
    finally:
        os.chmod(home, 0o755)

    assert result.exit_code == 0, result.output
    assert "Could not make ~/.anton private" not in result.output
    assert not (home / ".anton").exists()


def test_the_cli_start_pins_an_existing_project_anton_folder(home, tmp_path):
    """The project is already set up, so the start never reaches initialize()."""
    project = tmp_path / "project"
    anton_dir = project / ".anton"
    (anton_dir / "memory").mkdir(parents=True)
    (anton_dir / "anton.md").write_text("# mine\n")
    os.chmod(anton_dir, 0o755)
    run_chat = mock.Mock()

    result = _boot_cli(["--no-update", "--folder", str(project)], run_chat=run_chat)

    assert result.exit_code == 0, result.output
    run_chat.assert_called_once()
    assert _mode(anton_dir) == 0o700
    assert (anton_dir / "anton.md").read_text() == "# mine\n"


@pytest.mark.parametrize(("command", "expected"), [(["sessions"], 0), (["session", "missing"], 1), (["learnings"], 0)])
@pytest.mark.parametrize("existing", [False, True])
def test_cli_history_secures_project_before_opening_store(home, tmp_path, command, expected, existing):
    project = tmp_path / "project"
    project.mkdir()
    folder = project / ".anton"
    if existing:
        folder.mkdir(mode=0o755)

    result = _boot_cli(["--no-update", "--folder", str(project), *command])

    assert result.exit_code == expected, result.output
    assert _mode(folder) == 0o700


def test_first_run_detaches_linked_dotenv_without_changing_personal_dotfiles(home, tmp_path):
    from anton.chat import _persist_first_run_done

    folder = home / ".anton"
    folder.mkdir(mode=0o700)
    personal = tmp_path / "personal.env"
    personal.write_text("SETTING=value\n")
    personal.chmod(0o644)
    managed = folder / ".env"
    managed.symlink_to(personal)

    _persist_first_run_done(SimpleNamespace(first_run_done=False))

    assert personal.read_text() == "SETTING=value\n"
    assert _mode(personal) == 0o644
    assert not managed.is_symlink()
    assert _mode(managed) == 0o600
    assert managed.read_text() == "SETTING=value\nANTON_FIRST_RUN_DONE=true\n"


@pytest.mark.parametrize(("options", "mode"), [({}, 0o770), ({"private_workspace": True}, 0o700)])
async def test_shared_runtime_builder_uses_host_workspace_policy(home, tmp_path, monkeypatch, options, mode):
    from anton.core.runtime import build_chat_session

    monkeypatch.setenv("ANTON_ANTHROPIC_API_KEY", "test-local-session")
    project = tmp_path / "project"
    directory = project / ".anton"
    directory.mkdir(parents=True)
    directory.chmod(0o770)

    session = await build_chat_session(session_id="local-check", workspace_path=str(project), **options)
    try:
        assert _mode(directory) == mode
    finally:
        await session.close()
