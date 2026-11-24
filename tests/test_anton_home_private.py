"""``~/.anton`` opens only for its user, and its ``.env`` reads only for them.

The folder holds API keys, global memory and saved connections. Every step that
can create it creates it 0700, the terms and first-run steps create ``.env``
0600, and the CLI tightens an older folder and ``.env`` when it starts.
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


def _boot_cli(args: list[str]):
    """Run the real `anton` entry point up to the subcommand, with the network off."""
    from typer.testing import CliRunner

    import anton.cli as cli_mod

    with (
        mock.patch.object(cli_mod, "_reconcile_publish_identity", return_value=False),
        mock.patch.object(cli_mod, "_ensure_terms_consent"),
        # Real boot logging stops anton.usage propagating, which hides usage lines from later tests.
        mock.patch("anton.diagnostics.configure_logging"),
        mock.patch("anton.updater.check_and_update", return_value=False),
        mock.patch("anton.analytics.send_event"),
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


def test_the_cli_start_leaves_a_symlinked_home_and_its_env_alone(home, tmp_path):
    target = tmp_path / "dotfiles-anton"
    target.mkdir()
    os.chmod(target, 0o755)
    env = target / ".env"
    env.write_text("ANTON_ANTHROPIC_API_KEY=secret\n")
    os.chmod(env, 0o644)
    (home / ".anton").symlink_to(target, target_is_directory=True)

    result = _boot_cli(["--no-update", "version"])

    assert result.exit_code == 0, result.output
    assert _mode(target) == 0o755
    assert _mode(env) == 0o644


def test_the_cli_start_carries_on_when_the_home_cannot_be_made_private(home, monkeypatch):
    def refuse():
        raise PermissionError(13, "Permission denied", str(home / ".anton"))

    monkeypatch.setattr("anton.utils.private_dir.anton_home", refuse)

    result = _boot_cli(["--no-update", "version"])

    assert result.exit_code == 0, result.output
    assert "Could not make ~/.anton private" in result.output
