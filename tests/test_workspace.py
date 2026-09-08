from __future__ import annotations

import errno
import os
import stat
import sys
from pathlib import Path

import pytest

from anton.workspace import Workspace


@pytest.fixture()
def ws(tmp_path):
    return Workspace(tmp_path)


class TestFolderStateChecks:
    def test_not_initialized_by_default(self, ws):
        assert ws.is_initialized() is False

    def test_initialized_after_create(self, ws):
        ws.initialize()
        assert ws.is_initialized() is True

    def test_has_non_anton_files_empty_folder(self, ws):
        assert ws.has_non_anton_files() is False

    def test_has_non_anton_files_with_regular_files(self, ws, tmp_path):
        (tmp_path / "README.md").write_text("hello")
        assert ws.has_non_anton_files() is True

    def test_has_non_anton_files_ignores_anton_files(self, ws, tmp_path):
        (tmp_path / ".anton").mkdir()
        assert ws.has_non_anton_files() is False

    def test_has_non_anton_files_ignores_hidden_files(self, ws, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".gitignore").write_text("node_modules")
        assert ws.has_non_anton_files() is False

    def test_needs_confirmation_empty_folder(self, ws):
        assert ws.needs_confirmation() is False

    def test_needs_confirmation_non_empty_no_anton_md(self, ws, tmp_path):
        (tmp_path / "index.js").write_text("console.log('hi')")
        assert ws.needs_confirmation() is True

    def test_needs_confirmation_non_empty_with_anton_md(self, ws, tmp_path):
        (tmp_path / "index.js").write_text("console.log('hi')")
        (tmp_path / ".anton").mkdir()
        (tmp_path / ".anton" / "anton.md").write_text("context")
        assert ws.needs_confirmation() is False


class TestInitialization:
    def test_creates_anton_dir(self, ws, tmp_path):
        ws.initialize()
        assert (tmp_path / ".anton").is_dir()

    def test_creates_anton_md(self, ws, tmp_path):
        ws.initialize()
        assert (tmp_path / ".anton" / "anton.md").is_file()
        content = (tmp_path / ".anton" / "anton.md").read_text()
        assert "Anton Workspace" in content

    def test_creates_env_file(self, ws, tmp_path):
        ws.initialize()
        assert (tmp_path / ".anton" / ".env").is_file()

    def test_idempotent(self, ws, tmp_path):
        ws.initialize()
        (tmp_path / ".anton" / "anton.md").write_text("custom content")
        ws.initialize()
        # Should not overwrite existing anton.md
        assert (tmp_path / ".anton" / "anton.md").read_text() == "custom content"

    def test_returns_actions(self, ws, tmp_path):
        actions = ws.initialize()
        assert len(actions) == 4  # .anton/, anton.md, .env, artifacts/

    def test_creates_artifacts_dir(self, ws, tmp_path):
        ws.initialize()
        assert (tmp_path / ".anton" / "artifacts").is_dir()

    def test_artifacts_dir_property(self, ws, tmp_path):
        assert ws.artifacts_dir == tmp_path / ".anton" / "artifacts"

    def test_retries_once_on_stale_nfs_handle(self, ws, monkeypatch):
        calls = {"n": 0}
        real = ws._initialize_once

        def stale_then_ok(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError(errno.ESTALE, "Stale file handle")
            return real(**kwargs)

        monkeypatch.setattr(ws, "_initialize_once", stale_then_ok)
        actions = ws.initialize()
        assert calls["n"] == 2
        assert len(actions) == 4

    def test_does_not_retry_other_oserrors(self, ws, monkeypatch):
        def boom(**kwargs):
            raise OSError(errno.EACCES, "Permission denied")

        monkeypatch.setattr(ws, "_initialize_once", boom)
        with pytest.raises(OSError) as exc_info:
            ws.initialize()
        assert exc_info.value.errno == errno.EACCES

    def test_cloud_mode_skips_anton_md(self, ws, tmp_path):
        """ENG-1817: in the cloud, `.anton/anton.md` is owned by cowork-server's
        instruction staging. A pod-written template is treated as a stale staged
        copy and deleted before the next turn, and under gVisor the pod's cached
        NFS handle then fails every stat with ESTALE — so the cloud path must not
        create (or even stat) the file."""
        actions = ws.initialize(create_anton_md=False)
        assert not (tmp_path / ".anton" / "anton.md").exists()
        assert not any("anton.md" in a for a in actions)
        # Everything else is still set up.
        assert (tmp_path / ".anton" / ".env").is_file()
        assert (tmp_path / ".anton" / "artifacts").is_dir()

    def test_cloud_mode_leaves_staged_instructions_alone(self, ws, tmp_path):
        (tmp_path / ".anton").mkdir()
        (tmp_path / ".anton" / "anton.md").write_text("staged by cowork-server")
        ws.initialize(create_anton_md=False)
        assert (tmp_path / ".anton" / "anton.md").read_text() == "staged by cowork-server"


class TestAntonMd:
    def test_read_none_when_missing(self, ws):
        assert ws.read_anton_md() is None

    def test_read_content(self, ws, tmp_path):
        (tmp_path / ".anton").mkdir(exist_ok=True)
        (tmp_path / ".anton" / "anton.md").write_text("project info")
        assert ws.read_anton_md() == "project info"

    def test_tracked_read(self, ws, tmp_path):
        (tmp_path / ".anton").mkdir(exist_ok=True)
        (tmp_path / ".anton" / "anton.md").write_text("info")
        content = ws.read_anton_md_tracked()
        assert content == "info"
        # After tracked read, modified_since returns False (unless file changes)
        assert ws.anton_md_modified_since_last_read() is False

    def test_modified_since_first_read(self, ws, tmp_path):
        (tmp_path / ".anton").mkdir(exist_ok=True)
        (tmp_path / ".anton" / "anton.md").write_text("info")
        # Before any tracked read, should be True
        assert ws.anton_md_modified_since_last_read() is True

    def test_build_context_empty(self, ws):
        assert ws.build_anton_md_context() == ""

    def test_build_context_with_content(self, ws, tmp_path):
        (tmp_path / ".anton").mkdir(exist_ok=True)
        (tmp_path / ".anton" / "anton.md").write_text("Uses Python 3.11 and pytest")
        context = ws.build_anton_md_context()
        assert "Project Context" in context
        assert "Python 3.11" in context


class TestSecretVault:
    def test_load_env_empty(self, ws):
        assert ws.load_env() == {}

    def test_set_and_get_secret(self, ws, tmp_path):
        ws.initialize()
        ws.set_secret("MY_TOKEN", "abc123")
        assert ws.get_secret("MY_TOKEN") == "abc123"
        assert ws.has_secret("MY_TOKEN") is True
        assert ws.has_secret("OTHER") is False

    def test_set_secret_creates_env_dir(self, ws, tmp_path):
        # Even without initialize(), set_secret creates .anton/
        ws.set_secret("KEY", "value")
        assert (tmp_path / ".anton" / ".env").is_file()

    def test_set_secret_updates_existing(self, ws, tmp_path):
        ws.initialize()
        ws.set_secret("KEY", "old")
        ws.set_secret("KEY", "new")
        assert ws.get_secret("KEY") == "new"

    def test_set_secret_preserves_others(self, ws, tmp_path):
        ws.initialize()
        ws.set_secret("A", "1")
        ws.set_secret("B", "2")
        assert ws.get_secret("A") == "1"
        assert ws.get_secret("B") == "2"

    def test_set_secret_updates_environ(self, ws, tmp_path):
        ws.set_secret("ANTON_TEST_SECRET_XYZ", "secretval")
        assert os.environ.get("ANTON_TEST_SECRET_XYZ") == "secretval"
        # Clean up
        del os.environ["ANTON_TEST_SECRET_XYZ"]

    def test_apply_env_to_process(self, ws, tmp_path):
        ws.initialize()
        ws.set_secret("ANTON_TEST_APPLY_KEY", "applied")
        # Remove from environ to test apply
        del os.environ["ANTON_TEST_APPLY_KEY"]
        count = ws.apply_env_to_process()
        assert count >= 1
        assert os.environ.get("ANTON_TEST_APPLY_KEY") == "applied"
        # Clean up
        del os.environ["ANTON_TEST_APPLY_KEY"]

    def test_load_env_ignores_comments(self, ws, tmp_path):
        (tmp_path / ".anton").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".anton" / ".env").write_text(
            "# comment\nKEY=value\n\n# another\n"
        )
        env = ws.load_env()
        assert env == {"KEY": "value"}

    def test_remove_secret_existing(self, ws, tmp_path):
        ws.initialize()
        ws.set_secret("MY_KEY", "my_value")
        assert ws.has_secret("MY_KEY") is True
        result = ws.remove_secret("MY_KEY")
        assert result is True
        assert ws.has_secret("MY_KEY") is False

    def test_remove_secret_missing(self, ws, tmp_path):
        ws.initialize()
        result = ws.remove_secret("NONEXISTENT")
        assert result is False

    def test_remove_secret_no_env_file(self, ws):
        result = ws.remove_secret("ANYTHING")
        assert result is False

    def test_remove_secret_preserves_others(self, ws, tmp_path):
        ws.initialize()
        ws.set_secret("KEEP", "yes")
        ws.set_secret("DROP", "no")
        ws.remove_secret("DROP")
        assert ws.get_secret("KEEP") == "yes"
        assert ws.has_secret("DROP") is False

    def test_remove_secret_pops_environ(self, ws, tmp_path):
        ws.set_secret("ANTON_TEST_REMOVE_XYZ", "val")
        assert os.environ.get("ANTON_TEST_REMOVE_XYZ") == "val"
        ws.remove_secret("ANTON_TEST_REMOVE_XYZ")
        assert os.environ.get("ANTON_TEST_REMOVE_XYZ") is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
class TestSecretVaultFilePermissions:
    """The .env holds secrets, so other users must not be able to read it."""

    def _mode(self, tmp_path):
        return stat.S_IMODE((tmp_path / ".anton" / ".env").stat().st_mode)

    def test_initialize_creates_owner_only_env_file(self, ws, tmp_path):
        ws.initialize()
        assert self._mode(tmp_path) == 0o600

    def test_set_secret_creates_owner_only_env_file(self, ws, tmp_path):
        ws.set_secret("KEY", "value")
        assert self._mode(tmp_path) == 0o600

    def test_set_secret_tightens_a_world_readable_env_file(self, ws, tmp_path):
        ws.initialize()
        os.chmod(tmp_path / ".anton" / ".env", 0o644)
        ws.set_secret("KEY", "value")
        assert self._mode(tmp_path) == 0o600
        assert ws.get_secret("KEY") == "value"

    def test_remove_secret_tightens_a_world_readable_env_file(self, ws, tmp_path):
        ws.initialize()
        ws.set_secret("KEEP", "yes")
        ws.set_secret("DROP", "no")
        os.chmod(tmp_path / ".anton" / ".env", 0o644)
        ws.remove_secret("DROP")
        assert self._mode(tmp_path) == 0o600
        assert ws.get_secret("KEEP") == "yes"


class TestSecretVaultInjection:
    """A pasted value must not be able to add settings of its own.

    Every `set_secret` caller writes user-supplied input straight through —
    `cli.py`'s provider setup, `/remote`, `/connect`, `commands/setup.py`. A
    value carrying a newline would append further `ANTON_*` lines to the vault,
    which is a config injection with no legitimate use: no API key, URL or
    model name contains one. Guarded in the shared writer rather than at each
    prompt, so a new caller is covered by construction.
    """

    @pytest.mark.parametrize(
        "payload",
        ["k\nANTON_MINDS_URL=http://evil", "k\rANTON_BACKEND=remote"],
    )
    def test_a_newline_in_a_value_is_rejected(self, ws, payload):
        with pytest.raises(ValueError):
            ws.set_secret("ANTON_MINDS_API_KEY", payload)

        assert "ANTON_MINDS_URL" not in ws.load_env()
        assert "ANTON_BACKEND" not in ws.load_env()


class TestSecretVaultEncoding:
    # AntonSettings reads .anton/.env with env_file_encoding="utf-8"
    # (anton/config/settings.py). The vault must therefore be written as
    # UTF-8, not in the host locale: on a Windows code page (cp1252) a bare
    # write_text() stores a non-ASCII secret as bytes the settings loader
    # cannot decode, so every later AntonSettings() raises UnicodeDecodeError.

    def test_non_ascii_secret_is_stored_as_utf8(self, ws):
        ws.initialize()
        ws.set_secret("ANTON_MINDS_MIND_NAME", "café_mind")

        raw = ws.env_path.read_bytes()
        assert "café_mind" in raw.decode("utf-8")

    def test_secret_outside_the_host_code_page_is_storable(self, ws):
        # 密钥 has no cp1252 representation, so a bare write_text() raises
        # UnicodeEncodeError and the secret is never stored at all.
        ws.initialize()
        ws.set_secret("ANTON_MINDS_DATASOURCE", "密钥")
        assert ws.get_secret("ANTON_MINDS_DATASOURCE") == "密钥"

    def test_settings_can_load_a_vault_holding_a_non_ascii_secret(self, ws):
        from anton.config.settings import AntonSettings

        ws.initialize()
        ws.set_secret("ANTON_MINDS_MIND_NAME", "café_mind")
        assert AntonSettings(_env_file=str(ws.env_path)).minds_mind_name == "café_mind"

    def test_anton_md_is_read_as_utf8(self, ws):
        ws.initialize()
        ws.anton_md_path.write_text("café", encoding="utf-8")
        assert ws.read_anton_md() == "café"

    def test_workspace_file_io_never_relies_on_the_host_locale(self, tmp_path):
        # PEP 597: under `-X warn_default_encoding` every locale-default text
        # open() emits an EncodingWarning. Asserting that none names
        # workspace.py keeps this regression visible on UTF-8 CI too, where
        # the behavioural tests above pass whether or not the bug is present.
        import subprocess
        import sys

        import anton.workspace as workspace_mod

        repo_root = Path(workspace_mod.__file__).resolve().parent.parent
        probe = tmp_path / "probe.py"
        probe.write_text(
            "import tempfile, warnings\n"
            "from pathlib import Path\n"
            "from anton.workspace import Workspace\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.simplefilter('always')\n"
            "    ws = Workspace(Path(tempfile.mkdtemp()))\n"
            "    ws.initialize()\n"
            "    ws.set_secret('K', 'v')\n"
            "    ws.load_env()\n"
            "    ws.read_anton_md()\n"
            "    ws.remove_secret('K')\n"
            "for w in caught:\n"
            "    if w.category is EncodingWarning:\n"
            "        print(f'{w.filename}:{w.lineno}')\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, "-X", "warn_default_encoding", str(probe)],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(repo_root)},
        )

        assert proc.returncode == 0, proc.stderr
        offenders = [ln for ln in proc.stdout.splitlines() if "workspace.py" in ln]
        assert offenders == [], f"locale-default text I/O in workspace.py: {offenders}"
