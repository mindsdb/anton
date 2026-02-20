from __future__ import annotations

import os
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
        (tmp_path / "anton.md").write_text("context")
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
        (tmp_path / "anton.md").write_text("context")
        assert ws.needs_confirmation() is False


class TestInitialization:
    def test_creates_anton_dir(self, ws, tmp_path):
        ws.initialize()
        assert (tmp_path / ".anton").is_dir()

    def test_creates_anton_md(self, ws, tmp_path):
        ws.initialize()
        assert (tmp_path / "anton.md").is_file()
        content = (tmp_path / "anton.md").read_text()
        assert "Anton Workspace" in content

    def test_creates_env_file(self, ws, tmp_path):
        ws.initialize()
        assert (tmp_path / ".anton" / ".env").is_file()

    def test_idempotent(self, ws, tmp_path):
        ws.initialize()
        (tmp_path / "anton.md").write_text("custom content")
        ws.initialize()
        # Should not overwrite existing anton.md
        assert (tmp_path / "anton.md").read_text() == "custom content"

    def test_returns_actions(self, ws):
        actions = ws.initialize()
        assert len(actions) == 3  # .anton/, anton.md, .env


class TestAntonMd:
    def test_read_none_when_missing(self, ws):
        assert ws.read_anton_md() is None

    def test_read_content(self, ws, tmp_path):
        (tmp_path / "anton.md").write_text("project info")
        assert ws.read_anton_md() == "project info"

    def test_tracked_read(self, ws, tmp_path):
        (tmp_path / "anton.md").write_text("info")
        content = ws.read_anton_md_tracked()
        assert content == "info"
        # After tracked read, modified_since returns False (unless file changes)
        assert ws.anton_md_modified_since_last_read() is False

    def test_modified_since_first_read(self, ws, tmp_path):
        (tmp_path / "anton.md").write_text("info")
        # Before any tracked read, should be True
        assert ws.anton_md_modified_since_last_read() is True

    def test_build_context_empty(self, ws):
        assert ws.build_anton_md_context() == ""

    def test_build_context_with_content(self, ws, tmp_path):
        (tmp_path / "anton.md").write_text("Uses Python 3.11 and pytest")
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
