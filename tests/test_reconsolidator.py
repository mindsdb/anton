from __future__ import annotations

import json
from pathlib import Path

import pytest

from anton.memory.reconsolidator import needs_reconsolidation, reconsolidate


@pytest.fixture()
def project_dir(tmp_path):
    d = tmp_path / ".anton"
    d.mkdir()
    return d


class TestNeedsReconsolidation:
    def test_no_legacy_no_new(self, project_dir):
        assert needs_reconsolidation(project_dir) is False

    def test_legacy_context_exists(self, project_dir):
        ctx = project_dir / "context"
        ctx.mkdir()
        (ctx / "overview.md").write_text("Python project")
        assert needs_reconsolidation(project_dir) is True

    def test_legacy_learnings_exists(self, project_dir):
        learn = project_dir / "learnings"
        learn.mkdir()
        (learn / "api.md").write_text("API facts")
        assert needs_reconsolidation(project_dir) is True

    def test_already_migrated(self, project_dir):
        # Legacy exists
        ctx = project_dir / "context"
        ctx.mkdir()
        (ctx / "overview.md").write_text("Python project")
        # New also exists
        mem = project_dir / "memory"
        mem.mkdir()
        (mem / "lessons.md").write_text("# Lessons\n- Already migrated")
        assert needs_reconsolidation(project_dir) is False

    def test_empty_legacy_dirs(self, project_dir):
        (project_dir / "context").mkdir()
        (project_dir / "learnings").mkdir()
        # Dirs exist but are empty
        assert needs_reconsolidation(project_dir) is False


class TestReconsolidate:
    def test_migrates_context_files(self, project_dir):
        ctx = project_dir / "context"
        ctx.mkdir()
        (ctx / "stack.md").write_text("Python 3.11\nasyncio-based\nUses pytest")

        actions = reconsolidate(project_dir)
        assert len(actions) >= 1  # At least the summary + 1 file
        assert any("context/stack.md" in a for a in actions)

        # Check new format
        lessons = project_dir / "memory" / "lessons.md"
        assert lessons.exists()
        content = lessons.read_text()
        assert "Python 3.11" in content

    def test_migrates_learnings_files(self, project_dir):
        learn = project_dir / "learnings"
        learn.mkdir()
        (learn / "api_design.md").write_text("## API Design\n\nUse REST conventions\nAlways version your API")
        (learn / "index.json").write_text(json.dumps({
            "api_design": {"topic": "API Design", "summary": "REST conventions"}
        }))

        actions = reconsolidate(project_dir)
        assert any("learnings/api_design.md" in a for a in actions)

        lessons = project_dir / "memory" / "lessons.md"
        assert lessons.exists()
        content = lessons.read_text()
        assert "REST conventions" in content

    def test_no_legacy_returns_empty(self, project_dir):
        actions = reconsolidate(project_dir)
        assert actions == []

    def test_skips_short_lines(self, project_dir):
        ctx = project_dir / "context"
        ctx.mkdir()
        (ctx / "notes.md").write_text("OK\n\nThis is a real fact about the project")

        reconsolidate(project_dir)
        lessons = project_dir / "memory" / "lessons.md"
        content = lessons.read_text()
        assert "OK" not in content.split("\n")  # "OK" alone shouldn't be a lesson
        assert "This is a real fact" in content

    def test_skips_dotfiles_and_dirs(self, project_dir):
        ctx = project_dir / "context"
        ctx.mkdir()
        (ctx / ".hidden").write_text("secret")
        sub = ctx / "subdir"
        sub.mkdir()
        (sub / "nested.md").write_text("nested")
        (ctx / "visible.md").write_text("This is visible content here")

        actions = reconsolidate(project_dir)
        lessons = project_dir / "memory" / "lessons.md"
        content = lessons.read_text()
        assert "secret" not in content
        assert "nested" not in content
        assert "visible content" in content
