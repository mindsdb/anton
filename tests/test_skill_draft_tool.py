"""`create_skill_draft` — folder claiming, seeding from the store, and the
registration gate that keeps it from colliding with a host's own tool."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.memory.skills import Skill, SkillStore
from anton.core.tools.skill_draft import handle_create_skill_draft


def _call(session, **args) -> dict:
    """Unwrap the handler's ToolOutcome the way the tool loop does (ENG-2248).

    `.content` is what reaches the model and lands in history; `.ok` is the
    verdict the streak reads. These tests assert on the content, so they read
    exactly what they read before the migration — see
    `test_tool_verdict_migration.py` for the `.ok` half.
    """
    outcome = asyncio.run(handle_create_skill_draft(session, args))
    return json.loads(getattr(outcome, "content", outcome))


def _session(drafts_root, store=None):
    return SimpleNamespace(_skill_drafts_root=drafts_root, _skill_store=store)


def _store(root: Path, label: str = "csv-summary") -> SkillStore:
    store = SkillStore(root=root)
    store.save(Skill(
        label=label,
        name="CSV Summary",
        description="Summarize a CSV",
        declarative_md="1. Read it\n2. Summarize it",
        created_at="2026-01-01T00:00:00+00:00",
        provenance="manual",
    ))
    return store


def test_claims_a_folder_under_the_drafts_root(tmp_path):
    out = _call(_session(tmp_path), name="Competitive Analysis")
    assert out["slug"] == "competitive-analysis"
    assert Path(out["path"]) == tmp_path / "competitive-analysis"
    assert Path(out["path"]).is_dir()
    assert out["skill_file"].endswith("/competitive-analysis/SKILL.md")


def test_name_is_slugified_not_taken_literally(tmp_path):
    out = _call(_session(tmp_path), name="  Weekly   Report!!  ")
    assert out["slug"] == "weekly-report"


@pytest.mark.parametrize("name", ["", "   ", "!!!"])
def test_unusable_name_is_an_error_not_a_folder(tmp_path, name):
    assert "error" in _call(_session(tmp_path), name=name)
    assert list(tmp_path.iterdir()) == []


def test_no_drafts_root_is_an_error(tmp_path):
    assert "error" in _call(_session(None), name="anything")


def test_editing_a_saved_skill_seeds_the_draft_from_it(tmp_path):
    store = _store(tmp_path / "store")
    out = _call(_session(tmp_path / "drafts", store), name="csv-summary")

    body = (Path(out["path"]) / "SKILL.md").read_text()
    assert "Summarize it" in body, "an edit must start from the saved version"


def test_seeding_leaves_store_only_files_behind(tmp_path):
    store = _store(tmp_path / "store")
    assert (store.root / "csv-summary" / "stats.json").is_file()

    out = _call(_session(tmp_path / "drafts", store), name="csv-summary")
    # Recall counters belong to the store, not to the skill's content.
    assert not (Path(out["path"]) / "stats.json").exists()


def test_a_second_call_does_not_clobber_what_the_agent_wrote(tmp_path):
    store = _store(tmp_path / "store")
    session = _session(tmp_path / "drafts", store)

    out = _call(session, name="csv-summary")
    Path(out["skill_file"]).write_text("agent's own draft")
    _call(session, name="csv-summary")

    assert Path(out["skill_file"]).read_text() == "agent's own draft"


def test_unsaved_skill_yields_an_empty_folder_not_an_error(tmp_path):
    store = _store(tmp_path / "store")
    out = _call(_session(tmp_path / "drafts", store), name="brand-new")
    assert "error" not in out
    assert list(Path(out["path"]).iterdir()) == []
