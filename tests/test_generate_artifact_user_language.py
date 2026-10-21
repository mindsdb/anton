"""The language the calling agent states reaches every step by name.

An English request got a Russian or Spanish brief when a step had to infer
the language: "in the same language as the user request", and since the
section headings had to follow it, "translate the section names above".
Replaying recorded briefs on gpt-5.6-luna, that sentence switched 12 of 45
English briefs; the same sentence without "translate", or with the language
named, switched none.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from anton.core.tools.generate_artifact import engine, orchestrator, prompts
from anton.core.tools.generate_artifact.discovery import checkpoint as cp
from anton.core.tools.generate_artifact.discovery import prompts as discovery_prompts
from anton.core.tools.generate_artifact.discovery import sub_tools
from anton.core.tools.generate_artifact.state import GenState

DOCUMENT_STEPS = (
    sub_tools.STEP_GATHERING,
    sub_tools.STEP_DRAFT_BRIEF,
    sub_tools.STEP_REDRAW_BRIEF,
    sub_tools.STEP_WRITE_PRD,
)

# Wordings that leave the model to work out the language.
INFERRED_LANGUAGE = ("same language as the user request", "language of the user request", "translat")


def _session() -> AsyncMock:
    session = AsyncMock()
    session._scratchpads.cell_sandbox = None
    session.question_count = 0
    return session


def _state(tmp_path: Path, **kw) -> GenState:
    base = dict(
        session=_session(), artifact_type="html-app", artifact_path=tmp_path,
        slug="a", brief="Show sales", is_fullstack=False,
        user_request="show sales", agent_understanding="a sales page",
    )
    base.update(kw)
    return GenState(**base)


@pytest.mark.parametrize("act_first", [False, True])
@pytest.mark.parametrize("step", DOCUMENT_STEPS)
def test_every_document_step_names_the_language_and_leaves_nothing_to_infer(tmp_path, step, act_first):
    message = discovery_prompts.step_message(step, _state(tmp_path, user_language="English", act_first=act_first))

    assert "User language: English." in message
    for phrase in INFERRED_LANGUAGE:
        assert phrase not in message.lower(), (step, phrase)


async def test_generate_hands_the_stated_language_to_the_run(tmp_path, monkeypatch):
    seen = {}

    async def fake_run(state, *, entry):
        seen["state"] = state
        return {"status": "generated", "files_written": [], "internal_files": [], "trace": []}

    monkeypatch.setattr(orchestrator, "run", fake_run)
    monkeypatch.setattr(engine, "_scratchpads_context", lambda session: "")
    art = tmp_path / "art"
    art.mkdir()

    await engine.generate(
        session=_session(), slug="a", artifact_path=art, artifact_type="html-app",
        user_request="r", agent_understanding="u", user_language="Russian",
    )

    state = seen["state"]
    assert state.user_language == "Russian"
    for step in DOCUMENT_STEPS:
        assert "User language: Russian." in discovery_prompts.step_message(step, state), step


def test_a_saved_checkpoint_keeps_the_brief_for_the_same_language_only(tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    orchestrator._save_checkpoint(_state(tmp_path, artifact_path=art, user_language="Russian"), cp.STAGE_AWAITING_CONFIRMATION)
    stored = cp.load(art)

    same = _state(tmp_path, artifact_path=art, user_language="russian")
    engine._restore(same, stored)
    assert same.call_changed is False

    changed = _state(tmp_path, artifact_path=art, user_language="Spanish")
    engine._restore(changed, stored)
    assert changed.call_changed is True


def test_the_page_is_written_in_the_stated_language_unless_the_prd_asks_otherwise():
    html = prompts.build_subagent_system_prompt(Path("/tmp/a"), user_language="Russian")
    front = prompts.build_frontend_system_prompt(Path("/tmp/a"), user_language="Russian")

    for system in (html, front):
        assert "Write the UI text in Russian unless the PRD asks for\nanother language" in system
        assert "language of the PRD" not in system


def test_the_page_falls_back_to_english_without_a_language():
    assert "Write the UI text in English" in prompts.build_subagent_system_prompt(Path("/tmp/a"))


def test_a_redraw_restates_the_sources_by_their_declared_names():
    """`redraw_brief` treats a source whose name changed as a new, unverified
    one; written in another language, "orders table" would come back
    translated and send the run back to the data step."""
    suffix = discovery_prompts._redraw_suffix(False)
    assert "by exactly the name it was declared with" in suffix
    assert "by exactly the name it was declared with" in discovery_prompts._redraw_suffix(True)
