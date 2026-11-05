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
from anton.core.tools.generate_artifact.state import MAX_USER_LANGUAGE_CHARS, GenState, language_name

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
    back = prompts.build_backend_system_prompt(Path("/tmp/a"), user_language="Russian")

    for system in (html, front):
        assert "Write the UI text in Russian unless the PRD or the brief asks" in system
        assert "language of the PRD" not in system
    assert "is in Russian unless the PRD or the brief asks" in back


def test_the_page_language_comes_last_so_the_prompt_prefix_is_shared():
    """Everything before the language is the same in every run, so runs in
    different languages still share one cached prefix."""
    for build in (prompts.build_subagent_system_prompt, prompts.build_frontend_system_prompt):
        english, russian = build(Path("/tmp/a"), user_language="English"), build(Path("/tmp/a"), user_language="Russian")
        cut = english.index("## UI language")
        assert russian[:cut] == english[:cut]
        assert english.rstrip().endswith("the language of the UI text.")


def test_the_page_falls_back_to_english_without_a_language():
    assert "Write the UI text in English" in prompts.build_subagent_system_prompt(Path("/tmp/a"))


class _Built(Exception):
    """Stops a generation step once its system prompt is built."""


@pytest.mark.parametrize(
    ("builder", "step", "fullstack"),
    [
        ("build_subagent_system_prompt", "_gen_verify_frontend", False),
        ("build_frontend_system_prompt", "_gen_verify_frontend", True),
        ("build_backend_system_prompt", "_gen_verify_backend", True),
    ],
)
async def test_each_generation_step_builds_its_prompt_with_the_run_language(
    tmp_path, monkeypatch, builder, step, fullstack
):
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        raise _Built

    monkeypatch.setattr(prompts, builder, spy)
    state = _state(
        tmp_path, user_language="Russian", is_fullstack=fullstack,
        artifact_type="fullstack-stateless-app" if fullstack else "html-app",
    )

    with pytest.raises(_Built):
        await getattr(orchestrator, step)(state)

    assert seen["user_language"] == "Russian"


def test_language_name_collapses_inner_whitespace():
    assert language_name(" Brazilian\n  Portuguese ") == "Brazilian Portuguese"
    assert language_name("  ") == "English"


def test_the_schema_caps_the_language_like_the_handler():
    from anton.core.tools.tool_defs import GENERATE_ARTIFACT_TOOL

    prop = GENERATE_ARTIFACT_TOOL.input_schema["properties"]["user_language"]
    assert prop["maxLength"] == MAX_USER_LANGUAGE_CHARS


def test_a_redraw_restates_the_sources_by_their_declared_names():
    """`redraw_brief` treats a source whose name changed as a new, unverified
    one; written in another language, "orders table" would come back
    translated and send the run back to the data step."""
    for act_first in (False, True):
        suffix = discovery_prompts._redraw_suffix(act_first)
        assert "that the corrected artifact still needs, use exactly the name it was declared with" in suffix


def test_a_confirming_call_repeats_the_language():
    """The language is part of the call fingerprint, so a confirming call
    that restates it differently redraws the brief and asks again."""
    from anton.core.tools.tool_defs import GENERATE_ARTIFACT_TOOL
    from anton.core.tools.tool_handlers import _STATUS_INSTRUCTIONS

    for text in (
        _STATUS_INSTRUCTIONS["needs_confirmation"],
        _STATUS_INSTRUCTIONS["stopped_over_budget"],
        GENERATE_ARTIFACT_TOOL.description,
        GENERATE_ARTIFACT_TOOL.prompt,
    ):
        assert "SAME `user_request` and `user_language`" in text


def test_the_data_steps_write_their_reasoning_in_the_run_language(tmp_path):
    line = "Write `reasoning` and any summary in Spanish."
    st = _state(tmp_path, user_language="Spanish")
    for system, _ in (
        prompts.build_required_data_prompt(st),
        prompts.build_can_fetch_prompt(st, "- sites"),
    ):
        assert line in system
    assert line in prompts.build_fetch_data_system_prompt(tmp_path, user_language="Spanish")
    assert "Write `reasoning` and any summary in English." in prompts.build_fetch_data_system_prompt(tmp_path)
