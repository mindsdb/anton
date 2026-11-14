"""The project's anton.md reaches every generator step as rules, not content.

Rules that reached the page writer only inside the brief or the PRD were read
as content: pages printed the rules block as text. So the rules travel in each
step's system prompt, under a heading that says to follow them and never show
them, and never in a kickoff.

The rules and identity the conversation recalled from memory reach the steps
that never read the gathering kickoff (the two page writers and the cold tech
spec) the same way.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from anton.core.tools.generate_artifact import engine, orchestrator, prompts
from anton.core.tools.generate_artifact.state import GenState
from anton.workspace import Workspace

RULE = "Every page must show the text PROJECT-RULE-OK in its footer"
RULES = prompts.project_rules_section(RULE)
NEVER_SHOWN = "never print, quote or summarize the rules themselves"


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
        project_rules=RULES,
    )
    base.update(kw)
    return GenState(**base)


def _system_prompts(state: GenState, project_rules: str) -> dict[str, str]:
    """Every generator step's system prompt, built the way the run builds it."""
    path = state.artifact_path
    return {
        "pipeline": state.pipeline_system,
        "define_required_data": prompts.build_required_data_prompt(state)[0],
        "is_possible_to_fetch": prompts.build_can_fetch_prompt(state, "- sales")[0],
        "fetch_data_sample": prompts.build_fetch_data_system_prompt(path, project_rules=project_rules),
        "make_tech_spec (cold)": prompts.build_tech_spec_prompt(state)[0],
        "make_api_spec (cold)": prompts.build_api_spec_prompt("ctx", project_rules=project_rules)[0],
        "html-app page": prompts.build_subagent_system_prompt(path, project_rules=project_rules),
        "fullstack page": prompts.build_frontend_system_prompt(path, project_rules=project_rules),
        "backend": prompts.build_backend_system_prompt(path, project_rules=project_rules),
    }


def test_the_data_steps_put_the_language_line_before_the_rules(tmp_path):
    """A project rule about language comes after the run's language line,
    so the rule has the last word, as it does in the generators."""
    state = _state(tmp_path, user_language="Spanish")
    line = "Write `reasoning` and any summary in Spanish."
    for system in (
        prompts.build_required_data_prompt(state)[0],
        prompts.build_can_fetch_prompt(state, "- sales")[0],
        prompts.build_fetch_data_system_prompt(tmp_path, project_rules=RULES, user_language="Spanish"),
    ):
        assert system.index(line) < system.index(RULES)


def test_every_step_carries_the_rules_under_the_heading(tmp_path):
    for step, system in _system_prompts(_state(tmp_path), RULES).items():
        assert prompts.PROJECT_RULES_HEADER in system, step
        rules = system[system.index(prompts.PROJECT_RULES_HEADER):]
        assert RULE in rules, step
        assert NEVER_SHOWN in rules, step
        assert prompts.OVER_DEFAULTS in rules, step


def test_no_anton_md_leaves_every_prompt_as_it_was(tmp_path):
    state = _state(tmp_path, project_rules="")
    for step, system in _system_prompts(state, "").items():
        assert prompts.PROJECT_RULES_HEADER not in system, step
    path = state.artifact_path
    assert prompts.build_subagent_system_prompt(path, project_rules="") == prompts.build_subagent_system_prompt(path)
    assert prompts.build_backend_system_prompt(path, project_rules="") == prompts.build_backend_system_prompt(path)


@pytest.mark.parametrize("anton_md", ["", "   \n\t\n"])
def test_a_blank_anton_md_renders_no_section(anton_md):
    assert prompts.project_rules_section(anton_md) == ""


def test_the_rules_never_travel_in_a_kickoff(tmp_path):
    """The kickoff is the content channel the printed rules came from."""
    state = _state(tmp_path)
    context = orchestrator._spec_context(state)
    for kickoff in (
        prompts.build_user_kickoff(context),
        prompts.build_frontend_kickoff(context, "{}"),
        prompts.build_backend_kickoff(context, "{}"),
        prompts.build_fetch_data_kickoff(state),
    ):
        assert RULE not in kickoff


@pytest.mark.parametrize(
    "build", [prompts.build_subagent_system_prompt, prompts.build_frontend_system_prompt]
)
def test_the_rules_follow_the_fixed_blocks_and_the_language_stays_last(build):
    """Projects with different rules still share one cached prefix."""
    plain = build(Path("/tmp/a"), user_language="English")
    ruled = build(Path("/tmp/a"), user_language="English", project_rules=RULES)
    cut = ruled.index(prompts.PROJECT_RULES_HEADER)
    assert plain[:cut] == ruled[:cut]
    assert ruled.index("## UI language") > cut
    assert ruled.rstrip().endswith("the language of the UI text.")


class _Built(Exception):
    """Stops a step once its system prompt is built."""


@pytest.mark.parametrize(
    ("builder", "step", "fullstack"),
    [
        ("build_subagent_system_prompt", "_gen_verify_frontend", False),
        ("build_frontend_system_prompt", "_gen_verify_frontend", True),
        ("build_backend_system_prompt", "_gen_verify_backend", True),
        ("build_fetch_data_system_prompt", "_fetch_data_sample", False),
    ],
)
async def test_each_step_builds_its_prompt_with_the_run_rules(
    tmp_path, monkeypatch, builder, step, fullstack
):
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        raise _Built

    monkeypatch.setattr(prompts, builder, spy)
    state = _state(
        tmp_path, is_fullstack=fullstack, public_sources="",
        artifact_type="fullstack-stateless-app" if fullstack else "html-app",
    )

    with pytest.raises(_Built):
        await getattr(orchestrator, step)(state)

    assert seen["project_rules"] == RULES


async def test_the_cold_api_spec_gets_the_run_rules(tmp_path, monkeypatch):
    seen = {}

    async def spy(*args, **kwargs):
        seen.update(kwargs)
        raise _Built

    monkeypatch.setattr(engine, "_generate_api_spec", spy)
    state = _state(tmp_path, is_fullstack=True, artifact_type="fullstack-stateless-app")

    with pytest.raises(_Built):
        await orchestrator._make_api_spec(state)

    assert seen["project_rules"] == RULES


TEAL = "## Your Memory — Global Rules\n## Always\n- Always use teal (#008080) for chart bars"


@pytest.mark.parametrize(
    ("builder", "fullstack"),
    [("build_subagent_system_prompt", False), ("build_frontend_system_prompt", True)],
)
async def test_both_page_writers_carry_the_recalled_preferences(tmp_path, monkeypatch, builder, fullstack):
    """The page writers read only the PRD and spec otherwise, and those may
    leave a saved chart color out."""
    real = getattr(prompts, builder)
    built = {}

    def spy(*args, **kwargs):
        built["system"] = real(*args, **kwargs)
        raise _Built

    monkeypatch.setattr(prompts, builder, spy)
    state = _state(
        tmp_path, is_fullstack=fullstack, recalled_preferences=TEAL,
        artifact_type="fullstack-stateless-app" if fullstack else "html-app",
    )

    with pytest.raises(_Built):
        await orchestrator._gen_verify_frontend(state)

    system = built["system"]
    section = system[system.index(prompts.RECALLED_PREFERENCES_HEADER):]
    assert "#008080" in section
    assert "never print, quote or summarize them, or the user's name or profile" in section
    assert prompts.OVER_DEFAULTS in section


async def test_the_cold_tech_spec_carries_the_recalled_preferences(tmp_path, monkeypatch):
    seen = {}

    async def spy(*args, **kwargs):
        seen.update(kwargs)
        raise _Built

    monkeypatch.setattr(engine, "_plan_whole_document", spy)
    state = _state(tmp_path, recalled_preferences=TEAL)
    assert not state.messages                                  # the cold path

    with pytest.raises(_Built):
        await orchestrator._write_tech_spec(state)

    assert "#008080" in seen["system"]
    assert "#008080" not in seen["user"]


ORANGE = "Use orange chart bars this time"


def _stated_then_recalled(system: str) -> None:
    """The agent's stated preferences lead the recalled ones and win over them."""
    section = system[system.index(prompts.RECALLED_PREFERENCES_HEADER):]
    stated = section.index(f"Stated for this request:\n{ORANGE}")
    assert stated < section.index(f"Recalled from memory:\n{TEAL}")
    assert "or from the stated preferences, the request and the stated preferences win" in section


@pytest.mark.parametrize(
    ("builder", "fullstack"),
    [("build_subagent_system_prompt", False), ("build_frontend_system_prompt", True)],
)
async def test_both_page_writers_put_the_stated_preferences_ahead_of_the_recalled(
    tmp_path, monkeypatch, builder, fullstack,
):
    """The brief, PRD and spec may leave "orange this time" out; the page
    writer then still reads it ahead of the saved teal rule."""
    real = getattr(prompts, builder)
    built = {}

    def spy(*args, **kwargs):
        built["system"] = real(*args, **kwargs)
        raise _Built

    monkeypatch.setattr(prompts, builder, spy)
    state = _state(
        tmp_path, is_fullstack=fullstack, recalled_preferences=TEAL, user_preferences=ORANGE,
        artifact_type="fullstack-stateless-app" if fullstack else "html-app",
    )

    with pytest.raises(_Built):
        await orchestrator._gen_verify_frontend(state)

    _stated_then_recalled(built["system"])


def test_the_cold_tech_spec_puts_the_stated_preferences_ahead_of_the_recalled(tmp_path):
    system, _ = prompts.build_tech_spec_prompt(
        _state(tmp_path, recalled_preferences=TEAL, user_preferences=ORANGE)
    )
    _stated_then_recalled(system)


@pytest.mark.parametrize(
    "build", [prompts.build_subagent_system_prompt, prompts.build_frontend_system_prompt]
)
def test_the_recalled_preferences_follow_the_rules_and_skills_and_the_language_stays_last(build):
    skills = "## Artifact skills (follow, don't display)\nskill body"
    plain = build(Path("/tmp/a"), user_language="English", project_rules=RULES, artifact_skills=skills)
    recalled = build(
        Path("/tmp/a"), user_language="English", project_rules=RULES, artifact_skills=skills,
        recalled_preferences=TEAL,
    )
    cut = recalled.index(prompts.RECALLED_PREFERENCES_HEADER)
    assert plain[:cut] == recalled[:cut]
    assert recalled.index(skills) < cut < recalled.index("## UI language")
    assert f"{TEAL}\n{prompts.RECALLED_PREFERENCES_FOOTER}" in recalled


@pytest.mark.parametrize("recalled", ["", "   \n\t\n"])
def test_nothing_recalled_leaves_the_page_writers_and_the_cold_spec_as_they_were(tmp_path, recalled):
    # The stated preferences alone reach these steps through the PRD and spec.
    path = Path("/tmp/a")
    assert prompts.recalled_preferences_section(recalled, ORANGE) == ""
    assert prompts.build_subagent_system_prompt(
        path, recalled_preferences=recalled, stated_preferences=ORANGE,
    ) == prompts.build_subagent_system_prompt(path)
    assert prompts.build_frontend_system_prompt(
        path, recalled_preferences=recalled, stated_preferences=ORANGE,
    ) == prompts.build_frontend_system_prompt(path)
    assert prompts.build_tech_spec_prompt(_state(tmp_path, recalled_preferences=recalled)) == (
        prompts.build_tech_spec_prompt(_state(tmp_path))
    )
    assert prompts.build_tech_spec_prompt(
        _state(tmp_path, recalled_preferences=recalled, user_preferences=ORANGE)
    )[0] == prompts.build_tech_spec_prompt(_state(tmp_path))[0]


async def test_generate_reads_anton_md_into_the_run_without_marking_it_read(tmp_path, monkeypatch):
    """The main agent's `anton_md_modified_since_last_read` stays its own."""
    seen = {}

    async def fake_run(state, *, entry):
        seen["state"] = state
        return {"status": "generated", "files_written": [], "internal_files": [], "trace": []}

    monkeypatch.setattr(orchestrator, "run", fake_run)
    monkeypatch.setattr(engine, "_scratchpads_context", lambda session: "")
    workspace = Workspace(tmp_path)
    workspace.initialize()
    workspace.anton_md_path.write_text(RULE + "\n", encoding="utf-8")
    session = _session()
    session._workspace = workspace
    art = tmp_path / "art"
    art.mkdir()

    await engine.generate(
        session=session, slug="a", artifact_path=art, artifact_type="html-app",
        user_request="r", agent_understanding="u", user_language="English",
    )

    state = seen["state"]
    assert state.project_rules == RULES
    assert RULE in state.pipeline_system
    assert workspace.anton_md_modified_since_last_read() is True


def test_an_untouched_template_workspace_has_no_rules(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.initialize()
    session = _session()
    session._workspace = workspace
    assert orchestrator._project_rules(session) == ""


def test_a_session_without_a_workspace_has_no_rules():
    session = _session()
    session._workspace = None
    assert orchestrator._project_rules(session) == ""


def test_a_test_double_workspace_yields_no_rules_and_no_stray_coroutine(recwarn):
    assert orchestrator._project_rules(_session()) == ""
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_an_unreadable_anton_md_builds_without_rules_and_says_so(caplog):
    class Refusing:
        def read_anton_md_user_text(self):
            raise OSError("anton.md is a link")

    session = _session()
    session._workspace = Refusing()

    with caplog.at_level(logging.WARNING):
        assert orchestrator._project_rules(session) == ""

    records = [r for r in caplog.records if r.name == orchestrator.__name__]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
