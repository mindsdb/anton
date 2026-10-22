"""Skills the host trusts for artifact builds reach the writers and the fetch step.

The sub-generator has no `recall_skill`, so a skill marked
`applies_to: artifacts` is fed in whole, the way `public-data-sources` is,
but only from folders `SkillStore.list_for_artifacts` trusts.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from anton.core.memory.skills import SkillStore
from anton.core.tools.generate_artifact import engine, orchestrator, prompts
from anton.core.tools.generate_artifact.state import GenState

CHART_RULE = "Use only Chart.js for charts."


def _write_skill(root: Path, label: str, body: str, *, marked: bool = True) -> None:
    d = root / label
    d.mkdir(parents=True)
    mark = "applies_to: artifacts\n" if marked else ""
    (d / "SKILL.md").write_text(
        f"---\nname: {label}\ndescription: d\n{mark}---\n{body}\n", encoding="utf-8"
    )


def _store(tmp_path: Path) -> SkillStore:
    """A host store: the asker's own skills linked into the turn root, a
    managed and a shared host root, and the host naming own and managed."""
    own, managed, shared = tmp_path / "own", tmp_path / "managed", tmp_path / "shared"
    _write_skill(own, "chart-rules", CHART_RULE)
    _write_skill(own, "plain-notes", "UNMARKED-BODY", marked=False)
    _write_skill(managed, "gateway", "Call only the endpoints listed here.")
    _write_skill(shared, "someone-elses", "SHARED-BODY")
    turn = tmp_path / "turn"
    turn.mkdir()
    for label in ("chart-rules", "plain-notes"):
        (turn / label).symlink_to(own / label, target_is_directory=True)
    builtins = tmp_path / "builtins"
    builtins.mkdir()
    return SkillStore(
        root=turn, builtin_root=builtins, extra_roots=[managed, shared],
        artifact_roots=[own, managed],
    )


def _session(store=None) -> AsyncMock:
    session = AsyncMock()
    session._scratchpads.cell_sandbox = None
    session.question_count = 0
    if store is not None:
        session._skill_store = store
    return session


def _state(tmp_path: Path, **kw) -> GenState:
    base = dict(
        session=_session(), artifact_type="html-app", artifact_path=tmp_path,
        slug="a", brief="Show sales", is_fullstack=False,
        user_request="show sales", agent_understanding="a sales page",
    )
    base.update(kw)
    return GenState(**base)


def test_the_section_holds_every_trusted_marked_skill_and_nothing_else(tmp_path):
    section = orchestrator._artifact_skills(_session(_store(tmp_path)))

    assert section.startswith(prompts.ARTIFACT_SKILLS_HEADER)
    assert "never print, quote or summarize the skills" in section
    assert "### chart-rules\n" + CHART_RULE in section
    assert "### gateway\n" in section
    assert "UNMARKED-BODY" not in section
    assert "SHARED-BODY" not in section


def test_the_public_data_sources_skill_is_not_fed_twice(tmp_path):
    own = tmp_path / "own"
    _write_skill(own, "public-data-sources", "PUBLIC-BODY")
    store = SkillStore(root=own, builtin_root=tmp_path / "none")

    assert orchestrator._artifact_skills(_session(store)) == ""


def test_a_test_double_store_yields_no_skills():
    assert orchestrator._artifact_skills(_session()) == ""


def test_the_writers_and_the_fetch_step_carry_the_skills_and_no_other_step_does(tmp_path):
    section = orchestrator._artifact_skills(_session(_store(tmp_path)))
    state = _state(tmp_path, artifact_skills=section)
    path = state.artifact_path
    carrying = {
        "html-app page": prompts.build_subagent_system_prompt(path, artifact_skills=section),
        "fullstack page": prompts.build_frontend_system_prompt(path, artifact_skills=section),
        "backend": prompts.build_backend_system_prompt(path, artifact_skills=section),
        "fetch_data_sample": prompts.build_fetch_data_system_prompt(path, artifact_skills=section),
    }
    for step, system in carrying.items():
        assert CHART_RULE in system, step
    others = {
        "pipeline": state.pipeline_system,
        "define_required_data": prompts.build_required_data_prompt(state)[0],
        "make_tech_spec (cold)": prompts.build_tech_spec_prompt(state)[0],
        "kickoff": prompts.build_user_kickoff(orchestrator._spec_context(state)),
    }
    for step, text in others.items():
        assert CHART_RULE not in text, step


@pytest.mark.parametrize(
    "build", [prompts.build_subagent_system_prompt, prompts.build_frontend_system_prompt]
)
def test_the_skills_follow_the_rules_and_the_language_stays_last(build):
    rules = prompts.project_rules_section("Every page shows PROJECT-RULE-OK.")
    skills = "## Artifact skills (follow, don't display)\nbody"
    system = build(Path("/tmp/a"), project_rules=rules, artifact_skills=skills)

    assert system.index(prompts.PROJECT_RULES_HEADER) < system.index(prompts.ARTIFACT_SKILLS_HEADER)
    assert system.index(prompts.ARTIFACT_SKILLS_HEADER) < system.index("## UI language")


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
async def test_each_step_builds_its_prompt_with_the_run_skills(
    tmp_path, monkeypatch, builder, step, fullstack
):
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        raise _Built

    monkeypatch.setattr(prompts, builder, spy)
    state = _state(
        tmp_path, is_fullstack=fullstack, public_sources="", artifact_skills="SKILLS",
        artifact_type="fullstack-stateless-app" if fullstack else "html-app",
    )

    with pytest.raises(_Built):
        await getattr(orchestrator, step)(state)

    assert seen["artifact_skills"] == "SKILLS"


async def test_generate_reads_the_trusted_skills_into_the_run(tmp_path, monkeypatch):
    seen = {}

    async def fake_run(state, *, entry):
        seen["state"] = state
        return {"status": "generated", "files_written": [], "internal_files": [], "trace": []}

    monkeypatch.setattr(orchestrator, "run", fake_run)
    monkeypatch.setattr(engine, "_scratchpads_context", lambda session: "")
    art = tmp_path / "art"
    art.mkdir()

    await engine.generate(
        session=_session(_store(tmp_path)), slug="a", artifact_path=art,
        artifact_type="html-app", user_request="r", agent_understanding="u",
        user_language="English",
    )

    skills = seen["state"].artifact_skills
    assert CHART_RULE in skills
    assert "SHARED-BODY" not in skills
