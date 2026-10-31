"""Earth geography guidance reaches the prompts used to plan, build and repair."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from anton.core.artifacts.geography_guidance import GEOGRAPHY_GUIDANCE
from anton.core.llm.prompt_builder import (
    SESSION_CONTEXT_MARKER,
    ChatSystemPromptBuilder,
    SystemPromptContext,
)
from anton.core.tools.generate_artifact import engine, orchestrator, prompts
from anton.core.tools.generate_artifact.discovery import sub_tools
from anton.core.tools.generate_artifact.discovery.prompts import step_message
from anton.core.tools.generate_artifact.state import GenState


@pytest.mark.parametrize("single_call", [False, True])
@pytest.mark.parametrize("proactive", [False, True])
@pytest.mark.parametrize("direct_reports", [False, True])
@pytest.mark.parametrize("fullstack_offered", [False, True])
def test_main_agent_always_receives_geography_without_skill_recall(
    monkeypatch, single_call, proactive, direct_reports, fullstack_offered,
):
    monkeypatch.setenv("ANTON_STATIC_SINGLE_CALL", "on" if single_call else "off")
    tool = SimpleNamespace(
        name="generate_artifact", prompt=None,
        input_schema={"properties": {"name": {}}},
    )
    system = ChatSystemPromptBuilder().build(
        conversation_started="2026-10-30", system_prompt_context=SystemPromptContext(),
        proactive_dashboards=proactive, direct_reports=direct_reports,
        fullstack_offered=fullstack_offered, output_dir="/tmp/artifacts",
        tool_defs=[tool], skill_store=None, project_context="A project",
    )

    assert system.count(GEOGRAPHY_GUIDANCE) == 1
    assert GEOGRAPHY_GUIDANCE in system.split(SESSION_CONTEXT_MARKER)[0]
    assert "Procedural memory" not in system
    # Exercise the single-call switch, rather than accidentally testing only
    # the ordinary prompt with an incompatible tool schema.
    assert ("This one\noperation registers the folder" in system) is single_call


@pytest.mark.parametrize("builder", [
    prompts.build_subagent_system_prompt,
    prompts.build_frontend_system_prompt,
])
def test_standard_page_builders_receive_the_shared_geography_recipe(builder, tmp_path):
    system = builder(tmp_path)
    assert system.count(GEOGRAPHY_GUIDANCE) == 1
    assert "Never write GeoJSON into the page by hand" not in system


def _state(tmp_path, **overrides):
    values = dict(
        session=AsyncMock(question_count=0), artifact_type="html-app",
        artifact_path=tmp_path, slug="geography", user_request="Show an Earth globe",
        agent_understanding="A globe showing the supplied places",
        brief="A globe showing the supplied places", prd="# Globe requirements",
    )
    values.update(overrides)
    return GenState(**values)


@pytest.mark.parametrize("act_first", [False, True])
def test_discovery_retains_geography_but_keeps_implementation_out_of_the_brief(tmp_path, act_first):
    state = _state(tmp_path, act_first=act_first)
    system = state.pipeline_system
    assert system.count(GEOGRAPHY_GUIDANCE) == 1
    assert "user-facing brief" in system
    brief_instruction = step_message(sub_tools.STEP_DRAFT_BRIEF, state)
    assert "library/framework names" in brief_instruction
    assert "world-atlas@" not in brief_instruction
    state.final_artifact_type = "fullstack-stateless-app"
    assert state.pipeline_system == system


@pytest.mark.parametrize("hot", [False, True])
async def test_actual_spec_node_passes_geography_to_hot_and_cold_calls(monkeypatch, tmp_path, hot):
    plan = AsyncMock(return_value=("# Implementation notes\nUse the supplied places.", None))
    monkeypatch.setattr(engine, "_plan_whole_document", plan)
    state = _state(tmp_path, messages=[{"role": "user", "content": "Gathered places"}] if hot else [])

    assert await orchestrator._write_tech_spec(state) is None

    call = plan.call_args.kwargs
    assert call["user"].count(GEOGRAPHY_GUIDANCE) == 1
    if hot:
        assert call["messages"] is state.messages
        assert call["system"] == state.pipeline_system
        assert GEOGRAPHY_GUIDANCE in call["system"]
        assert call["user"] == prompts.build_tech_spec_instruction(state)
    else:
        assert call["messages"] is None
        assert "A globe showing the supplied places" in call["user"]


def test_geography_policy_keeps_authentic_data_schematics_and_repairs_supported():
    flat = " ".join(GEOGRAPHY_GUIDANCE.split())
    for rule in (
        "Earth geography only",
        "Never invent or hand-transcribe coastlines",
        "SVG itself is allowed",
        "Authentic user-supplied GeoJSON or other supplied geographic data is allowed",
        "non-geographic map",
        "explicitly requested schematic",
        "Load the atlas in the page's JavaScript, not in Python or a scratchpad",
        "preserve its real geography, requested views and interactions",
        "replacing it with hand-drawn shapes or a blank sphere",
        "during repairs without recalling a skill",
        "Await or return every promise the setup starts",
    ):
        assert rule in flat
    for package in (
        "d3@7.9.0", "topojson-client@3.1.0", "world-atlas@2.0.2",
        "echarts@5.5.1", "globe.gl@2.34.2",
    ):
        assert f"https://cdn.jsdelivr.net/npm/{package}/" in GEOGRAPHY_GUIDANCE


def test_manual_dashboard_skill_keeps_the_same_geography_and_repair_scope():
    skill = (Path(__file__).resolve().parents[1]
             / "anton/core/memory/builtin_skills/build-html-dashboard/SKILL.md").read_text()
    assert "authentic user-supplied GeoJSON" in skill
    assert "explicitly requested schematics" in skill
    assert "Load the atlas in the page's JavaScript, not in Python or a scratchpad" in skill
    assert "During repairs, preserve the real geography" in skill
    assert "Not needed to repair a page generate_artifact returned as browser_check_failed" in skill
