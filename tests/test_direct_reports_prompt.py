"""The DIRECT REPORTS guidance stays consistent with the code it documents."""
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.artifacts import report_tools
from anton.core.llm import prompts as llm_prompts
from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext
from anton.core.llm.prompts import ARTIFACTS_PROMPT
from anton.core.tools import tool_defs
from anton.core.tools.tool_defs import CREATE_ARTIFACT_TOOL, GENERATE_ARTIFACT_TOOL


def _direct_reports_block() -> str:
    return ARTIFACTS_PROMPT.split("DIRECT REPORTS:", 1)[1].split("WORKFLOW:", 1)[0]


def test_every_documented_helper_exists_with_the_documented_parameters():
    block = _direct_reports_block()
    assert "from anton.core.artifacts import report_tools as rt" in block
    documented = dict(re.findall(r"`rt\.(\w+)\(([^`]*)\)`", block.replace("\n", " ")))
    assert documented
    for name, args in documented.items():
        assert name in report_tools.__all__, name
        params = inspect.signature(getattr(report_tools, name)).parameters
        for arg in re.findall(r"(\w+)=", args):
            assert arg in params, f"rt.{name} has no parameter {arg}"


def test_routing_is_stated_once_with_the_criterion_first():
    """Each routing text names the work it is for; no 'Exception:' afterthoughts."""
    for text in (ARTIFACTS_PROMPT, CREATE_ARTIFACT_TOOL.description, GENERATE_ARTIFACT_TOOL.description,
                 GENERATE_ARTIFACT_TOOL.prompt):
        assert "fully specified" in text
        assert "Exception:" not in text


def test_the_block_is_short():
    assert len(_direct_reports_block()) < 2800


_NO_DIRECT_TEXTS = (
    llm_prompts.NO_DIRECT_REPORTS_ARTIFACTS_PROMPT,
    llm_prompts.NO_DIRECT_REPORTS_STATIC_ARTIFACTS_PROMPT,
    llm_prompts.NO_DIRECT_REPORTS_VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT,
    llm_prompts.NO_DIRECT_REPORTS_VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT,
)
_NO_DIRECT_TOOLS = (
    tool_defs.CREATE_ARTIFACT_TOOL_NO_DIRECT_REPORTS,
    tool_defs.STATIC_CREATE_ARTIFACT_TOOL_NO_DIRECT_REPORTS,
    tool_defs.GENERATE_ARTIFACT_TOOL_NO_DIRECT_REPORTS,
    tool_defs.STATIC_GENERATE_ARTIFACT_TOOL_NO_DIRECT_REPORTS,
)


@pytest.mark.parametrize("text", _NO_DIRECT_TEXTS)
def test_variants_without_direct_reports_drop_the_path(text):
    assert "DIRECT REPORTS" not in text
    assert "report_tools" not in text
    assert "fully specified" not in text


@pytest.mark.parametrize("tool", _NO_DIRECT_TOOLS)
def test_tools_without_direct_reports_drop_the_path(tool):
    for text in (tool.description, tool.prompt or ""):
        assert "DIRECT REPORTS" not in text and "fully specified" not in text


@pytest.mark.parametrize("tool", [
    tool_defs.GENERATE_ARTIFACT_TOOL_NO_DIRECT_REPORTS,
    tool_defs.STATIC_GENERATE_ARTIFACT_TOOL_NO_DIRECT_REPORTS,
])
def test_generate_without_direct_reports_takes_every_web_artifact(tool):
    for text in (tool.description, tool.prompt):
        assert "still need agreeing" not in text
    assert "web artifacts, INSTEAD of writing their files yourself" in tool.prompt


def test_every_html_app_goes_to_the_generator_without_direct_reports():
    assert "Every `html-app`" in llm_prompts.NO_DIRECT_REPORTS_ARTIFACTS_PROMPT
    assert "Every `html-app` goes to" in llm_prompts.NO_DIRECT_REPORTS_STATIC_ARTIFACTS_PROMPT


@pytest.mark.parametrize("proactive", [True, False])
def test_builder_follows_the_flag(proactive):
    def build(direct_reports):
        return ChatSystemPromptBuilder().build(
            conversation_started="2026-10-28T00:00:00+00:00",
            system_prompt_context=SystemPromptContext(runtime_context="test runtime"),
            proactive_dashboards=proactive, output_dir="", direct_reports=direct_reports,
        )

    assert "DIRECT REPORTS" in build(True)
    assert "DIRECT REPORTS" not in build(False)


def test_skill_list_drops_the_direct_reports_note_when_off():
    """The build-html-dashboard skill's description points at DIRECT REPORTS;
    with the path off, the skill list must not send the agent there."""
    import anton
    from anton.core.llm.prompt_builder import DIRECT_REPORTS_SKILL_NOTE

    skill_md = (
        Path(anton.__file__).parent / "core/memory/builtin_skills/build-html-dashboard/SKILL.md"
    ).read_text(encoding="utf-8")
    # The note is matched in the skill's real text, so a reworded skill fails here.
    assert DIRECT_REPORTS_SKILL_NOTE.search(" ".join(skill_md.split()))

    description = (
        "ONLY for writing dashboard HTML BY HAND. NOT needed to change the content of a report "
        "or add sections (report_tools does that; see DIRECT REPORTS). Recall this before rewriting."
    )
    store = SimpleNamespace(list_summaries=lambda: [{"label": "build-html-dashboard", "description": description}])

    def build(direct_reports):
        return ChatSystemPromptBuilder().build(
            conversation_started="2026-10-28T00:00:00+00:00",
            system_prompt_context=SystemPromptContext(runtime_context="test runtime"),
            proactive_dashboards=True, output_dir="", skill_store=store, direct_reports=direct_reports,
        )

    assert "see DIRECT REPORTS" in build(True)
    off = build(False)
    assert "DIRECT REPORTS" not in off and "report_tools" not in off
    assert "Recall this before rewriting." in off


@pytest.mark.parametrize("settings, expected", [
    (None, True),
    (SimpleNamespace(), True),
    (SimpleNamespace(direct_reports=False), False),
])
def test_session_reads_the_flag_from_its_settings(settings, expected):
    from anton.core.session import ChatSession

    fake = SimpleNamespace(_settings=settings)
    assert ChatSession._direct_reports_enabled(fake) is expected
