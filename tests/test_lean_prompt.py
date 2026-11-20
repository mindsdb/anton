"""The `lean_prompt` setting: a short system prompt and short tool descriptions."""

import pytest

from anton.config.settings import AntonSettings
from anton.core.llm import prompts
from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext
from anton.core.tools.tool_defs import lean_scratchpad_description

_ARTIFACT_TOOLS = ("create_artifact", "generate_artifact", "edit_artifact", "launch_backend",
                   "list_artifacts", "update_artifact", "publish_or_preview")


def _settings(**over) -> AntonSettings:
    return AntonSettings(_env_file=None, **over)


def _build(**over) -> str:
    kwargs = dict(conversation_started="Tuesday", system_prompt_context=SystemPromptContext(),
                  proactive_dashboards=True, output_dir="/tmp/out")
    return ChatSystemPromptBuilder().build(**{**kwargs, **over})


def test_the_setting_is_off_unless_the_env_turns_it_on(monkeypatch):
    monkeypatch.delenv("ANTON_LEAN_PROMPT", raising=False)
    assert AntonSettings(_env_file=None).lean_prompt is False
    monkeypatch.setenv("ANTON_LEAN_PROMPT", "true")
    assert AntonSettings(_env_file=None).lean_prompt is True


@pytest.mark.asyncio
async def test_a_lean_session_keeps_every_tool_and_schema_with_shorter_texts(make_session):
    full = make_session(settings=_settings())
    lean = make_session(settings=_settings(lean_prompt=True))
    full_tools = {t["name"]: t for t in full._build_tools()}
    lean_tools = {t["name"]: t for t in lean._build_tools()}

    assert set(lean_tools) == set(full_tools)
    assert all(lean_tools[n]["input_schema"] == full_tools[n]["input_schema"] for n in full_tools)
    assert sum(len(t["description"]) for t in lean_tools.values()) < sum(
        len(t["description"]) for t in full_tools.values()) / 2
    assert len(await lean._build_system_prompt("hi")) < len(await full._build_system_prompt("hi"))


def test_a_session_without_settings_keeps_the_full_texts(make_session):
    session = make_session()
    assert not session._lean_prompt_enabled()
    assert session._build_tools() == session.tool_registry.dump()


@pytest.mark.parametrize("over", [{}, {"direct_reports": False}, {"artifact_editor": "on"}])
def test_short_texts_name_only_tools_the_session_has(make_session, tmp_path, over):
    from anton.workspace import Workspace

    session = make_session(workspace=Workspace(tmp_path), settings=_settings(lean_prompt=True, **over))
    tools = session._build_tools()
    names = {t["name"] for t in tools}
    assert {"create_artifact", "generate_artifact"} <= names
    for tool in tools:
        named = {other for other in _ARTIFACT_TOOLS if other in tool["description"]}
        assert named <= names, (tool["name"], named - names)


@pytest.mark.parametrize("internet_research", [True, False])
@pytest.mark.parametrize("classify", [True, False])
def test_the_lean_scratchpad_offers_only_the_helpers_the_session_has(internet_research, classify):
    text = lean_scratchpad_description(internet_research=internet_research, classify=classify)
    assert ("web_search" in text) is internet_research
    assert ("classify" in text) is classify


def test_the_lean_prompt_keeps_the_artifact_sections_and_the_no_internet_block():
    assert prompts.ARTIFACTS_PROMPT in _build(lean=True)
    block = prompts.NO_INTERNET_DATA_SOURCES.strip()
    assert block not in _build(lean=True)
    assert block in _build(lean=True, internet_research=False)


def test_the_lean_prompt_follows_the_ask_first_setting():
    assert _build(lean=True, act_first=True) != _build(lean=True, act_first=False)
