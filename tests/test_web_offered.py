"""Where the deployment turns web search off, the agent is told the web is out
of reach: no "full internet access" sections, no web_search() line in the
prompt or the scratchpad tool. Unset, the prompt is what it was."""
from __future__ import annotations

from pathlib import Path

import pytest

from anton.core.backends.wire import WEB_SEARCH_ENV
from anton.core.llm.prompts import CHAT_SYSTEM_PROMPT, NO_WEB_CHAT_SYSTEM_PROMPT
from anton.core.memory import skills as skills_mod
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.generate_artifact import prompts as builder_prompts
from anton.core.tools.generate_artifact import sub_tools as builder_sub_tools
from anton.core.tools.generate_artifact.discovery import sub_tools as discovery_sub_tools
from anton.core.tools.generate_artifact.discovery.prompts import build_pipeline_system_prompt
from anton.core.tools.generate_artifact.state import GenState
from anton.core.tools.tool_defs import NO_WEB_GENERATE_ARTIFACT_TOOLS
from anton.workspace import Workspace
from tests.conftest import make_mock_llm, run_turn

_WEB_ON_TEXTS = (
    "You DO have access to the internet",
    "You have FULL internet access",
    "try 2-3 different approaches",
    'recall_skill("public-data-sources")',
    "web scraping",
    "web_search(query)",
    "web_search are already available",
    "different HTTP headers or user-agents",
    "validate it online FIRST",
    "- `public-data-sources`",
)


@pytest.fixture(autouse=True)
def builtin_skills(monkeypatch):
    """The real built-in skills, which the suite hides: the prompt lists each
    one, and `public-data-sources` is a catalog of web endpoints."""
    monkeypatch.setattr(skills_mod, "_BUILTIN_SKILLS_ROOT", Path(skills_mod.__file__).parent / "builtin_skills")


async def _first_request(tmp_path: Path) -> tuple[str, dict[str, dict]]:
    """The system prompt and tools, by name, of a turn's first model request."""
    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    llm = make_mock_llm()
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace))
    try:
        await run_turn(session, "Search the web for the opening hours of the Louvre museum.")
    finally:
        await session.close()
    kwargs = llm.plan.call_args_list[0].kwargs
    return kwargs["system"], {tool["name"]: tool for tool in kwargs["tools"]}


async def test_with_web_search_off_the_agent_is_told_the_web_is_out_of_reach(tmp_path, monkeypatch):
    monkeypatch.setenv(WEB_SEARCH_ENV, "off")
    system, tools = await _first_request(tmp_path)

    assert "NO WEB ACCESS:" in system
    assert "say plainly that web access is off on this install" in system
    assert "a host that a skill names" in system
    assert "even one the user names" in system
    assert "so none of them turns to another site or API" in system
    assert "it cannot be checked online" in system
    for text in _WEB_ON_TEXTS:
        assert text not in system, text
    assert "web_search" not in tools["scratchpad"]["description"]
    assert "sample(var)" in tools["scratchpad"]["description"]
    assert "(it can use the scratchpad), asks the user" in tools["generate_artifact"]["description"]


async def test_without_the_switch_the_prompt_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv(WEB_SEARCH_ENV, raising=False)
    system, tools = await _first_request(tmp_path)

    assert "NO WEB ACCESS:" not in system
    for text in _WEB_ON_TEXTS:
        assert text in system, text
    assert "web_search(query)" in tools["scratchpad"]["description"]
    assert "agentic_loop, web_search, sample" in tools["scratchpad"]["description"]
    assert "web search and web fetch" in tools["generate_artifact"]["description"]


def test_the_no_web_prompt_keeps_every_placeholder():
    assert NO_WEB_CHAT_SYSTEM_PROMPT.count("{") == CHAT_SYSTEM_PROMPT.count("{")


def _builder_texts() -> list[str]:
    """Everything the artifact builder's data steps are told."""
    state = GenState(
        session=object(), artifact_type="html-app", artifact_path=Path("/tmp/a"), slug="a",
        brief="A live BTC price dashboard", is_fullstack=False,
    )
    return [
        build_pipeline_system_prompt(state),
        builder_prompts.build_required_data_prompt(state)[0],
        builder_prompts.build_can_fetch_prompt(state, "- BTC price")[0],
        builder_prompts.build_fetch_data_system_prompt(Path("/tmp/a")),
    ]


def test_with_web_search_off_the_artifact_builder_is_told_the_web_is_out_of_reach(monkeypatch):
    monkeypatch.setenv(WEB_SEARCH_ENV, "off")
    for text in _builder_texts():
        assert builder_prompts.NO_WEB_DATA_NOTE in text
    for schema in (builder_sub_tools._scratchpad_schema(), discovery_sub_tools._scratchpad_schema()):
        assert "web_search" not in schema["description"]


def test_without_the_switch_the_artifact_builder_is_unchanged(monkeypatch):
    monkeypatch.delenv(WEB_SEARCH_ENV, raising=False)
    for text in _builder_texts():
        assert builder_prompts.NO_WEB_DATA_NOTE not in text
    for schema in (builder_sub_tools._scratchpad_schema(), discovery_sub_tools._scratchpad_schema()):
        assert "web_search(query)" in schema["description"]


def test_no_generate_artifact_variant_promises_the_web():
    assert len(NO_WEB_GENERATE_ARTIFACT_TOOLS) == 4
    for tool in NO_WEB_GENERATE_ARTIFACT_TOOLS.values():
        assert "web search" not in tool.description
        assert "(it can use the scratchpad), asks the user" in tool.description
