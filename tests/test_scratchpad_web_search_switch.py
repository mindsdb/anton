"""The scratchpad's web_search() is offered only when `scratchpad_web_search` is
on: zero data retention turns it off. Then the prompt and the scratchpad tool
leave it out and the pad's helper raises (see test_scratchpad_llm_loop.py)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from anton.core.llm.prompts import NO_INTERNET_SCRATCHPAD_INSTALL_NUDGE, SCRATCHPAD_INSTALL_NUDGE
from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.settings import CoreSettings
from anton.core.tools.generate_artifact.state import GenState
from anton.core.tools.tool_defs import scratchpad_tool
from anton.workspace import Workspace
from tests.conftest import make_mock_llm, run_turn


@pytest.mark.parametrize(("settings", "offered"), [({}, True), ({"scratchpad_web_search": "off"}, False)],
                         ids=["default", "off"])
async def test_web_search_is_offered_only_when_on(tmp_path, monkeypatch, settings, offered):
    monkeypatch.delenv("ANTON_SCRATCHPAD_WEB_SEARCH", raising=False)
    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    llm = make_mock_llm()
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace, settings=CoreSettings(**settings)))
    try:
        await run_turn(session, "What's new today?")
    finally:
        await session.close()
    kwargs = llm.plan.call_args_list[0].kwargs
    tools = {tool["name"]: tool for tool in kwargs["tools"]}

    assert session._scratchpad_web_search_enabled is offered
    assert ("web_search(query) answers" in kwargs["system"]) is offered
    assert ("web_search(query) routes" in tools["scratchpad"]["description"]) is offered
    assert session._scratchpads._scratchpad_web_search_enabled is offered
    # Only the helper goes: the internet guidance stays.
    assert "INTERNET & LIVE INFORMATION:" in kwargs["system"]


@pytest.mark.parametrize("classify", [True, False])
def test_the_prompt_without_web_search_cuts_only_its_item(classify):
    def build(web_search):
        return ChatSystemPromptBuilder().build(
            conversation_started="x", output_dir="/o", proactive_dashboards=False,
            system_prompt_context=SystemPromptContext(prefix="", runtime_context="", suffix=""),
            classify=classify, web_search=web_search,
        )

    on, off = build(True), build(False)
    assert "web_search(query) answers" in on
    assert "web_search(query) answers" not in off
    lines = on.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("  - web_search("))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("- "))
    assert lines[:start] + lines[end:] == off.splitlines()


@pytest.mark.parametrize("classify", [True, False])
def test_the_prompt_with_internet_research_and_web_search_off_is_the_no_internet_one(classify):
    """Internet research off already drops web_search(); the ZDR switch on top changes nothing."""
    def build(**flags):
        return ChatSystemPromptBuilder().build(
            conversation_started="x", output_dir="/o", proactive_dashboards=False,
            system_prompt_context=SystemPromptContext(prefix="", runtime_context="", suffix=""),
            classify=classify, **flags,
        )

    both_off = build(internet_research=False, web_search=False)
    assert both_off == build(internet_research=False)
    assert "web_search(" not in both_off
    assert "INTERNET & LIVE INFORMATION:" not in both_off


@pytest.mark.parametrize("classify", [True, False])
def test_the_scratchpad_tool_without_web_search(classify):
    on = scratchpad_tool(classify=classify, web_search=True).description
    off = scratchpad_tool(classify=classify, web_search=False).description
    assert "web_search" in on
    assert "web_search" not in off
    assert ("classify(texts, categories)" in off) is classify


@pytest.mark.parametrize(("session", "offered"), [({}, True), ({"_scratchpad_web_search_enabled": False}, False)],
                         ids=["predates-flag", "off"])
def test_the_gathering_loop_follows_the_session(tmp_path, session, offered):
    state = GenState(session=SimpleNamespace(**session), artifact_type="html-app",
                     artifact_path=tmp_path, slug="demo", brief="b")
    tools = {tool["name"]: tool for tool in state.pipeline_tools}
    assert ("web_search(query) routes" in tools["scratchpad"]["description"]) is offered


def test_the_manager_passes_the_switch_and_refuses_a_factory_without_it(monkeypatch):
    """Off is passed on; a factory that cannot take it would keep web_search(), so it is refused."""
    from anton.core.backends import manager as manager_mod

    seen = {}

    def factory(*, name, coding_provider, coding_model, coding_api_key, coding_base_url, cells,
                workspace_path, scratchpad_web_search_enabled=True):
        seen["scratchpad_web_search_enabled"] = scratchpad_web_search_enabled
        raise RuntimeError("stop")

    def old_factory(*, name, coding_provider, coding_model, coding_api_key, coding_base_url, cells, workspace_path):
        raise RuntimeError("stop")

    def swallowing_factory(**kwargs):
        raise RuntimeError("stop")

    monkeypatch.setattr(manager_mod.ScratchpadManager, "probe_packages", lambda self: [])
    args = dict(coding_provider="x", coding_model="x", coding_api_key="", coding_base_url="")
    manager = manager_mod.ScratchpadManager(runtime_factory=factory, scratchpad_web_search_enabled=False, **args)
    with pytest.raises(RuntimeError):
        asyncio.run(manager.get_or_create("p"))
    assert seen == {"scratchpad_web_search_enabled": False}
    for f in (old_factory, swallowing_factory):
        with pytest.raises(TypeError, match="scratchpad_web_search_enabled"):
            manager_mod.ScratchpadManager(runtime_factory=f, scratchpad_web_search_enabled=False, **args)
        # With web search on, an older factory still works.
        manager_mod.ScratchpadManager(runtime_factory=f, **args)


def test_the_install_nudge_leaves_out_web_search_when_off():
    """A missing-module nudge must not list web_search() as a global the pad lacks."""
    def install_nudge(**session):
        streak: dict[str, int] = {}
        nudged: set[str] = set()
        pad = SimpleNamespace(
            _resilience_nudge_at=2, _max_consecutive_errors=5,
            _select_resilience_nudge=ChatSession._select_resilience_nudge, **session,
        )
        error = "ModuleNotFoundError: No module named 'x'"
        ChatSession._apply_error_tracking(pad, error, "scratchpad", streak, nudged, ok=False)
        return ChatSession._apply_error_tracking(pad, error, "scratchpad", streak, nudged, ok=False)

    assert install_nudge().endswith(SCRATCHPAD_INSTALL_NUDGE)
    off = install_nudge(_scratchpad_web_search_enabled=False)
    assert off.endswith(NO_INTERNET_SCRATCHPAD_INSTALL_NUDGE)
    assert "web_search" not in off


def test_the_pad_env_is_not_the_settings_env(monkeypatch):
    """A nested Anton inherits the pad's env, so it must neither read it as the setting nor fail to load."""
    from anton.core.backends.wire import WEB_SEARCH_ENV

    monkeypatch.delenv("ANTON_SCRATCHPAD_WEB_SEARCH", raising=False)
    assert WEB_SEARCH_ENV.startswith("ANTON_SCRATCHPAD_")
    for value in ("off", "1"):
        monkeypatch.setenv(WEB_SEARCH_ENV, value)
        assert CoreSettings().scratchpad_web_search == "on"
