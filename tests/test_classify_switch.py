"""The scratchpad's classify() and classify_yes_no() are offered only when
`classify_jev` is on and the coding model runs on the MindsHub gateway, which
serves Jev. Otherwise the prompt and the scratchpad tool leave them out and
the pad's helpers raise (see test_scratchpad_llm_loop.py)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext
from anton.core.llm.provider import ProviderConnectionInfo
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.settings import CoreSettings
from anton.core.tools.generate_artifact.state import GenState
from anton.core.tools.tool_defs import scratchpad_tool
from anton.workspace import Workspace
from tests.conftest import make_mock_llm, run_turn

_MINDSHUB = ProviderConnectionInfo(provider="openai", api_key="k", base_url="https://api.mindshub.ai/v1")
_ELSEWHERE = ProviderConnectionInfo(provider="openai", api_key="k", base_url="https://api.openai.com/v1")


async def _first_request(tmp_path, connection, **settings) -> tuple[dict, ChatSession]:
    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    llm = make_mock_llm()
    llm.coding_provider.export_connection_info.return_value = connection
    config = ChatSessionConfig(llm_client=llm, workspace=workspace, settings=CoreSettings(**settings))
    session = ChatSession(config)
    try:
        await run_turn(session, "Label these reviews")
    finally:
        await session.close()
    return llm.plan.call_args_list[0].kwargs, session


@pytest.mark.parametrize(("connection", "settings", "offered"), [
    (_MINDSHUB, {}, True),
    (_MINDSHUB, {"classify_jev": "off"}, False),
    (_ELSEWHERE, {}, False),
], ids=["mindshub", "turned-off", "other-gateway"])
async def test_classify_is_offered_only_on_mindshub_when_on(tmp_path, monkeypatch, connection, settings, offered):
    monkeypatch.delenv("ANTON_CLASSIFY_JEV", raising=False)
    kwargs, session = await _first_request(tmp_path, connection, **settings)
    tools = {tool["name"]: tool for tool in kwargs["tools"]}

    assert session._classify_enabled is offered
    assert ("classify(texts, categories)" in kwargs["system"]) is offered
    assert ("classify(texts, categories)" in tools["scratchpad"]["description"]) is offered
    assert session._scratchpads._classify_enabled is offered


@pytest.mark.parametrize("internet_research", [True, False])
def test_the_prompt_without_classify_names_no_classify_helper(internet_research):
    def build(classify):
        return ChatSystemPromptBuilder().build(
            conversation_started="x", output_dir="/o", proactive_dashboards=False,
            system_prompt_context=SystemPromptContext(prefix="", runtime_context="", suffix=""),
            internet_research=internet_research, classify=classify,
        )

    on, off = build(True), build(False)
    assert "classify(texts, categories)" in on and "classify_yes_no" in on
    assert "classify" not in off.replace("classified", "")
    # Everything else is kept: the only removed lines are the classify ones.
    removed = set(on.splitlines()) - set(off.splitlines())
    assert all("classify" in line or "categor" in line or "label" in line for line in removed)


@pytest.mark.parametrize("internet_research", [True, False])
def test_the_scratchpad_tool_without_classify(internet_research):
    on = scratchpad_tool(internet_research=internet_research, classify=True).description
    off = scratchpad_tool(internet_research=internet_research, classify=False).description
    assert "classify(texts, categories)" in on and "classify_yes_no" in on
    assert "classify" not in off
    assert ("web_search(query)" in off) is internet_research


@pytest.mark.parametrize(("session", "offered"), [({}, True), ({"_classify_enabled": False}, False)],
                         ids=["predates-flag", "off"])
def test_the_gathering_loop_follows_the_session(tmp_path, session, offered):
    state = GenState(session=SimpleNamespace(**session), artifact_type="html-app",
                     artifact_path=tmp_path, slug="demo", brief="b")
    tools = {tool["name"]: tool for tool in state.pipeline_tools}
    assert ("classify(texts, categories)" in tools["scratchpad"]["description"]) is offered


def test_the_manager_passes_the_switch_only_when_off(monkeypatch):
    from anton.core.backends import manager as manager_mod

    seen = {}

    def factory(*, name, coding_provider, coding_model, coding_api_key, coding_base_url, cells,
                workspace_path, classify_enabled=True):
        seen["classify_enabled"] = classify_enabled
        raise RuntimeError("stop")

    def old_factory(*, name, coding_provider, coding_model, coding_api_key, coding_base_url, cells, workspace_path):
        seen["old"] = True
        raise RuntimeError("stop")

    monkeypatch.setattr(manager_mod.ScratchpadManager, "probe_packages", lambda self: [])
    for f in (factory, old_factory):
        manager = manager_mod.ScratchpadManager(
            runtime_factory=f, coding_provider="x", coding_model="x",
            coding_api_key="", coding_base_url="", classify_enabled=False,
        )
        with pytest.raises(RuntimeError):
            import asyncio
            asyncio.run(manager.get_or_create("p"))
    assert seen == {"classify_enabled": False, "old": True}
