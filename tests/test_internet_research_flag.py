"""With internet research off, the agent answers from the connected data only.

`ChatSessionConfig.internet_research_enabled` False takes away both web tools,
the scratchpad's web_search() (see test_scratchpad_llm_loop.py) and every
passage of the prompt that sends the agent to the internet. It is separate
from the web tool flags: a session with those off but internet research on
(cloud_turn's posture) keeps the internet guidance and the helper."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.llm.prompts import (
    NO_INTERNET_CHAT_SYSTEM_PROMPT,
    NO_INTERNET_RESILIENCE_NUDGE,
    NO_INTERNET_SCRATCHPAD_INSTALL_NUDGE,
    RESILIENCE_NUDGE,
    SCRATCHPAD_INSTALL_NUDGE,
)
from anton.core.memory import skills as skills_mod
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.generate_artifact.orchestrator import _public_data_sources_skill
from anton.core.tools.generate_artifact.state import GenState
from anton.workspace import Workspace
from tests.conftest import make_mock_llm, run_turn

# Phrases only the internet guidance has.
_INTERNET_GUIDANCE = (
    "You DO have access to the internet",
    "FULL internet access",
    "PUBLIC DATA AND WORLD EVENTS",
    "archive.org",
    "web scraping",
)


@pytest.fixture
def builtin_skills(monkeypatch):
    """The real built-in skills, which the suite hides: `public-data-sources`
    is one of them."""
    monkeypatch.setattr(skills_mod, "_BUILTIN_SKILLS_ROOT", Path(skills_mod.__file__).parent / "builtin_skills")


async def _first_request(tmp_path: Path, **config) -> dict:
    """The kwargs of a turn's first model request, on a provider with native web tools."""
    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    llm = make_mock_llm()
    llm.planning_provider.native_web_tools.return_value = {"web_search", "web_fetch"}
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace, **config))
    try:
        await run_turn(session, "What were Amy Jones's articles about?")
    finally:
        await session.close()
    return llm.plan.call_args_list[0].kwargs


async def test_without_internet_research_a_turn_has_no_web_tools_or_internet_guidance(tmp_path, builtin_skills):
    kwargs = await _first_request(tmp_path, internet_research_enabled=False)
    system = kwargs["system"]
    tools = {tool["name"]: tool for tool in kwargs["tools"]}

    assert not kwargs.get("native_web_tools")
    assert "web_search" not in tools and "web_fetch" not in tools
    assert "web_search" not in tools["scratchpad"]["description"]
    assert not any(phrase in system for phrase in _INTERNET_GUIDANCE)
    assert "web_search" not in system
    assert "public-data-sources" not in system
    assert "Internet research is turned off for this session" in system


async def test_web_tools_off_with_internet_research_on_keeps_the_guidance_and_helper(tmp_path, builtin_skills):
    """cloud_turn turns the web tools off, but its scratchpads keep the internet."""
    kwargs = await _first_request(tmp_path, web_search_enabled=False, web_fetch_enabled=False)
    system = kwargs["system"]
    tools = {tool["name"]: tool for tool in kwargs["tools"]}

    assert not kwargs.get("native_web_tools")
    assert all(phrase in system for phrase in _INTERNET_GUIDANCE)
    assert "web_search(query)" in system
    assert "- `public-data-sources` — " in system
    assert "web_search(query)" in tools["scratchpad"]["description"]
    assert "Internet research is turned off" not in system


async def test_by_default_the_web_tools_stay_native(tmp_path):
    kwargs = await _first_request(tmp_path)
    assert kwargs["native_web_tools"] == {"web_search", "web_fetch"}


def test_the_prompt_variant_differs_only_where_it_cuts():
    assert NO_INTERNET_CHAT_SYSTEM_PROMPT.startswith("{identity_lead}")
    for field in ("{artifacts_section}", "{conversation_discipline}", "{runtime_identity_section}"):
        assert field in NO_INTERNET_CHAT_SYSTEM_PROMPT
    assert "GENERAL RULES:" in NO_INTERNET_CHAT_SYSTEM_PROMPT
    assert "web_search" not in NO_INTERNET_RESILIENCE_NUDGE + NO_INTERNET_SCRATCHPAD_INSTALL_NUDGE
    assert "archive.org" not in NO_INTERNET_RESILIENCE_NUDGE


def _nudges(*, internet_research: bool | None, tool_name: str, error: str) -> str:
    """The text two failures of `tool_name` get, through the real streak tracker."""
    session = SimpleNamespace(
        _resilience_nudge_at=2, _max_consecutive_errors=5,
        _select_resilience_nudge=ChatSession._select_resilience_nudge,
    )
    if internet_research is not None:
        session._internet_research_enabled = internet_research
    streak: dict[str, int] = {}
    nudged: set[str] = set()
    ChatSession._apply_error_tracking(session, error, tool_name, streak, nudged, ok=False)
    return ChatSession._apply_error_tracking(session, error, tool_name, streak, nudged, ok=False)


@pytest.mark.parametrize(
    ("tool_name", "error", "online", "offline"),
    [
        ("web_fetch", "Fetch returned HTTP 403", RESILIENCE_NUDGE, NO_INTERNET_RESILIENCE_NUDGE),
        ("scratchpad", "ModuleNotFoundError: No module named 'x'",
         SCRATCHPAD_INSTALL_NUDGE, NO_INTERNET_SCRATCHPAD_INSTALL_NUDGE),
    ],
    ids=["generic", "install"],
)
def test_nudges_drop_internet_advice_only_without_internet_research(tool_name, error, online, offline):
    assert _nudges(internet_research=False, tool_name=tool_name, error=error).endswith(offline)
    assert _nudges(internet_research=True, tool_name=tool_name, error=error).endswith(online)
    # A session object that predates the flag keeps today's nudge.
    assert _nudges(internet_research=None, tool_name=tool_name, error=error).endswith(online)


def test_the_prompt_variant_drops_the_public_dataset_follow_ups():
    assert "PROACTIVE FOLLOW-UP SUGGESTIONS" not in NO_INTERNET_CHAT_SYSTEM_PROMPT
    assert "DATA SOURCES:" in NO_INTERNET_CHAT_SYSTEM_PROMPT


# --- generate_artifact's own gathering loop ---------------------------------

def _gen_state(tmp_path: Path, **session) -> GenState:
    return GenState(
        session=SimpleNamespace(**session), artifact_type="html-app",
        artifact_path=tmp_path, slug="demo", brief="b",
    )


@pytest.mark.parametrize("session", [{}, {"_internet_research_enabled": True}], ids=["predates-flag", "on"])
def test_the_gathering_loop_keeps_its_web_tools_with_internet_research(tmp_path, session):
    state = _gen_state(tmp_path, **session)
    tools = {tool["name"]: tool for tool in state.pipeline_tools}

    assert state.internet_research is True
    assert {"web_search", "web_fetch"} <= set(tools)
    assert "web_search(query)" in tools["scratchpad"]["description"]


def test_the_gathering_loop_offers_no_web_tools_without_internet_research(tmp_path):
    state = _gen_state(tmp_path, _internet_research_enabled=False)
    tools = {tool["name"]: tool for tool in state.pipeline_tools}

    assert state.internet_research is False
    assert "web_search" not in tools and "web_fetch" not in tools
    assert "web_search" not in tools["scratchpad"]["description"]


def test_the_public_data_sources_catalog_is_not_read_without_internet_research():
    store = SimpleNamespace(load_for_artifacts=lambda label: pytest.fail("the catalog was read"))
    session = SimpleNamespace(_internet_research_enabled=False, _skill_store=store)
    assert _public_data_sources_skill(session) == ""


def test_the_unsupported_factory_warning_is_logged_once(monkeypatch, caplog):
    from anton.core.backends import manager as manager_mod

    def old_factory(*, name, coding_provider, coding_model, coding_api_key, coding_base_url, cells, workspace_path):
        raise AssertionError("not called")

    monkeypatch.setattr(manager_mod, "_warned_no_internet_research_kwarg", False)
    monkeypatch.setattr(manager_mod.ScratchpadManager, "probe_packages", lambda self: [])
    with caplog.at_level("WARNING", logger=manager_mod.logger.name):
        for _ in range(3):
            manager_mod.ScratchpadManager(
                runtime_factory=old_factory, coding_provider="x", coding_model="x",
                coding_api_key="", coding_base_url="", internet_research_enabled=False,
            )
    assert sum("internet_research_enabled" in r.getMessage() for r in caplog.records) == 1
