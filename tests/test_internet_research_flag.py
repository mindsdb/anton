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
