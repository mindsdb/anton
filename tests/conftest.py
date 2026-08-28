from __future__ import annotations

import os

from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.llm.provider import LLMResponse, ProviderConnectionInfo, ToolCall, Usage

# Tests that drive full turns reach the turn-cost analytics sink (ENG-1288):
# _emit_turn_cost falls back to a fresh AntonSettings(), whose default sinks are
# the real collector and the real PostHog project. Kill analytics for the whole
# suite so no test run ever fires a real event. (CI is also dropped by
# send_event's own CI detection; this covers local dev runs.)
#
# Assignment, not setdefault (ENG-2055). setdefault writes only when the key is
# absent, so a developer with ANTON_ANALYTICS_ENABLED exported — which is exactly
# what someone working on an analytics ticket sets — cancelled this line without
# any warning, and the suite shipped real events to production for four months.
# Measured 2026-08-28 against a local capture server: one run with the variable
# exported emitted 254 events — tool_completed 147, ds_connect_* 89, ask_user_* 16,
# turn_completed 2.
#
# ENG-1692's script-traffic guard does not cover this. It lives inside
# _emit_turn_cost alone, so three of those four families have no guard at all, and
# it only takes effect once a developer updates their installed build. This line
# is the one that works on every build, immediately.
#
# The escape hatch is removed deliberately. Nothing loses coverage: the tests that
# exercise the analytics layer build their own settings objects and never read the
# environment (test_analytics.py::_Settings and ::S, test_tool_completed.py::
# _PosthogSettings, test_tool_outcome_tracking.py's SimpleNamespace), and
# tests/e2e/harness.py sets the variable explicitly per subprocess rather than
# relying on inheritance. Same call cowork-server's own conftest already makes for
# the database: "Force isolation (assignment, not setdefault, never touch a real DB)."
os.environ["ANTON_ANALYTICS_ENABLED"] = "false"


def make_mock_llm() -> AsyncMock:
    """Return an AsyncMock LLM client with coding_provider configured for sync use.

    ``AsyncMock`` makes all child attributes ``AsyncMock`` too, which means
    methods we call synchronously on the provider would otherwise return
    coroutines.  This helper fixes that for both providers — ``coding_provider``
    (whose ``export_connection_info()`` is read in ``ChatSession.__init__``) and
    ``planning_provider`` (whose ``native_web_tools()`` is read in the same
    constructor to resolve the per-session web tool routing).
    """
    mock = AsyncMock()
    mock.coding_provider = MagicMock()
    mock.coding_provider.export_connection_info = MagicMock(
        return_value=ProviderConnectionInfo(provider="anthropic", api_key="test")
    )
    mock.coding_model = "claude-sonnet-4-6"
    # The real client defaults the router role to the coding model; mirror it so
    # window-derived budgets read a real model id, not a Mock.
    mock.router_model = "claude-sonnet-4-6"
    mock.max_tokens = 8192  # LLMClient's own default output ceiling
    mock.planning_provider = MagicMock()
    # Default test posture: no native web tools — fallback tools also off
    # unless a specific test configures otherwise via ChatSessionConfig.
    mock.planning_provider.native_web_tools = MagicMock(return_value=set())
    return mock


@pytest.fixture()
def make_session():
    """A ChatSession with a mock LLM and no console — the minimum needed to
    exercise session-level wiring without touching a terminal or a provider.
    """

    def _factory(**over):
        from anton.core.session import ChatSession, ChatSessionConfig

        kwargs = dict(llm_client=make_mock_llm())
        kwargs.update(over)
        return ChatSession(ChatSessionConfig(**kwargs))

    return _factory


@pytest.fixture()
def make_llm_response():
    def _factory(
        content: str = "",
        tool_calls: list[ToolCall] | None = None,
        input_tokens: int = 10,
        output_tokens: int = 20,
        stop_reason: str | None = "end_turn",
    ) -> LLMResponse:
        return LLMResponse(
            content=content,
            tool_calls=tool_calls or [],
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            stop_reason=stop_reason,
        )

    return _factory


@pytest.fixture(autouse=True)
def _no_browser_windows(monkeypatch):
    """No test may pop a real browser window.

    Every /publish, /share and `anton setup` path ends in ``webbrowser.open``,
    and a stubbed prompt is all it takes to reach one: three tests in
    test_openai_setup.py answer ``Confirm.ask`` with a blanket ``False``, which
    the first prompt in ``_setup_minds`` reads as "no, I don't have an API key"
    and so opens the MindsHub signup page — three windows on every local run.

    Same *intent* as the analytics kill above — the suite never touches the
    world outside the process — but not the same mechanism, which matters for
    what it reaches: that one is a module-level env var, so it is live during
    collection and inherited by subprocesses, while this is per-test and
    parent-process only. Hence the one call site it cannot cover, the
    module-level ``webbrowser.open`` in demo_data/nvda_btc_scratchpad_backup.py
    that ``_agent_zero`` runs in a scratchpad subprocess (ENG-1453). No test
    goes near it today.

    Tests that assert on the call still patch it themselves. Deleting this
    fixture is caught by tests/test_suite_guards.py.
    """
    import webbrowser

    for name in ("open", "open_new", "open_new_tab"):
        monkeypatch.setattr(webbrowser, name, lambda *a, **kw: True)


@pytest.fixture(autouse=True)
def _no_builtin_skills(tmp_path_factory, monkeypatch):
    """Point the built-in skills root at an empty dir for all tests.

    Built-in skills (anton/core/memory/builtin_skills/) would otherwise appear
    in every SkillStore listing and break empty-store assumptions. Tests that
    exercise built-ins pass an explicit `builtin_root=` to SkillStore.
    """
    from anton.core.memory import skills as skills_mod

    empty = tmp_path_factory.mktemp("no-builtin-skills")
    monkeypatch.setattr(skills_mod, "_BUILTIN_SKILLS_ROOT", empty)
