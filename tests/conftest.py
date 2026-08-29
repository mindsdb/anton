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
# Measured 2026-08-28 with the variable exported, counted at two different
# points, because they do not agree and the difference is the interesting part:
#
#   on the wire (local capture server)   260 requests, 13 event names, 5 families
#                                        tool_completed 149, ds_connect_* 89,
#                                        ask_user_* 16, turn_completed 5,
#                                        scratchpad_package_installed 1
#   at the emitter (send_event calls)    277 invocations, 16 event names
#
# Quote the wire number: 260 is what reached production. Do NOT subtract the two
# — they count different populations, in both directions. Some send_event calls
# never send (they die inside its own try/except), and some wire requests have no
# send_event call in this process at all: test_cloud_turn_process.py does
# `env = os.environ.copy()` and runs the real entrypoint as a child, which
# produced 5 wire events against 0 in-process calls when measured alone. The tell
# is that the wire shows MORE tool_completed than the emitter does (149 vs 148),
# which no amount of swallowed exceptions can explain.
#
# That child is also why this line must be an assignment rather than setdefault
# for a second reason: children inherit os.environ, so the write reaches them.
# Same file, measured before and after this fix: 5 events -> 0.
#
# Two traps live in that gap, and both cost a review round here.
#
# First, enumerating leakers by GREPPING THE EVENT NAME is unsound. For
# scratchpad_package_installed the grep finds only TestPackageInstallTelemetry,
# whose four tests monkeypatch send_event and fire nothing — but a fifth caller,
# test_scratchpad_observer_dispatch.py::TestHandleScratchpadObserverIntegration::
# test_non_exec_actions_do_not_fire_observers, reaches send_package_install_event
# through the dispatch path while containing no occurrence of send_event, patch(,
# monkeypatch, or the event name. It is structurally invisible to that method.
# Instrument the emitter instead.
#
# Second, reaching send_event is NOT sending, and the same event demonstrates
# both halves. Instrumenting send_event itself (not send_package_install_event —
# that is what made an earlier revision name the wrong culprit) shows THREE
# callers reaching it with scratchpad_package_installed, with three outcomes:
#
#   test_analytics.py::test_scratchpad_package_installed_goes_to_posthog_...
#       _PosthogSettings, host ph.example.test — body builds, goes nowhere real
#   test_chat_scratchpad.py::TestScratchpadInstallViaChat::
#       test_install_action_dispatch
#       real AntonSettings — reads the environment and SENDS. This is the 1
#       scratchpad_package_installed on the wire above
#   test_scratchpad_observer_dispatch.py::TestHandleScratchpadObserverIntegration::
#       test_non_exec_actions_do_not_fire_observers
#       MagicMock session, so send_event sails past both guards below (every
#       MagicMock attribute is truthy) and dies in _posthog_body with "Object of
#       type MagicMock is not JSON serializable", swallowed by send_event's own
#       `except Exception: pass`
#
# The MagicMock one is safe BY ACCIDENT — not by either guard in this file — and
# an earlier revision generalised that into "it never leaked". It does leak, from
# the middle caller, which needs no more realistic settings object because it
# already builds a real one.
#
# That middle caller fires only on a COLD workspace, and this is the thing to
# know before re-measuring. The event is gated on
# `install_call_installed_something(result)` (tool_handlers.py:719), and the
# `workspace` fixture is a PERSISTENT directory in the repo —
# `<repo>/.pytest-workspace` (test_chat_scratchpad.py:20), not tmp_path. So the
# first run on a machine really pip-installs cowsay and emits the event; every
# run after that gets "already satisfied", emits nothing, and reports 259/four
# forever. Both counts are correct, for different machine states:
#
#   cold (.pytest-workspace absent)    260 requests, 5 families
#   warm (cowsay already in the venv)  259 requests, 4 families
#
# To reproduce the 260: `rm -rf .pytest-workspace` first, or you will measure
# 259 and conclude this comment is wrong. It is not ordering — ordering within a
# run cannot change it, because the gate is the state of a directory that
# outlives the run.
#
# Both points stand and neither subsumes the other: a name-grep cannot find the
# third caller, and reaching send_event does not mean sending. The accidental
# safety of the third is still worth its own ticket — a caller that stopped
# passing a MagicMock would start sending, with nothing in this file catching it.
#
# ENG-1692's script-traffic guard does not cover this. It lives inside
# _emit_turn_cost alone, so four of those five families have no guard at all, and
# it only takes effect once a developer updates their installed build. This line
# is read from the checkout, so it takes effect on the next test run after a pull
# or rebase — cheaper than a reinstall, but a long-lived branch cut before this
# landed does not get it automatically.
#
# The escape hatch is removed deliberately. Nothing loses coverage: the tests that
# exercise the analytics layer build their own settings objects and never read the
# environment (test_analytics.py::_Settings and ::S, test_tool_completed.py::
# _PosthogSettings, test_tool_outcome_tracking.py's SimpleNamespace), and
# tests/e2e/harness.py sets the variable explicitly per subprocess rather than
# relying on inheritance. Same call cowork-server's own conftest already makes for
# the database: "Force isolation (assignment, not setdefault, never touch a real DB)."
os.environ["ANTON_ANALYTICS_ENABLED"] = "false"
#
# Belt and braces: blank the sinks too, so the suite is fail-safe rather than
# fail-open. The flag above is honoured by exactly one `if` in `send_event`, and
# this whole bug exists because ENG-1288 added an emitter that reached a real
# sink — the next one, or any refactor of that single check, reopens it, and the
# test below would not notice because it asserts on a resolved setting rather
# than on "no bytes left the process". Both of these are documented kill
# switches in analytics.py: blanking `analytics_url` has always stopped every
# event, and an empty `posthog_key` disables the direct PostHog sink. Measured
# with the enabled-check bypassed: flag alone still emitted, flag plus these two
# emitted nothing. Costs no coverage — the full suite passes unchanged.
os.environ["ANTON_ANALYTICS_URL"] = ""
os.environ["ANTON_POSTHOG_KEY"] = ""


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
