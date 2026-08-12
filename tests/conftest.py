from __future__ import annotations

import os

from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.llm.provider import LLMResponse, ProviderConnectionInfo, ToolCall, Usage

# Tests that drive full turns now reach the turn-cost analytics sink
# (ENG-1288): _emit_turn_cost falls back to a fresh AntonSettings(), whose
# default analytics_url is the real collector. Kill analytics for the whole
# suite so no test run — local or CI — ever fires a real event. (CI is also
# dropped by send_event's own CI detection; this covers local dev runs.)
os.environ.setdefault("ANTON_ANALYTICS_ENABLED", "false")


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
    Same posture as the analytics kill above: the suite never touches the
    desktop. Tests that assert on the call still patch it themselves.
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
