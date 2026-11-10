"""The answer style rule reaches the shared prompt on every way the prompt is built."""
from __future__ import annotations

import pytest

from anton.core.llm.prompt_builder import SESSION_CONTEXT_MARKER, ChatSystemPromptBuilder, SystemPromptContext
from anton.core.tools import tool_defs

_TOOL_SETS = {
    "server": [tool_defs.CREATE_ARTIFACT_TOOL, tool_defs.GENERATE_ARTIFACT_TOOL],
    "self_hosted_static": [tool_defs.STATIC_CREATE_ARTIFACT_TOOL, tool_defs.STATIC_GENERATE_ARTIFACT_TOOL],
    "no_direct_reports": [
        tool_defs.CREATE_ARTIFACT_TOOL_NO_DIRECT_REPORTS,
        tool_defs.GENERATE_ARTIFACT_TOOL_NO_DIRECT_REPORTS,
    ],
    "no_workspace": [],
}

_RULES = (
    "Every money figure carries its currency",
    "still give the total, but say plainly that it mixes currencies",
    "State one fact in one sentence.",
    "not table or field names from a query",
)

_BRANCHES = [
    pytest.param(tools, internet, id=f"{name}-internet_{internet}")
    for name, tools in _TOOL_SETS.items()
    for internet in (True, False)
]


def _shared_prompt(tools: list, internet: bool) -> str:
    """The part of the built system prompt every session shares, before the session context."""
    prompt = ChatSystemPromptBuilder().build(
        conversation_started="2026-11-10",
        system_prompt_context=SystemPromptContext(),
        proactive_dashboards=False,
        output_dir="/tmp/out",
        tool_defs=tools,
        internet_research=internet,
        project_context="\n\nproject notes",
    )
    assert SESSION_CONTEXT_MARKER in prompt
    return prompt.split(SESSION_CONTEXT_MARKER, 1)[0]


@pytest.mark.parametrize(("tools", "internet"), _BRANCHES)
def test_the_rule_is_in_the_shared_prompt_once(tools, internet):
    shared = _shared_prompt(tools, internet)
    for rule in _RULES:
        assert shared.count(rule) == 1, rule
