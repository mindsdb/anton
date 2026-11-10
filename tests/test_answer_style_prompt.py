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
    "take it from the record it belongs to",
    "say so instead of guessing",
    "still give the total, but say plainly that it mixes currencies, name them, and give the amount per currency",
    "Convert only when the user asks or the data holds the rate, and name the rate.",
    "State one fact in one sentence.",
    "never repeat a single value in a table",
    "not table or field names from a query",
    "Use schema names when the user asks about the schema or data model.",
    "Record ids and codes (order numbers, part and supplier codes) stay exactly as written.",
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


_GENERIC_CLOSERS = (
    "always end with what the user might want next",
    "Always leave a door open",
    "always let the user know the detail is available",
)


@pytest.mark.parametrize(("tools", "internet"), _BRANCHES)
def test_closing_offers_are_asked_for_only_when_specific(tools, internet):
    shared = _shared_prompt(tools, internet)
    for closer in _GENERIC_CLOSERS:
        assert closer not in shared, closer
    assert shared.count("End with a next step only when it is specific to this answer") == 1


@pytest.mark.parametrize("internet", (True, False))
@pytest.mark.parametrize("name", ("no_direct_reports", "no_workspace"))
def test_a_single_number_is_not_put_in_a_summary_table(name, internet):
    shared = _shared_prompt(_TOOL_SETS[name], internet)
    assert "Start with a compact summary table" not in shared
    assert "A single number goes in the headline sentence, not a table." in shared


@pytest.mark.parametrize("internet", (True, False))
@pytest.mark.parametrize("name", ("server", "self_hosted_static"))
def test_inline_analysis_sections_are_for_several_figures(name, internet):
    shared = _shared_prompt(_TOOL_SETS[name], internet)
    assert "When the answer has several figures, present the main finding" in shared
    assert "A single fact is one sentence with its unit or currency." in shared
    assert "Keep units, source and time scope clear." not in shared


@pytest.mark.parametrize("internet", (True, False))
@pytest.mark.parametrize("name", ("no_direct_reports", "no_workspace"))
def test_the_insights_order_is_for_several_figures(name, internet):
    shared = _shared_prompt(_TOOL_SETS[name], internet)
    assert "ALWAYS follow this order for analysis and reports" not in shared
    assert "for analysis and reports with several figures, follow this order" in shared
    assert "A single fact is still fetched and computed, then answered in one sentence" in shared
