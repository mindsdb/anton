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
    "When the question asks for a total and the amounts are in different currencies, state the combined total",
    "the combined total as one number first, then say plainly that it mixes currencies, name them, and give the amount per currency",
    '"17,500 in total, mixing EUR and USD: EUR 10,000 and USD 7,500"',
    "That combined figure is the one money figure written without a single currency",
    "Never leave an asked-for total out, even when the amounts are split.",
    "Do not add amounts in different currencies when no total was asked for.",
    "Convert only when the user asks or the data holds the rate, and name the rate.",
    "State one fact in one sentence, unless another rule here asks for more",
    "Tables are for comparing several items or values;",
    "never repeat a single value in a table",
    "Use plain words (the user's own where they gave them), not table or field names from a query",
    "Use schema names when the user asks about the schema or data model.",
    "Record ids and codes (order numbers, part and supplier codes) stay exactly as written.",
)

_BRANCHES = [
    pytest.param(tools, internet, act_first, id=f"{name}-internet_{internet}-act_first_{act_first}")
    for name, tools in _TOOL_SETS.items()
    for internet in (True, False)
    for act_first in (True, False)
]


def _shared_prompt(tools: list, internet: bool, act_first: bool = True) -> str:
    """The part of the built system prompt every session shares, before the session context."""
    prompt = ChatSystemPromptBuilder().build(
        conversation_started="2026-11-10",
        system_prompt_context=SystemPromptContext(),
        proactive_dashboards=False,
        output_dir="/tmp/out",
        tool_defs=tools,
        internet_research=internet,
        project_context="\n\nproject notes",
        act_first=act_first,
    )
    assert SESSION_CONTEXT_MARKER in prompt
    return prompt.split(SESSION_CONTEXT_MARKER, 1)[0]


@pytest.mark.parametrize(("tools", "internet", "act_first"), _BRANCHES)
def test_the_rule_is_in_the_shared_prompt_once(tools, internet, act_first):
    shared = _shared_prompt(tools, internet, act_first)
    for rule in _RULES:
        assert shared.count(rule) == 1, rule


_GENERIC_CLOSERS = (
    "always end with what the user might want next",
    "Always leave a door open",
    "always let the user know the detail is available",
)


@pytest.mark.parametrize(("tools", "internet", "act_first"), _BRANCHES)
def test_closing_offers_are_asked_for_only_when_specific(tools, internet, act_first):
    shared = _shared_prompt(tools, internet, act_first)
    for closer in _GENERIC_CLOSERS:
        assert closer not in shared, closer
    assert shared.count("End with a next step only when it is specific to this answer") == 1
    assert "Otherwise end without one: a generic offer" in shared
    assert "a check the result depends on" not in shared
    assert shared.count(
        "A next step never replaces the answer: do the work, answer, then offer, "
        "unless these instructions say to ask first."
    ) == 1


@pytest.mark.parametrize("internet", (True, False))
@pytest.mark.parametrize("name", ("no_direct_reports", "no_workspace"))
def test_a_single_number_is_not_put_in_a_summary_table(name, internet):
    shared = _shared_prompt(_TOOL_SETS[name], internet)
    assert "Start with a compact summary table" not in shared
    assert "A single number goes in the headline sentence, not a table." in shared


_INLINE_ANALYSIS = (
    "For inline analysis without an artifact, fetch current data and calculate metrics "
    "with code. When the answer has several figures, present the main finding, exact "
    "figures, comparisons, assumptions and useful implications, and keep units, time "
    "scope and where the data came from clear, in plain words. A single fact is still "
    "computed from the data, then given in one sentence with its unit or currency.\n"
)


@pytest.mark.parametrize("internet", (True, False))
@pytest.mark.parametrize("name", ("server", "self_hosted_static"))
def test_inline_analysis_sections_are_for_several_figures(name, internet):
    """The server paragraph is pinned whole: no snapshot digest covers this branch."""
    shared = _shared_prompt(_TOOL_SETS[name], internet)
    start = shared.index("For inline analysis without an artifact")
    assert shared[start:start + len(_INLINE_ANALYSIS)] == _INLINE_ANALYSIS
    assert shared.count("For inline analysis without an artifact") == 1


@pytest.mark.parametrize("internet", (True, False))
@pytest.mark.parametrize("name", ("no_direct_reports", "no_workspace"))
def test_the_insights_order_is_for_several_figures(name, internet):
    shared = _shared_prompt(_TOOL_SETS[name], internet)
    assert "ALWAYS follow this order for analysis and reports" not in shared
    assert "for analysis and reports with several figures, follow this order" in shared
    assert "A single fact is still fetched and computed, then answered in one sentence" in shared
