"""Unit tests for the unified cowork + coding system prompt updates and
per-tool advertisement invariants.
"""

from __future__ import annotations

from types import SimpleNamespace

from anton.core.artifacts.geography_guidance import GEOGRAPHY_GUIDANCE
from anton.core.llm.prompt_builder import (
    SESSION_CONTEXT_MARKER,
    ChatSystemPromptBuilder,
    SystemPromptContext,
)
from anton.core.llm.prompts import (
    NO_INTERNET_RESILIENCE_NUDGE,
    RESILIENCE_NUDGE,
)


def _tool(tool_name: str, **props):
    return SimpleNamespace(name=tool_name, prompt=None, input_schema={"properties": props})


def test_coding_tools_section_gates_individual_tools_accurately():
    builder = ChatSystemPromptBuilder()
    common = dict(
        conversation_started="Monday, November 23, 2026",
        system_prompt_context=SystemPromptContext(runtime_context="- Model: kimi-k3"),
        proactive_dashboards=False,
        output_dir="artifacts",
        workspace_context="Workspace files: app.py",
    )

    # None or empty tool_defs must advertise zero coding tools
    for defs in (None, [], [_tool("scratchpad")]):
        p = builder.build(**common, tool_defs=defs)
        assert "CODING & WORKSPACE TOOLS:" not in p
        assert "`bash`" not in p
        assert "`edit_file`" not in p
        assert "`write_file`" not in p

    # Partial tool sets must only mention registered tools
    edit_only = builder.build(**common, tool_defs=[_tool("edit_file")])
    assert "CODING & WORKSPACE TOOLS:" in edit_only
    assert "`edit_file`" in edit_only
    assert "`write_file`" not in edit_only
    assert "`bash`" not in edit_only

    write_only = builder.build(**common, tool_defs=[_tool("write_file")])
    assert "`write_file`" in write_only
    assert "`edit_file`" not in write_only

    # Full coding tool set activates across Desktop/CLI (with direct_generation) and Cloud alike
    full_defs = [
        _tool("bash"),
        _tool("edit_file"),
        _tool("write_file"),
        _tool("grep_search"),
        _tool("find_files"),
        _tool("read_text_file"),
        _tool("generate_artifact", name={}),
    ]
    full_desktop = builder.build(**common, tool_defs=full_defs)
    for marker in (
        "CODING & WORKSPACE TOOLS:",
        "`bash`",
        "`edit_file`",
        "`write_file`",
        "`grep_search`",
        "`find_files`",
        "coworker and software engineer",
        "<user_addition> tag",
        "DS_<ENGINE>_<NAME>__<FIELD>",
        GEOGRAPHY_GUIDANCE,
        "This one\noperation registers the folder",
        "normal workspace source code",
        SESSION_CONTEXT_MARKER,
    ):
        assert marker in full_desktop, f"missing {marker!r}"


def test_offline_and_classify_invariants_preserved_with_coding_tools():
    builder = ChatSystemPromptBuilder()
    defs = [_tool("bash"), _tool("grep_search")]
    common = dict(
        conversation_started="2026-11-23",
        system_prompt_context=SystemPromptContext(),
        proactive_dashboards=False,
        output_dir="/o",
        tool_defs=defs,
    )
    off_internet = builder.build(**common, internet_research=False, classify=True)
    assert "CODING & WORKSPACE TOOLS:" in off_internet
    assert "Internet research is turned off for this session" in off_internet
    assert "DATA SOURCES:" in off_internet

    on = builder.build(**common, internet_research=True, classify=True)
    off = builder.build(**common, internet_research=True, classify=False)
    lines = on.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("  - classify("))
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("- "))
    assert lines[:start] + lines[end:] == off.splitlines()


def test_resilience_nudge_is_universal_and_respects_offline():
    assert "root cause" in RESILIENCE_NUDGE
    assert "explain clearly what failed" in RESILIENCE_NUDGE
    assert "archive.org" not in RESILIENCE_NUDGE
    assert "alternate source/API" in RESILIENCE_NUDGE
    assert "alternate source/API" not in NO_INTERNET_RESILIENCE_NUDGE
    assert "explain clearly what failed" in NO_INTERNET_RESILIENCE_NUDGE
