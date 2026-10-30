from types import SimpleNamespace

from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext
from anton.core.llm.prompts import CONVERSATION_DISCIPLINE_ASK_FIRST


def build(*, has_name=True, proactive=False, act_first=True):
    tool = SimpleNamespace(name="generate_artifact", prompt=None,
                           input_schema={"properties": {"name": {}} if has_name else {"slug": {}}})
    return ChatSystemPromptBuilder().build(
        conversation_started="2026-10-24", system_prompt_context=SystemPromptContext(),
        proactive_dashboards=proactive, output_dir="/tmp/unused", act_first=act_first,
        tool_defs=[tool],
    )


def enable(monkeypatch):
    monkeypatch.setenv("ANTON_STATIC_SINGLE_CALL", "on")


def test_delegated_artifacts_have_no_competing_registration_or_data_fetch_instruction(monkeypatch):
    enable(monkeypatch)
    prompt = build()
    assert "This one\noperation registers the folder" in prompt
    assert "REGISTER FIRST, always" not in prompt
    assert "DIRECT REPORTS:" not in prompt
    assert "FETCH DATA FIRST" not in prompt
    assert "For inline analysis without an artifact" in prompt
    assert "dataset for CSV/JSON/parquet" in prompt
    assert "Never create a duplicate artifact" in prompt


def test_old_tool_schema_and_default_configuration_keep_the_existing_workflow(monkeypatch):
    monkeypatch.delenv("ANTON_STATIC_SINGLE_CALL", raising=False)
    baseline = build(has_name=False)
    assert "REGISTER FIRST, always" in build()
    enable(monkeypatch)
    assert build(has_name=False) == baseline


def test_proactive_preference_and_ask_first_are_preserved(monkeypatch):
    enable(monkeypatch)
    assert "Do not proactively create HTML" in build(proactive=False)
    assert "benefits from a saved visual" in build(proactive=True)
    prompt = build(act_first=False)
    assert "configured ask-first/act-first behavior" in prompt
    assert CONVERSATION_DISCIPLINE_ASK_FIRST in prompt
