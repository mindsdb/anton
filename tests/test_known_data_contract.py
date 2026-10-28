import re

from anton.core.tools.tool_defs import GENERATE_ARTIFACT_TOOL
from anton.core.tools.tool_handlers import _STATUS_INSTRUCTIONS, _data_unavailable


def test_repair_instruction_keeps_the_data_blocks():
    assert "data-anton-data" in _STATUS_INSTRUCTIONS["browser_check_failed"]


def test_known_data_asks_for_a_described_pad_reference():
    description = GENERATE_ARTIFACT_TOOL.description
    schema = GENERATE_ARTIFACT_TOOL.input_schema["properties"]["known_data"]["description"]
    for text in (description, schema):
        assert "a few hundred rows" not in text
        # The shape of a reference: pad in backticks, variable in backticks, type in parentheses.
        assert re.search(r"scratchpad `\w+`, variable `\w+` \([A-Za-z ]+\)", text), text
        # Not the benchmark's own names: the example must not hand the agent a task's answer.
        assert "supply_kpi" not in text and "`frames`" not in text


def test_data_unavailable_asks_for_the_variables():
    assert "the variables that hold the data" in _data_unavailable("no rows")


def test_known_data_asks_to_say_when_pad_data_changed():
    # The saved data is reused when `known_data` is the same text, so a
    # refreshed pad has to show in the text.
    schema = GENERATE_ARTIFACT_TOOL.input_schema["properties"]["known_data"]["description"]
    assert "say when the data in the scratchpad changed" in schema
