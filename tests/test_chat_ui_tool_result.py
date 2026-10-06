from __future__ import annotations

import inspect

import pytest

from anton.core.llm.provider import StreamToolResult


@pytest.mark.parametrize("name,action,shown", [
    ("scratchpad", "dump", True),
    ("scratchpad", "exec", False),
    ("scratchpad", None, False),
    ("generate_artifact", "message", True),
    ("generate_artifact", None, False),
])
def test_which_tool_results_the_cli_prints(name, action, shown):
    from anton.chat_ui import is_displayed_tool_result

    assert is_displayed_tool_result(StreamToolResult(name=name, content="x", action=action)) is shown


def test_both_cli_loops_use_it():
    from anton import chat
    from anton.commands import goal

    assert "is_displayed_tool_result(event)" in inspect.getsource(chat)
    assert "is_displayed_tool_result(event)" in inspect.getsource(goal)
