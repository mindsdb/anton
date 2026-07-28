"""Explicit Anton ↔ V1 wire message conversion (item 2)."""

from __future__ import annotations

import pytest

from anton.cloud_turn.messages import (
    MessageConversionError,
    final_assistant_text,
    to_anton_history,
    to_anton_input,
    turn_output_messages,
)
from anton.cloud_turn.protocol import MessageV1, TextBlockV1, ToolUseBlockV1


def test_to_anton_input_string_and_blocks():
    assert to_anton_input("hi") == "hi"
    blocks = [TextBlockV1(text="a"), ToolUseBlockV1(id="t", name="scratchpad", input={})]
    dumped = to_anton_input(blocks)
    assert dumped == [
        {"type": "text", "text": "a"},
        {"type": "tool_use", "id": "t", "name": "scratchpad", "input": {}},
    ]


def test_to_anton_history_round_trips_to_dicts():
    hist = [MessageV1(role="user", content="q"), MessageV1(role="assistant", content="a")]
    assert to_anton_history(hist) == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


def test_turn_output_messages_drops_leading_input_echo():
    prior = [
        {"role": "user", "content": "prior"},        # input history
        {"role": "assistant", "content": "prior answer"},
    ]
    history = prior + [
        {"role": "user", "content": "current input"},  # this turn's echo
        {"role": "assistant", "content": "new answer"},
    ]
    out = turn_output_messages(history, prior)
    assert [m.role for m in out] == ["assistant"]
    assert out[0].content == "new answer"


def test_turn_output_messages_empty_when_no_assistant():
    history = [{"role": "user", "content": "current input"}]
    assert turn_output_messages(history, []) == []


def test_final_assistant_text_from_blocks():
    msgs = [
        MessageV1(role="assistant", content=[TextBlockV1(text="Hello "), TextBlockV1(text="world")]),
    ]
    assert final_assistant_text(msgs) == "Hello world"


def test_unsupported_block_type_raises_not_dropped():
    # V1 supports only text / tool_use / tool_result. An image block (not in the
    # cloud tool surface) must fail loudly rather than be silently dropped.
    history = [{"role": "assistant", "content": [{"type": "image", "source": {}}]}]
    with pytest.raises(MessageConversionError):
        turn_output_messages(history, [])


def test_compaction_summary_and_separator_are_excluded():
    """Simulates _summarize_history rewriting the prefix: the NEW summary +
    'Understood' separator (new identities) precede the surviving input anchor,
    so they must fall outside the current-turn region."""
    surviving_input = {"role": "assistant", "content": "recent input turn"}
    history = [
        {"role": "user", "content": "COMPACTED summary of old turns"},   # new artifact
        {"role": "assistant", "content": "Understood — using that as reference."},  # new artifact
        surviving_input,                                                  # anchor (pre-turn)
        {"role": "user", "content": "current input"},                     # echo
        {"role": "assistant", "content": "the answer"},                   # generated
    ]
    out = turn_output_messages(history, [surviving_input])
    assert [m.role for m in out] == ["assistant"]
    assert out[0].content == "the answer"
