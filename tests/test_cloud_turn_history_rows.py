"""Turn-history row extraction (ENG-1808).

These helpers live here rather than in cowork-server because both consumers
need them: the pod entrypoint (to build the wire payload) and cowork-server's
in-process harness (to build the same rows on the desktop path).
"""

from __future__ import annotations

from anton.cloud_turn.history_rows import (
    REPLAY_IMAGE_PLACEHOLDER,
    sanitize_tool_result,
    split_turn_into_rows,
)


def _turn_slice():
    """A realistic anton history slice: narration + tool call, its result,
    then the final answer."""
    return [
        {"role": "assistant", "content": [
            {"type": "text", "text": "let me look it up"},
            {"type": "tool_use", "id": "t1", "name": "recall_skill", "input": {"name": "pdf"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "SKILL BODY"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "final answer"}]},
    ]


def test_split_strips_text_and_keeps_tool_blocks():
    rows = split_turn_into_rows(_turn_slice())
    assert len(rows) == 2
    assert rows[0]["role"] == "assistant"
    assert [b["type"] for b in rows[0]["content"]] == ["tool_use"]
    assert rows[0]["content"][0]["input"] == {"name": "pdf"}
    assert rows[1]["role"] == "user"
    assert rows[1]["content"][0]["type"] == "tool_result"
    assert rows[1]["content"][0]["tool_use_id"] == "t1"


def test_split_drops_a_pure_text_message():
    # The trailing "final answer" message carries no tool block -> no row.
    rows = split_turn_into_rows(_turn_slice())
    assert all(
        b["type"] in ("tool_use", "tool_result")
        for row in rows for b in row["content"]
    )


def test_split_ignores_non_dict_entries_and_string_content():
    slice_ = ["junk", {"role": "assistant", "content": "plain text"}]
    assert split_turn_into_rows(slice_) == []


def test_split_of_an_empty_slice_is_empty():
    assert split_turn_into_rows([]) == []


def test_sanitize_replaces_an_image_with_the_replay_marker():
    block = {"type": "tool_result", "tool_use_id": "x", "content": [
        {"type": "image", "source": {"data": "AAAA"}},
        {"type": "text", "text": "kept"},
    ]}
    out = sanitize_tool_result(block)
    assert out["content"][0] == {"type": "text", "text": REPLAY_IMAGE_PLACEHOLDER}
    assert out["content"][1] == {"type": "text", "text": "kept"}
    # The input block is not mutated in place.
    assert block["content"][0]["type"] == "image"


def test_sanitize_leaves_a_string_result_untouched():
    block = {"type": "tool_result", "tool_use_id": "x", "content": "plain string"}
    assert sanitize_tool_result(block) == block


def test_split_sanitizes_images_inside_tool_results():
    slice_ = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "screenshot", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "image", "source": {"data": "AAAA"}}]}]},
    ]
    rows = split_turn_into_rows(slice_)
    assert rows[1]["content"][0]["content"] == [
        {"type": "text", "text": REPLAY_IMAGE_PLACEHOLDER}
    ]
