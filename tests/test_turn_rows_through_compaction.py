"""A turn's tool rows survive a mid-turn history swap.

Compaction (`_summarize_history`) and hard truncation (`hard_truncate_history`)
replace `ChatSession.history` with a new list and fold the turn's earlier
tool_use/tool_result messages out of it. `ChatSession.turn_messages()` still
returns every message the turn appended, in order, so the cloud pod and
cowork-server's in-process harness can build the turn's tool rows.

The scripted turns (tests/_turn_rows_helpers.py) run a tool round AFTER the
swap: a turn that compacts on its final round appends only text afterwards,
and `split_turn_into_rows` drops text, so such a fixture cannot tell a fix
from a session that only reads the final list.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from anton.cloud_turn.__main__ import stream_turn
from anton.cloud_turn.history_rows import split_turn_into_rows
from anton.core.llm.provider import LLMResponse, StreamContextCompacted
from anton.core.session import ChatSession, ChatSessionConfig
from tests._turn_rows_helpers import (
    PAGES,
    SEED,
    TOOL_IDS,
    run_turn,
    scripted_session,
    tool_result,
    tool_use,
)
from tests.conftest import make_mock_llm

# StreamContextCompacted events each mode yields.
COMPACTIONS = {"none": 0, "proactive": 1, "overflow": 1, "hard-truncate": 2}


def _trim(messages: list) -> list:
    """Drops the turn's user input, as both consumers do before splitting."""
    while messages and (
        not isinstance(messages[0], dict) or messages[0].get("role") != "assistant"
    ):
        messages = messages[1:]
    return messages


def _row_ids(rows: list[dict]) -> list[tuple[str, str]]:
    return [
        (row["role"], block.get("id") or block.get("tool_use_id"))
        for row in rows
        for block in row["content"]
    ]


def _live_result_ids(session: ChatSession) -> list[str]:
    return [
        block["tool_use_id"]
        for message in session.history
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]


_EXPECTED_IDS = [
    (role, tid) for tid in TOOL_IDS for role in ("assistant", "user")
]


@pytest.mark.parametrize("mode", ["proactive", "overflow", "hard-truncate"])
async def test_a_turn_that_swaps_history_returns_every_tool_row(mode):
    session = scripted_session(mode=mode)
    events = await run_turn(session)

    assert sum(isinstance(e, StreamContextCompacted) for e in events) == COMPACTIONS[mode]
    # Precondition: the swap folded the turn's first round out of the live list.
    assert "toolu_1" not in _live_result_ids(session)

    rows = split_turn_into_rows(_trim(session.turn_messages()))
    assert _row_ids(rows) == _EXPECTED_IDS
    roles = [row["role"] for row in rows]
    assert all(a != b for a, b in zip(roles, roles[1:]))
    results = {
        block["tool_use_id"]: block["content"]
        for row in rows
        for block in row["content"]
        if block["type"] == "tool_result"
    }
    assert results == PAGES


async def test_a_turn_that_does_not_swap_history_returns_the_seed_slice():
    """Without a swap the method is exactly the slice consumers used to take."""
    session = scripted_session(mode="none")
    seed_len = len(session.history)
    events = await run_turn(session)

    assert not any(isinstance(e, StreamContextCompacted) for e in events)
    assert session.turn_messages() == session.history[seed_len:]
    assert _row_ids(split_turn_into_rows(_trim(session.turn_messages()))) == _EXPECTED_IDS


async def test_a_turn_after_a_seed_ending_in_a_user_message_returns_the_seed_slice():
    """The turn's user input merges into the seed's last message. That message
    is not this turn's, so it stays out, as it did from the slice."""
    session = scripted_session(mode="none", seed=[*SEED, {"role": "user", "content": "pending"}])
    seed_len = len(session.history)
    events = await run_turn(session)

    assert not any(isinstance(e, StreamContextCompacted) for e in events)
    assert session.history[seed_len - 1]["role"] == "user"
    assert session.turn_messages() == session.history[seed_len:]
    assert split_turn_into_rows(_trim(session.turn_messages())) == split_turn_into_rows(
        _trim(session.history[seed_len:])
    )


async def test_turn_messages_starts_at_the_latest_turn():
    session = scripted_session(mode="none")
    await run_turn(session)
    first_turn_end = len(session.history)
    session._llm.plan_stream = make_mock_llm().plan_stream
    await run_turn(session)

    assert session.turn_messages() == session.history[first_turn_end:]


async def test_a_swap_no_event_announces_keeps_the_rows_appended_after_it():
    """A swap inside a tool call yields no StreamContextCompacted, so a host
    that keyed on that event would never learn of it. The session records the
    swap itself."""

    truncated: list[int] = []

    async def truncating_lookup(session, tc_input):
        if tc_input["id"] == "toolu_3":
            # Inside the tool call, so anton appends this call's tool_result
            # to the new list before the turn yields anything.
            session.hard_truncate_history()
            truncated.append(len(session.history))
        return PAGES[tc_input["id"]]

    session = scripted_session(mode="none", handler=truncating_lookup)
    events = await run_turn(session)

    assert truncated == [4]  # the placeholder, toolu_2's round, toolu_3's tool_use
    assert not any(isinstance(e, StreamContextCompacted) for e in events)
    assert "toolu_1" not in _live_result_ids(session)
    rows = split_turn_into_rows(_trim(session.turn_messages()))
    assert _row_ids(rows) == _EXPECTED_IDS


# ── swaps driven directly ────────────────────────────────────────────────────


def _seeded_session(*, messages: int = 10) -> ChatSession:
    llm = make_mock_llm()
    llm.summarize = AsyncMock(return_value=LLMResponse(content="SUMMARY"))
    seed = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 50}
        for i in range(messages)
    ]
    return ChatSession(ChatSessionConfig(llm_client=llm, initial_history=seed))


async def test_rows_appended_right_after_a_swap_are_kept_without_a_read_between():
    """Nothing reads the session between the swap and the appends. A host that
    noted the new list only when it next looked would start it past t2."""
    session = _seeded_session()
    session._start_turn_segments()
    session._append_history({"role": "user", "content": "q"})
    session._append_history(tool_use("t1"))
    session._append_history(tool_result("t1"))
    outgoing = session.history
    assert await session._summarize_history() is True
    assert session.history is not outgoing
    session._append_history(tool_use("t2"))
    session._append_history(tool_result("t2"))

    rows = split_turn_into_rows(_trim(session.turn_messages()))
    assert _row_ids(rows) == [
        ("assistant", "t1"), ("user", "t1"), ("assistant", "t2"), ("user", "t2"),
    ]


async def test_rows_appended_after_a_summary_with_a_separator_are_kept():
    """With an 8-message seed the compaction cut lands on a user message, so
    the summary swap puts an assistant separator after the summary. That
    branch swaps history too, and the rows appended after it are kept."""
    session = _seeded_session(messages=8)
    session._start_turn_segments()
    session._append_history({"role": "user", "content": "q"})
    session._append_history(tool_use("t1"))
    session._append_history(tool_result("t1"))
    assert await session._summarize_history() is True
    # Precondition: the separator branch ran.
    assert session.history[1] == {
        "role": "assistant", "content": "Understood — using that as reference.",
    }
    session._append_history(tool_use("t2"))
    session._append_history(tool_result("t2"))

    rows = split_turn_into_rows(_trim(session.turn_messages()))
    assert _row_ids(rows) == [
        ("assistant", "t1"), ("user", "t1"), ("assistant", "t2"), ("user", "t2"),
    ]


# `_append_history` merges a message into the previous one when both have the
# same role. Right after a swap the previous message sits before the new list's
# turn segment, so the merged message must move into the segment.


async def test_a_merge_into_the_message_a_compaction_carried_over_is_kept():
    session = _seeded_session()
    session._start_turn_segments()
    session._append_history({"role": "user", "content": "q"})
    session._append_history(tool_use("t1", "t2"))
    session._append_history(tool_result("t1"))
    assert await session._summarize_history() is True
    # The new list ends with the same tool_result message; this merges into it.
    session._append_history(tool_result("t2"))

    rows = split_turn_into_rows(_trim(session.turn_messages()))
    assert _row_ids(rows) == [
        ("assistant", "t1"), ("assistant", "t2"), ("user", "t1"), ("user", "t2"),
    ]
    assert len(rows) == 2


async def test_a_second_merge_into_the_claimed_message_keeps_each_row_once():
    """The first merge moves the carried-over message into the segment. A
    second merge into that same message must not move the segment again,
    which would read the message before it twice."""
    session = _seeded_session()
    session._start_turn_segments()
    session._append_history({"role": "user", "content": "q"})
    session._append_history(tool_use("t1", "t2"))
    session._append_history(tool_result("t1"))
    assert await session._summarize_history() is True
    session._append_history(tool_result("t2"))
    session._append_history({"role": "user", "content": "a note after the results"})

    messages = _trim(session.turn_messages())
    roles = [message["role"] for message in messages]
    assert all(a != b for a, b in zip(roles, roles[1:]))
    rows = split_turn_into_rows(messages)
    assert sorted(_row_ids(rows)) == sorted([
        ("assistant", "t1"), ("assistant", "t2"), ("user", "t1"), ("user", "t2"),
    ])


async def test_a_merge_into_the_hard_truncation_separator_is_kept():
    session = _seeded_session()
    session._start_turn_segments()
    session._append_history({"role": "user", "content": "q"})
    session._append_history(tool_use("t1"))
    session._append_history(tool_result("t1"))
    session.hard_truncate_history(keep=1)
    # The one kept message is t1's tool_result, whose tool_use was cut off, so
    # it goes too. The new list is the placeholder and the "Understood."
    # separator; the next assistant message merges into it.
    assert [m["role"] for m in session.history] == ["user", "assistant"]
    session._append_history(tool_use("t2"))
    session._append_history(tool_result("t2"))

    rows = split_turn_into_rows(_trim(session.turn_messages()))
    assert _row_ids(rows) == [
        ("assistant", "t1"), ("user", "t1"), ("assistant", "t2"), ("user", "t2"),
    ]


async def test_a_nudge_after_two_swaps_on_one_call_keeps_each_row_once():
    """Two overflows on one planning call swap history twice with nothing
    appended between, so the summary's list holds none of the turn's
    messages. The retry then comes back empty and cut at max_tokens, and
    anton's nudge merges into toolu_3's result, which both swaps carried
    over. The claim walks past the empty segment to the list toolu_3's
    result was appended to, so that result is read once."""
    session = scripted_session(mode="hard-truncate", cut=True)
    events = await run_turn(session)

    # Precondition: the summary and the hard truncation both ran, and the
    # nudge merged into toolu_3's result.
    assert sum(isinstance(e, StreamContextCompacted) for e in events) == 2
    carried = next(
        message
        for message in session.history
        if isinstance(message.get("content"), list)
        and any(block.get("tool_use_id") == "toolu_3" for block in message["content"])
    )
    assert [block["type"] for block in carried["content"]] == ["tool_result", "text"]

    rows = split_turn_into_rows(_trim(session.turn_messages()))
    assert _row_ids(rows) == _EXPECTED_IDS


async def test_a_swap_before_the_first_turn_is_not_recorded():
    session = _seeded_session()
    assert session.turn_messages() == []
    assert await session._summarize_history() is True
    session._append_history({"role": "user", "content": "between turns"})
    assert session.turn_messages() == []


async def test_a_swap_after_the_turn_ended_leaves_its_messages_alone():
    session = scripted_session(mode="none")
    await run_turn(session)
    finished = session.turn_messages()
    assert await session._summarize_history() is True
    session._append_history({"role": "user", "content": "between turns"})
    assert session.turn_messages() == finished


async def test_edits_between_turns_leave_the_finished_turn_alone():
    """CLI commands append to history between turns, and the memory command
    rewrites it in place. Neither belongs to the turn that already ended."""
    session = scripted_session(mode="none")
    await run_turn(session)
    finished = session.turn_messages()
    session.history.append({"role": "assistant", "content": "connected a datasource"})
    session.history[:] = [{"role": "user", "content": "rewritten"}, *session.history]
    assert session.turn_messages() == finished


async def test_the_turn_closes_even_when_closing_the_inner_turn_raises():
    session = _seeded_session()

    async def inner(user_input, **kwargs):
        session._append_history({"role": "user", "content": user_input})
        try:
            yield "event"
        finally:
            raise RuntimeError("close failed")

    session._turn_stream_inner = inner
    stream = session.turn_stream("q", enable_interaction=False)
    assert await stream.__anext__() == "event"
    with pytest.raises(RuntimeError, match="close failed"):
        await stream.aclose()
    finished = session.turn_messages()
    assert finished == [{"role": "user", "content": "q"}]

    # A swap after the turn is a between-turn swap, not part of this turn.
    assert await session._summarize_history() is True
    session._append_history({"role": "assistant", "content": "between turns"})
    assert session.turn_messages() == finished


# ── the cloud pod ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["proactive", "overflow", "hard-truncate"])
def test_the_pod_sends_its_history_frame_after_a_compaction(mode):
    session = scripted_session(mode=mode)
    frames: list[dict] = []
    asyncio.run(stream_turn(
        '{"protocol_version":1,"conversation_id":"c","input":"What does overtime cost?"}',
        frames.append,
        session_builder=lambda _req: session,
    ))
    kinds = [frame["kind"] for frame in frames if frame["kind"] != "heartbeat"]

    assert kinds[-1] == "turn_completed"
    assert "compacted" in kinds
    history = [frame for frame in frames if frame["kind"] == "history"]
    assert len(history) == 1
    assert _row_ids(history[0]["rows"]) == _EXPECTED_IDS
    assert kinds.index("history") < kinds.index("turn_completed")
    if mode != "hard-truncate":
        # A hard truncation clears the compaction record, so only a summary
        # sends a compaction frame, and it goes before the history frame.
        assert kinds.index("compaction") < kinds.index("history") < kinds.index("turn_completed")
