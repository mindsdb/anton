"""`turn_attempt_id`: a key that is unique per turn EXECUTION (ENG-2243).

`turn_index` is a POSITION in the history — `_turn_count` is seeded by counting
the user messages the session was handed — and cowork-server rebuilds the
ChatSession on every turn. So a retried or cancelled attempt is handed the same
history and stamps the same `turn_index`. Measured on prod 2026-08-28..09-01:
14.5% of desktop turn keys carried more than one row (worst: 16, spanning 34
hours) and 18.5% of `tool_completed` rows joined to more than one
`turn_completed` row — the join ENG-1486 stamped the pair to enable.

These tests drive the rebuild that causes it rather than asserting on the
field's existence, because "the field is present" would have passed before the
fix too. The load-bearing one is
`test_a_rebuilt_session_repeats_turn_index_but_not_the_attempt_id`.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_mock_llm

from anton.core.llm.provider import LLMResponse, StreamComplete, Usage
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.turn_cost import TurnCost


@pytest.fixture()
def workspace():
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _usage(n: int = 1_000) -> Usage:
    return Usage(input_tokens=n // 2, output_tokens=n // 2)


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _session(workspace, initial_history=None, session_id="conv-attempt"):
    """A session that completes one turn, optionally seeded with history.

    `initial_history` is the lever: it is what cowork-server hands a freshly
    rebuilt session, and what `_turn_count` is derived from.
    """
    llm = make_mock_llm()
    llm.usage_listener = None

    def plan_stream(**kw):
        async def gen():
            if llm.usage_listener:
                llm.usage_listener("planning", "m", _usage())
            yield StreamComplete(response=LLMResponse(
                content="done", tool_calls=[], usage=_usage(),
                stop_reason="end_turn",
            ))
        return gen()

    async def plan(**kw):
        if llm.usage_listener:
            llm.usage_listener("planning", "m", _usage())
        return LLMResponse(content="done", tool_calls=[], usage=_usage(),
                           stop_reason="end_turn")

    llm.plan_stream = plan_stream
    llm.plan = plan
    s = ChatSession(ChatSessionConfig(
        llm_client=llm, workspace=workspace, session_id=session_id,
        initial_history=list(initial_history) if initial_history else None,
    ))
    s._max_turn_tokens = 0
    return s


async def _turn_row(session, prompt="go") -> dict:
    """Run a turn; return the kwargs of its `turn_completed` event."""
    with patch("anton.analytics.send_event") as sent:
        async for _ in session.turn_stream(prompt):
            pass
    rows = [c.kwargs for c in sent.call_args_list if c.args[1] == "turn_completed"]
    assert len(rows) == 1, f"expected one turn_completed, got {len(rows)}"
    return rows[0]


async def _all_rows(session, prompt="go"):
    with patch("anton.analytics.send_event") as sent:
        async for _ in session.turn_stream(prompt):
            pass
    return [(c.args[1], c.kwargs) for c in sent.call_args_list]


# ── The defect ───────────────────────────────────────────────────────


async def test_a_rebuilt_session_repeats_turn_index_but_not_the_attempt_id(workspace):
    """The production shape: same history in, same turn_index, distinct attempts.

    Two sessions seeded with the SAME `initial_history` is exactly what happens
    when a turn fails and the user retries — cowork-server rebuilds and the
    failed attempt's message was never committed. `turn_index` repeating is
    CORRECT (it is a history position); the attempt id must not.
    """
    history = [_user("first"), _assistant("reply"), _user("second"), _assistant("reply")]

    first = await _turn_row(_session(workspace, history))
    retry = await _turn_row(_session(workspace, history))

    assert first["turn_index"] == retry["turn_index"], (
        "turn_index is a history position and is expected to repeat — if this "
        "fails the premise changed, not the fix"
    )
    assert first["turn_attempt_id"] != retry["turn_attempt_id"]
    assert first["turn_attempt_id"] and retry["turn_attempt_id"]


async def test_the_attempt_id_survives_a_non_completed_terminal(workspace):
    """A turn that dies must still be attributable — those ARE the collisions.

    Every observed colliding key in production was a failed/cancelled attempt
    followed by a success, so an id that only appears on `completed` rows would
    miss the entire population this exists for.
    """
    session = _session(workspace)

    def boom(**kw):
        async def gen():
            raise RuntimeError("planning blew up")
            yield  # pragma: no cover - generator shape
        return gen()

    session._llm.plan_stream = boom
    # anton's turn loop SWALLOWS the exception — the stream just ends and the
    # terminal is recorded on the event. So this asserts on the emitted row,
    # not on a raise: `pytest.raises` here passed vacuously in a first draft.
    with patch("anton.analytics.send_event") as sent:
        async for _ in session.turn_stream("go"):
            pass
    rows = [c.kwargs for c in sent.call_args_list if c.args[1] == "turn_completed"]
    assert len(rows) == 1, "a dead turn must still close its books"
    assert rows[0]["ended_by"] != "completed", (
        f"expected a failure terminal, got {rows[0]['ended_by']!r} — if this "
        "fails the harness stopped reaching the error path and the assertion "
        "below is vacuous"
    )
    assert rows[0]["turn_attempt_id"]


# ── The join it exists to fix ────────────────────────────────────────


async def test_tool_row_and_turn_row_agree_on_the_attempt_id(workspace):
    """Both read the same books, so the join cannot be ambiguous.

    Reading live session state in `_emit_tool_completed` instead would let a
    late finalizer pair a tool row with a different attempt — the same class of
    bug `turn_index`'s stamped-at-open comment records (#309 review).
    """
    from unittest.mock import AsyncMock

    from anton.core.tools.registry import ToolOutcome

    llm = make_mock_llm()
    llm.usage_listener = None
    seq = {"i": 0}

    def plan_stream(**kw):
        async def gen():
            seq["i"] += 1
            if llm.usage_listener:
                llm.usage_listener("planning", "m", _usage())
            if seq["i"] == 1:
                from anton.core.llm.provider import ToolCall
                yield StreamComplete(response=LLMResponse(
                    content="working",
                    tool_calls=[ToolCall(id="tc1", name="scratchpad",
                                         input={"action": "view", "name": "m"})],
                    usage=_usage(), stop_reason="tool_use"))
            else:
                yield StreamComplete(response=LLMResponse(
                    content="done", tool_calls=[], usage=_usage(),
                    stop_reason="end_turn"))
        return gen()

    llm.plan_stream = plan_stream
    session = ChatSession(ChatSessionConfig(
        llm_client=llm, workspace=workspace, session_id="conv-join"))
    session._max_turn_tokens = 0
    session.tool_registry.dispatch_tool = AsyncMock(
        side_effect=[ToolOutcome(content="ok", ok=True)])

    rows = await _all_rows(session)
    turn = [k for n, k in rows if n == "turn_completed"]
    tools = [k for n, k in rows if n == "tool_completed"]
    assert len(turn) == 1 and len(tools) == 1

    assert tools[0]["turn_attempt_id"] == turn[0]["turn_attempt_id"] != ""
    # The old key must still agree too — this fix adds a key, it does not
    # replace the one the Langfuse trace name and artifact index are built on.
    assert tools[0]["turn_index"] == turn[0]["turn_index"]


# ── Seam guard: no construction site can forget ──────────────────────


def test_every_turncost_gets_an_id_without_the_caller_passing_one():
    """A `default_factory`, not a per-call-site argument.

    There are two construction sites in `session.py` today (streaming and
    non-streaming). A third that forgot to pass an id would silently
    reintroduce the collision, so the guarantee lives on the dataclass.
    """
    ids = {TurnCost().attempt_id for _ in range(200)}
    assert len(ids) == 200, "attempt ids must be unique per instance"
    assert all(len(i) == 16 and all(c in "0123456789abcdef" for c in i) for i in ids)


def test_the_id_is_stamped_at_books_open_not_read_at_emit():
    """Stable for the life of the books, like `turn_index`.

    A late finalizer emits books whose turn ended long ago; if the id were
    derived at emit it would name whichever turn owns the slot now.
    """
    tc = TurnCost(turn_index=3)
    first = tc.attempt_id
    tc.add("planning", "m", _usage())
    tc.ended_by = "completed"
    assert tc.attempt_id == first


async def test_the_non_streaming_turn_path_stamps_an_id_too(workspace):
    """`turn()` opens its OWN books (`session.py:3632`) and emits separately.

    The `default_factory` makes this true by construction, but the path is real
    — the CLI's non-streaming API uses it — and "covered by construction" is
    how the streaming path got a field the other one lacked in the first place.
    Two runs, because the value being PRESENT is weaker than it being DISTINCT.
    """
    history = [_user("first"), _assistant("reply")]
    seen = []
    for _ in range(2):
        session = _session(workspace, history)
        with patch("anton.analytics.send_event") as sent:
            await session.turn("go")
        rows = [c.kwargs for c in sent.call_args_list
                if c.args[1] == "turn_completed"]
        assert len(rows) == 1
        seen.append(rows[0])

    assert seen[0]["turn_index"] == seen[1]["turn_index"]
    assert seen[0]["turn_attempt_id"] != seen[1]["turn_attempt_id"]
    assert all(r["turn_attempt_id"] for r in seen)
