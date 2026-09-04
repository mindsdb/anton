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
    """Both rows carry the same attempt, so the join cannot be ambiguous.

    How each side gets there differs, and the earlier version of this docstring
    had it backwards (#431 review). `_emit_tool_completed` DOES read live
    session state — `_tc = getattr(self, "_turn_cost", None)`, near the top of
    its `try`. It is safe because a tool row is always emitted
    synchronously inside its own live turn and the owning turn nulls the books
    at close, so the read is never late; not because it reads stamped books.

    The stamped-at-open guarantee is load-bearing on the TURN side, where the
    emit genuinely can be late — see
    `test_a_late_finalizer_reports_its_own_attempt_not_the_live_turns`.

    The distinction matters for whoever changes this next: if the tool emit
    were ever moved off the synchronous path (queued or batched), this test
    would still pass while the join silently started pairing tool rows with a
    neighbouring attempt.
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


def test_all_sixteen_characters_carry_entropy():
    """The width claim in `turn_cost.py` has to be true, not just plausible.

    The field first shipped as `uuid.uuid4().hex[:16]`, whose comment claimed
    64 bits. It is 60: hex position 12 is uuid4's version nibble, so it is the
    literal `4` in every id ever generated. The test above could not catch that
    — 200 uuid4-derived ids are all unique and all lowercase hex, and a fixed
    nibble costs 4 bits without failing either assertion.

    So assert the property the comment claims: every position varies. 200
    samples over 16 possible values makes a false failure ~(15/16)**200 ≈ 3e-6
    per position, and a positionally-fixed nibble fails deterministically.
    """
    ids = [TurnCost().attempt_id for _ in range(200)]
    fixed = [pos for pos in range(16) if len({i[pos] for i in ids}) == 1]
    assert not fixed, (
        f"hex positions {fixed} are constant across 200 ids — the field is "
        "narrower than the 64 bits its comment claims"
    )


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


def test_a_late_finalizer_reports_its_own_attempt_not_the_live_turns():
    """The invariant the stamped-at-open design exists for, at the emit.

    `_turn_stream_inner`'s `finally` calls
    `_emit_turn_cost(expected=_turn_cost_books, exc=_turn_exc)` on the
    abandoned-generator path — a fresh task, long after the fact, by
    which point a NEWER turn may own the shared slot. That is the case where
    `tc is not self._turn_cost`, and `_emit_turn_cost` reads the books it was
    handed (`tc = expected if expected is not None else self._turn_cost`)
    rather than live session state, precisely so the abandoned turn's row
    carries its own id.

    Nothing covered this. `test_the_id_is_stamped_at_books_open_not_read_at_emit`
    above never invokes `_emit_turn_cost` — it mutates a `TurnCost` and asserts
    the field is stable — so changing that line to `self._turn_cost.attempt_id`
    passed this whole module while stamping the live turn's id on the abandoned
    turn's row, reintroducing the #309 mis-attribution one layer down.
    `turn_index` has a guard for that class in `test_turn_cost_terminals.py`
    (`test_a_late_finalizer_cannot_close_a_newer_turns_books`); this is the
    matching one for `attempt_id`.
    """
    session = ChatSession.__new__(ChatSession)
    session._llm = MagicMock()
    session._llm.planning_model = "p"
    session._llm.coding_model = "c"
    session._session_id = "s"
    session._harness = "cowork"
    session._turn_count = 1
    session._cancel_event = MagicMock(is_set=lambda: False)
    session._settings = None

    abandoned, live = TurnCost(turn_index=4), TurnCost(turn_index=5)
    abandoned.add("planning", "sonnet", _usage())
    session._turn_cost = live               # the newer turn owns the slot

    with patch("anton.analytics.send_event") as send:
        session._emit_turn_cost(expected=abandoned)

        assert send.called, "the abandoned turn must still be counted (#309)"
        emitted = send.call_args.kwargs["turn_attempt_id"]
        assert emitted == abandoned.attempt_id, (
            "the late finalizer stamped the LIVE turn's attempt id on the "
            "abandoned turn's row"
        )
        assert emitted != live.attempt_id
        assert session._turn_cost is live, "only the owner clears the slot"


def test_the_structured_log_line_carries_the_attempt_too(caplog):
    """The only forensics surface that does not depend on the collector.

    `_emit_turn_cost`'s own comment says the `turn_cost` log line is what
    survives the allowlist that silently dropped `turn_completed`'s properties
    for weeks (ENG-1355), and per ENG-2193 it is the only channel a desktop
    customer has — `cowork-server.log`, not PostHog. Without the attempt id
    there, the analytics property this PR adds has no fallback, and two
    attempts of one turn still produce two indistinguishable log lines: the
    exact ambiguity ENG-2243 exists to remove.

    Asserted on the ABANDONED books so this also pins the source, not just the
    presence of the key.
    """
    import logging

    session = ChatSession.__new__(ChatSession)
    session._llm = MagicMock()
    session._llm.planning_model = "p"
    session._llm.coding_model = "c"
    session._session_id = "s"
    session._harness = "cowork"
    session._turn_count = 1
    session._cancel_event = MagicMock(is_set=lambda: False)
    session._settings = None

    abandoned, live = TurnCost(turn_index=4), TurnCost(turn_index=5)
    abandoned.add("planning", "sonnet", _usage())
    session._turn_cost = live

    with caplog.at_level(logging.INFO, logger="anton.core.session"):
        with patch("anton.analytics.send_event"):
            session._emit_turn_cost(expected=abandoned)

    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("turn_cost session=")]
    assert len(lines) == 1, lines
    assert f"attempt={abandoned.attempt_id}" in lines[0], lines[0]
    assert live.attempt_id not in lines[0]


async def test_two_turns_in_ONE_session_get_different_ids(workspace):
    """Per TURN, not per session — the CLI never rebuilds.

    Every other test here builds a fresh ChatSession per turn, which is what
    cowork-server does. The CLI does not: it keeps one session for the whole
    conversation. So an id that were merely per-SESSION would satisfy all of
    them while leaving every CLI turn sharing one id and the join ambiguous
    again. Found by mutation: making `attempt_id` a stable per-session value
    passed the entire suite before this test existed.
    """
    session = _session(workspace)
    seen = []
    for _ in range(3):
        with patch("anton.analytics.send_event") as sent:
            async for _ in session.turn_stream("go"):
                pass
        rows = [c.kwargs for c in sent.call_args_list
                if c.args[1] == "turn_completed"]
        assert len(rows) == 1
        seen.append(rows[0])

    ids = [r["turn_attempt_id"] for r in seen]
    assert len(set(ids)) == 3, f"expected 3 distinct ids across 3 turns, got {ids}"
    # turn_index DOES advance here — the session was never rebuilt — so this
    # also pins that the two fields stay independent.
    assert len({r["turn_index"] for r in seen}) == 3
