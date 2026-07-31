"""The four hazards of racing a tool dispatch against the emitter queue.

Every one of these is a bug that a naive implementation has, so each maps
to a numbered detail in the design doc's drain-loop section.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from anton.core.interaction.emitter import TurnEmitter
from anton.core.tools.tool_defs import ToolDef


def _tc(name="probe", **input_):
    return SimpleNamespace(id="tu_1", name=name, input=input_)


async def _collect(session, tc):
    events, result = [], None
    agen = session._dispatch_draining(tc)
    try:
        async for kind, payload in agen:
            if kind == "event":
                events.append(payload)
            else:
                result = payload
    finally:
        await agen.aclose()
    return events, result


@pytest.fixture()
def session(make_session):
    s = make_session()
    s.emitter = TurnEmitter()
    return s


async def test_all_events_arrive_in_order_before_the_result(session):
    """Hazard 2, base case: everything a running tool emits is forwarded, in
    order, and all of it before the result.

    Served entirely by the in-loop branch — this handler emits synchronously
    while the dispatch is still running — so it says nothing about events
    queued in the last instant. That claim belongs to the two dedicated
    last-instant tests below.
    """

    async def handler(sess, tc_input):
        for i in range(5):
            await sess.emit(f"ev{i}")
        return "done"

    session.tool_registry.register_tool(
        ToolDef(name="probe", description="", input_schema={}, handler=handler)
    )
    events, result = await _collect(session, _tc())
    assert events == ["ev0", "ev1", "ev2", "ev3", "ev4"]
    assert result == "done"


async def test_events_emitted_between_awaits_are_interleaved(session):
    async def handler(sess, tc_input):
        await sess.emit("first")
        await asyncio.sleep(0.01)
        await sess.emit("second")
        return "ok"

    session.tool_registry.register_tool(
        ToolDef(name="probe", description="", input_schema={}, handler=handler)
    )
    events, result = await _collect(session, _tc())
    assert events == ["first", "second"]
    assert result == "ok"


async def test_an_event_queued_in_the_last_instant_is_not_dropped(session):
    """Hazard 2, sharpened.

    The in-loop branch already forwards anything the tool emits while it is
    still running, so a handler that emits and returns synchronously cannot
    tell whether the tail drain exists. This one pins the ordering that makes
    the tail drain meaningful: an event that reaches the queue only after the
    dispatch task is already done must still be forwarded, and it must be
    forwarded before the result.

    At THIS timing — the emit lands before `asyncio.wait`'s waiter is
    resolved — the getter also completes, so `getter in done` is True and the
    in-loop branch carries the event. That makes this a test of the getter's
    priority over `task in done`, not of the tail drain: it passes with the
    tail drain removed. Do not delete the tail drain on its authority; see
    the one-hop-late test below, which fails without it.
    """

    async def handler(sess, tc_input):
        # Emitted from a separate task that runs after `handler` has already
        # returned, i.e. after the dispatch task is done. This is the shape
        # of anything emitted by a sub-agent outliving its tool call.
        asyncio.create_task(sess.emit("late"))
        return "done"

    session.tool_registry.register_tool(
        ToolDef(name="probe", description="", input_schema={}, handler=handler)
    )
    events, result = await _collect(session, _tc())
    assert events == ["late"]
    assert result == "done"


async def test_an_event_emitted_one_hop_late_is_recovered_by_the_tail_drain(session):
    """Hazard 2, the timing the tail drain exists for.

    Resolving `asyncio.wait`'s waiter takes two `call_soon` hops:
    `_on_completion` -> `waiter.set_result` -> the waiting coroutine's
    `__wakeup`. An emit performed by a callback scheduled BETWEEN those two
    hops — one `await asyncio.sleep(0)` later than the test above — schedules
    the getter's own wakeup after the drain loop has already resumed. So the
    getter is still pending, `getter in done` is False, the loop breaks, and
    the item is sitting untouched in `Queue._queue` (`put_nowait`'s
    `_wakeup_next` never hands it over). The tail drain is the only thing
    that recovers it; without it the event is silently dropped.
    """

    async def handler(sess, tc_input):
        async def _one_hop_late():
            await asyncio.sleep(0)
            await sess.emit("late")

        asyncio.create_task(_one_hop_late())
        return "done"

    session.tool_registry.register_tool(
        ToolDef(name="probe", description="", input_schema={}, handler=handler)
    )
    events, result = await _collect(session, _tc())
    assert events == ["late"]
    assert result == "done"


async def test_a_raising_tool_surfaces_its_exception_to_the_caller(session):
    """Hazard 3, helper half: asyncio.wait never re-raises a task's
    exception, so the result must be retrieved with task.result() rather than
    silently dropped into 'Task exception was never retrieved'."""

    async def handler(sess, tc_input):
        raise ValueError("tool exploded")

    session.tool_registry.register_tool(
        ToolDef(name="probe", description="", input_schema={}, handler=handler)
    )
    with pytest.raises(ValueError, match="tool exploded"):
        await _collect(session, _tc())


async def test_a_raising_tool_still_becomes_a_tool_result_for_the_model(make_session):
    """Hazard 3, the half that actually matters.

    The regression this guards is that the PRE-EXISTING handler in
    _stream_and_handle_tools keeps working:

        except Exception as exc:
            result_text = f"Tool '{tc.name}' failed: {exc}"

    That only holds if the `async for` over the helper sits INSIDE the same
    `try`. The helper-level test above passes either way, so it cannot catch
    a misplaced `try` — this one runs a whole turn and inspects the history.
    """
    session = make_session()
    # Keep the completion verifier out of this turn: it would call the mock
    # LLM's generate_object_code and leave an unawaited-coroutine warning
    # from the resulting MagicMock verdict. Irrelevant to what is under test.
    session._verify_min_tool_rounds = 99

    async def handler(sess, tc_input):
        raise ValueError("tool exploded")

    session.tool_registry.register_tool(
        ToolDef(name="probe", description="", input_schema={}, handler=handler)
    )

    # Scripted LLM. The seam is plan_stream_with_recovery — which
    # _stream_and_handle_tools consumes — NOT the non-streaming
    # plan_with_recovery, which turn_stream never reaches.
    calls = {"n": 0}

    async def _plan_stream(system=None, tools=None, **kwargs):
        from anton.core.llm.provider import (
            LLMResponse,
            StreamComplete,
            StreamToolUseEnd,
            StreamToolUseStart,
            ToolCall,
            Usage,
        )

        calls["n"] += 1
        if calls["n"] == 1:
            yield StreamToolUseStart(id="tu_1", name="probe")
            yield StreamToolUseEnd(id="tu_1")
            yield StreamComplete(
                response=LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="tu_1", name="probe", input={})],
                    usage=Usage(input_tokens=1, output_tokens=1),
                    stop_reason="tool_use",
                )
            )
            return
        yield StreamComplete(
            response=LLMResponse(
                content="I could not run that.",
                tool_calls=[],
                usage=Usage(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
            )
        )

    session.plan_stream_with_recovery = _plan_stream
    async for _ in session.turn_stream("go"):
        pass

    tool_results = [
        block
        for message in session._history
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert tool_results, "the failing tool produced no tool_result at all"
    assert "Tool 'probe' failed" in str(tool_results[-1]["content"])


async def test_cancelling_while_parked_on_a_yield_still_cancels_the_tool(session):
    """Hazard 1 + generator finalization: cancel while the consumer is
    handling an event, i.e. while the helper is suspended AT a yield.
    Cancelling on an inner await passes even when cleanup is broken."""
    cleaned = asyncio.Event()

    async def handler(sess, tc_input):
        await sess.emit("card")
        try:
            await asyncio.sleep(3600)  # waiting for a human
        except asyncio.CancelledError:
            cleaned.set()
            raise
        return "never"

    session.tool_registry.register_tool(
        ToolDef(name="probe", description="", input_schema={}, handler=handler)
    )

    async def consumer():
        agen = session._dispatch_draining(_tc())
        try:
            async for kind, payload in agen:
                if kind == "event":
                    await asyncio.sleep(3600)  # park AT the yield
        finally:
            await agen.aclose()

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(cleaned.wait(), timeout=1)


async def test_no_task_is_left_running_after_the_helper_closes(session):
    """Hazard 1: BOTH futures the helper owns must be gone afterwards.

    Asserted on the specific futures, not on a count of `all_tasks()` with
    slack: the helper owns exactly two, the dispatch task and the pending
    `emitter.get()` it parks on, and a count-with-slack assertion passes
    happily while leaking one of them. `asyncio.wait` cancels neither of its
    own futures on cancellation, so the getter needs its own `cancel()` in the
    helper's `finally` — otherwise it stays registered in the queue's
    `_getters` and asyncio eventually logs "Task was destroyed but it is
    pending!".

    The Stop shape reproduced here is the common one: a tool that is running
    and has not emitted, i.e. the helper parked inside `asyncio.wait`.
    """
    dispatch = {}

    async def handler(sess, tc_input):
        dispatch["task"] = asyncio.current_task()
        await asyncio.sleep(3600)

    session.tool_registry.register_tool(
        ToolDef(name="probe", description="", input_schema={}, handler=handler)
    )
    queue = session.emitter._queue
    before = len(asyncio.all_tasks())
    gen_task = asyncio.create_task(_collect(session, _tc()))
    await asyncio.sleep(0.05)
    # Parked in asyncio.wait: the tool is running, the getter is pending.
    assert not dispatch["task"].done()
    assert len(queue._getters) == 1

    gen_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await gen_task
    await asyncio.sleep(0.05)

    assert dispatch["task"].cancelled(), "the dispatch task outlived the helper"
    assert not queue._getters, "a pending emitter.get() is still registered"
    # Upper bound only: any unrelated task that happened to be alive at
    # `before` and finished during the test would push the count below it and
    # fail for a reason that has nothing to do with the helper. The real
    # guarantee is the two assertions above, which name the futures.
    assert len(asyncio.all_tasks()) <= before


async def test_a_failure_opening_the_first_getter_reports_its_own_cause(session):
    """The ``finally`` cancels ``getter``, which the first
    ``ensure_future(self.emitter.get())`` is what binds. If that call itself
    raises, an unbound ``getter`` in the ``finally`` replaces the real cause
    with a ``NameError`` and the tool failure the model sees names the wrong
    thing entirely. Unreachable with the real ``TurnEmitter``; reachable for any
    future emitter, so it is initialised to None and guarded.
    """

    async def handler(sess, tc_input):
        return "done"

    session.tool_registry.register_tool(
        ToolDef(name="probe", description="", input_schema={}, handler=handler)
    )

    class _BrokenEmitter(TurnEmitter):
        def get(self):  # not a coroutine -> ensure_future raises TypeError
            return object()

    session.emitter = _BrokenEmitter()
    with pytest.raises(TypeError):  # not NameError
        await _collect(session, _tc())


async def test_an_event_emitted_while_the_host_handles_the_result_is_not_lost(session):
    """The in-loop ``getter.cancel()`` before ``break``, not just the one in
    ``finally``.

    Between ``break`` and the ``finally`` the helper suspends: the tail drain
    yields, and then ``yield ("result", ...)``. A getter that is doomed but
    still registered in the queue's ``_getters`` is handed any event emitted
    while the host is processing one of those yields — ``Queue.put_nowait``
    -> ``_wakeup_next`` sets that pending getter's result instead of leaving
    the item in ``_queue``. The helper never reads that result (it has already
    broken out of the loop), ``gather`` in the ``finally`` swallows it, and the
    event is gone: not delivered, and not even left in the queue for a later
    drain to recover.

    Cancelling in the loop makes the inner waiter already-done, so
    ``_wakeup_next`` skips it and the item stays in ``_queue``. Without the
    in-loop cancel this fails with an empty queue and no delivery.

    The consumer below mirrors the real caller in ``_stream_and_handle_tools``,
    including its post-``aclose()`` drain, which is what turns "delivered or at
    least still recoverable from the queue" into an unconditional "delivered".
    """

    async def handler(sess, tc_input):
        return "done"

    session.tool_registry.register_tool(
        ToolDef(name="probe", description="", input_schema={}, handler=handler)
    )
    queue = session.emitter._queue

    events, result = [], None
    agen = session._dispatch_draining(_tc())
    try:
        async for kind, payload in agen:
            if kind == "event":
                events.append(payload)
            else:
                result = payload
                # Emitted while the helper is parked AT the ("result", ...)
                # yield, i.e. exactly the window the in-loop cancel closes.
                await session.emitter.emit("post")
                await asyncio.sleep(0)
    finally:
        await agen.aclose()
    # The caller's end-of-turn drain.
    while not session.emitter.empty():
        events.append(session.emitter.get_nowait())

    assert result == "done"
    assert "post" in events, (
        "the event emitted while the host handled the result was swallowed: "
        f"events={events!r} leftover_in_queue={list(queue._queue)!r}"
    )


# ─── turn_stream: the per-turn guards around the dispatch loop ───────────


def _script_one_tool_call(session, tool_name="probe", rounds=1):
    """Scripted `plan_stream_with_recovery`: call *tool_name* once per turn,
    then end the turn. The seam is the streaming planner — `turn_stream` never
    reaches the non-streaming `plan_with_recovery`."""
    from anton.core.llm.provider import (
        LLMResponse,
        StreamComplete,
        StreamToolUseEnd,
        StreamToolUseStart,
        ToolCall,
        Usage,
    )

    # Keep the completion verifier out of these turns: it would call the mock
    # LLM's generate_object_code and leave an unawaited-coroutine warning.
    session._verify_min_tool_rounds = 99
    state = {"n": 0}

    async def _plan_stream(system=None, tools=None, **kwargs):
        state["n"] += 1
        if state["n"] % (rounds + 1) == 1:
            yield StreamToolUseStart(id="tu_1", name=tool_name)
            yield StreamToolUseEnd(id="tu_1")
            yield StreamComplete(
                response=LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="tu_1", name=tool_name, input={})],
                    usage=Usage(input_tokens=1, output_tokens=1),
                    stop_reason="tool_use",
                )
            )
            return
        yield StreamComplete(
            response=LLMResponse(
                content="done",
                tool_calls=[],
                usage=Usage(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
            )
        )

    session.plan_stream_with_recovery = _plan_stream


async def test_turn_stream_attaches_the_emitter_for_the_turn_and_detaches_after(
    make_session,
):
    """`turn_stream` owns the emitter's lifetime, and both halves matter.

    Attached: without `self.emitter = TurnEmitter()` nothing a tool emits out of
    band can reach the host, and `elicit()` refuses every choice question as
    unavailable.

    Detached: without `self.emitter = None` in the `finally`, a later
    non-streaming `turn()` inherits a stale emitter, so `elicit()` passes its own
    availability gate and blocks on a question nobody will ever see — the exact
    failure that gate exists to prevent.
    """
    session = make_session()
    seen = {}

    async def handler(sess, tc_input):
        seen["emitter"] = sess.emitter
        return "ok"

    session.tool_registry.register_tool(
        ToolDef(name="probe", description="", input_schema={}, handler=handler)
    )
    _script_one_tool_call(session)

    assert session.emitter is None  # before the turn
    async for _ in session.turn_stream("go"):
        pass

    assert seen["emitter"] is not None, "no emitter was attached for the turn"
    assert isinstance(seen["emitter"], TurnEmitter)
    assert session.emitter is None, "the emitter outlived the streaming turn"


async def test_turn_stream_resets_the_question_budget_each_turn(make_session):
    """Per TURN, not per session: without the reset, `ask_user` dies permanently
    after the third question of a session and nothing else notices."""
    from anton.core.interaction.elicit import MAX_QUESTIONS_PER_TURN

    session = make_session()
    counts = []

    async def handler(sess, tc_input):
        counts.append(sess.question_count)
        # Spend the whole budget, as three questions in one turn would.
        sess.question_count = MAX_QUESTIONS_PER_TURN
        return "ok"

    session.tool_registry.register_tool(
        ToolDef(name="probe", description="", input_schema={}, handler=handler)
    )
    _script_one_tool_call(session)

    async for _ in session.turn_stream("first"):
        pass
    async for _ in session.turn_stream("second"):
        pass

    assert counts == [0, 0], (
        "the second turn inherited a spent question budget, so ask_user would be "
        f"permanently unavailable: {counts!r}"
    )


@pytest.mark.parametrize(
    "tool_name",
    ["probe", "connect_new_datasource"],
    ids=["general-branch", "interactive-branch"],
)
async def test_the_dispatch_loop_drains_what_the_helper_left_behind(
    make_session, tool_name
):
    """The end-of-turn drain after `agen.aclose()`, at both call sites.

    A stub helper stands in for the reachable-tomorrow case: a tool that queues
    an event after the helper has already handed over the result (a background
    task outliving its tool call). No tool does that today — every `ask_user`
    emit happens inside `elicit()` before the handler returns, and a
    `generate_artifact` sub-agent runs inside the drained window too — so the
    stub is the only way to reach the window at all. Without the drain the event
    sits in the queue until the next tool's drain, or forever if this was the
    last tool of the turn, leaving a published card that is never retired.
    """
    session = make_session()
    session.tool_registry.register_tool(
        ToolDef(name=tool_name, description="", input_schema={}, handler=None)
    )
    _script_one_tool_call(session, tool_name=tool_name)

    async def _stub_draining(tc):
        yield ("result", "ok")
        # Queued after the result was handed over — past the helper's own tail
        # drain, so only the caller's drain can still deliver it.
        session.emitter._queue.put_nowait("left-behind")

    session._dispatch_draining = _stub_draining

    events = [ev async for ev in session.turn_stream("go")]
    assert "left-behind" in events, (
        "an event left in the queue after the helper returned was never "
        f"forwarded: {[type(e).__name__ for e in events]!r}"
    )
