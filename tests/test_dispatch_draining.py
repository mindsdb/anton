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
    # No slack: `before` was taken with only the test's own task alive, and
    # both of the helper's futures are now accounted for above.
    assert len(asyncio.all_tasks()) == before
