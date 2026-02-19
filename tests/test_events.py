from __future__ import annotations

import asyncio

from anton.events.bus import EventBus
from anton.events.types import (
    Phase,
    PromptUser,
    StatusUpdate,
    TaskComplete,
    TaskFailed,
)


class TestPhaseEnum:
    def test_all_phases_exist(self):
        assert Phase.PLANNING == "planning"
        assert Phase.SKILL_DISCOVERY == "skill_discovery"
        assert Phase.EXECUTING == "executing"
        assert Phase.COMPLETE == "complete"
        assert Phase.FAILED == "failed"

    def test_phase_count(self):
        assert len(Phase) == 7


class TestEventModels:
    def test_status_update_defaults(self):
        e = StatusUpdate(phase=Phase.PLANNING, message="working")
        assert e.type == "status_update"
        assert e.eta_seconds is None

    def test_status_update_with_eta(self):
        e = StatusUpdate(phase=Phase.EXECUTING, message="step", eta_seconds=10.5)
        assert e.eta_seconds == 10.5

    def test_task_complete(self):
        e = TaskComplete(summary="done")
        assert e.type == "task_complete"
        assert e.summary == "done"

    def test_task_failed(self):
        e = TaskFailed(error_summary="oops")
        assert e.type == "task_failed"
        assert e.error_summary == "oops"

    def test_prompt_user(self):
        e = PromptUser(question="continue?")
        assert e.type == "prompt_user"
        assert e.question == "continue?"


class TestEventBus:
    async def test_publish_to_subscriber(self):
        bus = EventBus()
        q = bus.subscribe()
        event = StatusUpdate(phase=Phase.PLANNING, message="hi")
        await bus.publish(event)
        received = q.get_nowait()
        assert received == event

    async def test_multiple_subscribers(self):
        bus = EventBus()
        q1 = bus.subscribe()
        q2 = bus.subscribe()
        event = TaskComplete(summary="ok")
        await bus.publish(event)
        assert q1.get_nowait() == event
        assert q2.get_nowait() == event

    async def test_unsubscribe(self):
        bus = EventBus()
        q = bus.subscribe()
        bus.unsubscribe(q)
        event = StatusUpdate(phase=Phase.PLANNING, message="hi")
        await bus.publish(event)
        assert q.empty()

    async def test_publish_no_subscribers(self):
        bus = EventBus()
        # Should not raise
        await bus.publish(StatusUpdate(phase=Phase.PLANNING, message="no one listening"))
