"""A call's wait note leads the host's still-working line."""
from __future__ import annotations

from anton.core.llm.client import LLMClient
from anton.core.llm.liveness import ModelCallTracker, arm_turn_tracker, disarm_turn_tracker
from anton.core.llm.provider import LLMProvider, LLMResponse, StreamComplete, Usage


def test_snapshot_message_starts_with_the_note():
    tracker = ModelCallTracker()
    call = tracker.open(role="planning", idle_timeout_s=None, note="The answer was cut off — trying again")
    call.awaiting = True
    message = tracker.snapshot().message
    assert message.startswith("The answer was cut off — trying again — waiting for the model (")


def test_snapshot_without_a_note_is_unchanged():
    tracker = ModelCallTracker()
    call = tracker.open(role="planning", idle_timeout_s=None)
    call.awaiting = True
    assert tracker.snapshot().message.startswith("Waiting for the model (")


class _PeekingProvider(LLMProvider):
    name = "peek"

    def __init__(self, tracker):
        self.tracker = tracker
        self.messages: list[str] = []

    async def complete(self, **kw):
        return LLMResponse(content="ok", usage=Usage(output_tokens=1))

    async def stream(self, **kw):
        self.messages.append(self.tracker.snapshot().message)
        yield StreamComplete(response=LLMResponse(content="ok", usage=Usage(output_tokens=1)))


async def test_plan_and_code_stream_pass_the_note_to_the_tracker():
    tracker = ModelCallTracker()
    provider = _PeekingProvider(tracker)
    llm = LLMClient(planning_provider=provider, planning_model="p", coding_provider=provider, coding_model="c")
    token = arm_turn_tracker(tracker)
    try:
        async for _ in llm.plan_stream(system="s", messages=[], wait_note="Note A"):
            pass
        async for _ in llm.code_stream(system="s", messages=[], wait_note="Note B"):
            pass
        async for _ in llm.plan_stream(system="s", messages=[]):
            pass
    finally:
        tracker.close()
        disarm_turn_tracker(token)
    assert provider.messages[0].startswith("Note A — waiting for the model")
    assert provider.messages[1].startswith("Note B — waiting for the model")
    assert provider.messages[2].startswith("Waiting for the model")
