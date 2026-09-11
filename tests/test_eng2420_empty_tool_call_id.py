"""ENG-2420 — an empty tool-call id must never reach history.

A `ToolCall` with `id=""` is written into `session.history`, serialised back
out on the next request as `"call_id": ""`, and the provider then rejects the
WHOLE request (`400 Invalid 'input[N].call_id': empty string`) on that turn and
on every later turn of the conversation. Nothing recovers it: the only trimming
path runs on a caught `ContextOverflowError` or on pressure from a successful
response's usage, and a 400 produces neither.

These tests drive the readers with the provider shapes that produce it, and the
repair with a conversation that was already poisoned before the fix shipped.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.core.llm.openai import (
    OpenAIProvider,
    _parse_response_object,
    _translate_messages_to_responses_input,
)
from anton.core.llm.provider import StreamComplete, ToolCall, replayable_tool_call
from anton.memory.history_store import is_user_turn, repair_replayed_tool_ids


async def _fake_async_iter(items):
    for item in items:
        yield item


def _completed(events):
    return [e for e in events if isinstance(e, StreamComplete)][-1].response


async def _run_responses_stream(events, flavor=OpenAIProvider.FLAVOR_OPENAI):
    with patch("anton.core.llm.openai.openai") as mock_openai:
        client = AsyncMock()
        mock_openai.AsyncOpenAI.return_value = client
        client.responses.create = AsyncMock(return_value=_fake_async_iter(events))
        provider = OpenAIProvider(api_key="k", flavor=flavor)
        return [
            e
            async for e in provider.stream(
                model="gpt-5",
                system="s",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"name": "scratchpad", "description": "x", "input_schema": {}}],
            )
        ]


def _args_events(index=0):
    return [
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            output_index=index,
            delta='{"action": "exec"}',
        ),
        SimpleNamespace(type="response.function_call_arguments.done", output_index=index),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(usage=None, status="completed", model="gpt-5"),
        ),
    ]


class TestReadersNeverEmitAnUnreplayableCall:
    """Three provider shapes reach the empty id, not just the filed one.

    Only the first is what ENG-2420 describes. The other two never touch the
    `output_item.added`-missing buffer path, so a guard written for that path
    alone would miss them.
    """

    async def test_arguments_arrive_with_no_output_item_added(self):
        response = _completed(await _run_responses_stream(_args_events()))
        # Name is empty too on this shape — there is no tool to dispatch, so
        # the call is dropped rather than given a synthetic handle.
        assert response.tool_calls == []

    async def test_output_item_added_carries_a_blank_call_id(self):
        events = [
            SimpleNamespace(
                type="response.output_item.added",
                output_index=0,
                item=SimpleNamespace(type="function_call", call_id="", name="scratchpad"),
            ),
            *_args_events(),
        ]
        calls = _completed(await _run_responses_stream(events)).tool_calls
        # Name survived, so the call is salvageable: mint a handle and let it run.
        assert len(calls) == 1
        assert calls[0].name == "scratchpad"
        assert calls[0].id, "an id must have been minted"
        assert calls[0].input == {"action": "exec"}

    async def test_item_is_not_labelled_function_call(self):
        events = [
            SimpleNamespace(
                type="response.output_item.added",
                output_index=0,
                item=SimpleNamespace(type="custom_tool_call", call_id="c", name="scratchpad"),
            ),
            *_args_events(),
        ]
        assert _completed(await _run_responses_stream(events)).tool_calls == []

    async def test_healthy_call_is_untouched(self):
        events = [
            SimpleNamespace(
                type="response.output_item.added",
                output_index=0,
                item=SimpleNamespace(type="function_call", call_id="call_ok", name="scratchpad"),
            ),
            *_args_events(),
        ]
        calls = _completed(await _run_responses_stream(events)).tool_calls
        assert [(c.id, c.name) for c in calls] == [("call_ok", "scratchpad")]

    async def test_chat_completions_stream_where_the_id_never_arrives(self):
        """The gateway population runs this transport, not the Responses API."""

        def chunk(tool_calls, finish=None):
            return SimpleNamespace(
                model="m",
                usage=None,
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None, tool_calls=tool_calls, reasoning_content=None
                        ),
                        finish_reason=finish,
                    )
                ],
            )

        delta = SimpleNamespace(
            index=0,
            id=None,
            function=SimpleNamespace(name="scratchpad", arguments='{"a": 1}'),
        )
        with patch("anton.core.llm.openai.openai") as mock_openai:
            client = AsyncMock()
            mock_openai.AsyncOpenAI.return_value = client
            client.chat.completions.create = AsyncMock(
                return_value=_fake_async_iter([chunk([delta]), chunk(None, finish="tool_calls")])
            )
            provider = OpenAIProvider(
                api_key="k", flavor=OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC
            )
            events = [
                e
                async for e in provider.stream(
                    model="m",
                    system="s",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=[{"name": "scratchpad", "description": "x", "input_schema": {}}],
                )
            ]
        calls = _completed(events).tool_calls
        assert len(calls) == 1 and calls[0].id and calls[0].name == "scratchpad"

    def test_non_streaming_responses_item_with_no_call_id(self):
        item = SimpleNamespace(type="function_call", name="scratchpad", arguments='{"a": 1}')
        response = SimpleNamespace(
            output=[item], usage=None, status="completed", model="gpt-5"
        )
        calls = _parse_response_object(response, "gpt-5").tool_calls
        assert len(calls) == 1 and calls[0].id


class TestReplayableToolCall:
    def test_unnamed_call_is_dropped(self):
        assert replayable_tool_call(ToolCall(id="", name="", input={})) is None

    def test_named_call_keeps_its_input_and_gains_an_id(self):
        call = replayable_tool_call(ToolCall(id="", name="scratchpad", input={"a": 1}))
        assert call is not None and call.id and call.input == {"a": 1}

    def test_a_non_string_id_is_not_merely_falsy_checked(self):
        assert replayable_tool_call(ToolCall(id={"oops": 1}, name="", input={})) is None


class TestRepairOfAnAlreadyPoisonedConversation:
    """The bad ids are already in users' stored histories.

    Every assertion here is a constraint the repair may not break, and each one
    is a bug that a simpler repair actually causes.
    """

    @staticmethod
    def _poisoned():
        return [
            {"role": "user", "content": "[2026-09-01 10:00] load my csv"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "On it."},
                    {"type": "tool_use", "id": "", "name": "", "input": {"action": "exec"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "", "content": "Tool '' not found"}
                ],
            },
            {"role": "user", "content": "[2026-09-01 10:02] hello? are you there"},
        ]

    def test_nothing_unreplayable_survives_onto_the_wire(self):
        fixed, repaired = repair_replayed_tool_ids(self._poisoned())
        assert repaired == 2
        wire = _translate_messages_to_responses_input(fixed)
        assert not [i for i in wire if i.get("call_id") == "" or i.get("name") == ""]

    def test_message_count_is_preserved(self):
        """`last_compaction.covered_through` is a POSITIONAL count into
        `initial_history` that the host maps onto its own message list
        (cowork-server's `_persist_history_compaction` does
        `tail_start + covered - 1`). Removing a message here would move a saved
        compaction cutoff onto the wrong message and silently drop or duplicate
        real history on the next turn."""
        poisoned = self._poisoned()
        fixed, _ = repair_replayed_tool_ids(poisoned)
        assert len(fixed) == len(poisoned)

    def test_a_tool_reply_is_still_not_counted_as_a_user_turn(self):
        """Rewriting the reply to plain text inflates the session's turn count
        and leaves two consecutive user messages in exactly this shape."""
        poisoned = self._poisoned()
        fixed, _ = repair_replayed_tool_ids(poisoned)
        assert sum(1 for m in fixed if is_user_turn(m)) == sum(
            1 for m in poisoned if is_user_turn(m)
        )

    def test_no_new_consecutive_same_role_pair(self):
        poisoned = self._poisoned()
        fixed, _ = repair_replayed_tool_ids(poisoned)
        before = [i for i in range(1, len(poisoned)) if poisoned[i]["role"] == poisoned[i - 1]["role"]]
        after = [i for i in range(1, len(fixed)) if fixed[i]["role"] == fixed[i - 1]["role"]]
        assert after == before

    def test_the_pair_stays_paired(self):
        fixed, _ = repair_replayed_tool_ids(self._poisoned())
        assert fixed[1]["content"][1]["id"] == fixed[2]["content"][0]["tool_use_id"]

    def test_the_callers_list_is_not_mutated(self):
        poisoned = self._poisoned()
        snapshot = copy.deepcopy(poisoned)
        repair_replayed_tool_ids(poisoned)
        assert poisoned == snapshot

    def test_repair_is_idempotent(self):
        fixed, _ = repair_replayed_tool_ids(self._poisoned())
        again, repaired = repair_replayed_tool_ids(fixed)
        assert repaired == 0 and again == fixed

    def test_healthy_history_is_untouched(self):
        healthy = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_ok", "name": "scratchpad", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_ok", "content": "done"}
                ],
            },
        ]
        out, repaired = repair_replayed_tool_ids(copy.deepcopy(healthy))
        assert repaired == 0 and out == healthy

    def test_a_call_with_nothing_to_pair_with_becomes_text_not_an_orphan(self):
        """A minted id with no reply would be an orphan `tool_use`, which is
        its own 400 on Anthropic. Block count is still preserved."""
        history = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "", "name": "scratchpad", "input": {}}],
            },
        ]
        fixed, repaired = repair_replayed_tool_ids(history)
        assert repaired == 1
        assert len(fixed) == 2 and len(fixed[1]["content"]) == 1
        assert fixed[1]["content"][0]["type"] == "text"
