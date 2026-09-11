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
from anton.core.llm.provider import (
    UNNAMED_REPLAYED_TOOL,
    StreamComplete,
    ToolCall,
    ensure_replayable_tool_call,
    usable_call_id,
)
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
        """Both fields are empty on this shape, and BOTH are salvaged.

        Dropping the call looked tidier and was worse: the streaming paths do
        not call `raise_on_empty_response`, so a dropped sole call ends the
        turn silently. Renamed, it fails as an ordinary `Tool ... not found`
        tool_result the model re-emits from inside the same turn.
        """
        calls = _completed(await _run_responses_stream(_args_events())).tool_calls
        assert len(calls) == 1
        assert calls[0].id and calls[0].name == UNNAMED_REPLAYED_TOOL

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
        calls = _completed(await _run_responses_stream(events)).tool_calls
        assert len(calls) == 1
        assert calls[0].id and calls[0].name == UNNAMED_REPLAYED_TOOL

    async def test_chat_completions_non_streaming_with_a_blank_id(self):
        """The SDK types `id` as a required `str`, which "" satisfies."""
        tc = SimpleNamespace(
            id="", function=SimpleNamespace(name="scratchpad", arguments='{"a": 1}')
        )
        message = MagicMock()
        message.content = ""
        message.tool_calls = [tc]
        usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        response = MagicMock(
            choices=[MagicMock(message=message, finish_reason="tool_calls")], usage=usage
        )
        with patch("anton.core.llm.openai.openai") as mock_openai:
            client = AsyncMock()
            mock_openai.AsyncOpenAI.return_value = client
            client.chat.completions.create = AsyncMock(return_value=response)
            provider = OpenAIProvider(
                api_key="k", flavor=OpenAIProvider.FLAVOR_OPENAI_COMPATIBLE_GENERIC
            )
            out = await provider.complete(
                model="m", system="s", messages=[{"role": "user", "content": "hi"}],
                tools=[{"name": "scratchpad", "description": "x", "input_schema": {}}],
            )
        assert len(out.tool_calls) == 1 and out.tool_calls[0].id

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


class TestEnsureReplayableToolCall:
    def test_named_call_keeps_its_input_and_gains_an_id(self):
        call = ensure_replayable_tool_call(ToolCall(id="", name="scratchpad", input={"a": 1}))
        assert call.id and call.name == "scratchpad" and call.input == {"a": 1}

    def test_a_good_id_with_no_name_is_renamed_not_dropped(self):
        """Pre-fix this call was dispatched and came back as an ordinary
        "tool not found" result the model recovers from inside the turn.
        Dropping it was a behaviour change beyond ENG-2420 and a worse failure
        than the one being fixed (review: pnewsam on #471)."""
        call = ensure_replayable_tool_call(ToolCall(id="call_ok", name="", input={"a": 1}))
        assert call.id == "call_ok" and call.name == UNNAMED_REPLAYED_TOOL

    def test_a_non_string_id_is_not_merely_falsy_checked(self):
        """`name` is deliberately GOOD here: with an empty name this exits
        through the name branch and never exercises the id check at all, which
        is how a bare `bool(value)` mutation stayed green (review: pnewsam)."""
        call = ensure_replayable_tool_call(ToolCall(id={"oops": 1}, name="scratchpad", input={}))
        assert isinstance(call.id, str) and call.id.startswith("call_anton_")

    def test_usable_call_id_rejects_non_strings_directly(self):
        assert usable_call_id("call_x") is True
        assert usable_call_id("") is False
        assert usable_call_id({"nested": 1}) is False
        assert usable_call_id(None) is False
        assert usable_call_id(123) is False


class TestAnthropicReaderIsGuardedToo:
    """The Anthropic reader has no empty-id BUFFER path, but writes `block.id`
    straight through — so a relay sending a blank id poisons history the same
    way. Neither of these sites was pinned by any test (review: pnewsam)."""

    async def test_non_streaming_block_with_a_blank_id(self):
        from anton.core.llm.anthropic import AnthropicProvider

        block = SimpleNamespace(type="tool_use", id="", name="scratchpad", input={"a": 1})
        response = SimpleNamespace(
            content=[block], stop_reason="tool_use", model="claude",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        with patch("anton.core.llm.anthropic.anthropic") as mock_anthropic:
            client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = client
            client.messages.create = AsyncMock(return_value=response)
            provider = AnthropicProvider(api_key="k")
            out = await provider.complete(
                model="claude", system="s", messages=[{"role": "user", "content": "hi"}],
            )
        assert len(out.tool_calls) == 1 and out.tool_calls[0].id

    async def test_streaming_block_with_a_blank_id_and_no_orphan_ui_step(self):
        from anton.core.llm.anthropic import AnthropicProvider
        from anton.core.llm.provider import StreamToolUseEnd, StreamToolUseStart

        events = [
            SimpleNamespace(type="message_start", message=SimpleNamespace(
                model="claude",
                usage=SimpleNamespace(input_tokens=1, output_tokens=0,
                                      cache_read_input_tokens=0, cache_creation_input_tokens=0))),
            SimpleNamespace(type="content_block_start", index=0, content_block=SimpleNamespace(
                type="tool_use", id="", name="scratchpad")),
            SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(
                type="input_json_delta", partial_json='{"a": 1}')),
            SimpleNamespace(type="content_block_stop", index=0),
            SimpleNamespace(type="message_delta",
                            delta=SimpleNamespace(stop_reason="tool_use"),
                            usage=SimpleNamespace(output_tokens=1)),
        ]
        with patch("anton.core.llm.anthropic.anthropic") as mock_anthropic:
            client = AsyncMock()
            mock_anthropic.AsyncAnthropic.return_value = client
            # Reuses the canonical stand-in: `messages.stream()` is NOT
            # awaited, it is used as `async with ... as stream`.
            from tests.test_provider import _FakeAnthropicStream

            client.messages.stream = MagicMock(return_value=_FakeAnthropicStream(events))
            provider = AnthropicProvider(api_key="k")
            out = [e async for e in provider.stream(
                model="claude", system="s", messages=[{"role": "user", "content": "hi"}])]

        calls = _completed(out).tool_calls
        assert len(calls) == 1 and calls[0].id
        # A Start keyed "" could never be retired: the marker that closes a step
        # carries the call's id, which is now the minted one.
        assert not [e for e in out if isinstance(e, StreamToolUseStart)]
        assert not [e for e in out if isinstance(e, StreamToolUseEnd)]


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
        # 3, not 2: the fixture's call has an empty id AND an empty name, and
        # the name is now repaired as its own concern (review: pnewsam on #471).
        assert repaired == 3
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

    def test_two_separate_loads_produce_byte_identical_history(self):
        """The repair is re-applied on EVERY load and never written back —
        cowork-server leaves `history_store` unset and persists only
        `session.history[seed_len:]`, so the repaired seed never reaches the
        messages table. A random minted id would therefore change the replayed
        prefix every turn and miss the prompt cache on the whole conversation,
        forever, for exactly the conversations being healed.

        Note this is a strictly stronger property than `test_repair_is_idempotent`
        below, which re-repairs an ALREADY-repaired list and so cannot catch a
        non-deterministic mint.
        """
        import json

        first, _ = repair_replayed_tool_ids(self._poisoned())
        second, _ = repair_replayed_tool_ids(self._poisoned())
        assert json.dumps(first) == json.dumps(second)

    def test_two_identical_calls_still_get_distinct_ids(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "", "name": "scratchpad", "input": {"a": 1}},
                {"type": "tool_use", "id": "", "name": "scratchpad", "input": {"a": 1}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "", "content": "x"},
                {"type": "tool_result", "tool_use_id": "", "content": "y"}]},
        ]
        fixed, _ = repair_replayed_tool_ids(history)
        ids = [b["id"] for b in fixed[1]["content"]]
        assert len(set(ids)) == 2
        assert ids == [b["tool_use_id"] for b in fixed[2]["content"]]

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


class TestABadNameIsRepairedToo:
    """The second poisoned shape: a usable id with a blank name.

    Pre-fix anton wrote this whenever a provider sent a good `call_id` with no
    name. The first revision of this fix selected on the id alone, so the shape
    passed through all three layers untouched and reached the wire as
    `name: ''` (review: pnewsam on #471).
    """

    def test_a_stored_block_with_a_good_id_and_no_name_is_renamed(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_x", "name": "", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_x", "content": "nope"}]},
        ]
        fixed, repaired = repair_replayed_tool_ids(history)
        assert repaired == 1
        assert fixed[1]["content"][0]["name"] == UNNAMED_REPLAYED_TOOL
        # The id was fine and must be left exactly as it was — re-keying it
        # would needlessly change the replayed prefix.
        assert fixed[1]["content"][0]["id"] == "call_x"
        wire = _translate_messages_to_responses_input(fixed)
        assert not [i for i in wire if i.get("name") == ""]

    def test_a_non_dict_message_after_a_broken_call_does_not_raise(self):
        """Every other message access in the repair is isinstance-guarded;
        this one was not, and raised out of `ChatSession.__init__`, which is
        strictly worse than the 400 being fixed (review: pnewsam on #471)."""
        history = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "", "name": "scratchpad", "input": {}}]},
            "not a dict at all",
        ]
        fixed, _ = repair_replayed_tool_ids(history)  # must not raise
        assert len(fixed) == 2


class TestTheWiringIsPinned:
    """Each layer past the readers, asserted at its real call site.

    Reverting any of these to its pre-PR form left the full suite green, which
    is the same structural hole ENG-1808 came from (review: pnewsam on #471).
    """

    @staticmethod
    def _session():
        from anton.core.session import ChatSession, ChatSessionConfig
        from tests.conftest import make_mock_llm

        poisoned = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "", "name": "", "input": {"a": 1}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "", "content": "x"}]},
        ]
        return ChatSession(ChatSessionConfig(
            llm_client=make_mock_llm(), initial_history=poisoned,
        )), poisoned

    def test_constructing_a_session_repairs_poisoned_history(self):
        """The ingress repair is only reachable through `ChatSession.__init__`,
        and nothing constructed a session with poisoned history."""
        session, poisoned = self._session()
        block = session.history[1]["content"][0]
        assert block["id"] and block["name"] == UNNAMED_REPLAYED_TOOL
        assert session.history[2]["content"][0]["tool_use_id"] == block["id"]
        # Count-preserving, so `_seed_len` and the host's positional
        # `covered_through` mapping still line up.
        assert len(session.history) == len(poisoned)
        assert session._seed_len == len(poisoned)

    def test_the_session_belt_salvages_rather_than_empties(self):
        from anton.core.llm.provider import LLMResponse

        session, _ = self._session()
        response = LLMResponse(
            content="", tool_calls=[ToolCall(id="", name="", input={"a": 1})],
        )
        session._ensure_replayable_calls(response)
        # Salvaged, NOT dropped: emptying the list would leave the round with
        # no calls and no content, so both `_append_history` calls become
        # no-ops and the loop re-issues an identical request until the round
        # cap (review: pnewsam on #471).
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id and response.tool_calls[0].name

    def test_the_tool_loop_calls_the_belt(self):
        """AST, not behaviour: the belt is expected to be a permanent no-op, so
        only its presence at the call site can be asserted. Mirrors
        `tests/test_verifier_truncation.py`'s wiring pins."""
        import ast
        import inspect

        from anton.core.session import ChatSession

        src = inspect.getsource(ChatSession._stream_and_handle_tools)
        called = {
            n.func.attr
            for n in ast.walk(ast.parse(textwrap_dedent(src)))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert "_ensure_replayable_calls" in called

    def test_the_scratchpad_boot_loop_calls_its_belt(self):
        """AST for the same reason the sibling wiring tests use it: importing
        `scratchpad_boot` reads stdin at import time."""
        import ast
        import pathlib as _pathlib

        import anton

        src = (_pathlib.Path(anton.__file__).parent
               / "core" / "backends" / "scratchpad_boot.py").read_text()
        called = {
            n.func.id
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "_ensure_replayable_calls" in called


def textwrap_dedent(src: str) -> str:
    import textwrap

    return textwrap.dedent(src)
