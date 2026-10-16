"""A hard truncation inside a tool loop keeps the latest tool round.

`plan_stream_with_recovery` and `plan_with_recovery` hard-truncate history
after a second context overflow, then retry the planning call. In a tool loop
the last messages are all tool rows: an assistant tool_use, then the user
tool_result that answers it. `hard_truncate_history` puts its placeholder, a
user message, first and keeps a leading tool_use right after it. The retried
request then ends with the latest tool_result. These tests pin that the request
passes the checks of Anthropic, OpenAI Chat Completions and the OpenAI
Responses API.

The Anthropic adapter sends anton's messages unchanged, so the Anthropic check
runs on them as they are. The OpenAI adapter translates them, so the Chat
Completions and Responses checks run on its translations.
"""

from __future__ import annotations

import copy
from unittest.mock import AsyncMock

import pytest

from anton.core.llm.openai import _translate_messages, _translate_messages_to_responses_input
from anton.core.llm.provider import (
    ContextOverflowError,
    LLMResponse,
    StreamContextCompacted,
    Usage,
)
from anton.core.session import ChatSession, ChatSessionConfig
from tests.conftest import make_mock_llm
from tests.test_turn_rows_through_compaction import (
    ANSWER,
    PlanCalls,
    _run,
    _tool_result,
    _tool_use,
)
from tests.test_turn_rows_through_compaction import _session as _scripted_session

PLACEHOLDER_PREFIX = "[Earlier conversation was truncated"


def _block_ids(message: dict | None, *, block_type: str, key: str) -> set[str]:
    if message is None or not isinstance(message.get("content"), list):
        return set()
    return {
        block[key]
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == block_type
    }


def _anthropic_violations(messages: list[dict]) -> list[str]:
    """What the Anthropic Messages API rejects in a request's messages.

    The first message is the user's. Every tool_use is answered by a
    tool_result in the next message, and every tool_result answers a tool_use
    in the message before it. The last message is the user's: Anthropic models
    from 4.6 on reject a final assistant message as a prefill. Roles alternate,
    as anton's own pre-flight check expects.
    """
    if not messages:
        return ["no messages"]
    problems: list[str] = []
    if messages[0]["role"] != "user":
        problems.append(f"the first message is {messages[0]['role']}")
    if messages[-1]["role"] != "user":
        problems.append(f"the last message is {messages[-1]['role']}")
    for i, message in enumerate(messages):
        before = messages[i - 1] if i > 0 else None
        after = messages[i + 1] if i + 1 < len(messages) else None
        if before is not None and before["role"] == message["role"]:
            problems.append(f"two {message['role']} messages at {i - 1} and {i}")
        unanswered = _block_ids(message, block_type="tool_use", key="id") - _block_ids(
            after, block_type="tool_result", key="tool_use_id"
        )
        if unanswered:
            problems.append(f"tool_use {sorted(unanswered)} at {i} has no tool_result after it")
        orphaned = _block_ids(message, block_type="tool_result", key="tool_use_id") - _block_ids(
            before, block_type="tool_use", key="id"
        )
        if orphaned:
            problems.append(f"tool_result {sorted(orphaned)} at {i} has no tool_use before it")
    return problems


def _chat_completions_violations(messages: list[dict]) -> list[str]:
    """What OpenAI Chat Completions rejects in a request anton translated.

    An assistant message with tool_calls is followed by one tool message per
    call, and every tool message answers a call of that assistant message.
    The first message after the system prompt is the user's.
    """
    problems: list[str] = []
    conversation = [m for m in messages if m["role"] != "system"]
    if not conversation or conversation[0]["role"] != "user":
        problems.append("the first message after the system prompt is not the user's")
    pending: set[str] = set()
    for i, message in enumerate(conversation):
        if message["role"] == "tool":
            if message["tool_call_id"] not in pending:
                problems.append(f"tool message {message['tool_call_id']} at {i} answers no call")
            pending.discard(message["tool_call_id"])
            continue
        if pending:
            problems.append(f"tool_calls {sorted(pending)} have no tool message before {i}")
        pending = {call["id"] for call in message.get("tool_calls") or []}
    if pending:
        problems.append(f"tool_calls {sorted(pending)} have no tool message")
    return problems


def _responses_violations(items: list[dict]) -> list[str]:
    """What the OpenAI Responses API rejects in input anton translated.

    Every function_call_output answers an earlier function_call, and every
    function_call has an output.
    """
    problems: list[str] = []
    called: set[str] = set()
    answered: set[str] = set()
    for i, item in enumerate(items):
        if item.get("type") == "function_call":
            called.add(item["call_id"])
        elif item.get("type") == "function_call_output":
            if item["call_id"] not in called:
                problems.append(f"function_call_output {item['call_id']} at {i} answers no call")
            answered.add(item["call_id"])
    if called - answered:
        problems.append(f"function_call {sorted(called - answered)} has no output")
    return problems


def _provider_violations(messages: list[dict]) -> dict[str, list[str]]:
    return {
        "anthropic": _anthropic_violations(messages),
        "chat_completions": _chat_completions_violations(_translate_messages("system", messages)),
        "responses": _responses_violations(_translate_messages_to_responses_input(messages)),
    }


_VALID = {"anthropic": [], "chat_completions": [], "responses": []}


def _tool_rows(messages: list[dict]) -> list[tuple[str, str]]:
    """(role, tool id) for every tool_use and tool_result block, in order."""
    return [
        (message["role"], block.get("id") or block.get("tool_use_id"))
        for message in messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") in ("tool_use", "tool_result")
    ]


def _tool_loop(*tids: str) -> list[dict]:
    """A question, then one tool round per id, ending with the last tool_result."""
    history: list[dict] = [{"role": "user", "content": "How many lines does the plant run?"}]
    for tid in tids:
        history += [_tool_use(tid), _tool_result(tid)]
    return history


# ── a real turn ──────────────────────────────────────────────────────────────


async def test_the_retry_after_a_hard_truncation_in_a_tool_loop_is_a_valid_request():
    calls = PlanCalls()
    session = _scripted_session(mode="hard-truncate", calls=calls)
    events = await _run(session)

    # Precondition: the planning call that returns toolu_4 overflowed twice,
    # so the summary and then the hard truncation both ran.
    assert len(calls.overflowed) == 2
    assert sum(isinstance(e, StreamContextCompacted) for e in events) == 2
    retried = calls.requests[calls.overflowed[-1] + 1]
    assert retried[0]["content"].startswith(PLACEHOLDER_PREFIX)

    assert _provider_violations(retried) == _VALID
    # The round the retried call answers is still there, and is the request's end.
    assert _tool_rows(retried)[-2:] == [("assistant", "toolu_3"), ("user", "toolu_3")]
    # The turn goes on to run toolu_4 and answer.
    assert _tool_rows(session.history)[-2:] == [("assistant", "toolu_4"), ("user", "toolu_4")]
    assert session.history[-1] == {"role": "assistant", "content": ANSWER}
    for request in calls.requests[calls.overflowed[-1] + 1:]:
        assert _provider_violations(request) == _VALID


async def test_the_non_streaming_retry_after_a_hard_truncation_is_a_valid_request():
    llm = make_mock_llm()
    session = ChatSession(ChatSessionConfig(
        llm_client=llm, initial_history=_tool_loop("toolu_1", "toolu_2", "toolu_3"),
    ))
    session._summarize_history = AsyncMock(return_value=False)
    requests: list[list[dict]] = []

    async def plan(messages, **kwargs):
        requests.append(copy.deepcopy(messages))
        if len(requests) < 3:
            raise ContextOverflowError("too long")
        return LLMResponse(content=ANSWER, usage=Usage(input_tokens=100), stop_reason="end_turn")

    llm.plan = plan
    await session.plan_with_recovery(system="system")

    retried = requests[-1]
    assert retried[0]["content"].startswith(PLACEHOLDER_PREFIX)
    assert _provider_violations(retried) == _VALID
    assert _tool_rows(retried)[-2:] == [("assistant", "toolu_3"), ("user", "toolu_3")]


# ── hard_truncate_history on its own ─────────────────────────────────────────


def _session(*, history: list[dict]) -> ChatSession:
    return ChatSession(ChatSessionConfig(llm_client=make_mock_llm(), initial_history=history))


def test_a_tail_of_tool_rounds_is_kept_whole():
    session = _session(history=_tool_loop("toolu_1", "toolu_2", "toolu_3"))
    session.hard_truncate_history(keep=4)

    assert session.history[0]["content"].startswith(PLACEHOLDER_PREFIX)
    # No "Understood." separator: the placeholder is a user message, so the
    # kept tool_use can answer it directly.
    assert session.history[1:] == [
        _tool_use("toolu_2"), _tool_result("toolu_2"),
        _tool_use("toolu_3"), _tool_result("toolu_3"),
    ]
    assert _provider_violations(session.history) == _VALID


def test_a_tool_result_cut_from_its_tool_use_is_still_dropped():
    """An odd cut lands on a tool_result whose tool_use is in the dropped part.
    That result has nothing to answer, so it goes, and the round after it stays."""
    session = _session(history=_tool_loop("toolu_1", "toolu_2", "toolu_3"))
    session.hard_truncate_history(keep=3)

    assert session.history[1:] == [_tool_use("toolu_3"), _tool_result("toolu_3")]
    assert _provider_violations(session.history) == _VALID


def test_a_truncation_during_a_tool_call_keeps_the_call_being_run():
    """A truncation from inside a tool handler runs while the assistant message
    with the running call is last; its tool_result is appended after."""
    history = [*_tool_loop("toolu_1", "toolu_2"), _tool_use("toolu_3")]
    session = _session(history=history)
    session.hard_truncate_history(keep=4)

    assert session.history[1:] == [_tool_use("toolu_2"), _tool_result("toolu_2"), _tool_use("toolu_3")]
    session._append_history(_tool_result("toolu_3"))
    assert _provider_violations(session.history) == _VALID


@pytest.mark.parametrize("rounds", [2, 3, 5])
def test_the_truncated_tool_loop_ends_with_its_latest_tool_result(rounds):
    tids = [f"toolu_{n}" for n in range(1, rounds + 1)]
    session = _session(history=_tool_loop(*tids))
    session.hard_truncate_history()

    assert _provider_violations(session.history) == _VALID
    assert session.history[-2:] == [_tool_use(tids[-1]), _tool_result(tids[-1])]
