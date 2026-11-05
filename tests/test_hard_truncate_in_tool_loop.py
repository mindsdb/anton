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
    context_window,
)
from anton.core.session import ChatSession, ChatSessionConfig
from tests._turn_rows_helpers import (
    ANSWER,
    PAGES,
    TOO_LARGE_PAGE,
    TOOL_IDS,
    PlanCalls,
    run_turn,
    scripted_session,
    tool_result,
    tool_use,
)
from tests.conftest import make_mock_llm

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


def _result_content(*, messages: list[dict], tid: str) -> str | list[dict]:
    """The content of the tool_result in `messages` that answers `tid`."""
    for message in messages:
        for block in message["content"] if isinstance(message.get("content"), list) else []:
            if isinstance(block, dict) and block.get("tool_use_id") == tid:
                return block["content"]
    raise AssertionError(f"no tool_result for {tid}")


def _tool_loop(*tids: str) -> list[dict]:
    """A question, then one tool round per id, ending with the last tool_result."""
    history: list[dict] = [{"role": "user", "content": "How many lines does the plant run?"}]
    for tid in tids:
        history += [tool_use(tid), tool_result(tid)]
    return history


# ── a real turn ──────────────────────────────────────────────────────────────


async def test_the_retry_after_a_hard_truncation_in_a_tool_loop_is_a_valid_request():
    calls = PlanCalls()
    session = scripted_session(mode="hard-truncate", calls=calls)
    events = await run_turn(session)

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


async def _lookup_too_large(session, tc_input):
    return TOO_LARGE_PAGE if tc_input["id"] == "toolu_3" else PAGES[tc_input["id"]]


async def test_a_latest_tool_result_too_large_for_the_window_is_stubbed_and_the_turn_answers():
    """toolu_3's result alone is over the window, so every request that holds
    it overflows. The hard truncation keeps toolu_3's round. Kept whole, the
    result overflows every retry and the turn ends in ContextOverflowError.
    The truncation stubs it instead: the pairing stays valid, and the retry
    fits and asks the model for less."""
    calls = PlanCalls()
    session = scripted_session(mode="too-large", handler=_lookup_too_large, calls=calls)
    await run_turn(session)

    # Precondition: the planning call after toolu_3 overflowed twice, so the
    # hard truncation ran.
    assert len(calls.overflowed) == 2
    retried = calls.requests[calls.overflowed[-1] + 1]
    assert retried[0]["content"].startswith(PLACEHOLDER_PREFIX)

    assert _provider_violations(retried) == _VALID
    assert _tool_rows(retried)[-2:] == [("assistant", "toolu_3"), ("user", "toolu_3")]
    stub = _result_content(messages=retried, tid="toolu_3")
    assert "too large" in stub and "call the tool again" in stub
    # The other kept result fits, so it stays whole.
    assert _result_content(messages=retried, tid="toolu_2") == PAGES["toolu_2"]
    # The turn goes on to run toolu_4 and answer.
    assert _tool_rows(session.history)[-2:] == [("assistant", "toolu_4"), ("user", "toolu_4")]
    assert session.history[-1] == {"role": "assistant", "content": ANSWER}
    for request in calls.requests[calls.overflowed[-1] + 1:]:
        assert _provider_violations(request) == _VALID
    # The turn's messages hold each tool row once.
    assert _tool_rows(session.turn_messages()) == [
        (role, tid) for tid in TOOL_IDS for role in ("assistant", "user")
    ]


# Seed messages long enough that the summary after the first overflow folds
# them away, so the hard truncation after the second follows a real swap.
_LONG_SEED = [
    {"role": "user" if i % 2 == 0 else "assistant", "content": f"seed {i} " + "x" * 60_000}
    for i in range(10)
]


@pytest.mark.parametrize("cut", [False, True])
async def test_a_result_stubbed_right_after_a_summary_is_read_once(cut):
    """On a long seed, the summary after the first overflow swaps history and
    the hard truncation after the second stubs toolu_3's result, with nothing
    appended between. The stub's claim walks past the summary's empty list to
    the one toolu_3's result was appended to, so the turn's messages hold that
    result once, as the stub. With `cut`, a nudge then merges into the stub."""
    session = scripted_session(mode="too-large", cut=cut, handler=_lookup_too_large, seed=_LONG_SEED)
    events = await run_turn(session)

    # Precondition: the summary and the hard truncation both swapped history.
    assert sum(isinstance(e, StreamContextCompacted) for e in events) == 2
    assert session.history[-1] == {"role": "assistant", "content": ANSWER}
    assert _tool_rows(session.turn_messages()) == [
        (role, tid) for tid in TOOL_IDS for role in ("assistant", "user")
    ]
    assert "call the tool again" in _result_content(messages=session.turn_messages(), tid="toolu_3")


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
        tool_use("toolu_2"), tool_result("toolu_2"),
        tool_use("toolu_3"), tool_result("toolu_3"),
    ]
    assert _provider_violations(session.history) == _VALID


def test_a_tool_result_cut_from_its_tool_use_is_still_dropped():
    """An odd cut lands on a tool_result whose tool_use is in the dropped part.
    That result has nothing to answer, so it goes, and the round after it stays."""
    session = _session(history=_tool_loop("toolu_1", "toolu_2", "toolu_3"))
    session.hard_truncate_history(keep=3)

    assert session.history[1:] == [tool_use("toolu_3"), tool_result("toolu_3")]
    assert _provider_violations(session.history) == _VALID


def test_a_truncation_during_a_tool_call_keeps_the_call_being_run():
    """A truncation from inside a tool handler runs while the assistant message
    with the running call is last; its tool_result is appended after."""
    history = [*_tool_loop("toolu_1", "toolu_2"), tool_use("toolu_3")]
    session = _session(history=history)
    session.hard_truncate_history(keep=4)

    assert session.history[1:] == [tool_use("toolu_2"), tool_result("toolu_2"), tool_use("toolu_3")]
    session._append_history(tool_result("toolu_3"))
    assert _provider_violations(session.history) == _VALID


@pytest.mark.parametrize("rounds", [2, 3, 5])
def test_the_truncated_tool_loop_ends_with_its_latest_tool_result(rounds):
    tids = [f"toolu_{n}" for n in range(1, rounds + 1)]
    session = _session(history=_tool_loop(*tids))
    session.hard_truncate_history()

    assert _provider_violations(session.history) == _VALID
    assert session.history[-2:] == [tool_use(tids[-1]), tool_result(tids[-1])]


def _result_of(*, tid: str, content) -> dict:
    return {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tid, "content": content},
    ]}


_BUDGET = context_window("claude-sonnet-4-6")


def test_kept_results_that_fit_the_budget_together_stay_whole():
    """The kept results share one budget, so a result larger than an even
    share of it stays whole while all of them fit together."""
    parallel = ["toolu_3", "toolu_4", "toolu_5", "toolu_6"]
    sizes = dict(zip(parallel, [60_000, 1_000, 1_000, 1_000]))
    history = [
        *_tool_loop("toolu_1", "toolu_2"),
        tool_use(*parallel),
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": "x" * size}
            for tid, size in sizes.items()
        ]},
    ]
    session = _session(history=history)
    session.hard_truncate_history()

    assert _provider_violations(session.history) == _VALID
    for tid, size in sizes.items():
        assert _result_content(messages=session.history, tid=tid) == "x" * size


@pytest.mark.parametrize(("small", "stubbed"), [(1_000, False), (1_001, True)])
def test_the_largest_kept_result_is_stubbed_once_the_results_overflow_the_budget(small, stubbed):
    """The budget is one UTF-8 byte of text per window token, one ASCII char.
    Results that fill it exactly stay whole; one char more and the largest
    becomes a stub, which leaves the rest under it."""
    history = [
        *_tool_loop("toolu_1"),
        tool_use("toolu_2"), _result_of(tid="toolu_2", content="x" * (_BUDGET - 1_000)),
        tool_use("toolu_3"), _result_of(tid="toolu_3", content="x" * small),
    ]
    session = _session(history=history)
    session.hard_truncate_history()

    assert _provider_violations(session.history) == _VALID
    large = _result_content(messages=session.history, tid="toolu_2")
    assert ("call the tool again" in large) is stubbed
    assert _result_content(messages=session.history, tid="toolu_3") == "x" * small


def test_of_two_results_the_same_size_the_earlier_one_is_stubbed():
    """The retried planning call answers the latest result, so that one is
    kept when either would do."""
    page = "x" * (_BUDGET * 3 // 5)
    history = [
        *_tool_loop("toolu_1"),
        tool_use("toolu_2"), _result_of(tid="toolu_2", content=page),
        tool_use("toolu_3"), _result_of(tid="toolu_3", content=page),
    ]
    session = _session(history=history)
    session.hard_truncate_history()

    assert "call the tool again" in _result_content(messages=session.history, tid="toolu_2")
    assert _result_content(messages=session.history, tid="toolu_3") == page


def test_text_outside_ascii_counts_by_its_utf8_bytes():
    """CJK text packs more tokens into each char than English does. The
    budget counts UTF-8 bytes, three per CJK char, so half a budget of CJK
    chars is over it."""
    page = "数" * (_BUDGET // 2)
    history = [
        *_tool_loop("toolu_1", "toolu_2"), tool_use("toolu_3"), _result_of(tid="toolu_3", content=page),
    ]
    session = _session(history=history)
    session.hard_truncate_history()

    assert "call the tool again" in _result_content(messages=session.history, tid="toolu_3")


def test_an_image_result_is_measured_by_its_text_not_its_base64():
    """A provider bills an image by its pixels, so megabytes of base64 do not
    make a read_image result too large."""
    source = {"type": "base64", "media_type": "image/png", "data": TOO_LARGE_PAGE}
    image = [
        {"type": "image", "source": source},
        {"type": "text", "text": "Loaded chart.png (800x600, 960 KB)."},
    ]
    session = _session(history=[
        *_tool_loop("toolu_1", "toolu_2"), tool_use("toolu_3"), _result_of(tid="toolu_3", content=image),
    ])
    session.hard_truncate_history()

    assert _result_content(messages=session.history, tid="toolu_3") == image


def test_a_message_merged_into_a_stubbed_tool_result_is_read_once():
    """A retry that comes back empty at max_tokens appends a nudge right after
    the truncation, and the nudge merges into the stubbed message. The turn's
    messages hold toolu_3's result once, as the stub the model saw."""
    session = _session(history=_tool_loop("toolu_1"))
    session._start_turn_segments()
    for message in (
        tool_use("toolu_2"), tool_result("toolu_2"),
        tool_use("toolu_3"), _result_of(tid="toolu_3", content=TOO_LARGE_PAGE),
    ):
        session._append_history(message)
    session.hard_truncate_history()
    session._append_history({"role": "user", "content": "a nudge"})

    assert _tool_rows(session.turn_messages()) == [
        ("assistant", "toolu_2"), ("user", "toolu_2"),
        ("assistant", "toolu_3"), ("user", "toolu_3"),
    ]
    assert "call the tool again" in _result_content(messages=session.turn_messages(), tid="toolu_3")


def test_a_tool_result_the_seed_ends_with_is_stubbed_but_stays_the_seeds():
    """A seed can end with a tool_result, as when an earlier turn stopped
    after its tool ran. The turn's input merges into that seeded message, and
    the hard truncation stubs its result. The message is still the seed's, so
    the turn's messages stay empty, as they are before the truncation."""
    session = _session(history=[
        *_tool_loop("toolu_1"), tool_use("toolu_2"), _result_of(tid="toolu_2", content=TOO_LARGE_PAGE),
    ])
    session._start_turn_segments()
    session._append_history({"role": "user", "content": "continue"})
    assert session.turn_messages() == []
    session.hard_truncate_history()

    # Precondition: the stub went out in history.
    assert "call the tool again" in _result_content(messages=session.history, tid="toolu_2")
    assert session.turn_messages() == []
