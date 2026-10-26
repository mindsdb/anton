"""A scripted tool-loop turn and tool-row builders.

tests/test_turn_rows_through_compaction.py and
tests/test_hard_truncate_in_tool_loop.py both run this turn: the planner calls
`lookup` four times, then answers, and its mode picks whether history is
compacted, summarized after an overflow, or hard-truncated on SWAP_ROUND.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

from anton.core.llm.provider import (
    ContextOverflowError,
    LLMResponse,
    StreamComplete,
    StreamTextDelta,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
    ToolCall,
    Usage,
)
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.tool_defs import ToolDef
from tests.conftest import make_mock_llm

ANSWER = "Overtime costs $42 an hour, and the plant runs 9 lines."
PAGES = {
    "toolu_1": "Wage table: overtime rate is $42 per hour.",
    "toolu_2": "Shift calendar: three shifts.",
    "toolu_3": "Holiday list: none.",
    "toolu_4": "Line roster: the plant runs 9 lines.",
}
TOOL_IDS = list(PAGES)
# A page over a 200k-token window on its own: about 960,000 characters.
TOO_LARGE_PAGE = "Line roster: " + "the plant runs 9 lines. " * 40_000
SEED = [
    {"role": "user", "content": "seed question one"},
    {"role": "assistant", "content": "seed answer one"},
]
# The plan_stream call that returns toolu_4. Compaction or overflow lands on
# it, so toolu_4's tool_use and tool_result are appended after the swap.
SWAP_ROUND = 3


@dataclass
class PlanCalls:
    """What the scripted planner received.

    `requests` holds the messages of each planning call, copied when the call
    was made. `overflowed` holds the indexes in `requests` of the calls that
    raised ContextOverflowError.
    """

    requests: list[list[dict]] = field(default_factory=list)
    overflowed: list[int] = field(default_factory=list)


def scripted_llm(*, mode: str, cut: bool = False, calls: PlanCalls | None = None):
    """A planner that calls `lookup` four times, then answers.

    `mode` picks what happens on SWAP_ROUND: "proactive" reports a context
    pressure over the threshold, so the tool loop compacts; "overflow" raises
    ContextOverflowError once (summarize and retry); "hard-truncate" raises it
    twice (summarize, then hard-truncate and retry); "none" does neither.
    "too-large" raises it on every request that holds TOO_LARGE_PAGE, on any
    round, as a provider rejects a request over its window.
    With `cut`, the first planning call on SWAP_ROUND that does not overflow
    comes back empty and cut at max_tokens, so anton appends a nudge before
    the call is made again.
    Pass `calls` to read back what each planning call received.
    """
    llm = make_mock_llm()
    calls = calls if calls is not None else PlanCalls()
    state = {"round": 0, "cut_pending": cut}
    overflow_budget = {"overflow": 1, "hard-truncate": 2}.get(mode, 0)

    def plan_stream(messages, **kwargs):
        calls.requests.append(copy.deepcopy(messages))

        async def gen():
            r = state["round"]
            too_large = mode == "too-large" and TOO_LARGE_PAGE in json.dumps(messages)
            if too_large or (r == SWAP_ROUND and len(calls.overflowed) < overflow_budget):
                calls.overflowed.append(len(calls.requests) - 1)
                raise ContextOverflowError("too long")
            if r == SWAP_ROUND and state["cut_pending"]:
                state["cut_pending"] = False
                yield StreamComplete(response=LLMResponse(
                    content="",
                    usage=Usage(input_tokens=100, output_tokens=8192),
                    stop_reason="length",
                ))
                return
            state["round"] += 1
            pressure = 0.99 if mode == "proactive" and r == SWAP_ROUND else 0.0
            if r < len(TOOL_IDS):
                tid = TOOL_IDS[r]
                yield StreamToolUseStart(id=tid, name="lookup")
                yield StreamToolUseDelta(id=tid, json_delta=json.dumps({"id": tid}))
                yield StreamToolUseEnd(id=tid)
                yield StreamComplete(response=LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id=tid, name="lookup", input={"id": tid})],
                    usage=Usage(input_tokens=100, context_pressure=pressure),
                    stop_reason="tool_use",
                ))
            else:
                yield StreamTextDelta(text=ANSWER)
                yield StreamComplete(response=LLMResponse(
                    content=ANSWER,
                    usage=Usage(input_tokens=100, context_pressure=pressure),
                    stop_reason="end_turn",
                ))

        return gen()

    llm.plan_stream = plan_stream
    llm.summarize = AsyncMock(return_value=LLMResponse(content="SUMMARY of earlier turns"))
    return llm


async def _lookup(session, tc_input):
    return PAGES[tc_input["id"]]


def scripted_session(
    *,
    mode: str,
    cut: bool = False,
    handler=_lookup,
    seed: list[dict] = SEED,
    calls: PlanCalls | None = None,
) -> ChatSession:
    """A session whose planner is `scripted_llm(mode=mode, cut=cut)` and whose
    one tool, `lookup`, runs `handler`."""
    lookup = ToolDef(
        name="lookup",
        description="Look up a page by id.",
        input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
        handler=handler,
    )
    session = ChatSession(ChatSessionConfig(
        llm_client=scripted_llm(mode=mode, cut=cut, calls=calls),
        initial_history=list(seed),
        tools=[lookup],
    ))
    session._verify_min_tool_rounds = 99  # no verifier LLM call
    return session


async def run_turn(session: ChatSession) -> list:
    return [event async for event in session.turn_stream("What does overtime cost?")]


def tool_use(*ids: str) -> dict:
    return {"role": "assistant", "content": [
        {"type": "tool_use", "id": tid, "name": "lookup", "input": {}} for tid in ids
    ]}


def tool_result(*ids: str) -> dict:
    return {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tid, "content": tid} for tid in ids
    ]}
