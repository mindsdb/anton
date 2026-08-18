"""Tests for the cheap front-model thalamus (ENG-648).

Covers the three layers: history condensation, decision parsing in
`gate_turn`, and the ChatSession integration (direct answer, delegation
with skill preload, fail-open fallback, and the off-by-default flag).
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTextDelta,
    ToolCall,
    Usage,
)
from anton.core.llm.thalamus import (
    ACTION_DELEGATE,
    ACTION_RESPOND,
    condense_history,
    gate_turn,
)
from tests.conftest import make_mock_llm


def _response(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    stop_reason: str | None = "end_turn",
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason=stop_reason,
    )


def _delegate_call(reason: str = "needs tools", skills: list[str] | None = None) -> ToolCall:
    tc_input: dict = {"reason": reason}
    if skills is not None:
        tc_input["skills"] = skills
    return ToolCall(id="tc_1", name="delegate", input=tc_input)


class TestCondenseHistory:
    def test_tool_blocks_collapse_to_markers(self):
        history = [
            {"role": "user", "content": "crunch the numbers"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "On it."},
                    {"type": "tool_use", "id": "t1", "name": "scratchpad", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 9000}
                ],
            },
            {"role": "assistant", "content": "Done — the mean is 4.2."},
            {"role": "user", "content": "what was the mean again?"},
        ]
        condensed = condense_history(history)
        assert condensed[0] == {"role": "user", "content": "crunch the numbers"}
        assert "[ran tool: scratchpad]" in condensed[1]["content"]
        assert "[tool output omitted]" in condensed[2]["content"]
        assert "x" * 100 not in condensed[2]["content"]  # payload not leaked
        assert condensed[-1]["content"] == "what was the mean again?"

    def test_merges_consecutive_same_role_and_truncates(self):
        history = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b" * 5000},
            {"role": "assistant", "content": "ok"},
        ]
        condensed = condense_history(history, max_chars=200)
        assert len(condensed) == 2
        assert condensed[0]["role"] == "user"
        assert condensed[0]["content"].startswith("a\n")
        assert "[… truncated …]" in condensed[0]["content"]

    def test_window_keeps_recent_and_starts_with_user(self):
        history = []
        for i in range(20):
            history.append({"role": "user", "content": f"u{i}"})
            history.append({"role": "assistant", "content": f"a{i}"})
        condensed = condense_history(history, max_messages=5)
        # 5 kept, but the leading assistant left by the cut is dropped
        assert len(condensed) == 4
        assert condensed[0]["role"] == "user"
        assert condensed[-1]["content"] == "a19"

    def test_empty_and_nontext_history(self):
        assert condense_history([]) == []
        assert condense_history([{"role": "system", "content": "x"}]) == []

    def test_merged_block_respects_max_chars_cap(self):
        # Point 3: truncation runs AFTER merge, so a block merged from several
        # near-cap same-role messages still honours the per-entry cap (before
        # the fix each piece was capped but the merged sum blew past it).
        history = [{"role": "user", "content": "x" * 1000} for _ in range(5)]
        max_chars = 200
        condensed = condense_history(history, max_chars=max_chars)
        assert len(condensed) == 1  # all merged into one user block
        assert len(condensed[0]["content"]) <= max_chars
        # and the whole view is bounded by max_messages * max_chars
        assert all(len(m["content"]) <= max_chars for m in condensed)


class TestGateTurn:
    async def test_direct_answer(self):
        llm = make_mock_llm()
        llm.gate = AsyncMock(return_value=_response("Four."))
        decision = await gate_turn(llm, history=[{"role": "user", "content": "2+2?"}])
        assert decision.action == ACTION_RESPOND
        assert decision.text == "Four."

    async def test_delegate_tool_call_with_skills(self):
        llm = make_mock_llm()
        llm.gate = AsyncMock(
            return_value=_response(
                tool_calls=[_delegate_call(skills=["csv-summary", "  ", 42, "other"])]
            )
        )
        decision = await gate_turn(llm, history=[{"role": "user", "content": "analyze data.csv"}])
        assert decision.action == ACTION_DELEGATE
        assert decision.skills == ["csv-summary", "other"]
        assert decision.reason == "needs tools"

    async def test_truncated_answer_delegates(self):
        llm = make_mock_llm()
        llm.gate = AsyncMock(
            return_value=_response("a very long answer that got cut", stop_reason="max_tokens")
        )
        decision = await gate_turn(llm, history=[{"role": "user", "content": "explain X"}])
        assert decision.action == ACTION_DELEGATE

    async def test_empty_answer_delegates(self):
        llm = make_mock_llm()
        llm.gate = AsyncMock(return_value=_response(""))
        decision = await gate_turn(llm, history=[{"role": "user", "content": "hm"}])
        assert decision.action == ACTION_DELEGATE

    async def test_unroutable_history_delegates_without_llm_call(self):
        llm = make_mock_llm()
        llm.gate = AsyncMock()
        decision = await gate_turn(llm, history=[{"role": "system", "content": "x"}])
        assert decision.action == ACTION_DELEGATE
        llm.gate.assert_not_called()

    async def test_skill_summaries_listed_in_prompt(self):
        llm = make_mock_llm()
        llm.gate = AsyncMock(return_value=_response("hi"))
        await gate_turn(
            llm,
            history=[{"role": "user", "content": "hi"}],
            skill_summaries=[{"label": "csv-summary", "description": "Summarize CSVs"}],
        )
        system = llm.gate.call_args.kwargs["system"]
        assert "`csv-summary` — Summarize CSVs" in system


def _mock_skill_store():
    skill = SimpleNamespace(
        label="csv-summary",
        name="CSV Summary",
        description="Summarize CSVs",
        declarative_md="1. load the CSV\n2. describe it",
    )
    store = MagicMock()
    store.list_summaries.return_value = [
        {"label": "csv-summary", "description": "Summarize CSVs"}
    ]
    store.load.side_effect = lambda label: skill if label == "csv-summary" else None
    return store


class TestSessionThalamus:
    async def test_thalamus_off_by_default(self):
        llm = make_mock_llm()
        llm.plan = AsyncMock(return_value=_response("Hey!"))
        llm.gate = AsyncMock()
        session = ChatSession(ChatSessionConfig(llm_client=llm))
        reply = await session.turn("hi")
        assert reply == "Hey!"
        llm.gate.assert_not_called()

    async def test_direct_answer_skips_planning(self):
        llm = make_mock_llm()
        llm.plan = AsyncMock()
        llm.gate = AsyncMock(return_value=_response("Four."))
        session = ChatSession(ChatSessionConfig(llm_client=llm, router_enabled=True))
        reply = await session.turn("what is 2+2?")
        assert reply == "Four."
        llm.plan.assert_not_called()
        user_msg, assistant_msg = session.history
        # User turn carries its send-time stamp (ENG-1092); assert the format
        # rather than a live timestamp.
        assert user_msg["role"] == "user"
        assert re.fullmatch(
            r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\] what is 2\+2\?", user_msg["content"]
        )
        assert assistant_msg == {"role": "assistant", "content": "Four."}

    async def test_direct_answer_streaming(self):
        llm = make_mock_llm()
        llm.gate = AsyncMock(return_value=_response("Four."))

        async def _plan_stream(**kwargs):
            raise AssertionError("planning model must not be called")
            yield  # pragma: no cover

        llm.plan_stream = _plan_stream
        session = ChatSession(ChatSessionConfig(llm_client=llm, router_enabled=True))
        events = [e async for e in session.turn_stream("what is 2+2?")]
        deltas = [e for e in events if isinstance(e, StreamTextDelta)]
        completes = [e for e in events if isinstance(e, StreamComplete)]
        assert [d.text for d in deltas] == ["Four."]
        assert len(completes) == 1
        assert session.history[-1] == {"role": "assistant", "content": "Four."}

    async def test_delegate_preloads_skills_then_plans(self):
        llm = make_mock_llm()
        llm.gate = AsyncMock(
            return_value=_response(
                tool_calls=[_delegate_call(skills=["csv-summary", "nonexistent"])]
            )
        )
        llm.plan = AsyncMock(return_value=_response("Here's the summary."))
        session = ChatSession(ChatSessionConfig(llm_client=llm, router_enabled=True))
        session._skill_store = _mock_skill_store()

        reply = await session.turn("summarize data.csv")

        assert reply == "Here's the summary."
        llm.plan.assert_called_once()
        # history: user → assistant(recall tool_use) → user(tool_result) → assistant
        assert len(session.history) == 4
        tool_use = session.history[1]["content"][0]
        assert tool_use["name"] == "recall_skill"
        assert tool_use["input"] == {"label": "csv-summary"}
        tool_result = session.history[2]["content"][0]
        assert tool_result["tool_use_id"] == tool_use["id"]
        assert "1. load the CSV" in tool_result["content"]
        # only the existing skill was preloaded, and its counter bumped
        assert len(session.history[1]["content"]) == 1
        session._skill_store.increment_recommended.assert_called_once_with(
            "csv-summary", stage=1
        )
        # planning saw the preloaded exchange
        planned_messages = llm.plan.call_args.kwargs["messages"]
        assert planned_messages[1]["content"][0]["name"] == "recall_skill"

    async def test_preload_ids_unique_across_non_streaming_turns(self):
        # turn() (unlike turn_stream()) never increments _turn_count, so an
        # id derived from it would repeat on every call — regression guard
        # for that. Two *different* skills across the two turns, since the
        # same skill would be deduped on the second preload.
        skills = {
            "csv-summary": SimpleNamespace(
                label="csv-summary", name="CSV Summary",
                description="", declarative_md="1. load the CSV",
            ),
            "json-summary": SimpleNamespace(
                label="json-summary", name="JSON Summary",
                description="", declarative_md="1. load the JSON",
            ),
        }
        store = MagicMock()
        store.load.side_effect = lambda label: skills.get(label)

        llm = make_mock_llm()
        llm.gate = AsyncMock(
            side_effect=[
                _response(tool_calls=[_delegate_call(skills=["csv-summary"])]),
                _response(tool_calls=[_delegate_call(skills=["json-summary"])]),
            ]
        )
        llm.plan = AsyncMock(return_value=_response("ok"))
        session = ChatSession(ChatSessionConfig(llm_client=llm, router_enabled=True))
        session._skill_store = store

        await session.turn("summarize data.csv")
        first_id = session.history[1]["content"][0]["id"]

        await session.turn("now summarize data.json")
        second_id = session.history[5]["content"][0]["id"]

        assert first_id != second_id

    async def test_delegate_streaming_reaches_planning(self):
        llm = make_mock_llm()
        llm.gate = AsyncMock(return_value=_response(tool_calls=[_delegate_call()]))

        async def _plan_stream(**kwargs):
            yield StreamTextDelta(text="Working on it.")
            yield StreamComplete(response=_response("Working on it."))

        llm.plan_stream = _plan_stream
        session = ChatSession(ChatSessionConfig(llm_client=llm, router_enabled=True))
        events = [e async for e in session.turn_stream("analyze data.csv")]
        assert any(
            isinstance(e, StreamTextDelta) and e.text == "Working on it." for e in events
        )

    async def test_preload_skips_skill_already_in_history(self):
        # A skill whose full body is already in context must not be
        # re-injected — mirrors handle_recall_skill's stub path, so a
        # later delegate naming the same skill can't duplicate the
        # procedure (wasted tokens).
        llm = make_mock_llm()
        session = ChatSession(ChatSessionConfig(llm_client=llm, router_enabled=True))
        session._skill_store = _mock_skill_store()

        session._inject_recalled_skills(["csv-summary"])
        assert len(session.history) == 2  # tool_use + tool_result appended
        session._inject_recalled_skills(["csv-summary"])
        assert len(session.history) == 2  # second preload is a no-op

    async def test_gate_usage_counted_on_delegate_streaming(self):
        # The gate hits every text turn; on delegate its usage must reach the
        # consumer as its own StreamComplete (like a planning round), or token
        # counters that sum StreamComplete usage under-report every gate call.
        llm = make_mock_llm()
        llm.gate = AsyncMock(return_value=_response(tool_calls=[_delegate_call()]))

        async def _plan_stream(**kwargs):
            yield StreamTextDelta(text="Working on it.")
            yield StreamComplete(response=_response("Working on it."))

        llm.plan_stream = _plan_stream
        session = ChatSession(ChatSessionConfig(llm_client=llm, router_enabled=True))
        events = [e async for e in session.turn_stream("analyze data.csv")]
        completes = [e for e in events if isinstance(e, StreamComplete)]
        # two calls billed: the gate, then the planning round
        assert len(completes) == 2
        total_in = sum(e.response.usage.input_tokens for e in completes)
        assert total_in == 20  # 10 (gate) + 10 (planning)

    async def test_thalamus_failure_falls_back_to_planning(self):
        llm = make_mock_llm()
        llm.gate = AsyncMock(side_effect=RuntimeError("thalamus down"))
        llm.plan = AsyncMock(return_value=_response("Handled anyway."))
        session = ChatSession(ChatSessionConfig(llm_client=llm, router_enabled=True))
        reply = await session.turn("hi")
        assert reply == "Handled anyway."
        llm.plan.assert_called_once()

    async def test_image_turns_skip_thalamus(self):
        llm = make_mock_llm()
        llm.gate = AsyncMock()
        llm.plan = AsyncMock(return_value=_response("Nice chart."))
        session = ChatSession(ChatSessionConfig(llm_client=llm, router_enabled=True))
        reply = await session.turn(
            [
                {"type": "text", "text": "what's in this image?"},
                {"type": "image", "source": {"type": "base64", "data": "…"}},
            ]
        )
        assert reply == "Nice chart."
        llm.gate.assert_not_called()


class TestLLMClientGate:
    async def test_gate_defaults_to_coding_role(self):
        from anton.core.llm.client import LLMClient

        planning = AsyncMock()
        coding = AsyncMock()
        coding.complete = AsyncMock(return_value=_response("ok"))
        client = LLMClient(
            planning_provider=planning,
            planning_model="big-model",
            coding_provider=coding,
            coding_model="small-model",
        )
        assert client.router_model == "small-model"
        await client.gate(system="s", messages=[{"role": "user", "content": "x"}])
        assert coding.complete.call_args.kwargs["model"] == "small-model"
        planning.complete.assert_not_called()

    async def test_gate_uses_explicit_router_role(self):
        from anton.core.llm.client import LLMClient

        planning = AsyncMock()
        coding = AsyncMock()
        router = AsyncMock()
        router.complete = AsyncMock(return_value=_response("ok"))
        client = LLMClient(
            planning_provider=planning,
            planning_model="big-model",
            coding_provider=coding,
            coding_model="small-model",
            router_provider=router,
            router_model="tiny-model",
        )
        await client.gate(system="s", messages=[{"role": "user", "content": "x"}])
        assert router.complete.call_args.kwargs["model"] == "tiny-model"
        coding.complete.assert_not_called()
