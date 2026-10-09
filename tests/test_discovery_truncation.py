"""Discovery steps cut off by the output budget are retried, visibly, once."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest

from anton.core.llm.effort import TRUNCATION_RETRY_NOTE
from anton.core.llm.provider import LLMResponse, ToolCall, Usage
from anton.core.tools.generate_artifact.discovery import brief, prd, prompts, sub_tools
from anton.core.tools.generate_artifact.discovery.state import PrdState
from tests.streaming_llm import StreamingLLM

BUDGET = 16384


def _cut(content="", tokens=BUDGET):
    return LLMResponse(content=content, usage=Usage(output_tokens=tokens), stop_reason="length")


def _ok(content="## Goal\nAn analog clock.\n"):
    return LLMResponse(content=content, usage=Usage(output_tokens=600), stop_reason="stop")


def _state(*plan_outcomes, progress=True, winding_down=False, artifact_path=Path("/tmp/s")):
    llm = StreamingLLM(budget=BUDGET, plan=AsyncMock(side_effect=list(plan_outcomes)))
    session = SimpleNamespace(
        _llm=llm, question_count=0, elicitor=None, emit=AsyncMock(),
        _workspace=SimpleNamespace(artifacts_dir=artifact_path.parent),
    )
    state = PrdState(
        session=session, slug="s", artifact_path=artifact_path, artifact_type="html-app",
        user_request="build a clock", agent_understanding="an analog clock",
        known_data="", user_preferences="",
    )
    if progress:
        state.progress = asyncio.Queue()
    if winding_down:
        state.spend = SimpleNamespace(should_wind_down=lambda: True)
    return state, llm


def _lines(state):
    out = []
    while not state.progress.empty():
        out.append(state.progress.get_nowait())
    return out


async def test_cut_off_brief_is_retried_with_more_room_and_a_note():
    state, llm = _state(_cut(), _ok())
    await brief.draft_brief(state)

    assert state.brief.startswith("## Goal")
    first, retry = llm.stream_calls
    assert first["max_tokens"] == BUDGET and "wait_note" not in first
    assert retry["max_tokens"] == 2 * BUDGET
    assert retry["wait_note"] == TRUNCATION_RETRY_NOTE
    assert retry["messages"][-1] == {"role": "user", "content": prompts.DISCOVERY_TRUNCATED_NUDGE}
    assert "Preparing a short brief for you (attempt 2)" in _lines(state)


async def test_history_keeps_the_retry_answer_and_not_the_nudge():
    state, _ = _state(_cut(), _ok())
    await brief.draft_brief(state)
    assert all(m.get("content") != prompts.DISCOVERY_TRUNCATED_NUDGE for m in state.messages)
    assert state.messages[-1] == {"role": "assistant", "content": state.brief}


async def test_partial_brief_text_cut_off_is_retried_not_kept():
    state, llm = _state(_cut(content="## Goal\nAn ana"), _ok())
    await brief.draft_brief(state)
    assert len(llm.stream_calls) == 2
    assert state.brief.startswith("## Goal\nAn analog clock.")


async def test_brief_cut_off_twice_fails_with_the_real_reason():
    state, _ = _state(_cut(), _cut(tokens=2 * BUDGET))
    with pytest.raises(RuntimeError, match="cut off twice"):
        await brief.draft_brief(state)


async def test_winding_down_keeps_partial_text_without_a_retry():
    state, llm = _state(_cut(content="## Goal\nAn ana"), progress=False, winding_down=True)
    await brief.draft_brief(state)
    assert len(llm.stream_calls) == 1
    assert state.brief == "## Goal\nAn ana"


async def test_winding_down_with_nothing_written_still_fails():
    state, llm = _state(_cut(), progress=False, winding_down=True)
    with pytest.raises(RuntimeError, match="no text"):
        await brief.draft_brief(state)
    assert len(llm.stream_calls) == 1


async def test_retry_works_without_a_progress_channel():
    state, llm = _state(_cut(), _ok(), progress=False)
    await brief.draft_brief(state)
    assert len(llm.stream_calls) == 2


async def test_dropped_stream_in_discovery_is_retried_at_half_budget():
    state, llm = _state(httpx2.RemoteProtocolError("peer closed connection"), _ok())
    await brief.draft_brief(state)
    assert [c["max_tokens"] for c in llm.stream_calls] == [BUDGET, BUDGET // 2]


async def test_discovery_streams_instead_of_one_shot_calls():
    state, llm = _state(_ok())
    await brief.draft_brief(state)
    assert [c["role"] for c in llm.stream_calls] == ["planning"]


async def test_cut_off_redraw_is_retried():
    redrawn = LLMResponse(
        content="## Goal\nRedrawn.", usage=Usage(output_tokens=600), stop_reason="tool_calls",
        tool_calls=[ToolCall(id="f", name="finish_gathering",
                             input={"summary": "s", "artifact_type": "html-app", "data_sources": []})],
    )
    state, llm = _state(_cut(), redrawn)
    await brief.redraw_brief(state)
    assert len(llm.stream_calls) == 2
    assert state.brief.startswith("## Goal\nRedrawn.")


async def test_redraw_cut_while_winding_down_ignores_its_damaged_call():
    damaged = LLMResponse(
        content="## Goal\nRedrawn.", usage=Usage(output_tokens=BUDGET), stop_reason="length",
        tool_calls=[ToolCall(id="f", name="finish_gathering", repaired=True,
                             input={"summary": "s", "artifact_type": "html-app", "data_sources": ["half a na"]})],
    )
    state, llm = _state(damaged, progress=False, winding_down=True)
    state.declared_sources = ["orders table"]
    await brief.redraw_brief(state)
    assert len(llm.stream_calls) == 1
    assert state.brief.startswith("## Goal\nRedrawn.")
    assert state.declared_sources == ["orders table"]


async def test_cut_off_prd_is_retried(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "s"
    artifact_dir.mkdir(parents=True)
    state, llm = _state(_cut(), _ok("## Goal\nFull PRD text.\n"), artifact_path=artifact_dir)
    state.final_artifact_type = "html-app"
    text = await prd.write_prd(state)
    assert text == "## Goal\nFull PRD text."
    assert len(llm.stream_calls) == 2


async def test_room_result_reports_retry_and_truncation():
    state, _ = _state(_cut(), _ok())
    room = await sub_tools.call_with_room(
        state, "draft_brief", role="planning", system="s", messages=[], tools=None,
    )
    assert room.retried and not room.truncated and room.budget == 2 * BUDGET
