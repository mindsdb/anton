"""orchestrator.run: the full two-phase sequence, exercised end to end with
every LLM/elicit call mocked (unit-level, no real model)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from anton.core.artifacts import ArtifactStore
from anton.core.interaction.elicit import AskAnswer
from anton.core.llm.provider import LLMResponse, ToolCall, Usage
from anton.core.tools.generate_prd import orchestrator
from anton.core.tools.generate_prd.state import PrdState


def _text_response(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1))


def _tool_response(name: str, input: dict) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc1", name=name, input=input)],
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _make_state(tmp_path: Path, **over) -> PrdState:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.create(name="Clock", description="d", type="html-app")
    artifact_dir = store.folder_for(artifact.slug)
    base = dict(
        session=SimpleNamespace(_llm=SimpleNamespace(), question_count=0, elicitor=None),
        slug=artifact.slug,
        artifact_path=artifact_dir,
        artifact_type="html-app",
        user_request="build a clock",
        agent_understanding="an analog clock",
        known_data="",
        user_preferences="",
    )
    base.update(over)
    return PrdState(**base)


async def test_full_happy_path_accept_on_first_try(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    state.session._llm.plan = AsyncMock(
        side_effect=[
            _tool_response("finish_gathering", {"summary": "ok", "artifact_type": "html-app", "notes": "no data needed"}),
            _text_response("## Goal\nAn analog clock.\n"),  # draft_brief
            _text_response("## Goal\nAn analog clock, in full.\n"),  # write_prd
        ]
    )

    async def fake_ask_via_elicit(session, request):
        return AskAnswer(status="answered", values=("accept",))

    monkeypatch.setattr(orchestrator.sub_tools, "ask_via_elicit", fake_ask_via_elicit)

    result = await orchestrator.run(state)
    assert result["status"] == "prd_written"
    assert result["artifact_type"] == "html-app"
    assert (state.artifact_path / "prd.md").exists()


async def test_cancelled_never_writes_prd(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    state.session._llm.plan = AsyncMock(
        side_effect=[
            _tool_response("finish_gathering", {"summary": "ok", "artifact_type": "html-app"}),
            _text_response("## Goal\n...\n"),
        ]
    )

    async def fake_ask_via_elicit(session, request):
        return AskAnswer(status="answered", values=("cancel",))

    monkeypatch.setattr(orchestrator.sub_tools, "ask_via_elicit", fake_ask_via_elicit)

    result = await orchestrator.run(state)
    assert result["status"] == "cancelled"
    assert not (state.artifact_path / "prd.md").exists()


async def test_revise_then_accept_loops_back_through_draft_brief(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    state.session._llm.plan = AsyncMock(
        side_effect=[
            _tool_response("finish_gathering", {"summary": "ok", "artifact_type": "html-app"}),
            _text_response("## Goal\nfirst draft\n"),
            _text_response("## Goal\nrevised draft\n"),
            _text_response("## Goal\nfull revised PRD\n"),
        ]
    )
    state.session._llm.generate_object = AsyncMock(
        return_value=orchestrator.FeedbackVerdict(route="revise_brief", reasoning="just wording")
    )

    answers = iter([
        AskAnswer(status="answered", text="make it blue"),
        AskAnswer(status="answered", values=("accept",)),
    ])

    async def fake_ask_via_elicit(session, request):
        return next(answers)

    monkeypatch.setattr(orchestrator.sub_tools, "ask_via_elicit", fake_ask_via_elicit)

    result = await orchestrator.run(state)
    assert result["status"] == "prd_written"
    assert (state.artifact_path / "prd.md").read_text() == "full revised PRD" or "full revised PRD" in (state.artifact_path / "prd.md").read_text()


async def test_best_effort_when_gathering_never_calls_finish_gathering(tmp_path, monkeypatch):
    """No finish_gathering call → state.final_artifact_type stays "" after
    the gathering loop → orchestrator falls back to the originally
    registered artifact_type instead of failing."""
    state = _make_state(tmp_path)
    responses = iter(
        [
            _text_response("I looked around and I'm ready."),  # no tool_calls at all
            _text_response("## Goal\n...\n"),  # draft_brief
            _text_response("## Goal\n... full\n"),  # write_prd
        ]
    )
    # A snapshot (`list(...)`) of `messages` per call, not the live list:
    # `state.messages` is mutated in place and passed by reference, so
    # capturing `kwargs["messages"]` itself (or reading it back later from
    # `call_args_list`) would show every call's `messages` as the FINAL
    # history — including everything appended after that call — rather than
    # what draft_brief actually saw at call time. That would make the
    # assertion below pass even if the ordering regressed.
    messages_snapshots: list[list[dict]] = []

    async def fake_plan(**kwargs):
        messages_snapshots.append(list(kwargs["messages"]))
        return next(responses)

    state.session._llm.plan = fake_plan

    async def fake_ask_via_elicit(session, request):
        return AskAnswer(status="answered", values=("accept",))

    monkeypatch.setattr(orchestrator.sub_tools, "ask_via_elicit", fake_ask_via_elicit)

    result = await orchestrator.run(state)
    assert result["status"] == "prd_written"
    assert result["artifact_type"] == "html-app"  # fell back to the registered type
    # The gathering pass's best-effort summary must have reached draft_brief
    # via `state.messages` — not been silently dropped along with the empty
    # `final_artifact_type`.
    draft_brief_messages = messages_snapshots[1]
    assert any(
        m.get("content") == "I looked around and I'm ready." for m in draft_brief_messages
    )


async def test_unconfirmed_when_budget_runs_out_at_confirm(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    state.session._llm.plan = AsyncMock(
        side_effect=[
            _tool_response("finish_gathering", {"summary": "ok", "artifact_type": "html-app"}),
            _text_response("## Goal\n...\n"),
            _text_response("## Goal\n... full\n"),  # write_prd still runs (best-effort)
        ]
    )

    async def fake_ask_via_elicit(session, request):
        return AskAnswer(status="limit")

    monkeypatch.setattr(orchestrator.sub_tools, "ask_via_elicit", fake_ask_via_elicit)

    result = await orchestrator.run(state)
    assert result["status"] == "prd_written_unconfirmed"
    assert (state.artifact_path / "prd.md").exists()


async def test_qa_log_is_present_in_every_status(tmp_path, monkeypatch):
    state = _make_state(tmp_path)
    state.session._llm.plan = AsyncMock(
        side_effect=[
            _tool_response("finish_gathering", {"summary": "ok", "artifact_type": "html-app"}),
            _text_response("## Goal\n...\n"),
        ]
    )

    async def fake_ask_via_elicit(session, request):
        return AskAnswer(status="answered", values=("cancel",))

    monkeypatch.setattr(orchestrator.sub_tools, "ask_via_elicit", fake_ask_via_elicit)

    result = await orchestrator.run(state)
    assert "qa_log" in result


async def test_revise_loop_has_a_defensive_cap_and_ends_unconfirmed(tmp_path, monkeypatch):
    """If show_and_confirm somehow never returns "unconfirmed" (the shared
    question budget never reports "limit" — a test double, a future
    elicit() change, a custom elicitor that never increments
    session.question_count), the revise loop must still terminate instead
    of spinning forever. `responses` is padded well past
    MAX_REVISE_CYCLES so the mock never runs dry before the cap fires."""
    state = _make_state(tmp_path)
    responses = [
        _tool_response("finish_gathering", {"summary": "ok", "artifact_type": "html-app"}),
    ] + [_text_response("## Goal\n...\n")] * (orchestrator.MAX_REVISE_CYCLES + 5)
    state.session._llm.plan = AsyncMock(side_effect=responses)
    state.session._llm.generate_object = AsyncMock(
        return_value=orchestrator.FeedbackVerdict(route="revise_brief", reasoning="wording only")
    )

    async def fake_ask_via_elicit(session, request):
        return AskAnswer(status="answered", text="one more tweak")

    monkeypatch.setattr(orchestrator.sub_tools, "ask_via_elicit", fake_ask_via_elicit)

    result = await orchestrator.run(state)
    assert result["status"] == "prd_written_unconfirmed"
    assert (state.artifact_path / "prd.md").exists()
