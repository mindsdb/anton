"""Text the user adds while a turn runs (ENG-4194).

Drives the real ``turn_stream`` with a scripted LLM, because the feature is
wiring: where in the tool loop the text reaches the model, that history keeps
alternating, and what happens when the model finishes with text still unread.
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_mock_llm

from anton.core.llm.provider import (
    LLMResponse,
    StreamComplete,
    StreamTaskProgress,
    ToolCall,
    Usage,
)
from anton.core.session import (
    ChatSession,
    ChatSessionConfig,
    _MAX_USER_INPUT_ROUNDS,
    _USER_INPUT_PREFIX,
    _VerifierVerdict,
)
from anton.core.tools.registry import ToolOutcome


@pytest.fixture()
def workspace():
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _tool_call(i: int = 1) -> LLMResponse:
    return LLMResponse(
        content="working",
        tool_calls=[ToolCall(id=f"tc_{i}", name="scratchpad",
                             input={"action": "view", "name": "main"})],
        usage=Usage(input_tokens=100, output_tokens=100), stop_reason="tool_use",
    )


def _text(text: str = "done") -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[],
                       usage=Usage(input_tokens=100, output_tokens=100),
                       stop_reason="end_turn")


def _session(workspace, responses) -> tuple[ChatSession, list[list[dict]]]:
    """Session whose LLM replies with `responses` in order. A callable entry is
    called with the session first and returns the response, so a test can act
    at a precise point in the turn. Returns the session and the history each
    planning call was sent."""
    mock_llm = make_mock_llm()
    script = list(responses)
    sent: list[list[dict]] = []

    def _plan_stream(**kwargs):
        async def _gen():
            sent.append(copy.deepcopy(kwargs.get("messages") or []))
            resp = script.pop(0) if script else _text()
            if callable(resp):
                resp = resp(session)
            yield StreamComplete(response=resp)
        return _gen()

    mock_llm.usage_listener = None
    mock_llm.plan_stream = _plan_stream
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    session._jev_enabled = False
    session._llm.generate_object_code = AsyncMock(return_value=_VerifierVerdict(
        status="COMPLETE", reason="done", close_to_done=False,
    ))
    session.tool_registry.dispatch_tool = AsyncMock(
        return_value=ToolOutcome(content="stubbed tool result", ok=True)
    )
    return session, sent


async def _run(session, prompt="do the thing"):
    with patch("anton.analytics.send_event"):
        return [ev async for ev in session.turn_stream(prompt)]


def _texts(message: dict) -> list[str]:
    content = message.get("content")
    if isinstance(content, str):
        return [content]
    return [b.get("text", "") for b in content if b.get("type") == "text"]


def _added_blocks(messages: list[dict]) -> list[str]:
    return [
        t for m in messages if m.get("role") == "user"
        for t in _texts(m) if t.startswith(_USER_INPUT_PREFIX)
    ]


def _alternates(history: list[dict]) -> bool:
    roles = [m["role"] for m in history]
    return all(a != b for a, b in zip(roles, roles[1:]))


def _add_while_running(*inputs: tuple[str, str]):
    """A tool dispatch that adds `inputs` as if the user typed them meanwhile."""
    def _dispatch(session):
        async def _run_tool(*args, **kwargs):
            for addon_id, text in inputs:
                assert session.add_user_input(addon_id, text) is True
            return ToolOutcome(content="stubbed tool result", ok=True)
        return _run_tool
    return _dispatch


async def test_text_added_during_a_tool_reaches_the_next_step(workspace):
    session, sent = _session(workspace, [_tool_call(1), _text("done")])
    session.tool_registry.dispatch_tool = AsyncMock(
        side_effect=_add_while_running(("a1", "use the sales DB"), ("a2", "also Q3"))(session)
    )
    await _run(session)

    # The second planning call is the next step: it sees both, in one block
    # inside the tool-result message.
    assert _added_blocks(sent[0]) == []
    assert _added_blocks(sent[1]) == [
        f"{_USER_INPUT_PREFIX}\n- use the sales DB\n- also Q3"
    ]
    carrier = next(m for m in sent[1] if _added_blocks([m]))
    assert any(b.get("type") == "tool_result" for b in carrier["content"])
    assert _alternates(session.history)
    assert session.take_unread_user_inputs() == []


async def test_read_callback_gets_the_ids_once(workspace):
    session, _ = _session(workspace, [_tool_call(1), _tool_call(2), _text("done")])
    add = _add_while_running(("a1", "x"), ("a2", "y"))(session)
    calls = {"n": 0}

    async def _adds_on_the_first_round_only(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return await add(*args, **kwargs)
        return ToolOutcome(content="stubbed tool result", ok=True)

    session.tool_registry.dispatch_tool = AsyncMock(side_effect=_adds_on_the_first_round_only)
    seen: list[list[str]] = []
    session.on_user_input_read = seen.append
    await _run(session)
    assert seen == [["a1", "a2"]]


async def test_a_failing_read_callback_does_not_end_the_turn(workspace):
    session, sent = _session(workspace, [_tool_call(1), _text("done")])
    session.tool_registry.dispatch_tool = AsyncMock(
        side_effect=_add_while_running(("a1", "x"))(session)
    )

    def _boom(ids):
        raise RuntimeError("host bug")

    session.on_user_input_read = _boom
    await _run(session)
    assert len(sent) == 2
    assert session.history[-1] == {"role": "assistant", "content": "done"}


async def test_withdrawn_text_never_reaches_the_model(workspace):
    def _add_then_withdraw(session):
        async def _run_tool(*args, **kwargs):
            assert session.add_user_input("a1", "keep me")
            assert session.add_user_input("a2", "drop me")
            assert session.withdraw_user_input("a2") is True
            return ToolOutcome(content="stubbed tool result", ok=True)
        return _run_tool

    session, sent = _session(workspace, [_tool_call(1), _text("done")])
    session.tool_registry.dispatch_tool = AsyncMock(side_effect=_add_then_withdraw(session))
    await _run(session)
    assert _added_blocks(sent[1]) == [f"{_USER_INPUT_PREFIX}\n- keep me"]
    # Already read, and never known.
    assert session.withdraw_user_input("a1") is False
    assert session.withdraw_user_input("nope") is False


async def test_no_turn_running_refuses(workspace):
    session, _ = _session(workspace, [_text("done")])
    assert session.add_user_input("a1", "x") is False
    await _run(session)
    # Refused again once the turn has ended.
    assert session.add_user_input("a2", "y") is False
    assert session.take_unread_user_inputs() == []


async def test_the_flag_resets_when_the_turn_fails(workspace):
    def _boom(session):
        raise ValueError("deterministic failure")

    session, _ = _session(workspace, [_boom] * 10)
    try:
        await _run(session)
    except Exception:
        pass
    assert session.add_user_input("a1", "x") is False


async def test_the_flag_resets_when_the_turn_is_abandoned(workspace):
    session, _ = _session(workspace, [_tool_call(1), _text("done")])
    with patch("anton.analytics.send_event"):
        stream = session.turn_stream("do the thing")
        await stream.__anext__()
        assert session.add_user_input("a1", "x") is True
        await stream.aclose()
    assert session.add_user_input("a2", "y") is False
    assert session.take_unread_user_inputs() == [{"id": "a1", "text": "x"}]


async def test_text_added_during_the_final_answer_takes_one_more_round(workspace):
    def _final(session):
        assert session.add_user_input("a1", "make it shorter")
        return _text("long answer")

    session, sent = _session(workspace, [_final, _text("short answer")])
    events = await _run(session)

    assert len(sent) == 2
    assert _added_blocks(sent[1]) == [f"{_USER_INPUT_PREFIX}\n- make it shorter"]
    # The first answer stays in history, followed by the added text, and the
    # turn ends on the answer that took it into account.
    assert session.history[-1] == {"role": "assistant", "content": "short answer"}
    assert any(m["role"] == "assistant" and _texts(m) == ["long answer"]
               for m in session.history)
    assert _alternates(session.history)
    assert any(isinstance(e, StreamTaskProgress) and e.phase == "continuation"
               for e in events)


async def test_extra_rounds_are_capped_and_the_rest_stays_unread(workspace):
    counter = {"n": 0}

    def _keeps_adding(session):
        counter["n"] += 1
        session.add_user_input(f"a{counter['n']}", f"more {counter['n']}")
        return _text(f"answer {counter['n']}")

    session, sent = _session(workspace, [_keeps_adding] * 10)
    await _run(session)

    assert len(sent) == 1 + _MAX_USER_INPUT_ROUNDS
    assert session.history[-1]["content"] == f"answer {1 + _MAX_USER_INPUT_ROUNDS}"
    assert _alternates(session.history)
    last = f"a{1 + _MAX_USER_INPUT_ROUNDS}"
    assert session.take_unread_user_inputs() == [
        {"id": last, "text": f"more {1 + _MAX_USER_INPUT_ROUNDS}"}
    ]
    # Cleared by the take.
    assert session.take_unread_user_inputs() == []


async def test_the_extra_round_can_use_tools_and_still_gets_verified(workspace):
    def _final(session):
        session.add_user_input("a1", "check EMEA too")
        return _text("first answer")

    session, sent = _session(workspace, [_tool_call(1), _final, _tool_call(2), _text("final")])
    session._verify_min_tool_rounds = 2
    await _run(session)
    # One round before the addition and one after count as one attempt, so the
    # verifier runs once on the final answer.
    assert session._llm.generate_object_code.await_count == 1
    assert session.history[-1] == {"role": "assistant", "content": "final"}
    assert _alternates(session.history)


def test_add_from_another_thread(workspace):
    session, _ = _session(workspace, [])
    session._user_input_turn_running = True
    threads = [
        threading.Thread(target=session.add_user_input, args=(f"a{i}", f"t{i}"))
        for i in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(e["id"] for e in session.take_unread_user_inputs()) == sorted(
        f"a{i}" for i in range(20)
    )
