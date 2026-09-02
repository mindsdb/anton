"""Completion-verifier fail-safe (ENG-1079).

Before this fix, an exception from the verifier's ``generate_object_code``
call (provider hiccup, malformed structured output, etc.) was silently
treated as a COMPLETE verdict — the task loop just broke with no message,
making the agent look like it had died on the first hurdle. It should
instead behave like the STUCK path: append an honest SYSTEM note and let the
model generate a real message summarizing progress and asking the user how
to proceed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from tests.conftest import make_mock_llm

from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict
from anton.core.llm.provider import (
    LLMResponse,
    ModelUnavailableError,
    StreamComplete,
    StreamTaskProgress,
    TokenLimitExceeded,
    ToolCall,
    Usage,
)


@pytest.fixture()
def workspace():
    # Keep scratchpad venvs inside the repo workspace (pytest runs sandboxed and
    # can't write to the real home directory).
    base = Path(__file__).resolve().parents[1] / ".pytest-workspace"
    base.mkdir(parents=True, exist_ok=True)
    return MagicMock(base=base)


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )


def _scratchpad_response(text: str, action: str, name: str, code: str = "") -> LLMResponse:
    tc_input: dict = {"action": action, "name": name}
    if code:
        tc_input["code"] = code
    return LLMResponse(
        content=text,
        tool_calls=[ToolCall(id="tc_1", name="scratchpad", input=tc_input)],
        usage=Usage(input_tokens=10, output_tokens=20),
        stop_reason="tool_use",
    )


# Anchors into the verifier-failure hand-back (session.py). Shared because the
# denied-verdict tests assert this text NEVER enters history — anchored on a
# string the prompt no longer contains, those assertions would pass vacuously.
_HANDBACK_ANCHOR = "automatic check on whether the task is complete"
_HANDBACK_PROGRESS = "Confirming what was completed..."


class _FakeAsyncIter:
    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


async def test_verifier_exception_yields_real_message_not_silent_stop(workspace):
    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = AsyncMock(side_effect=RuntimeError("provider hiccup"))

    call_count = 0

    def fake_plan_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Tool-call round.
            return _FakeAsyncIter([
                StreamComplete(response=_scratchpad_response("Running.", "exec", "main", "print(1)"))
            ])
        if call_count == 2:
            # Final round of the tool loop — this text is what gets sent to
            # (the now-failing) verifier.
            return _FakeAsyncIter([
                StreamComplete(response=_text_response("Done running the script."))
            ])
        # Third call is the honest-diagnosis re-prompt (plan_stream_with_recovery),
        # only reached via the verifier-exception fail-safe.
        return _FakeAsyncIter([
            StreamComplete(
                response=_text_response(
                    "Here's what I've done so far — ran the script. An internal "
                    "check failed, so let me know how you'd like to proceed."
                )
            )
        ])

    mock_llm.plan_stream = fake_plan_stream

    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    try:
        events = [e async for e in session.turn_stream("run my script")]

        # Never silently stops: a progress event surfaces the check-in, and
        # exactly one more model call happens (the honest diagnosis) — not a
        # forced "Continue working" continuation, and not silence.
        progress_msgs = [e.message for e in events if isinstance(e, StreamTaskProgress)]
        assert _HANDBACK_PROGRESS in progress_msgs
        assert call_count == 3

        history_texts = [
            m["content"] for m in session.history
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ]
        assert any(_HANDBACK_ANCHOR in t for t in history_texts)
        # The hand-back must not read as a failed turn, and must not invite the
        # model to restate a path from memory: asked where the file went, it
        # named the path it MEANT to write and the user went looking there.
        assert any("Your reply above stands" in t for t in history_texts)
        assert any(
            "Never state a path you intended to write" in t for t in history_texts
        )
        # It must not claim the work was clean either. Only the tool-round count
        # gates this path, so a turn whose tool calls failed lands here too, and
        # a "nothing went wrong" framing would be the same misreport inverted.
        assert any(
            "including any tool that returned an error" in t for t in history_texts
        )
        assert not any("Continue working on the original request" in t for t in history_texts)
        # The model must be asked for a solvability self-assessment, not just
        # a status dump — otherwise "let me know how to proceed" is a vague
        # ask instead of an actual recommendation (ENG-1079 follow-up).
        assert any("whether you believe this is still solvable on your own" in t for t in history_texts)

        final_texts = [
            m["content"] for m in session.history
            if m.get("role") == "assistant" and isinstance(m.get("content"), str)
        ]
        assert any("how you'd like to proceed" in t for t in final_texts)
    finally:
        await session.close()


async def test_truncation_exhausting_every_budget_also_gets_the_diagnosis(workspace, caplog):
    """The ENG-1081 ladder can run dry: a verdict truncated at 2048 AND at the
    4096 retry (narrating model, huge session) used to fall into the same
    silent fake-COMPLETE this file exists to remove. Exhausted budgets are not
    a verdict either — the honest-diagnosis path must fire for them too.
    """
    import logging

    from anton.core.llm.provider import StructuredOutputError
    from anton.core.session import _VERIFIER_TOKEN_BUDGETS

    budgets_seen: list[int] = []

    async def always_truncated(_schema, *, system, messages, max_tokens):
        budgets_seen.append(max_tokens)
        raise StructuredOutputError(
            "no tool call", truncated=True, output_tokens=max_tokens,
            max_tokens=max_tokens, stop_reason="stop",
        )

    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = AsyncMock(side_effect=always_truncated)

    call_count = 0

    def fake_plan_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _FakeAsyncIter([
                StreamComplete(response=_scratchpad_response("Running.", "exec", "main", "print(1)"))
            ])
        if call_count == 2:
            return _FakeAsyncIter([
                StreamComplete(response=_text_response("Done running the script."))
            ])
        return _FakeAsyncIter([
            StreamComplete(
                response=_text_response(
                    "I ran the script; an internal check failed — how would "
                    "you like to proceed?"
                )
            )
        ])

    mock_llm.plan_stream = fake_plan_stream

    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    try:
        with caplog.at_level(logging.INFO):
            events = [e async for e in session.turn_stream("run my script")]

        assert budgets_seen == list(_VERIFIER_TOKEN_BUDGETS), (
            "the ladder must exhaust every budget before giving up"
        )
        progress_msgs = [e.message for e in events if isinstance(e, StreamTaskProgress)]
        assert _HANDBACK_PROGRESS in progress_msgs, (
            "exhausted budgets must fail toward the diagnosis, not a silent COMPLETE"
        )
        assert call_count == 3
        # Only the attempt that ends the ladder is a failure; the retried one is
        # not, so a default-level (WARNING) log carries exactly one of the two.
        attempts = [
            r for r in caplog.records
            if "completion-verifier verdict=" in r.getMessage()
            and "output_tokens=" in r.getMessage()
        ]
        ladder = [logging.INFO] * (len(_VERIFIER_TOKEN_BUDGETS) - 1)
        assert [r.levelno for r in attempts] == ladder + [logging.WARNING]
        # This clause carries the dominant free-tier failure, so the line has to
        # name the exception type or a support log cannot say what broke.
        assert all("error=StructuredOutputError" in r.getMessage() for r in attempts)
        # And the line naming the failure class must itself clear WARNING, or a
        # customer's default log cannot say why the turn handed back.
        gave_up = [
            r for r in caplog.records
            if "failing toward an honest diagnosis" in r.getMessage()
        ]
        assert [r.levelno for r in gave_up] == [logging.WARNING]
    finally:
        await session.close()


def _assistant_texts(session) -> list[str]:
    return [
        m["content"] for m in session.history
        if m.get("role") == "assistant" and isinstance(m.get("content"), str)
    ]


async def test_stuck_diagnosis_is_persisted_as_streamed(workspace):
    """ENG-1155: the STUCK path streams a model-generated diagnosis but never
    captures it, so the post-loop fallback re-appends the *stale
    pre-verification* reply. History — and therefore the model's memory of what
    it just told the user — ends up out of sync on exactly the turn where the
    user is being asked to make a decision.

    #276 fixed this for the verifier-failure path; STUCK never got the same
    treatment.
    """
    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="STUCK", reason="missing credentials")
    )

    call_count = 0

    def fake_plan_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _FakeAsyncIter([
                StreamComplete(response=_scratchpad_response("Running.", "exec", "main", "print(1)"))
            ])
        if call_count == 2:
            # Pre-verification reply. The user never sees this as the turn's
            # final word — the diagnosis below is what actually gets streamed.
            return _FakeAsyncIter([
                StreamComplete(response=_text_response("STALE_PRE_VERIFICATION_REPLY"))
            ])
        # The STUCK diagnosis — this is what the user reads.
        return _FakeAsyncIter([
            StreamComplete(
                response=_text_response(
                    "DIAGNOSIS: I couldn't reach the database — no credentials are "
                    "configured. Add DB_URL and I'll retry."
                )
            )
        ])

    mock_llm.plan_stream = fake_plan_stream

    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    try:
        async for _ in session.turn_stream("query my database"):
            pass

        assert call_count == 3, "the STUCK path must generate a diagnosis"
        texts = _assistant_texts(session)
        assert any("DIAGNOSIS:" in t for t in texts), (
            "the diagnosis the user actually read is missing from history"
        )
        # The turn's last assistant word in history must be what was streamed,
        # not the reply the verifier rejected.
        assert "DIAGNOSIS:" in texts[-1], (
            f"history ends on the stale reply, not the diagnosis: {texts[-1]!r}"
        )
        assert sum("STALE_PRE_VERIFICATION_REPLY" in t for t in texts) <= 1, (
            "the stale reply was appended twice (2831 + the post-loop fallback)"
        )
    finally:
        await session.close()


async def test_budget_exhausted_diagnosis_is_persisted_as_streamed(workspace):
    """ENG-1155, same desync on the budget-exhausted path. `_max_continuations
    = 0` makes the very first verification hit the exhausted branch, which
    fires before any verdict call.
    """
    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = AsyncMock(
        side_effect=AssertionError("no verdict call on the exhausted path")
    )

    call_count = 0

    def fake_plan_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _FakeAsyncIter([
                StreamComplete(response=_scratchpad_response("Running.", "exec", "main", "print(1)"))
            ])
        if call_count == 2:
            return _FakeAsyncIter([
                StreamComplete(response=_text_response("STALE_PRE_VERIFICATION_REPLY"))
            ])
        return _FakeAsyncIter([
            StreamComplete(
                response=_text_response(
                    "DIAGNOSIS: I tried three times and the export step keeps "
                    "failing. Here's what you can do next."
                )
            )
        ])

    mock_llm.plan_stream = fake_plan_stream

    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    session._max_continuations = 0
    try:
        async for _ in session.turn_stream("export my data"):
            pass

        assert call_count == 3, "the budget-exhausted path must generate a diagnosis"
        texts = _assistant_texts(session)
        assert any("DIAGNOSIS:" in t for t in texts), (
            "the diagnosis the user actually read is missing from history"
        )
        assert "DIAGNOSIS:" in texts[-1], (
            f"history ends on the stale reply, not the diagnosis: {texts[-1]!r}"
        )
        assert sum("STALE_PRE_VERIFICATION_REPLY" in t for t in texts) <= 1, (
            "the stale reply was appended twice (2831 + the post-loop fallback)"
        )
    finally:
        await session.close()


def _make_session(workspace, mock_llm) -> ChatSession:
    return ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))


def _tool_then_text_plan(diagnosis: str = "DIAGNOSIS: internal check failed."):
    """plan_stream factory replaying one tool round, a reply, then a diagnosis —
    reset per turn so the same session can run several turns."""
    state = {"n": 0}

    def plan(**kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return _FakeAsyncIter([
                StreamComplete(response=_scratchpad_response("Running.", "exec", "main", "print(1)"))
            ])
        if state["n"] == 2:
            return _FakeAsyncIter([StreamComplete(response=_text_response("REPLY"))])
        return _FakeAsyncIter([StreamComplete(response=_text_response(diagnosis))])

    return plan, state


async def test_deterministic_hard_failure_latches_after_one_diagnosis(workspace, caplog):
    """ENG-1155: kimi-K3 rejects the forced `tool_choice` with a 400 on EVERY
    verdict call (ENG-1095), so the ENG-1079 diagnosis path would fire on every
    multi-step turn — a full-history planning call plus a "checking in" message
    each time. The session must latch instead: one diagnosis, then skip.
    """
    import logging

    mock_llm = make_mock_llm()
    verdict_calls = 0

    async def always_400(_schema, *, system, messages, max_tokens):
        nonlocal verdict_calls
        verdict_calls += 1
        raise RuntimeError("400 tool_choice not supported")

    mock_llm.generate_object_code = AsyncMock(side_effect=always_400)
    session = _make_session(workspace, mock_llm)

    diagnoses = 0
    try:
        with caplog.at_level(logging.INFO):
            for turn in range(4):
                plan, state = _tool_then_text_plan()
                mock_llm.plan_stream = plan
                async for _ in session.turn_stream(f"do step {turn}"):
                    pass
                # A third plan_stream call on a turn == the diagnosis fired.
                if state["n"] >= 3:
                    diagnoses += 1

        assert diagnoses == 1, (
            f"expected exactly one diagnosis per session, got {diagnoses}"
        )
        # Turns 1 and 2 call the verifier (the 2nd establishes the pattern);
        # turns 3 and 4 must not pay for it at all.
        assert verdict_calls == 2, (
            f"latched session kept calling the verifier ({verdict_calls} calls)"
        )
        assert session._verifier_latched is True
        # Anchor on a substring unique to the ANNOUNCEMENT line. An earlier
        # assertion matched "latched after N consecutive hard failures", which
        # the per-turn *skip* log also contained — so it passed without ever
        # guarding the announcement it was written for.
        announcements = [
            r for r in caplog.records
            if "skipping verification until the next re-probe" in r.message
        ]
        assert len(announcements) == 1, (
            f"the latch must announce itself exactly once, got {len(announcements)}"
        )
        assert "no successful verdict between them" in announcements[0].message, (
            f"announcement must state the real reset semantics: {announcements[0].message!r}"
        )
        skips = [r for r in caplog.records
                 if "completion-verifier skipped — latched" in r.message]
        assert len(skips) == 2, (
            f"expected one skip log per skipped verification, got {len(skips)}"
        )
    finally:
        await session.close()


async def test_an_exhausted_ladder_latches_at_the_threshold(workspace):
    """A ladder exhausted at every budget counts toward the latch, the same as
    a provider that rejects the call. The ENG-1081 retry ladder still runs in
    full first, and the first such turn still gets its honest diagnosis.
    """
    from anton.core.llm.provider import StructuredOutputError

    mock_llm = make_mock_llm()

    async def always_truncated(_schema, *, system, messages, max_tokens):
        raise StructuredOutputError(
            "no tool call", truncated=True, output_tokens=max_tokens,
            max_tokens=max_tokens, stop_reason="stop",
        )

    mock_llm.generate_object_code = AsyncMock(side_effect=always_truncated)
    session = _make_session(workspace, mock_llm)

    diagnoses = 0
    try:
        for turn in range(3):
            plan, state = _tool_then_text_plan()
            mock_llm.plan_stream = plan
            async for _ in session.turn_stream(f"do step {turn}"):
                pass
            if state["n"] >= 3:
                diagnoses += 1

        # An exhausted ladder is evidence about the model, not a tail sample
        # of its verbosity: the first turn diagnoses, the second latches.
        assert session._verifier_latched is True, (
            "an exhausted ladder must count toward the latch"
        )
        # The ladder must run in FULL before anything counts, or this rule
        # silently becomes "one truncation latches" and halves the verifier's
        # real recovery rate. Two turns reach the verifier, two budgets each.
        assert mock_llm.generate_object_code.await_count == 4, (
            f"expected the full ladder on each verifying turn, got "
            f"{mock_llm.generate_object_code.await_count} calls"
        )
        assert session._verifier_latch_reason == "truncated", (
            "the books must show what actually latched, not a fixed 'hard'"
        )
        assert diagnoses == 1, (
            f"one honest diagnosis, then silence — not one per turn, got {diagnoses}"
        )
    finally:
        await session.close()


async def test_successful_verdict_clears_the_latch_counter(workspace):
    """One hard failure then a working verdict is a transient blip — the counter
    must reset, or two unrelated failures an hour apart would latch the session.
    """
    mock_llm = make_mock_llm()
    outcomes = [
        RuntimeError("400 once"),
        _VerifierVerdict(status="COMPLETE", reason="done"),
        RuntimeError("400 again"),
    ]

    async def scripted(_schema, *, system, messages, max_tokens):
        item = outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    mock_llm.generate_object_code = AsyncMock(side_effect=scripted)
    session = _make_session(workspace, mock_llm)
    try:
        for turn in range(3):
            plan, _ = _tool_then_text_plan()
            mock_llm.plan_stream = plan
            async for _ in session.turn_stream(f"do step {turn}"):
                pass

        # fail, succeed, fail → never two in a row.
        assert session._verifier_latched is False
        assert session._verifier_no_verdict_failures == 1
    finally:
        await session.close()


async def test_max_tool_rounds_diagnosis_is_persisted_as_streamed(workspace):
    """The tool-round circuit breaker is a FOURTH hand-back path with the same
    desync — not named in ENG-1155, and the highest-traffic of the four (the
    25-round cap fires on real sessions far more often than STUCK does).

    It appends the reply, injects a SYSTEM pause, streams a diagnosis, and never
    captures it — so the post-loop fallback re-appends the reply.
    """
    mock_llm = make_mock_llm()
    # Verifier is never reached on this path (_max_rounds_hit skips verification).
    mock_llm.generate_object_code = AsyncMock(
        side_effect=AssertionError("no verdict call on the max-rounds path")
    )

    call_count = 0

    def fake_plan_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            # Keep calling tools until the cap trips.
            return _FakeAsyncIter([
                StreamComplete(response=_scratchpad_response(
                    "STALE_PRE_VERIFICATION_REPLY", "exec", "main", "print(1)"))
            ])
        return _FakeAsyncIter([
            StreamComplete(response=_text_response(
                "DIAGNOSIS: I've used my step budget. Here's where I got to."
            ))
        ])

    mock_llm.plan_stream = fake_plan_stream

    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    session._max_tool_rounds = 1
    try:
        async for _ in session.turn_stream("do lots of steps"):
            pass

        texts = _assistant_texts(session)
        assert any("DIAGNOSIS:" in t for t in texts), (
            "the cap diagnosis the user read is missing from history"
        )
        assert "DIAGNOSIS:" in texts[-1], (
            f"history ends on the stale reply, not the diagnosis: {texts[-1]!r}"
        )
        assert sum("STALE_PRE_VERIFICATION_REPLY" in t for t in texts) <= 1, (
            "the stale reply was appended twice"
        )
    finally:
        await session.close()


async def test_text_only_continuation_answer_is_not_dropped(workspace):
    """Self-review catch. `_reply_persisted` must be per-iteration, not per-turn.

    INCOMPLETE → the continuation finishes the answer with no tool calls →
    `tool_round` stays 0 → the loop breaks at the verify_min_tool_rounds gate
    BEFORE appending the new reply. A turn-scoped flag would still be True from
    the first iteration, so the post-loop fallback would skip, and the answer the
    user just read would vanish from history. That is the same defect ENG-1155
    fixes, introduced on the continuation path.
    """
    mock_llm = make_mock_llm()
    verdicts = [_VerifierVerdict(status="INCOMPLETE", reason="not done yet")]

    async def verdict(_schema, *, system, messages, max_tokens):
        return verdicts.pop(0) if verdicts else _VerifierVerdict(
            status="COMPLETE", reason="done"
        )

    mock_llm.generate_object_code = AsyncMock(side_effect=verdict)

    call_count = 0

    def fake_plan_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _FakeAsyncIter([
                StreamComplete(response=_scratchpad_response("Working.", "exec", "main", "print(1)"))
            ])
        if call_count == 2:
            return _FakeAsyncIter([StreamComplete(response=_text_response("FIRST_DRAFT"))])
        # Continuation needs no tools — it just finishes writing.
        return _FakeAsyncIter([
            StreamComplete(response=_text_response("FINAL_ANSWER_USER_READ"))
        ])

    mock_llm.plan_stream = fake_plan_stream

    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    try:
        async for _ in session.turn_stream("write me a summary"):
            pass

        texts = _assistant_texts(session)
        assert any("FINAL_ANSWER_USER_READ" in t for t in texts), (
            "the continuation's answer — what the user actually read — is missing "
            f"from history; assistant turns held: {texts}"
        )
        assert "FINAL_ANSWER_USER_READ" in texts[-1]
    finally:
        await session.close()


async def test_latched_verifier_reprobes_and_can_recover(workspace):
    """Self-review catch. A latched session makes no verdict calls, so "reset on a
    successful verdict" is unreachable and the latch outlives its cause — the user
    switches off the broken model and verification stays dead for the session.

    After `_VERIFIER_LATCH_REPROBE_TURNS` skips it must spend one call; if that
    succeeds, the latch clears.
    """
    from anton.core.session import _VERIFIER_LATCH_REPROBE_TURNS

    mock_llm = make_mock_llm()
    calls = {"n": 0}
    # Two hard failures to latch, then every later call succeeds (the user has
    # switched to a model whose forced tool_choice works).
    async def scripted(_schema, *, system, messages, max_tokens):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("400 tool_choice not supported")
        return _VerifierVerdict(status="COMPLETE", reason="done")

    mock_llm.generate_object_code = AsyncMock(side_effect=scripted)
    session = _make_session(workspace, mock_llm)
    try:
        for turn in range(2):
            plan, _ = _tool_then_text_plan()
            mock_llm.plan_stream = plan
            async for _ in session.turn_stream(f"step {turn}"):
                pass
        assert session._verifier_latched is True, "two hard failures must latch"
        assert calls["n"] == 2

        # Run exactly enough turns to consume the skip budget and re-probe.
        for turn in range(_VERIFIER_LATCH_REPROBE_TURNS):
            plan, _ = _tool_then_text_plan()
            mock_llm.plan_stream = plan
            async for _ in session.turn_stream(f"more {turn}"):
                pass

        assert calls["n"] == 3, (
            f"expected exactly one re-probe call after {_VERIFIER_LATCH_REPROBE_TURNS} "
            f"skips, verifier was called {calls['n']}x total"
        )
        assert session._verifier_latched is False, (
            "a successful re-probe must clear the latch"
        )
        assert session._verifier_no_verdict_failures == 0
    finally:
        await session.close()


@pytest.mark.parametrize("exc_factory, label", [
    (lambda: __import__("anton.core.llm.provider", fromlist=["x"]).TransientProviderError(
        "provider returned an empty 200"), "typed"),
    # Review finding (pnewsam on #299): an UNTYPED transient reaching the generic
    # handler used to count as "hard". A timeout is not a statement about whether
    # the model can produce a verdict.
    (lambda: asyncio.TimeoutError("read timed out"), "asyncio-timeout"),
    (lambda: ConnectionResetError("peer reset the connection"), "connection-reset"),
    (lambda: httpx.ConnectError("failed to establish a connection"), "httpx-connect"),
])
async def test_transient_provider_errors_never_latch(workspace, exc_factory, label):
    """No transient failure may latch — typed or untyped. The latch is for a model
    that *cannot* produce a verdict (kimi-K3's forced-tool_choice 400, ENG-1095) —
    a capability claim. A dropped connection or a timeout asserts the opposite:
    retryable, and typed ones are already retried upstream (ENG-673). Counting
    them would let two provider blips disable verification for a whole re-probe
    window.

    Matters more once #297 (ENG-847) lands, which converts empty 200s into
    `TransientProviderError` on the same providers the verdict call uses.
    """
    mock_llm = make_mock_llm()
    calls = {"n": 0}

    async def always_transient(_schema, *, system, messages, max_tokens):
        calls["n"] += 1
        raise exc_factory()

    mock_llm.generate_object_code = AsyncMock(side_effect=always_transient)
    session = _make_session(workspace, mock_llm)
    try:
        for turn in range(4):
            plan, _ = _tool_then_text_plan()
            mock_llm.plan_stream = plan
            async for _ in session.turn_stream(f"step {turn}"):
                pass

        assert session._verifier_latched is False, (
            "transient provider errors must never latch the session"
        )
        assert session._verifier_no_verdict_failures == 0
        # Verification keeps being attempted every turn, rather than being
        # switched off after two blips.
        assert calls["n"] == 4, (
            f"verifier should still be called every turn, got {calls['n']}"
        )
    finally:
        await session.close()


async def test_failed_reprobe_stays_latched_without_rediagnosis(workspace, caplog):
    """The other half of the re-probe contract: when the cause is still there,
    the re-probe fails, the session stays latched, and — critically — no second
    diagnosis fires (one per session, ENG-1155). The next skip cycle then starts
    over, so the steady-state cost of a permanently broken verifier is one
    verdict call per `_VERIFIER_LATCH_REPROBE_TURNS` turns.
    """
    import logging

    from anton.core.session import _VERIFIER_LATCH_REPROBE_TURNS

    mock_llm = make_mock_llm()
    calls = {"n": 0}

    async def always_hard(_schema, *, system, messages, max_tokens):
        calls["n"] += 1
        raise RuntimeError("400 tool_choice not supported")

    mock_llm.generate_object_code = AsyncMock(side_effect=always_hard)
    session = _make_session(workspace, mock_llm)

    diagnoses = 0
    try:
        with caplog.at_level(logging.INFO):
            # 2 latching turns, then two full skip-and-reprobe cycles.
            for turn in range(2 + 2 * _VERIFIER_LATCH_REPROBE_TURNS):
                plan, state = _tool_then_text_plan()
                mock_llm.plan_stream = plan
                async for _ in session.turn_stream(f"step {turn}"):
                    pass
                if state["n"] >= 3:
                    diagnoses += 1

        assert session._verifier_latched is True, "failed re-probe must stay latched"
        assert diagnoses == 1, (
            f"failed re-probes must not re-diagnose, got {diagnoses}"
        )
        # 2 latching calls + 1 failed re-probe per cycle.
        assert calls["n"] == 4, (
            f"expected 2 latching + 2 re-probe calls, got {calls['n']}"
        )
        reprobe_failures = [
            r for r in caplog.records
            if "re-probe failed — staying latched" in r.message
        ]
        assert len(reprobe_failures) == 2, (
            "a failed re-probe must log its own line, not re-announce the latch"
        )
    finally:
        await session.close()


async def test_latched_truncating_reprobe_stays_latched_without_rediagnosing(workspace):
    """A latched session whose re-probe TRUNCATES (the user switched from a
    hard-failing model to a persistently-verbose one) stays latched and does
    not re-diagnose: an exhausted ladder is now evidence about the model, so it
    takes the failed-re-probe path like any other no-verdict outcome. Only a
    successful verdict clears the latch.
    """
    from anton.core.llm.provider import StructuredOutputError
    from anton.core.session import _VERIFIER_LATCH_REPROBE_TURNS

    mock_llm = make_mock_llm()
    calls = {"n": 0}

    async def hard_then_truncated(_schema, *, system, messages, max_tokens):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("400 tool_choice not supported")
        raise StructuredOutputError(
            "no tool call", truncated=True, output_tokens=max_tokens,
            max_tokens=max_tokens, stop_reason="stop",
        )

    mock_llm.generate_object_code = AsyncMock(side_effect=hard_then_truncated)
    session = _make_session(workspace, mock_llm)

    diagnoses = 0
    try:
        for turn in range(2 + _VERIFIER_LATCH_REPROBE_TURNS):
            plan, state = _tool_then_text_plan()
            mock_llm.plan_stream = plan
            async for _ in session.turn_stream(f"step {turn}"):
                pass
            if state["n"] >= 3:
                diagnoses += 1

        assert session._verifier_latched is True, (
            "a truncating re-probe must not clear the latch"
        )
        assert diagnoses == 1, (
            f"the latch-time diagnosis only: a truncating re-probe must not "
            f"re-diagnose, got {diagnoses}"
        )
    finally:
        await session.close()


async def test_empty_diagnosis_is_logged_not_circular(workspace, caplog):
    """Review follow-up: an empty diagnosis must not silently recreate the
    out-of-sync history this path fixes — it logs, and the post-loop fallback
    keeps history consistent (stale-but-true beats empty)."""
    import logging

    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = AsyncMock(side_effect=RuntimeError("hiccup"))

    call_count = 0

    def fake_plan_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _FakeAsyncIter([
                StreamComplete(response=_scratchpad_response("Running.", "exec", "main", "print(1)"))
            ])
        if call_count == 2:
            return _FakeAsyncIter([
                StreamComplete(response=_text_response("Done running the script."))
            ])
        # Diagnosis comes back empty (refusal / provider hiccup).
        return _FakeAsyncIter([StreamComplete(response=_text_response(""))])

    mock_llm.plan_stream = fake_plan_stream

    session = ChatSession(ChatSessionConfig(llm_client=mock_llm, workspace=workspace))
    try:
        with caplog.at_level(logging.WARNING):
            async for _ in session.turn_stream("run my script"):
                pass

        assert any(
            "diagnosis returned no content" in r.message for r in caplog.records
        ), "an empty diagnosis must be visible in logs, not silent"
        # History must not gain an empty assistant entry; the post-loop
        # fallback re-appends the (real) pre-verification reply instead.
        assert all(
            m.get("content") not in ("",)
            for m in session.history
            if m.get("role") == "assistant"
        )
    finally:
        await session.close()


# ── Deterministic denials latch silently on the FIRST occurrence (ENG-1632) ──
#
# A wallet 402 / allowance 429 (TokenLimitExceeded, code-exact per ENG-1169) or
# a 404'd/403'd model (ModelUnavailableError) recurs on every retry by
# construction. The turn's work already succeeded — telling the user an
# internal check failed and streaming an apology is a misreport (measured: 208 aux-call wallet-402s across 33 users in
# 14 days, every one surfaced as an internal error). The denied path must be
# SILENT: no history injection, no diagnosis stream, latch on call one.
#
# ModelUnavailableError additionally subclasses ConnectionError → OSError, so
# before ENG-1632 it was swallowed by the TRANSIENT clause — diagnose every
# turn, never latch. These tests pin the except-clause ordering that fixes that.

@pytest.mark.parametrize("exc_factory, label", [
    (lambda: TokenLimitExceeded(
        "402: Your wallet has no balance to cover the model 'haiku'."), "wallet-402"),
    (lambda: TokenLimitExceeded(
        "429: Your included token allowance for 'haiku' is exhausted."), "allowance-429"),
    (lambda: ModelUnavailableError(
        "404: The model 'gemini-3.6-flash' does not exist",
        code="model_not_found", model="gemini-3.6-flash"), "model-404"),
])
async def test_denied_verdict_latches_silently_on_first_occurrence(
    workspace, caplog, exc_factory, label
):
    import logging

    mock_llm = make_mock_llm()
    calls = {"n": 0}

    async def always_denied(_schema, *, system, messages, max_tokens):
        calls["n"] += 1
        raise exc_factory()

    mock_llm.generate_object_code = AsyncMock(side_effect=always_denied)
    session = _make_session(workspace, mock_llm)

    diagnoses = 0
    progress_messages: list[str] = []
    try:
        with caplog.at_level(logging.INFO):
            for turn in range(4):
                plan, state = _tool_then_text_plan()
                mock_llm.plan_stream = plan
                async for event in session.turn_stream(f"do step {turn}"):
                    if isinstance(event, StreamTaskProgress):
                        progress_messages.append(event.message or "")
                if state["n"] >= 3:
                    diagnoses += 1

        # Silent: no diagnosis stream ever fires, and the user never sees the
        # hand-back's progress line.
        assert diagnoses == 0, (
            f"denied verdicts must not stream a diagnosis, got {diagnoses}"
        )
        assert not any(_HANDBACK_PROGRESS in m for m in progress_messages)
        # The hand-back SYSTEM injection must never enter history — it persists
        # into every later turn's payload once appended.
        assert not any(
            _HANDBACK_ANCHOR in str(m.get("content", ""))
            for m in session._history
            if isinstance(m, dict)
        ), "the hand-back injection reached history on a denied verdict"
        # Latched on the FIRST call: turns 2-4 pay nothing.
        assert calls["n"] == 1, (
            f"denied verdict must latch on the first occurrence, got {calls['n']} calls"
        )
        assert session._verifier_latched is True
        # Denied is not a capability failure — the hard counter stays clean so
        # a later hard failure still gets its honest one-per-session diagnosis.
        assert session._verifier_no_verdict_failures == 0
        announcements = [
            r for r in caplog.records
            if "latched after a deterministic denial" in r.message
        ]
        assert len(announcements) == 1, (
            f"the denied latch must announce itself exactly once in the log, "
            f"got {len(announcements)}"
        )
        # The per-turn skip log names the real cause, not "0 hard failures".
        skips = [
            r for r in caplog.records
            if "completion-verifier skipped — latched" in r.getMessage()
        ]
        assert len(skips) == 3
        assert all("deterministic denial" in r.getMessage() for r in skips)
    finally:
        await session.close()


async def test_denied_reprobe_stays_latched_and_silent(workspace, caplog):
    """After _VERIFIER_LATCH_REPROBE_TURNS skips, the latch re-probes once; a
    still-denied re-probe must land back in the denied branch — stay latched,
    stay silent, no diagnosis — so a top-up self-heals on a later re-probe
    while a persistent denial costs one quiet call per re-probe window."""
    import logging

    from anton.core.session import _VERIFIER_LATCH_REPROBE_TURNS

    mock_llm = make_mock_llm()
    calls = {"n": 0}

    async def always_denied(_schema, *, system, messages, max_tokens):
        calls["n"] += 1
        raise TokenLimitExceeded(
            "402: Your wallet has no balance to cover the model 'haiku'."
        )

    mock_llm.generate_object_code = AsyncMock(side_effect=always_denied)
    session = _make_session(workspace, mock_llm)

    diagnoses = 0
    try:
        with caplog.at_level(logging.INFO):
            for turn in range(_VERIFIER_LATCH_REPROBE_TURNS + 2):
                plan, state = _tool_then_text_plan()
                mock_llm.plan_stream = plan
                async for _ in session.turn_stream(f"do step {turn}"):
                    pass
                if state["n"] >= 3:
                    diagnoses += 1

        # Call 1 latches; the single re-probe is call 2; everything else skips.
        assert calls["n"] == 2, (
            f"expected first call + one re-probe, got {calls['n']}"
        )
        assert session._verifier_latched is True
        assert diagnoses == 0
        assert not any(
            _HANDBACK_ANCHOR in str(m.get("content", ""))
            for m in session._history
            if isinstance(m, dict)
        )
    finally:
        await session.close()


# ── Carrying the latch across a rebuilt session ──────────────────────────────
#
# The latch is ChatSession state and Cowork rebuilds ChatSession per message, so
# the counter restarted at zero on every message: one message contributes at
# most one failure, the threshold is two, and every message paid a hand-back.
# Carrying the counter is what lets message one diagnose and the rest go quiet.
# The `verification_skipped` stamp on these paths is covered by
# test_turn_cost_terminals.py; nothing here adds a stamp site.


def _always_truncated():
    from anton.core.llm.provider import StructuredOutputError

    async def truncated(_schema, *, system, messages, max_tokens):
        raise StructuredOutputError(
            "no tool call", truncated=True, output_tokens=max_tokens,
            max_tokens=max_tokens, stop_reason="stop",
        )

    return truncated


async def _run_one_turn(session, mock_llm, message: str):
    plan, state = _tool_then_text_plan()
    mock_llm.plan_stream = plan
    progress = []
    async for event in session.turn_stream(message):
        if isinstance(event, StreamTaskProgress):
            progress.append(event.message or "")
    return progress, state


async def test_a_failed_reprobe_updates_the_latch_attribution(workspace):
    """Latching `hard` then re-probing into a truncation must not keep booking
    `latched_hard`: the books would name a cause that is no longer the cause."""
    from anton.core.llm.provider import StructuredOutputError
    from anton.core.session import _VERIFIER_LATCH_REPROBE_TURNS

    mock_llm = make_mock_llm()
    calls = {"n": 0}

    async def hard_then_truncated(_schema, *, system, messages, max_tokens):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("400 tool_choice not supported")
        raise StructuredOutputError(
            "no tool call", truncated=True, output_tokens=max_tokens,
            max_tokens=max_tokens, stop_reason="stop",
        )

    mock_llm.generate_object_code = AsyncMock(side_effect=hard_then_truncated)
    session = _make_session(workspace, mock_llm)
    try:
        for turn in range(2):
            await _run_one_turn(session, mock_llm, f"step {turn}")
        assert session._verifier_latched is True
        assert session._verifier_latch_reason == "hard"

        for turn in range(_VERIFIER_LATCH_REPROBE_TURNS):
            await _run_one_turn(session, mock_llm, f"later {turn}")

        # The re-probe truncated, so the attribution is no longer purely hard.
        assert session._verifier_latch_reason == "mixed", (
            "a re-probe failing a different way must reach the books"
        )
    finally:
        await session.close()


async def test_mixed_classes_do_not_depend_on_arrival_order(workspace):
    """One exhausted ladder plus one hard rejection is neither latched_hard nor
    latched_truncated, whichever arrived last."""
    from anton.core.llm.provider import StructuredOutputError

    state = {"turn": 0}

    async def truncated_then_hard(_schema, *, system, messages, max_tokens):
        if state["turn"] == 0:
            raise StructuredOutputError(
                "no tool call", truncated=True, output_tokens=max_tokens,
                max_tokens=max_tokens, stop_reason="stop",
            )
        raise RuntimeError("400 tool_choice not supported")

    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = AsyncMock(side_effect=truncated_then_hard)
    session = _make_session(workspace, mock_llm)
    try:
        await _run_one_turn(session, mock_llm, "step one")
        assert session._verifier_latch_reason == "truncated"

        state["turn"] = 1
        await _run_one_turn(session, mock_llm, "step two")
        assert session._verifier_latched is True
        assert session._verifier_latch_reason == "mixed"
    finally:
        await session.close()


async def test_a_denied_latch_accumulates_when_the_reprobe_fails_differently(workspace):
    """A denial latches on call one with the counter untouched. If its re-probe
    then fails for a capability reason, the denial does NOT drop out of the
    books: one call failing a different way is no evidence the wallet was topped
    up, and `denied` is the only class with a user remedy attached to it.
    """
    from anton.core.session import _VERIFIER_LATCH_REPROBE_TURNS

    mock_llm = make_mock_llm()
    calls = {"n": 0}

    async def denied_then_hard(_schema, *, system, messages, max_tokens):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TokenLimitExceeded(
                "402: Your wallet has no balance to cover the model 'haiku'."
            )
        raise RuntimeError("400 tool_choice not supported")

    mock_llm.generate_object_code = AsyncMock(side_effect=denied_then_hard)
    session = _make_session(workspace, mock_llm)
    try:
        await _run_one_turn(session, mock_llm, "message one")
        assert session._verifier_latched is True
        assert session._verifier_latch_reason == "denied"
        assert session._verifier_no_verdict_failures == 0, (
            "a denial latches on its own branch without counting"
        )

        for turn in range(_VERIFIER_LATCH_REPROBE_TURNS):
            await _run_one_turn(session, mock_llm, f"later {turn}")

        assert calls["n"] == 2, "exactly one re-probe should have spent a call"
        assert session._verifier_latched is True
        assert session._verifier_latch_reason == "mixed", (
            "a differently-failing re-probe is no evidence the denial was paid, "
            "so the denial stays in the evidence"
        )
        assert session._verifier_last_no_verdict == "hard", (
            "the window follows what failed last, not the accumulated evidence"
        )
    finally:
        await session.close()


def test_the_reprobe_window_follows_the_last_failure_not_the_evidence():
    """The window is a prediction about what a re-probe would hit, so it keys on
    the LAST no-verdict class. A truncation can recover, a rejected `tool_choice`
    cannot, and a truncation ten turns ago says nothing about a model that has
    been rejecting the call ever since."""
    from anton.core.session import (
        _VERIFIER_LATCH_REPROBE_TURNS,
        _VERIFIER_LATCH_REPROBE_TURNS_TRUNCATED,
        _reprobe_turns_for,
    )

    assert _VERIFIER_LATCH_REPROBE_TURNS_TRUNCATED < _VERIFIER_LATCH_REPROBE_TURNS
    assert _reprobe_turns_for("truncated") == _VERIFIER_LATCH_REPROBE_TURNS_TRUNCATED
    assert _reprobe_turns_for("hard") == _VERIFIER_LATCH_REPROBE_TURNS
    assert _reprobe_turns_for("denied") == _VERIFIER_LATCH_REPROBE_TURNS
    assert _reprobe_turns_for("") == _VERIFIER_LATCH_REPROBE_TURNS


async def test_the_short_window_is_given_back_when_truncation_stops(workspace):
    """Regression: keying the window on the ACCUMULATED reason made "mixed"
    absorbing, so one truncation during a hard latch's re-probe dropped the
    session to the short window for good — re-probing a deterministic 400 three
    times as often forever, against a cause that can never clear. The evidence
    stays mixed; the window follows what failed last and comes back."""
    from anton.core.llm.provider import StructuredOutputError
    from anton.core.session import (
        _VERIFIER_LATCH_REPROBE_TURNS,
        _VERIFIER_LATCH_REPROBE_TURNS_TRUNCATED,
    )

    mock_llm = make_mock_llm()
    calls = {"n": 0}

    async def hard_then_one_truncating_reprobe(_schema, *, system, messages, max_tokens):
        calls["n"] += 1
        # 1-2 latch `hard`; 3-4 are one re-probe's ladder, truncating; the 400
        # is back from then on.
        if calls["n"] in (3, 4):
            raise StructuredOutputError(
                "no tool call", truncated=True, output_tokens=max_tokens,
                max_tokens=max_tokens, stop_reason="stop",
            )
        raise RuntimeError("400 tool_choice not supported")

    mock_llm.generate_object_code = AsyncMock(
        side_effect=hard_then_one_truncating_reprobe
    )
    session = _make_session(workspace, mock_llm)
    try:
        for turn in range(2):
            await _run_one_turn(session, mock_llm, f"step {turn}")
        assert session._verifier_latch_reason == "hard"

        for turn in range(_VERIFIER_LATCH_REPROBE_TURNS):
            await _run_one_turn(session, mock_llm, f"later {turn}")
        assert session._verifier_latch_reason == "mixed", (
            "the truncating re-probe joins the evidence"
        )
        assert session._verifier_last_no_verdict == "truncated"

        # Short window now, correctly: a truncation is what failed last.
        for turn in range(_VERIFIER_LATCH_REPROBE_TURNS_TRUNCATED):
            await _run_one_turn(session, mock_llm, f"short {turn}")
        assert session._verifier_last_no_verdict == "hard", (
            "that re-probe hit the 400 again"
        )

        # ...and the long window is back, because the 400 is what fails now.
        spent = calls["n"]
        for turn in range(_VERIFIER_LATCH_REPROBE_TURNS - 1):
            await _run_one_turn(session, mock_llm, f"after {turn}")
        assert calls["n"] == spent, (
            "a mixed latch whose last failure was a 400 must get the long window "
            "back; an absorbing 'mixed' re-probes at 3 turns forever"
        )
        assert session._verifier_latch_reason == "mixed", (
            "the accumulated evidence still includes the truncation"
        )
    finally:
        await session.close()


async def test_a_truncation_latch_reprobes_within_the_short_window(workspace):
    """End to end: the latch engages on the second exhausted ladder, then the
    verifier is tried again after the short window rather than the long one."""
    from anton.core.session import _VERIFIER_LATCH_REPROBE_TURNS_TRUNCATED

    mock_llm = make_mock_llm()
    mock_llm.generate_object_code = AsyncMock(side_effect=_always_truncated())
    session = _make_session(workspace, mock_llm)
    try:
        for turn in range(2):
            await _run_one_turn(session, mock_llm, f"step {turn}")
        assert session._verifier_latched is True
        assert session._verifier_latch_reason == "truncated"
        # Two verifying turns, two budgets each.
        latch_calls = mock_llm.generate_object_code.await_count
        assert latch_calls == 4

        for turn in range(_VERIFIER_LATCH_REPROBE_TURNS_TRUNCATED - 1):
            await _run_one_turn(session, mock_llm, f"skip {turn}")
        assert mock_llm.generate_object_code.await_count == latch_calls, (
            "still inside the window: no verdict call"
        )

        await _run_one_turn(session, mock_llm, "re-probe turn")
        assert mock_llm.generate_object_code.await_count > latch_calls, (
            "the short window must let the re-probe fire"
        )
    finally:
        await session.close()
