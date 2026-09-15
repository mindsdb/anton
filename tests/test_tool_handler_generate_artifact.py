"""handle_generate_artifact: FSM failures must come back wrapped with an
instruction to report to the user (never DIY the artifact); input-validation
errors stay unwrapped so the agent fixes its call instead.

The handler is an async generator (ENG-970), so every case here drains it via
`_collect` and asserts on the single non-ToolProgress item — the tool result.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anton.core.tools.generate_artifact as gen_pkg
from anton.core.artifacts import ArtifactStore
from anton.core.tools.progress import ToolProgress
from anton.core.tools.tool_handlers import handle_generate_artifact


async def _collect_raw(session, tc_input, handler=handle_generate_artifact):
    """Drain the streaming handler into (progress lines, final item).

    The final item is whatever the handler yielded — a `ToolOutcome` on every
    path that reached the pipeline, a bare string for input validation, which
    the dispatcher still accepts.
    """
    progress: list[str] = []
    result = None
    async for item in handler(session, tc_input):
        if isinstance(item, ToolProgress):
            progress.append(item.text)
        else:
            result = item
    return progress, result


async def _collect(session, tc_input, handler=handle_generate_artifact):
    """Same, with the result flattened to its text for content assertions."""
    progress, result = await _collect_raw(session, tc_input, handler)
    return progress, getattr(result, "content", result)


def _session(tmp_path: Path):
    return SimpleNamespace(
        _workspace=SimpleNamespace(artifacts_dir=tmp_path / "artifacts")
    )


def _make_artifact(tmp_path: Path) -> str:
    store = ArtifactStore(tmp_path / "artifacts")
    return store.create(
        name="Clock", description="d", type="fullstack-stateless-app"
    ).slug


async def test_fsm_failure_is_wrapped_with_report_instruction(tmp_path: Path, monkeypatch):
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        return "Backend verification failed after retry: boom"

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    _, out = await _collect(_session(tmp_path), {"slug": slug, "user_request": "build it", "agent_understanding": "an app"})
    assert "artifact generation failed" in out
    assert "Backend verification failed after retry: boom" in out
    assert "do NOT build or repair the artifact yourself" in out
    assert "Report this failure to the user" in out


async def test_generator_crash_is_wrapped(tmp_path: Path, monkeypatch):
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        raise RuntimeError("kaput")

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    _, out = await _collect(_session(tmp_path), {"slug": slug, "user_request": "build it", "agent_understanding": "an app"})
    assert "artifact generation failed" in out
    assert "kaput" in out


async def test_input_validation_errors_are_not_wrapped(tmp_path: Path):
    _, out = await _collect(_session(tmp_path), {"slug": "nope", "user_request": "build it", "agent_understanding": "an app"})
    assert out.startswith("Error: no artifact found")
    assert "generation failed" not in out


async def test_success_returns_json_unchanged(tmp_path: Path, monkeypatch):
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        return {"files_written": ["backend.py"], "summary": "ok", "trace": []}

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    _, out = await _collect(_session(tmp_path), {"slug": slug, "user_request": "build it", "agent_understanding": "an app"})
    assert '"files_written"' in out
    assert "generation failed" not in out


async def test_handler_forwards_primary_to_generate(monkeypatch, tmp_path):
    """The primary from metadata must reach the generator."""
    import anton.core.tools.tool_handlers as th

    captured = {}

    async def fake_generate(**kw):
        captured.update(kw)
        return {"files_written": ["report.html"], "summary": "s", "trace": []}

    monkeypatch.setattr(
        "anton.core.tools.generate_artifact.generate", fake_generate, raising=False
    )

    class _Artifact:
        type = "html-app"
        slug = "a"
        primary = "report.html"

    class _Store:
        def open(self, slug):
            return _Artifact()

        def folder_for(self, slug):
            return tmp_path

    monkeypatch.setattr(th, "_artifact_store", lambda session: _Store())

    _, out = await _collect(
        object(), {"slug": "a", "user_request": "build it", "agent_understanding": "an app"},
        handler=th.handle_generate_artifact,
    )
    assert "report.html" in out
    assert captured["primary"] == "report.html"


# ── ENG-970: progress markers ────────────────────────────────────────────────

async def test_step_lines_are_yielded_as_progress_before_the_result(
    tmp_path: Path, monkeypatch
):
    """Lines the FSM pushes onto the channel must reach the consumer as
    ToolProgress markers, in order, ahead of the result — that ordering is
    the whole point: a marker arriving after the result would render as
    progress on a step that already finished."""
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        kw["progress"].put_nowait("Writing the backend")
        kw["progress"].put_nowait("Verifying the backend")
        return {"files_written": ["backend.py"], "summary": "ok", "trace": []}

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    progress, out = await _collect(
        _session(tmp_path), {"slug": slug, "user_request": "build it", "agent_understanding": "an app"}
    )
    assert progress == ["Writing the backend", "Verifying the backend"]
    assert '"files_written"' in out


async def test_progress_lines_are_relayed_while_generation_is_still_running(
    tmp_path: Path, monkeypatch
):
    """The channel must drain concurrently, not be flushed at the end: a
    marker's only value is arriving while the step it names is in flight."""
    slug = _make_artifact(tmp_path)
    released = asyncio.Event()

    async def fake_generate(**kw):
        kw["progress"].put_nowait("Writing the technical specification")
        await released.wait()  # generation cannot finish until the test says so
        return {"files_written": ["backend.py"], "summary": "ok", "trace": []}

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    agen = handle_generate_artifact(
        _session(tmp_path), {"slug": slug, "user_request": "build it", "agent_understanding": "an app"}
    )
    first = await agen.__anext__()
    assert first == ToolProgress("Writing the technical specification")
    released.set()
    rest = [item async for item in agen]
    assert len(rest) == 1 and '"files_written"' in rest[0].content


async def test_progress_lines_stop_at_a_crash_and_the_failure_still_arrives(
    tmp_path: Path, monkeypatch
):
    """A crash after some progress must still close the channel and produce
    the wrapped failure — the sentinel lives in a `finally` precisely so this
    does not hang."""
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        kw["progress"].put_nowait("Writing the backend")
        raise RuntimeError("kaput")

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    progress, out = await _collect(
        _session(tmp_path), {"slug": slug, "user_request": "build it", "agent_understanding": "an app"}
    )
    assert progress == ["Writing the backend"]
    assert "kaput" in out


async def test_abandoning_the_generator_cancels_generation(tmp_path: Path, monkeypatch):
    """Closing the generator mid-yield (turn cancelled) must not leave the FSM
    running: it would keep writing files into the artifact folder with nobody
    left to receive the result."""
    slug = _make_artifact(tmp_path)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def fake_generate(**kw):
        kw["progress"].put_nowait("Writing the backend")
        started.set()
        try:
            await asyncio.Event().wait()  # never completes on its own
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    agen = handle_generate_artifact(
        _session(tmp_path), {"slug": slug, "user_request": "build it", "agent_understanding": "an app"}
    )
    assert await agen.__anext__() == ToolProgress("Writing the backend")
    await started.wait()
    await agen.aclose()
    await asyncio.sleep(0)  # let the cancellation land
    assert cancelled.is_set()


async def test_validation_errors_yield_no_progress(tmp_path: Path):
    """A rejected call never reaches the FSM, so it must not open a step in
    the host's UI."""
    progress, out = await _collect(_session(tmp_path), {"user_request": "build it", "agent_understanding": "an app"})
    assert progress == []
    assert out == "Error: `slug` is required."


async def test_success_tells_the_agent_not_to_reverify(tmp_path: Path, monkeypatch):
    """Without this instruction the calling agent re-verifies the pipeline's
    work by hand — a measured 2026-08-27 run spent 11 planning-model calls
    re-reading an artifact the generator had already verified."""
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        return {"files_written": ["backend.py"], "summary": "ok", "trace": []}

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    _, out = await _collect(_session(tmp_path), {"slug": slug, "user_request": "build it", "agent_understanding": "an app"})
    assert '"instruction"' in out
    assert "Do NOT re-read" in out


# ── I-03: the handler states its own verdict ────────────────────────────────


def test_the_tool_schema_has_no_context_parameter():
    """`context` carried a three-section markdown contract duplicated across
    three ToolDef surfaces. Typed fields need no such contract."""
    from anton.core.tools.tool_defs import GENERATE_ARTIFACT_TOOL

    props = GENERATE_ARTIFACT_TOOL.input_schema["properties"]
    assert "context" not in props
    assert set(GENERATE_ARTIFACT_TOOL.input_schema["required"]) == {
        "slug", "user_request", "agent_understanding",
    }
    assert set(props) == {
        "slug", "user_request", "agent_understanding",
        "known_data", "user_preferences",
    }


def test_generate_prd_is_no_longer_a_tool():
    import anton.core.tools.tool_defs as td

    assert not hasattr(td, "GENERATE_PRD_TOOL")


async def test_a_successful_run_reports_ok_even_though_its_trace_says_failed(tmp_path, monkeypatch):
    """I-03. The legacy classifier substring-matched the result text, and a
    successful trace legitimately contains the word: "backend.py failed to
    import in venv" is what a SUCCESSFUL retry looks like. That counted as an
    error and fed the per-tool error streak."""
    from anton.core.tools.registry import ToolOutcome

    async def fake_generate(**kw):
        return {
            "status": "generated",
            "files_written": ["dashboard.html"],
            "internal_files": [],
            "summary": "verify_backend:fail; generate_backend:done",
            "trace": [{"node": "verify_backend", "outcome": "fail",
                       "detail": "backend.py failed to import in venv"}],
        }

    slug = _make_artifact(tmp_path)
    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    _, outcome = await _collect_raw(
        _session(tmp_path),
        {"slug": slug, "user_request": "build it", "agent_understanding": "an app"},
    )
    assert isinstance(outcome, ToolOutcome)
    assert outcome.ok is True
    assert "failed" in outcome.content  # the very substring that used to lie


async def test_a_pipeline_failure_reports_not_ok(tmp_path, monkeypatch):
    from anton.core.tools.registry import ToolOutcome

    async def fake_generate(**kw):
        return "make_tech_spec: the specification hit the output limit"

    slug = _make_artifact(tmp_path)
    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    _, outcome = await _collect_raw(
        _session(tmp_path),
        {"slug": slug, "user_request": "build it", "agent_understanding": "an app"},
    )
    assert isinstance(outcome, ToolOutcome)
    assert outcome.ok is False


# ── The question and the progress stream do not overlap ─────────────────────


async def test_progress_is_silent_while_a_question_is_open():
    """The pipeline asks the user from inside the task this handler drains.
    A marker printed over a live prompt corrupts it, and the CLI's Live
    context does not survive that."""
    import asyncio

    from anton.core.tools.generate_artifact.progress import (
        QUESTION_CLOSED,
        QUESTION_OPEN,
    )
    from anton.core.tools.tool_handlers import _drain_progress

    queue: asyncio.Queue = asyncio.Queue()
    for item in (
        "Gathering what the artifact needs",
        QUESTION_OPEN,
        "Writing down the agreed requirements",   # produced while the prompt was up
        "Designing the API",                      # ditto
        QUESTION_CLOSED,
        "Writing the page",
        None,
    ):
        queue.put_nowait(item)

    lines = [m.text async for m in _drain_progress(queue)]
    assert lines[0] == "Gathering what the artifact needs"
    # Everything produced while the question was open collapses into at most
    # one line, emitted only after the answer — and never a raw sentinel.
    assert QUESTION_OPEN not in lines and QUESTION_CLOSED not in lines
    assert lines[-1] == "Writing the page"
    assert "Writing down the agreed requirements" not in lines


async def test_nested_question_sentinels_do_not_unmute_early():
    """`show_and_confirm` wraps a call that wraps itself, so the sentinels
    arrive nested. A boolean flag would unmute on the inner CLOSED, while the
    brief is still on screen."""
    import asyncio

    from anton.core.tools.generate_artifact.progress import (
        QUESTION_CLOSED,
        QUESTION_OPEN,
    )
    from anton.core.tools.tool_handlers import _drain_progress

    queue: asyncio.Queue = asyncio.Queue()
    for item in (
        QUESTION_OPEN, QUESTION_OPEN,
        "Writing down the agreed requirements",   # deep inside the question
        QUESTION_CLOSED,
        "Designing the API",                      # still inside the outer one
        QUESTION_CLOSED,
        None,
    ):
        queue.put_nowait(item)

    lines = [m.text async for m in _drain_progress(queue)]
    assert lines == ["Designing the API"]


async def test_the_brief_confirmation_mutes_progress(tmp_path):
    """The longest question of the run goes through `show_and_confirm`, which
    reaches `elicit` on its own path. Wrapping only the `ask_user` sub-tool
    would leave exactly this one unprotected."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from anton.core.interaction.elicit import AskAnswer
    from anton.core.tools.generate_artifact.discovery import brief
    from anton.core.tools.generate_artifact.progress import (
        QUESTION_CLOSED,
        QUESTION_OPEN,
    )
    from anton.core.tools.generate_artifact.state import GenState

    queue: asyncio.Queue = asyncio.Queue()
    seen_when_asked: list[str] = []

    async def fake_ask(session, request):
        while not queue.empty():
            seen_when_asked.append(queue.get_nowait())
        return AskAnswer(status="answered", values=("accept",))

    session = SimpleNamespace(
        _llm=SimpleNamespace(), question_count=0, elicitor=None,
        emit=AsyncMock(), _artifact_progress=queue,
    )
    state = GenState(
        session=session, artifact_type="html-app", artifact_path=tmp_path,
        slug="s", brief="## Goal\nA clock.",
    )

    import unittest.mock

    with unittest.mock.patch.object(brief.sub_tools, "ask_via_elicit", fake_ask):
        outcome = await brief.show_and_confirm(state)

    assert outcome == "accepted"
    assert QUESTION_OPEN in seen_when_asked, "the channel was not muted before asking"
    remaining = []
    while not queue.empty():
        remaining.append(queue.get_nowait())
    assert QUESTION_CLOSED in remaining, "the channel was never unmuted"


# ── Status instructions: what the outer agent is told to do next ─────────────
#
# Re-homed from the deleted test_tool_handler_generate_prd.py. These strings
# are the whole interface between a non-terminal pipeline outcome and the
# agent holding the conversation: get them wrong and the agent either writes
# prd.md by hand or loops calling the tool forever. Asserted through the
# handler rather than against the dict, because the bug worth catching is a
# status the handler never routes.


async def _status_result(tmp_path: Path, monkeypatch, payload: dict):
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        return payload

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    _, out = await _collect(
        _session(tmp_path),
        {"slug": slug, "user_request": "build it", "agent_understanding": "an app"},
    )
    return out


async def test_cancelled_forbids_doing_it_by_hand(tmp_path: Path, monkeypatch):
    out = await _status_result(
        tmp_path, monkeypatch, {"status": "cancelled", "reason": "user declined the brief"}
    )
    assert "declined" in out
    assert "do NOT build the artifact by hand" in out
    assert "generation failed" not in out


async def test_needs_confirmation_tells_the_agent_the_repeat_call_confirms(
    tmp_path: Path, monkeypatch
):
    """The convergence rule. Without "the repeat call is the confirmation"
    the agent has no way to ever get past this status."""
    out = await _status_result(
        tmp_path, monkeypatch,
        {"status": "needs_confirmation", "brief_summary": "## Goal\nA clock."},
    )
    assert "brief_summary" in out
    assert "the repeat call is the confirmation" in out
    assert "SAME `user_request`" in out


async def test_a_budget_stop_asks_the_user_rather_than_reporting_failure(
    tmp_path: Path, monkeypatch
):
    out = await _status_result(
        tmp_path, monkeypatch, {"status": "stopped_over_budget", "reason": "ceiling"}
    )
    assert "budget" in out
    assert "resumes rather than restarting" in out
    assert "generation failed" not in out


async def test_an_unknown_status_falls_back_to_the_generated_instruction(
    tmp_path: Path, monkeypatch
):
    """A status added to the pipeline but not to the table must not drop the
    instruction field altogether — the agent would be left with raw JSON."""
    out = await _status_result(tmp_path, monkeypatch, {"status": "brand_new", "files_written": []})
    assert "instruction" in out
    assert "report the result to the user" in out


async def test_a_cancelled_run_is_not_tracked_as_generated_files(
    tmp_path: Path, monkeypatch
):
    """Cancelling writes nothing, so the turn has no artifact work to
    attribute. Every other outcome does."""
    import anton.core.tools.tool_handlers as th

    tracked: list[str] = []
    monkeypatch.setattr(
        th, "_track_artifact",
        lambda session, store, slug, summary="": tracked.append(summary),
    )

    await _status_result(tmp_path, monkeypatch, {"status": "cancelled", "reason": "declined"})
    assert tracked == []

    await _status_result(tmp_path, monkeypatch, {"status": "generated", "files_written": ["a.html"]})
    assert tracked == ["Generated artifact files"]


# ── Required fields ─────────────────────────────────────────────────────────

async def test_a_missing_user_request_is_rejected_without_starting_the_pipeline(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        gen_pkg, "generate",
        AsyncMock(side_effect=AssertionError("the pipeline must not start")),
    )
    slug = _make_artifact(tmp_path)
    _, out = await _collect(
        _session(tmp_path), {"slug": slug, "agent_understanding": "an app"}
    )
    assert out.startswith("Error:")
    assert "user_request" in out
    assert "generation failed" not in out


async def test_a_missing_agent_understanding_is_rejected(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        gen_pkg, "generate",
        AsyncMock(side_effect=AssertionError("the pipeline must not start")),
    )
    slug = _make_artifact(tmp_path)
    _, out = await _collect(
        _session(tmp_path), {"slug": slug, "user_request": "build it"}
    )
    assert out.startswith("Error:")
    assert "agent_understanding" in out
