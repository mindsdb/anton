"""handle_generate_artifact: FSM failures must come back wrapped with an
instruction to report to the user (never DIY the artifact); input-validation
errors stay unwrapped so the agent fixes its call instead.

The handler is an async generator, so every case here drains it via
`_collect` and asserts on the single non-ToolProgress item — the tool result.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import anton.core.tools.generate_artifact as gen_pkg
from anton.core.artifacts import ArtifactStore
from anton.core.llm.client import LLMClient
from anton.core.llm.provider import LLMProvider, LLMResponse, StreamComplete, Usage
from anton.core.tools.progress import ToolProgress
from anton.core.tools.tool_handlers import handle_generate_artifact
from tests.streaming_llm import StreamingLLM


async def _collect_raw(session, tc_input, handler=handle_generate_artifact):
    """Drain the streaming handler into (progress lines, final item).

    The final item is whatever the handler yielded — a `ToolOutcome` on every
    path that reached the pipeline, a bare string for input validation, which
    the dispatcher still accepts. `user_language` defaults to English so
    each case states only the fields it is about; pass it to override.
    """
    tc_input = {"user_language": "English", **tc_input}
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


def _make_artifact(tmp_path: Path, type: str = "fullstack-stateless-app") -> str:
    store = ArtifactStore(tmp_path / "artifacts")
    return store.create(name="Clock", description="d", type=type).slug


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


async def test_missing_data_tells_the_agent_what_to_ask_the_user_for(tmp_path: Path, monkeypatch):
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        return "not enough data: no connection holds the sites. Looked in: rest-gw."

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    _, out = await _collect(_session(tmp_path), {"slug": slug, "user_request": "build it", "agent_understanding": "an app"})
    assert out.startswith("Error: the data this artifact needs is not available.")
    assert "no connection holds the sites. Looked in: rest-gw." in out
    assert "do NOT build the artifact yourself" in out
    assert "pass the data in `known_data`" in out
    assert "artifact generation failed" not in out
    assert "ask how they want to proceed" not in out


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

    monkeypatch.setattr(th, "resolve_artifact_store", lambda session: _Store())

    _, out = await _collect(
        object(), {"slug": "a", "user_request": "build it", "agent_understanding": "an app"},
        handler=th.handle_generate_artifact,
    )
    assert "report.html" in out
    assert captured["primary"] == "report.html"


# ── progress markers ─────────────────────────────────────────────────────────

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
        _session(tmp_path), {"slug": slug, "user_request": "build it", "agent_understanding": "an app",
        "user_language": "English"}
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
        _session(tmp_path), {"slug": slug, "user_request": "build it", "agent_understanding": "an app",
        "user_language": "English"}
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
    work by hand — a live run spent 11 planning-model calls
    re-reading an artifact the generator had already verified."""
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        return {"files_written": ["backend.py"], "summary": "ok", "trace": [], "verified_in_browser": True}

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
        "slug", "user_request", "user_language", "agent_understanding",
    }
    assert set(props) == {
        "slug", "user_request", "user_language", "agent_understanding",
        "known_data", "user_preferences", "attachments",
    }


def test_the_tool_tells_the_agent_to_keep_the_users_language():
    """The pipeline writes in `user_language`; the agent relays the result
    in the same language."""
    from anton.core.tools.tool_defs import GENERATE_ARTIFACT_TOOL

    assert "the language the user writes to you in" in GENERATE_ARTIFACT_TOOL.description
    assert "talk to the user in that language" in GENERATE_ARTIFACT_TOOL.prompt


async def test_the_handler_passes_the_user_language_to_the_pipeline(
    tmp_path: Path, monkeypatch
):
    slug = _make_artifact(tmp_path)
    seen = {}

    async def fake_generate(**kw):
        seen.update(kw)
        return {"files_written": [], "summary": "ok", "trace": []}

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    await _collect(
        _session(tmp_path),
        {
            "slug": slug, "user_request": "build it", "agent_understanding": "an app",
            "user_language": " Brazilian\n  Portuguese ",
        },
    )
    assert seen["user_language"] == "Brazilian Portuguese"


async def test_the_handler_rejects_a_user_language_longer_than_a_name(
    tmp_path: Path, monkeypatch
):
    """The value is quoted into every step header and the call fingerprint,
    so a sentence about the user is refused before the pipeline starts."""
    from anton.core.tools.generate_artifact.state import MAX_USER_LANGUAGE_CHARS

    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        raise AssertionError("the pipeline must not start")

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    _, out = await _collect(
        _session(tmp_path),
        {
            "slug": slug, "user_request": "build it", "agent_understanding": "an app",
            "user_language": "x" * (MAX_USER_LANGUAGE_CHARS + 1),
        },
    )
    assert out.startswith("Error: `user_language` must be only the language's English name")
    assert f"got {MAX_USER_LANGUAGE_CHARS + 1} characters" in out


async def test_the_handler_accepts_a_user_language_at_the_cap(tmp_path: Path, monkeypatch):
    from anton.core.tools.generate_artifact.state import MAX_USER_LANGUAGE_CHARS

    slug = _make_artifact(tmp_path)
    seen = {}

    async def fake_generate(**kw):
        seen.update(kw)
        return {"files_written": [], "summary": "ok", "trace": []}

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    value = "x" * MAX_USER_LANGUAGE_CHARS
    await _collect(
        _session(tmp_path),
        {"slug": slug, "user_request": "build it", "agent_understanding": "an app", "user_language": value},
    )
    assert seen["user_language"] == value


def test_the_soft_input_fields_forbid_the_outer_agent_from_inventing_scope():
    """Repeated live runs on an eight-word request: the outer agent put ten
    features into `agent_understanding` and, by the fifth run, wrote into
    `user_preferences` that the user "likes combo scoring and localStorage
    records" — features it had itself built the time before. The pipeline
    labelled all of it as proposals, but seven proposals on a short
    request are accepted unread. The fix has to be at the source."""
    from anton.core.tools.tool_defs import GENERATE_ARTIFACT_TOOL

    props = GENERATE_ARTIFACT_TOOL.input_schema["properties"]
    understanding = props["agent_understanding"]["description"]
    assert "only what they said" in understanding
    assert "Do not add features" in understanding
    preferences = props["user_preferences"]["description"]
    assert "stated themselves" in preferences
    assert "Never infer a preference from an artifact you built before" in preferences
    # The prose description carries the same two rules.
    assert "Do NOT add features" in GENERATE_ARTIFACT_TOOL.description
    assert "Never infer a preference" in GENERATE_ARTIFACT_TOOL.description


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


async def test_a_message_line_becomes_a_message_marker_in_place():
    import asyncio

    from anton.core.tools.generate_artifact.progress import MESSAGE_PREFIX
    from anton.core.tools.tool_handlers import _drain_progress

    queue: asyncio.Queue = asyncio.Queue()
    for item in (
        "Preparing a short brief for you",
        MESSAGE_PREFIX + "## Brief\nline two",
        "Writing down the requirements",
        None,
    ):
        queue.put_nowait(item)

    markers = [(m.kind, m.text) async for m in _drain_progress(queue)]
    assert markers == [
        ("step", "Preparing a short brief for you"),
        ("message", "## Brief\nline two"),
        ("step", "Writing down the requirements"),
    ]


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


async def test_a_budget_stop_tells_the_agent_to_show_the_unseen_brief(
    tmp_path: Path, monkeypatch
):
    """A budget stop in phase B leaves the checkpoint
    at `awaiting_confirmation`, and the repeat call enters `ENTRY_CONFIRM`,
    which treats the call itself as agreement. The instruction has to make
    the agent show `brief_summary` first, or the app is built to a brief
    nobody saw."""
    out = await _status_result(
        tmp_path, monkeypatch,
        {"status": "stopped_over_budget", "reason": "ceiling", "brief_summary": "Goal: a clock"},
    )
    assert "`brief_summary`" in out
    assert "taken as their agreement" in out
    assert "`agent_understanding`" in out


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


@pytest.mark.parametrize("value", ["", "   ", None, ["English"]])
async def test_a_missing_or_malformed_user_language_is_rejected(tmp_path: Path, monkeypatch, value):
    """Without a named language the steps fall back to a default instead of
    the user's language, so the agent is sent back to state it."""
    monkeypatch.setattr(
        gen_pkg, "generate",
        AsyncMock(side_effect=AssertionError("the pipeline must not start")),
    )
    slug = _make_artifact(tmp_path)
    _, out = await _collect(
        _session(tmp_path),
        {"slug": slug, "user_request": "build it", "agent_understanding": "an app",
         "user_language": value},
    )
    assert out.startswith("Error:")
    assert "user_language" in out


async def test_a_peek_line_is_relayed_as_a_peek_and_dropped_while_a_question_is_open():
    """The live tail replaces itself in the footer, so it must not be printed
    as a step line — and unlike a step line it is not worth holding through a
    question: by the time the answer comes it is stale."""
    import asyncio

    from anton.core.tools.generate_artifact.progress import (
        PEEK_PREFIX,
        QUESTION_CLOSED,
        QUESTION_OPEN,
    )
    from anton.core.tools.tool_handlers import _drain_progress

    queue: asyncio.Queue = asyncio.Queue()
    for line in (
        "Writing the page (step 3 of 4)", PEEK_PREFIX + "<html>\n<body>",
        QUESTION_OPEN, PEEK_PREFIX + "<div>", "Verifying the page (step 4 of 4)",
        QUESTION_CLOSED, PEEK_PREFIX + "", None,
    ):
        queue.put_nowait(line)
    markers = [m async for m in _drain_progress(queue)]
    assert [(m.text, m.kind) for m in markers] == [
        ("Writing the page (step 3 of 4)", "step"),
        ("<html>\n<body>", "peek"),
        ("Verifying the page (step 4 of 4)", "step"),
        ("", "peek"),
    ]


def test_known_data_asks_the_outer_agent_to_name_the_scratchpad():
    """Without the pad name the gathering step cannot reuse the agent's
    data and guesses a new pad — refused by the single-scratchpad guard."""
    from anton.core.tools.tool_defs import GENERATE_ARTIFACT_TOOL

    schema_text = GENERATE_ARTIFACT_TOOL.input_schema["properties"]["known_data"]["description"]
    assert "a reference to data in a scratchpad" in schema_text
    assert "the scratchpad in backticks" in schema_text
    assert "pass a reference instead of the rows" in GENERATE_ARTIFACT_TOOL.description


def test_known_data_asks_for_the_rows_and_says_a_new_request_does_not_inherit_it():
    """The gathering rule counts only the values themselves, and a new
    iteration does not reuse the saved copy: the agent has to know both."""
    from anton.core.tools.tool_defs import GENERATE_ARTIFACT_TOOL

    schema_text = GENERATE_ARTIFACT_TOOL.input_schema["properties"]["known_data"]["description"]
    for text in (schema_text, GENERATE_ARTIFACT_TOOL.description):
        assert "values themselves" in text
    assert "pass it again" in schema_text
    assert "Pass it on every call that needs it" in GENERATE_ARTIFACT_TOOL.description


# ── S-03: the agent relays skipped checks and warnings ──────────────────────

async def test_success_instruction_tells_the_agent_to_relay_skipped_checks(tmp_path: Path, monkeypatch):
    """The same instruction that forbids re-verifying the files has to name
    the two fields that say what the pipeline itself did not check — or a
    skipped browser load is reported to the user as a passed one."""
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        return {
            "files_written": ["index.html"], "summary": "ok", "trace": [],
            "warnings": [], "checks_skipped": ["browser load of the page: no browser"],
        }

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    _, out = await _collect(_session(tmp_path), {"slug": slug, "user_request": "build it", "agent_understanding": "a page"})
    assert '"checks_skipped"' in out
    assert "never present a skipped check as passed" in out


def test_the_tool_description_names_the_two_report_fields():
    from anton.core.tools.tool_defs import GENERATE_ARTIFACT_TOOL

    assert "`checks_skipped`" in GENERATE_ARTIFACT_TOOL.description
    assert "`warnings`" in GENERATE_ARTIFACT_TOOL.description


# ── S-01: attachments reach the generator ───────────────────────────────────

async def test_handler_forwards_attachments_to_generate(monkeypatch, tmp_path):
    import anton.core.tools.tool_handlers as th

    captured = {}

    async def fake_generate(**kw):
        captured.update(kw)
        return {"files_written": [], "summary": "s", "trace": []}

    monkeypatch.setattr("anton.core.tools.generate_artifact.generate", fake_generate, raising=False)

    class _Artifact:
        type = "html-app"
        slug = "a"
        primary = "index.html"

    class _Store:
        def open(self, slug):
            return _Artifact()

        def folder_for(self, slug):
            return tmp_path

    monkeypatch.setattr(th, "resolve_artifact_store", lambda session: _Store())

    await _collect(
        object(),
        {"slug": "a", "user_request": "build it", "agent_understanding": "a page",
         "attachments": ["/tmp/u/logo.png", "  ", None, "/tmp/u/sales.csv"]},
        handler=th.handle_generate_artifact,
    )
    assert captured["attachments"] == ["/tmp/u/logo.png", "/tmp/u/sales.csv"]


async def test_handler_accepts_a_single_attachment_path_as_a_string(monkeypatch, tmp_path):
    import anton.core.tools.tool_handlers as th

    captured = {}

    async def fake_generate(**kw):
        captured.update(kw)
        return {"files_written": [], "summary": "s", "trace": []}

    monkeypatch.setattr("anton.core.tools.generate_artifact.generate", fake_generate, raising=False)

    class _Artifact:
        type = "html-app"
        slug = "a"
        primary = None

    class _Store:
        def open(self, slug):
            return _Artifact()

        def folder_for(self, slug):
            return tmp_path

    monkeypatch.setattr(th, "resolve_artifact_store", lambda session: _Store())
    await _collect(
        object(),
        {"slug": "a", "user_request": "build it", "agent_understanding": "a page", "attachments": "/tmp/u/logo.png"},
        handler=th.handle_generate_artifact,
    )
    assert captured["attachments"] == ["/tmp/u/logo.png"]


def test_the_attachments_field_tells_the_agent_what_qualifies():
    from anton.core.tools.tool_defs import GENERATE_ARTIFACT_TOOL

    desc = GENERATE_ARTIFACT_TOOL.input_schema["properties"]["attachments"]["description"]
    assert "outside the workspace" in desc and "`.anton/`" in desc
    assert "Only files the user actually provided" in desc
    assert "`attachments` (optional)" in GENERATE_ARTIFACT_TOOL.description


class _RecordingProvider(LLMProvider):
    """Records the effort and output budget of each call. A call that names no effort sends the
    provider's own, as the real providers do."""

    name = "fake"
    accepts_large_output = True

    def __init__(self):
        self._reasoning_effort = "max"
        self.sent: list[tuple[str | None, int]] = []

    def _record(self, kw):
        self.sent.append((kw.get("reasoning_effort") or self.reasoning_effort, kw["max_tokens"]))

    async def complete(self, **kw):
        self._record(kw)
        return LLMResponse(content="ok", usage=Usage(output_tokens=1))

    async def stream(self, **kw):
        self._record(kw)
        yield StreamComplete(response=LLMResponse(content="ok", usage=Usage(output_tokens=1)))


async def test_generation_runs_every_model_call_at_the_configured_effort(tmp_path: Path, monkeypatch):
    slug = _make_artifact(tmp_path)
    provider = _RecordingProvider()
    llm = LLMClient(planning_provider=provider, planning_model="p", coding_provider=provider, coding_model="c")
    session = SimpleNamespace(_workspace=SimpleNamespace(artifacts_dir=tmp_path / "artifacts"), _llm=llm)

    async def fake_generate(**kw):
        await llm.plan(system="s", messages=[])
        async for _ in llm.plan_stream(system="s", messages=[]):
            pass
        async for _ in llm.code_stream(system="s", messages=[]):
            pass
        return "Backend verification failed after retry: stop here"

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    await _collect(session, {"slug": slug, "user_request": "build it", "agent_understanding": "an app"})

    # Max for every call. A streamed call's budget follows it; a non-streamed call keeps 8192.
    assert provider.sent == [("max", 8192), ("max", 65536), ("max", 65536)]


async def test_act_first_relays_the_brief_between_the_steps_and_reports_it(
    tmp_path, monkeypatch, make_llm_response,
):
    """The whole chain: the session flag reaches `GenState` in
    `engine.generate`, the brief goes out as a message marker between the
    step lines, and the result tells the agent the user saw it."""
    import json

    from anton.core.llm.provider import ToolCall
    from anton.core.tools.generate_artifact import orchestrator as fsm

    slug = _make_artifact(tmp_path, type="html-app")
    session = _session(tmp_path)
    session.__dict__.update(
        _llm=StreamingLLM(plan=AsyncMock(side_effect=[
            make_llm_response(tool_calls=[ToolCall(
                id="tc1", name="finish_gathering",
                input={"summary": "ok", "artifact_type": "html-app"},
            )]),
            make_llm_response("## Goal\nAn analog clock."),       # draft_brief
            make_llm_response("## Goal\nAn analog clock, full."),  # write_prd
        ])),
        question_count=0, elicitor=None, emit=AsyncMock(),
        act_first=True, tool_messages=True, emitter=object(),
        # Not the developer's own vault: its connections would leak into the kickoff.
        _data_vault=SimpleNamespace(list_connections=lambda: []),
    )
    monkeypatch.delenv("ANTON_DEBUG_ARTIFACT_GENERATE_TOOL", raising=False)

    async def skipped(state, *args, **kwargs):
        return None

    for name in ("_data_phase", "_write_tech_spec", "_gen_verify_frontend"):
        monkeypatch.setattr(fsm, name, skipped)

    markers, result = [], None
    async for item in handle_generate_artifact(session, {
        "slug": slug, "user_request": "a clock", "agent_understanding": "an analog clock",
        "user_language": "English",
    }):
        if isinstance(item, ToolProgress):
            markers.append((item.kind, item.text))
        else:
            result = item

    kinds = [k for k, _ in markers]
    assert kinds.count("message") == 1
    at = kinds.index("message")
    assert markers[at][1] == "## Goal\nAn analog clock."
    assert any(t.startswith("Preparing a short brief") for k, t in markers[:at] if k == "step")
    assert any(t.startswith("Writing down the") for k, t in markers[at + 1:] if k == "step")

    payload = json.loads(result.content)
    assert payload["status"] == "generated"
    assert payload["brief_shown"] is True
    # Already shown: the text is not resent into the agent's history.
    assert "brief_summary" not in payload
    assert "brief_shown" in payload["instruction"]


def _browser_check_failed():
    from anton.core.artifacts.render_check import CheckedView, RenderScreenshot
    from anton.core.tools.generate_artifact.state import BrowserCheckFailed

    return BrowserCheckFailed(
        page="dashboard.html",
        findings=("Loaded in a headless browser, the page logged a console error: "
                  "console.error: Route network failed: ROUTES is not defined (desktop)",),
        views=(CheckedView(pressed=None, viewports=["desktop"]), CheckedView(pressed="Routes", viewports=["desktop"])),
        screenshot=RenderScreenshot(media_type="image/png", data="iVBORw0KGgo="),
        error="Frontend verification failed after 2 attempt(s): x",
    )


async def test_a_rebuild_that_failed_its_browser_check_names_what_it_broke_and_removed(tmp_path: Path, monkeypatch):
    """A rebuild of an existing page comes back to repair knowing which views drew before and fail
    now, and which it dropped: the same comparison a built page's result carries."""
    from dataclasses import replace

    from anton.core.artifacts.render_check import Comparison
    from anton.core.tools import tool_handlers

    comparison = Comparison.model_validate({
        "previous": "complete", "previous_status": "passed", "removal_checked": "all",
        "compared": [{"pressed": "Routes", "viewports": ["desktop"]}],
        "broken_views": [{"pressed": "Routes", "reason": "new_problem", "viewports": ["desktop"], "findings": []}],
        "removed_views": [{"pressed": "Globe"}], "removed_controls": ["Globe"],
    })
    slug = _make_artifact(tmp_path, type="html-app")

    async def fake_generate(**kw):
        return replace(_browser_check_failed(), comparison=comparison)

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    session = _session(tmp_path)
    session._render_screenshots_refused = True
    _, result = await _collect_raw(session, {"slug": slug, "user_request": "rebuild it", "agent_understanding": "a page"})

    body = json.loads(result.content)
    assert body["changes_from_previous"] == {"broke": ["“Routes”"], "blank": [], "removed": ["“Globe”"], "shrank": []}
    assert body["instruction"].endswith(tool_handlers._CHANGES_INSTRUCTION)
    assert list(body)[-1] == "instruction"


@pytest.mark.parametrize("refused", [False, True])
async def test_a_page_that_failed_its_browser_check_comes_back_to_repair(tmp_path: Path, monkeypatch, refused):
    """The result names the page the last build wrote, its findings, the
    views its check measured and the repair budget, with the check's
    screenshot unless the model refused one. Nothing in it tells the agent
    not to repair; it tells the agent how."""
    slug = _make_artifact(tmp_path, type="html-app")

    async def fake_generate(**kw):
        return _browser_check_failed()

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    session = _session(tmp_path)
    session._render_screenshots_refused = refused
    _, result = await _collect_raw(session, {"slug": slug, "user_request": "build it", "agent_understanding": "a page"})

    assert (result.ok, result.reason) == (True, "")
    if refused:
        assert isinstance(result.content, str)
        text = result.content
    else:
        text = result.content[0]["text"]
        assert any(block.get("type") == "image" for block in result.content[1:])
    body = json.loads(text)
    assert body["status"] == "browser_check_failed"
    assert body["page"] == "dashboard.html"
    assert body["findings"] == list(_browser_check_failed().findings)
    assert body["views_measured"] == "at desktop size, as it opens and behind “Routes”"
    assert body["repair_cells"] == 2
    # A first build has nothing to compare with.
    assert "changes_from_previous" not in body
    instruction = body["instruction"]
    assert "do NOT build or repair" not in text and "ask how they want to proceed" not in text
    for step in ("repair it yourself now", "`read_text_file`", "scratchpad cell", "`runView`",
                 "Do NOT call generate_artifact again", "needs no `recall_skill`"):
        assert step in instruction, step


async def test_a_view_whose_script_did_not_parse_comes_back_to_repair_in_its_own_script(tmp_path: Path, monkeypatch):
    """One view's script holds a syntax error: the browser skips that script, so its `defineView`
    never runs and the helper fails the view by name. The repair is told what that message means,
    to fix that view's script, and to keep each view's code in a script of its own, since merging the
    scripts would let the next syntax error blank every view again."""
    from dataclasses import replace

    findings = (
        "Loaded in a headless browser, the page threw an uncaught exception: "
        "SyntaxError: missing ) after argument list (index.html line 41) (desktop, phone)",
        "Loaded in a headless browser, the page logged a console error: "
        "console.error: Globe failed: no script defined this view (desktop, phone)",
        "Loaded in a headless browser, a chart, map or canvas stayed blank: "
        "Globe failed to draw: the page marked div#globe data-view-failed (desktop, phone)",
    )
    slug = _make_artifact(tmp_path, type="html-app")

    async def fake_generate(**kw):
        return replace(_browser_check_failed(), findings=findings)

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    session = _session(tmp_path)
    session._render_screenshots_refused = True
    _, result = await _collect_raw(session, {"slug": slug, "user_request": "build it", "agent_understanding": "a page"})

    body = json.loads(result.content)
    assert body["status"] == "browser_check_failed"
    assert body["findings"] == list(findings)
    instruction = " ".join(body["instruction"].split())
    for step in ('When a view failed with "no script defined this view", its own script did not parse',
                 "fix that view's script",
                 'When views failed with "two scripts defined this view", two views share that title',
                 "keep each view's code in a `<script>` of its own",
                 "never merges views' scripts"):
        assert step in instruction, step


def test_no_rule_forbids_the_repair_and_each_names_the_one_path():
    """The rules that send a generator failure back to the user each make
    one exception: a page that came back as `browser_check_failed`."""
    from anton.core.llm import prompts
    from anton.core.tools.tool_defs import GENERATE_ARTIFACT_TOOL

    skill = Path(__file__).resolve().parents[1] / "anton/core/memory/builtin_skills/build-html-dashboard/SKILL.md"
    frontmatter = skill.read_text(encoding="utf-8").split("---")[1]
    rules = {
        "tool": GENERATE_ARTIFACT_TOOL.description,
        "artifacts": prompts.ARTIFACTS_PROMPT,
        "html": prompts.VISUALIZATIONS_HTML_OUTPUT_FORMAT_PROMPT,
        "markdown": prompts.VISUALIZATIONS_MARKDOWN_OUTPUT_FORMAT_PROMPT,
        "skill": frontmatter,
    }
    for name, text in rules.items():
        flat = " ".join(text.split())
        assert "browser_check_failed" in flat, name
    assert "The one exception is `browser_check_failed`: repair that page in place" in " ".join(rules["tool"].split())
