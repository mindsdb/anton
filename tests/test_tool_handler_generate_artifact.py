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
