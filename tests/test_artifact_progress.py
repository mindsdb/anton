"""ENG-970: the generation FSM announces each step's START on a progress
channel, so a run that takes minutes is not silent.

Covers the label table, `GenState.step_started`, and the orchestrator's
call sites — including the AST check that keeps the two in sync.
"""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from anton.core.tools.generate_artifact import orchestrator
from anton.core.tools.generate_artifact.progress import STEP_LABELS, label_for
from anton.core.tools.generate_artifact.state import DataVerdict, GenState, VerifyResult


def _state(tmp_path, **kw):
    base = dict(
        session=AsyncMock(), artifact_type="html-app", artifact_path=tmp_path,
        slug="a", brief="Show current time", is_fullstack=False,
        progress=asyncio.Queue(),
    )
    base.update(kw)
    return GenState(**base)


def _drain(state) -> list[str]:
    lines = []
    while not state.progress.empty():
        lines.append(state.progress.get_nowait())
    return lines


# ── label table ──────────────────────────────────────────────────────────────

def test_label_is_plain_language_not_the_node_name():
    """Node names are graph vocabulary — `is_data_enough` must never be what
    a user reads."""
    assert label_for("is_data_enough", is_fullstack=True) == (
        "Working out whether that data is enough"
    )


def test_unknown_node_has_no_label():
    assert label_for("some_new_node") is None


def test_retry_is_called_out():
    """A repeated step explains why the run is longer than the step list
    suggests; without the suffix the same line just appears twice."""
    assert label_for("generate_backend", is_fullstack=True, attempt=0) == (
        "Writing the backend"
    )
    assert label_for("generate_backend", is_fullstack=True, attempt=1) == (
        "Writing the backend (retry)"
    )


def test_html_app_says_page_not_frontend():
    """An html-app has no backend for "frontend" to contrast with, so the
    word is jargon there — same node, different wording."""
    assert label_for("generate_frontend", is_fullstack=False) == "Writing the page"
    assert label_for("generate_frontend", is_fullstack=True) == "Writing the frontend"


def test_every_step_started_call_site_has_a_label():
    """The orchestrator and the label table must not drift: a node added to
    the FSM with a `step_started` call but no entry here would silently
    produce no progress line at all."""
    src = Path(orchestrator.__file__).read_text(encoding="utf-8")
    called: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step_started"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            called.add(node.args[0].value)
    assert called, "no step_started call sites found — did the FSM stop reporting?"
    assert called <= set(STEP_LABELS), (
        f"nodes with no label: {sorted(called - set(STEP_LABELS))}"
    )


# ── GenState.step_started ────────────────────────────────────────────────────

def test_step_started_is_a_noop_without_a_channel(tmp_path: Path):
    """`bench_generate.py`, the non-streaming path and most tests pass no
    queue — call sites must stay unconditional."""
    st = _state(tmp_path, progress=None)
    st.step_started("make_tech_spec")  # must not raise


def test_step_started_pushes_the_label(tmp_path: Path):
    st = _state(tmp_path)
    st.step_started("make_tech_spec")
    assert _drain(st) == ["Writing the technical specification"]


def test_step_started_skips_an_unlabelled_node(tmp_path: Path):
    st = _state(tmp_path)
    st.step_started("declare_datasources")
    assert _drain(st) == []


def test_step_started_does_not_touch_the_journal(tmp_path: Path):
    """Progress is a UI channel only: `record` remains the single source of
    the journal that later prompts read."""
    st = _state(tmp_path)
    st.step_started("make_tech_spec")
    assert st.trace == []
    assert st.journal() == ""


# ── orchestrator call sites ─────────────────────────────────────────────────

async def test_data_phase_and_tech_spec_report_before_they_run(tmp_path: Path):
    """Order matters: each line must arrive before its step's work, which is
    what `record` (fired on completion) cannot do."""
    st = _state(tmp_path)
    st.session._llm.generate_object = AsyncMock(
        return_value=DataVerdict(enough=True, reasoning="no data needed")
    )
    st.session._llm.plan = AsyncMock(return_value=type("R", (), {"content": "# Spec"})())

    assert await orchestrator._data_phase(st) is None
    assert _drain(st) == [
        "Looking at the data already gathered in this session",
        "Working out whether that data is enough",
    ]
    assert await orchestrator._write_tech_spec(st) is None
    assert _drain(st) == ["Writing the technical specification"]


async def test_backend_retry_reports_both_attempts(tmp_path: Path, monkeypatch):
    """The generate→verify loop reports every attempt, and the second one is
    marked as a retry."""
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.api_spec = "{}"
    verifies = {"n": 0}

    async def fake_loop(**kw):
        (tmp_path / "backend.py").write_text("x")
        return {"files_written": ["backend.py"], "rounds_used": 1, "summary": "s"}

    async def fake_verify(**kw):
        verifies["n"] += 1
        if verifies["n"] == 1:
            return VerifyResult(errors=["missing /api/health"]), []
        return VerifyResult(errors=[]), []

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_backend", fake_verify)
    monkeypatch.setattr(orchestrator, "_map_datasources", lambda session, keys: ([], []))

    assert await orchestrator._gen_verify_backend(st) is None
    assert _drain(st) == [
        "Writing the backend",
        "Verifying the backend",
        "Writing the backend (retry)",
        "Verifying the backend (retry)",
    ]
