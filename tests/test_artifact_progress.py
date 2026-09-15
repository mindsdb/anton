"""ENG-970: the generation FSM announces each step's START on a progress
channel, so a run that takes minutes is not silent.

Covers the label table, `GenState.step_started`, and the orchestrator's
call sites — including the AST check that keeps the two in sync.
"""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from anton.core.llm.provider import StreamComplete
from anton.core.tools.generate_artifact import orchestrator
from anton.core.tools.generate_artifact.progress import STEP_LABELS, label_for
from anton.core.tools.generate_artifact.state import GenState, VerifyResult


async def _one_event_stream(response):
    yield StreamComplete(response=response)


def _stream_mock(response):
    """`plan_stream` fake: every call returns a fresh one-event stream of the
    same response (mirrors `AsyncMock(return_value=...)`)."""
    return Mock(side_effect=lambda **kw: _one_event_stream(response))


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
    assert label_for("make_tech_spec", is_fullstack=True) == (
        "Writing the technical specification"
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
    """The pipeline and the label table must not drift: a step added with a
    `step_started` call but no entry here would silently produce no progress
    line at all.

    Walks the discovery phases as well as the FSM. They report too now, and a
    walk that stopped at the orchestrator would leave the newer half of the
    pipeline unchecked — which is exactly where new steps are being added.
    """
    from anton.core.tools.generate_artifact.discovery import (
        brief as discovery_brief,
        engine as discovery_engine,
        orchestrator as discovery_orchestrator,
        prd as discovery_prd,
    )

    modules = [
        orchestrator,
        discovery_engine,
        discovery_brief,
        discovery_prd,
        discovery_orchestrator,
    ]
    src = "\n".join(
        Path(m.__file__).read_text(encoding="utf-8") for m in modules
    )
    from anton.core.tools.generate_artifact.discovery import sub_tools

    def _node_name(arg):
        """The step name a `step_started` argument denotes, or None.

        Resolves `sub_tools.STEP_*` as well as bare strings: the discovery
        phases pass the constants, and a walk that only understood literals
        would wave them through unchecked — which is precisely the drift this
        test exists to catch.
        """
        if isinstance(arg, ast.Constant):
            return arg.value
        if isinstance(arg, ast.Attribute) and arg.attr.startswith("STEP_"):
            return getattr(sub_tools, arg.attr, None)
        return None

    called: set[str] = set()
    unresolved: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step_started"
            and node.args
        ):
            name = _node_name(node.args[0])
            if name is None:
                unresolved.append(ast.dump(node.args[0]))
            else:
                called.add(name)
    assert not unresolved, (
        "step_started called with something this walk cannot resolve, so its "
        f"label goes unchecked: {unresolved}"
    )
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

async def test_the_tech_spec_reports_before_it_runs(tmp_path: Path):
    """Order matters: the line must arrive before the step's work, which is
    what `record` (fired on completion) cannot do.

    The data phase no longer reports anything on the normal path — it does
    not run there. `inspect_scratchpads` and `is_data_enough` were the two
    steps that always announced themselves, and both are gone.
    """
    st = _state(tmp_path)
    st.gathering_complete = True
    st.session._llm.plan_stream = _stream_mock(type("R", (), {"content": "# Spec"})())

    assert await orchestrator._data_phase(st) is None
    assert _drain(st) == []
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
