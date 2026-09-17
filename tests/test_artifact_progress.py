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
from anton.core.tools.generate_artifact.discovery import checkpoint as cp
from anton.core.tools.generate_artifact import engine
from anton.core.tools.generate_artifact import sub_tools as file_tools
from anton.core.tools.generate_artifact.progress import (
    PEEK_GROUPS,
    PEEK_LINE_MAX,
    PEEK_PREFIX,
    QUESTION_CLOSED,
    QUESTION_OPEN,
    STEP_LABELS,
    LivePeek,
    StepCounter,
    label_for,
    plan_steps,
)
from anton.core.llm.provider import StreamTextDelta
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
        "Writing the backend (attempt 2)"
    )
    assert label_for("generate_backend", is_fullstack=True, attempt=1, position=(4, 9)) == (
        "Writing the backend (step 4 of 9, attempt 2)"
    )
    assert label_for("generate_backend", is_fullstack=True, position=(4, 9)) == (
        "Writing the backend (step 4 of 9)"
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
    assert _drain(st) == ["Writing the technical specification (step 1 of 4)"]


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
    assert _drain(st) == ["Writing the technical specification (step 1 of 4)"]


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
        "Writing the backend (step 1 of 9)",
        "Verifying the backend (step 2 of 9)",
        "Writing the backend (step 1 of 9, attempt 2)",
        "Verifying the backend (step 2 of 9, attempt 2)",
    ]


# ── N of M ──────────────────────────────────────────────────────────────────

def test_the_plan_follows_the_artifact_type_and_the_entry_point():
    """The count starts at `write_prd`: before it the type is not settled,
    and `gathering` / the brief steps report without one. A resumed run has
    fewer steps ahead, and the total says so."""
    assert plan_steps(is_fullstack=False) == [
        "write_prd", "make_tech_spec", "generate_frontend", "verify_frontend",
    ]
    assert plan_steps(is_fullstack=True) == [
        "write_prd", "make_tech_spec", "make_api_spec", "generate_backend",
        "verify_backend", "generate_frontend", "verify_frontend", "run_app",
        "verify_fullstack",
    ]
    assert plan_steps(is_fullstack=True, entry=cp.ENTRY_SPEC)[0] == "make_tech_spec"
    assert plan_steps(is_fullstack=True, entry=cp.ENTRY_GENERATE)[0] == "generate_backend"
    assert plan_steps(is_fullstack=False, entry=cp.ENTRY_GENERATE) == [
        "generate_frontend", "verify_frontend",
    ]
    for entry in (cp.ENTRY_FULL, cp.ENTRY_CONFIRM, cp.ENTRY_NEW_ITERATION):
        assert plan_steps(is_fullstack=True, entry=entry)[0] == "write_prd"


def test_every_planned_step_has_a_label_and_a_call_site():
    """A plan naming a step the orchestrator never announces would leave the
    count short of its total; one the label table lacks would print nothing."""
    src = Path(orchestrator.__file__).read_text(encoding="utf-8")
    # `write_prd` is announced by the discovery phase through
    # `sub_tools.STEP_WRITE_PRD`; the rest by the orchestrator itself.
    from anton.core.tools.generate_artifact.discovery import sub_tools
    assert sub_tools.STEP_WRITE_PRD == "write_prd"
    for step in plan_steps(is_fullstack=True):
        assert step in STEP_LABELS, step
        if step != "write_prd":
            assert f'step_started("{step}"' in src, step


def test_the_counter_numbers_steps_in_start_order():
    """The two generation loops run in parallel, so the plan's order is not
    the order steps start in — and a count that jumped back would read as a
    step lost. A retry keeps its number; an unplanned step has none."""
    c = StepCounter(plan_steps(is_fullstack=True))
    assert c.position("write_prd") == (1, 9)
    assert c.position("make_tech_spec") == (2, 9)
    assert c.position("make_api_spec") == (3, 9)
    assert c.position("generate_frontend") == (4, 9)  # plan says backend first
    assert c.position("generate_backend") == (5, 9)
    assert c.position("verify_frontend") == (6, 9)
    assert c.position("generate_frontend") == (4, 9)  # retry: same number
    assert c.position("verify_backend") == (7, 9)
    assert c.position("run_app") == (8, 9)
    assert c.position("verify_fullstack") == (9, 9)
    assert c.position("gathering") is None
    assert c.position("declare_datasources") is None


def test_a_data_phase_step_grows_the_total_when_it_appears():
    """The data phase is decided at run time and is off the plan; the honest
    thing when it fires is a larger total, not a wrong one."""
    c = StepCounter(plan_steps(is_fullstack=False))
    assert c.position("write_prd") == (1, 4)
    assert c.position("define_required_data") == (2, 5)
    assert c.position("is_possible_to_fetch") == (3, 6)
    assert c.position("fetch_data_sample") == (4, 7)
    # The second iteration is a repeat of the same steps, not new ones.
    assert c.position("define_required_data") == (2, 7)
    assert c.position("make_tech_spec") == (5, 7)


def test_step_started_counts_a_full_html_app_run(tmp_path: Path):
    st = _state(tmp_path)
    for node in ("gathering", "draft_brief", "write_prd", "make_tech_spec",
                 "generate_frontend", "verify_frontend"):
        st.step_started(node)
    assert _drain(st) == [
        "Gathering what the artifact needs",
        "Preparing a short brief for you",
        "Writing down the agreed requirements (step 1 of 4)",
        "Writing the technical specification (step 2 of 4)",
        "Writing the page (step 3 of 4)",
        "Verifying the page (step 4 of 4)",
    ]


def test_a_resumed_run_counts_only_the_steps_ahead(tmp_path: Path):
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.entry = cp.ENTRY_GENERATE
    st.step_started("generate_backend")
    st.step_started("generate_frontend")
    assert _drain(st) == ["Writing the backend (step 1 of 6)", "Writing the frontend (step 2 of 6)"]


def test_the_data_phase_reports_its_iteration_as_the_attempt():
    """Its three steps repeat up to `DATA_LOOP_MAX` times with the same
    names; the iteration number is what tells the lines apart."""
    src = Path(orchestrator.__file__).read_text(encoding="utf-8")
    for step in ("define_required_data", "is_possible_to_fetch", "fetch_data_sample"):
        assert f'step_started("{step}", attempt=state.data_iterations)' in src, step
    assert "state.entry = entry" in src


# ── live peek ───────────────────────────────────────────────────────────────

class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _peeks(queue: asyncio.Queue) -> list[str]:
    out = []
    while not queue.empty():
        line = queue.get_nowait()
        assert line.startswith(PEEK_PREFIX)
        out.append(line[len(PEEK_PREFIX):])
    return out


def test_the_peek_prefix_cannot_be_mistaken_for_a_step_line():
    """Same queue as the step lines and the question sentinels; a control
    byte keeps the three apart, and the markers the tail hides are the ones
    the file protocol really uses."""
    assert PEEK_PREFIX.startswith("\x00")
    assert not QUESTION_OPEN.startswith(PEEK_PREFIX)
    assert not QUESTION_CLOSED.startswith(PEEK_PREFIX)
    for marker in (file_tools.FILE_BEGIN_MARKER, file_tools.FILE_END_MARKER):
        assert marker.startswith("<<<ANTON_FILE_")


def test_the_tail_keeps_the_last_lines_and_hides_protocol_and_blanks():
    q: asyncio.Queue = asyncio.Queue()
    peek = LivePeek(q, lines=3, clock=_Clock())
    peek.feed("frontend", file_tools.FILE_BEGIN_MARKER + "\n<!DOCTYPE html>\n<html>\n\n  <head>\n    <ti")
    assert _peeks(q) == ["<html>\n  <head>\n    <ti"]
    # The partial last line grows as the stream continues.
    peek.feed("frontend", "tle>Clock</title>\n" + "x" * 300 + "\n")
    peek._sent_at = float("-inf")  # past the throttle
    peek.feed("frontend", "")
    tail = _peeks(q)[-1].split("\n")
    assert tail[0] == "  <head>"
    assert tail[1] == "    <title>Clock</title>"
    assert len(tail[2]) == PEEK_LINE_MAX and tail[2].endswith("\u2026")


def test_the_tail_is_pushed_at_most_once_per_interval_and_only_when_changed():
    clock = _Clock()
    q: asyncio.Queue = asyncio.Queue()
    peek = LivePeek(q, interval=0.3, clock=clock)
    peek.feed("spec", "# Spec\n")
    peek.feed("spec", "## Insights\n")          # 0.0s later: held back
    clock.now += 0.1
    peek.feed("spec", "- one\n")                # still inside the interval
    assert _peeks(q) == ["# Spec"]
    clock.now += 0.3
    peek.feed("spec", "")                        # interval over: the held tail goes out
    assert _peeks(q) == ["# Spec\n## Insights\n- one"]
    clock.now += 1
    peek.feed("spec", "")                        # nothing new: nothing sent
    assert _peeks(q) == []


def test_two_live_groups_get_headings_and_a_finished_one_drops_out():
    """The fullstack generators stream in parallel; one tail on top of the
    other would flicker between two files. Alone, a group needs no heading."""
    clock = _Clock()
    q: asyncio.Queue = asyncio.Queue()
    peek = LivePeek(q, clock=clock)
    peek.feed("backend", "import os\n")
    clock.now += 1
    peek.feed("frontend", "<html>\n")
    assert _peeks(q) == ["import os", "backend:\n  import os\nfrontend:\n  <html>"]
    peek.clear("backend")                        # forced: no wait for the interval
    assert _peeks(q) == ["<html>"]
    peek.clear("frontend")
    assert _peeks(q) == [""]                     # empty tail clears the footer
    peek.clear("frontend")
    assert _peeks(q) == []                       # already clear: nothing to say


def test_peek_for_is_silent_without_a_channel_and_feeds_it_when_there_is_one(tmp_path: Path):
    assert _state(tmp_path, progress=None).peek_for("generate_frontend") is None
    st = _state(tmp_path)
    on_text = st.peek_for("generate_frontend")
    on_text("<body>\n<h1>Hi</h1>\n")
    st.peek_done("generate_frontend")
    assert _peeks(st.progress) == ["<body>\n<h1>Hi</h1>", ""]
    assert set(PEEK_GROUPS) >= {"generate_frontend", "generate_backend", "make_tech_spec", "make_api_spec"}


async def test_the_stream_drain_hands_text_deltas_to_on_text():
    from anton.core.llm.provider import StreamComplete

    async def events():
        yield StreamTextDelta(text="<<<ANTON_FILE_BEGIN>>>\n")
        yield StreamTextDelta(text="<html>")
        yield StreamTextDelta(text="")
        yield StreamComplete(response=type("R", (), {"content": "done"})())

    seen: list[str] = []
    response = await engine._drain_stream(events(), seen.append)
    assert response.content == "done"
    assert seen == ["<<<ANTON_FILE_BEGIN>>>\n", "<html>"]
    # Without a taker the drain is what it always was.
    assert (await engine._drain_stream(events())).content == "done"


async def test_the_generation_loop_streams_its_tail_and_clears_it_after(tmp_path: Path, monkeypatch):
    st = _state(tmp_path)
    st.gathering_complete = True

    async def fake_loop(**kw):
        kw["on_text"]("<<<ANTON_FILE_BEGIN>>>\n<html>\n<body>\n")
        (tmp_path / "index.html").write_text("<html><body></body></html>")
        return {"files_written": ["index.html"], "rounds_used": 1, "summary": "s"}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend", lambda *a, **k: VerifyResult(errors=[]))
    monkeypatch.setattr(orchestrator, "_frontend_entry", lambda state, written: tmp_path / "index.html")
    await orchestrator._gen_verify_frontend(st)
    lines = _drain(st)
    assert lines[0] == "Writing the page (step 1 of 4)"
    assert PEEK_PREFIX + "<html>\n<body>" in lines
    assert lines.index(PEEK_PREFIX + "") > lines.index(PEEK_PREFIX + "<html>\n<body>")



def test_settling_a_new_type_rebuilds_the_plan_but_keeps_the_numbers(tmp_path):
    """The counter is created at the first `gathering` line, before the type is
    known. When `write_prd` settles a different type the total has to follow
    (4 → 9), while a number already shown is not renumbered."""
    import asyncio

    from anton.core.tools.generate_artifact.state import GenState

    q: asyncio.Queue = asyncio.Queue()
    st = GenState(
        session=object(), artifact_type="html-app", artifact_path=tmp_path,
        slug="a", progress=q,
    )
    st.step_started("gathering")
    assert q.get_nowait() == "Gathering what the artifact needs"

    st.settle_artifact_type("fullstack-stateless-app")
    assert st.is_fullstack is True
    st.step_started("write_prd")
    assert q.get_nowait() == "Writing down the agreed requirements (step 1 of 9)"

    # Settling the same type again is a no-op; a blank is ignored.
    st.settle_artifact_type("fullstack-stateless-app")
    st.settle_artifact_type("")
    assert st.artifact_type == "fullstack-stateless-app"
    st.step_started("make_tech_spec")
    assert q.get_nowait() == "Writing the technical specification (step 2 of 9)"
