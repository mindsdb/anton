"""`prd.md` and `discovery.json`: the record and the resume point.

The PRD stays the human-readable statement of what the user agreed to, and
is still what the generation nodes are told to treat as authoritative. What
changed with the merge is that it is no longer the CHANNEL: the pipeline
hands its own state forward in memory on the hot path, and reads
`discovery.json` back on a cold start. The `context` parameter — and the
three-surface markdown contract it needed — is gone with it.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

from anton.core.artifacts.internal_files import PRD_FILENAME
from anton.core.tools.generate_artifact import engine, orchestrator, prompts
from anton.core.tools.generate_artifact.discovery import checkpoint as cp
from anton.core.tools.generate_artifact.state import GenState


def _state(tmp_path, **kw):
    base = dict(
        session=AsyncMock(), artifact_type="html-app", artifact_path=tmp_path,
        slug="a", brief="## User request\nShow orders", is_fullstack=False,
    )
    base.update(kw)
    return GenState(**base)


def _write_prd(tmp_path: Path, body: str) -> None:
    (tmp_path / PRD_FILENAME).write_text(body, encoding="utf-8")


# ── loading ─────────────────────────────────────────────────────────────────

def test_prd_is_read_from_the_artifact_folder(tmp_path: Path):
    _write_prd(tmp_path, "## Goal\nShow the orders table.\n")
    st = _state(tmp_path)
    engine._load_prd(st)
    assert "Show the orders table." in st.prd
    assert [(s.node, s.outcome) for s in st.trace] == [("read_prd", "done")]


def test_a_missing_prd_is_not_a_failure(tmp_path: Path):
    """An agent may skip the PRD step and artifacts predating ENG-969 have no
    prd.md — generation then runs on `context` alone."""
    st = _state(tmp_path)
    engine._load_prd(st)
    assert st.prd == ""
    assert [(s.node, s.outcome) for s in st.trace] == [("read_prd", "skipped")]


def test_an_empty_prd_is_treated_as_absent(tmp_path: Path):
    _write_prd(tmp_path, "   \n\n")
    st = _state(tmp_path)
    engine._load_prd(st)
    assert st.prd == ""
    assert st.trace[0].outcome == "skipped"
    assert "empty" in st.trace[0].detail


def test_an_unreadable_prd_is_recorded_and_skipped(tmp_path: Path, monkeypatch):
    """A broken PRD must not cost a generation that `context` alone can still
    complete — but it must be visible in the trace, not silent."""
    _write_prd(tmp_path, "## Goal\nx")

    def boom(*a, **kw):
        raise OSError("disk gone")

    monkeypatch.setattr(Path, "read_text", boom)
    st = _state(tmp_path)
    engine._load_prd(st)
    assert st.prd == ""
    assert st.trace[0].outcome == "error"
    assert "disk gone" in st.trace[0].detail


def test_which_mode_ran_is_recorded_for_every_outcome(tmp_path: Path):
    """The record is the only way to tell afterwards which requirements a
    given artifact was built from."""
    st_without = _state(tmp_path)
    engine._load_prd(st_without)
    _write_prd(tmp_path, "## Goal\ny")
    st_with = _state(tmp_path)
    engine._load_prd(st_with)
    assert st_without.trace[0].detail != st_with.trace[0].detail


async def test_generate_loads_the_prd_before_running_the_fsm(tmp_path: Path, monkeypatch):
    """Wired into the public entry point, not just callable: the FSM must
    never see a state whose PRD has not been loaded yet."""
    _write_prd(tmp_path, "## Goal\nShow the orders table.")
    seen = {}

    async def fake_run(state, *, entry):
        seen["prd"] = state.prd
        seen["entry"] = entry
        return {"files_written": [], "internal_files": [], "summary": "", "trace": []}

    # A checkpoint has to exist for this to be a resume — without one the run
    # starts at gathering and writes its own PRD rather than reading this one.
    cp.save(tmp_path, cp.DiscoveryCheckpoint(
        request_fingerprint=cp.request_fingerprint("show orders"),
        pipeline_stage=cp.STAGE_PRD_WRITTEN,
    ))
    monkeypatch.setattr(orchestrator, "run", fake_run)
    await engine.generate(
        session=AsyncMock(), artifact_type="html-app", artifact_path=tmp_path,
        slug="a", user_request="show orders", agent_understanding="orders table",
    )
    assert seen["entry"] == cp.ENTRY_SPEC
    assert "Show the orders table." in seen["prd"]


async def test_a_resumed_generation_reads_the_api_contract_back(tmp_path: Path, monkeypatch):
    """`resume_generate` skips `_make_api_spec`, and nothing else set
    `state.api_spec`: both generators got `{}` as the contract, `_probe_app`
    probed nothing and the contract check saw no contract (I-48)."""
    from anton.core.artifacts.internal_files import API_SPEC_FILENAME

    _write_prd(tmp_path, "## Goal\nA clock.")
    contract = '{"paths": {"/api/time": {"get": {"responses": {"200": {}}}}}}'
    (tmp_path / API_SPEC_FILENAME).write_text(contract)
    seen = {}

    async def fake_run(state, *, entry):
        seen["api_spec"] = state.api_spec
        seen["internal"] = list(state.internal_files)
        seen["entry"] = entry
        return {"files_written": [], "internal_files": [], "summary": "", "trace": []}

    cp.save(tmp_path, cp.DiscoveryCheckpoint(
        request_fingerprint=cp.request_fingerprint("a clock"),
        pipeline_stage=cp.STAGE_SPEC_WRITTEN,
        artifact_type="fullstack-stateless-app",
    ))
    monkeypatch.setattr(orchestrator, "run", fake_run)
    await engine.generate(
        session=AsyncMock(), artifact_type="fullstack-stateless-app", artifact_path=tmp_path,
        slug="a", user_request="a clock", agent_understanding="clock",
    )
    assert seen["entry"] == cp.ENTRY_GENERATE
    assert seen["api_spec"] == contract
    assert API_SPEC_FILENAME in seen["internal"]


async def test_a_resumed_spec_phase_does_not_read_a_contract_it_will_rewrite(tmp_path: Path, monkeypatch):
    """At `resume_spec` the contract is regenerated; a file on disk (a stale
    one `_invalidate_specs` has not reached yet) must not pre-seed the state."""
    from anton.core.artifacts.internal_files import API_SPEC_FILENAME

    _write_prd(tmp_path, "## Goal\nA clock.")
    (tmp_path / API_SPEC_FILENAME).write_text('{"paths": {"/api/old": {"get": {}}}}')
    seen = {}

    async def fake_run(state, *, entry):
        seen["api_spec"] = state.api_spec
        seen["entry"] = entry
        return {"files_written": [], "internal_files": [], "summary": "", "trace": []}

    cp.save(tmp_path, cp.DiscoveryCheckpoint(
        request_fingerprint=cp.request_fingerprint("a clock"),
        pipeline_stage=cp.STAGE_PRD_WRITTEN,
        artifact_type="fullstack-stateless-app",
    ))
    monkeypatch.setattr(orchestrator, "run", fake_run)
    await engine.generate(
        session=AsyncMock(), artifact_type="fullstack-stateless-app", artifact_path=tmp_path,
        slug="a", user_request="a clock", agent_understanding="clock",
    )
    assert seen["entry"] == cp.ENTRY_SPEC
    assert seen["api_spec"] is None


# ── reaching the nodes ──────────────────────────────────────────────────────

def test_prd_section_states_its_own_standing(tmp_path: Path):
    """Nodes see the PRD next to `spec.md` and the data record; the header —
    not the position — is what tells them which wins."""
    st = _state(tmp_path, prd="## Goal\nShow orders")
    section = prompts.prd_section(st)
    assert "authoritative" in section
    assert "Show orders" in section


def test_prd_section_is_delimited_at_both_ends(tmp_path: Path):
    """The PRD's own `##` headings land as siblings of the wrapper header, so
    without a closing marker its last section runs into whatever the context
    puts next."""
    st = _state(tmp_path, prd="## Goal\nShow orders\n\n## UI/UX requirements\ndark")
    section = prompts.prd_section(st)
    assert section.startswith(prompts.PRD_SECTION_HEADER)
    assert section.endswith(prompts.PRD_SECTION_FOOTER)


def test_prd_section_is_empty_without_a_prd(tmp_path: Path):
    assert prompts.prd_section(_state(tmp_path)) == ""


def test_data_nodes_see_the_prd(tmp_path: Path):
    """`is_data_enough` and friends decide what data is needed — they cannot do
    that from a brief that no longer carries the requirements."""
    st = _state(tmp_path, prd="## Data model\nthe orders table")
    assert "the orders table" in prompts._brief_and_notes(st)


def test_generation_nodes_see_the_prd(tmp_path: Path):
    """`_spec_context` feeds the tech spec, the API spec and both generators."""
    st = _state(tmp_path, prd="## UI/UX requirements\ndark background")
    assert "dark background" in orchestrator._spec_context(st)


def test_generation_nodes_do_not_see_the_brief_next_to_a_prd(tmp_path: Path):
    """Since phase B the brief is the confirmation proposal shown to the user
    ("here is what I suggest — continue or say what to change"), with
    questions the accepted PRD has already settled. Ninth live run
    2026-09-16: 1 KB of every generation round and a second voice next to
    the authoritative document. The PRD supersedes it."""
    st = _state(tmp_path, prd="## Goal\nShow orders")
    st.brief = "Вот что предлагаю сделать.\n\n**Вопросы**\n- Нужен ли таймер?"
    context = orchestrator._spec_context(st)
    assert "## Brief" not in context
    assert "Нужен ли таймер" not in context
    assert "Show orders" in context


def test_generation_nodes_fall_back_to_the_brief_without_a_prd(tmp_path: Path):
    st = _state(tmp_path)
    st.brief = "An orders dashboard."
    context = orchestrator._spec_context(st)
    assert context.startswith("## Brief\nAn orders dashboard.")


def test_kickoffs_do_not_prepend_their_own_brief_header():
    """The context renders its own section headers; a kickoff that added
    `## Brief` on top would label the PRD as the brief."""
    ctx = "## Product requirements\n...\n--- end of prd.md ---"
    for kickoff in (
        prompts.build_user_kickoff(ctx),
        prompts.build_frontend_kickoff(ctx, "{}"),
        prompts.build_backend_kickoff(ctx, "{}"),
    ):
        assert not kickoff.startswith("## Brief")
        assert kickoff.startswith("## Product requirements")


# ── the resume point ────────────────────────────────────────────────────────

def test_a_cold_start_restores_the_notes_the_hot_path_had(tmp_path: Path):
    """`data_notes` and `web_notes` live only in memory during a run, and the
    pad-inspection step that used to rebuild them is gone. If they did not
    survive in `discovery.json`, a resumed run would build from a PRD that no
    longer restates the data-access code either."""
    from anton.core.tools.generate_artifact.engine import _restore

    stored = cp.DiscoveryCheckpoint(
        request_fingerprint=cp.request_fingerprint("show orders"),
        call_fingerprint=cp.call_fingerprint("an orders dashboard", "", ""),
        pipeline_stage=cp.STAGE_PRD_WRITTEN,
        artifact_type="html-app",
        gathering_complete=True,
        declared_sources=["orders table"],
        unverified_sources=[],
        brief_markdown="## Goal\nAn orders dashboard.",
        data_notes="Scratchpad `o`:\n```python\nrows = q()\n```",
        web_notes="### Sources read from the web\n- web_fetch: https://x/y",
    )
    st = _state(tmp_path, user_request="show orders",
                agent_understanding="an orders dashboard")
    _restore(st, stored)

    context = orchestrator._spec_context(st)
    assert "rows = q()" in context
    assert "https://x/y" in context
    assert st.declared_sources == ["orders table"]
    assert st.gathering_complete is True
    assert st.call_changed is False


def test_a_changed_understanding_marks_the_call_as_a_correction(tmp_path: Path):
    """`call_changed` decides whether the brief is redrawn. It is an
    optimization, never a confirmation signal — being wrong costs one cheap
    call in either direction."""
    from anton.core.tools.generate_artifact.engine import _restore

    stored = cp.DiscoveryCheckpoint(
        request_fingerprint=cp.request_fingerprint("show orders"),
        call_fingerprint=cp.call_fingerprint("an orders dashboard", "", ""),
        pipeline_stage=cp.STAGE_AWAITING_CONFIRMATION,
    )
    st = _state(tmp_path, user_request="show orders",
                agent_understanding="an orders dashboard, weekly not daily")
    _restore(st, stored)
    assert st.call_changed is True


def test_whitespace_noise_in_the_request_is_not_a_different_request(tmp_path: Path):
    """The outer model re-types these fields every call. A stray newline must
    not cost a full re-gather, questions to the user included."""
    stored = cp.DiscoveryCheckpoint(
        request_fingerprint=cp.request_fingerprint("show   orders"),
        pipeline_stage=cp.STAGE_PRD_WRITTEN,
    )
    entry = cp.decide_entry(
        stored, request_fp=cp.request_fingerprint("  show orders  ")
    )
    assert entry == cp.ENTRY_SPEC


# ── the tool's own contract ─────────────────────────────────────────────────

# ── the two tools agree on the file ─────────────────────────────────────────

async def test_what_generate_prd_writes_is_what_generate_artifact_reads(tmp_path: Path):
    """The whole handoff in one test. Both tools go through `PRD_FILENAME`, so
    they cannot drift apart into a silent fallback — a reader that finds
    nothing would build the artifact from a brief the user never confirmed.
    """
    from types import SimpleNamespace

    from anton.core.llm.provider import LLMResponse, Usage
    from anton.core.tools.generate_artifact.discovery import orchestrator as prd_orchestrator
    from anton.core.tools.generate_artifact.discovery.state import PrdState

    artifact_dir = tmp_path / "artifacts" / "s"
    artifact_dir.mkdir(parents=True)
    body = "## Goal\nShow the orders table.\n\n## Data model\nthe `orders` scratchpad, cell 1"
    prd_state = PrdState(
        session=SimpleNamespace(
            _llm=SimpleNamespace(plan=AsyncMock(return_value=LLMResponse(
                content=body, tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1),
            ))),
            question_count=0, elicitor=None, emit=AsyncMock(),
            _workspace=SimpleNamespace(artifacts_dir=artifact_dir.parent),
        ),
        slug="s", artifact_path=artifact_dir, artifact_type="html-app",
        user_request="show orders", agent_understanding="a table",
        known_data="", user_preferences="",
    )
    prd_state.final_artifact_type = "html-app"
    await prd_orchestrator.write_prd(prd_state)

    gen_state = _state(artifact_dir)
    engine._load_prd(gen_state)
    assert gen_state.prd == body.strip()


async def test_write_prd_settles_the_type_the_gathering_chose(tmp_path: Path, monkeypatch):
    """The type `finish_gathering` picked has to reach the object the rest of
    the call reads, not only metadata and the checkpoint: with `is_fullstack`
    still False the run skipped the API spec and the backend of the fullstack
    app it had just agreed to build (I-40)."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from anton.core.llm.provider import LLMResponse, Usage
    from anton.core.tools import tool_handlers
    from anton.core.tools.generate_artifact.discovery import orchestrator as prd_orchestrator
    from anton.core.tools.generate_artifact.discovery.state import PrdState

    store = Mock()
    monkeypatch.setattr(tool_handlers, "resolve_artifact_store", lambda session: store)
    prd_state = PrdState(
        session=SimpleNamespace(
            _llm=SimpleNamespace(plan=AsyncMock(return_value=LLMResponse(
                content="## Goal\nA room-based game.", tool_calls=[],
                usage=Usage(input_tokens=1, output_tokens=1),
            ))),
            question_count=0, elicitor=None, emit=AsyncMock(),
        ),
        slug="s", artifact_path=tmp_path, artifact_type="html-app",
        user_request="a game", agent_understanding="a", known_data="", user_preferences="",
    )
    assert prd_state.is_fullstack is False
    prd_state.final_artifact_type = "fullstack-stateful-app"

    await prd_orchestrator.write_prd(prd_state)

    assert prd_state.artifact_type == "fullstack-stateful-app"
    assert prd_state.is_fullstack is True
    store.update.assert_called_once_with("s", type="fullstack-stateful-app")


async def test_a_restored_run_does_not_re_enter_the_data_loop(tmp_path: Path):
    """What the pad-inspection step used to buy, now bought by the
    checkpoint: a resumed run whose sources were verified last time does not
    pay for the data loop again."""
    from anton.core.tools.generate_artifact.engine import _restore

    stored = cp.DiscoveryCheckpoint(
        request_fingerprint=cp.request_fingerprint("show orders"),
        pipeline_stage=cp.STAGE_PRD_WRITTEN,
        gathering_complete=True,
        declared_sources=["orders table"],
        unverified_sources=[],
        data_notes="Scratchpad `o`:\n```python\nprint(df.head())\n```",
    )
    st = _state(tmp_path, user_request="show orders", agent_understanding="x")
    _restore(st, stored)

    assert orchestrator._needs_data_loop(st) is False
    assert await orchestrator._data_phase(st) is None
    assert st.data_iterations == 0
    assert "print(df.head())" in st.data_notes
