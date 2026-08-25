"""I-01: `generate_artifact` takes its requirements from the `prd.md` that
`generate_prd` left in the artifact folder.

Before this, the file was only ever written — nothing read it, so the artifact
was built from whatever the calling agent chose to put in `context`, not from
the document the user actually accepted.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

from anton.core.artifacts.internal_files import PRD_FILENAME
from anton.core.tools.generate_artifact import engine, orchestrator, prompts
from anton.core.tools.generate_artifact.state import DataVerdict, GenState


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

    async def fake_run(state):
        seen["prd"] = state.prd
        return {"files_written": [], "internal_files": [], "summary": "", "trace": []}

    monkeypatch.setattr(orchestrator, "run", fake_run)
    await engine.generate(
        session=AsyncMock(), artifact_type="html-app", artifact_path=tmp_path,
        context="## User request\nx", slug="a",
    )
    assert "Show the orders table." in seen["prd"]


# ── reaching the nodes ──────────────────────────────────────────────────────

def test_prd_section_states_its_own_standing(tmp_path: Path):
    """Nodes see `## Brief` (written by the calling agent) and the PRD side by
    side; the header — not the position — is what tells them which wins."""
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


# ── scratchpad reuse ────────────────────────────────────────────────────────

class _FakeCell:
    def __init__(self, code, stdout=""):
        self.code = code
        self.stdout = stdout
        self.error = ""


class _FakePad:
    def __init__(self, cells):
        self.cells = cells


def _session_with_pads(pads: dict):
    session = AsyncMock()
    session._scratchpads.pads = pads
    return session


def test_a_pad_named_only_in_the_prd_is_still_inspected(tmp_path: Path):
    """`generate_prd` must cite the scratchpad and cell behind every data
    source it describes, so on the normal path the PRD — not the brief — is
    where a pad name appears. Matching the brief alone would re-fetch data the
    PRD already points at."""
    session = _session_with_pads(
        {"orders": _FakePad([_FakeCell("df = read_orders()", "1000 rows")])}
    )
    st = _state(
        tmp_path, session=session,
        brief="## User request\nShow me a dashboard",
        prd="## Data model\nOrders come from the `orders` scratchpad, cell 1.",
    )
    orchestrator._inspect_named_scratchpads(st)
    assert "read_orders()" in st.data_notes
    assert st.trace[0].outcome == "done"


def test_pads_are_still_matched_from_the_brief_alone(tmp_path: Path):
    """The pre-PRD path keeps working — a brief that names a pad with no PRD
    present must still be inspected."""
    session = _session_with_pads({"orders": _FakePad([_FakeCell("q = 1", "ok")])})
    st = _state(tmp_path, session=session, brief="data is in the orders scratchpad")
    orchestrator._inspect_named_scratchpads(st)
    assert "q = 1" in st.data_notes


# ── the tool's own contract ─────────────────────────────────────────────────

def _contract_surfaces() -> dict[str, str]:
    """The three places the `context` contract is stated for the model. All
    three are read by the LLM — the description and the schema as part of the
    tools array, `prompt` as a block spliced into the system prompt by
    `prompt_builder._build_tool_prompts_section` — so a change applied to one
    and forgotten in another is a contradiction shipped to production."""
    from anton.core.tools.tool_defs import GENERATE_ARTIFACT_TOOL as tool

    return {
        "description": tool.description,
        "prompt": tool.prompt or "",
        "schema": tool.input_schema["properties"]["context"]["description"],
    }


def test_no_surface_still_asks_for_a_requirements_section():
    """The FRS section is gone from `context`: requirements now live in the
    PRD, and asking for both invites two sources of truth that disagree."""
    for name, text in _contract_surfaces().items():
        assert "Functional Requirements Specification" not in text, name


def test_every_surface_points_at_the_prd():
    for name, text in _contract_surfaces().items():
        assert "PRD" in text or "prd.md" in text, name


def test_the_workflow_names_generate_prd_before_generate_artifact():
    """Without this the model has no way to know a PRD is supposed to exist —
    the file would simply never be there and every run would take the
    context-only fallback."""
    prompt = _contract_surfaces()["prompt"]
    assert "generate_prd" in prompt
    assert prompt.index("generate_prd") < prompt.index("generate_artifact(slug=")


def test_the_data_section_is_limited_to_what_already_exists():
    """`## Data` must describe fetched cells, not expected sources: anything
    the agent merely expects is a guess competing with the PRD, and a sample
    it has not observed is fabricated data."""
    for name, text in _contract_surfaces().items():
        if "## Data" not in text:
            continue
        assert "already" in text, name
        assert "scratchpad" in text, name


# ── the two tools agree on the file ─────────────────────────────────────────

async def test_what_generate_prd_writes_is_what_generate_artifact_reads(tmp_path: Path):
    """The whole handoff in one test. Both tools go through `PRD_FILENAME`, so
    they cannot drift apart into a silent fallback — a reader that finds
    nothing would build the artifact from a brief the user never confirmed.
    """
    from types import SimpleNamespace

    from anton.core.llm.provider import LLMResponse, Usage
    from anton.core.tools.generate_prd import orchestrator as prd_orchestrator
    from anton.core.tools.generate_prd.state import PrdState

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


async def test_prd_data_reference_lets_the_data_phase_short_circuit(tmp_path: Path):
    """The point of the two changes together: cells the PRD points at land in
    `data_notes`, so `is_data_enough` can answer from what is already there
    instead of starting its own fetch loop."""
    session = _session_with_pads(
        {"orders": _FakePad([_FakeCell("print(df.head())", "id,total\n1,99")])}
    )
    st = _state(
        tmp_path, session=session,
        brief="## User request\nShow a dashboard",
        prd="## Data model\nthe `orders` scratchpad, cell 1",
    )
    st.session._llm.generate_object = AsyncMock(
        return_value=DataVerdict(enough=True, reasoning="cells already show the data")
    )
    assert await orchestrator._data_phase(st) is None
    assert st.data_iterations == 0
    assert "print(df.head())" in st.data_notes
