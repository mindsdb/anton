"""prd.py (phase C): write_prd and the instruction that drives it.

Split out of the phase-B tests when the orchestrator was split: the file that
writes the agreed requirements has its own module now, and its tests follow
it."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.artifacts import ArtifactStore
from anton.core.interaction.elicit import AskAnswer
from anton.core.llm.provider import LLMResponse, Usage
from anton.core.tools.generate_artifact.discovery import prd
from anton.core.tools.generate_artifact.discovery.state import PrdState


def _response(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1))


def _state(artifact_path: Path, **over) -> PrdState:
    # `_workspace.artifacts_dir` defaults to `artifact_path`'s parent so
    # `resolve_artifact_store(session)` resolves to the same root a test's own
    # `ArtifactStore(tmp_path / "artifacts")` uses — needed by the
    # write_prd type-update tests below, harmless for the tests that never
    # reach that branch.
    base = dict(
        session=SimpleNamespace(
            _llm=SimpleNamespace(plan=AsyncMock()),
            question_count=0,
            elicitor=None,
            emit=AsyncMock(),
            _workspace=SimpleNamespace(artifacts_dir=artifact_path.parent),
        ),
        slug="s",
        artifact_path=artifact_path,
        artifact_type="html-app",
        user_request="build a clock",
        agent_understanding="an analog clock",
        known_data="",
        user_preferences="",
    )
    base.update(over)
    state = PrdState(**base)
    state.final_artifact_type = "html-app"
    state.gathering_notes = "no external data needed"
    return state


async def test_write_prd_saves_the_file_and_returns_its_text(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "s"
    artifact_dir.mkdir(parents=True)
    state = _state(artifact_dir)
    state.session._llm.plan = AsyncMock(return_value=_response("## Goal\nFull PRD text.\n"))
    text = await prd.write_prd(state)
    assert text == "## Goal\nFull PRD text."
    assert (artifact_dir / "prd.md").read_text(encoding="utf-8") == text


async def test_write_prd_restarts_the_spinner_before_the_llm_call(tmp_path):
    from anton.core.llm.provider import StreamTaskProgress

    artifact_dir = tmp_path / "artifacts" / "s"
    artifact_dir.mkdir(parents=True)
    state = _state(artifact_dir)
    state.session._llm.plan = AsyncMock(return_value=_response("## Goal\nFull PRD text.\n"))
    await prd.write_prd(state)
    state.session.emit.assert_awaited_with(StreamTaskProgress(phase="reasoning_start", message="Thinking..."))


async def test_write_prd_logs_an_llm_call_and_a_done_node(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "s"
    artifact_dir.mkdir(parents=True)
    trace = MagicMock()
    state = _state(artifact_dir, trace_log=trace)
    state.session._llm.plan = AsyncMock(return_value=_response("## Goal\nFull PRD text.\n"))
    await prd.write_prd(state)
    assert trace.llm_call.call_args.kwargs["node"] == "write_prd"
    trace.node.assert_called_once_with("write_prd", "done", detail=str(artifact_dir / "prd.md"))


async def test_write_prd_logs_a_fail_node_before_raising(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "s"
    artifact_dir.mkdir(parents=True)
    trace = MagicMock()
    state = _state(artifact_dir, trace_log=trace)
    state.session._llm.plan = AsyncMock(return_value=_response(""))
    with pytest.raises(RuntimeError):
        await prd.write_prd(state)
    trace.node.assert_called_once_with("write_prd", "fail", detail="model replied with no text")


async def test_write_prd_raises_on_an_empty_reply_instead_of_writing_an_empty_file(tmp_path):
    """Same failure shape as draft_brief's guard: an empty reply must never
    become an empty prd.md on disk reported back as `prd_written`."""
    artifact_dir = tmp_path / "artifacts" / "s"
    artifact_dir.mkdir(parents=True)
    state = _state(artifact_dir)
    state.session._llm.plan = AsyncMock(return_value=_response(""))
    with pytest.raises(RuntimeError, match="no text"):
        await prd.write_prd(state)
    assert not (artifact_dir / "prd.md").exists()


async def test_write_prd_updates_the_artifact_type_when_it_changed(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.create(name="Clock", description="d", type="html-app")
    artifact_dir = store.folder_for(artifact.slug)
    state = _state(artifact_dir, slug=artifact.slug)
    state.final_artifact_type = "fullstack-stateless-app"
    state.session._llm.plan = AsyncMock(return_value=_response("## Goal\n...\n"))
    await prd.write_prd(state)
    reloaded = store.open(artifact.slug)
    assert reloaded.type == "fullstack-stateless-app"


async def test_write_prd_does_not_touch_type_when_unchanged(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.create(name="Clock", description="d", type="html-app")
    artifact_dir = store.folder_for(artifact.slug)
    state = _state(artifact_dir, slug=artifact.slug)
    state.final_artifact_type = "html-app"  # same as state.artifact_type
    state.session._llm.plan = AsyncMock(return_value=_response("## Goal\n...\n"))

    calls: list[dict] = []
    original_update = ArtifactStore.update

    def spy_update(self, slug, **kwargs):
        calls.append(kwargs)
        return original_update(self, slug, **kwargs)

    monkeypatch.setattr(ArtifactStore, "update", spy_update)
    await prd.write_prd(state)
    assert calls == []
    assert store.open(artifact.slug).type == "html-app"


def test_write_prd_instruction_also_forbids_process_meta_commentary():
    """The full PRD is a continuation of the same conversation, so it is
    just as exposed to the redo-framing leak as the brief."""
    lowered = prd._WRITE_PRD_INSTRUCTION.lower()
    assert "redo" in lowered or "regeneration" in lowered



def test_write_prd_instruction_promotes_accepted_proposals_and_defaults():
    """The brief now separates proposals and questions from requirements;
    once accepted they have to become requirements, or the PRD loses them
    the way the 2026-09-16 run lost both open points."""
    text = prd._WRITE_PRD_INSTRUCTION
    assert "Proposals are accepted" in text
    assert "the stated default is the decision" in text
    assert "expanding the same five sections" not in text


def test_write_prd_instruction_asks_data_model_to_skip_negatives():
    """Fifth live run 2026-09-16: the brief already had this rule and was
    clean, but the PRD's Data model still said "the connected databases
    are not used" — the kickoff's connections section is in its history."""
    text = prd._WRITE_PRD_INSTRUCTION
    assert "State only what IS used" in text
    assert "does NOT use" in text


def test_write_prd_instruction_forbids_copying_long_form_source_content():
    """The 2026-08-27 live run retold one article three times (pad -> prd.md
    -> spec.md -> index.html). The PRD gives the structure, not the content.

    The citation half of the old rule is gone with the merge: the generator
    no longer finds the source by following a pointer in the PRD, because the
    spec node carries what it needs forward and `data_notes` holds the code
    that fetched it.
    """
    text = prd._WRITE_PRD_INSTRUCTION
    assert "long-form source content" in text
    assert "slide outline" in text


def test_the_prd_no_longer_demands_connection_code():
    """I-25: the PRD weighed 18-21KB because it was the only channel to the
    generator. It is not any more — `data_notes` carries the working code and
    `spec.md` carries the source material — so its mandate shrinks to the
    requirements the user agreed to.

    Asserts the removal of the exact REQUIREMENT, not the absence of a word.
    The new text legitimately says "do not restate connection code", so a
    substring check for "connection code" would be red on the very paragraph
    that implements this change — and the natural way to make it green would
    be deleting that paragraph. Same trap as `"6,000 characters"` being a
    substring of `"16,000 characters"`.
    """
    from anton.core.tools.generate_artifact.discovery.prd import _WRITE_PRD_INSTRUCTION

    # The old demand, verbatim from the pre-merge instruction.
    assert "PLUS connection code examples" not in _WRITE_PRD_INSTRUCTION
    assert "cite the scratchpad name and cell" not in _WRITE_PRD_INSTRUCTION
    # The new framing: this document records the agreement, it is not the
    # only thing downstream reads.
    assert "not the only thing the build step reads" in _WRITE_PRD_INSTRUCTION


async def test_write_prd_updates_the_state_not_only_the_file(tmp_path):
    """`prd_section` renders `state.prd` and declares it authoritative. A PRD
    rewritten during this call — which is what every user correction produces
    — has to reach the spec node, or the correction is lost inside one run."""
    artifact_dir = tmp_path / "artifacts" / "s"
    artifact_dir.mkdir(parents=True)
    state = _state(artifact_dir, prd="# Stale PRD from the previous call")
    state.session._llm.plan = AsyncMock(
        return_value=_response("## Goal\nWeekly, not daily.\n")
    )
    await prd.write_prd(state)
    assert state.prd != "# Stale PRD from the previous call"
    assert state.prd == (artifact_dir / "prd.md").read_text(encoding="utf-8")


def test_write_prd_instruction_names_the_latest_brief_as_the_source():
    """On the revise path the history holds brief v1, the user's feedback and
    brief v2, which already reflects that feedback. The old "anything the user
    changed in their reply overrides the brief" invited the model to re-apply
    the feedback on top of v2; the source is the latest brief, full stop."""
    text = prd._WRITE_PRD_INSTRUCTION
    assert "The latest brief in this conversation is the source" in text
    assert "overrides the brief" not in text
    assert "Do not call any tool" in text


def test_write_prd_instruction_lets_a_data_free_artifact_skip_the_data_model():
    """Sixth live run 2026-09-16: "fields and a few sample rows" was demanded
    unconditionally, so the PRD for a card game invented an in-memory card
    table and a fixed emoji list, which the spec then froze verbatim."""
    text = prd._WRITE_PRD_INSTRUCTION
    assert "If there is no external data and nothing stored, one line saying so" in text
    assert "Do not describe in-memory variables or invent content lists" in text


def test_write_prd_instruction_keeps_sensitive_sample_values_out():
    """Sample rows come from real connected sources. A PRD is a document a
    person reads and shares; real names, emails or salaries must not land in
    it — made-up values of the same shape carry the same information."""
    text = prd._WRITE_PRD_INSTRUCTION
    assert "looks sensitive" in text
    assert "made-up ones of the same shape" in text


def test_write_prd_instruction_leaves_visual_style_to_the_build_step():
    """Sixth live run: the UI/UX section added flip animation, hover feedback
    and a "calm, friendly" style nobody agreed to, under a header downstream
    reads as "accepted by the user". Layout follows the requirements; the
    look is the spec's decision."""
    text = prd._WRITE_PRD_INSTRUCTION
    assert "Do not invent a visual style, animations or effects here" in text
    assert "Do not add requirements the brief does not contain" in text


def test_write_prd_instruction_asks_for_brevity_without_a_line_count():
    """Same shape as the brief and gathering instructions: a task, sections,
    rules, the user's language, and a push toward brevity — phrased as
    "as short as the requirements allow" rather than a line count, because a
    presentation outline or a multi-table data model legitimately runs long."""
    text = prd._WRITE_PRD_INSTRUCTION
    assert text.startswith("## Your task")
    assert "## Sections" in text and "## Rules" in text
    assert "Write in the language of the user request" in text
    assert "as short as the requirements allow" in text
    assert "lines" not in text.lower()
