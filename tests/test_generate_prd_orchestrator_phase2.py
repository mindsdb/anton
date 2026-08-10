"""orchestrator.py phase 2: draft_brief, show_and_confirm, classify_feedback,
write_prd. Each step is tested in isolation with a fake `session._llm` and a
monkeypatched `elicit`."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from anton.core.artifacts import ArtifactStore
from anton.core.interaction.elicit import AskAnswer
from anton.core.llm.provider import LLMResponse, Usage
from anton.core.tools.generate_prd import orchestrator
from anton.core.tools.generate_prd.state import PrdState


def _response(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[], usage=Usage(input_tokens=1, output_tokens=1))


def _state(artifact_path: Path, **over) -> PrdState:
    # `_workspace.artifacts_dir` defaults to `artifact_path`'s parent so
    # `_artifact_store(session)` resolves to the same root a test's own
    # `ArtifactStore(tmp_path / "artifacts")` uses — needed by the
    # write_prd type-update tests below, harmless for the tests that never
    # reach that branch.
    base = dict(
        session=SimpleNamespace(
            _llm=SimpleNamespace(plan=AsyncMock()),
            question_count=0,
            elicitor=None,
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


async def test_draft_brief_sets_brief_markdown_and_appends_to_messages(tmp_path):
    state = _state(tmp_path)
    state.session._llm.plan = AsyncMock(return_value=_response("## Goal\nAn analog clock.\n"))
    await orchestrator.draft_brief(state)
    assert state.brief_markdown == "## Goal\nAn analog clock."
    assert state.messages[-1]["content"] == state.brief_markdown


async def test_draft_brief_raises_on_an_empty_reply(tmp_path):
    """A model that calls a tool instead of replying with text (tools stay
    defined in this phase for the Anthropic API's sake, see `_phase2_tools`)
    must not silently produce an empty brief that then sails through
    show_and_confirm as if it were real content."""
    state = _state(tmp_path)
    state.session._llm.plan = AsyncMock(return_value=_response(""))
    with pytest.raises(RuntimeError, match="no text"):
        await orchestrator.draft_brief(state)


async def test_show_and_confirm_accept(monkeypatch):
    state = _state(Path("/tmp/x"))
    state.brief_markdown = "## Goal\n..."

    async def fake_elicit(session, request):
        assert request.prompt == state.brief_markdown
        assert {o.value for o in request.options} == {"accept", "cancel"}
        return AskAnswer(status="answered", values=("accept",))

    monkeypatch.setattr(orchestrator.sub_tools, "ask_via_elicit", fake_elicit)
    outcome = await orchestrator.show_and_confirm(state)
    assert outcome == "accepted"


async def test_show_and_confirm_cancel(monkeypatch):
    async def fake_elicit(session, request):
        return AskAnswer(status="answered", values=("cancel",))

    monkeypatch.setattr(orchestrator.sub_tools, "ask_via_elicit", fake_elicit)
    state = _state(Path("/tmp/x"))
    state.brief_markdown = "## Goal\n..."
    outcome = await orchestrator.show_and_confirm(state)
    assert outcome == "cancelled"


async def test_show_and_confirm_user_declined_to_answer(monkeypatch):
    async def fake_elicit(session, request):
        return AskAnswer(status="cancelled")

    monkeypatch.setattr(orchestrator.sub_tools, "ask_via_elicit", fake_elicit)
    state = _state(Path("/tmp/x"))
    state.brief_markdown = "## Goal\n..."
    outcome = await orchestrator.show_and_confirm(state)
    assert outcome == "cancelled"


async def test_show_and_confirm_free_text_is_a_revision(monkeypatch):
    async def fake_elicit(session, request):
        return AskAnswer(status="answered", text="make it dark mode")

    monkeypatch.setattr(orchestrator.sub_tools, "ask_via_elicit", fake_elicit)
    state = _state(Path("/tmp/x"))
    state.brief_markdown = "## Goal\n..."
    outcome = await orchestrator.show_and_confirm(state)
    assert outcome == "revise"
    assert "make it dark mode" in state.messages[-1]["content"]
    assert "make it dark mode" in state.qa_log_markdown()


@pytest.mark.parametrize("status", ["limit", "unavailable", "error"])
async def test_show_and_confirm_budget_or_channel_failure_is_unconfirmed_not_cancelled(monkeypatch, status):
    async def fake_elicit(session, request):
        return AskAnswer(status=status)

    monkeypatch.setattr(orchestrator.sub_tools, "ask_via_elicit", fake_elicit)
    state = _state(Path("/tmp/x"))
    state.brief_markdown = "## Goal\n..."
    outcome = await orchestrator.show_and_confirm(state)
    assert outcome == "unconfirmed"


async def test_classify_feedback_returns_the_model_verdict(monkeypatch):
    state = _state(Path("/tmp/x"))

    async def fake_generate_object(schema, *, system, messages):
        return orchestrator.FeedbackVerdict(route="back_to_gathering", reasoning="needs more data")

    state.session._llm.generate_object = fake_generate_object
    route = await orchestrator.classify_feedback(state)
    assert route == "back_to_gathering"


async def test_classify_feedback_falls_back_to_revise_brief_on_an_unknown_route(monkeypatch):
    state = _state(Path("/tmp/x"))

    async def fake_generate_object(schema, *, system, messages):
        return orchestrator.FeedbackVerdict(route="something_else", reasoning="")

    state.session._llm.generate_object = fake_generate_object
    route = await orchestrator.classify_feedback(state)
    assert route == "revise_brief"


async def test_write_prd_saves_the_file_and_returns_its_text(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "s"
    artifact_dir.mkdir(parents=True)
    state = _state(artifact_dir)
    state.session._llm.plan = AsyncMock(return_value=_response("## Goal\nFull PRD text.\n"))
    text = await orchestrator.write_prd(state)
    assert text == "## Goal\nFull PRD text."
    assert (artifact_dir / "prd.md").read_text(encoding="utf-8") == text


async def test_write_prd_raises_on_an_empty_reply_instead_of_writing_an_empty_file(tmp_path):
    """Same failure shape as draft_brief's guard: an empty reply must never
    become an empty prd.md on disk reported back as `prd_written`."""
    artifact_dir = tmp_path / "artifacts" / "s"
    artifact_dir.mkdir(parents=True)
    state = _state(artifact_dir)
    state.session._llm.plan = AsyncMock(return_value=_response(""))
    with pytest.raises(RuntimeError, match="no text"):
        await orchestrator.write_prd(state)
    assert not (artifact_dir / "prd.md").exists()


async def test_write_prd_updates_the_artifact_type_when_it_changed(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.create(name="Clock", description="d", type="html-app")
    artifact_dir = store.folder_for(artifact.slug)
    state = _state(artifact_dir, slug=artifact.slug)
    state.final_artifact_type = "fullstack-stateless-app"
    state.session._llm.plan = AsyncMock(return_value=_response("## Goal\n...\n"))
    await orchestrator.write_prd(state)
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
    await orchestrator.write_prd(state)
    assert calls == []
    assert store.open(artifact.slug).type == "html-app"


def test_draft_brief_instruction_asks_for_a_lead_in_sentence():
    """Live-testing feedback (ENG-969): the brief used to land on the user
    with zero framing — just headers. The model must open with one
    sentence explaining what follows, in the reply's own language."""
    assert "lead-in" in orchestrator._DRAFT_BRIEF_INSTRUCTION.lower()
    assert "same language" in orchestrator._DRAFT_BRIEF_INSTRUCTION.lower()


def test_draft_brief_instruction_forbids_process_meta_commentary():
    """Live-testing feedback: a redo of an existing artifact leaked
    "recreating X, but going through the PRD step" into Goal — process
    narration, not what the artifact is for."""
    lowered = orchestrator._DRAFT_BRIEF_INSTRUCTION.lower()
    assert "redo" in lowered or "regeneration" in lowered
    assert "generation tool" in lowered or "prd workflow" in lowered


def test_draft_brief_instruction_forbids_technical_detail():
    """Live-testing feedback: the brief is for a non-technical user and
    must not carry CSS clamp() values, hex colors, or similar — that
    belongs only in the full PRD after acceptance."""
    assert "hex" in orchestrator._DRAFT_BRIEF_INSTRUCTION.lower()
    assert "css" in orchestrator._DRAFT_BRIEF_INSTRUCTION.lower()


def test_draft_brief_instruction_asks_data_model_to_skip_negatives():
    """Live-testing feedback: the brief listed "no APIs, no fetch, no DB"
    for a clock with no external data — negatives nobody asked about."""
    assert "not used" in orchestrator._DRAFT_BRIEF_INSTRUCTION.lower()


def test_write_prd_instruction_also_forbids_process_meta_commentary():
    """The full PRD is a continuation of the same conversation, so it is
    just as exposed to the redo-framing leak as the brief."""
    lowered = orchestrator._WRITE_PRD_INSTRUCTION.lower()
    assert "redo" in lowered or "regeneration" in lowered


def test_draft_brief_instruction_asks_for_a_closing_continue_line():
    """Live-testing feedback (ENG-969): the brief used to end abruptly on
    "1. Accept / 2. Cancel" with no framing sentence. The model must add a
    closing line, in-language, that explains a bare Enter continues."""
    lowered = orchestrator._DRAFT_BRIEF_INSTRUCTION.lower()
    assert "closing line" in lowered
    assert "press enter" in lowered


async def test_show_and_confirm_defaults_to_accept_on_a_bare_enter(monkeypatch):
    """A bare Enter (no selection, no free text) must resolve to "accept",
    not the channel's own default of "cancelled" — accepting the brief as
    drafted is the overwhelmingly common case."""
    seen_requests = []

    async def fake_elicit(session, request):
        seen_requests.append(request)
        return AskAnswer(status="answered", values=(request.default_value,))

    monkeypatch.setattr(orchestrator.sub_tools, "ask_via_elicit", fake_elicit)
    state = _state(Path("/tmp/x"))
    state.brief_markdown = "## Goal\n..."
    outcome = await orchestrator.show_and_confirm(state)
    assert outcome == "accepted"
    assert seen_requests[0].default_value == "accept"
