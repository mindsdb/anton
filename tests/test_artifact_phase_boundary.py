"""The D->E boundary: what phase A gathered has to reach phase E by code.

The shared message list is dropped once `spec.md` is written, so everything
a generation loop needs verbatim travels in `data_notes` / `web_notes` — two
strings rendered from what the discovery loop actually did, never from what
a model chose to summarise.

These tests deliberately lock the FILLING of those channels, not only their
transport. A test that hands `_spec_context` a pre-filled state proves the
renderer works and says nothing about whether anything ever calls it — which
is exactly how the channel shipped dead once already (I-06).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from anton.core.llm.provider import LLMResponse, ToolCall, Usage
from anton.core.tools.generate_artifact import orchestrator
from anton.core.tools.generate_artifact.discovery import checkpoint as cp
from anton.core.tools.generate_artifact.discovery.engine import run_gathering_loop
from anton.core.tools.generate_artifact.discovery.notes import WEB_NOTES_MAX
from anton.core.tools.generate_artifact.state import GenState

ARTICLE_URL = "https://example.com/the-article"


def _state(tmp_path: Path, **kw) -> GenState:
    session = AsyncMock()
    session.question_count = 0
    base = dict(
        session=session, artifact_type="html-app", artifact_path=tmp_path,
        slug="a", user_request="turn that article into a dashboard",
        agent_understanding="a dashboard built from one article",
        is_fullstack=False,
    )
    base.update(kw)
    return GenState(**base)


def _response(*calls: tuple[str, dict], text: str = "") -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[
            ToolCall(id=str(i), name=name, input=inp)
            for i, (name, inp) in enumerate(calls)
        ],
        usage=Usage(input_tokens=1, output_tokens=1),
    )


async def _gather_one_web_fetch(state: GenState, monkeypatch, page: str) -> None:
    """Drive phase A through a single `web_fetch` and `finish_gathering`."""
    import anton.core.tools.web_tools as web_tools

    monkeypatch.setattr(
        web_tools, "handle_web_fetch_fallback", AsyncMock(return_value=page)
    )
    state.session._llm.plan = AsyncMock(
        return_value=_response(("web_fetch", {"url": ARTICLE_URL}))
    )
    state.session._llm.code = AsyncMock(
        return_value=_response(
            ("finish_gathering", {
                "artifact_type": "html-app",
                "notes": "read the article",
                "data_sources": ["the article"],
            })
        )
    )
    await run_gathering_loop(state)


# ── Filling: the calls a phase made become the channels phase E reads ────────

async def test_a_fetched_page_reaches_the_generation_context(tmp_path, monkeypatch):
    """The URL survives the boundary because code recorded it.

    This is the reference-request-1 case: a dashboard built from an article
    must link back to its source, and a model summarising a page into
    `spec.md` is precisely where a URL goes missing.
    """
    state = _state(tmp_path)
    await _gather_one_web_fetch(state, monkeypatch, "Body of the article. " * 20)

    assert state.web_calls, "phase A must record the web call it made"
    assert not state.web_notes, "nothing renders the notes during gathering"

    orchestrator._absorb_discovery_notes(state)

    assert ARTICLE_URL in state.web_notes
    assert ARTICLE_URL in orchestrator._spec_context(state)


async def test_the_code_that_fetched_the_data_reaches_the_generation_context(tmp_path):
    """`render_exec_notes` exists so the backend generator gets the exact
    working data-access code instead of the model's recollection of it."""
    state = _state(tmp_path)
    state.scratchpad_execs = [
        {"name": "d", "code": "rows = vault.query('select * from orders')", "output": "12 rows"}
    ]

    orchestrator._absorb_discovery_notes(state)

    assert "select * from orders" in state.data_notes
    assert "select * from orders" in orchestrator._spec_context(state)


async def test_absorbing_keeps_what_an_earlier_call_gathered(tmp_path):
    """Additive on purpose: `web_calls` holds only THIS call's work, while
    the notes may already carry a previous call's, restored from
    `discovery.json`. Re-rendering from scratch would drop the restored half.
    """
    state = _state(tmp_path)
    state.web_notes = "### Sources read from the web\n- web_fetch: https://old.example/x"
    state.data_notes = "Scratchpad `old`:\n```python\nprevious = 1\n```"
    state.web_calls = [{"kind": "web_fetch", "url": ARTICLE_URL, "title": "T", "excerpt": "e"}]
    state.scratchpad_execs = [{"name": "new", "code": "fresh = 2", "output": ""}]

    orchestrator._absorb_discovery_notes(state)

    assert "https://old.example/x" in state.web_notes
    assert ARTICLE_URL in state.web_notes
    assert "previous = 1" in state.data_notes
    assert "fresh = 2" in state.data_notes


# ── The call site: absorbing happens before anything is persisted ────────────

async def test_the_checkpoint_carries_what_this_call_gathered(tmp_path, monkeypatch):
    """A run that stops for confirmation has already paid for the gathering.

    If the notes were rendered after the checkpoint save — or only on the
    path that reaches generation — the continuation would restore a
    checkpoint whose channels are empty and re-fetch everything.
    """
    state = _state(tmp_path)

    async def _fake_discovery(st, *, entry):
        st.web_calls.append(
            {"kind": "web_fetch", "url": ARTICLE_URL, "title": "T", "excerpt": "body"}
        )
        st.brief = "## Goal\nA dashboard."
        return cp.STAGE_AWAITING_CONFIRMATION

    monkeypatch.setattr(orchestrator, "run_discovery", _fake_discovery)

    result = await orchestrator.run(state, entry=cp.ENTRY_FULL)
    assert result["status"] == "needs_confirmation"

    stored = cp.load(tmp_path)
    assert stored is not None
    assert ARTICLE_URL in stored.web_notes


async def test_a_cold_start_restores_the_notes_the_hot_path_had(tmp_path, monkeypatch):
    """The boundary has to survive the process, not just the phase switch."""
    hot = _state(tmp_path)
    await _gather_one_web_fetch(hot, monkeypatch, "Body of the article. " * 20)
    hot.scratchpad_execs = [{"name": "d", "code": "rows = q()", "output": "ok"}]
    hot.brief = "## Goal\nA dashboard."
    orchestrator._absorb_discovery_notes(hot)
    orchestrator._save_checkpoint(hot, cp.STAGE_PRD_WRITTEN)

    cold = _state(tmp_path)
    stored = cp.load(tmp_path)
    assert stored is not None
    cold.brief = stored.brief_markdown
    cold.data_notes = stored.data_notes
    cold.web_notes = stored.web_notes

    assert orchestrator._spec_context(cold) == orchestrator._spec_context(hot)
    assert ARTICLE_URL in orchestrator._spec_context(cold)
    assert "rows = q()" in orchestrator._spec_context(cold)


# ── What must NOT cross ──────────────────────────────────────────────────────

async def test_the_full_page_does_not_cross_the_boundary(tmp_path, monkeypatch):
    """`make_tech_spec` is the last node that sees the page body. Phase E
    re-sends its context on every write round, so what travels is the pointer
    plus enough text to recognise the source."""
    page = "ARTICLE " * 4000
    state = _state(tmp_path)
    await _gather_one_web_fetch(state, monkeypatch, page)
    orchestrator._absorb_discovery_notes(state)

    assert len(state.web_notes) <= WEB_NOTES_MAX + 200
    assert len(orchestrator._spec_context(state)) < len(page)


async def test_no_phase_a_message_reaches_the_generation_context(tmp_path, monkeypatch):
    """The shared history is dropped at the boundary; only the rendered
    channels cross. A phase-A message leaking in would restore exactly the
    context growth this merge exists to avoid."""
    state = _state(tmp_path)
    await _gather_one_web_fetch(state, monkeypatch, "unique-marker-in-the-page")
    orchestrator._absorb_discovery_notes(state)
    assert state.messages, "phase A must have built a shared history"

    state.messages = []  # what run() does at the D->E boundary
    context = orchestrator._spec_context(state)

    assert "tool_use" not in context
    assert "tool_result" not in context
