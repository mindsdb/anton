"""Tests for the /skill slash-command handlers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

from anton.commands.skills import (
    _format_history_turns,
    _format_scratchpad_cells,
    _gather_session_scratchpad_cells,
    _SkillDraft,
    handle_skill_remove,
    handle_skill_save,
    handle_skill_show,
    handle_skills_list,
)
from anton.core.memory.skills import Skill, SkillStore


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def store(tmp_path: Path) -> SkillStore:
    return SkillStore(root=tmp_path / "skills")


@pytest.fixture()
def console() -> Console:
    # Capture-friendly console; record=True lets us read output if needed
    return Console(record=True, width=120)


def _fake_cell(code: str, stdout: str = "ok", stderr: str = "", error=None):
    return SimpleNamespace(
        code=code, stdout=stdout, stderr=stderr, error=error, description=""
    )


def _make_session(
    *,
    draft: _SkillDraft | None = None,
    raises: Exception | None = None,
    cells: list | None = None,
    history: list | None = None,
) -> MagicMock:
    """Build a fake session whose `_llm.generate_object()` returns a known draft.

    Pass `draft` to return a specific `_SkillDraft`, or `raises` to make
    the call raise that exception.
    """
    session = MagicMock()
    session._history = history or []
    pad = SimpleNamespace(cells=cells or [])
    session._scratchpads = SimpleNamespace(_pads={"main": pad})
    session._llm = MagicMock()
    if raises is not None:
        session._llm.generate_object = AsyncMock(side_effect=raises)
    else:
        session._llm.generate_object = AsyncMock(return_value=draft)
    return session


def _draft(
    *,
    label: str = "csv_summary",
    name: str = "CSV Summary",
    description: str = "",
    when_to_use: str = "When summarizing CSV files.",
    declarative_md: str = "1. Load.\n2. Describe.",
) -> _SkillDraft:
    """Convenience constructor for test drafts."""
    return _SkillDraft(
        label=label,
        name=name,
        description=description,
        when_to_use=when_to_use,
        declarative_md=declarative_md,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper unit tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatScratchpadCells:
    def test_empty(self):
        assert "no scratchpad" in _format_scratchpad_cells([])

    def test_renders_code_and_output(self):
        cells = [_fake_cell(code="print(1+1)", stdout="2")]
        out = _format_scratchpad_cells(cells)
        assert "print(1+1)" in out
        assert "2" in out
        assert "Cell 1" in out

    def test_truncates_long_code(self):
        long_code = "x = 1\n" * 1000
        cells = [_fake_cell(code=long_code, stdout="")]
        out = _format_scratchpad_cells(cells)
        assert "[truncated]" in out


class TestFormatHistoryTurns:
    def test_empty(self):
        assert "no conversation" in _format_history_turns([])

    def test_string_content(self):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello back"},
        ]
        out = _format_history_turns(history)
        assert "hi" in out
        assert "hello back" in out

    def test_structured_content_with_text_blocks(self):
        history = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "the answer"},
                    {"type": "tool_use", "id": "x", "name": "y", "input": {}},
                ],
            }
        ]
        out = _format_history_turns(history)
        assert "the answer" in out

    def test_skips_empty_turns(self):
        history = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "hello"},
        ]
        out = _format_history_turns(history)
        assert "hello" in out

    def test_max_turns_limit(self):
        history = [
            {"role": "user", "content": f"msg {i}"} for i in range(20)
        ]
        out = _format_history_turns(history, max_turns=3)
        # Should only include the last 3
        assert "msg 19" in out
        assert "msg 17" in out
        assert "msg 15" not in out


class TestGatherSessionCells:
    def test_collects_from_multiple_pads(self):
        pad_a = SimpleNamespace(cells=[_fake_cell("a")])
        pad_b = SimpleNamespace(cells=[_fake_cell("b"), _fake_cell("c")])
        session = SimpleNamespace(
            _scratchpads=SimpleNamespace(_pads={"a": pad_a, "b": pad_b})
        )
        cells = _gather_session_scratchpad_cells(session)
        assert len(cells) == 3


# ─────────────────────────────────────────────────────────────────────────────
# /skill save
# ─────────────────────────────────────────────────────────────────────────────


class TestSkillSave:
    @pytest.mark.asyncio
    async def test_happy_path(self, console, store):
        session = _make_session(
            draft=_draft(
                label="csv_summary",
                name="CSV Summary",
                description="Load and summarize a CSV.",
                when_to_use="User asks to summarize a CSV file.",
                declarative_md="1. Load.\n2. Describe.\n3. Plot.",
            ),
            cells=[_fake_cell("import pandas as pd; df = pd.read_csv('x.csv')")],
            history=[{"role": "user", "content": "summarize x.csv"}],
        )

        await handle_skill_save(console, session, store=store)

        loaded = store.load("csv_summary")
        assert loaded is not None
        assert loaded.name == "CSV Summary"
        assert loaded.when_to_use == "User asks to summarize a CSV file."
        assert "Load." in loaded.declarative_md
        assert loaded.provenance == "manual"
        assert loaded.created_at  # ISO timestamp set

    @pytest.mark.asyncio
    async def test_passes_skill_draft_schema_to_llm(self, console, store):
        """generate_object should be called with the _SkillDraft Pydantic class."""
        session = _make_session(
            draft=_draft(),
            cells=[_fake_cell("x")],
            history=[{"role": "user", "content": "go"}],
        )
        await handle_skill_save(console, session, store=store)
        session._llm.generate_object.assert_called_once()
        call_args = session._llm.generate_object.call_args
        assert call_args.args[0] is _SkillDraft

    @pytest.mark.asyncio
    async def test_label_collision_appends_number(self, console, store):
        # Pre-existing skill with the label the LLM will return
        store.save(
            Skill(
                label="csv_summary",
                name="Existing",
                description="",
                when_to_use="",
                declarative_md="prior",
                created_at="2026-04-09T00:00:00+00:00",
                provenance="manual",
            )
        )
        session = _make_session(
            draft=_draft(
                label="csv_summary",
                name="New CSV Summary",
                description="",
                when_to_use="User asks for CSV stats.",
                declarative_md="step 1",
            ),
            cells=[_fake_cell("x")],
            history=[{"role": "user", "content": "go"}],
        )

        await handle_skill_save(console, session, store=store)

        # Original is unchanged
        original = store.load("csv_summary")
        assert original is not None
        assert original.declarative_md == "prior"
        # New one was saved with a unique label
        new = store.load("csv_summary_2")
        assert new is not None
        assert new.declarative_md == "step 1"

    @pytest.mark.asyncio
    async def test_name_hint_is_passed_to_llm(self, console, store):
        session = _make_session(
            draft=_draft(
                label="data_loader",
                name="Data Loader",
                when_to_use="User asks to load data.",
            ),
            cells=[_fake_cell("x")],
            history=[{"role": "user", "content": "go"}],
        )

        await handle_skill_save(
            console, session, name_hint="data loader", store=store
        )

        # The hint should appear in the prompt sent to the LLM
        call_args = session._llm.generate_object.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "data loader" in user_msg

    @pytest.mark.asyncio
    async def test_empty_procedure_refuses_save(self, console, store):
        session = _make_session(
            draft=_draft(
                label="empty",
                name="Empty",
                when_to_use="x",
                declarative_md="",  # blank — refuse
            ),
            cells=[_fake_cell("x")],
            history=[{"role": "user", "content": "go"}],
        )
        await handle_skill_save(console, session, store=store)
        assert store.list_all() == []

    @pytest.mark.asyncio
    async def test_no_work_in_session_aborts(self, console, store):
        session = _make_session(
            draft=_draft(),
            cells=[],
            history=[],
        )
        await handle_skill_save(console, session, store=store)
        # No LLM call was even made
        session._llm.generate_object.assert_not_called()
        assert store.list_all() == []

    @pytest.mark.asyncio
    async def test_llm_exception_is_caught(self, console, store):
        session = _make_session(
            raises=RuntimeError("network down"),
            cells=[_fake_cell("x")],
            history=[{"role": "user", "content": "go"}],
        )
        # Must not raise
        await handle_skill_save(console, session, store=store)
        assert store.list_all() == []

    @pytest.mark.asyncio
    async def test_validation_error_is_caught(self, console, store):
        """If the LLM produces output that fails Pydantic validation
        (rare with forced tool_choice but possible), we surface a
        warning and don't save anything."""
        from pydantic import ValidationError as _PVE

        try:
            _SkillDraft.model_validate({})  # missing required fields
        except _PVE as exc:
            session = _make_session(
                raises=exc,
                cells=[_fake_cell("x")],
                history=[{"role": "user", "content": "go"}],
            )
        await handle_skill_save(console, session, store=store)
        assert store.list_all() == []


# ─────────────────────────────────────────────────────────────────────────────
# /skills list, /skill show, /skill remove
# ─────────────────────────────────────────────────────────────────────────────


class TestListShowRemove:
    def test_list_empty(self, console, store):
        # Should not raise
        handle_skills_list(console, store=store)

    def test_list_with_skills(self, console, store):
        store.save(
            Skill(
                label="csv_summary",
                name="CSV Summary",
                description="",
                when_to_use="When the user asks about a CSV",
                declarative_md="step",
                created_at="2026-04-10T00:00:00+00:00",
                provenance="manual",
            )
        )
        handle_skills_list(console, store=store)
        # Sanity-check that the rendered output mentions the label
        out = console.export_text()
        assert "csv_summary" in out

    def test_show_existing(self, console, store):
        store.save(
            Skill(
                label="csv_summary",
                name="CSV Summary",
                description="A CSV utility.",
                when_to_use="when needed",
                declarative_md="1. Load\n2. Describe",
                created_at="2026-04-10T00:00:00+00:00",
                provenance="manual",
            )
        )
        handle_skill_show(console, "csv_summary", store=store)
        out = console.export_text()
        assert "CSV Summary" in out
        assert "Load" in out

    def test_show_unknown_suggests_closest(self, console, store):
        store.save(
            Skill(
                label="csv_summary",
                name="CSV",
                description="",
                when_to_use="",
                declarative_md="x",
                created_at="2026-04-10T00:00:00+00:00",
                provenance="manual",
            )
        )
        handle_skill_show(console, "csv_sumary", store=store)  # typo
        out = console.export_text()
        assert "csv_summary" in out

    def test_show_no_args(self, console, store):
        handle_skill_show(console, "", store=store)
        out = console.export_text()
        assert "Usage" in out

    def test_remove_existing(self, console, store):
        store.save(
            Skill(
                label="zap",
                name="Zap",
                description="",
                when_to_use="",
                declarative_md="x",
                created_at="2026-04-10T00:00:00+00:00",
                provenance="manual",
            )
        )
        handle_skill_remove(console, "zap", store=store)
        assert store.load("zap") is None

    def test_remove_unknown(self, console, store):
        handle_skill_remove(console, "nope", store=store)
        out = console.export_text()
        assert "No skill" in out
