"""Characterization tests for select_path — written BEFORE the elicit()
migration so the refactor cannot change observable behaviour silently.

The tool's JSON result is a contract with the LLM: statuses resolved /
invalid / no_matches / cancelled / picker_unavailable / error, plus the
auto_resolved and path fields. Nothing here may change in Task 7.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from anton.core.interaction.elicit import MAX_QUESTIONS_PER_TURN, AskAnswer
from anton.core.tools.tool_handlers import handle_select_path


class _FakeElicitor:
    """Terminal-free stand-in: returns a scripted choice."""

    supported_kinds = ("choice", "path")
    answer_hint = "hint"
    timeout_s = None

    def __init__(self, chosen: str | None) -> None:
        self.chosen = chosen
        self.requests: list = []

    async def begin(self, question_id, request):
        return None

    async def end(self, question_id):
        return None

    async def ask(self, question_id, request):
        self.requests.append(request)
        if self.chosen is None:
            return AskAnswer(status="cancelled")
        return AskAnswer(status="answered", values=(self.chosen,))


async def _noop_emit(event):
    """Stand in for ChatSession.emit, which is a no-op with no emitter attached.

    elicit() calls ``await session.emit(...)`` unconditionally, so a fake
    session without this method raises AttributeError before ever reaching
    elicitor.ask().
    """
    return None


def _session(tmp_path: Path, elicitor=None):
    return SimpleNamespace(
        _console=None,
        elicitor=elicitor,
        emitter=None,
        emit=_noop_emit,
        question_count=0,
        answer_wait_s=0.0,
        escape_watcher=None,
        _workspace=SimpleNamespace(base=tmp_path),
    )


async def test_pick_single_candidate_auto_resolves_without_elicitor(tmp_path):
    (tmp_path / "report.csv").write_text("a,b\n")
    result = json.loads(
        await handle_select_path(
            _session(tmp_path), {"prompt": "Which one?", "pattern": "*.csv"}
        )
    )
    assert result["status"] == "resolved"
    assert result["auto_resolved"] is True
    assert result["path"].endswith("report.csv")


async def test_pick_no_candidates_reports_no_matches(tmp_path):
    result = json.loads(
        await handle_select_path(
            _session(tmp_path), {"prompt": "Which one?", "pattern": "*.parquet"}
        )
    )
    assert result["status"] == "no_matches"


async def test_pick_ambiguous_without_elicitor_is_picker_unavailable(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "report.csv").write_text("x")
    (tmp_path / "b" / "report.csv").write_text("y")
    result = json.loads(
        await handle_select_path(
            _session(tmp_path), {"prompt": "Which one?", "pattern": "**/report.csv"}
        )
    )
    assert result["status"] == "picker_unavailable"
    assert len(result["candidates"]) == 2


async def test_pick_ambiguous_resolves_through_elicitor(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "report.csv").write_text("x")
    (tmp_path / "b" / "report.csv").write_text("y")
    chosen = str((tmp_path / "b" / "report.csv").resolve())
    elicitor = _FakeElicitor(chosen)
    result = json.loads(
        await handle_select_path(
            _session(tmp_path, elicitor),
            {"prompt": "Which one?", "pattern": "**/report.csv"},
        )
    )
    assert result == {"status": "resolved", "path": chosen}
    assert len(elicitor.requests[0].options) == 2


async def test_pick_cancelled(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "report.csv").write_text("x")
    (tmp_path / "b" / "report.csv").write_text("y")
    result = json.loads(
        await handle_select_path(
            _session(tmp_path, _FakeElicitor(None)),
            {"prompt": "Which one?", "pattern": "**/report.csv"},
        )
    )
    assert result["status"] == "cancelled"


async def test_pick_rejects_a_path_that_was_never_offered(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "report.csv").write_text("x")
    (tmp_path / "b" / "report.csv").write_text("y")
    result = json.loads(
        await handle_select_path(
            _session(tmp_path, _FakeElicitor("/etc/passwd")),
            {"prompt": "Which one?", "pattern": "**/report.csv"},
        )
    )
    assert result["status"] == "invalid"


async def test_browse_without_elicitor_is_picker_unavailable(tmp_path):
    result = json.loads(
        await handle_select_path(_session(tmp_path), {"prompt": "Find the folder"})
    )
    assert result["status"] == "picker_unavailable"


async def test_browse_resolves_a_typed_folder(tmp_path):
    target = tmp_path / "data"
    target.mkdir()
    result = json.loads(
        await handle_select_path(
            _session(tmp_path, _FakeElicitor(str(target.resolve()))),
            {"prompt": "Find the folder", "kind": "folder"},
        )
    )
    assert result["status"] == "resolved"
    assert result["path"] == str(target.resolve())


class _StatusElicitor:
    """Terminal-free stand-in: always answers with a scripted non-answered status."""

    supported_kinds = ("choice", "path")
    answer_hint = "hint"
    timeout_s = None

    def __init__(self, status: str) -> None:
        self.status = status
        self.requests: list = []

    async def begin(self, question_id, request):
        return None

    async def end(self, question_id):
        return None

    async def ask(self, question_id, request):
        self.requests.append(request)
        return AskAnswer(status=self.status)


async def test_browse_limit_is_error_not_cancelled(tmp_path):
    result = json.loads(
        await handle_select_path(
            _session(tmp_path, _StatusElicitor("limit")), {"prompt": "Find the folder"}
        )
    )
    assert result["status"] == "error"
    assert "dismissed" not in result["message"]


async def test_browse_timeout_is_error(tmp_path):
    result = json.loads(
        await handle_select_path(
            _session(tmp_path, _StatusElicitor("timeout")), {"prompt": "Find the folder"}
        )
    )
    assert result["status"] == "error"


async def test_browse_unavailable_is_picker_unavailable(tmp_path):
    result = json.loads(
        await handle_select_path(
            _session(tmp_path, _StatusElicitor("unavailable")), {"prompt": "Find the folder"}
        )
    )
    assert result["status"] == "picker_unavailable"


async def test_pick_ambiguous_limit_is_error_not_cancelled(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "report.csv").write_text("x")
    (tmp_path / "b" / "report.csv").write_text("y")
    result = json.loads(
        await handle_select_path(
            _session(tmp_path, _StatusElicitor("limit")),
            {"prompt": "Which one?", "pattern": "**/report.csv"},
        )
    )
    assert result["status"] == "error"
    assert "dismissed" not in result["message"]


async def test_pick_ambiguous_timeout_is_error(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "report.csv").write_text("x")
    (tmp_path / "b" / "report.csv").write_text("y")
    result = json.loads(
        await handle_select_path(
            _session(tmp_path, _StatusElicitor("timeout")),
            {"prompt": "Which one?", "pattern": "**/report.csv"},
        )
    )
    assert result["status"] == "error"


async def test_pick_ambiguous_unavailable_is_picker_unavailable(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "report.csv").write_text("x")
    (tmp_path / "b" / "report.csv").write_text("y")
    result = json.loads(
        await handle_select_path(
            _session(tmp_path, _StatusElicitor("unavailable")),
            {"prompt": "Which one?", "pattern": "**/report.csv"},
        )
    )
    assert result["status"] == "picker_unavailable"


async def test_browse_budget_exhausted_through_real_elicit_is_error(tmp_path):
    """Drives the real elicit() budget check (not a faked status) so the path
    that actually produced the defect is pinned: exhausting question_count
    before asking must not be reported to the model as a dismissed picker.
    """
    session = _session(tmp_path, _FakeElicitor("irrelevant"))
    session.question_count = MAX_QUESTIONS_PER_TURN
    result = json.loads(
        await handle_select_path(session, {"prompt": "Find the folder"})
    )
    assert result["status"] == "error"
    assert "dismissed" not in result["message"]


async def test_elicitor_exception_becomes_error_status(tmp_path):
    class _Boom:
        supported_kinds = ("choice", "path")
        answer_hint = "hint"
        timeout_s = None

        async def begin(self, question_id, request):
            return None

        async def end(self, question_id):
            return None

        async def ask(self, question_id, request):
            raise RuntimeError("picker died")

    target = tmp_path / "data"
    target.mkdir()
    result = json.loads(
        await handle_select_path(
            _session(tmp_path, _Boom()), {"prompt": "Find the folder"}
        )
    )
    assert result["status"] == "error"
