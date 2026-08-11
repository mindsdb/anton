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


# ─── ENG-1357: don't advertise a picker that cannot render ───────────────
#
# Browse mode is physically impossible without a path elicitor (cowork-server
# injects none). Advertising it anyway — while the injected system prompt
# forbids asking for a path and forbids guessing — left the model with no
# legitimate move when a user referred to an unattached file. It fabricated
# the user's data instead. These tests pin the two halves of the fix: the
# tool's remedies, and which definition each host gets.


class _PathlessElicitor(_FakeElicitor):
    """A host that can render choices but has no file browser."""

    supported_kinds = ("choice",)


async def test_browse_without_elicitor_points_at_attachment_not_a_typed_path(tmp_path):
    result = json.loads(
        await handle_select_path(_session(tmp_path), {"prompt": "Find the folder"})
    )
    assert result["status"] == "picker_unavailable"
    msg = result["message"]
    # The one route that works on this host.
    assert "attach" in msg.lower()
    # NOT the route the harness file-access policy forbids anyway.
    assert "plain text" not in msg.lower()
    assert "paste" not in msg.lower() or "Do not ask them to type or paste" in msg


async def test_ambiguous_pick_still_asks_which_candidate(tmp_path):
    """The browse remedy changed; the PICK remedy must NOT. Asking which of
    several paths you already found inside the project is legitimate — they are
    files the agent may read. Guards against a blanket message rewrite."""
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
    assert "which of these" in result["message"].lower()
    assert "attach" not in result["message"].lower()


async def test_no_matches_does_not_suggest_browsing_when_browse_cannot_run(tmp_path):
    result = json.loads(
        await handle_select_path(
            _session(tmp_path), {"prompt": "Which one?", "pattern": "*.parquet"}
        )
    )
    assert result["status"] == "no_matches"
    # Must not tell the model to fall back to browse mode — that lands on
    # picker_unavailable here. (Careful: the message legitimately contains the
    # word "browser", so match the suggestion, not the substring.)
    assert "let the user browse" not in result["message"].lower()
    assert "attach" in result["message"].lower()


async def test_no_matches_still_suggests_browsing_where_browse_works(tmp_path):
    """The honest advice must survive on hosts that can actually browse."""
    result = json.loads(
        await handle_select_path(
            _session(tmp_path, _FakeElicitor(None)),
            {"prompt": "Which one?", "pattern": "*.parquet"},
        )
    )
    assert result["status"] == "no_matches"
    assert "browse" in result["message"].lower()


# ─── which definition each host is given ────────────────────────────────


def _registered_select_path(session):
    session._build_tools()
    return next(t for t in session.tool_registry.get_tool_defs() if t.name == "select_path")


def test_pathless_host_gets_the_pick_only_definition(make_session):
    tool = _registered_select_path(make_session(elicitor=_PathlessElicitor(None)))
    # Browse is not offered as a mode — the full definition advertises it as a
    # "• BROWSE —" bullet, which must be gone. It may still be *named* in order
    # to say it does not exist here, so match the advertisement, not the word.
    assert "• BROWSE" not in tool.description
    assert "there is no BROWSE mode" in tool.description
    # ...and the injected system prompt names attachment instead of steering
    # the model at a picker that cannot render. This is the sentence whose
    # absence produced the fabrication.
    assert "attach" in tool.prompt.lower()
    assert "no file browser" in tool.prompt.lower()
    # ...and the FUNCTION-CALLING SCHEMA must agree with the prose. `replace()`
    # copies input_schema by reference, so without an explicit override the
    # model reads "there is no BROWSE mode" in the description while the schema
    # still offers `start_dir`, documented as "BROWSE mode only" (caught in
    # review of #331 — the description-only assertions above missed it).
    assert "start_dir" not in tool.input_schema["properties"]
    # The parameters pick mode actually needs are still there.
    for key in ("prompt", "candidates", "pattern", "base_dir"):
        assert key in tool.input_schema["properties"]


def test_host_with_a_path_elicitor_keeps_browse_mode(make_session):
    tool = _registered_select_path(make_session(elicitor=_FakeElicitor(None)))
    assert "BROWSE" in tool.description
    assert "Browse mode" in tool.prompt
    # The browse-only parameter belongs here, and only here.
    assert "start_dir" in tool.input_schema["properties"]


def test_no_elicitor_at_all_gets_the_pick_only_definition(make_session):
    tool = _registered_select_path(make_session())
    assert "there is no BROWSE mode" in tool.description


def test_picker_unavailable_is_a_documented_status_in_both_definitions():
    """The model was handed a status the description never mentioned, carrying
    a remedy in prose. Both variants must name it."""
    from anton.core.tools.tool_defs import SELECT_PATH_TOOL, SELECT_PATH_TOOL_PICK_ONLY

    assert "picker_unavailable" in SELECT_PATH_TOOL.description
    assert "picker_unavailable" in SELECT_PATH_TOOL_PICK_ONLY.description
    # The full definition must say what to do in each mode.
    assert "attach" in SELECT_PATH_TOOL.description.lower()


def test_module_singletons_are_not_mutated(make_session):
    """ToolDefs are module-level singletons shared across every session in the
    process — the pick-only variant must be a copy, never an in-place edit."""
    from anton.core.tools.tool_defs import SELECT_PATH_TOOL, SELECT_PATH_TOOL_PICK_ONLY

    pristine_full = SELECT_PATH_TOOL.description
    pristine_pick = SELECT_PATH_TOOL_PICK_ONLY.description
    _registered_select_path(make_session())
    _registered_select_path(make_session(elicitor=_FakeElicitor(None)))
    assert SELECT_PATH_TOOL.description == pristine_full
    assert SELECT_PATH_TOOL_PICK_ONLY.description == pristine_pick
    assert SELECT_PATH_TOOL is not SELECT_PATH_TOOL_PICK_ONLY
    # input_schema is a module-level dict shared by every session in the
    # process: the pick-only variant must rebuild it, never pop from it.
    assert "start_dir" in SELECT_PATH_TOOL.input_schema["properties"]
    assert (
        SELECT_PATH_TOOL.input_schema["properties"]
        is not SELECT_PATH_TOOL_PICK_ONLY.input_schema["properties"]
    )
