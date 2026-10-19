"""With `write_approval` on, a cell that changes connected data, or deletes or
renames a file the project held when the turn began, waits for the user's
Approve, and anything else refuses it.

Before this, such a cell ran in the same turn with no question: asked to move
a supplier's lead time, the agent ran the UPDATE 6 times in 6; asked to rename
a PDF after "Do not change any data.", it renamed it 8 times in 10. Only the
model's own choice to ask stood in the way.

These drive the real `turn_stream` with a scripted model, a fake pad that
records what it runs, and an elicitor that answers only once the test has seen
the card arrive in the turn's stream. So a gate whose card sits undelivered
until it times out, which is what an ask from the inline exec without draining
does, hangs here and fails on the timeout instead of passing late.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.core.backends.base import Cell
from anton.core.interaction.elicit import AskAnswer
from anton.core.interaction.emitter import TurnEmitter
from anton.core.llm.provider import (
    LLMResponse,
    StreamAskUser,
    StreamComplete,
    ToolCall,
    Usage,
)
from anton.core.session import ChatSession, ChatSessionConfig, _VerifierVerdict
from anton.core.settings import CoreSettings
from anton.core.tools.generate_artifact.discovery.engine import _exec_ran
from anton.core.tools.registry import ToolOutcome
from tests.conftest import make_mock_llm

UPDATE_CELL = (
    "import os, psycopg2\n"
    "conn = psycopg2.connect(host=os.environ['DS_POSTGRES_QA__HOST'])\n"
    "cur = conn.cursor()\n"
    'cur.execute("""\n'
    "    UPDATE suppliers\n"
    "    SET lead_time_days = %s\n"
    "    WHERE supplier_id = %s\n"
    '""", (21, "S005"))\n'
    "conn.commit()\n"
)
RENAME_CELL = "import os\nos.rename('report.pdf', 'old.pdf')\n"

APPROVE = AskAnswer(status="answered", values=("approve",))
CANCEL = AskAnswer(status="answered", values=("cancel",))
SKIP = AskAnswer(status="cancelled")
TIMEOUT = AskAnswer(status="timeout")

# Long enough for a slow CI box; a gate that never delivers its card hangs
# until this, and the test then fails rather than passing late.
DEADLINE_S = 10


class _Pad:
    """A scratchpad runtime with no subprocess. Records each cell it runs."""

    def __init__(self) -> None:
        self.ran: list[str] = []
        self.cells: list[Cell] = []

    async def execute_streaming(self, code, **kwargs):
        self.ran.append(code)
        cell = Cell(code=code, stdout="done", stderr="", error=None)
        self.cells.append(cell)
        yield cell

    async def cancel(self) -> None:
        pass


class _GatedElicitor:
    """Answers each question only after the test releases it, and records the
    requests. The test releases an answer once it has seen the card."""

    supported_kinds = ("choice",)
    answer_hint = "hint"
    timeout_s = 300

    def __init__(self) -> None:
        self.requests = []
        self.question_ids: list[str] = []
        self._answers: asyncio.Queue[AskAnswer] = asyncio.Queue()

    def release(self, answer: AskAnswer) -> None:
        self._answers.put_nowait(answer)

    async def begin(self, question_id, request):
        pass

    async def ask(self, question_id, request):
        self.requests.append(request)
        self.question_ids.append(question_id)
        return await self._answers.get()

    async def end(self, question_id):
        pass


def _llm(*cells: str) -> AsyncMock:
    """A model that runs each of `cells` as one exec, one round each, then answers."""
    llm = make_mock_llm()
    llm.generate_object_code = AsyncMock(
        return_value=_VerifierVerdict(status="COMPLETE", reason="done")
    )
    script = list(cells)
    calls = 0

    def plan_stream(**kwargs):
        nonlocal calls
        calls += 1
        if script:
            code = script.pop(0)
            response = LLMResponse(
                content="Working.",
                tool_calls=[ToolCall(
                    id=f"tc_{calls}", name="scratchpad",
                    input={"action": "exec", "name": "main", "code": code,
                           "one_line_description": "step"},
                )],
                usage=Usage(input_tokens=10, output_tokens=20), stop_reason="tool_use",
            )
        else:
            response = LLMResponse(
                content="Done.", tool_calls=[],
                usage=Usage(input_tokens=10, output_tokens=20), stop_reason="end_turn",
            )

        async def gen():
            yield StreamComplete(response=response)

        return gen()

    llm.plan_stream = plan_stream
    return llm


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "report.pdf").write_text("terms")
    (tmp_path / "stock.csv").write_text("site,qty\nA,1\n")
    return tmp_path


def _session(project: Path, *cells: str, write_approval: bool = True) -> tuple[ChatSession, _Pad, _GatedElicitor]:
    workspace = MagicMock(base=project)
    workspace.artifacts_dir = project / "artifacts"
    session = ChatSession(ChatSessionConfig(
        llm_client=_llm(*cells),
        workspace=workspace,
        settings=CoreSettings(write_approval=write_approval),
    ))
    pad = _Pad()
    session._scratchpads.get_or_create = AsyncMock(return_value=pad)
    elicitor = _GatedElicitor()
    session.elicitor = elicitor
    return session, pad, elicitor


async def _run(session: ChatSession, pad: _Pad, elicitor: _GatedElicitor, answers: list[AskAnswer]):
    """Drive one turn. On each card, check nothing has run yet for it, then
    release the next answer."""
    seen: list[StreamAskUser] = []
    ran_before_answer: list[int] = []

    async def drive():
        async for event in session.turn_stream("go"):
            if isinstance(event, StreamAskUser):
                seen.append(event)
                ran_before_answer.append(len(pad.ran))
                elicitor.release(answers.pop(0) if answers else TIMEOUT)

    try:
        with patch("anton.analytics.send_event"):
            await asyncio.wait_for(drive(), DEADLINE_S)
    finally:
        await session.close()
    return seen, ran_before_answer


def _history_text(session: ChatSession) -> str:
    out = []
    for message in session._history:
        content = message.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    out.append(str(block.get("content") or block.get("text") or ""))
    return "\n".join(out)


# ── Connected data ────────────────────────────────────────────────────────

async def test_an_update_waits_for_approve_and_then_runs_once(project):
    session, pad, elicitor = _session(project, UPDATE_CELL)
    seen, ran_before = await _run(session, pad, elicitor, [APPROVE])
    assert len(seen) == 1, "the card reaches the host while the cell waits"
    assert ran_before == [0], "nothing ran before the user answered"
    assert pad.ran == [UPDATE_CELL], "it runs exactly once after Approve"
    prompt = seen[0].request.prompt
    quoted = "UPDATE suppliers\nSET lead_time_days = %s\nWHERE supplier_id = %s\nwith values (21, 'S005')"
    assert f"```\n{quoted}\n```" in prompt, "the card quotes the statement line by line, with its values"
    assert seen[0].id.startswith("write:")
    assert seen[0].request.allow_custom is False
    assert {option.style for option in seen[0].request.options} == {"default"}, "no host renders a style"


@pytest.mark.parametrize("answer", [CANCEL, SKIP, TIMEOUT], ids=["cancel", "skip", "timeout"])
async def test_an_update_without_approve_never_runs(project, answer):
    session, pad, elicitor = _session(project, UPDATE_CELL)
    seen, _ = await _run(session, pad, elicitor, [answer])
    assert len(seen) == 1
    assert pad.ran == [], "nothing is written"
    history = _history_text(session)
    assert "The user didn't approve this change, so it must not be made another way" in history
    assert "try one or two more genuine workarounds" not in history


# ── Project files ─────────────────────────────────────────────────────────

async def test_a_rename_of_an_existing_file_waits_and_names_it(project):
    session, pad, elicitor = _session(project, RENAME_CELL)
    seen, ran_before = await _run(session, pad, elicitor, [APPROVE])
    assert len(seen) == 1 and ran_before == [0]
    assert "This step would rename:\n\n```\nreport.pdf\n```" in seen[0].request.prompt
    assert pad.ran == [RENAME_CELL]


@pytest.mark.parametrize("answer", [CANCEL, TIMEOUT], ids=["cancel", "timeout"])
async def test_a_rename_without_approve_never_runs(project, answer):
    session, pad, elicitor = _session(project, RENAME_CELL)
    await _run(session, pad, elicitor, [answer])
    assert pad.ran == []
    assert (project / "report.pdf").exists()


async def test_after_a_timeout_later_cells_in_the_turn_are_refused_without_a_card(project):
    session, pad, elicitor = _session(project, UPDATE_CELL, RENAME_CELL)
    seen, _ = await _run(session, pad, elicitor, [TIMEOUT])
    assert len(seen) == 1, "one unanswered card, not one per cell"
    assert pad.ran == []
    assert "an earlier approval request in this turn went unanswered" in _history_text(session)


@pytest.mark.parametrize("code", [
    "import os, psycopg2\nconn = psycopg2.connect(host=os.environ['DS_POSTGRES_QA__HOST'])\n"
    "conn.cursor().execute('SELECT name, lead_time_days FROM suppliers ORDER BY 2 DESC')\n",
    "import duckdb\ncon = duckdb.connect()\n"
    "con.execute(\"CREATE TABLE stock AS SELECT * FROM read_csv('stock.csv')\")\n",
    "open('summary.md', 'w').write('One line.')\n",
    "import os\nopen('summary.md.tmp', 'w').write('x')\nos.replace('summary.md.tmp', 'summary.md')\n",
], ids=["select", "duckdb-from-csv", "new-file", "temp-renamed-into-place"])
async def test_read_only_work_and_new_files_raise_no_card(project, code):
    session, pad, elicitor = _session(project, code)
    seen, _ = await _run(session, pad, elicitor, [])
    assert seen == []
    assert pad.ran == [code]


# ── Off ───────────────────────────────────────────────────────────────────

async def test_with_the_setting_off_the_same_cells_run_with_no_card(project):
    session, pad, elicitor = _session(project, UPDATE_CELL, RENAME_CELL, write_approval=False)
    assert getattr(session, "_write_gate", None) is None
    seen, _ = await _run(session, pad, elicitor, [])
    assert seen == []
    assert pad.ran == [UPDATE_CELL, RENAME_CELL]


# ── The hold on its own ───────────────────────────────────────────────────
#
# Imported here, not at the top, so the turn tests above still run, and fail
# on what they check, against an anton that has no gate.

def _hold():
    from anton.core.utils.scratchpad import hold_unapproved_write

    return hold_unapproved_write

async def _gated_session(project: Path, *, elicitor=None, emitter: bool = True) -> ChatSession:
    session, _pad, default = _session(project)
    session.elicitor = elicitor if elicitor is not None else default
    session.emitter = TurnEmitter() if emitter else None
    await session._write_gate.begin_turn(manager=session._scratchpads)
    return session


class _AnswersAtOnce:
    supported_kinds = ("choice",)
    answer_hint = "hint"
    timeout_s = 300

    def __init__(self, answer=None, *, raises: Exception | None = None) -> None:
        self.answer = answer
        self.raises = raises
        self.asked = 0
        self.prompts: list[str] = []

    async def begin(self, question_id, request):
        pass

    async def ask(self, question_id, request):
        self.asked += 1
        self.prompts.append(request.prompt)
        if self.raises is not None:
            raise self.raises
        return self.answer

    async def end(self, question_id):
        pass


async def test_the_question_limit_refuses_with_no_card(project):
    elicitor = _AnswersAtOnce(APPROVE)
    session = await _gated_session(project, elicitor=elicitor)
    session.question_count = 8
    outcome = await _hold()(session, pad=_Pad(), code=UPDATE_CELL)
    assert isinstance(outcome, ToolOutcome) and outcome.reason == "write_not_approved"
    assert outcome.ok is True, "a refusal is no tool failure: it feeds no error streak"
    assert "asked all the questions it may" in outcome.content
    assert elicitor.asked == 0


async def test_no_way_to_ask_refuses(project):
    session = await _gated_session(project, emitter=False)
    outcome = await _hold()(session, pad=_Pad(), code=RENAME_CELL)
    assert outcome is not None and "there is no way to ask them here" in outcome.content
    session.elicitor = None
    session.emitter = TurnEmitter()
    outcome = await _hold()(session, pad=_Pad(), code=RENAME_CELL)
    assert outcome is not None and outcome.reason == "write_not_approved"


async def test_a_failed_question_refuses_like_a_cancel(project):
    session = await _gated_session(project, elicitor=_AnswersAtOnce(raises=RuntimeError("broker down")))
    outcome = await _hold()(session, pad=_Pad(), code=UPDATE_CELL)
    assert outcome is not None and outcome.ok is True
    assert "the approval request failed" in outcome.content


async def test_a_typed_answer_is_no_approval(project):
    session = await _gated_session(project, elicitor=_AnswersAtOnce(AskAnswer(status="answered", text="yes")))
    outcome = await _hold()(session, pad=_Pad(), code=UPDATE_CELL)
    assert outcome is not None and "they declined it" in outcome.content


async def test_a_check_that_fails_asks_instead_of_clearing_the_cell(project, monkeypatch):
    def boom(**kwargs):
        raise RecursionError("too deep")

    from anton.core.utils import scratchpad as scratchpad_mod

    monkeypatch.setattr(scratchpad_mod, "find_cell_writes", boom)
    elicitor = _AnswersAtOnce(TIMEOUT)
    session = await _gated_session(project, elicitor=elicitor)
    outcome = await _hold()(session, pad=_Pad(), code="print('hello')")
    assert elicitor.asked == 1
    assert outcome is not None and "the check couldn't read" in outcome.content


async def test_each_decision_is_logged_without_the_code(project, caplog):
    session = await _gated_session(project, elicitor=_AnswersAtOnce(CANCEL))
    with caplog.at_level(logging.WARNING, logger="anton.core.utils.scratchpad"):
        await _hold()(session, pad=_Pad(), code=UPDATE_CELL)
    records = [r for r in caplog.records
               if r.name == "anton.core.utils.scratchpad" and r.levelno == logging.WARNING]
    assert len(records) == 1
    message = records[0].getMessage()
    assert '"status":"cancelled"' in message and '"kinds":["sql"]' in message
    assert "suppliers" not in message, "the log line holds kinds and counts, never the statement"


async def test_an_earlier_cells_connection_still_counts(project):
    """The DS_ read was in an earlier cell; this one only writes."""
    session = await _gated_session(project, elicitor=_AnswersAtOnce(CANCEL))
    pad = _Pad()
    pad.cells.append(Cell(
        code="import os, psycopg2\nconn = psycopg2.connect(host=os.environ['DS_POSTGRES_QA__HOST'])",
        stdout="", stderr="", error=None,
    ))
    outcome = await _hold()(
        session, pad=pad, code="conn.cursor().execute(\"DELETE FROM suppliers WHERE id = 5\")",
    )
    assert outcome is not None and outcome.reason == "write_not_approved"


async def test_a_session_with_no_saved_connection_still_holds_a_database_write(project):
    """No vault connection and no DS_ name: the write still waits."""
    session = await _gated_session(project, elicitor=_AnswersAtOnce(CANCEL))
    code = "import psycopg2\nconn = psycopg2.connect(dsn)\nconn.cursor().execute('UPDATE t SET a = 1')"
    outcome = await _hold()(session, pad=_Pad(), code=code)
    assert outcome is not None and outcome.reason == "write_not_approved"
    local = "import duckdb\ncon = duckdb.connect()\ncon.execute(\"CREATE TABLE stock AS SELECT * FROM read_csv('stock.csv')\")"
    assert await _hold()(session, pad=_Pad(), code=local) is None


async def test_the_shared_folder_is_held_too(project, tmp_path_factory):
    from anton.core.utils.write_check import snapshot_project_files

    shared = tmp_path_factory.mktemp("shared")
    (shared / "forecast.xlsx").write_text("x")
    session = await _gated_session(project, elicitor=_AnswersAtOnce(CANCEL))
    session._write_gate.snapshots += (
        snapshot_project_files(root=shared, label="shared", taken_at=10**10),
    )
    outcome = await _hold()(
        session, pad=_Pad(), code=f"import os\nos.remove({str(shared / 'forecast.xlsx')!r})",
    )
    assert outcome is not None and "shared folder: forecast.xlsx" in outcome.content


@pytest.mark.parametrize("code", [
    "import shutil\nshutil.rmtree({root!r})",
    "import subprocess\nsubprocess.run(['rm', '-rf', {root!r}])",
], ids=["rmtree", "rm-rf"])
async def test_the_whole_shared_folder_is_named_on_the_card(project, tmp_path_factory, code):
    from anton.core.utils.write_check import snapshot_project_files

    shared = tmp_path_factory.mktemp("shared")
    (shared / "forecast.xlsx").write_text("x")
    elicitor = _AnswersAtOnce(CANCEL)
    session = await _gated_session(project, elicitor=elicitor)
    session._write_gate.snapshots += (
        snapshot_project_files(root=shared, label="shared", taken_at=10**10),
    )
    outcome = await _hold()(session, pad=_Pad(), code=code.format(root=str(shared)))
    assert outcome is not None and outcome.reason == "write_not_approved"
    assert elicitor.prompts and "This step would delete the whole shared folder." in elicitor.prompts[0]
    assert (shared / "forecast.xlsx").exists()


def test_a_refused_cell_is_not_working_code_for_the_artifact_pipeline():
    refused = ToolOutcome(content="Not run", ok=True, reason="write_not_approved")
    assert _exec_ran(refused) is False
