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
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.core.backends.base import Cell
from anton.core.backends.local import LocalScratchpadRuntime
from anton.core.backends.sandbox import CellSandbox
from anton.core.interaction.elicit import AskAnswer, AskRequest
from anton.core.interaction.emitter import TurnEmitter
from anton.core.llm.provider import (
    LLMResponse,
    StreamAskUser,
    StreamComplete,
    StreamEvent,
    StreamTextDelta,
    StreamToolResult,
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

GATE_LOG = "anton.core.utils.scratchpad"

# Long enough for a slow CI box; a gate that never delivers its card hangs
# until this, and the test then fails rather than passing late.
DEADLINE_S = 10


class _Pad:
    """A scratchpad runtime with no subprocess. Records each cell it runs and
    each package install it is asked for."""

    def __init__(self) -> None:
        self.ran: list[str] = []
        self.installed: list[list[str]] = []
        self.cells: list[Cell] = []

    async def install_packages(self, packages: list[str]) -> str:
        self.installed.append(list(packages))
        return "All packages already installed."

    async def execute_streaming(self, code: str, **kwargs: object):
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
        self.requests: list[AskRequest] = []
        self.question_ids: list[str] = []
        self._answers: asyncio.Queue[AskAnswer] = asyncio.Queue()

    def release(self, answer: AskAnswer) -> None:
        self._answers.put_nowait(answer)

    async def begin(self, question_id: str, request: AskRequest) -> None:
        pass

    async def ask(self, question_id: str, request: AskRequest) -> AskAnswer:
        self.requests.append(request)
        self.question_ids.append(question_id)
        return await self._answers.get()

    async def end(self, question_id: str) -> None:
        pass


class _AnswersAtOnce:
    """Answers every question with `answer` as soon as it is asked, or raises `raises`."""

    supported_kinds = ("choice",)
    answer_hint = "hint"
    timeout_s = 300

    def __init__(self, *, answer: AskAnswer | None = None, raises: Exception | None = None) -> None:
        self.answer = answer
        self.raises = raises
        self.asked = 0
        self.prompts: list[str] = []

    async def begin(self, question_id: str, request: AskRequest) -> None:
        pass

    async def ask(self, question_id: str, request: AskRequest) -> AskAnswer:
        self.asked += 1
        self.prompts.append(request.prompt)
        if self.raises is not None:
            raise self.raises
        return self.answer

    async def end(self, question_id: str) -> None:
        pass


def _llm(*cells: str, packages: tuple[str, ...] = ()) -> AsyncMock:
    """A model that runs each of `cells` as one exec, one round each, then
    answers. Each exec declares `packages`."""
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
                           "one_line_description": "step",
                           **({"packages": list(packages)} if packages else {})},
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


@dataclass
class _Harness:
    """A session whose cells run on `pad` and whose cards `elicitor` answers."""

    session: ChatSession
    pad: _Pad
    elicitor: _GatedElicitor


@dataclass
class _TurnSeen:
    """What one turn streamed: every event, the cards among them, and how many
    cells had run when each card arrived."""

    events: list[StreamEvent] = field(default_factory=list)
    cards: list[StreamAskUser] = field(default_factory=list)
    ran_before: list[int] = field(default_factory=list)


def _session(
    project: Path, *cells: str, write_approval: bool = True, packages: tuple[str, ...] = (),
) -> _Harness:
    workspace = MagicMock(base=project)
    workspace.artifacts_dir = project / "artifacts"
    session = ChatSession(ChatSessionConfig(
        llm_client=_llm(*cells, packages=packages),
        workspace=workspace,
        settings=CoreSettings(write_approval=write_approval),
    ))
    pad = _Pad()
    session._scratchpads.get_or_create = AsyncMock(return_value=pad)
    elicitor = _GatedElicitor()
    session.elicitor = elicitor
    return _Harness(session=session, pad=pad, elicitor=elicitor)


async def _turn(*, harness: _Harness, answers: list[AskAnswer]) -> _TurnSeen:
    """Drive one turn. On each card, note how many cells have run, then
    release the next answer, or a timeout once `answers` runs out."""
    seen = _TurnSeen()

    async def drive() -> None:
        async for event in harness.session.turn_stream("go"):
            seen.events.append(event)
            if isinstance(event, StreamAskUser):
                seen.cards.append(event)
                seen.ran_before.append(len(harness.pad.ran))
                harness.elicitor.release(answers.pop(0) if answers else TIMEOUT)

    with patch("anton.analytics.send_event"):
        await asyncio.wait_for(drive(), DEADLINE_S)
    return seen


async def _run(*, harness: _Harness, answers: list[AskAnswer]) -> _TurnSeen:
    """Drive one turn, then close the session."""
    try:
        return await _turn(harness=harness, answers=answers)
    finally:
        await harness.session.close()


def _history_text(session: ChatSession) -> str:
    # `ChatSession._history` holds raw message dicts; there is no message model to read instead.
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


def _decisions(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Each logged write approval decision, as the JSON it logged."""
    prefix = "write approval: {"
    return [
        r.getMessage().removeprefix("write approval: ") for r in caplog.records
        if r.name == GATE_LOG and r.levelno == logging.WARNING and r.getMessage().startswith(prefix)
    ]


def _spy_on_draining(session: ChatSession) -> list[str]:
    """The name of each coroutine the session runs through `_run_draining`, as it runs them."""
    names: list[str] = []
    original = session._run_draining

    def spy(coro):
        names.append(coro.__name__)
        return original(coro)

    session._run_draining = spy
    return names


# ── Connected data ────────────────────────────────────────────────────────

async def test_an_update_waits_for_approve_and_then_runs_once(project):
    harness = _session(project, UPDATE_CELL)
    drained = _spy_on_draining(harness.session)
    seen = await _run(harness=harness, answers=[APPROVE])
    assert len(seen.cards) == 1, "the card reaches the host while the cell waits"
    assert seen.ran_before == [0], "nothing ran before the user answered"
    assert harness.pad.ran == [UPDATE_CELL], "it runs exactly once after Approve"
    assert drained == ["prepare_scratchpad_exec"], "the gated exec is drained so its card reaches the host"
    prompt = seen.cards[0].request.prompt
    quoted = "UPDATE suppliers\nSET lead_time_days = %s\nWHERE supplier_id = %s\nwith values (21, 'S005')"
    assert f"```\n{quoted}\n```" in prompt, "the card quotes the statement line by line, with its values"
    assert seen.cards[0].id.startswith("write:")
    assert seen.cards[0].request.allow_custom is False
    assert {option.style for option in seen.cards[0].request.options} == {"default"}, "no host renders a style"


@pytest.mark.parametrize("answer", [CANCEL, SKIP, TIMEOUT], ids=["cancel", "skip", "timeout"])
async def test_an_update_without_approve_never_runs(project, answer):
    harness = _session(project, UPDATE_CELL)
    seen = await _run(harness=harness, answers=[answer])
    assert len(seen.cards) == 1
    assert harness.pad.ran == [], "nothing is written"
    history = _history_text(harness.session)
    assert "The user didn't approve this change, so it must not be made another way" in history
    assert "try one or two more genuine workarounds" not in history


async def test_a_refused_cell_closes_its_step_with_a_result_the_replay_skips(project):
    """The host draws a step from the exec call before the card shows. The
    refusal is that step's result, so it doesn't read as a finished run. It
    is plain text, so cowork-server's cell replay, which rebuilds a cell only
    from a result's JSON, never reads it as one that ran."""
    harness = _session(project, UPDATE_CELL)
    seen = await _run(harness=harness, answers=[CANCEL])
    results = [e for e in seen.events if isinstance(e, StreamToolResult)]
    assert [(r.name, r.action, r.id) for r in results] == [("scratchpad", "exec", "tc_1")]
    assert results[0].content.startswith("Not run: this step would change data")
    with pytest.raises(json.JSONDecodeError):
        json.loads(results[0].content)


async def test_a_refused_cell_installs_no_packages(project):
    harness = _session(project, UPDATE_CELL, packages=("tabulate",))
    seen = await _run(harness=harness, answers=[CANCEL])
    assert len(seen.cards) == 1
    assert harness.pad.ran == []
    assert harness.pad.installed == [], "the hold comes before the installs"


async def test_a_refused_write_is_listed_as_not_run_when_anton_writes_the_handback(project, monkeypatch):
    """The check fails and the content filter stops both hand-back tries, so
    anton writes the report itself. The write the user cancelled is listed
    as not run, not as a step it did."""
    from tests.test_spend_ceiling import _REFUSAL

    monkeypatch.setenv("ANTON_VERIFY_MIN_TOOL_ROUNDS", "1")
    harness = _session(project)
    llm = harness.session._llm
    llm.generate_object_code = AsyncMock(side_effect=RuntimeError("provider hiccup"))
    update = ToolCall(id="tc_1", name="scratchpad", input={
        "action": "exec", "name": "main", "code": UPDATE_CELL,
        "one_line_description": "Set supplier S005 lead time to 21 days",
    })
    usage = Usage(input_tokens=10, output_tokens=20)
    replies = [
        [StreamComplete(response=LLMResponse(content="", tool_calls=[update], usage=usage, stop_reason="tool_use"))],
        [StreamComplete(response=LLMResponse(content="", tool_calls=[], usage=usage, stop_reason="end_turn"))],
        *[[StreamTextDelta(text=_REFUSAL),
           StreamComplete(response=LLMResponse(content=_REFUSAL, usage=usage, stop_reason="content_filter"))]] * 2,
    ]

    def plan_stream(**kwargs):
        events = replies.pop(0)

        async def gen():
            for event in events:
                yield event

        return gen()

    llm.plan_stream = plan_stream
    seen = await _run(harness=harness, answers=[CANCEL])
    shown = "".join(e.text for e in seen.events if isinstance(e, StreamTextDelta))
    assert harness.pad.ran == []
    assert shown == (
        "The model provider's content filter stopped my summary of this request. "
        "This is what I did before I stopped:\n\n"
        "- Set supplier S005 lead time to 21 days (not run)\n\n"
        'Reply "keep going" to continue from where I stopped.'
    )


# ── Project files ─────────────────────────────────────────────────────────

async def test_a_rename_of_an_existing_file_waits_and_names_it(project):
    harness = _session(project, RENAME_CELL)
    seen = await _run(harness=harness, answers=[APPROVE])
    assert len(seen.cards) == 1 and seen.ran_before == [0]
    assert "This step would rename:\n\n```\nreport.pdf\n```" in seen.cards[0].request.prompt
    assert harness.pad.ran == [RENAME_CELL]


@pytest.mark.parametrize("answer", [CANCEL, TIMEOUT], ids=["cancel", "timeout"])
async def test_a_rename_without_approve_never_runs(project, answer):
    harness = _session(project, RENAME_CELL)
    await _run(harness=harness, answers=[answer])
    assert harness.pad.ran == []


# ── One unanswered card ───────────────────────────────────────────────────

async def test_after_a_timeout_later_cells_in_the_turn_are_refused_without_a_card(project):
    harness = _session(project, UPDATE_CELL, RENAME_CELL)
    seen = await _run(harness=harness, answers=[TIMEOUT])
    assert len(seen.cards) == 1, "one unanswered card, not one per cell"
    assert harness.pad.ran == []
    assert "an earlier approval request in this turn went unanswered" in _history_text(harness.session)


async def test_the_next_turn_after_a_timeout_asks_again(project):
    """The CLI keeps one session across turns: an unanswered card refuses
    later cells in its own turn only."""
    harness = _session(project, UPDATE_CELL)
    try:
        first = await _turn(harness=harness, answers=[TIMEOUT])
        harness.session._llm.plan_stream = _llm(UPDATE_CELL).plan_stream
        second = await _turn(harness=harness, answers=[APPROVE])
    finally:
        await harness.session.close()
    assert (len(first.cards), len(second.cards)) == (1, 1)
    assert harness.pad.ran == [UPDATE_CELL]


async def test_a_cancel_leaves_later_cells_their_own_card(project):
    harness = _session(project, UPDATE_CELL, RENAME_CELL)
    seen = await _run(harness=harness, answers=[CANCEL, APPROVE])
    assert len(seen.cards) == 2, "a Cancel is an answer, so the next flagged cell still asks"
    assert harness.pad.ran == [RENAME_CELL]


# ── Stop ──────────────────────────────────────────────────────────────────

async def test_stop_while_the_card_waits_runs_nothing_and_logs_stopped(project, caplog):
    """The host ends the turn (Stop, a delete, shutdown) while the card waits:
    anton sees a cancellation, not an answer."""
    harness = _session(project, UPDATE_CELL)
    card = asyncio.Event()

    async def drive() -> None:
        async for event in harness.session.turn_stream("go"):
            if isinstance(event, StreamAskUser):
                card.set()

    with caplog.at_level(logging.WARNING, logger=GATE_LOG), patch("anton.analytics.send_event"):
        task = asyncio.create_task(drive())
        await asyncio.wait_for(card.wait(), DEADLINE_S)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    await harness.session.close()
    assert harness.pad.ran == []
    assert _decisions(caplog) == ['{"kinds":["sql"],"count":1,"status":"stopped"}']


# ── What raises no card ───────────────────────────────────────────────────

@pytest.mark.parametrize("code", [
    "import os, psycopg2\nconn = psycopg2.connect(host=os.environ['DS_POSTGRES_QA__HOST'])\n"
    "conn.cursor().execute('SELECT name, lead_time_days FROM suppliers ORDER BY 2 DESC')\n",
    "import duckdb\ncon = duckdb.connect()\n"
    "con.execute(\"CREATE TABLE stock AS SELECT * FROM read_csv('stock.csv')\")\n",
    "open('summary.md', 'w').write('One line.')\n",
    "import os\nopen('summary.md.tmp', 'w').write('x')\nos.replace('summary.md.tmp', 'summary.md')\n",
], ids=["select", "duckdb-from-csv", "new-file", "temp-renamed-into-place"])
async def test_read_only_work_and_new_files_raise_no_card(project, code):
    harness = _session(project, code)
    seen = await _run(harness=harness, answers=[])
    assert seen.cards == []
    assert harness.pad.ran == [code]


# ── Off ───────────────────────────────────────────────────────────────────

async def test_with_the_setting_off_the_same_cells_run_with_no_card(project):
    harness = _session(project, UPDATE_CELL, RENAME_CELL, write_approval=False)
    assert getattr(harness.session, "_write_gate", None) is None
    drained = _spy_on_draining(harness.session)
    seen = await _run(harness=harness, answers=[])
    assert seen.cards == []
    assert harness.pad.ran == [UPDATE_CELL, RENAME_CELL]
    assert "prepare_scratchpad_exec" not in drained, "the inline exec keeps its plain await"


# ── Approval cards and the turn's questions ───────────────────────────────

async def test_approval_cards_spend_none_of_the_turn_s_questions(project):
    """Nine approved writes, one more than a turn's questions: each gets its
    card and runs, and the model still has all of its own questions."""
    harness = _session(project, *[UPDATE_CELL] * 9)
    seen = await _run(harness=harness, answers=[APPROVE] * 9)
    assert len(seen.cards) == 9
    assert harness.pad.ran == [UPDATE_CELL] * 9
    assert harness.session.question_count == 0


# ── The hold on its own ───────────────────────────────────────────────────
#
# Imported here, not at the top, so the turn tests above still run, and fail
# on what they check, against an anton that has no gate.

def _hold():
    from anton.core.utils.scratchpad import hold_unapproved_write

    return hold_unapproved_write


async def _gated_session(
    project: Path, *, elicitor: _AnswersAtOnce | None = None, emitter: bool = True,
) -> ChatSession:
    harness = _session(project)
    session = harness.session
    if elicitor is not None:
        session.elicitor = elicitor
    session.emitter = TurnEmitter() if emitter else None
    await session._write_gate.begin_turn(manager=session._scratchpads)
    return session


async def test_a_full_question_budget_still_shows_the_card(project):
    elicitor = _AnswersAtOnce(answer=APPROVE)
    session = await _gated_session(project, elicitor=elicitor)
    session.question_count = 8
    outcome = await _hold()(session, pad=_Pad(), code=UPDATE_CELL)
    assert outcome is None, "Approve runs the cell"
    assert elicitor.asked == 1
    assert session.question_count == 8, "the card spends none of the turn's questions"


async def test_a_refusal_is_no_tool_failure(project):
    session = await _gated_session(project, elicitor=_AnswersAtOnce(answer=CANCEL))
    outcome = await _hold()(session, pad=_Pad(), code=UPDATE_CELL)
    assert isinstance(outcome, ToolOutcome) and outcome.reason == "write_not_approved"
    assert outcome.ok is True, "a refusal is no tool failure: it feeds no error streak"


async def test_no_way_to_ask_refuses(project):
    session = await _gated_session(project, emitter=False)
    outcome = await _hold()(session, pad=_Pad(), code=RENAME_CELL)
    assert outcome is not None and "there is no way to ask them here" in outcome.content
    session.elicitor = None
    session.emitter = TurnEmitter()
    outcome = await _hold()(session, pad=_Pad(), code=RENAME_CELL)
    assert outcome is not None and outcome.reason == "write_not_approved"


async def test_a_failed_question_refuses_like_a_cancel_and_logs_no_traceback(project, caplog):
    session = await _gated_session(project, elicitor=_AnswersAtOnce(raises=RuntimeError("broker down")))
    with caplog.at_level(logging.WARNING, logger=GATE_LOG):
        outcome = await _hold()(session, pad=_Pad(), code=UPDATE_CELL)
    assert outcome is not None and outcome.ok is True
    assert "the approval request failed" in outcome.content
    failed = [r for r in caplog.records if r.name == GATE_LOG and "the question failed" in r.getMessage()]
    assert [r.getMessage() for r in failed] == ["write approval: the question failed (RuntimeError)"]
    assert failed[0].exc_info is None, "a traceback's frames would print the card and its statement"


async def test_a_typed_answer_is_no_approval(project):
    session = await _gated_session(
        project, elicitor=_AnswersAtOnce(answer=AskAnswer(status="answered", text="yes")),
    )
    outcome = await _hold()(session, pad=_Pad(), code=UPDATE_CELL)
    assert outcome is not None and "they declined it" in outcome.content


async def test_a_check_that_fails_asks_instead_of_clearing_the_cell(project, monkeypatch, caplog):
    def boom(**kwargs):
        raise RecursionError("too deep")

    from anton.core.utils import scratchpad as scratchpad_mod

    monkeypatch.setattr(scratchpad_mod, "find_cell_writes", boom)
    elicitor = _AnswersAtOnce(answer=TIMEOUT)
    session = await _gated_session(project, elicitor=elicitor)
    with caplog.at_level(logging.WARNING, logger=GATE_LOG):
        outcome = await _hold()(session, pad=_Pad(), code="print('hello')")
    assert elicitor.asked == 1
    assert outcome is not None and "the check couldn't read" in outcome.content
    failed = [r for r in caplog.records if r.name == GATE_LOG and "the check failed" in r.getMessage()]
    assert [r.getMessage() for r in failed] == ["write approval: the check failed (RecursionError)"]
    assert failed[0].exc_info is None, "a traceback's frames would print the cell's code"


async def test_each_decision_is_logged_without_the_code(project, caplog):
    session = await _gated_session(project, elicitor=_AnswersAtOnce(answer=CANCEL))
    with caplog.at_level(logging.WARNING, logger=GATE_LOG):
        await _hold()(session, pad=_Pad(), code=UPDATE_CELL)
    assert _decisions(caplog) == ['{"kinds":["sql"],"count":1,"status":"cancelled"}']
    assert "suppliers" not in caplog.text, "the log line holds kinds and counts, never the statement"


async def test_an_overwrite_runs_with_no_card_and_is_logged_not_held(project, caplog):
    elicitor = _AnswersAtOnce(answer=CANCEL)
    session = await _gated_session(project, elicitor=elicitor)
    with caplog.at_level(logging.WARNING, logger=GATE_LOG):
        outcome = await _hold()(session, pad=_Pad(), code="open('report.pdf', 'w').write('x')")
    assert outcome is None and elicitor.asked == 0
    assert _decisions(caplog) == ['{"kinds":["overwrite"],"count":1,"status":"not_held"}']


async def test_an_earlier_cells_connection_still_counts(project):
    """The DS_ read was in an earlier cell; this one only writes."""
    session = await _gated_session(project, elicitor=_AnswersAtOnce(answer=CANCEL))
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
    session = await _gated_session(project, elicitor=_AnswersAtOnce(answer=CANCEL))
    code = "import psycopg2\nconn = psycopg2.connect(dsn)\nconn.cursor().execute('UPDATE t SET a = 1')"
    outcome = await _hold()(session, pad=_Pad(), code=code)
    assert outcome is not None and outcome.reason == "write_not_approved"
    local = "import duckdb\ncon = duckdb.connect()\ncon.execute(\"CREATE TABLE stock AS SELECT * FROM read_csv('stock.csv')\")"
    assert await _hold()(session, pad=_Pad(), code=local) is None


async def test_the_shared_folder_is_held_too(project, tmp_path_factory):
    from anton.core.utils.write_check import snapshot_project_files

    shared = tmp_path_factory.mktemp("shared")
    (shared / "forecast.xlsx").write_text("x")
    session = await _gated_session(project, elicitor=_AnswersAtOnce(answer=CANCEL))
    session._write_gate.snapshots += (
        snapshot_project_files(root=shared, label="shared", taken_at=10**10),
    )
    outcome = await _hold()(
        session, pad=_Pad(), code=f"import os\nos.remove({str(shared / 'forecast.xlsx')!r})",
    )
    assert outcome is not None and "shared folder: forecast.xlsx" in outcome.content


async def test_begin_turn_snapshots_the_sandbox_s_shared_folder(project, tmp_path_factory):
    """The shared folder comes from the session's sandbox policy, the way a host sets it."""
    shared = tmp_path_factory.mktemp("shared")
    (shared / "forecast.xlsx").write_text("x")
    harness = _session(project)
    session = harness.session
    session._scratchpads._cell_sandbox = CellSandbox(
        project_dir=project, home_dir=tmp_path_factory.mktemp("home"),
        tmp_dir=tmp_path_factory.mktemp("tmp"), shared_dir=shared, user_name="tester",
    )
    session.elicitor = _AnswersAtOnce(answer=CANCEL)
    session.emitter = TurnEmitter()
    await session._write_gate.begin_turn(manager=session._scratchpads)
    assert [snap.label for snap in session._write_gate.snapshots] == ["project", "shared"]
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
    elicitor = _AnswersAtOnce(answer=CANCEL)
    session = await _gated_session(project, elicitor=elicitor)
    session._write_gate.snapshots += (
        snapshot_project_files(root=shared, label="shared", taken_at=10**10),
    )
    outcome = await _hold()(session, pad=_Pad(), code=code.format(root=str(shared)))
    assert outcome is not None and outcome.reason == "write_not_approved"
    assert elicitor.prompts and "This step would delete the whole shared folder." in elicitor.prompts[0]


# ── Compaction ────────────────────────────────────────────────────────────
#
# Under context pressure the session swaps all but a pad's last few cells for
# one summary cell. The pad's process keeps what those cells imported and
# bound, so the check must still read their code.

def _runtime(project: Path, *codes: str) -> LocalScratchpadRuntime:
    """A real, unstarted local runtime whose cells are `codes`."""
    pad = LocalScratchpadRuntime(
        name="main", coding_provider="anthropic", coding_model="", coding_api_key="",
        coding_base_url="", workspace_path=project, _venvs_base=project.parent / f"{project.name}-venvs",
    )
    pad.cells = [Cell(code=code, stdout="", stderr="", error=None) for code in codes]
    return pad


_FILLERS = tuple(f"total_{i} = {i}" for i in range(6))


@pytest.mark.parametrize("earlier,later,held", [
    ("from os import remove", "remove('report.pdf')", True),
    ("import os\nos.chdir('docs')", "import os\nos.remove('only.txt')", True),
    ("import duckdb\ncon = duckdb.connect()",
     "con.execute(\"CREATE TABLE summary AS SELECT * FROM read_csv('stock.csv')\")", False),
    ("import os, duckdb\nduckdb.sql(f\"ATTACH '{os.environ['DS_PG__URL']}' AS pg (TYPE POSTGRES)\")",
     "duckdb.sql(\"UPDATE pg.public.suppliers SET lead_time_days = 21 WHERE supplier_id = 'S005'\")", True),
], ids=["import-alias", "chdir", "local-duckdb-handle", "attach"])
async def test_compaction_leaves_the_check_every_earlier_cell(project, earlier, later, held):
    (project / "docs").mkdir()
    (project / "docs" / "only.txt").write_text("x")
    elicitor = _AnswersAtOnce(answer=CANCEL)
    session = await _gated_session(project, elicitor=elicitor)
    pad = _runtime(project, earlier, *_FILLERS)
    assert pad._compact_cells(), "the cell that binds the name is folded into the summary"
    assert all(cell.code != earlier for cell in pad.cells)
    outcome = await _hold()(session, pad=pad, code=later)
    assert (outcome is not None, elicitor.asked) == (held, int(held))


def test_compaction_keeps_the_code_it_folds_in_order(project):
    pad = _runtime(project, *_FILLERS, "total_6 = 6")
    assert pad._compact_cells()
    pad.cells.append(Cell(code="total_7 = 7", stdout="", stderr="", error=None))
    assert pad._compact_cells(), "a second compaction folds the first one's summary in too"
    assert pad.compacted_code == ("total_0 = 0", "total_1 = 1", "total_2 = 2")


async def test_a_reset_forgets_the_compacted_code(project):
    pad = _runtime(project, *_FILLERS, "total_6 = 6")
    assert pad._compact_cells()
    with patch.object(pad, "_stop_process", AsyncMock()), \
            patch.object(pad, "_run_venv_work", AsyncMock()), \
            patch.object(pad, "_discard_session_snapshot"), \
            patch.object(pad, "start", AsyncMock()):
        await pad.reset()
    assert (pad.cells, pad.compacted_code) == ([], ())


def test_a_refused_cell_is_not_working_code_for_the_artifact_pipeline():
    refused = ToolOutcome(content="Not run", ok=True, reason="write_not_approved")
    assert _exec_ran(refused) is False


def test_every_cell_that_never_ran_says_so_first():
    """A host marks a step as not run by its result's opening "Not run: "."""
    from anton.core.utils.scratchpad import _refusal
    from anton.core.utils.write_check import CellWrites

    assert _refusal(CellWrites(unchecked=True), "cancelled").content.startswith("Not run: ")
