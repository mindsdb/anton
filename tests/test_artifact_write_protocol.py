"""The file-body protocol: content as text, `write_file` for the path only.

Why the protocol exists at all is in
docs/artifact-generation-tools/2026-09-15-tool-call-argument-not-streamed.md —
a body sent as a tool argument leaves the connection silent for the whole
generation, because the API buffers a parameter before streaming it.

What this file guards is the other half: a protocol built out of markers in
free text can fail in ways a JSON schema could not, and every one of those
failures must be loud. A file written from a body we guessed at is worse than
no file, because nothing downstream can tell.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock

from anton.core.llm.provider import LLMResponse, StreamComplete, ToolCall, Usage
from anton.core.tools.generate_artifact.engine import _run_loop
from anton.core.tools.generate_artifact.sub_tools import (
    FILE_BEGIN_MARKER,
    FILE_END_MARKER,
    NOTE_TEXT_AFTER,
    NOTE_TEXT_BEFORE,
    extract_file_body,
)


def _wrap(text: str, *, closed: bool = True) -> str:
    tail = f"\n{FILE_END_MARKER}" if closed else ""
    return f"{FILE_BEGIN_MARKER}\n{text}{tail}"


def _resp(tool_calls=(), body: str = "") -> LLMResponse:
    return LLMResponse(
        content=body,
        tool_calls=[ToolCall(id=str(i), name=n, input=inp) for i, (n, inp) in enumerate(tool_calls)],
        usage=Usage(input_tokens=1, output_tokens=1),
        stop_reason="tool_use",
    )


async def _one(response):
    yield StreamComplete(response=response)


async def _drive(tmp_path: Path, responses, **kw):
    """Run the loop over a fixed sequence of replies."""
    it = iter(responses)
    session = AsyncMock()
    session._llm.plan_stream = Mock(side_effect=lambda **k: _one(next(it)))
    session._llm.code_stream = Mock(side_effect=lambda **k: _one(next(it)))
    kw.setdefault("require_files", False)
    return await _run_loop(
        session=session, system="s", kickoff="k", artifact_path=tmp_path,
        node_label="generate_frontend", **kw,
    )


class _Trace:
    """Records only what these tests ask about; everything else is a no-op."""

    def __init__(self) -> None:
        self.notes: list[str] = []

    def protocol_note(self, *, node, kind):
        self.notes.append(kind)

    def __getattr__(self, _name):
        return lambda *a, **k: None


# ── The parser ──────────────────────────────────────────────────────────────

def test_a_well_formed_body_comes_back_verbatim():
    body, err, notes = extract_file_body(_wrap("<html>\n  <p>hi</p>\n</html>"))
    assert err is None and notes == []
    # Interior newlines and indentation survive; only the two newlines that
    # belong to the marker lines are removed.
    assert body == "<html>\n  <p>hi</p>\n</html>"


def test_no_markers_at_all_is_an_error_naming_them():
    body, err, _ = extract_file_body("I will now write the file.")
    assert body is None
    assert FILE_BEGIN_MARKER in err and FILE_END_MARKER in err


def test_a_duplicated_marker_refuses_instead_of_taking_the_first_pair():
    """The one case where guessing would be silent data loss.

    If the file's own content contains a marker, every rule for picking "the
    real pair" is a rule for deciding where the file ends — and picking wrong
    truncates it with nothing to show for it.
    """
    two = _wrap("first") + "\n" + _wrap("second")
    body, err, _ = extract_file_body(two)
    assert body is None
    assert "exactly once" in err


def test_an_empty_body_is_an_error():
    body, err, _ = extract_file_body(_wrap("   \n  "))
    assert body is None
    assert "empty" in err


def test_markers_in_the_wrong_order_are_an_error():
    body, err, _ = extract_file_body(f"{FILE_END_MARKER}\nx\n{FILE_BEGIN_MARKER}")
    assert body is None
    assert "before" in err


def test_prose_around_the_body_is_accepted_and_noted():
    """Not a failure — an observation.

    A live run (2026-09-15) opened a continuation round with "Let me append the
    JavaScript section." before the body. Under the plan's first draft that was
    an error and would have cost a round; it costs nothing, because the body was
    complete and the file was written. The note exists so a CHANGE in the
    model's formatting is visible, not so anything acts on it.
    """
    body, err, notes = extract_file_body("Sure!\n" + _wrap("x") + "\nDone.")
    assert err is None
    assert body == "x"
    assert notes == [NOTE_TEXT_BEFORE, NOTE_TEXT_AFTER]


# ── The loop ────────────────────────────────────────────────────────────────

async def test_the_body_reaches_disk_and_the_call_carries_no_content(tmp_path: Path):
    result = await _drive(tmp_path, [
        _resp([("write_file", {"path": "index.html"})], body=_wrap("<html></html>")),
        _resp([("finish", {"summary": "ok"})]),
    ])
    assert result["files_written"] == ["index.html"]
    assert (tmp_path / "index.html").read_text(encoding="utf-8") == "<html></html>"


async def test_a_content_argument_is_refused_and_not_written(tmp_path: Path):
    """Two ways to deliver a body would drift: the model would sometimes send
    both and sometimes neither. There is one way."""
    result = await _drive(tmp_path, [
        # A body IS present, so silently ignoring the stray argument would look
        # identical to refusing it — the file would land either way. The call
        # has to be refused outright, or the model keeps sending both and the
        # day one of them disagrees with the other is the day a wrong file is
        # written with nothing to notice it.
        _resp([("write_file", {"path": "a.html", "content": "from the argument"})],
              body=_wrap("from the body")),
        _resp([("finish", {"summary": "gave up"})]),
    ])
    assert result["files_written"] == []
    assert not (tmp_path / "a.html").exists()


async def test_a_body_with_no_write_file_is_retried_without_regenerating_it(tmp_path: Path):
    """The expensive half is the body; the cheap half is the call.

    A retry that asked for both would pay twice for the same file, so the body
    is kept and the second reply only has to name the path.
    """
    result = await _drive(tmp_path, [
        _resp([], body=_wrap("<html>kept</html>")),          # forgot the call
        _resp([("write_file", {"path": "i.html"})]),          # no body this time
        _resp([("finish", {"summary": "ok"})]),
    ])
    assert result["files_written"] == ["i.html"]
    assert (tmp_path / "i.html").read_text(encoding="utf-8") == "<html>kept</html>"


async def test_a_body_alongside_finish_does_not_end_the_run(tmp_path: Path):
    """The failure this branch exists for, and it is the model's normal ending.

    "Final part plus `finish` in one reply" is how it finishes today. Accept the
    `finish` and the last section never reaches disk — while `require_files`
    passes on the parts written by earlier rounds, so the artifact ships
    truncated and is reported as a success.
    """
    result = await _drive(tmp_path, [
        _resp([("write_file", {"path": "i.html"})], body=_wrap("<html>part1")),
        _resp([("finish", {"summary": "done"})], body=_wrap("part2</html>")),  # trap
        _resp([("write_file", {"path": "i.html", "mode": "a"})]),
        _resp([("finish", {"summary": "ok"})]),
    ])
    assert result["finished"] is True
    assert (tmp_path / "i.html").read_text(encoding="utf-8") == "<html>part1part2</html>"


async def test_a_reply_with_neither_body_nor_calls_still_ends_the_loop(tmp_path: Path):
    """Unchanged behaviour: a model that simply stopped is a failed node, not a
    protocol slip to retry forever."""
    result = await _drive(tmp_path, [_resp([], body="I am not sure what to do.")])
    assert isinstance(result, str)
    assert "without writing files" in result


async def test_prose_around_the_body_reaches_the_trace(tmp_path: Path):
    """Countable, and countable APART from refusals.

    A refusal costs a round and belongs in the acceptance thresholds; this does
    not and must stay out of them — lumping the two made a healthy run read as
    50% failing.
    """
    trace = _Trace()
    await _drive(tmp_path, [
        _resp([("write_file", {"path": "i.html"})],
              body="Here you go:\n" + _wrap("<html></html>") + "\nAll set."),
        _resp([("finish", {"summary": "ok"})]),
    ], trace=trace)
    assert trace.notes == [NOTE_TEXT_BEFORE, NOTE_TEXT_AFTER]
    assert (tmp_path / "i.html").read_text(encoding="utf-8") == "<html></html>"
