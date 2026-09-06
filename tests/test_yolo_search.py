"""Finding a file when its name does not tell you anything.

The map lists paths. For a five-file artifact that is enough — the model
reads the names and knows where the title lives. For forty files it is
not, and no amount of retrying helps, because a retry can only widen to
files the model already guessed at.

Search closes that. It is deterministic, calls no model, and its results
feed straight into the read set.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from anton.core.yolo import Change, ReadRequest, Workspace, YoloEditor
from anton.core.yolo.workspace import (
    MAX_MATCHES_PER_FILE,
    MAX_MATCHES_PER_QUERY,
    MAX_QUERIES,
)

from tests.test_yolo_agent import StubCoder


def haystack(tmp_path: Path) -> Workspace:
    """A folder where the names give nothing away."""
    workspace = Workspace(tmp_path)
    workspace.write("a.js", "export const NAV = 1;\n")
    workspace.write("b.js", "// nothing interesting\n")
    workspace.write("c.js", "document.title = 'Old Title';\n")
    workspace.write("d.js", "// also nothing\n")
    return workspace


# ─── The search itself ──────────────────────────────────────────────────


def test_search_finds_the_file_the_name_did_not_reveal(tmp_path: Path):
    [match] = haystack(tmp_path).search("Old Title").matches
    assert match.path == "c.js"
    assert match.line == 1
    assert "document.title" in match.text


def test_search_ignores_case(tmp_path: Path):
    assert haystack(tmp_path).search("OLD TITLE").matches
    assert haystack(tmp_path).search("old title").matches


def test_search_skips_generated_data(tmp_path: Path):
    """Searching two megabytes of rows for a word returns thousands of
    lines of noise and buries the one line of code that reads them."""
    workspace = haystack(tmp_path)
    workspace.write("prices.data.js", "window.X=['Old Title','Old Title'];\n")
    assert [m.path for m in workspace.search("Old Title").matches] == ["c.js"]


def test_one_common_word_cannot_flood_the_prompt(tmp_path: Path):
    workspace = Workspace(tmp_path)
    for name in "abcdefghij":
        workspace.write(f"{name}.js", "hit\n" * 50)
    matches = workspace.search("hit").matches
    assert len(matches) <= MAX_MATCHES_PER_QUERY
    per_file = [m for m in matches if m.path == "a.js"]
    assert len(per_file) <= MAX_MATCHES_PER_FILE


def test_a_very_long_line_is_clipped(tmp_path: Path):
    workspace = Workspace(tmp_path)
    workspace.write("min.js", "x" * 5000 + "needle" + "y" * 5000)
    [match] = workspace.search("needle").matches
    assert len(match.text) < 400


def test_no_matches_is_reported_not_omitted(tmp_path: Path):
    """'That string is nowhere in this folder' is a real answer, and the
    one that stops the model looking for it."""
    rendered, hits = haystack(tmp_path).search_many(["Old Title", "not here at all"])
    assert "c.js:1" in rendered
    assert '"not here at all" — no matches' in rendered
    assert hits == ["c.js"]


def test_an_empty_query_matches_nothing(tmp_path: Path):
    assert haystack(tmp_path).search("   ").matches == []


# ─── Search inside the loop ─────────────────────────────────────────────


async def test_a_search_at_the_pick_step_finds_the_file_to_edit(tmp_path: Path):
    """The model cannot tell from a.js/b.js/c.js/d.js which sets the
    title, so it searches instead of guessing. One LLM call still."""
    workspace = haystack(tmp_path)
    coder = StubCoder(
        ReadRequest(paths=[], search=["Old Title"]),
        Change(
            summary="Retitled",
            files=["c.js"],
            diff="--- a/c.js\n+++ b/c.js\n@@\n"
            "-document.title = 'Old Title';\n+document.title = 'New Title';\n",
        ),
    )
    outcome = await YoloEditor(workspace=workspace, llm_client=coder).edit("retitle it")

    assert outcome.applied
    assert len(coder.calls) == 2, "search must not cost an extra model call"
    # The change request was shown both the hit and the file it is in.
    request = coder.calls[1]["content"]
    assert "SEARCH RESULTS:" in request
    assert "c.js:1" in request
    assert "document.title" in request
    assert "New Title" in workspace.read("c.js")


async def test_the_model_can_search_to_recover_from_a_wrong_pick(tmp_path: Path):
    """The recovery gap search was meant to close: it read the wrong file
    and does not know the name of the right one."""
    workspace = haystack(tmp_path)
    coder = StubCoder(
        ReadRequest(paths=["a.js"]),  # wrong file
        Change(summary="", files=[], diff="", need_search=["Old Title"]),
        Change(
            summary="Retitled",
            files=["c.js"],
            diff="--- a/c.js\n+++ b/c.js\n@@\n"
            "-document.title = 'Old Title';\n+document.title = 'New Title';\n",
        ),
    )
    outcome = await YoloEditor(workspace=workspace, llm_client=coder).edit("retitle it")

    assert outcome.applied
    assert outcome.attempts == 2
    retry = coder.calls[2]["content"]
    assert "c.js:1" in retry
    assert "document.title = 'Old Title';" in retry  # the file itself, now read


async def test_a_fruitless_search_still_terminates(tmp_path: Path):
    """Searching for something that is not there must not loop."""
    workspace = haystack(tmp_path)
    asking = Change(summary="", files=[], diff="", need_search=["not here at all"])
    coder = StubCoder(ReadRequest(paths=[]), asking, asking, asking)
    outcome = await YoloEditor(workspace=workspace, llm_client=coder).edit("x")

    assert not outcome.applied
    assert len(coder.calls) <= 4
    # And it was told the search found nothing, rather than left guessing.
    assert "no matches" in coder.calls[2]["content"]


# ─── Regex, and the one hazard that comes with it ───────────────────────


def test_search_takes_a_regular_expression(tmp_path: Path):
    """The reason regex is worth the trouble: finding where a thing is
    defined, not just where its exact spelling appears."""
    workspace = Workspace(tmp_path)
    workspace.write("a.js", "function drawChart(rows) {}\n")
    workspace.write("b.js", "function drawLegend(rows) {}\n")
    workspace.write("c.js", "const notAFunction = 1;\n")

    found = workspace.search(r"function\s+draw\w+")
    assert sorted(m.path for m in found.matches) == ["a.js", "b.js"]
    assert not found.note


def test_a_plain_word_is_still_a_valid_pattern(tmp_path: Path):
    assert haystack(tmp_path).search("Old Title").matches


def test_an_invalid_pattern_comes_back_as_feedback_not_a_crash(tmp_path: Path):
    """A bad pattern is something the model can fix next attempt, so it is
    handed back rather than raised."""
    found = haystack(tmp_path).search("unclosed (group")
    assert found.matches == []
    assert "invalid pattern" in found.note


def test_a_runaway_pattern_is_cut_off_rather_than_hung(tmp_path: Path):
    """The hazard that made this worth thinking about. `(a+)+b` against
    thirty characters takes ~46 seconds and grows exponentially, so no cap
    on line or file size contains it — only a wall-clock interrupt does."""
    workspace = Workspace(tmp_path)
    workspace.write("evil.txt", "a" * 40 + "!")

    started = time.monotonic()
    found = workspace.search(r"(a+)+b")
    elapsed = time.monotonic() - started

    assert elapsed < 5, f"took {elapsed:.1f}s — the interrupt did not fire"
    assert "timed out" in found.note
    # And it says what to do about it.
    assert "nested quantifiers" in found.note


async def test_a_runaway_pattern_does_not_take_the_run_down(tmp_path: Path):
    """It has to degrade to 'no matches', not an exception."""
    workspace = haystack(tmp_path)
    workspace.write("evil.txt", "a" * 40 + "!")
    coder = StubCoder(
        ReadRequest(paths=[], search=[r"(a+)+b"]),
        Change(
            summary="Retitled",
            files=["c.js"],
            diff="--- a/c.js\n+++ b/c.js\n@@\n"
            "-document.title = 'Old Title';\n+document.title = 'New';\n",
        ),
    )
    outcome = await YoloEditor(workspace=workspace, llm_client=coder).edit("retitle")
    assert outcome.applied
    assert "timed out" in coder.calls[1]["content"]


def test_too_many_queries_are_dropped_and_said_so(tmp_path: Path):
    """The worst case is timeout x queries, so the count is capped too."""
    rendered, _ = haystack(tmp_path).search_many([f"q{n}" for n in range(12)])
    assert "not run" in rendered
    assert f"at most {MAX_QUERIES}" in rendered
