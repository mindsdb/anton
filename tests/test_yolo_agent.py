"""Coverage for the yolo workspace and edit loop.

The LLM is a stub that returns whatever the test queues up, so the loop's
behaviour — how many calls it makes, what it retries, when it gives up and
what it says on the way out — is asserted exactly, with no provider and no
network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anton.core.yolo import (
    MAX_ATTEMPTS,
    Change,
    Outcome,
    ReadRequest,
    Workspace,
    WorkspaceError,
    YoloEditor,
)

# ─── Workspace ──────────────────────────────────────────────────────────


def test_a_path_outside_the_folder_is_refused(tmp_path: Path):
    """A hallucinated ../.. in a diff must be an error, not a write."""
    workspace = Workspace(tmp_path / "art")
    workspace.root.mkdir(parents=True)
    for escape in ["../secrets.txt", "a/../../secrets.txt", "/etc/hosts"]:
        with pytest.raises(WorkspaceError, match="outside"):
            workspace.resolve(escape)


def test_reading_and_writing_round_trips(tmp_path: Path):
    workspace = Workspace(tmp_path)
    workspace.write("nested/deep/file.txt", "hello")
    assert workspace.read("nested/deep/file.txt") == "hello"
    assert workspace.exists("nested/deep/file.txt")
    assert not workspace.exists("nope.txt")


def test_the_map_lists_paths_and_sizes_but_never_contents(tmp_path: Path):
    workspace = Workspace(tmp_path)
    workspace.write("index.html", "<h1>secret marker</h1>")
    workspace.write("src/app.js", "x")
    file_map = workspace.map()
    assert "index.html" in file_map and "src/app.js" in file_map
    assert "secret marker" not in file_map


def test_noise_directories_are_not_mapped(tmp_path: Path):
    workspace = Workspace(tmp_path)
    workspace.write("real.py", "x")
    workspace.write("node_modules/pkg/index.js", "x")
    workspace.write("__pycache__/real.pyc", "x")
    assert [info.path for info in workspace.files()] == ["real.py"]


def test_an_unreadable_file_explains_itself_instead_of_failing_the_batch(
    tmp_path: Path,
):
    workspace = Workspace(tmp_path)
    workspace.write("good.txt", "content")
    text = workspace.read_many(["good.txt", "missing.txt"])
    assert "content" in text
    assert "does not exist" in text


# ─── A stub coder ───────────────────────────────────────────────────────


class StubCoder:
    """Returns queued objects and records what it was asked."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls: list[dict] = []

    async def generate_object_code(
        self, schema_class, *, system, messages, max_tokens=None
    ):
        self.calls.append(
            {
                "schema": schema_class,
                "system": system,
                "content": messages[0]["content"],
            }
        )
        if not self.queue:
            raise AssertionError("the loop asked for more calls than the test queued")
        return self.queue.pop(0)


def make_workspace(tmp_path: Path) -> Workspace:
    workspace = Workspace(tmp_path)
    workspace.write("index.html", "<title>Old Title</title>\n<p>body</p>\n")
    workspace.write("style.css", "body { color: red; }\n")
    return workspace


# ─── The happy path ─────────────────────────────────────────────────────


async def test_a_change_that_applies_takes_two_calls(tmp_path: Path):
    """One to pick files, one to plan and patch. That is the budget."""
    workspace = make_workspace(tmp_path)
    coder = StubCoder(
        ReadRequest(paths=["index.html"]),
        Change(
            summary="Retitled it",
            files=["index.html"],
            diff="--- a/index.html\n+++ b/index.html\n@@\n"
            "-<title>Old Title</title>\n+<title>New Title</title>\n",
        ),
    )
    outcome = await YoloEditor(workspace=workspace, llm_client=coder).edit("retitle it")

    assert outcome.applied
    assert outcome.attempts == 1
    assert outcome.files == ["index.html"]
    assert len(coder.calls) == 2
    assert "New Title" in workspace.read("index.html")
    # The body it did not touch is still there.
    assert "<p>body</p>" in workspace.read("index.html")


async def test_only_the_files_asked_for_are_inlined(tmp_path: Path):
    """Reading a file the change does not need spends the room the change
    itself will want."""
    workspace = make_workspace(tmp_path)
    coder = StubCoder(
        ReadRequest(paths=["index.html"]),
        Change(summary="s", files=[], diff="--- a/index.html\n+++ b/index.html\n@@\n-<p>body</p>\n+<p>hi</p>\n"),
    )
    await YoloEditor(workspace=workspace, llm_client=coder).edit("change the body")

    change_request = coder.calls[1]["content"]
    assert "Old Title" in change_request  # index.html was inlined
    assert "color: red" not in change_request  # style.css was not


async def test_a_file_the_model_invents_is_dropped_not_read(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    coder = StubCoder(
        ReadRequest(paths=["index.html", "does_not_exist.js"]),
        Change(summary="s", files=[], diff="--- a/index.html\n+++ b/index.html\n@@\n-<p>body</p>\n+<p>hi</p>\n"),
    )
    outcome = await YoloEditor(workspace=workspace, llm_client=coder).edit("x")
    assert outcome.applied
    assert "does_not_exist.js" not in coder.calls[1]["content"]


async def test_a_new_file_is_created_by_the_patch(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    coder = StubCoder(
        ReadRequest(paths=[]),
        Change(
            summary="Added French",
            files=["fr.json"],
            diff='*** Begin Patch\n*** Add File: fr.json\n+{\n+  "hello": "bonjour"\n+}\n*** End Patch\n',
        ),
    )
    outcome = await YoloEditor(workspace=workspace, llm_client=coder).edit("add french")
    assert outcome.applied
    assert "bonjour" in workspace.read("fr.json")


# ─── The repair ladder ──────────────────────────────────────────────────


async def test_a_failed_diff_is_retried_with_its_own_output_as_evidence(
    tmp_path: Path,
):
    """Told only that it failed, a model reproduces the same diff. It has
    to see what it wrote."""
    workspace = make_workspace(tmp_path)
    bad = "--- a/index.html\n+++ b/index.html\n@@\n-<title>WRONG</title>\n+<title>New</title>\n"
    coder = StubCoder(
        ReadRequest(paths=["index.html"]),
        Change(summary="try one", files=["index.html"], diff=bad),
        Change(
            summary="try two",
            files=["index.html"],
            diff="--- a/index.html\n+++ b/index.html\n@@\n"
            "-<title>Old Title</title>\n+<title>New</title>\n",
        ),
    )
    outcome = await YoloEditor(workspace=workspace, llm_client=coder).edit("retitle")

    assert outcome.applied
    assert outcome.attempts == 2
    retry = coder.calls[2]["content"]
    assert "WHAT WENT WRONG LAST TIME" in retry
    assert "could not find" in retry  # the diagnosis
    assert "<title>WRONG</title>" in retry  # its own failed diff


async def test_it_gives_up_after_three_diffs_and_says_why(tmp_path: Path):
    """The handoff contract. After MAX_ATTEMPTS the outcome carries the
    diagnosis so the caller can pass the job on with it attached."""
    workspace = make_workspace(tmp_path)
    bad = Change(
        summary="nope",
        files=["index.html"],
        diff="--- a/index.html\n+++ b/index.html\n@@\n-<title>WRONG</title>\n+<title>x</title>\n",
    )
    coder = StubCoder(ReadRequest(paths=["index.html"]), bad, bad, bad)
    outcome = await YoloEditor(workspace=workspace, llm_client=coder).edit("retitle")

    assert not outcome.applied
    assert outcome.status == "patch_failed"
    assert outcome.attempts == MAX_ATTEMPTS
    assert "could not find" in outcome.detail
    # And the file was left exactly as it was.
    assert workspace.read("index.html") == "<title>Old Title</title>\n<p>body</p>\n"


async def test_a_promised_file_that_was_not_created_is_caught(tmp_path: Path):
    """A patch that omits the new file it promised applies perfectly and
    looks like success. It is not success."""
    workspace = make_workspace(tmp_path)
    lying = Change(
        summary="added a helper",
        files=["index.html", "helper.js"],  # helper.js never appears in the diff
        diff="--- a/index.html\n+++ b/index.html\n@@\n-<p>body</p>\n+<p>new</p>\n",
    )
    honest = Change(
        summary="added a helper",
        files=["helper.js"],
        diff="*** Begin Patch\n*** Add File: helper.js\n+export const x = 1;\n*** End Patch\n",
    )
    coder = StubCoder(ReadRequest(paths=[]), lying, honest)
    outcome = await YoloEditor(workspace=workspace, llm_client=coder).edit("add a helper")

    assert outcome.applied
    assert outcome.attempts == 2
    assert "did not create helper.js" in coder.calls[2]["content"]
    assert workspace.exists("helper.js")


async def test_an_empty_diff_stops_immediately(tmp_path: Path):
    """There is nothing to apply and nothing to repair from; another
    attempt is the same question asked again."""
    workspace = make_workspace(tmp_path)
    coder = StubCoder(
        ReadRequest(paths=[]),
        Change(summary="I could not do it", files=[], diff="   "),
    )
    outcome = await YoloEditor(workspace=workspace, llm_client=coder).edit("x")

    assert not outcome.applied
    assert outcome.status == "no_diff"
    assert outcome.attempts == 1
    assert len(coder.calls) == 2  # it did not retry


async def test_nothing_is_written_when_a_later_file_will_not_apply(tmp_path: Path):
    """All files or none. A hunk that will not place in the second file
    must not leave the first one edited and the change half-done."""
    workspace = make_workspace(tmp_path)
    coder = StubCoder(
        ReadRequest(paths=[]),
        Change(
            summary="two files",
            files=["index.html", "style.css"],
            diff=(
                "--- a/index.html\n+++ b/index.html\n@@\n-<p>body</p>\n+<p>ok</p>\n"
                "--- a/style.css\n+++ b/style.css\n@@\n-body { color: blue; }\n+body { color: green; }\n"
            ),
        ),
        Change(summary="x", files=[], diff="   "),
    )
    await YoloEditor(workspace=workspace, llm_client=coder).edit("x")

    assert workspace.read("index.html") == "<title>Old Title</title>\n<p>body</p>\n"
    assert workspace.read("style.css") == "body { color: red; }\n"


async def test_progress_is_reported(tmp_path: Path):
    class Recorder:
        def __init__(self):
            self.lines: list[str] = []

        def status(self, message: str) -> None:
            self.lines.append(f"status: {message}")

        def log(self, message: str) -> None:
            self.lines.append(f"log: {message}")

    workspace = make_workspace(tmp_path)
    recorder = Recorder()
    coder = StubCoder(
        ReadRequest(paths=["index.html"]),
        Change(
            summary="Retitled it",
            files=["index.html"],
            diff="--- a/index.html\n+++ b/index.html\n@@\n-<title>Old Title</title>\n+<title>New</title>\n",
        ),
    )
    await YoloEditor(workspace=workspace, llm_client=coder, progress=recorder).edit("x")

    trail = "\n".join(recorder.lines)
    assert "reading index.html" in trail
    assert "plan: Retitled it" in trail
    assert "wrote index.html" in trail


async def test_a_broken_workspace_is_an_outcome_not_an_exception(tmp_path: Path):
    """Callers get one shape back whatever went wrong."""
    workspace = Workspace(tmp_path)
    workspace.write("f.txt", "x")
    escape = Change(
        summary="s",
        files=[],
        diff="*** Begin Patch\n*** Add File: ../escape.txt\n+x\n*** End Patch\n",
    )
    coder = StubCoder(ReadRequest(paths=[]), escape, escape, escape)
    outcome = await YoloEditor(workspace=workspace, llm_client=coder).edit("x")
    assert isinstance(outcome, Outcome)
    assert not outcome.applied
    assert not (tmp_path.parent / "escape.txt").exists()


# ─── Recovering from a wrong file pick ──────────────────────────────────
#
# The bounded stand-in for a read/search tool loop. The loop was never
# really about the first pick — models are good at that — it was about
# recovering when the first pick turned out wrong. Without it, a model
# rewrites the same doomed diff until the attempts run out.


def three_file_project(tmp_path: Path) -> Workspace:
    workspace = Workspace(tmp_path)
    workspace.write("index.html", "<script src='chart.js'></script>\n")
    workspace.write("chart.js", "import { fmt } from './utils.js';\n")
    workspace.write("utils.js", "export const fmt = (n) => n.toFixed(2);\n")
    return workspace


async def test_a_failed_diff_against_an_unread_file_widens_the_read_set(
    tmp_path: Path,
):
    """The first pick missed utils.js. The model writes against it anyway
    and fails. It must be shown the file, not lectured about copying."""
    workspace = three_file_project(tmp_path)
    blind = Change(
        summary="round to 0 dp",
        files=["utils.js"],
        diff="--- a/utils.js\n+++ b/utils.js\n@@\n"
        "-export const fmt = (n) => n.toFixed(1);\n"   # wrong — never saw it
        "+export const fmt = (n) => n.toFixed(0);\n",
    )
    sighted = Change(
        summary="round to 0 dp",
        files=["utils.js"],
        diff="--- a/utils.js\n+++ b/utils.js\n@@\n"
        "-export const fmt = (n) => n.toFixed(2);\n"
        "+export const fmt = (n) => n.toFixed(0);\n",
    )
    coder = StubCoder(ReadRequest(paths=["chart.js"]), blind, sighted)
    outcome = await YoloEditor(workspace=workspace, llm_client=coder).edit("round it")

    assert outcome.applied
    assert outcome.attempts == 2
    retry = coder.calls[2]["content"]
    # utils.js is now in front of it, with its real contents.
    assert "toFixed(2)" in retry
    assert "had not been shown" in retry
    assert "toFixed(0)" in workspace.read("utils.js")


async def test_the_model_can_ask_for_files_instead_of_guessing(tmp_path: Path):
    """An empty diff plus need_files is a legitimate answer, not a dead
    end. Previously it ended the run as `no_diff`."""
    workspace = three_file_project(tmp_path)
    asking = Change(summary="", files=[], diff="", need_files=["utils.js"])
    answering = Change(
        summary="round to 0 dp",
        files=["utils.js"],
        diff="--- a/utils.js\n+++ b/utils.js\n@@\n"
        "-export const fmt = (n) => n.toFixed(2);\n"
        "+export const fmt = (n) => n.toFixed(0);\n",
    )
    coder = StubCoder(ReadRequest(paths=["chart.js"]), asking, answering)
    outcome = await YoloEditor(workspace=workspace, llm_client=coder).edit("round it")

    assert outcome.applied, "asking for a file should not end the run"
    assert outcome.attempts == 2
    assert "toFixed(2)" in coder.calls[2]["content"]


async def test_asking_for_a_file_that_does_not_exist_still_ends_the_run(
    tmp_path: Path,
):
    """Widening only ever adds real files, so an invented name cannot loop."""
    workspace = three_file_project(tmp_path)
    coder = StubCoder(
        ReadRequest(paths=["chart.js"]),
        Change(summary="", files=[], diff="", need_files=["imaginary.js"]),
    )
    outcome = await YoloEditor(workspace=workspace, llm_client=coder).edit("x")
    assert outcome.status == "no_diff"
    assert len(coder.calls) == 2


async def test_widening_cannot_run_past_the_attempt_budget(tmp_path: Path):
    """Recovery is bounded by the same 3 attempts — it is not a loop."""
    workspace = three_file_project(tmp_path)
    asking = Change(summary="", files=[], diff="", need_files=["utils.js", "index.html"])
    coder = StubCoder(ReadRequest(paths=[]), asking, asking, asking)
    outcome = await YoloEditor(workspace=workspace, llm_client=coder).edit("x")

    assert not outcome.applied
    # 1 pick + at most 3 changes. It cannot keep asking forever.
    assert len(coder.calls) <= 4


async def test_a_plain_bad_diff_still_gets_the_copy_carefully_note(tmp_path: Path):
    """Widening must not swallow the ordinary case. When every named file
    was already read, the failure really is careless copying."""
    workspace = three_file_project(tmp_path)
    bad = Change(
        summary="x",
        files=["chart.js"],
        diff="--- a/chart.js\n+++ b/chart.js\n@@\n-NOT IN THE FILE\n+y\n",
    )
    coder = StubCoder(ReadRequest(paths=["chart.js"]), bad, bad, bad)
    await YoloEditor(workspace=workspace, llm_client=coder).edit("x")

    retry = coder.calls[2]["content"]
    assert "character for character" in retry
    assert "had not been shown" not in retry
