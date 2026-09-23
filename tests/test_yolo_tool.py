"""The `edit_artifact` tool: yolo wired into the agent.

Covers the seams rather than the engine — the engine has its own tests.
What matters here is what a tool call does to the world: whether it
streams, whether the store finds out, and what the model is told when it
does not work.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from anton.core.artifacts import ArtifactStore
from anton.core.tools.progress import ToolProgress
from anton.core.tools.registry import ToolOutcome
from anton.core.tools.tool_defs import EDIT_ARTIFACT_TOOL
from anton.core.tools.tool_handlers import handle_edit_artifact
from anton.core.yolo import Change, Outcome, ReadRequest


class FakeSession:
    """Enough ChatSession for the handler, and nothing more."""

    def __init__(self, workspace_dir: Path, llm) -> None:
        class _W:
            artifacts_dir = workspace_dir / "artifacts"

        self._workspace = _W()
        self.llm_client = llm
        self._session_id = "conv-1"
        self._turn_count = 3
        self._artifacts_touched: set[str] = set()


class StubLLM:
    def __init__(self, *responses):
        self.queue = list(responses)

    async def generate_object_code(self, schema_class, *, system, messages, max_tokens=None):
        if not self.queue:
            raise AssertionError("more model calls than the test queued")
        return self.queue.pop(0)


def artifact_with(tmp_path: Path, llm) -> tuple[FakeSession, ArtifactStore, str]:
    session = FakeSession(tmp_path, llm)
    store = ArtifactStore(tmp_path / "artifacts")
    art = store.create(name="Chart", description="a chart", type="html-app")
    folder = store.folder_for(art.slug)
    (folder / "index.html").write_text("<title>Old Title</title>\n")
    store.rescan_files(art.slug)
    return session, store, art.slug


async def drain(session, tc_input) -> tuple[list[str], ToolOutcome]:
    """Run the handler, separating progress lines from the result."""
    lines, result = [], None
    async for item in handle_edit_artifact(session, tc_input):
        if isinstance(item, ToolProgress):
            lines.append(item.text)
        else:
            result = item
    return lines, result


RETITLE = Change(
    summary="Retitled it",
    files=["index.html"],
    diff="--- a/index.html\n+++ b/index.html\n@@\n"
    "-<title>Old Title</title>\n+<title>TicTacTris</title>\n",
)


# ─── The tool contract ──────────────────────────────────────────────────


def test_the_tool_tells_the_model_when_not_to_use_it():
    """Routing is done by the description, not a classifier, so the
    description has to carry the boundary."""
    text = EDIT_ARTIFACT_TOOL.description
    assert "already exists" in text
    assert "scratchpad" in text  # named as the alternative, twice over
    assert "data.js" in text  # and the one thing it will not write


async def test_a_change_is_applied_and_reported(tmp_path: Path):
    llm = StubLLM(ReadRequest(paths=["index.html"]), RETITLE)
    session, store, slug = artifact_with(tmp_path, llm)

    _, result = await drain(session, {"slug": slug, "task": "retitle it"})

    assert result.ok is True
    assert "Retitled it" in result.content
    assert "index.html" in result.content
    assert "TicTacTris" in (store.folder_for(slug) / "index.html").read_text()


async def test_progress_streams_while_the_work_runs(tmp_path: Path):
    """A run makes two model calls. Without streaming it is a silent
    pause, which is what the whole bridge exists to prevent."""
    llm = StubLLM(ReadRequest(paths=["index.html"]), RETITLE)
    session, _, slug = artifact_with(tmp_path, llm)

    lines, result = await drain(session, {"slug": slug, "task": "retitle it"})

    assert result.ok is True
    trail = "\n".join(lines)
    assert "reading index.html" in trail
    assert "plan: Retitled it" in trail
    assert "wrote index.html" in trail


async def test_the_store_is_told_what_changed(tmp_path: Path):
    """Yolo writes to the folder directly, so nothing updates
    metadata.json unless the handler says so. The symptom of forgetting
    is a silently stale file list."""
    llm = StubLLM(ReadRequest(paths=["index.html"]), RETITLE)
    session, store, slug = artifact_with(tmp_path, llm)

    await drain(session, {"slug": slug, "task": "retitle it"})

    art = store.open(slug)
    assert slug in session._artifacts_touched
    [entry] = art.provenance
    [turn] = entry.turns
    assert turn.summary == "Retitled it"
    assert turn.files_touched == ["index.html"]


async def test_a_new_file_shows_up_in_the_metadata(tmp_path: Path):
    llm = StubLLM(
        ReadRequest(paths=[]),
        Change(
            summary="Added a stylesheet",
            files=["style.css"],
            diff="*** Begin Patch\n*** Add File: style.css\n+body { margin: 0; }\n*** End Patch\n",
        ),
    )
    session, store, slug = artifact_with(tmp_path, llm)

    _, result = await drain(session, {"slug": slug, "task": "add a stylesheet"})

    assert result.ok is True
    assert "style.css" in {f.path for f in store.open(slug).files}


# ─── Failing back to the scratchpad ─────────────────────────────────────


async def test_a_failure_writes_nothing_and_points_at_the_scratchpad(tmp_path: Path):
    """The handoff. The model already has the scratchpad; it needs to be
    told plainly that nothing was written and what to do next."""
    doomed = Change(
        summary="nope",
        files=["index.html"],
        diff="--- a/index.html\n+++ b/index.html\n@@\n-<title>WRONG</title>\n+<title>x</title>\n",
    )
    llm = StubLLM(ReadRequest(paths=["index.html"]), doomed, doomed, doomed)
    session, store, slug = artifact_with(tmp_path, llm)
    before = (store.folder_for(slug) / "index.html").read_text()

    _, result = await drain(session, {"slug": slug, "task": "retitle it"})

    assert result.ok is False
    assert "Nothing was written" in result.content
    assert "scratchpad" in result.content
    assert "could not find" in result.content  # the diagnosis travels with it
    assert result.reason == "yolo_patch_failed"
    assert (store.folder_for(slug) / "index.html").read_text() == before


async def test_a_failed_run_is_not_recorded_as_a_turn(tmp_path: Path):
    """Provenance says which turns changed the artifact. One that changed
    nothing did not."""
    doomed = Change(summary="", files=[], diff="")
    llm = StubLLM(ReadRequest(paths=[]), doomed)
    session, store, slug = artifact_with(tmp_path, llm)

    await drain(session, {"slug": slug, "task": "x"})

    assert store.open(slug).provenance == []
    assert session._artifacts_touched == set()


async def test_an_unknown_slug_is_an_error_not_a_crash(tmp_path: Path):
    session, _, _ = artifact_with(tmp_path, StubLLM())
    _, result = await drain(session, {"slug": "nope", "task": "x"})
    assert result.ok is False
    assert result.reason == "artifact_not_found"


async def test_missing_arguments_are_reported_as_the_agents_own_fault(tmp_path: Path):
    """These reasons feed root_cause tiering, where a bad argument is
    self-fixable and must not be counted as an environment wall."""
    session, _, slug = artifact_with(tmp_path, StubLLM())
    for tc_input, reason in (
        ({"task": "x"}, "missing_slug"),
        ({"slug": slug}, "missing_task"),
    ):
        _, result = await drain(session, tc_input)
        assert result.ok is False
        assert result.reason == reason


async def test_no_workspace_means_no_store(tmp_path: Path):
    session = FakeSession(tmp_path, StubLLM())
    session._workspace = None
    _, result = await drain(session, {"slug": "x", "task": "y"})
    assert result.ok is False
    assert result.reason == "store_unavailable"


async def test_a_crash_inside_the_editor_surfaces(tmp_path: Path):
    """A silently empty result would be worse than an exception."""

    class Exploding:
        async def generate_object_code(self, *a, **k):
            raise RuntimeError("provider is down")

    session, _, slug = artifact_with(tmp_path, Exploding())
    with pytest.raises(RuntimeError, match="provider is down"):
        await drain(session, {"slug": slug, "task": "x"})
