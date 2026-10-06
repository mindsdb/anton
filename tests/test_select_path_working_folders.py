"""select_path with working folders the host handed the session.

Without working folders every message must stay exactly what it was before
they existed; the expected strings below are copied from that version.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.interaction.elicit import AskAnswer
from anton.core.tools import tool_handlers
from anton.core.tools.tool_handlers import handle_select_path


class _ChoiceElicitor:
    """A host with choice cards and a path picker that picks a scripted option."""

    supported_kinds = ("choice", "path")
    answer_hint = "hint"
    timeout_s = None

    def __init__(self, chosen: str | None = None, *, kinds=("choice", "path"), text: str = "") -> None:
        self.supported_kinds = kinds
        self.chosen = chosen
        self.text = text
        self.requests: list = []

    async def begin(self, question_id, request):
        return None

    async def end(self, question_id):
        return None

    async def ask(self, question_id, request):
        self.requests.append(request)
        if self.text:
            return AskAnswer(status="answered", text=self.text)
        if self.chosen is None:
            return AskAnswer(status="cancelled")
        return AskAnswer(status="answered", values=(self.chosen,))


async def _noop_emit(event):
    return None


def _session(workspace: Path, folders=(), elicitor=None, *, emitter=None):
    return SimpleNamespace(
        _console=None,
        elicitor=elicitor,
        emitter=emitter,
        emit=_noop_emit,
        question_count=0,
        answer_wait_s=0.0,
        escape_watcher=None,
        _workspace=SimpleNamespace(base=workspace),
        _working_folders=tuple(Path(f).resolve() for f in folders),
    )


@pytest.fixture()
def layout(tmp_path):
    workspace = tmp_path / "project"
    folder = tmp_path / "docs"
    outside = tmp_path / "elsewhere"
    for d in (workspace, folder, outside):
        d.mkdir()
    return workspace, folder, outside


async def _call(session, tc_input):
    return json.loads(await handle_select_path(session, tc_input))


async def test_a_pattern_alone_finds_a_file_in_a_working_folder(layout):
    workspace, folder, _ = layout
    (folder / "q3" ).mkdir()
    (folder / "q3" / "sales.csv").write_text("a")

    result = await _call(_session(workspace, [folder]), {"pattern": "**/sales*.csv"})

    assert result["status"] == "resolved"
    assert result["path"] == str((folder / "q3" / "sales.csv").resolve())


async def test_an_absolute_base_dir_inside_a_working_folder_is_searched(layout):
    workspace, folder, _ = layout
    (folder / "plan.md").write_text("a")

    result = await _call(
        _session(workspace, [folder]), {"pattern": "*.md", "base_dir": str(folder)}
    )

    assert result["path"] == str((folder / "plan.md").resolve())


async def test_a_path_outside_every_root_is_still_dropped(layout):
    workspace, folder, outside = layout
    (outside / "secret.csv").write_text("a")

    by_candidate = await _call(
        _session(workspace, [folder]), {"candidates": [str(outside / "secret.csv")]}
    )
    by_base_dir = await _call(
        _session(workspace, [folder]), {"pattern": "*.csv", "base_dir": str(outside)}
    )

    assert by_candidate["status"] == "no_matches"
    assert by_base_dir["status"] == "no_matches"


async def test_an_anton_directory_inside_a_working_folder_is_never_offered(layout):
    workspace, folder, _ = layout
    (folder / ".anton").mkdir()
    (folder / ".anton" / "notes.md").write_text("a")

    result = await _call(_session(workspace, [folder]), {"pattern": "**/*.md"})

    assert result["status"] == "no_matches"


@pytest.mark.skipif(sys.platform != "darwin", reason="case-insensitive volume")
async def test_a_case_variant_of_a_working_folder_path_fails_safe_on_macos(layout):
    workspace, folder, _ = layout
    (folder / "plan.md").write_text("a")
    variant = folder.parent / folder.name.upper() / "plan.md"

    result = await _call(_session(workspace, [folder]), {"candidates": [str(variant)]})

    assert result["status"] == "no_matches"


async def test_options_label_working_folder_files_by_folder_name(layout):
    workspace, folder, _ = layout
    (workspace / "a.csv").write_text("a")
    (folder / "b.csv").write_text("b")
    elicitor = _ChoiceElicitor(chosen=str((folder / "b.csv").resolve()))

    result = await _call(_session(workspace, [folder], elicitor), {"pattern": "*.csv"})

    labels = sorted(option.label for option in elicitor.requests[0].options)
    assert labels == ["a.csv", "docs/b.csv"]
    assert result["path"] == str((folder / "b.csv").resolve())


async def test_the_scan_budget_is_shared_across_roots(layout, monkeypatch):
    workspace, folder, _ = layout
    for i in range(5):
        (workspace / f"w{i}.txt").write_text("a")
    (folder / "late.txt").write_text("a")
    monkeypatch.setattr(tool_handlers, "_SELECTION_SCAN_LIMIT", 5)

    elicitor = _ChoiceElicitor()

    found = await _call(_session(workspace, [folder]), {"pattern": "late.txt"})
    await _call(_session(workspace, [folder], elicitor), {"pattern": "*.txt"})

    assert found["status"] == "resolved"
    offered = [option.label for option in elicitor.requests[0].options]
    assert len(offered) == 5
    assert "docs/late.txt" not in offered


_NO_MATCHES_PATHLESS = (
    "No match found in the project. Refine the pattern, or — if the file "
    "is not in the project at all — ask the user to attach it to the "
    "conversation. This host has no file browser, and you cannot read a "
    "path the user types."
)
_BROWSE_PICKER_UNAVAILABLE = (
    "This host cannot render a file browser. Ask the user to attach the "
    "file to the conversation — that is how they grant you access to a "
    "file outside the project. Do not ask them to type or paste a path, "
    "and do not proceed with invented or example data in its place."
)


def _needs_confirmation_text(label: str) -> str:
    return (
        f"You supplied '{label}' as the only candidate — that is your guess, not "
        "the user's choice, so it was not accepted. Ask the user to confirm it "
        "before using it, and do not present it as chosen or connected until "
        "they do. If they asked for a file or folder outside the project, no "
        "path inside the project is an answer: tell them plainly that this host "
        "cannot reach paths outside the project, and ask them to attach the "
        "relevant files to the conversation."
    )


def _declined_text(label: str) -> str:
    return (
        f"The user declined '{label}'. Do not use this path and do not present "
        "it as connected. Ask what they actually meant — and if it is a file or "
        "folder outside the project, tell them plainly that this host cannot "
        "reach paths outside the project, and ask them to attach the relevant "
        "files to the conversation."
    )


def _typed_text(typed: str, label: str) -> str:
    return (
        f'The user typed a reply instead of choosing an option: "{typed}". '
        f"'{label}' is not confirmed — do not use it or present it as "
        "connected; act on their reply. If it reads as agreement, call "
        "select_path again for an explicit confirmation. If they want a "
        "file or folder outside the project, tell them plainly that this "
        "host cannot reach it and ask them to attach the relevant files "
        "to the conversation."
    )


async def test_without_working_folders_every_message_is_unchanged(layout):
    workspace, _, _ = layout
    (workspace / "skills").mkdir()
    pathless = ("choice",)
    one_candidate = {"kind": "folder", "candidates": ["skills"]}

    no_matches = await _call(_session(workspace, elicitor=_ChoiceElicitor(kinds=pathless)), {"pattern": "*.nope"})
    browse = await _call(_session(workspace), {"prompt": "Find it"})
    needs = await _call(_session(workspace), one_candidate)
    declined = await _call(
        _session(workspace, elicitor=_ChoiceElicitor(chosen="no", kinds=pathless), emitter=object()),
        one_candidate,
    )
    typed = await _call(
        _session(workspace, elicitor=_ChoiceElicitor(kinds=pathless, text="the other one"), emitter=object()),
        one_candidate,
    )

    assert no_matches["message"] == _NO_MATCHES_PATHLESS
    assert browse["message"] == _BROWSE_PICKER_UNAVAILABLE
    assert needs["message"] == _needs_confirmation_text("skills")
    assert declined["message"] == _declined_text("skills")
    assert typed["message"] == _typed_text("the other one", "skills")


async def test_with_working_folders_the_messages_name_them(layout):
    workspace, folder, _ = layout
    (folder / "data").mkdir()
    pathless = ("choice",)
    one_candidate = {"kind": "folder", "candidates": [str(folder / "data")]}

    no_matches = await _call(
        _session(workspace, [folder], _ChoiceElicitor(kinds=pathless)), {"pattern": "*.nope"}
    )
    browse = await _call(_session(workspace, [folder]), {"prompt": "Find it"})
    needs = await _call(_session(workspace, [folder]), one_candidate)

    assert no_matches["message"].startswith("No match found in the project and its working folders.")
    assert "outside the project and its working folders" in browse["message"]
    assert browse["message"].endswith(
        "First search the working folders by calling select_path with a pattern."
    )
    assert "You supplied 'docs/data'" in needs["message"]
    assert "outside the project and its working folders" in needs["message"]


def _registered_select_path(session):
    session._build_tools()
    return next(t for t in session.tool_registry.get_tool_defs() if t.name == "select_path")


@pytest.mark.parametrize("kinds", [("choice", "path"), ("choice",)])
def test_without_working_folders_the_registered_tool_is_the_module_definition(make_session, kinds):
    from anton.core.tools.tool_defs import SELECT_PATH_TOOL, SELECT_PATH_TOOL_PICK_ONLY

    tool = _registered_select_path(make_session(elicitor=_ChoiceElicitor(kinds=kinds)))

    assert tool is (SELECT_PATH_TOOL if "path" in kinds else SELECT_PATH_TOOL_PICK_ONLY)


def test_every_widened_phrase_exists_in_the_definitions():
    """A phrase reworded upstream would otherwise stop being widened silently."""
    from anton.core.tools.tool_defs import (
        _SELECT_PATH_WIDENED_PROPERTIES,
        _SELECT_PATH_WIDENED_TEXT,
        SELECT_PATH_TOOL,
        SELECT_PATH_TOOL_PICK_ONLY,
    )

    texts = " ".join(
        part for tool in (SELECT_PATH_TOOL, SELECT_PATH_TOOL_PICK_ONLY) for part in (tool.description, tool.prompt)
    )
    for old, _new in _SELECT_PATH_WIDENED_TEXT:
        assert old in texts, old
    for key, (old, _new) in _SELECT_PATH_WIDENED_PROPERTIES.items():
        assert old in SELECT_PATH_TOOL.input_schema["properties"][key]["description"], key


@pytest.mark.parametrize("kinds", [("choice", "path"), ("choice",)])
def test_with_working_folders_the_registered_tool_names_them(make_session, tmp_path, kinds):
    import copy

    from anton.core.tools.tool_defs import SELECT_PATH_TOOL, SELECT_PATH_TOOL_PICK_ONLY

    folder = tmp_path / "docs"
    folder.mkdir()
    pristine = copy.deepcopy((SELECT_PATH_TOOL, SELECT_PATH_TOOL_PICK_ONLY))

    tool = _registered_select_path(
        make_session(elicitor=_ChoiceElicitor(kinds=kinds), working_folders=(folder,))
    )

    assert str(folder.resolve()) in tool.description
    if "path" not in kinds:
        # The browse-capable prompt never names the project, so only this one changes.
        assert "working folders" in tool.prompt
    assert "the absolute path of a working folder" in tool.input_schema["properties"]["base_dir"]["description"]
    assert "working folder" in tool.input_schema["properties"]["pattern"]["description"]
    assert "working folder" in tool.input_schema["properties"]["candidates"]["description"]
    assert (SELECT_PATH_TOOL, SELECT_PATH_TOOL_PICK_ONLY) == pristine
