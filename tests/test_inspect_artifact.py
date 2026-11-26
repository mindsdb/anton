"""ENG-4684: inspect_artifact, the agent's view of what the user did in an inline page.

The page reports through two files the Cowork preview shim writes into the
artifact folder: `state.json` (anton.setState) and `snapshot.json` (form
controls and visible text, captured even when the page never calls setState).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from anton.core.artifacts import ArtifactStore
from anton.core.artifacts.internal_files import (
    INLINE_SNAPSHOT_FILENAME,
    INLINE_STATE_FILENAME,
    NON_CONTENT_NAMES,
)
from anton.core.artifacts.listing import SUMMARY_FIELDS, render_listing
from anton.core.artifacts.page_state import (
    NOTHING_REPORTED,
    PAGE_FILE_MAX_BYTES,
    STATE_SHOWN_MAX_CHARS,
    TEXT_SHOWN_MAX_CHARS,
    relative_time,
)
from anton.core.llm.prompts import IN_REPLY_VISUALS_PROMPT
from anton.core.tools import tool_defs
from anton.core.tools.tool_handlers import handle_create_artifact, handle_inspect_artifact
from tests.test_artifact_turn_tracking import FakeSession

SNAPSHOT = {
    "controls": [
        {"label": "Exponent p", "name": "p", "type": "range", "value": "0.47"},
        {"label": "", "name": "show_grid", "type": "checkbox", "value": True},
    ],
    "text": "Power curve\n  y = x^p   p = 0.47",
}


@pytest.fixture
def session(tmp_path: Path) -> FakeSession:
    return FakeSession(tmp_path / "artifacts")


async def _page(session: FakeSession, *, kind: str = "html-app", display: str | None = "inline") -> Path:
    args = {"name": "Power curve", "description": "d", "type": kind}
    if display:
        args["display"] = display
    await handle_create_artifact(session, args)
    store = ArtifactStore(session._workspace.artifacts_dir)
    (artifact,) = [a for a in store.list() if a.name == "Power curve"]
    return store.root / artifact.slug


async def _inspect(session: FakeSession, folder: Path):
    return await handle_inspect_artifact(session, {"slug": folder.name})


def _write(folder: Path, name: str, body) -> None:
    (folder / name).write_text(body if isinstance(body, str) else json.dumps(body), encoding="utf-8")


def test_snapshot_is_content_named_snapshot_json():
    assert INLINE_SNAPSHOT_FILENAME == "snapshot.json"
    assert INLINE_SNAPSHOT_FILENAME not in NON_CONTENT_NAMES


async def test_no_files_says_the_page_has_not_reported(session):
    folder = await _page(session)
    out = await _inspect(session, folder)
    assert out.ok is True
    assert out.content.splitlines() == [
        f"## {folder.name}",
        "title: Power curve",
        "display: inline",
        NOTHING_REPORTED,
    ]
    assert "once the user changes one" in NOTHING_REPORTED
    assert "only if the page calls anton.setState" in NOTHING_REPORTED


async def test_state_only(session):
    folder = await _page(session)
    _write(folder, INLINE_STATE_FILENAME, '{\n  "people": [{"name": "Ana Li", "picked": true}]\n}')
    lines = (await _inspect(session, folder)).content.splitlines()
    assert lines[3] == "last saved: just now"
    assert lines[4] == 'page state (state.json, from anton.setState): {"people":[{"name":"Ana Li","picked":true}]}'
    assert lines[5] == "controls: none reported yet (the user has not changed a form control)"
    assert len(lines) == 6


async def test_snapshot_only(session):
    folder = await _page(session)
    _write(folder, INLINE_SNAPSHOT_FILENAME, SNAPSHOT)
    lines = (await _inspect(session, folder)).content.splitlines()
    assert lines[4] == "page state: none (the page has not called anton.setState)"
    assert lines[5:] == [
        "controls (2):",
        "  Exponent p (range): 0.47",
        "  show_grid (checkbox): true",
        "visible text: Power curve y = x^p p = 0.47",
    ]


async def test_both_and_last_saved_is_the_newest(session):
    folder = await _page(session)
    _write(folder, INLINE_STATE_FILENAME, {"p": 0.47})
    _write(folder, INLINE_SNAPSHOT_FILENAME, SNAPSHOT)
    old = time.time() - 3 * 3600
    os.utime(folder / INLINE_STATE_FILENAME, (old, old))
    recent = time.time() - 5 * 60
    os.utime(folder / INLINE_SNAPSHOT_FILENAME, (recent, recent))
    text = (await _inspect(session, folder)).content
    assert "last saved: 5 minutes ago" in text
    assert 'page state (state.json, from anton.setState): {"p":0.47}' in text
    assert "  Exponent p (range): 0.47" in text


async def test_a_large_state_is_cut_and_says_so(session):
    folder = await _page(session)
    _write(folder, INLINE_STATE_FILENAME, {"rows": ["x" * 50] * 200})
    line = next(l for l in (await _inspect(session, folder)).content.splitlines() if l.startswith("page state"))
    assert "… (cut:" in line and "characters in all)" in line
    assert len(line) < STATE_SHOWN_MAX_CHARS + 200


async def test_long_visible_text_is_cut(session):
    folder = await _page(session)
    _write(folder, INLINE_SNAPSHOT_FILENAME, {"controls": [], "text": "word " * 1000})
    line = (await _inspect(session, folder)).content.splitlines()[-1]
    assert line.startswith("visible text: ") and line.endswith("… (cut)")
    assert len(line) <= TEXT_SHOWN_MAX_CHARS + 30


async def test_an_oversized_file_is_not_read(session):
    folder = await _page(session)
    _write(folder, INLINE_SNAPSHOT_FILENAME, json.dumps({"text": "y" * PAGE_FILE_MAX_BYTES}))
    text = (await _inspect(session, folder)).content
    assert "controls: snapshot.json" in text and "over the 256 KB cap; not read" in text
    assert "yyyy" not in text


async def test_bad_json_is_reported_not_raised(session):
    folder = await _page(session)
    _write(folder, INLINE_STATE_FILENAME, "{not json")
    _write(folder, INLINE_SNAPSHOT_FILENAME, "[1, 2")
    out = await _inspect(session, folder)
    assert out.ok is True
    assert "page state: state.json 9 bytes, not valid JSON" in out.content
    assert "controls: snapshot.json 5 bytes, not valid JSON" in out.content


async def test_a_wrong_shape_snapshot_shows_no_controls(session):
    folder = await _page(session)
    _write(folder, INLINE_SNAPSHOT_FILENAME, {"controls": "nope", "text": 3})
    assert (await _inspect(session, folder)).content.splitlines()[-1] == "controls: none"


async def test_a_symlink_is_refused(session, tmp_path):
    folder = await _page(session)
    secret = tmp_path / "secret.json"
    secret.write_text('{"token": "s3cret"}', encoding="utf-8")
    (folder / INLINE_STATE_FILENAME).symlink_to(secret)
    (folder / INLINE_SNAPSHOT_FILENAME).symlink_to(secret)
    text = (await _inspect(session, folder)).content
    assert "s3cret" not in text
    assert NOTHING_REPORTED in text


async def test_a_control_value_cannot_fake_a_heading(session):
    folder = await _page(session)
    _write(folder, INLINE_SNAPSHOT_FILENAME,
           {"controls": [{"label": "x\n## other", "type": "text", "value": "a\n## b"}], "text": ""})
    assert "\n## other" not in (await _inspect(session, folder)).content
    assert "\n## b" not in (await _inspect(session, folder)).content


async def test_a_non_html_app_is_refused(session):
    folder = await _page(session, kind="document", display=None)
    out = await _inspect(session, folder)
    assert (out.ok, out.reason) == (False, "invalid_type")
    assert "only html-app" in out.content


async def test_missing_and_unknown_slugs(session):
    out = await handle_inspect_artifact(session, {})
    assert (out.ok, out.reason) == (False, "missing_slug")
    out = await handle_inspect_artifact(session, {"slug": "nope"})
    assert (out.ok, out.reason) == (False, "artifact_not_found")


async def test_a_full_page_html_app_still_inspects(session):
    folder = await _page(session, display=None)
    assert "display: - (a full page" in (await _inspect(session, folder)).content


@pytest.mark.parametrize("seconds, expected", [
    (5, "just now"), (60, "1 minute ago"), (600, "10 minutes ago"),
    (3600, "1 hour ago"), (7 * 3600, "7 hours ago"), (2 * 86400, "2 days ago"),
])
def test_relative_time(seconds, expected):
    assert relative_time(1000.0, now=1000.0 + seconds) == expected


# ── listing ─────────────────────────────────────────────────────


async def test_listing_counts_captured_controls(session):
    folder = await _page(session)
    _write(folder, INLINE_STATE_FILENAME, {"p": 0.47})
    _write(folder, INLINE_SNAPSHOT_FILENAME, SNAPSHOT)
    store = ArtifactStore(session._workspace.artifacts_dir)
    text = render_listing([(store.root, store.list())], SUMMARY_FIELDS)
    assert 'state: {"p":0.47}' in text
    assert "controls: 2 (snapshot.json; inspect_artifact shows their values)" in text


async def test_listing_has_no_controls_line_without_a_snapshot(session):
    await _page(session)
    store = ArtifactStore(session._workspace.artifacts_dir)
    assert "controls:" not in render_listing([(store.root, store.list())], SUMMARY_FIELDS)


# ── tool and prompt ─────────────────────────────────────────────


def test_the_tool_takes_a_slug():
    tool = tool_defs.INSPECT_ARTIFACT_TOOL
    assert tool.name == "inspect_artifact"
    assert tool.input_schema["required"] == ["slug"]
    assert "state.json" in tool.description and "snapshot.json" in tool.description


def test_the_prompt_sends_the_agent_to_inspect_artifact():
    flat = " ".join(IN_REPLY_VISUALS_PROMPT.split())
    for needle in (
        "When the user refers to something they did or picked in an inline artifact or widget",
        '"what did I pick", "use these settings", "go with what I selected"',
        "call `inspect_artifact` first and answer from it",
        "Never claim you can't see the page.",
    ):
        assert needle in flat, needle
    assert "read it for the current selections" not in flat


def test_the_prompt_fixes_a_page_that_hides_the_pick():
    flat = " ".join(IN_REPLY_VISUALS_PROMPT.split())
    for needle in (
        "If it shows nothing that answers",
        "such as a clicked chart bar",
        "update the page in place (same slug) so it reports that choice through anton.setState",
        "tell the user plainly that you couldn't see that choice, have fixed the page, "
        "and ask them to make the pick again.",
    ):
        assert needle in flat, needle


def test_the_skill_asks_for_labelled_controls():
    skill = (Path(tool_defs.__file__).parents[1] / "memory" / "builtin_skills"
             / "build-html-dashboard" / "SKILL.md").read_text(encoding="utf-8")
    assert "captures the page's form controls" in skill
    assert "`<label for>`" in skill
    assert "use `anton.setState` for any choice that is not a form control" in skill
