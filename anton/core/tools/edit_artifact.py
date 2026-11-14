"""`edit_artifact`: change an existing artifact with the artifact editor (ENG-4207).

The agent names the artifact and says what to change; :class:`ArtifactEditor`
makes the change in one patch-writing conversation (usually one model call),
checks the page in a browser when one is available, repairs it once if the
check fails, and puts every file back if it cannot make the change safely.

This handler joins that to the turn's existing page safety net instead of
building a parallel one:

* the turn's copy of each page (`capture_turn_pages`), so the render check
  compares the edited page with how the turn found it;
* the repair accounting (`REPAIR_CELLS`, `_out_of_repairs`, `_page_checks`
  via `record_page_checks`), so an edit of a page whose last check failed is a
  repair, and a page out of repairs is refused here as generate_artifact
  refuses it;
* attribution (`track_edits_since`) and the store's file list
  (`rescan_files`);
* a running live app is relaunched on its new files (`_relaunch_live_apps`),
  and its address is what the render check's URL mode loads.

Progress streams while the edit runs: the editor reports by calling a
`Progress` object, a streaming handler reports by yielding, so a queue sits
between them and is drained alongside the work.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from anton.core.tools.registry import ToolOutcome

if TYPE_CHECKING:
    from anton.core.artifact_editor.editor import Outcome
    from anton.core.session import ChatSession

_log = logging.getLogger(__name__)

EDITOR_ENV_VAR = "ANTON_ARTIFACT_EDITOR"
ROLE_ENV_VAR = "ANTON_ARTIFACT_EDITOR_ROLE"


def editor_enabled(settings) -> bool:
    """Whether `edit_artifact` is offered: `ANTON_ARTIFACT_EDITOR` (settings first, then the environment)."""
    import os

    value = getattr(settings, "artifact_editor", None)
    if value is None:
        value = os.environ.get(EDITOR_ENV_VAR, "on")
    return str(value).strip().lower() not in ("off", "0", "false", "no")


def editor_role(settings) -> str:
    import os

    value = getattr(settings, "artifact_editor_role", None) or os.environ.get(ROLE_ENV_VAR, "planning")
    return "coding" if str(value).strip().lower() == "coding" else "planning"


class _QueueProgress:
    """The editor's Progress, writing onto a queue the handler drains."""

    def __init__(self, queue: asyncio.Queue) -> None:
        self._queue = queue

    def status(self, text: str) -> None:
        pass  # transient activity lines: the tool's own spinner covers them

    def log(self, text: str) -> None:
        self._queue.put_nowait(text.strip())


async def _run_with_progress(work: asyncio.Task, queue: asyncio.Queue):
    """Yield progress lines as they arrive, until WORK finishes; then whatever is left."""
    from anton.core.tools.progress import ToolProgress

    reader: asyncio.Task | None = asyncio.ensure_future(queue.get())
    try:
        while True:
            done, _ = await asyncio.wait({reader, work}, return_when=asyncio.FIRST_COMPLETED)
            if reader in done:
                yield ToolProgress(reader.result())
                reader = asyncio.ensure_future(queue.get())
            if work in done:
                break
    finally:
        if reader is not None and not reader.done():
            reader.cancel()
    while not queue.empty():
        yield ToolProgress(queue.get_nowait())


def _check_lines(outcome: "Outcome", label: str) -> list[str]:
    check = outcome.check
    if check is None:
        return []
    if check.status == "passed":
        lines = [f"{label} — loaded in a browser ({check.channel}): no problems found."]
    elif check.status == "unverified":
        # Internal: why nothing could load the page here. Not something to relay.
        why = f" ({check.note})" if check.note else ""
        lines = [f"{label} — the page could not be checked in a browser from here{why}; the user's open viewer "
                 "reloads with the change and shows any errors itself."]
    else:
        lines = [f"{label} — the {check.channel} check failed:"]
        lines += [f"  {finding.line()}" for finding in check.findings[:8]]
    if check.status != "unverified" and check.note:
        lines += [f"{label} — {line}" for line in check.note.splitlines() if line.strip()]
    return lines


def _resolve(store, key: str):
    """(artifact, ambiguous matches, every artifact) for `key`: its slug or id, else its name or its slug
    without the id suffix, else a name whose words all appear in `key` (or the reverse).

    Accepting the name the user used saves the agent a list_artifacts round
    when it knows which artifact is meant but not its slug. Only an
    unambiguous match is taken; several come back as candidates.
    """
    artifact = store.open(key)
    if artifact is not None:
        return artifact, [], []
    found, _ = store.find([key])
    if found:
        return store.open(found[0].slug), [], []
    def words(text: str) -> list[str]:
        return [w for w in re.split(r"[^0-9a-z]+", text.casefold()) if w]

    wanted = words(key)
    if not wanted or not store.root.is_dir():
        return None, [], []
    exact, loose, everything = [], [], []
    for child in sorted(store.root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        meta = store.read_metadata(child.name)
        if meta is None:
            continue
        everything.append(meta)
        name = words(meta.name)
        base = words(child.name.rsplit("-", 1)[0])
        if wanted in (name, base):
            exact.append(meta)
        elif name and (set(name) <= set(wanted) or set(wanted) <= set(name)):
            # "the Customer Survey Results report" for "Customer Survey Results"
            loose.append(meta)
    for found in (exact, loose):
        if len(found) == 1:
            return store.open(found[0].slug), [], everything
        if found:
            return None, found, everything
    return None, [], everything


async def handle_edit_artifact(session: "ChatSession", tc_input: dict):
    """Change an existing artifact as `task` says. An async generator: progress, then the result."""
    from anton.core.artifact_editor.checkers import select_checker
    from anton.core.artifact_editor.editor import ArtifactEditor
    from anton.core.artifact_editor.workspace import ArtifactWorkspace
    from anton.core.artifacts import live_app_slots
    from anton.core.artifacts.page_copies import turn_pages_of
    from anton.core.backends.sandbox import cell_sandbox_of
    from anton.core.tools.tool_handlers import (
        REPAIR_CELLS,
        PageKey,
        _display_name,
        _relaunch_live_apps,
        _track_artifact,
        capture_turn_pages,
        images_refused,
        record_page_checks,
        render_screenshots_refused,
        resolve_artifact_store,
        snapshot_cell_artifacts,
        track_edits_since,
    )

    store = resolve_artifact_store(session)
    if store is None:
        yield ToolOutcome("Artifact store unavailable (no workspace bound to this session).", ok=False,
                          reason="store_unavailable")
        return
    slug = str(tc_input.get("slug") or "").strip()
    task = str(tc_input.get("task") or "").strip()
    if not slug:
        yield ToolOutcome("Error: `slug` (the artifact to change) is required.", ok=False, reason="missing_slug")
        return
    if not task:
        yield ToolOutcome("Error: `task` (what to change, in plain words) is required.", ok=False,
                          reason="missing_task")
        return
    artifact, ambiguous, known = _resolve(store, slug)
    if artifact is None:
        if ambiguous:
            listed = ", ".join(f"`{c.slug}` ({c.name})" for c in ambiguous[:20])
            text = f"Error: `{slug}` matches several artifacts: {listed}. Call edit_artifact again with one slug."
        elif known:
            listed = ", ".join(f"`{c.slug}` ({c.name})" for c in known[:20])
            text = (f"Error: no artifact is called `{slug}`. The artifacts here: {listed}. Call edit_artifact "
                    "again with the right slug.")
        else:
            text = f"Error: no artifact found for `{slug}`, and this workspace has none."
        yield ToolOutcome(text, ok=False, reason="artifact_not_found")
        return
    slug = artifact.slug
    spent = getattr(session, "_out_of_repairs", None)
    out_of_repairs = sorted(key.page for key in spent if key.slug == slug) if isinstance(spent, set) else []
    if out_of_repairs:
        pages = ", ".join(f"`{slug}/{page}`" for page in out_of_repairs)
        yield ToolOutcome(
            f"Error: `{slug}` has a page with no repair cells left this turn ({pages}). Do not edit it again this "
            "turn: tell the user what its last browser check found.", ok=False, reason="out_of_repairs")
        return

    folder = store.folder_for(slug)
    sandbox = cell_sandbox_of(getattr(session, "_scratchpads", None))
    kind = artifact.type
    primary = artifact.primary if artifact.primary and (Path(folder) / artifact.primary).is_file() else None
    try:
        ws = ArtifactWorkspace(folder, kind=kind, primary=primary, sandbox=sandbox)
    except ValueError as exc:
        yield ToolOutcome(f"Error: `{slug}` cannot be edited this way: {exc}", ok=False, reason="invalid_type")
        return

    await capture_turn_pages(session, store)
    before = await snapshot_cell_artifacts(session=session, store=store)
    turn_pages = turn_pages_of(session)
    key = PageKey(slug=slug, page=ws.primary or "index.html")
    previous_html = turn_pages.previous(key) if turn_pages is not None else None
    fullstack = kind in ("fullstack-stateless-app", "fullstack-stateful-app")

    live_address = None
    if fullstack and live_app_slots.live_apps_enabled() and sandbox is not None:
        async def live_address() -> str | None:
            # The app must serve the edited files before a browser loads it.
            await live_app_slots.relaunch_changed_apps(session, store, [slug])
            return await live_app_slots.running_app_address(session, folder=Path(folder))

    screenshots = not render_screenshots_refused(session)
    checker = select_checker(
        kind, slug=slug, cell_sandbox=sandbox, previous_html=previous_html, screenshot=screenshots,
        preview_checker=getattr(session, "_preview_checker", None), live_app_address=live_address,
    )
    settings = getattr(session, "_settings", None)
    queue: asyncio.Queue = asyncio.Queue()
    editor = ArtifactEditor(
        ws, getattr(session, "_llm"), checker, _QueueProgress(queue), role=editor_role(settings), slug=slug,
        name=artifact.name, images=screenshots and not images_refused(session),
        conversation_id=str(getattr(session, "_session_id", "") or ""),
    )
    checks_before = getattr(session, "_page_checks", None)
    was_failing = isinstance(checks_before, dict) and checks_before.get(key) == "failed"

    work = asyncio.ensure_future(editor.edit(task))
    try:
        async for line in _run_with_progress(work, queue):
            yield line
    finally:
        if not work.done():
            # The turn was abandoned mid-edit: the editor restores its snapshot on cancellation.
            work.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await work
    outcome = work.result()

    with contextlib.suppress(Exception):
        store.rescan_files(slug)
    after = await snapshot_cell_artifacts(session=session, store=store)
    label = f"{slug}/{key.page}"
    name = _display_name(artifact, slug)

    if outcome.status == "restored":
        failed_check = outcome.check is not None and outcome.check.status == "failed"
        if was_failing:
            used = getattr(session, "_repair_cells", {}).get(key, 0) if isinstance(
                getattr(session, "_repair_cells", None), dict) else 0
            record_page_checks(
                session, checks={key: "failed"}, repaired=frozenset({key}),
                out_of_repairs=frozenset({key}) if used + 1 >= REPAIR_CELLS else frozenset(),
            )
        lines = [f"Error: the edit of “{name}” (`{slug}`) could not be made safely, so every file is back as it "
                 f"was before this call. Why: {outcome.error}."]
        lines += _check_lines(outcome, label)
        if failed_check:
            lines.append(
                "The change failed its browser check twice. Do not call edit_artifact for this page again this "
                "turn. Tell the user the change could not be made safely, what the check found, and that the "
                "artifact is unchanged.")
            reason = "edit_check_failed"
        else:
            lines.append(
                "Nothing was changed. You may call edit_artifact once more with a more specific task (name the "
                "exact element, label or function), or tell the user what could not be done.")
            reason = "edit_not_applied"
        yield ToolOutcome("\n".join(lines), ok=False, reason=reason)
        return

    if outcome.status == "answered":
        yield ToolOutcome(
            f"The artifact editor changed nothing in “{name}” (`{slug}`) and said:\n{outcome.reply}\n\n"
            "Relay that to the user. If the request needs data or work the artifact's files cannot provide, say "
            "so; do not make the change another way unless the user asks.", ok=True)
        return

    _track_artifact(session, store, slug, summary=f"Edited: {task[:160]}")
    track_edits_since(session, store, before, after=after)
    live_lines = await _relaunch_live_apps(session, store, before, after=after)
    check = outcome.check
    status = check.status if check is not None else "unverified"
    # A page-level record only for a page; non-web artifacts have none. An
    # unverified check of a page that was not failing records nothing: on a
    # sandboxed deployment every recorded non-pass becomes a "may not display
    # correctly" line under the reply, and the user's own viewer shows the
    # page anyway. A failing page's record is replaced, so a repair the check
    # could not see still counts as one.
    unremarkable = status == "unverified" and not was_failing
    if kind in ("html-app", "fullstack-stateless-app", "fullstack-stateful-app") and not unremarkable:
        record_page_checks(
            session, checks={key: status}, repaired=frozenset({key}) if was_failing else frozenset(),
        )
        if turn_pages is not None and check is not None and check.comparison is not None and status != "unverified":
            with contextlib.suppress(Exception):
                turn_pages.record({key: check.comparison}, edited=[key])

    changed = ", ".join(outcome.files) or "no file"
    calls = outcome.model_calls
    lines = [
        f"Edited “{name}” (`{slug}`): {changed}, in {calls} model call{'s' if calls != 1 else ''}"
        + (" (the editor rewrote the file whole)" if outcome.rewrote else "") + ".",
    ]
    if outcome.reply:
        lines.append(f"What changed: {outcome.reply}")
    lines += _check_lines(outcome, label)
    lines += live_lines
    if status == "passed":
        lines.append(
            "The change is applied and a browser loaded the page without finding a problem. Do NOT re-read, "
            "re-parse or re-verify the files: report the change to the user, and act only on problems they "
            "report. Say \"browser-verified\" only of what the check names.")
    else:
        lines.append(
            "The change is applied. It could not be checked in a browser from here; the user's open viewer "
            "reloads with it and shows any errors there. Tell the user what changed. Do not bring up checks, "
            "verification or browser availability (mention a check only when it found a problem), and never say "
            "the change was tested or verified. Do not open the files looking for problems.")
    yield ToolOutcome("\n".join(lines), ok=True)


EDIT_ARTIFACT_PROMPT = """EDITING AN EXISTING ARTIFACT:
For any change to an artifact that already exists — a color, a label, a chart \
type, a new filter or sort, a bug the user reports, a repair after a failed \
browser check — call edit_artifact(slug, task) with the user's request in your \
own clear words. `slug` may be the artifact's name as the user calls it, so \
no list_artifacts call is needed first. It reads the artifact, edits only what the request needs, \
checks the page in a browser and repairs it, all in one call. Do NOT read the \
files with read_text_file or patch them in a scratchpad cell first, and do not \
rebuild the artifact with generate_artifact for a change like that. Use \
generate_artifact(slug) only when the user asks for a different artifact \
(a new page structure, new data sources, a rebuild). If edit_artifact says it \
could not make the change, follow its instruction."""


def _tool():
    from anton.core.tools.tool_defs import ToolDef

    return ToolDef(
        name="edit_artifact",
        description=(
            "Change an existing artifact: restyle it, rename or rewrite labels and text, add or change a filter, "
            "sort, toggle, column or chart, or fix a bug the user reports or a failed browser check. It reads the "
            "artifact's files, edits only what the task needs (usually in one model call), checks the page in a "
            "browser when one is available and repairs it, and puts every file back if it cannot make the change "
            "safely. Use it instead of reading the files and patching them in a scratchpad cell, and instead of "
            "generate_artifact for any change short of a rebuild.\n\n"
            "Inputs:\n"
            "- `slug`: the artifact to change: its slug, or its exact name when you do not know the slug "
            "(no need to call list_artifacts first).\n"
            "- `task`: what to change, in plain words, with every concrete requirement from the user (exact "
            "labels, colors, names, behaviour). One task may ask for several changes."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "slug": {"type": "string", "description": "The artifact's slug, or its exact name."},
                "task": {"type": "string", "description": "What to change, in plain words."},
            },
            "required": ["slug", "task"],
        },
        handler=handle_edit_artifact,
        prompt=EDIT_ARTIFACT_PROMPT,
    )


EDIT_ARTIFACT_TOOL = _tool()


# ── routing: where the agent is told how to change an artifact ─────────

_GENERATE_EDIT_SENTENCE = (
    "For an EDIT, pass the existing slug instead of name so the existing artifact is updated."
)
_GENERATE_EDIT_WITH_EDITOR = (
    "To CHANGE an existing artifact (a style, a label, a filter, a chart, a bug fix, a failed browser check), "
    "call edit_artifact instead; pass an existing slug here only to rebuild it."
)

# (old, new) pairs applied to the system prompt's ARTIFACTS section when
# `edit_artifact` is offered. Each `old` is a sentence of one of the
# section's variants; `tests/test_edit_artifact_tool.py` checks that every
# variant has its edit sentence replaced.
SECTION_ROUTES: tuple[tuple[str, str], ...] = (
    # Finding the artifact: edit_artifact takes its name, so no list_artifacts round first.
    (
        "Find the existing artifact with list_artifacts, using match when its slug or\nid is known.",
        "To change an artifact the user names, call edit_artifact with that name as\n`slug` directly; "
        "otherwise find it with list_artifacts, using match when its\nslug or id is known.",
    ),
    (
        "4. EDITING an existing artifact: call `list_artifacts` to find it; if you have its id, pass the id in "
        "`match`.",
        "4. EDITING an existing artifact: when the user names it, call `edit_artifact` with that name as `slug` "
        "directly; otherwise call `list_artifacts` to find it (pass its id in `match` when you have it).",
    ),
    # static_artifact_prompt.ARTIFACTS (the single-call generator's section)
    (
        "For a small edit, read and patch only the relevant content in place;\n"
        "report_tools.update/insert can update an existing report. For a substantial\n"
        "rebuild, use generate_artifact with the existing slug and requested change.",
        "For any change short of a rebuild (a style, label, text, filter, sort,\n"
        "chart or bug fix, or a repair after a failed browser check), call\n"
        "edit_artifact(slug, task): it edits, checks and repairs the page in one call;\n"
        "do not read and patch the files in the scratchpad first, and pass the\n"
        "artifact's name as `slug` when you do not know the slug. For a substantial\n"
        "rebuild, use generate_artifact with the existing slug and requested change.",
    ),
    # prompts.ARTIFACTS_PROMPT step 4 and its browser_check_failed exception
    (
        "Read the files you will change with `read_text_file`, then edit them in the scratchpad.",
        "Change it with `edit_artifact(slug, task)`, which edits, checks and repairs it in one call; do not "
        "read and patch its files in the scratchpad first.",
    ),
    (
        "you repair it in a scratchpad cell as its `instruction` says.",
        "you repair it with `edit_artifact` as its `instruction` says.",
    ),
)


def route_edits_to_editor(tool):
    """GENERATE_ARTIFACT's definition with its "for an EDIT" sentence pointing at edit_artifact."""
    from dataclasses import replace

    return replace(
        tool,
        description=tool.description.replace(_GENERATE_EDIT_SENTENCE, _GENERATE_EDIT_WITH_EDITOR),
        prompt=(tool.prompt or "").replace(_GENERATE_EDIT_SENTENCE, _GENERATE_EDIT_WITH_EDITOR) or tool.prompt,
    )


def route_section_to_editor(section: str) -> str:
    """The ARTIFACTS prompt section with its edit instructions pointing at edit_artifact."""
    for old, new in SECTION_ROUTES:
        section = section.replace(old, new)
    return section
