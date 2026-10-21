from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NamedTuple

from anton.core.backends.base import Cell
from anton.core.backends.sandbox import CellSandbox, cell_sandbox_of
from anton.core.tools.registry import ToolOutcome
from anton.core.tools.side_effect import SideEffectResult, now_iso
from anton.core.utils.scratchpad import (
    SCRATCHPAD_ACTIONS,
    scratchpad_action,
    prepare_scratchpad_exec,
    format_cell_result,
    observe_scratchpad_cell,
    cell_failure_reason,
    install_call_installed_something,
    refuse_pad_name,
    reject_invalid_packages,
    send_package_install_event,
)

if TYPE_CHECKING:
    from anton.chat_session import ChatSession
    from anton.core.artifacts.render_check import RenderScreenshot, RenderStatus


_log = logging.getLogger(__name__)


def preliminary_cell(
    *,
    code: str,
    description: str,
    estimated_time: str,
    estimated_seconds: int,
) -> Cell:
    """The cell pre-execute observers see: the code and its intent, no output yet.

    Both exec paths (`handle_scratchpad` and `ChatSession`'s inline streaming
    exec) build it here, so observers get the same shape from either one.
    """
    return Cell(
        code=code,
        stdout="",
        stderr="",
        error=None,
        description=description,
        estimated_time=estimated_time or str(estimated_seconds),
    )


async def _fire_pre_execute(session: "ChatSession", cell: Cell) -> None:
    """Notify pre-execute observers (e.g. cerebellum) before a cell runs.

    Best-effort: a buggy observer never kills a cell. The list of
    observers is owned by the session — typically populated in
    ChatSession.__init__. Empty list (or attribute missing) means no
    observers and this is a no-op.
    """
    observers = getattr(session, "_scratchpad_observers", None) or []
    for obs in observers:
        on_pre = getattr(obs, "on_pre_execute", None)
        if on_pre is None:
            continue
        try:
            await on_pre(cell)
        except Exception as exc:
            _log.warning(
                "scratchpad pre-execute observer %s failed: %s",
                type(obs).__name__,
                exc,
            )


async def _fire_post_execute(session: "ChatSession", cell: Cell) -> None:
    """Notify post-execute observers (e.g. cerebellum) after a cell finishes.

    Same best-effort contract as `_fire_pre_execute`.
    """
    observers = getattr(session, "_scratchpad_observers", None) or []
    for obs in observers:
        on_post = getattr(obs, "on_post_execute", None)
        if on_post is None:
            continue
        try:
            await on_post(cell)
        except Exception as exc:
            _log.warning(
                "scratchpad post-execute observer %s failed: %s",
                type(obs).__name__,
                exc,
            )


def resolve_artifact_store(session: "ChatSession"):
    """Return the artifact store rooted at the session's workspace.

    Returns None when the session has no workspace (e.g. CLI calls
    that don't go through `resolve_workspace`). Tool handlers fall
    back to a clear error string in that case rather than raising.
    """
    workspace = getattr(session, "_workspace", None)
    if workspace is None:
        return None
    from anton.core.artifacts import ArtifactStore
    return ArtifactStore(workspace.artifacts_dir)


def _track_artifact(session: "ChatSession", store, slug: str, *, summary: str = "") -> None:
    """Record that THIS turn created, generated or updated `slug`.

    Two records, for two different readers:

    * `session._artifacts_touched` — in-memory, per-turn. The host reads it
      after the turn to build the turn's artifact cards. This is what makes
      attribution correct without diffing the artifacts directory: several
      turns can share one project-wide artifacts folder and each still knows
      exactly what it touched.
    * `provenance` in the artifact's own `metadata.json`, via
      `record_turn()` — durable, and the only record that survives the
      process. Answers "which conversations have ever worked on this?" for
      any later reader.

    Best-effort: attribution is bookkeeping, so a failure here must never
    fail the tool call the agent actually made.
    """
    try:
        session._artifacts_touched.add(slug)
    except AttributeError:
        # A session predating this field (or a test double) still gets
        # durable provenance below.
        pass
    conversation_id = str(getattr(session, "_session_id", "") or "")
    if not conversation_id:
        # No host conversation to attribute to (a bare CLI session). The
        # in-memory set above is still useful; provenance would be a row
        # keyed by nothing.
        return
    try:
        store.record_turn(
            slug,
            conversation_id=conversation_id,
            conversation_title=None,
            turn_index=getattr(session, "_turn_count", 0) + 1,
            summary=summary,
            files_touched=[],
        )
    except Exception:
        _log.warning("could not record turn provenance for artifact %s", slug, exc_info=True)


class FileStamp(NamedTuple):
    """When a content file last changed, in nanoseconds, and its size."""

    mtime_ns: int
    size: int


@dataclass(frozen=True)
class ArtifactFiles:
    """An artifact folder's content files at one moment, each by its path
    inside the folder (`index.html`, `static/app.js`).

    Two snapshots differ when a cell wrote, replaced, renamed, copied in or
    deleted a file. `os.replace` of an older draft onto `index.html` keeps
    the draft's mtime, so the folder's newest mtime does not move, but the
    stamp under `index.html` changes.
    """

    stamps: dict[str, FileStamp] = field(default_factory=dict)

    def edited_since(self, *, before: "ArtifactFiles", rel: str) -> bool:
        """Whether the file at `rel` is new, or changed, since `before`."""
        return before.stamps.get(rel) != self.stamps.get(rel)


def _artifact_files(folder: Path) -> ArtifactFiles:
    """The stamps of an artifact folder's content files; empty if none."""
    from anton.core.artifacts.store import iter_content_files

    return ArtifactFiles(stamps={
        path.relative_to(folder).as_posix(): FileStamp(mtime_ns=st.st_mtime_ns, size=st.st_size)
        for path, st in iter_content_files(folder)
    })


def snapshot_existing_artifacts(store) -> dict[str, ArtifactFiles]:
    """slug -> its content files, for every artifact folder that exists right now."""
    root = store.root
    if not root.is_dir():
        return {}
    snapshot: dict[str, ArtifactFiles] = {}
    for child in root.iterdir():
        if child.is_dir() and (child / "metadata.json").is_file():
            snapshot[child.name] = _artifact_files(child)
    return snapshot


# Bounded: never lint a huge file inline on the agent's turn.
_ARTIFACT_LINT_SIZE_CEILING = 10 * 1024 * 1024  # 10 MB

# The render checks after one cell share this budget: the agent waits for
# them before it sees the cell's result. A page that would start with less
# than `_RENDER_CHECK_MIN_SECONDS` left is reported as not checked.
_RENDER_CHECK_CELL_BUDGET_SECONDS = 45.0
_RENDER_CHECK_MIN_SECONDS = 5.0


# Said once per file when a check that would start a browser or LibreOffice
# is skipped because agent code runs in a sandbox on this host.
_SANDBOX_BROWSER_SKIP = (
    "not checked in a browser: agent code runs in a sandbox on this deployment, "
    "which starts no browser"
)
_SANDBOX_OFFICE_SKIP = "LibreOffice check skipped: agent code runs in a sandbox on this deployment"
# Said on each page of a fullstack app a cell edits, which the render-check
# service does not load (`_check_pages`).
_FULLSTACK_TYPES = frozenset({"fullstack-stateless-app", "fullstack-stateful-app"})
_FULLSTACK_PAGE_SKIP = (
    "not checked in a browser: a fullstack app's page calls its backend, which "
    "the render check does not run"
)


class PageKey(NamedTuple):
    """One page of one artifact: its slug and its path inside the artifact
    folder (`index.html`, `static/index.html`)."""

    slug: str
    page: str


@dataclass(frozen=True)
class ArtifactLint:
    """What the checks after one cell found in the artifacts it changed."""

    # `<slug>/<file> — <finding>` lines for the `[artifact lint]` block.
    messages: list[str] = field(default_factory=list)
    # The first page that failed the render check, or the first page it
    # loaded when none failed, as the app preview shows it, and that page's
    # `<slug>/<file>`.
    screenshot: "RenderScreenshot | None" = None
    screenshot_page: str = ""
    # How each page this cell checked fared in a browser.
    page_checks: "dict[PageKey, RenderStatus]" = field(default_factory=dict)
    # The pages whose unverified check listed what it still found.
    findings_listed: frozenset[PageKey] = frozenset()


_STATUS_RANK = {"passed": 0, "unverified": 1, "failed": 2}


def _worse(a: "RenderStatus | None", b: "RenderStatus") -> "RenderStatus":
    return b if a is None or _STATUS_RANK[b] > _STATUS_RANK[a] else a


def _artifact_linters(*, cell_sandbox: CellSandbox | None = None) -> dict[str, Callable[[Path], list[str] | None]]:
    """suffix -> checker. `None` means it could not run at all (e.g. no
    LibreOffice); `[]` means it ran and found nothing; a non-empty list is
    real findings — only that last case is ever appended to the agent-facing
    message list.

    Add a format by adding one entry here. Only `.xlsx`/`.pptx`/`.docx` are
    registered here and `.html` has its own pass (`_check_pages`) — every
    other extension is silently unchecked, which is honest as-is: nothing
    ever claimed to validate a `.csv`.

    A `cell_sandbox` policy skips the checks that start LibreOffice on the
    agent's file (each of them also refuses under it); the in-process lints
    still run. Where the skipped tool is installed, the result says the check
    was skipped.
    """
    from anton.core.artifacts.office_open_check import check_opens_via_office
    from anton.core.artifacts.ooxml_lint import lint_ooxml
    from anton.core.artifacts.xlsx_lint import lint_xlsx
    from anton.core.artifacts.xlsx_office_check import _discover_office, check_xlsx_via_office

    sandboxed = cell_sandbox is not None

    def _office_skip() -> list[str]:
        return [_SANDBOX_OFFICE_SKIP] if _discover_office() is not None else []

    def _xlsx_linter(path: Path) -> list[str] | None:
        # Both always run — LibreOffice recalculates and catches any
        # formula error; the structural lint runs regardless (not only as
        # a fallback), because it names the actual fix ("missing cross-
        # sheet prefix") where the oracle only names the symptom
        # ("#VALUE!"), and some cells only the oracle catches at all
        # (a genuine #REF!/#DIV/0! unrelated to same-sheet circularity).
        if sandboxed:
            structural_findings = lint_xlsx(path)
            skipped = _office_skip()
            if structural_findings is None and not skipped:
                return None
            return [f.message() for f in (structural_findings or [])] + skipped
        office_findings = check_xlsx_via_office(path, cell_sandbox=cell_sandbox)
        structural_findings = lint_xlsx(path)

        if office_findings is None and structural_findings is None:
            # Neither could even check this file (no office AND it doesn't
            # parse with openpyxl either) — nothing to report.
            return None

        # A cell both catch gets the structural message only: it's the more
        # actionable one, and reporting the same cell twice would just be
        # noise. A cell only the oracle catches still gets through.
        structural_cells = {(f.sheet, f.cell) for f in (structural_findings or [])}
        messages = [f.message() for f in (structural_findings or [])]
        messages += [
            f.message() for f in (office_findings or [])
            if (f.sheet, f.cell) not in structural_cells
        ]
        return messages

    def _ooxml_linter(kind: str) -> Callable[[Path], list[str] | None]:
        def _lint(path: Path) -> list[str] | None:
            # Structure first (ENG-2175): a truncated or hand-rolled package
            # is named without any Office install. Only a package that is
            # structurally sound goes on to LibreOffice, since a 30-second
            # export of a file already known to be broken adds nothing.
            structural_findings = lint_ooxml(path, kind)
            if structural_findings:
                return [f.message() for f in structural_findings]
            if sandboxed:
                skipped = _office_skip()
                return skipped if structural_findings is not None or skipped else None
            office_findings = check_opens_via_office(path, cell_sandbox=cell_sandbox)
            if structural_findings is None and office_findings is None:
                return None
            return [f.message() for f in (office_findings or [])]

        return _lint

    return {
        ".xlsx": _xlsx_linter,
        ".pptx": _ooxml_linter("pptx"),
        ".docx": _ooxml_linter("docx"),
    }


@dataclass(frozen=True)
class _ChangedPage:
    """An `.html` content file of an artifact the cell changed."""

    slug: str
    path: Path
    # The cell wrote, replaced or added this file itself.
    edited: bool
    size: int

    def key(self, store) -> PageKey:
        return PageKey(slug=self.slug, page=self.path.relative_to(store.root / self.slug).as_posix())


def _primary_name(store, slug: str) -> str | None:
    """The artifact's primary file as its metadata names it, or None.

    `read_metadata`, not `open`: `open` reconciles the file list and may
    rewrite the metadata, which a check must not do.
    """
    metadata = store.read_metadata(slug)
    return metadata.primary if metadata is not None else None


def _check_pages(
    store,
    pages: list[_ChangedPage],
    *,
    cell_sandbox: CellSandbox | None,
    screenshot: bool,
    findings_listed_earlier: frozenset[PageKey] = frozenset(),
) -> ArtifactLint:
    """Load the changed artifacts' pages in a browser.

    With the render-check service configured: the pages this cell edited,
    and the main page of a changed artifact other than a fullstack app
    whose pages the cell left alone (it rewrote only a file the page loads,
    say, or deleted one), each artifact's main page first, within one budget
    per cell. An artifact's main page is its primary, else its
    `index.html`, else its first page by name. The screenshot kept is the
    first failed page's, or the first page's when none failed; with
    `screenshot` False the service is asked for none. A fullstack app's
    page is not sent: it calls its backend, which the service does not
    run, and the service fails every request to the page's own origin.
    generate_artifact loads that page through its running backend instead
    (`verifiers.verify_app_live`). Otherwise the local Electron browser
    loads the same pages (`lint_html`), and a sandbox policy, which starts
    no browser, says so on each of them.

    Every page checked gets a line: its findings, that the service passed
    it, or that it was not checked in a browser and why (no browser here, a
    sandbox, a fullstack app, the service could not vouch for it, the budget
    ran out, the page is too large), with what an unverified check still
    found. The local browser adds no line for a page that passed: the
    result stays in history and is resent on every model call, and a page
    it could not load still gets a line, so silence there means a pass.
    Those findings are listed only for a page not in
    `findings_listed_earlier`, the pages whose findings an earlier
    unverified check this turn already listed, so the agent fixes them once
    and does not edit a page no check can vouch for again on every cell. A
    check that listed nothing (the service was busy, the cell's time ran
    out) leaves the next one free to list. The pages this call lists
    findings for come back in `findings_listed`.
    """
    from anton.core.artifacts import render_check
    from anton.core.artifacts.html_lint import discover_browser, lint_html

    messages: list[str] = []
    page_checks: "dict[PageKey, RenderStatus]" = {}
    findings_listed: set[PageKey] = set()
    shot = None
    shot_page = ""

    # `read_metadata`, not `open`: `open` reconciles the file list and may
    # rewrite the metadata, which a check must not do.
    metadata = {slug: store.read_metadata(slug) for slug in {page.slug for page in pages}}

    def _type(page: _ChangedPage) -> str | None:
        return metadata[page.slug].type if metadata[page.slug] is not None else None

    def _primary(page: _ChangedPage) -> str | None:
        return metadata[page.slug].primary if metadata[page.slug] is not None else None

    def _order(page: _ChangedPage) -> tuple:
        rel = page.key(store).page
        return (rel != _primary(page), rel != "index.html", rel)

    edited_slugs = {page.slug for page in pages if page.edited}
    # Each artifact's main page, its first by `_order`: its primary, else
    # `index.html`, else its first page by name.
    main_page: dict[str, _ChangedPage] = {}
    for page in sorted(pages, key=_order):
        main_page.setdefault(page.slug, page)
    to_check = sorted((
        page for page in pages
        if page.edited or (
            page.slug not in edited_slugs and _type(page) not in _FULLSTACK_TYPES
            and page == main_page[page.slug]
        )
    ), key=_order)

    if render_check.configured_url() is None:
        for page in to_check:
            label = f"{page.slug}/{page.path.name}"
            if cell_sandbox is not None:
                messages.append(f"{label} — {_SANDBOX_BROWSER_SKIP}")
                page_checks[page.key(store)] = "unverified"
                continue
            if page.size > _ARTIFACT_LINT_SIZE_CEILING:
                messages.append(
                    f"{label} — not checked in a browser: the page is over "
                    f"{_ARTIFACT_LINT_SIZE_CEILING // (1024 * 1024)} MB"
                )
                page_checks[page.key(store)] = "unverified"
                continue
            findings = lint_html(page.path, cell_sandbox=None)
            if findings is None:
                reason = (
                    "no browser on this deployment" if discover_browser() is None
                    else "the local browser check gave no answer"
                )
                messages.append(f"{label} — not checked in a browser: {reason}")
                page_checks[page.key(store)] = "unverified"
                continue
            messages += [f"{label} — {f.message()}" for f in findings]
            page_checks[page.key(store)] = "failed" if findings else "passed"
        return ArtifactLint(messages=messages, page_checks=page_checks)

    # Whether `shot` is a failed page's: until one is, every page is asked
    # for a screenshot, so the agent sees the page it has to fix.
    shot_failed = False
    deadline = time.monotonic() + _RENDER_CHECK_CELL_BUDGET_SECONDS
    for page in to_check:
        label = f"{page.slug}/{page.path.name}"
        if _type(page) in _FULLSTACK_TYPES:
            messages.append(f"{label} — {_FULLSTACK_PAGE_SKIP}")
            page_checks[page.key(store)] = "unverified"
            continue
        remaining = deadline - time.monotonic()
        if remaining < _RENDER_CHECK_MIN_SECONDS:
            outcome = render_check.RenderCheckUnavailable(
                reason="the browser checks after this cell ran out of time before this page"
            )
        else:
            outcome = render_check.check_file(
                page.path, cell_sandbox=cell_sandbox, screenshot=screenshot and not shot_failed,
                timeout=min(render_check.TIMEOUT_SECONDS, remaining),
            )
        page_checks[page.key(store)] = outcome.status
        failed = outcome.status == "failed"
        if outcome.screenshot is not None and (shot is None or (failed and not shot_failed)):
            shot, shot_page, shot_failed = outcome.screenshot, label, failed
        if outcome.status == "passed":
            messages.append(f"{label} — loaded in a browser the way the app preview loads it: no problems found")
            continue
        if outcome.status == "unverified":
            messages.append(f"{label} — not checked in a browser: {outcome.unverified_reason}")
            if page.key(store) in findings_listed_earlier:
                # An earlier unverified check this turn already listed what
                # it found on this page. They are not listed again: the page
                # may already work, and listing them would send the agent
                # back to edit it.
                continue
            # What the check saw before it gave up (a blank map measured
            # before the deadline, say) is the agent's to fix once.
            found = render_check.unverified_findings(outcome)
            lines, more = [f"{found.lead} {line}" for line in found.lines], found.more
            if lines:
                findings_listed.add(page.key(store))
        else:
            grouped = render_check.grouped_findings(outcome.findings)
            lines, more = [group.text for group in grouped.shown], grouped.more
        messages += [f"{label} — {line}" for line in lines]
        if more:
            messages.append(f"{label} — and {more} more distinct findings, not listed")
    return ArtifactLint(
        messages=messages, screenshot=shot, screenshot_page=shot_page, page_checks=page_checks,
        findings_listed=frozenset(findings_listed),
    )


def lint_changed_artifact_files(
    store,
    before: dict[str, ArtifactFiles],
    *,
    cell_sandbox: CellSandbox | None = None,
    screenshot: bool = True,
    findings_listed_earlier: frozenset[PageKey] = frozenset(),
) -> ArtifactLint:
    """Run the format-appropriate checker on artifact folders this cell
    changed (any content file written, replaced, added or removed since
    `before`), so findings reach the agent as tool-result text right away,
    not only via a later open()/list().

    Under a sandbox policy the checks that start a browser or LibreOffice
    locally are skipped: they would run the agent's file outside the
    sandbox. The render-check service still loads pages there: only the
    page's text leaves this process (`_check_pages`). `screenshot=False`
    asks it for no screenshot. An unverified check of a page in
    `findings_listed_earlier` lists no findings (`_check_pages`).
    """
    linters = _artifact_linters(cell_sandbox=cell_sandbox)
    after = snapshot_existing_artifacts(store)
    messages: list[str] = []
    pages: list[_ChangedPage] = []
    for slug, files_before in before.items():
        files = after.get(slug)
        if files is None or files == files_before:
            continue

        artifact_dir = store.root / slug
        for rel, stamp in files.stamps.items():
            path = artifact_dir / rel
            if path.suffix.lower() == ".html":
                # Whatever its size: `_check_pages` says why a page too large
                # to check was not checked.
                pages.append(_ChangedPage(
                    slug=slug, path=path, edited=files.edited_since(before=files_before, rel=rel), size=stamp.size,
                ))
                continue
            if stamp.size > _ARTIFACT_LINT_SIZE_CEILING:
                continue
            linter = linters.get(path.suffix.lower())
            if linter is None:
                continue
            file_messages = linter(path)
            if not file_messages:
                # None (couldn't check) or [] (checked, clean) — either way,
                # nothing worth telling the agent about this file.
                continue
            for message in file_messages:
                messages.append(f"{slug}/{path.name} — {message}")

    checked = _check_pages(
        store, pages, cell_sandbox=cell_sandbox, screenshot=screenshot,
        findings_listed_earlier=findings_listed_earlier,
    )
    return ArtifactLint(
        messages=messages + checked.messages,
        screenshot=checked.screenshot,
        screenshot_page=checked.screenshot_page,
        page_checks=checked.page_checks,
        findings_listed=checked.findings_listed,
    )


def record_page_checks(
    session: "ChatSession",
    *,
    checks: "dict[PageKey, RenderStatus]",
    findings_listed: frozenset[PageKey] = frozenset(),
) -> None:
    """Remember, for this turn, how each page last fared in a browser, and
    which pages an unverified check already listed findings for.

    Per page: a check of one page replaces that page's earlier result and
    never another page's. The session reads it when the turn ends
    (`ChatSession._unverified_page_lines`). A page stays in
    `_unverified_findings_listed` for the rest of the turn, so a later
    unverified check of it lists nothing (`check_cell_artifacts`).
    Best-effort, like `_track_artifact`: a session without the record (a
    test double) loses nothing else.
    """
    record = getattr(session, "_page_checks", None)
    if isinstance(record, dict):
        record.update(checks)
    listed = getattr(session, "_unverified_findings_listed", None)
    if isinstance(listed, set):
        listed.update(findings_listed)


def render_screenshots_refused(session: "ChatSession") -> bool:
    """Whether the model refused a request carrying a render-check screenshot
    in this session and took it without one
    (`ChatSession._confirm_render_screenshots_refused`), so checks ask the
    service for none and the findings go to the agent without one. A test
    double without the flag gets screenshots."""
    return getattr(session, "_render_screenshots_refused", False) is True


def images_refused(session: "ChatSession") -> bool:
    """Whether the model refused a request carrying an image a tool returned
    in this session and took it without that image
    (`ChatSession._confirm_tool_images_refused`), so `read_image` sends no
    more images. Render screenshots follow their own refusal
    (`render_screenshots_refused`): one corrupt file draws a
    refusal from a model that takes images too. A test double without the
    flag gets images."""
    return getattr(session, "_images_refused", False) is True


def record_render_screenshots_refused(session: "ChatSession") -> None:
    """Remember that the model refused a request carrying a render-check
    screenshot and took it without one, for a caller outside the session's
    own turn loop (generate_artifact's retry). Later checks then ask for no
    screenshot (`render_screenshots_refused`)."""
    session._render_screenshots_refused = True


def page_check_lines(store, *, checks: "dict[PageKey, RenderStatus]") -> list[str]:
    """One line for each artifact with a page that did not pass its last
    browser check, naming the artifact, for the end of the agent's final
    message. A failed page wins over an unverified one."""
    by_slug: "dict[str, RenderStatus]" = {}
    for key, status in checks.items():
        by_slug[key.slug] = _worse(by_slug.get(key.slug), status)
    lines: list[str] = []
    for slug, status in by_slug.items():
        if status == "passed":
            continue
        metadata = store.read_metadata(slug) if store is not None else None
        name = metadata.name if metadata is not None and metadata.name.strip() else slug
        if status == "failed":
            lines.append(f"“{name}” failed the browser check, so it may not display correctly in the preview.")
        else:
            lines.append(f"“{name}” was not verified in a browser, so it may not display correctly in the preview.")
    return lines


def cell_content_with_lint(text: str, lint: ArtifactLint, *, cell_ok: bool) -> "str | list[dict]":
    """A cell's result with the checks' `[artifact lint]` block, and the
    render screenshot after it when the cell succeeded.

    Only a successful cell carries the image: a list result with a failure
    verdict skips the error streak's nudge text (see `handle_read_image`).
    """
    from anton.core.artifacts import render_check

    if lint.messages:
        # [artifact lint] follows format_cell_result's own [output]/[error]
        # labeling convention so the agent can tell the finding apart from
        # its own code's output. cowork-server's figure sources cut this
        # block from a replayed cell's output.
        text += "\n\n[artifact lint]\n" + "\n".join(lint.messages)
    if lint.screenshot is None or not cell_ok:
        return text
    return [{"type": "text", "text": text}, *render_check.screenshot_blocks(lint.screenshot, page=lint.screenshot_page)]


async def check_cell_artifacts(
    session: "ChatSession", *, store, before: dict[str, ArtifactFiles]
) -> ArtifactLint:
    """After an exec cell: attribute its artifact edits, check the changed
    files and record how their pages fared. Both exec paths (`handle_scratchpad`
    and `ChatSession`'s inline streaming exec) call this. Never raises.

    What an unverified check still found reaches the agent once per page
    per turn: once an earlier check this turn listed it (here, or in
    generate_artifact's `checks_skipped`), the page gets only the line that
    says it was not checked. A check that listed nothing does not count.
    """
    import asyncio

    track_edits_since(session, store, before)
    refused = render_screenshots_refused(session)
    listed = getattr(session, "_unverified_findings_listed", None)
    findings_listed_earlier = frozenset(listed) if isinstance(listed, set) else frozenset()
    try:
        # The checkers take seconds; run them off the event loop.
        lint = await asyncio.to_thread(
            lint_changed_artifact_files, store, before,
            cell_sandbox=cell_sandbox_of(getattr(session, "_scratchpads", None)), screenshot=not refused,
            findings_listed_earlier=findings_listed_earlier,
        )
    except Exception:
        # Best-effort: a lint crash must never fail the cell's own result.
        _log.warning("artifact lint failed for this cell", exc_info=True)
        return ArtifactLint()
    record_page_checks(session, checks=lint.page_checks, findings_listed=lint.findings_listed)
    if refused:
        # A service that sends a screenshot it was not asked for.
        return replace(lint, screenshot=None, screenshot_page="")
    return lint


def track_edits_since(session: "ChatSession", store, before: dict[str, ArtifactFiles]) -> None:
    """Attribute the artifact edits one scratchpad cell made.

    The agent edits artifact files from scratchpad cells, which the tool layer
    never sees, so this is how such an edit reaches `_artifacts_touched` and the
    artifact's provenance. Without it the host could never card the edit, even
    though the file genuinely changed this turn.

    Scoped to THIS cell's own execution window — the snapshot taken right
    before `pad.execute` vs. right after — rather than the whole turn, to
    keep the diff-based race (a concurrent sibling conversation happening to
    bump some other artifact's mtime) as narrow as possible: seconds, not
    minutes. Not zero — the same trade-off `index_turn_artifacts` already
    accepts for Hermes's edits.
    """
    already_touched = getattr(session, "_artifacts_touched", None) or ()
    after = snapshot_existing_artifacts(store)
    for slug, files_before in before.items():
        files = after.get(slug)
        if files is not None and files != files_before and slug not in already_touched:
            _track_artifact(session, store, slug, summary="Edited via scratchpad")


async def handle_create_artifact(session: "ChatSession", tc_input: dict) -> ToolOutcome:
    """Create a fresh artifact folder + metadata.json + README.md.

    Returns a `SideEffectResult` whose `message` carries the artifact path the
    agent writes output files under (`<path>/...`) after this call returns.
    """
    store = resolve_artifact_store(session)
    if store is None:
        return SideEffectResult.failed(
            "Artifact store unavailable (no workspace bound to this session).",
            reason="store_unavailable",
        )

    name = (tc_input.get("name") or "").strip()
    description = (tc_input.get("description") or "").strip()
    artifact_type = (tc_input.get("type") or "").strip()
    primary = tc_input.get("primary")
    if not name:
        return SideEffectResult.failed("Error: `name` is required.", reason="missing_name")
    if not description:
        return SideEffectResult.failed(
            "Error: `description` is required.", reason="missing_description"
        )

    from anton.core.artifacts.models import ARTIFACT_TYPES

    if artifact_type not in ARTIFACT_TYPES:
        return SideEffectResult.failed(
            f"Error: `type` must be one of {ARTIFACT_TYPES}. Got: {artifact_type!r}.",
            reason="invalid_type",
        )

    artifact = store.create(  # type: ignore[arg-type]
        name=name,
        description=description,
        type=artifact_type,
        primary=primary if isinstance(primary, str) else None,
    )
    folder = store.folder_for(artifact.slug)
    _track_artifact(session, store, artifact.slug, summary=f"Created artifact: {name}")
    return SideEffectResult(
        success=True,
        message=(
            f"Created artifact `{artifact.slug}` ({artifact.type}). "
            f"Write output files under: {folder}"
            + (f" (primary: {artifact.primary})" if artifact.primary else "")
        ),
        resource_id=artifact.slug,
        idempotency_key=artifact.slug,
        committed_at=now_iso(),
        details={
            "slug": artifact.slug,
            "path": str(folder),
            "name": artifact.name,
            "type": artifact.type,
            "primary": artifact.primary,
        },
    ).to_outcome()


async def handle_update_artifact_metadata(session: "ChatSession", tc_input: dict) -> ToolOutcome:
    """Update mutable metadata fields on an existing artifact.

    Only fields present in the input are modified. Supports:
    - `primary`: entry-point file path (empty string to clear)
    - `port`: backend port number (fullstack apps only)
    - `datasources`: list of vault-connection slugs the backend reads from.
      `engine`, `name`, and `env_prefix` are derived from the vault.
    """
    store = resolve_artifact_store(session)
    if store is None:
        return SideEffectResult.failed(
            "Artifact store unavailable (no workspace bound to this session).",
            reason="store_unavailable",
        )

    slug = (tc_input.get("slug") or "").strip()
    if not slug:
        return SideEffectResult.failed("Error: `slug` is required.", reason="missing_slug")

    kwargs: dict = {}
    if "primary" in tc_input:
        kwargs["primary"] = tc_input["primary"]
    if "port" in tc_input:
        try:
            kwargs["port"] = int(tc_input["port"]) if tc_input["port"] is not None else None
        except (TypeError, ValueError):
            return SideEffectResult.failed("Error: `port` must be a number.", reason="invalid_port")

    if "datasources" in tc_input:
        from anton.core.artifacts.models import DatasourceRef
        from anton.core.datasources.data_vault import LocalDataVault

        raw_list = tc_input.get("datasources") or []
        if not isinstance(raw_list, list):
            return SideEffectResult.failed(
                "Error: `datasources` must be a list of slug strings.",
                reason="invalid_datasources",
            )

        vault = session._data_vault or LocalDataVault()
        known = {f"{c['engine']}-{c['name']}": (c["engine"], c["name"])
                 for c in vault.list_connections()}

        refs: list[DatasourceRef] = []
        unknown: list[str] = []
        for item in raw_list:
            if not isinstance(item, str):
                return SideEffectResult.failed(
                    "Error: each entry in `datasources` must be a slug string.",
                    reason="invalid_datasources",
                )
            ref_slug = item.strip()
            if not ref_slug:
                continue
            if ref_slug not in known:
                unknown.append(ref_slug)
                continue
            engine, name = known[ref_slug]
            refs.append(DatasourceRef(engine=engine, name=name))
        if unknown:
            return SideEffectResult.failed(
                f"Error: unknown datasource slug(s): {', '.join(unknown)}. "
                f"Each slug must match an existing vault connection "
                f"(format: `<engine>-<name>`).",
                reason="unknown_datasource",
            )
        kwargs["datasources"] = refs

    artifact = store.update(slug, **kwargs)
    if artifact is None:
        return SideEffectResult.failed(
            f"Error: no artifact found for slug `{slug}`.", reason="artifact_not_found"
        )
    datasources = [d.slug for d in artifact.datasources]
    _track_artifact(session, store, artifact.slug, summary=f"Updated artifact metadata: {artifact.name}")
    return SideEffectResult(
        success=True,
        message=(
            f"Updated artifact `{artifact.slug}` "
            f"(primary={artifact.primary}, port={artifact.port}, "
            f"datasources={datasources})."
        ),
        resource_id=artifact.slug,
        idempotency_key=artifact.slug,
        committed_at=now_iso(),
        details={
            "slug": artifact.slug,
            "primary": artifact.primary,
            "port": artifact.port,
            "datasources": datasources,
        },
    ).to_outcome()


async def handle_launch_backend(session: "ChatSession", tc_input: dict) -> ToolOutcome:
    """Launch the artifact's backend script as a standalone subprocess.

    Thin wrapper over `launch_artifact_backend`: validates tool-call shape,
    resolves the artifact folder via the session's ArtifactStore, hands
    the session's scratchpad pool + tracked-backends dict to the helper,
    then persists the discovered port into metadata.json.

    The actual subprocess lifecycle (free-port discovery, dependency
    install, health probe, idempotent reap) lives in
    `anton.core.artifacts.backend_launcher.launch_artifact_backend` so
    other entry points (e.g. cowork's auto-relaunch) can reuse it.
    """
    from anton.core.artifacts.backend_launcher import launch_artifact_backend

    store = resolve_artifact_store(session)
    if store is None:
        return SideEffectResult.failed(
            "Artifact store unavailable (no workspace bound to this session).",
            reason="store_unavailable",
        )

    slug = (tc_input.get("slug") or "").strip()
    if not slug:
        return SideEffectResult.failed("Error: `slug` is required.", reason="missing_slug")
    artifact = store.open(slug)
    if artifact is None:
        return SideEffectResult.failed(
            f"Error: no artifact found for slug `{slug}`.", reason="artifact_not_found"
        )

    rel_path = (tc_input.get("path") or "backend.py").strip()
    extra_args = tc_input.get("extra_args") or []
    health_path = tc_input.get("health_path") or "/"
    try:
        health_timeout = float(tc_input.get("health_timeout", 10))
    except (TypeError, ValueError):
        return SideEffectResult.failed(
            "Error: `health_timeout` must be a number.", reason="invalid_health_timeout"
        )

    tracked = getattr(session, "_tracked_backends", None)
    if tracked is None:
        tracked = {}
        session._tracked_backends = tracked

    # A subprocess, so its credentials go in its own env — and only for the
    # datasources the artifact declared. Shared with the generator's own launch
    # (generate_artifact/orchestrator.py), which never reaches this handler.
    from anton.core.artifacts.backend_launcher import build_datasource_env

    ds_env = build_datasource_env(
        getattr(session, "_data_vault", None), artifact.datasources, slug=slug
    )

    # Only-if-unset, like the scratchpad, so a project .env cannot override
    # PATH or a key this process already has. The cloud turn's own state (its
    # bearer, its connection references) is for the scratchpad alone.
    overlay = getattr(session, "_workspace_env_overlay", None) or {}
    extra_env = {k: v for k, v in overlay.items() if k not in os.environ and not k.startswith("ANTON_CLOUD_")}

    result = await launch_artifact_backend(
        slug=slug,
        artifact_folder=store.folder_for(slug),
        scratchpad_pool=session._scratchpads,
        tracked_backends=tracked,
        path=rel_path,
        extra_args=extra_args,
        extra_env=extra_env,
        ds_env=ds_env,
        health_path=health_path,
        health_timeout=health_timeout,
    )
    # The launcher rolls back on failure (kills the process, never tracks it),
    # so a string result means nothing committed.
    if isinstance(result, str):
        return SideEffectResult.failed(result, reason="launch_failed")

    store.update(slug, port=result["port"])
    url = result.get("url", "")
    return SideEffectResult(
        success=True,
        message=(
            f"Backend for `{slug}` is running at {url} "
            f"(pid {result.get('pid')}, port {result.get('port')}, "
            f"log {result.get('log_path')})."
        ),
        resource_id=slug,
        external_url=url or None,
        idempotency_key=slug,
        committed_at=now_iso(),
        details={
            "slug": slug,
            "port": result.get("port"),
            "pid": result.get("pid"),
            "log_path": result.get("log_path"),
        },
    ).to_outcome()


def _generation_failed(reason: str) -> str:
    """Wrap an FSM failure so the outer agent reports it instead of DIY-ing.

    Without this instruction the agent treats a pipeline failure as a cue to
    build the artifact by hand (write_file/scratchpad fallback), silently
    bypassing the whole verified pipeline. Input-validation errors are NOT
    wrapped — those the agent should fix by correcting its call.
    """
    return (
        "Error: artifact generation failed.\n\n"
        f"{reason}\n\n"
        "IMPORTANT: do NOT build or repair the artifact yourself — no "
        "write_file / scratchpad fallback — and do not re-call "
        "generate_artifact with the same input. Report this failure to the "
        "user: state in plain language that artifact generation failed, quote "
        "the reason above, and ask how they want to proceed."
    )


async def _drain_progress(queue):
    """Yield ToolProgress markers, staying silent while a question is open.

    The pipeline asks the user from inside the task this handler is draining,
    so a marker emitted mid-question lands on top of a live prompt — and the
    CLI's Live context does not survive that.

    Lines produced during that window are not dropped silently: the last one
    is emitted once the answer arrives, so the user still learns where the
    pipeline got to. They are not replayed in full, because by then they
    describe steps that already finished.
    """
    from anton.core.tools.generate_artifact.progress import (
        MESSAGE_PREFIX,
        PEEK_PREFIX,
        QUESTION_CLOSED,
        QUESTION_OPEN,
    )
    from anton.core.tools.progress import ToolProgress

    depth = 0
    pending: str | None = None
    while True:
        line = await queue.get()
        if line is None:
            return
        if line.startswith(PEEK_PREFIX):
            # The live tail of a streaming LLM call. Not a step line: it is
            # replaced, never kept, and while a question is open it is
            # dropped rather than held — by the time the answer arrives it
            # describes text that has long since moved on.
            if not depth:
                yield ToolProgress(line[len(PEEK_PREFIX):], kind="peek")
            continue
        if line.startswith(MESSAGE_PREFIX):
            yield ToolProgress(line[len(MESSAGE_PREFIX):], kind="message")
            continue
        if line == QUESTION_OPEN:
            # A counter, not a flag: `show_and_confirm` wraps a call that
            # wraps itself, so the sentinels arrive nested. A flag would let
            # the inner CLOSED unmute the channel while the outer question is
            # still on screen — exactly the bug this prevents.
            depth += 1
            continue
        if line == QUESTION_CLOSED:
            depth = max(0, depth - 1)
            if depth == 0 and pending is not None:
                yield ToolProgress(pending)
                pending = None
            continue
        if depth:
            pending = line
            continue
        yield ToolProgress(line)


async def _generate_with_progress_close(coro, queue) -> "dict | str":
    """Await `coro`, then close the progress channel exactly once.

    The sentinel goes in a `finally` so the handler's drain loop terminates
    on a crash or a cancellation too — otherwise it would sit awaiting a
    queue nobody will ever write to again, turning any generator exception
    into a hung turn instead of a reported failure.
    """
    try:
        return await coro
    finally:
        queue.put_nowait(None)


# What every `generated` instruction says after its opening, verified in a
# browser or not.
_GENERATED_REPORTING = (
    "When the result carries `url`, the backend is running there and that URL "
    "is the app's entry point: give the user the URL, not the path of "
    "`static/index.html` — opened from disk the page cannot reach its "
    "`/api/*`. Without `url` (html-app), the entry point is "
    "`<path>/<primary>`. `checks_skipped` lists verifier checks that "
    "could not run (for example, no headless browser to load the page) "
    "and `warnings` what the verifier flagged without failing the step: "
    "when either is non-empty, tell the user in one sentence what was "
    "not checked or flagged — never present a skipped check as passed."
    " When the result carries `brief_shown: true`, the user has already "
    "seen the brief: do not repeat it. When it carries `brief_summary`, "
    "the user has not seen it: sum up its assumptions in a sentence or two."
)

_STATUS_INSTRUCTIONS = {
    "generated": (
        "Every file listed was statically verified by the generation "
        "pipeline, and a browser loaded the page without finding a problem. "
        "Do NOT re-read, re-parse or re-verify them, and do not "
        "open them looking for problems — report the result to the user and "
        "act only on problems the user actually reports. "
        + _GENERATED_REPORTING
    ),
    # `generated` when no browser check passed: `verified_in_browser` false.
    "generated_unverified": (
        "The files listed passed the pipeline's static checks, but no browser "
        "verified the page: `checks_skipped` says why. When it also lists what "
        "the check found (a blank view, an overflow, an exception), fix those "
        "once; if a later check still cannot vouch for the page, report it as "
        "unverified instead of editing it again. Reading the files again "
        "cannot verify how they render, so do not open them looking for other "
        "problems; report the result to the user and otherwise act only on "
        "problems the user actually reports. Your reply MUST say, in those "
        "words, that the page is unverified in a browser and may not display "
        "correctly in the preview; never say it works, was tested or renders "
        "correctly. "
        + _GENERATED_REPORTING
    ),
    "cancelled": (
        "The user declined the brief. Do NOT write prd.md yourself and do "
        "NOT build the artifact by hand — report back to the user."
    ),
    "needs_confirmation": (
        "A brief was drafted but the user could not be asked to confirm it, "
        "or did not answer in time. "
        "Show `brief_summary` to the user. If they agree, call this tool "
        "again with the SAME `user_request` — the repeat call is the "
        "confirmation and the pipeline continues from where it stopped. If "
        "they ask for a change or answer an open question, call again with "
        "the same `user_request` and that in `agent_understanding`. If they "
        "decline, do not call "
        "again."
    ),
    "stopped_over_budget": (
        "This turn reached its token budget and the pipeline stopped "
        "cleanly; what it had finished is on disk. Tell the user how far it "
        "got and ask whether to continue. When the result carries "
        "`brief_shown: true`, the user already saw the brief and was not "
        "asked to confirm it: do not show it again. Otherwise, when it "
        "carries a non-empty `brief_summary`, show it to the user as part of "
        "that question: the pipeline stopped before they could confirm it, "
        "and the repeat call is taken as their agreement to that brief. If "
        "they agree, call this tool again with the SAME `user_request` — it "
        "resumes rather than restarting. A requested change goes in "
        "`agent_understanding`, with `user_request` left as it was."
    ),
}


async def handle_generate_artifact(session: "ChatSession", tc_input: dict):
    """Build an already-registered artifact end to end.

    The handler validates input and reads the artifact's `type` (rejecting
    types outside {html-app, fullstack-stateless-app, fullstack-stateful-app}),
    then hands off to `generate_artifact.generate`, which runs all five
    phases — gather, brief, PRD, spec, code — and writes into the artifact
    folder. Pipeline failures come back wrapped by `_generation_failed` so
    the agent surfaces them instead of hand-building the artifact.

    An async generator rather than a plain coroutine: a full run
    takes minutes, and a tool that emits nothing for minutes reads as a hang.
    Step starts arrive on an `asyncio.Queue` and are yielded as `ToolProgress`
    markers; `ToolRegistry.dispatch_tool_stream` forwards those and takes the
    LAST non-marker item as the tool result. The queue exists because the FSM
    cannot yield from here: progress originates several frames down,
    including inside the `asyncio.gather` that runs backend and frontend
    generation at once.
    """
    import asyncio
    import json

    from anton.core.artifacts.models import GENERATOR_ARTIFACT_TYPES
    from anton.core.tools.registry import ToolOutcome

    store = resolve_artifact_store(session)
    if store is None:
        yield "Artifact store unavailable (no workspace bound to this session)."
        return

    slug = (tc_input.get("slug") or "").strip()
    if not slug:
        yield "Error: `slug` is required."
        return
    artifact = store.open(slug)
    if artifact is None:
        yield f"Error: no artifact found for slug `{slug}`."
        return

    if artifact.type not in GENERATOR_ARTIFACT_TYPES:
        yield (
            "Error: generate_artifact only supports html-app / "
            "fullstack-stateless-app / fullstack-stateful-app. "
            f"Got: {artifact.type}."
        )
        return

    user_request = (tc_input.get("user_request") or "").strip()
    if not user_request:
        yield "Error: `user_request` is required."
        return
    agent_understanding = (tc_input.get("agent_understanding") or "").strip()
    if not agent_understanding:
        yield "Error: `agent_understanding` is required."
        return

    raw_attachments = tc_input.get("attachments") or []
    if isinstance(raw_attachments, str):  # a single path, not wrapped in a list
        raw_attachments = [raw_attachments]
    attachments = [str(p).strip() for p in raw_attachments if str(p or "").strip()] \
        if isinstance(raw_attachments, list) else []

    folder = store.folder_for(slug)
    from anton.core.tools.generate_artifact import generate

    queue: "asyncio.Queue[str | None]" = asyncio.Queue()
    task = asyncio.create_task(
        _generate_with_progress_close(
            generate(
                session=session,
                slug=slug,
                artifact_path=folder,
                artifact_type=artifact.type,
                user_request=user_request,
                agent_understanding=agent_understanding,
                known_data=(tc_input.get("known_data") or "").strip(),
                user_preferences=(tc_input.get("user_preferences") or "").strip(),
                primary=artifact.primary,
                progress=queue,
                attachments=attachments,
            ),
            queue,
        )
    )
    try:
        async for marker in _drain_progress(queue):
            yield marker
        result = await task
    except Exception as exc:  # last-resort: never escalate to the dispatcher
        yield ToolOutcome(
            content=_generation_failed(f"generator crashed: {exc}"),
            ok=False,
            reason=type(exc).__name__,
        )
        return
    finally:
        # A no-op once the task is done. Reached with the task still running
        # when the consumer abandons this generator mid-yield (turn cancelled
        # → `aclose()` → GeneratorExit): without it, generation would carry on
        # writing files into the artifact folder with nobody left to report to.
        task.cancel()

    if isinstance(result, str):
        # A failed run still worked inside the artifact folder on its way to
        # failing, so the turn touched the artifact.
        _track_artifact(session, store, slug, summary="Generated artifact files")
        yield ToolOutcome(content=_generation_failed(result), ok=False, reason="pipeline")
        return

    status = result.get("status", "generated")
    # `cancelled` is the one outcome that wrote nothing at all: the user
    # declined the brief before phase C, so there is no prd.md and no
    # generated file to attribute, and claiming otherwise would put "Generated
    # artifact files" on a turn that generated none. Every other outcome —
    # including a budget stop, which leaves a PRD and a checkpoint behind —
    # did write, and `create_artifact` may have happened in an earlier turn,
    # leaving this one otherwise unattributed.
    if status != "cancelled":
        _track_artifact(session, store, slug, summary="Generated artifact files")
    # The render check's view of the page travels beside the text, as an
    # image the agent can look at, never inside the JSON, unless the model
    # refused a render screenshot earlier in the session.
    screenshot = result.pop("screenshot", None)
    screenshot_page = result.pop("screenshot_page", "")
    if render_screenshots_refused(session):
        screenshot = None
    instruction_key = status
    if status == "generated" and not result.get("verified_in_browser"):
        instruction_key = "generated_unverified"
    text = json.dumps(
        {
            "slug": slug,
            "path": str(folder),
            **result,
            "instruction": _STATUS_INSTRUCTIONS.get(
                instruction_key, _STATUS_INSTRUCTIONS["generated_unverified"]
            ),
        },
        indent=2,
    )
    # I-03: an explicit verdict rather than the dispatcher substring-matching
    # the text. A successful run's `trace` legitimately contains the word
    # "failed" (e.g. "backend.py failed to import in venv" after a successful
    # retry), which the legacy classifier counted as an error and fed into the
    # per-tool error streak.
    if screenshot is None:
        yield ToolOutcome(content=text, ok=True)
        return
    from anton.core.artifacts import render_check

    # The page the check loaded, which the generator may have named
    # differently from the primary `artifact` held before the run.
    page = f"{slug}/{screenshot_page or _primary_name(store, slug) or 'index.html'}"
    yield ToolOutcome(
        content=[{"type": "text", "text": text}, *render_check.screenshot_blocks(screenshot, page=page)],
        ok=True,
    )


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _field_names(value) -> list:
    """`fields` as a list; a string may name several, separated by commas."""
    names: list = []
    for item in _as_list(value):
        if isinstance(item, str):
            names += [part.strip() for part in item.split(",") if part.strip()]
        else:
            names.append(item)
    return names


async def handle_list_artifacts(
    session: "ChatSession", tc_input: dict
) -> "str | ToolOutcome":
    """List artifacts as text, grouped by artifacts root, newest first.

    Without `match` the default fields are a summary for finding an artifact;
    with `match` they are the full record for editing it.
    """
    from anton.core.artifacts.listing import FIELDS, SUMMARY_FIELDS, render_listing

    store = resolve_artifact_store(session)
    if store is None:
        # Tier 2 (ENG-2248): the tool cannot operate at all, so a retry
        # cannot help and repetition is thrash. Reuses the existing
        # `store_unavailable` sentinel key (external_wall/service_unavailable).
        return ToolOutcome(
            content="Artifact store unavailable (no workspace bound to this session).",
            ok=False, reason="store_unavailable",
        )

    keys = [str(key).strip() for key in _as_list(tc_input.get("match"))]
    keys = [key for key in keys if key]
    if tc_input.get("fields") is None:
        fields = FIELDS if keys else SUMMARY_FIELDS
    else:
        requested = _field_names(tc_input.get("fields"))
        valid = ", ".join(FIELDS)
        if not requested:
            return ToolOutcome(
                content=f"Error: fields is empty. Valid fields: {valid}.",
                ok=False, reason="invalid_fields",
            )
        for name in requested:
            if name not in FIELDS and name != "slug":
                return ToolOutcome(
                    content=f'Error: unknown field "{name}". Valid fields: {valid}.',
                    ok=False, reason="invalid_fields",
                )
        # Every entry is headed by its slug, so asking for it adds nothing.
        fields = [name for name in requested if name != "slug"]

    if keys:
        artifacts, unmatched = store.find(keys)
    else:
        artifacts, unmatched = store.list(), []
    root = Path(store.root).absolute()
    # Tier 1: a listing was produced. No artifacts, or unmatched keys, is still
    # a correct answer, not a failure.
    return ToolOutcome(content=render_listing([(root, artifacts)], fields, unmatched), ok=True)


async def handle_recall(session: ChatSession, tc_input: dict) -> str:
    """Process a recall tool call — search episodic memory."""
    if session._episodic is None or not session._episodic.enabled:
        return "Episodic memory is not available."

    query = tc_input.get("query", "")
    if not query:
        return "No query provided."

    kwargs: dict = {}
    if "max_results" in tc_input:
        kwargs["max_results"] = int(tc_input["max_results"])
    if "days_back" in tc_input:
        kwargs["days_back"] = int(tc_input["days_back"])

    return session._episodic.recall_formatted(query, **kwargs)


async def handle_memorize(
    session: ChatSession, tc_input: dict
) -> "str | ToolOutcome":
    """Process a memorize tool call and return a result string.

    Encoding is fire-and-forget so it never blocks scratchpad execution.
    """
    import asyncio

    if session._cortex is None:
        # Tier 2 (ENG-2248): no memory system wired, so no entry can ever be
        # stored. Retrying cannot help.
        return ToolOutcome(
            content="Memory system not available.",
            ok=False, reason="store_unavailable",
        )

    if session._cortex.mode == "off":
        # Tier 3 (ENG-2248): deliberately ok=None. This is a CONFIGURED state,
        # not a failure — the user turned memory off, and the tool reported that
        # correctly. Marking it ok=False would nudge and then break the tool for
        # every user who has memory disabled on purpose.
        return "Memory encoding is disabled. Change memory mode via /setup to enable."

    from anton.core.memory.base import Engram

    raw_entries = tc_input.get("entries", [])
    if not raw_entries:
        # Tier 2: a malformed call.
        return ToolOutcome(
            content="No entries provided.",
            ok=False, reason="missing_name",
        )

    engrams: list[Engram] = []
    for entry in raw_entries:
        if not isinstance(entry, dict) or "text" not in entry:
            continue

        kind = entry.get("kind", "lesson")
        if kind not in ("always", "never", "when", "lesson", "profile"):
            kind = "lesson"

        scope = entry.get("scope", "project")
        if scope not in ("global", "project"):
            scope = "project"

        # User-sourced memories (via explicit tool call) get high confidence
        engrams.append(
            Engram(
                text=entry["text"],
                kind=kind,
                scope=scope,
                confidence="high",
                topic=entry.get("topic", ""),
                source="user",
            )
        )

    if not engrams:
        # Tier 2: every entry was rejected by the shape checks above, so the
        # call carried nothing usable.
        return ToolOutcome(
            content="No valid entries provided.",
            ok=False, reason="invalid_type",
        )

    # Always encode immediately via fire-and-forget — the LLM explicitly
    # chose to memorize these, so we never interrupt the user mid-turn
    # with confirmation prompts.  Confirmations are reserved for the
    # post-turn consolidator (lessons extracted from scratchpad sessions).
    async def _encode_bg(cortex, entries):
        try:
            await cortex.encode(entries)
        except Exception:
            pass  # Best-effort; don't disrupt the conversation

    # Tracked so a host that tears down at end of turn can await it.
    session._track_memory_write(asyncio.create_task(_encode_bg(session._cortex, engrams)))

    descriptions = [f"Encoded {e.kind}: {e.text}" for e in engrams]
    # Tier 1: at least one entry was stored.
    return ToolOutcome(
        content="Memory updated: " + "; ".join(descriptions), ok=True
    )


async def handle_scratchpad(
    session: ChatSession, tc_input: dict
) -> str | ToolOutcome:
    """Dispatch a scratchpad tool call by action.

    The exec path returns a ``ToolOutcome`` carrying the runtime's own
    failure verdict, so the error streak doesn't re-classify the result by
    reading it (ENG-1276). The other actions still return plain strings
    (legacy substring classification).
    """
    action = scratchpad_action(tc_input)
    name = tc_input.get("name", "")

    if not name:
        # Explicit failure: this text has no legacy marker phrase, so the
        # substring fallback used to RESET the streak on it (ENG-1276).
        return ToolOutcome(
            content="Scratchpad name is required.",
            ok=False,
            reason="scratchpad_missing_name",
        )
    refused_name = refuse_pad_name(name)
    if refused_name is not None:
        return refused_name

    # ACC emit helper: use the session's safe wrapper if it exists,
    # otherwise no-op. Defined as a local closure so each emit site
    # stays a single line.
    def _acc_observe(kind: str, detail: dict, *, severity: int = 1) -> None:
        fn = getattr(session, "_acc_observe", None)
        if fn is not None:
            fn(kind, detail, severity=severity)

    if action == "exec":
        # The single-scratchpad guard and the pre-execute ACC events
        # (scratchpad_empty_code / scratchpad_call) live in
        # prepare_scratchpad_exec — the SHARED entry point that the streaming
        # path (ChatSession.turn_stream) also calls — so they fire on both
        # paths. A str return is a message the call should not run past
        # (empty code, single-scratchpad challenge, or install failure).
        result = await prepare_scratchpad_exec(session, tc_input)
        if isinstance(result, ToolOutcome):
            return result
        pad, code, description, estimated_time, estimated_seconds = result

        # Notify pre-execute observers (e.g. cerebellum). The runtime
        # never sees these — observation is an orchestration concern,
        # so it lives at the dispatcher layer where the data is most
        # natural and where local/remote runtimes stay interchangeable.
        prelim_cell = preliminary_cell(
            code=code,
            description=description,
            estimated_time=estimated_time,
            estimated_seconds=estimated_seconds,
        )
        await _fire_pre_execute(session, prelim_cell)

        # Snapshot existing artifacts' content files before the cell runs, so
        # an artifact edit the cell makes is attributed below (see
        # `track_edits_since`).
        artifact_store = resolve_artifact_store(session)
        before_artifacts = (
            snapshot_existing_artifacts(artifact_store)
            if artifact_store is not None
            else {}
        )

        cell = await pad.execute(
            code,
            description=description,
            estimated_time=estimated_time,
            estimated_seconds=estimated_seconds,
        )
        if cell is not None:
            session._record_cell_explainability(
                pad_name=name, description=description, cell=cell,
            )
            await _fire_post_execute(session, cell)
            # Post-execute ACC event (killed vs result) via the shared helper —
            # the streaming path emits the same.
            observe_scratchpad_cell(session, name, cell)
            lint = (
                await check_cell_artifacts(session, store=artifact_store, before=before_artifacts)
                if artifact_store is not None
                else ArtifactLint()
            )
        else:
            lint = ArtifactLint()
        # The runtime's verdict: a raised error/timeout/kill is a failure;
        # stderr-only output (warnings) is not, and stdout containing words
        # like "failed" is not either — the streak reads this flag, never the
        # text (ENG-1276). The reason is the traceback's LAST line (the cause),
        # the machine-comparable key ENG-1286's thrash breaker will consume.
        error = (cell.error or "").strip() if cell is not None else ""
        return ToolOutcome(
            content=cell_content_with_lint(format_cell_result(cell), lint, cell_ok=not error),
            ok=not error,
            reason=cell_failure_reason(error),
        )

    elif action == "view":
        # get_or_create: new ChatSession has empty _pads but replayed cells on the
        # manager — same hydration path as exec so view works on the first tool call.
        pad = await session._scratchpads.get_or_create(name)
        # ok=True: viewing succeeded even when the notebook being viewed
        # contains old "[error]" cells — the substring fallback used to count
        # a successful view of a failed cell as a fresh tool failure
        # (ENG-1276 false positive).
        return ToolOutcome(content=pad.view(), ok=True)

    elif action == "reset":
        pad = session._scratchpads.pads.get(name)
        if pad is None:
            return f"No scratchpad named '{name}'."
        await pad.reset()
        _acc_observe(
            "scratchpad_reset",
            {"name": name, "reason": "manual"},
            severity=5,
        )
        return f"Scratchpad '{name}' reset. All state cleared."

    elif action == "remove":
        return await session._scratchpads.remove(name)

    elif action == "dump":
        # get_or_create: dump must materialize the runtime from replayed cells when this
        # is the first scratchpad call in a new session (pads.get would miss every time).
        pad = await session._scratchpads.get_or_create(name)
        # ok=True for the same reason as view: rendering a notebook whose
        # cells include past "[error]" output is a success, not a failure.
        return ToolOutcome(content=pad.render_notebook(), ok=True)

    elif action == "install":
        packages = tc_input.get("packages", [])
        if not packages:
            return "No packages specified."
        refused = reject_invalid_packages(packages)
        if refused:
            return ToolOutcome(
                content=refused, ok=False, reason="package_install_rejected"
            )
        pad = await session._scratchpads.get_or_create(name)
        result = await pad.install_packages(packages)
        if install_call_installed_something(result):
            send_package_install_event(session, packages)
        return result

    else:
        # Every call that reaches here was refused before doing anything, so
        # it is a failure the error streak counts, not a result.
        named = f"`{action}` is not a scratchpad action" if action else "No `action` given and no `code` to run"
        return ToolOutcome(
            content=f"{named}. Valid actions: {', '.join(SCRATCHPAD_ACTIONS)}.",
            ok=False,
            reason="scratchpad_unknown_action",
        )


async def handle_read_image(
    session: "ChatSession", tc_input: dict
) -> "ToolOutcome":
    """Read an image file from disk and return it as an image content block.

    Returns a list of content blocks (image + text) on success so the model
    sees the picture on its next turn. Returns a plain error string on
    failure (missing file, non-image extension, bytes that are no PNG, JPEG,
    GIF or WebP, oversized image, etc.).
    """
    import base64
    from pathlib import Path

    from anton.clipboard import is_image_path
    from anton.utils.clipboard import (
        MAX_IMAGE_BYTES,
        _media_type_for,
        human_size,
        sniff_image_format,
    )

    file_path = (tc_input.get("file_path") or "").strip()
    if not file_path:
        # Tier 2 (ENG-2248): a malformed call; a retry cannot fix it.
        return ToolOutcome(
            content="Error: file_path is required.",
            ok=False, reason="missing_name",
        )

    if images_refused(session):
        # The model refused a request carrying an image a tool returned and
        # took it without one, so another image would cost one more refused
        # request and reach the model no better. Before the file is read.
        return ToolOutcome(
            content=(
                f"Error: {file_path} was not loaded. The model provider refused a "
                "request that carried an image earlier in this conversation, so "
                "read_image sends no more images. Tell the user you could not "
                "view this image."
            ),
            ok=False, reason="images_refused",
        )

    # Set only by a multi-person host (Cowork with the Landlock sandbox).
    # The desktop, the CLI and the cloud pod carry none and read as before.
    policy = cell_sandbox_of(getattr(session, "_scratchpads", None))
    try:
        path = _expand_home(file_path, policy)
        if not path.is_absolute():
            # Resolve relative paths against the project workspace (where
            # artifacts/decks live), not the process cwd. Under the desktop
            # app the agent's cwd is NOT the project, so a project-relative
            # image path would otherwise land in the wrong directory and
            # come back "file not found". Mirrors publish_or_preview. Falls
            # back to cwd for the CLI, where cwd already is the project.
            base = getattr(getattr(session, "_workspace", None), "base", None)
            root = Path(base) if base else Path.cwd()
            path = (root / path).resolve()
        if policy is not None:
            path = path.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        # Tier 2: the model supplied a path that will not parse.
        return ToolOutcome(
            content=f"Error: invalid path '{file_path}': {exc}",
            ok=False, reason="invalid_type",
        )

    if policy is not None and not policy.allows_read(path):
        # This process runs outside the sandbox, so it admits only what the
        # person's own cells may read. Checked before the file is looked at,
        # and worded without the resolved path, so the reply is the same
        # whether another person's file exists or not.
        return ToolOutcome(
            content=(
                f"Error: access denied: {file_path} is outside the files this "
                "conversation may read (the project, the scratchpad's home and "
                "temp folders, the shared folder and the conversation's "
                "attachments). If the user wants you to see this image, ask "
                "them to attach it to the conversation."
            ),
            ok=False, reason="access_denied",
        )

    if not path.is_file():
        # Tier 2: the file is genuinely absent. Unlike `recall_skill`'s NO
        # MATCH, there is no listing the model can consult to self-correct and
        # nothing here tells it to proceed regardless, so repeating the same
        # path IS thrash and belongs in the streak.
        return ToolOutcome(
            content=f"Error: file not found: {path}",
            ok=False, reason="path_not_found",
        )

    if not is_image_path(path.name):
        # Tier 2: the model pointed the image tool at a non-image. Its own
        # argument, and the message names the accepted extensions.
        return ToolOutcome(
            content=(
                f"Error: '{path.name}' is not a supported image format "
                "(expected .png/.jpg/.jpeg/.gif/.webp/.bmp)."
            ),
            ok=False, reason="not_an_image",
        )

    try:
        raw = path.read_bytes()
    except OSError as exc:
        # Tier 2: the read itself failed. `ok=False` is certain — the model got
        # no image. The CAUSE is not: this wraps a bare `except Exception`, so
        # `read_failed` is mapped TIER_UNCLASSIFIED rather than guessing at a
        # permissions wall.
        return ToolOutcome(
            content=f"Error: cannot read '{path}': {exc}",
            ok=False, reason="read_failed",
        )

    # The format the bytes show, not the extension: a JPEG saved as .png
    # sent as image/png draws a refusal, and a refused image turns
    # `read_image` off for the rest of the session. A BMP, whatever its name,
    # goes out as PNG. Bytes in none of the formats the API takes get a text
    # answer, and nothing is sent.
    fmt = sniff_image_format(raw[:16])
    if fmt == "BMP":
        try:
            import io
            from PIL import Image as _PILImage

            buf = io.BytesIO()
            _PILImage.open(io.BytesIO(raw)).save(buf, format="PNG")
            raw = buf.getvalue()
            fmt = "PNG"
        except Exception as exc:
            # Tier 2 for the verdict, unclassified for the cause: a bare
            # `except Exception` around a PIL call covers a missing Pillow (a
            # wall) and a corrupt BMP (self-inflicted) with one sentinel.
            return ToolOutcome(
                content=f"Error: failed to convert BMP to PNG: {exc}",
                ok=False, reason="bmp_convert_failed",
            )

    if fmt not in ("PNG", "JPEG", "GIF", "WEBP"):
        return ToolOutcome(
            content=f"Error: '{path.name}' is not a readable PNG, JPEG, GIF or WebP image.",
            ok=False, reason="not_an_image",
        )

    if len(raw) * 4 // 3 > MAX_IMAGE_BYTES:
        # Tier 2: over the API's hard limit. The model chose the file and the
        # message tells it what to do instead.
        return ToolOutcome(
            content=(
                f"Error: image is too large ({human_size(len(raw))}); "
                "the API limit is ~3.7 MB raw / 5 MB base64. "
                "Resize the image and try again."
            ),
            ok=False, reason="image_too_large",
        )

    b64 = base64.standard_b64encode(raw).decode("ascii")
    media_type = _media_type_for(fmt)

    summary = f"Loaded {path.name} ({human_size(len(raw))})."
    try:
        import io
        from PIL import Image as _PILImage

        # The bytes already read, not a second open of the path.
        with _PILImage.open(io.BytesIO(raw)) as im:
            summary = f"Loaded {path.name} ({im.width}x{im.height}, {human_size(len(raw))})."
    except Exception:
        pass

    # Tier 1, and the ONLY multimodal verdict in the tree. `ok=True` is free
    # here: the list arms of both tool loops already treat a list result as a
    # success unless a handler says otherwise, so this changes the emitted
    # `ok` from "unknown" to `true` and nothing else.
    #
    # It must stay `ok=True`. A list carrying `ok=False` would reach two
    # documented gaps in the list arms — the nudge/breaker text is never
    # appended there, and `_record_root_cause` is never called — so the model
    # would be silently retried past the breaker. `_tool_failure_cause`'s
    # docstring in session.py describes that shape as unreachable; it stays
    # unreachable because every failure above returns a plain string.
    # Pinned by `test_read_image_never_pairs_a_list_with_a_failure_verdict`.
    return ToolOutcome(
        content=[
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            },
            {"type": "text", "text": summary},
        ],
        ok=True,
    )


def _expand_home(file_path: str, policy: "CellSandbox | None") -> Path:
    """`file_path` with `~` expanded to the HOME the person's cells run with.

    Under a policy the process HOME is the server's, which holds every
    person's space; the cell's HOME is the policy's. Without one, `~` is the
    process HOME as before.
    """
    if policy is not None and (file_path == "~" or file_path.startswith("~/")):
        return policy.home_dir / file_path[1:].lstrip("/")
    return Path(file_path).expanduser()


def _read_roots(session: "ChatSession") -> "tuple[Path, tuple[Path, ...]]":
    """The base for relative paths, and the folders anton manages that may be read.

    Hosts and tests pass partial sessions; anything that is not a real path is
    ignored rather than trusted.
    """
    workspace = getattr(session, "_workspace", None)
    base = getattr(workspace, "base", None)
    base = Path(base) if isinstance(base, (str, Path)) else Path.cwd()
    owned: list[Path] = []
    artifacts_dir = getattr(workspace, "artifacts_dir", None)
    if isinstance(artifacts_dir, (str, Path)):
        owned.append(Path(artifacts_dir))
    drafts_root = getattr(session, "_skill_drafts_root", None)
    if isinstance(drafts_root, (str, Path)):
        owned.append(Path(drafts_root))
    # Both hosts stage skill drafts here; the desktop host keeps its own draft
    # tool and does not pass `_skill_drafts_root`.
    owned.append(base / ".anton" / "skill_drafts")
    return base, tuple(owned)


def _line_arg(value) -> "int | None":
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("a line number cannot be a boolean")
    # `int(2.5)` is 2: a fractional line number would silently read another range.
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("a line number must be a whole number")
    return int(value)


def _flag(value) -> bool:
    # `bool("false")` is True, and a model or gateway may send the flag as a
    # string or a number.
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return isinstance(value, str) and value.strip().lower() in ("true", "1")


def _shown_path(path: Path, display_root: "Path | None") -> str:
    """`path` as the result names it: relative to `display_root` when inside it."""
    if display_root is None:
        return str(path)
    roots = [display_root]
    try:
        roots.append(display_root.resolve())
    except (OSError, RuntimeError):
        pass
    for root in roots:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            continue
    return str(path)


async def handle_read_text_file(
    session: "ChatSession", tc_input: dict, *, display_root: "Path | None" = None
) -> ToolOutcome:
    """Read a text file the access policy allows, as a range of lines.

    Writes nothing: reading an artifact is not attributed to the turn.
    `display_root`: name files inside it relative to it in the header and in
    errors (the artifact generator passes its folder).
    """
    import asyncio

    raw = tc_input.get("path")
    if not isinstance(raw, str) or not raw.strip():
        return ToolOutcome(content="Error: `path` is required.", ok=False, reason="missing_path")
    # Resolving the path, the policy check and the read all touch the
    # filesystem, which on a network mount is too slow for the event loop.
    return await asyncio.to_thread(_read_text_file, session, raw.strip(), tc_input, display_root)


def _read_text_file(
    session: "ChatSession", raw: str, tc_input: dict, display_root: "Path | None"
) -> ToolOutcome:
    import stat

    from anton.core.tools import file_access, text_file

    base, owned_roots = _read_roots(session)
    policy = cell_sandbox_of(getattr(session, "_scratchpads", None))
    try:
        path = _expand_home(raw, policy)
        if not path.is_absolute():
            path = base / path
        resolved = path.resolve()
    except (ValueError, RuntimeError, OSError) as exc:
        return ToolOutcome(
            content=f"Error: invalid path {raw!r}: {exc}", ok=False, reason="invalid_path"
        )
    shown = _shown_path(path, display_root)

    refusal = file_access.under_policy(
        file_access.read_refusal(resolved, workspace=base, owned_roots=owned_roots),
        resolved, policy, outside=file_access.REFUSED_OUTSIDE,
    )
    if refusal is not None:
        _log.debug("read_text_file refused %s: %s", resolved, refusal)
        resolved_shown = _shown_path(resolved, display_root)
        # Under a policy the target can be another person's file: name only
        # the path as given.
        same = resolved_shown == shown or policy is not None
        where = shown if same else f"{shown} (it resolves to {resolved_shown})"
        message = f"Error: access denied: {where} is {refusal}."
        if refusal == file_access.REFUSED_OUTSIDE:
            message += (
                " If the user wants you to use this file, ask them to attach it"
                " to the conversation."
            )
        return ToolOutcome(content=message, ok=False, reason="access_denied")

    try:
        mode = resolved.stat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        return ToolOutcome(content=f"Error: file not found: {shown}", ok=False, reason="path_not_found")
    except OSError as exc:
        return ToolOutcome(content=f"Error: cannot read {shown}: {exc}", ok=False, reason="read_failed")
    if not stat.S_ISREG(mode):
        return ToolOutcome(content=f"Error: {shown} is not a file.", ok=False, reason="not_a_file")

    try:
        start_line = _line_arg(tc_input.get("start_line"))
        end_line = _line_arg(tc_input.get("end_line"))
    except (TypeError, ValueError, OverflowError):
        return ToolOutcome(
            content="Error: `start_line` and `end_line` must be integers.",
            ok=False, reason="invalid_range",
        )

    try:
        result = text_file.read_text_range(
            resolved, shown, start_line, end_line,
            line_numbers=_flag(tc_input.get("line_numbers")),
        )
    except text_file.FileTooLargeError:
        limit_mb = text_file.MAX_FILE_BYTES // (1024 * 1024)
        return ToolOutcome(
            content=f"Error: {shown} is larger than {limit_mb} MB. Read the part you need in the scratchpad.",
            ok=False, reason="file_too_large",
        )
    except text_file.NotTextError:
        return ToolOutcome(
            content=f"Error: {shown} is not a UTF-8 text file. Read it in the scratchpad.",
            ok=False, reason="not_text",
        )
    except text_file.InvalidRangeError as exc:
        return ToolOutcome(content=f"Error: {exc}.", ok=False, reason="invalid_range")
    except OSError as exc:
        return ToolOutcome(content=f"Error: cannot read {shown}: {exc}", ok=False, reason="read_failed")

    _log.debug(
        "read_text_file %s lines %d-%d of %d",
        resolved, result.first, result.last, result.total_lines,
    )
    return ToolOutcome(content=result.render(), ok=True)


# ---------------------------------------------------------------------------
# select_path — interactive file/folder disambiguation
# ---------------------------------------------------------------------------

# Hard caps keep the picker fast and the prompt readable: never offer more
# than _SELECTION_MAX_CANDIDATES options, and stop walking a glob after
# _SELECTION_SCAN_LIMIT entries so a `**` pattern on a huge tree can't stall.
_SELECTION_MAX_CANDIDATES = 50
_SELECTION_SCAN_LIMIT = 5000


def _selection_root(session: "ChatSession") -> "Path":
    """The directory candidates are confined to: the project root (or cwd)."""
    workspace = getattr(session, "_workspace", None)
    return (workspace.base if workspace is not None else Path.cwd()).resolve()


def _is_within(root: "Path", candidate: "Path") -> bool:
    """True when *candidate* is inside *root* — the path-traversal guard."""
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _collect_selection_candidates(tc_input: dict, root: "Path", kind: str) -> "list[Path]":
    """Resolve candidate paths: confined to *root*, deduped, kind-filtered, capped.

    Prefers the model's explicit ``candidates`` list; otherwise globs
    ``pattern`` (under an optional ``base_dir``). The private ``.anton``
    workspace is never exposed.
    """
    seen: set[Path] = set()
    found: list[Path] = []

    def consider(path: Path) -> bool:
        """Add *path* if it qualifies. Returns False once the cap is hit."""
        if len(found) >= _SELECTION_MAX_CANDIDATES:
            return False
        resolved = path.resolve()
        if resolved in seen or not _is_within(root, resolved) or not resolved.exists():
            return True
        if ".anton" in resolved.relative_to(root).parts:
            return True
        if (kind == "file" and not resolved.is_file()) or (kind == "folder" and not resolved.is_dir()):
            return True
        seen.add(resolved)
        found.append(resolved)
        return True

    explicit = tc_input.get("candidates")
    if isinstance(explicit, list) and explicit:
        for raw in explicit:
            if not isinstance(raw, str) or not raw.strip():
                continue
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            if not consider(candidate):
                break
    else:
        pattern = (tc_input.get("pattern") or "").strip()
        if pattern:
            base_dir = (tc_input.get("base_dir") or "").strip()
            search_root = root
            if base_dir:
                rel = Path(base_dir).expanduser()
                search_root = (rel if rel.is_absolute() else root / rel).resolve()
            if _is_within(root, search_root):
                for scanned, match in enumerate(search_root.glob(pattern)):
                    if scanned >= _SELECTION_SCAN_LIMIT or not consider(match):
                        break

    found.sort(key=lambda p: str(p).lower())
    return found


def _selection_option(path: "Path", root: "Path"):
    """One picker entry for *path*, labelled relative to the project root."""
    from anton.core.interaction.elicit import AskOption

    try:
        label = str(path.relative_to(root))
    except ValueError:
        label = str(path)
    return AskOption(
        value=str(path),
        label=label,
        kind="folder" if path.is_dir() else "file",
    )


def _chosen_path(answer) -> "str | None":
    """The single path an ``AskAnswer`` carries, or None if there is none."""
    if answer.status != "answered":
        return None
    if answer.values:
        return answer.values[0]
    return (answer.text or "").strip() or None


def _path_answer_failure(answer) -> "str | None":
    """Map a non-``answered`` outcome onto ``select_path``'s own status, or None.

    Collapsing every failure into ``cancelled`` would tell the model "the user
    dismissed the picker" when the user was never asked — the per-turn question
    budget was spent, the question timed out, or no host could render it. The
    model's documented reaction to ``cancelled`` is to ask how to proceed, so a
    false ``cancelled`` costs a turn on a question nobody ever saw.
    """
    if answer.status == "answered":
        return None
    if answer.status == "cancelled":
        return _status(
            "cancelled",
            "The user dismissed the picker without choosing. Ask how they would like to proceed.",
        )
    if answer.status == "unavailable":
        return _status(
            "picker_unavailable",
            "An interactive picker is unavailable here; ask the user for the path in plain text.",
        )
    if answer.status == "limit":
        return _status(
            "error",
            "Too many questions this turn; choose a path yourself and state which you picked, "
            "or ask in plain text.",
        )
    return _status("error", f"The picker did not return a selection ({answer.status}).")


def _browse_start_dir(tc_input: dict, root: "Path") -> "Path":
    """Resolve the browse-mode starting directory (defaults to the project root)."""
    raw = (tc_input.get("start_dir") or "").strip()
    if not raw:
        return root
    start = Path(raw).expanduser()
    if not start.is_absolute():
        start = root / start
    start = start.resolve()
    return start if start.is_dir() else root


def _status(status: str, message: str = "", **extra) -> str:
    """Serialize a select_path tool result, omitting an empty message."""
    return json.dumps({"status": status, **({"message": message} if message else {}), **extra})


def _needs_confirmation(candidate: "Path", label: str) -> str:
    """The model-supplied candidate cannot be silently accepted and no
    confirmation card can render on this host: hand the decision back."""
    return _status(
        "needs_confirmation",
        f"You supplied '{label}' as the only candidate — that is your guess, not "
        "the user's choice, so it was not accepted. Ask the user to confirm it "
        "before using it, and do not present it as chosen or connected until "
        "they do. If they asked for a file or folder outside the project, no "
        "path inside the project is an answer: tell them plainly that this host "
        "cannot reach paths outside the project, and ask them to attach the "
        "relevant files to the conversation.",
        candidates=[str(candidate)],
    )


async def _confirm_single_candidate(
    session: "ChatSession", candidate: "Path", root: "Path", timeout_s: "int | None"
) -> str:
    """Confirm a model-supplied single candidate with the user (ENG-1852).

    A lone entry in ``candidates`` is the model's own guess echoed back, not a
    discovery: in a fresh project exactly one directory exists (``skills``), so
    silently resolving it reported folders nobody chose as connected — to the
    first-run prompt "give you access to a folder on my computer", no less.
    Confirmation is enforced here rather than in the prompt, because the model
    already narrates such guesses as decisions ("I'll use it as the folder").

    Renders a yes/no choice card (supported on every cowork host, which has no
    path picker). Where no card can render either, returns
    ``needs_confirmation`` so the model must ask in plain text.
    """
    from anton.core.interaction.elicit import AskOption, AskRequest, elicit

    try:
        label = str(candidate.relative_to(root))
    except ValueError:
        label = str(candidate)
    noun = "folder" if candidate.is_dir() else "file"
    request = AskRequest(
        prompt=f"Only one candidate was found — use this {noun}?",
        kind="choice",
        timeout_s=timeout_s,
        options=(
            AskOption(value="yes", label=f"Yes — use {label}", kind=noun, style="primary"),
            AskOption(value="no", label="No — that's not it"),
        ),
    )
    try:
        answer = await elicit(session, f"path:{uuid.uuid4().hex}", request)
    except Exception as exc:  # noqa: BLE001 — the card is host code
        _log.warning("select_path confirmation elicitor failed: %s", exc, exc_info=True)
        return _status("error", f"Selection failed: {exc}")
    if answer.status == "unavailable":
        return _needs_confirmation(candidate, label)
    if answer.status == "cancelled":
        return _status(
            "cancelled",
            "The user dismissed the confirmation without choosing. Ask how they "
            "would like to proceed.",
        )
    if answer.status == "limit":
        return _status(
            "error",
            "Too many questions this turn; do not use the candidate without the "
            "user's confirmation — ask in plain text.",
        )
    if answer.status != "answered":
        return _status("error", f"The confirmation did not return an answer ({answer.status}).")
    typed = (answer.text or "").strip()
    if "yes" in answer.values:
        # Deliberately NOT auto_resolved: the user confirmed this path.
        return _status(
            "resolved",
            f'The user added: "{typed}"' if typed else "",
            path=str(candidate),
        )
    if typed and "no" not in answer.values:
        # A free-typed reply is not a decline the user made — it may even be
        # the word "yes" — so assert nothing; hand it over verbatim. Parsing
        # typed affirmations here would be a worse failure than one re-ask.
        return _status(
            "cancelled",
            f'The user typed a reply instead of choosing an option: "{typed}". '
            f"'{label}' is not confirmed — do not use it or present it as "
            "connected; act on their reply. If it reads as agreement, call "
            "select_path again for an explicit confirmation. If they want a "
            "file or folder outside the project, tell them plainly that this "
            "host cannot reach it and ask them to attach the relevant files "
            "to the conversation.",
        )
    return _status(
        "cancelled",
        f"The user declined '{label}'. Do not use this path and do not present "
        "it as connected. Ask what they actually meant — and if it is a file or "
        "folder outside the project, tell them plainly that this host cannot "
        "reach paths outside the project, and ask them to attach the relevant "
        "files to the conversation."
        + (f' The user said: "{typed}"' if typed else ""),
    )


def _finalize_browse_choice(chosen: "str | None", kind: str, root: "Path") -> str:
    """Validate a browse-mode pick: any existing path of the requested kind."""
    if chosen is None:
        return _status("cancelled", "The user dismissed the picker without choosing. Ask how they would like to proceed.")
    path = Path(chosen).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return _status("invalid", "The selected path no longer exists.")
    if kind == "file" and not path.is_file():
        return _status("invalid", "A folder was selected but a file was expected.")
    if kind == "folder" and not path.is_dir():
        return _status("invalid", "A file was selected but a folder was expected.")
    return _status("resolved", path=str(path.resolve()))


async def handle_select_path(session: "ChatSession", tc_input: dict) -> str:
    """Have the user choose a file/folder; return the chosen path as JSON.

    Two modes, chosen automatically from the inputs:

    * **browse** — no ``candidates``/``pattern`` given: the location is unknown,
      so the user navigates a picker to locate it. Use this instead of asking
      the user to type or paste a path. **Host-gated:** only reachable where an
      elicitor supports ``kind="path"``. Elsewhere — every cowork session today
      — it returns ``picker_unavailable`` pointing at attachment, and the model
      is not offered browse at all, because ``session.py`` registers
      ``SELECT_PATH_TOOL_PICK_ONLY`` there (ENG-1357).
    * **pick** — ``candidates`` or ``pattern`` given: disambiguate concrete
      matches within the project. A single *pattern* match auto-resolves and
      zero matches reports "no matches"; a single model-supplied candidate is
      confirmed with the user first (ENG-1852) — via the path picker where one
      renders, a yes/no choice card elsewhere, or ``needs_confirmation`` when
      neither can.

    The result is fed back as the tool result, so the agent continues without a
    separate user message.
    """
    from anton.core.interaction.elicit import AskRequest, elicit

    prompt = (tc_input.get("prompt") or "Select a file or folder.").strip()
    kind = (tc_input.get("kind") or "any").strip().lower()
    if kind not in ("file", "folder", "any"):
        kind = "any"

    root = _selection_root(session)
    elicitor = getattr(session, "elicitor", None)
    # Early-out only: elicit() re-checks kind support itself and answers
    # "unavailable" (mapped to picker_unavailable below) if this check were
    # ever removed or bypassed, so this is not the sole guard.
    can_pick = elicitor is not None and "path" in getattr(elicitor, "supported_kinds", ())
    timeout_s = getattr(elicitor, "timeout_s", None)
    has_candidates = isinstance(tc_input.get("candidates"), list) and bool(tc_input.get("candidates"))
    has_pattern = bool((tc_input.get("pattern") or "").strip())

    # ── browse — locate an unspecified path ──────────────────────────────
    if not has_candidates and not has_pattern:
        if not can_pick:
            # NOT "ask for the path in plain text" (ENG-1357). Browse means the
            # file's location is unknown, so it is very likely outside the
            # project — and every host that lands here forbids reading files
            # that are neither in the project nor attached, so a typed path is
            # unusable even when the user supplies it. Naming the one route
            # that works matters: with no legitimate exit offered, the model
            # fabricated the user's data instead of asking.
            return _status(
                "picker_unavailable",
                "This host cannot render a file browser. Ask the user to attach the "
                "file to the conversation — that is how they grant you access to a "
                "file outside the project. Do not ask them to type or paste a path, "
                "and do not proceed with invented or example data in its place.",
            )
        request = AskRequest(
            prompt=prompt,
            kind="path",
            timeout_s=timeout_s,
            path_kind=kind,
            path_mode="browse",
            root=str(_browse_start_dir(tc_input, root)),
        )
        try:
            answer = await elicit(session, f"path:{uuid.uuid4().hex}", request)
        except Exception as exc:  # noqa: BLE001 — the picker is host code
            _log.warning("select_path elicitor failed: %s", exc, exc_info=True)
            return _status("error", f"Selection failed: {exc}")
        failure = _path_answer_failure(answer)
        if failure is not None:
            return failure
        browse_root = Path(request.root) if request.root else root
        return _finalize_browse_choice(_chosen_path(answer), kind, browse_root)

    # ── pick — disambiguate concrete candidates within the project ───────
    candidates = _collect_selection_candidates(tc_input, root, kind)
    if not candidates:
        # The browse suggestion is only honest where browse can actually run —
        # on a host without a path elicitor it leads straight to
        # picker_unavailable, and "ask in plain text" is unusable for a file
        # outside the project (ENG-1357).
        return _status(
            "no_matches",
            (
                "No match found. Refine the pattern, omit candidates/pattern to let "
                "the user browse, or ask in plain text."
            )
            if can_pick
            else (
                "No match found in the project. Refine the pattern, or — if the file "
                "is not in the project at all — ask the user to attach it to the "
                "conversation. This host has no file browser, and you cannot read a "
                "path the user types."
            ),
        )
    if len(candidates) == 1:
        if not has_candidates:
            # A lone *pattern* match is a genuine discovery — the glob searched
            # the project and found exactly one thing — so resolving it without
            # a prompt is safe. A lone *explicit* candidate is not (ENG-1852):
            # it is the model's own guess echoed back, so it must be confirmed
            # by the user — via the confirmation card below, or by picking it
            # in the path picker where one renders.
            return _status("resolved", auto_resolved=True, path=str(candidates[0]))
        if not can_pick:
            return await _confirm_single_candidate(session, candidates[0], root, timeout_s)
    if not can_pick:
        return _status(
            "picker_unavailable",
            "An interactive picker is unavailable here; ask the user which of these paths they meant.",
            candidates=[str(p) for p in candidates],
        )

    options = tuple(_selection_option(p, root) for p in candidates)
    request = AskRequest(
        prompt=prompt,
        kind="path",
        timeout_s=timeout_s,
        options=options,
        path_kind=kind,
    )
    try:
        answer = await elicit(session, f"path:{uuid.uuid4().hex}", request)
    except Exception as exc:  # noqa: BLE001
        _log.warning("select_path elicitor failed: %s", exc, exc_info=True)
        return _status("error", f"Selection failed: {exc}")
    failure = _path_answer_failure(answer)
    if failure is not None:
        return failure
    chosen = _chosen_path(answer)
    if chosen not in {option.value for option in options}:
        return _status("invalid", "The returned selection was not one of the offered options.")
    return _status("resolved", path=chosen)


# ---------------------------------------------------------------------------
# ask_user — a multiple-choice question answered inside the current turn
# ---------------------------------------------------------------------------


def build_ask_request(tc_input: dict, timeout_s: "int | None"):
    """Parse the tool input into an ``AskRequest``, or None if it is junk.

    Parsing only — the well-formedness rules (option count, uniqueness) live
    in ``validate_request`` so the orchestrator, which builds requests in
    Python and never comes through here, gets them too.
    """
    from anton.core.interaction.elicit import AskOption, AskRequest

    question = (tc_input.get("question") or "").strip()
    if not question:
        return None

    raw_options = tc_input.get("options")
    if not isinstance(raw_options, list) or not raw_options:
        return None
    options = []
    for raw in raw_options:
        if not isinstance(raw, dict):
            return None
        value = (raw.get("value") or "").strip()
        if not value:
            return None
        options.append(
            AskOption(
                value=value,
                label=(raw.get("label") or value).strip(),
                detail=(raw.get("detail") or "").strip(),
            )
        )

    select = (tc_input.get("select") or "one").strip().lower()
    if select not in ("one", "many"):
        return None

    allow_custom = tc_input.get("allow_custom")
    return AskRequest(
        prompt=question,
        kind="choice",
        timeout_s=timeout_s,
        options=tuple(options),
        select=select,
        allow_custom=True if allow_custom is None else bool(allow_custom),
    )


_ASK_USER_UNAVAILABLE = (
    "Interactive questions are unavailable here, or the question was malformed "
    "(it needs 2-10 options with unique values). Ask in plain text instead and "
    "end your turn."
)
_ASK_USER_LIMIT = (
    "Question limit reached for this turn. Do not act on an assumed answer: "
    "ask in plain text and end your turn."
)
# Read by the model in place of an answer. "Proceed on a stated assumption"
# turned an unanswered approval into the approval itself, so both say to stop.
# The card stays answerable after the turn, and the answer comes back as the
# user's next message.
_ASK_USER_UNANSWERED = {
    "timeout": (
        "The user has not answered yet. Do not act on an assumed answer. End "
        "your turn now and say the question is still waiting for them; their "
        "answer will arrive as their next message."
    ),
    "cancelled": (
        "The user skipped this question. Do not act on an assumed answer. End "
        "your turn and say what you need from them to continue."
    ),
}


def _ask_user_telemetry_settings(session: "ChatSession"):
    """Best-effort settings object for `send_event`, mirroring the pattern
    already used at every other call site (see `anton/tools.py`)."""
    settings = getattr(session, "_settings", None)
    if settings is not None:
        return settings
    try:
        from anton.config.settings import AntonSettings

        return AntonSettings()
    except Exception:
        return None


def _send_ask_user_event(session: "ChatSession", action: str, props: dict) -> None:
    """Fire one telemetry event. Never raises — analytics must not break a turn."""
    try:
        settings = _ask_user_telemetry_settings(session)
        if not settings:
            return
        from anton.analytics import send_event

        send_event(settings, action, **props)
    except Exception:
        pass


async def handle_ask_user(session: "ChatSession", tc_input: dict) -> str:
    """Ask the user to choose, and return the answer as the tool result."""
    from anton.core.interaction.elicit import elicit

    elicitor = getattr(session, "elicitor", None)
    request = build_ask_request(
        tc_input, timeout_s=getattr(elicitor, "timeout_s", None)
    )
    if request is None:
        return _status("error", _ASK_USER_UNAVAILABLE)

    # send_event only accepts string extras (see every existing call site).
    props = {"select": request.select, "options": str(len(request.options))}
    _send_ask_user_event(session, "ask_user_asked", props)

    # The question id doubles as the correlation key the host echoes back
    # with the answer. The originating tool_use.id is not visible here
    # (dispatch_tool passes only name + input), so mint one.
    question_id = f"ask:{uuid.uuid4().hex}"
    answer = await elicit(session, question_id, request)
    _send_ask_user_event(session, f"ask_user_{answer.status}", props)

    if answer.status == "limit":
        return _status("error", _ASK_USER_LIMIT)
    if answer.status == "unavailable":
        return _status("error", _ASK_USER_UNAVAILABLE)
    if answer.status in _ASK_USER_UNANSWERED:
        return _status(answer.status, _ASK_USER_UNANSWERED[answer.status])

    if answer.status == "answered":
        extra: dict = {}
        if answer.values:
            extra["values"] = list(answer.values)
        if answer.text:
            extra["text"] = answer.text
        return _status("answered", **extra)

    # Explicit rather than a fall-through to "answered": `Elicitor` is a
    # structural Protocol implemented out of tree, so an unlisted status (a
    # host-side typo, a future status) is reachable without touching this
    # repo — and telling the LLM the user answered and chose nothing is the
    # worst failure shape a decision tool has. Same shape as
    # `_path_answer_failure`.
    return _status("error", f"The question did not return an answer ({answer.status}).")
