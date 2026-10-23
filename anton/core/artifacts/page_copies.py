"""Each artifact's main page as it was when the turn began.

A follow-up edit can break or drop a view that worked before the turn. The
check after each cell and generate_artifact's browser check send the
render-check service the page kept here (`previous_html`), and the service
compares the two (`render_check.Comparison`). When an edit breaks a page
past its repair cells, or a rebuild ends without a page, anton puts the
copy back (`TurnPages.restore`). What the turn broke, blanked, removed or
put back closes the agent's final message (`TurnPages.end_lines`).

Copies are kept only with the render-check service configured: without it
nothing compares them. `ChatSession` holds one `TurnPages` per turn.
"""

from __future__ import annotations

import logging
import os
import stat
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from anton.core.artifacts import render_check
from anton.core.artifacts.render_check import Comparison, ComparisonNotes, agree, comparison_notes, joined
from anton.utils.nofollow import read_bytes_nofollow, replace_bytes_nofollow

if TYPE_CHECKING:
    from anton.core.artifacts.render_check import RenderStatus
    from anton.core.backends.sandbox import CellSandbox
    from anton.core.tools.tool_handlers import PageKey

_log = logging.getLogger(__name__)

# The most page text one turn keeps. Pages are taken newest first while
# they fit, so a page that does not fit goes without a copy: it is neither
# compared nor put back.
MAX_TURN_COPY_BYTES = 64 * 1024 * 1024
# A fullstack app's page calls its backend, which the render check does not
# run, so no check loads it and nothing compares it.
_FULLSTACK_TYPES = frozenset({"fullstack-stateless-app", "fullstack-stateful-app"})


def main_page_name(folder: Path, primary: str | None) -> str | None:
    """The artifact's main page in FOLDER, as the cell check picks it: its
    primary when that is an `.html` file there, else `index.html`, else its
    first `.html` content file by name. None without a page. The primary
    and `index.html` are looked up directly; only without either is the
    folder walked."""
    from anton.core.artifacts.store import iter_content_files

    for name in (primary, "index.html"):
        if name is not None and _is_content_page(folder, name):
            return name
    pages = sorted(
        path.relative_to(folder).as_posix()
        for path, _ in iter_content_files(folder)
        if path.suffix.lower() == ".html"
    )
    return pages[0] if pages else None


def _is_content_page(folder: Path, name: str) -> bool:
    """Whether NAME, a path relative to FOLDER, is an `.html` file that
    `iter_content_files` would yield: a regular file reached through real
    folders, never a link, outside the housekeeping names and bytecode.
    Each part must be listed under its exact name, as the walk lists it,
    also where the file system ignores case."""
    from anton.core.artifacts.internal_files import NON_CONTENT_NAMES, is_bytecode_name

    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts) or parts[0] in NON_CONTENT_NAMES:
        return False
    if any(is_bytecode_name(part) for part in parts) or PurePosixPath(name).suffix.lower() != ".html":
        return False
    path = folder
    try:
        for index, part in enumerate(parts):
            if part not in os.listdir(path):
                return False
            path = path / part
            mode = path.lstat().st_mode
            if not (stat.S_ISREG(mode) if index == len(parts) - 1 else stat.S_ISDIR(mode)):
                return False
    except OSError:
        return False
    return True


def _read_page(path: Path, *, cell_sandbox: CellSandbox | None) -> bytes | None:
    """PATH's bytes, read the way `render_check.check_file` reads a page:
    only where the person's own cells may read, never through a link, never
    past `render_check.MAX_PAGE_BYTES`. None when it can't be read so."""
    try:
        real = Path(os.path.realpath(path))
        if cell_sandbox is not None and not cell_sandbox.allows_read(real):
            return None
        return read_bytes_nofollow(real.anchor, real, limit=render_check.MAX_PAGE_BYTES)
    except OSError:
        return None


class TurnPages:
    """One turn's copies of the artifacts' main pages, the render check's
    latest comparison of each page with its copy, and the pages anton put
    back.

    The check after a cell runs in a worker thread while the session
    records on the event loop, so one lock guards the state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Held for a whole `capture`, so a second call (a cell and a
        # generate_artifact run in parallel tool calls) waits for the copies
        # instead of finding none.
        self._capture_lock = threading.Lock()
        self._captured = False
        self._copies: dict[PageKey, bytes] = {}
        # Each page's latest comparison whose previous page the service
        # checked, in the order the turn's last lines name the pages.
        self._comparisons: dict[PageKey, Comparison] = {}
        # Per page, the `blank` and `removed` labels last shown to the agent.
        self._shown: dict[PageKey, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        self._restored: dict[PageKey, None] = {}

    def capture(self, store, *, cell_sandbox: CellSandbox | None) -> None:
        """Keep the main page of every artifact but a fullstack app, the
        first time this turn something may change one. Later calls keep
        nothing, so a page made this turn has no copy and every copy is the
        page as the turn found it. Newest pages first, up to
        `MAX_TURN_COPY_BYTES`; a page that can't be read the way the check
        reads it, or isn't UTF-8, gets none. Nothing without the render-check
        service. Blocks on file reads: call it off the event loop."""
        if render_check.configured_url() is None:
            return
        with self._capture_lock:
            if self._captured:
                return
            try:
                self._read_copies(store, cell_sandbox=cell_sandbox)
            finally:
                # A later capture would copy pages the turn may have edited
                # by then: no copies is safer than late ones.
                self._captured = True

    @property
    def captured(self) -> bool:
        """Whether this turn already took its copies (or tried and failed)."""
        return self._captured

    def _read_copies(self, store, *, cell_sandbox: CellSandbox | None) -> None:
        """`capture`'s reads, under its lock."""
        from anton.core.tools.tool_handlers import PageKey

        root = store.root
        if not root.is_dir():
            return
        candidates: list[tuple[int, int, PageKey, Path]] = []
        for child in root.iterdir():
            if not child.is_dir() or not (child / "metadata.json").is_file():
                continue
            metadata = store.read_metadata(child.name)
            if metadata is None or metadata.type in _FULLSTACK_TYPES:
                continue
            name = main_page_name(child, metadata.primary)
            if name is None:
                continue
            try:
                info = (child / name).lstat()
            except OSError:
                continue
            candidates.append((info.st_mtime_ns, info.st_size, PageKey(slug=child.name, page=name), child / name))
        copies: dict[PageKey, bytes] = {}
        total = 0
        for _, size, key, path in sorted(candidates, key=lambda candidate: candidate[0], reverse=True):
            # A page that cannot fit is not read at all; the read below is
            # still measured, since the page may have grown since.
            if size > render_check.MAX_PAGE_BYTES or total + size > MAX_TURN_COPY_BYTES:
                _log.debug("no copy of %s/%s for this turn", key.slug, key.page)
                continue
            data = _read_page(path, cell_sandbox=cell_sandbox)
            if data is not None:
                try:
                    data.decode("utf-8")
                except UnicodeDecodeError:
                    data = None
            if data is None or total + len(data) > MAX_TURN_COPY_BYTES:
                _log.debug("no copy of %s/%s for this turn", key.slug, key.page)
                continue
            total += len(data)
            copies[key] = data
        with self._lock:
            self._copies.update(copies)

    def previous(self, key: PageKey) -> str | None:
        """KEY's page as the turn found it, or None without a copy."""
        with self._lock:
            data = self._copies.get(key)
        return None if data is None else data.decode("utf-8")

    def main_previous(self, slug: str) -> tuple[PageKey, str] | None:
        """SLUG's main page as the turn found it, with its key, or None."""
        with self._lock:
            key = next((key for key in self._copies if key.slug == slug), None)
        if key is None:
            return None
        previous = self.previous(key)
        return None if previous is None else (key, previous)

    def _known(self, key: PageKey, comparison: Comparison | None) -> Comparison | None:
        """COMPARISON when the service checked the previous page, else the
        turn's latest one for KEY that it did. Call under the lock."""
        if comparison is not None and comparison.informative:
            return comparison
        return self._comparisons.get(key)

    def should_restore(self, key: PageKey, comparison: Comparison | None) -> bool:
        """Whether to put KEY back after its check failed with no repair
        cells left: yes with a copy, unless the comparison shows the page
        failed before the turn and no view broke since. Without a comparison
        that says something, yes."""
        with self._lock:
            if key not in self._copies:
                return False
            known = self._known(key, comparison)
        return known is None or not (known.previous_status == "failed" and not known.has_new_problem)

    def status_before(self, key: PageKey, comparison: Comparison | None) -> RenderStatus:
        """How the page as the turn found it fared in the render check:
        COMPARISON's `previous_status` whenever the service checked the
        previous page, however far the new page's check got (that status
        describes the previous page alone), else the turn's latest
        comparison that says something, else `unverified`."""
        if comparison is not None and comparison.previous != "not_checked" and comparison.previous_status is not None:
            return comparison.previous_status
        with self._lock:
            known = self._comparisons.get(key)
        if known is None or known.previous_status is None:
            return "unverified"
        return known.previous_status

    def has_restored(self) -> bool:
        """Whether anton put a page back this turn."""
        with self._lock:
            return bool(self._restored)

    def restore(self, key: PageKey, store) -> bool:
        """Put KEY's copy back, byte for byte, through a rename that never
        follows a link. Whether it did; a failure is logged. The page then
        reads as put back in the turn's last lines, and its comparison and
        the lines the agent saw about it are forgotten: they described the
        page anton replaced."""
        with self._lock:
            data = self._copies.get(key)
        if data is None:
            return False
        try:
            replace_bytes_nofollow(store.root, store.root / key.slug / key.page, data)
        except OSError:
            _log.warning("could not put %s/%s back as it was before this turn", key.slug, key.page, exc_info=True)
            return False
        with self._lock:
            self._restored[key] = None
            self._comparisons.pop(key, None)
            self._shown.pop(key, None)
        return True

    def is_as_found(self, key: PageKey, store) -> bool:
        """Whether KEY's page on disk is, byte for byte, its copy. False
        without a copy or when the page can't be read: read under the store
        root, never through a link, never past `render_check.MAX_PAGE_BYTES`.
        Blocks on a file read: call it off the event loop."""
        with self._lock:
            data = self._copies.get(key)
        if data is None:
            return False
        try:
            current = read_bytes_nofollow(
                store.root, store.root / key.slug / key.page, limit=render_check.MAX_PAGE_BYTES
            )
        except OSError:
            return False
        return current == data

    def restore_if_changed(self, key: PageKey, store) -> bool:
        """`restore`, unless the page on disk already is its copy: for the
        outcomes of a generate_artifact run, which may have left it alone."""
        if self.is_as_found(key, store):
            return False
        return self.restore(key, store)

    def record(self, comparisons: Mapping[PageKey, Comparison], *, edited: Iterable[PageKey] = ()) -> None:
        """Keep each page's latest comparison that says something; one whose
        previous page the service did not check leaves the earlier one. A
        page compared after anton put it back, or in EDITED (rewritten by a
        cell, whatever its check said), was edited since, so it no longer
        reads as put back."""
        with self._lock:
            for key, comparison in comparisons.items():
                if not comparison.informative:
                    continue
                self._comparisons[key] = comparison
                self._restored.pop(key, None)
            for key in edited:
                self._restored.pop(key, None)

    def notes_to_show(self, key: PageKey, notes: ComparisonNotes) -> ComparisonNotes:
        """NOTES for the agent, less the `blank` and `removed` it was already
        shown for KEY this turn: a view drawn late reads as blank on every
        check, and the same line on every cell would send the agent back to
        fix it each time. A changed set is shown whole. `broke` always
        stays: the page failed, and its repair cells bound the edits."""
        shown = (tuple(notes.blank), tuple(notes.removed))
        with self._lock:
            if self._shown.get(key) == shown:
                return notes._replace(blank=[], removed=[])
            self._shown[key] = shown
        return notes

    def end_lines(self, store, *, checks: Mapping[PageKey, RenderStatus]) -> list[str]:
        """The lines that close the turn's final message, written by anton
        whatever the reply claims: each page put back, and what the turn
        broke, blanked or removed on each compared page. A page whose last
        check in CHECKS passed gets no line for a view that broke: that
        check may have seen it fixed without comparing."""
        with self._lock:
            restored = list(self._restored)
            comparisons = dict(self._comparisons)
        lines: list[str] = []
        for key in dict.fromkeys([*restored, *comparisons]):
            name = _artifact_name(store, key.slug)
            if key in restored:
                lines.append(
                    f"“{name}”: the edit could not be made safely, so the page is back as it was before this turn."
                )
                continue
            notes = comparison_notes(comparisons[key])
            if notes.broke and checks.get(key) != "passed":
                lines.append(
                    f"“{name}”: {joined(notes.broke)} drew before this turn and "
                    f"{agree(notes.broke, 'fails', 'fail')} now."
                )
            if notes.blank:
                lines.append(
                    f"“{name}”: {joined(notes.blank)} showed a graphic before this turn and "
                    f"{agree(notes.blank, 'shows', 'show')} none now."
                )
            if notes.removed:
                lines.append(
                    f"“{name}” no longer has {joined(notes.removed)}: this turn removed "
                    f"{agree(notes.removed, 'it', 'them')}."
                )
        return lines


def _artifact_name(store, slug: str) -> str:
    """The artifact's name for a line the user reads, else its slug."""
    metadata = store.read_metadata(slug) if store is not None else None
    return metadata.name if metadata is not None and metadata.name.strip() else slug
