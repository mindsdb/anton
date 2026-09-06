"""The folder a yolo run is allowed to touch.

Every path the model names is resolved against one root and refused if it
lands outside it. This is not a sandbox — the process can still do
anything — but it does mean a hallucinated `../../etc/hosts` in a diff is
an error rather than a write, and that is the failure worth closing.

Nothing here knows about anton. It is a folder, some files, and a size
budget.
"""

from __future__ import annotations

import contextlib
import re
import signal
import threading
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "DATA_SUFFIX",
    "Match",
    "SearchResult",
    "SCHEMA_SUFFIX",
    "Workspace",
    "WorkspaceError",
    "is_generated_data",
    "is_schema",
    "schema_for",
]

# The naming convention that divides the two tools' territory.
#
# `prices.data.js` is written by the scratchpad, which is what should be
# producing data: extraction, transformation, arithmetic. It is `.js`
# rather than `.json` on purpose — an artifact opened over file:// cannot
# fetch() a JSON file, so the data has to arrive as a <script> defining a
# global.
#
# `prices.schema.json` is its sidecar: what the columns are, and what
# global the data file defines. It is JSON rather than JavaScript because
# nothing loads it — only the agent reads it — and a file that is never
# executed should not look executable.
#
# Sidecar rather than a central registry: it survives the folder being
# copied, it is visible to whoever opens the artifact, and there is only
# one place to look.
DATA_SUFFIX = ".data.js"
SCHEMA_SUFFIX = ".schema.json"


def is_generated_data(path: str) -> bool:
    """Whether this is a generated data file, which yolo must not edit."""
    return path.endswith(DATA_SUFFIX)


def is_schema(path: str) -> bool:
    return path.endswith(SCHEMA_SUFFIX)


def schema_for(data_path: str) -> str:
    """The sidecar that should describe this data file."""
    return data_path[: -len(DATA_SUFFIX)] + SCHEMA_SUFFIX


class WorkspaceError(Exception):
    """A path that escaped the workspace, or a file that could not be read."""


# Files above this size are listed in the map but never inlined: a single
# bundled asset can otherwise eat the whole context window and leave no
# room for the change itself.
MAX_FILE_BYTES = 256 * 1024

# Directories that are never worth mapping. Their contents are numerous,
# uninteresting, and not what anyone means by "the artifact".
SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "dist",
        "build",
        ".next",
        ".idea",
        ".vscode",
    }
)


# Caps on what one search can return. A common word must not be able to
# flood the prompt: three lines from any one file is enough to say "it is
# in here", and twenty across the folder is enough to say "here is where".
MAX_MATCHES_PER_FILE = 3
MAX_MATCHES_PER_QUERY = 20
MAX_MATCH_LINE = 200

# How long one pattern may run before it is cut off, and how many
# patterns one request may ask for. The product is the worst case a
# pathological regex can cost, and 5s is a pause rather than a hang.
SEARCH_TIMEOUT = 1.0
MAX_QUERIES = 5


def _can_interrupt() -> bool:
    """Whether a runaway match can be stopped here.

    SIGALRM exists only on Unix, and a handler can only be installed from
    the main thread. Both are checked because the alternative to knowing
    is a hang with no way out.
    """
    return (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )


@contextlib.contextmanager
def _time_limit(seconds: float):
    """Raise TimeoutError if the block runs longer than seconds.

    This works on a regex specifically because `sre` checks for pending
    signals while it matches — a plain thread could not interrupt it.
    """
    def ring(*_):
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, ring)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _clip(line: str) -> str:
    return line if len(line) <= MAX_MATCH_LINE else line[:MAX_MATCH_LINE] + " …"


@dataclass
class Match:
    """One line of one file that matched."""

    path: str
    line: int
    text: str


@dataclass
class SearchResult:
    """What one query found, and anything the model should know about it.

    `note` carries the things worth saying out loud: a pattern that would
    not compile, one that had to be cut short, or a platform where the
    search fell back to literal text. All three are feedback the model
    can act on, which is why they travel with the result instead of being
    logged and lost.
    """

    query: str
    matches: list[Match] = field(default_factory=list)
    note: str = ""


@dataclass
class FileInfo:
    path: str  # workspace-relative, forward slashes
    bytes: int


class Workspace:
    """Read and write access confined to one folder."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    # ── paths ────────────────────────────────────────────────────────

    def resolve(self, relative: str) -> Path:
        """Resolve a workspace-relative path, refusing anything outside.

        The check is on the resolved path, so `a/../../b` is caught as
        readily as `../b`. Symlinks are resolved too: a link pointing out
        of the workspace is a way out of the workspace.
        """
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError(f"{relative} is outside the workspace")
        return candidate

    # ── reading ──────────────────────────────────────────────────────

    def read(self, relative: str) -> str:
        path = self.resolve(relative)
        if not path.is_file():
            raise WorkspaceError(f"{relative} does not exist")
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise WorkspaceError(f"{relative} is not text") from error

    def exists(self, relative: str) -> bool:
        try:
            return self.resolve(relative).is_file()
        except WorkspaceError:
            return False

    def read_many(self, paths: list[str]) -> str:
        """Read several files into one labelled block for the model.

        A file that cannot be read reports why in place rather than
        failing the batch: the model asked for five files and getting
        four plus an explanation is more useful than getting none.
        """
        chunks = []
        for relative in paths:
            try:
                body = self.read(relative)
            except WorkspaceError as error:
                chunks.append(f"--- {relative} ---\n({error})\n")
                continue
            if len(body.encode("utf-8")) > MAX_FILE_BYTES:
                chunks.append(
                    f"--- {relative} ---\n"
                    f"(too large to show: {len(body.encode('utf-8'))} bytes)\n"
                )
                continue
            chunks.append(f"--- {relative} ---\n{body}\n")
        return "".join(chunks)

    # ── writing ──────────────────────────────────────────────────────

    def write(self, relative: str, content: str) -> None:
        path = self.resolve(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    # ── searching ────────────────────────────────────────────────────

    def search(self, query: str) -> SearchResult:
        """Find a regular expression in the workspace, case-insensitively.

        Regex rather than literal text, because models write it well and
        it is what makes the difference between finding a thing and
        finding where a thing is defined: `function\\s+draw\\w*` beats
        guessing at the exact spelling.

        The one real hazard is catastrophic backtracking. It is not
        hypothetical — `(a+)+b` against thirty characters takes 46
        seconds on this machine, and grows exponentially from there, so
        no cap on line length or file size contains it. What does contain
        it is a wall-clock interrupt: `sre` checks for signals while
        matching, so an alarm cuts a runaway pattern short.

        Where an alarm cannot be armed — Windows, or off the main thread —
        the pattern is escaped and searched for literally instead. That is
        a real loss of power, but it is the honest trade against a hang,
        and the result says which one happened.

        Generated data is skipped. Searching two megabytes of rows for
        "price" returns eight thousand lines of noise and buries the one
        line of code that reads them.
        """
        pattern = query.strip()
        if not pattern:
            return SearchResult(query, [], "empty query")

        note = ""
        if not _can_interrupt():
            # No way to stop a runaway match here, so do not start one.
            pattern, note = re.escape(pattern), "searched literally (no regex on this platform)"
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as error:
            # Handed back to the model rather than raised: a bad pattern
            # is something it can fix on the next attempt, and "invalid
            # pattern" is more use to it than a stack trace.
            return SearchResult(query, [], f"invalid pattern: {error}")

        matches: list[Match] = []
        try:
            with _time_limit(SEARCH_TIMEOUT):
                self._scan(compiled, matches)
        except TimeoutError:
            note = (
                f"timed out after {SEARCH_TIMEOUT:g}s — the pattern backtracks badly; "
                f"simplify it (avoid nested quantifiers like (x+)+)"
            )
        return SearchResult(query, matches, note)

    def _scan(self, compiled: re.Pattern, matches: list[Match]) -> None:
        """Fill matches, respecting the caps. Interruptible by the alarm."""
        for info in self.files():
            if is_generated_data(info.path) or info.bytes > MAX_FILE_BYTES:
                continue
            try:
                content = self.read(info.path)
            except WorkspaceError:
                continue
            found = 0
            for number, line in enumerate(content.split("\n"), start=1):
                if not compiled.search(line):
                    continue
                matches.append(Match(info.path, number, _clip(line.strip())))
                found += 1
                # Both caps are checked here. Testing the total only
                # between files lets a single file overshoot it by up to
                # MAX_MATCHES_PER_FILE - 1.
                if found >= MAX_MATCHES_PER_FILE or len(matches) >= MAX_MATCHES_PER_QUERY:
                    break
            if len(matches) >= MAX_MATCHES_PER_QUERY:
                return

    def search_many(self, queries: list[str]) -> tuple[str, list[str]]:
        """Run several searches, returning what to show and where to look.

        A query with no matches is reported as such rather than omitted.
        "that pattern matches nothing in this folder" is a real answer,
        and the one that stops the model looking for it.
        """
        blocks: list[str] = []
        hits: list[str] = []
        for query in queries[:MAX_QUERIES]:
            result = self.search(query)
            header = f'"{result.query}"'
            if result.note:
                header += f"  [{result.note}]"
            if not result.matches:
                blocks.append(f"{header} — no matches")
                continue
            lines = [f"{header} —"]
            for match in result.matches:
                lines.append(f"  {match.path}:{match.line}  {match.text}")
                if match.path not in hits:
                    hits.append(match.path)
            blocks.append("\n".join(lines))
        if len(queries) > MAX_QUERIES:
            blocks.append(
                f"({len(queries) - MAX_QUERIES} further quer"
                f"{'y was' if len(queries) - MAX_QUERIES == 1 else 'ies were'} not run: "
                f"at most {MAX_QUERIES} per request)"
            )
        return "\n".join(blocks), hits

    # ── mapping ──────────────────────────────────────────────────────

    def files(self) -> list[FileInfo]:
        """Every text-ish file in the workspace, sorted by path."""
        found: list[FileInfo] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.relative_to(self.root).parts):
                continue
            found.append(
                FileInfo(
                    path=path.relative_to(self.root).as_posix(),
                    bytes=path.stat().st_size,
                )
            )
        return found

    def map(self) -> str:
        """The listing the model starts from.

        Paths and sizes, never contents — with one deliberate exception.
        Which files matter is the one judgement models are reliably good
        at: given the names, they ask for the right two or three out of
        thirty. Inlining everything up front spends the context window to
        save a decision they were going to make correctly anyway.

        The exception is schema sidecars. A line reading
        `prices.data.js (2.1 MB)` tells the model nothing it can write
        code against, and the file itself can never be inlined. Its
        sidecar can: a few hundred bytes that say what the columns are
        and — the part that actually matters — what global the data file
        defines. So schemas are always included in full, and the data
        files they describe are always listed and never read.
        """
        entries = self.files()
        if not entries:
            return "(the folder is empty)"

        lines = []
        for entry in entries:
            note = ""
            if is_generated_data(entry.path):
                sidecar = schema_for(entry.path)
                note = (
                    f"  ← generated data, see {sidecar}"
                    if self.exists(sidecar)
                    else "  ← generated data (no schema sidecar)"
                )
            lines.append(f"{entry.path} ({entry.bytes} bytes){note}")

        schemas = [entry.path for entry in entries if is_schema(entry.path)]
        if not schemas:
            return "\n".join(lines)

        blocks = ["\n".join(lines), "\nDATA SCHEMAS (read these instead of the data files):"]
        for path in schemas:
            try:
                blocks.append(f"--- {path} ---\n{self.read(path).strip()}")
            except WorkspaceError as error:
                blocks.append(f"--- {path} ---\n({error})")
        return "\n".join(blocks)
