"""Read the patches a model writes, and place them by content.

A port of yolocoder's applier (``internal/repo/compact.go`` and
``internal/repo/patch.go``), cross-checked against Mochi's newer TypeScript
port of the same code. Every tolerance and repair here exists because a real
run lost a round trip without it; the comments say which.

The lesson behind all of it: **models get the content right and the
bookkeeping wrong.** They reproduce a file's lines faithfully and then
miscount ``@@ -14,7 +14,9 @@``, start a hunk one line off, or stop a hunk at
its own change. None of that bookkeeping needs the model, because the file is
right here, so every line number and count is ignored and each edit is placed
by finding its text.

Three formats are read. The compact one is what the model is taught::

    @path            modify this file
    @+path           write this whole file; every following line is literal content
     context         (a leading space)
    -removed
    +added

A blank line (or a bare ``@@`` at column 0) separates one edit from the next.
OpenAI's apply_patch (V4A) and unified diffs are read too, because models fall
back to them on their own; unified diffs are placed by content as well, since
there is no ``git apply`` here.

Where the reading *refuses* matters as much: a block that matches nothing, or
matches in several places with nothing to say which, is reported rather than
guessed at, and every report is written to be handed straight back to the
model, quoting the file's own lines for it to copy.

Pure: no I/O. :func:`apply_patch` reads through the callable it is given and
returns the new text of every file it could fully place.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

__all__ = [
    "ApplyResult",
    "Diagnosis",
    "FileEdit",
    "Hunk",
    "PatchParseError",
    "apply_patch",
    "parse_patch",
]


class PatchParseError(ValueError):
    """The patch could not be read at all, or asks for something refused
    outright (deleting or moving a file). The message is model-facing."""


# A line of a hunk and what it was doing there: " " context, "-" removed,
# "+" added.
Op = tuple[str, str]


@dataclass(frozen=True)
class Hunk:
    """One edit: ``before`` is what must be found in the file (context and
    removed lines, in order), ``after`` is what replaces it (context and
    added lines). Line numbers are deliberately absent.

    ``ops`` keeps the same lines with their shape (which were context, which
    removed, which added). Placing needs only before/after; splitting a hunk a
    model bunched together, or repairing one, needs the shape. It is empty for
    a created file's literal content.
    """

    before: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    ops: tuple[Op, ...] = field(default=(), repr=False)

    @classmethod
    def from_ops(cls, ops: Iterable[Op]) -> Hunk:
        ops = tuple(ops)
        return cls(
            before=tuple(text for kind, text in ops if kind != "+"),
            after=tuple(text for kind, text in ops if kind != "-"),
            ops=ops,
        )

    @property
    def empty(self) -> bool:
        return not self.before and not self.after


@dataclass(frozen=True)
class FileEdit:
    path: str
    create: bool
    hunks: tuple[Hunk, ...]


@dataclass(frozen=True)
class Diagnosis:
    path: str
    # Index of the hunk within its FileEdit; -1 for a file-level problem.
    hunk_index: int
    kind: str
    message: str


@dataclass
class ApplyResult:
    written: dict[str, str] = field(default_factory=dict)
    failed: dict[str, list[Diagnosis]] = field(default_factory=dict)
    repairs: list[str] = field(default_factory=list)
    noop: bool = False

    @property
    def ok(self) -> bool:
        return not self.failed and not self.noop

    def feedback(self, tool: str = "apply_diff") -> str:
        """One model-facing message naming every failure in the patch.

        Validating the whole patch before giving up is what makes a repair
        cost one round trip instead of several: three bad hunks reported one
        at a time is three model calls to learn what one pass already knew.
        """
        if self.ok:
            return ""
        if not self.failed:
            return NOOP_MESSAGE
        failures = [one for group in self.failed.values() for one in group]
        if len(failures) == 1:
            text = _describe(failures[0])
        else:
            parts = [
                (
                    f"{len(failures)} edits in this patch could not be placed. "
                    f"Fix every one of them, all in one {tool} call:\n"
                )
            ]
            for number, one in enumerate(failures, 1):
                parts.append(f"\n[{number}] {_describe(one)}\n")
            text = "".join(parts)
        if self.written:
            # The files that did place are written; a repair that resends
            # them is skipped as already in place, but says so up front.
            names = ", ".join(sorted(self.written))
            text = text.rstrip("\n") + (
                f"\n\nEvery edit to {names} placed and was applied; send only the fixes above."
            )
        return text


def _describe(one: Diagnosis) -> str:
    return f"{one.path}: {one.message}" if one.path else one.message


NOOP_MESSAGE = (
    "this patch places correctly but changes nothing — every line in it is already "
    'exactly as written. Check that the lines you meant to change are on "-" and "+" '
    "lines rather than context lines"
)

# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


class _Builder:
    """A hunk being read: its ops so far."""

    def __init__(self) -> None:
        self.ops: list[Op] = []

    def add(self, kind: str, text: str) -> None:
        self.ops.append((kind, text))

    def hunk(self) -> Hunk:
        return Hunk.from_ops(self.ops)


@dataclass
class _File:
    path: str
    create: bool = False
    hunks: list[Hunk] = field(default_factory=list)
    # For a created file: its literal content lines.
    content: list[str] | None = None


def parse_patch(text: str) -> list[FileEdit]:
    """Read a compact patch, an apply_patch (V4A) patch, or a unified diff."""
    lines = _lines_of(text)
    if _is_compact(lines):
        files = _parse_compact(lines)
    else:
        files = _parse_diff(lines)
    return _finish(files)


def _lines_of(text: str) -> list[str]:
    # Trailing blank lines are the patch text ending in a newline, not empty
    # context lines; counted as context they make every last hunk unmatchable.
    lines = [
        line.rstrip("\r")
        for line in text.rstrip("\n").replace("\r\n", "\n").split("\n")
    ]
    # A whole patch wrapped in a markdown fence, the way chat replies come.
    # Only an outer pair: a created markdown file may end in a fence of its own.
    nonblank = [index for index, line in enumerate(lines) if line.strip()]
    if len(nonblank) >= 2:
        first, last = nonblank[0], nonblank[-1]
        if lines[first].strip().startswith("```") and lines[last].strip() == "```":
            lines = lines[first + 1 : last]
    return lines


def _finish(files: list[_File]) -> list[FileEdit]:
    if not files:
        raise PatchParseError("no file headers in patch")
    edits = []
    for one in files:
        if one.create:
            content = list(one.content or [])
            # A unified or apply_patch create carries its content as added
            # lines, perhaps over several hunks; it is one literal body.
            for current in one.hunks:
                content.extend(current.after)
            edits.append(FileEdit(one.path, True, (Hunk(after=tuple(content)),)))
            continue
        # A hunk that collected nothing carries no instruction, and keeping
        # one would read as "replace this file with nothing".
        hunks = tuple(current for current in one.hunks if not current.empty)
        edits.append(FileEdit(one.path, False, hunks))
    return edits


# --- compact ---------------------------------------------------------------


def _looks_like_path(text: str) -> bool:
    """What keeps a created file's own contents from being read as the next
    file's header. A CSS file full of ``@media (...)`` and ``@import url(...)``,
    or TypeScript ``@Component({...})``, would otherwise cut the file short at
    its first at-rule and start a second file named after it. Real paths
    carry no spaces, brackets, quotes or semicolons; the at-rules that matter
    all do. A bare ``@Component`` alone on a line would still be misread."""
    if not text or len(text) > 200 or text.startswith("@"):
        return False
    return not any(ch in text for ch in " \t(){}[]\"';,")


# "@END", "*** END ***", "EOF": bare end markers models add, which are not
# files. Prefixed forms in any case; a bare word only in capitals, so a Ruby
# or Lua "end" that lost its leading space is still read as context.
_PREFIXED_END = re.compile(
    r"^(?:\*\*\*\s*|@)(?:END|EOF)(?:\s+PATCH)?\s*(?:\*\*\*)?$", re.IGNORECASE
)
_PLAIN_END = re.compile(r"^(?:END|EOF)(?:\s+PATCH)?$")


def _is_bare_end(trimmed: str) -> bool:
    return bool(_PREFIXED_END.match(trimmed) or _PLAIN_END.match(trimmed))


def _is_trailer(line: str) -> bool:
    """A borrowed wrapper line that must never become file content. A model
    that closed a created file with ``*** End Patch`` had the marker written
    into the file, and every later repair burned the edit budget scrubbing it
    back out. A bare ``***`` is not one: that is a markdown rule."""
    trimmed = line.strip()
    return trimmed.startswith(("*** Begin Patch", "*** End Patch")) or _is_bare_end(
        trimmed
    )


def _is_terminator(line: str) -> bool:
    trimmed = line.strip()
    return trimmed.startswith("*** End Patch") or _is_bare_end(trimmed)


def _is_separator(line: str) -> bool:
    """Markers a model writes between edits when it slips into a format it
    knows better. Anchored at column 0 on purpose: a context line carries a
    leading space, so a file that really contains ``@@`` or ``***`` still
    reads as content. A bare ``***`` counts: requiring ``*** `` cost a whole
    turn reported as ``expected: ***`` against real code."""
    return line.startswith(("@@", "***"))


def _compact_header(line: str) -> tuple[str, str] | None:
    """(mode, path) for a compact file header, else None."""
    if _is_trailer(line):
        return None
    # A header prefixed with another format's marker: "*** @path", or
    # apply_patch's "*** path". Read as a separator, the header was dropped
    # and nine edits went looking for their lines in the previous file.
    if line.startswith("***"):
        rest = line[3:].strip()
        if rest.startswith("@") or _looks_like_path(rest):
            line = rest if rest.startswith("@") else "@" + rest
    if not line.startswith("@"):
        return None
    rest = line[1:]
    mode = " "
    if rest and rest[0] in "+->":
        mode, rest = rest[0], rest[1:]
    if not _looks_like_path(rest):
        return None
    return mode, _clean_path(rest)


def _borrowed_header(line: str) -> tuple[str, str] | None:
    """Another format's file header partway through a compact patch:
    ``*** Update File: path`` or ``@@ path``. Taken as plain separators, the
    edits under them went looking in the file named before — a run spent
    three rounds on App.tsx edits reported missing from backend/index.ts.
    ``@@ path`` counts only with a "/" or "." in it: a bare ``@@ render`` is
    apply_patch's function hint, and a unified hunk header carries spaces."""
    line = line.strip()
    for prefix, mode in (("*** Update File:", " "), ("*** Add File:", "+")):
        if line.startswith(prefix):
            rest = line[len(prefix) :].strip()
            return (mode, _clean_path(rest)) if _looks_like_path(rest) else None
    for prefix in ("*** Delete File:", "*** Move to:"):
        if line.startswith(prefix):
            rest = line[len(prefix) :].strip()
            return ("-" if "Delete" in prefix else ">", rest) if rest else None
    if line.startswith("@@"):
        rest = line[2:].strip()
        rest = rest.removesuffix("@@").strip()
        if _looks_like_path(rest) and ("/" in rest or "." in rest):
            return " ", _clean_path(rest)
    return None


def _is_compact(lines: Sequence[str]) -> bool:
    """Decided by the first line that says anything at all."""
    for line in lines:
        if line.strip():
            return _compact_header(line) is not None
    return False


def _plus_only_follows(lines: Sequence[str], start: int) -> bool:
    """Whether the edit after a blank line is added lines and nothing else.
    Such an edit has nothing to place it by, so it can only be the rest of
    the addition before the blank."""
    while start < len(lines) and not lines[start].strip():
        start += 1
    seen = False
    for line in lines[start:]:
        if (
            not line.strip()
            or _is_separator(line)
            or _compact_header(line) is not None
            or _borrowed_header(line) is not None
        ):
            break
        if line[0] != "+":
            return False
        seen = True
    return seen


def _more_blocks_after(lines: Sequence[str], start: int) -> bool:
    """Whether anything from here on is patch syntax rather than whatever the
    model emitted once it had stopped writing one."""
    for line in lines[start:]:
        if not line.strip():
            continue
        if (
            line.startswith(("*** ", "diff --git ", "--- ", "+++ "))
            or _compact_header(line) is not None
        ):
            return True
    return False


def _strip_uniform_plus(content: list[str]) -> list[str]:
    """A created file written with "+" on every non-blank line means diff
    lines, not literal plus signs. Mixed content (a markdown "+" bullet) is
    left alone."""
    nonblank = [line for line in content if line.strip()]
    if not nonblank or not all(line.startswith("+") for line in nonblank):
        return content
    return [line.removeprefix("+") for line in content]


def _created_content_ends(lines: Sequence[str], index: int) -> bool:
    """Whether line ``index`` ends a created file's literal content."""
    line = lines[index]
    if _compact_header(line) is not None:
        return True
    stripped = line.strip()
    if stripped.startswith(
        ("*** Begin Patch", "*** End Patch", "*** Update File:", "*** Add File:")
    ):
        return True
    if _is_bare_end(stripped):
        # "EOF" / "END" close a heredoc in a shell script too; only the last
        # thing in the patch (or before the next header) is a trailer.
        for after in lines[index + 1 :]:
            if _compact_header(after) is not None:
                return True
            if after.strip():
                return False
        return True
    return False


def _refuse(mode: str, path: str) -> PatchParseError:
    verb = "deletes" if mode == "-" else "moves"
    return PatchParseError(
        f"the patch {verb} {path}, which the artifact editor does not do; "
        "edit or create files only"
    )


def _parse_compact(lines: list[str]) -> list[_File]:
    files: list[_File] = []
    current: _File | None = None
    active: _Builder | None = None
    # Whether the previous line was a "+": a "-" straight after one ends the
    # edit and starts the next.
    added = False

    def flush() -> None:
        # An empty edit is spacing, not an edit that changes nothing.
        nonlocal active
        if current is not None and active is not None and active.ops:
            current.hunks.append(active.hunk())
        active = None

    index = 0
    while index < len(lines):
        line = lines[index]
        header = _compact_header(line)
        if header is None and current is not None:
            header = _borrowed_header(line)
        if header is not None:
            flush()
            mode, path = header
            if mode in "->":
                raise _refuse(mode, path)
            added = False
            current = _File(path=path, create=mode == "+")
            files.append(current)
            if mode == "+":
                # A created file's lines are literal, unprefixed content, so a
                # file that really begins with "+", "-" or a space can be
                # written down; and blank lines are content, not separators.
                content = []
                while index + 1 < len(lines) and not _created_content_ends(
                    lines, index + 1
                ):
                    index += 1
                    content.append(lines[index])
                current.content = _strip_uniform_plus(content)
            index += 1
            continue

        if current is None:
            if line.strip():
                raise PatchParseError(
                    f"an edit appears before any file header: {_clip(line)!r}"
                )
            index += 1
            continue

        # An explicit end means the end. A run emitted a burst of corrupted
        # tokens after "*** End Patch"; read as a hunk it could not be placed,
        # and the retry was the same patch without the garbage.
        if _is_terminator(line):
            if not _more_blocks_after(lines, index + 1):
                break
            flush()
            added = False
            index += 1
            continue

        # "@@" (and "*** End Patch" and friends) is what a model reaches for
        # between edits. Read as content, "@@" is a line nowhere in the file,
        # and a run lost its whole edit budget reporting `expected: "@@"`.
        if _is_separator(line):
            flush()
            added = False
            index += 1
            continue

        if line.startswith("\\"):  # "\ No newline at end of file"
            index += 1
            continue

        if not line.strip():
            if added and active is not None and _plus_only_follows(lines, index + 1):
                # A blank inside added code, left unprefixed, belongs to the
                # addition. Split there, the rest is added lines alone, which
                # can never be placed: a run lost a round to two new files'
                # worth of code rejected as having no context.
                active.add("+", "")
            else:
                # Otherwise it separates, even though it might be a context
                # line whose leading space was dropped. Splitting is the safe
                # way to be wrong: a hunk that loses a line of context still
                # places, while two merged edits match nothing anywhere.
                flush()
                added = False
            index += 1
            continue

        # A "-" straight after a "+" with no context between ends one edit and
        # starts the next: models write two unrelated one-line changes under
        # one header like this constantly. A genuine replacement writes its
        # removals first ("-a -b +x +y"), so it never hits this.
        if line[0] == "-" and added:
            flush()
        added = line[0] == "+"
        if active is None:
            active = _Builder()
        if line[0] in " -+":
            active.add(line[0], line[1:])
        else:
            # An unprefixed line is context that lost its leading space; the
            # whitespace-insensitive match finds it either way.
            active.add(" ", line)
        index += 1
    flush()
    if not files:
        raise PatchParseError("no file headers in patch")
    return files


# --- apply_patch (V4A) and unified diff ------------------------------------


def _clean_path(raw: str) -> str:
    raw = raw.strip()
    while raw.startswith("./"):
        raw = raw[2:]
    return raw


def _diff_path(raw: str) -> str:
    """A diff header path, made relative: no tab-separated timestamp, no
    ``a/``/``b/`` prefix; empty for /dev/null."""
    raw = raw.strip().split("\t", 1)[0].strip()
    if raw == "/dev/null":
        return ""
    for prefix in ("a/", "b/"):
        if raw.startswith(prefix):
            return _clean_path(raw[len(prefix) :])
    return _clean_path(raw)


def _parse_diff(lines: list[str]) -> list[_File]:
    files: list[_File] = []
    current: _File | None = None
    active: _Builder | None = None
    # The file named by "--- ", kept in case the "+++ " that normally takes
    # priority never comes: a model wrote "--- path" then went straight to
    # "@@", which used to cost a whole regenerated diff.
    pending = ""
    # Whether the previous line was read as a "--- " header: only then is a
    # "+++ " inside a hunk a header rather than an added "++ " line.
    after_minus = False
    pending_create = False

    def close() -> None:
        nonlocal active
        if current is not None and active is not None:
            current.hunks.append(active.hunk())
        active = None

    for position, line in enumerate(lines):
        nxt = lines[position + 1] if position + 1 < len(lines) else ""
        was_minus, after_minus = after_minus, False
        if line.startswith("diff --git "):
            close()
            pending, pending_create = "", False
        elif line.startswith("--- ") and (active is None or nxt.startswith("+++ ")):
            # Inside a hunk, "--- x" is a removed "-- x" (an SQL comment, say)
            # unless the "+++ " of a new file header follows it.
            close()
            raw = line[4:]
            pending = _diff_path(raw)
            after_minus = True
            pending_create = raw.strip().split("\t", 1)[0] == "/dev/null"
        elif line.startswith("+++ ") and (active is None or was_minus):
            close()
            path = _diff_path(line[4:])
            if not path:
                raise _refuse("-", pending or "a file")
            current = _File(path=path, create=pending_create)
            files.append(current)
            pending, pending_create = "", False
        elif line.startswith("*** End Patch"):
            close()
            # It closes a block, and models write several; anything that is
            # not another block is not part of the patch at all.
            if not _more_blocks_after(lines, position + 1):
                break
        elif line.startswith(("*** Begin Patch", "*** End of File")):
            close()
        elif line.startswith(("*** Update File:", "*** Add File:")):
            close()
            path = _diff_path(line.partition(":")[2])
            if not path:
                current = None
                continue
            current = _File(path=path, create=line.startswith("*** Add File:"))
            files.append(current)
            # "*** Add File:" is followed straight by its "+" lines with no
            # "@@"; start collecting now or the whole file is read as noise.
            active = _Builder()
        elif line.startswith("*** Delete File:"):
            raise _refuse("-", line.partition(":")[2].strip())
        elif line.startswith("*** Move to:"):
            raise _refuse(
                ">", current.path if current else line.partition(":")[2].strip()
            )
        elif line.startswith("@@"):
            close()
            if current is None and pending:
                current = _File(path=pending)
                files.append(current)
            pending = ""
            if current is None:
                raise PatchParseError("a hunk appears before any file header")
            active = _Builder()
        elif active is None:
            pass  # preamble, index lines, trailing noise
        elif line.startswith(("-", "+", " ")):
            active.add(line[0], line[1:])
        elif line == "":
            active.add(" ", "")
        elif line.startswith("\\"):
            pass  # "\ No newline at end of file"
        else:
            close()
    close()
    if not files:
        raise PatchParseError("no file headers in patch")
    return files


# ---------------------------------------------------------------------------
# placing
# ---------------------------------------------------------------------------


@dataclass
class _Failure:
    kind: str
    reason: str
    detail: str
    block: tuple[str, ...] = ()
    index: int = 0


def apply_patch(
    edits: list[FileEdit], read: Callable[[str], str | None]
) -> ApplyResult:
    """Place every edit by content and report every one that will not place.

    ``read(path)`` returns a file's current text, or None if it does not
    exist. Nothing is written: the result carries the new text of every file
    whose edits all placed (written even when another file failed, so a
    repair does not have to resend them), and a diagnosis for every hunk
    that did not.
    """
    result = ApplyResult()
    files: dict[str, _FileState] = {}
    failures: dict[str, list[_Failure]] = {}
    missing: set[str] = set()

    for edit in edits:
        state = files.get(edit.path)
        if state is None and edit.path not in missing:
            raw = read(edit.path)
            if raw is None and not edit.create:
                missing.add(edit.path)
                failures.setdefault(edit.path, []).append(
                    _Failure(
                        "no_such_file",
                        "no such file",
                        f"{edit.path} does not exist. To create it, use @+{edit.path} "
                        "followed by its whole content.",
                        index=-1,
                    )
                )
                continue
            state = files[edit.path] = _FileState.of(raw)
        if state is None:
            continue
        # A patch may touch the same file more than once (several "*** Begin
        # Patch" blocks for one file); each block builds on the last, or the
        # final one silently discards every change before it. Kept even when
        # some hunks failed, so a later block is checked against what did
        # place rather than cascading into spurious failures.
        state.lines, hunk_failures = _apply_file_edit(state.lines, edit, result.repairs)
        if hunk_failures:
            failures.setdefault(edit.path, []).extend(hunk_failures)

    if failures:
        _misrouted(
            failures, {path: state.original_lines for path, state in files.items()}
        )

    for path, state in files.items():
        if path in failures:
            continue
        text = state.render()
        if state.original is None or text != state.original:
            result.written[path] = text
    for path, group in failures.items():
        result.failed[path] = [
            Diagnosis(
                path=path, hunk_index=one.index, kind=one.kind, message=one.detail
            )
            for one in group
        ]
    result.noop = not result.failed and not result.written
    return result


@dataclass
class _FileState:
    original: str | None
    original_lines: list[str]
    lines: list[str]
    eol: str
    trailing_newline: bool

    @classmethod
    def of(cls, raw: str | None) -> _FileState:
        if raw is None:
            return cls(None, [], [], "\n", True)
        crlf = raw.count("\r\n")
        eol = "\r\n" if crlf and crlf >= raw.count("\n") - crlf else "\n"
        text = raw.replace("\r\n", "\n") if eol == "\r\n" else raw
        trailing = text.endswith("\n") or not text.strip()
        body = text.removesuffix("\n")
        lines = body.split("\n") if body else []
        return cls(raw, list(lines), lines, eol, trailing)

    def render(self) -> str:
        if not self.lines:
            return ""
        text = "\n".join(self.lines)
        if self.trailing_newline:
            text += "\n"
        return text.replace("\n", self.eol) if self.eol != "\n" else text


# Shares of the file an added-only patch must reach, in characters and in
# lines, to be read as the whole file rewritten rather than an insertion that
# forgot its anchor (Mochi's rule, with the line share added so that two
# added lines in a three-line file stay an anchorless insertion).
_REWRITE_SHARE = 0.6


def _apply_file_edit(
    lines: list[str], edit: FileEdit, repairs: list[str]
) -> tuple[list[str], list[_Failure]]:
    if edit.create:
        content = list(edit.hunks[0].after) if edit.hunks else []
        while content and not content[-1].strip():
            content.pop()
        if content and content == lines:
            repairs.append("already_in_place")
        # "@+" on a file that exists writes it whole: a rewrite, the way a new
        # look is often easiest to give.
        return content, []

    hunks = list(edit.hunks)
    if hunks and all(not current.before for current in hunks):
        # Nothing but added lines. Into a blank file they are its content;
        # about as long as a nonblank file, they are a whole new version of it
        # written as "+" lines (what @+ is for), split into several edits by
        # its own blank lines.
        whole: list[str] = []
        for number, current in enumerate(hunks):
            whole.extend(([""] if number else []) + list(current.after))
        current_text = "\n".join(lines)
        if not current_text.strip() or (
            len("\n".join(whole)) >= len(current_text) * _REWRITE_SHARE
            and len(whole) >= len(lines) * _REWRITE_SHARE
        ):
            if current_text.strip():
                repairs.append("whole_file_rewrite")
            return whole, []
    return _apply_hunks(lines, hunks, repairs)


def _apply_hunks(
    lines: list[str],
    hunks: Sequence[Hunk],
    repairs: list[str],
    indexes: Sequence[int] | None = None,
) -> tuple[list[str], list[_Failure]]:
    """Place every hunk it can and report every one it cannot. A failed hunk
    is skipped rather than aborting, so the rest are still checked."""
    lines = list(lines)
    failures: list[_Failure] = []
    # Where the last placed edit ended, or -1 before any has: an edit whose
    # lines appear more than once takes the first match from here on.
    cursor = -1
    for number, current in enumerate(hunks):
        hunk_index = indexes[number] if indexes is not None else number
        # An edit of context alone points at where the next one goes, the way
        # apply_patch's "@@ const createRoom" does. A real run lost five edits
        # when such a pointer, matching twice in a stylesheet, was rejected as
        # ambiguous and took the patch down with it.
        if _changes_nothing(current):
            at = _first_after(lines, current.before, cursor)
            if at is not None:
                cursor = at + len(current.before)
            continue
        # The files of a failed patch that did place are written, so a repair
        # resending the whole patch carries edits that have landed. Taking
        # them again would fail a replacement and double an insertion.
        if _already_in_place(lines, current):
            repairs.append("already_in_place")
            continue
        if not current.before:
            if not any(line.strip() for line in lines):
                lines = list(current.after)
                continue
            reason = "a hunk has no context to place it by"
            failures.append(
                _Failure(
                    "no_anchor_insert",
                    reason,
                    reason
                    + ". Added lines alone do not locate an insertion in this nonempty file. "
                    "Include a short, unique unchanged line copied from the current file, prefixed "
                    "with a space, and place the added lines before or after it. For a replacement, "
                    "include the exact old lines prefixed with '-'. To rewrite the whole file, use "
                    "@+path followed by its new content.",
                    index=hunk_index,
                )
            )
            continue
        found = _locate_after(lines, current.before, cursor)
        if isinstance(found, _Failure):
            # Mechanical repairs, each from a traced failure, tried in order.
            placed = _place_truncated(lines, current)
            if placed is not None:
                lines = placed
                repairs.append("truncated_line")
                continue
            parts = _split_by_anchor(lines, current)
            if parts is not None:
                split, part_failures = _apply_hunks(lines, parts, [])
                if not part_failures:
                    lines = split
                    repairs.append("split_by_anchor")
                    continue
            replaced = _within_line(lines, current)
            if replaced is not None:
                lines = replaced
                repairs.append("within_line")
                continue
            trimmed = _without_stray_removals(lines, current)
            if trimmed is not None:
                at = _locate(lines, trimmed.before)
                if isinstance(at, int):
                    lines = (
                        lines[:at]
                        + list(trimmed.after)
                        + lines[at + len(trimmed.before) :]
                    )
                    repairs.append("stray_removals")
                    continue
            found.index = hunk_index
            failures.append(found)
            continue
        lines = (
            lines[:found] + list(current.after) + lines[found + len(current.before) :]
        )
        cursor = found + len(current.after)
    return lines, failures


def _changes_nothing(current: Hunk) -> bool:
    return bool(current.before) and current.before == current.after


def _already_in_place(lines: list[str], current: Hunk) -> bool:
    """The file already holds this edit: its after lines sit, exactly and
    once, where its before lines would be. Only an edit that adds something
    counts; one that only removes would read as in place whenever its
    context survives, including when its removed lines were copied wrong."""
    if current.ops:
        adds = any(kind == "+" for kind, _ in current.ops)
    else:
        adds = bool(current.after) and current.after != current.before
    if not adds:
        return False
    at = _find_all(lines, current.after, _exact)
    if len(at) != 1:
        return False
    befores = _find_all(lines, current.before, _exact) if current.before else []
    if not befores:
        return True
    return any(
        start >= at[0] and start + len(current.before) <= at[0] + len(current.after)
        for start in befores
    )


def _locate_after(
    lines: list[str], block: Sequence[str], cursor: int
) -> int | _Failure:
    """Like _locate, except that when the block's lines appear more than once
    and an earlier edit in this file has placed, the first match after that
    edit is taken. Edits in a file are read in order, as apply_patch reads
    them: an anchor, "@@", then a lone "};" means the "};" after the anchor —
    placed independently it matched three places and cost two rounds."""
    if cursor >= 0:
        matches = _candidates(lines, block)
        if len(matches) > 1:
            for index in matches:
                if index >= cursor:
                    return index
    return _locate(lines, block)


def _first_after(lines: list[str], block: Sequence[str], cursor: int) -> int | None:
    for index in _candidates(lines, block):
        if index >= cursor:
            return index
    return None


def _exact(a: str, b: str) -> bool:
    return a == b


def _stripped(a: str, b: str) -> bool:
    return a.strip() == b.strip()


_ENTITIES = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#39;": "'",
    "&apos;": "'",
}
_ENTITY = re.compile(r"&(?:amp|lt|gt|quot|#39|apos);")
_SPACES = re.compile(r"\s+")


def _decode_entities(line: str) -> str:
    return _ENTITY.sub(lambda match: _ENTITIES[match.group(0)], line.strip())


def _entities(a: str, b: str) -> bool:
    return _decode_entities(a) == _decode_entities(b)


def _squeeze(line: str) -> str:
    return _SPACES.sub(" ", _decode_entities(line))


def _squeezed(a: str, b: str) -> bool:
    return _squeeze(a) == _squeeze(b)


def _candidates(lines: list[str], block: Sequence[str]) -> list[int]:
    """Where block matches, by the strictest test that finds it anywhere:
    exactly; ignoring surrounding whitespace (indentation drift); with HTML
    entities decoded (the mistake the prompt warns about, recovered when the
    warning does not work); with runs of whitespace squeezed (aligned columns
    in docs and tables counted wrong)."""
    for equal in (_exact, _stripped, _entities, _squeezed):
        matches = _find_all(lines, block, equal)
        if matches:
            return matches
    return []


def _find_all(
    lines: Sequence[str], block: Sequence[str], equal: Callable[[str, str], bool]
) -> list[int]:
    if not block:
        return []
    height = len(block)
    return [
        start
        for start in range(len(lines) - height + 1)
        if all(equal(lines[start + offset], want) for offset, want in enumerate(block))
    ]


def _locate(lines: list[str], block: Sequence[str]) -> int | _Failure:
    """The one place block occurs, or why there is not exactly one.

    Unlike both earlier ports, a block that matches more than once is never
    placed at its first match, however long it is: a repeated three-line
    block (two identical cards in a page) is exactly where editing the wrong
    one goes unnoticed. The ordered placement in _locate_after resolves the
    common case; anything it cannot is sent back with the places to choose
    between.
    """
    matches = _candidates(lines, block)
    if not matches:
        # One or two lines are nowhere in the file while the rest are all
        # there: almost always a marker this format did not recognise ("@@",
        # "***" and "*** End Patch" have each cost a whole turn) or an
        # invented line. Naming it needs no list of conventions.
        absent = _absent_lines(lines, block)
        if absent:
            reason = "this hunk has a line that is nowhere in the file"
            return _Failure(
                "absent_lines",
                reason,
                reason
                + ":"
                + absent
                + "\n\nEverything else in the hunk was found. If that line was meant to divide two "
                "edits, a blank line does that; otherwise copy it from the file exactly.",
            )
        # Every line is in the file, just not together: anchors gathered from
        # all over a file into one hunk. Said as such, because the near-miss
        # report would print a line as differing from itself.
        scattered = _scattered_lines(lines, block)
        if scattered:
            reason = (
                "this hunk's lines are each in the file, but not next to each other"
            )
            return _Failure(
                "scattered",
                reason,
                reason
                + ":"
                + scattered
                + "\n\nA hunk's lines have to be consecutive in the file. Separate edits go in "
                "separate hunks, each under its own header or separated by a blank line.",
            )
        expected, found, at = _near_miss(lines, block)
        reason = "could not find this hunk's lines in the file"
        detail = f"{reason}:\n{_preview(block)}"
        if expected:
            detail += (
                f"\n\nthe closest match is at line {at}, where it differs:\n"
                f"  expected: {_quote(expected)}\n  in file:  {_quote(found)}"
            )
            return _Failure("near_miss", reason, detail, block=tuple(block))
        return _Failure("not_found", reason, detail, block=tuple(block))
    if len(matches) > 1:
        reason = (
            f"this hunk's lines appear {len(matches)} times, too ambiguous to place"
        )
        return _Failure(
            "ambiguous",
            reason,
            f"{reason}:\n{_preview(block)}{_disambiguate(lines, matches, block)}",
        )
    return matches[0]


def _quote(text: str) -> str:
    """Go's %q, near enough: double-quoted with escapes."""
    return (
        '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\t", "\\t") + '"'
    )


# How far the ambiguity report reaches for something that tells candidates
# apart. Past this they are genuinely alike and more lines only hurt.
_MAX_DISAMBIGUATE = 6


def _window(
    lines: Sequence[str], index: int, height: int, span: int
) -> tuple[int, int]:
    return max(0, index - span), min(len(lines), index + height + span)


def _distinct_around(
    lines: Sequence[str], matches: Sequence[int], span: int, height: int
) -> bool:
    seen = set()
    for index in matches:
        low, high = _window(lines, index, height, span)
        key = "\n".join(lines[low:high])
        if key in seen:
            return False
        seen.add(key)
    return True


def _disambiguate(lines: list[str], matches: list[int], block: Sequence[str]) -> str:
    """Each place the hunk could go, with enough of the real file around it to
    tell them apart. One line above was not enough: a stylesheet had the same
    selector twice, both preceded by "}", so the report offered the same text
    for both, and the model invented context and failed again. The window
    widens until the candidates differ, and the file's own text is printed to
    be copied rather than recalled."""
    span = 1
    while span < _MAX_DISAMBIGUATE and not _distinct_around(
        lines, matches, span, len(block)
    ):
        span += 1
    where = "either place" if len(matches) == 2 else "any of the places"
    parts = [
        (
            f"\n\nit could go in {where} below. Add context until only one matches, "
            "copying these lines exactly as they appear here:"
        )
    ]
    for index in matches:
        parts.append(f"\n\n  at line {index + 1}:")
        low, high = _window(lines, index, len(block), span)
        for at in range(low, high):
            marker = "  > " if index <= at < index + len(block) else "    "
            parts.append(f"\n{marker}{lines[at]}")
    parts.append(
        "\n\nThe lines marked > are the ones you already wrote; the rest is what surrounds "
        "them. Lines above and below go in as context, each starting with a space."
    )
    return "".join(parts)


def _near_miss(lines: list[str], block: Sequence[str]) -> tuple[str, str, int]:
    """The closest place the block almost fits and the first line that
    differs there: expected, in-file, 1-based line. Empty when nothing
    resembles it — a low-quality guess is worse than none."""
    best_start, best_score = -1, 0
    for start in range(len(lines) - len(block) + 1):
        score = sum(
            1
            for offset, want in enumerate(block)
            if lines[start + offset].strip() == want.strip()
        )
        if score > best_score:
            best_start, best_score = start, score
    # At least half the block must already line up.
    if best_start != -1 and best_score * 2 >= len(block):
        for offset, want in enumerate(block):
            got = lines[best_start + offset]
            if got != want:
                return want, got, best_start + offset + 1
    # A one-line block can never clear that bar, and one-line edits are the
    # common case in the compact format.
    return _closest_line(lines, block)


def _closest_line(lines: list[str], block: Sequence[str]) -> tuple[str, str, int]:
    want = next((line for line in block if line.strip()), "")
    if not want:
        return "", "", 0
    best_index, best_score = -1, 0.0
    for index, line in enumerate(lines):
        score = _similarity(line, want)
        if score > best_score:
            best_index, best_score = index, score
    # Below half, it points at an unrelated line. An exact match is worse:
    # reporting a line as differing from itself sends the model hunting for
    # a difference that is not there.
    if best_index == -1 or best_score < 0.5 or lines[best_index] == want:
        return "", "", 0
    return want, lines[best_index], best_index + 1


def _similarity(a: str, b: str) -> float:
    """How much of two lines agrees from each end: the shape of a line
    transcribed almost right. Deliberately not an edit distance; this runs
    against every line of every file in a patch."""
    if a == b:
        return 1.0
    longest = max(len(a), len(b))
    if not longest:
        return 0.0
    prefix = 0
    while prefix < len(a) and prefix < len(b) and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < len(a) - prefix
        and suffix < len(b) - prefix
        and a[-1 - suffix] == b[-1 - suffix]
    ):
        suffix += 1
    return (prefix + suffix) / longest


def _absent_lines(lines: list[str], block: Sequence[str]) -> str:
    """The few lines of a block with no relative anywhere in the file, when
    everything else is there. A line transcribed almost right is left to the
    near-miss report, which says more about it."""
    present = {line.strip() for line in lines}
    missing = []
    counted = 0
    for want in block:
        if not want.strip():
            continue
        counted += 1
        if want.strip() in present:
            continue
        if max((_similarity(line, want) for line in lines), default=0.0) < 0.5:
            missing.append(want)
    if not missing or len(missing) > 2 or counted - len(missing) < 2:
        return ""
    return "".join(f"\n  not in the file: {_clip(line.strip())}" for line in missing)


def _scattered_lines(lines: list[str], block: Sequence[str]) -> str:
    """Where each of a block's lines sits, when every one is in the file but
    they are not consecutive."""
    parts = []
    for want in block:
        if not want.strip():
            continue
        at = next(
            (
                index + 1
                for index, line in enumerate(lines)
                if line == want or line.strip() == want.strip()
            ),
            -1,
        )
        if at == -1:
            return ""
        parts.append(f"\n  line {at}: {_clip(want.strip())}")
    return "".join(parts) if len(parts) >= 2 else ""


def _misrouted(
    failures: dict[str, list[_Failure]], contents: dict[str, list[str]]
) -> None:
    """Rewrite any failure whose lines sit, whole and findable, in another file
    of this patch. Under the wrong header an edit reads as lines that are not
    there, and the model rewrites text that was right all along: a dropped
    header sent nine edits to look in a file they had never been in."""
    for path, group in failures.items():
        for failure in group:
            if not failure.block:
                continue
            for other, lines in contents.items():
                if other == path or not isinstance(_locate(lines, failure.block), int):
                    continue
                failure.kind = "misrouted"
                failure.reason = f"these lines are in {other}, not in {path}"
                failure.detail = f"{failure.reason}:\n{_preview(failure.block)}\n\nPut this edit under a header naming {other}."
                break


# --- repairs ---------------------------------------------------------------

# How much of a line must be present before a prefix is read as that line
# cut short: long enough that it cannot be the start of another by accident.
_TRUNCATED_FLOOR = 40
# The shortest removed text matched inside a line rather than at its start;
# shorter, and "}" or "return" would match by accident.
_MIN_FRAGMENT = 16


def _place_truncated(lines: list[str], current: Hunk) -> list[str] | None:
    """A hunk whose last removed line was cut off part-way: to change eight
    characters of a 300-character JSX line the format asks for all 300 twice,
    and a model that stops at 280 matches nothing (three traced runs). Every
    line but the last must match exactly, the last must be a long prefix of
    exactly one candidate, and the hunk must add something, because the tail
    is carried onto the last added line."""
    if not current.before or not current.after:
        return None
    head, last = current.before[:-1], current.before[-1]
    if len(last) < _TRUNCATED_FLOOR:
        return None
    index, found = -1, 0
    for start in range(len(lines) - len(current.before) + 1):
        if any(lines[start + offset] != want for offset, want in enumerate(head)):
            continue
        candidate = lines[start + len(head)]
        if len(candidate) <= len(last) or not candidate.startswith(last):
            continue
        found += 1
        if found > 1:
            return None
        index = start
    if found != 1:
        return None
    tail = lines[index + len(head)][len(last) :]
    after = list(current.after)
    after[-1] += tail
    return lines[:index] + after + lines[index + len(current.before) :]


def _split_by_anchor(lines: list[str], current: Hunk) -> list[Hunk] | None:
    """Break a hunk into the separate edits it was meant to be: wherever two
    consecutive context/removed lines are not adjacent in the file, cut.
    Added lines stay with the anchor above them. Every anchor must occur
    exactly once; a repeated line gives no honest answer about where its
    edit belongs."""
    if not current.ops:
        return None

    def at(text: str) -> int:
        found = -1
        for index, line in enumerate(lines):
            if line == text or line.strip() == text.strip():
                if found != -1:
                    return -1
                found = index
        return found

    parts: list[list[Op]] = []
    part: list[Op] = []
    previous = -1
    for kind, text in current.ops:
        if kind == "+":
            part.append((kind, text))
            continue
        position = at(text)
        if position == -1:
            return None
        if previous != -1 and position != previous + 1:
            parts.append(part)
            part = []
        part.append((kind, text))
        previous = position
    parts.append(part)
    if len(parts) < 2:
        return None
    return [Hunk.from_ops(ops) for ops in parts]


def _within_line(lines: list[str], current: Hunk) -> list[str] | None:
    """Replace part of one long line: one-line JSX puts a whole component on
    a line, and a model changing one button writes the button as the line it
    removes. Nowhere as a line, found once inside one, so where it goes is
    not in doubt (rejected in a real run on the last edit of its budget)."""
    removed: list[str] = []
    added: list[str] = []
    for kind, text in current.ops:
        if kind == "-":
            if added:
                return None
            removed.append(text)
        elif kind == "+":
            added.append(text)
        else:
            return None
    # Something must be added: removing a piece of a line with nothing in its
    # place is more likely a line copied short than a real deletion.
    if len(removed) != 1 or not added:
        return None
    fragment = removed[0].strip()
    if len(fragment) < _MIN_FRAGMENT:
        return None
    at, count = -1, 0
    for index, line in enumerate(lines):
        occurrences = line.count(fragment)
        if occurrences:
            at, count = index, count + occurrences
    if count != 1 or lines[at].strip() == fragment:
        return None
    start = lines[at].index(fragment)
    # A piece the line starts with is a line cut short: _place_truncated's case.
    if not lines[at][:start].strip():
        return None
    edited = (
        lines[at][:start]
        + "\n".join(added).strip()
        + lines[at][start + len(fragment) :]
    )
    return lines[:at] + edited.split("\n") + lines[at + 1 :]


def _without_stray_removals(lines: list[str], current: Hunk) -> Hunk | None:
    """The edit with removed lines at its edges dropped, when those lines are
    nowhere in the file and what remains places exactly once. Removing a line
    that is not there changes nothing: a model recalling a file's head wrote
    ``-import React from "react";`` where the file had none, and the rest of
    the edit was right. Only lines absent from the whole file are dropped
    (and blanks beside them), so a real line is never quietly left behind."""
    ops = current.ops
    lead = 0
    while lead < len(ops) and ops[lead][0] == "-":
        lead += 1
    trail = 0
    while trail < len(ops) - lead and ops[len(ops) - 1 - trail][0] == "-":
        trail += 1
    present = {line.strip() for line in lines}

    def stray(dropped: Sequence[Op]) -> bool:
        absent = False
        for _, text in dropped:
            text = text.strip()
            if not text:
                continue
            if text in present:
                return False
            absent = True
        return absent

    for total in range(1, lead + trail + 1):
        front = min(total, lead)
        while front >= 0 and total - front <= trail:
            back = total - front
            kept = ops[front : len(ops) - back]
            dropped = list(ops[:front]) + list(ops[len(ops) - back :])
            front -= 1
            if not stray(dropped):
                continue
            candidate = Hunk.from_ops(kept)
            if not candidate.before or _changes_nothing(candidate):
                continue
            if isinstance(_locate(lines, candidate.before), int):
                return candidate
    return None


def _clip(line: str) -> str:
    line = line.rstrip("\r\n")
    return line if len(line) <= 110 else line[:110] + "..."


def _preview(block: Sequence[str]) -> str:
    return "\n".join(block[:4])
