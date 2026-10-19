"""Find the steps in a scratchpad cell that change a database or existing files.

The host holds a cell that has any of them until the user approves it
(`hold_unapproved_write` in `anton.core.utils.scratchpad`). This module only
reads the cell's code; it never runs it.

What it finds:

- **SQL writes**, in every conversation: each statement whose first word is a
  write verb and whose clause shape matches it (UPDATE ... SET col =, DELETE
  FROM, INSERT INTO ... VALUES, CREATE TABLE, TRUNCATE, GRANT ... TO, and so
  on), that reaches something that could run it. The statement may be a
  string or bytes literal, an f-string, a ``+`` chain, ``str.format`` or ``%``
  on a template, a ``str.join``, a name built up with ``+=`` or ``.append``,
  a name an earlier cell bound, or code passed to ``exec`` or ``eval`` as a
  literal. Each statement in a string that holds a write is its own
  finding, writes or not, since one Approve runs them all. A
  statement only printed, logged or written to a file doesn't count, and
  neither does one run on an in-process DuckDB or SQLite handle. Any other
  handle, known or not, counts. ``to_sql`` into a database that isn't
  in-process counts too.
- **Deletes, renames and moves** of a file or folder that was in the project
  folder (artifacts and the host's own files under ``.anton`` included) or the
  shared folder when the turn began, either folder itself included. The call
  may be a Python one (``os``, ``shutil``, ``pathlib``, also reached through
  ``getattr``, ``vars`` or ``functools.partial`` or passed as a value) or a
  shell ``rm``, ``mv``, ``find -delete``, ``xargs rm`` or ``git clean`` in a
  command the cell builds, wrapped in ``sh -c``, piped into a shell or not. A
  list of paths the cell builds up counts with everything it adds, through
  any name or function it passes the list to, and a relative path counts
  from every folder the pad changes into. A target the code picks when it
  runs, from a root the check can't place, counts as well, because the check
  can't rule it out.
- Both kinds inside a **function, lambda or method an earlier cell
  defined**, at each place this cell calls it or passes it on, with its
  parameters read from the call: by its name, through an instance or the
  class, a list or dict that holds it, a ``functools.partial`` of it, or a
  function that returns it.

What it can't find: a statement or path with no literal behind it, a stored
procedure or function call (CALL, EXEC, SELECT f()) unless it is given the
statement it runs as a literal, a write with no SQL text other than
``to_sql``, a write through an HTTP or other non-SQL API, a shell command it
can't read at all, and code in a module or script the cell imports or runs.
An overwrite of an existing file's contents is reported as ``overwrite`` and
isn't held, and so is a delete that a write to the same path follows on every
path through the code.
"""

from __future__ import annotations

import ast
import copy
import fnmatch
import os
import posixpath
import re
import shlex
import string
import textwrap
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from anton.core.backends.wire import heal_surrogate_source
from anton.utils.datasources import scrub_output

__all__ = [
    "CellWrites",
    "FileChange",
    "ProjectSnapshot",
    "SqlWrite",
    "find_cell_writes",
    "snapshot_project_files",
    "write_statements",
]

# Folders recorded as entries but not walked: big, and never the agent's
# answer. A path under one still counts as one that existed.
_UNWALKED = frozenset({".git", "node_modules", ".venv", "venv", "__pycache__"})
# The folders under the project's .anton that the agent writes in, walked like
# any other: `output` is where the CLI saves what it makes when it has no
# artifacts. Every other folder there is the host's (scratchpad venvs, memory,
# saved state) and is recorded but not walked.
_ANTON_WALKED = frozenset({"artifacts", "skill_drafts", "output"})
# A listing names entries under these after the project's own files and folders.
_HOST_FOLDERS = _UNWALKED | {".anton"}
# Entries a walk records before it stops; past it, a path on disk counts
# unless the turn's cells made it (`ProjectSnapshot.named`).
_SNAPSHOT_CAP = 200_000
# Entries a glob reads from disk, under a folder the walk didn't record,
# before it gives up and counts the pattern itself.
_SCAN_LIMIT = 50_000
_WORKSPACE_ENV = "ANTON_SCRATCHPAD_WORKSPACE_PATH"


class SqlWrite(BaseModel):
    """One statement that changes a database, whole and scrubbed, as the
    card quotes it."""

    model_config = ConfigDict(frozen=True)

    statement: str
    values: str = ""
    line: int
    # A part of `statement` in braces is the code that fills it in when the
    # cell runs, such as `{r.new}` for an f-string's placeholder.
    run_time_parts: bool = False


class FileChange(BaseModel):
    """One delete, rename, move or overwrite of an existing file."""

    model_config = ConfigDict(frozen=True)

    action: Literal["delete", "rename", "move", "overwrite"]
    # Snapshot paths, the shared folder's prefixed with "shared folder: ".
    # Empty when the code picks the path when it runs.
    paths: tuple[str, ...] = ()
    whole_project: bool = False
    whole_shared: bool = False
    call: str
    line: int


class CellWrites(BaseModel):
    """What a cell would change. ``unchecked`` means the check itself failed."""

    model_config = ConfigDict(frozen=True)

    sql: tuple[SqlWrite, ...] = ()
    files: tuple[FileChange, ...] = ()
    unchecked: bool = False

    @property
    def held_files(self) -> tuple[FileChange, ...]:
        return tuple(f for f in self.files if f.action != "overwrite")

    @property
    def overwrites(self) -> tuple[FileChange, ...]:
        return tuple(f for f in self.files if f.action == "overwrite")

    def needs_approval(self) -> bool:
        return bool(self.sql or self.held_files or self.unchecked)

    def kinds(self) -> tuple[str, ...]:
        kinds = ["sql"] if self.sql else []
        kinds += sorted({f.action for f in self.files})
        if self.unchecked:
            kinds.append("unchecked")
        return tuple(kinds)


# What one check read from disk under folders the walk didn't walk: the
# matches for a root, a pattern, a folder, and whether only older ones count.
_DiskReads = dict[tuple[str, str, str, bool], list[str]]


class ProjectSnapshot(BaseModel):
    """The paths under one root when the turn began, relative and POSIX."""

    model_config = ConfigDict(frozen=True)

    root: Path
    label: Literal["project", "shared"]
    files: frozenset[str] = frozenset()
    dirs: frozenset[str] = frozenset()
    unwalked: frozenset[str] = frozenset()
    taken_at: float
    # Past the entry cap the walk stops, and a path on disk counts as existing
    # unless the turn changed it and the pad's cells made it new (`named`).
    complete: bool = True
    # The walk couldn't read the root or a folder under it: every path under
    # the root counts as existing.
    failed: bool = False
    # The root as the host spelled it, when that isn't its realpath `root`.
    given_root: Path | None = None
    # The root's filesystem ignores case: paths match whatever their case, and
    # `spellings` maps each recorded path, casefolded, to how it was recorded.
    casefold: bool = False
    spellings: dict[str, str] = {}

    def is_empty(self) -> bool:
        return not (self.files or self.dirs) and self.complete and not self.failed

    def roots(self) -> tuple[str, ...]:
        """The root's spellings an absolute path may start with."""
        return tuple(dict.fromkeys(str(r) for r in (self.root, self.given_root) if r is not None))

    def key(self, path: str) -> str:
        """`path` as the root's filesystem compares it."""
        return path.casefold() if self.casefold else path

    def recorded(self, *, rel: str) -> str | None:
        """`rel` as the walk recorded it; None when it didn't."""
        if rel in self.files or rel in self.dirs:
            return rel
        return self.spellings.get(rel.casefold()) if self.casefold else None

    def named(self, *, rel: str, made: _Made | None = None) -> str | None:
        """The name a card gives `rel` when it existed at turn start: its
        recorded spelling, or `rel` itself when only a fallback counts it.

        Past the cap, a path on disk the walk didn't record counts, unless the
        turn changed it and `made`, what the pad's cells write, made it new.
        """
        found = self.recorded(rel=rel)
        if found is not None:
            return found
        if self.failed or self._under_unwalked(rel):
            return rel
        if not self.complete:
            try:
                changed = (self.root / rel).lstat().st_mtime >= self.taken_at
            except OSError:
                return None
            return None if changed and made is not None and made.new(snap=self, rel=rel) else rel
        return None

    def holds(self, *, rel: str, made: _Made | None = None) -> bool:
        return self.named(rel=rel, made=made) is not None

    def is_file(self, *, rel: str) -> bool:
        return self.recorded(rel=rel) in self.files

    def is_dir(self, *, rel: str) -> bool:
        return self.recorded(rel=rel) in self.dirs

    def matching(self, *, pattern: str, made: _Made | None = None, disk: _DiskReads | None = None) -> list[str]:
        """Snapshot paths a glob-style pattern names, `*` within one folder and
        `**` across folders, as glob reads them: files first, then folders,
        then entries under the host's folders.

        Under a folder the walk recorded but didn't walk, the matches on disk
        count, as `holds` counts any path there. Past the cap, the matches on
        disk count unless the turn changed one and `made` made it new. `disk`
        keeps what one check read from disk, so each pattern reads each folder
        once per check: one check reads with one `made`.
        """
        if self.failed:
            return [pattern]

        def on_disk(start: str, *, older: bool) -> list[str]:
            if disk is None:
                return self._on_disk(pattern=pattern, start=start, older=older, made=made)
            read = (str(self.root), pattern, start, older)
            if read not in disk:
                disk[read] = self._on_disk(pattern=pattern, start=start, older=older, made=made)
            return disk[read]

        key = self.key(pattern)
        if self.casefold:
            hits = [p for folded, p in self.spellings.items() if glob_match(folded, key)]
        else:
            hits = [p for p in (*self.files, *self.dirs) if glob_match(p, key)]
        for folder in sorted(self.unwalked):
            if _reaches_under(key.split("/"), self.key(folder).split("/")):
                hits += on_disk(folder, older=False)
        if not self.complete:
            hits += on_disk("", older=True)
        return sorted(dict.fromkeys(hits), key=self._listing_order)

    def _under_unwalked(self, rel: str) -> bool:
        key = self.key(rel)
        return any(key == self.key(u) or key.startswith(self.key(u) + "/") for u in self.unwalked)

    def _listing_order(self, path: str) -> tuple[int, str]:
        head, _, rest = path.partition("/")
        host = head in _HOST_FOLDERS and not (head == ".anton" and rest.split("/")[0] in _ANTON_WALKED)
        return (2 if host else 1 if path in self.dirs else 0, path)

    def _on_disk(self, *, pattern: str, start: str, older: bool, made: _Made | None = None) -> list[str]:
        """Entries under `start` on disk that `pattern` names; with `older`,
        not those the turn changed and `made` made new. Past the scan limit,
        or on a folder it can't read, the pattern itself counts too."""
        key = self.key(pattern)
        parts = key.split("/")
        out: list[str] = []
        unread = False
        seen = 0

        def on_error(_exc: OSError) -> None:
            nonlocal unread
            unread = True

        for dirpath, dirnames, filenames in os.walk(self.root / start, onerror=on_error):
            rel_dir = os.path.relpath(dirpath, self.root).replace(os.sep, "/")
            rel_dir = "" if rel_dir == "." else rel_dir
            seen += len(dirnames) + len(filenames)
            if seen > _SCAN_LIMIT:
                return [*out, pattern]
            for name in (*dirnames, *filenames):
                rel = posixpath.join(rel_dir, name) if rel_dir else name
                if glob_match(self.key(rel), key) and not (
                    older and not self._older(rel) and made is not None and made.new(snap=self, rel=rel)
                ):
                    out.append(rel)
            dirnames[:] = [
                d for d in dirnames
                if _reaches_under(parts, self.key(posixpath.join(rel_dir, d) if rel_dir else d).split("/"))
            ]
        return [*out, pattern] if unread else out

    def _older(self, rel: str) -> bool:
        try:
            return (self.root / rel).lstat().st_mtime < self.taken_at
        except OSError:
            return False


def glob_match(path: str, pattern: str) -> bool:
    return _match_parts(path.split("/"), pattern.split("/"))


def _match_parts(parts: list[str], patterns: list[str]) -> bool:
    if not patterns:
        return not parts
    head, rest = patterns[0], patterns[1:]
    if head == "**" and "**" not in rest:
        # The rest names one part each, so `**` takes the parts before them.
        return len(parts) >= len(rest) and _match_parts(parts[len(parts) - len(rest):], rest)
    if head == "**":
        return any(_match_parts(parts[i:], rest) for i in range(len(parts) + 1))
    # A part spelled exactly as the pattern's matches it, so a name holding
    # `[` or `*` matches itself.
    return (
        bool(parts)
        and (parts[0] == head or fnmatch.fnmatchcase(parts[0], head))
        and _match_parts(parts[1:], rest)
    )


def _reaches_under(pattern: list[str], folder: list[str]) -> bool:
    """Whether a pattern's parts can name an entry below `folder`'s parts."""
    for i, part in enumerate(folder):
        if i >= len(pattern):
            return False
        if pattern[i] == "**":
            return True
        if not (pattern[i] == part or fnmatch.fnmatchcase(part, pattern[i])):
            return False
    return len(pattern) > len(folder)


def _folds_case(folder: Path) -> bool:
    """Whether `folder`'s filesystem ignores case: an entry in it, its case
    flipped, is the same file."""
    try:
        with os.scandir(folder) as entries:
            names = [e.name for e in entries if e.name.swapcase() != e.name][:8]
    except OSError:
        return False
    for name in names:
        try:
            return os.path.samefile(folder / name, folder / name.swapcase())
        except OSError:
            continue
    return False


def snapshot_project_files(
    *,
    root: Path,
    label: Literal["project", "shared"],
    taken_at: float,
    cap: int = _SNAPSHOT_CAP,
) -> ProjectSnapshot:
    """Record the files and folders under `root`.

    In the project, ``.anton`` holds the host's own files and folders: its
    files are recorded, ``.anton/artifacts``, ``.anton/skill_drafts`` and
    ``.anton/output``, where the agent writes, are walked, and every other
    folder there is recorded unwalked, so a path under it counts as existing
    without walking the scratchpad venvs. A folder the walk can't read marks
    the snapshot failed, which makes every path under the root count as
    existing: an unread folder must not let a delete through.
    """
    real = Path(os.path.realpath(root))
    given = Path(os.path.abspath(root))
    files: set[str] = set()
    dirs: set[str] = set()
    unwalked: set[str] = set()
    failed = not real.is_dir()
    complete = True

    def _on_error(_exc: OSError) -> None:
        nonlocal failed
        failed = True

    if not failed:
        for dirpath, dirnames, filenames in os.walk(real, onerror=_on_error):
            rel_dir = os.path.relpath(dirpath, real)
            rel_dir = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")
            hosts = label == "project" and rel_dir == ".anton"
            walk = []
            for name in dirnames:
                rel = posixpath.join(rel_dir, name) if rel_dir else name
                dirs.add(rel)
                if name in _UNWALKED or hosts and name not in _ANTON_WALKED:
                    unwalked.add(rel)
                else:
                    walk.append(name)
            dirnames[:] = walk
            for name in filenames:
                files.add(posixpath.join(rel_dir, name) if rel_dir else name)
            if len(files) + len(dirs) >= cap:
                complete = False
                break
    casefold = not failed and _folds_case(real)
    return ProjectSnapshot(
        root=real,
        label=label,
        files=frozenset(files),
        dirs=frozenset(dirs),
        unwalked=frozenset(unwalked),
        taken_at=taken_at,
        complete=complete,
        failed=failed,
        given_root=given if given != real else None,
        casefold=casefold,
        spellings={p.casefold(): p for p in (*files, *dirs)} if casefold else {},
    )


# ── SQL ───────────────────────────────────────────────────────────────────

# Every token after a comment. A hole marker (`_HOLE`) reads as part of a word.
_TOKEN_TAIL = (
    r"""|(?P<text>'(?:[^']|'')*'|\$(?P<tag>\w*)\$.*?\$(?P=tag)\$)"""
    r"""|(?P<name>"(?:[^"]|"")*"|`[^`]*`|\[[A-Za-z_@#$][^\]\n]*\])"""
    r"""|(?P<hole>\{[^{}\s;]*\}|%(?:\(\w+\))?[sd])"""
    r"""|(?P<end>;)|(?P<word>[A-Za-z_\ue000][A-Za-z_0-9\ue000\ue001]*)|(?P<open>\()|(?P<close>\))|(?P<punct>\S)"""
)
# A `--` comment ends at a line feed or a carriage return, as Postgres ends
# it. `#` is code: Postgres's XOR, or a T-SQL temp table's name.
_TOKEN = re.compile(r"""(?P<comment>--[^\n\r]*|/\*.*?\*/)""" + _TOKEN_TAIL, re.S | re.M)
# MySQL's reading: a `#` at the start of a line starts a comment too.
_HASH_COMMENT_TOKEN = re.compile(
    r"""(?P<comment>--[^\n\r]*|/\*.*?\*/|^[ \t]*\#[^\n]*)""" + _TOKEN_TAIL, re.S | re.M,
)
_HASH_LINE = re.compile(r"^[ \t]*#", re.M)
_LINE_BREAK = re.compile(r"\r\n|\r|\n")
_OBJECTS = frozenset({
    "TABLE", "VIEW", "INDEX", "SCHEMA", "DATABASE", "FUNCTION", "PROCEDURE",
    "SEQUENCE", "TRIGGER", "TYPE", "EXTENSION", "ROLE", "USER", "STAGE",
    "STREAM", "PIPE", "WAREHOUSE", "TABLESPACE", "SYNONYM", "PUBLICATION",
    "SUBSCRIPTION",
})
# Words that may sit between CREATE and the object, as in CREATE OR ALTER VIEW
# or CREATE OR REPLACE DYNAMIC TABLE. TEMP is not one, so a temporary object
# stops the rule before it reaches the object word.
_CREATE_MODIFIERS = frozenset({
    "OR", "REPLACE", "ALTER", "UNIQUE", "MATERIALIZED", "EXTERNAL", "TRANSIENT",
    "SECURE", "RECURSIVE", "UNLOGGED", "VIRTUAL", "LOCAL", "GLOBAL", "CLUSTERED",
    "NONCLUSTERED", "FORCE", "NOFORCE", "BITMAP", "DYNAMIC", "ICEBERG", "FOREIGN",
    "HYBRID", "EDITIONABLE", "NONEDITIONABLE",
})
_DROP_MODIFIERS = frozenset({"MATERIALIZED", "EXTERNAL", "DYNAMIC", "FOREIGN", "ICEBERG", "HYBRID"})
_TEMP = frozenset({"TEMP", "TEMPORARY", "VOLATILE"})
_DML = frozenset({"INSERT", "UPDATE", "DELETE", "MERGE"})
_TRUNCATE_TAIL = frozenset({
    "RESTART", "CONTINUE", "IDENTITY", "CASCADE", "RESTRICT", "DROP", "REUSE",
    "STORAGE", "PRESERVE", "PURGE", "MATERIALIZED", "VIEW", "LOG", "PARTITION",
    "PARTITIONS", "WITH",
})
# What may follow DELETE FROM's table, directly or after an alias.
_DELETE_FOLLOWERS = frozenset({
    "WHERE", "USING", "RETURNING", "OUTPUT", "FROM", "JOIN", "INNER", "LEFT",
    "RIGHT", "FULL", "CROSS", "STRAIGHT_JOIN", "PARTITION", "ORDER", "LIMIT",
    "WITH", "OPTION", "AS",
})
_INSERT_MODIFIERS = frozenset({"LOW_PRIORITY", "DELAYED", "HIGH_PRIORITY", "IGNORE", "OVERWRITE"})
# English words that stand where a statement names its table only in prose,
# as in "Insert the values ..." or "Create table of contents": SQL reserves
# each, or no table is named it.
_NOT_NAMES = frozenset({
    "THE", "AN", "OF", "FOR", "TO", "AND", "OR", "WITH", "FROM", "BY", "INTO", "IN", "ON", "AS", "AT",
})
# What may follow the object's name in a CREATE written in sentence case,
# besides a column or parameter list, `option = value` or a placeholder.
_CREATE_FOLLOWERS = frozenset({"AS", "LIKE", "CLONE", "ON"})
# What may follow a dropped object's name in a DROP written in sentence case.
_DROP_FOLLOWERS = frozenset({"CASCADE", "RESTRICT", "PURGE", "ON", "FORCE"})
# What may follow the object's name in an ALTER written in sentence case.
_ALTER_ACTIONS = frozenset({
    "ADD", "DROP", "ALTER", "RENAME", "SET", "RESET", "UNSET", "OWNER", "MODIFY", "CHANGE",
    "ENABLE", "DISABLE", "ATTACH", "DETACH", "REFRESH", "SWAP", "SUSPEND", "RESUME",
})
# Words a T-SQL script starts a statement with, on its own line, with no `;`
# before the next one.
_SCRIPT_WORDS = frozenset({
    "SELECT", "SET", "DECLARE", "USE", "BEGIN", "IF", "ELSE", "WHILE", "PRINT", "GO", "END",
    "COMMIT", "ROLLBACK", "SAVE", "RAISERROR", "THROW", "RETURN", "WAITFOR", "OPEN", "FETCH",
    "CLOSE", "DEALLOCATE", "DBCC", "BULK", "DENY", "BREAK", "CONTINUE",
})
_GRANTEE_KINDS = frozenset({"ROLE", "USER", "GROUP", "SHARE", "APPLICATION", "DATABASE"})
# First words of a privilege granted with no ON, such as Oracle's CREATE SESSION.
_PRIVILEGE_WORDS = frozenset({
    "CREATE", "ALTER", "DROP", "SELECT", "INSERT", "UPDATE", "DELETE", "EXECUTE",
    "ALL", "UNLIMITED", "USAGE", "READ", "WRITE",
})
_EXPLAINED = frozenset({"SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "MERGE", "CREATE", "REPLACE", "UPSERT"})
# Stands in for an f-string placeholder or a name in a `+` chain. The reader
# numbers each one between two private-use characters, which source text
# doesn't hold, so the card can show the code that fills it in and never
# rewrites a literal that looks like one.
_HOLE = "\ue000"
_HOLE_END = "\ue001"
_HOLE_MARK = re.compile(r"\ue000\d*\ue001")
_NUMBERED_HOLE = re.compile(r"\ue000(\d+)\ue001")


@dataclass(frozen=True)
class _Token:
    """One top-level token of a SQL statement.

    `kind` is "word" (a bare name or keyword, in upper case, `raw` as
    written), "name" (a quoted, backticked or bracketed name), "hole" (a part
    picked at run time), "text" (a quoted literal), "group" (a parenthesized
    part: `text` its first word, `words` every word and `texts` every quoted
    literal in it) or "punct" (any other character).
    """

    kind: Literal["word", "name", "hole", "text", "group", "punct"]
    text: str
    raw: str = ""
    line_start: bool = False
    words: tuple[str, ...] = ()
    texts: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Statement:
    """One `;`-separated statement: its top-level tokens and its span in the text."""

    tokens: tuple[_Token, ...]
    start: int
    end: int


def _statements(text: str, *, hash_comments: bool = False) -> Iterator[_Statement]:
    """Each statement in `text`, comments dropped and each parenthesized part
    folded into one "group" token. With `hash_comments`, a line that starts
    with `#` is a comment, as MySQL reads it."""
    tokens: list[_Token] = []
    inner: list[str] = []
    inner_texts: list[str] = []
    depth = 0
    group_line_start = False
    start = end = 0
    gap_from = 0

    def close_group() -> None:
        tokens.append(_Token(kind="group", text=inner[0] if inner else "", line_start=group_line_start,
                             words=tuple(inner), texts=tuple(inner_texts)))

    for m in (_HASH_COMMENT_TOKEN if hash_comments else _TOKEN).finditer(text):
        kind = m.lastgroup
        line_start = _LINE_BREAK.search(text, gap_from, m.start()) is not None
        gap_from = m.end()
        if kind == "comment":
            continue
        if kind == "end":
            if depth:
                close_group()
            if tokens:
                yield _Statement(tokens=tuple(tokens), start=start, end=end)
            tokens, depth = [], 0
            continue
        if not tokens and not depth:
            start = m.start()
        end = m.end()
        if kind == "open":
            if not depth:
                inner, inner_texts, group_line_start = [], [], line_start
            depth += 1
        elif kind == "close":
            if depth:
                depth -= 1
                if not depth:
                    close_group()
        elif depth:
            if kind == "word":
                inner.append(m.group().upper())
            elif kind == "text":
                inner_texts.append(m.group())
        elif kind == "word":
            word = m.group().upper()
            tokens.append(_Token(kind="hole" if _HOLE_MARK.fullmatch(word) else "word", text=word,
                                 raw=m.group(), line_start=line_start))
        else:
            tokens.append(_Token(kind=kind, text=m.group(), line_start=line_start))
    if depth:
        close_group()
    if tokens:
        yield _Statement(tokens=tuple(tokens), start=start, end=end)


def _word(tokens: Sequence[_Token], i: int) -> str:
    return tokens[i].text if i < len(tokens) and tokens[i].kind == "word" else ""


def _punct(tokens: Sequence[_Token], i: int, chars: str) -> bool:
    return i < len(tokens) and tokens[i].kind == "punct" and tokens[i].text in chars


def _name_end(tokens: Sequence[_Token], i: int) -> int | None:
    """The index after a name such as `t`, `s.t`, `"T"`, `[dbo].[t]` or a
    hole starting at `i`, or None when no name starts there."""
    if i >= len(tokens) or tokens[i].kind not in ("word", "name", "hole"):
        return None
    i += 1
    while _punct(tokens, i, ".") and i + 1 < len(tokens) and (
        tokens[i + 1].kind in ("word", "name", "hole") or _punct(tokens, i + 1, "*")
    ):
        i += 2
    return i


def _table_end(tokens: Sequence[_Token], i: int) -> int | None:
    """`_name_end` for the table or object a statement names, None for an
    English word standing there (`_NOT_NAMES`)."""
    j = _name_end(tokens, i)
    if j == i + 1 and tokens[i].kind == "word" and tokens[i].text in _NOT_NAMES:
        return None
    return j


def _ends_a_sentence(tokens: Sequence[_Token]) -> bool:
    """Whether the text ends as a sentence does, with a period after a word:
    no SQL dialect parses that."""
    return len(tokens) > 1 and _punct(tokens, len(tokens) - 1, ".") and (
        tokens[-2].kind in ("word", "name", "text", "group")
    )


def _items(tokens: Sequence[_Token]) -> list[Sequence[_Token]]:
    """`tokens` split at each top-level comma."""
    out: list[Sequence[_Token]] = []
    begin = 0
    for i, t in enumerate(tokens):
        if t.kind == "punct" and t.text == ",":
            out.append(tokens[begin:i])
            begin = i + 1
    out.append(tokens[begin:])
    return out


def _assignment_at(tokens: Sequence[_Token], i: int) -> bool:
    """Whether a SET clause's assignment (`col = ...`, `(a, b) = ...`, `@v = ...`
    or a hole) starts at `i`."""
    if i < len(tokens) and (tokens[i].kind in ("hole", "group") or _punct(tokens, i, "@")):
        return True
    j = _name_end(tokens, i)
    if j is None:
        return False
    # An array or map subscript, as in SET a[1] = 2.
    while j < len(tokens) and (tokens[j].kind == "name" and tokens[j].text.startswith("[") or _punct(tokens, j, "[")):
        if tokens[j].kind == "name":
            j += 1
        else:
            j = next((k for k in range(j, len(tokens)) if _punct(tokens, k, "]")), len(tokens)) + 1
    if _punct(tokens, j, "+-*/%&|^"):
        j += 1
    return _punct(tokens, j, "=")


def _update_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    i = 1
    while _word(tokens, i) in ("LOW_PRIORITY", "IGNORE", "ONLY", "TOP"):
        i += 2 if _word(tokens, i) == "TOP" else 1
    j = _name_end(tokens, i)
    if j is None:
        return False
    if j < len(tokens) and tokens[j].kind == "hole":
        return True  # the SET clause is picked at run time, as in f"UPDATE {t} {changes}"
    return any(_word(tokens, k) == "SET" and _assignment_at(tokens, k + 1) for k in range(j, len(tokens)))


def _delete_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    i = 1
    while _word(tokens, i) in ("LOW_PRIORITY", "QUICK", "IGNORE"):
        i += 1
    if _word(tokens, i) == "TOP":
        i += 3 if _word(tokens, i + 2) == "PERCENT" else 2
    if _word(tokens, i) == "FROM":
        j = _table_end(tokens, i + 1)
        if j is None:
            return False
        if j == len(tokens):
            return not (prose and _sentence_case(tokens[0]))
        if tokens[j].kind in ("hole", "group") or _punct(tokens, j, ","):
            return True
        if _word(tokens, j) in _DELETE_FOLLOWERS:
            return True
        # An alias, then the end or what may follow.
        return tokens[j].kind == "word" and (
            j + 1 == len(tokens) or _word(tokens, j + 1) in _DELETE_FOLLOWERS or tokens[j + 1].kind == "hole"
        )
    # DELETE t, DELETE t WHERE ..., or MySQL's DELETE a, b FROM a JOIN b ...
    # A sentence takes this shape too, as in "Delete orders where the
    # quantity is zero", and JavaScript's `delete params.page` the bare one.
    if prose and _sentence_case(tokens[0]):
        return False
    j = _table_end(tokens, i)
    while j is not None and _punct(tokens, j, ","):
        j = _table_end(tokens, j + 1)
    if j is None:
        return False
    if j == len(tokens):
        return not prose or tokens[0].raw.isupper()
    return _word(tokens, j) in ("FROM", "WHERE", "OUTPUT") or tokens[j].kind == "hole"


def _insert_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    """INSERT, and REPLACE and UPSERT, which take the same shape."""
    i = 3 if _word(tokens, 1) == "OR" else 1  # SQLite's INSERT OR REPLACE
    while _word(tokens, i) in _INSERT_MODIFIERS:
        i += 1
    if _word(tokens, i) in ("ALL", "FIRST"):
        # Oracle's multi-table insert: INSERT ALL INTO t1 VALUES (...) INTO t2 ...
        return any(_word(tokens, k) == "INTO" for k in range(i + 1, len(tokens)))
    into = _word(tokens, i) == "INTO"
    if into:
        i += 1
    if _word(tokens, i) == "TABLE":
        i += 1
    j = _table_end(tokens, i)
    # With no INTO, "a" is an article too in prose, as in "Insert a table of totals".
    if j is None or prose and not into and j == i + 1 and _word(tokens, i) == "A":
        return False
    while j < len(tokens):
        if _word(tokens, j) in ("AS", "PARTITION"):
            j += 2
        elif _word(tokens, j) == "BY" and _word(tokens, j + 1) in ("NAME", "POSITION"):
            j += 2
        elif _word(tokens, j) == "WITH" and j + 1 < len(tokens) and tokens[j + 1].kind == "group":
            j += 2  # a T-SQL table hint
        elif tokens[j].kind == "group" and tokens[j].text not in ("SELECT", "WITH", "VALUES"):
            j += 1  # the column list
        else:
            break
    return _rows_at(tokens, j, into=into)


def _rows_at(tokens: Sequence[_Token], j: int, *, into: bool) -> bool:
    """Whether the rows of an INSERT, REPLACE or UPSERT start at `j`, after
    its table: VALUES (...), a query, MySQL's SET col = ..., DEFAULT VALUES,
    Postgres's TABLE t, T-SQL's OUTPUT or EXEC, ClickHouse's FORMAT, or a
    placeholder. Prose such as "Replace missing values" or "Replace nulls
    with zeros" stops here."""
    if j >= len(tokens):
        return False
    if tokens[j].kind in ("hole", "group"):
        return True
    word = _word(tokens, j)
    if word in ("VALUES", "VALUE"):
        # ClickHouse's driver passes the rows apart from `INSERT INTO t VALUES`.
        return j + 1 < len(tokens) and tokens[j + 1].kind in ("group", "hole") or into and j + 1 == len(tokens)
    if word == "SET":
        return _assignment_at(tokens, j + 1)
    if word == "DEFAULT":
        return _word(tokens, j + 1) == "VALUES"
    if word == "TABLE":
        return _name_end(tokens, j + 1) == len(tokens)
    if word == "WITH":
        k = _name_end(tokens, j + 2 if _word(tokens, j + 1) == "RECURSIVE" else j + 1)
        if k is not None and k < len(tokens) and tokens[k].kind == "group":
            k += 1  # the CTE's column list
        return k is not None and _word(tokens, k) == "AS"
    if word == "OUTPUT":
        return any(_word(tokens, k) in ("VALUES", "SELECT", "DEFAULT", "EXEC", "EXECUTE")
                   for k in range(j + 1, len(tokens)))
    if word in ("EXEC", "EXECUTE"):
        return _name_end(tokens, j + 1) is not None
    if word == "FORMAT":
        return into
    return word in ("SELECT", "OVERRIDING")


def _merge_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    """MERGE [INTO] t USING s ON ... WHEN [NOT] MATCHED ..., or a placeholder
    for what follows USING or ON."""
    using = next((k for k in range(1, len(tokens)) if _word(tokens, k) == "USING"), None)
    if using is None:
        return False
    on = next((k for k in range(using + 1, len(tokens)) if _word(tokens, k) == "ON"), None)
    rest = range(using + 1 if on is None else on + 1, len(tokens))
    return any(tokens[k].kind == "hole" for k in rest) or on is not None and any(
        _word(tokens, k) == "WHEN" and _word(tokens, k + 1) in ("MATCHED", "NOT") for k in rest
    )


def _sentence_case(token: _Token) -> bool:
    """`Delete` or `Truncate` as a sentence or heading starts it. In prose, a
    bare `Truncate labels` or `Delete from backlog` reads as a label, while
    SQL writes TRUNCATE or truncate. A SQL runner takes either as SQL."""
    return token.raw[:1].isupper() and token.raw[1:].islower()


def _truncate_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    if prose and _word(tokens, 1) != "TABLE" and _sentence_case(tokens[0]):
        return False
    i = 2 if _word(tokens, 1) == "TABLE" else 1
    if _word(tokens, i) == "IF" and _word(tokens, i + 1) == "EXISTS":
        i += 2
    while True:
        if _word(tokens, i) == "ONLY":
            i += 1
        j = _name_end(tokens, i)
        if j is None:
            return False
        if _punct(tokens, j, "*"):
            j += 1
        if not _punct(tokens, j, ","):
            break
        i = j + 1
    return all(t.kind in ("group", "hole") or t.kind == "word" and t.text in _TRUNCATE_TAIL for t in tokens[j:])


def _object_name_end(tokens: Sequence[_Token], i: int) -> int | None:
    """The index after the name in CREATE, DROP or ALTER <object> <name>, `i`
    at the object word, past IF [NOT] EXISTS and CONCURRENTLY: at ON for
    Postgres's unnamed index, after a quoted user such as MySQL's 'bob'. None
    when no name follows."""
    index = tokens[i].text == "INDEX"
    i += 1
    if index and _word(tokens, i) == "CONCURRENTLY":
        i += 1
    if _word(tokens, i) == "IF" and _word(tokens, i + 1) in ("NOT", "EXISTS"):
        i += 3 if _word(tokens, i + 1) == "NOT" else 2
    if index and _word(tokens, i) == "ON":
        return i
    if i < len(tokens) and tokens[i].kind == "text":
        return i + 1
    if _punct(tokens, i, "$") and i + 1 < len(tokens) and tokens[i + 1].kind in ("word", "hole"):
        return i + 2  # a string.Template placeholder such as $table
    return _table_end(tokens, i)


def _option_at(tokens: Sequence[_Token], j: int) -> bool:
    """A column or parameter list, `option = value`, T-SQL's `@param` or a
    placeholder at `j`: what follows an object's name only in SQL."""
    return j < len(tokens) and (
        tokens[j].kind in ("group", "hole") or _punct(tokens, j, "@")
        or tokens[j].kind == "word" and _punct(tokens, j + 1, "=")
    )


def _create_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    """CREATE <object> <name> ... In prose, a sentence-case one such as
    "Create user accounts for ..." counts only with SQL's own shape after
    the name."""
    i = 1
    while _word(tokens, i) in _CREATE_MODIFIERS:
        i += 1
    if _word(tokens, i) not in _OBJECTS:
        return False
    j = _object_name_end(tokens, i)
    if j is None:
        return False
    if not (prose and _sentence_case(tokens[0])):
        return True
    return _option_at(tokens, j) or _word(tokens, j) in _CREATE_FOLLOWERS


def _drop_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    """DROP and ALTER <object> <name> ... In prose, a sentence-case one such
    as "Drop index funds below 2% weight" counts only with SQL's own shape
    after the name, or as DROP TABLE <name> alone, as TRUNCATE takes it."""
    i = 1
    while _word(tokens, i) in _DROP_MODIFIERS:
        i += 1
    if _word(tokens, i) not in _OBJECTS:
        return False
    j = _object_name_end(tokens, i)
    if j is None:
        return False
    if not (prose and _sentence_case(tokens[0])):
        return True
    if _word(tokens, 0) == "ALTER":
        return _option_at(tokens, j) or _word(tokens, j) in _ALTER_ACTIONS
    while j is not None and _punct(tokens, j, ","):
        j = _table_end(tokens, j + 1)  # DROP TABLE a, b
    if j is None:
        return False
    if j == len(tokens):
        return _word(tokens, i) == "TABLE"
    return _option_at(tokens, j) or _word(tokens, j) in _DROP_FOLLOWERS


def _role_item(item: Sequence[_Token]) -> bool:
    """`analyst`, `ROLE analyst`, or a privilege such as `CREATE SESSION`."""
    if len(item) == 1:
        return item[0].kind in ("word", "name", "hole")
    if not item or item[0].kind != "word":
        return False
    return item[0].text in _PRIVILEGE_WORDS or item[0].text in _GRANTEE_KINDS and _role_item(item[1:])


def _grantee_item(item: Sequence[_Token]) -> bool:
    """`bob`, `ROLE analyst` or MySQL's `'bob'@'%'`."""
    while len(item) > 1 and item[0].kind == "word" and item[0].text in _GRANTEE_KINDS:
        item = item[1:]
    named = ("word", "name", "hole", "text")
    if len(item) == 1:
        return item[0].kind in named
    return len(item) == 3 and _punct(item, 1, "@") and item[0].kind in named and item[2].kind in named


def _grant_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    """GRANT ... [ON ...] TO ..., and REVOKE ... [ON ...] FROM ... In prose,
    a sentence-case one such as "Grant access to Finance" counts only when
    it grants a privilege ON something."""
    to = "TO" if _word(tokens, 0) == "GRANT" else "FROM"
    k = next((i for i in range(1, len(tokens)) if _word(tokens, i) == to), None)
    if k is None:
        return False
    on = any(_word(tokens, i) == "ON" for i in range(1, k))
    if prose and _sentence_case(tokens[0]) and not (on and _word(tokens, 1) in _PRIVILEGE_WORDS):
        return False
    if not on and not all(_role_item(item) for item in _items(tokens[1:k])):
        return False
    stop = next((i for i in range(k + 1, len(tokens))
                 if _word(tokens, i) in ("WITH", "GRANTED", "CASCADE", "RESTRICT", "AS")), len(tokens))
    return all(_grantee_item(item) for item in _items(tokens[k + 1:stop]))


def _with_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    """WITH name AS (...), ... then the main statement: a write when a CTE
    changes data or the main statement does."""
    i = 2 if _word(tokens, 1) == "RECURSIVE" else 1
    while True:
        j = _name_end(tokens, i)
        if j is None:
            return False
        if j < len(tokens) and tokens[j].kind == "group":
            j += 1  # the CTE's column list
        if _word(tokens, j) != "AS":
            return False
        j += 1
        while _word(tokens, j) in ("NOT", "MATERIALIZED"):
            j += 1
        if j >= len(tokens) or tokens[j].kind != "group":
            return False
        if tokens[j].text in _DML:
            return True
        j += 1
        if not _punct(tokens, j, ","):
            break
        i = j + 1
    return _writes(tokens[j:], prose=prose)


def _select_into(tokens: Sequence[_Token], *, prose: bool) -> bool:
    into = next((i for i in range(len(tokens)) if _word(tokens, i) == "INTO"), None)
    if into is None:
        return False
    first_from = next((i for i in range(len(tokens)) if _word(tokens, i) == "FROM"), None)
    if first_from is not None and first_from < into:
        return False
    # A variable (@v, :v) or a T-SQL temp table (#t) holds nothing that lasts.
    if _punct(tokens, into + 1, "@#:") or _word(tokens, into + 1) in _TEMP:
        return False
    if into + 1 == len(tokens) and not (prose and _sentence_case(tokens[0])) or (
        _word(tokens, into + 1) in ("OUTFILE", "DUMPFILE")
    ):
        # The table comes after the text ends; MySQL writes a file on the
        # database's server.
        return True
    i = into + 1
    while _word(tokens, i) in ("UNLOGGED", "TABLE"):
        i += 1
    # The new table, then the query's FROM or the end, not prose such as
    # "Select suppliers into the watchlist for Q3".
    j = _table_end(tokens, i)
    return j is not None and (
        j == len(tokens) or _word(tokens, j) in ("FROM", "WHERE") or tokens[j].kind == "hole" or _punct(tokens, j, ",")
    )


def _copy_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    """COPY [INTO] t FROM a file, STDIN, a program, a stage, a query or a
    placeholder, not prose such as "Copy data from the March export"; into a
    Snowflake stage, from any table."""
    i = 2 if _word(tokens, 1) == "INTO" else 1
    stage = _punct(tokens, i, "@")
    if stage:
        i += 1
    j = _table_end(tokens, i)
    if j is None:
        return False
    if j < len(tokens) and tokens[j].kind == "group":
        j += 1
    if _word(tokens, j) != "FROM":
        return False
    k = j + 1
    return stage or k < len(tokens) and (
        tokens[k].kind in ("text", "hole", "group") or _word(tokens, k) in ("STDIN", "PROGRAM") or _punct(tokens, k, "@")
    )


def _explain_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    """EXPLAIN ANALYZE runs the statement it explains."""
    k = next((i for i in range(1, len(tokens)) if _word(tokens, i) in _EXPLAINED), None)
    if k is None:
        return False
    options = {w for t in tokens[1:k] for w in (t.words if t.kind == "group" else (t.text,))}
    return bool(options & {"ANALYZE", "ANALYSE"}) and _writes(tokens[k:], prose=prose)


def _load_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    if _word(tokens, 1) not in ("DATA", "XML"):
        return False
    i = 2
    while _word(tokens, i) in ("LOW_PRIORITY", "CONCURRENT", "LOCAL"):
        i += 1
    if _word(tokens, i) in ("INFILE", "INPATH"):
        return True
    # BigQuery: LOAD DATA INTO|OVERWRITE t ... FROM FILES (...)
    return _word(tokens, 2) in ("INTO", "OVERWRITE") and any(
        _word(tokens, k) == "FROM" and _word(tokens, k + 1) == "FILES" for k in range(3, len(tokens))
    )


def _rename_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    """MySQL's RENAME TABLE a TO b[, c TO d]."""
    if _word(tokens, 1) != "TABLE":
        return False
    i = 2
    while True:
        j = _name_end(tokens, i)
        if j is None or _word(tokens, j) != "TO":
            return False
        k = _name_end(tokens, j + 1)
        if k is None:
            return False
        if k == len(tokens):
            return True
        if not _punct(tokens, k, ","):
            return False
        i = k + 1


def _comment_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    """COMMENT ON <object> IS '...'."""
    if _word(tokens, 1) != "ON":
        return False
    k = next((i for i in range(2, len(tokens)) if _word(tokens, i) == "IS"), None)
    return k is not None and k + 1 < len(tokens) and (
        tokens[k + 1].kind in ("text", "hole") or _word(tokens, k + 1) == "NULL"
    )


def _refresh_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    return _word(tokens, 1) == "MATERIALIZED" and _word(tokens, 2) == "VIEW"


def _do_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    """A Postgres DO block: its body is a quoted literal the check can't read."""
    i = 3 if _word(tokens, 1) == "LANGUAGE" else 1
    return i < len(tokens) and tokens[i].kind == "text"


def _call_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    """CALL, EXEC or EXECUTE given a write as a quoted literal, as DuckDB's
    postgres_execute, T-SQL's sp_executesql and Snowflake's EXECUTE
    IMMEDIATE take one. A plain procedure call stays unread."""
    literals = [t.text for t in tokens if t.kind == "text"] + [s for t in tokens for s in t.texts]
    return any(write_statements(_unquoted(s), prose=prose) for s in literals)


def _unquoted(literal: str) -> str:
    if literal.startswith("'"):
        return literal[1:-1].replace("''", "'")
    tag = literal[: literal.index("$", 1) + 1]
    return literal[len(tag):-len(tag)]


class _Rule(Protocol):
    """A verb's rule: whether its statement changes data. With `prose`, the
    text may be English, so a sentence-case verb needs SQL's own shape after
    it (`write_statements`)."""

    def __call__(self, tokens: Sequence[_Token], *, prose: bool) -> bool: ...


_RULES: dict[str, _Rule] = {
    "UPDATE": _update_writes,
    "DELETE": _delete_writes,
    "INSERT": _insert_writes,
    "REPLACE": _insert_writes,
    "UPSERT": _insert_writes,
    "MERGE": _merge_writes,
    "TRUNCATE": _truncate_writes,
    "CREATE": _create_writes,
    "ALTER": _drop_writes,
    "DROP": _drop_writes,
    "GRANT": _grant_writes,
    "REVOKE": _grant_writes,
    "WITH": _with_writes,
    "SELECT": _select_into,
    "COPY": _copy_writes,
    "EXPLAIN": _explain_writes,
    "LOAD": _load_writes,
    "RENAME": _rename_writes,
    "COMMENT": _comment_writes,
    "REFRESH": _refresh_writes,
    "DO": _do_writes,
    "CALL": _call_writes,
    "EXEC": _call_writes,
    "EXECUTE": _call_writes,
}


def _writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    rule = _RULES.get(_word(tokens, 0))
    return rule is not None and not (prose and _ends_a_sentence(tokens)) and rule(tokens, prose=prose)


def _statement_writes(statement: _Statement, *, prose: bool) -> bool:
    """Whether a statement changes data. A line inside it that starts a write
    counts too: T-SQL runs statements that have no `;` between them. In
    prose, only a script's lines count: a report's, an email's or an HTML
    page's lines aren't a script, and a sentence-case line starts a sentence."""
    tokens = statement.tokens
    if _writes(tokens, prose=prose):
        return True
    if prose:
        head = tokens[0]
        # A `#` that doesn't start its line is a MySQL comment too.
        script = _punct(tokens, 0, "#") or head.kind == "word" and not _sentence_case(head) and (
            head.text in _RULES or head.text in _SCRIPT_WORDS
        )
        if not script:
            return False
    return any(
        t.line_start and t.kind == "word" and t.text in _RULES and not (prose and _sentence_case(t))
        and _writes(tokens[i:], prose=prose)
        for i, t in enumerate(tokens) if i
    )


def write_statements(text: str, *, prose: bool = True) -> tuple[str, ...]:
    """Each statement in `text` that changes data, as written. A line that
    starts with `#` is read as code, as Postgres reads it, and then as a
    comment, as MySQL reads it: a write in either reading counts.

    With `prose`, for text that may go somewhere other than a SQL runner, the
    text may be English: a sentence-case verb such as "Delete orders where
    the quantity is zero" needs SQL's own shape after it, and text that ends
    as a sentence does, with a period, is no statement. Without it, for text
    a SQL runner gets, a statement that parses as a write counts whatever
    its case or closing period."""
    found = tuple(text[s.start:s.end] for s in _statements(text) if _statement_writes(s, prose=prose))
    if found or not _HASH_LINE.search(text):
        return found
    return tuple(
        text[s.start:s.end] for s in _statements(text, hash_comments=True) if _statement_writes(s, prose=prose)
    )


def _quoted_statements(text: str) -> tuple[str, ...]:
    """Every statement in `text` as the card quotes it, writes or not: all
    the text from one top-level `;` to the next, comments and `#` lines
    included, so nothing between them is left off. One Approve runs them all."""
    out: list[str] = []
    begin = 0
    for m in _TOKEN.finditer(text):
        if m.lastgroup == "end":
            out.append(text[begin:m.start()].strip())
            begin = m.end()
    out.append(text[begin:].strip())
    return tuple(s for s in out if s)


# An ATTACH statement, not the English word: ATTACH or attach (in prose, a
# sentence's "Attach" is English), a space and its optional keywords, then
# what it attaches. That is a quoted literal or a placeholder followed by
# AS, options, `;` or the end; the start of a literal the code finishes when
# it runs; or the end of the literal.
_ATTACH_TAIL = (
    r"""\s+(?i:OR\s+REPLACE\s+)?(?i:DATABASE\s+)?(?i:IF\s+NOT\s+EXISTS\s+)?"""
    r"""(?:(?:(?P<quoted>'(?:[^']|'')*'|"(?:[^"]|"")*")|\{[^{}]*\}|%s|%\(\w+\)s|\$\w+|:\w+|\?)"""
    r"""(?=\s*(?:(?i:AS)\b|\(|;|\)|\Z))|['"][^'"]*\Z|\Z)"""
)
_ATTACH = re.compile(r"""(?:^|[;'"(])\s*(?:ATTACH|attach)""" + _ATTACH_TAIL, re.M)
# The same, in any case, for text a SQL runner gets.
_ATTACH_ANY_CASE = re.compile(r"""(?:^|[;'"(])\s*(?i:ATTACH)""" + _ATTACH_TAIL, re.M)
# An ATTACH's options: (TYPE postgres, ...) or TYPE postgres.
_ATTACH_TYPE = re.compile(r"""\bTYPE\b\s*=?\s*['"]?(\w+)""", re.I)
# A scheme such as `postgres:` or `md:` before what an ATTACH names, but not
# a Windows drive.
_SCHEME = re.compile(r"(?![A-Za-z]:[\\/])[A-Za-z][\w+.-]*:")


def _attaches_elsewhere(text: str, *, prose: bool = True) -> bool:
    """Whether `text` holds an ATTACH of a database outside the process. An
    in-memory database, a local file, or a SQLite one, named in full, is in
    it, as `duckdb.connect` on one is. With `prose`, as `write_statements`
    takes it, a sentence-case "Attach" is English."""
    for m in (_ATTACH if prose else _ATTACH_ANY_CASE).finditer(text):
        quoted = m.group("quoted")
        if quoted is None:
            return True  # what it attaches is picked when the code runs
        target = quoted[1:-1]
        options = _ATTACH_TYPE.search(text[m.end():].split(";", 1)[0])
        scheme = _SCHEME.match(target)
        local = target == ":memory:" or bool(target) and "=" not in target and (
            scheme is None or scheme.group().lower() in ("sqlite:", "file:")
        )
        if not local or options is not None and options.group(1).lower() not in ("sqlite", "duckdb"):
            return True
    return False


# ── The reader ────────────────────────────────────────────────────────────

_EXECUTE = frozenset({
    "execute", "executemany", "executescript", "exec_driver_sql", "sql", "query",
})
_READ_SQL = frozenset({"read_sql", "read_sql_query"})
# The keywords that pass an execute's or read_sql's statement, beside the first argument.
_STATEMENT_KEYWORDS = frozenset({"query", "sql", "statement", "operation", "sql_script"})
_HARMLESS_CALLS = frozenset({"print", "pprint", "display", "len", "input"})
_HARMLESS_METHODS = frozenset({
    "write", "write_text", "writelines", "debug", "info", "warning", "warn",
    "error", "exception", "critical", "log", "startswith", "endswith", "find",
    "count", "index", "split", "splitlines", "lower", "upper", "dump", "print", "pprint",
    "writerow", "writerows",
})
_LOGGER_NAMES = frozenset({"logger", "log", "logging", "_log", "_logger"})
# Calls that make a statement object of their text: what they wrap is SQL.
_SQL_OBJECTS = frozenset({"text", "sqlalchemy.text", "SQL", "sql.SQL", "psycopg2.sql.SQL", "psycopg.sql.SQL"})
_SQL_WRAPPERS = _SQL_OBJECTS | {"dedent", "textwrap.dedent", "str"}
# Calls whose result carries their argument on: a statement inside one goes
# wherever the result goes.
_PASS_THROUGH = frozenset({
    "repr", "list", "tuple", "set", "sorted", "reversed", "iter", "next", "enumerate", "json.dumps",
    "IPython.display.Markdown", "IPython.display.HTML", "IPython.display.Code",
    "zip", "dict", "textwrap.indent", "textwrap.fill", "textwrap.wrap", "textwrap.shorten",
    "html.escape", "pandas.DataFrame", "pandas.Series",
})
# Methods whose result carries what their receiver holds: a dict's items, or
# a DataFrame as text. to_sql isn't one: it writes a database.
_CARRIERS = frozenset({
    "items", "values", "tolist", "to_csv", "to_excel", "to_string", "to_markdown", "to_html", "to_json",
})
# Calls that build a new object, so no other name holds what they return.
_FRESH_CALLS = frozenset({
    "dict", "list", "set", "collections.defaultdict", "collections.OrderedDict", "copy.copy", "copy.deepcopy",
    "pandas.DataFrame", "pandas.Series", "pandas.concat", "pandas.merge", "pandas.read_csv", "pandas.read_excel",
    "pandas.read_parquet", "pandas.read_json", "pandas.read_sql", "pandas.read_sql_query",
})
# Calls that hand each item to the function they're given, and return what
# it returns (map, a DataFrame's apply) or the items it picks (filter).
_MAPPERS = frozenset({"map", "filter", "sorted", "min", "max"})
_MAPPER_METHODS = frozenset({"apply", "map", "transform", "agg", "aggregate", "applymap"})
_STR_METHODS = frozenset({
    "format", "strip", "lstrip", "rstrip", "replace", "upper", "lower", "decode", "encode",
})
# List and set methods that add an item: the item goes wherever the list goes.
_ADDERS = frozenset({"append", "extend", "insert", "add"})
_EXEC_CALLS = frozenset({"exec", "eval", "compile"})
_EXEC_LIMIT = 8
_FLOW_DEPTH = 12

_DELETE_FUNCS = frozenset({
    "os.remove", "os.unlink", "os.rmdir", "os.removedirs", "shutil.rmtree", "send2trash.send2trash",
    "pathlib.Path.unlink", "pathlib.Path.rmdir", "pathlib.PosixPath.unlink", "pathlib.PosixPath.rmdir",
})
# Deletes that can't remove a folder: the folders a listing names drop out.
_FILE_DELETES = frozenset({"os.remove", "os.unlink", "pathlib.Path.unlink", "pathlib.PosixPath.unlink"})
_MOVE_FUNCS = {
    "os.rename": "rename", "os.renames": "rename", "os.replace": "rename", "shutil.move": "move",
    "pathlib.Path.rename": "rename", "pathlib.Path.replace": "rename",
    "pathlib.PosixPath.rename": "rename", "pathlib.PosixPath.replace": "rename",
}
_COPY_FUNCS = frozenset({"shutil.copy", "shutil.copy2", "shutil.copyfile"})
# Modules and classes that hold a delete, a move or a shell call, for a
# lookup by a name the check computes (`_FileFinder.lookup`).
_LOOKUP_MODULES = frozenset({
    "os", "shutil", "pathlib", "pathlib.Path", "pathlib.PosixPath", "send2trash", "subprocess",
})
_SHELL_FUNCS = frozenset({
    "subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output",
    "subprocess.Popen", "subprocess.getoutput", "subprocess.getstatusoutput", "os.system", "os.popen",
    "asyncio.create_subprocess_shell",
})
# Calls that run a program from its words: where the words start, either
# spread over the arguments (the l forms) or as one list (the v forms).
_SPREAD_ARGV_FUNCS = {
    "os.execl": 1, "os.execle": 1, "os.execlp": 1, "os.execlpe": 1,
    "os.spawnl": 2, "os.spawnle": 2, "os.spawnlp": 2, "os.spawnlpe": 2,
    "asyncio.create_subprocess_exec": 0,
}
_LIST_ARGV_FUNCS = {
    "os.execv": 1, "os.execve": 1, "os.execvp": 1, "os.execvpe": 1,
    "os.spawnv": 2, "os.spawnve": 2, "os.spawnvp": 2, "os.spawnvpe": 2,
    "os.posix_spawn": 1, "os.posix_spawnp": 1,
}
_CHDIR_FUNCS = frozenset({"os.chdir", "contextlib.chdir"})
# Calls that end the process, so nothing after them runs.
_EXITS = frozenset({"sys.exit", "os._exit", "os.abort", "exit", "quit"})
_SHELL_DELETE = frozenset({"rm", "unlink", "rmdir", "shred"})
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})
# Words that run the command after them, and shell keywords a command follows.
_SHELL_PREFIXES = frozenset({
    "sudo", "doas", "env", "nohup", "time", "command", "exec", "nice", "xargs",
    "timeout", "stdbuf", "ionice", "busybox",
})
# Flags of one runner that take the next word as their value, beyond
# `_VALUE_FLAGS`, and the operands a runner takes before the command, as
# timeout takes its duration.
_PREFIX_VALUE_FLAGS = {
    "timeout": frozenset({"-k"}), "stdbuf": frozenset({"-i", "-o", "-e"}), "ionice": frozenset({"-c", "-p"}),
}
_PREFIX_OPERANDS = {"timeout": 1}
_SHELL_KEYWORDS = frozenset({"if", "then", "else", "elif", "do", "while", "until", "!", "{"})
# Shell words that end one command and start the next.
_SHELL_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "(", ")", "|&", ";;"})
# Folders the pad changes into that the check tracks; past this many, any.
_CWD_LIMIT = 16
# Calls into earlier cells' functions followed from one call in this cell.
_CALL_DEPTH = 4
_WRITER_METHODS = frozenset({
    "write_text", "write_bytes", "to_csv", "to_excel", "to_parquet", "to_json", "savefig",
})
_PATH_TYPES = frozenset({"pathlib.Path", "pathlib.PurePath", "pathlib.PosixPath", "pathlib.PurePosixPath"})
_PATH_MAKERS = frozenset({"glob", "rglob", "iterdir", "resolve", "absolute", "with_suffix",
                          "with_name", "with_stem", "joinpath", "expanduser"})
_ITER_WRAPPERS = frozenset({"next", "sorted", "list", "max", "min", "reversed", "iter", "tuple", "set", "frozenset"})

_SCOPES = (
    ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
)
# A binding under one of these may not run.
_BRANCHES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.TryStar, ast.Match)


class _Unknown:
    """A path the code picks when it runs, from a root the check can't place."""


_UNKNOWN = _Unknown()
_TEMP_PATH = "\x00temp"
# Stands in for a literal-command operand that isn't a literal.
_NOT_LITERAL = "\x00"
_PARAM_DEPTH = 5


@dataclass(frozen=True, eq=False)
class _Binding:
    """One place a cell gives a name a value.

    `how` says what the value is: "value" (the name holds it), "element" (an
    element of it), "context" (a with-statement's resource), "param" (a
    parameter of the function or lambda in `value`) or "unpacked" (a part of
    it the check can't place). `scope` is the def, lambda, class or
    comprehension the binding sits in, None at module level. `conditional`
    means it sits under an if, loop, try or match, so it may not run.
    """

    line: int
    value: ast.AST
    how: Literal["value", "element", "context", "param", "unpacked"]
    scope: ast.AST | None
    conditional: bool
    cell: _Cell


@dataclass(frozen=True, eq=False)
class _Growth:
    """One place a cell adds to what a name holds: `x.append(v)`, `x.add(v)`,
    `x.insert(i, v)`, `x.extend(vs)`, `x.update(vs)`, `x += vs`, `x |= vs`,
    `x[i] = v`, `x[:] = vs` and `x[0].append(v)` ("add"), or `x /= part`
    ("join")."""

    site: ast.AST
    value: ast.AST
    how: Literal["add", "join"]


@dataclass(frozen=True, eq=False)
class _Arg:
    """What a call passes for one parameter, and the cell the call is in."""

    node: ast.AST
    cell: _Cell


Function = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda


@dataclass(frozen=True, eq=False)
class _Callee:
    """A function or lambda an earlier cell defined, where the code read
    calls it or passes it on.

    `call` is the call, None when the code passes it on as a value; `use`
    is the code that names it, where the card reports what it changes. For a
    method, `bound` is what its first parameter gets: the instance or class
    it is called on, or the call that makes the instance. None for a plain
    function, a static method, or a method called on its class.
    """

    func: Function
    call: ast.Call | None
    use: ast.expr
    bound: ast.AST | None = None


class _Cell:
    """One parsed cell: its tree, parents, bindings and source lines, and
    what `_Reader` gathers from every cell. Nothing changes it once built, so
    one earlier cell serves every check that reads it (`_EarlierCells`)."""

    def __init__(self, tree: ast.Module, code: str, index: int, *, host_line: int | None = None) -> None:
        self.tree = tree
        self.index = index
        # For code a cell runs through exec() or eval(): the line of that call
        # in the cell, which findings report.
        self.host_line = host_line
        self.lines = code.splitlines()
        self.parents: dict[ast.AST, ast.AST] = {}
        # Every name the cell imports, by the name it binds; the functions and
        # classes it defines, by name; every call, in walk order; the calls
        # to a name or method that pass a name, by the name passed; whether a
        # string in it attaches a database outside the process
        # (`_attaches_elsewhere`); and the strings that do so only when a SQL
        # runner gets them, such as `Attach 'postgres:...' AS pg`, which
        # `_Reader` follows.
        self.imports: dict[str, str] = {}
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.calls: list[ast.Call] = []
        self.passed: dict[str, list[ast.Call]] = {}
        self.attaches = False
        self.run_attaches: list[ast.Constant] = []
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        self.imports[a.asname] = a.name
                    else:
                        top = a.name.split(".")[0]
                        self.imports[top] = top
            elif isinstance(node, ast.ImportFrom) and node.module:
                for a in node.names:
                    self.imports[a.asname or a.name] = f"{node.module}.{a.name}"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions[node.name] = node
            elif isinstance(node, ast.ClassDef):
                self.classes[node.name] = node
            elif isinstance(node, ast.Call):
                self.calls.append(node)
                if isinstance(node.func, (ast.Name, ast.Attribute)):
                    for arg in (*node.args, *(k.value for k in node.keywords)):
                        if isinstance(arg, ast.Name):
                            self.passed.setdefault(arg.id, []).append(node)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if _attaches_elsewhere(node.value):
                    self.attaches = True
                elif _attaches_elsewhere(node.value, prose=False):
                    self.run_attaches.append(node)
        self.bindings: dict[str, list[_Binding]] = {}
        self.loads: dict[str, list[ast.Name]] = {}
        # `name += ...` and `name.append(...)`: what builds a name's text or list up.
        self.augments: dict[str, list[ast.AugAssign]] = {}
        self.additions: dict[str, list[ast.Call]] = {}
        # Every place the cell adds to a name's list, set or path.
        self.grows: dict[str, list[_Growth]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                self.loads.setdefault(node.id, []).append(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    self._bind(t, node.value, at=node, how="value")
                    root = _subscript_root(node=t)
                    if isinstance(t, ast.Subscript) and root is not None:
                        # `x[i] = v` and `x[:] = vs` change the list `x` holds in place.
                        self.grows.setdefault(root, []).append(_Growth(site=node, value=node.value, how="add"))
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.op, ast.Add) and isinstance(node.target, ast.Name):
                    self.augments.setdefault(node.target.id, []).append(node)
                if isinstance(node.op, (ast.Add, ast.BitOr, ast.Div)) and isinstance(node.target, ast.Name):
                    how = "join" if isinstance(node.op, ast.Div) else "add"
                    self.grows.setdefault(node.target.id, []).append(_Growth(site=node, value=node.value, how=how))
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                self._bind(node.target, node.iter, at=node, how="element")
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                self._bind(node.optional_vars, node.context_expr, at=node, how="context")
            elif isinstance(node, ast.NamedExpr):
                self._bind(node.target, node.value, at=node, how="value")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                args = node.args
                for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                    self.bindings.setdefault(a.arg, []).append(_Binding(
                        line=node.lineno, value=node, how="param", scope=node, conditional=False, cell=self,
                    ))
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name)):
                if node.func.attr in _ADDERS:
                    self.additions.setdefault(node.func.value.id, []).append(node)
                if node.func.attr in (*_ADDERS, "update") and node.args:
                    self.grows.setdefault(node.func.value.id, []).append(
                        _Growth(site=node, value=node.args[-1], how="add"),
                    )
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Subscript) and node.func.attr in (*_ADDERS, "update")
                  and node.args and _subscript_root(node=node.func.value) is not None):
                # `x[0].append(v)`: what a list inside the one `x` holds grows.
                self.grows.setdefault(_subscript_root(node=node.func.value), []).append(
                    _Growth(site=node, value=node.args[-1], how="add"),
                )
        # The names each lambda the cell binds is bound to, by the lambda's id;
        # and the names bound to another name's value, as `b` in `b = a`, by `a`.
        self.lambda_names: dict[int, list[str]] = {}
        self.aliases: dict[str, list[str]] = {}
        for name, found in self.bindings.items():
            for b in found:
                if b.how == "value" and isinstance(b.value, ast.Lambda):
                    self.lambda_names.setdefault(id(b.value), []).append(name)
                if b.how == "value" and isinstance(b.value, ast.Name) and b.value.id != name:
                    self.aliases.setdefault(b.value.id, []).append(name)

    def _bind(self, target: ast.AST, value: ast.AST, *, at: ast.AST, how: Literal["value", "element", "context"]) -> None:
        if self._dead(at):
            return
        if isinstance(target, ast.Name):
            names, kind = [target.id], how
        elif isinstance(target, (ast.Tuple, ast.List)):
            # An unpacked tuple: each name gets a part the check can't place.
            names = [inner.id for elt in target.elts
                     for inner in [elt.value if isinstance(elt, ast.Starred) else elt]
                     if isinstance(inner, ast.Name)]
            kind = "unpacked"
        else:
            return
        scope = self.scope_of(at)
        conditional = self._under_branch(at, scope=scope)
        line = getattr(at, "lineno", 0) or getattr(value, "lineno", 0)
        for name in names:
            self.bindings.setdefault(name, []).append(_Binding(
                line=line, value=value, how=kind, scope=scope, conditional=conditional, cell=self,
            ))

    def builds_up(self, name: str, *, scope: ast.AST | None) -> bool:
        """Whether this cell adds to `name` in `scope` with `+=` or `.append`."""
        return any(
            self.scope_of(n) is scope
            for n in (*self.augments.get(name, ()), *self.additions.get(name, ()))
        )

    def _dead(self, node: ast.AST) -> bool:
        """Whether `node` sits in a branch that never runs, as under `if False:`."""
        child, parent = node, self.parents.get(node)
        while parent is not None:
            if isinstance(parent, (ast.If, ast.While)) and isinstance(parent.test, ast.Constant):
                if child in parent.body and not parent.test.value:
                    return True
                if isinstance(parent, ast.If) and child in parent.orelse and parent.test.value:
                    return True
            child, parent = parent, self.parents.get(parent)
        return False

    def _under_branch(self, node: ast.AST, *, scope: ast.AST | None) -> bool:
        current = self.parents.get(node)
        while current is not None and current is not scope:
            if isinstance(current, _BRANCHES):
                return True
            current = self.parents.get(current)
        return False

    def scope_of(self, node: ast.AST) -> ast.AST | None:
        """The def, lambda, class or comprehension `node` sits in; None at module level."""
        current = self.parents.get(node)
        while current is not None and not isinstance(current, _SCOPES):
            current = self.parents.get(current)
        return current

    def scopes_around(self, node: ast.AST) -> list[ast.AST]:
        """The scopes `node` sits in, innermost first. A class body is seen
        only from directly inside it, as Python sees it."""
        out: list[ast.AST] = []
        scope = self.scope_of(node)
        while scope is not None:
            if not (isinstance(scope, ast.ClassDef) and out):
                out.append(scope)
            scope = self.scope_of(scope)
        return out

    def source(self, node: ast.AST) -> str:
        """Every line of the call `node` is or sits in, so a call split
        across lines is quoted whole, with the indentation the lines share
        and each line's trailing space removed."""
        while not isinstance(node, ast.Call) and isinstance(self.parents.get(node), (ast.expr, ast.keyword)):
            node = self.parents[node]
        first = getattr(node, "lineno", 0)
        last = getattr(node, "end_lineno", None) or first
        if not 0 < first <= len(self.lines):
            return ""
        text = textwrap.dedent("\n".join(self.lines[first - 1:last]))
        return "\n".join(line.rstrip() for line in text.strip("\n").splitlines())


def _inside(node: ast.AST, ancestor: ast.AST, cell: _Cell) -> bool:
    current = cell.parents.get(node)
    while current is not None:
        if current is ancestor:
            return True
        current = cell.parents.get(current)
    return False


def _class_member(*, cls: ast.ClassDef, name: str) -> Function | None:
    """The function `cls`'s body gives `name`: a def, or a lambda assigned
    to it, bare or wrapped in staticmethod or classmethod."""
    for item in cls.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
            return item
        if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in item.targets):
            value = item.value
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id in ("staticmethod", "classmethod") and value.args):
                value = value.args[0]
            if isinstance(value, ast.Lambda):
                return value
    return None


def _method_kind(*, func: Function, cell: _Cell) -> Literal["static", "class", "plain"]:
    """How a method takes its first parameter: a static method takes none, a
    class method gets the class, a plain one the instance."""
    if isinstance(func, ast.Lambda):
        wrapper = cell.parents.get(func)
        names = {wrapper.func.id} if isinstance(wrapper, ast.Call) and isinstance(wrapper.func, ast.Name) else set()
    else:
        names = {d.id for d in func.decorator_list if isinstance(d, ast.Name)}
    return "static" if "staticmethod" in names else "class" if "classmethod" in names else "plain"


def _subscript_root(*, node: ast.AST) -> str | None:
    """The name `x` in `x[i]`, `x[i][j]` or `x[:]`; None for anything else."""
    while isinstance(node, ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _follows(*, node: ast.AST, value: ast.AST) -> bool:
    """Whether `node` starts after `value` ends, both in one cell."""
    start = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
    return start >= (getattr(value, "end_lineno", 0) or 0, getattr(value, "end_col_offset", 0) or 0)


def _reads_another(found: list[_Binding], given: set[int]) -> bool:
    """Whether a read that gets `found` reads only bindings other than the
    ones in `given`, by id."""
    return bool(found) and not any(id(b) in given for b in found)


def _since_last_sure(bindings: list[_Binding], *, name: str) -> list[_Binding]:
    """One cell's module-level `bindings` of `name`, in line order, from the
    last that surely ran and doesn't build on the value before it, as `sql =
    sql.rstrip(', ')` does: a read after them gets that value or one an if,
    loop or try after it gave."""
    ordered = sorted(bindings, key=lambda b: b.line)
    for i in range(len(ordered) - 1, -1, -1):
        if not ordered[i].conditional and not _loads_in([ordered[i].value], name):
            return ordered[i:]
    return ordered


def _ran_before(node: ast.AST, binding: _Binding, *, cell: _Cell) -> bool:
    """Whether `node`, in `cell`, surely ran before `binding` gave its name a
    value: in the binding's scope, and in an earlier cell, or earlier in the
    binding's cell (`_follows`) outside any loop around the binding. Code in
    a function runs when it is called, a function's body after its
    parameters get their values, and code a cell passes to exec() later than
    all, so none of those counts."""
    if cell.scope_of(node) is not binding.scope or binding.cell.host_line is not None:
        return False
    if cell is not binding.cell:
        return cell.index < binding.cell.index
    if binding.how == "param" or _follows(node=node, value=binding.value):
        return False
    current = cell.parents.get(binding.value)
    while current is not None and current is not binding.scope:
        if isinstance(current, (ast.For, ast.AsyncFor, ast.While)) and _inside(node, current, cell):
            return False
        current = cell.parents.get(current)
    return True


class _Reader:
    """Reads one cell against the conversation's earlier cells."""

    def __init__(self, cells: list[_Cell], *, current: _Cell) -> None:
        self.cells = cells
        self.current = current
        self._cells_by_node: dict[ast.AST, _Cell] = {}
        self.imports: dict[str, str] = {}
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        # The names each lambda is bound to, by the lambda's id.
        self.lambda_names: dict[int, list[str]] = {}
        for cell in cells:
            self._cells_by_node[cell.tree] = cell
            self._cells_by_node.update(dict.fromkeys(cell.parents, cell))
            self.imports.update(cell.imports)
            self.functions.update(cell.functions)
            self.classes.update(cell.classes)
            self.lambda_names.update(cell.lambda_names)
        self.function_names = {*self.functions, *(n for names in self.lambda_names.values() for n in names)}
        # The names any cell binds, and the members the pad's classes define:
        # a call through any other name or attribute reaches no pad function.
        self.bound_names = {name for cell in cells for name in cell.bindings}
        self.method_names: set[str] = set()
        for cls in self.classes.values():
            for item in cls.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.method_names.add(item.name)
                elif isinstance(item, ast.Assign):
                    self.method_names.update(t.id for t in item.targets if isinstance(t, ast.Name))
        self.attaches = any(cell.attaches for cell in cells)
        self._held: dict[tuple[int, str, bool, tuple[tuple[int, str, int], ...]], tuple[str, ...]] = {}
        # What `_name_statements` found at each read of a name, by the read's
        # id, `prose`, the function read and `call_key`: a value appended to
        # a list asks it of every read of the list.
        self.named: dict[tuple[int, bool, int, tuple[tuple[int, str, int], ...]], tuple[str, ...]] = {}
        # The code behind each numbered placeholder, None for a part with no
        # code to show, and each node's number by its id.
        self._holes: list[ast.AST | None] = []
        self._hole_numbers: dict[int, int] = {}
        # Pad functions this cell calls: their bodies run now.
        self.called = {
            n.func.id for cell in cells[current.index:] for n in cell.calls
            if isinstance(n.func, ast.Name) and n.func.id in self.functions
        }
        # While the check reads an earlier cell's function for one call: what
        # that call passes for each parameter, by (function id, name), None
        # for one it can't read; and the call its return value goes to.
        self.param_args: dict[tuple[int, str], _Arg | None] = {}
        self.returns_to: dict[int, ast.AST] = {}
        # What the pad's cells write (`_made`), for a snapshot past its cap.
        self.made: _Made | None = None
        # What this check read from disk for a glob, once per pattern and folder.
        self.disk: _DiskReads = {}
        # The folders the pad's code may run in, relative to the project or
        # absolute, the project itself first; unknown once a cell changes into
        # one the check can't place.
        self.cwds: list[str] | _Unknown = ["."]
        # A call `dotted` reads as a chdir ends in `.chdir` or uses a name
        # bound to one: only those are read through it.
        chdir_names = {name for name, full in self.imports.items() if full in _CHDIR_FUNCS}
        for cell in cells:
            for node in cell.calls:
                func = node.func
                named = func.attr == "chdir" if isinstance(func, ast.Attribute) else (
                    isinstance(func, ast.Name) and func.id in chdir_names
                )
                if named and self.dotted(func) in _CHDIR_FUNCS:
                    self.cwds = self.changed_into(self.cwds, _target_arg(node, keywords=("path",)), cell=cell)
        # A sentence-case `Attach '...' AS pg` attaches where a SQL runner,
        # on any handle, gets it. Read last: the read needs the rest.
        if not self.attaches:
            self.attaches = any(
                _reaches_a_runner(node, self, any_handle=True) is _Reach.RUNNER
                for cell in cells for node in cell.run_attaches
            )

    def cell_of(self, node: ast.AST) -> _Cell:
        return self._cells_by_node.get(node, self.current)

    def hole(self, node: ast.AST | None) -> str:
        """The placeholder for a part the code picks at run time: `node`'s
        own number, so `show_holes` can put its code back."""
        if node is not None and id(node) in self._hole_numbers:
            return f"{_HOLE}{self._hole_numbers[id(node)]}{_HOLE_END}"
        self._holes.append(node)
        if node is not None:
            self._hole_numbers[id(node)] = len(self._holes) - 1
        return f"{_HOLE}{len(self._holes) - 1}{_HOLE_END}"

    def show_holes(self, text: str) -> str:
        """`text` with each numbered placeholder replaced by the code that
        fills it in, in braces: `{r.new}`, or `{...}` when there is none."""

        def code(m: re.Match[str]) -> str:
            if int(m.group(1)) >= len(self._holes):
                return m.group()
            node = self._holes[int(m.group(1))]
            try:
                if isinstance(node, ast.FormattedValue):
                    return ast.unparse(node)
                return "{...}" if node is None else "{" + ast.unparse(node) + "}"
            except Exception:
                return "{...}"

        return _NUMBERED_HOLE.sub(code, text)

    def statements_held(self, name: str, binding: _Binding, *, prose: bool) -> tuple[str, ...]:
        """Every statement of each text `binding` gives `name` that holds a
        write, read as `write_statements` reads it with `prose`, once per
        check and set of call arguments."""
        key = (id(binding), name, prose, self.call_key())
        if key not in self._held:
            texts = _held_texts(name, binding, self, depth=0)
            self._held[key] = tuple(
                s for t in texts if write_statements(t, prose=prose) for s in _quoted_statements(t)
            )
        return self._held[key]

    def call_key(self) -> tuple[tuple[int, str, int], ...]:
        """What the call being read passes for each parameter, as a cache key."""
        return tuple(sorted((k[0], k[1], id(a.node) if a else 0) for k, a in self.param_args.items()))

    def changed_into(
        self, cwds: list[str] | _Unknown, target: ast.AST | None, *, cell: _Cell,
    ) -> list[str] | _Unknown:
        """`cwds` with the folder a chdir, `cwd=` or `cd` to `target` reaches
        from each of them added: a folder already there stays, since the
        change may not run."""
        if isinstance(cwds, _Unknown):
            return cwds
        got = self.paths(target, cell=cell) if target is not None else _UNKNOWN
        if isinstance(got, _Unknown):
            return _UNKNOWN
        out = list(dict.fromkeys([*cwds, *(posixpath.normpath(_join(c, p)) for c in cwds for p in got)]))
        return out if len(out) <= _CWD_LIMIT else _UNKNOWN

    # ── calls into earlier cells' functions ──────────────────────────────

    def earlier_function(self, node: ast.Name) -> Function | None:
        """The function or lambda an earlier cell bound to `node`'s name, when
        that is what `node` reads."""
        found = self.bindings_of(node.id, node=node)
        lambdas = [b.value for b in found if b.how == "value" and isinstance(b.value, ast.Lambda)]
        if any(b.scope is not None for b in found) and not lambdas:
            # A local name, such as a parameter, shadows the pad's function.
            return None
        func: Function | None = lambdas[0] if lambdas else self.functions.get(node.id)
        if func is None or self.cell_of(func).index >= self.current.index:
            return None
        return func

    @contextmanager
    def calling(self, func: Function, call: ast.Call | None, *, bound: ast.AST | None = None) -> Iterator[None]:
        """Read `func`'s parameters as the arguments `call` passes, after
        `bound` for a method's first, and its return value as going to `call`.
        With no call (the function passed on as a value), every other
        parameter is one the check can't read."""
        passed = _call_arguments(func, call, call_cell=self.cell_of(call) if call else self.current,
                                 func_cell=self.cell_of(func), bound=bound)
        saved = (dict(self.param_args), dict(self.returns_to))
        self.param_args.update({(id(func), name): arg for name, arg in passed.items()})
        if call is not None:
            self.returns_to[id(func)] = call
        try:
            yield
        finally:
            self.param_args, self.returns_to = saved

    def argument_of(self, node: ast.Name) -> _Arg | None:
        """What the call being read passes for the parameter `node` reads."""
        if not self.param_args:
            return None
        for b in self.bindings_of(node.id, node=node):
            if b.how == "param" and (id(b.value), node.id) in self.param_args:
                return self.param_args[(id(b.value), node.id)]
        return None

    def literal_argument(self, node: ast.Name) -> str | None:
        """The text of a string or number literal the call being read passes
        for the parameter `node` reads."""
        arg = self.argument_of(node)
        value = arg.node.value if arg is not None and isinstance(arg.node, ast.Constant) else None
        if isinstance(value, str) or isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return None

    def callees(self, *, node: ast.expr, call: ast.Call | None, use: ast.expr, depth: int = 0) -> list[_Callee]:
        """The functions and lambdas an earlier cell defined that calling
        `node` with `call` runs: one a name, an alias, a list or dict, a
        `functools.partial` or a function's return value holds, and a method
        of an earlier cell's class, called on the class, an instance, `self`
        or the call that makes one (`__init__`, `__call__`). A method called
        on something the check can't place counts in every earlier class
        that defines it."""
        if depth > 6:
            return []
        if isinstance(node, ast.Lambda):
            return [_Callee(func=node, call=call, use=use)] if self.cell_of(node).index < self.current.index else []
        if isinstance(node, ast.Name):
            if node.id in self.function_names:
                func = self.earlier_function(node)
                if func is not None:
                    return [_Callee(func=func, call=call, use=use)]
            cls = self._earlier_class(node=node)
            if cls is not None:
                # Calling a class runs its __init__ on the new instance.
                return self._methods(cls=cls, name="__init__", bound=call or node, call=call, use=use)
            if node.id not in self.bound_names:
                return []
            return [c for b in self.bindings_of(node.id, node=node) if b.how == "value"
                    for c in self.callees(node=b.value, call=call, use=use, depth=depth + 1)]
        if isinstance(node, ast.Subscript):
            items = self._items_of(node=node.value, key=node.slice, depth=depth + 1) or []
            return [c for item in items for c in self.callees(node=item, call=call, use=use, depth=depth + 1)]
        if isinstance(node, ast.Attribute) and node.attr in self.method_names:
            receiver = node.value
            if isinstance(receiver, ast.Name) and receiver.id in self.imports:
                return []
            cls = self._earlier_class(node=receiver)
            if cls is not None:
                return self._methods(cls=cls, name=node.attr, bound=receiver, on_class=True, call=call, use=use)
            classes = self._instance_classes(node=receiver, depth=depth + 1)
            if classes is None:
                classes = list(self.classes.values())
            return [c for k in classes for c in self._methods(cls=k, name=node.attr, bound=receiver, call=call, use=use)]
        if isinstance(node, ast.Call):
            if self.dotted(node.func) == "functools.partial" and node.args:
                made = None if call is None else _partial_call(partial=node, call=call)
                return self.callees(node=node.args[0], call=made, use=use, depth=depth + 1)
            cls = self._earlier_class(node=node.func)
            if cls is not None:
                # An instance called: its __call__.
                return self._methods(cls=cls, name="__call__", bound=node, call=call, use=use)
            # What a function returns, called.
            return [c for made in self.callees(node=node.func, call=node, use=use, depth=depth + 1)
                    for ret in self._returns(func=made.func)
                    for c in self.callees(node=ret, call=call, use=use, depth=depth + 1)]
        return []

    def _earlier_class(self, *, node: ast.AST) -> ast.ClassDef | None:
        """The class an earlier cell defined that the name `node` reads."""
        if not isinstance(node, ast.Name):
            return None
        cls = self.classes.get(node.id)
        if cls is None or self.cell_of(cls).index >= self.current.index:
            return None
        cell = self.cell_of(node)
        scopes = cell.scopes_around(node)
        if any(b.scope in scopes for b in cell.bindings.get(node.id, ())):
            # A local name, such as a parameter, shadows the pad's class.
            return None
        return cls

    def _methods(
        self, *, cls: ast.ClassDef, name: str, bound: ast.AST, call: ast.Call | None, use: ast.expr,
        on_class: bool = False, depth: int = 0,
    ) -> list[_Callee]:
        """`cls`'s method `name`, or the one a base class an earlier cell
        defined gives it. Its first parameter gets `bound`, except in a static
        method, or in a plain one called on the class (`on_class`)."""
        func = _class_member(cls=cls, name=name)
        if func is not None:
            if self.cell_of(func).index >= self.current.index:
                return []
            kind = _method_kind(func=func, cell=self.cell_of(func))
            gets_bound = kind == "class" or kind == "plain" and not on_class
            return [_Callee(func=func, call=call, use=use, bound=bound if gets_bound else None)]
        for base in cls.bases:
            parent = self._earlier_class(node=base)
            if parent is not None and parent is not cls and depth < 6:
                found = self._methods(cls=parent, name=name, bound=bound, call=call, use=use, on_class=on_class, depth=depth + 1)
                if found:
                    return found
        return []

    def _returns(self, *, func: Function) -> list[ast.expr]:
        """What `func` itself returns: a lambda's body, or the value of each
        return statement in the function, not in a function inside it."""
        if isinstance(func, ast.Lambda):
            return [func.body]
        cell = self.cell_of(func)
        return [n.value for n in ast.walk(func)
                if isinstance(n, ast.Return) and n.value is not None and cell.scope_of(n) is func]

    def _items_of(self, *, node: ast.AST, key: ast.AST | None, depth: int) -> list[ast.AST] | None:
        """The elements of the list, tuple, set or dict values `node` holds,
        with what the pad adds to it, only the one a literal `key` names when
        the literal says which; None when the check can't place them."""
        if depth > 6:
            return None
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            elements = [e.value if isinstance(e, ast.Starred) else e for e in node.elts]
            if (isinstance(key, ast.Constant) and type(key.value) is int and not isinstance(node, ast.Set)
                    and -len(elements) <= key.value < len(elements)):
                return [elements[key.value]]
            return elements
        if isinstance(node, ast.Dict):
            if isinstance(key, ast.Constant) and all(isinstance(k, ast.Constant) for k in node.keys):
                hits = [v for k, v in zip(node.keys, node.values) if k.value == key.value]
                if hits:
                    return hits
            return list(node.values)
        if isinstance(node, ast.Name):
            out: list[ast.AST] = []
            for b in self.bindings_of(node.id, node=node):
                got = self._items_of(node=b.value, key=key, depth=depth + 1) if b.how == "value" else None
                grown = self.list_growth(name=node.id, binding=b) if got is not None else _UNKNOWN
                if got is None or isinstance(grown, _Unknown):
                    return None
                out += got + [g.value for g, _ in grown]
            return out
        if isinstance(node, ast.Call) and self.dotted(node.func) in ("dict", "list", "tuple"):
            # A dict of keywords, or a list or tuple made from another.
            hits = [k.value for k in node.keywords if isinstance(key, ast.Constant) and k.arg == key.value]
            if hits:
                return hits
            out = [k.value for k in node.keywords]
            for arg in node.args:
                got = self._items_of(node=arg, key=None, depth=depth + 1)
                if got is None:
                    return None
                out += got
            return out
        if isinstance(node, ast.Call):
            made = self.callees(node=node.func, call=node, use=node, depth=depth + 1)
            if not made:
                return None
            out = []
            for callee in made:
                for ret in self._returns(func=callee.func):
                    got = self._items_of(node=ret, key=key, depth=depth + 1)
                    if got is None:
                        return None
                    out += got
            return out
        return None

    def _instance_classes(self, *, node: ast.AST, depth: int) -> list[ast.ClassDef] | None:
        """The earlier cells' classes `node` holds an instance of: one made by
        calling the class, held in a name, a list or dict, or `self` in a
        method. None when the check can't place what `node` holds."""
        if depth > 6:
            return None
        if isinstance(node, ast.Call):
            cls = self._earlier_class(node=node.func)
            if cls is not None:
                return [cls]
            if isinstance(node.func, ast.Name) and node.func.id == "super":
                return None
            # A pad function's return value may be an instance; any other call's isn't.
            return None if self.callees(node=node.func, call=node, use=node, depth=depth + 1) else []
        if isinstance(node, ast.Name):
            if node.id in self.imports:
                return []
            arg = self.argument_of(node)
            if arg is not None:
                return self._instance_classes(node=arg.node, depth=depth + 1)
            found = self.bindings_of(node.id, node=node)
            if not found:
                return None
            out: list[ast.ClassDef] = []
            for b in found:
                if b.how == "value":
                    got = self._instance_classes(node=b.value, depth=depth + 1)
                elif b.how == "element":
                    got = self._element_classes(items=self._items_of(node=b.value, key=None, depth=depth + 1), depth=depth)
                elif b.how == "param":
                    got = self._self_class(func=b.value, param=node.id)
                else:
                    got = None
                if got is None:
                    return None
                out += got
            return out
        if isinstance(node, ast.Subscript):
            return self._element_classes(items=self._items_of(node=node.value, key=node.slice, depth=depth + 1), depth=depth)
        if isinstance(node, (ast.Constant, ast.JoinedStr, ast.List, ast.Tuple, ast.Set, ast.Dict,
                             ast.ListComp, ast.SetComp, ast.DictComp, ast.BinOp, ast.Compare)):
            return []
        return None

    def _element_classes(self, *, items: list[ast.AST] | None, depth: int) -> list[ast.ClassDef] | None:
        if items is None:
            return None
        out: list[ast.ClassDef] = []
        for item in items:
            got = self._instance_classes(node=item, depth=depth + 1)
            if got is None:
                return None
            out += got
        return out

    def _self_class(self, *, func: ast.AST, param: str) -> list[ast.ClassDef] | None:
        """`[the class]` when `param` is the first parameter of a method,
        `self`; None for any other parameter, which the check can't place."""
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        cls = self.cell_of(func).parents.get(func)
        first = [*func.args.posonlyargs, *func.args.args][:1]
        if isinstance(cls, ast.ClassDef) and first and first[0].arg == param:
            return [cls]
        return None

    def partial_of(self, *, node: ast.expr) -> ast.Call | None:
        """The `functools.partial(f, ...)` call `node` is, or a name holds."""
        if isinstance(node, ast.Name):
            if node.id not in self.bound_names:
                return None
            values = [b.value for b in self.bindings_of(node.id, node=node) if b.how == "value"]
        else:
            values = [node]
        return next((v for v in values if isinstance(v, ast.Call) and v.args
                     and self.dotted(v.func) == "functools.partial"), None)

    # ── names ────────────────────────────────────────────────────────────

    def dotted(self, node: ast.AST) -> str:
        """`os.path.join` for a call through an alias such as `osp.join`."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return ""
        head = self.imports.get(node.id, node.id)
        return ".".join([head, *reversed(parts)])

    def bindings_of(self, name: str, *, node: ast.AST) -> list[_Binding]:
        """What `name` holds where `node` reads it.

        Inside a function, the function's own bindings first. At module level,
        this cell's bindings before the read, else the latest earlier cell's,
        each from the last one that surely ran. A function body runs when it
        is called, so it sees every cell's module bindings, the latest first.
        A cell that binds the name only under an if, loop or try doesn't hide
        an older binding: both count.
        """
        cell = self.cell_of(node)
        scopes = cell.scopes_around(node)
        for scope in scopes:
            local = [b for b in cell.bindings.get(name, ()) if b.scope is scope]
            if local:
                return local
        if scopes:
            layers = (self._module_bindings(c, name) for c in reversed(self.cells))
        else:
            line = getattr(node, "lineno", 0)
            here = [b for b in self._module_bindings(cell, name) if b.line <= line]
            layers = iter([_since_last_sure(here, name=name), *(
                _since_last_sure(self._module_bindings(c, name), name=name) for c in reversed(self.cells[: cell.index])
            )])
        found: list[_Binding] = []
        for layer in layers:
            found += layer
            if any(not b.conditional for b in layer):
                break
        return found

    @staticmethod
    def _module_bindings(cell: _Cell, name: str) -> list[_Binding]:
        return [b for b in cell.bindings.get(name, ()) if b.scope is None]

    def call_sites(self, func: Function, param: str) -> list[_Arg | None]:
        """What each call of `func` in the pad's cells passes as `param`, its
        default when the call passes nothing; None where the check can't read it."""
        names = [func.name] if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) else (
            self.lambda_names.get(id(func), [])
        )
        home = self.cell_of(func)
        out: list[_Arg | None] = []
        for cell in self.cells:
            for node in cell.calls:
                if isinstance(node.func, ast.Name) and node.func.id in names:
                    out.append(_call_arguments(func, node, call_cell=cell, func_cell=home).get(param))
        return out

    # ── paths ────────────────────────────────────────────────────────────

    def paths(self, node: ast.AST, *, cell: _Cell, depth: int = 0) -> list[str] | _Unknown:
        """The glob-style path patterns `node` can hold, `*` for a part picked at
        run time, or unknown when the check can't place its root."""
        if depth > 8:
            return _UNKNOWN
        if isinstance(node, ast.Constant):
            return [node.value] if isinstance(node.value, str) else _UNKNOWN
        if isinstance(node, ast.JoinedStr):
            parts = [""]
            for v in node.values:
                if isinstance(v, ast.Constant):
                    parts = [p + str(v.value) for p in parts]
                    continue
                inner = self.paths(v.value, cell=cell, depth=depth + 1)
                if isinstance(inner, _Unknown) or _TEMP_PATH in inner:
                    if parts == [""]:
                        return inner
                    inner = ["*"]
                parts = [p + i for p in parts for i in inner]
            return parts
        if isinstance(node, ast.BinOp):
            left = self.paths(node.left, cell=cell, depth=depth + 1)
            if isinstance(left, _Unknown) or _TEMP_PATH in left:
                return left
            right = self.paths(node.right, cell=cell, depth=depth + 1)
            if isinstance(node.op, ast.Add) and (
                not left or self.is_sequence(node.left, cell=cell) or self.is_sequence(node.right, cell=cell)
            ):
                # Two lists added: each element of either.
                return right if isinstance(right, _Unknown) else left + right
            if isinstance(right, _Unknown):
                right = ["*"]
            if isinstance(node.op, ast.Div):
                return [_join(a, b) for a in left for b in right]
            if isinstance(node.op, ast.Add):
                return [a + b for a in left for b in right]
            return _UNKNOWN
        if isinstance(node, ast.Name):
            return self._name_paths(node, cell=cell, depth=depth)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            # A loop over these names each element.
            out: list[str] = []
            for elt in node.elts:
                got = self.paths(elt, cell=cell, depth=depth + 1)
                if isinstance(got, _Unknown):
                    return got
                out += got
            return out
        if isinstance(node, ast.Attribute):
            base = self.paths(node.value, cell=cell, depth=depth + 1)
            if not isinstance(base, _Unknown) and _TEMP_PATH in base:
                # `tempfile.NamedTemporaryFile(...).name`
                return [_TEMP_PATH]
            if node.attr == "path":
                return base
            if node.attr == "parent" and not isinstance(base, _Unknown):
                return [posixpath.dirname(p) or "." for p in base]
            return _UNKNOWN
        if isinstance(node, ast.Subscript):
            if self._is_workspace_env(node):
                return ["."]
            return self.paths(node.value, cell=cell, depth=depth + 1)
        if isinstance(node, ast.Call):
            return self._call_paths(node, cell=cell, depth=depth)
        return _UNKNOWN

    def _name_paths(self, node: ast.Name, *, cell: _Cell, depth: int) -> list[str] | _Unknown:
        found = self.bindings_of(node.id, node=node)
        if not found:
            return _UNKNOWN
        out: list[str] = []
        for b in found:
            if b.how == "param":
                got = self._param_paths(node.id, b, depth=depth)
            elif b.how == "unpacked":
                got = self._unpacked_paths(b.value, cell=b.cell, depth=depth)
            else:
                got = self.paths(b.value, cell=b.cell, depth=depth + 1)
                if b.how == "value" and not isinstance(got, _Unknown):
                    got = self._grown_paths(node.id, b, got, use=node, depth=depth)
            if isinstance(got, _Unknown):
                return got
            out += got
        return out

    def _param_paths(self, name: str, binding: _Binding, *, depth: int) -> list[str] | _Unknown:
        """A parameter: what the call being read passes for it, else what
        every call site passes. A parameter met at a call site is unknown."""
        key = (id(binding.value), name)
        if key in self.param_args:
            arg = self.param_args[key]
            return _UNKNOWN if arg is None else self.paths(arg.node, cell=arg.cell, depth=depth + 1)
        sites = self.call_sites(binding.value, name) if depth < _PARAM_DEPTH else []
        if not sites:
            return _UNKNOWN
        out: list[str] = []
        for site in sites:
            got = _UNKNOWN if site is None else self.paths(site.node, cell=site.cell, depth=_PARAM_DEPTH)
            if isinstance(got, _Unknown):
                return got
            out += got
        return out

    def growth(self, name: str, binding: _Binding) -> list[tuple[_Growth, _Cell]]:
        """Each place the pad adds to what `binding` gives `name` that can run
        after it binds it: in any cell for a module-level name, in its own
        function for a local one. An addition that surely ran before the
        binding (`_ran_before`) grew the value the name held then, so it
        doesn't count."""
        out = []
        for cell in self.cells if binding.scope is None else [binding.cell]:
            for grown in cell.grows.get(name, ()):
                if self._binding_scope(cell, grown.site, name) is binding.scope and not _ran_before(
                    grown.site, binding, cell=cell,
                ):
                    out.append((grown, cell))
        return out

    def list_growth(self, *, name: str, binding: _Binding, depth: int = 0) -> list[tuple[_Growth, _Cell]] | _Unknown:
        """`growth` for a list or set, which the pad changes in place: also
        what it adds through another name bound to it, as `b` in `b = a`, and
        what a pad function adds to the parameter a call passes it as.
        Unknown when such a function passes the parameter on."""
        out = self.growth(name, binding)
        if depth > 3:
            return out
        for cell in self.cells if binding.scope is None else [binding.cell]:
            for alias in cell.aliases.get(name, ()):
                for b in cell.bindings.get(alias, ()):
                    if (b.how == "value" and isinstance(b.value, ast.Name) and b.value.id == name
                            and self._binding_scope(cell, b.value, name) is binding.scope
                            and not _ran_before(b.value, binding, cell=cell)):
                        more = self.list_growth(name=alias, binding=b, depth=depth + 1)
                        if isinstance(more, _Unknown):
                            return more
                        out += more
            for call in cell.passed.get(name, ()):
                if self._binding_scope(cell, call, name) is not binding.scope or _ran_before(call, binding, cell=cell):
                    continue
                method = isinstance(call.func, ast.Attribute)
                for func in self._pad_callables(node=call.func):
                    # A method gets the list as the parameter after its first or,
                    # called on the class, as the one it is passed for.
                    for bound in (None, call.func.value) if method else (None,):
                        passed = _call_arguments(func, call, call_cell=cell, func_cell=self.cell_of(func), bound=bound)
                        for param, arg in passed.items():
                            if arg is not None and isinstance(arg.node, ast.Name) and arg.node.id == name:
                                filled = self._filled(func=func, param=param)
                                if isinstance(filled, _Unknown):
                                    return filled
                                out += filled
        return out

    def _pad_callables(self, *, node: ast.expr) -> list[Function]:
        """The pad's functions and lambdas a call to the name `node` runs, or
        to the method `node` names, in every pad class that defines it."""
        if isinstance(node, ast.Attribute) and node.attr in self.method_names:
            members = (_class_member(cls=cls, name=node.attr) for cls in self.classes.values())
            return [m for m in members if m is not None]
        if not isinstance(node, ast.Name) or node.id not in self.function_names:
            return []
        lambdas = [b.value for b in self.bindings_of(node.id, node=node)
                   if b.how == "value" and isinstance(b.value, ast.Lambda)]
        func = self.functions.get(node.id)
        return lambdas or ([func] if func is not None else [])

    def _filled(self, *, func: Function, param: str) -> list[tuple[_Growth, _Cell]] | _Unknown:
        """What `func` adds to the list its parameter `param` holds; unknown
        when it passes the parameter on to another pad function or method,
        which may."""
        cell = self.cell_of(func)
        if any(self._binding_scope(cell, call, param) is func and self._pad_callables(node=call.func)
               for call in cell.passed.get(param, ())):
            return _UNKNOWN
        return [(grown, cell) for grown in cell.grows.get(param, ())
                if self._binding_scope(cell, grown.site, param) is func]

    @staticmethod
    def _binding_scope(cell: _Cell, node: ast.AST, name: str) -> ast.AST | None:
        """The innermost scope around `node` that binds `name`; None for module level."""
        for scope in cell.scopes_around(node):
            if any(b.scope is scope for b in cell.bindings.get(name, ())):
                return scope
        return None

    def _grown_paths(
        self, name: str, binding: _Binding, base: list[str], *, use: ast.AST, depth: int,
    ) -> list[str] | _Unknown:
        """`base` with what the pad adds to it after `binding`: each element
        appended, extended or added to a list or set, and a path built on with
        `+=` or `/=`. The unbuilt path stays too unless the addition surely
        ran before `use`: earlier in its cell or in an earlier cell, in the
        binding's own scope and not under a branch or loop."""
        out = list(base)
        sequence = self.is_sequence(binding.value, cell=binding.cell)
        here = self.cell_of(use)
        growth = self.list_growth(name=name, binding=binding) if sequence else self.growth(name, binding)
        if isinstance(growth, _Unknown):
            return growth
        for grown, cell in growth:
            got = self.paths(grown.value, cell=cell, depth=depth + 1)
            if isinstance(got, _Unknown):
                if sequence:
                    return got
                got = ["*"]
            if sequence:
                out += got
                continue
            built = [_join(b, p) if grown.how == "join" else b + p for b in out for p in got]
            ran = cell.index < here.index or cell is here and grown.site.lineno < getattr(use, "lineno", 0)
            # One in a function runs only if it is called.
            in_order = cell.scope_of(grown.site) is binding.scope
            sure = ran and in_order and not cell._under_branch(grown.site, scope=binding.scope)
            out = built if sure else out + built
            if len(out) > 256:
                return _UNKNOWN
        return list(dict.fromkeys(out))

    def is_sequence(self, node: ast.AST, *, cell: _Cell, depth: int = 0) -> bool:
        """Whether `node` is visibly a list, tuple or set of paths, so `+`
        on it adds elements rather than text."""
        if depth > 6:
            return False
        if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.ListComp, ast.SetComp)):
            return True
        if isinstance(node, ast.Call):
            name = self.dotted(node.func)
            return name in ("list", "set", "tuple", "sorted", "glob.glob", "os.listdir") or (
                isinstance(node.func, ast.Attribute) and node.func.attr == "split"
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self.is_sequence(node.left, cell=cell, depth=depth + 1)
        if isinstance(node, ast.Name):
            return any(b.how == "value" and self.is_sequence(b.value, cell=b.cell, depth=depth + 1)
                       for b in self.bindings_of(node.id, node=node))
        return False

    def _unpacked_paths(self, value: ast.AST, *, cell: _Cell, depth: int) -> list[str] | _Unknown:
        # `fd, path = tempfile.mkstemp()`: every part of a temporary is one.
        whole = self.paths(value, cell=cell, depth=depth + 1)
        if not isinstance(whole, _Unknown) and _TEMP_PATH in whole:
            return [_TEMP_PATH]
        # `for i, p in enumerate(paths)`: one of the names is the element.
        if isinstance(value, ast.Call) and self.dotted(value.func) == "enumerate" and value.args:
            got = self.paths(value.args[0], cell=cell, depth=depth + 1)
            return got if isinstance(got, _Unknown) else got + ["*"]
        return _UNKNOWN

    def _call_paths(self, node: ast.Call, *, cell: _Cell, depth: int) -> list[str] | _Unknown:
        name = self.dotted(node.func)
        args = node.args
        if name.startswith("tempfile.") or name in ("tempfile",):
            return [_TEMP_PATH]
        if name in ("os.getcwd", "pathlib.Path.cwd", "os.getcwdb"):
            return ["."]
        if name in ("os.environ.get", "os.getenv"):
            if args and isinstance(args[0], ast.Constant) and args[0].value == _WORKSPACE_ENV:
                return ["."]
            return _UNKNOWN
        if name in _PATH_TYPES or name in ("os.path.join", "posixpath.join"):
            if not args:
                return ["."]
            out = self.paths(args[0], cell=cell, depth=depth + 1)
            if isinstance(out, _Unknown) or _TEMP_PATH in out:
                return out
            for a in args[1:]:
                more = self.paths(a, cell=cell, depth=depth + 1)
                more = ["*"] if isinstance(more, _Unknown) else more
                out = [_join(p, m) for p in out for m in more]
            return out
        if name in ("os.path.abspath", "os.path.realpath", "os.path.normpath",
                    "os.path.expanduser", "os.fspath", "str") and args:
            return self.paths(args[0], cell=cell, depth=depth + 1)
        if name in ("os.path.dirname",) and args:
            base = self.paths(args[0], cell=cell, depth=depth + 1)
            return base if isinstance(base, _Unknown) else [posixpath.dirname(p) or "." for p in base]
        if name in ("glob.glob", "glob.iglob") and args:
            return self.paths(args[0], cell=cell, depth=depth + 1)
        if name == "os.listdir":
            folder = self.paths(args[0], cell=cell, depth=depth + 1) if args else ["."]
            # Bare names: they mean the folder's entries once joined to it,
            # and root entries if used as they are.
            return folder if isinstance(folder, _Unknown) else ["*"]
        if name == "os.scandir":
            folder = self.paths(args[0], cell=cell, depth=depth + 1) if args else ["."]
            return folder if isinstance(folder, _Unknown) else [_join(f, "*") for f in folder]
        if name in _ITER_WRAPPERS and args:
            return self.paths(args[0], cell=cell, depth=depth + 1)
        if name in ("list", "tuple", "set", "frozenset") and not args:
            # Empty: what the cell adds to it counts where it is used.
            return []
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            base = self.paths(node.func.value, cell=cell, depth=depth + 1)
            if isinstance(base, _Unknown) or _TEMP_PATH in base:
                return base
            if method in ("glob", "rglob") and args:
                pattern = self.paths(args[0], cell=cell, depth=depth + 1)
                pattern = ["*"] if isinstance(pattern, _Unknown) else pattern
                deep = "**/" if method == "rglob" else ""
                return [_join(b, deep + p) for b in base for p in pattern]
            if method == "iterdir":
                return [_join(b, "*") for b in base]
            if method in ("resolve", "absolute", "expanduser"):
                return base
            if method == "joinpath":
                out = base
                for a in args:
                    more = self.paths(a, cell=cell, depth=depth + 1)
                    more = ["*"] if isinstance(more, _Unknown) else more
                    out = [_join(p, m) for p in out for m in more]
                return out
            if method in ("with_name", "with_suffix", "with_stem"):
                part = self.paths(args[0], cell=cell, depth=depth + 1) if args else _UNKNOWN
                parts = ["*"] if isinstance(part, _Unknown) else part
                return [_with_part(b, method, p) for b in base for p in parts]
        return _UNKNOWN

    def _is_workspace_env(self, node: ast.Subscript) -> bool:
        return (
            self.dotted(node.value) == "os.environ"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == _WORKSPACE_ENV
        )

    def is_path(self, node: ast.AST, *, cell: _Cell, depth: int = 0) -> bool:
        """Whether `node` is visibly a pathlib path, so `.rename` and `.replace`
        on it are file calls rather than a DataFrame's or a str's."""
        if depth > 6:
            return False
        if isinstance(node, ast.Name):
            arg = self.argument_of(node)
            if arg is not None:
                return self.is_path(arg.node, cell=arg.cell, depth=depth + 1)
            return any(
                b.how in ("value", "element") and self.is_path(b.value, cell=b.cell, depth=depth + 1)
                for b in self.bindings_of(node.id, node=node)
            )
        if isinstance(node, ast.Call):
            name = self.dotted(node.func)
            if name in _PATH_TYPES or name in ("pathlib.Path.cwd", "pathlib.Path.home"):
                return True
            if name in _ITER_WRAPPERS and node.args:
                return self.is_path(node.args[0], cell=cell, depth=depth + 1)
            if isinstance(node.func, ast.Attribute) and node.func.attr in _PATH_MAKERS:
                return self.is_path(node.func.value, cell=cell, depth=depth + 1)
            return False
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return self.is_path(node.left, cell=cell, depth=depth + 1)
        if isinstance(node, ast.Subscript):
            return self.is_path(node.value, cell=cell, depth=depth + 1)
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            return self.is_path(node.value, cell=cell, depth=depth + 1)
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            return self.is_path(node.elt, cell=cell, depth=depth + 1)
        return False

    # ── handles ──────────────────────────────────────────────────────────

    def is_local_handle(self, node: ast.AST, *, depth: int = 0) -> bool:
        """Whether `node` is an in-process DuckDB or SQLite handle, which
        changes no database outside the pad unless the pad ATTACHes one
        outside the process."""
        if self.attaches or depth > 6:
            return False
        if isinstance(node, ast.Name):
            if self.imports.get(node.id, node.id) == "duckdb":
                return True
            found = self.bindings_of(node.id, node=node)
            return bool(found) and all(
                b.how in ("value", "context") and self.is_local_handle(b.value, depth=depth + 1)
                for b in found
            )
        if isinstance(node, ast.Call):
            name = self.dotted(node.func)
            if name in ("duckdb.connect", "sqlite3.connect"):
                target = node.args[0] if node.args else next(
                    (k.value for k in node.keywords if k.arg in ("database", "path")), None
                )
                if target is None:
                    return True
                if not (isinstance(target, ast.Constant) and isinstance(target.value, str)):
                    return False
                return not _is_motherduck(target.value)
            if name in ("sqlalchemy.create_engine", "create_engine") and node.args:
                url = node.args[0]
                if not (isinstance(url, ast.Constant) and isinstance(url.value, str)):
                    return False
                scheme, _, database = url.value.partition(":")
                driver = scheme.split("+")[0]
                return driver == "sqlite" or driver == "duckdb" and not _is_motherduck(database.lstrip("/"))
            if isinstance(node.func, ast.Attribute) and node.func.attr in (
                "cursor", "connect", "begin", "raw_connection",
            ):
                return self.is_local_handle(node.func.value, depth=depth + 1)
            if isinstance(node.func, ast.Name) and node.func.id in self.functions:
                returns = [n.value for n in ast.walk(self.functions[node.func.id])
                           if isinstance(n, ast.Return) and n.value is not None]
                return bool(returns) and all(
                    self.is_local_handle(r, depth=depth + 1) for r in returns
                )
        return False

    def runs_locally(self, func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Whether a function the pad defines runs every statement it gets on a
        local handle."""
        calls = [n for n in ast.walk(func) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr in _EXECUTE]
        return bool(calls) and all(self.is_local_handle(c.func.value) for c in calls)


def _is_motherduck(database: str) -> bool:
    return database.lower().startswith(("md:", "motherduck:"))


def _join(a: str, b: str) -> str:
    if b.startswith("/"):
        return b
    if a in ("", "."):
        return b
    return a.rstrip("/") + "/" + b


def _call_arguments(
    func: Function, call: ast.Call | None, *, call_cell: _Cell, func_cell: _Cell, bound: ast.AST | None = None,
) -> dict[str, _Arg | None]:
    """What `call` passes for each of `func`'s parameters, a parameter's
    default when it passes nothing, None for one the check can't read: past
    a `*args` or `**kwargs`, or with no call at all. With `bound`, a
    method's first parameter gets it and the call's arguments fill the rest."""
    a = func.args
    positional = [*a.posonlyargs, *a.args]
    defaults = dict(zip([p.arg for p in positional][len(positional) - len(a.defaults):], a.defaults, strict=True))
    defaults.update({p.arg: d for p, d in zip(a.kwonlyargs, a.kw_defaults, strict=True) if d is not None})
    out: dict[str, _Arg | None] = {}
    if bound is not None and positional:
        out[positional[0].arg] = _Arg(node=bound, cell=call_cell)
        positional = positional[1:]
    if call is None:
        return {**out, **{p.arg: None for p in (*positional, *a.kwonlyargs)}}
    starred = next((i for i, x in enumerate(call.args) if isinstance(x, ast.Starred)), len(call.args))
    spread = any(k.arg is None for k in call.keywords)
    keywords = {k.arg: k.value for k in call.keywords if k.arg is not None}
    for i, p in enumerate((*positional, *a.kwonlyargs)):
        if p.arg in keywords:
            out[p.arg] = _Arg(node=keywords[p.arg], cell=call_cell)
        elif i < len(positional) and i < starred:
            out[p.arg] = _Arg(node=call.args[i], cell=call_cell)
        elif i < len(positional) and starred < len(call.args) or spread:
            out[p.arg] = None
        elif p.arg in defaults:
            out[p.arg] = _Arg(node=defaults[p.arg], cell=func_cell)
        else:
            out[p.arg] = None
    return out


def _partial_call(*, partial: ast.Call, call: ast.Call) -> ast.Call:
    """The call `call` makes through `partial`, a `functools.partial(f, ...)`:
    `f` given the partial's arguments, then the call's."""
    made = ast.Call(func=partial.args[0], args=[*partial.args[1:], *call.args],
                    keywords=[*partial.keywords, *call.keywords])
    return ast.copy_location(made, call)


def _with_part(path: str, method: str, part: str) -> str:
    """`Path.with_name`, `with_suffix` or `with_stem` on a path pattern."""
    folder, name = posixpath.split(path)
    stem, suffix = posixpath.splitext(name)
    if method == "with_name":
        name = part
    elif method == "with_suffix":
        name = stem + part
    else:
        name = part + suffix
    return _join(folder or ".", name)


# ── SQL text ──────────────────────────────────────────────────────────────

_PERCENT = re.compile(r"%(?:\((\w+)\))?[-#0 +]*(?:\d+|\*)?(?:\.\d+)?([diouxXeEfFgGcrsa%])")


def _sql_text(
    node: ast.AST, reader: _Reader, *, parts: list[ast.AST] | None = None, depth: int = 0,
) -> str | None:
    """The text a string-building expression makes, a placeholder for each
    part chosen at run time; None when it isn't one. Each node read into the
    text goes into `parts`."""
    parts = [] if parts is None else parts
    if depth > 40:
        return None
    text: str | None = None
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            text = node.value
        elif isinstance(node.value, bytes):
            text = node.value.decode("utf-8", errors="replace")
    elif isinstance(node, ast.JoinedStr):
        text = "".join(str(v.value) if isinstance(v, ast.Constant) else _field_text(v, reader) for v in node.values)
        parts.extend(v for v in node.values if isinstance(v, ast.Constant))
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        text = _chain_text(node, reader, parts=parts, depth=depth)
    elif isinstance(node, ast.Call):
        text = _call_text(node, reader, parts=parts, depth=depth)
    if text is not None:
        parts.append(node)
    return text


def _field_text(field: ast.AST, reader: _Reader) -> str:
    """An f-string's `{...}`: the literal the call being read passes for a
    parameter, else a placeholder."""
    if (isinstance(field, ast.FormattedValue) and isinstance(field.value, ast.Name)
            and field.conversion == -1 and field.format_spec is None):
        filled = reader.literal_argument(field.value)
        if filled is not None:
            return filled
    return reader.hole(field)


def _chain_text(node: ast.BinOp, reader: _Reader, *, parts: list[ast.AST], depth: int) -> str | None:
    """A `+` chain or a `%` format. Flattened with a stack, so a long chain
    can't exhaust recursion, and each link goes into `parts` so the caller
    reads the chain once."""
    pieces: list[str] = []
    stack: list[ast.AST] = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, ast.BinOp) and isinstance(item.op, ast.Add):
            parts.append(item)
            stack.append(item.right)
            stack.append(item.left)
        elif isinstance(item, ast.BinOp) and isinstance(item.op, ast.Mod):
            template = _sql_text(item.left, reader, parts=parts, depth=depth + 1)
            if template is None:
                # Not a format: its right side is read on its own.
                pieces.append(reader.hole(item))
            else:
                parts.append(item)
                pieces.append(_percent_text(template, item.right, reader, parts=parts, depth=depth))
        elif isinstance(item, ast.Name):
            filled = reader.literal_argument(item)
            pieces.append(reader.hole(item) if filled is None else filled)
        else:
            text = _sql_text(item, reader, parts=parts, depth=depth + 1)
            pieces.append(reader.hole(item) if text is None else text)
    joined = "".join(pieces)
    return joined if _HOLE_MARK.sub("", joined).strip() else None


def _call_text(node: ast.Call, reader: _Reader, *, parts: list[ast.AST], depth: int) -> str | None:
    if reader.dotted(node.func) in _SQL_WRAPPERS and node.args:
        return _sql_text(node.args[0], reader, parts=parts, depth=depth + 1)
    if not isinstance(node.func, ast.Attribute):
        return None
    method = node.func.attr
    if method == "join" and len(node.args) == 1:
        separator = _sql_text(node.func.value, reader, parts=parts, depth=depth + 1)
        if separator is None:
            return None
        return _join_text(separator, node.args[0], reader, parts=parts, depth=depth)
    if method in _STR_METHODS:
        receiver = _sql_text(node.func.value, reader, parts=parts, depth=depth + 1)
        if receiver is None or method != "format":
            return receiver
        return _format_text(receiver, node, reader, parts=parts, depth=depth)
    return None


def _argument_text(arg: ast.AST | None, reader: _Reader, *, parts: list[ast.AST], depth: int) -> str | None:
    """A format argument's text: a string the check can build, or a number."""
    if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)) and not isinstance(arg.value, bool):
        parts.append(arg)
        return str(arg.value)
    if arg is None:
        return None
    if isinstance(arg, ast.Name):
        return reader.literal_argument(arg)
    return _sql_text(arg, reader, parts=parts, depth=depth + 1)


def _item(node: ast.AST, key: str | int) -> ast.Subscript:
    """`node[key]`: the code that fills in one field from a tuple or mapping
    the code builds at run time."""
    return ast.Subscript(value=node, slice=ast.Constant(value=key), ctx=ast.Load())


def _percent_text(template: str, right: ast.AST, reader: _Reader, *, parts: list[ast.AST], depth: int) -> str:
    """`template % right`, each literal argument put in its place and each
    other one a placeholder for the code that fills it in."""
    fields = [m for m in _PERCENT.finditer(template) if m.group(2) != "%"]
    if isinstance(right, ast.Dict):
        named = {k.value: v for k, v in zip(right.keys, right.values)
                 if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        positional: list[ast.AST] = []
    elif isinstance(right, ast.Tuple) or len(fields) == 1 and fields[0].group(1) is None:
        named = {}
        positional = list(right.elts) if isinstance(right, ast.Tuple) else [right]
    else:
        # A tuple or mapping the code builds at run time: each field is one item of it.
        named = {m.group(1): _item(right, m.group(1)) for m in fields if m.group(1) is not None}
        positional = [_item(right, i) for i in range(sum(m.group(1) is None for m in fields))]
    position = 0

    def fill(m: re.Match[str]) -> str:
        nonlocal position
        key, conversion = m.group(1), m.group(2)
        if conversion == "%":
            return "%"
        if key is not None:
            arg = named.get(key)
        else:
            arg = positional[position] if position < len(positional) else None
            position += 1
        value = _argument_text(arg, reader, parts=parts, depth=depth)
        return reader.hole(arg) if value is None else value

    return _PERCENT.sub(fill, template)


def _format_text(template: str, call: ast.Call, reader: _Reader, *, parts: list[ast.AST], depth: int) -> str:
    """`template.format(...)`, each literal argument put in its place and
    each other one a placeholder for the code that fills it in, or for no
    code when the field takes an attribute or item of it."""
    pieces: list[str] = []
    position = 0
    try:
        fields = list(string.Formatter().parse(template))
    except ValueError:
        return template
    for literal, field, spec, conversion in fields:
        pieces.append(literal)
        if field is None:
            continue
        key = re.split(r"[.\[]", field, maxsplit=1)[0]
        arg: ast.AST | None
        if key == "":
            arg = call.args[position] if position < len(call.args) else None
            position += 1
        elif key.isdigit():
            arg = call.args[int(key)] if int(key) < len(call.args) else None
        else:
            arg = next((k.value for k in call.keywords if k.arg == key), None)
        if field != key or arg is None:
            pieces.append(reader.hole(None))
            continue
        # A conversion or format spec changes the literal's text, so it shows as code.
        value = None if conversion or spec else _argument_text(arg, reader, parts=parts, depth=depth)
        pieces.append(reader.hole(arg) if value is None else value)
    return "".join(pieces)


def _join_text(separator: str, arg: ast.AST, reader: _Reader, *, parts: list[ast.AST], depth: int) -> str | None:
    """`separator.join(arg)` for a list of literals, a comprehension, or a
    name holding a list built from literals."""
    nodes: list[ast.AST | None]
    items: list[str | None]
    if isinstance(arg, (ast.List, ast.Tuple, ast.Set)):
        nodes = list(arg.elts)
        items = [_sql_text(e, reader, parts=parts, depth=depth + 1) for e in arg.elts]
    elif isinstance(arg, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        nodes = [arg.elt]
        items = [_sql_text(arg.elt, reader, parts=parts, depth=depth + 1)]
    elif isinstance(arg, ast.Name) and depth < 20:
        nodes = [item for b in reader.bindings_of(arg.id, node=arg) for item in (_list_items(arg.id, b) or [])]
        items = [None if item is None else _sql_text(item, reader, depth=depth + 1) for item in nodes]
    else:
        return None
    if not items:
        return None
    parts.append(arg)
    joined = separator.join(reader.hole(n) if t is None else t for t, n in zip(items, nodes))
    return joined if _HOLE_MARK.sub("", joined).strip() else None


def _list_items(name: str, binding: _Binding) -> list[ast.AST | None] | None:
    """The elements of the list `binding` gives `name`, with what `.append`,
    `.insert`, `.add` and `.extend` add (None for an element the check can't
    read); None when it gives no list."""
    value = binding.value
    if binding.how != "value":
        return None
    items: list[ast.AST | None]
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        items = list(value.elts)
    elif isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        items = [value.elt]
    else:
        return None
    cell = binding.cell
    for call in cell.additions.get(name, ()):
        if cell.scope_of(call) is not binding.scope or not call.args or _ran_before(call, binding, cell=cell):
            continue
        added = call.args[-1]
        if call.func.attr != "extend":
            items.append(added)
        elif isinstance(added, (ast.List, ast.Tuple, ast.Set)):
            items += added.elts
        else:
            items.append(None)
    return items


def _name_statements(
    use: ast.Name, reader: _Reader, *, prose: bool, root: ast.AST | None = None,
) -> tuple[str, ...]:
    """The statements of each text with a write, read as `write_statements`
    reads it with `prose`, a name holds where this cell reads it, when an
    earlier cell bound it or this cell builds it up with `+=` or `.append`.
    The literal pass reads every other binding where it is written.

    With `root`, an earlier cell's function read for a call: a binding outside
    it counts unless the checked cells made it, since their literal pass
    already follows their values into the functions they call.

    Read once per check, read, `root` and set of call arguments.
    """
    key = (id(use), prose, id(root), reader.call_key())
    if key in reader.named:
        return reader.named[key]
    cell = reader.cell_of(use)
    found: list[str] = []
    for b in reader.bindings_of(use.id, node=use):
        own = b.cell is cell and not cell.builds_up(use.id, scope=b.scope)
        if root is None and own:
            continue
        if root is not None and (b.cell.index >= reader.current.index or own and _inside(b.value, root, cell)):
            continue
        found += reader.statements_held(use.id, b, prose=prose)
    reader.named[key] = tuple(dict.fromkeys(found))
    return reader.named[key]


def _held_texts(name: str, binding: _Binding, reader: _Reader, *, depth: int) -> list[str]:
    """Every text `binding` can give `name`: its string with what `+=` adds,
    each element of a list it holds, or each element a loop gives it."""
    if depth > 6:
        return []
    items = _list_items(name, binding)
    if binding.how == "element":
        it = binding.value
        if isinstance(it, (ast.List, ast.Tuple, ast.Set)):
            items = list(it.elts)
        elif isinstance(it, ast.Name):
            return [t for b in reader.bindings_of(it.id, node=it)
                    for t in _held_texts(it.id, b, reader, depth=depth + 1)]
    if items is not None:
        return [t for t in (_sql_text(i, reader, depth=depth + 1) for i in items if i is not None) if t is not None]
    if binding.how != "value":
        return []
    cell = binding.cell
    nodes = [binding.value, *(a.value for a in cell.augments.get(name, ())
                              if cell.scope_of(a) is binding.scope and not _ran_before(a, binding, cell=cell))]
    pieces = [_sql_text(n, reader, depth=depth + 1) for n in nodes]
    if all(p is None for p in pieces):
        return []
    return ["".join(reader.hole(n) if p is None else p for p, n in zip(pieces, nodes))]


# ── SQL findings ──────────────────────────────────────────────────────────

def _is_harmless_call(call: ast.Call, reader: _Reader) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in _HARMLESS_CALLS
    if isinstance(func, ast.Attribute):
        if func.attr in _HARMLESS_METHODS:
            return True
        # A logger-named root is logging, unless the method runs SQL: a cursor
        # someone named `log` still runs `log.execute(...)`.
        if func.attr in _EXECUTE or func.attr in _READ_SQL:
            return False
        root = func.value
        while isinstance(root, ast.Attribute):
            root = root.value
        return isinstance(root, ast.Name) and (
            root.id in _LOGGER_NAMES or reader.imports.get(root.id) == "logging"
        )
    return False


def _executes_on_local(call: ast.Call, reader: _Reader) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in _EXECUTE:
        return reader.is_local_handle(func.value)
    if isinstance(func, ast.Attribute) and func.attr in _READ_SQL:
        con = call.args[1] if len(call.args) > 1 else next(
            (k.value for k in call.keywords if k.arg == "con"), None
        )
        return con is not None and reader.is_local_handle(con)
    if isinstance(func, ast.Name) and func.id in reader.functions:
        return reader.runs_locally(reader.functions[func.id])
    return False


def _runs_sql(call: ast.Call, reader: _Reader) -> bool:
    """Whether `call` runs the statement it is given: an execute, read_sql,
    .sql() or .query() call."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr in _EXECUTE or func.attr in _READ_SQL
    return reader.dotted(func) in ("pandas.read_sql", "pandas.read_sql_query")


def _is_statement(arg: ast.AST, call: ast.Call) -> bool:
    """Whether `arg`, one of `call`'s arguments or a keyword it passes, is
    the statement it runs, not a parameter or a connection."""
    if isinstance(arg, ast.keyword):
        return arg.arg in _STATEMENT_KEYWORDS
    return bool(call.args) and arg is call.args[0]


def _parameter_uses(call: ast.Call, value: ast.AST, reader: _Reader) -> list[ast.AST]:
    """Where a function the pad defines, called by name, reads the
    parameter `call` passes `value` for; empty for any other call."""
    if not (isinstance(call.func, ast.Name) and call.func.id in reader.functions):
        return []
    func = reader.functions[call.func.id]
    passed = _call_arguments(func, call, call_cell=reader.cell_of(call), func_cell=reader.cell_of(func))
    params = [name for name, arg in passed.items() if arg is not None and arg.node is value]
    return [use for name in params for use in _loads_in(func.body, name)]


def _name_uses(names: set[str], *, at: ast.AST, reader: _Reader) -> list[ast.AST]:
    """Where a value bound to `names` at `at` is read: in its cell, inside the
    pad functions this cell calls (a module-level value is their global),
    and as an attribute when `at` sits in a class body."""
    cell = reader.cell_of(at)
    uses: list[ast.AST] = [u for n in names for u in cell.loads.get(n, ())]
    scope = cell.scope_of(at)
    if scope is None:
        for called in reader.called:
            func = reader.functions[called]
            home = reader.cell_of(func)
            for n in names:
                if any(b.scope is func for b in home.bindings.get(n, ())):
                    continue
                uses += [u for u in home.loads.get(n, ()) if _inside(u, func, home)]
    elif isinstance(scope, ast.ClassDef):
        uses += [a for c in {cell, reader.current} for a in ast.walk(c.tree)
                 if isinstance(a, ast.Attribute) and a.attr in names and isinstance(a.ctx, ast.Load)]
    return uses


def _loads_in(nodes: Sequence[ast.AST], name: str) -> list[ast.AST]:
    return [n for root in nodes for n in ast.walk(root)
            if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)]


def _target_names(target: ast.AST) -> list[str] | None:
    """The names a loop's target binds, as in `for i, (a, b) in ...`; None
    when it binds anything else, such as an attribute."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if not isinstance(target, (ast.Tuple, ast.List)):
        return None
    names: list[str] = []
    for elt in target.elts:
        inner = _target_names(elt)
        if inner is None:
            return None
        names += inner
    return names


def _stored_name(target: ast.AST) -> str | None:
    """The name an assignment stores into: `x`, or `df` for `df['sql']` or
    `cfg.sql`; None for anything else, such as a tuple."""
    while isinstance(target, (ast.Subscript, ast.Attribute)):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def _built_here(name: str, *, at: ast.AST, reader: _Reader) -> bool:
    """Whether `name` holds, where `at` stores into an entry or attribute of
    it, an object the pad built that no other name can hold: a literal, a
    comprehension, or what a `_FRESH_CALLS` call or `.copy()` returns, and
    every read of it in the pad reads an entry or attribute of it or shows it."""

    def fresh(value: ast.AST) -> bool:
        if isinstance(value, (ast.Dict, ast.List, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
            return True
        return isinstance(value, ast.Call) and (
            reader.dotted(value.func) in _FRESH_CALLS
            or isinstance(value.func, ast.Attribute) and value.func.attr == "copy"
        )

    found = reader.bindings_of(name, node=at)
    if not found or not all(b.how == "value" and fresh(b.value) for b in found):
        return False
    for cell in reader.cells:
        for use in cell.loads.get(name, ()):
            parent = cell.parents.get(use)
            if isinstance(parent, (ast.Attribute, ast.Subscript)) and parent.value is use:
                continue
            if isinstance(parent, ast.Call) and use in parent.args and _is_harmless_call(parent, reader):
                continue
            return False
    return True


def _mapping_call(func: Function, cell: _Cell, reader: _Reader) -> ast.Call | None:
    """The map(), filter() or DataFrame apply() a lambda is passed to: what
    the lambda returns goes on in what that call returns."""
    if not isinstance(func, ast.Lambda):
        return None
    call = cell.parents.get(func)
    if isinstance(call, ast.keyword):
        call = cell.parents.get(call)
    if not isinstance(call, ast.Call) or call.func is func:
        return None
    if reader.dotted(call.func) in _MAPPERS or isinstance(call.func, ast.Attribute) and (
        call.func.attr in _MAPPER_METHODS
    ):
        return call
    return None


def _own_calls(func: Function, cell: _Cell, reader: _Reader) -> list[ast.Call] | None:
    """Each call of a function or named lambda the checked cells define, in
    the checked cells; None when its return value may go anywhere else: an
    earlier cell's, a method, a decorated one, or one passed on by name."""
    if cell.index < reader.current.index or isinstance(cell.scope_of(func), ast.ClassDef):
        return None
    if isinstance(func, ast.Lambda):
        names = cell.lambda_names.get(id(func), [])
    else:
        names = [] if func.decorator_list else [func.name]
    if not names:
        return None
    calls: list[ast.Call] = []
    for checked in reader.cells[reader.current.index:]:
        for use in (u for n in names for u in checked.loads.get(n, ())):
            call = checked.parents.get(use)
            if not (isinstance(call, ast.Call) and call.func is use):
                return None
            calls.append(call)
    return calls


class _Reach(IntEnum):
    """Where a statement's value can go, ordered so that the furthest of a
    value's uses is the largest."""

    # Only uses the check sees are harmless.
    NOWHERE = 0
    # A use the check can't follow, such as a report, a template or a
    # function it doesn't know: the text may be English there.
    ELSEWHERE = 1
    # A call that runs it as SQL.
    RUNNER = 2


def _reaches_a_runner(
    node: ast.AST, reader: _Reader, *, depth: int = 0, followed: set[int] | None = None, any_handle: bool = False,
) -> _Reach:
    """Where a statement's value can go: the furthest of its uses.

    NOWHERE only when every use is visibly harmless: a bare expression (a
    docstring), a print, log or file write, a comparison or condition, a
    list nothing runs, a name nothing reads, or an execute on a local
    handle. RUNNER when a use is the statement an execute, read_sql, .sql()
    or .query() call runs (`_runs_sql`), what text() or sql.SQL() wraps,
    or a parameter a pad function runs so. ELSEWHERE for anything else,
    because the check can't follow it. With `any_handle`, an execute on a
    local handle is a runner too.
    """
    if depth > _FLOW_DEPTH:
        return _Reach.ELSEWHERE
    followed = set() if followed is None else followed
    cell = reader.cell_of(node)
    parent = cell.parents.get(node)

    def on(target: ast.AST) -> _Reach:
        return _reaches_a_runner(target, reader, depth=depth + 1, followed=followed, any_handle=any_handle)

    def via(uses: Sequence[ast.AST]) -> _Reach:
        fresh = [u for u in uses if id(u) not in followed]
        followed.update(id(u) for u in fresh)
        furthest = _Reach.NOWHERE
        for u in fresh:
            furthest = max(furthest, on(u))
            if furthest is _Reach.RUNNER:
                break
        return furthest

    def via_names(names: set[str], *, at: ast.AST) -> _Reach:
        uses = _name_uses(names, at=at, reader=reader)
        checked = cell.index >= reader.current.index
        if checked and any(cell.builds_up(n, scope=cell.scope_of(at)) for n in names):
            # The name pass reports this cell's reads of a built-up name with
            # its whole text, so a part isn't reported again on its own.
            uses = [u for u in uses if not (
                reader.cell_of(u) is cell and isinstance(u, ast.Name) and _name_statements(u, reader, prose=False)
            )]
        given = {id(b) for n in names for b in reader.bindings_of(n, node=at)} if cell.scope_of(at) is None else set()
        if given:
            # A read in the cell after the name is bound again gets the new value.
            uses = [u for u in uses if not (
                isinstance(u, ast.Name) and reader.cell_of(u) is cell and cell.scope_of(u) is None
                and _reads_another(reader.bindings_of(u.id, node=u), given)
            )]
        return via(uses)

    def wrapped(name: str, call: ast.Call) -> _Reach:
        """Where what a wrapper such as str() or text() returns goes: for a
        statement object, a runner wherever it goes."""
        reach = on(call)
        return _Reach.RUNNER if reach and name in _SQL_OBJECTS else reach

    def into(func: ast.AST, call: ast.Call) -> _Reach:
        """Where the items `call` hands `func` go: into a lambda's first
        parameter, and for filter() into what `call` returns; through str(),
        text() or a str method into what `call` returns; into a runner for
        an execute; anywhere for any other."""
        if isinstance(func, ast.Lambda):
            params = [*func.args.posonlyargs, *func.args.args]
            reach = via(_loads_in([func.body], params[0].arg)) if params else _Reach.NOWHERE
            if reader.dotted(call.func) == "filter":
                reach = max(reach, on(call))
            return reach
        name = reader.dotted(func)
        if name in _SQL_WRAPPERS or name in _PASS_THROUGH:
            return wrapped(name, call)
        if isinstance(func, ast.Attribute) and reader.dotted(func.value) == "str" and func.attr in _STR_METHODS:
            return on(call)
        if isinstance(func, ast.Attribute) and (func.attr in _EXECUTE or func.attr in _READ_SQL) and (
            any_handle or not reader.is_local_handle(func.value)
        ):
            return _Reach.RUNNER
        return _Reach.ELSEWHERE

    if parent is None or isinstance(parent, ast.Expr):
        return _Reach.NOWHERE
    if isinstance(parent, ast.Return) or isinstance(parent, ast.Lambda) and node is parent.body:
        # A function's return value: it goes where the call being read sends
        # it; for a lambda map() or apply() calls, into what that returns;
        # for a function the checked cells define, to each call of it there;
        # and anywhere at all for any other.
        func = parent if isinstance(parent, ast.Lambda) else cell.scope_of(parent)
        call = reader.returns_to.get(id(func)) or _mapping_call(func, cell, reader)
        if call is not None:
            return on(call)
        calls = _own_calls(func, cell, reader)
        return _Reach.ELSEWHERE if calls is None else via(calls)
    if isinstance(parent, (ast.JoinedStr, ast.FormattedValue, ast.BinOp, ast.BoolOp, ast.Starred,
                           ast.List, ast.Tuple, ast.Set, ast.Dict,
                           ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        return on(parent)
    if isinstance(parent, ast.IfExp):
        return on(parent) if node is not parent.test else _Reach.NOWHERE
    if isinstance(parent, ast.Subscript):
        # Storing into an entry of it reads nothing out of it.
        if node is parent.value and not isinstance(parent.ctx, (ast.Store, ast.Del)):
            return on(parent)
        return _Reach.NOWHERE
    if isinstance(parent, ast.comprehension):
        if node is not parent.iter:
            return _Reach.NOWHERE
        names = _target_names(parent.target)
        if names is None:
            return _Reach.ELSEWHERE
        comp = cell.parents[parent]
        body = [comp.key, comp.value] if isinstance(comp, ast.DictComp) else [comp.elt]
        body += [g.iter for g in comp.generators[comp.generators.index(parent) + 1:]]
        return via([u for n in names for u in _loads_in(body, n)])
    if isinstance(parent, (ast.For, ast.AsyncFor)):
        names = _target_names(parent.target) if node is parent.iter else None
        if names is None:
            return _Reach.ELSEWHERE
        return via([u for n in names for u in _loads_in(parent.body, n)])
    if isinstance(parent, ast.Attribute):
        if parent.attr in _ADDERS or isinstance(parent.ctx, (ast.Store, ast.Del)):
            # Adding to a list, or setting an attribute: the value's own uses
            # carry what it holds.
            return _Reach.NOWHERE
        call = cell.parents.get(parent)
        called = isinstance(call, ast.Call) and call.func is parent
        if parent.attr in _MAPPER_METHODS and called:
            return into(call.args[0], call) if call.args else _Reach.ELSEWHERE
        if parent.attr in _STR_METHODS or parent.attr == "join" or parent.attr in _CARRIERS:
            return on(call if called else parent)
        if called:
            return _Reach.ELSEWHERE  # any other method may run it
        return on(parent)  # a field, such as a row's `r.days`, carries what it holds on
    value = node
    if isinstance(parent, ast.keyword):
        node, parent = parent, cell.parents.get(parent)
    if isinstance(parent, ast.Call):
        if parent.func is node:
            return _Reach.ELSEWHERE
        name = reader.dotted(parent.func)
        if name in _SQL_WRAPPERS or name in _PASS_THROUGH:
            return wrapped(name, parent)
        if name in ("map", "filter") and len(parent.args) > 1 and node is not parent.args[0]:
            return into(parent.args[0], parent)
        func = parent.func
        if isinstance(func, ast.Attribute) and func.attr == "join":
            joined = _sql_text(parent, reader) if reader.cell_of(parent).index >= reader.current.index else None
            if joined is not None and write_statements(joined, prose=False):
                # The literal pass reports this join whole, as its own statement.
                return _Reach.NOWHERE
            return on(parent)
        if isinstance(func, ast.Attribute) and func.attr == "format":
            return on(parent)
        if isinstance(func, ast.Attribute) and func.attr in _ADDERS:
            if not isinstance(func.value, ast.Name):
                return _Reach.ELSEWHERE
            return via_names({func.value.id}, at=parent)
        if _is_harmless_call(parent, reader) or not any_handle and _executes_on_local(parent, reader):
            return _Reach.NOWHERE
        if _runs_sql(parent, reader):
            return _Reach.RUNNER if _is_statement(node, parent) else _Reach.ELSEWHERE
        # A pad function may run the parameter it gets; any other call may too.
        return max(_Reach.ELSEWHERE, via(_parameter_uses(parent, value, reader)))
    if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr, ast.AugAssign)):
        targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
        # Into a name, or an entry or attribute of one such as df['sql'] when
        # no other name holds what it does: the name's uses carry it on.
        names: set[str] = set()
        for t in targets:
            stored = _stored_name(t)
            if stored is None or not isinstance(t, ast.Name) and not _built_here(stored, at=t, reader=reader):
                return _Reach.ELSEWHERE
            names.add(stored)
        return via_names(names, at=parent)
    if isinstance(parent, (ast.Compare, ast.UnaryOp, ast.If, ast.While, ast.Assert)):
        return _Reach.NOWHERE
    return _Reach.ELSEWHERE


# The keywords that pass a statement's parameters: psycopg2's vars, psycopg's
# and pandas' params, SQLAlchemy's and sqlite3's parameters, PyMySQL's args.
_VALUES_KEYWORDS = frozenset({"vars", "params", "parameters", "args"})


def _values_text(call: ast.Call, *, reader: _Reader) -> str:
    """The parameters a call passes with its statement, whole: a literal, the
    tuple literal a name holds, or else a placeholder for the code."""
    given = next((k.value for k in call.keywords if k.arg in _VALUES_KEYWORDS), None)
    if given is None and len(call.args) > 1 and isinstance(call.func, ast.Attribute) and call.func.attr in _EXECUTE:
        given = call.args[1]
    if isinstance(given, ast.Name):
        # Only a tuple: a list or dict can change after it is bound.
        found = reader.bindings_of(given.id, node=given)
        if (len(found) == 1 and found[0].how == "value" and isinstance(found[0].value, ast.Tuple)
                and not reader.growth(given.id, found[0])):
            given = found[0].value
    if isinstance(given, (ast.Tuple, ast.List, ast.Dict)):
        # Scrub each string as the cell holds it, before unparse escapes its
        # line breaks: "access_token=\n<value>" only reads as a key and its
        # value with the real line break in place. A value is plain text, not
        # SQL, so it takes the line-aware scrubber.
        shown = copy.deepcopy(given)
        for node in ast.walk(shown):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = _scrub_lines(node.value)
        try:
            return ast.unparse(shown)
        except Exception:
            return ""
    return "" if given is None else reader.hole(given)


def _card_lines(text: str) -> str:
    """`text` line by line, as the card quotes it. Each line loses its
    trailing space and the lines after the first lose the indentation they
    share, so a `--` comment keeps to its own line. A line break inside a
    quoted literal, quoted name or block comment stays as written, with the
    spaces around it."""
    kept = [m.span() for m in _TOKEN.finditer(text)
            if m.lastgroup in ("comment", "text", "name") and ("\n" in m.group() or "\r" in m.group())]
    lines: list[str] = []
    begin = k = 0
    for m in _LINE_BREAK.finditer(text):
        while k < len(kept) and kept[k][1] <= m.start():
            k += 1
        if k < len(kept) and kept[k][0] < m.start():
            continue
        lines.append(text[begin:m.start()])
        begin = m.end()
    lines.append(text[begin:])
    rest = [line.rstrip() for line in lines[1:]]
    margin = os.path.commonprefix([line[:len(line) - len(line.lstrip(" \t"))] for line in rest if line])
    return "\n".join([lines[0].rstrip(), *(line[len(margin):] for line in rest)])


# A line break and the spaces and tabs around it.
_BREAK_RUN = re.compile(r"([ \t]*(?:\r\n|\r|\n)[ \t]*)")
# The white space between two words of code.
_SPACE_RUN = re.compile(r"(\s+)")
# Each key whose value `scrub_output` reads only on the key's own line (an
# access, refresh or id token, a client secret) holds one of these words.
_TOKEN_KEY_WORD = re.compile("token|secret", re.I)


def _scrub(text: str) -> str:
    """A statement or its values as the card quotes them, each secret in it
    redacted and the rest as written.

    `scrub_output` runs a token key's value on through spaces, so on a whole
    statement it takes what follows the value with it, a WHERE included.
    Here each quoted literal, quoted name and comment is scrubbed on its own,
    after the text since the piece before it, so a key there, as in
    `access_token = '...'` or `{'access_token': '...'}`, still names its
    value, and keeps the quote or `*/` that ends it. The code between them is
    scrubbed one stretch with no space at a time, so a word that is a secret
    is redacted and every other word stays.
    """
    if _scrub_lines(text) == text:
        return text
    out: list[str] = []
    position = context_from = 0
    for m in _TOKEN.finditer(text):
        if m.lastgroup not in ("text", "name", "comment"):
            continue
        out.append(_scrub_code(text[position:m.start()]))
        scrubbed = _scrub_after(text[context_from:m.start()], m.group())
        closer = _closer(m)
        out.append(scrubbed if scrubbed.endswith(closer) else scrubbed + closer)
        context_from, position = m.start(), m.end()
    out.append(_scrub_code(text[position:]))
    return "".join(out)


def _closer(m: re.Match[str]) -> str:
    """What ends the quoted literal, quoted name or comment `_TOKEN` matched."""
    if m.lastgroup == "comment":
        return "*/" if m.group().startswith("/*") else ""
    if m.lastgroup == "text" and m.group("tag") is not None:
        return f"${m.group('tag')}$"
    return m.group()[-1]


def _scrub_after(context: str, piece: str) -> str:
    """`piece` scrubbed after `context`, the text before it, so a token key
    there names a value in it. A redaction that starts in `context` and
    runs on into `piece` takes `piece` whole."""
    head = _scrub_lines(context)
    whole = _scrub_lines(context + piece)
    return whole[len(head):] if whole.startswith(head) else "[REDACTED_TOKEN]"


def _scrub_code(code: str) -> str:
    """Code outside quotes and comments, each stretch with no space in it
    scrubbed on its own."""
    return "".join(
        part if not part or part.isspace() else scrub_output(part) for part in _SPACE_RUN.split(code)
    )


def _scrub_lines(text: str) -> str:
    """`text` scrubbed, a token value on the line after its key included.
    `scrub_output` reads a key and its value across spaces and tabs only, so
    a second pass reads each line break, with the spaces around it, as a tab
    and then puts the line breaks back."""
    scrubbed = scrub_output(text)
    pieces = _BREAK_RUN.split(scrubbed)
    if len(pieces) == 1 or not _TOKEN_KEY_WORD.search(scrubbed):
        return scrubbed
    lines, breaks = pieces[0::2], pieces[1::2]
    parts = scrub_output("\t".join(lines)).split("\t")
    sizes = [line.count("\t") + 1 for line in lines]
    if len(parts) != sum(sizes):
        # A redaction took a line break with it: the breaks it left can't be placed.
        return "\n".join(parts)
    out: list[str] = []
    taken = 0
    for size, brk in zip(sizes, [*breaks, ""], strict=True):
        out.append("\t".join(parts[taken:taken + size]) + brk)
        taken += size
    return "".join(out)


def _sql_write(text: str, *, reader: _Reader, values: str = "", line: int) -> SqlWrite:
    """One statement as the card quotes it: whole, scrubbed, and line by line
    (`_card_lines`). A part the code picks at run time shows as that code,
    in braces."""
    return SqlWrite(
        statement=_scrub(reader.show_holes(_card_lines(text))),
        values=_scrub(reader.show_holes(values)) if values else "",
        line=line, run_time_parts=any(_NUMBERED_HOLE.search(t) for t in (text, values)),
    )


def _sql_findings(
    reader: _Reader, cell: _Cell, *, root: ast.AST | None = None, report: _Report | None = None,
) -> list[SqlWrite]:
    """The statements `cell` runs; with `root`, the ones a function of that
    cell runs, reported at `report`, the call in the checked cell."""
    out: list[SqlWrite] = []
    top = cell.tree if root is None else root

    def add(statements: Sequence[str], node: ast.AST) -> None:
        parent = cell.parents.get(node)
        if isinstance(parent, ast.keyword):
            parent = cell.parents.get(parent)
        values = _values_text(parent, reader=reader) if isinstance(parent, ast.Call) else ""
        line = report.line if report else cell.host_line or getattr(node, "lineno", 0)
        out.extend(_sql_write(s, reader=reader, values=values, line=line) for s in statements)

    # Statements written in this cell. A walk meets an expression before its
    # parts, so each is read whole once and its parts are skipped.
    read: set[int] = set()
    for node in ast.walk(top):
        if id(node) in read or not isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp, ast.Call)):
            continue
        parts: list[ast.AST] = []
        text = _sql_text(node, reader, parts=parts)
        read.update(id(p) for p in parts)
        if text is None or not write_statements(text, prose=False):
            continue
        # A string that holds a write is held whole: the card quotes every
        # statement in it, since one Approve runs them all. Where it goes
        # decides how it reads: a SQL runner runs it whatever its case, and
        # anywhere else it may be English (`write_statements`).
        reach = _reaches_a_runner(node, reader)
        if reach is _Reach.RUNNER or reach is _Reach.ELSEWHERE and write_statements(text):
            add(_quoted_statements(text), node)
    # A name an earlier cell bound to a statement, or one this cell builds up.
    for uses in cell.loads.values():
        for use in uses:
            if id(use) in read or root is not None and not _inside(use, root, cell):
                continue
            statements = _name_statements(use, reader, prose=False, root=root)
            if not statements:
                continue
            reach = _reaches_a_runner(use, reader)
            if reach is _Reach.ELSEWHERE:
                statements = _name_statements(use, reader, prose=True, root=root)
            if reach is not _Reach.NOWHERE and statements:
                add(statements, use)
    # DataFrame.to_sql into a database that isn't in-process.
    for node in ast.walk(top):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "to_sql"):
            continue
        con = node.args[1] if len(node.args) > 1 else next(
            (k.value for k in node.keywords if k.arg == "con"), None
        )
        if con is not None and reader.is_local_handle(con):
            continue
        table = node.args[0] if node.args else next(
            (k.value for k in node.keywords if k.arg == "name"), None
        )
        label = table.value if isinstance(table, ast.Constant) and isinstance(table.value, str) else "a table"
        out.append(_sql_write(f"to_sql into {label}", reader=reader,
                              line=report.line if report else cell.host_line or node.lineno))
    return out


# ── File findings ─────────────────────────────────────────────────────────

class _Targets(BaseModel):
    """Where a path pattern lands: snapshot paths, a root folder whole, or a
    root the check can't place."""

    model_config = ConfigDict(frozen=True)

    paths: tuple[str, ...] = ()
    whole_project: bool = False
    whole_shared: bool = False
    unknown: bool = False

    def found(self) -> bool:
        return bool(self.paths or self.whole_project or self.whole_shared or self.unknown)


@dataclass(frozen=True)
class _Made:
    """What the pad's cells write, read for a snapshot past its cap: the path
    patterns of writes that leave a file new (`created`) and of writes that
    keep what a file held (`kept`: an append, `os.utime`, `Path.touch`),
    unknown once one of those has a path the check can't read; and the
    folders they may run in."""

    created: tuple[str, ...]
    kept: tuple[str, ...] | _Unknown
    cwds: Sequence[str] | _Unknown

    def new(self, *, snap: ProjectSnapshot, rel: str) -> bool:
        """Whether the cells made `rel` new: a write that leaves a file new
        names it, and no write that keeps what a file held can."""
        if isinstance(self.kept, _Unknown):
            return False
        return self._names(patterns=self.created, snap=snap, rel=rel) and not self._names(patterns=self.kept, snap=snap, rel=rel)

    def _names(self, *, patterns: Sequence[str], snap: ProjectSnapshot, rel: str) -> bool:
        key = snap.key(rel)
        for raw in patterns:
            pattern = raw.replace("\\", "/")
            if _TEMP_PATH in pattern:
                continue
            if os.path.isabs(pattern):
                candidates = [pattern]
            elif isinstance(self.cwds, _Unknown):
                candidates = ["**/" + _without_parents(pattern)]
            else:
                candidates = [_join(c, pattern) for c in self.cwds]
            for candidate in candidates:
                found = _relative(candidate, snap)
                if found is not None and glob_match(key, snap.key(found)):
                    return True
        return False


@dataclass(frozen=True)
class _Report:
    """Where a change made inside an earlier cell's function is reported:
    the line of the call in the checked cell, and that line's text."""

    line: int
    call: str


@dataclass(frozen=True)
class _ShellStep:
    """One delete or move a shell command runs: its operands as the command
    spells them, `_NOT_LITERAL` for one the check can't read, and the
    folders it may run in."""

    action: Literal["delete", "move"]
    operands: tuple[str, ...]
    cwds: Sequence[str] | _Unknown
    files_only: bool


def _locate(
    patterns: list[str] | _Unknown,
    snapshots: Sequence[ProjectSnapshot],
    *,
    cwds: Sequence[str] | _Unknown,
    files_only: bool = False,
    made: _Made | None = None,
    disk: _DiskReads | None = None,
) -> _Targets:
    """The snapshot entries `patterns` name. A relative pattern counts from
    each folder in `cwds`, or from under any folder of either root when the
    check can't place them. With `files_only`, for a delete that can't
    remove a folder, the folders a listing names drop out. `made` reads a
    snapshot past its cap (`ProjectSnapshot.named`), and `disk` is the
    check's reads from disk (`ProjectSnapshot.matching`)."""
    if isinstance(patterns, _Unknown):
        return _Targets(unknown=True)
    hits: list[str] = []
    whole: set[str] = set()
    unknown = False
    project = next((s for s in snapshots if s.label == "project"), None)
    for raw in patterns:
        pattern = raw.replace("\\", "/")
        if _TEMP_PATH in pattern:
            continue
        if not os.path.isabs(pattern) and isinstance(cwds, _Unknown):
            # From a folder the check can't place: under any folder of either root.
            rest = _without_parents(pattern)
            if not rest:
                unknown = True
                continue
            for snap in snapshots:
                hits += _card_paths(snap, snap.matching(pattern="**/" + rest, made=made, disk=disk), files_only=files_only)
            continue
        candidates = [pattern] if os.path.isabs(pattern) else [_join(c, pattern) for c in cwds]
        for candidate in candidates:
            if project is not None and not os.path.isabs(candidate) and (
                posixpath.normpath(candidate).startswith("../")
            ):
                # Out of the project and maybe back in, or into the shared folder.
                candidate = posixpath.join(str(project.root), candidate)
            for snap in snapshots:
                rel = _relative(candidate, snap)
                if rel is None or rel.startswith("../") and not _above(rel):
                    continue
                if rel == "" or _above(rel):
                    whole.add(snap.label)
                elif any(ch in rel for ch in "*?["):
                    hits += _card_paths(snap, snap.matching(pattern=rel, made=made, disk=disk), files_only=files_only)
                else:
                    named = snap.named(rel=rel, made=made)
                    hits += _card_paths(snap, [named] if named is not None else [], files_only=False)
    return _Targets(
        paths=tuple(dict.fromkeys(hits)),
        whole_project="project" in whole,
        whole_shared="shared" in whole,
        unknown=unknown,
    )


def _card_paths(snap: ProjectSnapshot, found: list[str], *, files_only: bool) -> list[str]:
    """Snapshot paths as a card names them, the shared folder's prefixed."""
    prefix = "" if snap.label == "project" else "shared folder: "
    return [prefix + p for p in found if not (files_only and snap.is_dir(rel=p))]


def _without_parents(pattern: str) -> str:
    """A relative pattern with its leading `..` parts dropped; "" when only
    those, or the folder itself, are left."""
    parts = [p for p in posixpath.normpath(pattern).split("/") if p not in ("", ".")]
    while parts and parts[0] == "..":
        parts.pop(0)
    return "/".join(parts)


def _above(rel: str) -> bool:
    """Whether a relative path made only of `..` parts names the root or above."""
    return all(part in ("..", "") for part in rel.split("/"))


def _relative(pattern: str, snap: ProjectSnapshot) -> str | None:
    """`pattern` relative to the snapshot's root: "" for the root itself, a
    "../" path for above it, None when it lies outside.

    An absolute pattern may start with the root as the host spelled it or as
    its realpath, and is tried again with its parent folder resolved, for a
    path spelled through a link, then resolved whole, for a link to a folder
    above the root. The spelling as written wins, so a link inside the root
    is its own entry.
    """
    if os.path.isabs(pattern):
        path = posixpath.normpath(pattern)
        head, tail = posixpath.split(path)
        resolved = posixpath.join(os.path.realpath(head), tail) if tail else path
        for candidate in dict.fromkeys((path, resolved)):
            for root in snap.roots():
                rel = _under(candidate, root, fold=snap.casefold)
                if rel is not None:
                    return rel
        full = os.path.realpath(path)
        above = any(_under(full, root, fold=snap.casefold) == ".." for root in snap.roots())
        return ".." if above else None
    if snap.label != "project":
        return None
    rel = posixpath.normpath(pattern)
    return "" if rel == "." else rel


def _under(path: str, root: str, *, fold: bool) -> str | None:
    """`path` relative to `root`, ".." when it is above it, None when apart."""
    root = root.rstrip("/") or "/"
    p, r = (path.casefold(), root.casefold()) if fold else (path, root)
    if len(p) != len(path) or len(r) != len(root):
        p, r = path, root
    if p == r:
        return ""
    if p.startswith(r.rstrip("/") + "/"):
        return posixpath.normpath(path[len(root.rstrip("/")) + 1:])
    if r.startswith(p.rstrip("/") + "/"):
        return ".."
    return None


def _target_arg(call: ast.Call, *, keywords: tuple[str, ...]) -> ast.AST | None:
    if call.args:
        return call.args[0]
    return next((k.value for k in call.keywords if k.arg in keywords), None)


def _open_mode_writes(call: ast.Call) -> bool:
    mode = call.args[1] if len(call.args) > 1 else next(
        (k.value for k in call.keywords if k.arg == "mode"), None
    )
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and any(
        c in mode.value for c in "wax"
    )


def _open_mode_writes_method(call: ast.Call) -> bool:
    mode = call.args[0] if call.args else next(
        (k.value for k in call.keywords if k.arg == "mode"), None
    )
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and any(
        c in mode.value for c in "wax"
    )


def _overwrite_target(call: ast.Call, reader: _Reader) -> ast.AST | None:
    name = reader.dotted(call.func)
    if name in ("open", "io.open") and call.args and _open_mode_writes(call):
        return call.args[0]
    if isinstance(call.func, ast.Attribute):
        method = call.func.attr
        if method in ("write_text", "write_bytes"):
            return call.func.value
        if method == "open" and _open_mode_writes_method(call):
            return call.func.value
        if method in _WRITER_METHODS and call.args:
            return call.args[0]
    return None


def _move_destination(call: ast.Call) -> ast.AST | None:
    return call.args[1] if len(call.args) > 1 else next(
        (k.value for k in call.keywords if k.arg in ("dst", "dest", "target")), None
    )


def _written_target(call: ast.Call, reader: _Reader, cell: _Cell) -> ast.AST | None:
    """The path `call` writes: a file it opens for writing, or the
    destination of a rename, move or copy."""
    name = reader.dotted(call.func)
    if name in _MOVE_FUNCS or name in _COPY_FUNCS:
        return _move_destination(call)
    if isinstance(call.func, ast.Attribute) and call.func.attr in ("rename", "replace") and (
        _is_path_rename(call, reader, cell)
    ):
        return call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == "target"), None)
    return _overwrite_target(call, reader)


def _keeps_contents(*, call: ast.Call, reader: _Reader) -> bool:
    """Whether a write keeps what its file held: an open to append or to
    update in place, or a writer given `mode="a"`."""
    mode: ast.AST | None = None
    if reader.dotted(call.func) in ("open", "io.open"):
        mode = call.args[1] if len(call.args) > 1 else None
    elif isinstance(call.func, ast.Attribute) and call.func.attr == "open":
        mode = call.args[0] if call.args else None
    mode = next((k.value for k in call.keywords if k.arg == "mode"), mode)
    text = mode.value if isinstance(mode, ast.Constant) and isinstance(mode.value, str) else ""
    return "a" in text or "+" in text and not set(text) & set("wx")


def _made(*, reader: _Reader) -> _Made:
    """What the pad's cells write, every cell's writes read where they are
    written: the files a write leaves new, and those an append, `os.utime`
    or `Path.touch` keeps."""
    created: list[str] = []
    kept: list[str] | _Unknown = []
    for cell in reader.cells:
        for call in cell.calls:
            name = reader.dotted(call.func)
            method = call.func.attr if isinstance(call.func, ast.Attribute) else ""
            if name == "os.utime":
                target, keeps = _target_arg(call, keywords=("path",)), True
            elif method == "touch":
                target, keeps = call.func.value, True
            else:
                target = _written_target(call, reader, cell)
                keeps = target is not None and _keeps_contents(call=call, reader=reader)
            if target is None:
                continue
            got = reader.paths(target, cell=cell)
            if not keeps:
                created += [] if isinstance(got, _Unknown) else got
            elif isinstance(got, _Unknown) or isinstance(kept, _Unknown):
                kept = _UNKNOWN
            else:
                kept += got
    return _Made(created=tuple(created), kept=kept if isinstance(kept, _Unknown) else tuple(kept), cwds=reader.cwds)


def _is_path_rename(call: ast.Call, reader: _Reader, cell: _Cell) -> bool:
    """`.rename(x)` and `.replace(x)` on a pathlib path, not a DataFrame's or a str's."""
    receiver = call.func.value
    if isinstance(receiver, (ast.Constant, ast.JoinedStr)):
        return False
    one_arg = len(call.args) == 1 and not call.keywords
    target_kw = not call.args and any(k.arg == "target" for k in call.keywords)
    if not (one_arg or target_kw):
        return False
    if reader.is_path(receiver, cell=cell):
        return True
    dest = call.args[0] if call.args else next(k.value for k in call.keywords if k.arg == "target")
    if reader.is_path(dest, cell=cell):
        return True
    return (
        isinstance(dest, ast.Constant) and isinstance(dest.value, str)
        and bool(re.search(r"\.[A-Za-z0-9]{1,5}$", dest.value))
    )


def _runs_whenever(node: ast.AST, stmt: ast.stmt, cell: _Cell) -> bool:
    """Whether `node` runs every time `stmt` does: not under a branch, loop,
    try, function, comprehension or the lazy side of a condition."""
    child, parent = node, cell.parents.get(node)
    while child is not stmt and parent is not None:
        if isinstance(child, ast.stmt) and not isinstance(parent, (ast.With, ast.AsyncWith)):
            return False
        if isinstance(parent, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return False
        if isinstance(parent, ast.IfExp) and child is not parent.test:
            return False
        if isinstance(parent, ast.BoolOp) and child is not parent.values[0]:
            return False
        child, parent = parent, cell.parents.get(parent)
    return child is stmt


def _may_leave(stmt: ast.stmt, *, reader: _Reader) -> bool:
    """Whether `stmt` can return, raise, break, continue, fail an assert or
    end the process past what follows it."""
    return any(
        isinstance(n, (ast.Return, ast.Raise, ast.Break, ast.Continue, ast.Assert))
        or isinstance(n, ast.Call) and reader.dotted(n.func) in _EXITS
        for n in ast.walk(stmt)
    )


class _FileFinder:
    """Collects the file changes code in one cell makes. With a report, the
    code is an earlier cell's function, read for one call in the checked
    cell, and each change is reported at that call."""

    def __init__(
        self, reader: _Reader, snapshots: Sequence[ProjectSnapshot], *, cell: _Cell, report: _Report | None,
    ) -> None:
        self.reader = reader
        self.snapshots = snapshots
        self.cell = cell
        self.report = report
        self.out: list[FileChange] = []

    def paths(self, node: ast.AST) -> list[str] | _Unknown:
        return self.reader.paths(node, cell=self.cell)

    def locate(
        self, got: list[str] | _Unknown, *, cwds: Sequence[str] | _Unknown | None = None, files_only: bool = False,
    ) -> _Targets:
        return _locate(got, self.snapshots, cwds=self.reader.cwds if cwds is None else cwds, files_only=files_only,
                       made=self.reader.made, disk=self.reader.disk)

    def add(self, action: Literal["delete", "rename", "move", "overwrite"], targets: _Targets, node: ast.AST) -> None:
        if not targets.found():
            return
        if self.report is not None:
            line, call = self.report.line, self.report.call
        else:
            line, call = self.cell.host_line or getattr(node, "lineno", 0), scrub_output(self.cell.source(node))
        self.out.append(FileChange(
            action=action, paths=targets.paths, whole_project=targets.whole_project,
            whole_shared=targets.whole_shared, call=call, line=line,
        ))

    def walk(self, root: ast.AST) -> None:
        reader, cell = self.reader, self.cell
        for node in ast.walk(root):
            if isinstance(node, ast.Attribute) and reader.dotted(node) in (*_DELETE_FUNCS, *_MOVE_FUNCS):
                parent = cell.parents.get(node)
                if not (isinstance(parent, ast.Call) and parent.func is node):
                    # Passed as a value, as in map(os.remove, ...): the target is
                    # picked when it runs.
                    name = reader.dotted(node)
                    self.add("delete" if name in _DELETE_FUNCS else _MOVE_FUNCS[name], _Targets(unknown=True), node)
                continue
            if self.lookup(node=node):
                continue
            if not isinstance(node, ast.Call):
                continue
            partial = reader.partial_of(node=node.func) if isinstance(node.func, (ast.Name, ast.Call)) else None
            if partial is not None:
                # A call through functools.partial, read as the call it makes.
                node = _partial_call(partial=partial, call=node)
            name = reader.dotted(node.func)
            method = node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name in _DELETE_FUNCS:
                target = _target_arg(node, keywords=("path", "self"))
                if target is not None:
                    self.delete(target, node, files_only=name in _FILE_DELETES)
            elif name in _MOVE_FUNCS:
                source = _target_arg(node, keywords=("src", "self"))
                if source is not None:
                    self.add(_MOVE_FUNCS[name], self.locate(self.paths(source)), node)
                self.note_overwrite(node)
            elif name in _COPY_FUNCS:
                self.note_overwrite(node)
            elif name in _SHELL_FUNCS or name in _SPREAD_ARGV_FUNCS or name in _LIST_ARGV_FUNCS:
                self.shell(node)
            elif method == "unlink" and not node.args:
                if not isinstance(node.func.value, (ast.Constant, ast.JoinedStr)):
                    self.delete(node.func.value, node, files_only=True)
            elif method == "rmdir" and reader.is_path(node.func.value, cell=cell):
                self.add("delete", self.locate(self.paths(node.func.value)), node)
            elif method in ("rename", "replace") and _is_path_rename(node, reader, cell):
                self.add("rename", self.locate(self.paths(node.func.value)), node)
                dest = node.args[0] if node.args else next(
                    (k.value for k in node.keywords if k.arg == "target"), None
                )
                if dest is not None:
                    self.note_overwrite_target(dest, node)
            else:
                target = _overwrite_target(node, reader)
                if target is not None:
                    self.note_overwrite_target(target, node)

    def lookup(self, *, node: ast.AST) -> bool:
        """Whether `node` is a lookup `_plain_lookup` can't read, on a module,
        class or path that holds a delete, a move or a shell call:
        `getattr(m, name)` with a name that isn't a literal, `vars(m)[name]`
        or `m.__dict__[name]`. Each text the name can hold is read as the call
        it reaches; a name the check can't read is a delete picked when it
        runs."""
        reader, cell = self.reader, self.cell
        receiver: ast.AST | None = None
        key: ast.AST | None = None
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
                and len(node.args) in (2, 3)):
            receiver, key = node.args[0], node.args[1]
        elif isinstance(node, ast.Subscript):
            inner = node.value
            if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "vars"
                    and len(inner.args) == 1):
                receiver, key = inner.args[0], node.slice
            elif isinstance(inner, ast.Attribute) and inner.attr == "__dict__":
                receiver, key = inner.value, node.slice
        if receiver is None or key is None:
            return False
        module = reader.dotted(receiver)
        on_path = module not in _LOOKUP_MODULES and reader.is_path(receiver, cell=cell)
        if module not in _LOOKUP_MODULES and not on_path:
            return False
        parent = cell.parents.get(node)
        call = parent if isinstance(parent, ast.Call) and parent.func is node else None
        names = reader.paths(key, cell=cell)
        if isinstance(names, _Unknown) or any(ch in n for n in names for ch in "*?["):
            self.add("delete", _Targets(unknown=True), node)
            return True
        for name in names:
            full = f"pathlib.Path.{name}" if on_path else f"{module}.{name}"
            if call is None:
                if full in _DELETE_FUNCS or full in _MOVE_FUNCS or full in _SHELL_FUNCS:
                    self.add("delete" if full not in _MOVE_FUNCS else _MOVE_FUNCS[full], _Targets(unknown=True), node)
            elif full in _DELETE_FUNCS:
                target = receiver if on_path else _target_arg(call, keywords=("path", "self"))
                if target is not None:
                    self.delete(target, call, files_only=full in _FILE_DELETES)
            elif full in _MOVE_FUNCS:
                source = receiver if on_path else _target_arg(call, keywords=("src", "self"))
                if source is not None:
                    self.add(_MOVE_FUNCS[full], self.locate(self.paths(source)), call)
            elif full in _SHELL_FUNCS:
                self.shell(call)
        return True

    def delete(self, target: ast.AST, call: ast.Call, *, files_only: bool) -> None:
        """A delete; an overwrite when a write to the same path follows it."""
        got = self.paths(target)
        rewritten = not isinstance(got, _Unknown) and bool(got) and set(got) <= self.written_after(call)
        self.add("overwrite" if rewritten else "delete", self.locate(got, files_only=files_only), call)

    def written_after(self, call: ast.Call) -> set[str]:
        """The paths written after `call` on every path through the code: by a
        later statement of its block, or of a block around it up to its
        function, that writes whenever it runs. A statement that can return,
        raise, break, continue, fail an assert or exit ends the search."""
        cell = self.cell
        node: ast.AST | None = call
        while node is not None and not isinstance(node, ast.stmt):
            if isinstance(node, ast.Lambda):
                return set()
            node = cell.parents.get(node)
        written: set[str] = set()
        while node is not None and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parent = cell.parents.get(node)
            block = next((b for b in (getattr(parent, f, None) for f in ("body", "orelse", "finalbody"))
                          if isinstance(b, list) and node in b), [])
            later_ones = block[block.index(node) + 1:] if node in block else []
            for later in later_ones:
                if _may_leave(later, reader=self.reader):
                    return written
                for inner in ast.walk(later):
                    target = _written_target(inner, self.reader, cell) if isinstance(inner, ast.Call) else None
                    if target is None or not _runs_whenever(inner, later, cell):
                        continue
                    got = self.paths(target)
                    if not isinstance(got, _Unknown):
                        written.update(got)
            node = parent
        return written

    def note_overwrite(self, call: ast.Call) -> None:
        """A move or copy onto an existing file, logged as an overwrite. A
        folder destination takes the source's name."""
        dest = _move_destination(call)
        if dest is None:
            return
        got = self.paths(dest)
        if isinstance(got, _Unknown):
            return
        source = call.args[0] if call.args else None
        names = self.paths(source) if source is not None else _UNKNOWN
        expanded = []
        for d in got:
            for snap in self.snapshots:
                rel = _relative(d, snap)
                if rel is not None and snap.is_dir(rel=rel) and not isinstance(names, _Unknown):
                    expanded += [_join(d, posixpath.basename(n)) for n in names]
                    break
            else:
                expanded.append(d)
        self.record_overwrite(expanded, call)

    def note_overwrite_target(self, node: ast.AST, call: ast.Call) -> None:
        got = self.paths(node)
        if not isinstance(got, _Unknown):
            self.record_overwrite(got, call)

    def record_overwrite(self, patterns: list[str], call: ast.Call) -> None:
        hits = []
        for pattern in patterns:
            for snap in self.snapshots:
                rel = _relative(pattern, snap)
                if rel and not rel.startswith("..") and snap.is_file(rel=rel):
                    hits += _card_paths(snap, [snap.recorded(rel=rel) or rel], files_only=False)
        if hits:
            self.add("overwrite", _Targets(paths=tuple(dict.fromkeys(hits))), call)

    def shell(self, call: ast.Call) -> None:
        """Each rm, mv and the like a shell command runs. One whose operand
        the check can't read, or that reads them from its input, is held as
        picked when it runs."""
        cwd = next((k.value for k in call.keywords if k.arg == "cwd"), None)
        cwds = self.reader.cwds
        if cwd is not None and not isinstance(cwds, _Unknown):
            got = self.paths(cwd)
            cwds = _UNKNOWN if isinstance(got, _Unknown) else [
                posixpath.normpath(_join(c, p)) for c in cwds for p in got
            ]
        for step in _shell_steps(call, self.reader, cwds=cwds):
            if _NOT_LITERAL in step.operands:
                self.add(step.action, _Targets(unknown=True), call)
            else:
                self.add(step.action, self.locate(list(step.operands), cwds=step.cwds,
                                                  files_only=step.files_only), call)


def _file_findings(
    reader: _Reader,
    cell: _Cell,
    snapshots: Sequence[ProjectSnapshot],
    *,
    root: ast.AST | None = None,
    report: _Report | None = None,
) -> list[FileChange]:
    """The file changes `cell` makes; with `root`, the ones a function of
    that cell makes, reported at `report`, the call in the checked cell."""
    if not snapshots or all(s.is_empty() for s in snapshots):
        return []
    finder = _FileFinder(reader, snapshots, cell=cell, report=report)
    finder.walk(cell.tree if root is None else root)
    return finder.out


# ── Shell commands ────────────────────────────────────────────────────────

# A redirection such as `>`, `2>>` or `<`, alone or glued to its file.
_REDIRECT = re.compile(r"^(?:\d*|&)(?:>>?|<<?<?)(?:&\d*)?")
# Shell flags that take the next word as their value.
_VALUE_FLAGS = frozenset({"-u", "-g", "-n", "-I", "-L", "-P", "-d", "-E", "-s", "-a", "-C"})


def _shell_steps(call: ast.Call, reader: _Reader, *, cwds: Sequence[str] | _Unknown) -> list[_ShellStep]:
    """The deletes and moves the command `call` runs: a list of words or a
    script, written in the call, held in a name, or built with an f-string,
    `+`, `%` or `.format`, each part picked at run time a placeholder."""
    name = reader.dotted(call.func)
    if name in _SPREAD_ARGV_FUNCS:
        words = call.args[_SPREAD_ARGV_FUNCS[name]:]
        # The e forms end with the environment, not a word.
        words = words[:-1] if name.endswith("e") else words
        commands: list[str | list[str]] = [[_command_word(w, reader) for w in words]]
    else:
        index = _LIST_ARGV_FUNCS.get(name, 0)
        arg = call.args[index] if len(call.args) > index else (
            None if index else _target_arg(call, keywords=("args", "cmd", "command"))
        )
        commands = [] if arg is None else _commands(arg, reader)
    given = next((k.value for k in call.keywords if k.arg == "input"), None)
    stdin = None if given is None else _sql_text(given, reader)
    out: list[_ShellStep] = []
    for command in commands:
        if isinstance(command, str):
            out += _script_steps(command, cwds=cwds, depth=0)
        else:
            out += _words_steps(command, cwds=cwds, depth=0, stdin=stdin)
    return out


def _commands(node: ast.AST, reader: _Reader, *, depth: int = 0) -> list[str | list[str]]:
    """Each command `node` can hold: a list of words, or a script's text."""
    if depth > 6:
        return []
    if isinstance(node, (ast.List, ast.Tuple)):
        return [[_command_word(e, reader) for e in node.elts]]
    words_made = _word_lists(node=node, reader=reader, depth=depth)
    if words_made is not None:
        return list(words_made)
    if isinstance(node, ast.Name):
        arg = reader.argument_of(node)
        if arg is not None:
            return _commands(arg.node, reader, depth=depth + 1)
        out: list[str | list[str]] = []
        for b in reader.bindings_of(node.id, node=node):
            if b.how == "param":
                # What each call of the function passes for it.
                for site in reader.call_sites(b.value, node.id):
                    out += [] if site is None else _commands(site.node, reader, depth=depth + 1)
            elif b.how == "value" and (
                isinstance(b.value, (ast.List, ast.Tuple)) or _word_lists(node=b.value, reader=reader, depth=depth) is not None
            ):
                grown = reader.list_growth(name=node.id, binding=b)
                for words in _commands(b.value, reader, depth=depth + 1):
                    if isinstance(words, list) and (isinstance(grown, _Unknown) or grown):
                        # Words the cell adds to the list.
                        words = [*words, _NOT_LITERAL]
                    out.append(words)
            else:
                out += _held_texts(node.id, b, reader, depth=0)
        return out
    text = _sql_text(node, reader)
    return [] if text is None else [text]


def _word_lists(*, node: ast.AST, reader: _Reader, depth: int) -> list[list[str]] | None:
    """The word lists an expression that builds one can give a command:
    `text.split()`, `shlex.split(text)`, or lists added with `+`, a part the
    check can't read one word it can't read. None for any other expression."""
    shell_split = isinstance(node, ast.Call) and bool(node.args) and reader.dotted(node.func) == "shlex.split"
    if shell_split or isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "split":
        source = node.args[0] if shell_split else node.func.value
        sep: str | None = None
        if node.args and not shell_split:
            given = node.args[0]
            literal = isinstance(given, ast.Constant) and isinstance(given.value, str) and given.value
            sep = given.value if literal else _NOT_LITERAL
        texts = [c for c in _commands(source, reader, depth=depth + 1) if isinstance(c, str)]
        if not texts or sep == _NOT_LITERAL:
            return [[_NOT_LITERAL]]
        out: list[list[str]] = []
        for text in texts:
            try:
                out.append(shlex.split(text) if shell_split else text.split(sep))
            except ValueError:
                out.append(text.split())
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add) and (
        reader.is_sequence(node.left, cell=reader.cell_of(node))
        or reader.is_sequence(node.right, cell=reader.cell_of(node))
    ):
        sides = []
        for side in (node.left, node.right):
            lists = [c for c in _commands(side, reader, depth=depth + 1) if isinstance(c, list)]
            sides.append(lists or [[_NOT_LITERAL]])
        return [left + right for left in sides[0] for right in sides[1]]
    return None


def _command_word(node: ast.AST, reader: _Reader) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    text = _sql_text(node, reader)
    return _NOT_LITERAL if text is None or _HOLE in text else text


def _script_steps(text: str, *, cwds: Sequence[str] | _Unknown, depth: int) -> list[_ShellStep]:
    """The deletes and moves a shell script runs, command by command, each
    after the folders its `cd`s reach."""
    try:
        lexer = shlex.shlex(text.replace("\n", " ; "), posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        tokens = text.replace("\n", " ; ").split()
    commands: list[list[str]] = [[]]
    for token in tokens:
        if token in _SHELL_SEPARATORS or token and set(token) <= set(";&|()"):
            commands.append([])
        else:
            commands[-1].append(token)
    out: list[_ShellStep] = []
    for words in commands:
        bare, _ = _command_words(words)
        if bare and bare[0] in ("cd", "pushd"):
            cwds = _shell_cd(cwds, bare[1] if len(bare) > 1 else "~")
        else:
            out += _words_steps(words, cwds=cwds, depth=depth)
    return out


def _shell_cd(cwds: Sequence[str] | _Unknown, target: str) -> Sequence[str] | _Unknown:
    """`cwds` with the folder a `cd target` reaches from each added; a
    folder the check can't place makes them unknown."""
    if isinstance(cwds, _Unknown) or _unreadable_word(target) or target == "-":
        return _UNKNOWN
    if target.startswith("~"):
        return [*cwds, os.path.expanduser(target)]
    out = list(dict.fromkeys([*cwds, *(posixpath.normpath(_join(c, target)) for c in cwds)]))
    return out if len(out) <= _CWD_LIMIT else _UNKNOWN


def _unreadable_word(word: str) -> bool:
    """A word whose value the shell or Python picks at run time: a
    placeholder, a variable, a command substitution, or a `.format` or `%`
    field left unfilled."""
    return (
        _HOLE in word or _NOT_LITERAL in word or "$" in word or "`" in word
        or bool(re.search(r"\{[^{}]*\}|%(?:\(\w+\))?[sdr]", word))
    )


def _command_words(words: list[str]) -> tuple[list[str], bool]:
    """A command's words from the verb on, with redirections, `VAR=value`
    settings and runners such as sudo, env and xargs dropped; and whether
    xargs feeds it operands from its input."""
    out: list[str] = []
    skip = False
    for word in words:
        if skip:
            skip = False
            continue
        redirect = _REDIRECT.match(word)
        if redirect and redirect.end() > 0:
            skip = redirect.end() == len(word)
            continue
        out.append(word)
    from_input = False
    while out:
        head = out[0].rsplit("/", 1)[-1]
        if re.match(r"^[A-Za-z_]\w*=", out[0]) or out[0] in _SHELL_KEYWORDS:
            out.pop(0)
        elif head in _SHELL_PREFIXES:
            from_input = from_input or head == "xargs"
            out.pop(0)
            while out and out[0].startswith("-"):
                flag = out.pop(0)
                if (flag in _VALUE_FLAGS or flag in _PREFIX_VALUE_FLAGS.get(head, ())) and out:
                    out.pop(0)
            del out[:_PREFIX_OPERANDS.get(head, 0)]
        else:
            break
    return out, from_input


def _words_steps(
    words: list[str], *, cwds: Sequence[str] | _Unknown, depth: int, stdin: str | None = None,
) -> list[_ShellStep]:
    """The delete or move one command runs. `stdin` is the text the call
    gives the command as its input, when the check can read it."""
    words, from_input = _command_words(words)
    if not words:
        return []
    verb, args = words[0].rsplit("/", 1)[-1], words[1:]
    picked = [_ShellStep(action="delete", operands=(_NOT_LITERAL,), cwds=cwds, files_only=False)]
    if "$" in verb or "`" in verb:
        # A command the shell picks when it runs, as in $(which rm) or $RM.
        return picked
    if verb in _SHELLS:
        script = next((args[i + 1] for i, a in enumerate(args[:-1])
                       if a.startswith("-") and not a.startswith("--") and "c" in a), None)
        if script is None and ("-s" in args or not [a for a in args if not a.startswith("-")]):
            # The script comes from the shell's input, piped in or passed as `input=`.
            script = stdin
            if script is None:
                return picked
        return [] if script is None or depth >= 4 else _script_steps(script, cwds=cwds, depth=depth + 1)
    if verb == "git":
        while args and args[0].startswith("-"):
            flag = args.pop(0)
            if flag == "-C" and args:
                cwds = _shell_cd(cwds, args.pop(0))
            elif flag in ("-c", "--git-dir", "--work-tree") and args:
                args.pop(0)
        if not args:
            return []
        verb, args = args[0], args[1:]
        if verb == "clean":
            forced = any(a == "--force" or a.startswith("-") and not a.startswith("--") and "f" in a for a in args)
            if not forced:
                return []
            return [_ShellStep(action="delete", operands=(_NOT_LITERAL,), cwds=cwds, files_only=False)]
    if verb == "find":
        runs = [args[i + 1].rsplit("/", 1)[-1] for i, a in enumerate(args[:-1])
                if a in ("-exec", "-execdir", "-ok", "-okdir")]
        if "mv" in runs:
            return [_ShellStep(action="move", operands=(_NOT_LITERAL,), cwds=cwds, files_only=False)]
        if "-delete" in args or any(r in _SHELL_DELETE for r in runs):
            return [_ShellStep(action="delete", operands=(_NOT_LITERAL,), cwds=cwds, files_only=False)]
        return []
    flags, operands, target_dir = _split_flags(args, takes_target=verb == "mv")
    if from_input:
        operands.append(_NOT_LITERAL)
    operands = [_NOT_LITERAL if _unreadable_word(o) else o for o in operands]
    if verb in _SHELL_DELETE and operands:
        recursive = any(f in ("--recursive", "--dir") or not f.startswith("--") and bool(set(f) & set("rRd"))
                        for f in flags)
        files_only = verb != "rmdir" and not recursive
        return [_ShellStep(action="delete", operands=tuple(operands), cwds=cwds, files_only=files_only)]
    if verb == "mv" or verb == "rsync" and "--remove-source-files" in flags:
        sources = operands if target_dir is not None else operands[:-1]
        return [_ShellStep(action="move", operands=tuple(sources), cwds=cwds, files_only=False)] if sources else []
    return []


def _split_flags(args: list[str], *, takes_target: bool) -> tuple[list[str], list[str], str | None]:
    """A command's flags, its operands, and, for mv (`takes_target`), the
    folder `-t` or `--target-directory` names. Words after `--` are all
    operands."""
    flags: list[str] = []
    operands: list[str] = []
    target_dir: str | None = None
    words = iter(args)
    for word in words:
        if word == "--":
            operands += list(words)
            break
        if takes_target and word in ("-t", "--target-directory", "-S", "--suffix"):
            value = next(words, None)
            target_dir = value if word in ("-t", "--target-directory") else target_dir
        elif takes_target and word.startswith("--target-directory="):
            target_dir = word.split("=", 1)[1]
        elif takes_target and re.match(r"-[^-]*[tS]", word):
            # A cluster such as -tDIR, -vt DIR or -fS.bak: the first of t and S
            # takes the rest of the word, or the next word.
            at = next(i for i, ch in enumerate(word) if i and ch in "tS")
            value = word[at + 1:] or next(words, None)
            target_dir = value if word[at] == "t" else target_dir
            flags.append(word[:at])
        elif word.startswith("-") and word != "-":
            flags.append(word)
        else:
            operands.append(word)
    return flags, operands, target_dir


# ── Entry point ───────────────────────────────────────────────────────────

def _plain_lookup(call: ast.Call) -> ast.expr:
    """`getattr(x, "name")` as `x.name`, also with a constant default as in
    `getattr(x, "name", None)`, and `__import__("os")` or
    `importlib.import_module("os")` as `os`, so a call made through one is
    read like the plain spelling; any other call as it is."""
    func, args = call.func, call.args
    named = bool(args) and isinstance(args[-1], ast.Constant) and isinstance(args[-1].value, str)
    attr = len(args) in (2, 3) and isinstance(args[1], ast.Constant) and isinstance(args[1].value, str)
    if (isinstance(func, ast.Name) and func.id == "getattr" and attr and not call.keywords
            and (len(args) == 2 or isinstance(args[2], ast.Constant))):
        return ast.copy_location(ast.Attribute(value=args[0], attr=args[1].value, ctx=ast.Load()), call)
    importer = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
    if importer in ("__import__", "import_module") and len(args) == 1 and named and args[0].value:
        parts = args[0].value.split(".")
        parts = parts[:1] if importer == "__import__" else parts
        out: ast.expr = ast.copy_location(ast.Name(id=parts[0], ctx=ast.Load()), call)
        for part in parts[1:]:
            out = ast.copy_location(ast.Attribute(value=out, attr=part, ctx=ast.Load()), call)
        return out
    return call


def _read_lookups_plainly(tree: ast.Module) -> ast.Module:
    """`tree` with each literal lookup `_plain_lookup` reads swapped in place.
    Deepest nodes first, without recursion, so a nested lookup is read
    before the one around it and a long expression can't exhaust the stack."""
    for node in reversed(list(ast.walk(tree))):
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.Call):
                setattr(node, field, _plain_lookup(value))
            elif isinstance(value, list):
                value[:] = [_plain_lookup(v) if isinstance(v, ast.Call) else v for v in value]
    return tree


def _parse(code: str) -> ast.Module:
    """The cell's tree, from the same healed text the runner compiles, with
    literal `getattr` and import lookups read as plain names.

    Quiet: the agent's own invalid escape sequences would otherwise print a
    SyntaxWarning in the host's log for every check; the runner reports them
    where they belong, in the cell's output.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return _read_lookups_plainly(ast.parse(heal_surrogate_source(code)))


def _read_cell(*, code: str, index: int) -> _Cell | None:
    """`code` read as the cell at `index` among the parsed cells; None when it doesn't parse."""
    try:
        return _Cell(_parse(code), code, index)
    except (SyntaxError, ValueError):
        return None


@dataclass(eq=False)
class _CachedCell:
    """One earlier cell `_EarlierCells` keeps, and when a check last read it."""

    cell: _Cell | None
    read_at: float


class _EarlierCells:
    """Earlier cells, each parsed once per process instead of once per check.

    Every check reads all of the conversation's earlier cells, hundreds in a
    long one, and they don't change from one check to the next. A cell is
    keyed by its code and its place among the parsed cells, since what it
    records holds both. A cell that would take the cached code past `budget`
    characters makes room only from cells no check has read for `idle`
    seconds, those read least recently first, and otherwise isn't kept. A
    check reads the cells in order, so dropping the oldest for the newest
    would leave the next check nothing to find. Checks run in worker
    threads, so a lock guards the table.
    """

    def __init__(self, *, budget: int, idle: float = 600.0) -> None:
        self._budget = budget
        self._idle = idle
        self._size = 0
        self._cells: OrderedDict[tuple[str, int], _CachedCell] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, *, code: str, index: int) -> _Cell | None:
        key = (code, index)
        with self._lock:
            cached = self._cells.get(key)
            if cached is not None:
                cached.read_at = time.monotonic()
                self._cells.move_to_end(key)
                return cached.cell
        cell = _read_cell(code=code, index=index)
        if len(code) > self._budget:
            return cell
        with self._lock:
            now = time.monotonic()
            while self._cells and self._size + len(code) > self._budget:
                (old, _), oldest = next(iter(self._cells.items()))
                if now - oldest.read_at < self._idle:
                    break
                self._cells.popitem(last=False)
                self._size -= len(old)
            if key not in self._cells and self._size + len(code) <= self._budget:
                self._cells[key] = _CachedCell(cell=cell, read_at=now)
                self._size += len(code)
        return cell


# A parsed cell takes about 80 bytes per character of its code, so this holds
# about 80 MB: some 1,400 cells of 700 characters, a long conversation's worth.
_EARLIER_CELLS = _EarlierCells(budget=1_000_000)


def _exec_cells(host: _Cell, *, index: int) -> list[_Cell]:
    """Code `host` passes to exec(), eval() or compile() as a string literal,
    each read as one more cell."""
    out: list[_Cell] = []
    pending = [host]
    while pending and len(out) < _EXEC_LIMIT:
        cell = pending.pop()
        for node in ast.walk(cell.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in _EXEC_CALLS and node.args):
                continue
            source = node.args[0]
            if not (isinstance(source, ast.Constant) and isinstance(source.value, (str, bytes))):
                continue
            text = source.value if isinstance(source.value, str) else source.value.decode("utf-8", errors="replace")
            try:
                tree = _parse(text)
            except (SyntaxError, ValueError):
                continue
            ran = _Cell(tree, text, index + len(out), host_line=cell.host_line or node.lineno)
            out.append(ran)
            pending.append(ran)
            if len(out) >= _EXEC_LIMIT:
                break
    return out


def _earlier_function_uses(reader: _Reader, root: ast.AST, cell: _Cell) -> Iterator[_Callee]:
    """Each place in `root` that calls a function or lambda an earlier cell
    defined, or passes one on as a value: by its name anywhere, or through
    whatever else `_Reader.callees` reads where it is called or passed to a
    call."""
    for node in ast.walk(root):
        if not (isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)) and isinstance(node.ctx, ast.Load)):
            continue
        parent = cell.parents.get(node)
        call = parent if isinstance(parent, ast.Call) and parent.func is node else None
        if isinstance(node, ast.Name) and node.id in reader.function_names:
            func = reader.earlier_function(node)
            if func is not None:
                yield _Callee(func=func, call=call, use=node)
                continue
        if call is not None or isinstance(parent, ast.Call) and node in parent.args:
            yield from reader.callees(node=node, call=call, use=node)


def find_cell_writes(
    *,
    code: str,
    earlier_code: Sequence[str],
    snapshots: Sequence[ProjectSnapshot],
) -> CellWrites:
    """What `code` would change, read against the conversation's earlier cells.

    A function or lambda an earlier cell defined is read again at each place
    this cell calls it or passes it on, with its parameters read from that
    call, and what it would change is reported at that place.

    A cell that doesn't parse returns nothing to hold: the runner compiles it
    the same way and refuses it before any of it runs.
    """
    try:
        tree = _parse(code)
    except (SyntaxError, ValueError):
        return CellWrites()
    cells: list[_Cell] = []
    for text in earlier_code:
        earlier = _EARLIER_CELLS.get(code=text, index=len(cells))
        if earlier is not None:
            cells.append(earlier)
    current = _Cell(tree, code, len(cells))
    cells.append(current)
    ran = _exec_cells(current, index=len(cells))
    cells += ran
    reader = _Reader(cells, current=current)
    if any(not s.complete for s in snapshots):
        reader.made = _made(reader=reader)
    sql: list[SqlWrite] = []
    files: list[FileChange] = []

    def read_function(callee: _Callee, *, report: _Report, chain: tuple[int, ...]) -> None:
        func = callee.func
        if id(func) in chain or len(chain) >= _CALL_DEPTH:
            return
        home = reader.cell_of(func)
        with reader.calling(func, callee.call, bound=callee.bound):
            sql.extend(_sql_findings(reader, home, root=func, report=report))
            files.extend(_file_findings(reader, home, snapshots, root=func, report=report))
            for inner in _earlier_function_uses(reader, func, home):
                read_function(inner, report=report, chain=(*chain, id(func)))

    for cell in (current, *ran):
        sql += _sql_findings(reader, cell)
        files += _file_findings(reader, cell, snapshots)
        for callee in _earlier_function_uses(reader, cell.tree, cell):
            use = callee.use
            report = _Report(line=cell.host_line or use.lineno, call=scrub_output(cell.source(use)))
            read_function(callee, report=report, chain=())
    return CellWrites(sql=tuple(sorted(sql, key=lambda w: w.line)), files=tuple(files))
