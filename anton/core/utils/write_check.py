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
  neither does one run on an in-process DuckDB or SQLite handle: one that
  opens a file the turn began with counts as that file's overwrite, below.
  Any other handle, known or not, counts. ``to_sql`` into a database that isn't
  in-process counts too. A statement hidden in a base64, base32, base16, hex
  or rot13 literal that the cell decodes, or read from a literal-path SQL
  file that exists (up to 1 MB), is decoded or read now and checked too.
  CALL, EXEC and EXECUTE (``?`` and ``:1`` placeholders and ODBC's
  ``{CALL p(?)}`` included), an Oracle block that calls a procedure
  (``BEGIN p; END;``), a SELECT of a side-effecting function (``setval``,
  ``pg_terminate_backend``, ``dblink_exec``, ``GET_LOCK`` and friends), a
  SELECT ... INTO OUTFILE and a COPY ... TO a file all count.
- **Data changes with no SQL text to quote**: a ``cursor.callproc`` or
  ``cursor.callfunc``, called or handed on, a
  SQLAlchemy Core or ORM write (``insert``/``update``/``delete`` constructs
  passed to ``execute``, ``Table.drop``/``create`` and ``MetaData``'s
  ``drop_all``/``create_all`` with an engine, ``session.query(...).delete()``,
  and ``session.add``/``merge`` and the like with a commit or flush in the
  cell), a non-SQL store's own method (pymongo, boto3 DynamoDB, BigQuery's
  table, dataset, insert and load calls, a Redis client, an Elasticsearch
  client), or a SQL file past the 1 MB cap, held unchecked. The card quotes
  the line. A routine bound in an earlier cell is held where it runs or is
  handed on.
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
- **Overwrites** of a file that existed when the turn began: ``open`` in a
  ``w``, ``w+`` or ``r+`` mode, ``Path.write_text`` and ``write_bytes``, a
  pandas ``to_csv``, ``to_excel``, ``to_parquet``, ``to_json``,
  ``to_feather``, ``to_pickle`` or ``to_html``, a ``pd.ExcelWriter``, a chart's
  ``savefig``, any ``.save(path)`` (openpyxl, docx, pptx), a copy, rename,
  replace or move onto it, a delete that a write to the same path follows on
  every path through the code, and a write statement or ``to_sql`` on an
  in-process SQLite or DuckDB handle that opens it, which the card quotes
  with the file, each statement whole, as it quotes a SQL write. An append,
  and an overwrite of a file the agent owns (one under ``.anton/artifacts``,
  ``.anton/skill_drafts`` or ``.anton/output``, or one an earlier cell in the
  conversation wrote, unless that cell stopped on an error or picked the
  name when it ran, as a dated export does), are reported but not held. ``open`` in an ``x`` mode
  can't replace a file, so it isn't one.
- **HTTP writes** through requests, httpx (their module functions, a
  Session's or Client's methods, ``.request(method, ...)``) and urllib: every
  PUT, PATCH and DELETE, a request whose method the code picks when it runs,
  and any request whose method and URL match an endpoint the operator lists
  (``HttpWriteRule``). A plain GET or POST isn't one, since many business APIs
  read through POST.
- All of these inside a **function, lambda or method an earlier cell
  defined**, at each place this cell calls it or passes it on, with its
  parameters read from the call: by its name, through an instance or the
  class, a list or dict that holds it, a ``functools.partial`` of it, or a
  function that returns it. All of these, too, in **literal code the cell
  runs**: a str or bytes literal handed to ``exec``, ``eval`` or
  ``compile``, a literal statement handed to ``timeit``, ``cProfile.run``,
  ``profile.run``, ``pdb.run`` or ``pdb.runeval``, and the literal after
  ``python -c``, each read like one more cell, placed right after the cell
  that runs it; and in a function a literal exec in an earlier cell defined,
  where this cell calls it. A program's or module's name held in a name
  bound only to literal text reads as that text.
- **Code the check can't read**, which holds the cell unchecked: its card
  ends "This step can't be checked before it runs." That is ``exec``,
  ``eval`` or ``compile`` reached through the builtins however the code
  names them and handed anything but a literal (a name, a file read, a
  decode, a join) or a literal too long to read, or passed or stored as a
  value; code the pad's own process runs through ``runpy``, ``importlib``
  (``reload``, a spec or a loader from a file, an import by a name the code
  picks when it runs), ``zipimport``, ``site`` (the ``.pth`` files it reads),
  a code object, an interactive console, a debugger, a profiler, ``trace``,
  ``doctest``, ``pytest.main``, a notebook runner, IPython's ``run_cell``,
  the C API's ``PyRun_`` functions, or a runner of text handed code that
  isn't a literal or a function it reads, each under any name or lookup and
  also passed or stored as a value; a write of a ``.pth`` file; an import of
  a module a cell wrote, or one new in the project since the turn began, or
  of a project module once a cell changed ``sys.path`` or wrote where the
  check can't place; a Python program the cell starts
  (``python`` with a script, a module, code from its input
  or ``-c`` in another folder or environment; a ``.py`` script; ``pytest``,
  ``papermill``, ``jupyter nbconvert --execute``, ``streamlit run``,
  ``uvicorn``, ``gunicorn``; what ``uv run``, ``poetry run`` and the like
  start), pip installing from a file, folder, URL, repository or index, or
  from a word the check can't read, and a program the check can't name at
  all; a call to a name, or through a namespace, module or value, that such
  code may have filled, once such code ran, a name an errored cell may not
  have bound among them; and a read a depth limit stopped where a write may
  lie past it. A helper the pad's boot injects, such as ``progress`` or
  ``sample``, is a name read code binds, and a function nothing calls runs
  no program its parameters name.

What it can't find: a statement or path with no literal behind it, a plain
procedure call the check can't tie to a write it runs, a write through an API
other than HTTP PUT, PATCH or DELETE or an operator-listed endpoint (aiohttp,
an SDK such as boto3 outside DynamoDB), a store method on a receiver the check
can't place, a shell command it can't read at all, a shell redirect or ``cp``
onto an existing file, an overwrite whose path the code picks from a root the
check can't place, what a program other than Python does (a shell script
file, ``node``, ``Rscript``), a program launcher handed on as a value, a write
through a trusted name that code the check couldn't read rebinds (``print =
requests.post``), what a ``pickle``, ``dill`` or ``joblib`` load runs, a call
to what unread code defined through ``.get(key)()`` or a list's element, and
code or a statement the cell reads from a file when it runs beyond the cases
above. An overwrite through a name the code picks when it runs, as in
``f"exports/orders_{date.today()}.csv"``, counts as a new file, unless a
listing (``glob``, ``os.listdir``, ``iterdir``) gave the name.
"""

from __future__ import annotations

import ast
import bisect
import builtins
import copy
import fnmatch
import itertools
import logging
import os
import posixpath
import re
import shlex
import stat
import string
import textwrap
import threading
import time
import warnings
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import IntEnum
from functools import cached_property
from pathlib import Path
from typing import Literal, NamedTuple, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from anton.core.backends.wire import EXPLAINABILITY_QUERIES_NAME, INJECTED_HELPER_NAMES, heal_surrogate_source
from anton.utils.datasources import scrub_output

__all__ = [
    "CallVia",
    "CallWrite",
    "CellWrites",
    "FileChange",
    "HttpWrite",
    "HttpWriteRule",
    "PACKAGE_FILE_SUFFIXES",
    "ProjectSnapshot",
    "SqlWrite",
    "find_cell_writes",
    "http_write_rule_matches",
    "parse_http_write_rules",
    "snapshot_project_files",
    "write_statements",
]

logger = logging.getLogger(__name__)

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


# How a `CallWrite` changes data: a stored procedure or function, a SQLAlchemy
# construct or session, a non-SQL store's own client ("data_store" for a table
# call on a client the check can't trace to DynamoDB or BigQuery), or SQL read
# from a file the check couldn't read before the cell runs.
CallVia = Literal[
    "procedure", "sqlalchemy", "mongodb", "dynamodb", "bigquery", "redis", "elasticsearch", "data_store", "sql_file",
]


class CallWrite(BaseModel):
    """One call that changes data with no SQL text to quote: the card quotes
    its code, scrubbed. When `call` calls a function an earlier cell defined,
    `runs` is the code inside it that makes the change."""

    model_config = ConfigDict(frozen=True)

    via: CallVia
    call: str
    line: int
    runs: str = ""


class FileChange(BaseModel):
    """One delete, rename, move, overwrite or append of an existing file."""

    model_config = ConfigDict(frozen=True)

    action: Literal["delete", "rename", "move", "overwrite", "append"]
    # Snapshot paths, the shared folder's prefixed with "shared folder: ".
    # Empty when the code picks the path when it runs.
    paths: tuple[str, ...] = ()
    whole_project: bool = False
    whole_shared: bool = False
    call: str
    line: int
    # False for a change that runs with no card and is logged as not held:
    # an append, or an overwrite of a file the agent owns.
    held: bool = True
    # For an overwrite through an in-process SQLite or DuckDB handle: each
    # statement it runs, as the card quotes a SQL write. A `to_sql` reads
    # "to_sql into <table>".
    statements: tuple[SqlWrite, ...] = ()


class HttpWrite(BaseModel):
    """One HTTP request that may change data on a server, as the card names it."""

    model_config = ConfigDict(frozen=True)

    # Upper case, or the code that picks it in braces, such as `{verb}`.
    method: str
    # Scrubbed, a part picked at run time shown as its code in braces.
    url: str
    call: str
    line: int
    # A part of `url` is shown as its code in braces.
    run_time_parts: bool = False
    # The code that picks the method when the step runs, scrubbed; "" when
    # the check cannot show it; None when the method is written out.
    method_code: str | None = None
    # The same for the whole address, when no part of it is written out.
    url_code: str | None = None


_HTTP_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})


class HttpWriteRule(BaseModel):
    """One endpoint the operator lists as a write: a request whose method and
    URL match it is held, a POST or GET included. `method` is upper case, or
    `*` for any. In `url_glob`, `*` stands for any text within one path
    segment, `**` for any text, and `?` for one character other than `/`;
    it is matched, ignoring case, against the request's URL without its
    query string or fragment (`http_write_rule_matches`)."""

    model_config = ConfigDict(frozen=True)

    method: str
    url_glob: str

    @field_validator("method")
    @classmethod
    def _known_method(cls, value: str) -> str:
        method = value.upper()
        if method != "*" and method not in _HTTP_METHODS:
            raise ValueError(f"the method must be * or one of {', '.join(sorted(_HTTP_METHODS))}")
        return method

    @field_validator("url_glob")
    @classmethod
    def _one_word(cls, value: str) -> str:
        if not value or any(ch.isspace() for ch in value):
            raise ValueError("the URL must be one word")
        return value


def parse_http_write_rules(text: str) -> tuple[HttpWriteRule, ...]:
    """The rules in the `write_approval_http` setting: entries separated by
    newlines or commas, each a method and a URL glob, as in
    `POST https://planner.example.com/api/*/orders/*`. A bad entry is skipped
    with a WARNING that names it."""
    rules: list[HttpWriteRule] = []
    for entry in (e.strip() for e in re.split(r"[\n,]", text)):
        if not entry:
            continue
        words = entry.split()
        try:
            if len(words) != 2:
                raise ValueError("an entry is a method and a URL")
            rules.append(HttpWriteRule(method=words[0], url_glob=words[1]))
        except (ValidationError, ValueError) as exc:
            reason = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
            logger.warning("write approval: skipped the HTTP write entry %r: %s", entry, reason)
    return tuple(rules)


class CellWrites(BaseModel):
    """What a cell would change. ``unchecked`` means the check couldn't read
    all the code the cell runs: code it hands ``exec`` or a runner, a module
    or a Python program it runs, what such code may have defined, or a read
    a depth limit stopped short; or the check itself failed. The card then
    says the step can't be checked before it runs, beside what it did read."""

    model_config = ConfigDict(frozen=True)

    sql: tuple[SqlWrite, ...] = ()
    calls: tuple[CallWrite, ...] = ()
    files: tuple[FileChange, ...] = ()
    http: tuple[HttpWrite, ...] = ()
    unchecked: bool = False

    @property
    def held_files(self) -> tuple[FileChange, ...]:
        return tuple(f for f in self.files if f.held)

    @property
    def not_held_files(self) -> tuple[FileChange, ...]:
        return tuple(f for f in self.files if not f.held)

    def needs_approval(self) -> bool:
        return bool(self.sql or self.calls or self.held_files or self.http or self.unchecked)

    def kinds(self) -> tuple[str, ...]:
        """What the card holds, for the log."""
        kinds = ["sql"] if self.sql else []
        kinds += sorted({c.via for c in self.calls})
        kinds += sorted({f.action for f in self.held_files})
        if self.http:
            kinds.append("http")
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

    @cached_property
    def module_names(self) -> frozenset[str]:
        """The names an import may reach a Python file under the root by
        (`_module_names`), read once per snapshot. A file in an
        environment's own package folder (`_PACKAGE_FOLDERS`) is an
        installed package."""
        return frozenset(
            name for path in self.files if _PACKAGE_FOLDERS.isdisjoint(path.split("/"))
            for name in _module_names(path)
        )

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

    def file_at_start(self, *, rel: str, made: _Made | None = None) -> str | None:
        """The name a card gives `rel` when it was a file at turn start: its
        recorded spelling, or, where the walk didn't record it (under an
        unwalked folder, past the cap, or after a failed walk), `rel` when a
        file is there, unless the turn changed it and `made`, what the pad's
        cells write, made it new (`named`). None for a folder or a new file."""
        found = self.recorded(rel=rel)
        if found is not None:
            return found if found in self.files else None
        if not (self.failed or not self.complete or self._under_unwalked(rel)):
            return None
        try:
            st = (self.root / rel).stat()
        except OSError:
            return None
        if not stat.S_ISREG(st.st_mode):
            return None
        changed = st.st_mtime >= self.taken_at
        return None if changed and made is not None and made.new(snap=self, rel=rel) else rel

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
# Procedures that run the SQL text they are given: the check reads that text.
_SQL_RUNNERS = frozenset({"POSTGRES_EXECUTE", "MYSQL_EXECUTE", "SP_EXECUTESQL", "IMMEDIATE"})
# Functions a SELECT can call that change data, the server's state or its files.
_SIDE_EFFECTS = frozenset({
    # Postgres
    "SETVAL", "NEXTVAL", "SET_CONFIG", "PG_TERMINATE_BACKEND", "PG_CANCEL_BACKEND", "PG_RELOAD_CONF",
    "PG_ROTATE_LOGFILE", "PG_SWITCH_WAL", "PG_PROMOTE", "PG_STAT_RESET", "PG_CREATE_RESTORE_POINT",
    "PG_CREATE_LOGICAL_REPLICATION_SLOT", "PG_CREATE_PHYSICAL_REPLICATION_SLOT", "PG_DROP_REPLICATION_SLOT",
    "PG_FILE_WRITE", "LO_UNLINK", "LO_IMPORT", "LO_EXPORT", "LO_CREATE", "LO_FROM_BYTEA", "LO_PUT",
    "DBLINK_EXEC", "PG_ADVISORY_LOCK", "PG_ADVISORY_LOCK_SHARED", "PG_ADVISORY_XACT_LOCK",
    "PG_ADVISORY_XACT_LOCK_SHARED", "PG_TRY_ADVISORY_LOCK", "PG_TRY_ADVISORY_LOCK_SHARED",
    "PG_TRY_ADVISORY_XACT_LOCK", "PG_TRY_ADVISORY_XACT_LOCK_SHARED", "PG_ADVISORY_UNLOCK",
    "PG_ADVISORY_UNLOCK_SHARED", "PG_ADVISORY_UNLOCK_ALL",
    # MySQL
    "GET_LOCK", "RELEASE_LOCK", "RELEASE_ALL_LOCKS",
})
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
    literal in it) or "punct" (any other character). `glued` means no space
    separates it from the token before it, as in `refresh(1)`.
    """

    kind: Literal["word", "name", "hole", "text", "group", "punct"]
    text: str
    raw: str = ""
    line_start: bool = False
    glued: bool = False
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
    group_line_start = group_glued = False
    start = end = 0
    gap_from = 0

    def close_group() -> None:
        tokens.append(_Token(kind="group", text=inner[0] if inner else "", line_start=group_line_start,
                             glued=group_glued, words=tuple(inner), texts=tuple(inner_texts)))

    for m in (_HASH_COMMENT_TOKEN if hash_comments else _TOKEN).finditer(text):
        kind = m.lastgroup
        line_start = _LINE_BREAK.search(text, gap_from, m.start()) is not None
        glued = gap_from == m.start()
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
                inner, inner_texts, group_line_start, group_glued = [], [], line_start, glued
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
        if tokens[j].text in _DML or _SIDE_EFFECTS & set(tokens[j].words):
            return True
        j += 1
        if not _punct(tokens, j, ","):
            break
        i = j + 1
    return _writes(tokens[j:], prose=prose)


def _select_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    """SELECT ... INTO a new table or a server file, or a SELECT that calls a
    function that changes something."""
    return _select_into(tokens, prose=prose) or _calls_side_effect(tokens, prose=prose)


def _calls_side_effect(tokens: Sequence[_Token], *, prose: bool) -> bool:
    """Whether a statement calls a function in `_SIDE_EFFECTS`, at its top
    level or in a parenthesized part, takes a sequence's `.NEXTVAL`, or gives
    dblink a write as a quoted literal."""
    for k, t in enumerate(tokens):
        if t.kind == "group" and _SIDE_EFFECTS & set(t.words):
            return True
        if t.kind != "word":
            continue
        called = k + 1 < len(tokens) and tokens[k + 1].kind == "group"
        if called and t.text in _SIDE_EFFECTS or t.text == "NEXTVAL" and _punct(tokens, k - 1, "."):
            return True
        if called and t.text == "DBLINK" and any(
            write_statements(_unquoted(s), prose=prose) for s in tokens[k + 1].texts
        ):
            return True
    return False


def _select_into(tokens: Sequence[_Token], *, prose: bool) -> bool:
    # MySQL writes a file on the database server, wherever the INTO sits.
    if any(_word(tokens, i) == "INTO" and _word(tokens, i + 1) in ("OUTFILE", "DUMPFILE") for i in range(len(tokens))):
        return True
    into = next((i for i in range(len(tokens)) if _word(tokens, i) == "INTO"), None)
    if into is None:
        return False
    first_from = next((i for i in range(len(tokens)) if _word(tokens, i) == "FROM"), None)
    if first_from is not None and first_from < into:
        return False
    # A variable (@v, :v) or a T-SQL temp table (#t) holds nothing that lasts.
    if _punct(tokens, into + 1, "@#:") or _word(tokens, into + 1) in _TEMP:
        return False
    if into + 1 == len(tokens) and not (prose and _sentence_case(tokens[0])):
        # The table comes after the text ends.
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
    Snowflake stage, from any table; or Postgres's COPY ... TO a file or
    program on the database server."""
    to = next((k for k in range(1, len(tokens)) if _word(tokens, k) == "TO"), None)
    if to is not None and not _sentence_case(tokens[0]) and (
        to + 1 < len(tokens) and tokens[to + 1].kind == "text" or _word(tokens, to + 1) == "PROGRAM"
    ):
        return True
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
    """CALL, EXEC or EXECUTE. A procedure can change anything, so a call
    counts, except one that runs SQL text the check reads instead (DuckDB's
    postgres_execute, T-SQL's sp_executesql and EXEC ('...'), EXECUTE
    IMMEDIATE) and DuckDB's pragma and catalog functions.

    In prose, a lowercase verb counts only before `name(`: `call volume` and
    `execute the plan` are prose, and a capitalized one never does. Text a
    SQL runner gets is read as SQL whatever its case."""
    if prose and _sentence_case(tokens[0]):
        return False
    i = 4 if _punct(tokens, 1, "@") and _punct(tokens, 3, "=") else 1  # T-SQL's EXEC @status = p
    if i < len(tokens) and tokens[i].kind == "group":
        return _runs_writes(tokens[i:], prose=prose)
    if _punct(tokens, i, "@"):
        return not prose or tokens[0].raw.isupper()  # T-SQL's EXEC @procedure: picked when it runs
    j = _name_end(tokens, i)
    if j is None:
        return False
    name = tokens[j - 1].text.upper()
    if name in _SQL_RUNNERS:
        return _runs_writes(tokens[j:], prose=prose)
    if name.startswith(("PRAGMA_", "DUCKDB_")):
        return False
    if j < len(tokens) and tokens[j].kind == "group" and tokens[j].glued:
        return True
    if prose and not tokens[0].raw.isupper():
        return False
    # MySQL's CALL p, or T-SQL's EXEC p @a = 1, 'x', 2.
    return j == len(tokens) or tokens[j].kind in ("group", "hole", "text") or (
        tokens[j].kind == "punct" and (tokens[j].text in "@-" or tokens[j].text.isdigit())
    ) or not prose and _placeholder_at(tokens, j)


def _placeholder_at(tokens: Sequence[_Token], i: int) -> bool:
    """Whether a driver's placeholder starts at `i`: pyodbc's `?`, or
    Oracle's `:1` or `:name`."""
    if _punct(tokens, i, "?"):
        return True
    return _punct(tokens, i, ":") and i + 1 < len(tokens) and (
        tokens[i + 1].kind == "word" or tokens[i + 1].kind == "punct" and tokens[i + 1].text.isdigit()
    )


# What follows BEGIN when it starts a transaction or a T-SQL TRY or CATCH
# block, or makes an empty block, rather than an Oracle block that runs a
# procedure: BEGIN TRANSACTION, BEGIN WORK, SQLite's BEGIN IMMEDIATE,
# Postgres's BEGIN ISOLATION LEVEL ..., Snowflake's BEGIN NAME t, BEGIN NULL.
_BEGIN_FOLLOWERS = frozenset({
    "TRANSACTION", "TRAN", "WORK", "DEFERRED", "IMMEDIATE", "EXCLUSIVE", "ISOLATION", "READ", "NOT",
    "DEFERRABLE", "DISTRIBUTED", "TRY", "CATCH", "NAME", "ATOMIC", "NULL",
})


def _begin_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    """An Oracle or Snowflake block, `BEGIN ...; END;`, up to its first `;`:
    a write that opens it counts, and so does a bare procedure call, as in
    `BEGIN p; END;` or `BEGIN pkg.proc(:1); END;`. BEGIN alone, or before a
    word in `_BEGIN_FOLLOWERS`, changes nothing. In prose, only an
    upper-case BEGIN counts, and a call needs its `name(`."""
    if len(tokens) == 1 or _word(tokens, 1) in _BEGIN_FOLLOWERS or prose and not tokens[0].raw.isupper():
        return False
    if _writes(tokens[1:], prose=prose):
        return True
    if tokens[1].kind == "word" and (tokens[1].text in _RULES or tokens[1].text in _SCRIPT_WORDS):
        return False
    j = _name_end(tokens, 1)
    if j is None:
        return False
    called = j + 1 == len(tokens) and tokens[j].kind == "group" and tokens[j].glued
    return called or not prose and j == len(tokens)


def _unescaped(tokens: Sequence[_Token]) -> Sequence[_Token]:
    """ODBC's escape for a procedure call, `{CALL p(?)}` or `{? = CALL f(?)}`,
    as the CALL inside it; any other tokens as they are."""
    if not _punct(tokens, 0, "{"):
        return tokens
    i = 3 if _punct(tokens, 1, "?") and _punct(tokens, 2, "=") else 1
    if _word(tokens, i) != "CALL":
        return tokens
    return tokens[i:-1] if _punct(tokens, len(tokens) - 1, "}") else tokens[i:]


def _runs_writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    """Whether the SQL text a runner is given writes: one of its quoted
    literals does, or it has none, so what it runs is picked when it runs."""
    literals = [t.text for t in tokens if t.kind == "text"] + [s for t in tokens for s in t.texts]
    return not literals or any(write_statements(_unquoted(s), prose=prose) for s in literals)


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
    "SELECT": _select_writes,
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
    "BEGIN": _begin_writes,
}


def _writes(tokens: Sequence[_Token], *, prose: bool) -> bool:
    tokens = _unescaped(tokens)
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
# Characters of one literal source the check reads as one more cell; past
# this, a cell that runs it is unchecked.
_LITERAL_CODE_LIMIT = 100_000
# Calls that run code in this process that the check can't read: a script,
# a module, a notebook, a code object, or what a console, debugger, profiler
# or test runner is handed.
_CODE_RUNNERS = frozenset({
    "runpy.run_path", "runpy.run_module", "importlib.reload",
    "importlib.util.spec_from_file_location", "importlib.util.spec_from_loader",
    "importlib.machinery.SourceFileLoader", "importlib.machinery.SourcelessFileLoader", "zipimport.zipimporter",
    "types.FunctionType", "types.CodeType", "code.interact", "code.InteractiveInterpreter",
    "code.InteractiveConsole", "pdb.runctx", "pdb.runcall", "cProfile.runctx", "profile.runctx",
    "doctest.testmod", "doctest.testfile", "doctest.run_docstring_examples", "doctest.DocTestRunner",
    "pytest.main", "papermill.execute_notebook", "nbclient.execute",
    "runpy._run_code", "runpy._run_module_code", "runpy._run_module_as_main",
    # Each runs the import lines of the .pth files in the folders it adds.
    "site.addsitedir", "site.addpackage", "site.main",
})
# Modules whose members run code the check can't read, so a member looked up
# by a name it can't read may be one.
_RUNNER_MODULES = frozenset(
    name.rpartition(".")[0] for name in (*_CODE_RUNNERS, "importlib.util.module_from_spec")
)
# The C API's PyRun_* functions run the text they're handed: the start of
# their names, and of their full names.
_PYRUN_PREFIX = "PyRun_"
_PYRUN_FUNCS = f"ctypes.pythonapi.{_PYRUN_PREFIX}"
# Calls that run text as code, by the parameters that take it, in order: a
# literal text is read as one more cell, as exec()'s is.
_STRING_RUNNERS = {
    "timeit.timeit": ("stmt", "setup"), "timeit.repeat": ("stmt", "setup"), "timeit.Timer": ("stmt", "setup"),
    "cProfile.run": ("command",), "profile.run": ("statement",), "pdb.run": ("statement",),
    "pdb.runeval": ("expression",),
}
_STRING_RUNNER_NAMES = frozenset(name.rpartition(".")[2] for name in _STRING_RUNNERS)
# Methods that run code, by the call that makes the object they're called on.
_IPYTHON_RUNS = frozenset({"run_cell", "run_line_magic", "run_cell_magic"})
_INSTANCE_RUNNERS = {
    "cProfile.Profile": frozenset({"run", "runctx", "runcall"}),
    "profile.Profile": frozenset({"run", "runctx", "runcall"}),
    "pdb.Pdb": frozenset({"run", "runeval", "runctx", "runcall"}),
    "bdb.Bdb": frozenset({"run", "runeval", "runctx", "runcall"}),
    "trace.Trace": frozenset({"run", "runctx", "runfunc"}),
    "nbclient.NotebookClient": frozenset({"execute"}),
    "nbclient.client.NotebookClient": frozenset({"execute"}),
    "nbconvert.preprocessors.ExecutePreprocessor": frozenset({"preprocess"}),
    "nbconvert.preprocessors.execute.ExecutePreprocessor": frozenset({"preprocess"}),
    "get_ipython": _IPYTHON_RUNS, "IPython.get_ipython": _IPYTHON_RUNS,
    "IPython.core.getipython.get_ipython": _IPYTHON_RUNS,
}
_RUNNER_METHODS = frozenset().union(*_INSTANCE_RUNNERS.values())
# Methods that run the module a loader or importer loads, whatever made it.
_LOADER_METHODS = frozenset({"exec_module", "load_module"})
# Calls that import a module by its name. `pkgutil.resolve_name` and
# `pydoc.locate` take the name as "pkg.mod:attr" or "pkg.mod"; the module
# before the colon is the one they load.
_IMPORTERS = frozenset({
    "importlib.import_module", "importlib.__import__", "__import__", "builtins.__import__",
    "pkgutil.resolve_name", "pydoc.locate",
})
_IMPORTER_LAST_NAMES = frozenset(name.rpartition(".")[2] for name in _IMPORTERS)
# The importers whose result is the module itself, so a name bound to one
# reads as that module (`_Reader._importer_module_texts`).
_MODULE_IMPORTERS = frozenset({
    "importlib.import_module", "importlib.__import__", "__import__", "builtins.__import__",
})
_MODULE_IMPORTER_LAST_NAMES = frozenset(name.rpartition(".")[2] for name in _MODULE_IMPORTERS)
# The last name of each call `_runs_unread` reads, so any other is passed by.
_RUNNER_NAMES = frozenset(
    name.rpartition(".")[2] for name in (*_CODE_RUNNERS, *_STRING_RUNNERS, *_IMPORTERS)
)
# Names that reach a runner or an importer with no import: the builtin
# `__import__`, and `get_ipython`, which an IPython shell binds.
_PAD_BUILTINS = frozenset({"__import__", "get_ipython"})
# What changes the folders `import` looks in, and the methods that add to a
# list, as `sys.path` is one.
_LIST_CHANGES = frozenset({"insert", "append", "extend", "__setitem__", "__iadd__"})
_SYS_PATH_CHANGES = frozenset({
    "sys.path", "site.addsitedir", *(f"sys.path.{method}" for method in _LIST_CHANGES),
})
_PY_FILE = re.compile(r"\.py\b")
# Every name Python, or the pad's boot, binds before a cell runs, so a call
# to one is no sign that code the check couldn't read defined it.
_BUILTIN_NAMES = frozenset({*vars(builtins), *INJECTED_HELPER_NAMES, EXPLAINABILITY_QUERIES_NAME})
# Runners and importers that run code the check can't read wherever they are
# passed on or stored to be called later.
_RUNNER_VALUES = frozenset({*_CODE_RUNNERS, *_STRING_RUNNERS, *_IMPORTERS})
# Runners whose return value is a namespace a later call reaches through.
_NAMESPACE_RUNNERS = frozenset({"runpy.run_path", "runpy.run_module"})
# The calls that hand out the cell's own namespace to fill or read by key.
_GLOBALS_CALLS = frozenset({"globals", "vars", "locals"})
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
# Modules and classes that hold a delete, a move, a copy or a shell call, for
# a lookup by a name the check computes (`_FileFinder.lookup`).
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
    "os.posix_spawn": 1, "os.posix_spawnp": 1, "pty.spawn": 0,
}
# Every call that runs a program, and the last names they are called by.
_LAUNCHERS = frozenset({*_SHELL_FUNCS, *_SPREAD_ARGV_FUNCS, *_LIST_ARGV_FUNCS})
_LAUNCH_NAMES = frozenset(name.rpartition(".")[2] for name in _LAUNCHERS)
# The modules that hold a launcher, so a member looked up on one by a name
# the check can't read may be a launcher.
_LAUNCHER_MODULES = frozenset(name.rpartition(".")[0] for name in _LAUNCHERS)
# The last names that hint a cell may run code the check can't read
# (`_CodeSigns.hints`), and the calls `_names_a_runner` reads such a name as
# through the imports. A conversation whose cells name none of these that
# way, and write and import no module the check can't read, runs only code
# it reads.
_BUILTINS_MODULE_NAMES = frozenset({"builtins", "__builtins__"})
_CODE_HINTS = frozenset({
    *_EXEC_CALLS, *_GLOBALS_CALLS, *_RUNNER_NAMES,
    *_RUNNER_METHODS, *_LOADER_METHODS, *_LAUNCH_NAMES, *_BUILTINS_MODULE_NAMES,
})
# Every call that runs code the check can't read, passed on or stored.
_RUN_CALLS = frozenset({*_RUNNER_VALUES, *_LAUNCHERS})
# The runners and launchers each module holds, by the bare name a
# `from module import *` binds them to (`_Reader.star_runners`).
_RUNNERS_BY_MODULE: dict[str, dict[str, str]] = {}
for _full in _RUN_CALLS:
    _module, _, _name = _full.rpartition(".")
    if _module:
        _RUNNERS_BY_MODULE.setdefault(_module, {})[_name] = _full
# The modules that hold something that runs code, so a member named off one
# by a name the check can't read may run code (`_Reader._pick_runner`).
_RUNNER_HOLDERS = frozenset({*_RUNNER_MODULES, *_LAUNCHER_MODULES, *_BUILTINS_MODULE_NAMES})
# The last names of the calls that may run code the check can't read, however
# the imports name them (`_Cell.unread_candidates`): every hint that names a
# call, not the cell's own namespace or the builtins module, and `getattr`.
_UNREAD_CALL_NAMES = frozenset({*(_CODE_HINTS - _GLOBALS_CALLS - _BUILTINS_MODULE_NAMES), "getattr"})


def _is_hint(last: str) -> bool:
    """Whether `last`, the last name of a node, may name something that runs
    code the check can't read: a hint (`_CODE_HINTS`) or a PyRun_ function."""
    return last in _CODE_HINTS or last.startswith(_PYRUN_PREFIX)


# The Python interpreter by any of its names, and the scripts it runs as a
# program of their own.
_PYTHON_PROGRAM = re.compile(r"(?:python|pypy|ipython)(?:\d+(?:\.\d+)*)?w?")
_PYTHON_SCRIPTS = (".py", ".pyw", ".pyz")
# Python's long options that print and exit, as the short -V and -h do.
_PYTHON_EXITS = frozenset({"--version", "--help", "--help-env", "--help-xoptions", "--help-all"})
# Programs that load a project's code, a test or a notebook: by name, and by
# name and command.
_PYTHON_TOOLS = frozenset({"pytest", "py.test", "papermill", "uvicorn", "gunicorn", "jupyter-execute"})
_PYTHON_TOOL_COMMANDS = frozenset({("streamlit", "run"), ("jupyter", "execute")})
# Runners that run the command after them in a Python environment, by the
# head word to the subcommand words that follow it. Leading global flags may
# come before the subcommand, as in `uv -q run`.
_ENV_RUNNERS: dict[str, tuple[tuple[str, ...], ...]] = {
    "uv": (("run",), ("tool", "run")),
    "uvx": ((),),
    "poetry": (("run",),),
    "pipx": (("run",),),
    "pdm": (("run",),),
    "hatch": (("run",),),
    "conda": (("run",),),
    "pipenv": (("run",),),
}
_ENV_RUNNER_VALUE_FLAGS = frozenset({
    "--with", "--with-requirements", "--with-editable", "--python", "-p", "--project", "--directory",
    "--package", "--extra", "--group", "--env-file", "--index", "--from", "--spec", "-n", "--name",
    "--prefix", "--cwd",
})
# uv run's flags that run a Python module or script, as `python -m` and a
# script path do.
_ENV_RUN_MODULE_FLAGS = frozenset({"-m", "--module"})
_ENV_RUN_SCRIPT_FLAGS = frozenset({"-s", "--script"})
# Modules `python -m` runs that only set up an environment or fetch data.
_SETUP_MODULES = frozenset({
    "ensurepip", "venv", "spacy", "nltk", "nltk.downloader", "playwright", "textblob.download_corpora",
})
_PIP_PROGRAM = re.compile(r"pip(?:\d+(?:\.\d+)*)?")
# All of pip's commands, those of them that take packages, and the ones that
# install from a file of requirements whatever follows, as `uv pip sync`
# does. A word the check reads as the command that is none of these means a
# flag ate it, so the check can't tell what pip does.
_PIP_COMMANDS = frozenset({
    "install", "download", "uninstall", "wheel", "list", "show", "freeze", "config", "cache", "check",
    "debug", "hash", "help", "inspect", "index", "search", "completion", "sync",
})
_PIP_PACKAGE_COMMANDS = frozenset({"install", "download", "uninstall", "wheel"})
_PIP_FILE_COMMANDS = frozenset({"sync"})
# pip's flags that name where a package comes from, the long ones matched by
# unique prefix (`--req` is `--requirement`), and the short ones' letters.
_PIP_SOURCE_FLAGS = frozenset({
    "-r", "--requirement", "--requirements", "-e", "--editable", "-i", "--index-url", "--extra-index-url",
    "-f", "--find-links",
})
_PIP_SOURCE_LONG = frozenset(flag for flag in _PIP_SOURCE_FLAGS if flag.startswith("--"))
_PIP_SOURCE_LETTERS = frozenset("reif")
# pip's flags that take a value, so their value isn't a source; the short
# ones' letters; and the global ones that come before the command.
_PIP_VALUE_FLAGS = frozenset({
    "-t", "--target", "--prefix", "--root", "--src", "-c", "--constraint", "--cache-dir", "--log", "--proxy",
    "--retries", "--timeout", "--trusted-host", "--cert", "--client-cert", "--exists-action",
    "--python-version", "--platform", "--implementation", "--abi", "--only-binary", "--no-binary",
    "--progress-bar", "--upgrade-strategy", "--root-user-action", "--report", "-d", "--dest", "--python",
    "-C", "--config-settings", "--global-option", "--use-feature", "--use-deprecated", "-w", "--wheel-dir",
    "--keyring-provider", "--config-file", "--local-log",
})
_PIP_VALUE_LONG = frozenset(flag for flag in _PIP_VALUE_FLAGS if flag.startswith("--"))
_PIP_VALUE_LETTERS = frozenset("ctdwC")
# Endings of a file pip and uv install as a package from where it lies, with
# the code it runs to build or install it, rather than from the index. Any
# other name, `discord.py` or `job.egg` among them, is a name on the index.
# The scratchpad refuses a package entry with one too.
PACKAGE_FILE_SUFFIXES = (
    ".whl", ".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tar.xz", ".txz", ".tlz", ".tar.lz",
    ".tar.lzma",
)
_CHDIR_FUNCS = frozenset({"os.chdir", "contextlib.chdir"})
# Calls that end the process, so nothing after them runs.
_EXITS = frozenset({"sys.exit", "os._exit", "os.abort", "exit", "quit"})
_SHELL_DELETE = frozenset({"rm", "unlink", "rmdir", "shred"})
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})
# Words that change the working folder, how many shells deep a script is read
# before its commands count as unreadable, and the `find` flags that run a
# command on each match: read the same way by the delete, move and launch walks.
_CD_WORDS = ("cd", "pushd")
_SHELL_DEPTH = 4
_FIND_EXEC_FLAGS = ("-exec", "-execdir", "-ok", "-okdir")
# Words that run the command after them, and shell keywords a command follows.
_SHELL_PREFIXES = frozenset({
    "sudo", "doas", "env", "nohup", "time", "command", "exec", "nice", "xargs",
    "timeout", "stdbuf", "ionice", "busybox", "setsid",
})
# Flags of one runner that take the next word as their value, beyond
# `_VALUE_FLAGS`, and the operands a runner takes before the command, as
# timeout takes its duration.
_PREFIX_VALUE_FLAGS = {
    "timeout": frozenset({"-k"}), "stdbuf": frozenset({"-i", "-o", "-e"}), "ionice": frozenset({"-c", "-p"}),
    "xargs": frozenset({"-E"}),
}
# A shell word that sets a variable for the command after it.
_SHELL_ASSIGNMENT = re.compile(r"^[A-Za-z_]\w*=")
_PREFIX_OPERANDS = {"timeout": 1}
_SHELL_KEYWORDS = frozenset({"if", "then", "else", "elif", "do", "while", "until", "!", "{"})
# Shell words that end one command and start the next.
_SHELL_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "(", ")", "|&", ";;"})
# Folders the pad changes into that the check tracks; past this many, any.
_CWD_LIMIT = 16
# Calls into earlier cells' functions followed from one call in this cell.
_CALL_DEPTH = 4
# Methods that write the file their first argument names, and the keywords
# that can name it instead.
_WRITER_METHODS = {
    "to_csv": ("path_or_buf",), "to_json": ("path_or_buf",), "to_excel": ("excel_writer",),
    "to_parquet": ("path",), "to_feather": ("path",), "to_pickle": ("path",), "to_html": ("buf",),
    "savefig": ("fname",), "save": ("filename", "fp", "path", "file"),
}
# The last names of the calls `_written_target` reads a written path from.
_FILE_WRITE_NAMES = frozenset({
    "open", "write_text", "write_bytes", "ExcelWriter", "rename", "replace", *_WRITER_METHODS,
    *(name.rpartition(".")[2] for name in (*_MOVE_FUNCS, *_COPY_FUNCS)),
})
# Folders under the project whose files are the agent's own output.
_AGENT_FOLDERS = tuple(f".anton/{name}/" for name in sorted(_ANTON_WALKED))
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
# Stands in for a part of a path the code picks when it runs from a value the
# check can't read, as in f"orders_{date.today()}.csv". `_locate` reads it as
# `*`; for an overwrite it makes the path a new file (`_FileFinder.overwrite`).
# A listing's own wildcard, as from glob() or os.listdir(), stays `*`: it
# names files that exist.
_RUN_TIME_PART = "\x01"
# Stands in for a literal-command operand that isn't a literal.
_NOT_LITERAL = "\x00"
_PARAM_DEPTH = 5
# How far a read that stopped at its depth limit is read on, each node once,
# to tell whether it stopped short of what it looks for (`_Reader.read_past`).
_PAST_DEPTH = 64
# Paths one value may hold that the check keeps; past this many, unknown.
_PATH_LIMIT = 256
# Bindings the reading of errored cells as maybe stopped anywhere reads in one
# check before it gives up, which keeps only the run-to-end reading's findings.
_WIDE_BINDINGS = 16 * _PATH_LIMIT


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

_T = TypeVar("_T")


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
    guessed: bool = False


class _Literal(NamedTuple):
    """A literal a call passes for one parameter, by the parameter's name and
    the literal's repr."""

    param: str
    value: str


@dataclass(frozen=True)
class _Visit:
    """One read of a function at a call (`find_cell_writes`): the function,
    by id, and the literal arguments the call passes, by parameter, a
    parameter it passes on read as the literal the call being read gave it,
    so a recursive call that passes other literals is read again with them."""

    func: int
    literals: tuple[_Literal, ...]

    @classmethod
    def of(cls, callee: _Callee, reader: _Reader) -> _Visit:
        literals: list[_Literal] = []
        if callee.call is not None:
            passed = _call_arguments(callee.func, callee.call, call_cell=reader.cell_of(callee.call),
                                     func_cell=reader.cell_of(callee.func), bound=callee.bound)
            for name, arg in passed.items():
                if arg is not None and isinstance(arg.node, ast.Constant):
                    literals.append(_Literal(name, repr(arg.node.value)))
                elif arg is not None and isinstance(arg.node, ast.Name):
                    # A parameter the call being read passes on.
                    value = reader.literal_argument(arg.node)
                    if value is not None:
                        literals.append(_Literal(name, repr(value)))
        return cls(func=id(callee.func), literals=tuple(sorted(literals)))


class _ListChange(NamedTuple):
    """One place a cell adds to a bare name's list, or stores to it: the name,
    and the node that does it."""

    name: str
    node: ast.AST


@dataclass(frozen=True)
class _CodeSigns:
    """What one walk of a cell reads for the check of code it can't read
    (`_Cell.signs`). `_runs_code` reads which of the hints run such code."""

    # The nodes that may name something that runs such code, by their last
    # name (`_CODE_HINTS`, and the C API's PyRun_ functions), each entry
    # looked up under a key one of those names, and each import of the
    # builtins module.
    hints: tuple[ast.AST, ...]
    # The cell's import statements, in walk order.
    imports: tuple[ast.Import | ast.ImportFrom, ...]
    # Whether the cell names exec, eval, compile or the builtins module at
    # all, by a bare name, a string or an import: a conversation with no such
    # cell can't reach those builtins (`_Reader.names_builtins`), since
    # `x.exec` reaches one only where `x` holds the builtins module.
    names_builtins: bool
    # Whether a string names a Python file, `.py` at the end of a word, as a
    # write of a module the pad imports needs.
    names_a_py_file: bool
    # Whether a string names a `.pth` file, whose import lines Python runs
    # as it starts.
    names_a_pth_file: bool
    # Whether a string holds `-c`, as the code `python -c` runs follows.
    names_dash_c: bool
    # Code that may change where `import` looks, which `_Modules` reads
    # through the imports as `sys.path` or `site.addsitedir`: the function of
    # a call to a method that adds to a list on a `path`, or to `addsitedir`,
    # and a `path`, or an item of one, that an assignment or `+=` stores to.
    path_changes: tuple[ast.AST, ...]
    # The same change made on a bare name, by the name, which counts where
    # the name holds `sys.path` (`path_aliases`).
    list_changes: tuple[_ListChange, ...]
    # The names an assignment binds to a `path`, or `from sys import path` does.
    path_aliases: frozenset[str]
    # The names an assignment binds to a hint, as `rp = runpy.run_path` binds `rp`.
    hint_aliases: frozenset[str]
    # The names an `except ... as`, `global` or `nonlocal` statement, a `*`
    # or `**` parameter, a match pattern or a walrus gives, which
    # `_Cell.bindings` doesn't hold where they bind.
    bound_aside: frozenset[str]
    # Each `global` statement.
    globals_declared: tuple[ast.Global, ...]
    # The strings it holds that could be a name, as a lookup in the cell
    # namespace by name spells one.
    spelled: frozenset[str]
    # The names a `del` statement unbinds.
    deleted: frozenset[str]
    # The names an import reaches each Python file a string names by
    # (`_module_names`).
    py_files: frozenset[str]


class _Cell:
    """One parsed cell: its tree, parents, bindings and source lines, and
    what `_Reader` gathers from every cell. Nothing changes it once built, so
    one earlier cell serves every check that reads it (`_EarlierCells`)."""

    def __init__(
        self, tree: ast.Module, code: str, index: int, *, host_line: int | None = None, launched: bool = False,
        host: int | None = None, shared: bool = True, namespace: str | None = None,
    ) -> None:
        # Literal getattr and import lookups read as plain names, and the
        # plain names that now stand for a literal import (`_read_lookups_plainly`).
        self.looked_up = _read_lookups_plainly(tree)
        self.tree = tree
        self.index = index
        # For code a cell runs through exec() or eval(): the line of that call
        # in the cell, which findings report; the index of the conversation's
        # cell whose run runs it, which places it among them (`_Reader.before`);
        # whether what it binds lands in the cell namespace, as for an exec()
        # at a cell's top level given no namespace or `globals()`; and the
        # name of the namespace it fills otherwise, as `ns` for `exec(src, ns)`.
        # A conversation's own cell is its own host, and shares its names.
        self.host_line = host_line
        self.host = index if host is None else host
        self.shared = shared
        self.namespace = namespace
        # For code a cell hands `python -c`, and the code that code runs: the
        # Python it runs in has its working folder on `sys.path`.
        self.launched = launched
        self.lines = code.splitlines()
        self.parents: dict[ast.AST, ast.AST] = {}
        self._run_nodes: dict[int, tuple[ast.AST, ...]] = {}
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
        # What `h.run = v` stores, by the name and the member: `v`.
        self.member_values: dict[tuple[str, str], list[ast.expr]] = {}
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
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and not self._dead(node):
                        self.member_values.setdefault((t.value.id, t.attr), []).append(node.value)
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
            if _never_runs(parent=parent, child=child):
                return True
            child, parent = parent, self.parents.get(parent)
        return False

    def under_branch(self, node: ast.AST) -> bool:
        """Whether `node` sits under an if, loop, try or match in its scope,
        so it may not run."""
        return self._under_branch(node, scope=self.scope_of(node))

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

    @cached_property
    def module_bound(self) -> frozenset[str]:
        """The names the cell binds at its top level, outside any function,
        class or comprehension."""
        return frozenset(name for name, found in self.bindings.items() if any(b.scope is None for b in found))

    @cached_property
    def names_a_routine(self) -> bool:
        """Whether the parsed code names `callproc` or `callfunc`, as an
        attribute or a string, however the source spells it."""
        return any(
            isinstance(node, ast.Attribute) and node.attr in _ROUTINE_CALLS
            or isinstance(node, ast.Constant) and node.value in _ROUTINE_CALLS
            for node in ast.walk(self.tree)
        )

    @cached_property
    def function_execs(self) -> list[ast.Call]:
        """The calls inside a function, lambda or class the cell defines that
        are spelled as exec(), eval() or compile() (`_spelled_exec`), or as a
        runner of text such as `timeit.timeit` (`_STRING_RUNNER_NAMES`): a
        check that calls the function reads the literal code they run."""
        return [
            c for c in self.calls
            if (_spelled_exec(c.func) or _last_name(c.func) in _STRING_RUNNER_NAMES) and self.scope_of(c) is not None
        ]

    @cached_property
    def function_launches(self) -> list[ast.Call]:
        """The calls inside a function, lambda or class the cell defines that
        are spelled as a call that runs a program (`_LAUNCH_NAMES`) and hold
        a string with `-c` in it: they may hand `python -c` literal code,
        which a check that calls the function reads."""
        return [
            c for c in self.calls
            if _last_name(c.func) in _LAUNCH_NAMES and self.scope_of(c) is not None and any(
                isinstance(n, ast.Constant) and isinstance(n.value, str) and "-c" in n.value for n in ast.walk(c)
            )
        ]

    def nodes_run_by(self, root: ast.AST) -> tuple[ast.AST, ...]:
        """The nodes of `root`, the cell's tree or a function in it, that run
        when it runs (`_nodes_that_run`), read once per root: an earlier
        cell serves every check that reads it."""
        found = self._run_nodes.get(id(root))
        if found is None:
            found = self._run_nodes[id(root)] = tuple(_nodes_that_run(root))
        return found

    @cached_property
    def unread_candidates(self) -> tuple[ast.Call, ...]:
        """The calls spelled as one that may run code the check can't read:
        by a last name or key a runner, launcher, loader or the builtin
        exec(), eval() or compile() has (`_UNREAD_CALL_NAMES`), a PyRun_
        function, a lookup by `getattr` or a call of what one returns, and
        a write of a file where the cell names a `.pth` file."""
        out: list[ast.Call] = []
        for call in self.calls:
            name = _member_name(call.func)
            if (isinstance(call.func, ast.Call) or name in _UNREAD_CALL_NAMES or name.startswith(_PYRUN_PREFIX)
                    or self.signs.names_a_pth_file and name in _FILE_WRITE_NAMES):
                out.append(call)
        return tuple(out)

    @cached_property
    def hands_literal_code(self) -> bool:
        """Whether the cell may hand literal code to something that runs it:
        a call spelled as exec(), eval(), compile() or a runner of text
        (`_spelled_exec`, `_STRING_RUNNER_NAMES`) given a literal or a call,
        or a string with `-c` in it, as `python -c` takes code. A cell with
        none gives `_exec_cells` nothing to read at its own top level."""
        return self.signs.names_dash_c or any(
            (_spelled_exec(c.func) or _last_name(c.func) in _STRING_RUNNER_NAMES) and any(
                _literal_source(a) is not None or isinstance(a, ast.Call)
                for a in (*c.args, *(k.value for k in c.keywords))
            )
            for c in self.calls
        )

    @cached_property
    def fills_that_run(self) -> tuple[ast.AST, ...]:
        """The code that runs when the cell runs (`_nodes_that_run`) that may
        fill a name or a namespace with what code the check can't read made,
        read once per cell: each import; each assignment and walrus; and
        each call that hands on a namespace or updates one, by a second
        argument, a `globals` or `locals` keyword, or `.update(...)`.
        `_build_defined` and `_poisons_namespace` read no other."""
        return tuple(
            node for node in self.nodes_run_by(self.tree)
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.NamedExpr))
            or isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
            or isinstance(node, ast.Call) and (
                len(node.args) > 1 or any(k.arg in ("globals", "locals") for k in node.keywords)
                or isinstance(node.func, ast.Attribute) and node.func.attr == "update"
            )
        )

    @cached_property
    def writer_calls(self) -> list[ast.Call]:
        """The calls spelled as one that may write a module: a write, copy
        or move of a file (`_FILE_WRITE_NAMES`), a call that makes a file
        (`_FILE_MAKERS`) or unpacks a tree of them (`_TREE_WRITERS`), and a
        call that runs a program (`_LAUNCH_NAMES`). `_written_modules` reads
        no other."""
        return [c for c in self.calls if _last_name(c.func) in _WRITER_CALL_NAMES]

    @cached_property
    def signs(self) -> _CodeSigns:
        """What the check of code it can't read needs from the cell, read in
        one walk once per cell (`_CodeSigns`)."""
        hints: list[ast.AST] = []
        imports: list[ast.Import | ast.ImportFrom] = []
        names_builtins = names_a_py_file = names_a_pth_file = names_dash_c = False
        path_changes: list[ast.AST] = []
        list_changes: list[_ListChange] = []
        path_aliases: set[str] = set()
        hint_aliases: set[str] = set()
        aside: set[str] = set()
        declared: list[ast.Global] = []
        spelled: set[str] = set()
        deleted: set[str] = set()
        py_files: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Name, ast.Attribute)):
                if _is_hint(_last_name(node)):
                    hints.append(node)
                # A member `.exec` reaches the builtin where its holder is the
                # builtins module, which a name may hold with no literal
                # "builtins" in the cell, as `import_module("built" + "ins")`.
                names_builtins = names_builtins or (
                    node.id in _EXEC_CALLS or node.id in _BUILTINS_MODULE_NAMES
                    if isinstance(node, ast.Name) else node.attr in _EXEC_CALLS
                )
            elif isinstance(node, ast.Subscript):
                key = node.slice.value if isinstance(node.slice, ast.Constant) else None
                if key in _CODE_HINTS:
                    hints.append(node)
                    names_builtins = names_builtins or key in _EXEC_CALLS or key in _BUILTINS_MODULE_NAMES
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names_builtins = names_builtins or node.value in _EXEC_CALLS or node.value in _BUILTINS_MODULE_NAMES
                names_a_py_file = names_a_py_file or bool(_PY_FILE.search(node.value))
                names_a_pth_file = names_a_pth_file or ".pth" in node.value
                names_dash_c = names_dash_c or "-c" in node.value
                if _PY_FILE.search(node.value):
                    py_files.update(n for word in node.value.split() for n in _module_names(word.strip("'\";,()")))
                if node.value.isidentifier():
                    spelled.add(node.value)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                if "builtins" in modules:
                    hints.append(node)
                    names_builtins = True
                if isinstance(node, ast.ImportFrom) and node.module == "sys":
                    path_aliases.update(a.asname or a.name for a in node.names if a.name == "path")
            elif isinstance(node, ast.ExceptHandler) and node.name:
                aside.add(node.name)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                aside.update(node.names)
                if isinstance(node, ast.Global):
                    declared.append(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                aside.update(a.arg for a in (node.args.vararg, node.args.kwarg) if a is not None)
            elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
                aside.add(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                aside.add(node.rest)
            elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                # A walrus in a comprehension binds in the scope around it.
                aside.add(node.target.id)
            elif isinstance(node, ast.Delete):
                deleted.update(t.id for t in node.targets if isinstance(t, ast.Name))
            if isinstance(node, ast.Call):
                func = node.func
                if _last_name(func) == "addsitedir":
                    path_changes.append(func)
                elif isinstance(func, ast.Attribute) and func.attr in _LIST_CHANGES:
                    if _last_name(func.value) == "path":
                        path_changes.append(func)
                    elif isinstance(func.value, ast.Name):
                        list_changes.append(_ListChange(func.value.id, func))
            elif isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    stored = target.value if isinstance(target, ast.Subscript) else target
                    if _last_name(stored) == "path":
                        path_changes.append(stored)
                    elif isinstance(stored, ast.Name) and (stored is not target or isinstance(node, ast.AugAssign)):
                        list_changes.append(_ListChange(stored.id, stored))
                value = node.value
                if isinstance(node, ast.Assign) and isinstance(value, (ast.Name, ast.Attribute)):
                    names = [t.id for t in targets if isinstance(t, ast.Name)]
                    last = _last_name(value)
                    if last == "path":
                        path_aliases.update(names)
                    if _is_hint(last):
                        hint_aliases.update(names)
        return _CodeSigns(
            hints=tuple(hints), imports=tuple(imports), names_builtins=names_builtins,
            names_a_py_file=names_a_py_file, names_a_pth_file=names_a_pth_file, names_dash_c=names_dash_c,
            path_changes=tuple(path_changes), list_changes=tuple(list_changes),
            path_aliases=frozenset(path_aliases), hint_aliases=frozenset(hint_aliases),
            bound_aside=frozenset(aside), globals_declared=tuple(declared), spelled=frozenset(spelled),
            deleted=frozenset(deleted), py_files=frozenset(py_files),
        )

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


def _never_runs(*, parent: ast.AST, child: ast.AST) -> bool:
    """Whether `child` sits in a branch of `parent` that never runs: the body
    of an if or while whose test is a constant false (`if False:`, `if 0:`,
    `while False:`), or the else of an if whose test is a constant true."""
    if not (isinstance(parent, (ast.If, ast.While)) and isinstance(parent.test, ast.Constant)):
        return False
    if child in parent.body:
        return not parent.test.value
    return isinstance(parent, ast.If) and child in parent.orelse and bool(parent.test.value)


def _nodes_that_run(root: ast.AST) -> Iterator[ast.AST]:
    """The nodes of `root`, in `ast.walk` order, that run when `root` runs.
    Code under an if, loop, try or with counts. A function or lambda defined
    inside `root` runs only when called, so its code doesn't count, and nor
    does code in a branch that never runs (`_never_runs`)."""
    todo = deque([root])
    while todo:
        node = todo.popleft()
        if node is not root and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        todo.extend(c for c in ast.iter_child_nodes(node) if not _never_runs(parent=node, child=c))
        yield node


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


def _in_value(node: ast.AST, binding: _Binding) -> bool:
    """Whether `node` sits in the value `binding` gives its name, as the
    right side of `exec = exec` does: it runs before the binding."""
    value = binding.value
    if not getattr(value, "lineno", 0) <= getattr(node, "lineno", 0) <= (getattr(value, "end_lineno", 0) or 0):
        return False
    return node is value or _inside(node, value, binding.cell)


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

    def __init__(
        self, cells: list[_Cell], *, current: _Cell, project_root: Path | None = None,
        errored: frozenset[int] = frozenset(), errored_paths: Literal["any", "unknown"] = "any",
        stopped_early: frozenset[int] = frozenset(),
    ) -> None:
        self.cells = cells
        self.current = current
        # The turn's project root, to read a literal relative SQL-file path.
        self.project_root = project_root
        # The earlier cells that stopped on an error, by index, for a reader
        # that reads them as maybe stopped anywhere: a path read through a
        # name one binds may hold any value it gives the name, or the one
        # before it ("any"), or be one the check can't place ("unknown").
        # Empty for the reader that reads every earlier cell as run to its end.
        self.errored = errored
        self.errored_paths = errored_paths
        # The earlier cells that stopped on an error, by index, for the
        # names they bind: a definition in one may not have replaced the one
        # before it (`replaced`), and a name only one binds may never have
        # been bound (`binds`).
        self.stopped_early = stopped_early
        # The "any" reading gives a name every value each errored cell gives
        # it, so a path built from several names grows fast: past
        # `_PATH_LIMIT` paths for one value, or past `_WIDE_BINDINGS`
        # bindings read, it gives up (`past_limit`). It reads each name once
        # per read site (`_wide_names`).
        self.wide = bool(errored) and errored_paths == "any"
        self.past_limit = False
        # How many times a depth limit stopped a read (`stops`), and whether
        # such a stop may have hidden a write (`stopped`), which leaves the
        # cell unchecked. While a read goes on past its limits (`read_past`),
        # the nodes each read reached, by the read and the node's id.
        self.cuts = 0
        self.stopped = False
        self._past: set[tuple[str, int]] | None = None
        self.bindings_read = 0
        self._wide_names: dict[tuple[int, int, tuple[tuple[int, str, int], ...]], list[str] | _Unknown] = {}
        self._cells_by_node: dict[ast.AST, _Cell] = {}
        self.imports: dict[str, str] = {}
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        # The names each lambda is bound to, by the lambda's id.
        self.lambda_names: dict[int, list[str]] = {}
        # The cells whose names land in the cell namespace, in the order they
        # run: literal code an exec() runs right after the cell that runs it
        # (`_Cell.host`). The tables above take each name's latest.
        self.in_order = sorted((c for c in cells if c.shared), key=lambda c: (c.host, c.index))
        self._places = [(c.host, c.index) for c in self.in_order]
        # The cells that bind each name at their top level, in that order.
        self._binders: dict[str, list[_Cell]] = {}
        for cell in self.in_order:
            for name in cell.module_bound:
                self._binders.setdefault(name, []).append(cell)
        # The checked cell and the literal code it runs.
        self.checked = [c for c in cells if c.host >= current.index]
        for cell in cells:
            self._cells_by_node[cell.tree] = cell
            self._cells_by_node.update(dict.fromkeys(cell.parents, cell))
            self.lambda_names.update(cell.lambda_names)
        # The definitions a later one in a cell that stopped early may not
        # have replaced, by the later one's id.
        self.replaced: dict[int, list[Function | ast.ClassDef]] = {}
        for cell in self.in_order:
            if cell.host in stopped_early:
                for table, defined in ((self.functions, cell.functions), (self.classes, cell.classes)):
                    for name, node in defined.items():
                        if name in table:
                            self.replaced[id(node)] = [table[name], *self.replaced.get(id(table[name]), [])]
            self.imports.update(cell.imports)
            self.functions.update(cell.functions)
            self.classes.update(cell.classes)
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
            n.func.id for cell in self.checked for n in cell.calls
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

    def read_too(self, cell: _Cell) -> None:
        """Reads the names of `cell`, literal code a cell runs, with the
        rest, so the literal code it runs in turn resolves through its own
        imports while `_exec_cells` reads on. The check then reads every
        such cell with a reader of its own."""
        self._cells_by_node[cell.tree] = cell
        self._cells_by_node.update(dict.fromkeys(cell.parents, cell))
        self.imports.update(cell.imports)
        self.bound_names.update(cell.bindings)

    def before(self, cell: _Cell) -> list[_Cell]:
        """The cells whose names land in the cell namespace that run before
        `cell`, in the order they run."""
        return self.in_order[: bisect.bisect_left(self._places, (cell.host, cell.index))]

    def earlier(self, node: ast.AST) -> bool:
        """Whether `node` sits in code that ran before the checked cell: an
        earlier cell, or literal code one ran."""
        return self.cell_of(node).host < self.current.index

    def hole(self, node: ast.AST | None) -> str:
        """The placeholder for a part the code picks at run time: `node`'s
        own number, so `show_holes` can put its code back."""
        if node is not None and id(node) in self._hole_numbers:
            return f"{_HOLE}{self._hole_numbers[id(node)]}{_HOLE_END}"
        self._holes.append(node)
        if node is not None:
            self._hole_numbers[id(node)] = len(self._holes) - 1
        return f"{_HOLE}{len(self._holes) - 1}{_HOLE_END}"

    def hole_node(self, text: str) -> ast.AST | None:
        """The code that fills in `text` when it is one placeholder and
        nothing else, an f-string's field as the code in its braces; None
        for any other text, or a placeholder no code fills."""
        m = _NUMBERED_HOLE.fullmatch(text)
        if m is None or int(m.group(1)) >= len(self._holes):
            return None
        node = self._holes[int(m.group(1))]
        return node.value if isinstance(node, ast.FormattedValue) else node

    def hole_code(self, text: str) -> str | None:
        """When `text` is one placeholder and nothing else, the code that
        fills it in, without braces ("" when there is none to show); None
        when `text` holds anything written out."""
        m = _NUMBERED_HOLE.fullmatch(text)
        if m is None:
            return None
        node = self._holes[int(m.group(1))] if int(m.group(1)) < len(self._holes) else None
        if node is None:
            return ""
        try:
            return ast.unparse(node.value if isinstance(node, ast.FormattedValue) else node)
        except Exception:
            return ""

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

    @cached_property
    def names_a_routine(self) -> bool:
        """Whether any cell's parsed code names `callproc` or `callfunc`, so a
        name may hold a stored procedure or function (`_routine_of`)."""
        return any(cell.names_a_routine for cell in self.cells)

    @cached_property
    def routine_names(self) -> frozenset[str]:
        """The names, and `h.run` members, that some cell binds in any scope
        to a value naming `callproc` or `callfunc` or reading another such
        name: the only ones `_routine_of` reads the bindings of."""
        if not self.names_a_routine:
            return frozenset()
        values: list[tuple[str, ast.AST]] = []
        for cell in self.cells:
            values += [(name, b.value) for name, found in cell.bindings.items() for b in found
                       if b.how in ("value", "element", "unpacked")]
            values += [(name, g.value) for name, grown in cell.grows.items() for g in grown]
            values += [(f"{base}.{attr}", v) for (base, attr), stored in cell.member_values.items() for v in stored]
        held: set[str] = set()
        # The names whose values read each name.
        readers: dict[str, set[str]] = {}
        for name, value in values:
            for node in ast.walk(value):
                if isinstance(node, ast.Attribute) and node.attr in _ROUTINE_CALLS or (
                    isinstance(node, ast.Constant) and node.value in _ROUTINE_CALLS
                ):
                    held.add(name)
                elif isinstance(node, ast.Name):
                    readers.setdefault(node.id, set()).add(name)
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    readers.setdefault(f"{node.value.id}.{node.attr}", set()).add(name)
        todo = list(held)
        while todo:
            for name in readers.get(todo.pop(), ()):
                if name not in held:
                    held.add(name)
                    todo.append(name)
        return frozenset(held)

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
        if func is None or not self.earlier(func):
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
        that defines it. It stops at the depth limit (`stops`)."""
        if self.stops("callees", node, depth=depth, limit=6):
            return []
        if isinstance(node, ast.Lambda):
            return [_Callee(func=node, call=call, use=use)] if self.earlier(node) else []
        if isinstance(node, ast.Name):
            if node.id in self.function_names:
                func = self.earlier_function(node)
                if func is not None:
                    return [_Callee(func=f, call=call, use=use) for f in (func, *self.replaced.get(id(func), ()))
                            if not isinstance(f, ast.ClassDef)]
            cls = self._earlier_class(node=node)
            if cls is not None:
                # Calling a class runs its __init__ on the new instance.
                return [c for k in (cls, *self.replaced.get(id(cls), ())) if isinstance(k, ast.ClassDef)
                        for c in self._methods(cls=k, name="__init__", bound=call or node, call=call, use=use)]
            if node.id not in self.bound_names:
                return []
            return [c for b in self.bindings_of(node.id, node=node) if b.how == "value"
                    for c in self.callees(node=b.value, call=call, use=use, depth=depth + 1)]
        if isinstance(node, ast.Subscript):
            items = self.items_of(node=node.value, key=node.slice, depth=depth + 1) or []
            return [c for item in items for c in self.callees(node=item, call=call, use=use, depth=depth + 1)]
        if isinstance(node, ast.Attribute) and node.attr in self.method_names:
            receiver = node.value
            if isinstance(receiver, ast.Name) and receiver.id in self.imports:
                return []
            cls = self._earlier_class(node=receiver)
            if cls is not None:
                return self._methods(cls=cls, name=node.attr, bound=receiver, on_class=True, call=call, use=use)
            classes = self._instance_classes(node=receiver, depth=depth + 1)
            found = [c for k in (self.classes.values() if classes is None else classes)
                     for c in self._methods(cls=k, name=node.attr, bound=receiver, call=call, use=use)]
            # A receiver the check can't place: a guess at every class.
            return found if classes is not None else [replace(c, guessed=True) for c in found]
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
                    for ret in self.returns(func=made.func)
                    for c in self.callees(node=ret, call=call, use=use, depth=depth + 1)]
        return []

    def _earlier_class(self, *, node: ast.AST) -> ast.ClassDef | None:
        """The class an earlier cell defined that the name `node` reads."""
        if not isinstance(node, ast.Name):
            return None
        cls = self.classes.get(node.id)
        if cls is None or not self.earlier(cls):
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
        method, or in a plain one called on the class (`on_class`). It stops
        at the depth limit (`stops`)."""
        func = _class_member(cls=cls, name=name)
        if func is not None:
            if not self.earlier(func):
                return []
            kind = _method_kind(func=func, cell=self.cell_of(func))
            gets_bound = kind == "class" or kind == "plain" and not on_class
            return [_Callee(func=func, call=call, use=use, bound=bound if gets_bound else None)]
        for base in cls.bases:
            parent = self._earlier_class(node=base)
            if parent is not None and parent is not cls and not self.stops("methods", base, depth=depth, limit=5):
                found = self._methods(cls=parent, name=name, bound=bound, call=call, use=use, on_class=on_class, depth=depth + 1)
                if found:
                    return found
        return []

    def returns(self, *, func: Function) -> list[ast.expr]:
        """What `func` itself returns: a lambda's body, or the value of each
        return statement in the function, not in a function inside it."""
        if isinstance(func, ast.Lambda):
            return [func.body]
        cell = self.cell_of(func)
        return [n.value for n in ast.walk(func)
                if isinstance(n, ast.Return) and n.value is not None and cell.scope_of(n) is func]

    def items_of(self, *, node: ast.AST, key: ast.AST | None, depth: int) -> list[ast.AST] | None:
        """The elements of the list, tuple, set or dict values `node` holds,
        with what the pad adds to it, only the one a literal `key` names when
        the literal says which; None when the check can't place them, or
        at the depth limit (`stops`)."""
        if self.stops("items", node, depth=depth, limit=6):
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
                got = self.items_of(node=b.value, key=key, depth=depth + 1) if b.how == "value" else None
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
                got = self.items_of(node=arg, key=None, depth=depth + 1)
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
                for ret in self.returns(func=callee.func):
                    got = self.items_of(node=ret, key=key, depth=depth + 1)
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
                    got = self._element_classes(items=self.items_of(node=b.value, key=None, depth=depth + 1), depth=depth)
                elif b.how == "param":
                    got = self._self_class(func=b.value, param=node.id)
                else:
                    got = None
                if got is None:
                    return None
                out += got
            return out
        if isinstance(node, ast.Subscript):
            return self._element_classes(items=self.items_of(node=node.value, key=node.slice, depth=depth + 1), depth=depth)
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

    def bindings_of(self, name: str, *, node: ast.AST, partly_run: bool = False) -> list[_Binding]:
        """What `name` holds where `node` reads it.

        Inside a function, the function's own bindings first. At module level,
        this cell's bindings before the read, else the latest earlier cell's,
        each from the last one that surely ran. A function body runs when it
        is called, so it sees every cell's module bindings, the latest first.
        A cell that binds the name only under an if, loop or try doesn't hide
        an older binding: both count. With `partly_run`, an earlier cell in
        `errored` may have stopped before any of its bindings, or after any:
        each of them counts, and so do the older ones.
        """
        cell = self.cell_of(node)
        scopes = cell.scopes_around(node)
        for scope in scopes:
            local = [b for b in cell.bindings.get(name, ()) if b.scope is scope]
            if local:
                return local
        stopped = self.errored if partly_run else frozenset()
        binders = self._binders.get(name, [])
        layers: Iterator[list[_Binding]]
        if scopes:
            layers = (self._module_bindings(c, name) for c in reversed(binders))
        else:
            line = getattr(node, "lineno", 0)
            here = [b for b in self._module_bindings(cell, name) if b.line <= line]
            place = (cell.host, cell.index)
            layers = itertools.chain([_since_last_sure(here, name=name)], (
                self._module_bindings(c, name) if c.index in stopped
                else _since_last_sure(self._module_bindings(c, name), name=name)
                for c in reversed(binders) if (c.host, c.index) < place
            ))
        found: list[_Binding] = []
        for layer in layers:
            found += layer
            # At module level, the line `node` is on ran, so a sure binding
            # before it in its own cell did too.
            if any(not b.conditional and (b.cell.index not in stopped or b.cell is cell and not scopes)
                   for b in layer):
                break
        return found

    @staticmethod
    def _module_bindings(cell: _Cell, name: str) -> list[_Binding]:
        return [b for b in cell.bindings.get(name, ()) if b.scope is None]

    def binds(self, node: ast.Name, *, surely: bool = False) -> bool:
        """Whether read code gives the name `node` reads a value where it
        reads it, hiding any builtin of that name: an import, an assignment,
        parameter, loop, with, comprehension or walrus (`bindings_of`), or a
        def or class in a scope around it or at module level, before it when
        `node` is at module level of the same cell. With `surely`, a cell
        that stopped early (`stopped_early`) binds nothing."""
        cell = self.cell_of(node)
        skip = self.stopped_early if surely else frozenset()
        if node.id in self.imports and (
            not skip or any(node.id in c.imports for c in (*self.before(cell), cell) if c.host not in skip)
        ):
            return True
        if node.id in self.bound_names and self._bound_where(node, skip=skip):
            return True
        if not any(node.id in names for names in (self.functions, self.classes, cell.functions, cell.classes)):
            return False
        scopes = cell.scopes_around(node)
        for other in (*self.before(cell), cell):
            if other.host in skip:
                continue
            for defined in (other.functions.get(node.id), other.classes.get(node.id)):
                if defined is None:
                    continue
                scope = other.scope_of(defined)
                if scope in scopes or scope is None and (
                    other is not cell or scopes or _follows(node=node, value=defined)
                ):
                    return True
        return False

    def _bound_where(self, node: ast.Name, *, skip: frozenset[int]) -> bool:
        """Whether a binding of the name `node` reads is where it reads it
        (`bindings_of` finds one): in a scope around it, else at module
        level in its cell before it or in a cell that runs before it, or in
        any cell for a read inside a function. A read inside the value a
        binding gives, as the right side of `exec = exec` is, runs before
        that binding. A cell in `skip` binds nothing."""
        name = node.id
        cell = self.cell_of(node)
        scopes = cell.scopes_around(node)
        own = cell.bindings.get(name, ())
        if any(b.scope is not None and b.scope in scopes for b in own):
            return True
        binders = self._binders.get(name, [])
        if scopes:
            return any(c.host not in skip for c in binders)
        line = getattr(node, "lineno", 0)
        if cell.host not in skip and any(
            b.scope is None and b.line <= line and not _in_value(node, b) for b in own
        ):
            return True
        place = (cell.host, cell.index)
        return any((c.host, c.index) < place and c.host not in skip for c in binders)

    @cached_property
    def deleted(self) -> frozenset[str]:
        """The names a `del` statement in any cell unbinds (`_CodeSigns.deleted`)."""
        return frozenset(name for cell in self.cells for name in cell.signs.deleted)

    @cached_property
    def names_builtins(self) -> bool:
        """Whether any cell names exec, eval, compile or the builtins module
        at all (`_CodeSigns.names_builtins`): with none, nothing reaches those
        builtins, and `exec_builtin` and `builtins_lookup` read nothing."""
        return any(cell.signs.names_builtins for cell in self.cells)

    def exec_builtin(self, node: ast.AST) -> str | None:
        """The builtin exec, eval or compile `node` reads, by name: by its
        own name where no read code binds the name (`binds`), by a name
        imported from builtins, as a member of the builtins module
        (`builtins.exec`, `b.exec` after `import builtins as b`,
        `__builtins__.exec`, and `getattr(builtins, "exec")`, which `_parse`
        reads as `builtins.exec`), or as an entry of its dict
        (`__builtins__["exec"]`, `vars(builtins)["exec"]`,
        `builtins.__dict__["exec"]`). Where `node` is called, also a name an
        assignment in any cell binds to one (`run = exec`), or an entry of a
        list or dict that holds one. None for anything else, such as
        `re.compile`, `df.eval`, a method, or `compile` after
        `from re import compile`. A read the depth limit stopped short of
        one stops the check (`read_past`)."""
        if not self.names_builtins:
            return None
        return self.read_through(lambda: self._exec_found(node))

    def _exec_found(self, node: ast.AST) -> str | None:
        """The builtin `exec_builtin` names: `node`'s own, or, where it is
        called, one a name or an entry holds (`_exec_held`)."""
        builtin = self._exec_named(node)
        parent = self.cell_of(node).parents.get(node)
        if builtin is None and isinstance(parent, ast.Call) and parent.func is node:
            builtin = self._exec_held(node, depth=0)
        return builtin

    def _exec_held(self, node: ast.AST, *, depth: int) -> str | None:
        """The builtin `exec_builtin` names that a name's assigned values, or
        the entries of a list or dict, hold. It stops at the depth limit
        (`stops`)."""
        if self.stops("exec", node, depth=depth, limit=_PARAM_DEPTH):
            return None
        if isinstance(node, ast.Name):
            if node.id in self.imports or node.id not in self.bound_names:
                return None
            values: list[ast.AST] = [b.value for b in self.bindings_of(node.id, node=node) if b.how == "value"]
        elif isinstance(node, ast.Subscript):
            values = self.items_of(node=node.value, key=node.slice, depth=0) or []
        else:
            return None
        return next(
            (found for v in values for found in [self._exec_named(v) or self._exec_held(v, depth=depth + 1)] if found),
            None,
        )

    def _exec_named(self, node: ast.AST) -> str | None:
        """The builtin `exec_builtin` names that `node` reads by its own name,
        or as a member or entry of the builtins module."""
        if isinstance(node, ast.Name):
            if node.id in _EXEC_CALLS and self._builtin_kept(node):
                return node.id
            head, _, name = self.imports.get(node.id, "").rpartition(".")
            return name if head == "builtins" and name in _EXEC_CALLS else None
        if isinstance(node, ast.Attribute):
            return node.attr if node.attr in _EXEC_CALLS and self._builtins_module(node.value) else None
        if isinstance(node, ast.Subscript):
            key = node.slice
            if isinstance(key, ast.Constant) and key.value in _EXEC_CALLS and self._builtins_dict(node.value):
                return key.value
        return None

    def _builtin_kept(self, node: ast.Name) -> bool:
        """Whether the bare name `node` reads may still hold the builtin of
        that name: no read code binds it (`binds`), a `del` unbinds it, or
        no binding of it that reaches the read surely ran. A binding under
        an if, loop, try or match, a loop's own variable and an import under
        a branch may not have run. A def, a class, and a name the function
        the read is in binds replace the builtin surely."""
        name = node.id
        if name in self.deleted or not self.binds(node):
            return True
        cell = self.cell_of(node)
        scopes = cell.scopes_around(node)
        if any(b.scope is not None and b.scope in scopes for b in cell.bindings.get(name, ())):
            return False
        if name in self.functions or name in self.classes:
            return False
        if any(
            name in (a.asname or a.name.partition(".")[0] for a in stmt.names) and not c.under_branch(stmt)
            for c in self.cells for stmt in c.signs.imports
        ):
            return False
        return not any(not b.conditional and b.how != "element" for b in self.bindings_of(name, node=node))

    def builtins_lookup(self, node: ast.AST) -> bool:
        """Whether `node` looks a member of the builtins module up by a name
        the check can't read, or reads as exec, eval or compile:
        `getattr(builtins, name)` that `_parse` left as it is, or an entry of
        the module's dict (`_builtins_dict`) under any key but a literal,
        which `exec_builtin` reads. A read the depth limit stopped short of
        the module stops the check (`read_past`)."""
        if not self.names_builtins:
            return False
        key = self.read_through(lambda: self._builtins_key(node))
        if key is None:
            return False
        names = self.paths(key, cell=self.cell_of(node))
        if isinstance(names, _Unknown) and isinstance(key, ast.Subscript):
            # A name picked from a list or dict of literals, as a type map gives one.
            items = self.items_of(node=key.value, key=key.slice, depth=0)
            texts = [i.value for i in items or () if isinstance(i, ast.Constant) and isinstance(i.value, str)]
            if items is not None and len(texts) == len(items):
                names = texts
        return isinstance(names, _Unknown) or any(
            n in _EXEC_CALLS or any(ch in n for ch in "*?[" + _RUN_TIME_PART) for n in names
        )

    def _builtins_key(self, node: ast.AST) -> ast.AST | None:
        """The key `node` looks a member of the builtins module up by, as
        `builtins_lookup` reads one: `getattr(builtins, key)`, or an entry
        of the module's dict under any key but a literal. None for anything
        else."""
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
                and len(node.args) >= 2 and self._builtins_module(node.args[0])):
            return node.args[1]
        if (isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Constant)
                and self._builtins_dict(node.value)):
            return node.slice
        return None

    def _builtins_module(self, node: ast.AST, *, depth: int = 0) -> bool:
        """Whether `node` holds the builtins module: `builtins` under any
        import name, `__builtins__`, `sys.modules["builtins"]`,
        `globals()["__builtins__"]`, or a name bound to one of them, as
        `m = __import__("builtins")` is once `_parse` reads the import as
        the plain name. It stops at the depth limit (`stops`)."""
        if self.stops("builtins", node, depth=depth, limit=_PARAM_DEPTH):
            return False
        if isinstance(node, ast.Name) and node.id not in self.imports and node.id in self.bound_names:
            found = self.bindings_of(node.id, node=node)
            if found:
                return any(b.how == "value" and self._builtins_module(b.value, depth=depth + 1) for b in found)
        if isinstance(node, ast.Call):
            # `importlib.import_module("built" + "ins")` or `__import__(n)`
            # with `n` bound to that text.
            texts = self._importer_module_texts(node)
            return texts is not None and any(text == "builtins" for text in texts)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            key, holder = node.slice.value, node.value
            return key == "builtins" and self.dotted(holder) == "sys.modules" or key == "__builtins__" and (
                isinstance(holder, ast.Call) and self.dotted(holder.func) == "globals"
            )
        return self.dotted(node) in ("builtins", "__builtins__")

    def _builtins_dict(self, node: ast.AST) -> bool:
        """Whether `node` holds the builtins module's dict: `__builtins__`
        (the dict itself outside the main module), `vars(builtins)` or
        `builtins.__dict__`."""
        if isinstance(node, ast.Call):
            return self.dotted(node.func) == "vars" and len(node.args) == 1 and self._builtins_module(node.args[0])
        if isinstance(node, ast.Attribute) and node.attr == "__dict__":
            return self._builtins_module(node.value)
        return self.dotted(node) == "__builtins__"

    def loads_builtins_value(self, node: ast.AST) -> bool:
        """Whether `node` loads the builtins module, or its dict, as a value
        anywhere the check doesn't already read as reaching a literal exec,
        eval or compile: handed to a call, held in a list or dict, returned,
        or read through `.get`, `.__getattribute__` or another member. As
        `exec` itself, the builtins module loaded as a value may run any code
        the check can't read. Its safe uses are `builtins.exec` and its kin
        (read by `exec_builtin`), `builtins.__dict__`, `vars(builtins)` and
        an entry of either, and `getattr(builtins, ...)` (read by
        `builtins_lookup`)."""
        if not self.names_builtins:
            return False
        is_value = isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)) and isinstance(node.ctx, ast.Load)
        if not (is_value or isinstance(node, ast.Call)):
            return False
        as_module = is_value and self._builtins_module(node)
        as_dict = not as_module and self._builtins_dict(node)
        if not (as_module or as_dict):
            return False
        parent = self.cell_of(node).parents.get(node)
        if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            # Bound to a name (`b = builtins`); that name's own uses are checked.
            return False
        if isinstance(parent, ast.Subscript) and parent.value is node:
            # `__builtins__["exec"]`, read by `exec_builtin` or `builtins_lookup`.
            return False
        if as_module and isinstance(parent, ast.Attribute):
            # `builtins.exec`, read by `exec_builtin`, or `builtins.__dict__`.
            return not (parent.attr in _EXEC_CALLS or parent.attr == "__dict__")
        if as_module and isinstance(parent, ast.Call) and node in parent.args[:1]:
            # `vars(builtins)` or `getattr(builtins, ...)`.
            return self.dotted(parent.func) not in ("vars", "getattr")
        return True

    def module_member(self, node: ast.AST) -> str | None:
        """The dotted name `node` reads, as `dotted` spells it, when its root
        name holds a module the code imports: by an import, as the plain name
        a literal `importlib.import_module("x")` or `__import__("x")` reads
        as (`_Cell.looked_up`), or through names bound to either, as
        `rp = importlib.import_module("runpy")` binds `rp`. An entry of
        `sys.modules`, of `vars(m)` or of `m.__dict__`, under a literal key
        or a name bound to one (`_entry_key`), reads as the module or member
        it names. A name bound to a literal `importlib.import_module` or
        `__import__` reads as the module it loads. A builtin of the pad no
        read code binds counts as itself (`_PAD_BUILTINS`). None for a root
        nothing imports, as a local named `code` or `profile` is."""
        todo: list[tuple[ast.AST, tuple[str, ...]]] = [(node, ())]
        seen: set[int] = set()
        while todo:
            node, tail = todo.pop()
            parts: list[str] = []
            while isinstance(node, (ast.Attribute, ast.Subscript)):
                if isinstance(node, ast.Attribute):
                    parts.append(node.attr)
                    node = node.value
                    continue
                holder = node.value
                sys_mod = self.dotted(holder) == "sys.modules"
                through_dict = isinstance(holder, ast.Attribute) and holder.attr == "__dict__"
                through_vars = (isinstance(holder, ast.Call) and self.dotted(holder.func) == "vars"
                                and len(holder.args) == 1)
                key = self._entry_key(node.slice) if sys_mod or through_dict or through_vars else None
                if key is None:
                    break
                if sys_mod:
                    # `sys.modules["runpy"]`: the module by its full name.
                    return ".".join([key, *reversed(parts), *tail])
                # `runpy.__dict__["run_path"]`, read as `runpy.run_path`.
                parts.append(key)
                node = holder.value if through_dict else holder.args[0]
            rest = (*reversed(parts), *tail)
            if isinstance(node, ast.Call):
                texts = self._importer_module_texts(node)
                if texts:
                    return ".".join([self._pick_runner(texts), *rest])
                continue
            if not isinstance(node, ast.Name) or id(node) in seen:
                continue
            seen.add(id(node))
            head = self.imports.get(node.id)
            if head is None and (node in self.cell_of(node).looked_up
                                 or node.id in _PAD_BUILTINS and not self.binds(node)):
                head = node.id
            if head is not None:
                return ".".join([head, *rest])
            if not rest and node.id in self.star_runners and not self.binds(node):
                # `run_path` after `from runpy import *`.
                return self.star_runners[node.id]
            if node.id in self.bound_names:
                todo += [(b.value, rest) for b in self.bindings_of(node.id, node=node) if b.how == "value"]
        return None

    def _entry_key(self, slice_node: ast.AST) -> str | None:
        """The literal key an entry is read under: a string constant, or a
        name bound only to literal texts (`_literal_texts`), the one that
        names a runner holder where several could. None otherwise."""
        if isinstance(slice_node, ast.Constant):
            return slice_node.value if isinstance(slice_node.value, str) else None
        texts = _literal_texts(slice_node, self)
        return self._pick_runner(texts) if texts else None

    def _importer_module_texts(self, call: ast.AST) -> list[str] | None:
        """The modules a call to `importlib.import_module` or `__import__`
        loads, read from a target that is literal texts (`_literal_texts`),
        each `__import__`'s top package. None when `call` isn't one of those,
        or its target isn't literal texts."""
        if not (isinstance(call, ast.Call) and _member_name(call.func) in _MODULE_IMPORTER_LAST_NAMES):
            return None
        name = self.module_member(call.func) if _runner_named(call.func, self) else None
        if name not in _MODULE_IMPORTERS:
            return None
        target = _target_arg(call, keywords=("name",))
        texts = _literal_texts(target, self) if target is not None else None
        if not texts:
            return None
        return [t.partition(".")[0] for t in texts] if name.endswith("__import__") else texts

    @staticmethod
    def _pick_runner(texts: Sequence[str]) -> str:
        """The one of `texts` whose top package holds something that runs
        code (`_RUNNER_HOLDERS`), else the first: where a name holds several
        literal module names, the reader resolves it to the one that may run
        code."""
        return next((t for t in texts if t.partition(".")[0] in _RUNNER_HOLDERS), texts[0])

    @cached_property
    def star_runners(self) -> dict[str, str]:
        """The runners and launchers a `from module import *` of a module
        that holds one binds, by the bare name it binds each to
        (`_RUNNERS_BY_MODULE`): `run_path` after `from runpy import *`."""
        out: dict[str, str] = {}
        for cell in self.cells:
            for node in cell.signs.imports:
                if (isinstance(node, ast.ImportFrom) and node.module in _RUNNERS_BY_MODULE
                        and any(a.name == "*" for a in node.names)):
                    out.update(_RUNNERS_BY_MODULE[node.module])
        return out

    @cached_property
    def runner_aliases(self) -> frozenset[str]:
        """The names that may hold something that runs code by another name
        than its own: an import under another name, as
        `from runpy import run_path as rp` binds `rp`, and a name an
        assignment binds to a hint (`_CodeSigns.hint_aliases`)."""
        out = {name for cell in self.cells for name in cell.signs.hint_aliases}
        for name, full in self.imports.items():
            last = full.rpartition(".")[2]
            if name != last and _is_hint(last):
                out.add(name)
        return frozenset(out)

    def uncalled(self, func: Function) -> bool:
        """Whether nothing can call `func`: a def, or a lambda bound to
        names, that no decorator wraps and no class holds, whose names no
        cell loads or holds as a string, as `globals()["sh"]` does. The
        cell namespace is no module's, so no member names reach it."""
        if isinstance(func, ast.Lambda):
            names = self.lambda_names.get(id(func), [])
            if not names or not isinstance(self.cell_of(func).parents.get(func), (ast.Assign, ast.AnnAssign)):
                return False
        else:
            if func.decorator_list:
                return False
            names = [func.name]
        if isinstance(self.cell_of(func).scope_of(func), ast.ClassDef):
            return False
        return not any(name in cell.loads or name in cell.signs.spelled for cell in self.cells for name in names)

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

    # ── depth limits ─────────────────────────────────────────────────────

    def stops(self, read: str, node: ast.AST, *, depth: int, limit: int) -> bool:
        """Whether the read named `read` stops at `node`, which `cuts`
        counts: past `limit` levels, or, while it goes on past its limits
        (`read_past`), at a node it reached already, which ends a loop of
        bindings such as `df = df.dropna()` reading its own name. Past
        `_PAST_DEPTH` levels even then it gives up, which stops the check."""
        if self._past is None:
            if depth > limit:
                self.cuts += 1
                return True
            return False
        if depth > _PAST_DEPTH:
            self.stopped = True
            return True
        key = (read, id(node))
        if key in self._past:
            self.cuts += 1
            return True
        self._past.add(key)
        return False

    def read_past(self, mark: int, read: Callable[[], object]) -> None:
        """After a read that came back short: when a depth limit stopped it
        (`cuts` past `mark`), `read` reads again with no limit, each node
        once, and says whether that found what the first read missed. Then
        a write may be past the limit, unread, and the check stops
        (`stopped`). Nothing it finds goes on the card. The wide reading,
        which only adds files to the first read's, doesn't read past."""
        if self.cuts == mark or self._past is not None or self.wide:
            return
        self._past = set()
        try:
            if read():
                self.stopped = True
        finally:
            self._past = None

    def read_through(self, read: Callable[[], _T]) -> _T:
        """Runs `read`, and when a depth limit stopped it short of a truthy
        result (`read_past`), reads again with no limit; returns `read`'s own
        result either way. The one wrapper for the reads that record the cut
        count, read, and re-read on a falsy result."""
        mark = self.cuts
        result = read()
        if not result:
            self.read_past(mark, read)
        return result

    # ── paths ────────────────────────────────────────────────────────────

    def paths(self, node: ast.AST, *, cell: _Cell, depth: int = 0) -> list[str] | _Unknown:
        """The glob-style path patterns `node` can hold, `*` for a part picked at
        run time, or unknown when the check can't place its root."""
        if depth > 8 or self.past_limit:
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
                    inner = [_RUN_TIME_PART]
                if self._too_many(count=len(parts) * len(inner)):
                    return _UNKNOWN
                parts = [p + i for p in parts for i in inner]
            return parts
        if isinstance(node, ast.BinOp):
            left = self.paths(node.left, cell=cell, depth=depth + 1)
            if isinstance(left, _Unknown) or _TEMP_PATH in left:
                return left
            right = self.paths(node.right, cell=cell, depth=depth + 1)
            if isinstance(node.op, ast.Add) and (
                not left or self.sequence(node.left, node.right, cell=cell)
            ):
                # Two lists added: each element of either.
                return right if isinstance(right, _Unknown) else left + right
            if isinstance(right, _Unknown):
                right = [_RUN_TIME_PART]
            if self._too_many(count=len(left) * len(right)):
                return _UNKNOWN
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

    def _too_many(self, *, count: int) -> bool:
        """Whether `count` paths for one value pass `_PATH_LIMIT` on the wide
        reading, which then gives up whole (`past_limit`). Always False on
        any other reader."""
        if self.wide and count > _PATH_LIMIT:
            self.past_limit = True
        return self.past_limit

    def _name_paths(self, node: ast.Name, *, cell: _Cell, depth: int) -> list[str] | _Unknown:
        if not self.wide:
            return self._name_paths_read(node, cell=cell, depth=depth)
        key = (id(node), depth, self.call_key())
        if key not in self._wide_names:
            self._wide_names[key] = self._name_paths_read(node, cell=cell, depth=depth)
        got = self._wide_names[key]
        return got if isinstance(got, _Unknown) else list(got)

    def _name_paths_read(self, node: ast.Name, *, cell: _Cell, depth: int) -> list[str] | _Unknown:
        found = self.bindings_of(node.id, node=node, partly_run=True)
        if not found or (self.errored_paths == "unknown" and any(b.cell.index in self.errored for b in found)):
            return _UNKNOWN
        if self.wide:
            self.bindings_read += len(found)
            if self.bindings_read > _WIDE_BINDINGS:
                self.past_limit = True
                return _UNKNOWN
        out: list[str] = []
        unread = False
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
                if not self.wide:
                    return got
                # The wide reading only adds held files: one value it can't
                # read leaves the files the others name.
                unread = True
                continue
            out += got
        if not self.wide:
            return out
        out = list(dict.fromkeys(out))
        if (unread and not out) or self._too_many(count=len(out)):
            return _UNKNOWN
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
        Unknown when such a function passes the parameter on. At the depth
        limit it reads only the name's own additions (`stops`)."""
        out = self.growth(name, binding)
        if self.stops("growth", binding.value, depth=depth, limit=3):
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
                for func in self.pad_callables(node=call.func):
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

    def list_growth_for_write(self, *, name: str, binding: _Binding) -> list[tuple[_Growth, _Cell]] | _Unknown:
        """`list_growth` of a list whose elements a call that writes reads, as
        paths or a command's words: when the depth limit stopped it short of
        another name's additions, the check stops (`read_past`)."""
        mark = self.cuts
        out = self.list_growth(name=name, binding=binding)
        if isinstance(out, _Unknown):
            return out
        read = {id(grown) for grown, _ in out}

        def grows_past() -> bool:
            past = self.list_growth(name=name, binding=binding)
            return isinstance(past, _Unknown) or any(id(grown) not in read for grown, _ in past)

        self.read_past(mark, grows_past)
        return out

    def pad_callables(self, *, node: ast.expr) -> list[Function]:
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
        if any(self._binding_scope(cell, call, param) is func and self.pad_callables(node=call.func)
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
        sequence = self.sequence(binding.value, cell=binding.cell)
        here = self.cell_of(use)
        growth = self.list_growth_for_write(name=name, binding=binding) if sequence else self.growth(name, binding)
        if isinstance(growth, _Unknown):
            return growth
        for grown, cell in growth:
            got = self.paths(grown.value, cell=cell, depth=depth + 1)
            if isinstance(got, _Unknown):
                if sequence:
                    return got
                got = [_RUN_TIME_PART]
            if sequence:
                out += got
                continue
            if self._too_many(count=len(out) * len(got)):
                return _UNKNOWN
            # A cell that stopped on an error may have stopped before it built on the path.
            stopped = cell.index in self.errored
            built = [_join(b, p) if grown.how == "join" else b + p for b in out for p in got]
            ran = cell.index < here.index or cell is here and grown.site.lineno < getattr(use, "lineno", 0)
            # One in a function runs only if it is called.
            in_order = cell.scope_of(grown.site) is binding.scope
            sure = ran and in_order and not stopped and not cell._under_branch(grown.site, scope=binding.scope)
            out = built if sure else out + built
            if self._too_many(count=len(out)) or len(out) > _PATH_LIMIT:
                return _UNKNOWN
        return list(dict.fromkeys(out))

    def sequence(self, *nodes: ast.AST, cell: _Cell) -> bool:
        """Whether any of `nodes` is visibly a list, tuple or set of paths
        (`is_sequence`). A read the depth limit stopped short of one stops
        the check (`read_past`)."""
        return self.read_through(lambda: any(self.is_sequence(n, cell=cell) for n in nodes))

    def is_sequence(self, node: ast.AST, *, cell: _Cell, depth: int = 0) -> bool:
        """Whether `node` is visibly a list, tuple or set of paths, so `+`
        on it adds elements rather than text. It stops at the depth limit
        (`stops`)."""
        if self.stops("sequence", node, depth=depth, limit=6):
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
                more = [_RUN_TIME_PART] if isinstance(more, _Unknown) else more
                if self._too_many(count=len(out) * len(more)):
                    return _UNKNOWN
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
                if self._too_many(count=len(base) * len(pattern)):
                    return _UNKNOWN
                return [_join(b, deep + p) for b in base for p in pattern]
            if method == "iterdir":
                return [_join(b, "*") for b in base]
            if method in ("resolve", "absolute", "expanduser"):
                return base
            if method == "joinpath":
                out = base
                for a in args:
                    more = self.paths(a, cell=cell, depth=depth + 1)
                    more = [_RUN_TIME_PART] if isinstance(more, _Unknown) else more
                    if self._too_many(count=len(out) * len(more)):
                        return _UNKNOWN
                    out = [_join(p, m) for p in out for m in more]
                return out
            if method in ("with_name", "with_suffix", "with_stem"):
                part = self.paths(args[0], cell=cell, depth=depth + 1) if args else _UNKNOWN
                parts = [_RUN_TIME_PART] if isinstance(part, _Unknown) else part
                if self._too_many(count=len(base) * len(parts)):
                    return _UNKNOWN
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
        on it are file calls rather than a DataFrame's or a str's. It stops
        at the depth limit (`stops`)."""
        if self.stops("path", node, depth=depth, limit=6):
            return False
        if isinstance(node, ast.Name):
            arg = self.argument_of(node)
            if arg is not None:
                return self.is_path(arg.node, cell=arg.cell, depth=depth + 1)
            return any(
                b.how in ("value", "element") and self.is_path(b.value, cell=b.cell, depth=depth + 1)
                for b in self.bindings_of(node.id, node=node, partly_run=True)
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

    # ── SQL read from a file ───────────────────────────────────────────────

    def read_sql_file(self, path: str) -> str | None:
        """The text of the file a literal path names, read at check time to
        check its statements; None when no file sits there.

        An absolute path is read as it is, a relative one from the turn's
        project root. Past the 1 MB cap it reads as None here, and
        `oversized_sql_file` holds the read unchecked instead.
        """
        if self.project_root is None and not os.path.isabs(path):
            return None
        full = Path(path) if os.path.isabs(path) else self.project_root / path
        try:
            if not full.is_file() or full.stat().st_size > _SQL_FILE_CAP:
                return None
            return full.read_text(errors="replace")
        except OSError:
            return None

    def oversized_sql_file(self, path: str) -> bool:
        """Whether a literal path names a file past the 1 MB cap."""
        if self.project_root is None and not os.path.isabs(path):
            return False
        full = Path(path) if os.path.isabs(path) else self.project_root / path
        try:
            return full.is_file() and full.stat().st_size > _SQL_FILE_CAP
        except OSError:
            return False


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

# Calls that decode a literal into text at run time: the check decodes it now
# and checks its statements, so SQL hidden in one still shows.
_DECODERS = frozenset({
    "base64.b64decode", "base64.standard_b64decode", "base64.urlsafe_b64decode",
    "base64.b32decode", "base64.b16decode", "bytes.fromhex", "codecs.decode",
})
# Encodings `codecs.decode(literal, enc)` is read through, text in and out.
_CODEC_ENCODINGS = frozenset({"hex", "hex_codec", "rot13", "rot_13", "rot-13", "base64", "base_64", "base64_codec"})
# A SQL file read whole past this many bytes is held unchecked, not read.
_SQL_FILE_CAP = 1_000_000
_FILE_READERS = frozenset({"read", "read_text"})
_OPEN_CALLS = frozenset({"open", "io.open"})


def _decoded_text(name: str, literal: ast.AST, encoding: ast.AST | None) -> str | None:
    """`name(literal)` decoded at check time, as the cell would decode it; None
    when `literal` isn't a string or bytes literal, or the decode fails."""
    import base64
    import codecs

    if not (isinstance(literal, ast.Constant) and isinstance(literal.value, (str, bytes))):
        return None
    value = literal.value
    try:
        if name == "bytes.fromhex":
            return bytes.fromhex(value if isinstance(value, str) else value.decode()).decode("utf-8", "replace")
        if name == "codecs.decode":
            enc = encoding.value if isinstance(encoding, ast.Constant) and isinstance(encoding.value, str) else ""
            if enc.lower().replace("-", "_") not in {e.replace("-", "_") for e in _CODEC_ENCODINGS}:
                return None
            got = codecs.decode(value, enc)
            return got if isinstance(got, str) else got.decode("utf-8", "replace")
        fn = {
            "base64.b64decode": base64.b64decode, "base64.standard_b64decode": base64.standard_b64decode,
            "base64.urlsafe_b64decode": base64.urlsafe_b64decode,
            "base64.b32decode": base64.b32decode, "base64.b16decode": base64.b16decode,
        }[name]
        return fn(value).decode("utf-8", "replace")
    except (ValueError, TypeError, UnicodeError, LookupError, KeyError):
        return None


def _file_path_literal(node: ast.AST, reader: _Reader) -> str | None:
    """The literal path `open('x')` or `Path('x')` names; None otherwise."""
    if not isinstance(node, ast.Call) or not node.args:
        return None
    name = reader.dotted(node.func)
    if name in _OPEN_CALLS or name in _PATH_TYPES:
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return None


def _sql_file_path(node: ast.Call, reader: _Reader) -> str | None:
    """The literal path a file read names: `open('x').read()` or
    `Path('x').read_text()`; None when the call isn't one."""
    if not (isinstance(node.func, ast.Attribute) and node.func.attr in _FILE_READERS and not node.args):
        return None
    receiver = node.func.value
    if node.func.attr == "read" and not (isinstance(receiver, ast.Call)
                                         and reader.dotted(receiver.func) in _OPEN_CALLS):
        return None
    return _file_path_literal(receiver, reader)


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
    name = reader.dotted(node.func)
    if name in _SQL_WRAPPERS and node.args:
        return _sql_text(node.args[0], reader, parts=parts, depth=depth + 1)
    if name in _DECODERS and node.args:
        decoded = _decoded_text(name, node.args[0], node.args[1] if len(node.args) > 1 else None)
        if decoded is not None:
            parts.append(node)
            return decoded
    path = _sql_file_path(node, reader)
    if path is not None:
        text = reader.read_sql_file(path)
        if text is not None:
            parts.append(node)
        return text
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
        if root is not None and (not reader.earlier(b.value) or own and _inside(b.value, root, cell)):
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
    root = _root_of(target)
    return root.id if isinstance(root, ast.Name) else None


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
    if reader.earlier(func) or isinstance(cell.scope_of(func), ast.ClassDef):
        return None
    if isinstance(func, ast.Lambda):
        names = cell.lambda_names.get(id(func), [])
    else:
        names = [] if func.decorator_list else [func.name]
    if not names:
        return None
    calls: list[ast.Call] = []
    for checked in reader.checked:
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
        checked = cell.host >= reader.current.index
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
        builtin = reader.exec_builtin(parent.func)
        whole = _literal_source(value) is not None or isinstance(value, (ast.Name, ast.Attribute, ast.Subscript))
        if builtin is not None and whole and _exec_source(parent, builtin) is value:
            # Python source exec(), eval() or compile() takes whole is no SQL
            # statement, though `exec(...)` reads as a T-SQL EXEC; a statement
            # spliced into the source, as by `repr(q)`, may be one it runs.
            return _Reach.NOWHERE
        name = reader.dotted(parent.func)
        if name in _SQL_WRAPPERS or name in _PASS_THROUGH:
            return wrapped(name, parent)
        if name in ("map", "filter") and len(parent.args) > 1 and node is not parent.args[0]:
            return into(parent.args[0], parent)
        func = parent.func
        if isinstance(func, ast.Attribute) and func.attr == "join":
            joined = _sql_text(parent, reader) if not reader.earlier(parent) else None
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


def _to_sql_statement(call: ast.Call) -> str:
    """A DataFrame `to_sql` call as the card names it: the table it writes,
    or "a table" when the code picks the name when it runs."""
    table = call.args[0] if call.args else _keyword(call, "name")
    label = table.value if isinstance(table, ast.Constant) and isinstance(table.value, str) else "a table"
    return f"to_sql into {label}"


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
        out.append(_sql_write(_to_sql_statement(node), reader=reader,
                              line=report.line if report else cell.host_line or node.lineno))
    return out


# ── Call findings (no SQL text) ─────────────────────────────────────────────

# A DB-API cursor's stored procedure and stored function calls.
_ROUTINE_CALLS = frozenset({"callproc", "callfunc"})
# pymongo methods distinctive enough to hold on any receiver.
_MONGO_WRITES = frozenset({
    "insert_one", "insert_many", "update_one", "update_many", "replace_one", "delete_one", "delete_many",
    "bulk_write", "find_one_and_update", "find_one_and_replace", "find_one_and_delete",
    "drop_collection", "drop_database",
})
# boto3 DynamoDB writes, distinctive enough to hold on any receiver.
_DYNAMO_WRITES = frozenset({
    "put_item", "update_item", "delete_item", "batch_write_item", "transact_write_items",
})
# Table calls DynamoDB's and BigQuery's clients share: held on any receiver,
# and named for the client the receiver traces to.
_TABLE_WRITES = frozenset({"delete_table", "create_table"})
# BigQuery client writes distinctive enough to hold on any receiver.
_BIGQUERY_DISTINCT = frozenset({
    "delete_dataset", "insert_rows_json", "load_table_from_dataframe", "load_table_from_json",
    "load_table_from_file", "load_table_from_uri",
})
# A BigQuery write held only on a BigQuery client, since an openpyxl
# worksheet's `insert_rows` changes only the workbook in memory.
_BIGQUERY_WRITES = frozenset({"insert_rows"})
# Elasticsearch writes distinctive enough to hold on any receiver.
_ES_DISTINCT = frozenset({"delete_by_query", "update_by_query"})
# Redis writes, held only on a redis client, since the names are shared.
_REDIS_WRITES = frozenset({
    "flushdb", "flushall", "set", "delete", "hset", "hdel", "lpush", "rpush",
    "sadd", "srem", "zadd", "zrem", "expire", "incr",
})
# Elasticsearch writes held only on an Elasticsearch client, since the names are shared.
_ES_WRITES = frozenset({"index", "delete", "update", "bulk"})
# SQLAlchemy session writes that take effect on a commit or flush in the cell.
_SA_SESSION_WRITES = frozenset({"add", "add_all", "delete", "merge", "bulk_save_objects"})
_SA_CONSTRUCTS = frozenset({"insert", "update", "delete"})
_SA_SESSION_MAKERS = frozenset({
    "Session", "sessionmaker", "scoped_session", "async_sessionmaker", "AsyncSession",
})
_ENGINE_NAMES = frozenset({"engine", "conn", "connection", "bind", "con", "e", "eng", "cx"})
_ENGINE_MAKERS = frozenset({"sqlalchemy.create_engine", "create_engine"})


def _constructor_kind(name: str) -> CallVia | None:
    """The store a constructor opens: `redis`, `elasticsearch`, `mongodb` or `bigquery`."""
    parts = name.split(".")
    last = parts[-1]
    if "redis" in parts and last in ("Redis", "StrictRedis", "from_url"):
        return "redis"
    if "elasticsearch" in parts and last in ("Elasticsearch", "AsyncElasticsearch"):
        return "elasticsearch"
    if last == "MongoClient":
        return "mongodb"
    if "bigquery" in parts and last == "Client":
        return "bigquery"
    return None


def _opens_dynamodb(call: ast.Call) -> bool:
    """Whether `call` is boto3's `client("dynamodb")` or `resource("dynamodb")`,
    on the module or a session."""
    if not (isinstance(call.func, ast.Attribute) and call.func.attr in ("client", "resource")):
        return False
    service = call.args[0] if call.args else _keyword(call, "service_name")
    return isinstance(service, ast.Constant) and service.value == "dynamodb"


def _client_kind(node: ast.AST, reader: _Reader, *, depth: int = 0) -> CallVia | None:
    """The store a receiver belongs to, followed through subscripts, attribute
    chains, bindings and call arguments to the constructor that opened it.
    It stops at the depth limit (`_Reader.stops`)."""
    if reader.stops("client", node, depth=depth, limit=8):
        return None
    if isinstance(node, ast.Call):
        kind = _constructor_kind(reader.dotted(node.func))
        if kind is not None:
            return kind
        if _opens_dynamodb(node):
            return "dynamodb"
        return _client_kind(node.func.value, reader, depth=depth + 1) if isinstance(node.func, ast.Attribute) else None
    if isinstance(node, (ast.Subscript, ast.Attribute)):
        return _client_kind(node.value, reader, depth=depth + 1)
    if isinstance(node, ast.Name):
        arg = reader.argument_of(node)
        if arg is not None:
            return _client_kind(arg.node, reader, depth=depth + 1)
        for b in reader.bindings_of(node.id, node=node):
            if b.how in ("value", "context", "element"):
                kind = _client_kind(b.value, reader, depth=depth + 1)
                if kind is not None:
                    return kind
    return None


def _is_sa_construct(arg: ast.AST, reader: _Reader, *, depth: int = 0) -> bool:
    """Whether `arg` is a SQLAlchemy `insert`/`update`/`delete` construct or a
    `table.insert()`/`update()`/`delete()`, directly, through a name, or at the
    base of a `.where(...).values(...)` builder chain. It stops at the depth
    limit (`_Reader.stops`)."""
    if reader.stops("construct", arg, depth=depth, limit=8):
        return False
    if isinstance(arg, ast.Call):
        parts = reader.dotted(arg.func).split(".")
        if parts[0] == "sqlalchemy" and parts[-1] in _SA_CONSTRUCTS:
            return True
        if isinstance(arg.func, ast.Attribute):
            if arg.func.attr in _SA_CONSTRUCTS and not arg.args and not arg.keywords:
                return True
            return _is_sa_construct(arg.func.value, reader, depth=depth + 1)
        return False
    if isinstance(arg, ast.Name):
        return any(b.how == "value" and _is_sa_construct(b.value, reader, depth=depth + 1)
                   for b in reader.bindings_of(arg.id, node=arg))
    return False


def _engine_arg(call: ast.Call, reader: _Reader) -> bool:
    """Whether `.drop(x)`/`.create(x)` passes an engine or connection, so it is
    a SQLAlchemy table drop or create, not a DataFrame's `.drop('col')`."""
    arg = call.args[0] if call.args else next(
        (k.value for k in call.keywords if k.arg in ("bind", "engine", "con", "connection")), None
    )
    if isinstance(arg, ast.Name):
        if arg.id.lower() in _ENGINE_NAMES:
            return True
        return any(b.how == "value" and isinstance(b.value, ast.Call)
                   and reader.dotted(b.value.func) in _ENGINE_MAKERS
                   for b in reader.bindings_of(arg.id, node=arg))
    if isinstance(arg, ast.Call):
        return reader.dotted(arg.func) in _ENGINE_MAKERS
    return False


def _is_sa_session(node: ast.AST, reader: _Reader, *, depth: int = 0) -> bool:
    """Whether `node` is a SQLAlchemy session: a name that reads as one, or a
    value a session maker gave. It stops at the depth limit
    (`_Reader.stops`)."""
    if reader.stops("session", node, depth=depth, limit=6):
        return False
    if isinstance(node, ast.Name):
        if "sess" in node.id.lower():
            return True
        arg = reader.argument_of(node)
        if arg is not None:
            return _is_sa_session(arg.node, reader, depth=depth + 1)
        return any(b.how in ("value", "context") and _is_sa_session(b.value, reader, depth=depth + 1)
                   for b in reader.bindings_of(node.id, node=node))
    if isinstance(node, ast.Call):
        if reader.dotted(node.func).split(".")[-1] in _SA_SESSION_MAKERS:
            return True
        # A session a maker built, as in `sessionmaker(engine)()`.
        return isinstance(node.func, ast.Call) and _is_sa_session(node.func, reader, depth=depth + 1)
    return False


def _sqlalchemy_write(call: ast.Call, reader: _Reader, *, commits: bool) -> bool:
    func = call.func
    if not isinstance(func, ast.Attribute):
        return False
    method, recv = func.attr, func.value
    if method in _EXECUTE and call.args and not reader.is_local_handle(recv):
        return reader.read_through(lambda: _is_sa_construct(call.args[0], reader))
    if method in ("drop_all", "create_all") and (call.args or call.keywords):
        return True
    if method in ("drop", "create") and _engine_arg(call, reader):
        return True
    if method in ("delete", "update") and _on_a_query(recv):
        return True
    if not (method in _SA_SESSION_WRITES and commits):
        return False
    return reader.read_through(lambda: _is_sa_session(recv, reader))


def _on_a_query(node: ast.AST) -> bool:
    """Whether `node` is a `session.query(...)`, down a `.filter(...)` chain."""
    while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "query":
            return True
        node = node.func.value
    return False


@dataclass(frozen=True)
class _Routine:
    """A stored procedure or function a callee holds: the `cur.callfunc` or
    `getattr(cur, "callfunc")` that names it, and the code of each binding
    the check followed to reach it, the one nearest the routine first.
    `unfollowed` means the check stopped at the depth limit on code that
    names one or a name that may hold one, which is then `source`."""

    source: ast.AST
    bindings: tuple[str, ...] = ()
    unfollowed: bool = False


# How many bindings, displays and wrappers `_routine_of` follows.
_ROUTINE_DEPTH = 8
# Methods that read what a dict or list holds: `ops.get("purge")`,
# `steps.pop()` and `for f in ops.values():`.
_ENTRY_READS = frozenset({"get", "pop", "popitem", "setdefault", "values", "items", "copy"})


def _routine_getattr(*, call: ast.Call, reader: _Reader) -> bool:
    """Whether `call` is `getattr(cur, "callfunc")` on a handle outside the
    process: it hands on the same bound method."""
    return isinstance(call.func, ast.Name) and call.func.id == "getattr" and len(call.args) >= 2 and (
        isinstance(call.args[1], ast.Constant) and call.args[1].value in _ROUTINE_CALLS
        and not reader.is_local_handle(call.args[0])
    )


def _routine_key(*, node: ast.AST) -> str | None:
    """How `_Reader.routine_names` names what `node` reads: `run` for a
    name, `h.run` for a member of one, None for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def _routine_of(
    *, node: ast.AST, reader: _Reader, depth: int = 0, seen: set[int] | None = None,
) -> _Routine | None:
    """The stored procedure or function that calling `node` runs: a
    `cur.callfunc` itself, or one that a name, an alias, a
    `functools.partial`, a list, tuple or dict element or entry read, a
    conditional, a tuple unpack or a member store holds, in this cell or an
    earlier one. An in-process handle's doesn't count. Code past the depth
    limit that names a routine, or a name that may hold one, counts as
    holding it.

    `seen` holds the id of each binding value this question followed. One
    reached again has nothing more to give: the first visit found nothing,
    from any depth, or its answer would already have ended the walk. So
    many reads of one binding follow it once."""
    if not reader.names_a_routine:
        return None
    if seen is None:
        seen = set()
    if depth > _ROUTINE_DEPTH:
        if any(
            _routine_key(node=n) in reader.routine_names
            or isinstance(n, ast.Attribute) and n.attr in _ROUTINE_CALLS and not reader.is_local_handle(n.value)
            or isinstance(n, ast.Call) and _routine_getattr(call=n, reader=reader)
            for n in ast.walk(node)
        ):
            return _Routine(source=node, unfollowed=True)
        return None
    if isinstance(node, ast.Starred):
        return _routine_of(node=node.value, reader=reader, depth=depth + 1, seen=seen)
    if isinstance(node, ast.Attribute) and node.attr in _ROUTINE_CALLS:
        return None if reader.is_local_handle(node.value) else _Routine(source=node)
    if isinstance(node, ast.Call):
        func = node.func
        if _routine_getattr(call=node, reader=reader):
            return _Routine(source=node)
        if reader.dotted(func) == "functools.partial" and node.args:
            return _routine_of(node=node.args[0], reader=reader, depth=depth + 1, seen=seen)
        if isinstance(func, ast.Attribute) and func.attr in _ENTRY_READS and (
            _routine_key(node=func.value) in reader.routine_names
        ):
            return _routine_of(node=func.value, reader=reader, depth=depth + 1, seen=seen)
        return None
    if isinstance(node, ast.Subscript):
        return _routine_of(node=node.value, reader=reader, depth=depth + 1, seen=seen)
    # Each value to follow, with the cell whose binding holds it: None for
    # a part of an expression the binding already shows.
    follow: list[tuple[ast.AST, _Cell | None]] = []
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        follow = [(e, None) for e in node.elts]
    elif isinstance(node, ast.Dict):
        follow = [(v, None) for v in node.values]
    elif isinstance(node, ast.IfExp):
        # `run = cur.callfunc if ok else None` and `run = cur.callfunc or None`.
        follow = [(node.body, None), (node.orelse, None)]
    elif isinstance(node, ast.BoolOp):
        follow = [(v, None) for v in node.values]
    elif isinstance(node, ast.Name) and node.id in reader.routine_names:
        follow = [(b.value, b.cell) for b in reader.bindings_of(node.id, node=node)
                  if b.how in ("value", "element", "unpacked")]
        # `ops["purge"] = cur.callproc` and `ops.append(cur.callproc)`.
        follow += [(g.value, cell) for cell in reader.cells for g in cell.grows.get(node.id, ()) if g.how == "add"]
    elif isinstance(node, ast.Attribute) and _routine_key(node=node) in reader.routine_names:
        # `h.run = cur.callfunc`, then `h.run(...)`.
        follow = [(v, cell) for cell in reader.cells for v in cell.member_values.get((node.value.id, node.attr), ())]
    for value, home in follow:
        if id(value) in seen:
            continue
        seen.add(id(value))
        found = _routine_of(node=value, reader=reader, depth=depth + 1, seen=seen)
        if found is not None:
            shown = () if home is None else (home.source(value),)
            return _Routine(source=found.source, bindings=(*found.bindings, *shown), unfollowed=found.unfollowed)
    return None


def _bound_into(*, node: ast.AST, cell: _Cell, reader: _Reader) -> str | None:
    """The name a statement binds `node` to, directly or as an element of a
    list, tuple, set or dict display or a `functools.partial`'s function:
    `run` in `run = cur.callfunc` or `run = partial(cur.callfunc, "p")`."""
    parent = cell.parents.get(node)
    while isinstance(parent, (ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Starred)) or (
        isinstance(parent, ast.Call) and parent.args and parent.args[0] is node
        and reader.dotted(parent.func) == "functools.partial"
    ):
        if isinstance(parent, ast.Dict) and not any(v is node for v in parent.values):
            return None
        node, parent = parent, cell.parents.get(parent)
    if isinstance(parent, ast.Assign) and parent.value is node and len(parent.targets) == 1 and (
        isinstance(parent.targets[0], ast.Name)
    ):
        return parent.targets[0].id
    return None


def _call_write_kind(call: ast.Call, reader: _Reader, *, commits: bool) -> CallVia | None:
    """What a call changes with no SQL text to quote, or None when it changes
    nothing the check holds."""
    func = call.func
    if not isinstance(func, ast.Attribute):
        return None
    method, recv = func.attr, func.value
    if method in _ROUTINE_CALLS and not reader.is_local_handle(recv):
        return "procedure"
    if method in _MONGO_WRITES:
        return "mongodb"
    if method == "drop" and _store_kind(recv, reader) == "mongodb":
        return "mongodb"
    if method in _DYNAMO_WRITES:
        return "dynamodb"
    if method in _BIGQUERY_DISTINCT:
        return "bigquery"
    if method in _TABLE_WRITES:
        kind = _client_kind(recv, reader)
        return kind if kind in ("bigquery", "dynamodb") else "data_store"
    if method in _BIGQUERY_WRITES and _store_kind(recv, reader) == "bigquery":
        return "bigquery"
    if method in _ES_DISTINCT:
        return "elasticsearch"
    if method == "delete" and isinstance(recv, ast.Attribute) and recv.attr == "indices":
        return "elasticsearch"
    if method in _REDIS_WRITES or method in _ES_WRITES:
        kind = _store_kind(recv, reader)
        if kind == "redis" and method in _REDIS_WRITES:
            return "redis"
        if kind == "elasticsearch" and method in _ES_WRITES:
            return "elasticsearch"
    return "sqlalchemy" if _sqlalchemy_write(call, reader, commits=commits) else None


def _store_kind(recv: ast.AST, reader: _Reader) -> CallVia | None:
    """`_client_kind` of a receiver whose method changes data only on a store
    of the right kind: a read the depth limit stopped short of the store's
    constructor leaves the cell unchecked (`_Reader.read_past`)."""
    return reader.read_through(lambda: _client_kind(recv, reader))


def _call_findings(
    reader: _Reader, cell: _Cell, *, root: ast.AST | None = None, report: _Report | None = None,
) -> list[CallWrite]:
    """The data changes a cell makes with no SQL text to quote: a stored
    procedure, a SQLAlchemy construct, a non-SQL store's own method, or SQL
    read from a file too big to check. With `root`, the ones a function of the
    cell makes, reported at `report`, the call in the checked cell."""
    top = cell.tree if root is None else root
    commits = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("commit", "flush")
        for n in ast.walk(top)
    )
    out: list[CallWrite] = []

    def add(via: CallVia, node: ast.AST, *, bindings: tuple[str, ...] = ()) -> CallWrite:
        code = cell.source(node)
        if report is None:
            write = CallWrite(via=via, call=scrub_output(code), line=cell.host_line or getattr(node, "lineno", 0),
                              runs=scrub_output("\n".join(dict.fromkeys(bindings))))
        else:
            write = CallWrite(via=via, call=report.call, line=report.line,
                              runs=scrub_output("\n".join(dict.fromkeys((*bindings, code)))))
        out.append(write)
        return write

    # Each routine the code binds or hands on, with its finding; each call
    # that runs one through a binding, with what it runs; and each value
    # handed on that holds one.
    bound: list[tuple[ast.AST, CallWrite]] = []
    called: list[tuple[ast.Call, _Routine]] = []
    handed: list[tuple[ast.AST, _Routine]] = []
    for node in ast.walk(top):
        if isinstance(node, ast.Attribute) and node.attr in _ROUTINE_CALLS and not reader.is_local_handle(node.value):
            parent = cell.parents.get(node)
            if not (isinstance(parent, ast.Call) and parent.func is node):
                # Bound to a name, wrapped in a partial or passed on: the
                # routine runs where the check may not follow it.
                bound.append((node, add("procedure", node)))
            continue
        if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)) and node.value is not None:
            found = _routine_of(node=node.value, reader=reader)
            if found is not None:
                handed.append((node, found))
            continue
        if not isinstance(node, ast.Call):
            continue
        if _routine_getattr(call=node, reader=reader):
            # `getattr(cur, "callfunc")` hands on the same bound method.
            bound.append((node, add("procedure", node)))
            continue
        follows = reader.dotted(node.func) == "functools.partial"
        for i, arg in enumerate((*node.args, *(k.value for k in node.keywords))):
            found = None if follows and i == 0 else _routine_of(node=arg, reader=reader)
            if found is not None:
                handed.append((node, found))
        found = _routine_of(node=node.func, reader=reader)
        if found is not None:
            # `run(...)` for a routine a name, a partial, a container or a
            # member holds: the card shows the code that bound it.
            called.append((node, found))
            add("procedure", node, bindings=found.bindings)
            continue
        via = _call_write_kind(node, reader, commits=commits)
        if via is not None:
            add(via, node)
        elif isinstance(node.func, ast.Attribute) and node.func.attr in _FILE_READERS:
            path = _sql_file_path(node, reader)
            if path is not None and reader.oversized_sql_file(path) and _reaches_a_runner(node, reader):
                add("sql_file", node)
    here = {id(n) for n in ast.walk(top)}
    for node, found in handed:
        # A routine bound outside this code, handed on where the check may
        # not follow it. One bound here already has its own finding.
        if found.unfollowed or id(found.source) not in here:
            add("procedure", node, bindings=found.bindings)

    def only_called(source: ast.AST) -> bool:
        """Whether every use of the routine `source` here is a call this
        code reports: inside the called expression, or through the one
        name it is bound to, read only as a call or a call's subscript."""
        calls = [c for c, found in called if found.source is source]
        if any(_inside(source, c.func, cell) or source is c.func for c in calls):
            return True
        name = _bound_into(node=source, cell=cell, reader=reader)
        if name is None or not calls:
            return False
        funcs = {id(c.func) for c in calls}
        uses = [n for n in ast.walk(top) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)]
        return all(
            id(n) in funcs or isinstance(cell.parents.get(n), ast.Subscript) and id(cell.parents[n]) in funcs
            for n in uses
        )

    # A routine bound and called here is reported once, at each call.
    dropped = {id(write) for source, write in bound if only_called(source)}
    return [w for w in out if id(w) not in dropped]


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
                # A part picked at run time names any text, as in `_locate`.
                found = _relative(candidate.replace(_RUN_TIME_PART, "*"), snap)
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
    existing_files: bool = False,
    made: _Made | None = None,
    disk: _DiskReads | None = None,
) -> _Targets:
    """The snapshot entries `patterns` name. A relative pattern counts from
    each folder in `cwds`, or from under any folder of either root when the
    check can't place them. With `files_only`, for a delete that can't
    remove a folder, the folders a listing names drop out. With
    `existing_files`, for an overwrite, only the files the turn began with
    count: a write can't replace a folder, a root, or a file the turn made.
    `made` reads a snapshot past its cap (`ProjectSnapshot.named`), and
    `disk` is the check's reads from disk (`ProjectSnapshot.matching`)."""
    if isinstance(patterns, _Unknown):
        return _Targets(unknown=not existing_files)
    hits: list[str] = []
    whole: set[str] = set()
    unknown = False
    project = next((s for s in snapshots if s.label == "project"), None)
    for raw in patterns:
        pattern = raw.replace("\\", "/")
        if _TEMP_PATH in pattern:
            continue
        pattern = pattern.replace(_RUN_TIME_PART, "*")
        if not os.path.isabs(pattern) and isinstance(cwds, _Unknown):
            # From a folder the check can't place: under any folder of either root.
            rest = _without_parents(pattern)
            if not rest:
                unknown = not existing_files
                continue
            for snap in snapshots:
                hits += _card_paths(snap, snap.matching(pattern="**/" + rest, made=made, disk=disk),
                                    files_only=files_only, existing_files=existing_files, made=made)
            continue
        candidates = [pattern] if os.path.isabs(pattern) else [_join(c, pattern) for c in cwds]
        for candidate in candidates:
            candidate = candidate.replace(_RUN_TIME_PART, "*")
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
                    if not existing_files:
                        whole.add(snap.label)
                elif any(ch in rel for ch in "*?["):
                    hits += _card_paths(snap, snap.matching(pattern=rel, made=made, disk=disk),
                                        files_only=files_only, existing_files=existing_files, made=made)
                else:
                    named = snap.named(rel=rel, made=made)
                    hits += _card_paths(snap, [named] if named is not None else [], files_only=False,
                                        existing_files=existing_files, made=made)
    return _Targets(
        paths=tuple(dict.fromkeys(hits)),
        whole_project="project" in whole,
        whole_shared="shared" in whole,
        unknown=unknown,
    )


def _card_paths(
    snap: ProjectSnapshot, found: list[str], *, files_only: bool, existing_files: bool = False,
    made: _Made | None = None,
) -> list[str]:
    """Snapshot paths as a card names them, the shared folder's prefixed.
    With `existing_files`, only the files the turn began with, `made` read
    as `ProjectSnapshot.file_at_start` reads it."""
    prefix = "" if snap.label == "project" else "shared folder: "
    if existing_files:
        found = [f for f in (snap.file_at_start(rel=p, made=made) for p in found) if f is not None]
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


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


@dataclass(frozen=True)
class _FileWrite:
    """A call that writes a file: the node that names the file, and whether
    the call replaces what the file holds, adds to it, or only creates it,
    failing when it exists."""

    target: ast.AST
    kind: Literal["replace", "append", "create"]


_MODE = re.compile(r"[rwxabtU+]{1,5}")


def _mode_kind(
    mode: ast.AST | None, *, absent: str, unread: str,
) -> Literal["replace", "append", "create"] | None:
    """What a file mode does to an existing file: `w` and `r+` replace what
    it holds, `a` adds to it, `x` only creates a file; None for a read.
    `absent` is the mode when the call passes none, `unread` when it passes
    one the check can't read."""
    if mode is None:
        text = absent
    elif isinstance(mode, ast.Constant) and isinstance(mode.value, str) and _MODE.fullmatch(mode.value):
        text = mode.value
    else:
        text = unread
    if "a" in text:
        return "append"
    if "x" in text:
        return "create"
    if "w" in text or "+" in text:
        return "replace"
    return None


def _file_write(call: ast.Call, reader: _Reader) -> _FileWrite | None:
    """The file `call` writes into: one it opens for writing, a
    `Path.write_text`, a DataFrame or chart writer, a `pd.ExcelWriter`, or a
    `.save(path)`."""
    name = reader.dotted(call.func)
    method = call.func.attr if isinstance(call.func, ast.Attribute) else ""
    if name in ("open", "io.open"):
        target = _target_arg(call, keywords=("file",))
        mode = call.args[1] if len(call.args) > 1 else _keyword(call, "mode")
        kind = _mode_kind(mode, absent="r", unread="r")
    elif name == "pandas.ExcelWriter":
        target = _target_arg(call, keywords=("path",))
        kind = _mode_kind(_keyword(call, "mode"), absent="w", unread="w")
    elif method in ("write_text", "write_bytes"):
        target, kind = call.func.value, "replace"
    elif method == "open":
        target = call.func.value
        kind = _mode_kind(_target_arg(call, keywords=("mode",)), absent="r", unread="r")
    elif method in _WRITER_METHODS:
        target = _target_arg(call, keywords=_WRITER_METHODS[method])
        kind = _mode_kind(_keyword(call, "mode"), absent="w", unread="w")
    else:
        return None
    return None if target is None or kind is None else _FileWrite(target=target, kind=kind)


def _move_destination(call: ast.Call) -> ast.AST | None:
    return call.args[1] if len(call.args) > 1 else next(
        (k.value for k in call.keywords if k.arg in ("dst", "dest", "target")), None
    )


def _written_target(call: ast.Call, reader: _Reader, cell: _Cell) -> ast.AST | None:
    """The path `call` writes: a file it writes into or creates, or the
    destination of a rename, move or copy."""
    name = reader.dotted(call.func)
    if name in _MOVE_FUNCS or name in _COPY_FUNCS:
        return _move_destination(call)
    if isinstance(call.func, ast.Attribute) and call.func.attr in ("rename", "replace") and (
        _is_path_rename(call, reader, cell)
    ):
        return call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == "target"), None)
    write = _file_write(call, reader)
    return None if write is None else write.target


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


def _on_a_path(node: ast.AST, reader: _Reader, cell: _Cell) -> bool:
    """`_Reader.is_path` for a call that changes a file only when `node` is a
    path: a read the depth limit stopped short of one leaves the cell
    unchecked (`_Reader.read_past`)."""
    return reader.read_through(lambda: reader.is_path(node, cell=cell))


def _is_path_rename(call: ast.Call, reader: _Reader, cell: _Cell) -> bool:
    """`.rename(x)` and `.replace(x)` on a pathlib path, not a DataFrame's or a str's."""
    receiver = call.func.value
    if isinstance(receiver, (ast.Constant, ast.JoinedStr)):
        return False
    one_arg = len(call.args) == 1 and not call.keywords
    target_kw = not call.args and any(k.arg == "target" for k in call.keywords)
    if not (one_arg or target_kw):
        return False
    mark = reader.cuts
    if reader.is_path(receiver, cell=cell):
        return True
    dest = call.args[0] if call.args else next(k.value for k in call.keywords if k.arg == "target")
    if reader.is_path(dest, cell=cell):
        return True
    if isinstance(dest, ast.Constant) and isinstance(dest.value, str) and (
        re.search(r"\.[A-Za-z0-9]{1,5}$", dest.value)
    ):
        return True
    # A path the depth limit stopped short of.
    reader.read_past(mark, lambda: reader.is_path(receiver, cell=cell) or reader.is_path(dest, cell=cell))
    return False


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


def _database_file(database: str) -> list[str]:
    """The file a SQLite or DuckDB database name opens: none for `:memory:`,
    a MotherDuck database, or a read-only or in-memory `file:` URI."""
    if database.startswith("file:"):
        database, _, query = database[len("file:"):].partition("?")
        if re.search(r"(?:^|&)mode=(?:ro|memory)(?:&|$)", query):
            return []
    else:
        database = database.partition("?")[0]
    if database in ("", ":memory:") or database.startswith(":memory:") or _is_motherduck(database):
        return []
    return [database]


def _database_files(node: ast.AST, reader: _Reader, *, depth: int = 0) -> list[str]:
    """The files an in-process SQLite or DuckDB handle may open, as the code
    spells them, followed the way `_Reader.is_local_handle` follows a handle:
    none for an in-memory or read-only one."""
    if depth > 6:
        return []
    if isinstance(node, ast.Name):
        return [f for b in reader.bindings_of(node.id, node=node) if b.how in ("value", "context")
                for f in _database_files(b.value, reader, depth=depth + 1)]
    if not isinstance(node, ast.Call):
        return []
    name = reader.dotted(node.func)
    if name in ("duckdb.connect", "sqlite3.connect"):
        target = node.args[0] if node.args else next(
            (k.value for k in node.keywords if k.arg in ("database", "path")), None
        )
        read_only = _keyword(node, "read_only")
        if isinstance(read_only, ast.Constant) and read_only.value is True:
            return []
        if not (isinstance(target, ast.Constant) and isinstance(target.value, str)):
            return []
        return _database_file(target.value)
    if name in ("sqlalchemy.create_engine", "create_engine") and node.args:
        url = node.args[0]
        if not (isinstance(url, ast.Constant) and isinstance(url.value, str)):
            return []
        scheme, sep, rest = url.value.partition("://")
        if not sep or scheme.split("+")[0] not in ("sqlite", "duckdb"):
            return []
        # `sqlite:///a.db` names a.db, `sqlite:////tmp/a.db` names /tmp/a.db.
        return _database_file(rest[1:] if rest.startswith("/") else rest)
    if isinstance(node.func, ast.Attribute) and node.func.attr in ("cursor", "connect", "begin", "raw_connection"):
        return _database_files(node.func.value, reader, depth=depth + 1)
    if isinstance(node.func, ast.Name) and node.func.id in reader.functions:
        return [f for n in ast.walk(reader.functions[node.func.id]) if isinstance(n, ast.Return) and n.value is not None
                for f in _database_files(n.value, reader, depth=depth + 1)]
    return []


def _written_statements(node: ast.AST, reader: _Reader, *, depth: int = 0) -> tuple[str, ...]:
    """Every statement of each text `node` holds that has a write, as the
    card quotes it: written out, held in a name, or passed in as a
    parameter. It goes to an execute, so it reads as SQL whatever its case.
    Empty when no text holds a write. It stops at the depth limit
    (`_Reader.stops`)."""
    text = _sql_text(node, reader)
    if text is not None:
        return _quoted_statements(text) if write_statements(text, prose=False) else ()
    if not isinstance(node, ast.Name) or reader.stops("statements", node, depth=depth, limit=4):
        return ()
    found: list[str] = []
    for b in reader.bindings_of(node.id, node=node):
        if b.how != "param":
            found += reader.statements_held(node.id, b, prose=False)
            continue
        key = (id(b.value), node.id)
        args = [reader.param_args[key]] if key in reader.param_args else reader.call_sites(b.value, node.id)
        for a in args:
            if a is not None:
                found += _written_statements(a.node, reader, depth=depth + 1)
    return tuple(dict.fromkeys(found))


def _errored_names_read(cells: list[_Cell], *, errored: frozenset[int]) -> bool:
    """Whether a name an errored cell binds or adds to is read where that
    cell may have stopped before it: at module level in a later cell, or in
    any function, which runs when it is called. Only then can reading the
    errored cells as maybe stopped anywhere name another path. A history
    with no errored cell, or none whose names a later cell or a function
    reads, pays for no extra read."""
    for stopped in cells:
        if stopped.index not in errored:
            continue
        bound = {*stopped.bindings, *stopped.grows}
        for cell in cells:
            for name in bound & cell.loads.keys():
                if cell.index > stopped.index or any(cell.scope_of(n) is not None for n in cell.loads[name]):
                    return True
    return False


class _OwnedFiles:
    """The files the turn began with that the agent owns, so it may rewrite
    them with no card: those under `.anton/artifacts`, `.anton/skill_drafts`
    and `.anton/output` (`_AGENT_FOLDERS`), and those a write that ran in an
    earlier cell of the conversation replaced. The earlier cells are read the
    first time a check asks, since most cells overwrite nothing. A cell in
    `errored`, by index, stopped on an error, so the check can't tell which
    of its writes ran: none of them makes a file the agent's, and nor does a
    write through a name it binds."""

    def __init__(
        self, earlier: list[_Cell], snapshots: Sequence[ProjectSnapshot], *, errored: frozenset[int] = frozenset(),
    ) -> None:
        self._earlier = earlier
        self._snapshots = snapshots
        self._errored = errored
        # The files the earlier cells overwrote: read whole, read with each
        # errored cell as never run, and read with a path through a name an
        # errored cell binds as one the check can't place. Each is read the
        # first time a check needs it.
        self._written: frozenset[str] | None = None
        self._written_unerrored: frozenset[str] | None = None
        self._written_surely: frozenset[str] | None = None

    def owns(self, path: str) -> bool:
        """Whether the agent owns `path`, a snapshot path as a card names it.

        Only a write that ran counts, read the way the checked cell is,
        against the earlier cells alone: one in a cell's own code
        (`_nodes_that_run`), or in a function or lambda that code calls,
        defined before the call, or in the literal code either passes to
        exec() (`_exec_cells`). A function no cell calls, one only passed on
        as a value, or a write under `if False:` makes no file the agent's,
        and nor does an append."""
        if path.startswith(_AGENT_FOLDERS):
            return True
        if not self._earlier:
            return False
        if self._written is None:
            self._written = self._writes_read_from(self._earlier)
        if path not in self._written or not self._errored:
            return path in self._written
        # The check can't tell where an errored cell stopped, so a file is the
        # agent's only when the earlier cells make it so both with each errored
        # cell read whole and with it read as never run: a function it defines,
        # or what it adds to a list, may never have been there for a later
        # cell to use.
        if self._written_unerrored is None:
            empty = [_Cell(ast.Module(body=[], type_ignores=[]), "", c.index) if c.index in self._errored else c
                     for c in self._earlier]
            self._written_unerrored = self._writes_read_from(empty)
        if path not in self._written_unerrored:
            return False
        # Nor may a name it binds have pointed a later write at the file.
        if self._written_surely is None:
            self._written_surely = (
                self._writes_read_from(self._earlier, errored=self._errored)
                if _errored_names_read(self._earlier, errored=self._errored) else self._written
            )
        return path in self._written_surely

    def _writes_read_from(self, earlier: list[_Cell], *, errored: frozenset[int] = frozenset()) -> frozenset[str]:
        # Read from an empty cell after them all, so every earlier cell's
        # functions count as earlier ones (`_Reader.earlier_function`).
        after = _Cell(ast.Module(body=[], type_ignores=[]), "", len(earlier))
        cells = [*earlier, after]
        reader = _Reader(cells, current=after, errored=errored, errored_paths="unknown")
        unerrored = [c for c in earlier if c.index not in self._errored]
        exec_read = _exec_cells(reader, unerrored, index=len(cells), ran_only=True)
        if exec_read.cells:
            reader = _Reader([*cells, *exec_read.cells], current=after, errored=errored, errored_paths="unknown")
        changes: list[FileChange] = []

        def read_function(callee: _Callee, *, cell: _Cell, at: ast.AST, chain: tuple[int, ...]) -> None:
            """`callee`, called from the code of `cell` at `at`, read for
            what it overwrites, with the literal code it passes to exec(),
            and each function either calls in turn."""
            func = callee.func
            if callee.call is None or id(func) in chain or len(chain) >= _CALL_DEPTH:
                return
            home = reader.cell_of(func)
            if not _defined_before(func, home=home, cell=cell, at=at):
                # A definition after the call: the call ran an older one, or none.
                return
            with reader.calling(func, callee.call, bound=callee.bound):
                changes.extend(_file_findings(reader, home, self._snapshots, owned=None, root=func))
                for executed in exec_read.of(func):
                    changes.extend(_file_findings(reader, executed, self._snapshots, owned=None))
                for code in exec_read.with_ran(home, func):
                    for inner in _earlier_function_uses(reader, code.root, code.cell, ran_only=True):
                        read_function(inner, cell=cell, at=at, chain=(*chain, id(func)))

        for cell in unerrored:
            changes += _file_findings(reader, cell, self._snapshots, owned=None)
            for executed in exec_read.of(cell.tree):
                changes += _file_findings(reader, executed, self._snapshots, owned=None)
            for code in exec_read.with_ran(cell, cell.tree):
                for callee in _earlier_function_uses(reader, code.root, code.cell, ran_only=True):
                    # exec()'s code reads as a cell after them all, so a
                    # call in it reaches a function any earlier cell defines.
                    read_function(callee, cell=code.cell, at=callee.use, chain=())
        return frozenset(path for change in changes if change.action == "overwrite" for path in change.paths)


def _defined_before(func: Function, *, home: _Cell, cell: _Cell, at: ast.AST) -> bool:
    """Whether `func`, defined in `home`, was defined before the call at
    `at`, code of `cell`, ran: earlier in the same cell, or in a cell that
    ran before it. Literal code exec() runs ran where its exec() call is,
    in the cell that ran it (`_Cell.host`)."""
    if home is cell:
        return _follows(node=at, value=func)
    if home.host != cell.host:
        return home.host < cell.host
    if home.index == home.host:
        # The exec() that ran `cell` is in `home`, after the definition.
        return (cell.host_line or 0) > (getattr(func, "end_lineno", None) or func.lineno)
    if cell.index == cell.host:
        # An exec() in `cell` ran `home`, before the call.
        return getattr(at, "lineno", 0) > (home.host_line or 0)
    return home.index < cell.index


class _FileFinder:
    """Collects the file changes code in one cell makes. With a report, the
    code is an earlier cell's function, read for one call in the checked
    cell, and each change is reported at that call. With `owned`, the cell is
    the checked one: an overwrite of a file the agent owns is reported but
    not held. Without it, the cell is an earlier one, or a function it calls,
    read for the files it overwrote (`_OwnedFiles`) in only the code that
    runs when it runs (`_nodes_that_run`)."""

    def __init__(
        self, reader: _Reader, snapshots: Sequence[ProjectSnapshot], *, cell: _Cell, report: _Report | None,
        owned: _OwnedFiles | None,
    ) -> None:
        self.reader = reader
        self.snapshots = snapshots
        self.cell = cell
        self.report = report
        self.owned = owned
        self.out: list[FileChange] = []
        # Files already reported as overwritten, so a delete and the write
        # that follows it name a file once. A database write keeps its own
        # changes (`overwrite`), so it never hides another overwrite.
        self.overwritten: set[str] = set()

    def paths(self, node: ast.AST) -> list[str] | _Unknown:
        return self.reader.paths(node, cell=self.cell)

    def locate(
        self, got: list[str] | _Unknown, *, cwds: Sequence[str] | _Unknown | None = None, files_only: bool = False,
        existing_files: bool = False,
    ) -> _Targets:
        return _locate(got, self.snapshots, cwds=self.reader.cwds if cwds is None else cwds, files_only=files_only,
                       existing_files=existing_files, made=self.reader.made, disk=self.reader.disk)

    def line(self, node: ast.AST) -> int:
        """The line a change `node` makes is reported at."""
        return self.report.line if self.report is not None else self.cell.host_line or getattr(node, "lineno", 0)

    def add(
        self, action: Literal["delete", "rename", "move", "overwrite", "append"], targets: _Targets, node: ast.AST,
        *, held: bool = True, statements: tuple[SqlWrite, ...] = (),
    ) -> None:
        if not targets.found():
            return
        call = self.report.call if self.report is not None else scrub_output(self.cell.source(node))
        self.out.append(FileChange(
            action=action, paths=targets.paths, whole_project=targets.whole_project,
            whole_shared=targets.whole_shared, call=call, line=self.line(node), held=held, statements=statements,
        ))

    def walk(self, root: ast.AST) -> None:
        reader, cell = self.reader, self.cell
        # The walk goes breadth first, so a database write is read after it,
        # in source order: the card quotes a file's statements as they run.
        database_writes: list[ast.Call] = []
        for node in ast.walk(root) if self.owned is not None else _nodes_that_run(root):
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
            elif method == "rmdir" and _on_a_path(node.func.value, reader, cell):
                self.add("delete", self.locate(self.paths(node.func.value)), node)
            elif method in ("rename", "replace") and _is_path_rename(node, reader, cell):
                self.add("rename", self.locate(self.paths(node.func.value)), node)
                dest = node.args[0] if node.args else next(
                    (k.value for k in node.keywords if k.arg == "target"), None
                )
                if dest is not None:
                    self.overwrite(self.paths(dest), node)
            elif method in _EXECUTE or method in _READ_SQL or method == "to_sql":
                database_writes.append(node)
            else:
                write = _file_write(node, reader)
                if write is not None and write.kind != "create":
                    self.overwrite(self.paths(write.target), node, append=write.kind == "append")
        for call in sorted(database_writes, key=lambda c: (c.lineno, c.col_offset)):
            self.database_write(call)

    def lookup(self, *, node: ast.AST) -> bool:
        """Whether `node` is a lookup `_plain_lookup` can't read, on a module,
        class or path that holds a delete, a move, a copy or a shell call:
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
        on_path = module not in _LOOKUP_MODULES and _on_a_path(receiver, reader, cell)
        if module not in _LOOKUP_MODULES and not on_path:
            return False
        parent = cell.parents.get(node)
        call = parent if isinstance(parent, ast.Call) and parent.func is node else None
        names = reader.paths(key, cell=cell)
        if isinstance(names, _Unknown) or any(ch in n for n in names for ch in "*?[" + _RUN_TIME_PART):
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
                # What it lands on is overwritten, as `walk` reads the plain call.
                dest = _target_arg(call, keywords=("target",)) if on_path else None
                if dest is not None:
                    self.overwrite(self.paths(dest), call)
                elif not on_path:
                    self.note_overwrite(call)
            elif full in _COPY_FUNCS:
                self.note_overwrite(call)
            elif full in _SHELL_FUNCS:
                self.shell(call)
        return True

    def delete(self, target: ast.AST, call: ast.Call, *, files_only: bool) -> None:
        """A delete; an overwrite when a write to the same path follows it."""
        got = self.paths(target)
        rewritten = not isinstance(got, _Unknown) and bool(got) and set(got) <= self.written_after(call)
        if rewritten:
            self.overwrite(got, call)
        else:
            self.add("delete", self.locate(got, files_only=files_only), call)

    def overwrite(
        self, got: list[str] | _Unknown, call: ast.Call, *, append: bool = False, statements: tuple[SqlWrite, ...] = (),
    ) -> None:
        """A write into the files `got` names that the turn began with: held
        as an overwrite unless the agent owns the file. With `append`, the
        write adds to the file, and is reported but not held. A path from a
        root the check can't place counts as a new file, and so does one
        with a part the code picks when it runs, as a dated export or a chart
        per site has. In an earlier cell, such a path makes no file the
        agent's own: the snapshot holds the files the turn began with, not
        those the conversation began with, so it can't tell an older day's
        export from one an earlier cell wrote.

        With `statements`, the write runs them on a SQLite or DuckDB file,
        and the card quotes them with it. Such a write is reported even when
        another overwrite already named the file, and a file an earlier one
        in the cell changes takes the new statements on that change
        (`run_too`), so the card names it once with all of them."""
        if isinstance(got, _Unknown):
            return
        got = [p for p in got if _RUN_TIME_PART not in p]
        located = self.locate(got, existing_files=True).paths
        if statements:
            found = self.run_too(located, statements)
        else:
            found = [p for p in located if append or p not in self.overwritten]
        if append:
            self.add("append", _Targets(paths=tuple(found)), call, held=False)
            return
        if not statements:
            self.overwritten.update(found)
        owned = tuple(p for p in found if self.owned is not None and self.owned.owns(p))
        self.add("overwrite", _Targets(paths=tuple(p for p in found if p not in owned)), call, statements=statements)
        self.add("overwrite", _Targets(paths=owned), call, held=False, statements=statements)

    def run_too(self, paths: Sequence[str], statements: tuple[SqlWrite, ...]) -> list[str]:
        """`statements` added to each change an earlier database write in the
        cell made to files among `paths`; the paths no such change names."""
        rest = list(paths)
        for i, change in enumerate(self.out):
            if change.statements and change.paths and set(change.paths) <= set(rest):
                self.out[i] = change.model_copy(update={"statements": (*change.statements, *statements)})
                rest = [p for p in rest if p not in change.paths]
        return rest

    def database_write(self, call: ast.Call) -> None:
        """A write statement, or `to_sql`, on an in-process SQLite or DuckDB
        handle that opens a file: an overwrite of that file, with each
        statement it runs for the card. The SQL check lets every in-process
        handle through (`_executes_on_local`)."""
        method = call.func.attr
        if method == "to_sql":
            handle = call.args[1] if len(call.args) > 1 else _keyword(call, "con")
            statement = None
        elif method in _READ_SQL:
            handle = call.args[1] if len(call.args) > 1 else _keyword(call, "con")
            statement = _target_arg(call, keywords=("sql",))
        else:
            handle = call.func.value
            statement = _target_arg(call, keywords=("sql", "query", "statement", "sql_script", "operation"))
        if handle is None or method != "to_sql" and statement is None:
            return
        files = _database_files(handle, self.reader)
        if not files or not self.reader.is_local_handle(handle):
            return
        if statement is None:
            texts, values = (_to_sql_statement(call),), ""
        else:
            mark = self.reader.cuts
            texts, values = _written_statements(statement, self.reader), _values_text(call, reader=self.reader)
            if not texts:
                # A statement the depth limit stopped short of.
                self.reader.read_past(mark, lambda: _written_statements(statement, self.reader))
        if texts:
            line = self.line(call)
            self.overwrite(list(files), call, statements=tuple(
                _sql_write(text, reader=self.reader, values=values, line=line) for text in texts
            ))

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
        """A move or copy onto an existing file, an overwrite of it. A folder
        destination takes the source's name."""
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
        self.overwrite(expanded, call)

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
    owned: _OwnedFiles | None,
    root: ast.AST | None = None,
    report: _Report | None = None,
) -> list[FileChange]:
    """The file changes `cell` makes; with `root`, the ones a function of
    that cell makes, reported at `report`, the call in the checked cell.
    `owned` is None for an earlier cell read for what it overwrote, as
    `_FileFinder` says."""
    if not snapshots or all(s.is_empty() for s in snapshots):
        return []
    finder = _FileFinder(reader, snapshots, cell=cell, report=report, owned=owned)
    finder.walk(cell.tree if root is None else root)
    return finder.out


# ── Shell commands ────────────────────────────────────────────────────────

# A redirection such as `>`, `2>>` or `<`, alone or glued to its file.
_REDIRECT = re.compile(r"^(?:\d*|&)(?:>>?|<<?<?)(?:&\d*)?")
# Shell flags that take the next word as their value.
_VALUE_FLAGS = frozenset({"-u", "-g", "-n", "-I", "-L", "-P", "-d", "-s", "-a", "-C"})


def _shell_steps(call: ast.Call, reader: _Reader, *, cwds: Sequence[str] | _Unknown) -> list[_ShellStep]:
    """The deletes and moves the command `call` runs: a list of words or a
    script, written in the call, held in a name, or built with an f-string,
    `+`, `%` or `.format`, each part picked at run time a placeholder."""
    commands = _shell_commands(call, reader.dotted(call.func), reader)
    given = next((k.value for k in call.keywords if k.arg == "input"), None)
    stdin = None if given is None else _sql_text(given, reader)
    out: list[_ShellStep] = []
    for command in commands:
        if isinstance(command, str):
            out += _script_steps(command, cwds=cwds, depth=0)
        else:
            out += _words_steps(command, cwds=cwds, depth=0, stdin=stdin)
    return out


def _shell_commands(call: ast.Call, name: str, reader: _Reader) -> list[str | list[str]]:
    """Each command `call`, a call to `name` that runs a program, can run:
    its words, spread over the arguments or as one list, or a script's text.
    A command a stop at the depth limit left unread stops the check
    (`_Reader.read_past`)."""
    if name in _SPREAD_ARGV_FUNCS:
        words = call.args[_SPREAD_ARGV_FUNCS[name]:]
        # The e forms end with the environment, not a word.
        words = words[:-1] if name.endswith("e") else words
        return [[_command_word(w, reader) for w in words]]
    arg = _command_arg(call, name)
    if arg is None:
        return []
    mark = reader.cuts
    commands = _commands(arg, reader)
    # A command the depth limit stopped short of.
    reader.read_past(mark, lambda: any(c not in commands for c in _commands(arg, reader)))
    return commands


def _command_arg(call: ast.Call, name: str) -> ast.AST | None:
    """The argument of `call`, a call to `name` that runs a program from one
    list of words or a script's text, that holds them; None when it passes none."""
    index = _LIST_ARGV_FUNCS.get(name, 0)
    if len(call.args) > index:
        return call.args[index]
    return None if index else _target_arg(call, keywords=("args", "cmd", "command", "argv"))


def _commands(node: ast.AST, reader: _Reader, *, depth: int = 0) -> list[str | list[str]]:
    """Each command `node` can hold: a list of words, or a script's text. It
    stops at the depth limit (`_Reader.stops`)."""
    if reader.stops("commands", node, depth=depth, limit=6):
        return []
    if isinstance(node, (ast.List, ast.Tuple)):
        return list(_listed_commands(node, reader, depth=depth))
    words_made = _word_lists(node=node, reader=reader, depth=depth)
    if words_made is not None:
        return list(words_made)
    if isinstance(node, ast.Call) and len(node.args) == 1 and reader.dotted(node.func) == "shlex.join":
        return [shlex.join(c) for c in _commands(node.args[0], reader, depth=depth + 1) if isinstance(c, list)]
    made = reader.pad_callables(node=node.func) if isinstance(node, ast.Call) else []
    if made:
        # What a pad function that builds the command returns.
        return [c for func in made for ret in reader.returns(func=func)
                for c in _commands(ret, reader, depth=depth + 1)]
    if isinstance(node, ast.Name):
        arg = reader.argument_of(node)
        if arg is not None:
            return _commands(arg.node, reader, depth=depth + 1)
        out: list[str | list[str]] = []
        for b in reader.bindings_of(node.id, node=node):
            walked = _walked_items(b.value, reader, depth=depth) if b.how in ("element", "unpacked") else None
            if b.how == "param":
                # What each call of the function passes for it; a command the
                # check can't read where one call passes it may be any.
                for site in reader.call_sites(b.value, node.id):
                    passed = [] if site is None else _commands(site.node, reader, depth=depth + 1)
                    out += passed or [[_NOT_LITERAL]]
            elif b.how == "value" and (
                isinstance(b.value, (ast.List, ast.Tuple)) or _word_lists(node=b.value, reader=reader, depth=depth) is not None
            ):
                grown = reader.list_growth_for_write(name=node.id, binding=b)
                for words in _commands(b.value, reader, depth=depth + 1):
                    if isinstance(words, list) and (isinstance(grown, _Unknown) or grown):
                        # Words the cell adds to the list.
                        words = [*words, _NOT_LITERAL]
                    out.append(words)
            elif b.how == "value" and isinstance(b.value, ast.Name):
                # Another name for the command, as `c1 = c0` gives one.
                out += _commands(b.value, reader, depth=depth + 1)
            elif walked:
                # Each command a loop walks.
                out += [c for item in walked for c in _commands(item, reader, depth=depth + 1)]
            else:
                out += [f for t in _held_texts(node.id, b, reader, depth=0) for f in _program_filled(t, reader)]
        return out
    text = _command_text(node, reader)
    return [] if text is None else list(_program_filled(text, reader))


def _listed_commands(node: ast.List | ast.Tuple, reader: _Reader, *, depth: int) -> list[list[str]]:
    """The commands a list of words written out gives: each word as
    `_command_word` reads it, a starred list's words in its place, and a
    word read through a name as each literal text it can hold
    (`_literal_texts`), up to `_COMMAND_LIMIT` commands; so none when the
    program's word holds none. Code handed to a `-c` stays a word the
    check can't read: code through a name isn't read as a literal."""
    commands: list[list[str]] = [[]]
    for at, e in enumerate(node.elts):
        if isinstance(e, ast.Starred):
            items = reader.items_of(node=e.value, key=None, depth=depth + 1)
            words = [_NOT_LITERAL] if items is None else [_command_word(i, reader) for i in items]
            commands = [[*c, *words] for c in commands]
            continue
        word = _command_word(e, reader)
        code = at > 0 and bool(_CODE_FLAG.fullmatch(commands[0][-1] if commands[0] else ""))
        texts = _literal_texts(e, reader) if _unreadable_word(word) and not code else None
        if texts == [] and at == 0:
            return []
        options = list(dict.fromkeys(texts)) if texts and len(commands) * len(texts) <= _COMMAND_LIMIT else [word]
        commands = [[*c, option] for c in commands for option in options]
    return commands


def _word_lists(*, node: ast.AST, reader: _Reader, depth: int) -> list[list[str]] | None:
    """The word lists an expression that builds one can give a command:
    `text.split()`, `shlex.split(text)`, or lists added with `+`, a part the
    check can't read one word it can't read. A part a stop at the depth
    limit left empty gives none: in a loop of bindings, such as
    `cmd = cmd + ["-l"]` reading its own name, the other bindings give the
    command, and `_shell_commands` holds one past the limit. None for any
    other expression."""
    shell_split = isinstance(node, ast.Call) and bool(node.args) and reader.dotted(node.func) == "shlex.split"
    if shell_split or isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "split":
        source = node.args[0] if shell_split else node.func.value
        sep: str | None = None
        if node.args and not shell_split:
            given = node.args[0]
            literal = isinstance(given, ast.Constant) and isinstance(given.value, str) and given.value
            sep = given.value if literal else _NOT_LITERAL
        mark = reader.cuts
        texts = [c for c in _commands(source, reader, depth=depth + 1) if isinstance(c, str)]
        if not texts and reader.cuts > mark:
            return []
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
            mark = reader.cuts
            lists = [c for c in _commands(side, reader, depth=depth + 1) if isinstance(c, list)]
            if not lists and reader.cuts > mark:
                return []
            sides.append(lists or [[_NOT_LITERAL]])
        return [left + right for left in sides[0] for right in sides[1]]
    return None


# Commands one call's words or text read through names may give; past this
# many, a word stays one the check can't read.
_COMMAND_LIMIT = 16
# The flag a program takes code to run after, as `python -c` or `sh -uc`
# does, as a word and at the end of a command's text.
_CODE_FLAG = re.compile(r"-[A-Za-z]*c")
_CODE_FLAG_BEFORE = re.compile(r"(?:^|\s)-[A-Za-z]*c\s+['\"]?$")
# Calls that give back the text they're handed, or None: `shutil.which("ffmpeg")`
# is the path of that same program.
_SAME_TEXT_CALLS = frozenset({"shutil.which", "str", "os.fspath"})


def _literal_texts(node: ast.AST, reader: _Reader, *, depth: int = 0) -> list[str] | None:
    """The literal texts `node` can hold, as a program's or a module's name:
    a literal; each name's values (`_name_texts`); either side of an if-else
    and each value of an `or`; the entries of a list or dict it looks up
    (`_Reader.items_of`); what `shutil.which` or `str` is handed; and any
    other value `_Reader.paths` reads whole. Empty when it holds none: a
    parameter of a function nothing calls. None when a part is a value the
    check can't read, or past the depth limit (`stops`)."""
    if reader.stops("texts", node, depth=depth, limit=_PARAM_DEPTH):
        return None
    parts: Sequence[ast.AST]
    if isinstance(node, ast.Constant):
        return [node.value] if isinstance(node.value, str) else None
    if isinstance(node, ast.Name):
        return _name_texts(node, reader, depth=depth)
    if isinstance(node, ast.IfExp):
        parts = [node.body, node.orelse]
    elif isinstance(node, ast.BoolOp):
        parts = node.values
    elif (isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords
          and reader.dotted(node.func) in _SAME_TEXT_CALLS):
        parts = node.args
    elif isinstance(node, ast.Subscript):
        items = reader.items_of(node=node.value, key=node.slice, depth=depth + 1)
        if items is None:
            return None
        parts = items
    else:
        got = reader.paths(node, cell=reader.cell_of(node))
        if isinstance(got, _Unknown) or any(_unreadable_text(t) for t in got):
            return None
        return got
    return _all_texts(parts, reader, depth=depth + 1)


def _all_texts(parts: Sequence[ast.AST], reader: _Reader, *, depth: int) -> list[str] | None:
    """The literal texts each of `parts` can hold, together (`_literal_texts`)."""
    out: list[str] = []
    for part in parts:
        texts = _literal_texts(part, reader, depth=depth)
        if texts is None:
            return None
        out += texts
    return out


def _name_texts(node: ast.Name, reader: _Reader, *, depth: int) -> list[str] | None:
    """The literal texts the name `node` reads can hold (`_literal_texts`):
    each value an assignment gives it, each element of what a loop walks,
    and, for a parameter, what the call being read passes, else what each
    call of the function passes. A function nothing calls passes none."""
    cell = reader.cell_of(node)
    # `cmd = cmd + " -y"` reads the value before it.
    found = [b for b in reader.bindings_of(node.id, node=node)
             if not (b.cell is cell and b.how == "value" and _inside(node, b.value, cell))]
    if not found:
        return None
    out: list[str] = []
    for b in found:
        if b.how == "value":
            texts = _literal_texts(b.value, reader, depth=depth + 1)
        elif b.how in ("element", "unpacked"):
            texts = _element_texts(b.value, reader, depth=depth + 1)
        elif b.how == "param":
            texts = _param_texts(node.id, b, reader, depth=depth)
        else:
            texts = None
        if texts is None:
            return None
        out += texts
    return out


def _element_texts(walked: ast.AST, reader: _Reader, *, depth: int) -> list[str] | None:
    """The literal texts a loop over `walked` gives its names, read from
    each item it walks (`_walked_items`)."""
    items = _walked_items(walked, reader, depth=depth)
    return None if items is None else _all_texts(items, reader, depth=depth)


def _walked_items(walked: ast.AST, reader: _Reader, *, depth: int) -> list[ast.AST] | None:
    """What a loop over `walked` gives its names: each element of a list or
    dict it walks (`_Reader.items_of`), each key and value of a dict's
    `.items()`, and each element `enumerate` pairs with its count. None
    when the check can't place them."""
    if isinstance(walked, ast.Call) and walked.args and reader.dotted(walked.func) == "enumerate":
        walked = walked.args[0]
    elif (isinstance(walked, ast.Call) and isinstance(walked.func, ast.Attribute) and not walked.args
          and walked.func.attr in ("items", "keys", "values")):
        held = walked.func.value
        values = [b.value for b in reader.bindings_of(held.id, node=held)] if isinstance(held, ast.Name) else [held]
        if not values or not all(isinstance(v, ast.Dict) and None not in v.keys for v in values):
            return None
        return [part for v in values for part in (*v.keys, *v.values) if part is not None]
    return reader.items_of(node=walked, key=None, depth=depth)


def _param_texts(name: str, binding: _Binding, reader: _Reader, *, depth: int) -> list[str] | None:
    """The literal texts a parameter can hold (`_name_texts`)."""
    key = (id(binding.value), name)
    if key in reader.param_args:
        arg = reader.param_args[key]
        return None if arg is None else _literal_texts(arg.node, reader, depth=depth + 1)
    sites = reader.call_sites(binding.value, name)
    if not sites:
        return [] if reader.uncalled(binding.value) else None
    out: list[str] = []
    for site in sites:
        texts = None if site is None else _literal_texts(site.node, reader, depth=depth + 1)
        if texts is None:
            return None
        out += texts
    return out


def _unreadable_text(text: str) -> bool:
    """Whether `text`, read as a path, holds a part picked at run time, or a
    listing's wildcard: no name of a program or a module does."""
    return _RUN_TIME_PART in text or _HOLE in text or _NOT_LITERAL in text or any(ch in text for ch in "*?[")


def _command_word(node: ast.AST, reader: _Reader) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    text = _command_text(node, reader)
    return _NOT_LITERAL if text is None or _HOLE in text else text


def _command_text(node: ast.AST, reader: _Reader) -> str | None:
    """The text a command, or one of its words, is (`_sql_text`), with
    `sys.executable`, whole or as a part, read as `python`, the program it
    names (`_executable_filled`)."""
    if _names_executable(node, reader):
        return "python"
    text = _sql_text(node, reader)
    return None if text is None else _executable_filled(text, reader)


def _program_filled(text: str, reader: _Reader) -> list[str]:
    """`text`, a command's text, with `sys.executable` read as `python`
    (`_executable_filled`), and each placeholder whose code holds only
    literal texts read as each of them (`_literal_texts`), up to
    `_COMMAND_LIMIT` texts. Code handed to a `-c` stays unread."""
    texts = [_executable_filled(text, reader).lstrip()]
    for m in _NUMBERED_HOLE.finditer(texts[0]):
        if _CODE_FLAG_BEFORE.search(texts[0][: m.start()]):
            # Code handed to a `-c` through a name isn't read as a literal.
            continue
        node = reader.hole_node(m.group())
        options = None if node is None else _literal_texts(node, reader)
        if options and len(texts) * len(options) <= _COMMAND_LIMIT:
            texts = [t.replace(m.group(), option, 1) for t in texts for option in dict.fromkeys(options)]
    return texts


def _executable_filled(text: str, reader: _Reader) -> str:
    """`text` with each placeholder for `sys.executable` read as `python`."""
    if _HOLE not in text:
        return text
    return _NUMBERED_HOLE.sub(
        lambda m: "python" if _names_executable(reader.hole_node(m.group()), reader) else m.group(), text,
    )


def _names_executable(node: ast.AST | None, reader: _Reader) -> bool:
    """Whether `node` reads `sys.executable`, there or through a name bound
    to it (`_Reader.module_member`)."""
    return node is not None and (
        reader.dotted(node) == "sys.executable" or reader.module_member(node) == "sys.executable"
    )


def _script_commands(text: str) -> list[list[str]]:
    """A shell script's commands, each the list of its words: split where a
    separator such as `;`, `&&`, `|` or a parenthesis, or a line break, ends
    one. A quoted line break stays in its word, and a word that starts with
    `#` starts a comment to the end of its line."""
    try:
        lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|()\n")
        lexer.whitespace = " \t\r"
        lexer.commenters = ""
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        tokens = re.findall(r"\n|[^ \t\r\n]+", text)
    commands: list[list[str]] = [[]]
    comment = False
    for token in tokens:
        if token in _SHELL_SEPARATORS or token and set(token) <= set(";&|()\n"):
            if comment and "\n" not in token:
                continue
            comment = False
            commands.append([])
        elif comment or token.startswith("#"):
            comment = True
        else:
            commands[-1].append(token)
    return commands


def _script_steps(text: str, *, cwds: Sequence[str] | _Unknown, depth: int) -> list[_ShellStep]:
    """The deletes and moves a shell script runs, command by command, each
    after the folders its `cd`s reach."""
    out: list[_ShellStep] = []
    for words in _script_commands(text):
        bare, _ = _command_words(words)
        if bare and bare[0] in _CD_WORDS:
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
        if _SHELL_ASSIGNMENT.match(out[0]) or out[0] in _SHELL_KEYWORDS:
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
        script = _shell_script(args, stdin)
        if isinstance(script, _Unknown):
            return picked
        return [] if script is None or depth >= _SHELL_DEPTH else _script_steps(script, cwds=cwds, depth=depth + 1)
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
                if a in _FIND_EXEC_FLAGS]
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


def _shell_script(args: list[str], stdin: str | None) -> str | _Unknown | None:
    """The script a shell runs, given its arguments `args`: the text after
    `-c`; or, when it reads its script from its input, `stdin`, the text
    the call gives it, or unknown when the check can't read one. None when
    it runs a script file."""
    script = next((args[i + 1] for i, a in enumerate(args[:-1])
                   if a.startswith("-") and not a.startswith("--") and "c" in a), None)
    if script is None and ("-s" in args or not [a for a in args if not a.startswith("-")]):
        # The script comes from the shell's input, piped in or passed as `input=`.
        return _UNKNOWN if stdin is None else stdin
    return script


@dataclass(frozen=True)
class _Launch:
    """The Python a command launches: whether it runs code the check can't
    read, and the literal code it hands `python -c`, each read as one more
    cell (`_exec_cells`)."""

    unread: bool = False
    sources: tuple[str, ...] = ()

    @classmethod
    def joined(cls, launches: Sequence[_Launch]) -> _Launch:
        """What `launches`, the commands of one call or script, launch together."""
        return cls(
            unread=any(launch.unread for launch in launches),
            sources=tuple(source for launch in launches for source in launch.sources),
        )


def _python_launch(call: ast.Call, reader: _Reader) -> _Launch | None:
    """The Python `call` launches when it runs a program (`_LAUNCHERS`),
    named through the imports or a name bound to it (`_Reader.module_member`),
    or made through `functools.partial`: each command it can run, read by
    `_command_launch`. The program is the path the exec and spawn forms
    take before the program's words, or `executable=` when the call gives
    it. A call whose command the check can't read at all may run any
    program. None for any other call."""
    partial = reader.partial_of(node=call.func) if isinstance(call.func, (ast.Name, ast.Call)) else None
    if partial is not None:
        call = _partial_call(partial=partial, call=call)
    last = _last_name(call.func)
    if last not in _LAUNCH_NAMES and last not in reader.runner_aliases:
        return None
    name = reader.dotted(call.func)
    if name not in _LAUNCHERS:
        name = reader.module_member(call.func) or ""
        if name not in _LAUNCHERS:
            return None
    commands = _shell_commands(call, name, reader)
    index = _SPREAD_ARGV_FUNCS.get(name, _LIST_ARGV_FUNCS.get(name, 0))
    program = call.args[index - 1] if 0 < index <= len(call.args) else _keyword(call, "executable")
    if program is not None:
        word = _command_word(program, reader)
        words = _literal_texts(program, reader) if _unreadable_word(word) else [word]
        if words == []:
            # A parameter of a function nothing calls: it never runs.
            return _Launch()
        # The words given start with the name the program runs under, which
        # it doesn't read; a command's text runs as the program's `-c`, as
        # `shell=True` runs it.
        commands = [
            [w, *c[1:]] if isinstance(c, list) else [w, "-c", c] for w in words or [word] for c in commands
        ] or [[w] for w in words or [word]]
    if not commands:
        arg = _command_arg(call, name)
        first = arg.elts[0] if isinstance(arg, (ast.List, ast.Tuple)) and arg.elts else arg
        # A command a function nothing calls takes from its parameters never runs.
        return _Launch(unread=first is None or _literal_texts(first, reader) != [])
    given = _keyword(call, "input")
    stdin = None if given is None else _sql_text(given, reader)
    moved = _keyword(call, "cwd") is not None or _keyword(call, "env") is not None
    return _Launch.joined([
        _script_launch(c, depth=0, stdin=stdin, moved=moved) if isinstance(c, str)
        else _command_launch(c, depth=0, stdin=stdin, moved=moved)
        for c in commands
    ])


def _script_launch(text: str, *, depth: int, stdin: str | None, moved: bool) -> _Launch:
    """The Python a shell script launches, command by command
    (`_command_launch`). After a `cd`, a command runs in another folder
    than the cell's, as `moved` says it does from the start. Only the first
    command reads `stdin`."""
    launches: list[_Launch] = []
    for words in (c for c in _script_commands(text) if c):
        bare, _ = _command_words(words)
        if bare and bare[0] in ("cd", "pushd"):
            moved = True
        else:
            launches.append(_command_launch(words, depth=depth, stdin=None if launches else stdin, moved=moved))
    return _Launch.joined(launches)


def _command_launch(words: list[str], *, depth: int, stdin: str | None, moved: bool) -> _Launch:
    """The Python one command launches: python itself (`_python_arguments`),
    a `.py` script, a tool that loads a project's code, a test or a
    notebook (`_PYTHON_TOOLS`), the command a runner such as `uv run` runs
    (`_ENV_RUNNERS`), pip installing from a source the check can't read
    (`_pip_unread`), and what a shell's script or a `find -exec` runs in
    turn. A program word the check can't read may be any program, and so
    may a shell's script past four shells deep. `stdin` is the text the
    call gives the command as its input, when the check can read it;
    `moved` says the command runs in another folder or environment than
    the cell's, as a `VAR=value` before the program makes it do."""
    bare, _ = _command_words(words)
    if not bare:
        return _Launch()
    moved = moved or any(_SHELL_ASSIGNMENT.match(w) for w in words[: words.index(bare[0])])
    if _unreadable_word(bare[0]):
        return _Launch(unread=True)
    verb, args = bare[0].rsplit("/", 1)[-1], bare[1:]
    if verb in _SHELLS:
        script = _shell_script(args, stdin)
        if script is None:
            # A shell script file: not Python.
            return _Launch()
        if isinstance(script, _Unknown) or depth >= _SHELL_DEPTH:
            return _Launch(unread=True)
        return _script_launch(script, depth=depth + 1, stdin=None, moved=moved)
    if verb == "find":
        launches: list[_Launch] = []
        for i, word in enumerate(args):
            if word in _FIND_EXEC_FLAGS:
                command: list[str] = []
                for part in args[i + 1:]:
                    if part in (";", "+"):
                        break
                    command.append(part)
                launches.append(_command_launch(command, depth=depth, stdin=None, moved=moved or "dir" in word))
        return _Launch.joined(launches)
    target = _env_runner_target(verb, args)
    if target is not None:
        return _command_launch(target, depth=depth, stdin=stdin, moved=moved)
    pip_args = _pip_arguments(verb, args)
    if pip_args is not None:
        return _Launch(unread=_pip_unread(pip_args))
    if _PYTHON_PROGRAM.fullmatch(verb):
        return _python_arguments(args, moved=moved)
    first = args[0] if args else ""
    return _Launch(unread=(
        verb.endswith(_PYTHON_SCRIPTS) or verb in _PYTHON_TOOLS or (verb, first) in _PYTHON_TOOL_COMMANDS
        or (verb == "jupyter-nbconvert" or verb == "jupyter" and first == "nbconvert") and "--execute" in args
    ))


def _drop_leading_flags(args: list[str], value_flags: frozenset[str]) -> list[str]:
    """`args` past its leading flags: a `--flag value` or `-f value` whose
    flag takes a value (`value_flags`, long ones matched by unique prefix)
    drops its value too, a `--flag=value` drops nothing more."""
    rest = list(args)
    while rest and rest[0].startswith("-"):
        flag = rest.pop(0)
        name = flag.partition("=")[0]
        takes = flag in value_flags or (
            name.startswith("--") and any(o.startswith(name) for o in value_flags if o.startswith("--"))
        )
        if takes and "=" not in flag and rest:
            rest.pop(0)
    return rest


def _env_runner_target(verb: str, args: list[str]) -> list[str] | None:
    """The command a runner such as `uv run` or `conda run` (`_ENV_RUNNERS`)
    runs: `args` past the runner's subcommand and its own flags. A `uv run
    -m module` or `-s script` runs Python, read as `python -m` and a script
    do. None when the command isn't a runner's."""
    subcommands = _ENV_RUNNERS.get(verb)
    if subcommands is None:
        return None
    lead = _drop_leading_flags(args, _ENV_RUNNER_VALUE_FLAGS)
    for own in subcommands:
        if lead[: len(own)] != list(own):
            continue
        rest = lead[len(own):]
        while rest and rest[0].startswith("-"):
            flag, equals, value = rest[0].partition("=")
            if flag in _ENV_RUN_MODULE_FLAGS:
                return ["python", "-m", value if equals else (rest[1] if len(rest) > 1 else "")]
            if flag in _ENV_RUN_SCRIPT_FLAGS:
                return ["python", value if equals else (rest[1] if len(rest) > 1 else "")]
            rest.pop(0)
            takes = flag in _ENV_RUNNER_VALUE_FLAGS or (
                flag.startswith("--") and any(o.startswith(flag) for o in _ENV_RUNNER_VALUE_FLAGS))
            if takes and not equals and rest:
                rest.pop(0)
        return rest
    return None


def _pip_arguments(verb: str, args: list[str]) -> list[str] | None:
    """The arguments pip is given, as the `pip`/`pip3` program or as
    `uv pip`, past uv's own global flags before `pip`; None when the command
    isn't pip's."""
    if _PIP_PROGRAM.fullmatch(verb):
        return args
    if verb == "uv":
        rest = _drop_leading_flags(args, _ENV_RUNNER_VALUE_FLAGS)
        if rest[:1] == ["pip"]:
            return rest[1:]
    return None


def _python_arguments(args: list[str], *, moved: bool) -> _Launch:
    """What `python` runs given its arguments `args`. The literal code after
    `-c` is read as one more cell; it is unread when the check can't read
    the text, or when the command runs in another folder or environment
    (`moved`), where the files and modules it reaches differ. A module after
    `-m` is unread, but for `_SETUP_MODULES` and pip installing what
    `_pip_unread` lets through. So is a script, and code read from the
    input, with no script or with `-` or `-i`. Asked only for its version or
    its help, it runs nothing."""
    words = iter(args)
    for word in words:
        if word in _PYTHON_EXITS:
            return _Launch()
        if word == "--check-hash-based-pycs":
            next(words, None)
            continue
        if word == "--" or word == "-" or not word.startswith("-"):
            # A script, or the code its input gives it.
            return _Launch(unread=True)
        if word.startswith("--"):
            continue
        # A cluster of short options, such as -uc or -Wignore.
        for at, letter in enumerate(word[1:], start=2):
            value = word[at:]
            if letter in "Vh?":
                return _Launch()
            if letter == "i":
                return _Launch(unread=True)
            if letter in "cm":
                given = value or next(words, None)
                if given is None:
                    # Python stops with an error before it runs anything.
                    return _Launch()
                if letter == "m":
                    if _unreadable_word(given):
                        return _Launch(unread=True)
                    if given == "pip":
                        return _Launch(unread=_pip_unread(list(words)))
                    return _Launch(unread=given not in _SETUP_MODULES)
                if _HOLE in given or _NOT_LITERAL in given or "$" in given or "`" in given:
                    return _Launch(unread=True)
                return _Launch(unread=moved, sources=(given,))
            if letter in "WX":
                if not value:
                    next(words, None)
                break
    return _Launch(unread=True)


def _pip_long_kind(flag: str) -> Literal["source", "value", "other"]:
    """What a long pip option `flag` is, matched by unique prefix: a source
    flag (`--req` is `--requirement`), a value flag whose value isn't a
    source, or neither. A prefix of both reads as a source, since any source
    flag means the install can't be read."""
    name = flag.partition("=")[0]
    if name in _PIP_SOURCE_LONG or any(o.startswith(name) for o in _PIP_SOURCE_LONG):
        return "source"
    if name in _PIP_VALUE_LONG or any(o.startswith(name) for o in _PIP_VALUE_LONG):
        return "value"
    return "other"


def _pip_unread(args: list[str]) -> bool:
    """Whether pip, given `args`, installs from a source the check can't
    read: a requirements file, a folder, an archive (`PACKAGE_FILE_SUFFIXES`)
    or a word named like a script, a URL, a git repository, or an index or
    page of links it is pointed at, or a word the check can't read, which
    may be any of those. A command that installs from a file whatever
    follows (`_PIP_FILE_COMMANDS`, `uv pip sync`) always reads one. Its
    commands that take packages (`_PIP_PACKAGE_COMMANDS`) given names and
    version specs, and all its other commands, are none. A word the check
    reads as the command that is none of pip's means a global flag ate it,
    so the check can't tell what pip does."""
    rest = _drop_leading_flags(args, _PIP_VALUE_FLAGS)
    if not rest:
        return False
    command = rest[0]
    if command not in _PIP_COMMANDS:
        return True
    if command in _PIP_FILE_COMMANDS:
        return True
    if command not in _PIP_PACKAGE_COMMANDS:
        return False
    words = iter(rest[1:])
    for word in words:
        if word == "--":
            return any(_pip_source_operand(operand) for operand in words)
        if word.startswith("--"):
            kind = _pip_long_kind(word)
            if kind == "source":
                return True
            if kind == "value" and "=" not in word:
                next(words, None)
        elif word.startswith("-") and word != "-":
            # A cluster of short flags: the first that takes a value takes the rest.
            for letter in word[1:]:
                if letter in _PIP_SOURCE_LETTERS:
                    return True
                if letter in _PIP_VALUE_LETTERS:
                    break
        elif _pip_source_operand(word):
            return True
    return False


def _pip_source_operand(word: str) -> bool:
    """Whether `word`, an operand of a pip install, names a source the check
    can't read rather than a package on the index: a word it can't read, an
    archive or script, a path, a URL or a git repository."""
    return _unreadable_word(word) or (
        word.lower().endswith((*PACKAGE_FILE_SUFFIXES, *_PYTHON_SCRIPTS)) or "/" in word
        or word.startswith((".", "~")) or "://" in word or word.lower().startswith("git+")
    )


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


# ── HTTP ──────────────────────────────────────────────────────────────────

_HTTP_LIBRARIES = frozenset({"requests", "httpx"})
_HTTP_VERBS = frozenset({"get", "post", "put", "patch", "delete", "head", "options"})
# Calls that take the method as their first argument and the URL as their second.
_HTTP_ANY_VERB = frozenset({"request", "stream", "Request"})
_HTTP_CLIENTS = frozenset({
    "requests.Session", "requests.session", "requests.sessions.Session", "httpx.Client", "httpx.AsyncClient",
})
# Methods held with no rule: the ones that change or remove what a URL names.
_HELD_METHODS = frozenset({"PUT", "PATCH", "DELETE"})
_URL_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")
# A URL glob's wildcards, as `_url_could_match` reads them.
_ONE, _SEGMENT, _ANY = "\x00?", "\x00*", "\x00**"


@dataclass(frozen=True)
class _HttpCall:
    """A call that sends, or builds, an HTTP request: its method, named in
    the call or as the node that picks it, the node that names its URL, and
    the base URL its client joins a relative one to."""

    method: str | ast.AST
    url: ast.AST | None
    base: ast.AST | None = None


def _http_client(node: ast.AST, reader: _Reader, *, depth: int = 0) -> ast.Call | None:
    """The requests Session or httpx Client call that made what `node`
    holds, when it visibly holds one. It stops at the depth limit
    (`_Reader.stops`)."""
    if reader.stops("http", node, depth=depth, limit=6):
        return None
    values: list[ast.AST] = []
    if isinstance(node, ast.Call):
        if reader.dotted(node.func) in _HTTP_CLIENTS:
            return node
        if isinstance(node.func, ast.Name) and node.func.id in reader.functions:
            values = [n.value for n in ast.walk(reader.functions[node.func.id])
                      if isinstance(n, ast.Return) and n.value is not None]
    elif isinstance(node, ast.Name):
        for b in reader.bindings_of(node.id, node=node):
            if b.how in ("value", "context"):
                values.append(b.value)
            elif b.how == "param":
                key = (id(b.value), node.id)
                args = [reader.param_args[key]] if key in reader.param_args else reader.call_sites(b.value, node.id)
                values += [a.node for a in args if a is not None]
    for value in values:
        client = _http_client(value, reader, depth=depth + 1)
        if client is not None:
            return client
    return None


def _is_urllib_request(node: ast.AST, reader: _Reader, *, depth: int = 0) -> bool:
    """Whether `node` holds a urllib `Request`. It stops at the depth limit
    (`_Reader.stops`)."""
    if isinstance(node, ast.Call):
        return reader.dotted(node.func) == "urllib.request.Request"
    return isinstance(node, ast.Name) and not reader.stops("request", node, depth=depth, limit=5) and any(
        b.how == "value" and _is_urllib_request(b.value, reader, depth=depth + 1)
        for b in reader.bindings_of(node.id, node=node)
    )


def _given(node: ast.AST | None) -> ast.AST | None:
    """`node`, or None when it is a literal None."""
    return None if isinstance(node, ast.Constant) and node.value is None else node


def _urllib_call(call: ast.Call, name: str, reader: _Reader) -> _HttpCall | None:
    """urllib's `Request` or `urlopen`: its method is POST when it sends
    data and GET when it doesn't, unless a `Request` names one."""
    url = _target_arg(call, keywords=("url",))
    if name.endswith("urlopen") and url is not None and reader.read_through(
        lambda: _is_urllib_request(url, reader)):
        # The Request it opens is read where it is made; one the depth limit
        # stopped short of leaves the cell unchecked (`read_through`).
        return None
    method = None
    if name.endswith("Request"):
        method = _given(call.args[5] if len(call.args) > 5 else _keyword(call, "method"))
    data = _given(call.args[1] if len(call.args) > 1 else _keyword(call, "data"))
    return _HttpCall(method=method or ("GET" if data is None else "POST"), url=url)


def _http_call(call: ast.Call, reader: _Reader) -> _HttpCall | None:
    """The request `call` sends or builds: a requests or httpx function or
    `Request`, a Session's or Client's method, or urllib's `Request` or
    `urlopen`."""
    name = reader.dotted(call.func)
    if name in ("urllib.request.Request", "urllib.request.urlopen"):
        return _urllib_call(call, name, reader)
    library, _, verb = name.rpartition(".")
    base = None
    if library not in _HTTP_LIBRARIES:
        if not isinstance(call.func, ast.Attribute):
            return None
        verb = call.func.attr
        if verb not in _HTTP_VERBS and verb not in ("request", "stream"):
            return None
        client = _http_client(call.func.value, reader)
        if client is None:
            return None
        base = _keyword(client, "base_url")
    if verb in _HTTP_VERBS:
        return _HttpCall(method=verb.upper(), url=_target_arg(call, keywords=("url",)), base=base)
    if verb not in _HTTP_ANY_VERB:
        return None
    method = _given(call.args[0] if call.args else _keyword(call, "method"))
    url = call.args[1] if len(call.args) > 1 else _keyword(call, "url")
    return None if method is None else _HttpCall(method=method, url=url, base=base)


def _http_methods(node: ast.AST, reader: _Reader, *, depth: int = 0) -> list[str] | _Unknown:
    """The methods `node` can name, in upper case; unknown when the code
    picks one when it runs. A parameter of a function nothing calls names
    none: the request is read where the function is called."""
    if depth > 6:
        return _UNKNOWN
    if isinstance(node, ast.Constant):
        return [node.value.upper()] if isinstance(node.value, str) else _UNKNOWN
    if isinstance(node, ast.IfExp):
        body = _http_methods(node.body, reader, depth=depth + 1)
        orelse = _http_methods(node.orelse, reader, depth=depth + 1)
        return _UNKNOWN if isinstance(body, _Unknown) or isinstance(orelse, _Unknown) else body + orelse
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("upper", "lower", "strip") and not node.args):
        return _http_methods(node.func.value, reader, depth=depth + 1)
    if not isinstance(node, ast.Name):
        return _UNKNOWN
    found = reader.bindings_of(node.id, node=node)
    if not found:
        return _UNKNOWN
    out: list[str] = []
    for b in found:
        if b.how == "param":
            key = (id(b.value), node.id)
            args = [reader.param_args[key]] if key in reader.param_args else reader.call_sites(b.value, node.id)
            values: list[ast.AST | None] = [None if a is None else a.node for a in args]
        elif b.how == "value":
            values = [b.value]
        elif b.how == "element" and isinstance(b.value, (ast.List, ast.Tuple, ast.Set)):
            values = list(b.value.elts)
        else:
            return _UNKNOWN
        for value in values:
            got = _UNKNOWN if value is None else _http_methods(value, reader, depth=depth + 1)
            if isinstance(got, _Unknown):
                return got
            out += got
    return out


def _url_text(node: ast.AST, reader: _Reader, *, depth: int = 0) -> str:
    """A URL's text, each part the code picks when it runs a numbered
    placeholder (`_Reader.hole`). A name bound to one value, or a parameter
    the call being read passes, is read through to it."""
    if depth <= 8:
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float)) and not isinstance(
            node.value, bool
        ):
            return str(node.value)
        if isinstance(node, ast.JoinedStr):
            return "".join(
                str(v.value) if isinstance(v, ast.Constant)
                else _url_text(v.value, reader, depth=depth + 1)
                if isinstance(v, ast.FormattedValue) and v.conversion == -1 and v.format_spec is None
                else reader.hole(v)
                for v in node.values
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return _url_text(node.left, reader, depth=depth + 1) + _url_text(node.right, reader, depth=depth + 1)
        if isinstance(node, ast.Name):
            arg = reader.argument_of(node)
            if arg is not None:
                return _url_text(arg.node, reader, depth=depth + 1)
            found = reader.bindings_of(node.id, node=node)
            if len(found) == 1 and found[0].how == "value":
                return _url_text(found[0].value, reader, depth=depth + 1)
    return reader.hole(node)


def _url_could_match(url: str, glob: str) -> bool:
    """Whether `url`, a placeholder standing for any text, can match
    `glob`, ignoring case and the URL's query string and fragment. In the
    glob, `*` stands for text within one path segment, `**` for any text
    and `?` for one character other than `/`."""
    text: list[str | None] = []  # a character, or None for a part picked at run time
    for i, part in enumerate(_NUMBERED_HOLE.split(url)):
        if i % 2:
            text.append(None)
            continue
        cut = re.search(r"[?#]", part)
        text += list((part[:cut.start()] if cut else part).casefold())
        if cut:
            break
    pattern: list[str] = []
    for m in re.finditer(r"\*\*|\*|\?|.", glob.casefold(), re.S):
        pattern.append({"**": _ANY, "*": _SEGMENT, "?": _ONE}.get(m.group(), m.group()))
    # A walk over pairs of positions: (i, j) when some text takes the URL
    # to `i` and the glob to `j` at once.
    seen: set[tuple[int, int]] = set()
    stack = [(0, 0)]
    while stack:
        i, j = stack.pop()
        if (i, j) in seen:
            continue
        seen.add((i, j))
        if i == len(text) and j == len(pattern):
            return True
        u = text[i] if i < len(text) else ""
        g = pattern[j] if j < len(pattern) else ""
        if i < len(text) and u is None:
            stack.append((i + 1, j))  # the part stands for nothing
        if g in (_SEGMENT, _ANY):
            stack.append((i, j + 1))  # the wildcard stands for nothing
        if i == len(text) or j == len(pattern):
            continue
        if u is None:
            if g not in (_SEGMENT, _ANY):
                stack.append((i, j + 1))  # the part supplies the glob's next character
        elif g == _ANY or g == _SEGMENT and u != "/":
            stack.append((i + 1, j))
        elif g == _ONE and u != "/" or g == u:
            stack.append((i + 1, j + 1))
    return False


def http_write_rule_matches(rule: HttpWriteRule, *, method: str, url: str) -> bool:
    """Whether a request with `method` to `url` matches `rule`, as the gate
    matches a cell's request: the rule's method or `*`, and its glob against
    the URL, ignoring case, the query string and the fragment. Public for a
    service that applies the same rules to requests it forwards. The URL is
    matched as given, so such a service passes it decoded and normalized,
    as it forwards it."""
    return rule.method in ("*", method.upper()) and _url_could_match(url, rule.url_glob)


def _http_findings(
    reader: _Reader,
    cell: _Cell,
    *,
    rules: Sequence[HttpWriteRule],
    root: ast.AST | None = None,
    report: _Report | None = None,
) -> list[HttpWrite]:
    """The requests `cell` sends that may change data on a server: every
    PUT, PATCH and DELETE, one whose method the code picks when it runs, and
    one that matches a rule in `rules`. With `root`, the ones a function of
    that cell sends, reported at `report`, the call in the checked cell."""
    out: list[HttpWrite] = []
    for node in ast.walk(cell.tree if root is None else root):
        if not isinstance(node, ast.Call):
            continue
        mark = reader.cuts
        request = _http_call(node, reader)
        if request is None:
            if _may_send_held(node, rules):
                # A client the depth limit stopped short of.
                reader.read_past(mark, lambda: _http_call(node, reader))
            continue
        url = reader.hole(None) if request.url is None else _url_text(request.url, reader)
        if request.base is not None and not (_URL_SCHEME.match(url) or _HOLE_MARK.match(url)):
            url = _url_text(request.base, reader).rstrip("/") + "/" + url.lstrip("/")
        if isinstance(request.method, str):
            methods: list[str] | _Unknown = [request.method]
        else:
            methods = _http_methods(request.method, reader)
        if isinstance(methods, _Unknown):
            held = [reader.hole(_deciding_node(request.method, reader))]
        else:
            held = [m for m in dict.fromkeys(methods) if m in _HELD_METHODS or any(
                http_write_rule_matches(r, method=m, url=url) for r in rules
            )]
        line = report.line if report else cell.host_line or node.lineno
        call = report.call if report else scrub_output(cell.source(node))
        url_code = reader.hole_code(url)
        out += [
            HttpWrite(
                method=reader.show_holes(method), url=_scrub(reader.show_holes(url)), call=call, line=line,
                run_time_parts=url_code is None and _NUMBERED_HOLE.search(url) is not None,
                method_code=_scrub_code_or_none(reader.hole_code(method)),
                url_code=_scrub_code_or_none(url_code),
            )
            for method in held
        ]
    return out


def _may_send_held(call: ast.Call, rules: Sequence[HttpWriteRule]) -> bool:
    """Whether `call`, a client's method, sends a request the card may hold
    to an address the check can't place: a PUT, PATCH or DELETE, a method a
    rule in `rules` names, or `request` and `stream`, which take the method
    as an argument."""
    verb = call.func.attr if isinstance(call.func, ast.Attribute) else ""
    if verb in ("request", "stream"):
        return True
    method = verb.upper()
    return verb in _HTTP_VERBS and (method in _HELD_METHODS or any(r.method in ("*", method) for r in rules))


def _deciding_node(node: ast.AST, reader: _Reader) -> ast.AST:
    """The code that decides `node` as the checked call writes it: a
    parameter is followed to the argument the call passes for it, so the
    card shows `verb` from `api(verb, ...)`, not the helper's `method`.

    Only a parameter the helper never assigns again: one it may reassign
    (`if ...: method = 'DELETE'`) is decided inside the helper too, so the
    card names the helper's own `method`, not the caller's value alone."""
    for _ in range(8):
        if not isinstance(node, ast.Name):
            break
        bindings = reader.bindings_of(node.id, node=node)
        if len(bindings) != 1 or bindings[0].how != "param":
            break
        arg = reader.argument_of(node)
        if arg is None:
            break
        node = arg.node
    return node


def _scrub_code_or_none(code: str | None) -> str | None:
    """The run-time code the card quotes for a method or address, scrubbed
    as one text: `ast.unparse` writes a space after a dict's colon, so a
    token held in a dict is a word of its own, which `_scrub_code` would
    pass as written."""
    return None if code is None else scrub_output(code)


# ── Entry point ───────────────────────────────────────────────────────────

def _plain_lookup(call: ast.Call) -> ast.expr:
    """`getattr(x, "name")` as `x.name`, also with a constant default as in
    `getattr(x, "name", None)`, and `__import__("os")` or
    `importlib.import_module("os")` as `os`, so a call made through one is
    read like the plain spelling; any other call as it is."""
    func, args = call.func, call.args
    attr = len(args) in (2, 3) and isinstance(args[1], ast.Constant) and isinstance(args[1].value, str)
    if (isinstance(func, ast.Name) and func.id == "getattr" and attr and not call.keywords
            and (len(args) == 2 or isinstance(args[2], ast.Constant))):
        return ast.copy_location(ast.Attribute(value=args[0], attr=args[1].value, ctx=ast.Load()), call)
    return _plain_import(call) or call


def _plain_import(call: ast.Call) -> ast.expr | None:
    """`__import__("os")` or `importlib.import_module("os")` as `os`, and
    `importlib.import_module("os.path")` as `os.path`; None for any other call."""
    args = call.args
    named = bool(args) and isinstance(args[-1], ast.Constant) and isinstance(args[-1].value, str)
    importer = _last_name(call.func)
    if not (importer in ("__import__", "import_module") and len(args) == 1 and named and args[0].value):
        return None
    parts = args[0].value.split(".")
    parts = parts[:1] if importer == "__import__" else parts
    out: ast.expr = ast.copy_location(ast.Name(id=parts[0], ctx=ast.Load()), call)
    for part in parts[1:]:
        out = ast.copy_location(ast.Attribute(value=out, attr=part, ctx=ast.Load()), call)
    return out


def _read_lookups_plainly(tree: ast.Module) -> frozenset[ast.Name]:
    """Swaps each literal lookup `_plain_lookup` reads into `tree` in place,
    deepest nodes first, without recursion, so a nested lookup is read
    before the one around it and a long expression can't exhaust the stack.
    Returns the plain names that stand for a literal import now, each the
    name of the module it imports, as `job` stands for
    `importlib.import_module("job")`: an import the check reads (`_Modules`)."""
    imported: set[ast.Name] = set()

    def plain(call: ast.Call) -> ast.expr:
        module = _plain_import(call)
        if module is None:
            return _plain_lookup(call)
        root = module
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name):
            imported.add(root)
        return module

    for node in reversed(list(ast.walk(tree))):
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.Call):
                setattr(node, field, plain(value))
            elif isinstance(value, list):
                value[:] = [plain(v) if isinstance(v, ast.Call) else v for v in value]
    return frozenset(imported)


def _parse(code: str) -> ast.Module:
    """The cell's tree, from the same healed text the runner compiles.
    `_Cell` reads its literal `getattr` and import lookups as plain names.

    Quiet: the agent's own invalid escape sequences would otherwise print a
    SyntaxWarning in the host's log for every check; the runner reports them
    where they belong, in the cell's output.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(heal_surrogate_source(code))


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


# The place a cell of literal code takes among the parsed cells, which a
# conversation fixes: the exec cache keys a parsed cell by it and the source.
class _ExecCellKey(NamedTuple):
    text: str
    index: int
    host_line: int | None
    launched: bool
    host: int
    shared: bool
    namespace: str | None


class _ExecCells:
    """Parsed cells of literal code exec(), a runner of text or `python -c`
    runs, kept across checks instead of parsed once per check.

    Each check reads the same literal in an earlier cell, which doesn't
    change, so parsing an 83k-character script and walking it once per
    process, not once per check, saves that work. A cell is keyed by its
    source and the place it takes among the parsed cells (`_ExecCellKey`),
    which a given conversation fixes; a source that doesn't parse is kept as
    None, so it isn't parsed again. Budgeted and locked like `_EarlierCells`.
    """

    def __init__(self, *, budget: int, idle: float = 600.0) -> None:
        self._budget = budget
        self._idle = idle
        self._size = 0
        self._cells: OrderedDict[_ExecCellKey, _CachedCell] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: _ExecCellKey, make: Callable[[], _Cell | None]) -> _Cell | None:
        with self._lock:
            cached = self._cells.get(key)
            if cached is not None:
                cached.read_at = time.monotonic()
                self._cells.move_to_end(key)
                return cached.cell
        cell = make()
        if len(key.text) > self._budget:
            return cell
        with self._lock:
            now = time.monotonic()
            while self._cells and self._size + len(key.text) > self._budget:
                old, oldest = next(iter(self._cells.items()))
                if now - oldest.read_at < self._idle:
                    break
                self._cells.popitem(last=False)
                self._size -= len(old.text)
            if key not in self._cells and self._size + len(key.text) <= self._budget:
                self._cells[key] = _CachedCell(cell=cell, read_at=now)
                self._size += len(key.text)
        return cell


_EXEC_CELLS = _ExecCells(budget=1_000_000)


def _last_name(node: ast.AST) -> str:
    """The last name `node` is spelled with: `f` for `f`, `run` for
    `m.run`; "" for anything else."""
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""


def _spelled_exec(func: ast.AST) -> bool:
    """Whether a call's `func` is spelled as exec(), eval() or compile(): the
    bare name, a member of that name, or an entry under that literal key.
    `_Reader.exec_builtin` says which of them reach the builtin."""
    if isinstance(func, ast.Name):
        return func.id in _EXEC_CALLS
    if isinstance(func, ast.Attribute):
        return func.attr in _EXEC_CALLS
    return isinstance(func, ast.Subscript) and isinstance(func.slice, ast.Constant) and func.slice.value in _EXEC_CALLS


def _exec_source(call: ast.Call, builtin: str) -> ast.AST | None:
    """The code `call` hands the builtin `builtin` to run or compile: its
    first argument, or compile's `source=` or `**` keywords. None when it
    hands none, so nothing runs."""
    if call.args:
        return call.args[0]
    if builtin != "compile":
        return None
    return next((k.value for k in call.keywords if k.arg in ("source", None)), None)


def _literal_source(node: ast.AST) -> str | None:
    """The text of a str or bytes literal, or of an f-string with nothing to
    fill in; None for any other code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
        return node.value if isinstance(node.value, str) else node.value.decode("utf-8", errors="replace")
    if isinstance(node, ast.JoinedStr) and all(isinstance(v, ast.Constant) for v in node.values):
        return "".join(str(v.value) for v in node.values)
    return None


@dataclass(frozen=True, eq=False)
class _Code:
    """Code the check reads, and the cell it is in: a cell's whole tree, or
    a function in it."""

    cell: _Cell
    root: ast.AST


@dataclass(eq=False)
class _ExecRead:
    """The literal code exec(), eval(), compile(), the runners of text and
    `python -c` run, each source read as one more cell (`_exec_cells`).
    `cells` holds them all in the order read, which is their order among
    the parsed cells. `ran` holds them by the id of the code whose calls run
    them, a cell's tree or a function an earlier cell defined, with the
    cells their own sources make in turn. `tried` holds the id of each call
    whose literal sources were read, whether they parsed or not; a call
    past `_EXEC_LIMIT` isn't in it."""

    cells: list[_Cell]
    ran: dict[int, list[_Cell]]
    tried: set[int]

    @classmethod
    def empty(cls) -> _ExecRead:
        """A read that read nothing."""
        return cls(cells=[], ran={}, tried=set())

    @classmethod
    def joined(cls, reads: Sequence[_ExecRead]) -> _ExecRead:
        """What `reads` read together, their cells in the order given."""
        return cls(
            cells=[c for read in reads for c in read.cells],
            ran={key: ran for read in reads for key, ran in read.ran.items()},
            tried={key for read in reads for key in read.tried},
        )

    def of(self, root: ast.AST) -> list[_Cell]:
        """The cells the code `root` runs, and those they run in turn."""
        return self.ran.get(id(root), [])

    def with_ran(self, cell: _Cell, root: ast.AST) -> list[_Code]:
        """`root`, code of `cell`, and each cell it runs: all the code a call
        `root` runs may be made from."""
        return [_Code(cell=cell, root=root), *(_Code(cell=c, root=c.tree) for c in self.of(root))]


def _exec_cells(
    reader: _Reader, starts: Sequence[_Cell], *, index: int, ran_only: bool = False,
) -> _ExecRead:
    """The literal code each of `starts` passes to exec(), eval() or
    compile(), to a runner of text such as `timeit.timeit`
    (`_runner_code`), or to `python -c` in a program it runs
    (`_python_launch`), each source read as one more cell numbered from
    `index`, and the literal code those cells pass on in turn. A call
    spelled with one of the builtins' names counts whatever binds the name;
    any other counts where `_Reader.exec_builtin` reads it as the builtin.
    When a function an earlier cell defined spells one, the literal code
    each such function a start calls runs is read too, within `_CALL_DEPTH`
    calls, under that function. Past `_EXEC_LIMIT` cells it stops. With
    `ran_only`, only the code that runs when its cell runs counts
    (`_nodes_that_run`), and a function only passed on as a value isn't
    followed."""
    read = _ExecRead.empty()

    def exec_named(func: ast.AST) -> str | None:
        # A bare exec, eval or compile counts whatever binds the name, so a
        # def that shadows one keeps the read it had.
        if isinstance(func, ast.Name) and func.id in _EXEC_CALLS:
            return func.id
        return reader.exec_builtin(func)

    def sources(cell: _Cell, root: ast.AST, *, host: int) -> None:
        into = read.ran.setdefault(id(root), [])
        pending = [_Code(cell=cell, root=root)]
        while pending and len(read.cells) < _EXEC_LIMIT:
            code = pending.pop()
            # A cell's own calls are listed in walk order already.
            whole = code.root is code.cell.tree
            nodes = code.cell.nodes_run_by(code.root) if ran_only else code.cell.calls if whole else ast.walk(code.root)
            for node in nodes:
                if not isinstance(node, ast.Call):
                    continue
                builtin = exec_named(node.func)
                launch: _Launch | None = None
                if builtin:
                    source = _exec_source(node, builtin)
                    given: Sequence[ast.AST] = () if source is None else (source,)
                else:
                    runner = _runner_code(node, reader)
                    given = () if runner is None else runner.sources
                    launch = None if runner is not None else _python_launch(node, reader)
                texts = [t for t in map(_literal_source, given) if t is not None]
                if launch is not None:
                    texts = list(launch.sources)
                if not texts or any(len(text) > _LITERAL_CODE_LIMIT for text in texts):
                    # Past the limit, literal code is left unread.
                    continue
                if len(read.cells) + len(texts) > _EXEC_LIMIT:
                    return
                read.tried.add(id(node))
                place = _exec_place(node, code.cell, reader) if builtin else _ExecPlace(shared=False)
                for text in texts:
                    key = _ExecCellKey(
                        text=text, index=index + len(read.cells),
                        host_line=code.cell.host_line or node.lineno,
                        launched=launch is not None or code.cell.launched, host=host,
                        shared=place.shared and code.cell.shared,
                        namespace=place.namespace if code.cell.shared else code.cell.namespace)
                    # Literal code an earlier cell runs doesn't change, so its
                    # parsed cell is kept across checks; the checked cell's own
                    # changes each cell, so it is built fresh.
                    if host < reader.current.index:
                        ran = _EXEC_CELLS.get(key, lambda k=key: _exec_cell(k))
                    else:
                        ran = _exec_cell(key)
                    if ran is None:
                        continue
                    reader.read_too(ran)
                    read.cells.append(ran)
                    into.append(ran)
                    pending.append(_Code(cell=ran, root=ran.tree))

    # The shortest chain of calls each function was followed at.
    followed: dict[int, int] = {}

    def follow(cell: _Cell, root: ast.AST, chain: tuple[int, ...], *, host: int) -> None:
        for callee in _earlier_function_uses(reader, root, cell, ran_only=ran_only):
            func = callee.func
            if (id(func) in chain or len(chain) >= _CALL_DEPTH or ran_only and callee.call is None
                    or followed.get(id(func), _CALL_DEPTH) <= len(chain)):
                continue
            followed[id(func)] = len(chain)
            home = reader.cell_of(func)
            with reader.calling(func, callee.call, bound=callee.bound):
                if id(func) not in read.ran:
                    sources(home, func, host=host)
                for code in read.with_ran(home, func):
                    follow(code.cell, code.root, (*chain, id(func)), host=host)

    def runs_text(call: ast.Call) -> bool:
        return exec_named(call.func) is not None or _runner_code(call, reader) is not None

    def hands_python_code(call: ast.Call) -> bool:
        launch = _python_launch(call, reader)
        return launch is not None and bool(launch.sources)

    for start in starts:
        sources(start, start.tree, host=start.host)
    earlier = [c for c in reader.cells if c.host < reader.current.index]
    if any(runs_text(c) for e in earlier for c in e.function_execs) or any(
        hands_python_code(c) for e in earlier for c in e.function_launches
    ) or any(full.startswith("builtins.") for full in reader.imports.values()):
        for start in starts:
            for code in read.with_ran(start, start.tree):
                follow(code.cell, code.root, (), host=start.host)
    return read


@dataclass(frozen=True)
class _ExecPlace:
    """Where the names literal code exec() or eval() runs land: in the cell
    namespace (`shared`), or in the namespace a name holds (`namespace`),
    or neither, as for a function's own locals."""

    shared: bool
    namespace: str | None = None


def _exec_place(call: ast.Call, cell: _Cell, reader: _Reader) -> _ExecPlace:
    """Where the code `call`, code of `cell`, hands exec(), eval() or
    compile() binds its names. exec() at the cell's top level given no
    namespace, or `globals()`, binds them in the cell namespace; given a
    dict or a module's `__dict__`, in that; inside a function, in the
    function's own locals. A compile() binds where the exec() or eval() it
    is handed to does, and nowhere the check can follow otherwise."""
    parent = cell.parents.get(call)
    if reader.exec_builtin(call.func) == "compile":
        runs = isinstance(parent, ast.Call) and call in parent.args[:1] and reader.exec_builtin(parent.func) in (
            "exec", "eval",
        )
        if not runs:
            return _ExecPlace(shared=False)
        call = parent
    given = [*call.args[1:3], *(k.value for k in call.keywords if k.arg in ("globals", "locals"))]
    own = [g for g in given if not (isinstance(g, ast.Constant) and g.value is None)]
    if not own:
        return _ExecPlace(shared=cell.scope_of(call) is None)
    if all(isinstance(g, ast.Call) and reader.dotted(g.func) == "globals" for g in own):
        return _ExecPlace(shared=True)
    return _ExecPlace(shared=False, namespace=_namespace_root(own[-1]))


def _parse_ran(text: str) -> ast.Module | None:
    """`text`, literal code a cell runs, parsed; None when it doesn't parse,
    or is too deep or long for the parser, so it fails when it runs too."""
    try:
        return _parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def _exec_cell(key: _ExecCellKey) -> _Cell | None:
    """The parsed cell of literal code a cell runs, placed by `key`; None
    when the code doesn't parse."""
    tree = _parse_ran(key.text)
    if tree is None:
        return None
    return _Cell(tree, key.text, key.index, host_line=key.host_line, launched=key.launched, host=key.host,
                 shared=key.shared, namespace=key.namespace)


def _unread_exec_at(node: ast.AST, reader: _Reader, cell: _Cell, *, read: _ExecRead) -> ast.AST | None:
    """Where `node`, code of `cell`, runs code through exec(), eval() or
    compile() that the check can't read (`_unread_code`): the call when the
    builtin (`_Reader.exec_builtin`) is handed a source that is neither a
    literal (`_literal_source`) nor a compile() of one, or a literal `read`
    left unread at `_EXEC_LIMIT`; `node` when it loads the builtin as a
    value, as in `run = exec` or `map(exec, parts)`, loads the builtins
    module as a value (`_Reader.loads_builtins_value`), or looks the builtins
    module up by a name the check can't read (`_Reader.builtins_lookup`).
    A literal that doesn't parse runs nothing: it fails when it runs too.
    None anywhere else."""
    if not isinstance(node, (ast.Name, ast.Attribute, ast.Subscript, ast.Call)):
        return None
    if reader.builtins_lookup(node) or reader.loads_builtins_value(node):
        return node
    if isinstance(node, ast.Call) or not isinstance(node.ctx, ast.Load):
        return None
    builtin = reader.exec_builtin(node)
    if builtin is None:
        return None
    call = cell.parents.get(node)
    if not (isinstance(call, ast.Call) and call.func is node):
        return node
    source = _exec_source(call, builtin)
    if source is None or id(call) in read.tried:
        return None
    compiled = isinstance(source, ast.Call) and reader.exec_builtin(source.func) == "compile"
    return None if compiled and builtin != "compile" else call


@dataclass(frozen=True)
class _RunnerCode:
    """What a call to a runner of text (`_STRING_RUNNERS`) hands it to run:
    each argument it takes as code, and whether the call passes more
    through `*` or `**`, which the check can't read."""

    sources: tuple[ast.AST, ...]
    spread: bool


def _runner_code(call: ast.Call, reader: _Reader) -> _RunnerCode | None:
    """The code `call` hands a runner of text, such as `timeit.timeit`'s
    statement and setup; None when it calls none. A runner given nothing
    runs its default, `pass`."""
    if not _runner_named(call.func, reader):
        return None
    params = _STRING_RUNNERS.get(reader.module_member(call.func) or "")
    if params is None:
        return None
    sources = [*call.args[: len(params)], *(k.value for k in call.keywords if k.arg in params)]
    spread = any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords)
    return _RunnerCode(sources=tuple(sources), spread=spread)


def _runner_methods(node: ast.AST, reader: _Reader) -> frozenset[str]:
    """The methods that run code on what `node` holds: those of the runner
    whose call made it (`_INSTANCE_RUNNERS`), there or through the names a
    value or a with-statement binds to it."""
    out: set[str] = set()
    todo = [node]
    seen: set[int] = set()
    while todo:
        node = todo.pop()
        if isinstance(node, ast.Call):
            out |= _INSTANCE_RUNNERS.get(reader.module_member(node.func) or "", frozenset())
        elif isinstance(node, ast.Name) and node.id in reader.bound_names and id(node) not in seen:
            seen.add(id(node))
            todo += [b.value for b in reader.bindings_of(node.id, node=node) if b.how in ("value", "context")]
    return frozenset(out)


def _module_names(path: str) -> set[str]:
    """The names an import may reach a Python file at `path` by: its stem,
    and each folder above it, as a package or a folder on `sys.path`."""
    if not path.endswith(".py"):
        return set()
    *folders, name = path.split("/")
    return {part for part in (*folders, name[: -len(".py")]) if part.isidentifier()}


# Folders an environment installs packages into: what an import reaches in
# one is an installed package, not a project's module.
_PACKAGE_FOLDERS = frozenset({"site-packages", "dist-packages"})
# Calls that write files whose paths the check doesn't read, as an archive's
# extraction does, or a copy of a folder.
_TREE_WRITERS = frozenset({"extractall", "extract", "unpack_archive", "copytree"})
# Calls that may write the file one argument names, by that argument's place,
# beyond those `_written_target` reads.
_FILE_MAKERS = {"os.open": 0, "codecs.open": 0, "urllib.request.urlretrieve": 1}
_FILE_MAKER_NAMES = frozenset(name.rpartition(".")[2] for name in _FILE_MAKERS)
_WRITER_CALL_NAMES = frozenset({*_FILE_WRITE_NAMES, *_FILE_MAKER_NAMES, *_TREE_WRITERS, *_LAUNCH_NAMES})


@dataclass(frozen=True)
class _WrittenModules:
    """The modules the pad's cells may write (`_written_modules`): the names
    an import reaches each by (`names`), and whether a cell writes where the
    check can't place, or unpacks files it can't name (`anywhere`), so any
    module in the project may be one a cell wrote."""

    names: frozenset[str]
    anywhere: bool


def _written_modules(reader: _Reader, cells: Sequence[_Cell]) -> _WrittenModules:
    """The modules the pad's cells write: each Python file a write, copy or
    move of a file reaches (`_written_target`), or a call that writes the
    file one argument names (`_FILE_MAKERS`), by the names an import
    reaches it by (`_module_names`); each a program a cell runs names as a
    word, as `cp job.txt job.py` or `curl -o job.py` does; each folder a
    copy of a folder makes. A write to a path the check can't place, or an
    archive a cell unpacks (`_TREE_WRITERS`), may write any module, and
    each Python file the cell names in a string. Read from `cells`."""
    names: set[str] = set()
    anywhere = False
    for cell in cells:
        unplaced = False
        for call in cell.writer_calls:
            last = _last_name(call.func)
            target = None
            if last in _FILE_WRITE_NAMES:
                target = _written_target(call, reader, cell)
            if target is None and last in _FILE_MAKER_NAMES:
                index = _FILE_MAKERS.get(reader.module_member(call.func) or reader.dotted(call.func))
                target = call.args[index] if index is not None and len(call.args) > index else None
            if target is not None:
                got = reader.paths(target, cell=cell)
                if isinstance(got, _Unknown):
                    unplaced = True
                else:
                    names.update(n for path in got for n in _module_names(path.replace("\\", "/")))
            elif last in _TREE_WRITERS:
                got = reader.paths(call.args[1], cell=cell) if last == "copytree" and len(call.args) > 1 else _UNKNOWN
                if isinstance(got, _Unknown):
                    unplaced = True
                else:
                    names.update(posixpath.basename(p.rstrip("/")) for p in got)
            elif last in _LAUNCH_NAMES and cell.signs.names_a_py_file:
                name = reader.module_member(call.func) or reader.dotted(call.func)
                for command in _shell_commands(call, name, reader) if name in _LAUNCHERS else ():
                    words = command if isinstance(command, list) else [
                        w for c in _script_commands(command) for w in c
                    ]
                    names.update(n for w in words for n in _module_names(_REDIRECT.sub("", w).replace("\\", "/")))
        if unplaced:
            anywhere = True
            names.update(cell.signs.py_files)
    return _WrittenModules(names=frozenset(names), anywhere=anywhere)


class _Modules:
    """The modules an import in the pad may load that the check can't
    read: one a cell writes, one a Python file that is in the project now
    but wasn't when the turn began, and, once a cell changes `sys.path`,
    writes where the check can't place, or in code `python -c` runs, one in
    the project or shared folder. The pad runs `python -P`, so neither
    folder is on `sys.path` by itself. The writes are read the first time a
    check asks, outside any call it reads, so a write in a function counts
    with what each call passes it (`_Reader.call_sites`); the folders the
    first time an import asks, since most cells import only installed
    packages."""

    def __init__(self, reader: _Reader, snapshots: Sequence[ProjectSnapshot]) -> None:
        self._reader = reader
        self._snapshots = snapshots
        self._new: dict[str, bool] = {}

    @cached_property
    def _naming(self) -> list[_Cell]:
        """The cells whose writes may name a module: every cell, once one
        names a Python file, as a name one cell binds may name the file
        another writes; else those that copy or unpack a tree of files."""
        cells = self._reader.cells
        if any(c.signs.names_a_py_file for c in cells):
            return list(cells)
        return [c for c in cells if any(_last_name(call.func) in _TREE_WRITERS for call in c.writer_calls)]

    @cached_property
    def _written(self) -> _WrittenModules:
        """What the cells whose writes may name a module write (`_written_modules`)."""
        return _written_modules(self._reader, self._naming)

    @cached_property
    def _anywhere(self) -> bool:
        """Whether a cell writes where the check can't place, read from the
        rest of the cells only once an import asks about a project module."""
        naming = {id(c) for c in self._naming}
        rest = [c for c in self._reader.cells if id(c) not in naming]
        return self._written.anywhere or _written_modules(self._reader, rest).anywhere

    @property
    def written(self) -> frozenset[str]:
        """The names an import reaches a module a cell writes by."""
        return self._written.names

    def unread(self, module: str, *, cell: _Cell) -> bool:
        """Whether importing `module` in `cell` may run code the check can't
        read. A relative name counts, since the check can't place its
        package. Code `python -c` runs (`_Cell.launched`) has its working
        folder on `sys.path`, as if a cell had put it there."""
        if module.startswith("."):
            return True
        top = module.partition(".")[0]
        if not top:
            return False
        if top in self.written or self._new_on_disk(top):
            return True
        return self._in_folders(top) and (self.path_changed or cell.launched or self._anywhere)

    def any_unread(self, cells: Sequence[_Cell]) -> bool:
        """Whether an import in one of `cells`, by a statement or a literal
        importer (`_Cell.looked_up`), loads a module the check can't read."""
        return any(
            any(self.unread(a.name, cell=cell) for a in node.names) if isinstance(node, ast.Import)
            else not node.level and bool(node.module) and self.unread(node.module, cell=cell)
            for cell in cells for node in cell.signs.imports
        ) or any(self.unread(node.id, cell=cell) for cell in cells for node in cell.looked_up)

    @cached_property
    def path_changed(self) -> bool:
        """Whether any cell changes `sys.path` (`_CodeSigns.path_changes`),
        there or through a name bound to it (`_CodeSigns.list_changes`), as
        the imports and those names read it (`_Reader.module_member`)."""
        reader = self._reader
        aliases = {name for cell in reader.cells for name in cell.signs.path_aliases}
        return any(
            reader.dotted(node) in _SYS_PATH_CHANGES or reader.module_member(node) in _SYS_PATH_CHANGES
            for cell in reader.cells
            for node in (*cell.signs.path_changes, *(n for name, n in cell.signs.list_changes if name in aliases))
        )

    def _new_on_disk(self, top: str) -> bool:
        """Whether the project or shared folder holds `top.py` or
        `top/__init__.py` now, though the walk when the turn began didn't:
        a cell this turn wrote it, however the check read its write."""
        if top not in self._new:
            self._new[top] = any(
                rel not in snap.files and (snap.root / rel).is_file()
                for snap in self._snapshots if snap.complete and not snap.failed
                for rel in (f"{top}.py", f"{top}/__init__.py")
            )
        return self._new[top]

    def _in_folders(self, top: str) -> bool:
        """Whether a Python file in the project or shared folder is reached
        by the name `top`, at any depth, since a cell may put any folder on
        `sys.path` (`ProjectSnapshot.module_names`). Where the walk stopped
        short, the folder's own top level is read on disk."""
        if any(top in snap.module_names for snap in self._snapshots):
            return True
        return any(
            (snap.root / f"{top}.py").is_file() or (snap.root / top / "__init__.py").is_file()
            for snap in self._snapshots if snap.failed or not snap.complete
        )


def _member_name(node: ast.AST) -> str:
    """The last name `node` is spelled with (`_last_name`), or the literal
    key of an entry, as "run_path" for `vars(runpy)["run_path"]`."""
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return _last_name(node)


def _runner_named(node: ast.AST, reader: _Reader) -> bool:
    """Whether `node` is spelled as something that may run code: by the last
    name of a runner or importer (`_RUNNER_NAMES`), a PyRun_ function, or a
    name that may hold one by another name (`_Reader.runner_aliases`); or an
    entry read under a name the check resolves, as `vars(runpy)[k]` is.
    `_Reader.module_member` says which of them is one."""
    last = _member_name(node)
    if last in _RUNNER_NAMES or last.startswith(_PYRUN_PREFIX) or last in reader.runner_aliases:
        return True
    return (isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Constant)
            and reader.module_member(node) is not None)


def _looks_up_a_runner(call: ast.Call, reader: _Reader) -> bool:
    """Whether `call` looks a member up by a name the check can't read on a
    loader, or on a module that holds something that runs code:
    `getattr(runpy, name)` may be `runpy.run_path`. A runner module's
    members mostly run code, so any name there holds; a launcher module
    (`os`, `subprocess`) holds only when the key names a launcher member
    (`_RUNNERS_BY_MODULE`), read with `_literal_texts`, or can't be read."""
    if not (isinstance(call.func, ast.Name) and call.func.id == "getattr" and len(call.args) >= 2):
        return False
    key = call.args[1]
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return False
    held = call.args[0]
    if _last_name(held) == "loader":
        return True
    module = reader.module_member(held)
    if module in _RUNNER_MODULES:
        return True
    if module in _LAUNCHER_MODULES:
        texts = _literal_texts(key, reader)
        return texts is None or any(text in _RUNNERS_BY_MODULE.get(module, {}) for text in texts)
    return False


def _imports_unread(call: ast.Call, reader: _Reader, cell: _Cell, *, modules: _Modules) -> bool:
    """Whether `call`, an importer's, imports a module `modules` can't read,
    or one by a name the check can't read (`_literal_texts`). A
    `pkgutil.resolve_name` or `pydoc.locate` target names the module before
    its colon."""
    target = _target_arg(call, keywords=("name", "path"))
    if target is None:
        return False
    texts = _literal_texts(target, reader)
    return texts is None or any(modules.unread(text.partition(":")[0], cell=cell) for text in texts)


def _runs_unread(call: ast.Call, reader: _Reader, cell: _Cell, *, read: _ExecRead, modules: _Modules) -> bool:
    """Whether `call`, code of `cell`, runs code in this process that the
    check can't read (`_unread_run_at`). A runner of text handed a lambda
    or a function the check reads runs read code."""
    func = call.func
    last = _member_name(func)
    if isinstance(func, ast.Attribute) and last in _LOADER_METHODS:
        return True
    if isinstance(func, ast.Call) and _looks_up_a_runner(func, reader):
        return True
    name = reader.module_member(func) if _runner_named(func, reader) else None
    if name in _CODE_RUNNERS or name is not None and name.startswith(_PYRUN_FUNCS):
        return True
    if name in _IMPORTERS:
        return _imports_unread(call, reader, cell, modules=modules)
    if name in _STRING_RUNNERS:
        code = _runner_code(call, reader)
        if code is None:
            return False
        literal = [s for s in code.sources if _literal_source(s) is not None]
        return code.spread or any(
            _literal_source(s) is None and not _names_read_code(s, reader) for s in code.sources
        ) or bool(literal) and id(call) not in read.tried
    return isinstance(func, ast.Attribute) and last in _RUNNER_METHODS and last in _runner_methods(func.value, reader)


def _names_read_code(node: ast.AST, reader: _Reader) -> bool:
    """Whether `node` is a lambda, or names a function or lambda a cell
    defines that no other value of the name can stand for: code the check
    reads where it is written or passed on."""
    if isinstance(node, ast.Lambda):
        return True
    if not isinstance(node, ast.Name) or node.id not in reader.function_names:
        return False
    return all(b.how == "value" and isinstance(b.value, ast.Lambda) for b in reader.bindings_of(node.id, node=node))


def _launches_unread(call: ast.Call, reader: _Reader, *, read: _ExecRead) -> bool:
    """Whether `call` launches Python that runs code the check can't read
    (`_python_launch`), or hands `python -c` literal code `read` left unread
    at `_EXEC_LIMIT`."""
    launch = _python_launch(call, reader)
    return launch is not None and (launch.unread or bool(launch.sources) and id(call) not in read.tried)


def _unread_run_at(node: ast.AST, reader: _Reader, cell: _Cell, *, read: _ExecRead, modules: _Modules) -> bool:
    """Whether `node`, code of `cell`, runs code in this process, or in a
    Python process it launches, that the check can't read (`_unread_code`):
    a call that runs a script, a module, a notebook, a code object, or text
    a console, debugger or test runner gets (`_CODE_RUNNERS`,
    `_PYRUN_FUNCS`, `_INSTANCE_RUNNERS`, and any `.exec_module()` or
    `.load_module()`); a runner of text (`_STRING_RUNNERS`) handed code that
    isn't a literal, or a literal `read` left unread at `_EXEC_LIMIT`; an
    import by a name that isn't a literal; an import of a module `modules`
    can't read, whether by an import statement or a literal
    `importlib.import_module` or `__import__` (`_Cell.looked_up`); and a
    program it runs that runs Python the check can't read
    (`_launches_unread`). Each name a runner, importer or launcher is called
    by resolves through the imports (`_Reader.module_member`)."""
    if isinstance(node, ast.Import):
        return any(modules.unread(a.name, cell=cell) for a in node.names)
    if isinstance(node, ast.ImportFrom):
        # A relative import fails in the pad's cells: they are no package's.
        return not node.level and bool(node.module) and modules.unread(node.module, cell=cell)
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        if node in cell.looked_up and modules.unread(node.id, cell=cell):
            return True
        # A runner or importer passed on or stored, as `map(runpy.run_path, paths)`.
        call = cell.parents.get(node)
        return (isinstance(node.ctx, ast.Load) and not (isinstance(call, ast.Call) and call.func is node)
                and _runner_named(node, reader) and reader.module_member(node) in _RUNNER_VALUES)
    if not isinstance(node, ast.Call):
        return False
    if _looks_up_a_runner(node, reader) or (
        _last_name(node.func) in _FILE_WRITE_NAMES and _writes_a_pth_file(node, reader, cell)
    ):
        return True
    return _runs_unread(node, reader, cell, read=read, modules=modules) or _launches_unread(node, reader, read=read)


def _writes_a_pth_file(call: ast.Call, reader: _Reader, cell: _Cell) -> bool:
    """Whether `call`, a write of a file, may write a `.pth` file, whose
    import lines `site` runs as a Python starts, in a cell that names one."""
    if not cell.signs.names_a_pth_file:
        return False
    target = _written_target(call, reader, cell)
    got = reader.paths(target, cell=cell) if target is not None else _UNKNOWN
    return isinstance(got, _Unknown) or any(".pth" in path for path in got)


def _names_a_runner(node: ast.AST, reader: _Reader) -> bool:
    """Whether `node`, one of a cell's hints (`_CodeSigns.hints`), names
    something that may run code the check can't read: the builtin exec,
    eval or compile (`_Reader.exec_builtin`), the builtins module, the
    cell's own namespace (`globals`, `vars`, `locals`), a loader's method, a
    runner, importer or launcher, or the C API's PyRun_ functions as the
    imports name them (`_Reader.module_member`), or a runner's method on
    what such a runner made (`_runner_methods`). Not `re.compile`,
    `cursor.execute` or `pipeline.run`."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        # An import of the builtins module, which may bind exec to any name.
        return True
    if isinstance(node, ast.Subscript):
        key = node.slice.value if isinstance(node.slice, ast.Constant) else None
        return key in _BUILTINS_MODULE_NAMES or reader.exec_builtin(node) is not None
    last = _last_name(node)
    if last in _LOADER_METHODS or last in _BUILTINS_MODULE_NAMES or reader.dotted(node) in _GLOBALS_CALLS:
        return True
    if last in _EXEC_CALLS:
        return reader.exec_builtin(node) is not None
    name = reader.module_member(node)
    if name is not None and (name in _RUN_CALLS or name.startswith(_PYRUN_FUNCS)):
        return True
    return isinstance(node, ast.Attribute) and last in _RUNNER_METHODS and last in _runner_methods(node.value, reader)


def _runs_code(cell: _Cell, reader: _Reader) -> bool:
    """Whether a name in `cell` names something that may run code the check
    can't read (`_names_a_runner`), so a call that reaches what such code
    defined may hold the cell, and a literal exec in it is read for what it
    defines."""
    return any(_names_a_runner(node, reader) for node in cell.signs.hints)


def _unread_code(
    reader: _Reader, root: ast.AST, cell: _Cell, *, read: _ExecRead, modules: _Modules, spelled_only: bool = False,
) -> list[ast.AST]:
    """The places in `root`, code of `cell`, that run code the check can't
    read, read in one pass: through exec(), eval() or compile()
    (`_unread_exec_at`), or in this process or a Python process they launch
    (`_unread_run_at`). Code `read` read as literal code counts as read.
    Where a whole cell names none of those builtins itself
    (`_CodeSigns.names_builtins`), only an import, a call or what it calls,
    a hint or a name that may hold a runner by another name
    (`_Reader.runner_aliases`) can run such code, and the cell lists each
    already (`_CodeSigns.imports`, `_Cell.calls`, `_CodeSigns.hints`,
    `_Cell.loads`), so the pass reads those alone. With `spelled_only`, of
    the calls it reads only those spelled as one that may run such code
    (`_Cell.unread_candidates`) or called by such a name."""
    nodes: Iterable[ast.AST] = ast.walk(root)
    if root is cell.tree and not cell.signs.names_builtins:
        aliases = [load for name in reader.runner_aliases for load in cell.loads.get(name, ())]
        if spelled_only:
            called = [cell.parents[load] for load in aliases if isinstance(cell.parents.get(load), ast.Call)]
            calls: Sequence[ast.AST] = (*cell.unread_candidates, *called)
        else:
            calls = (*cell.calls, *(c.func for c in cell.calls))
        nodes = (*cell.signs.imports, *cell.looked_up, *calls, *cell.signs.hints, *aliases)
    out: list[ast.AST] = []
    for node in nodes:
        site = _unread_exec_at(node, reader, cell, read=read)
        if site is None and _unread_run_at(node, reader, cell, read=read, modules=modules):
            site = node
        if site is not None:
            out.append(site)
    return out


@dataclass(frozen=True)
class _StarImport:
    """The first star import of a module the check can read that read code
    ran, as `from math import *`: the cell it is in, and its line there."""

    cell: _Cell
    line: int


@dataclass(frozen=True)
class _Defined:
    """What code the check couldn't read may have defined before this cell.

    `tainted` names hold a namespace, module or value such code filled, so a
    call through one runs code the check never read. `namespaces` are the
    names a literal exec filled, each with the functions it defined there, so
    a call through one resolves to the function it holds. `values` are the
    names an assignment binds to a function a literal exec defined
    (`go = ns["go"]`) or to a lambda a literal eval made
    (`push = eval("lambda: ...")`), so a call to one reads that function.
    `bound` names a statement gives beyond what `_Reader.binds` knows
    (`_CodeSigns.bound_aside`), and `star` is the first star import read code
    ran, which may bind any name after it.
    """

    tainted: frozenset[str]
    namespaces: dict[str, dict[str, Function]]
    values: dict[str, Function]
    bound: frozenset[str]
    star: _StarImport | None


@dataclass(frozen=True)
class _DefinedUses:
    """What code reaches of what code the check couldn't read may have
    defined (`_defined_uses`): the places that call or run such a name or
    value, which hold the cell when such code ran (`sites`); and each call
    through a namespace a literal exec in an earlier cell filled, read at
    that call so its request shows on the card (`callees`)."""

    sites: tuple[ast.AST, ...] = ()
    callees: tuple[_Callee, ...] = ()


def _star_binds(reader: _Reader, node: ast.Name, star: _StarImport | None) -> bool:
    """Whether a star import ran before the code `node` is in reads it: in a
    cell that runs before it, or earlier in its own cell, or anywhere in it
    for a read inside a function, which runs when called."""
    if star is None:
        return False
    cell = reader.cell_of(node)
    if star.cell is not cell:
        return (star.cell.host, star.cell.index) < (cell.host, cell.index)
    return cell.scope_of(node) is not None or star.line < node.lineno


def _unvouched(reader: _Reader, node: ast.Name, defined: _Defined, tainted: frozenset[str] | set[str]) -> bool:
    """Whether the name `node` reads may come from code the check can't read:
    a `tainted` name, or one no read code gives a value (a builtin or a name
    the pad binds, a name `defined.bound` holds, or what `_Reader.binds`
    covers), unless a star import of a readable module ran before it."""
    if node.id in tainted:
        return True
    return not (node.id in _BUILTIN_NAMES or node.id in defined.bound or reader.binds(node, surely=True)
                or node in reader.cell_of(node).looked_up or _star_binds(reader, node, defined.star))


def _root_of(node: ast.AST) -> ast.AST:
    """What a chain of members and entries starts from: `ns` for
    `ns["f"].run`, the call for `runpy.run_path("job.py")["main"]`."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node


@dataclass(frozen=True)
class _Unread:
    """What `_from_unread` reads a value against: the cell it is in, the
    modules an import may load unread, what such code may have defined, and
    the names known to hold what it filled so far."""

    reader: _Reader
    cell: _Cell
    modules: _Modules
    defined: _Defined
    tainted: frozenset[str] | set[str]


def _from_unread(value: ast.AST, at: _Unread) -> bool:
    """Whether `value` holds what code the check can't read made: a tainted
    name or one no read code binds (`_unvouched`), a member or entry of one,
    or what calling one returns; a module an import loads unread; what
    exec(), eval() or compile() makes of a source that isn't a literal; the
    namespace a runner returns (`_NAMESPACE_RUNNERS`); a module an importer
    loads unread or by a name the check can't read; and a tuple, list,
    branch or `or` with one of those in it."""
    reader = at.reader
    while isinstance(value, (ast.NamedExpr, ast.Await, ast.Starred)):
        value = value.value
    if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
        return any(_from_unread(e, at) for e in value.elts)
    if isinstance(value, ast.IfExp):
        return _from_unread(value.body, at) or _from_unread(value.orelse, at)
    if isinstance(value, ast.BoolOp):
        return any(_from_unread(v, at) for v in value.values)
    if isinstance(value, (ast.Attribute, ast.Subscript)):
        root = _root_of(value)
        return isinstance(root, (ast.Name, ast.Call)) and _from_unread(root, at)
    if isinstance(value, ast.Name):
        if value in at.cell.looked_up:
            # A literal import, as `importlib.import_module("json")` reads.
            return at.modules.unread(value.id, cell=at.cell)
        return _unvouched(reader, value, at.defined, at.tainted)
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    builtin = reader.exec_builtin(func)
    if builtin is not None:
        source = _exec_source(value, builtin)
        if source is None or _literal_source(source) is not None:
            return False
        compiled = isinstance(source, ast.Call) and reader.exec_builtin(source.func) == "compile"
        return _from_unread(source, at) if compiled else True
    if isinstance(func, ast.Name) and func.id == "getattr" and value.args:
        return _from_unread(value.args[0], at)
    name = reader.module_member(func) if _runner_named(func, reader) else None
    if name in _NAMESPACE_RUNNERS:
        return True
    if name in _IMPORTERS:
        return _imports_unread(value, reader, at.cell, modules=at.modules)
    root = _root_of(func)
    return isinstance(root, (ast.Name, ast.Call)) and _from_unread(root, at)


class _Store(NamedTuple):
    """The names one binding gives, and the value it gives them."""

    names: list[str]
    value: ast.AST


def _stores(node: ast.AST) -> Iterator[_Store]:
    """The names `node` binds, each with the value it gives them: an
    assignment's (a tuple assigned a tuple of the same length pairs them up,
    any other unpacking gives every name the whole value), a walrus's, and
    the defaults of a function's or lambda's parameters."""
    if isinstance(node, ast.Assign) or isinstance(node, ast.AnnAssign) and node.value is not None:
        for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
            value = node.value
            if (isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List))
                    and len(target.elts) == len(value.elts)
                    and not any(isinstance(e, ast.Starred) for e in (*target.elts, *value.elts))):
                for t, v in zip(target.elts, value.elts):
                    if isinstance(t, ast.Name):
                        yield _Store([t.id], v)
            elif isinstance(target, ast.Name):
                yield _Store([target.id], value)
            elif isinstance(target, (ast.Tuple, ast.List)):
                yield _Store([n.id for e in target.elts for n in [e.value if isinstance(e, ast.Starred) else e]
                              if isinstance(n, ast.Name)], value)
    elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
        yield _Store([node.target.id], node.value)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = node.args
        positional = [*args.posonlyargs, *args.args]
        for param, default in zip(positional[len(positional) - len(args.defaults):], args.defaults):
            yield _Store([param.arg], default)
        for param, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None:
                yield _Store([param.arg], default)


def _defined_uses(
    reader: _Reader, root: ast.AST, cell: _Cell, defined: _Defined | None, modules: _Modules,
) -> _DefinedUses:
    """What `root`, code of `cell`, reaches of what code the check couldn't
    read may have defined, in one walk. A site is a call to a name such code
    may have defined (`_unvouched`), a call on one as a receiver or key, a
    call through the cell's own namespace dict (`globals()["x"]()`), such a
    name or a member of one handed to a call that may run it, and one used
    as a decorator, a base class or a class keyword, whose code runs as the
    class is made. A name `root` binds, in any scope, to what such code made
    (`_from_unread`), as `f = push` does, counts as such a name. A callee
    is a call through a namespace a literal exec in an earlier cell filled
    (`ns["f"]()`, `m.f()`). Nothing when no code the check can't read may
    run (`defined` is None)."""
    if defined is None:
        return _DefinedUses()
    nodes = list(ast.walk(root))
    tainted = set(defined.tainted)
    at = _Unread(reader=reader, cell=cell, modules=modules, defined=defined, tainted=tainted)
    for node in nodes:
        for names, value in _stores(node):
            if names and _from_unread(value, at):
                tainted.update(names)
    sites: list[ast.AST] = []
    callees: list[_Callee] = []

    def unvouched(node: ast.AST) -> bool:
        head = _root_of(node)
        return isinstance(head, ast.Name) and _unvouched(reader, head, defined, tainted)

    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            made = [dec.func if isinstance(dec, ast.Call) else dec for dec in node.decorator_list]
            if isinstance(node, ast.ClassDef):
                made += [*node.bases, *(k.value for k in node.keywords)]
            sites += [m for m in made if unvouched(m)]
            continue
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        while isinstance(func, ast.NamedExpr):
            func = func.value
        if isinstance(func, ast.Subscript) and _cell_namespace(func.value, reader):
            sites.append(node)
            continue
        if isinstance(func, ast.Name) and func.id in defined.values:
            # A name bound to a function a literal exec or eval made.
            callees.append(_Callee(func=defined.values[func.id], call=node, use=func))
        holder = func.value if isinstance(func, (ast.Attribute, ast.Subscript)) else None
        member = (func.attr if isinstance(func, ast.Attribute)
                  else func.slice.value if isinstance(func, ast.Subscript) and isinstance(func.slice, ast.Constant)
                  else None)
        filled = defined.namespaces.get(holder.id, {}) if isinstance(holder, ast.Name) else {}
        if isinstance(member, str) and member in filled:
            callees.append(_Callee(func=filled[member], call=node, use=func))
        if unvouched(func):
            sites.append(node)
        elif not _is_harmless_call(node, reader):
            for arg in (*node.args, *(k.value for k in node.keywords)):
                named = arg.value if isinstance(arg, ast.Starred) else arg
                if isinstance(named, (ast.Name, ast.Attribute, ast.Subscript)) and unvouched(named):
                    sites.append(named)
    return _DefinedUses(sites=tuple(sites), callees=tuple(callees))


def _namespace_root(node: ast.AST | None) -> str | None:
    """The name of the object a namespace argument names: `ns` for `ns`, `mod`
    for `mod.__dict__`; None for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and node.attr == "__dict__" and isinstance(node.value, ast.Name):
        return node.value.id
    return None


def _filled_names(node: ast.AST, at: _Unread) -> set[str]:
    """The names `node`, code that runs in a cell, fills with what code the
    check can't read made: those an import of a module `_Modules` can't
    read binds, those an assignment or walrus binds to such a value
    (`_from_unread`), and the namespace a non-literal exec() or eval() is
    handed."""
    if isinstance(node, ast.Import):
        return {a.asname or a.name.partition(".")[0] for a in node.names if at.modules.unread(a.name, cell=at.cell)}
    if isinstance(node, ast.ImportFrom):
        if node.level or not node.module or not at.modules.unread(node.module, cell=at.cell):
            return set()
        return {a.asname or a.name for a in node.names if a.name != "*"}
    if isinstance(node, ast.Call):
        if at.reader.exec_builtin(node.func) not in ("exec", "eval"):
            return set()
        source = _exec_source(node, "exec")
        namespace = node.args[1] if len(node.args) > 1 else _keyword(node, "globals")
        root = _namespace_root(namespace)
        return {root} if source is not None and _literal_source(source) is None and root is not None else set()
    return {name for names, value in _stores(node) if _from_unread(value, at) for name in names}


def _build_defined(
    reader: _Reader, cells: Sequence[_Cell], modules: _Modules, literal: _ExecRead, *, errored: frozenset[int],
    taint_from: _Cell | None,
) -> _Defined:
    """What code the check couldn't read may have defined before this cell,
    read from the code of each of `cells` that runs (`_Cell.fills_that_run`),
    and from what a function sets a name it declares `global` to: the names
    unread code tainted (`_filled_names`), read in order until no more
    turn up, so an alias of a tainted name is tainted too; the namespaces a
    literal exec in `literal` filled, with what it defined in each; the
    names read code binds beyond `_Reader.binds`; and the first star import
    of a readable module that ran. An errored cell may have stopped before
    its star import. Names are tainted from `taint_from` on, the first cell
    where such code ran (`_first_unread`): before it a name no read code
    bound held nothing yet. With no such cell, nothing is tainted."""
    namespaces: dict[str, dict[str, Function]] = {}
    for executed in literal.cells:
        if executed.namespace is not None and executed.host < reader.current.index:
            namespaces.setdefault(executed.namespace, {}).update(executed.functions)
    values = _literal_exec_values(reader, cells, literal, namespaces, errored=errored)
    star: _StarImport | None = None
    for cell in sorted((c for c in cells if c.shared and c.host not in errored), key=lambda c: (c.host, c.index)):
        lines = [n.lineno for n in cell.signs.imports if isinstance(n, ast.ImportFrom) and n.module
                 and any(a.name == "*" for a in n.names) and not modules.unread(n.module, cell=cell)]
        if lines:
            star = _StarImport(cell=cell, line=min(lines))
            break
    base = _Defined(
        tainted=frozenset(), namespaces=namespaces, values=values, star=star,
        bound=frozenset(name for cell in cells if cell.host not in errored for name in cell.signs.bound_aside),
    )
    tainted: set[str] = set()
    start = None if taint_from is None else (taint_from.host, taint_from.index)
    tainting = [c for c in cells if start is not None and (c.host, c.index) >= start]
    grew = bool(tainting)
    while grew:
        grew = False
        for cell in tainting:
            at = _Unread(reader=reader, cell=cell, modules=modules, defined=base, tainted=tainted)
            new = {name for node in cell.fills_that_run for name in _filled_names(node, at)}
            new |= {name for name, value in _global_fills(cell) if _from_unread(value, at)}
            if new - tainted:
                tainted |= new
                grew = True
    return replace(base, tainted=frozenset(tainted))


class _NameValue(NamedTuple):
    """A name and a value bound to it."""

    name: str
    value: ast.AST


def _literal_exec_values(
    reader: _Reader, cells: Sequence[_Cell], literal: _ExecRead, namespaces: dict[str, dict[str, Function]],
    *, errored: frozenset[int],
) -> dict[str, Function]:
    """The names an earlier cell binds to a function or lambda a literal exec
    or eval made, so a call to one reads it: `go = ns["go"]` for a function a
    literal `exec(src, ns)` defined (`namespaces`), and `push = eval("lambda:
    ...")` for the lambda a literal eval makes. The source was read, so this
    carries its findings to the call rather than holding the cell."""
    # A literal eval of one lambda, by the cell and line of its call.
    lambdas: dict[tuple[int, int], ast.Lambda] = {}
    for executed in literal.cells:
        if executed.host_line is None or executed.host >= reader.current.index:
            continue
        body = executed.tree.body
        if len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Lambda):
            lambdas[(executed.host, executed.host_line)] = body[0].value
    values: dict[str, Function] = {}
    for cell in cells:
        if not cell.shared or cell.host in errored or cell.host >= reader.current.index:
            continue
        for node in cell.fills_that_run:
            for names, value in _stores(node):
                func = _namespace_function(value, namespaces) or (
                    lambdas.get((cell.host, value.lineno)) if _is_literal_eval(value, reader) else None
                )
                if func is not None:
                    for name in names:
                        values[name] = func
    return values


def _namespace_function(value: ast.AST, namespaces: dict[str, dict[str, Function]]) -> Function | None:
    """The function a literal exec defined that `value`, an entry of a
    namespace it filled (`ns["go"]`), reads; None otherwise."""
    if (isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name)
            and isinstance(value.slice, ast.Constant) and isinstance(value.slice.value, str)):
        return namespaces.get(value.value.id, {}).get(value.slice.value)
    return None


def _is_literal_eval(value: ast.AST, reader: _Reader) -> bool:
    """Whether `value` is a call to the builtin eval on a literal source."""
    return (isinstance(value, ast.Call) and reader.exec_builtin(value.func) == "eval"
            and (source := _exec_source(value, "eval")) is not None and _literal_source(source) is not None)


def _global_fills(cell: _Cell) -> list[_NameValue]:
    """Each name a function `cell` defines declares `global` and sets, with
    the value it sets it to: it fills the cell namespace when called."""
    out: list[_NameValue] = []
    for declared in cell.signs.globals_declared:
        scope = cell.scope_of(declared)
        out += [_NameValue(name, b.value) for name in declared.names for b in cell.bindings.get(name, ())
                if b.scope is scope and b.how == "value"]
    return out


def _poisons_namespace(node: ast.AST, reader: _Reader) -> bool:
    """Whether `node` fills the cell's own namespace with code the check
    can't trace: `globals()[k] = v` or `vars()[k] = v` under a computed key,
    `globals().update(...)` or `vars().update(...)`, or exec() or eval()
    handed `globals()`. `vars(obj)` is the object's own, not the cell's."""
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return any(
            isinstance(t, ast.Subscript) and _cell_namespace(t.value, reader) and not isinstance(t.slice, ast.Constant)
            for t in targets
        )
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "update" and _cell_namespace(func.value, reader):
            return True
        if reader.exec_builtin(func) in ("exec", "eval"):
            args = (*node.args[1:], *(k.value for k in node.keywords if k.arg in ("globals", "locals")))
            return any(_cell_namespace(a, reader) for a in args)
    return False


def _cell_namespace(node: ast.AST, reader: _Reader) -> bool:
    """Whether `node` hands out the cell's own namespace: `globals()`,
    `locals()`, or `vars()` with no argument."""
    return isinstance(node, ast.Call) and reader.dotted(node.func) in _GLOBALS_CALLS and not node.args


def _first_unread(reader: _Reader, cells: Sequence[_Cell], modules: _Modules, read: _ExecRead) -> _Cell | None:
    """The first of `cells`, in the order they run, where code the check
    couldn't read ran before the checked cell.

    A cell counts when `_unread_code` finds such a site anywhere in it,
    reading the calls spelled as one that may run such code
    (`_Cell.unread_candidates`), a function the cell defines among them. A
    cell counts too when it fills the cell's own namespace with what the
    check can't trace (`_poisons_namespace`), if it names that namespace or
    exec() or eval() at all. None when no cell does. The checked cell's own
    unread code holds it already, so this looks only at earlier cells."""
    for cell in sorted(cells, key=lambda c: (c.host, c.index)):
        if reader.earlier(cell.tree) and _unread_code(reader, cell.tree, cell, read=read, modules=modules,
                                                      spelled_only=True):
            return cell
        signs = cell.signs
        if (signs.names_builtins or any(_last_name(h) in _GLOBALS_CALLS for h in signs.hints)) and any(
            _poisons_namespace(node, reader) for node in ast.walk(cell.tree)
        ):
            return cell
    return None


def _earlier_function_uses(
    reader: _Reader, root: ast.AST, cell: _Cell, *, ran_only: bool = False,
) -> Iterator[_Callee]:
    """Each place in `root` that calls a function or lambda an earlier cell
    defined, or passes one on as a value: by its name anywhere, or through
    whatever else `_Reader.callees` reads where it is called or passed to a
    call. With `ran_only`, only the calls in code that runs when `root` runs
    (`_nodes_that_run`): a function passed on as a value may never run."""
    for node in cell.nodes_run_by(root) if ran_only else ast.walk(root):
        if not (isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)) and isinstance(node.ctx, ast.Load)):
            continue
        parent = cell.parents.get(node)
        call = parent if isinstance(parent, ast.Call) and parent.func is node else None
        if ran_only and call is None:
            continue
        if isinstance(node, ast.Name) and node.id in reader.function_names:
            func = reader.earlier_function(node)
            if func is not None:
                # And the one before it, where a cell that stopped early defined it.
                for f in (func, *reader.replaced.get(id(func), ())):
                    if not isinstance(f, ast.ClassDef):
                        yield _Callee(func=f, call=call, use=node)
                continue
        if call is not None or isinstance(parent, ast.Call) and node in parent.args:
            mark = reader.cuts
            found = reader.callees(node=node, call=call, use=node)
            if reader.cuts > mark:
                # A function the depth limit stopped short of, beside any it reached.
                ids = {id(c.func) for c in found}
                reader.read_past(mark, lambda: any(
                    id(c.func) not in ids for c in reader.callees(node=node, call=call, use=node)
                ))
            yield from found


def _added_holds(*, found: list[FileChange], maybe: list[FileChange]) -> list[FileChange]:
    """The held changes in `maybe`, the checked cell read again with each
    errored earlier cell as maybe stopped anywhere, that `found`, its first
    read, doesn't hold: each with only the files `found` doesn't hold at
    that call. A target the code picks when it runs, with no path, counts
    as one more file unless `found` holds one at that call too."""

    def unplaced(change: FileChange) -> bool:
        return not (change.paths or change.whole_project or change.whole_shared)

    held: dict[tuple[str, int, str], list[FileChange]] = {}
    for change in found:
        if change.held:
            held.setdefault((change.action, change.line, change.call), []).append(change)
    out: list[FileChange] = []
    for change in maybe:
        if not change.held:
            continue
        same = held.get((change.action, change.line, change.call))
        if same is None or unplaced(change) and not any(unplaced(s) for s in same):
            out.append(change)
            continue
        paths = tuple(p for p in change.paths if all(p not in s.paths for s in same))
        project = change.whole_project and not any(s.whole_project for s in same)
        shared = change.whole_shared and not any(s.whole_shared for s in same)
        if paths or project or shared:
            out.append(change.model_copy(update={"paths": paths, "whole_project": project, "whole_shared": shared}))
    return out


def find_cell_writes(
    *,
    code: str,
    earlier_code: Sequence[str],
    snapshots: Sequence[ProjectSnapshot],
    http_rules: Sequence[HttpWriteRule] = (),
    earlier_errored: Sequence[bool] = (),
) -> CellWrites:
    """What `code` would change, read against the conversation's earlier cells.

    A function or lambda an earlier cell defined is read again at each place
    this cell calls it or passes it on, with its parameters read from that
    call, and what it would change is reported at that place. A request that
    matches one of `http_rules` counts as a write, whatever its method.
    `earlier_errored`, beside `earlier_code`, marks each earlier cell that
    stopped on an error: its writes make no file the agent's own
    (`_OwnedFiles`), and a path read through a name it binds may hold any
    value it gives the name, or the one before it: each file those name is
    held too (`_added_holds`), up to `_PATH_LIMIT` paths for one value. A
    cell with no entry ran to its end.

    Literal code the cell, or an earlier function it calls, hands exec(),
    a runner of text or `python -c` is read as one more cell, and so is a
    literal exec in an earlier cell, so a call to what it defined is read
    where this cell makes it (`_exec_cells`). The result is `unchecked` when
    the cell runs code the check can't read (`_unread_code`); when it calls
    a name, namespace or module that such code may have filled and such
    code ran in this cell or an earlier one (`_defined_uses`,
    `_first_unread`); or when a depth limit stopped a read where a write may
    lie past it, a function past `_CALL_DEPTH` calls among them
    (`_Reader.stopped`). Whatever else it read stays on the card.

    A cell that doesn't parse returns nothing to hold: the runner compiles it
    the same way and refuses it before any of it runs.
    """
    try:
        tree = _parse(code)
    except (SyntaxError, ValueError):
        return CellWrites()
    cells: list[_Cell] = []
    errored: set[int] = set()
    for i, text in enumerate(earlier_code):
        earlier = _EARLIER_CELLS.get(code=text, index=len(cells))
        if earlier is not None:
            if i < len(earlier_errored) and earlier_errored[i]:
                errored.add(earlier.index)
            cells.append(earlier)
    current = _Cell(tree, code, len(cells))
    cells.append(current)
    project = next((s for s in snapshots if s.label == "project"), None)
    project_root = project.root if project is not None else None
    reader = _Reader(cells, current=current, project_root=project_root, stopped_early=frozenset(errored))
    # A literal exec in an earlier cell is read as more cells, placed right
    # after the cell that ran it, so a later call to what it defined is read.
    # Then the literal code the cell runs through exec() and its kin is read
    # the same way, a function such earlier code defined among those it may
    # call. The reader then reads them with the rest.
    earlier_cells = cells[: current.index]
    with_code = [c for c in earlier_cells if _runs_code(c, reader)]
    # Only a cell that hands a runner literal code at its top level, or calls
    # what an earlier one defined, gives that read anything.
    follows = any(c.function_execs or c.function_launches for c in earlier_cells)
    with_literal = [c for c in with_code if follows or c.hands_literal_code
                    or any(_last_name(call.func) in reader.runner_aliases for call in c.calls)]
    earlier_exec = (
        _exec_cells(reader, with_literal, index=len(cells), ran_only=True) if with_literal else _ExecRead.empty()
    )

    def with_cells(old: _Reader, read: _ExecRead) -> _Reader:
        """A reader of `cells` with what `read` read added, stopped where
        `old` stopped; `old` itself when `read` read nothing."""
        if not read.cells:
            return old
        cells.extend(read.cells)
        new = _Reader(cells, current=current, project_root=project_root, stopped_early=frozenset(errored))
        new.stopped = old.stopped
        return new

    reader = with_cells(reader, earlier_exec)
    exec_read = _exec_cells(reader, [current], index=len(cells))
    reader = with_cells(reader, exec_read)
    literal_code = _ExecRead.joined([earlier_exec, exec_read])
    ran = exec_read.of(current.tree)
    if any(not s.complete for s in snapshots):
        reader.made = _made(reader=reader)
    owned = _OwnedFiles(earlier_cells, snapshots, errored=frozenset(errored))
    modules = _Modules(reader, snapshots)
    # When the conversation may run code the check can't read, what such code
    # could have defined, so a call that reaches it holds the cell; None when
    # it can't.
    conversation = [*earlier_cells, current]
    may_run_unread = bool(with_code) or _runs_code(current, reader) or modules.any_unread(conversation)
    # Whether such code ran in this cell or an earlier one: only then does a
    # call that reaches what it may have defined hold the cell, and only then
    # is what it tainted read.
    everything = [*conversation, *literal_code.cells]
    first_unread = _first_unread(reader, everything, modules, literal_code) if may_run_unread else None
    defined = (
        _build_defined(reader, everything, modules, literal_code, errored=frozenset(errored), taint_from=first_unread)
        if may_run_unread else None
    )
    sql: list[SqlWrite] = []
    calls: list[CallWrite] = []
    files: list[FileChange] = []
    http: list[HttpWrite] = []
    # Where the cell runs code the check can't read (`_unread_code`).
    unread: list[ast.AST] = []
    # Where the cell reaches what code the check couldn't read may have
    # defined, held only when such code ran (`_first_unread`).
    reaches_defined: list[ast.AST] = []

    # A function past `_CALL_DEPTH` calls that the reading of errored cells as
    # maybe stopped anywhere finds a held file change in.
    past_cap: list[ast.AST] = []

    def read_cells(reader: _Reader, *, found: list[FileChange], files_only: bool) -> None:
        """Reads the checked cell, and each earlier function it calls, with
        `reader`: its file changes into `found`, the rest too unless
        `files_only`."""
        # The shortest chain each function was read at for one set of literal
        # arguments, so a self-recursive function is read once per set, not
        # once per call that reaches it (`_Visit`).
        read_at: dict[_Visit, int] = {}

        def read_function(
            callee: _Callee, *, report: _Report, chain: tuple[_Visit, ...], guessed: bool = False,
        ) -> None:
            """Reads `callee` at its call, then each earlier function it calls
            in turn. A function the chain already reads is read again only
            for a call that passes other literal arguments (`_Visit`), and
            one read at a chain no longer than before isn't read again."""
            func = callee.func
            visit = _Visit.of(callee, reader)
            guessed = guessed or callee.guessed
            if any(v.func == visit.func and (not visit.literals or v.literals == visit.literals) for v in chain):
                return
            read_before = read_at.get(visit)
            if read_before is not None and read_before <= len(chain):
                return
            read_at[visit] = len(chain)
            if len(chain) >= _CALL_DEPTH:
                if files_only:
                    if past_the_limit(callee, report=report, chain=chain, guessed=guessed, files_only=True):
                        past_cap.append(func)
                elif not unread and past_the_limit(callee, report=report, chain=chain, guessed=guessed):
                    unread.append(func)
                return
            home = reader.cell_of(func)
            with reader.calling(func, callee.call, bound=callee.bound):
                # The function's own code, then each cell of literal code it
                # passes to exec(), read whole.
                for code in literal_code.with_ran(home, func):
                    cell, root = code.cell, func if code.cell is home else None
                    uses = _defined_uses(reader, code.root, cell, defined, modules)
                    found.extend(_file_findings(reader, cell, snapshots, root=root, report=report, owned=owned))
                    if not files_only:
                        sql.extend(_sql_findings(reader, cell, root=root, report=report))
                        calls.extend(_call_findings(reader, cell, root=root, report=report))
                        http.extend(_http_findings(reader, cell, rules=http_rules, root=root, report=report))
                        unread.extend(_unread_code(reader, code.root, cell, read=literal_code, modules=modules))
                        reaches_defined.extend(uses.sites)
                    for inner in (*_earlier_function_uses(reader, code.root, cell), *uses.callees):
                        read_function(inner, report=report, chain=(*chain, visit), guessed=guessed)

        def past_the_limit(
            callee: _Callee, *, report: _Report, chain: tuple[_Visit, ...], guessed: bool, files_only: bool = False,
        ) -> bool:
            """Whether `callee`, a function past `_CALL_DEPTH` calls, writes,
            runs code the check can't read, or calls another earlier
            function the chain hasn't read: it is read for that alone, with
            no call followed, and nothing it finds goes on the card. With
            `files_only`, only a held file change counts. A chain that
            started at a method of every class that defines its name, on a
            receiver the check can't place (`_Callee.guessed`), holds only
            on what it finds: such a chain may be none the cell runs."""
            func = callee.func
            home = reader.cell_of(func)
            read_ids = {visit.func for visit in chain}
            with reader.calling(func, callee.call, bound=callee.bound):
                for code in literal_code.with_ran(home, func):
                    cell, root = code.cell, func if code.cell is home else None
                    changes = _file_findings(reader, cell, snapshots, root=root, report=report, owned=owned)
                    if any(change.held for change in changes):
                        return True
                    if files_only:
                        continue
                    if (_sql_findings(reader, cell, root=root, report=report)
                            or _call_findings(reader, cell, root=root, report=report)
                            or _http_findings(reader, cell, rules=http_rules, root=root, report=report)
                            or _unread_code(reader, code.root, cell, read=literal_code, modules=modules)):
                        return True
                    uses = _defined_uses(reader, code.root, cell, defined, modules)
                    reaches_defined.extend(uses.sites)
                    inner = [*_earlier_function_uses(reader, code.root, cell), *uses.callees]
                    if not guessed and any(c.func is not func and id(c.func) not in read_ids for c in inner):
                        return True
            return False

        for cell in (current, *ran):
            uses = _defined_uses(reader, cell.tree, cell, defined, modules)
            found.extend(_file_findings(reader, cell, snapshots, owned=owned))
            if not files_only:
                sql.extend(_sql_findings(reader, cell))
                calls.extend(_call_findings(reader, cell))
                http.extend(_http_findings(reader, cell, rules=http_rules))
                unread.extend(_unread_code(reader, cell.tree, cell, read=literal_code, modules=modules))
                reaches_defined.extend(uses.sites)
            # Each earlier function the cell calls, and each function a
            # literal exec in an earlier cell defined, read at the call.
            for callee in (*_earlier_function_uses(reader, cell.tree, cell), *uses.callees):
                use = callee.use
                report = _Report(line=cell.host_line or use.lineno, call=scrub_output(cell.source(use)))
                read_function(callee, report=report, chain=())

    read_cells(reader, found=files, files_only=False)
    if _errored_names_read(cells, errored=frozenset(errored)):
        # Read again with each errored cell as maybe stopped anywhere, for
        # the files only that reading holds. Nothing else is taken from it:
        # a wider set of paths may hold less somewhere, never more. Past
        # `_PATH_LIMIT` paths for one value it gives up, and the first read
        # stands alone.
        wide = _Reader(cells, current=current, project_root=project_root, errored=frozenset(errored))
        wide.made = reader.made
        maybe: list[FileChange] = []
        read_cells(wide, found=maybe, files_only=True)
        if not wide.past_limit:
            files += _added_holds(found=files, maybe=maybe)
            unread += past_cap
    # A call that reaches what code the check couldn't read may have defined
    # holds the cell only when such code ran.
    if first_unread is not None:
        unread += reaches_defined
    return CellWrites(
        sql=tuple(sorted(sql, key=lambda w: w.line)),
        calls=tuple(sorted(calls, key=lambda c: c.line)),
        files=tuple(files),
        http=tuple(sorted(http, key=lambda w: w.line)),
        # A depth limit that may have stopped a read short of a write
        # (`_Reader.stopped`) leaves the cell unchecked too.
        unchecked=bool(unread) or reader.stopped,
    )
