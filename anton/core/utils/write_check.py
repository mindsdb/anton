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
  ``compile``, bytes decoded by their coding comment as Python decodes them,
  a literal statement handed to ``timeit``, ``cProfile.run``,
  ``profile.run``, ``pdb.run`` or ``pdb.runeval``, and the literal after
  ``python -c``, each read like one more cell, placed where the call that
  runs it runs, with its own imports; and in a function a literal exec in
  an earlier cell defined, where this cell calls it. A program's or
  module's name, or a package pip installs, held in a name bound only to
  literal text, or in a list built from such text or grown in place with
  it, reads as that text.
- **Code the check can't read**, which holds the cell unchecked: its card
  ends "This step can't be checked before it runs." That is ``exec``,
  ``eval`` or ``compile`` reached through the builtins however the code
  names them (a builtin function's ``__self__`` and a frame's
  ``f_builtins`` among them) and handed anything but a literal (a name, a
  file read, a decode, a join) or a literal too long to read, or passed or stored as a
  value; code the pad's own process runs through ``runpy``, ``importlib``
  (``reload``, a spec or a loader from a file, an import by a name the code
  picks when it runs), ``zipimport``, ``site`` (the ``.pth`` files it reads),
  a code object, or one a function is given as its ``__code__``, an
  interactive console, a debugger, a profiler or a subclass of one,
  ``trace``, ``doctest``, ``pytest.main``, a notebook runner, IPython's
  ``run_cell``, the C API's ``PyRun_`` functions, or a runner of text handed
  code that isn't a literal or a function it reads, each under any name or
  lookup (``getattr``, ``operator.methodcaller`` and ``attrgetter`` among
  them) and also passed or stored as a value, a loader's or a runner's own
  method too; a write of a startup hook a new Python process runs as it
  starts, a ``.pth`` file in the installed packages, the user site, on
  Windows the environment's own folder or its base Python's, or a folder
  the check can't place but for a literal with no line it runs, or
  a ``sitecustomize`` or ``usercustomize`` module or package, by a copy, a
  move or a download that lands one too (a ``.pth`` file in the project,
  as PyTorch saves weights, runs only once ``site.addsitedir`` adds its
  folder, which is held itself);
  once a cell wrote a hook a new Python process runs as it starts (such a
  module in any folder, or a ``.pth`` file in the installed packages, the
  user site, a folder with a ``site-packages`` or ``dist-packages`` part,
  on Windows the environment's own folder or its base Python's, or a
  folder the check can't place, or a file written where the check
  can't place it under a name such a hook may have, read through a name
  the path is bound to or a temporary file's suffix, a frame's or a
  model's only inside the installed packages), or unpacked or copied a
  folder's files into the installed packages, a Python process the cell
  starts, and the first cell after the pad starts a new process of its
  own, one that doesn't parse among them: after a cell whose process
  died, was killed or was canceled, a remote pad's own among them, after
  a remote pad's error, or when the pad's process isn't running as the
  cell comes;
  ``site.addsitedir`` of a folder that holds a
  ``.pth`` file, on disk or as a cell wrote it; an import of a module a
  cell wrote (to the project, or to a temporary folder, as source or as
  bytecode with no source), or one code the check can't read stored in
  ``sys.modules``, or one new since the turn began in the project or a
  folder a cell put on ``sys.path``, or of a project module a folder on
  ``sys.path`` reaches once a cell changed it or wrote where the check
  can't place; tests ``unittest`` discovers in the working folder; a
  Python program the cell starts, itself or through a launcher, or a
  ``functools.partial`` of one, it hands to a caller that runs it, a
  function the pad defines among them (``python`` with a script, a module, code from its input
  or ``-c`` in another folder or environment; IPython with the same, an
  option that loads code or a ``-c`` of its own syntax; a ``.py`` script;
  ``pytest``, ``papermill``, ``jupyter nbconvert --execute``, ``streamlit run``, a web
  app server, a task worker, a migration tool, a profiler; what ``uv run``,
  ``poetry run`` and the like start; behind ``xvfb-run``, ``caffeinate``,
  ``flock -c``, ``eval`` or backticks; through ``anyio``, ``trio``,
  ``pexpect`` or ``invoke``; under its Windows names too, as
  ``python.exe``, the ``py`` launcher, ``cmd /c`` and PowerShell's
  ``-Command``), pip or another installer building a package from a file,
  folder, URL, repository or index, or from a word the check can't read,
  or pointed at another index or a folder by a setting in its environment
  (``PIP_INDEX_URL``, ``UV_FIND_LINKS`` and their kin, given to the call or
  set in the pad's own), or building the project itself (``uv sync``, ``poetry
  install``, ``uv add`` into a project that builds), and a program the check
  can't name at all; a call to a name, or through a namespace, module or
  value, that such code may have filled, once such code ran and before read
  code surely gave the name a value of its own on every path to the call,
  a name an errored cell may not have bound, or an if, loop or try may
  skip binding, among them, and so may a with whose context may swallow an
  error (a swallowing context under any name a cell binds to it, a class
  a cell defines with ``__exit__``, or a context such code made or a
  module a cell wrote gives), where one from the standard library or an
  installed package lets it through; a with that enters a context such
  code made, whose enter and exit run it; and a read a depth limit stopped
  where a write may lie past it. A helper the pad's boot injects, such as
  ``progress`` or ``sample``, is a name read code binds, and a function
  nothing calls runs no program its parameters name.

What it can't find: a statement or path with no literal behind it, a plain
procedure call the check can't tie to a write it runs, a write through an API
other than HTTP PUT, PATCH or DELETE or an operator-listed endpoint (aiohttp,
an SDK such as boto3 outside DynamoDB), a store method on a receiver the check
can't place, a shell command it can't read at all, a shell redirect or ``cp``
onto an existing file, an overwrite whose path the code picks from a root the
check can't place, what a program other than Python does (a shell script
file, a ``make`` target, ``node``, ``Rscript``), a program launcher stored as
a value rather than handed to a caller that runs it, a write
through a trusted name that code the check couldn't read rebinds (``print =
requests.post``), what a ``pickle``, ``dill`` or ``joblib`` load runs, a call
to what unread code defined through ``.get(key)()`` or a list's element,
code or a statement the cell reads from a file when it runs beyond the cases
above, and a startup hook a cell wrote before the pad was reset: a reset
clears the cells the check reads, so the processes that start after it run
that hook with no card. An overwrite through a name the code picks when it
runs, as in ``f"exports/orders_{date.today()}.csv"``, counts as a new file,
unless a listing (``glob``, ``os.listdir``, ``iterdir``) gave the name.
"""

from __future__ import annotations

import ast
import bisect
import builtins
import copy
import errno
import fnmatch
import importlib.machinery
import importlib.util
import itertools
import logging
import ntpath
import os
import posixpath
import re
import shlex
import stat
import string
import sys
import textwrap
import threading
import time
import types
import warnings
from collections import ChainMap, OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import IntEnum
from functools import cached_property
from pathlib import Path
from typing import Any, Generic, Literal, NamedTuple, Protocol, TypeVar, overload

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from anton.core.backends.wire import (
    BOOT_IMPORTS,
    EXPLAINABILITY_QUERIES_NAME,
    INJECTED_HELPER_NAMES,
    NOTHING_KEPT,
    heal_surrogate_source,
)
from anton.utils.datasources import scrub_output

__all__ = [
    "CallVia",
    "CallWrite",
    "CellWrites",
    "EarlierCell",
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
    or a Python program it runs, what such code may have defined, a startup
    hook a cell wrote that a new Python process runs, the cell's own or the
    pad's as it starts again before the cell, or a read a depth limit
    stopped short; or the check itself failed. The card then says the step
    can't be checked before it runs, beside what it did read."""

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


class EarlierCell(BaseModel):
    """One earlier cell of the conversation, as `find_cell_writes` reads it:
    its code, whether it stopped on an error (`errored`), and, for one that
    did, the line of its own code it stopped at, as the pad's boot read it
    from the traceback (`cell_stopped_at`); None where the runtime doesn't
    say. `ended_process` says its process died, was killed, was canceled
    or never ran it, or that the pad's process wasn't running as the next
    cell came (`hold_unapproved_write`), so the pad starts a new process
    before the next cell, and that start runs the startup hooks a cell
    wrote."""

    model_config = ConfigDict(frozen=True)

    code: str
    errored: bool = False
    stopped_at: int | None = None
    ended_process: bool = False


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
            name for path in self.files if not _names_package_folder(path)
            for name in _module_names(path)
        )

    @cached_property
    def module_places(self) -> frozenset[str]:
        """Where an import reaches each module file under the root
        (`_module_place`), as `lib/erp` for `lib/erp.py`, read once per
        snapshot."""
        return frozenset(place for path in self.files for place in [_module_place(path)] if place is not None)

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
# The methods of a test loader that import the test modules they find or
# are named: `unittest.defaultTestLoader`'s, and a `unittest.TestLoader`'s
# (`_INSTANCE_RUNNERS`).
_TEST_LOADER_RUNS = frozenset({"discover", "loadTestsFromName", "loadTestsFromNames"})
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
    # Each runs the import lines of the .pth files in the folders it adds;
    # `site.addsitedir` only where the folder may hold one (`_adds_unread_site`).
    "site.addsitedir", "site.addpackage", "site.main",
    # The default loader's own discovery, which imports the test modules it
    # finds. `unittest.TestLoader()` goes through `_INSTANCE_RUNNERS`.
    *(f"unittest.defaultTestLoader.{run}" for run in _TEST_LOADER_RUNS),
    # Importers and loaders that run a module from a path or a name.
    "pydoc.importfile", "imp.load_source", "imp.load_module", "imp.load_compiled",
    "imp.load_package", "imp.reload", "importlib._bootstrap._load", "importlib._bootstrap._exec",
    # The doctest suites a test runner runs, and its own debuggers.
    "doctest.DocFileSuite", "doctest.DocTestSuite", "doctest.DebugRunner", "doctest.debug_src",
    "doctest.debug_script", "doctest.debug",
    # The command-line mains that run the script or module named on sys.argv.
    "cProfile.main", "profile.main", "trace.main", "pdb.main",
    # IPython shells and notebook runners made outside `get_ipython`.
    "IPython.start_ipython", "IPython.embed", "testbook.testbook",
    "nbconvert.preprocessors.ExecutePreprocessor", "nbconvert.preprocessors.execute.ExecutePreprocessor",
    # Subinterpreters that run a text, the site hooks that import a
    # customize module, and setup scripts a build runs.
    "_xxsubinterpreters.run_string", "_interpreters.run_string", "_interpreters.exec",
    "site.execsitecustomize", "site.execusercustomize",
    "distutils.core.run_setup", "setuptools._distutils.core.run_setup",
})
# Modules whose members run code the check can't read, so a member looked up
# by a name it can't read may be one.
_RUNNER_MODULES = frozenset(
    name.rpartition(".")[0] for name in (*_CODE_RUNNERS, "importlib.util.module_from_spec")
)
# The C API functions that run code the check can't read: PyRun_* run the
# text they're handed, PyImport_* import a module by its name. The start of
# their bare names, and of their full names on `ctypes.pythonapi`.
_PYAPI_PREFIXES = ("PyRun_", "PyImport_")
_PYAPI_FUNCS = tuple(f"ctypes.pythonapi.{prefix}" for prefix in _PYAPI_PREFIXES)
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
# The methods that run code on an IPython shell `.instance()` makes, beyond
# those a magic reaches: it runs a cell, a line, a module or a code object.
_IPYTHON_SHELL_RUNS = _IPYTHON_RUNS | {
    "ex", "ev", "safe_execfile", "safe_run_module", "run_code", "runcode", "run_cell_async",
}
_INSTANCE_RUNNERS = {
    "cProfile.Profile": frozenset({"run", "runctx", "runcall"}),
    "profile.Profile": frozenset({"run", "runctx", "runcall"}),
    "pdb.Pdb": frozenset({"run", "runeval", "runctx", "runcall"}),
    "bdb.Bdb": frozenset({"run", "runeval", "runctx", "runcall"}),
    "trace.Trace": frozenset({"run", "runctx", "runfunc"}),
    "nbclient.NotebookClient": frozenset({"execute"}),
    "nbclient.client.NotebookClient": frozenset({"execute"}),
    "unittest.TestLoader": _TEST_LOADER_RUNS,
    "unittest.loader.TestLoader": _TEST_LOADER_RUNS,
    "IPython.core.interactiveshell.InteractiveShell.instance": _IPYTHON_SHELL_RUNS,
    "IPython.terminal.interactiveshell.TerminalInteractiveShell.instance": _IPYTHON_SHELL_RUNS,
    "IPython.terminal.embed.InteractiveShellEmbed": _IPYTHON_SHELL_RUNS,
    "get_ipython": _IPYTHON_RUNS, "IPython.get_ipython": _IPYTHON_RUNS,
    "IPython.core.getipython.get_ipython": _IPYTHON_RUNS,
}
_RUNNER_METHODS = frozenset().union(*_INSTANCE_RUNNERS.values())
# Methods that run the module a loader or importer loads, whatever made it.
_LOADER_METHODS = frozenset({"exec_module", "load_module"})
# Calls that import a module by its name. `pkgutil.resolve_name` and
# `pydoc.locate` take the name as "pkg.mod:attr" or "pkg.mod"; the module
# before the colon is the one they load. `pkgutil.walk_packages` imports
# every package it finds under the folders it is given (`_walk_packages_unread`).
_IMPORTERS = frozenset({
    "importlib.import_module", "importlib.__import__", "__import__", "builtins.__import__",
    "pkgutil.resolve_name", "pydoc.locate", "pydoc.safeimport", "pydoc.render_doc", "pydoc.doc",
    "help", "pkgutil.walk_packages",
})
# The importers that import only a string argument, so an object given to one
# imports nothing: the builtin `help` and pydoc's renderers take a thing or
# its name, and a thing is no import the check reads.
_STRING_IMPORTERS = frozenset({"help", "pydoc.render_doc", "pydoc.doc"})
# The importers whose result is the module itself, so a name bound to one
# reads as that module (`_Reader._importer_module_texts`): all but those that
# return an attribute, a rendering or a walk rather than the module.
_MODULE_IMPORTERS = _IMPORTERS - _STRING_IMPORTERS - {
    "pkgutil.resolve_name", "pydoc.locate", "pkgutil.walk_packages",
}
_MODULE_IMPORTER_LAST_NAMES = frozenset(name.rpartition(".")[2] for name in _MODULE_IMPORTERS)
# The test-program entry points that load and run a module or the tests
# named on their argv (`_unittest_main_unread`), held unless they run the
# cell's own TestCase from `__main__` with no argv past the program word.
_UNITTEST_MAINS = frozenset({"unittest.main", "unittest.TestProgram", "unittest.main.TestProgram"})
# The command-line main that runs the statements on its argv list, read as a
# runner of text (`_timeit_main_code`).
_TIMEIT_MAIN = "timeit.main"
# The last name of each call `_runs_unread` reads, so any other is passed by.
_RUNNER_NAMES = frozenset(
    name.rpartition(".")[2]
    for name in (*_CODE_RUNNERS, *_STRING_RUNNERS, *_IMPORTERS, *_UNITTEST_MAINS, _TIMEIT_MAIN)
)
# Names that reach a runner or an importer with no import: the builtins
# `__import__` and `help`, and `get_ipython`, which an IPython shell binds.
_PAD_BUILTINS = frozenset({"__import__", "help", "get_ipython"})
# What changes the folders `import` looks in, and the methods that add to a
# list, as `sys.path` is one.
_LIST_CHANGES = frozenset({"insert", "append", "extend", "__setitem__", "__iadd__"})
_SYS_PATH_CHANGES = frozenset({
    "sys.path", "site.addsitedir", *(f"sys.path.{method}" for method in _LIST_CHANGES),
})
# The call that adds a folder to `sys.path` and runs its `.pth` files.
_SITE_DIR_ADDER = "site.addsitedir"
# A word that names a Python file: its source, bytecode Python loads with no
# source, or a compiled extension (`_module_stem`).
_PY_FILE = re.compile(r"\.(?:pyc?|so|pyd)\b")
# The startup hooks a Python process runs as it starts, from each folder
# `site` adds: the import lines of a `.pth` file, and these modules; and
# how a string may name one (`_CodeSigns.names_a_startup_hook`).
_STARTUP_HOOK_MODULES = frozenset({"sitecustomize.py", "usercustomize.py"})
# The folders of those modules written as packages, as `sitecustomize/__init__.py`.
_STARTUP_HOOK_PACKAGES = frozenset(name.removesuffix(".py") for name in _STARTUP_HOOK_MODULES)
_NAMES_A_STARTUP_HOOK = re.compile(r"\.pth|(?:site|user)customize")
# Every name Python, or the pad's boot, binds before a cell runs, so a call
# to one is no sign that code the check couldn't read defined it.
_BUILTIN_NAMES = frozenset({*vars(builtins), *INJECTED_HELPER_NAMES, EXPLAINABILITY_QUERIES_NAME})
# Runners that import and run an app from a string target, as
# `uvicorn.run("module:app")` imports `module`, and the config a
# `uvicorn.Server` runs; given an app object instead, they run read code
# (`_app_runner_unread`).
_APP_RUNNERS = frozenset({"uvicorn.run", "uvicorn.Config", "uvicorn.config.Config"})
# The calls whose result may be text, so an app runner or an importer given
# one may be given a name to import the check can't read (`_may_be_text`).
_TEXT_SOURCES = frozenset({
    *_STR_METHODS, "read", "read_text", "readline", "join", "get", "getenv", "pop", "input", "str", "getattr",
    "loads",
})
# Runners and importers that run code the check can't read wherever they are
# passed on or stored to be called later.
_RUNNER_VALUES = frozenset({*_CODE_RUNNERS, *_STRING_RUNNERS, *_IMPORTERS})
# The runners that are types: as the class `isinstance` checks, a side of a
# comparison, a dict key or an annotation, nothing calls them
# (`_type_runner_aside`).
_RUNNER_TYPES = frozenset({"types.FunctionType", "types.CodeType"})
# Runners whose return value is a namespace a later call reaches through.
_NAMESPACE_RUNNERS = frozenset({"runpy.run_path", "runpy.run_module"})
# The calls that hand out the cell's own namespace to fill or read by key.
_GLOBALS_CALLS = frozenset({"globals", "vars", "locals"})
# What an import binds to reach `sys.modules`, which holds every module by name.
_SYS_MODULE_NAMES = frozenset({"sys", "sys.modules"})
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
# Calls that make a link at the path their second argument names, which an
# import reaches like the file it points at; the path methods that make one
# at the path they are called on (`_written_target`).
_LINK_FUNCS = frozenset({"os.symlink", "os.link"})
_LINK_METHODS = frozenset({"symlink_to", "hardlink_to"})
_LINK_NAMES = frozenset({*(name.rpartition(".")[2] for name in _LINK_FUNCS), *_LINK_METHODS})
# Calls that copy or move a file and, given a folder, keep its own name
# (`_copied_module_stems`); `copyfile` needs a file destination, so it keeps
# no name.
_COPY_INTO_FOLDER = (_COPY_FUNCS - {"shutil.copyfile"}) | {"shutil.move"}
# Modules and classes that hold a delete, a move, a copy or a shell call, for
# a lookup by a name the check computes (`_FileFinder.lookup`).
_LOOKUP_MODULES = frozenset({
    "os", "shutil", "pathlib", "pathlib.Path", "pathlib.PosixPath", "send2trash", "subprocess",
})
# `os.startfile` opens a file with the program its type is bound to, running
# a `.py` it names (`_python_launch`).
_STARTFILE = "os.startfile"
_SHELL_FUNCS = frozenset({
    "subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output",
    "subprocess.Popen", "subprocess.getoutput", "subprocess.getstatusoutput", "os.system", "os.popen",
    "asyncio.create_subprocess_shell", "anyio.run_process", "anyio.open_process", "trio.run_process",
    "trio.lowlevel.open_process", "pexpect.run", "pexpect.spawn", "invoke.run", "IPython.utils.process.system",
    "IPython.utils.process.getoutput", "psutil.Popen", _STARTFILE,
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
# An asyncio event loop's own launcher methods, placed by their name since
# the loop `get_running_loop` or `get_event_loop` returns is one the check
# can't read: one runs a program's words after the protocol, the other a
# shell command (`_loop_launch`).
_LoopLaunch = Literal["exec", "shell"]
_LOOP_LAUNCHERS: dict[str, _LoopLaunch] = {"subprocess_exec": "exec", "subprocess_shell": "shell"}
# Every call that runs a program, and the last names they are called by.
_LAUNCHERS = frozenset({*_SHELL_FUNCS, *_SPREAD_ARGV_FUNCS, *_LIST_ARGV_FUNCS})
_LAUNCH_NAMES = frozenset(name.rpartition(".")[2] for name in _LAUNCHERS)
# The modules that hold a launcher, so a member looked up on one by a name
# the check can't read may be a launcher.
_LAUNCHER_MODULES = frozenset(name.rpartition(".")[0] for name in _LAUNCHERS)
# The names the builtins module goes by in a cell.
_BUILTINS_MODULE_NAMES = frozenset({"builtins", "__builtins__"})
# The members that hold the builtins module, or its dict, on any holder, as
# `sys.modules["__main__"].__builtins__`, a frame's `f_builtins` and a
# builtin function's `__self__` do (`_Reader._builtins_module`).
_BUILTINS_HOLDER_MEMBERS = frozenset({"__builtins__", "f_builtins", "__self__"})
# The builtin functions, whose `__self__` is the builtins module.
_BUILTIN_FUNCTIONS = frozenset(
    name for name, value in vars(builtins).items() if isinstance(value, types.BuiltinFunctionType)
)
# The members of the builtins module that run no code the check can't read,
# as `builtins.open` and `builtins.print`: every builtin but exec and its
# kin, which `exec_builtin` reads, and those that reach code or the cell's
# own namespace. A member that isn't a builtin is one a cell may have set
# (`_Reader.loads_builtins_value`).
_BUILTINS_HARMLESS = frozenset(vars(builtins)) - _EXEC_CALLS - frozenset({
    "getattr", "setattr", "vars", "globals", "locals", "__import__", "__loader__", "__spec__", "breakpoint",
    "__build_class__",
})
# The last names that hint a cell may run code the check can't read
# (`_CodeSigns.hints`), and the calls `_names_a_runner` reads such a name as
# through the imports. A conversation whose cells name none of these that
# way, and write and import no module the check can't read, runs only code
# it reads.
_CODE_HINTS = frozenset({
    *_EXEC_CALLS, *_GLOBALS_CALLS, *_RUNNER_NAMES,
    *_RUNNER_METHODS, *_LOADER_METHODS, *_LAUNCH_NAMES, *_BUILTINS_MODULE_NAMES,
})
# Every call that runs code the check can't read, passed on or stored.
_RUN_CALLS = frozenset({*_RUNNER_VALUES, *_LAUNCHERS})


def _runners_by_module() -> dict[str, dict[str, str]]:
    """The runners and launchers each module holds (`_RUN_CALLS`), by the
    bare name a `from module import *` binds them to."""
    out: dict[str, dict[str, str]] = {}
    for full in _RUN_CALLS:
        module, _, name = full.rpartition(".")
        if module:
            out.setdefault(module, {})[name] = full
    return out


# The runners and launchers each module holds, by the bare name a
# `from module import *` binds them to (`_Reader.star_runners`).
_RUNNERS_BY_MODULE = _runners_by_module()
# The modules that hold something that runs code, so a member named off one
# by a name the check can't read may run code (`_Reader._pick_runner`).
_RUNNER_HOLDERS = frozenset({*_RUNNER_MODULES, *_LAUNCHER_MODULES, *_BUILTINS_MODULE_NAMES})
# The modules that hold a runner of text or a class or call whose result runs
# code, so a member looked up on one by a name bound to texts may be one
# (`_looks_up_a_runner`).
_MAKER_MODULES = frozenset(
    name.rpartition(".")[0] for name in (*_STRING_RUNNERS, *_INSTANCE_RUNNERS) if "." in name
)
# Calls that make a getter of a member by its name, called later on any
# holder: `methodcaller("run_path", "job.py")(runpy)` runs `runpy.run_path`.
_MEMBER_GETTERS = frozenset({"operator.methodcaller", "operator.attrgetter"})
_MEMBER_GETTER_NAMES = frozenset(name.rpartition(".")[2] for name in _MEMBER_GETTERS)
# The last names of the members that run code on what holds them, as a
# getter of members may name them: a runner, an importer, a launcher, a
# loader's or an instance runner's method, exec and its kin, and the members
# that hold the builtins module.
_RUNNING_MEMBERS = frozenset({
    *_RUNNER_NAMES, *_LOADER_METHODS, *_RUNNER_METHODS, *_LAUNCH_NAMES, *_EXEC_CALLS,
    *_BUILTINS_MODULE_NAMES, *_BUILTINS_HOLDER_MEMBERS,
})
# The last names of the calls that look a member up by its name: `getattr`
# and the getters of members.
_LOOKUP_CALL_NAMES = frozenset({"getattr", *_MEMBER_GETTER_NAMES})
# The last names of the calls that may run code the check can't read, however
# the imports name them (`_Cell.unread_candidates`): every hint that names a
# call, not the cell's own namespace or the builtins module, `getattr`, and
# the getters of members.
_UNREAD_CALL_NAMES = frozenset({*(_CODE_HINTS - _GLOBALS_CALLS - _BUILTINS_MODULE_NAMES), *_LOOKUP_CALL_NAMES})


def _is_hint(last: str) -> bool:
    """Whether `last`, the last name of a node, may name something that runs
    code the check can't read: a hint (`_CODE_HINTS`) or a PyRun_ function."""
    return last in _CODE_HINTS or last.startswith(_PYAPI_PREFIXES)


# The Python interpreter by any of its names, and the scripts it runs as a
# program of their own.
_PYTHON_PROGRAM = re.compile(r"(?:python|pypy|ipython)(?:\d+(?:\.\d+)*t?)?w?")
_PYTHON_SCRIPTS = (".py", ".pyw", ".pyz")
# Python's long options that print and exit, as the short -V and -h do.
_PYTHON_EXITS = frozenset({"--version", "--help", "--help-env", "--help-xoptions", "--help-all"})
# Programs that load a project's code, a test, a notebook, an app or a
# migration: by name, and by name and command. Asked only for their version
# or help (`_TOOL_EXITS`), they run nothing, but for pytest, which loads the
# project's conftest.py files first.
_PYTHON_TOOLS = frozenset({
    "pytest", "py.test", "papermill", "uvicorn", "gunicorn", "jupyter-execute", "hypercorn", "celery", "gradio",
    "pyinstrument", "scalene", "tox", "nox", "alembic", "flask", "sphinx-build", "dbt",
    # Launchers that start a training run, a worker or a server from a
    # script, a module or an app the project holds.
    "torchrun", "accelerate", "deepspeed", "locust", "invoke", "inv", "fab", "doit", "viztracer", "kernprof",
    "behave", "django-admin", "prefect", "airflow", "snakemake", "luigi", "voila", "mkdocs", "dask", "daphne",
    "fire", "datasette", "waitress-serve", "functions-framework",
})
_PYTHON_TOOL_COMMANDS = frozenset({
    ("streamlit", "run"), ("jupyter", "execute"), ("jupyter", "run"), ("coverage", "run"), ("fastapi", "dev"),
    ("fastapi", "run"), ("panel", "serve"), ("bokeh", "serve"), ("marimo", "run"), ("marimo", "edit"),
    ("dagster", "dev"),
    ("shiny", "run"), ("chainlit", "run"), ("solara", "run"), ("textual", "run"), ("modal", "run"),
    ("scrapy", "crawl"), ("scrapy", "runspider"), ("kedro", "run"), ("mlflow", "run"), ("memray", "run"),
    ("quarto", "render"), ("pre-commit", "run"), ("wandb", "agent"), ("sqlmesh", "run"), ("hatch", "build"),
    ("jupytext", "--execute"),
})
_TOOL_EXITS = frozenset({"--version", "-V", "--help", "-h"})
_CONFTEST_TOOLS = frozenset({"pytest", "py.test"})
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
    "mamba": (("run",),),
    "micromamba": (("run",),),
    "pipenv": (("run",),),
    "pixi": (("run",), ("exec",)),
    "rye": (("run",),),
}
# The flags of each runner that run its command in another folder, project
# or environment than the cell's, and those of the words that run the
# command after them, as `env -C` does.
_RUNNER_FOLDER_FLAGS: dict[str, frozenset[str]] = {
    "uv": frozenset({"--directory", "--project", "--env-file"}),
    "poetry": frozenset({"-C", "--directory", "-P", "--project"}),
    "pdm": frozenset({"-p", "--project"}),
    "conda": frozenset({"--cwd"}),
    "mamba": frozenset({"--cwd"}),
    "micromamba": frozenset({"--cwd"}),
    "pixi": frozenset({"--manifest-path"}),
    "rye": frozenset({"--pyproject"}),
}
_PREFIX_FOLDER_FLAGS = frozenset({"-C", "--chdir", "-D"})
# Installers besides pip, by the head word to the commands that install the
# packages named after them, and to those that install the project in the
# folder they run in, which runs its build code. A package named by a path,
# an archive or a URL is built from where it lies too (`_pip_source_operand`).
_PACKAGE_INSTALLS: dict[str, tuple[tuple[str, ...], ...]] = {
    "uv": (("add",), ("tool", "install")),
    "pipx": (("install",), ("inject",)),
    "poetry": (("add",),),
    "pdm": (("add",),),
    "pipenv": (("install",),),
    "conda": (("install",), ("create",)),
    "mamba": (("install",), ("create",)),
    "micromamba": (("install",), ("create",)),
    "pixi": (("add",),),
    "rye": (("add",),),
}
# Commands that add a package and then install the project in the folder
# they run in too, which runs its build code where it has one
# (`_Reader.project_builds`), unless a flag says not to sync or install it.
_SYNCING_ADDS: dict[str, tuple[tuple[str, ...], ...]] = {"uv": (("add",),)}
_NO_PROJECT_SYNC_FLAGS = frozenset({"--no-sync", "--frozen", "--no-install-project"})
# A pyproject.toml line that makes the project one an install builds.
_BUILDS_A_PROJECT = re.compile(r"^\s*(?:\[build-system\]|package\s*=\s*true\b)", re.M)
_PROJECT_INSTALLS: dict[str, tuple[tuple[str, ...], ...]] = {
    "uv": (("sync",),),
    "poetry": (("install",),),
    "pdm": (("install",), ("sync",)),
    "pipenv": (("sync",),),
    "hatch": (("env", "create"),),
    "pixi": (("install",),),
    "rye": (("sync",),),
}
# Installers' flags that take a value, beyond the runners' own.
_INSTALL_VALUE_FLAGS = frozenset({"-c", "--channel", "--suffix", "--source"})
_ENV_RUNNER_VALUE_FLAGS = frozenset({
    "--with", "--with-requirements", "--with-editable", "--python", "-p", "--project", "--directory",
    "--package", "--extra", "--group", "--env-file", "--index", "--from", "--spec", "-n", "--name",
    "--prefix", "--cwd", "-C", "-P", "--manifest-path", "--pyproject",
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
# pip's own entry points, which run a command in this process from an argv
# list read the way `-m pip`'s is (`_pip_main_unread`).
_PIP_MAINS = frozenset({"pip.main", "pip._internal.main", "pip._internal.cli.main.main"})
# The commands that fetch the packages they take from an index (`_pip_launch`).
_PIP_FETCH_COMMANDS = _PIP_PACKAGE_COMMANDS - {"uninstall"}
# The settings, in a program's environment, that point pip or uv at another
# index or a folder of links, or at a config file that may: as the source
# flags do (`_PIP_SOURCE_FLAGS`), and the folders their config files lie in.
_INSTALLER_SOURCE_SETTING = re.compile(
    r"(?:PIP|UV)_\w*(?:INDEX|FIND_LINKS|CONFIG)\w*|HOME|XDG_CONFIG_\w+|APPDATA", re.IGNORECASE,
)
# The settings, in a Python process's environment, that change what it
# imports or runs as it starts: the folders it imports from, its home and
# user site, the script an interactive start runs, and the modules a warning
# filter or `breakpoint()` names.
_PYTHON_SETTING = re.compile(
    r"PYTHON(?:PATH|HOME|USERBASE|PLATLIBDIR|STARTUP|INSPECT|WARNINGS|BREAKPOINT)", re.IGNORECASE,
)
# pip's flags that name where a package comes from, the long ones matched by
# unique prefix (`--req` is `--requirement`), and the short ones' letters.
_PIP_SOURCE_FLAGS = frozenset({
    "-r", "--requirement", "--requirements", "-e", "--editable", "-i", "--index-url", "--extra-index-url",
    "-f", "--find-links",
})
_PIP_SOURCE_LONG = frozenset(flag for flag in _PIP_SOURCE_FLAGS if flag.startswith("--"))
_PIP_SOURCE_LETTERS = frozenset(flag[1] for flag in _PIP_SOURCE_FLAGS if len(flag) == 2)
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
_PIP_VALUE_LETTERS = frozenset(flag[1] for flag in _PIP_VALUE_FLAGS if len(flag) == 2)
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
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish", "tcsh", "csh"})
# Windows' shells: cmd runs the text after `/c` or `/k` as a script, and
# PowerShell the text after `-Command` (`_inner_script`).
_CMD_SHELLS = frozenset({"cmd"})
_POWERSHELLS = frozenset({"powershell", "pwsh"})
# Commands that run a script in turn: a shell, and `eval`, which runs its words.
_SCRIPT_RUNNERS = frozenset({*_SHELLS, *_CMD_SHELLS, *_POWERSHELLS, "eval"})
# PowerShell's flags, by their lowercase names: those that take the script
# itself, a script encoded in base64, a script file, or another value.
_POWERSHELL_COMMAND_FLAGS = frozenset({"-command", "-c"})
_POWERSHELL_ENCODED_FLAGS = frozenset({"-encodedcommand", "-enc", "-e", "-ec"})
_POWERSHELL_FILE_FLAGS = frozenset({"-file", "-f"})
_POWERSHELL_VALUE_FLAGS = frozenset({
    "-executionpolicy", "-ep", "-ex", "-windowstyle", "-w", "-version", "-v", "-inputformat", "-if",
    "-outputformat", "-of", "-o", "-configurationname", "-config", "-workingdirectory", "-wd",
    "-psconsolefile", "-settingsfile", "-settings", "-custompipename",
})
# The Windows launcher `py`'s flags that pick the Python it runs, as `-3` or
# `-V:3.12`, and those that list the Pythons it knows and run none.
_PY_LAUNCHER_FLAG = re.compile(r"-(?:\d[\w.-]*|V:\S+)")
_PY_LAUNCHER_LISTS = frozenset({"-0", "-0p", "--list", "--list-paths"})
# Words that change the working folder, how many shells deep a script is read
# before its commands count as unreadable, and the `find` flags that run a
# command on each match: read the same way by the delete, move and launch walks.
_CD_WORDS = ("cd", "pushd")
_SHELL_DEPTH = 4
_FIND_EXEC_FLAGS = ("-exec", "-execdir", "-ok", "-okdir")
# Words that run the command after them, and shell keywords a command follows.
_SHELL_PREFIXES = frozenset({
    "sudo", "doas", "env", "nohup", "time", "command", "exec", "nice", "xargs",
    "timeout", "stdbuf", "ionice", "busybox", "setsid", "xvfb-run", "caffeinate", "flock", "watch", "chrt",
    "taskset", "unbuffer", "parallel",
})
# Flags of one runner that take the next word as their value, beyond
# `_VALUE_FLAGS`; those of `_VALUE_FLAGS` that take none for one runner; and
# the operands a runner takes before the command, as timeout takes its
# duration.
_PREFIX_VALUE_FLAGS = {
    "timeout": frozenset({"-k"}), "stdbuf": frozenset({"-i", "-o", "-e"}), "ionice": frozenset({"-c", "-p"}),
    "xargs": frozenset({"-E"}), "xvfb-run": frozenset({"-e", "-f", "-p", "--server-num", "--server-args"}),
    "caffeinate": frozenset({"-t", "-w"}), "flock": frozenset({"-w", "--timeout", "-E"}),
    "watch": frozenset({"--interval"}), "parallel": frozenset({"-j", "--jobs"}),
}
_PREFIX_PLAIN_FLAGS = {
    "xvfb-run": frozenset({"-a"}), "caffeinate": frozenset({"-d", "-i", "-m", "-s", "-u"}),
    "flock": frozenset({"-s", "-n", "-u"}), "watch": frozenset({"-d"}), "chrt": frozenset({"-a"}),
    "taskset": frozenset({"-a"}),
}
# A shell word that sets a variable for the command after it.
_SHELL_ASSIGNMENT = re.compile(r"^[A-Za-z_]\w*=")
_PREFIX_OPERANDS = {"timeout": 1, "flock": 1, "chrt": 1, "taskset": 1}
# The flags that hand flock a shell command to run, in place of a program and its words.
_FLOCK_COMMAND_FLAGS = frozenset({"-c", "--command"})
_SHELL_KEYWORDS = frozenset({"if", "then", "else", "elif", "do", "while", "until", "!", "{"})
# Shell words that end one command and start the next.
_SHELL_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "(", ")", "|&", ";;"})
# Folders the pad changes into that the check tracks; past this many, any.
_CWD_LIMIT = 16
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
# Calls into earlier cells' functions followed from one call in this cell.
_CALL_DEPTH = 4
# How many times one check reads a function again for a recursive call that
# passes other literals; past this, the call is read only for whether it may
# write or call on, as one past `_CALL_DEPTH` calls is (`_CellRead`).
_REREAD_LIMIT = 1
# Methods that write the file their first argument names, and the keywords
# that can name it instead.
_WRITER_METHODS = {
    "to_csv": ("path_or_buf",), "to_json": ("path_or_buf",), "to_excel": ("excel_writer",),
    "to_parquet": ("path",), "to_feather": ("path",), "to_pickle": ("path",), "to_html": ("buf",),
    "savefig": ("fname",), "save": ("filename", "fp", "path", "file"),
}
# The last names of the calls `_written_target` reads a written path from.
_FILE_WRITE_NAMES = frozenset({
    "open", "write_text", "write_bytes", "ExcelWriter", "rename", "replace", *_WRITER_METHODS, *_LINK_METHODS,
    *(name.rpartition(".")[2] for name in (*_MOVE_FUNCS, *_COPY_FUNCS, *_LINK_FUNCS)),
})
# Folders under the project whose files are the agent's own output.
_AGENT_FOLDERS = tuple(f".anton/{name}/" for name in sorted(_ANTON_WALKED))
_PATH_TYPES = frozenset({"pathlib.Path", "pathlib.PurePath", "pathlib.PosixPath", "pathlib.PurePosixPath"})
_PATH_MAKERS = frozenset({"glob", "rglob", "iterdir", "resolve", "absolute", "with_suffix",
                          "with_name", "with_stem", "joinpath", "expanduser"})
_ITER_WRAPPERS = frozenset({"next", "sorted", "list", "max", "min", "reversed", "iter", "tuple", "set", "frozenset"})

# The comprehensions, each a scope of its own, and every scope a name may bind in.
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, *_COMPREHENSIONS)
# Statements that give a name a value: those with a value of their own, and
# loops and withs, whose head gives it (`_stores`).
_VALUE_STORES = (ast.Assign, ast.AnnAssign, ast.NamedExpr)
_HEAD_STORES = (ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)
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
# Stands in, in a command's words, for words the check can't place, any
# number of them in any order, drawn from the words after it, as one list
# for many does (`_capped`). Read as a word the check can't read, it is no
# runner's operand, and pip reads each word after it on its own.
_SOME_WORDS = _NOT_LITERAL + "*"
# Words the check can't read at all, any number of them, as a spread of a
# sequence it can't read gives.
_UNREAD_WORDS = (_SOME_WORDS, _NOT_LITERAL)
# Names a read follows through their values, and parameters through the
# calls that pass them, before it stops (`_Reader.stops`).
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
    means it sits under an if, loop, try or match, so it may not run
    (`_Cell._under_branch`).
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
    function, a static method, or a method called on its class. `guessed`
    marks a method of every class that defines its name, on a receiver the
    check can't place; `past` one found only by reading on past a depth
    limit (`_Reader.read_on`), which is read for whether it may write alone.
    """

    func: Function
    call: ast.Call | None
    use: ast.expr
    bound: ast.AST | None = None
    guessed: bool = False
    past: bool = False


class _Literal(NamedTuple):
    """A literal a call passes for one parameter, by the parameter's name and
    the literal's repr."""

    param: str
    value: str


class _Passed(NamedTuple):
    """Code a call passes for one parameter that isn't a literal, by the
    parameter's name and the id of the code that gives it its value."""

    param: str
    node_id: int


@dataclass(frozen=True)
class _Visit:
    """One read of a function at a call (`find_cell_writes`): the function,
    by id, and what the call passes for each parameter: the literals, a
    parameter it passes on read as what the call being read gave it, so a
    recursive call that passes other literals is read again with them; and
    the other code it passes, by the node that gives each value (`passed`),
    so two calls that pass different names are two reads."""

    func: int
    literals: tuple[_Literal, ...]
    passed: tuple[_Passed, ...] = ()

    @classmethod
    def of(cls, callee: _Callee, reader: _Reader) -> _Visit:
        """The visit `callee` makes: its function, and what its call passes
        for each parameter, a literal by its repr, other code by its node."""
        literals: list[_Literal] = []
        others: list[_Passed] = []
        if callee.call is not None:
            passed = _call_arguments(callee.func, callee.call, call_cell=reader.cell_of(callee.call),
                                     func_cell=reader.cell_of(callee.func), bound=callee.bound)
            for name, arg in passed.items():
                if arg is None:
                    continue
                node = reader.passed_node(arg.node)
                if isinstance(node, ast.Constant):
                    literals.append(_Literal(name, repr(node.value)))
                else:
                    others.append(_Passed(name, id(node)))
        return cls(func=id(callee.func), literals=tuple(sorted(literals)), passed=tuple(sorted(others)))


class _ParamValues(NamedTuple):
    """What a parameter can hold (`_Reader.param_values`): the arguments, None
    for one the check can't read, and whether they are what the call being
    read passes (`passed`) or what each call site passes."""

    args: tuple[_Arg | None, ...]
    passed: bool


@dataclass(frozen=True, eq=False)
class _Remembered(Generic[_T]):
    """What one read gave (`_Reader.recalled`), and how many times a depth
    limit stopped it (`_Reader.cuts`)."""

    value: _T
    cuts: int


class _FillReads(NamedTuple):
    """What one of a cell's `_Cell.fills_that_run` reads: the names it
    loads, and the methods it calls, None when it also calls something else,
    as a bare name, or is an import. It is read again for what code the
    check can't read filled only when a name it loads was filled more, and
    not at all when none of those may hold such code (`_plainly_read`)."""

    names: frozenset[str]
    methods: frozenset[str] | None


class _ListChange(NamedTuple):
    """One place a cell adds to a bare name's list, or stores to it: the name,
    and the node that does it."""

    name: str
    node: ast.AST


class _ModuleStoreSite(NamedTuple):
    """One place a cell stores a module in a mapping spelled `.modules`
    (`_Cell.module_stores`): the mapping, which `_stored_modules` reads as
    `sys.modules` through the imports; the code of the name it stores the
    module under, None where the cell hands over a mapping of them the
    check can't read; and the code of the module."""

    holder: ast.AST
    key: ast.AST | None
    value: ast.AST


class _Maker(NamedTuple):
    """One def, class or import a cell binds a name by (`_Reader.makers`)."""

    cell: _Cell
    statement: ast.AST


class _AsideSite(NamedTuple):
    """One place a cell gives a name a value that `_Cell.bindings` leaves
    out (`_CodeSigns.aside_sites`): the name, and the node that gives it,
    the handler of an `except ... as`, a `global` or `nonlocal` statement,
    the function of a `*` or `**` parameter, a match pattern or a walrus."""

    name: str
    node: ast.AST


@dataclass(frozen=True)
class _CodeSigns:
    """What one walk of a cell reads for the check of code it can't read
    (`_Cell.signs`). `_runs_code` reads which of the hints run such code."""

    # The nodes that may name something that runs such code, by their last
    # name (`_CODE_HINTS`, and the C API's PyRun_ functions), each entry
    # looked up under a key one of those names or, on a module's own dict,
    # under a key that isn't a literal (`_module_entry_by_name`), each
    # function's `__code__` an assignment stores to (`_stores_code`), and
    # each import of the builtins module.
    hints: tuple[ast.AST, ...]
    # The cell's import statements, in walk order.
    imports: tuple[ast.Import | ast.ImportFrom, ...]
    # Whether the cell names exec, eval, compile or the builtins module at
    # all, by a bare name, a string or an import: a conversation with no such
    # cell can't reach those builtins (`_Reader.names_builtins`), since
    # `x.exec` reaches one only where `x` holds the builtins module.
    names_builtins: bool
    # Whether a string names a Python file, `.py` or `.pyc` at the end of a
    # word (`_PY_FILE`), as a write of a module the pad imports needs.
    names_a_py_file: bool
    # Whether a string names a startup hook, a file a Python process runs as
    # it starts (`_NAMES_A_STARTUP_HOOK`): a `.pth` file's import lines, or a
    # `sitecustomize` or `usercustomize` module.
    names_a_startup_hook: bool
    # Whether a string names a program that unpacks an archive or copies a
    # tree of files into a folder (`_NAMES_UNPACK`), as a write of a module
    # the pad imports may land there.
    names_unpack: bool
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
    # The names an assignment binds to a hint, as `rp = runpy.run_path` binds
    # `rp`, or to a lookup of a member by name (`_LOOKUP_CALL_NAMES`), as
    # `go = operator.methodcaller("run_path")` binds `go`.
    hint_aliases: frozenset[str]
    # Each name an assignment binds to a `functools.partial` of another name,
    # with that name, as `job` with `base` for `job = partial(base, cmd)`:
    # `job` holds a runner once `base` does (`_Reader.runner_aliases`).
    partial_heads: tuple[_NameValue, ...]
    # The names an `except ... as`, `global` or `nonlocal` statement, a `*`
    # or `**` parameter, a match pattern or a walrus gives, which
    # `_Cell.bindings` doesn't hold where they bind, and each place one
    # gives one (`_AsideSite`).
    bound_aside: frozenset[str]
    aside_sites: tuple[_AsideSite, ...]
    # Each `global` statement.
    globals_declared: tuple[ast.Global, ...]
    # The strings it holds that could be a name, as a lookup in the cell
    # namespace by name spells one, each with the constants that spell it.
    spelled: Mapping[str, tuple[ast.Constant, ...]]
    # The names a `del` statement unbinds.
    deleted: frozenset[str]
    # The names an import reaches each Python file a string names by
    # (`_module_names`).
    py_files: frozenset[str]
    # Each with and async with, in walk order.
    withs: tuple[ast.With | ast.AsyncWith, ...]


class _FirstSureKey(NamedTuple):
    """What `_Cell._first_sure_binding` reads once: a block, by its id, a
    name, and the ids of the cell's own calls that may set it (`setters`)
    and withs that may swallow its binding (`swallowers`)."""

    block: int
    name: str
    setters: frozenset[int]
    walrus: bool
    swallowers: frozenset[int]


class _Cell:
    """One parsed cell: its tree, parents, bindings and source lines, and
    what `_Reader` gathers from every cell. Nothing changes it once built, so
    one earlier cell serves every check that reads it (`_EarlierCells`)."""

    def __init__(
        self, tree: ast.Module, code: str, index: int, *, host_line: int | None = None, launched: bool = False,
        host: int | None = None, shared: bool = True, namespace: str | None = None,
        order: tuple[int, ...] | None = None, inherits: bool = False, surely_run: bool = True,
        parent_order: tuple[int, ...] | None = None,
    ) -> None:
        # Literal getattr and import lookups read as plain names, and the
        # plain names that now stand for a literal import (`_read_lookups_plainly`).
        self.looked_up = _read_lookups_plainly(tree)
        self.tree = tree
        self.index = index
        # For code a cell runs through exec() or eval(): the line of that call
        # in the cell, which findings report; the index of the conversation's
        # cell whose run runs it, which places it among them (`_Reader.position`);
        # whether what it binds lands in the cell namespace, as for an exec()
        # at a cell's top level given no namespace or `globals()`; and the
        # name of the namespace it fills otherwise, as `ns` for `exec(src, ns)`.
        # A conversation's own cell is its own host, and shares its names.
        # `order` places it among the cells in the order they run: a
        # conversation's cell by its index; literal code by the order of the
        # code that runs it, then the line and column of that call there,
        # then its place among the texts the call hands on.
        self.host_line = host_line
        self.host = index if host is None else host
        self.order = (index,) if order is None else order
        # For literal code: whether the call that runs it runs whenever the
        # code that holds the call does, as no if, loop or try holds it,
        # and each call that leads to it did too.
        self.surely_run = surely_run
        # For literal code a cell whose names don't land in the cell
        # namespace runs with no namespace of its own: it reads and binds
        # names in that cell's namespace (`_Reader.namespace_parents`), the
        # cell placed at `parent_order`.
        self.inherits = inherits
        self.parent_order = parent_order
        self.shared = shared
        self.namespace = namespace
        # For code a cell hands `python -c`, and the code that code runs: the
        # Python it runs in has its working folder on `sys.path`.
        self.launched = launched
        # The cell's code as written, and its lines.
        self.code = code
        self.lines = code.splitlines()
        self.parents: dict[ast.AST, ast.AST] = {}
        self._run_nodes: dict[int, tuple[ast.AST, ...]] = {}
        # What each def of the cell returns, by the def's id (`returns_of`).
        self._returns: dict[int, tuple[ast.expr, ...]] = {}
        # The loops around each binding's value, by the ids of the value and
        # its scope (`loops_around`), and the values each function's run
        # returns, by its id (`returns_run_by`).
        self._loops: dict[tuple[int, int], tuple[ast.AST, ...]] = {}
        self._run_returns: dict[int, tuple[ast.expr, ...]] = {}
        self._loads_within: dict[tuple[int, str], tuple[ast.Name, ...]] = {}
        # Every load inside each function the check reads, by its id (`loads_in`).
        self._loads_in: dict[int, tuple[ast.Name, ...]] = {}
        # The loads of a name outside a function, by the function's id and the
        # name (`loads_outside`).
        self._loads_outside: dict[tuple[int, str], tuple[ast.Name, ...]] = {}
        self._statements_at: dict[int, ast.stmt | None] = {}
        self._first_sure: dict[_FirstSureKey, int] = {}
        self._names_within: dict[int, frozenset[str]] = {}
        self._spread_args: dict[tuple[int, int], ast.Tuple] = {}
        # Every name the cell imports, by the name it binds; the functions and
        # classes it defines, by name, a method aside, since a bare name never
        # reaches one; every call, in walk order; the calls
        # to a name or method that pass a name, by the name passed; whether a
        # string in it attaches a database outside the process
        # (`_attaches_elsewhere`); and the strings that do so only when a SQL
        # runner gets them, such as `Attach 'postgres:...' AS pg`, which
        # `_Reader` follows.
        self.imports: dict[str, str] = {}
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.calls: list[ast.Call] = []
        # Every def and class, and every assignment, walrus, loop, with, def
        # and lambda, each in walk order, nested ones among them (`_defined_uses`).
        self.definitions: list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef] = []
        self.stores: list[ast.AST] = []
        # Every `|=`, which merges a mapping into what its target holds.
        self.merges: list[ast.AugAssign] = []
        self.passed: dict[str, list[ast.Call]] = {}
        self.attaches = False
        self.run_attaches: list[ast.Constant] = []
        # Every node of the tree in walk order, walked once for every pass
        # over the whole cell below.
        self.walked = tuple(ast.walk(tree))
        for node in self.walked:
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
                self.definitions.append(node)
                if not isinstance(self.parents.get(node), ast.ClassDef):
                    self.functions[node.name] = node
            elif isinstance(node, ast.ClassDef):
                self.definitions.append(node)
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
        for node in self.walked:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                self.loads.setdefault(node.id, []).append(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                self.stores.append(node)
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
                if isinstance(node.op, ast.BitOr):
                    self.merges.append(node)
                if isinstance(node.op, ast.Add) and isinstance(node.target, ast.Name):
                    self.augments.setdefault(node.target.id, []).append(node)
                if isinstance(node.op, (ast.Add, ast.BitOr, ast.Div)) and isinstance(node.target, ast.Name):
                    how = "join" if isinstance(node.op, ast.Div) else "add"
                    self.grows.setdefault(node.target.id, []).append(_Growth(site=node, value=node.value, how=how))
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                if not isinstance(node, ast.comprehension):
                    self.stores.append(node)
                self._bind(node.target, node.iter, at=node, how="element")
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                self.stores.append(node)
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                self._bind(node.optional_vars, node.context_expr, at=node, how="context")
            elif isinstance(node, ast.NamedExpr):
                self.stores.append(node)
                self._bind(node.target, node.value, at=node, how="value")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                self.stores.append(node)
                args = node.args
                for a in (*args.posonlyargs, *args.args, *args.kwonlyargs, *_spread_params(args)):
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
            # An unpacked tuple, nested ones included (`fig, (ax1, ax2) = ...`):
            # each name gets a part the check can't place.
            names = _stored_names(target)
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

    def _dead(self, node: ast.AST, *, functions: bool = False) -> bool:
        """Whether `node` sits in a branch that never runs, as under `if
        False:`; with `functions`, or in a function or lambda, which runs
        only when something calls it."""
        child, parent = node, self.parents.get(node)
        while parent is not None:
            if functions and isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) or _never_runs(
                parent=parent, child=child,
            ):
                return True
            child, parent = parent, self.parents.get(parent)
        return False

    def _under_branch(self, node: ast.AST, *, scope: ast.AST | None) -> bool:
        """Whether an if, loop, try or match in `scope` holds `node`, so it
        may not run when `scope` does, but for the body of an if whose test
        is a true constant, which always runs."""
        child, current = node, self.parents.get(node)
        while current is not None and current is not scope:
            if isinstance(current, _BRANCHES) and not (
                isinstance(current, ast.If) and isinstance(current.test, ast.Constant) and current.test.value
                and child in current.body
            ):
                return True
            child, current = current, self.parents.get(current)
        return False

    def runs_surely(self, node: ast.AST) -> bool:
        """Whether `node` runs whenever its scope runs (`_under_branch`)."""
        return not self._under_branch(node, scope=self.scope_of(node))

    def runs_to_its_end(self, node: ast.AST, *, swallowers: frozenset[int]) -> bool:
        """Whether `node` surely runs to its end whenever its scope runs: it
        runs surely (`runs_surely`), and no with around it in its scope may
        swallow an error and stop its body short (`_may_swallow`, by
        `swallowers`), so a binding in it surely binds."""
        if not self.runs_surely(node):
            return False
        scope = self.scope_of(node)
        current = self.parents.get(node)
        while current is not None and current is not scope:
            if isinstance(current, (ast.With, ast.AsyncWith)) and _may_swallow(current, swallowers):
                return False
            current = self.parents.get(current)
        return True

    def binds_loop_target(self, node: ast.AST) -> bool:
        """Whether the statement holding `node`, the iterable of a loop or
        the context a with enters, binds its target once it ran: a with
        does, a loop only where it surely starts its body
        (`_runs_body_surely`), as a loop over no item binds nothing."""
        stmt = self.statement_of(node)
        return not isinstance(stmt, (ast.For, ast.AsyncFor)) or _runs_body_surely(stmt)

    @cached_property
    def swallowing_aliases(self) -> frozenset[str]:
        """The names the cell imports a context that may swallow an error
        its body raises (`_SWALLOWING_CONTEXTS`) as, as `quietly` for `from
        contextlib import suppress as quietly` (`_swallowing_withs`)."""
        return frozenset(
            a.asname for stmt in self.signs.imports if isinstance(stmt, ast.ImportFrom) for a in stmt.names
            if a.asname and a.name in _SWALLOWING_CONTEXTS
        )

    @cached_property
    def swallowing_classes(self) -> frozenset[str]:
        """The classes the cell defines with their own `__exit__` or
        `__aexit__`, which may return true and so swallow an error a with's
        body raises (`_swallowing_withs`)."""
        return frozenset(
            d.name for d in self.definitions if isinstance(d, ast.ClassDef) and any(
                isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name in ("__exit__", "__aexit__")
                for s in d.body
            )
        )

    @cached_property
    def node_ids(self) -> frozenset[int]:
        """The ids of the cell's calls and withs, the nodes a set of setter
        calls (`_Reader.global_setters`) or swallowing withs
        (`_Reader.swallowers`) names: the part of either set that is the
        cell's own keys what the cell reads of a block
        (`_first_sure_binding`), so another cell's check reads it again only
        where that part differs."""
        return frozenset(id(node) for node in (*self.calls, *self.signs.withs))

    @cached_property
    def value_heads(self) -> tuple[_NameValue, ...]:
        """Each name the cell binds to a value by an assignment, in any
        scope, with the last name the value is spelled with or calls
        (`_last_name`), as `suppress` for `cm = contextlib.suppress(E)`:
        what `_swallowing_withs` follows from a swallowing context to a name."""
        out: list[_NameValue] = []
        for name, found in self.bindings.items():
            for b in found:
                head = b.value.func if isinstance(b.value, ast.Call) else b.value
                if b.how == "value" and isinstance(head, (ast.Name, ast.Attribute)):
                    out.append(_NameValue(name, head))
        return tuple(out)

    @cached_property
    def names_bound(self) -> frozenset[str]:
        """Every name the cell's code binds, in any scope: by an assignment,
        parameter, loop, with, comprehension or walrus (`bindings`), a def
        or class, an import, or a place `bindings` leaves out
        (`_CodeSigns.bound_aside`)."""
        return frozenset({
            *self.bindings, *(d.name for d in self.definitions),
            *(name for stmt in self.signs.imports for name in _import_names(stmt)), *self.signs.bound_aside,
        })

    def runs_before(self, node: ast.AST, use: ast.AST) -> bool:
        """Whether `node` surely ran whenever `use`, code of this cell in the
        same scope, runs: the statement that holds `node` sits before `use`
        in the statement list that holds `use` or a statement around it, or
        is the with-statement or loop whose body holds `use`."""
        stmt: ast.AST | None = node
        while stmt is not None and not isinstance(stmt, ast.stmt):
            stmt = self.parents.get(stmt)
        if stmt is None or self.scope_of(stmt) is not self.scope_of(use):
            return False
        current: ast.AST | None = use
        while current is not None and current is not stmt:
            parent = self.parents.get(current)
            if isinstance(current, ast.stmt) and parent is not None:
                for block in (getattr(parent, field, None) for field in ("body", "orelse", "finalbody")):
                    if isinstance(block, list) and current in block and stmt in block:
                        return block.index(stmt) < block.index(current)
            current = parent
        return isinstance(stmt, (ast.With, ast.AsyncWith, ast.For, ast.AsyncFor)) and any(
            use is n or _inside(use, n, self) for n in stmt.body
        )

    def binds_whole(self, name: str, *, setters: frozenset[int], swallowers: frozenset[int]) -> bool:
        """Whether the cell's code, run to its end, surely binds `name` at
        its top level (`_first_sure_binding`)."""
        body = self.tree.body
        walrus = name in self.signs.bound_aside
        return self._first_sure_binding(body, name, setters=setters, walrus=walrus, swallowers=swallowers) < len(body)

    def bound_on_the_way(
        self, node: ast.AST, name: str, *, setters: frozenset[int], swallowers: frozenset[int],
        within: ast.AST | None = None,
    ) -> bool:
        """Whether the code of the cell that runs before `node`, a read at
        module level, surely binds `name` on every path to it: a walrus
        before it in its own statement, then, in each block that holds it,
        the statements before it there (`_binds_on_every_path`), and what
        the statement that holds that block binds before the block runs: an
        if's or while's test, a loop's target or iterable, a with's
        targets or contexts, a handler's `as` name, a case's pattern or
        guard, or a try's whole body before its else. A with whose context
        may swallow an error (`_may_swallow`, by `swallowers`) binds nothing
        its body binds. With `within`, only the blocks inside that scope
        count."""
        walrus = name in self.signs.bound_aside
        own = self.statement_of(node)
        if walrus and any(
            isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name) and n.target.id == name
            and _follows(node=node, value=n) for n in ast.walk(own)
        ):
            return True
        child: ast.AST = node
        current = self.parents.get(node)
        while current is not None and child is not within:
            if isinstance(current, ast.match_case) and child is current.guard:
                if _captures(current.pattern, name):
                    return True
            for field in ("body", "orelse", "finalbody"):
                block = getattr(current, field, None)
                if not (isinstance(block, list) and child in block):
                    continue
                first = self._first_sure_binding(block, name, setters=setters, walrus=walrus, swallowers=swallowers)
                if first < block.index(child):
                    return True
                if _binds_before(current, field, name, setters=setters, walrus=walrus, swallowers=swallowers):
                    return True
                break
            child, current = current, self.parents.get(current)
        return False

    def _first_sure_binding(
        self, block: list[ast.stmt], name: str, *, setters: frozenset[int], walrus: bool,
        swallowers: frozenset[int],
    ) -> int:
        """The index of the first statement of `block`, a statement list of
        the cell, that surely binds `name` when it runs to its end
        (`_binds_on_every_path`), a with read by `swallowers`, or the
        block's length where none does. Read once per block, name and
        swallowers, past each statement that can't bind it
        (`_bound_under`), unless a call in `setters` may; a name the cell
        can't bind at all, with no such call in it, needs no read."""
        own = self.node_ids
        mine = setters & own
        if not mine and name not in self.signs.bound_aside and name not in self._names_bound_under:
            return len(block)
        key = _FirstSureKey(block=id(block), name=name, setters=mine, walrus=walrus, swallowers=swallowers & own)
        if key not in self._first_sure:
            self._first_sure[key] = next((
                i for i, stmt in enumerate(block)
                if (setters or name in self.signs.bound_aside or name in self._bound_under.get(id(stmt), ()))
                and _binds_on_every_path([stmt], name, setters=setters, walrus=walrus, swallowers=swallowers)
            ), len(block))
        return self._first_sure[key]

    @cached_property
    def _names_bound_under(self) -> frozenset[str]:
        """Every name a statement of the cell may bind (`_bound_under`)."""
        return frozenset().union(*self._bound_under.values())

    @cached_property
    def _bound_under(self) -> dict[int, set[str]]:
        """The names each statement of the cell may bind, in any scope, by
        the statement's id, read from the lists the cell keeps (`bindings`,
        `definitions`, `signs`, `stores`) rather than by walking it again:
        what each binding, def, class, import or store into the cell's own
        namespace by a literal key binds goes to every statement that holds
        it. A name `_CodeSigns.bound_aside` holds may bind in places these
        lists leave out, as a match capture does."""
        out: dict[int, set[str]] = {}

        def mark(node: ast.AST | None, name: str) -> None:
            while node is not None:
                if isinstance(node, ast.stmt):
                    out.setdefault(id(node), set()).add(name)
                node = self.parents.get(node)

        for name, found in self.bindings.items():
            for b in found:
                mark(b.value, name)
        for d in self.definitions:
            mark(d, d.name)
        for stmt in self.signs.imports:
            for name in _import_names(stmt):
                mark(stmt, name)
        for node in self.stores:
            for target in node.targets if isinstance(node, ast.Assign) else ():
                key = _namespace_key(target)
                if key is not None:
                    mark(node, key)
        return out

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

    def binds_in_scope(self, node: ast.AST, name: str, *, swallowers: frozenset[int]) -> bool:
        """Whether a def, lambda, class or comprehension around `node` binds
        `name` in its own scope, so a read of it there never reaches the
        module's names: an assignment, parameter, loop, with, comprehension
        or walrus (`bindings`), a def or class, an import, or a place
        `bindings` leaves out (`_aside_scopes`), so `_Reader.binds` reads
        the name there. A scope that declares the name `global` or
        `nonlocal` binds the outer name, not its own, and a class body's
        binding counts only where it surely came before the read, since the
        body reads the module's name where its own isn't bound yet, a with
        read by `swallowers` (`bound_on_the_way`)."""
        scopes = [scope for scope in self.scopes_around(node) if not self.declares_outer(scope, name)]
        if not scopes:
            return False
        own = [scope for scope in scopes if not isinstance(scope, ast.ClassDef)]
        body = scopes[0] if isinstance(scopes[0], ast.ClassDef) else None
        return (
            any(b.scope in own for b in self.bindings.get(name, ()))
            or any(d.name == name and self.scope_of(d) in own for d in self.definitions)
            or any(name in _import_names(stmt) and self.scope_of(stmt) in own for stmt in self.signs.imports)
            or name in self.signs.bound_aside and any(
                scope in own for site in self.signs.aside_sites if site.name == name
                and not isinstance(site.node, (ast.Global, ast.Nonlocal))
                for scope in self._aside_scopes(site)
            )
            or body is not None and self.bound_on_the_way(
                node, name, setters=frozenset(), swallowers=swallowers, within=body,
            )
        )

    def declares_outer(self, scope: ast.AST, name: str) -> bool:
        """Whether `scope`, a def of the cell, declares `name` `global` or
        `nonlocal`, so its bindings of the name bind the outer one."""
        return name in self.signs.bound_aside and any(
            site.name == name and isinstance(site.node, (ast.Global, ast.Nonlocal))
            and self.scope_of(site.node) is scope for site in self.signs.aside_sites
        )

    def _aside_scopes(self, site: _AsideSite) -> list[ast.AST]:
        """The scopes `site` binds its name in: a function's own for its `*`
        or `**` parameter, else the scope around it, and, for a walrus in a
        comprehension, each scope around that up to the first that isn't
        one, where Python binds it."""
        if isinstance(site.node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return [site.node]
        out: list[ast.AST] = []
        scope = self.scope_of(site.node)
        while scope is not None:
            out.append(scope)
            if not (isinstance(site.node, ast.NamedExpr) and isinstance(scope, _COMPREHENSIONS)):
                break
            scope = self.scope_of(scope)
        return out

    @cached_property
    def module_bound(self) -> frozenset[str]:
        """The names the cell binds at its top level, outside any function,
        class or comprehension: by an assignment, parameter, loop, with,
        comprehension or walrus (`bindings`), a def or class, or an import."""
        return frozenset({
            *(name for name, found in self.bindings.items() if any(b.scope is None for b in found)),
            *(d.name for d in self.definitions if self.scope_of(d) is None),
            *(name for stmt in self.signs.imports if self.scope_of(stmt) is None for name in _import_names(stmt)),
        })

    @cached_property
    def compiles(self) -> bool:
        """Whether Python compiles the cell's code as the pad's boot does
        before it runs any of it. Code can parse and still not compile, as
        `await`, `return` or `break` at the top level, `nonlocal` there, a
        duplicate parameter or a name assigned before its `global` do: such
        a cell runs none of its code. Quiet, as `_parse` is."""
        _quiet_parses()
        try:
            compile(heal_surrogate_source(self.code), _CHECK_FILENAME, "exec", dont_inherit=True)
        except (SyntaxError, ValueError, RecursionError):
            return False
        return True

    @cached_property
    def leading_definitions(self) -> tuple[ast.stmt, ...]:
        """The defs and classes at the top of the cell, before its first
        statement of any other kind, that can't raise as they run
        (`_cannot_raise`). A cell that compiled and stopped on an error at a
        line the host can't tell ran them; an import, a decorator, a
        default, a class's base, or any other statement may have raised it."""
        out: list[ast.stmt] = []
        for stmt in self.tree.body:
            if not (isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and _cannot_raise(stmt)):
                break
            out.append(stmt)
        return tuple(out)

    def statement_of(self, node: ast.AST) -> ast.AST:
        """The statement of the cell that holds `node`, or `node` itself
        when no statement does."""
        current: ast.AST | None = node
        while current is not None and not isinstance(current, ast.stmt):
            current = self.parents.get(current)
        return node if current is None else current

    def statement_at(self, line: int) -> ast.stmt | None:
        """The statement a traceback that names `line` of the cell's own
        frame stopped in: the first top-level statement whose lines hold
        it, then, in each block that statement holds, the first statement
        whose lines hold it in turn, down to a def or class, whose body
        runs in a frame of its own, or a statement with no such block, as
        the test of an if that raised. None when no statement holds the line."""
        if line in self._statements_at:
            return self._statements_at[line]
        found: ast.stmt | None = None
        block: list[ast.stmt] = self.tree.body
        while block:
            stmt = next((s for s in block if s.lineno <= line <= (s.end_lineno or s.lineno)), None)
            if stmt is None:
                break
            found = stmt
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                break
            block = [*getattr(stmt, "body", ()), *getattr(stmt, "orelse", ()), *getattr(stmt, "finalbody", ())]
            for part in (*getattr(stmt, "handlers", ()), *getattr(stmt, "cases", ())):
                block += part.body
        self._statements_at[line] = found
        return found

    @cached_property
    def names_a_routine(self) -> bool:
        """Whether the parsed code names `callproc` or `callfunc`, as an
        attribute or a string, however the source spells it."""
        return any(
            isinstance(node, ast.Attribute) and node.attr in _ROUTINE_CALLS
            or isinstance(node, ast.Constant) and node.value in _ROUTINE_CALLS
            for node in self.walked
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

    @cached_property
    def _loads_by_line(self) -> list[tuple[int, ast.AST]]:
        """Every load of a name in the cell, by its line."""
        return sorted(((n.lineno, n) for found in self.loads.values() for n in found), key=lambda p: p[0])

    def _within(self, node: ast.AST, found: list[tuple[int, ast.AST]]) -> list[ast.AST]:
        """The nodes of `found`, nodes of this cell by line, that are or sit
        inside `node`, looked up by its lines rather than by walking it."""
        first, last = getattr(node, "lineno", 0), getattr(node, "end_lineno", None) or getattr(node, "lineno", 0)
        lo = bisect.bisect_left(found, first, key=lambda p: p[0])
        hi = bisect.bisect_right(found, last, key=lambda p: p[0])
        return [n for _, n in found[lo:hi] if n is node or _inside(n, node, self)]

    def spread_args(self, call: ast.Call, start: int) -> ast.Tuple | None:
        """The tuple a function's `*args` parameter gets from `call`, a call
        in this cell: its positional arguments from `start` on, made once, so
        every read of it reads the same node. None for a call the cell
        doesn't hold."""
        if call not in self.parents:
            return None
        made = ast.copy_location(ast.Tuple(elts=call.args[start:], ctx=ast.Load()), call)
        return self._spread_args.setdefault((id(call), start), made)

    def names_within(self, value: ast.AST) -> frozenset[str]:
        """The names `value`, code of this cell, loads, found among the
        cell's loads by line (`_within`) rather than by walking `value`,
        which may be a large literal; read once per value."""
        found = self._names_within.get(id(value))
        if found is None:
            found = self._names_within[id(value)] = frozenset(
                n.id for n in self._within(value, self._loads_by_line) if isinstance(n, ast.Name)
            )
        return found

    def loads_within(self, value: ast.AST, name: str) -> tuple[ast.Name, ...]:
        """The loads of `name` inside `value`, code of this cell, found among
        the cell's loads of the name (`loads`) rather than by walking
        `value`, which may be a large literal; read once per value and name,
        as a name rebuilt many times asks it of each of its values again."""
        key = (id(value), name)
        found = self._loads_within.get(key)
        if found is None:
            first = getattr(value, "lineno", 0)
            last = getattr(value, "end_lineno", None) or first
            found = self._loads_within[key] = tuple(
                n for n in self.loads.get(name, ())
                if first <= n.lineno <= last and (n is value or _inside(n, value, self))
            )
        return found

    def loads_outside(self, func: ast.AST, name: str) -> tuple[ast.Name, ...]:
        """The loads of `name` in the cell outside `func`, a def or lambda
        of it, as those that run or hand on the function `name` holds
        (`_runs_from`), read once per function and name: an earlier cell
        serves every check that reads it."""
        key = (id(func), name)
        found = self._loads_outside.get(key)
        if found is None:
            found = self._loads_outside[key] = tuple(
                load for load in self.loads.get(name, ()) if not _inside(load, func, self)
            )
        return found

    def loads_in(self, root: ast.AST) -> tuple[ast.Name, ...]:
        """Every load of a name inside `root`, code of this cell, in the
        order `loads` holds them, read once per root, as each function a
        read reaches asks it of a cell that may hold many."""
        found = self._loads_in.get(id(root))
        if found is None:
            first = getattr(root, "lineno", 0)
            last = getattr(root, "end_lineno", None) or first
            found = self._loads_in[id(root)] = tuple(
                n for uses in self.loads.values() for n in uses
                if first <= n.lineno <= last and _inside(n, root, self)
            )
        return found

    def returns_of(self, func: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.expr, ...]:
        """The value of each return statement in `func`, a def of the cell,
        not in a function inside it, read once per def: an earlier cell
        serves every check that reads it."""
        found = self._returns.get(id(func))
        if found is None:
            found = self._returns[id(func)] = tuple(
                n.value for n in ast.walk(func)
                if isinstance(n, ast.Return) and n.value is not None and self.scope_of(n) is func
            )
        return found

    def returns_run_by(self, func: Function) -> tuple[ast.expr, ...]:
        """The value of each return statement that runs when `func`, a def or
        lambda of the cell, runs (`nodes_run_by`), read once per function."""
        found = self._run_returns.get(id(func))
        if found is None:
            found = self._run_returns[id(func)] = tuple(
                n.value for n in self.nodes_run_by(func) if isinstance(n, ast.Return) and n.value is not None
            )
        return found

    def loops_around(self, value: ast.AST, *, scope: ast.AST | None) -> tuple[ast.AST, ...]:
        """The loops of the cell around `value` inside `scope`, innermost
        first, found once per value."""
        key = (id(value), id(scope))
        found = self._loops.get(key)
        if found is None:
            loops: list[ast.AST] = []
            current = self.parents.get(value)
            while current is not None and current is not scope:
                if isinstance(current, (ast.For, ast.AsyncFor, ast.While)):
                    loops.append(current)
                current = self.parents.get(current)
            found = self._loops[key] = tuple(loops)
        return found

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
        a write of a file where the cell names a startup hook."""
        out: list[ast.Call] = []
        for call in self.calls:
            name = _member_name(call.func)
            if (isinstance(call.func, ast.Call) or name in _UNREAD_CALL_NAMES or name.startswith(_PYAPI_PREFIXES)
                    or self.signs.names_a_startup_hook and name in _FILE_WRITE_NAMES):
                out.append(call)
        return tuple(out)

    @cached_property
    def module_calls(self) -> _PlacedCalls:
        """The cell's calls outside every def and lambda
        (`_outer_function`), which run as the cell runs, in the order they
        sit in its code (`_PlacedCalls`), read once per cell: an earlier
        cell serves every check that reads it."""
        calls = sorted((c for c in self.calls if _outer_function(c, self) is None),
                       key=lambda c: (c.lineno, c.col_offset))
        return _PlacedCalls(places=tuple((c.lineno, c.col_offset) for c in calls), calls=tuple(calls))

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
        read once per cell: each import; each assignment, walrus, loop and
        with; and each call that hands on a namespace or updates one, by a
        second argument, a `globals` or `locals` keyword, or `.update(...)`.
        `_build_defined` and `_poisons_namespace` read no other. Read from
        the lists the cell keeps (`signs`, `stores`, `calls`) rather than by
        walking it again, which a cell of large literals pays for."""
        candidates = (
            *self.signs.imports,
            *(node for node in self.stores if isinstance(node, (*_VALUE_STORES, *_HEAD_STORES))),
            *(node for node in self.calls
              if len(node.args) > 1 or any(k.arg in ("globals", "locals") for k in node.keywords)
              or isinstance(node.func, ast.Attribute) and node.func.attr == "update"),
        )
        return tuple(node for node in candidates if self._runs_with_cell(node))

    def _runs_with_cell(self, node: ast.AST) -> bool:
        """Whether `node` runs when the cell runs (`_nodes_that_run`): no
        function or lambda holds it, and no branch that never runs (`_dead`)."""
        return not self._dead(node, functions=True)

    @cached_property
    def fill_reads(self) -> tuple[_FillReads, ...]:
        """What each of `fills_that_run` reads, by the same place
        (`_FillReads`), for `_build_defined`."""
        # The calls by line, so each fill reads only those on its own lines
        # rather than walking its code again, as it does the loads.
        calls = sorted(((c.lineno, c) for c in self.calls), key=lambda p: p[0])
        out: list[_FillReads] = []
        for node in self.fills_that_run:
            # A loop or with fills its names from its head alone, not its body.
            parts = _store_heads(node)
            names = frozenset(name for part in parts for name in self.names_within(part))
            funcs = [c.func for part in parts for c in self._within(part, calls) if isinstance(c, ast.Call)]
            plain = not isinstance(node, (ast.Import, ast.ImportFrom)) and all(
                isinstance(f, (ast.Attribute, ast.Call)) for f in funcs
            )
            methods = frozenset(f.attr for f in funcs if isinstance(f, ast.Attribute)) if plain else None
            out.append(_FillReads(names=names, methods=methods))
        return tuple(out)

    @cached_property
    def module_stores(self) -> tuple[_ModuleStoreSite, ...]:
        """Each store of a module in a mapping that may be `sys.modules`
        (`_may_hold_modules`), in any scope (`_ModuleStoreSite`): `m[k] = v`,
        `m |= {...}`, `m.setdefault(k, v)`, `m.__setitem__(k, v)` and
        `m.update(...)` of a dict, keywords or a mapping it can't read, and
        the same handed the mapping first, as `operator.setitem(m, k, v)`,
        `setitem(m, k, v)` and `dict.__setitem__(m, k, v)` are.
        `_stored_modules` reads which mapping is `sys.modules`. Read from
        the lists the cell keeps (`stores`, `merges`, `calls`)."""
        out: list[_ModuleStoreSite] = []
        for node in self.stores:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                for t in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    if isinstance(t, ast.Subscript) and _may_hold_modules(t.value):
                        out.append(_ModuleStoreSite(holder=t.value, key=t.slice, value=node.value))
        for merge in self.merges:
            if _may_hold_modules(merge.target):
                out += _merged_modules(merge.target, merge.value)
        for call in self.calls:
            func = call.func
            if isinstance(func, ast.Name) and func.id in ("setitem", "__setitem__"):
                # `setitem(m, k, v)` once `operator`'s is imported by its name.
                if len(call.args) >= 3 and _may_hold_modules(call.args[0]):
                    out.append(_ModuleStoreSite(holder=call.args[0], key=call.args[1], value=call.args[2]))
                continue
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in _MAPPING_STORES:
                continue
            if _last_name(func.value) in ("dict", "operator"):
                if not (call.args and _may_hold_modules(call.args[0])):
                    continue
                holder, args = call.args[0], list(call.args[1:])
            elif _may_hold_modules(func.value):
                holder, args = func.value, list(call.args)
            else:
                continue
            if func.attr != "update" and len(args) >= 2:
                out.append(_ModuleStoreSite(holder=holder, key=args[0], value=args[1]))
            elif func.attr == "update":
                for arg in args:
                    out += _merged_modules(holder, arg)
                out += [_ModuleStoreSite(holder=holder, key=None if k.arg is None else ast.Constant(value=k.arg),
                                         value=k.value) for k in call.keywords]
        return tuple(out)

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
        names_builtins = names_a_py_file = names_a_startup_hook = names_dash_c = names_unpack = False
        path_changes: list[ast.AST] = []
        list_changes: list[_ListChange] = []
        path_aliases: set[str] = set()
        hint_aliases: set[str] = set()
        partial_heads: list[_NameValue] = []
        aside: list[_AsideSite] = []
        declared: list[ast.Global] = []
        spelled: dict[str, list[ast.Constant]] = {}
        deleted: set[str] = set()
        py_files: set[str] = set()
        withs: list[ast.With | ast.AsyncWith] = []
        for node in self.walked:
            if isinstance(node, (ast.Name, ast.Attribute)):
                if _is_hint(_last_name(node)) or _stores_code(node):
                    hints.append(node)
                # A member `.exec` reaches the builtin where its holder is the
                # builtins module, which a name may hold with no literal
                # "builtins" in the cell, as `import_module("built" + "ins")`,
                # and so may a member that holds the module, as `__self__`.
                names_builtins = names_builtins or (
                    node.id in _EXEC_CALLS or node.id in _BUILTINS_MODULE_NAMES
                    if isinstance(node, ast.Name) else node.attr in _EXEC_CALLS or node.attr in _BUILTINS_HOLDER_MEMBERS
                )
            elif isinstance(node, ast.Subscript):
                key = node.slice.value if isinstance(node.slice, ast.Constant) else None
                if key in _CODE_HINTS or _module_entry_by_name(node):
                    hints.append(node)
                    names_builtins = names_builtins or key in _EXEC_CALLS or key in _BUILTINS_MODULE_NAMES
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names_builtins = names_builtins or node.value in _EXEC_CALLS or node.value in _BUILTINS_MODULE_NAMES
                names_a_py_file = names_a_py_file or bool(_PY_FILE.search(node.value))
                names_a_startup_hook = names_a_startup_hook or bool(_NAMES_A_STARTUP_HOOK.search(node.value))
                names_unpack = names_unpack or bool(_NAMES_UNPACK.search(node.value))
                names_dash_c = names_dash_c or "-c" in node.value
                if _PY_FILE.search(node.value):
                    py_files.update(n for word in node.value.split() for n in _module_names(word.strip("'\";,()")))
                if node.value.isidentifier():
                    spelled.setdefault(node.value, []).append(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                if "builtins" in modules:
                    hints.append(node)
                    names_builtins = True
                if isinstance(node, ast.ImportFrom) and node.module == "sys":
                    path_aliases.update(a.asname or a.name for a in node.names if a.name == "path")
            elif isinstance(node, ast.ExceptHandler) and node.name:
                aside.append(_AsideSite(name=node.name, node=node))
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                aside += [_AsideSite(name=name, node=node) for name in node.names]
                if isinstance(node, ast.Global):
                    declared.append(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                aside += [_AsideSite(name=a.arg, node=node) for a in (node.args.vararg, node.args.kwarg) if a is not None]
            elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
                aside.append(_AsideSite(name=node.name, node=node))
            elif isinstance(node, ast.MatchMapping) and node.rest:
                aside.append(_AsideSite(name=node.rest, node=node))
            elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                # A walrus in a comprehension binds in the scope around it.
                aside.append(_AsideSite(name=node.target.id, node=node))
            elif isinstance(node, ast.Delete):
                deleted.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                withs.append(node)
            if isinstance(node, ast.Call):
                func = node.func
                if _sets_code(node):
                    hints.append(node)
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
                wrapped = _partial_head(value)
                if isinstance(node, ast.Assign) and isinstance(value, (ast.Name, ast.Attribute)):
                    names = [t.id for t in targets if isinstance(t, ast.Name)]
                    last = _last_name(value)
                    if last == "path":
                        path_aliases.update(names)
                    if _is_hint(last):
                        hint_aliases.update(names)
                elif isinstance(node, ast.Assign) and (
                    isinstance(value, ast.Subscript) and _module_entry_by_name(value)
                    or isinstance(value, ast.Call) and _last_name(value.func) in _LOOKUP_CALL_NAMES
                    or wrapped is not None and _is_hint(_last_name(wrapped))
                ):
                    # `f = vars(runpy)[k]`, `go = operator.methodcaller(k)`, `f = getattr(m, k)`,
                    # `job = partial(subprocess.run, cmd)`.
                    hint_aliases.update(t.id for t in targets if isinstance(t, ast.Name))
                elif isinstance(node, ast.Assign) and isinstance(wrapped, ast.Name):
                    partial_heads += [_NameValue(t.id, wrapped) for t in targets if isinstance(t, ast.Name)]
        return _CodeSigns(
            hints=tuple(hints), imports=tuple(imports), names_builtins=names_builtins,
            names_a_py_file=names_a_py_file, names_a_startup_hook=names_a_startup_hook, names_unpack=names_unpack,
            names_dash_c=names_dash_c, path_changes=tuple(path_changes), list_changes=tuple(list_changes),
            path_aliases=frozenset(path_aliases), hint_aliases=frozenset(hint_aliases),
            partial_heads=tuple(partial_heads), bound_aside=frozenset(site.name for site in aside),
            aside_sites=tuple(aside),
            globals_declared=tuple(declared), spelled={k: tuple(v) for k, v in spelled.items()},
            deleted=frozenset(deleted), py_files=frozenset(py_files), withs=tuple(withs),
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


def _stores_code(node: ast.AST) -> bool:
    """Whether `node` is an assignment's target that gives a function a
    code object as its own, `f.__code__`: calling `f` then runs that code."""
    return isinstance(node, ast.Attribute) and node.attr == "__code__" and isinstance(node.ctx, ast.Store)


def _sets_code(node: ast.AST) -> bool:
    """Whether `node` is a call `setattr(f, "__code__", v)`, which gives `f`
    the code object `v` as `f.__code__ = v` does (`_stores_code`)."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setattr"
            and len(node.args) == 3 and not node.keywords and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "__code__")


def _swaps_in_unread_code(target: ast.AST, cell: _Cell) -> bool:
    """Whether `target`, code of `cell`, gives a function code the check
    can't read (`_stores_code`): any code object but the `__code__` of a
    function the cell defines, which the check reads with the cell, as
    `g.__code__ = f.__code__` gives (`_unread_code_object`)."""
    if not _stores_code(target):
        return False
    store = cell.parents.get(target)
    value = store.value if isinstance(store, (ast.Assign, ast.AnnAssign)) else None
    return _unread_code_object(value, cell)


def _unread_code_object(value: ast.AST | None, cell: _Cell) -> bool:
    """Whether `value`, a code object code of `cell` gives a function, holds
    code the check can't read: any but the `__code__` of a function the
    cell defines."""
    return not (isinstance(value, ast.Attribute) and value.attr == "__code__"
                and isinstance(value.value, ast.Name) and value.value.id in cell.functions)


# The methods, and `operator`'s functions, that store an entry in a mapping.
_MAPPING_STORES = frozenset({"setdefault", "__setitem__", "setitem", "update"})


def _may_hold_modules(node: ast.AST) -> bool:
    """Whether `node`, the mapping a store goes into, may be `sys.modules`:
    one spelled `.modules`, or as the entry `["modules"]` of a mapping, as
    `vars(sys)["modules"]` is, or a bare name, which may be bound to it."""
    return isinstance(node, ast.Name) or _last_name(node) == "modules" or (
        isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == "modules"
    )


def _merged_modules(holder: ast.AST, merged: ast.AST) -> list[_ModuleStoreSite]:
    """The stores `merged`, a mapping merged into `holder`, makes: one per
    entry of a dict, or one under a key the check can't read."""
    if isinstance(merged, ast.Dict):
        return [_ModuleStoreSite(holder=holder, key=k, value=v) for k, v in zip(merged.keys, merged.values)]
    return [_ModuleStoreSite(holder=holder, key=None, value=merged)]


def _module_entry_by_name(node: ast.Subscript) -> bool:
    """Whether `node` is spelled as an entry of a module's own dict under a
    key that isn't a literal: of `vars(m)`, `m.__dict__` or `sys.modules`,
    which `_Reader.module_member` reads through a name bound to literal
    texts, as `vars(runpy)[k]` with `k = "run_path"`."""
    if isinstance(node.slice, ast.Constant):
        return False
    holder = node.value
    return (isinstance(holder, ast.Call) and _last_name(holder.func) == "vars" and len(holder.args) == 1
            or isinstance(holder, ast.Attribute) and holder.attr in ("__dict__", "modules"))


def _cannot_raise(stmt: ast.stmt) -> bool:
    """Whether running `stmt` runs no code that may raise: a def with no
    decorator, default, annotation or type parameter, each of which runs
    as the def does; a class with none of those, no base or keyword, and a
    body of such defs, `pass`, constants and names bound to them; `pass`;
    or a constant, as a docstring is."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = stmt.args
        params = (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg)
        return not (stmt.decorator_list or args.defaults or any(d is not None for d in args.kw_defaults)
                    or stmt.returns or getattr(stmt, "type_params", None)
                    or any(p is not None and p.annotation is not None for p in params))
    if isinstance(stmt, ast.ClassDef):
        return not (stmt.decorator_list or stmt.bases or stmt.keywords or getattr(stmt, "type_params", None)) and all(
            _cannot_raise(s) for s in stmt.body
        )
    if isinstance(stmt, ast.Assign):
        return isinstance(stmt.value, ast.Constant) and all(isinstance(t, ast.Name) for t in stmt.targets)
    return isinstance(stmt, ast.Pass) or isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)


# The context managers that may swallow an error raised in their body, by the
# last name they are called by, so a with of one may stop its body short and
# run on after it: contextlib's `suppress`, a test's `raises` and its kin,
# and an exit stack, which may enter one.
_SWALLOWING_CONTEXTS = frozenset({
    "suppress", "raises", "assertRaises", "assertRaisesRegex", "assertWarns", "assertWarnsRegex", "ExitStack",
    "AsyncExitStack",
})


def _binds_on_every_path(
    block: Sequence[ast.stmt], name: str, *, setters: AbstractSet[int] = frozenset(), walrus: bool = False,
    swallowers: AbstractSet[int],
) -> bool:
    """Whether running `block`, a list of statements, to its end surely
    binds `name` in its scope: one of them assigns, imports, defines or
    declares it as a class, or stores it as an entry of `globals()` or
    `vars()`, or is a with one of whose targets is it or whose body does,
    unless the with may swallow an error its body raises (`_may_swallow`),
    which may stop its body short, an if each arm of which that may run
    does (`_if_binds`), a loop that surely starts its body
    (`_runs_body_surely`) whose target is it or whose body does before its
    first break or continue (`_before_a_jump`), a match each of whose
    cases does by its pattern or its body and whose last case matches
    anything, or a try whose body or else does and each of whose handlers
    does too or ends in a raise, or whose finally does. A call in
    `setters`, a call of a function that sets it as a global on every path
    (`_Reader.global_setters`), binds it too, and, with `walrus`, so does a
    walrus the statement surely runs (`_walrus_runs`). Any other loop may
    run no time. Each block is read once, and an elif chain arm by arm,
    with no deep recursion."""
    def every(stmts: Sequence[ast.stmt]) -> bool:
        return _binds_on_every_path(stmts, name, setters=setters, walrus=walrus, swallowers=swallowers)

    for stmt in block:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and stmt.name == name:
            return True
        if isinstance(stmt, (ast.Import, ast.ImportFrom)) and name in _import_names(stmt):
            return True
        if isinstance(stmt, ast.Assign) and any(_assigns(t, name) for t in stmt.targets):
            return True
        if isinstance(stmt, ast.AnnAssign) and stmt.value is not None and _assigns(stmt.target, name):
            return True
        if isinstance(stmt, (ast.With, ast.AsyncWith)) and (
            any(item.optional_vars is not None and name in _stored_names(item.optional_vars) for item in stmt.items)
            or not _may_swallow(stmt, swallowers) and every(stmt.body)
        ):
            return True
        if isinstance(stmt, ast.If) and _if_binds(stmt, every, tested=lambda s: walrus and _walrus_runs(s, name)):
            return True
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)) and _runs_body_surely(stmt) and (
            isinstance(stmt, (ast.For, ast.AsyncFor)) and name in _stored_names(stmt.target)
            or every(_before_a_jump(stmt.body))
        ):
            return True
        if isinstance(stmt, ast.Match) and stmt.cases and _matches_anything(stmt.cases[-1]) and all(
            _captures(case.pattern, name) or every(case.body) for case in stmt.cases
        ):
            return True
        if isinstance(stmt, (ast.Try, ast.TryStar)) and (every(stmt.finalbody) or (
            every(stmt.body) or every(stmt.orelse)
        ) and all(every(h.body) or isinstance(h.body[-1], ast.Raise) for h in stmt.handlers)):
            return True
        if isinstance(stmt, ast.Expr) and id(stmt.value) in setters:
            return True
        if walrus and _walrus_runs(stmt, name):
            return True
    return False


def _if_binds(
    stmt: ast.If, every: Callable[[Sequence[ast.stmt]], bool], *, tested: Callable[[ast.If], bool],
) -> bool:
    """Whether an if binds a name on every path `every` reads: the arm that
    runs where its test is a constant, else both arms. An elif chain is
    read arm by arm in turn, so a long one needs no deep recursion, and an
    elif whose test binds the name (`tested`), as a walrus there does,
    binds it on each path that reaches it."""
    current = stmt
    while True:
        if isinstance(current.test, ast.Constant):
            if current.test.value:
                return every(current.body)
        elif not every(current.body):
            return False
        rest = current.orelse
        if len(rest) == 1 and isinstance(rest[0], ast.If):
            current = rest[0]
            if tested(current):
                return True
            continue
        return every(rest)


def _may_swallow(stmt: ast.With | ast.AsyncWith, swallowers: AbstractSet[int]) -> bool:
    """Whether a with may swallow an error its body raises, so the body may
    stop short and the code after it run: it is one of `swallowers`, by id
    (`_Reader.swallowers`, `_swallowing_withs`)."""
    return id(stmt) in swallowers


class _ContextNames(NamedTuple):
    """The names a with's context is read by (`_may_swallow`): the last name
    it is spelled with, or that it calls, as `suppress` for
    `contextlib.suppress(E)`; and the name it starts from, as `contextlib`
    there or `cm` for `cm`, None where it starts from no name."""

    last: str | None
    root: ast.Name | None

    @property
    def root_name(self) -> str | None:
        """The name it starts from, by its text."""
        return None if self.root is None else self.root.id


def _context_names(context: ast.expr) -> _ContextNames:
    """The names `context`, what a with enters, is read by (`_ContextNames`)."""
    head = context.func if isinstance(context, ast.Call) else context
    root = _root_of(head)
    return _ContextNames(last=_last_name(head) or None, root=root if isinstance(root, ast.Name) else None)


def _swallowing_withs(reader: _Reader) -> frozenset[int]:
    """The withs of `reader`'s cells, by id, that may swallow an error their
    body raises (`_may_swallow`): one of whose contexts (`_context_names`)
    is called by, or spelled with, the last name of a context that may
    (`_SWALLOWING_CONTEXTS`), or of a name a cell imports one as
    (`_Cell.swallowing_aliases`); or starts from a class a cell defines
    with `__exit__` (`_Cell.swallowing_classes`), so `requests.Session()`
    is no cell's `Session`; from a name any cell binds to a call or an
    instance of one of them (`_Cell.value_heads`), as `cm` for `cm =
    suppress(E)`, followed until no more turn up; from a name read code
    doesn't bind where the with runs (`_Reader.binds`), nor Python or the
    pad's boot (`_BUILTIN_NAMES`), since only code the check can't read may
    have bound it; or from a name that holds what such code made
    (`_Reader.tainted_contexts`). A with at module level reads the aliases
    and classes of the cells that run up to it (`_Reader.ran_ahead`); one
    inside a function, which runs when it is called, those of every
    cell."""
    cells = reader.cells
    aliasing = [cell for cell in cells if cell.swallowing_aliases]
    defining = [cell for cell in cells if cell.swallowing_classes]
    aliases = frozenset().union(*(cell.swallowing_aliases for cell in aliasing))
    classes = frozenset().union(*(cell.swallowing_classes for cell in defining))
    named = _SWALLOWING_CONTEXTS | aliases
    followed: set[str] = set()
    heads = [(name, _context_names(head)) for cell in cells for name, head in cell.value_heads]
    grew = True
    while grew:
        grew = False
        for name, spelled in heads:
            if name not in followed and (
                spelled.last in named or spelled.root_name in classes or spelled.root_name in followed
            ):
                followed.add(name)
                grew = True
    bound = set(_BUILTIN_NAMES).union(*(cell.names_bound for cell in cells))
    out: set[int] = set()
    for cell in cells:
        for stmt in cell.signs.withs:
            ahead = {id(c) for c in (cells if cell.scope_of(stmt) is not None else reader.ran_ahead(stmt))} if (
                aliasing or defining
            ) else set()
            for item in stmt.items:
                spelled = _context_names(item.context_expr)
                root = spelled.root
                if (
                    spelled.last in _SWALLOWING_CONTEXTS
                    or any(spelled.last in c.swallowing_aliases for c in aliasing if id(c) in ahead)
                    or root is not None and (
                        root.id in followed or root.id in reader.tainted_contexts
                        or any(root.id in c.swallowing_classes for c in defining if id(c) in ahead)
                        or root.id not in _BUILTIN_NAMES and (root.id not in bound or not reader.binds(root))
                    )
                ):
                    out.add(id(stmt))
                    break
    return frozenset(out)


def _tainted_contexts(reader: _Reader, defined: _Defined | None) -> frozenset[str]:
    """The names a with's context starts from (`_context_names`) that hold
    what code the check can't read made (`_Defined.tainted`), as a name
    imported from a module a cell wrote does, for a with `reader` doesn't
    read as one that may swallow an error already (`_Reader.swallowers`):
    such a context may swallow one too."""
    if defined is None or not defined.tainted:
        return frozenset()
    return frozenset(
        root for cell in reader.cells for stmt in cell.signs.withs if id(stmt) not in reader.swallowers
        for item in stmt.items for root in [_context_names(item.context_expr).root_name] if root in defined.tainted
    )


def _assigns(target: ast.AST, name: str) -> bool:
    """Whether an assignment to `target` binds `name`: the name itself, a
    tuple or list that holds it, or an entry of the cell's own namespace
    under it (`_namespace_key`)."""
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, ast.Subscript):
        return _namespace_key(target) == name
    return isinstance(target, (ast.Tuple, ast.List, ast.Starred)) and name in _stored_names(target)


def _namespace_key(target: ast.AST) -> str | None:
    """The name a store into the cell's own namespace by a literal key
    binds, as `globals()["push"] = f` binds `push`; None for any other
    target."""
    if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Call) and not target.value.args
            and _last_name(target.value.func) in _GLOBALS_CALLS and isinstance(target.slice, ast.Constant)
            and isinstance(target.slice.value, str)):
        return target.slice.value
    return None


def _runs_body_surely(node: ast.AST) -> bool:
    """Whether `node`, an if, a loop or a comprehension's loop, surely
    starts its body once it runs: an if or a while whose test is a true
    constant, or a loop over a literal of items, none of them spread, with
    no filter on a comprehension's."""
    if isinstance(node, (ast.If, ast.While)):
        return isinstance(node.test, ast.Constant) and bool(node.test.value)
    if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
        items = node.iter
        return isinstance(items, (ast.List, ast.Tuple, ast.Set)) and bool(items.elts) and not any(
            isinstance(e, ast.Starred) for e in items.elts
        ) and not getattr(node, "ifs", ())
    return False


# The statements that may end a loop's pass, and those that may end a
# function's run while its caller runs on, before the statements after them
# (`_before_a_jump`). A raise ends the caller's run too.
_LOOP_JUMPS = (ast.Break, ast.Continue)
_RETURNS = (ast.Return,)


def _before_a_jump(body: Sequence[ast.stmt], *, jumps: tuple[type[ast.AST], ...] = _LOOP_JUMPS) -> list[ast.stmt]:
    """The statements of `body` before the first that runs one of `jumps`
    (`_nodes_that_run`): a `break` or a `continue`, which may end a loop's
    pass there, or a `return`, which may end a function's run. One in a
    function or lambda defined inside, or in a branch that never runs,
    doesn't count."""
    def runs_a_jump(stmt: ast.stmt) -> bool:
        return not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            isinstance(n, jumps) for n in _nodes_that_run(stmt)
        )

    return list(itertools.takewhile(lambda s: not runs_a_jump(s), body))


def _sets_global(func: ast.FunctionDef | ast.AsyncFunctionDef, cell: _Cell, name: str, *,
                 swallowers: AbstractSet[int]) -> bool:
    """Whether a call of `func`, a def of `cell`, binds `name` at module
    level: it declares `name` global and binds it on every path through its
    body before its first return (`_binds_on_every_path`,
    `_before_a_jump`). Not an async def, whose call only makes a coroutine,
    nor a generator, whose call runs none of its body."""
    if isinstance(func, ast.AsyncFunctionDef) or any(
        isinstance(n, (ast.Yield, ast.YieldFrom)) and cell.scope_of(n) is func for n in ast.walk(func)
    ):
        return False
    return any(name in g.names and cell.scope_of(g) is func for g in cell.signs.globals_declared) and (
        _binds_on_every_path(_before_a_jump(func.body, jumps=_RETURNS), name, swallowers=swallowers)
    )


def _matches_anything(case: ast.match_case) -> bool:
    """Whether `case` matches any subject: a bare capture or `_`, with no guard."""
    return case.guard is None and isinstance(case.pattern, ast.MatchAs) and case.pattern.pattern is None


def _captures(pattern: ast.AST, name: str) -> bool:
    """Whether the match pattern `pattern` binds `name` once it matches."""
    return any(
        isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name == name or isinstance(n, ast.MatchMapping)
        and n.rest == name for n in ast.walk(pattern)
    )


def _walrus_runs(stmt: ast.stmt, name: str) -> bool:
    """Whether running `stmt` surely runs a walrus that binds `name`
    (`_walrus_binds`): anywhere in a simple statement, and in the part of a
    compound one that always runs, an if's or while's test, a loop's
    iterable, a with's contexts or a match's subject."""
    if isinstance(stmt, (ast.If, ast.While)):
        parts: list[ast.AST] = [stmt.test]
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        parts = [stmt.iter]
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        parts = [item.context_expr for item in stmt.items]
    elif isinstance(stmt, ast.Match):
        parts = [stmt.subject]
    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Try, ast.TryStar)):
        parts = []
    else:
        parts = [stmt]
    return any(_walrus_binds(part, name) for part in parts)


def _walrus_binds(node: ast.AST, name: str) -> bool:
    """Whether running `node` surely runs a walrus that binds `name` in the
    scope `node` is in: not past the first operand of `and` or `or`, nor
    past the first two of a chained comparison, which may stop there, nor in
    a branch of an if-expression, a lambda or a generator, which may not run
    it, and in a list, set or dict comprehension only where each of its
    loops surely runs its element (`_runs_body_surely`). Read with a stack
    of parts to read, so a deep expression needs no deep recursion."""
    todo: list[ast.AST] = [node]
    while todo:
        current = todo.pop()
        if isinstance(current, ast.NamedExpr):
            if isinstance(current.target, ast.Name) and current.target.id == name:
                return True
            todo.append(current.value)
        elif isinstance(current, ast.BoolOp):
            todo.append(current.values[0])
        elif isinstance(current, ast.Compare):
            todo += [current.left, current.comparators[0]]
        elif isinstance(current, ast.IfExp):
            todo.append(current.test)
        elif isinstance(current, (ast.ListComp, ast.SetComp, ast.DictComp)):
            if all(_runs_body_surely(g) for g in current.generators):
                todo += [current.key, current.value] if isinstance(current, ast.DictComp) else [current.elt]
        elif not isinstance(current, (ast.Lambda, ast.GeneratorExp)):
            todo.extend(ast.iter_child_nodes(current))
    return False


def _binds_before(
    stmt: ast.AST, field: str, name: str, *, setters: AbstractSet[int], walrus: bool, swallowers: AbstractSet[int],
) -> bool:
    """Whether `stmt`, a statement, handler or case, surely binds `name`
    before the block its `field` holds runs: a loop's target for its body,
    a with's targets, a handler's `as` name, a case's pattern, a try's
    whole body for its else (`_binds_on_every_path`), and, with `walrus`, a
    walrus in a case's guard or in what the statement runs first
    (`_walrus_runs`)."""
    if isinstance(stmt, (ast.For, ast.AsyncFor)) and field == "body" and name in _stored_names(stmt.target):
        return True
    if isinstance(stmt, (ast.With, ast.AsyncWith)) and any(
        item.optional_vars is not None and name in _stored_names(item.optional_vars) for item in stmt.items
    ):
        return True
    if isinstance(stmt, ast.ExceptHandler):
        return stmt.name == name
    if isinstance(stmt, ast.match_case):
        return _captures(stmt.pattern, name) or walrus and stmt.guard is not None and _walrus_binds(stmt.guard, name)
    if isinstance(stmt, (ast.Try, ast.TryStar)):
        return field == "orelse" and _binds_on_every_path(
            stmt.body, name, setters=setters, walrus=walrus, swallowers=swallowers,
        )
    return walrus and isinstance(stmt, ast.stmt) and _walrus_runs(stmt, name)


def _ends_before(node: ast.AST, cell: _Cell, line: int) -> bool:
    """Whether the statement of `cell` that holds `node` ends before `line`:
    a traceback names the line of a statement's part that raised, which may
    be any line of it."""
    while not isinstance(node, ast.stmt) and node in cell.parents:
        node = cell.parents[node]
    return (getattr(node, "end_lineno", None) or getattr(node, "lineno", 0)) < line


def _import_names(stmt: ast.Import | ast.ImportFrom) -> list[str]:
    """The names an import statement binds: each `as` name, else a plain
    import's top package or a from-import's name. A star import's names
    aren't listed."""
    return [a.asname or a.name.partition(".")[0] for a in stmt.names if a.name != "*"]


def _command_word_constant(node: ast.Constant, cell: _Cell) -> bool:
    """Whether `node`, a string of `cell`, is a word of a command list a
    program is handed: an item of a list or tuple, or of lists joined with
    `+`, that a call by a launcher's name (`_LAUNCH_NAMES`) takes, as
    `subprocess.run(["pip", "install", name])` does."""
    current: ast.AST = node
    parent = cell.parents.get(current)
    if not isinstance(parent, (ast.List, ast.Tuple)):
        return False
    while isinstance(parent, (ast.List, ast.Tuple, ast.BinOp, ast.Starred)):
        current, parent = parent, cell.parents.get(parent)
    return isinstance(parent, ast.Call) and current in parent.args and _last_name(parent.func) in _LAUNCH_NAMES


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


def _latest(found: list[_Binding]) -> list[_Binding]:
    """`found`, what a read gets (`_Reader.bindings_of`), from each cell's
    last module-level value that surely ran: one built on the value before
    it, as `cmd = cmd + " -y"` is, reads that value itself, so a reader that
    follows a name through its values needs no other."""
    last = {id(b.cell): b.line for b in found if b.scope is None and b.how == "value" and not b.conditional}
    return [b for b in found if not (b.scope is None and b.line < last.get(id(b.cell), 0))]


def _module_alias(binding: _Binding) -> bool:
    """Whether `binding` gives its name, at module level, another name's
    value, as `h1 = h0` does. That name reads only the bindings before this
    one (`_Reader.bindings_of`), so a chain of them ends at its first value."""
    return binding.scope is None and binding.how == "value" and isinstance(binding.value, ast.Name)


def _rebuilt(binding: _Binding, name: str) -> bool:
    """Whether `binding` gives `name`, at module level, a value built on the
    value before it, as `cmd = cmd + " -y"` does. Such a value reads only
    bindings before its own (`_Reader.bindings_of`), so a reader follows a
    chain of them with no depth limit: the chain ends at its first value."""
    return binding.scope is None and binding.how == "value" and _value_loads(binding, name)


def _value_loads(binding: _Binding, name: str) -> bool:
    """Whether the value `binding` gives reads `name` (`_Cell.loads_within`)."""
    return bool(binding.cell.loads_within(binding.value, name))


def _since_last_sure(bindings: list[_Binding], *, name: str) -> list[_Binding]:
    """One cell's module-level `bindings` of `name`, in line order, from the
    last that surely ran and doesn't build on the value before it, as `sql =
    sql.rstrip(', ')` does: a read after them gets that value or one an if,
    loop or try after it gave."""
    ordered = sorted(bindings, key=lambda b: b.line)
    for i in range(len(ordered) - 1, -1, -1):
        if not ordered[i].conditional and not _value_loads(ordered[i], name):
            return ordered[i:]
    return ordered


def _ran_before(node: ast.AST, binding: _Binding, *, cell: _Cell) -> bool:
    """Whether `node`, in `cell`, surely ran before `binding` gave its name a
    value: in the binding's scope, and in an earlier cell, or earlier in the
    binding's cell (`_follows`) outside any loop around the binding
    (`_loop_holds_both`). Code in
    a function runs when it is called, a function's body after its
    parameters get their values, and code a cell passes to exec() later than
    all, so none of those counts."""
    if cell.scope_of(node) is not binding.scope or binding.cell.host_line is not None:
        return False
    if cell is not binding.cell:
        return cell.order < binding.cell.order
    return not (binding.how == "param" or _follows(node=node, value=binding.value) or _loop_holds_both(node, binding))


def _loop_holds_both(node: ast.AST, binding: _Binding) -> bool:
    """Whether a loop around `binding`, in its scope, holds `node` too, code
    of the binding's cell, so each pass after the first runs `node` after
    the binding gave its name a value. A loop whose lines don't hold
    `node`'s can't hold it."""
    line = getattr(node, "lineno", None)
    return any(
        (line is None or loop.lineno <= line <= (loop.end_lineno or loop.lineno)) and _inside(node, loop, binding.cell)
        for loop in binding.cell.loops_around(binding.value, scope=binding.scope)
    )


class _Reader:
    """Reads one cell against the conversation's earlier cells."""

    def __init__(
        self, cells: list[_Cell], *, current: _Cell, project_root: Path | None = None,
        errored: frozenset[int] = frozenset(), errored_paths: Literal["any", "unknown"] = "any",
        stopped_early: frozenset[int] = frozenset(), stopped_at: Mapping[int, int] | None = None,
        tainted_contexts: frozenset[str] = frozenset(),
    ) -> None:
        self.cells = cells
        self.current = current
        # What each name holds where a node reads it, by the name, the node's
        # id and `partly_run`, read once per check (`bindings_of`).
        self._bindings_read: dict[tuple[str, int, bool], tuple[_Binding, ...]] = {}
        # The names a with's context may start from that hold what code the
        # check can't read made (`_tainted_contexts`): such a with may
        # swallow an error its body raises (`swallowers`).
        self.tainted_contexts = tainted_contexts
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
        # The line each of those stopped at, by index, where the host knows
        # it: what ran before that line bound its names (`bound_before_stop`).
        self.stopped_at: Mapping[int, int] = stopped_at or {}
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
        # What each read `recalled` gave, by the read, the node, the depth
        # and `call_key`. While a read reads a name rebuilt from itself ahead
        # of its own read (`reading_ahead`), what it read at each load of the
        # name, by the read, the load's id and the depth; and the reads and
        # names it reads so.
        self._remembered: dict[tuple[str, int, int, tuple[tuple[int, str, int], ...]], _Remembered[Any]] = {}
        self._read_ahead: dict[tuple[str, int, int], Any] = {}
        self._reading_ahead: set[tuple[str, str]] = set()
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
        self.in_order = sorted((c for c in cells if c.shared), key=lambda c: c.order)
        self._places = [c.order for c in self.in_order]
        # The cells that bind each name at their top level, in that order,
        # and the first host of those that didn't stop early, read once per
        # name (`bound_before`).
        self._binders: dict[str, list[_Cell]] = {}
        self._first_binder_hosts: dict[str, int] = {}
        # The member each `getattr` by a key bound to literal texts reads, by
        # the call's id (`looked_up_member`).
        self._looked_up: dict[int, str | None] = {}
        self._setters: dict[str, frozenset[int]] = {}
        for cell in self.in_order:
            for name in cell.module_bound:
                self._binders.setdefault(name, []).append(cell)
        # The checked cell and the literal code it runs; and every cell by
        # its place (`_Cell.order`).
        self.checked = [c for c in cells if c.host >= current.index]
        self._by_order = {c.order: c for c in cells}
        for cell in cells:
            self._cells_by_node[cell.tree] = cell
            self._cells_by_node.update(dict.fromkeys(cell.parents, cell))
            self.lambda_names.update(cell.lambda_names)
        # The definitions a later one in a cell that stopped early may not
        # have replaced, as it may not have run before the cell stopped
        # (`ran_before_stop`), by the later one's id.
        self.replaced: dict[int, list[Function | ast.ClassDef]] = {}
        # The cells that define each function name, in the order they run.
        self._definers: dict[str, list[_Cell]] = {}
        for cell in self.in_order:
            for name in cell.functions:
                self._definers.setdefault(name, []).append(cell)
            if cell.host in stopped_early:
                for table, defined in ((self.functions, cell.functions), (self.classes, cell.classes)):
                    for name, node in defined.items():
                        if name in table and not self.ran_before_stop(cell, node):
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
        # Pad functions this cell calls, by name: their bodies run now.
        self.called: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {
            n.func.id: func for cell in self.checked for n in cell.calls
            if isinstance(n.func, ast.Name) and (func := self.function_named(n.func)) is not None
        }
        # While the check reads an earlier cell's function for one call: what
        # that call passes for each parameter, by (function id, name), None
        # for one it can't read; and the call its return value goes to.
        self.param_args: dict[tuple[int, str], _Arg | None] = {}
        self.returns_to: dict[int, ast.AST] = {}
        # How many times a read asked what the call being read passes, or
        # where its return value goes (`given`, `returned_to`): a scan that
        # asked nothing reads alike at any call. While such a scan counts its
        # asks (`afresh`), the cache key of its reads, which no read before
        # it used.
        self.call_asks = 0
        self._fresh_key: tuple[tuple[int, str, int], ...] | None = None
        self._fresh_scans = 0
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
        chdir_names = {name for cell in cells for name, full in cell.imports.items() if full in _CHDIR_FUNCS}
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

    def function_named(self, node: ast.Name) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """The def a call to the name `node` reaches by the function table:
        its own cell's first when that cell's names don't land in the cell
        namespace, as for code `exec(src, {})` runs, then the latest of the
        conversation's (`functions`)."""
        cell = self.cell_of(node)
        own = None if cell.shared else cell.functions.get(node.id)
        return own or self.functions.get(node.id)

    def read_too(self, cell: _Cell) -> None:
        """Reads the names of `cell`, literal code a cell runs, with the
        rest, so the literal code it runs in turn resolves through its own
        imports while `_exec_cells` reads on. The check then reads every
        such cell with a reader of its own."""
        self._cells_by_node[cell.tree] = cell
        self._cells_by_node.update(dict.fromkeys(cell.parents, cell))
        self.imports.update(cell.imports)
        self.bound_names.update(cell.bindings)

    def position(self, node: ast.AST) -> tuple[int, ...]:
        """Where `node` runs among the cells: its cell's place (`_Cell.order`),
        then its line and column. Literal code an exec() runs comes at the
        exec()'s line, before what follows it in that cell."""
        return (*self.cell_of(node).order, getattr(node, "lineno", 0), getattr(node, "col_offset", 0))

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
        """What the call being read passes for each parameter, as a cache key;
        under `afresh`, that scan's own key."""
        if self._fresh_key is not None:
            return self._fresh_key
        return tuple(sorted((k[0], k[1], id(a.node) if a else 0) for k, a in self.param_args.items()))

    @cached_property
    def points_installers(self) -> bool:
        """Whether a cell sets, in the pad's own environment, a setting that
        points pip or uv at another index or a folder
        (`_INSTALLER_SOURCE_SETTING`, `_Reader.environ_sets`). Read once per
        check, where an install needs it."""
        return self.environ_sets(_INSTALLER_SOURCE_SETTING)

    @cached_property
    def points_python(self) -> bool:
        """Whether a cell sets, in the pad's own environment, a setting that
        changes what a Python process imports or runs as it starts
        (`_PYTHON_SETTING`, `_Reader.environ_sets`), as `PYTHONPATH` does.
        Read once per check, where a launch of Python needs it."""
        return self.environ_sets(_PYTHON_SETTING)

    def environ_sets(self, setting: re.Pattern[str]) -> bool:
        """Whether a cell sets, in the pad's own environment, a setting
        `setting` matches (`_changes_set`), as `os.environ["PIP_INDEX_URL"] =
        ...` or `os.putenv(...)` does."""

        def environ(node: ast.AST) -> bool:
            return self.dotted(node) == "os.environ"

        return any(
            _changes_set(cell, self, holds=environ, setting=setting, depth=0) or any(
                self.dotted(c.func) == "os.putenv" and (not c.args or _names_setting(c.args[0], setting))
                for c in cell.calls
            )
            for cell in self.cells
        )

    @cached_property
    def project_builds(self) -> bool:
        """Whether the project folder holds a Python project with build code
        an install of it runs, as uv reads one: a `pyproject.toml` with a
        `[build-system]` table or a `package = true` setting. With neither,
        uv installs only what the project depends on, a `setup.py` beside it
        or not. Read from disk once per check; yes where the check can't
        read it."""
        if self.project_root is None:
            return True
        try:
            text = (self.project_root / "pyproject.toml").read_text(errors="replace")
        except FileNotFoundError:
            return False
        except OSError:
            return True
        return bool(_BUILDS_A_PROJECT.search(text))

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
        func: Function | None = lambdas[0] if lambdas else self._function_at(node)
        if func is None or not self.earlier(func):
            return None
        return func

    def _function_at(self, node: ast.Name) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        """The function a def gave `node`'s name where `node` reads it: at
        module level, the latest one that ran before the read
        (`ran_ahead`), as a call before an exec() that redefines a name
        still runs the old one; inside a function, which runs when called,
        the latest of all (`functions`)."""
        cell = self.cell_of(node)
        if cell.scope_of(node) is not None:
            return self.functions.get(node.id)
        read_at = self.position(node)
        definers = [c for c in self._definers.get(node.id, []) if c.order < read_at]
        for other in reversed([*definers, *([] if cell.shared or node.id not in cell.functions else [cell])]):
            func = other.functions[node.id]
            if other.scope_of(func) is not None or self.defined_ahead(other, func, node):
                return func
        return None

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
        if not self.param_args and self._fresh_key is None:
            return None
        for b in self.bindings_of(node.id, node=node):
            if b.how == "param" and self.given(b.value, node.id):
                return self.param_args[(id(b.value), node.id)]
        return None

    def given(self, func: ast.AST, name: str) -> bool:
        """Whether the call being read passes `func`'s parameter `name`
        (`param_args`). Each ask counts (`call_asks`): what the read gives
        may differ at another call."""
        self.call_asks += 1
        return (id(func), name) in self.param_args

    def args_for(self, func: ast.AST, name: str) -> list[_Arg | None]:
        """What the call being read passes for `func`'s parameter `name`,
        else what each of its call sites passes (`call_sites`)."""
        if self.given(func, name):
            return [self.param_args[(id(func), name)]]
        return self.call_sites(func, name)

    def returned_to(self, func: ast.AST) -> ast.AST | None:
        """The call `func`'s return value goes to while that call is read,
        an ask `call_asks` counts as `given` does."""
        self.call_asks += 1
        return self.returns_to.get(id(func))

    @contextmanager
    def afresh(self) -> Iterator[None]:
        """While a scan reads under it, every read under it reads afresh: its
        cache key (`call_key`) is one no read before it used, so `call_asks`
        counts each ask the scan makes, a read the cache would give included."""
        saved = self._fresh_key
        self._fresh_scans += 1
        self._fresh_key = ((-self._fresh_scans, "", 0),)
        try:
            yield
        finally:
            self._fresh_key = saved

    def passed_node(self, node: ast.AST) -> ast.AST:
        """The code that gives `node` its value when `node` reads a parameter
        of a function being read: what the call being read passes for it,
        followed through each parameter it passes on in turn. `node` itself
        for any other code."""
        for _ in range(_CALL_DEPTH + 1):
            arg = self.argument_of(node) if isinstance(node, ast.Name) else None
            if arg is None:
                break
            node = arg.node
        return node

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
            if self._past is not None:
                node = self._alias_start(node)
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
            if isinstance(receiver, ast.Name) and self.import_of(receiver) is not None:
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

    def _alias_start(self, node: ast.Name) -> ast.Name:
        """The name a chain of module-level aliases that ends at `node`
        starts from, as `h0` for `h70` after `h1 = h0` to `h70 = h69`: each
        link a name one sure binding gives another name's value, followed
        in a loop, so a read past its limits (`read_past`) reaches the start
        of a chain of any length without running out of depth."""
        seen: set[int] = set()
        while (id(node) not in seen and node.id in self.bound_names and node.id not in self.function_names
               and node.id not in self.classes and self.import_of(node) is None):
            seen.add(id(node))
            found = self.bindings_of(node.id, node=node)
            value = found[0].value if len(found) == 1 and not found[0].conditional else None
            if not (isinstance(value, ast.Name) and _module_alias(found[0])):
                break
            node = value
        return node

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
        return statement in the function, not in a function inside it
        (`_Cell.returns_of`)."""
        if isinstance(func, ast.Lambda):
            return [func.body]
        return list(self.cell_of(func).returns_of(func))

    def items_of(self, *, node: ast.AST, key: ast.AST | None, depth: int) -> list[ast.AST] | None:
        """The elements of the list, tuple, set or dict values `node` holds,
        with what the pad adds to it, only those `key` names where the check
        reads which (`_key_values`); None when the check can't place them,
        or at the depth limit (`stops`)."""
        if self.stops("items", node, depth=depth, limit=6):
            return None
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            elements = [e.value if isinstance(e, ast.Starred) else e for e in node.elts]
            keys = None if key is None or isinstance(node, ast.Set) else _key_values(key, self, depth=depth + 1)
            if keys and not any(isinstance(e, ast.Starred) for e in node.elts) and all(
                type(k) is int and -len(elements) <= k < len(elements) for k in keys
            ):
                return [elements[k] for k in dict.fromkeys(keys)]
            return elements
        if isinstance(node, ast.Dict):
            keys = None if key is None else _key_values(key, self, depth=depth + 1)
            if keys and all(isinstance(k, ast.Constant) for k in node.keys):
                hits = [v for k, v in zip(node.keys, node.values) if k.value in keys]
                if hits:
                    return hits
            return list(node.values)
        if isinstance(node, ast.Name):
            found = self.bindings_of(node.id, node=node)
            if not found:
                # A name no read code binds holds what code the check can't read gave it.
                return None
            out: list[ast.AST] = []
            for b in found:
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
            if self.import_of(node) is not None:
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

    def partials_of(self, *, node: ast.expr, depth: int = 0) -> list[ast.Call]:
        """Each `functools.partial(f, ...)` call `node` is, or a name holds;
        one whose `f` is a partial, or a name that holds partials, reads as
        each of those with its own arguments after theirs, as
        `partial(base, cmd)` after `base = partial(subprocess.run,
        check=True)` reads as `partial(subprocess.run, cmd, check=True)`, up
        to `_PARAM_DEPTH` partials deep."""
        if isinstance(node, ast.Name):
            if node.id not in self.bound_names:
                return []
            values = [b.value for b in self.bindings_of(node.id, node=node) if b.how == "value"]
        else:
            values = [node]
        out: list[ast.Call] = []
        for v in values:
            if not (isinstance(v, ast.Call) and v.args and self.dotted(v.func) == "functools.partial"):
                continue
            inner = self.partials_of(node=v.args[0], depth=depth + 1) if depth < _PARAM_DEPTH else []
            out += [ast.copy_location(ast.Call(func=v.func, args=[*i.args, *v.args[1:]],
                                               keywords=[*i.keywords, *v.keywords]), v) for i in inner] or [v]
        return out

    # ── names ────────────────────────────────────────────────────────────

    def dotted(self, node: ast.AST) -> str:
        """`os.path.join` for a call through an alias such as `osp.join`."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return ""
        head = self.import_of(node) or node.id
        return ".".join([head, *reversed(parts)])

    def import_of(self, node: ast.Name) -> str | None:
        """What an import binds the name `node` reads to, as the dotted name
        it imports: its own cell's import first when that cell's names don't
        land in the cell namespace (`_Cell.shared`), as for the code
        `exec(src, {})`, a function's exec() or `python -c` runs, which its
        own imports reach; else the conversation's (`imports`). None when
        no import binds it."""
        cell = self.cell_of(node)
        if not cell.shared and node.id in cell.imports:
            return cell.imports[node.id]
        return self.imports.get(node.id)

    def bindings_of(self, name: str, *, node: ast.AST, partly_run: bool = False) -> list[_Binding]:
        """What `name` holds where `node` reads it.

        Inside a function, the function's own bindings first. At module level,
        this cell's bindings before the read, else the latest earlier cell's,
        each from the last one that surely ran (`_module_layers`). A function
        body runs when it is called, so it sees every cell's module bindings,
        the latest first.
        A cell that binds the name only under an if, loop or try doesn't hide
        an older binding: both count. With `partly_run`, an earlier cell in
        `errored` may have stopped before any of its bindings, or after any:
        each of them counts, and so do the older ones. Read once per check:
        a loop that reads a name many earlier loops bind asks it again and
        again.
        """
        key = (name, id(node), partly_run)
        found = self._bindings_read.get(key)
        if found is None:
            found = self._bindings_read[key] = tuple(self._bindings_afresh(name, node=node, partly_run=partly_run))
        return list(found)

    def _bindings_afresh(self, name: str, *, node: ast.AST, partly_run: bool) -> list[_Binding]:
        """`bindings_of`, read afresh."""
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
            # A cell whose names don't land in the cell namespace reads its own first.
            own = [] if cell.shared else [cell, *self.namespace_parents(cell)]
            layers = (self._module_bindings(c, name) for c in (*own, *reversed(binders)))
        else:
            layers = self._module_layers(name, node=node, stopped=stopped)
        found: list[_Binding] = []
        for layer in layers:
            found += layer
            # At module level, the line `node` is on ran, so a sure binding
            # before it in its own cell did too.
            if any(not b.conditional and (b.cell.index not in stopped or b.cell is cell and not scopes)
                   for b in layer):
                break
        return found

    def _module_layers(self, name: str, *, node: ast.AST, stopped: frozenset[int]) -> Iterator[list[_Binding]]:
        """The module-level bindings of `name` that ran before `node`, a read
        at module level, in layers from the latest: one for each run of
        bindings one cell gives, in the order they run (`position`), so the
        literal code an exec() runs comes between its cell's bindings before
        the exec() and those after it. Each layer runs from its last binding
        that surely ran (`_since_last_sure`), or whole for an earlier cell in
        `stopped`. A read in the value a binding gives, as the right side of
        `cmd = cmd + ["-l"]`, runs before that binding, and so do one
        before a binding later on its line, as `cmd = cmd + " -a"; cmd = cmd
        + " -l"` gives one, and one on a line before a later binding, but
        for a loop's binding that a later pass of the loop reads, on its
        line or a later one (`_loop_holds_both`)."""
        cell = self.cell_of(node)
        read_at = self.position(node)
        line = getattr(node, "lineno", 0)
        binders = self._binders.get(name, [])

        def ran(c: _Cell) -> list[_Binding]:
            if c is cell:
                return [b for b in self._module_bindings(c, name) if (b.line <= line or _loop_holds_both(node, b))
                        and not _in_value(node, b) and not (_follows(node=b.value, value=node)
                                                            and not _loop_holds_both(node, b))]
            if c.host != cell.host:
                return self._module_bindings(c, name)
            return [b for b in self._module_bindings(c, name) if (*c.order, b.line) < read_at]

        def layers(group: list[_Cell]) -> Iterator[list[_Binding]]:
            found = [b for c in group for b in ran(c)]
            if len(group) > 1:
                found.sort(key=lambda b: (*b.cell.order, b.line))
            runs: list[list[_Binding]] = []
            for b in found:
                if runs and runs[-1][-1].cell is b.cell:
                    runs[-1].append(b)
                else:
                    runs.append([b])
            for bindings in reversed(runs):
                whole = bindings[0].cell is not cell and bindings[0].cell.index in stopped
                yield bindings if whole else _since_last_sure(bindings, name=name)

        same_host = [c for c in binders if c.host == cell.host and c is not cell]
        yield from layers([cell, *self.namespace_parents(cell), *same_host])
        earlier = [c for c in binders if c.host < cell.host]
        for _, group in itertools.groupby(reversed(earlier), key=lambda c: c.host):
            yield from layers(list(group))

    def namespace_parents(self, cell: _Cell) -> list[_Cell]:
        """The cells whose namespace `cell`, literal code that has none of
        its own, reads and binds names in (`_Cell.inherits`): the cell whose
        code runs it, and so on up, until one whose names land in the cell
        namespace, which every read reaches anyway."""
        out: list[_Cell] = []
        while cell.inherits and not cell.shared:
            parent = None if cell.parent_order is None else self._by_order.get(cell.parent_order)
            if parent is None or parent.shared:
                break
            out.append(parent)
            cell = parent
        return out

    def ran_ahead(self, node: ast.AST) -> list[_Cell]:
        """The cells whose names land in the cell namespace that run before
        `node` reads a name (`position`), its own cell among them, and that
        cell itself when its names don't land there."""
        cell = self.cell_of(node)
        ahead = self.in_order[: bisect.bisect_left(self._places, self.position(node))]
        return ahead if cell.shared else [*ahead, cell]

    @staticmethod
    def _module_bindings(cell: _Cell, name: str) -> list[_Binding]:
        return [b for b in cell.bindings.get(name, ()) if b.scope is None]

    def binds(self, node: ast.Name, *, surely: bool = False) -> bool:
        """Whether read code gives the name `node` reads a value where it
        reads it, hiding any builtin of that name: an import
        (`_imported_where`), an assignment, parameter, loop, with,
        comprehension or walrus (`bindings_of`), a def or class in a scope
        around it or at module level, before it (`ran_ahead`,
        `defined_ahead`) when `node` is at module level, or a def above it
        in the class body it sits in, which the function table leaves out
        (`_Cell.functions`). With `surely`, a cell that stopped early
        (`stopped_early`) binds only what it bound before it stopped
        (`bound_before_stop`)."""
        cell = self.cell_of(node)
        skip = self.stopped_early if surely else frozenset()
        if self.import_of(node) is not None and self._imported_where(node, skip=skip):
            return True
        if node.id in self.bound_names and self._bound_where(node, skip=skip):
            return True
        body = cell.scope_of(node)
        if isinstance(body, ast.ClassDef) and any(
            isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == node.id
            and _follows(node=node, value=stmt) for stmt in body.body
        ):
            return True
        if not any(node.id in names for names in (self.functions, self.classes, cell.functions, cell.classes)):
            return False
        scopes = cell.scopes_around(node)
        for other in self.ran_ahead(node):
            if other.host in skip and not self.bound_before_stop(other, node.id):
                continue
            for defined in (other.functions.get(node.id), other.classes.get(node.id)):
                if defined is None:
                    continue
                scope = other.scope_of(defined)
                if scope in scopes or scope is None and (scopes or self.defined_ahead(other, defined, node)):
                    return True
        return False

    def defined_ahead(self, other: _Cell, defined: ast.AST, node: ast.AST) -> bool:
        """Whether `defined`, a module-level statement of `other`, ran before
        `node` reads a name: earlier in its cell, or in a cell that runs
        before it, where literal code an exec() runs comes at that line."""
        if other is self.cell_of(node):
            return _follows(node=node, value=defined)
        last = getattr(defined, "end_lineno", None) or getattr(defined, "lineno", 0)
        return (*other.order, last) < self.position(node)

    def bound_before_stop(self, cell: _Cell, name: str) -> bool:
        """Whether `cell`, a cell that stopped on an error or literal code
        one runs, surely gave `name` a value before it stopped: by a
        module-level binding, def, class or import that surely ran before
        then (`ran_before_stop`), or by an if or try that did, every path
        through which binds it (`_binds_on_every_path`), as an import with
        an `except ImportError` that sets the name does."""
        made = [
            *(d for d in cell.definitions if d.name == name and cell.scope_of(d) is None),
            *(stmt for stmt in cell.signs.imports if name in _import_names(stmt) and cell.scope_of(stmt) is None),
        ]
        if any(self.ran_before_stop(cell, d) for d in made):
            return True
        found = [b for b in cell.bindings.get(name, ()) if b.scope is None]
        if any(self.ran_before_stop(cell, b.value, head=b.how in ("element", "context")) for b in found):
            return True
        branches: dict[int, ast.AST] = {}
        for node in (*made, *(b.value for b in found)):
            current = cell.parents.get(node)
            while current is not None and current is not cell.tree:
                if isinstance(current, (ast.If, ast.Try, ast.TryStar)):
                    branches[id(current)] = current
                current = cell.parents.get(current)
        return any(_binds_on_every_path([b], name, swallowers=self.swallowers) and self.ran_before_stop(cell, b)
                   for b in branches.values())

    def ran_before_stop(self, cell: _Cell, node: ast.AST, *, head: bool = False) -> bool:
        """Whether the module-level statement that holds `node`, code of
        `cell`, a cell that stopped on an error or literal code one runs,
        surely ran to its end before the cell stopped: it ends before the
        line the host says the cell stopped at (`stopped_at`), and runs
        whenever the cell does (`_Cell.runs_surely`) or whenever the
        statement the cell stopped in does (`_Cell.statement_at`,
        `_Cell.runs_before`), as one before it in a block that holds that
        statement does. With `head`, `node` is the iterable of a loop or
        the context a with enters, whose target is bound once `node` ran,
        so a loop or with whose body holds that statement counts too.
        Where the host doesn't say, only the top defs and classes ran
        (`_Cell.leading_definitions`), and only in a cell that compiled,
        since one that doesn't ran none of its code (`find_cell_writes`).
        Literal code ran whole when the line that runs it ran before that
        line (`_Cell.host_line`) and the call that runs it surely ran
        (`_Cell.surely_run`), and none of it did otherwise, nor where the
        host doesn't say."""
        line = self.stopped_at.get(cell.host)
        stmt = cell.statement_of(node)
        if cell.index != cell.host:
            return (line is not None and cell.host_line is not None and cell.host_line < line and cell.surely_run
                    and cell.runs_surely(stmt))
        if line is None:
            return stmt in cell.leading_definitions
        stop = cell.statement_at(line)
        whole = cell.runs_to_its_end(stmt, swallowers=self.swallowers)
        if not (whole or stop is not None and cell.runs_before(stmt, stop)):
            return False
        if head:
            # A loop's target is bound once its body starts: surely for a
            # loop over a literal of items, or where its body holds the stop.
            in_body = stop is not None and stop is not stmt and _inside(stop, stmt, cell)
            return (getattr(node, "end_lineno", None) or getattr(node, "lineno", 0)) < line and (
                in_body or cell.binds_loop_target(node)
            )
        return _ends_before(stmt, cell, line)

    def _imported_where(self, node: ast.Name, *, skip: frozenset[int]) -> bool:
        """Whether an import binds the name `node` reads where it reads it:
        in a scope around it, or at module level of a cell that runs before
        it, or of its own cell, before it unless `node` is inside a
        function. A cell in `skip` binds only what it imported before it
        stopped (`bound_before_stop`)."""
        cell = self.cell_of(node)
        scopes = cell.scopes_around(node)
        for other in self.ran_ahead(node):
            if other.host in skip and not self.bound_before_stop(other, node.id):
                continue
            for stmt in other.signs.imports:
                if node.id not in _import_names(stmt):
                    continue
                scope = other.scope_of(stmt)
                if scope is not None and scope in scopes or scope is None and (
                    scopes or self.defined_ahead(other, stmt, node)
                ):
                    return True
        return False

    def bound_before(self, name: str, cell: _Cell) -> bool:
        """Whether a cell that didn't stop early, and surely binds `name` at
        its top level when it runs (`_Cell.binds_whole`), runs in a
        conversation's cell before the one whose run runs `cell`: then
        `binds_surely` holds for a read of it in `cell`, as a cheaper read
        of it for a name many cells read."""
        first = self._first_binder_hosts.get(name)
        if first is None:
            setters = self.global_setters(name)
            first = self._first_binder_hosts[name] = next((
                c.host for c in self._binders.get(name, []) if c.host < self.current.index
                and c.host not in self.stopped_early and c.surely_run
                and c.binds_whole(name, setters=setters, swallowers=self.swallowers)
            ), self.current.index + 1)
        return first < cell.host

    @cached_property
    def swallowers(self) -> frozenset[int]:
        """The withs of the cells, by id, that may swallow an error their
        body raises (`_swallowing_withs`): each, read by `_may_swallow`,
        vouches for nothing its body binds. Read once per check."""
        return _swallowing_withs(self)

    def global_setters(self, name: str) -> frozenset[int]:
        """The calls, by id, that bind `name` at module level: a call as a
        statement, by its bare name, of a function whose def that is live
        where the call runs sets the name (`_live_setter`). Read once per
        name."""
        if name not in self._setters:
            declaring = {
                scope.name for cell in self.cells for declared in cell.signs.globals_declared
                if name in declared.names for scope in [cell.scope_of(declared)]
                if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            self._setters[name] = frozenset(
                id(call) for cell in self.cells if declaring for call in cell.calls
                if isinstance(call.func, ast.Name) and call.func.id in declaring
                and isinstance(cell.parents.get(call), ast.Expr) and self._live_setter(call, call.func.id, name)
            ) if declaring else frozenset()
        return self._setters[name]

    def _live_setter(self, call: ast.Call, func: str, name: str) -> bool:
        """Whether `call`, a call of the function `func` by its bare name,
        runs a def that sets `name` as a global (`_sets_global`). Read back
        from the last module-level binding of `func`, a def, class, import
        or assignment, made before the call where the call is at module
        level (`defined_ahead`), or anywhere for a call inside a function,
        which runs later: each must be such a def, up to one that surely
        ran to its end (`_Cell.runs_to_its_end`, `ran_before_stop`), since
        a later def replaces an earlier one, and one an if may skip leaves
        the one before it live."""
        made: list[tuple[_Cell, ast.AST]] = []
        for other in self.cells:
            made += [(other, d) for d in other.definitions if d.name == func and other.scope_of(d) is None]
            made += [(other, stmt) for stmt in other.signs.imports
                     if func in _import_names(stmt) and other.scope_of(stmt) is None]
            made += [(other, b.value) for b in self._module_bindings(other, func)]
        if self.cell_of(call).scope_of(call) is None:
            made = [(other, node) for other, node in made if self.defined_ahead(other, node, call)]
        for other, node in sorted(made, key=lambda m: self.position(m[1]), reverse=True):
            if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and _sets_global(node, other, name, swallowers=self.swallowers)):
                return False
            if other.surely_run and (
                self.ran_before_stop(other, node) if other.host in self.stopped_early
                else other.runs_to_its_end(node, swallowers=self.swallowers)
            ):
                return True
        return False

    def binds_surely(self, node: ast.AST, name: str) -> bool:
        """Whether read code surely gave `name` a value on every path to
        `node`, a place at module level of a conversation's cell: its own
        cell's code before it (`_Cell.bound_on_the_way`), or a cell whose
        names land in the cell namespace that ran before it, whole
        (`_Cell.binds_whole`) or, for one that stopped early, up to its stop
        (`bound_before_stop`). Literal code counts only where the call that
        runs it surely runs (`_Cell.surely_run`), and for a read in literal
        code, the cell that runs it counts only up to the line that does.
        """
        cell = self.cell_of(node)
        setters = self.global_setters(name)
        swallowers = self.swallowers
        if cell.bound_on_the_way(node, name, setters=setters, swallowers=swallowers):
            return True
        for other in self.ran_ahead(node):
            if other is cell:
                continue
            if other.index == cell.host != cell.index:
                # The cell that runs this literal code ran only up to the
                # line that runs it.
                at = other.statement_at(cell.host_line) if cell.host_line is not None else None
                if at is not None and other.bound_on_the_way(at, name, setters=setters, swallowers=swallowers):
                    return True
                continue
            if other.host in self.stopped_early:
                if self.bound_before_stop(other, name):
                    return True
            elif other.surely_run and other.binds_whole(name, setters=setters, swallowers=swallowers):
                return True
        return False

    def _bound_where(self, node: ast.Name, *, skip: frozenset[int]) -> bool:
        """Whether a binding of the name `node` reads is where it reads it
        (`bindings_of` finds one): in a scope around it, else at module
        level in its cell before it or in a cell that runs before it, or in
        any cell for a read inside a function. A read inside the value a
        binding gives, as the right side of `exec = exec` is, runs before
        that binding. A cell in `skip` binds only what it bound before it
        stopped (`bound_before_stop`), and a read in it sees only what
        surely ran before that (`ran_before_stop`)."""
        name = node.id
        cell = self.cell_of(node)
        scopes = cell.scopes_around(node)
        own = cell.bindings.get(name, ())
        if any(b.scope is not None and b.scope in scopes for b in own):
            return True
        binders = [c for c in self._binders.get(name, []) if c.host not in skip or self.bound_before_stop(c, name)]
        if scopes:
            return bool(binders)
        line = getattr(node, "lineno", 0)
        if any(
            b.scope is None and (b.line <= line or _loop_holds_both(node, b)) and not _in_value(node, b) and (
                cell.host not in skip or self.ran_before_stop(cell, b.value, head=b.how in ("element", "context"))
            )
            for b in own
        ):
            return True
        read_at = self.position(node)
        return any(
            c is not cell and c.order < read_at and (
                c.host != cell.host or any((*c.order, b.line) < read_at for b in self._module_bindings(c, name))
            )
            for c in binders
        )

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
        if not self.names_builtins or isinstance(node, ast.Attribute) and node.attr not in _EXEC_CALLS:
            # A member by another name is none of them, nor a call of one.
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
            if self.import_of(node) is not None or node.id not in self.bound_names:
                return None
            values: list[ast.AST] = [b.value for b in self.bindings_of(node.id, node=node) if b.how == "value"]
        elif isinstance(node, ast.Subscript):
            found = self.items_of(node=node.value, key=node.slice, depth=0)
            key = node.slice.value if isinstance(node.slice, ast.Constant) else None
            if found is None and key in _EXEC_CALLS:
                # An entry named for the builtin, of a holder the check can't place.
                return key
            values = found or []
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
            head, _, name = (self.import_of(node) or "").rpartition(".")
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
        a branch may not have run, unless the branch is an if whose test is
        a true constant (`_Cell.runs_surely`) or holds the read too, after
        it (`_Cell.runs_before`). A def, a class, and a name the function
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
            name in _import_names(stmt) and self._ran_for(stmt, c, node)
            for c in self.cells for stmt in c.signs.imports
        ):
            return False
        return not any(
            b.how != "element" and (not b.conditional or self._ran_for(b.value, b.cell, node))
            for b in self.bindings_of(name, node=node)
        )

    @staticmethod
    def _ran_for(bound: ast.AST, cell: _Cell, use: ast.Name) -> bool:
        """Whether `bound`, code of `cell` that binds the name `use` reads,
        surely ran where `use` reads it (`_Cell.runs_surely`,
        `_Cell.runs_before`)."""
        return cell.runs_surely(bound) or use in cell.parents and cell.runs_before(bound, use)

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
        import name, `__builtins__`, `sys.modules["builtins"]`, the
        `__builtins__` of any holder (`_builtins_held`), a builtin
        function's `__self__`, as `print.__self__`, or a name bound to one
        of them, as `m = __import__("builtins")` is once `_parse` reads the
        import as the plain name. It stops at the depth limit (`stops`)."""
        if self.stops("builtins", node, depth=depth, limit=_PARAM_DEPTH):
            return False
        if isinstance(node, ast.Name) and self.import_of(node) is None and node.id in self.bound_names:
            found = self.bindings_of(node.id, node=node)
            if found:
                return any(b.how == "value" and self._builtins_module(b.value, depth=depth + 1) for b in found)
        if isinstance(node, ast.Call):
            # `importlib.import_module("built" + "ins")` or `__import__(n)`
            # with `n` bound to that text.
            texts = self._importer_module_texts(node)
            return texts is not None and any(text == "builtins" for text in texts)
        if self._builtins_held(node):
            return True
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            return node.slice.value == "builtins" and self.dotted(node.value) == "sys.modules"
        if isinstance(node, ast.Attribute) and node.attr == "__self__" and isinstance(node.value, ast.Name):
            return node.value.id in _BUILTIN_FUNCTIONS and self._builtin_kept(node.value)
        return self.dotted(node) in ("builtins", "__builtins__")

    @staticmethod
    def _builtins_held(node: ast.AST) -> bool:
        """Whether `node` is the `__builtins__` of a holder, which holds the
        builtins module or its dict: as a member (`m.__builtins__`), as an
        entry of a namespace (`globals()["__builtins__"]`,
        `f.__globals__["__builtins__"]`), or as a frame's `f_builtins`."""
        if isinstance(node, ast.Attribute):
            return node.attr in ("__builtins__", "f_builtins")
        return (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                and node.slice.value == "__builtins__")

    def _builtins_dict(self, node: ast.AST) -> bool:
        """Whether `node` holds the builtins module's dict: `__builtins__`
        (the dict itself outside the main module) on its own or of any
        holder (`_builtins_held`), `vars(builtins)` or `builtins.__dict__`."""
        if isinstance(node, ast.Call):
            return self.dotted(node.func) == "vars" and len(node.args) == 1 and self._builtins_module(node.args[0])
        if isinstance(node, ast.Attribute) and node.attr == "__dict__":
            return self._builtins_module(node.value)
        return self._builtins_held(node) or self.dotted(node) == "__builtins__"

    def loads_builtins_value(self, node: ast.AST) -> bool:
        """Whether `node` loads the builtins module, or its dict, as a value
        anywhere the check doesn't already read as reaching a literal exec,
        eval or compile: handed to a call, held in a list or dict, returned,
        or read through `.get`, `.__getattribute__` or another member that
        reaches code or the cell's namespace, or one that isn't a builtin
        (`_BUILTINS_HARMLESS`). As `exec` itself, the builtins module loaded as
        a value may run any code the check can't read. Its safe uses are
        `builtins.exec` and its kin (read by `exec_builtin`), any other
        builtin, as `builtins.open`, `builtins.__dict__`, `vars(builtins)`
        and an entry of either, and `getattr(builtins, ...)` (read by
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
            # `builtins.exec`, read by `exec_builtin`, `builtins.__dict__`, or
            # a builtin that reaches no code, as `builtins.open`.
            return not (parent.attr in _EXEC_CALLS or parent.attr == "__dict__" or parent.attr in _BUILTINS_HARMLESS)
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
        `__import__` reads as the module it loads, and a `getattr` of a
        module by a key bound to literal texts as the member it reads
        (`looked_up_member`). A builtin of the pad no read code binds counts
        as itself (`_PAD_BUILTINS`). None for a root nothing imports, as a
        local named `code` or `profile` is."""
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
                member = self.looked_up_member(node)
                if member is not None:
                    return ".".join([member, *rest])
                continue
            if not isinstance(node, ast.Name) or id(node) in seen:
                continue
            seen.add(id(node))
            head = self.import_of(node)
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

    def looked_up_member(self, call: ast.Call) -> str | None:
        """The member of a module `call` reads where it is a `getattr(m,
        key)`, with no default or a constant one, `m` a module
        (`module_member`) and `key` a name or code the check reads as
        literal texts (`_literal_texts`): `m.<text>`, as the literal key
        reads (`_plain_lookup`), for the one text that names a member of
        `m` that runs code (`_runs_on_holder`), or the first where none
        does. None for any other call, and where two texts run code, since
        each may run its own way (`_looks_up_a_runner`). Read once per
        call."""
        key = id(call)
        if key not in self._looked_up:
            self._looked_up[key] = None
            args = call.args
            if (isinstance(call.func, ast.Name) and call.func.id == "getattr" and len(args) in (2, 3)
                    and not call.keywords and not isinstance(args[1], ast.Constant)
                    and (len(args) == 2 or isinstance(args[2], ast.Constant))):
                module = self.module_member(args[0])
                texts = _literal_texts(args[1], self) if module is not None else None
                running = [text for text in dict.fromkeys(texts or ()) if _runs_on_holder(module or "", text)]
                if texts and len(running) <= 1:
                    self._looked_up[key] = f"{module}.{(running or texts)[0]}"
        return self._looked_up[key]

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
        assignment binds to a hint (`_CodeSigns.hint_aliases`), or to a
        `functools.partial` of a name that is one (`_CodeSigns.partial_heads`),
        followed until no more turn up."""
        out = {name for cell in self.cells for name in cell.signs.hint_aliases}
        for cell in self.cells:
            for name, full in cell.imports.items():
                last = full.rpartition(".")[2]
                if name != last and _is_hint(last):
                    out.add(name)
        heads = [(made, head) for cell in self.cells for made, head in cell.signs.partial_heads
                 if isinstance(head, ast.Name)]
        grew = True
        while grew:
            grew = False
            for made, head in heads:
                if made not in out and head.id in out:
                    out.add(made)
                    grew = True
        return frozenset(out)

    def uncalled(self, func: Function) -> bool:
        """Whether nothing can call `func`: a def, or a lambda bound to bare
        names alone, that no decorator wraps and no class holds, whose names
        no cell loads or holds as a string, as `globals()["sh"]` does, in a
        conversation that looks no name up in the cell namespace under a
        key the check can't read (`looks_up_namespace`). The cell namespace
        is no module's, so no member names reach it. A string inside the
        function itself, or a word of a command a program runs
        (`_command_word_constant`), as pip's own "install" is, spells no
        lookup of it."""
        if isinstance(func, ast.Lambda):
            names = self.lambda_names.get(id(func), [])
            parent = self.cell_of(func).parents.get(func)
            if not names or not isinstance(parent, (ast.Assign, ast.AnnAssign)) or not all(
                isinstance(t, ast.Name) for t in (parent.targets if isinstance(parent, ast.Assign) else [parent.target])
            ):
                return False
        else:
            if func.decorator_list:
                return False
            names = [func.name]
        home = self.cell_of(func)
        if isinstance(home.scope_of(func), ast.ClassDef) or self.looks_up_namespace:
            return False
        return not any(
            name in cell.loads or any(
                not (cell is home and _inside(text, func, cell)) and not _command_word_constant(text, cell)
                for text in cell.signs.spelled.get(name, ())
            )
            for cell in self.cells for name in names
        )

    @cached_property
    def makers(self) -> dict[str, list[_Maker]]:
        """The defs, classes and imports each cell binds a name by, indexed
        once per check by the name (`_Maker`): a cell's latest def and class
        of the name, and each import of it, so a read asks only the cells
        that bind its name (`_filled_again`)."""
        out: dict[str, list[_Maker]] = {}
        for cell in self.cells:
            for table in (cell.functions, cell.classes):
                for name, made in table.items():
                    out.setdefault(name, []).append(_Maker(cell=cell, statement=made))
            for stmt in cell.signs.imports:
                for name in _import_names(stmt):
                    out.setdefault(name, []).append(_Maker(cell=cell, statement=stmt))
        return out

    @cached_property
    def module_table_names(self) -> frozenset[str]:
        """The bare names a cell may bind to `sys.modules`, so a store into a
        mapping one holds may store a module (`_stored_modules`): one
        assigned a value spelled `.modules`, or one `modules` is imported as."""
        out: set[str] = set()
        for cell in self.cells:
            out.update(name for name, found in cell.bindings.items()
                       if any(b.how == "value" and _last_name(b.value) == "modules" for b in found))
            out.update(a.asname or a.name for stmt in cell.signs.imports if isinstance(stmt, ast.ImportFrom)
                       for a in stmt.names if a.name == "modules")
        return frozenset(out)

    @cached_property
    def looks_up_namespace(self) -> bool:
        """Whether a cell looks a name up in its own namespace (`globals()`,
        `vars()` or `locals()`) under a key the check can't read, as
        `globals()["run_" + step]` or `globals().get(name)` do, reads every
        entry of it, as `globals().values()` does (`_ENTRY_READS`), or hands
        it on whole (`_hands_on_namespace`): such a lookup may reach any
        function the cells define."""
        for cell in self.cells:
            for call in cell.calls:
                if _last_name(call.func) not in _GLOBALS_CALLS or not _cell_namespace(call, self):
                    continue
                parent = cell.parents.get(call)
                if isinstance(parent, ast.Subscript) and not isinstance(parent.slice, ast.Constant):
                    return True
                if isinstance(parent, ast.Attribute) and parent.attr in _ENTRY_READS:
                    used = cell.parents.get(parent)
                    if not (isinstance(used, ast.Call) and used.args and isinstance(used.args[0], ast.Constant)):
                        return True
                if _hands_on_namespace(call, parent):
                    return True
        return False

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
        (`cuts` past `mark`), `read` reads again with no limit (`read_on`)
        and says whether that found what the first read missed. Then a write
        may be past the limit, unread, and the check stops (`stopped`).
        Nothing it finds goes on the card."""
        if self.read_on(mark, read):
            self.stopped = True

    def read_on(self, mark: int, read: Callable[[], _T]) -> _T | None:
        """What `read` gives read again with no limit, each node once, after
        a read a depth limit stopped (`cuts` past `mark`); None when no limit
        stopped it, or while a read already goes on past its limits. The
        wide reading, which only adds files to the first read's, doesn't
        read on."""
        if self.cuts == mark or self._past is not None or self.wide:
            return None
        self._past = set()
        try:
            return read()
        finally:
            self._past = None

    @overload
    def remembered(self, read: str, node: ast.AST, *, depth: int, compute: Callable[[], list[_T]]) -> list[_T]: ...

    @overload
    def remembered(
        self, read: str, node: ast.AST, *, depth: int, compute: Callable[[], list[_T] | None],
    ) -> list[_T] | None: ...

    def remembered(
        self, read: str, node: ast.AST, *, depth: int, compute: Callable[[], list[_T] | None],
    ) -> list[_T] | None:
        """What `compute`, a read that gives a list, gives (`recalled`), as a
        list of the caller's own."""

        def kept() -> list[_T] | None:
            value = compute()
            return None if value is None else list(value)

        got = self.recalled(read, node, depth=depth, compute=kept)
        return None if got is None else list(got)

    def recalled(self, read: str, node: ast.AST, *, depth: int, compute: Callable[[], _T]) -> _T:
        """What `compute`, the read named `read` of `node` at `depth`, gives,
        read once per check and set of call arguments: a name rebound from
        itself, as `cmd = cmd + ["-l"]` is, reads each earlier binding again
        at every later one. A recalled read counts its stops again, so a read
        that stopped short still says so. What a read read ahead of a name
        rebuilt from itself (`reading_ahead`) it takes as read. While a read
        goes on past its limits (`read_past`), it reads afresh."""
        ahead = self._read_ahead.get((read, id(node), depth))
        if ahead is not None:
            self.cuts += ahead.cuts
            return ahead.value
        if self._past is not None:
            return compute()
        key = (read, id(node), depth, self.call_key())
        hit = self._remembered.get(key)
        if hit is None:
            mark = self.cuts
            hit = self._remembered[key] = _Remembered(value=compute(), cuts=self.cuts - mark)
        else:
            self.cuts += hit.cuts
        return hit.value

    def read_bindings(
        self, read: str, node: ast.Name, *, depth: int, compute: Callable[[_Binding, int], _T],
    ) -> list[_T]:
        """What `compute`, the read named `read`, gives for each binding the
        name `node` reads (`bindings_of`), from each cell's last value that
        surely ran (`_latest`), handed the depth to read the binding's value
        at: the same for a value built on the one before it, as `out =
        os.path.join(out, "a")` is (`_rebuilt`), so a chain of them reads to
        its end, else one deeper. Read once per check and depth
        (`remembered`), down a name rebuilt many times first, in a loop
        (`reading_ahead`), so it recurses no deeper than the chain's other
        names."""

        def through() -> list[_T]:
            found = _latest(self.bindings_of(node.id, node=node))
            with self.reading_ahead(read, node, depth=depth, compute=lambda load: self.read_bindings(
                read, load, depth=depth, compute=compute,
            )):
                return [compute(b, depth if _rebuilt(b, node.id) else depth + 1) for b in found]

        return self.remembered(read, node, depth=depth, compute=through)

    @contextmanager
    def reading_ahead(
        self, read: str, node: ast.Name, *, depth: int, compute: Callable[[ast.Name], object],
    ) -> Iterator[None]:
        """While the read named `read` of the name `node` at `depth` runs:
        what `compute` gives at each load of the name in the value of a
        binding built on the value before it that the read reaches
        (`rebuilt_loads`), read first, in a loop, earliest first, and taken
        as read at that load (`recalled`). So a name rebuilt in many cells,
        as `cmd = cmd + " -v"` is, reads each binding once, with no deep
        recursion. A read that reads ahead of the name already reads no more
        ahead."""
        key = (read, node.id)
        if key in self._reading_ahead:
            yield
            return
        self._reading_ahead.add(key)
        added: list[tuple[str, int, int]] = []

        def known(load: ast.Name) -> bool:
            return self._past is None and (read, id(load), depth, self.call_key()) in self._remembered

        try:
            for load in self.rebuilt_loads(node, known=known):
                mark = self.cuts
                value = compute(load)
                added.append((read, id(load), depth))
                self._read_ahead[added[-1]] = _Remembered(value=value, cuts=self.cuts - mark)
            yield
        finally:
            self._reading_ahead.discard(key)
            for done in added:
                self._read_ahead.pop(done, None)

    def rebuilt_loads(self, node: ast.Name, *, known: Callable[[ast.Name], bool]) -> list[ast.Name]:
        """The loads of `node`'s name in the value of each binding `node`
        reads that is built on the value before it (`_rebuilt`), as the
        right side of `cmd = cmd + " -v"` is, and in turn in the value of
        each such binding those loads read, earliest first (`position`). A
        load `known` holds is read already, and so are those before it."""
        found: dict[int, ast.Name] = {}
        todo = [node]
        while todo:
            current = todo.pop()
            for b in _latest(self.bindings_of(current.id, node=current)):
                if not _rebuilt(b, current.id):
                    continue
                for load in b.cell.loads_within(b.value, current.id):
                    if load is not node and id(load) not in found and not known(load):
                        found[id(load)] = load
                        todo.append(load)
        return sorted(found.values(), key=self.position)

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

    def param_values(self, name: str, binding: _Binding, *, depth: int) -> _ParamValues | None:
        """What the parameter `name` of `binding`'s function can hold, as
        `_param_paths` and `_param_texts` read it: what the call being read
        passes for it, else what each call site passes, None for one the
        check can't read. None past `_PARAM_DEPTH` names, where the read of
        call sites stops (`stops`)."""
        if self.given(binding.value, name):
            return _ParamValues(args=(self.param_args[(id(binding.value), name)],), passed=True)
        if self.stops(f"param {name}", binding.value, depth=depth, limit=_PARAM_DEPTH - 1):
            return None
        return _ParamValues(args=tuple(self.call_sites(binding.value, name)), passed=False)

    def _param_paths(self, name: str, binding: _Binding, *, depth: int) -> list[str] | _Unknown:
        """A parameter: what the call being read passes for it, else what
        every call site passes (`param_values`). A parameter met at a call
        site is unknown."""
        values = self.param_values(name, binding, depth=depth)
        if values is None or not values.args:
            return _UNKNOWN
        # A call site's own parameters are not read on.
        at = depth + 1 if values.passed else _PARAM_DEPTH
        out: list[str] = []
        for arg in values.args:
            got = _UNKNOWN if arg is None else self.paths(arg.node, cell=arg.cell, depth=at)
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
        found = self._name_changes.get(name, ()) if binding.scope is None else [
            (binding.cell, change) for change in binding.cell.grows.get(name, ())
        ]
        for cell, grown in found:
            if self._binding_scope(cell, grown.site, name) is binding.scope and not _ran_before(
                grown.site, binding, cell=cell,
            ):
                out.append((grown, cell))
        return out

    @cached_property
    def _name_changes(self) -> dict[str, list[tuple[_Cell, _Growth]]]:
        """Each place any cell adds to what a name holds (`_Cell.grows`), by
        the name, in the order of the cells: built once per check, so a read
        of one name's additions doesn't walk every cell."""
        out: dict[str, list[tuple[_Cell, _Growth]]] = {}
        for cell in self.cells:
            for name, changes in cell.grows.items():
                out.setdefault(name, []).extend((cell, change) for change in changes)
        return out

    @cached_property
    def _cells_naming(self) -> dict[str, list[_Cell]]:
        """The cells that bind another name to a name's value or pass the name
        to a call (`_Cell.aliases`, `_Cell.passed`), by the name, in the order
        of the cells: built once per check (`list_growth`)."""
        out: dict[str, list[_Cell]] = {}
        for cell in self.cells:
            for name in {*cell.aliases, *cell.passed}:
                out.setdefault(name, []).append(cell)
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
        for cell in self._cells_naming.get(name, ()) if binding.scope is None else [binding.cell]:
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
        cell = self.cell_of(node)
        # A cell whose names don't land in the cell namespace calls its own first.
        own = None if cell.shared or not isinstance(node, ast.Name) else cell.functions.get(node.id)
        if not isinstance(node, ast.Name) or node.id not in self.function_names and own is None:
            return []
        lambdas = [b.value for b in self.bindings_of(node.id, node=node)
                   if b.how == "value" and isinstance(b.value, ast.Lambda)]
        func = own or self.functions.get(node.id)
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
            ran = cell.order < here.order or cell is here and grown.site.lineno < getattr(use, "lineno", 0)
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
        on it adds elements rather than text, read once per check
        (`recalled`). It stops at the depth limit (`stops`)."""
        return self.recalled("sequence", node, depth=depth, compute=lambda: self._is_sequence(node, depth=depth))

    def _is_sequence(self, node: ast.AST, *, depth: int) -> bool:
        """What `is_sequence` reads `node` to be. The leftmost side of a `+`
        chain, and the value before a binding built on it, as `cmd = cmd +
        ["-l"]` is, are read in a loop at the same depth: their code is this
        value's own, and a name read already (`recalled`) gives what it gave.
        Any other name's value costs a level."""
        start = node
        todo, seen = [node], set()
        while todo:
            node = todo.pop()
            if id(node) in seen or self.stops("sequence", node, depth=depth, limit=6):
                continue
            seen.add(id(node))
            hit = None if node is start or self._past is not None else self._remembered.get(
                ("sequence", id(node), depth, self.call_key()),
            )
            if hit is not None:
                self.cuts += hit.cuts
                if hit.value:
                    return True
                continue
            if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.ListComp, ast.SetComp)):
                return True
            if isinstance(node, ast.Call):
                name = self.dotted(node.func)
                if name in ("list", "set", "tuple", "sorted", "glob.glob", "os.listdir") or (
                    isinstance(node.func, ast.Attribute) and node.func.attr == "split"
                ):
                    return True
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    node = node.left
                todo.append(node)
            elif isinstance(node, ast.Name):
                for b in _latest(self.bindings_of(node.id, node=node)):
                    if b.how != "value":
                        continue
                    if _rebuilt(b, node.id):
                        todo.append(b.value)
                    elif self.is_sequence(b.value, cell=b.cell, depth=depth + 1):
                        return True
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
            if (self.import_of(node) or node.id) == "duckdb":
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
            made = self.function_named(node.func) if isinstance(node.func, ast.Name) else None
            if made is not None:
                returns = [n.value for n in ast.walk(made) if isinstance(n, ast.Return) and n.value is not None]
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
    a `*args` or `**kwargs`, or with no call at all. A `*args` parameter gets
    the positional arguments past the others as one tuple
    (`_Cell.spread_args`); a `**kwargs` one is one the check can't read.
    With `bound`, a method's first parameter gets it and the call's
    arguments fill the rest."""
    a = func.args
    positional = [*a.posonlyargs, *a.args]
    defaults = dict(zip([p.arg for p in positional][len(positional) - len(a.defaults):], a.defaults, strict=True))
    defaults.update({p.arg: d for p, d in zip(a.kwonlyargs, a.kw_defaults, strict=True) if d is not None})
    out: dict[str, _Arg | None] = {}
    if bound is not None and positional:
        out[positional[0].arg] = _Arg(node=bound, cell=call_cell)
        positional = positional[1:]
    if call is None:
        return {**out, **{p.arg: None for p in (*positional, *a.kwonlyargs, *_spread_params(a))}}
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
    if a.vararg is not None:
        # The positional arguments past the named ones, or from a spread one
        # on where one comes sooner.
        rest = call_cell.spread_args(call, min(len(positional), starred))
        out[a.vararg.arg] = None if rest is None else _Arg(node=rest, cell=call_cell)
    if a.kwarg is not None:
        out[a.kwarg.arg] = None
    return out


def _spread_params(args: ast.arguments) -> list[ast.arg]:
    """A function's `*args` and `**kwargs` parameters, those it has."""
    return [p for p in (args.vararg, args.kwarg) if p is not None]


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
            root.id in _LOGGER_NAMES or reader.import_of(root) == "logging"
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
    made = reader.function_named(func) if isinstance(func, ast.Name) else None
    return made is not None and reader.runs_locally(made)


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
    parameter `call` passes `value` for, a `*args` one among them, which
    gets it as an item of the tuple `_call_arguments` gives it; empty for
    any other call."""
    func = reader.function_named(call.func) if isinstance(call.func, ast.Name) else None
    if func is None:
        return []
    passed = _call_arguments(func, call, call_cell=reader.cell_of(call), func_cell=reader.cell_of(func))
    spread = func.args.vararg.arg if func.args.vararg is not None else None
    params = [name for name, arg in passed.items() if arg is not None and (
        arg.node is value or name == spread and isinstance(arg.node, ast.Tuple) and any(e is value for e in arg.node.elts)
    )]
    return [use for name in params for use in _loads_in(func.body, name)]


def _name_uses(names: set[str], *, at: ast.AST, reader: _Reader) -> list[ast.AST]:
    """Where a value bound to `names` at `at` is read: in its cell, inside the
    pad functions this cell calls (a module-level value is their global),
    and as an attribute when `at` sits in a class body."""
    cell = reader.cell_of(at)
    uses: list[ast.AST] = [u for n in names for u in cell.loads.get(n, ())]
    scope = cell.scope_of(at)
    if scope is None:
        for func in reader.called.values():
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
        call = reader.returned_to(func) or _mapping_call(func, cell, reader)
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
    for use in (u for uses in cell.loads.values() for u in uses) if root is None else cell.loads_in(root):
        if id(use) in read:
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
    `Path.write_text`, a DataFrame or chart writer, a `pd.ExcelWriter`, a
    writer that takes the object before its file (`_WRITERS_OF_AN_OBJECT`),
    as `torch.save(obj, f)` does, or a `.save(path)`."""
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
    elif name in _WRITERS_OF_AN_OBJECT:
        target, kind = _WRITERS_OF_AN_OBJECT[name].given(call), "replace"
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
    """The path `call` writes: a file it writes into or creates, the
    destination of a rename, move or copy, or the link a symlink or hardlink
    makes, which an import reaches like the file it points at."""
    name = reader.dotted(call.func)
    if name in _MOVE_FUNCS or name in _COPY_FUNCS:
        return _move_destination(call)
    if name in _LINK_FUNCS:
        return call.args[1] if len(call.args) > 1 else _keyword(call, "dst")
    if isinstance(call.func, ast.Attribute) and call.func.attr in _LINK_METHODS:
        return call.func.value
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
    made = reader.function_named(node.func) if isinstance(node.func, ast.Name) else None
    if made is not None:
        return [f for n in ast.walk(made) if isinstance(n, ast.Return) and n.value is not None
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
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        # An item of what a parameter is passed, as `stmts[0]` of `*stmts`,
        # or of the list or dict a name holds, as `Q["purge"]`.
        items = reader.items_of(node=node.value, key=node.slice, depth=depth + 1) or ()
        return tuple(dict.fromkeys((
            *_parameter_items(node.value, reader, depth=depth, key=node.slice),
            *(s for item in items for s in _written_statements(item, reader, depth=depth + 1)),
        )))
    if not isinstance(node, ast.Name) or reader.stops("statements", node, depth=depth, limit=4):
        return ()
    found: list[str] = []
    for b in reader.bindings_of(node.id, node=node):
        if b.how == "element" and isinstance(b.value, ast.Name):
            # A loop over a parameter, as `for s in stmts` of `*stmts`.
            found += _parameter_items(b.value, reader, depth=depth + 1)
        if b.how in ("element", "unpacked") and isinstance(b.value, ast.Call):
            # A loop over a dict's entries, as `for name, sql in Q.items()`.
            for item in _walked_items(b.value, reader, depth=depth + 1) or ():
                found += _written_statements(item, reader, depth=depth + 1)
        if b.how != "param":
            found += reader.statements_held(node.id, b, prose=False)
            continue
        args = reader.args_for(b.value, node.id)
        for a in args:
            if a is not None:
                found += _written_statements(a.node, reader, depth=depth + 1)
    return tuple(dict.fromkeys(found))


def _parameter_items(
    node: ast.Name, reader: _Reader, *, depth: int, key: ast.AST | None = None,
) -> tuple[str, ...]:
    """The statements with a write that each item of what the parameter
    `node` reads is passed holds (`_written_statements`): the items of the
    tuple a `*args` one gets (`_call_arguments`), or of a list, tuple or
    dict another is passed, written out or held in a name
    (`_Reader.items_of`), only the one a literal `key` names when it says
    which, from the call being read or each call site
    (`_Reader.args_for`). Empty where `node` reads no parameter."""
    found: list[str] = []
    for b in reader.bindings_of(node.id, node=node):
        if b.how != "param":
            continue
        for a in reader.args_for(b.value, node.id):
            items = None if a is None else reader.items_of(node=a.node, key=key, depth=depth + 1)
            for item in items or ():
                found += _written_statements(item, reader, depth=depth + 1)
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
                if cell.order > stopped.order or any(
                    cell.scope_of(n) is not None for n in cell.loads[name]
                ):
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
        exec_read = _exec_cells(reader, unerrored, ran_only=True)
        if exec_read.cells:
            reader = _Reader([*cells, *exec_read.cells], current=after, errored=errored, errored_paths="unknown")
        changes: list[FileChange] = []

        def read_function(callee: _Callee, *, cell: _Cell, at: ast.AST, chain: tuple[int, ...]) -> None:
            """`callee`, called from the code of `cell` at `at`, read for
            what it overwrites, with the literal code it passes to exec(),
            and each function either calls in turn."""
            func = callee.func
            if callee.call is None or callee.past or id(func) in chain or len(chain) >= _CALL_DEPTH:
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
    return home.order < cell.order


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
            if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load) and (
                _last_name(node) in _LAUNCH_NAMES and reader.module_member(node) in _LAUNCHERS
                or _partial_launcher(node, reader)
            ):
                # A launcher, or a partial of one, handed to a caller that
                # runs it later, read as each call the caller makes of it
                # (`_passed_calls`).
                passed = _passed_calls(node, cell, reader)
                for call in () if isinstance(passed, _Unknown) else passed:
                    self.shell(call)
            if self.lookup(node=node):
                continue
            if not isinstance(node, ast.Call):
                continue
            partials = reader.partials_of(node=node.func) if isinstance(node.func, (ast.Name, ast.Call)) else []
            # A call through functools.partial, read as each call it makes.
            for made in [_partial_call(partial=partial, call=node) for partial in partials] or [node]:
                self.call_change(made, database_writes)
        for call in sorted(database_writes, key=lambda c: (c.lineno, c.col_offset)):
            self.database_write(call)

    def call_change(self, node: ast.Call, database_writes: list[ast.Call]) -> None:
        """The change `node`, a call `walk` reaches, makes to a file: a
        delete, a move, a rename, an overwrite, or what a program it runs
        does (`shell`); a call that runs SQL goes on `database_writes`, read
        once the walk is done."""
        reader, cell = self.reader, self.cell
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
        cwds = _run_folders(call, self.reader, self.cell)
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


def _run_folders(call: ast.Call, reader: _Reader, cell: _Cell) -> Sequence[str] | _Unknown:
    """The folders `call`, code of `cell` that runs a program, runs it in:
    each folder the pad's code may run in (`_Reader.cwds`), joined with the
    `cwd=` the call gives, if any; unknown where the check can't place one."""
    cwd = _keyword(call, "cwd")
    cwds = reader.cwds
    if cwd is None or isinstance(cwds, _Unknown):
        return cwds
    got = reader.paths(cwd, cell=cell)
    return _UNKNOWN if isinstance(got, _Unknown) else [posixpath.normpath(_join(c, p)) for c in cwds for p in got]


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
    """Each command `node` can hold: a list of words, or a script's text,
    read once per check (`_Reader.remembered`). It stops at the depth limit
    (`_Reader.stops`)."""
    return reader.remembered("commands", node, depth=depth, compute=lambda: _node_commands(node, reader, depth=depth))


def _node_commands(node: ast.AST, reader: _Reader, *, depth: int) -> list[str | list[str]]:
    """The commands `_commands` reads `node` to hold."""
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
        # Whether a binding of the name gives a value the check reads no
        # command from, as `cmd = todo.pop()` does.
        unread = False
        # A binding built on the value before it reads that value itself, at
        # this depth, so a name rebuilt from itself is read down its
        # bindings first, in a loop (`_Reader.reading_ahead`).
        with reader.reading_ahead("commands", node, depth=depth,
                                  compute=lambda load: _commands(load, reader, depth=depth)):
            for b in _latest(reader.bindings_of(node.id, node=node)):
                here = depth if _rebuilt(b, node.id) else depth + 1
                walked = _walked_items(b.value, reader, depth=depth) if b.how in ("element", "unpacked") else None
                if b.how == "param":
                    # What each call of the function passes for it; a command the
                    # check can't read where one call passes it may be any.
                    for site in reader.call_sites(b.value, node.id):
                        passed = [] if site is None else _commands(site.node, reader, depth=depth + 1)
                        out += passed or [[_NOT_LITERAL]]
                elif b.how == "value" and (
                    isinstance(b.value, (ast.List, ast.Tuple))
                    or _word_lists(node=b.value, reader=reader, depth=here) is not None
                ):
                    grown = reader.list_growth_for_write(name=node.id, binding=b)
                    for words in _commands(b.value, reader, depth=here):
                        if isinstance(words, list) and (isinstance(grown, _Unknown) or grown):
                            # The words the cell adds to the list, or words the check
                            # can't read, the program itself where a change may move it.
                            added = None if isinstance(grown, _Unknown) else _grown_words(words, grown, reader,
                                                                                          depth=depth)
                            if added is not None:
                                out += added
                            elif not isinstance(grown, _Unknown) and any(_moves_program(g) for g, _ in grown):
                                out.append([_NOT_LITERAL])
                            else:
                                out.append([*words, *_UNREAD_WORDS])
                        else:
                            out.append(words)
                elif b.how == "value" and isinstance(b.value, ast.Name):
                    # Another name for the command, as `c1 = c0` gives one.
                    out += _commands(b.value, reader, depth=depth + 1)
                elif walked:
                    # Each command a loop walks.
                    out += [c for item in walked for c in _commands(item, reader, depth=depth + 1)]
                else:
                    held = [f for t in _held_texts(node.id, b, reader, depth=0) for f in _program_filled(t, reader)]
                    if not held and b.how == "value":
                        held = _derived_commands(b.value, reader, depth=here) or []
                    unread = unread or not held and not isinstance(b.value, ast.Constant)
                    out += held
        if unread and out:
            # The command that binding gives may be any, beside those the
            # others give: on its own, the caller reads a name that gives no
            # command as one it can't read.
            out.append([_NOT_LITERAL])
        return _capped(out)
    text = _command_text(node, reader)
    if text is None:
        return _derived_commands(node, reader, depth=depth) or []
    return list(_program_filled(text, reader))


# Methods of a text that give back a command read from it: the same words,
# or the same words in another case.
_SAME_COMMAND_METHODS = frozenset({"strip", "lstrip", "rstrip"})
_CASE_METHODS = {"lower": str.lower, "upper": str.upper, "casefold": str.casefold}


def _derived_commands(node: ast.AST, reader: _Reader, *, depth: int) -> list[str | list[str]] | None:
    """The commands a value taken from other commands holds, read from
    those (`_commands`): an entry of a list or dict of them
    (`_Reader.items_of`), by `[k]`, `.get(k)` with its default, or a
    list's `.pop()`; each arm of an if-else; a text's `strip` and its kin,
    or its `lower`, `upper` or `casefold` in that case; `str` of a text and
    `list` or `tuple` of a list; a full slice, the same command; and any
    other slice of a list, or a comprehension that keeps some of its
    words, as each of the list's tails, since either may drop the words
    before any of them. Empty where a part holds no command the check
    reads; None for any other value."""
    parts: list[ast.AST]
    if isinstance(node, ast.IfExp):
        parts = [node.body, node.orelse]
    elif isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Slice):
        items = reader.items_of(node=node.value, key=node.slice, depth=depth + 1)
        if items is None:
            return []
        parts = items
    elif isinstance(node, ast.Subscript):
        whole = node.slice.lower is None and node.slice.upper is None and node.slice.step is None
        sliced = _commands(node.value, reader, depth=depth + 1)
        if whole:
            return sliced
        return [words[at:] for words in sliced if isinstance(words, list) for at in range(len(words))]
    elif (isinstance(node, (ast.ListComp, ast.GeneratorExp)) and len(node.generators) == 1
          and isinstance(node.elt, ast.Name) and isinstance(node.generators[0].target, ast.Name)
          and node.elt.id == node.generators[0].target.id):
        walked = node.generators[0].iter
        # The first iterable runs where the comprehension does, so a name
        # there reads what it holds before the comprehension.
        found = _latest(reader.bindings_of(walked.id, node=node)) if isinstance(walked, ast.Name) else None
        if found is not None and (not found or any(b.how != "value" for b in found)):
            return []
        held = [walked] if found is None else [b.value for b in found]
        lists = [c for value in held for c in _commands(value, reader, depth=depth + 1) if isinstance(c, list)]
        return [words[at:] for words in lists for at in range(len(words))]
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and (
        node.func.attr in ("pop", "get") and len(node.args) <= 2 and not node.keywords
    ):
        key = node.args[0] if node.args else None
        items = reader.items_of(node=node.func.value, key=key, depth=depth + 1)
        if items is None:
            return []
        parts = [*items, *node.args[1:]] if node.func.attr == "get" else items
    elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and not node.args
          and not node.keywords and node.func.attr in (*_SAME_COMMAND_METHODS, *_CASE_METHODS)):
        texts = [c for c in _commands(node.func.value, reader, depth=depth + 1) if isinstance(c, str)]
        change = _CASE_METHODS.get(node.func.attr, str.strip)
        return [change(text) for text in texts]
    elif isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords and (
        reader.dotted(node.func) in ("str", "list", "tuple")
    ):
        kind = str if reader.dotted(node.func) == "str" else list
        return [c for c in _commands(node.args[0], reader, depth=depth + 1) if isinstance(c, kind)]
    else:
        return None
    out: list[str | list[str]] = []
    for part in parts:
        got = _commands(part, reader, depth=depth + 1)
        if not got:
            return []
        out += got
    return out


def _capped(commands: list[str | list[str]]) -> list[str | list[str]]:
    """`commands`, as a name's bindings give them, each once; past
    `_COMMAND_LIMIT` different word lists, those lists as one: the words
    they all start with, then `_SOME_WORDS` and every word any of them has
    after those. A name rebuilt under many ifs, as `cmd = cmd + ["-v"]` is,
    gives a list for each choice of them. Lists one word apart, as a loop
    over a literal list of names gives, stay as they are: there are only as
    many as the code writes out."""
    unique = list(dict.fromkeys(tuple(c) if isinstance(c, list) else c for c in commands))
    lists = [list(c) for c in unique if isinstance(c, tuple)]
    if len(lists) <= _COMMAND_LIMIT or _one_word_apart(lists):
        return [list(c) if isinstance(c, tuple) else c for c in unique]
    return [*(c for c in unique if isinstance(c, str)), _collapsed(lists)]


def _collapsed(lists: list[list[str]]) -> list[str]:
    """`lists`, a command's word lists, as one: the words they all start
    with, then `_SOME_WORDS` and every word any of them has after those."""
    prefix = os.path.commonprefix(lists)
    return [*prefix, _SOME_WORDS, *dict.fromkeys(word for words in lists for word in words[len(prefix):])]


def _one_word_apart(lists: list[list[str]]) -> bool:
    """Whether `lists`, a command's word lists, are the same length and
    differ at one place at most."""
    return len({len(words) for words in lists}) == 1 and sum(len(set(at)) > 1 for at in zip(*lists)) <= 1


def _listed_commands(node: ast.List | ast.Tuple, reader: _Reader, *, depth: int) -> list[list[str]]:
    """The commands a list of words written out gives: each word as
    `_command_word` reads it, a starred list's words in its place, words
    the check can't read (`_UNREAD_WORDS`) for a starred value it can't
    read, and a word read through a name as each literal text it can hold
    (`_literal_texts`): one command for each text of the first such word,
    then up to `_COMMAND_LIMIT` commands; so none when the program's word
    holds none, or a word or a spread reads a parameter of a function
    nothing calls (`_never_given`). Code handed to a `-c` stays a word the
    check can't read: code through a name isn't read as a literal."""
    commands: list[list[str]] = [[]]
    for at, e in enumerate(node.elts):
        if isinstance(e, ast.Starred):
            items = reader.items_of(node=e.value, key=None, depth=depth + 1)
            words = [_command_word(i, reader) for i in items] if items is not None else None
            if words is None or any(_unreadable_word(w) for w in words):
                # Words spread from a sequence of literal texts, as a list
                # of the packages a cell finds missing gives.
                texts = _sequence_texts(e.value, reader)
                if not texts and _never_given(e.value, reader):
                    # Words a function nothing calls spreads from its
                    # parameters: the command never runs.
                    return []
                words = texts if texts is not None else words or [*_UNREAD_WORDS]
            commands = [[*c, *words] for c in commands]
            continue
        word = _command_word(e, reader)
        code = at > 0 and bool(_CODE_FLAG.fullmatch(commands[0][-1] if commands[0] else ""))
        texts = _literal_texts(e, reader) if _unreadable_word(word) and not code else None
        if texts == [] and _never_given(e, reader):
            # A word a function nothing calls takes from its parameters: the
            # command never runs.
            return []
        # One command for each text of the first word that holds many, as a
        # loop over a literal list of names gives; past that, up to the limit.
        fits = texts and (len(commands) == 1 or len(commands) * len(texts) <= _COMMAND_LIMIT)
        options = list(dict.fromkeys(texts)) if fits else [word]
        commands = [[*c, option] for c in commands for option in options]
    return commands


def _word_lists(*, node: ast.AST, reader: _Reader, depth: int) -> list[list[str]] | None:
    """The word lists an expression that builds one can give a command:
    `text.split()`, `shlex.split(text)`, or lists added with `+`, a part the
    check can't read words it can't read (`_UNREAD_WORDS`). The sides of a
    `+` chain are this expression's own code, so reading them costs no
    depth. A part a stop at the depth limit left empty gives none, and
    `_shell_commands` holds one past the limit. None for any other
    expression."""
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
        # Flattened with a stack, so a long chain can't exhaust recursion.
        sides: list[ast.AST] = []
        stack: list[ast.AST] = [node]
        while stack:
            item = stack.pop()
            if isinstance(item, ast.BinOp) and isinstance(item.op, ast.Add):
                stack += [item.right, item.left]
            else:
                sides.append(item)
        words: list[list[str]] = [[]]
        for side in sides:
            mark = reader.cuts
            lists = [c for c in _commands(side, reader, depth=depth) if isinstance(c, list)]
            if not lists and reader.cuts > mark:
                return []
            if not lists:
                # A sequence of literal texts, as a comprehension of the
                # missing packages or `sorted(names)` gives.
                texts = _sequence_texts(side, reader, depth=depth + 1)
                lists = [[*_UNREAD_WORDS] if texts is None else texts]
            if len(words) > 1 and len(lists) > 1:
                # Many lists on both sides: a side past the limit, as a loop
                # over a long literal list gives, joins the other as one list.
                words = [_collapsed(words)] if len(words) > _COMMAND_LIMIT else words
                lists = [_collapsed(lists)] if len(lists) > _COMMAND_LIMIT else lists
            words = [left + right for left in words for right in lists]
        return words
    return None



def _literal_texts(node: ast.AST, reader: _Reader, *, depth: int = 0) -> list[str] | None:
    """The literal texts `node` can hold, as a program's or a module's name:
    a literal; each name's values (`_name_texts`); either side of an if-else
    and each value of an `or`; each text of one side of a `+` joined to each
    of the other's, up to `_COMMAND_LIMIT`; the entries of a list or dict it
    looks up (`_Reader.items_of`); what `shutil.which` or `str` is handed;
    and any other value `_Reader.paths` reads whole. Empty when it holds
    none: a `None`, or a parameter of a function nothing calls. None when a
    part is a value the check can't read, or past the depth limit (`stops`).
    Read once per check (`_Reader.remembered`)."""
    return reader.remembered("texts", node, depth=depth, compute=lambda: _node_texts(node, reader, depth=depth))


def _node_texts(node: ast.AST, reader: _Reader, *, depth: int) -> list[str] | None:
    """The literal texts `_literal_texts` reads `node` to hold."""
    if reader.stops("texts", node, depth=depth, limit=_PARAM_DEPTH):
        return None
    parts: Sequence[ast.AST]
    if isinstance(node, ast.Constant):
        # A `None`, as a parameter's default, holds no text: `pkg or name`
        # gives `name` where the call passes no `pkg`.
        return [node.value] if isinstance(node.value, str) else [] if node.value is None else None
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
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # The sides of a `+` chain, flattened with a stack so a long chain
        # can't exhaust recursion: their code is part of this tree, so
        # reading them costs no depth.
        sides: list[ast.AST] = []
        stack: list[ast.AST] = [node]
        while stack:
            item = stack.pop()
            if isinstance(item, ast.BinOp) and isinstance(item.op, ast.Add):
                stack += [item.right, item.left]
            else:
                sides.append(item)
        return _joined_texts(sides, reader, depth=depth)
    else:
        made = _made_texts(node, reader, depth=depth)
        if made is not None:
            return made
        got = reader.paths(node, cell=reader.cell_of(node))
        if isinstance(got, _Unknown) or any(_unreadable_text(t) for t in got):
            return None
        return got
    return _all_texts(parts, reader, depth=depth + 1)


def _key_values(key: ast.AST, reader: _Reader, *, depth: int) -> list[str | int] | None:
    """The keys or indexes a subscript's `key` can be: a literal whole
    number, a negative one among them (`_whole_number`), or each value a
    name is given that is one; else each literal text it can hold
    (`_literal_texts`), as a loop over literal keys or a parameter the
    calls pass texts gives. None where the check can't read them."""
    number = _whole_number(key)
    if number is not None:
        return [number]
    if isinstance(key, ast.Name):
        found = reader.bindings_of(key.id, node=key)
        numbers = [_whole_number(b.value) if b.how == "value" else None for b in found]
        if found and None not in numbers:
            return [n for n in numbers if n is not None]
    texts = _literal_texts(key, reader, depth=depth)
    return None if texts is None else list(texts)


def _whole_number(node: ast.AST) -> int | None:
    """The whole number `node` writes out, as `2` or `-1`; None for anything
    else, `True` among them."""
    negative = isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
    value = node.operand if isinstance(node, ast.UnaryOp) and negative else node
    if isinstance(value, ast.Constant) and type(value.value) is int:
        return -value.value if negative else value.value
    return None


def _joined_texts(parts: Sequence[ast.AST], reader: _Reader, *, depth: int) -> list[str] | None:
    """Each text of `parts` joined end to end, each part read through
    `_literal_texts`, up to `_COMMAND_LIMIT` texts; None past that, or when
    a part isn't literal texts."""
    out = [""]
    for part in parts:
        texts = _literal_texts(part, reader, depth=depth)
        if texts is None or len(out) * len(texts) > _COMMAND_LIMIT:
            return None
        out = [a + b for a in out for b in texts]
    return out


def _made_texts(node: ast.AST, reader: _Reader, *, depth: int) -> list[str] | None:
    """The texts an f-string whose fields hold literal texts makes, or a
    literal separator's `.join` of a sequence of them (`_sequence_texts`);
    None for anything else, which `_Reader.paths` reads."""
    if isinstance(node, ast.JoinedStr):
        parts: list[ast.AST] = []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                if value.format_spec is not None or value.conversion != -1:
                    return None
                parts.append(value.value)
            else:
                parts.append(value)
        return _joined_texts(parts, reader, depth=depth)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join"
            and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str)
            and len(node.args) == 1 and not node.keywords):
        items = _sequence_texts(node.args[0], reader, depth=depth + 1)
        return None if items is None else [node.func.value.value.join(items)]
    return None


# Calls that give back the elements of the one sequence they're handed, or
# some of them: `_ITER_WRAPPERS` but for those that give back one element
# or an iterator.
_SEQUENCE_CALLS = _ITER_WRAPPERS - {"next", "max", "min", "iter"}


def _sequence_texts(node: ast.AST, reader: _Reader, *, depth: int = 0) -> list[str] | None:
    """The literal texts the elements of the sequence `node` can be, as a
    `*` in a command spreads them: each element of a list, tuple or set
    (`_literal_texts`); each one a comprehension makes, its filters aside,
    since a filter only drops elements; what `list`, `sorted`, `filter` and
    their kin give of a sequence of them; a set difference's left side; the
    keys and values of a literal dict; and, for a name, each value it is
    given with what `.append`, `.extend`, `.add` or `+=` adds to it, or
    what the calls of its function pass. Empty for a parameter of a function
    nothing calls. None when the check can't read them, or past the depth
    limit (`stops`)."""
    if reader.stops("sequence", node, depth=depth, limit=_PARAM_DEPTH):
        return None
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: list[str] = []
        for e in node.elts:
            got = (_sequence_texts(e.value, reader, depth=depth + 1) if isinstance(e, ast.Starred)
                   else _literal_texts(e, reader, depth=depth + 1))
            if got is None:
                return None
            out += got
        return out
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)) and len(node.generators) == 1:
        loop = node.generators[0]
        if isinstance(node.elt, ast.Name) and isinstance(loop.target, ast.Name) and node.elt.id == loop.target.id:
            return _sequence_texts(loop.iter, reader, depth=depth + 1)
        return _literal_texts(node.elt, reader, depth=depth + 1)
    if isinstance(node, ast.Call) and not node.keywords:
        name = reader.dotted(node.func)
        if name in _SEQUENCE_CALLS and len(node.args) == 1:
            return _sequence_texts(node.args[0], reader, depth=depth + 1)
        if name == "filter" and len(node.args) == 2:
            return _sequence_texts(node.args[1], reader, depth=depth + 1)
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("keys", "values") and not node.args:
            items = _walked_items(node, reader, depth=depth + 1)
            return None if items is None else _all_texts(items, reader, depth=depth + 1)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return _sequence_texts(node.left, reader, depth=depth + 1)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.BitOr)):
        left = _sequence_texts(node.left, reader, depth=depth + 1)
        right = None if left is None else _sequence_texts(node.right, reader, depth=depth + 1)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.Name):
        return _name_sequence_texts(node, reader, depth=depth)
    return None


def _spreads(growth: _Growth) -> bool:
    """Whether `growth` adds each element of its value, as `extend`,
    `update`, `+=`, `|=` and a store into the list do, rather than the value
    itself, as `append`, `add` and `insert` do."""
    site = growth.site
    return isinstance(site, (ast.AugAssign, ast.Assign)) or (
        isinstance(site, ast.Call) and isinstance(site.func, ast.Attribute) and site.func.attr in ("extend", "update")
    )


def _moves_program(growth: _Growth) -> bool:
    """Whether `growth`, a change to a command's list, may put another word
    where its program is: an `insert`, or a store at an index, at the front
    or where the check can't tell."""
    site = growth.site
    if isinstance(site, ast.Call) and isinstance(site.func, ast.Attribute) and site.func.attr == "insert":
        where: list[ast.AST] = site.args[:1]
    elif isinstance(site, ast.Assign):
        where = [t.slice.lower if isinstance(t.slice, ast.Slice) else t.slice
                 for t in site.targets if isinstance(t, ast.Subscript)]
    else:
        return False
    return not where or any(
        not (isinstance(w, ast.Constant) and isinstance(w.value, int) and w.value > 0) for w in where
    )


def _grown_words(
    words: list[str], growth: Sequence[tuple[_Growth, _Cell]], reader: _Reader, *, depth: int,
) -> list[list[str]] | None:
    """The word lists `words`, a command's list where a name is bound to it,
    can be with the words the cell adds to the list in place after them, in
    the order they run: each element `extend` or `+=` adds
    (`_sequence_texts`), and an element `append` or `add` adds, read as a
    word of a list written out is (`_command_word`, `_literal_texts`), one
    list for each text it can hold, up to `_COMMAND_LIMIT` lists. None when
    an addition isn't literal texts; when it may move the program, as an
    `insert` or a store at an index may; or when it hands code to a `-c`
    through a name, which isn't read as a literal."""
    out = [list(words)]
    for grown, _ in sorted(growth, key=lambda g: reader.position(g[0].site)):
        site, value = grown.site, grown.value
        method = site.func.attr if isinstance(site, ast.Call) and isinstance(site.func, ast.Attribute) else None
        if grown.how != "add" or isinstance(site, ast.Assign) or method == "insert":
            return None
        if _spreads(grown):
            texts = _sequence_texts(value, reader, depth=depth + 1)
            options = None if texts is None else [texts]
        else:
            word = _command_word(value, reader)
            texts = [word] if not _unreadable_word(word) else _literal_texts(value, reader, depth=depth + 1)
            options = [[t] for t in dict.fromkeys(texts)] if texts else None
        literal = all(isinstance(n, (ast.Constant, ast.List, ast.Tuple, ast.Set)) for n in ast.walk(value)
                      if isinstance(n, ast.expr))
        if options is None or len(out) * len(options) > _COMMAND_LIMIT:
            return None
        out = [[*w, *option] for w in out for option in options]
        if not literal and any(_CODE_FLAG.fullmatch(w) for w in out[0]):
            return None
    return out


def _name_sequence_texts(node: ast.Name, reader: _Reader, *, depth: int) -> list[str] | None:
    """The literal texts the elements of the sequence the name `node` reads
    can be (`_sequence_texts`)."""
    arg = reader.argument_of(node)
    if arg is not None:
        return _sequence_texts(arg.node, reader, depth=depth + 1)
    found = reader.bindings_of(node.id, node=node)
    if not found:
        return None
    out: list[str] = []
    for b in found:
        if b.how == "param":
            sites = reader.call_sites(b.value, node.id)
            if not sites and not reader.uncalled(b.value):
                return None
            for site in sites:
                got = None if site is None else _sequence_texts(site.node, reader, depth=depth + 1)
                if got is None:
                    return None
                out += got
            continue
        if b.how != "value":
            return None
        got = _sequence_texts(b.value, reader, depth=depth + 1)
        grown = reader.list_growth(name=node.id, binding=b) if got is not None else _UNKNOWN
        if got is None or isinstance(grown, _Unknown):
            return None
        for growth, _ in grown:
            more = (None if growth.how != "add" else _sequence_texts(growth.value, reader, depth=depth + 1)
                    if _spreads(growth) else _literal_texts(growth.value, reader, depth=depth + 1))
            if more is None:
                return None
            got += more
        out += got
    return out


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
    call of the function passes. A function nothing calls passes none. A
    name rebuilt from itself is read down its bindings first, in a loop
    (`_Reader.reading_ahead`)."""
    cell = reader.cell_of(node)
    # `cmd = cmd + " -y"` reads the value before it.
    found = [b for b in reader.bindings_of(node.id, node=node)
             if not (b.cell is cell and b.how == "value" and _inside(node, b.value, cell))]
    if not found:
        return None
    out: list[str] = []
    with reader.reading_ahead("texts", node, depth=depth,
                              compute=lambda load: _literal_texts(load, reader, depth=depth)):
        for b in _latest(found):
            if b.how == "value":
                texts = _literal_texts(b.value, reader, depth=depth if _rebuilt(b, node.id) else depth + 1)
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
    each item it walks (`_walked_items`), or from a sequence it builds of
    literal texts, as a comprehension of the missing ones does
    (`_sequence_texts`)."""
    items = _walked_items(walked, reader, depth=depth)
    if items is None:
        return _sequence_texts(walked, reader, depth=depth)
    return _all_texts(items, reader, depth=depth)


def _walked_items(walked: ast.AST, reader: _Reader, *, depth: int) -> list[ast.AST] | None:
    """What a loop over `walked` gives its names: each element of a list or
    dict it walks (`_Reader.items_of`), each key and value of a dict's
    `.items()`, and each element `enumerate` pairs with its count. None
    when the check can't place them, as for a dict a cell fills in place
    (`_changed_in_place`)."""
    if isinstance(walked, ast.Call) and walked.args and reader.dotted(walked.func) == "enumerate":
        walked = walked.args[0]
    elif (isinstance(walked, ast.Call) and isinstance(walked.func, ast.Attribute) and not walked.args
          and walked.func.attr in ("items", "keys", "values")):
        held = walked.func.value
        found = reader.bindings_of(held.id, node=held) if isinstance(held, ast.Name) else []
        values = [b.value for b in found] if isinstance(held, ast.Name) else [held]
        if not values or not all(isinstance(v, ast.Dict) and None not in v.keys for v in values):
            return None
        if isinstance(held, ast.Name) and _changed_in_place(held.id, found, reader):
            return None
        return [part for v in values for part in (*v.keys, *v.values) if part is not None]
    return reader.items_of(node=walked, key=None, depth=depth)


# A dict's methods that read it and leave it as it is.
_DICT_READS = frozenset({"items", "keys", "values", "get", "copy"})


def _changed_in_place(name: str, found: Sequence[_Binding], reader: _Reader) -> bool:
    """Whether a cell may change in place what `name` holds, as `found`
    binds it: an addition `_Reader.list_growth` reads, a store to an entry,
    a call of a method that does more than read it (`_DICT_READS`), or a
    call it is handed to that may fill it, as `fill(jobs)` may, but for one
    that hands nothing back (`_is_harmless_call`)."""
    for b in found:
        grown = reader.list_growth(name=name, binding=b)
        if isinstance(grown, _Unknown) or grown:
            return True
    for cell in reader.cells:
        for load in cell.loads.get(name, ()):
            parent = cell.parents.get(load)
            if isinstance(parent, ast.Subscript) and isinstance(parent.ctx, ast.Store):
                return True
            if (isinstance(parent, ast.Attribute) and parent.attr not in _DICT_READS
                    and isinstance(cell.parents.get(parent), ast.Call)):
                return True
            call = cell.parents.get(parent) if isinstance(parent, ast.keyword) else parent
            if isinstance(call, ast.Call) and call.func is not load and not _is_harmless_call(call, reader):
                return True
    return False


def _param_texts(name: str, binding: _Binding, reader: _Reader, *, depth: int) -> list[str] | None:
    """The literal texts a parameter can hold (`_name_texts`), read from what
    `_Reader.param_values` gives: none for a function nothing calls."""
    values = reader.param_values(name, binding, depth=depth)
    if values is None:
        return None
    if not values.args:
        return [] if reader.uncalled(binding.value) else None
    out: list[str] = []
    for arg in values.args:
        texts = None if arg is None else _literal_texts(arg.node, reader, depth=depth + 1)
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
    names, and a path built from it as the program that path names
    (`_interpreter_text`, `_executable_filled`)."""
    built = _interpreter_text(node, reader)
    if built is not None:
        return built
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
        if options == [] and node is not None and _never_given(node, reader):
            # A part a function nothing calls takes from its parameters: the
            # command never runs.
            return []
        if options and len(texts) * len(options) <= _COMMAND_LIMIT:
            texts = [t.replace(m.group(), option, 1) for t in texts for option in dict.fromkeys(options)]
    return texts


def _executable_filled(text: str, reader: _Reader) -> str:
    """`text` with each placeholder for `sys.executable` read as `python`,
    and for a path built from it as that path (`_interpreter_text`)."""
    if _HOLE not in text:
        return text
    return _NUMBERED_HOLE.sub(lambda m: _interpreter_text(reader.hole_node(m.group()), reader) or m.group(), text)


def _names_executable(node: ast.AST | None, reader: _Reader) -> bool:
    """Whether `node` reads `sys.executable`, there or through a name bound
    to it (`_Reader.module_member`)."""
    return node is not None and (
        reader.dotted(node) == "sys.executable" or reader.module_member(node) == "sys.executable"
    )


# The folders of the running environment, which hold its interpreter and its
# other programs, as `bin/pip` under `sys.prefix` (`_interpreter_text`).
_PYTHON_FOLDERS = frozenset({"sys.prefix", "sys.exec_prefix", "sys.base_prefix", "sys.base_exec_prefix"})
# Calls that give back the one path they are handed, as a text or a path,
# and the methods of a path that do; the calls that join parts onto the path
# they are handed first, and those that give its folder.
_SAME_PATH_CALLS = frozenset({
    *_SAME_TEXT_CALLS, "os.fsdecode", "os.path.abspath", "os.path.realpath", "os.path.normpath",
    "os.path.expanduser",
})
_SAME_PATH_METHODS = frozenset({"resolve", "absolute", "expanduser", "as_posix", "__fspath__"})
_PATH_JOINS = frozenset({
    "os.path.join", "posixpath.join", "ntpath.join", "pathlib.Path", "pathlib.PurePath", "pathlib.PosixPath",
    "pathlib.WindowsPath",
})
_PATH_FOLDERS = frozenset({"os.path.dirname", "posixpath.dirname", "ntpath.dirname"})


def _interpreter_text(node: ast.AST | None, reader: _Reader, *, depth: int = 0) -> str | None:
    """The path `node` builds from the running interpreter's, as a program
    word reads it: `python` for `sys.executable` (`_names_executable`), "."
    for its folder or one of `_PYTHON_FOLDERS`, and literal parts joined onto
    one of those, so `os.path.join(os.path.dirname(sys.executable), "pip")`
    names pip. A call that gives back the path it is handed, a join, a
    path's `.parent`, `.with_name` and `.joinpath`, `/`, `+` and an f-string
    read through, and a name through the values its bindings give it where
    they all read the same, up to `_PARAM_DEPTH` names, each name read once
    per check (`_Reader.recalled`), as `cmd = cmd + " -v"` rebuilt many
    times asks it of every binding before it again. None for any other
    code."""
    if node is None or depth > _PARAM_DEPTH:
        return None
    if _names_executable(node, reader):
        return "python"
    if isinstance(node, ast.Name):

        def through_bindings() -> str | None:
            bindings = reader.bindings_of(node.id, node=node)
            if not bindings or any(b.how != "value" for b in bindings):
                return None
            # The first value that reads otherwise, or reads as no path,
            # settles it.
            first: str | None = None
            for b in bindings:
                text = _interpreter_text(b.value, reader, depth=depth + 1)
                if text is None or first is not None and text != first:
                    return None
                first = text
            return first

        return reader.recalled("interpreter", node, depth=depth, compute=through_bindings)
    if isinstance(node, ast.Attribute):
        if reader.dotted(node) in _PYTHON_FOLDERS or reader.module_member(node) in _PYTHON_FOLDERS:
            return "."
        base = _interpreter_text(node.value, reader, depth=depth) if node.attr == "parent" else None
        return None if base is None else posixpath.dirname(base) or "."
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
        # A chain of parts joined on its left, read from its first part on,
        # so a long one needs no deep recursion.
        joins: list[ast.BinOp] = []
        start: ast.AST = node
        while isinstance(start, ast.BinOp) and isinstance(start.op, (ast.Div, ast.Add)):
            joins.append(start)
            start = start.left
        base = _interpreter_text(start, reader, depth=depth)
        for join in reversed(joins):
            right = join.right
            part = right.value if isinstance(right, ast.Constant) and isinstance(right.value, str) else None
            if base is None or part is None:
                return None
            base = posixpath.join(base, part) if isinstance(join.op, ast.Div) else base + part
        return base
    if isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.FormattedValue):
        head, *rest = node.values
        base = _interpreter_text(head.value, reader, depth=depth) if head.conversion == -1 else None
        if base is None or head.format_spec is not None or not all(isinstance(v, ast.Constant) for v in rest):
            return None
        return base + "".join(str(v.value) for v in rest if isinstance(v, ast.Constant))
    if not isinstance(node, ast.Call) or node.keywords:
        return None
    func, parts = node.func, list(node.args)
    if isinstance(func, ast.Attribute) and func.attr in _SAME_PATH_METHODS:
        return None if parts else _interpreter_text(func.value, reader, depth=depth)
    if isinstance(func, ast.Attribute) and func.attr in ("with_name", "joinpath"):
        base = _interpreter_text(func.value, reader, depth=depth)
        if func.attr == "with_name" and base is not None:
            base = posixpath.dirname(base) or "."
    else:
        name = reader.module_member(func) or reader.dotted(func)
        if not parts or not (name in _SAME_PATH_CALLS or name in _PATH_JOINS or name in _PATH_FOLDERS):
            return None
        base, parts = _interpreter_text(parts[0], reader, depth=depth), parts[1:]
        if base is None or parts and name not in _PATH_JOINS:
            return None
        if name in _PATH_FOLDERS:
            return posixpath.dirname(base) or "."
    texts = [p.value for p in parts if isinstance(p, ast.Constant) and isinstance(p.value, str)]
    return None if base is None or len(texts) != len(parts) else posixpath.join(base, *texts)


def _script_commands(text: str, *, escapes: bool = True) -> list[list[str]]:
    """A shell script's commands, each the list of its words: split where a
    separator such as `;`, `&&`, `|`, a parenthesis or a backtick, or a line
    break, ends one, so a command `$(...)` or backticks substitute is one of
    them. A quoted line break stays in its word, and a word that starts with
    `#` starts a comment to the end of its line. A backslash escapes the
    character after it, as in sh, unless not `escapes`, as in cmd."""
    try:
        lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|()\n`")
        lexer.whitespace = " \t\r"
        lexer.commenters = ""
        lexer.whitespace_split = True
        if not escapes:
            lexer.escape = ""
        tokens = list(lexer)
    except ValueError:
        tokens = re.findall(r"\n|`|[^ \t\r\n`]+", text)
    commands: list[list[str]] = [[]]
    comment = False
    for token in tokens:
        if token in _SHELL_SEPARATORS or token and set(token) <= set(";&|()\n`"):
            if comment and "\n" not in token:
                continue
            comment = False
            commands.append([])
        elif comment or token.startswith("#"):
            comment = True
        else:
            commands[-1].append(token)
    return commands


class _ScriptCommand(NamedTuple):
    """One command of a shell script (`_script_walk`): its words, and the
    folder it changes into when it is a `cd`, None for any other."""

    words: list[str]
    cd: str | None


def _script_walk(text: str, *, escapes: bool = True) -> Iterator[_ScriptCommand]:
    """Each command of a shell script (`_script_commands`), read the same way
    by the delete and move walk and the launch walk: a `cd` or `pushd` with
    the folder it changes into, `~` with none given."""
    for words in _script_commands(text, escapes=escapes):
        if not words:
            continue
        bare, _ = _command_words(words)
        cd = (bare[1] if len(bare) > 1 else "~") if bare and bare[0] in _CD_WORDS else None
        yield _ScriptCommand(words=words, cd=cd)


class _FindCommand(NamedTuple):
    """A command `find` runs on each match (`_find_exec_commands`): the flag
    that runs it, as `-exec` or `-execdir`, and its words."""

    flag: str
    words: list[str]


def _find_exec_commands(args: list[str]) -> list[_FindCommand]:
    """Each command `find`, given `args`, runs on each match: the words after
    one of `_FIND_EXEC_FLAGS`, up to the `;` or `+` that ends them. The
    delete and move walk and the launch walk read them the same way."""
    return [
        _FindCommand(flag=word, words=list(itertools.takewhile(lambda w: w not in (";", "+"), args[i + 1:])))
        for i, word in enumerate(args) if word in _FIND_EXEC_FLAGS
    ]


def _program_name(word: str) -> str:
    """The program a command's first word names: its last part past a `/`
    or a `\\`, and, for a Windows program, lowercase and without `.exe`, as
    `venv\\Scripts\\python.exe` names `python`."""
    name = ntpath.basename(word)
    if "\\" in word or name.lower().endswith(".exe") or sys.platform == "win32":
        name = name.lower().removesuffix(".exe")
    return name


def _script_steps(text: str, *, cwds: Sequence[str] | _Unknown, depth: int) -> list[_ShellStep]:
    """The deletes and moves a shell script runs, command by command
    (`_script_walk`), each after the folders its `cd`s reach."""
    out: list[_ShellStep] = []
    for command in _script_walk(text):
        if command.cd is not None:
            cwds = _shell_cd(cwds, command.cd)
        else:
            out += _words_steps(command.words, cwds=cwds, depth=depth)
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
        head = _program_name(out[0])
        if _SHELL_ASSIGNMENT.match(out[0]) or out[0] in _SHELL_KEYWORDS:
            out.pop(0)
        elif head in _SHELL_PREFIXES:
            from_input = from_input or head == "xargs"
            out.pop(0)
            command: str | None = None
            while out and out[0].startswith("-"):
                flag = out.pop(0)
                if head == "flock" and flag.partition("=")[0] in _FLOCK_COMMAND_FLAGS:
                    command = flag.partition("=")[2] if "=" in flag else out.pop(0) if out else ""
                    continue
                takes = flag in _VALUE_FLAGS and flag not in _PREFIX_PLAIN_FLAGS.get(head, ())
                if (takes or flag in _PREFIX_VALUE_FLAGS.get(head, ())) and out and out[0] != _SOME_WORDS:
                    out.pop(0)
            operands = out[:_PREFIX_OPERANDS.get(head, 0)]
            # Words the check can't place may hold the runner's operand and
            # the program both, so the program is one it can't read.
            del out[:operands.index(_SOME_WORDS) if _SOME_WORDS in operands else len(operands)]
            if head == "flock" and out and out[0].partition("=")[0] in _FLOCK_COMMAND_FLAGS:
                flag = out.pop(0)
                command = flag.partition("=")[2] if "=" in flag else out.pop(0) if out else ""
            if command is not None:
                # `flock lock -c "..."` runs the text as a shell script.
                out = ["sh", "-c", command]
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
    verb, args = _program_name(words[0]), words[1:]
    picked = [_ShellStep(action="delete", operands=(_NOT_LITERAL,), cwds=cwds, files_only=False)]
    if "$" in verb or "`" in verb:
        # A command the shell picks when it runs, as in $(which rm) or $RM.
        return picked
    if verb in _SCRIPT_RUNNERS:
        script = _inner_script(verb, args, stdin)
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
        # Each command it runs on a match deletes or moves what it matched;
        # `-execdir` runs it in the match's own folder.
        steps = [_ShellStep(action="delete", operands=(_NOT_LITERAL,), cwds=cwds, files_only=False)] if (
            "-delete" in args
        ) else []
        for command in _find_exec_commands(args):
            steps += _words_steps(command.words, cwds=_UNKNOWN if "dir" in command.flag else cwds, depth=depth)
        return steps
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


def _inner_script(verb: str, args: list[str], stdin: str | None) -> str | _Unknown | None:
    """The script `verb`, one of `_SCRIPT_RUNNERS`, runs given `args`: its
    words joined for `eval`, the one cmd or PowerShell runs
    (`_cmd_script`, `_powershell_script`), or the one a POSIX shell runs
    (`_shell_script`). The delete and move walk and the launch walk read it
    the same way."""
    if verb == "eval":
        return " ".join(args)
    if verb in _CMD_SHELLS:
        return _cmd_script(args, stdin)
    if verb in _POWERSHELLS:
        return _powershell_script(verb, args, stdin)
    return _shell_script(args, stdin)


def _cmd_script(args: list[str], stdin: str | None) -> str | _Unknown:
    """The script cmd runs, given its arguments `args`: the words after `/c`
    or `/k`, joined; or, with neither, the text its input gives it, unknown
    when the check can't read one."""
    for i, word in enumerate(args):
        if word.lower() in ("/c", "/k", "/r"):
            return " ".join(args[i + 1:])
    return _UNKNOWN if stdin is None else stdin


def _powershell_script(verb: str, args: list[str], stdin: str | None) -> str | _Unknown | None:
    """The script PowerShell runs, given its arguments `args`: the words
    after `-Command`, joined, or, for Windows PowerShell, the words from its
    first that isn't a flag or a flag's value; unknown for a script encoded
    in base64. None for a script file, as `-File` and pwsh's first word name.
    With no script, the text its input gives it, unknown when the check
    can't read one."""
    skip = False
    for i, word in enumerate(args):
        if skip:
            skip = False
            continue
        flag = "-" + word[1:].lower() if word[:1] in ("-", "/") else None
        if flag in _POWERSHELL_COMMAND_FLAGS:
            return " ".join(args[i + 1:])
        if flag in _POWERSHELL_ENCODED_FLAGS:
            return _UNKNOWN
        if flag in _POWERSHELL_FILE_FLAGS:
            return None
        if flag is not None:
            skip = flag in _POWERSHELL_VALUE_FLAGS
            continue
        return None if verb == "pwsh" else " ".join(args[i:])
    return _UNKNOWN if stdin is None else stdin


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
    read; whether it installs the project in the cell's folder, which runs
    such code only where the project builds (`_Reader.project_builds`);
    whether it installs packages by name, from an index a setting in its
    environment may point elsewhere (`_INSTALLER_SOURCE_SETTING`); the
    literal code it hands `python -c`, each read as one more cell
    (`_exec_cells`); and whether it starts a Python interpreter, which runs
    the startup hooks a cell wrote (`_Modules.hooks_written`)."""

    unread: bool = False
    sources: tuple[str, ...] = ()
    syncs: bool = False
    installs: bool = False
    starts_python: bool = False

    @classmethod
    def joined(cls, launches: Sequence[_Launch]) -> _Launch:
        """What `launches`, the commands of one call or script, launch together."""
        return cls(
            unread=any(launch.unread for launch in launches),
            sources=tuple(source for launch in launches for source in launch.sources),
            syncs=any(launch.syncs for launch in launches),
            installs=any(launch.installs for launch in launches),
            starts_python=any(launch.starts_python for launch in launches),
        )

    def pointed(self, points: bool) -> _Launch:
        """This launch where its environment may point an install elsewhere
        (`points`): one that installs packages runs code the check can't read."""
        return replace(self, unread=True) if points and self.installs else self


# The calls that hand their arguments on to `subprocess.Popen` in order, so
# the third is the program it runs (`executable`).
_POPEN_ARGS_FUNCS = frozenset({
    "subprocess.Popen", "subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output",
})


def _python_launch(call: ast.Call, reader: _Reader) -> _Launch | None:
    """The Python `call` launches when it runs a program (`_LAUNCHERS`),
    named through the imports or a name bound to it (`_Reader.module_member`),
    or made through `functools.partial`, each partial a name may hold in
    turn (`_Reader.partials_of`): each command it can run, read by
    `_command_launch`. The program is the path the exec and spawn forms
    take before the program's words, or `executable=` when the call gives
    it. A call whose command the check can't read at all may run any
    program. None for any other call."""
    partials = reader.partials_of(node=call.func) if isinstance(call.func, (ast.Name, ast.Call)) else []
    if partials:
        made = [_python_launch(_partial_call(partial=partial, call=call), reader) for partial in partials]
        launches = [launch for launch in made if launch is not None]
        return _Launch.joined(launches) if launches else None
    last = _last_name(call.func)
    loop_kind = _LOOP_LAUNCHERS.get(last) if isinstance(call.func, ast.Attribute) else None
    if loop_kind is not None:
        return _loop_launch(call, reader, kind=loop_kind)
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
    if program is None and name in _POPEN_ARGS_FUNCS and len(call.args) > 2:
        # Popen takes its program as `executable`, the third argument, and
        # `run` and the others hand theirs on to it.
        program = call.args[2]
    if program is not None:
        word = _command_word(program, reader)
        words = _literal_texts(program, reader) if _unreadable_word(word) else [word]
        if words == [] and _never_given(program, reader):
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
        # A command a function nothing calls takes from its parameters never runs.
        return _Launch(unread=arg is None or not _never_given(arg, reader))
    given = _keyword(call, "input")
    stdin = None if given is None else _sql_text(given, reader)
    env = _keyword(call, "env")
    # Another folder or environment, as `cwd=` and `env=` give, the pad's
    # own after a cell changes into another folder, or one a cell set a
    # setting in that changes what Python imports (`_Reader.points_python`).
    moved = (_keyword(call, "cwd") is not None or env is not None or reader.cwds != ["."]
             or reader.points_python)
    launch = _Launch.joined([
        _script_launch(c, depth=0, stdin=stdin, moved=moved) if isinstance(c, str)
        else _command_launch(c, depth=0, stdin=stdin, moved=moved)
        for c in commands
    ])
    # An install where the call's environment, or the pad's own, may point
    # it at another index or a folder of the cell's.
    return launch.pointed(launch.installs and (
        env is not None and _env_sets(env, reader, setting=_INSTALLER_SOURCE_SETTING) or reader.points_installers
    ))


def _loop_launch(call: ast.Call, reader: _Reader, *, kind: _LoopLaunch) -> _Launch:
    """The Python an asyncio loop's launcher method runs (`_LOOP_LAUNCHERS`):
    `subprocess_exec` takes the program's words after its protocol factory,
    read as a command (`_command_launch`); `subprocess_shell` takes a shell
    command after it, read as a script (`_script_launch`). The pad's own
    folder or a setting that changes what Python imports makes it `moved`."""
    moved = reader.cwds != ["."] or reader.points_python
    if kind == "exec":
        words = [_command_word(w, reader) for w in call.args[1:]]
        return _command_launch(words, depth=0, stdin=None, moved=moved) if words else _Launch()
    command = call.args[1] if len(call.args) > 1 else None
    text = None if command is None else _command_text(command, reader)
    if command is None:
        return _Launch()
    if text is None or _HOLE in text:
        return _Launch(unread=True)
    return _script_launch(text, depth=0, stdin=None, moved=moved)


def _never_given(node: ast.AST, reader: _Reader, *, depth: int = 0) -> bool:
    """Whether `node`, a command or a word of one, reads a parameter of a
    function nothing calls (`_uncalled_param`), so the code it sits in never
    runs: a name that is one, a name a value or loop gives such a part, or
    code with such a name in it. An empty sequence the code walks is no sign
    of one: a cell may fill it in place. Past `_PARAM_DEPTH` names, no."""
    if depth > _PARAM_DEPTH:
        return False
    if not isinstance(node, ast.Name):
        return any(_never_given(n, reader, depth=depth + 1) for n in ast.walk(node)
                   if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
    return any(
        _uncalled_param(node.id, b, reader)
        or b.how in ("value", "element", "unpacked") and _never_given(b.value, reader, depth=depth + 1)
        for b in reader.bindings_of(node.id, node=node)
    )


def _uncalled_param(name: str, binding: _Binding, reader: _Reader) -> bool:
    """Whether `binding` makes `name` a parameter of a function nothing calls
    and the call being read doesn't pass (`_Reader.uncalled`), as
    `_param_texts` reads it."""
    return (binding.how == "param" and not reader.given(binding.value, name)
            and not reader.call_sites(binding.value, name) and reader.uncalled(binding.value))


def _script_launch(text: str, *, depth: int, stdin: str | None, moved: bool) -> _Launch:
    """The Python a shell script launches, command by command (`_script_walk`,
    `_command_launch`). After a `cd`, a command runs in another folder than
    the cell's, as `moved` says it does from the start. Only the first
    command reads `stdin`. A script with a backslash is read again as cmd,
    the shell a Windows pad runs a command's text in, reads it, with each
    backslash kept, as in `venv\\Scripts\\python job.py`: either reading
    that runs code the check can't read counts."""
    launch = _commands_launch(_script_walk(text), depth=depth, stdin=stdin, moved=moved)
    if "\\" in text and not launch.unread:
        kept = _commands_launch(_script_walk(text, escapes=False), depth=depth, stdin=stdin, moved=moved)
        return replace(launch, unread=kept.unread)
    return launch


def _commands_launch(
    commands: Iterable[_ScriptCommand], *, depth: int, stdin: str | None, moved: bool,
) -> _Launch:
    """The Python a script's `commands` launch together (`_script_launch`).
    A setting that points installs elsewhere (`_points_installers`), as an
    `export` sets one, holds for each command after it."""
    launches: list[_Launch] = []
    points = False
    for command in commands:
        if command.cd is not None:
            moved = True
            continue
        points = points or _points_installers(command.words)
        launch = _command_launch(command.words, depth=depth, stdin=None if launches else stdin, moved=moved)
        launches.append(launch.pointed(points))
    return _Launch.joined(launches)


def _command_launch(words: list[str], *, depth: int, stdin: str | None, moved: bool) -> _Launch:
    """The Python one command launches: python itself (`_python_arguments`),
    a `.py` script, a tool that loads a project's code, a test or a
    notebook (`_PYTHON_TOOLS`), the command a runner such as `uv run` runs
    (`_ENV_RUNNERS`), pip or another installer building code from a source
    the check can't read (`_pip_unread`, `_install_unread`), and what a
    shell's script, an `eval` or a `find -exec` runs in turn. A program
    word the check can't read may be any program, and so may a shell's
    script past `_SHELL_DEPTH` shells deep. The Windows launcher `py` runs
    Python, past the flags that pick which. `stdin` is the text the
    call gives the command as its input, when the check can read it;
    `moved` says the command runs in another folder or environment than
    the cell's, as a `VAR=value` before the program makes it do."""
    bare, _ = _command_words(words)
    if not bare:
        return _Launch()
    # The words before the program, all of them for a command a runner hands a shell.
    before = words[: words.index(bare[0])] if bare[0] in words else words
    moved = moved or any(_SHELL_ASSIGNMENT.match(w) or _moves_folder(w, _PREFIX_FOLDER_FLAGS) for w in before)
    if _unreadable_word(bare[0]):
        return _Launch(unread=True)
    verb, args = _program_name(bare[0]), bare[1:]
    if verb in _SCRIPT_RUNNERS:
        script = _inner_script(verb, args, stdin)
        if script is None:
            # A shell script file: not Python.
            return _Launch()
        if isinstance(script, _Unknown) or depth >= _SHELL_DEPTH:
            return _Launch(unread=True)
        return _script_launch(script, depth=depth + 1, stdin=None, moved=moved)
    if verb == "find":
        return _Launch.joined([
            _command_launch(command.words, depth=depth, stdin=None, moved=moved or "dir" in command.flag)
            for command in _find_exec_commands(args)
        ])
    target = _env_runner_target(verb, args)
    if target is not None:
        # The runner's own words, which may move it into another folder or
        # environment, as `uv run --directory sub` does.
        own = args[: len(args) - len(target)] if args[len(args) - len(target):] == target else args
        flags = _RUNNER_FOLDER_FLAGS.get(verb, frozenset())
        return _command_launch(target, depth=depth, stdin=stdin,
                               moved=moved or any(_moves_folder(w, flags) for w in own))
    installs = _install_unread(verb, args)
    if installs is not None:
        # In another folder, the project it installs may be any.
        return _Launch(unread=True) if installs.syncs and moved else installs.pointed(_points_installers(before))
    pip_args = _pip_arguments(verb, args)
    if pip_args is not None:
        return replace(_pip_launch(pip_args).pointed(_points_installers(before)), starts_python=True)
    if verb == "py":
        if args and args[0] in _PY_LAUNCHER_LISTS:
            return _Launch()
        launch = _python_arguments(list(itertools.dropwhile(_PY_LAUNCHER_FLAG.fullmatch, args)), moved=moved)
        return replace(launch, starts_python=True)
    if verb.startswith("ipython") and _PYTHON_PROGRAM.fullmatch(verb):
        return replace(_ipython_launch(args, moved=moved), starts_python=True)
    if _PYTHON_PROGRAM.fullmatch(verb):
        return replace(_python_arguments(args, moved=moved), starts_python=True)
    first = args[0] if args else ""
    if verb.endswith(_PYTHON_SCRIPTS):
        return _Launch(unread=True)
    if args and all(a in _TOOL_EXITS for a in args) and verb not in _CONFTEST_TOOLS:
        return _Launch()
    return _Launch(unread=(
        verb in _PYTHON_TOOLS or (verb, first) in _PYTHON_TOOL_COMMANDS
        or (verb == "jupyter-nbconvert" or verb == "jupyter" and first == "nbconvert") and "--execute" in args
    ))


# IPython's own options that run code the check can't read as it starts: a
# profile, a directory of startup scripts, a config file or an extension it
# loads, a module it runs, and `-i`, which goes on to run its input.
_IPYTHON_LOADING_OPTIONS = frozenset({
    "--profile", "--profile-dir", "--ipython-dir", "--config", "--ext", "-m", "--m", "-i", "--i",
})
# How IPython's `-c` is spelled, as its short option or its long alias.
_IPYTHON_CODE_OPTIONS = frozenset({"-c", "--c"})


def _ipython_launch(args: list[str], *, moved: bool) -> _Launch:
    """What IPython runs given its arguments `args`. Asked for its version
    or its help (`_TOOL_EXITS`), it runs nothing. An option that loads code
    as it starts (`_IPYTHON_LOADING_OPTIONS`), or one that sets an option of
    one of its classes (`--Class.option=`, as `exec_lines` and
    `file_to_run` do), runs code the check can't read. Its `-c` text is
    read as `python -c` (`_python_arguments`) only where it parses as
    Python and names no `get_ipython`: a magic (`%run`), a shell escape
    (`!`) and IPython's other syntax don't parse, and the shell
    `get_ipython` gives runs commands and code. With no `-c`, it runs a
    script, or the code its input gives it, as `python` does. Its options
    end at the first `--`: the words after it are the script's own. Of two
    `-c` options the last one counts, as for any of its options, and a
    first word `help` asks for its help, unless a `--` follows."""
    if args[:1] == ["help"] and "--" not in args:
        return _Launch()
    code: str | None = None
    words = iter(args)
    for word in words:
        if word == "--":
            break
        option, equals, value = word.partition("=")
        if word in _TOOL_EXITS:
            return _Launch()
        if option in _IPYTHON_LOADING_OPTIONS or option.startswith("--") and "." in option:
            return _Launch(unread=True)
        if option in _IPYTHON_CODE_OPTIONS:
            code = value if equals else next(words, None)
        elif word.startswith("-c") and not word.startswith("--"):
            code = word[2:]
    if code is None:
        return _Launch(unread=True)
    if "get_ipython" in code or _parse_ran(code) is None:
        return _Launch(unread=True)
    return _python_arguments(["-c", code], moved=moved)


def _moves_folder(word: str, flags: AbstractSet[str]) -> bool:
    """Whether `word`, a flag before a command's program, is one of `flags`,
    which run the program in another folder or environment, as `--flag
    value` or `--flag=value`."""
    return word.partition("=")[0] in flags


def _takes_value(flag: str, value_flags: AbstractSet[str]) -> bool:
    """Whether `flag`, a flag word, `--flag=value` or not, is one of
    `value_flags`, which take a value: long ones matched by unique prefix,
    as `--req` is `--requirement`."""
    name = flag.partition("=")[0]
    return name in value_flags or name.startswith("--") and any(
        o.startswith(name) for o in value_flags if o.startswith("--")
    )


def _after_subcommand(lead: list[str], subcommands: Sequence[tuple[str, ...]]) -> list[str] | None:
    """The words of `lead`, a command's words past its program and leading
    flags, after the first of `subcommands` it starts with; None when it
    starts with none."""
    own = next((own for own in subcommands if lead[: len(own)] == list(own)), None)
    return None if own is None else lead[len(own):]


def _drop_leading_flags(args: list[str], value_flags: frozenset[str]) -> list[str]:
    """`args` past its leading flags: a `--flag value` or `-f value` whose
    flag takes a value (`_takes_value`) drops its value too, a
    `--flag=value` drops nothing more."""
    rest = list(args)
    while rest and rest[0].startswith("-"):
        flag = rest.pop(0)
        if _takes_value(flag, value_flags) and "=" not in flag and rest:
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
    rest = _after_subcommand(_drop_leading_flags(args, _ENV_RUNNER_VALUE_FLAGS), subcommands)
    if rest is None:
        return None
    while rest and rest[0].startswith("-"):
        flag, equals, value = rest[0].partition("=")
        if flag in _ENV_RUN_MODULE_FLAGS:
            return ["python", "-m", value if equals else (rest[1] if len(rest) > 1 else "")]
        if flag in _ENV_RUN_SCRIPT_FLAGS:
            return ["python", value if equals else (rest[1] if len(rest) > 1 else "")]
        rest.pop(0)
        if _takes_value(flag, _ENV_RUNNER_VALUE_FLAGS) and not equals and rest:
            rest.pop(0)
    return rest


def _install_unread(verb: str, args: list[str]) -> _Launch | None:
    """What an installer besides pip, given `args`, launches: code the check
    can't read for a command that installs the project in the folder it
    runs in (`_PROJECT_INSTALLS`), or one that installs packages
    (`_PACKAGE_INSTALLS`) from a requirements file, a local copy or an index
    it is pointed at (pip's source flags), or from a source
    `_pip_source_operand` names, as `pipx install .` does. `pipenv install`
    with no package installs the project's own list, so the project. A
    `uv add` installs the project too, unless told not to sync it
    (`_SYNCING_ADDS`), which runs code the check can't read only where the
    project builds (`_Launch.syncs`). None when the command is none of
    these."""
    lead = _drop_leading_flags(args, _ENV_RUNNER_VALUE_FLAGS)
    if _after_subcommand(lead, _PROJECT_INSTALLS.get(verb, ())) is not None:
        return _Launch(unread=True)
    rest = _after_subcommand(lead, _PACKAGE_INSTALLS.get(verb, ()))
    if rest is None:
        return None
    syncs = _after_subcommand(lead, _SYNCING_ADDS.get(verb, ())) is not None
    operands: list[str] = []
    words = iter(rest)
    for word in words:
        flag = word.partition("=")[0]
        if flag in _PIP_SOURCE_FLAGS:
            return _Launch(unread=True)
        if word.startswith("-"):
            syncs = syncs and flag not in _NO_PROJECT_SYNC_FLAGS
            if _takes_value(flag, _ENV_RUNNER_VALUE_FLAGS | _INSTALL_VALUE_FLAGS) and "=" not in word:
                next(words, None)
            continue
        operands.append(word)
    if not operands:
        return _Launch(unread=verb == "pipenv")
    return _Launch(unread=any(_pip_source_operand(word) for word in operands), syncs=syncs, installs=True)


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
                        return _pip_launch(list(words))
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
        if word == _SOME_WORDS:
            # Words in any order: each read on its own, so a value flag's
            # value counts as an operand.
            return any(_pip_unread([command, w]) for w in words if w != _SOME_WORDS)
        if word == "--":
            return any(_pip_source_operand(operand) for operand in words)
        if word.startswith("--"):
            kind = _pip_long_kind(word)
            if kind == "source":
                return True
            if kind == "value" and "=" not in word:
                next(words, None)
        elif word.startswith("-") and word != "-":
            # A cluster of short flags: the first that takes a value takes the
            # rest of the word, or the next word when it ends the cluster, as
            # `-t ./vendor` and `-qt ./vendor` do.
            for at, letter in enumerate(word[1:], start=2):
                if letter in _PIP_SOURCE_LETTERS:
                    return True
                if letter in _PIP_VALUE_LETTERS:
                    if at == len(word):
                        next(words, None)
                    break
        elif _pip_source_operand(word):
            return True
    return False


def _pip_launch(args: list[str]) -> _Launch:
    """What pip, given `args`, launches (`_pip_unread`): an install,
    download or wheel build of packages by name installs from an index."""
    rest = _drop_leading_flags(args, _PIP_VALUE_FLAGS)
    return _Launch(unread=_pip_unread(args), installs=bool(rest) and rest[0] in _PIP_FETCH_COMMANDS)


def _points_installers(words: Iterable[str]) -> bool:
    """Whether `words`, a command's, set a setting that points pip or uv at
    another index or a folder (`_INSTALLER_SOURCE_SETTING`), as
    `PIP_INDEX_URL=... pip install` or `export PIP_FIND_LINKS=...` does."""
    return any(
        _INSTALLER_SOURCE_SETTING.fullmatch(word.partition("=")[0].removeprefix("$env:"))
        for word in words if "=" in word
    )


def _env_sets(env: ast.AST, reader: _Reader, *, setting: re.Pattern[str], depth: int = 0) -> bool:
    """Whether `env`, the environment a call hands a program or the pad's
    own takes, may hold a setting `setting` matches, as one that points pip
    or uv at another index or a folder (`_INSTALLER_SOURCE_SETTING`): a
    dict, or `dict()`'s keywords, with such a key or a key the check can't
    read, or one the check can't read whole. The pad's own environment, as
    `os.environ` or a copy of it, holds what a cell set in it
    (`_Reader.points_installers`, `_Reader.points_python`)."""
    if depth > _PARAM_DEPTH:
        return True
    name = reader.dotted(env.func) if isinstance(env, ast.Call) else reader.dotted(env)
    if name in ("os.environ", "os.environ.copy"):
        return False
    if isinstance(env, ast.Dict):
        return any(
            _env_sets(v, reader, setting=setting, depth=depth + 1) if k is None else _names_setting(k, setting)
            for k, v in zip(env.keys, env.values)
        )
    if isinstance(env, ast.Call) and name == "dict":
        return any(_env_sets(a, reader, setting=setting, depth=depth + 1) for a in env.args) or any(
            _env_sets(k.value, reader, setting=setting, depth=depth + 1) if k.arg is None
            else bool(setting.fullmatch(k.arg)) for k in env.keywords
        )
    if not isinstance(env, ast.Name):
        return True
    found = reader.bindings_of(env.id, node=env)
    if not found or any(
        b.how != "value" or _env_sets(b.value, reader, setting=setting, depth=depth + 1) for b in found
    ):
        return True
    name = env.id

    def holds(node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id == name

    return any(_changes_set(cell, reader, holds=holds, setting=setting, depth=depth) for cell in reader.cells)


def _changes_set(
    cell: _Cell, reader: _Reader, *, holds: Callable[[ast.AST], bool], setting: re.Pattern[str], depth: int,
) -> bool:
    """Whether `cell` changes in place an environment's dict, the one each
    node `holds` reads, to hold a setting `setting` matches
    (`_names_setting`): a store under such a key, a `setdefault` of one, or
    an `update` with one (`_env_sets`)."""
    for node in cell.stores:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Subscript) and holds(t.value) and _names_setting(t.slice, setting) for t in node.targets
        ):
            return True
    for call in cell.calls:
        func = call.func
        if not (isinstance(func, ast.Attribute) and holds(func.value)):
            continue
        if func.attr == "update" and (
            any(_env_sets(a, reader, setting=setting, depth=depth + 1) for a in call.args)
            or any(k.arg is None or setting.fullmatch(k.arg) for k in call.keywords)
        ):
            return True
        if func.attr in ("setdefault", "__setitem__") and (not call.args or _names_setting(call.args[0], setting)):
            return True
    return False


def _names_setting(key: ast.AST, setting: re.Pattern[str]) -> bool:
    """Whether `key`, an environment's key, names a setting `setting`
    matches, or is one the check can't read."""
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return bool(setting.fullmatch(key.value))
    return True


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
        made = reader.function_named(node.func) if isinstance(node.func, ast.Name) else None
        if made is not None:
            values = [n.value for n in ast.walk(made) if isinstance(n, ast.Return) and n.value is not None]
    elif isinstance(node, ast.Name):
        for b in reader.bindings_of(node.id, node=node):
            if b.how in ("value", "context"):
                values.append(b.value)
            elif b.how == "param":
                args = reader.args_for(b.value, node.id)
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
    `Request`, a Session's or Client's method, one of either looked up by a
    name, as `getattr(requests, verb)(url)` is, whose method is each text
    the name holds (`_http_methods`), or urllib's `Request` or `urlopen`."""
    looked = call.func
    if (isinstance(looked, ast.Call) and isinstance(looked.func, ast.Name) and looked.func.id == "getattr"
            and len(looked.args) == 2 and not looked.keywords):
        holder, verb_node = looked.args
        client = None if reader.dotted(holder) in _HTTP_LIBRARIES else _http_client(holder, reader)
        if reader.dotted(holder) not in _HTTP_LIBRARIES and client is None:
            return None
        return _HttpCall(method=verb_node, url=_target_arg(call, keywords=("url",)),
                         base=None if client is None else _keyword(client, "base_url"))
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
            args = reader.args_for(b.value, node.id)
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
    """`__import__("os")` or `importlib.import_module("os")` as `os`,
    `importlib.import_module("os.path")` as `os.path`, and a relative
    `importlib.import_module(".helpers", "pkg")` as `pkg.helpers`, with the
    package given by position or as `package=`. None for any other call: an
    `__import__` given more than the name, a relative name with no literal
    package, or any other keyword."""
    args, importer = call.args, _last_name(call.func)
    if importer == "import_module":
        given = [*args[1:2], *(k.value for k in call.keywords if k.arg == "package")]
        if len(args) > 2 or len(given) > 1 or any(k.arg != "package" for k in call.keywords):
            return None
    elif importer != "__import__" or len(args) != 1 or call.keywords:
        return None
    else:
        given = []
    texts = [a.value if isinstance(a, ast.Constant) and isinstance(a.value, str) else "" for a in (*args[:1], *given)]
    name = texts[0] if texts else ""
    if not name:
        return None
    if name.startswith("."):
        package = texts[1] if len(texts) > 1 else ""
        if not package:
            return None
        try:
            name = importlib.util.resolve_name(name, package)
        except (ImportError, ValueError):
            return None
    parts = name.split(".")
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
                value[:] = [plain(v) if isinstance(v, ast.Call) else _plain_code_store(v) for v in value]
    return frozenset(imported)


def _plain_code_store(stmt: ast.AST) -> ast.AST:
    """`stmt` read plainly where it is `setattr(f, "__code__", v)`
    (`_sets_code`), as the assignment `f.__code__ = v` it makes
    (`_stores_code`); `stmt` itself otherwise. One inside another
    expression is read where it is (`_unread_run_at`, `_filled_names`)."""
    call = stmt.value if isinstance(stmt, ast.Expr) else None
    if not (isinstance(call, ast.Call) and _sets_code(call)):
        return stmt
    target = ast.copy_location(ast.Attribute(value=call.args[0], attr="__code__", ctx=ast.Store()), call)
    return ast.copy_location(ast.Assign(targets=[target], value=call.args[2]), stmt)


# The file name the check parses and compiles a cell's code under, so the
# warning filter `_quiet_parses` adds quiets its own parses and no other code's.
_CHECK_FILENAME = "<write-check>"
_CHECK_MODULE = re.escape(_CHECK_FILENAME) + r"\Z"


def _quiet_parses() -> None:
    """Have the process's warning filters ignore the SyntaxWarnings the
    check's own parses and compiles raise, by their file name
    (`_CHECK_FILENAME`), ahead of any other filter, so one that turns
    warnings into errors doesn't refuse a cell the runner compiles. The
    filter goes first once, and again only where another went ahead of it
    or a `warnings.catch_warnings` block that began before it put the
    filters back without it. It names the check's file alone, so a check
    running in any thread changes no warning other code raises, and no
    filter is saved and put back around each parse."""
    first = warnings.filters[0] if warnings.filters else None
    if not (first is not None and first[0] == "ignore" and first[2] is SyntaxWarning and first[3] is not None
            and first[3].pattern == _CHECK_MODULE):
        warnings.filterwarnings("ignore", category=SyntaxWarning, module=_CHECK_MODULE)


def _parse(code: str) -> ast.Module:
    """The cell's tree, from the same healed text the runner compiles.
    `_Cell` reads its literal `getattr` and import lookups as plain names.

    Quiet: the agent's own invalid escape sequences would otherwise print a
    SyntaxWarning in the host's log for every check; the runner reports them
    where they belong, in the cell's output (`_quiet_parses`).
    """
    _quiet_parses()
    return ast.parse(heal_surrogate_source(code), filename=_CHECK_FILENAME)


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


class _ExecCellKey(NamedTuple):
    """A cell of literal code by its source and the place it takes among the
    cells (`_Cell.order`), which a conversation fixes as it grows: the exec
    cache keys a parsed cell by it."""

    text: str
    order: tuple[int, ...]
    host_line: int | None
    launched: bool
    host: int
    shared: bool
    namespace: str | None
    inherits: bool
    surely_run: bool
    parent_order: tuple[int, ...]


class _ExecCells:
    """Parsed cells of literal code exec(), a runner of text or `python -c`
    runs, kept across checks instead of parsed once per check.

    Each check reads the same literal in an earlier cell, which doesn't
    change, so parsing an 83k-character script and walking it once per
    process, not once per check, saves that work. A cell is keyed by its
    source and the place it takes among the cells (`_ExecCellKey`), which a
    given conversation fixes as it grows; a source that doesn't parse is
    kept as None, so it isn't parsed again. Budgeted and locked like
    `_EarlierCells`.
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
    """The text of a str or bytes literal (`_decoded_source`), or of an
    f-string with nothing to fill in; None for any other code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Constant) and isinstance(node.value, bytes):
        return _decoded_source(node.value)
    if isinstance(node, ast.JoinedStr) and all(isinstance(v, ast.Constant) for v in node.values):
        return "".join(str(v.value) for v in node.values)
    return None


# A coding comment, as Python reads one in a source's first two lines.
_CODING_COMMENT = re.compile(rb"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")


def _decoded_source(value: bytes) -> str | None:
    """Bytes of Python code decoded as exec() and compile() decode them: by
    a byte order mark or a coding comment, else as UTF-8. "" for bytes that
    don't decode, which fail when they run too, so they run nothing; None
    for a coding comment that names an encoding Python doesn't ship, which a
    cell may have registered, so the code is one the check can't read."""
    import codecs

    try:
        return importlib.util.decode_source(value)
    except (SyntaxError, UnicodeDecodeError):
        pass
    for line in value.splitlines()[:2]:
        found = _CODING_COMMENT.match(line)
        if found is None:
            continue
        try:
            codecs.lookup(found.group(1).decode("ascii"))
        except LookupError:
            return None
        break
    return ""


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
    `cells` holds them all in the order read; `_Cell.order` places them
    where they run. `ran` holds them by the id of the code whose calls run
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


def _exec_cells(reader: _Reader, starts: Sequence[_Cell], *, ran_only: bool = False) -> _ExecRead:
    """The literal code each of `starts` passes to exec(), eval() or
    compile(), to a runner of text such as `timeit.timeit`
    (`_runner_code`), or to `python -c` in a program it runs
    (`_python_launch`), each source read as one more cell placed where its
    call runs (`_Cell.order`), and the literal code those cells pass on in
    turn (`_ExecReading`). When a function an earlier cell defined spells
    one, the literal code each such function a start calls runs is read
    too, within `_CALL_DEPTH` calls, under that function. With `ran_only`,
    only the code that runs when its cell runs counts (`_nodes_that_run`),
    and a function only passed on as a value isn't followed."""
    reading = _ExecReading(reader, ran_only=ran_only)
    for start in starts:
        reading.sources(start, start.tree, host=start.host)
    if reading.follows():
        for start in starts:
            for code in reading.read.with_ran(start, start.tree):
                reading.follow(code.cell, code.root, (), host=start.host)
    return reading.read


class _ExecReading:
    """One read of the literal code a set of cells runs (`_exec_cells`),
    with `reader`: what it read so far (`read`), how many cells each start's
    literal code made, by the start's host (`made`), past `_EXEC_LIMIT` of
    which it stops, and the shortest chain of calls each function was
    followed at (`followed`)."""

    def __init__(self, reader: _Reader, *, ran_only: bool) -> None:
        self.reader = reader
        self.ran_only = ran_only
        self.read = _ExecRead.empty()
        self.made: dict[int, int] = {}
        self.followed: dict[int, int] = {}

    def exec_named(self, func: ast.AST) -> str | None:
        """The builtin exec, eval or compile a call to `func` runs. A bare
        name counts whatever binds it, so a def that shadows one keeps the
        read it had; any other counts where `_Reader.exec_builtin` reads it
        as the builtin."""
        if isinstance(func, ast.Name) and func.id in _EXEC_CALLS:
            return func.id
        return self.reader.exec_builtin(func)

    def nodes(self, code: _Code) -> Iterable[ast.AST]:
        """The nodes of `code` read for the calls that run literal code: only
        those that run when its cell runs with `ran_only`, else a cell's own
        calls, listed in walk order already, or every node of a function."""
        if self.ran_only:
            return code.cell.nodes_run_by(code.root)
        if code.root is code.cell.tree:
            return code.cell.calls
        return ast.walk(code.root)

    def sources(
        self, cell: _Cell, root: ast.AST, *, host: int, at_call: tuple[int, ...] | None = None,
        at_line: int | None = None, at_sure: bool = True,
    ) -> None:
        """Reads the literal code `root`, code of `cell`, runs, and the code
        that code runs in turn, each as one more cell. Literal code a
        function runs is placed at the call that runs the function
        (`at_call`), not where the def sits, then by the def's cell, so two
        functions one call reaches place theirs apart; it runs at the line
        of that call in the host (`at_line`), and surely runs only when that
        call does (`at_sure`) and so does the exec() in the function
        (`_Cell.surely_run`). Literal code a def or lambda in a whole cell
        runs is placed where the cell first runs that function
        (`_first_run_at`)."""
        reader, read, made = self.reader, self.read, self.made
        into = read.ran.setdefault(id(root), [])
        pending = [_Code(cell=cell, root=root)]
        while pending and made.get(host, 0) < _EXEC_LIMIT:
            code = pending.pop()
            called = at_call is not None and code.root is root
            place_at = (*at_call, *code.cell.order) if at_call is not None and called else code.cell.order
            host_line = at_line if at_call is not None and called else code.cell.host_line
            sure = at_sure if at_call is not None and called else code.cell.surely_run
            for node in self.nodes(code):
                if not isinstance(node, ast.Call):
                    continue
                builtin = self.exec_named(node.func)
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
                elif not builtin and len(texts) > 1:
                    # A runner's setup and statement run in one namespace.
                    texts = ["\n".join(texts)]
                if not texts or any(len(text) > _LITERAL_CODE_LIMIT for text in texts):
                    # Past the limit, literal code is left unread.
                    continue
                if made.get(host, 0) + len(texts) > _EXEC_LIMIT:
                    return
                made[host] = made.get(host, 0) + len(texts)
                read.tried.add(id(node))
                place = _exec_place(node, code.cell, reader) if builtin else _ExecPlace(shared=False)
                # A call inside a def runs when the def is called.
                runs_at = (node.lineno, node.col_offset)
                if code.root is code.cell.tree and code.cell.scope_of(node) is not None:
                    runs_at = (*_first_run_at(node, code.cell, reader), *runs_at)
                for at, text in enumerate(texts):
                    key = _ExecCellKey(
                        text=text, order=(*place_at, *runs_at, at),
                        host_line=host_line or node.lineno,
                        launched=launch is not None or code.cell.launched, host=host,
                        shared=place.shared and code.cell.shared,
                        namespace=place.namespace if code.cell.shared else code.cell.namespace,
                        inherits=(place.shared or place.reads_host) and not code.cell.shared,
                        surely_run=sure and code.cell.runs_surely(node),
                        parent_order=code.cell.order)
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

    def follows(self) -> bool:
        """Whether a function an earlier cell defined may run literal code: an
        earlier cell's function hands exec() or a runner text
        (`runs_text`), or hands `python -c` code (`hands_python_code`), or a
        cell imports a member of the builtins module, which may be exec under
        another name."""
        reader = self.reader
        earlier = [c for c in reader.cells if c.host < reader.current.index]
        return (
            any(self.runs_text(c) for e in earlier for c in e.function_execs)
            or any(self.hands_python_code(c) for e in earlier for c in e.function_launches)
            or any(full.startswith("builtins.") for cell in reader.cells for full in cell.imports.values())
        )

    def runs_text(self, call: ast.Call) -> bool:
        """Whether `call` hands exec(), eval(), compile() or a runner of text its code."""
        return self.exec_named(call.func) is not None or _runner_code(call, self.reader) is not None

    def hands_python_code(self, call: ast.Call) -> bool:
        """Whether `call` runs a program that hands `python -c` literal code."""
        launch = _python_launch(call, self.reader)
        return launch is not None and bool(launch.sources)

    def follow(
        self, cell: _Cell, root: ast.AST, chain: tuple[int, ...], *, host: int, at_call: tuple[int, ...] | None = None,
        at_line: int | None = None, at_sure: bool = True,
    ) -> None:
        """Reads the literal code each earlier function `root` calls runs
        (`sources`), under that function, then the functions it calls in
        turn, within `_CALL_DEPTH` calls, each at the shortest chain it is
        reached at. That code runs where the start's own call that leads
        to it first runs (`_first_run_at`), at that call's line in the host
        (`at_line`), and surely runs only when each call that leads to it
        does (`at_sure`)."""
        reader = self.reader
        for callee in _earlier_function_uses(reader, root, cell, ran_only=self.ran_only):
            func = callee.func
            if (id(func) in chain or len(chain) >= _CALL_DEPTH or self.ran_only and callee.call is None
                    or self.followed.get(id(func), _CALL_DEPTH) <= len(chain)):
                continue
            self.followed[id(func)] = len(chain)
            home = reader.cell_of(func)
            # Where the start's own call that leads here runs.
            use = callee.call or callee.use
            # A call inside a def of the start runs when the def is called.
            here = at_call or (*cell.order, *_first_run_at(use, cell, reader), use.lineno, use.col_offset)
            line = at_line or use.lineno
            sure = at_sure and cell.surely_run and cell.runs_surely(use)
            with reader.calling(func, callee.call, bound=callee.bound):
                if id(func) not in self.read.ran:
                    self.sources(home, func, host=host, at_call=here, at_line=line, at_sure=sure)
                for code in self.read.with_ran(home, func):
                    self.follow(code.cell, code.root, (*chain, id(func)), host=host, at_call=here, at_line=line,
                                at_sure=sure)


@dataclass(frozen=True)
class _ExecPlace:
    """Where the names literal code exec() or eval() runs land: in the cell
    namespace (`shared`), or in the namespace a name holds (`namespace`),
    or neither, as for a function's own locals. Given no namespace inside a
    function, the code still reads the names of the namespace the
    function's own code runs in, its globals, not the function's locals or
    parameters (`reads_host`), and so does such code a later call runs
    through that function (`_ExecReading.follow`)."""

    shared: bool
    namespace: str | None = None
    reads_host: bool = False


def _exec_place(call: ast.Call, cell: _Cell, reader: _Reader) -> _ExecPlace:
    """Where the code `call`, code of `cell`, hands exec(), eval() or
    compile() binds its names. exec() at the cell's top level given no
    namespace, or the cell's own (`globals()`, and there `locals()` and
    `vars()` with no argument, the same dict), binds them in the cell
    namespace; given a dict or a module's `__dict__`, in that; inside a
    function, in the function's own locals. A compile() binds where the
    exec() or eval() it is handed to does, and nowhere the check can follow
    otherwise."""
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
    top = cell.scope_of(call) is None
    if not own:
        return _ExecPlace(shared=top, reads_host=not top)
    if all(_cell_namespace(g, reader) and (top or reader.dotted(g.func) == "globals") for g in own):
        return _ExecPlace(shared=True)
    return _ExecPlace(shared=False, namespace=_namespace_root(own[-1]))


def _parse_ran(text: str) -> ast.Module | None:
    """`text`, literal code a cell runs, parsed; None when it doesn't parse,
    or is too deep or long for the parser, so it fails when it runs too."""
    try:
        return _parse(text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


# Indexes for cells of literal code, past any conversation's own: each such
# cell keeps one for good, so a cell the exec cache serves to every check
# that reads it is told apart from the conversation's cells.
_EXEC_INDEXES = itertools.count(1 << 40)


def _exec_cell(key: _ExecCellKey) -> _Cell | None:
    """The parsed cell of literal code a cell runs, placed by `key`; None
    when the code doesn't parse."""
    tree = _parse_ran(key.text)
    if tree is None:
        return None
    return _Cell(tree, key.text, next(_EXEC_INDEXES), host_line=key.host_line, launched=key.launched,
                 host=key.host, shared=key.shared, namespace=key.namespace, order=key.order, inherits=key.inherits,
                 surely_run=key.surely_run, parent_order=key.parent_order)


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
    each argument it takes as code, in the order it runs them, a setup
    before the statement, and whether the call passes more through `*` or
    `**`, which the check can't read."""

    sources: tuple[ast.AST, ...]
    spread: bool


def _runner_code(call: ast.Call, reader: _Reader) -> _RunnerCode | None:
    """The code `call` hands a runner of text, such as `timeit.timeit`'s
    statement and setup, or `timeit.main`'s argv (`_timeit_main_code`);
    None when it calls none. A runner given nothing runs its default,
    `pass`."""
    if not _runner_named(call.func, reader):
        return None
    member = reader.module_member(call.func)
    if member == _TIMEIT_MAIN:
        return _timeit_main_code(call, reader)
    params = _STRING_RUNNERS.get(member or "")
    if params is None:
        return None
    given = {**dict(zip(params, call.args)), **{k.arg: k.value for k in call.keywords if k.arg in params}}
    sources = [given[p] for p in reversed(params) if p in given]
    spread = any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords)
    return _RunnerCode(sources=tuple(sources), spread=spread)


# The options `timeit.main`'s `getopt` knows, long and short, each with
# whether it takes a value, so the word that gives it isn't a statement: the
# count, the repeats, the unit, and the setup, which runs before the
# statements and may repeat.
_TIMEIT_MAIN_LONG = {"number": True, "setup": True, "repeat": True, "process": False, "verbose": False,
                     "unit": True, "help": False}
_TIMEIT_MAIN_SHORT = {"n": True, "u": True, "s": True, "r": True, "p": False, "v": False, "h": False}


def _timeit_main_code(call: ast.Call, reader: _Reader) -> _RunnerCode:
    """The code `timeit.main` runs: the setups and statements in its argv
    list, setups first, as they share one namespace. Its options are read
    as its `getopt` reads them (`_TIMEIT_MAIN_SHORT`, `_TIMEIT_MAIN_LONG`):
    a short option's value glued on or the next word, as `-sCODE` and `-s
    CODE` give a setup, a long one's after `=` or the next word, matched by
    unique prefix, as `--se=CODE` is `--setup=CODE`; the first word that
    isn't an option, or the words after `--`, are the statements. `spread`
    is set when the argv isn't a readable list, a word in it isn't a
    literal, or an option is none `getopt` knows, so the cell is held
    unchecked; a readable argv of literal statements is read."""
    argv = call.args[0] if call.args else _keyword(call, "args")
    items = reader.items_of(node=argv, key=None, depth=0) if argv is not None else None
    if items is None:
        return _RunnerCode(sources=(), spread=True)
    setups: list[ast.AST] = []
    statements: list[ast.AST] = []
    words = iter(items)
    spread = False

    for word in words:
        text = _literal_source(word)
        if text is None:
            spread = True
            continue
        if text == "--" or not text.startswith("-") or text == "-":
            statements += [word] if text != "--" else []
            break
        if text.startswith("--"):
            name, equals, glued = text[2:].partition("=")
            known = [name] if name in _TIMEIT_MAIN_LONG else [o for o in _TIMEIT_MAIN_LONG if o.startswith(name)]
            if len(known) != 1 or equals and not _TIMEIT_MAIN_LONG[known[0]]:
                spread = True
                continue
            if _TIMEIT_MAIN_LONG[known[0]]:
                given = ast.copy_location(ast.Constant(value=glued), word) if equals else next(words, None)
                if known[0] == "setup" and given is not None:
                    setups.append(given)
            continue
        for at, letter in enumerate(text[1:], start=2):
            takes = _TIMEIT_MAIN_SHORT.get(letter)
            if takes is None:
                spread = True
                break
            if takes:
                given = ast.copy_location(ast.Constant(value=text[at:]), word) if text[at:] else next(words, None)
                if letter == "s" and given is not None:
                    setups.append(given)
                break
    statements += list(words)
    spread = spread or any(_literal_source(word) is None for word in statements)
    return _RunnerCode(sources=(*setups, *statements), spread=spread)


def _runner_methods(node: ast.AST, reader: _Reader) -> frozenset[str]:
    """The methods that run code on what `node` holds: those of the runner
    whose call made it (`_INSTANCE_RUNNERS`), there or through the names a
    value or a with-statement binds to it, and those of a runner a class
    read code defines is made from (`_class_runner_methods`)."""
    out: set[str] = set()
    todo = [node]
    seen: set[int] = set()
    while todo:
        node = todo.pop()
        if isinstance(node, ast.Call):
            out |= _class_runner_methods(node.func, reader)
        elif isinstance(node, ast.Name) and node.id in reader.bound_names and id(node) not in seen:
            seen.add(id(node))
            todo += [b.value for b in reader.bindings_of(node.id, node=node) if b.how in ("value", "context")]
    return frozenset(out)


def _class_runner_methods(cls: ast.AST, reader: _Reader) -> frozenset[str]:
    """The methods that run code on what calling `cls` makes: those of the
    runner it names (`_INSTANCE_RUNNERS`), or, for a class read code
    defines, those of each runner among its bases, and their bases in turn,
    as `class Quiet(cProfile.Profile)` makes a profiler, and as a class
    `type(name, bases, ns)` makes, bound to a name, does too."""
    out: set[str] = set()
    todo = [cls]
    seen: set[int] = set()
    while todo:
        node = todo.pop()
        out |= _INSTANCE_RUNNERS.get(reader.module_member(node) or "", frozenset())
        if not isinstance(node, ast.Name) or id(node) in seen:
            continue
        seen.add(id(node))
        defined = reader.classes.get(node.id)
        if defined is not None and id(defined) not in seen:
            seen.add(id(defined))
            todo += defined.bases
        for b in reader.bindings_of(node.id, node=node) if node.id in reader.bound_names else ():
            made = b.value
            if (b.how == "value" and isinstance(made, ast.Call) and reader.dotted(made.func) == "type"
                    and len(made.args) == 3 and isinstance(made.args[1], (ast.Tuple, ast.List))):
                todo += made.args[1].elts
    return frozenset(out)


def _holds_a_running_member(node: ast.AST, reader: _Reader) -> bool:
    """Whether `node` holds a member that runs code on what holds it: a
    loader's method (`_LOADER_METHODS`), a runner's own on what an instance
    runner made (`_runner_methods`), or what a lookup of a member that may
    run code gives (`_looks_up_a_runner`), there or through the names a
    value binds to it, as `run = spec.loader.exec_module` binds `run`."""
    todo = [node]
    seen: set[int] = set()
    while todo:
        node = todo.pop()
        if isinstance(node, ast.Attribute) and (
            node.attr in _LOADER_METHODS
            or node.attr in _RUNNER_METHODS and node.attr in _runner_methods(node.value, reader)
        ) or isinstance(node, ast.Call) and _looks_up_a_runner(node, reader, called=False):
            return True
        if isinstance(node, ast.Name) and node.id in reader.bound_names and id(node) not in seen:
            seen.add(id(node))
            todo += [b.value for b in reader.bindings_of(node.id, node=node) if b.how == "value"]
    return False


# The endings of a compiled extension module, which an import loads like a
# `.py`: its name runs up to the first dot, as `erp.cpython-311.so` is `erp`.
_EXTENSION_ENDS = (".so", ".pyd")
# The endings of a source module: `.py`, and `.pyw`, which Python on Windows
# imports as source too (`importlib.machinery.SOURCE_SUFFIXES` there).
_SOURCE_ENDS = (".py", ".pyw")
# What follows a module's name in the path of a file an import loads it from,
# in a folder on `sys.path`, for a lookup on disk by that name: its source,
# its bytecode with no source beside it, a compiled extension by an ending
# this Python loads (`importlib.machinery.EXTENSION_SUFFIXES`), each a module
# file's ending `_module_stem` reads, or a package's `__init__` with one.
_MODULE_ENDS = (*_SOURCE_ENDS, ".pyc", *importlib.machinery.EXTENSION_SUFFIXES)
_MODULE_FILE_ENDS = (*_MODULE_ENDS, *(f"/__init__{end}" for end in _MODULE_ENDS))


def _module_stem(path: str) -> str | None:
    """The name an import reaches a module file at `path` by, in the folder
    that holds it: the stem of a source file (`_SOURCE_ENDS`), of a `.pyc` file outside
    `__pycache__` (which Python loads with no source beside it), or of a
    compiled extension (`_EXTENSION_ENDS`), whose name runs to its first dot.
    None for any other file, a `.pyc` in `__pycache__` among them: Python
    reads one only for the source it was compiled from."""
    folders, _, name = path.rpartition("/")
    if name.endswith(_SOURCE_ENDS):
        return name.rpartition(".")[0]
    if name.endswith(".pyc") and "__pycache__" not in folders.split("/"):
        return name[: -len(".pyc")]
    if name.endswith(_EXTENSION_ENDS):
        return name.split(".", 1)[0] or None
    return None


def _may_name_a_module(name: str) -> bool:
    """Whether a file named `name` may be one an import loads
    (`_module_stem`): its ending may be picked at run time (`_RUN_TIME_PART`)."""
    return _module_stem(name) is not None or _RUN_TIME_PART in name.rpartition(".")[2]


def _may_write_a_module(call: ast.Call, target: ast.AST, reader: _Reader, cell: _Cell) -> bool:
    """Whether `call`, code of `cell` that writes `target`, may put a file an
    import loads there. A write of a file's contents (`_file_write`) makes one
    when a name it writes may be one (`_may_name_a_module`), or when the check
    can't read the names (`_written_names`), so `f"out_{row.origin}.csv"`
    makes none. A copy or move into a folder lands each file under its own
    name (`_copied_names`), and a move, rename or link may land a package
    folder, a name with no ending: so a copied name or a destination with
    none counts too, and a copy of `stock.csv` into a folder doesn't."""
    names = _written_names(target, reader, cell)
    if _file_write(call, reader) is not None:
        return names is None or any(_may_name_a_module(name) for name in names)
    copied = _copied_names(call, reader, cell)
    if isinstance(copied, _Unknown) or any(_may_name_a_module(name) or "." not in name for name in copied):
        return True
    into_folder = reader.dotted(call.func) in _COPY_INTO_FOLDER
    return names is None or any(_may_name_a_module(name) or not into_folder and "." not in name for name in names)


def _module_place(path: str) -> str | None:
    """Where an import reaches the module file at `path`, as a path with no
    ending: its folder joined to its name (`_module_stem`), `lib/erp` for
    `lib/erp.py`, `lib/erp.pyc` or `lib/erp.cpython-311-darwin.so`, and a
    package's own folder for its `__init__`, `lib/erp` for
    `lib/erp/__init__.py`. None for any other file."""
    stem = _module_stem(path)
    if stem is None:
        return None
    folder = path.rpartition("/")[0]
    if stem == "__init__":
        return folder or None
    return f"{folder}/{stem}" if folder else stem


def _module_names(path: str) -> set[str]:
    """The names an import may reach a Python file at `path` by: its stem
    (`_module_stem`), and each folder above it, as a package or a folder on
    `sys.path`."""
    stem = _module_stem(path)
    if stem is None:
        return set()
    *folders, _ = path.split("/")
    return {part for part in (*folders, stem) if part.isidentifier()}


# Folders an environment installs packages into: what an import reaches in
# one is an installed package, not a project's module.
_PACKAGE_FOLDERS = frozenset({"site-packages", "dist-packages"})


def _names_package_folder(path: str) -> bool:
    """Whether `path` has a part that names a folder an environment installs
    packages into (`_PACKAGE_FOLDERS`), as a virtual environment's
    `lib/python3.12/site-packages` does."""
    return not _PACKAGE_FOLDERS.isdisjoint(re.split(r"[/\\]", path))


# The endings of an archive an import reads modules out of once it is on
# `sys.path` (`_Modules.archive_on_path`).
_ARCHIVE_PATH_ENDS = (".zip", ".whl", ".egg", ".pyz")


class _ArgPlace(NamedTuple):
    """Where a call takes one of its arguments: its place among the
    arguments, and its keyword."""

    place: int
    keyword: str

    def given(self, call: ast.Call) -> ast.AST | None:
        """What `call` passes there: the argument at `place`, else the one
        by `keyword`; None where it passes neither."""
        return call.args[self.place] if len(call.args) > self.place else _keyword(call, self.keyword)


# Calls that write files whose paths the check doesn't read, as an archive's
# extraction does, or a copy of a folder: each by its last name, with where
# it takes the folder it writes; with neither, the working folder.
_TREE_DESTINATIONS = {
    "extractall": _ArgPlace(0, "path"), "extract": _ArgPlace(1, "path"),
    "unpack_archive": _ArgPlace(1, "extract_dir"), "copytree": _ArgPlace(1, "dst"),
}
_TREE_WRITERS = frozenset(_TREE_DESTINATIONS)
# Calls that may write the file one argument names, beyond those
# `_written_target` reads, with where each takes that file;
# `py_compile.compile` writes the bytecode its `cfile` names, which Python
# loads with no source.
_FILE_MAKERS = {
    "os.open": _ArgPlace(0, "path"), "codecs.open": _ArgPlace(0, "filename"),
    "urllib.request.urlretrieve": _ArgPlace(1, "filename"), "py_compile.compile": _ArgPlace(1, "cfile"),
}
# A zip archive's `writestr`, which writes a member under the name it takes first.
_WRITESTR = _ArgPlace(0, "zinfo_or_arcname")
# Writers that take the object they save first and its file after it, with
# where each takes the file (`_file_write`).
_WRITERS_OF_AN_OBJECT = {"torch.save": _ArgPlace(1, "f")}
_FILE_MAKER_NAMES = frozenset({*(name.rpartition(".")[2] for name in _FILE_MAKERS), "writestr"})
_WRITER_CALL_NAMES = frozenset({*_FILE_WRITE_NAMES, *_FILE_MAKER_NAMES, *_TREE_WRITERS, *_LAUNCH_NAMES, "touch"})


@dataclass(frozen=True)
class _WrittenModules:
    """The modules the pad's cells may write (`_written_modules`): the path
    of each Python file a write the check places reaches (`paths`); the
    names an import reaches the others by (`names`); whether a cell writes
    where the check can't place, or unpacks files it can't name
    (`anywhere`), so any module in the project may be one a cell wrote; and
    the folders a cell unpacks an archive or clones a repository into
    (`unpacked`), "." for the working folder, with whether one is a folder
    the check can't place (`unpacked_anywhere`), and where the first such
    unpack or clone runs (`_Reader.position`), None with none; whether a
    write the check can't place, or an unpack, may write a startup hook a
    new Python process runs (`hooks_anywhere`, `_unplaced_hook`); and
    whether a cell unpacks an archive or copies a folder's files into the
    installed packages (`trees_in_packages`, `_lands_in_packages`), which
    may put a startup hook there that each later Python process runs."""

    paths: frozenset[str]
    names: frozenset[str]
    anywhere: bool
    unpacked: frozenset[str] = frozenset()
    unpacked_anywhere: bool = False
    first_unpack: tuple[int, ...] | None = None
    hooks_anywhere: bool = False
    trees_in_packages: bool = False


def _made_file(call: ast.Call, reader: _Reader) -> ast.AST | None:
    """The file `call`, one that writes the file an argument names
    (`_FILE_MAKERS`), writes, as `urlretrieve`'s second argument; None for
    any other call."""
    where = _FILE_MAKERS.get(reader.module_member(call.func) or reader.dotted(call.func))
    return None if where is None else where.given(call)


def _written_modules(reader: _Reader, cells: Sequence[_Cell]) -> _WrittenModules:
    """The modules the pad's cells write: each Python file a write, copy or
    move of a file reaches (`_written_target`), a call that writes the file
    one argument names (`_FILE_MAKERS`), or a file an archive is given by
    name (`writestr`); each a program a cell runs names as a word, as
    `cp job.txt job.py` or `curl -o job.py` does; each folder a copy of a
    folder makes. A write to a path the check can't place or to a temporary
    folder, or an archive a cell unpacks (`_TREE_WRITERS`), may write any
    module, and each Python file the cell names in a string; the folder it
    unpacks into, or a `git clone` clones into, is kept too. An empty file
    `.touch()` makes counts, as a package's `__init__.py`. Read from
    `cells`."""
    paths: set[str] = set()
    names: set[str] = set()
    unpacked: set[str] = set()
    anywhere = unpacked_anywhere = hooks_anywhere = trees_in_packages = False
    # Where each unpack or clone runs.
    unpacks: list[tuple[int, ...]] = []
    for cell in cells:
        unplaced = False
        for call in cell.writer_calls:
            last = _last_name(call.func)
            if last in _WRITER_METHODS and _literal_non_module_args(call):
                # A frame or a chart saved to a file a literal names, not a module.
                continue
            target = None
            if last in _FILE_WRITE_NAMES:
                target = _written_target(call, reader, cell)
            elif last == "touch" and isinstance(call.func, ast.Attribute):
                target = call.func.value
            if target is None and last in _FILE_MAKER_NAMES:
                target = _WRITESTR.given(call) if last == "writestr" else _made_file(call, reader)
            if target is not None:
                got = reader.paths(target, cell=cell)
                if isinstance(got, _Unknown) or _TEMP_PATH in got:
                    unplaced = True
                    hooks_anywhere = hooks_anywhere or _unplaced_hook(call, target, reader, cell)
                else:
                    places = [path.replace("\\", "/") for path in got]
                    paths.update(p for p in places if _module_stem(p) is not None)
                    # A copy or move into a folder keeps the source's own
                    # name, so `erp.py` lands under it.
                    for stem in _copied_module_stems(call, reader, cell):
                        paths.update(f"{p.rstrip('/')}/{stem}" for p in places if _module_stem(p) is None)
                    # A link to a folder is a package by its own last name,
                    # as `os.symlink("/pkg", "lib/erp")` makes `erp`.
                    if last in _LINK_NAMES:
                        names.update(posixpath.basename(p.rstrip("/")) for p in places
                                     if _module_stem(p) is None and posixpath.basename(p.rstrip("/")).isidentifier())
                    # A write to a folder under a name the check can't read,
                    # as a loop over an archive's members does, may put any
                    # module there: the folder holds any, like an unpack.
                    for p in places:
                        if _RUN_TIME_PART in p:
                            unpacks.append(reader.position(call))
                            unpacked.add(posixpath.dirname(p.split(_RUN_TIME_PART, 1)[0] + "x") or ".")
            elif last in _TREE_WRITERS:
                dest = _TREE_DESTINATIONS[last].given(call)
                got = ["."] if dest is None else reader.paths(dest, cell=cell)
                unpacks.append(reader.position(call))
                trees_in_packages = trees_in_packages or dest is not None and _writes_site_packages(dest, reader) or (
                    not isinstance(got, _Unknown) and _lands_in_packages(got, reader.cwds)
                )
                if isinstance(got, _Unknown):
                    unplaced = unpacked_anywhere = hooks_anywhere = True
                    continue
                unpacked.update(got)
                if last == "copytree":
                    names.update(posixpath.basename(p.rstrip("/")) for p in got)
                else:
                    unplaced = hooks_anywhere = True
            elif last in _LAUNCH_NAMES and (
                cell.signs.names_a_py_file or cell.signs.names_unpack or "clone" in cell.signs.spelled
            ):
                name = reader.module_member(call.func) or reader.dotted(call.func)
                for command in _shell_commands(call, name, reader) if name in _LAUNCHERS else ():
                    folders = _run_folders(call, reader, cell)
                    steps = ([_ScriptCommand(words=command, cd=None)] if isinstance(command, list)
                             else _script_walk(command))
                    for words, cd in steps:
                        paths.update(path for w in words for path in [_REDIRECT.sub("", w).replace("\\", "/")]
                                     if _module_stem(path) is not None)
                        into = _cloned_into(words) or _unpacked_into(words) or _copied_tree_into(words)
                        if into is not None:
                            unpacks.append(reader.position(call))
                            unpacked.add(into)
                            unpacked_anywhere = unpacked_anywhere or _unreadable_word(into)
                        into = _unpacked_into(words) or _copied_contents_into(words)
                        trees_in_packages = trees_in_packages or (
                            into is not None and _lands_in_packages([into], folders)
                        )
                        if cd is not None:
                            folders = _shell_cd(folders, cd)
        if unplaced:
            anywhere = True
            names.update(cell.signs.py_files)
    return _WrittenModules(paths=frozenset(paths), names=frozenset(names), anywhere=anywhere,
                           unpacked=frozenset(unpacked), unpacked_anywhere=unpacked_anywhere,
                           first_unpack=min(unpacks, default=None), hooks_anywhere=hooks_anywhere,
                           trees_in_packages=trees_in_packages)


def _lands_in_packages(folders: Iterable[str], cwds: Sequence[str] | _Unknown) -> bool:
    """Whether one of `folders`, each read against the folders `cwds` a
    relative one is in, is inside the installed packages
    (`_names_package_folder`)."""
    return any(
        _names_package_folder(folder)
        or not isinstance(cwds, _Unknown) and any(_names_package_folder(_join(cwd, folder)) for cwd in cwds)
        for folder in folders
    )



# Programs that unpack an archive into a folder, so a module the pad imports
# may land there (`_unpacked_into`, `_CodeSigns.names_unpack`), by the flag
# each takes the folder after; `7z` takes it glued, as `-olib`.
_UNPACK_DEST_FLAGS: dict[str, frozenset[str]] = {
    "unzip": frozenset({"-d"}),
    "tar": frozenset({"-C", "--directory"}),
    "bsdtar": frozenset({"-C", "--directory"}),
    "unar": frozenset({"-o", "-output-directory", "--output-directory"}),
}
_SEVENZIP = frozenset({"7z", "7za", "7zr"})
_UNPACK_PROGRAMS = frozenset({*_UNPACK_DEST_FLAGS, *_SEVENZIP})
# The modes of tar and bsdtar that make, list, add to or compare an archive
# and unpack nothing, by their letters and their long flags; and the long
# flags of the one that unpacks (`_unpacks`).
_TAR_OTHER_MODES = frozenset("ctruAd")
_TAR_OTHER_LONG = frozenset({
    "--create", "--list", "--append", "--update", "--concatenate", "--catenate", "--delete", "--diff",
    "--compare",
})
_TAR_EXTRACT_LONG = frozenset({"--extract", "--get"})
# The letters of tar's and bsdtar's options that take a value, which may be
# glued to the letter (`-farchive.tar`, `-Cdest`), so the letters after one
# in a word are its value, not modes.
_TAR_VALUE_LETTERS = {"tar": frozenset("bCfFgHIKLNTVX"), "bsdtar": frozenset("bCfIsTX")}
# unzip's options that list, test or print an archive and unpack nothing.
_UNZIP_OTHER_MODES = frozenset({"-l", "-t", "-v", "-p", "-z", "-Z"})
# 7z's commands that unpack nothing: add, benchmark, delete, hash, info,
# list, rename, test and update.
_SEVENZIP_OTHER = frozenset({"a", "b", "d", "h", "i", "l", "rn", "t", "u"})


def _unpacks(verb: str, args: list[str]) -> bool:
    """Whether an archive program, by its name `verb` and the words after it,
    unpacks files: tar or bsdtar in extract mode, unzip but for listing,
    testing or printing (`_UNZIP_OTHER_MODES`), 7z but for its commands that
    unpack nothing (`_SEVENZIP_OTHER`), and unar. A tar mode counts only
    where tar reads one: a long flag, or a prefix of `--extract` or `--get`,
    which tar takes for it; a dash word's letters up to the first that takes
    a value (`_TAR_VALUE_LETTERS`); and every letter of tar's first word with
    no dash. Words that name no mode the check reads count as one that
    unpacks."""
    if verb in ("tar", "bsdtar"):
        modes: set[str] = set()
        for i, word in enumerate(args):
            if word.startswith("--"):
                flag = word.partition("=")[0]
                if len(flag) > 2 and any(name.startswith(flag) for name in _TAR_EXTRACT_LONG):
                    modes.add("x")
                elif flag in _TAR_OTHER_LONG:
                    modes.add("c")
            elif word.startswith("-") or i == 0:
                # Short letters, bundled as `-xzf`, or as tar's first word
                # with no dash, as `czf`; in a dash word, a value glued to a
                # letter that takes one, as `-farchive.tar`, ends them.
                letters = word.lstrip("-")
                if word.startswith("-"):
                    letters = next((letters[:k + 1] for k, letter in enumerate(letters)
                                    if letter in _TAR_VALUE_LETTERS[verb]), letters)
                modes.update(set(letters) & {"x", *_TAR_OTHER_MODES})
        return "x" in modes or not modes
    if verb == "unzip":
        return not any(word in _UNZIP_OTHER_MODES for word in args)
    if verb in _SEVENZIP:
        command = next((word for word in args if not word.startswith("-")), None)
        return command not in _SEVENZIP_OTHER
    return True
# Programs that copy files into a folder, keeping their own names, so a
# module lands there (`_copied_tree_into`).
_TREE_COPY_PROGRAMS = frozenset({"cp", "mv", "rsync", "install"})
# A word that names one of them, or a program that unpacks an archive.
_NAMES_UNPACK = re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, sorted(_UNPACK_PROGRAMS | _TREE_COPY_PROGRAMS))))


def _unpacked_into(words: list[str]) -> str | None:
    """The folder an archive-extraction command, by its words, unpacks into
    (`_UNPACK_PROGRAMS`, `_unpacks`): the folder after its destination
    flag, the one glued to `7z`'s `-o`, or "." for the working folder when
    it gives none; None for any other command, one that makes or lists an
    archive among them."""
    bare, _ = _command_words(words)
    if not bare:
        return None
    verb = _program_name(bare[0])
    if verb not in _UNPACK_PROGRAMS:
        return None
    args = bare[1:]
    if not _unpacks(verb, args):
        return None
    flags = _UNPACK_DEST_FLAGS.get(verb, frozenset())
    for i, word in enumerate(args):
        if verb in _SEVENZIP and word.startswith("-o") and len(word) > 2:
            return word[2:]
        flag, equals, value = word.partition("=")
        if flag in flags:
            return value if equals else (args[i + 1] if i + 1 < len(args) else ".")
    return "."


# Calls and members that name a folder inside an environment's installed
# packages, where a written module joins the packages already there
# (`_writes_site_packages`): `site`'s and `sysconfig`'s folder lookups, a
# spec a `find_spec` makes, and the location of a package already imported.
_SITE_PACKAGES_CALLS = frozenset({
    "site.getsitepackages", "site.getusersitepackages", "site.getuserbase", "sysconfig.get_path",
    "sysconfig.get_paths", "importlib.util.find_spec", "importlib.find_loader", "pkgutil.get_loader",
})
_SITE_PACKAGES_MEMBERS = frozenset({"__file__", "origin", "submodule_search_locations", "__path__"})


def _writes_site_packages(target: ast.AST, reader: _Reader, *, depth: int = 0) -> bool:
    """Whether `target`, a path a write names, lands inside an environment's
    installed packages: it reads a folder lookup of `site` or `sysconfig`
    (`_SITE_PACKAGES_CALLS`), the origin or path of a spec or an imported
    package (`_SITE_PACKAGES_MEMBERS`), or a text with a `site-packages` or
    `dist-packages` part (`_names_package_folder`), as a path into a
    virtual environment a cell made does, there or through a value a name
    is bound to (`_Reader.read_bindings`), up to `_PARAM_DEPTH` names."""
    if depth > _PARAM_DEPTH:
        return False
    for node in ast.walk(target):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _names_package_folder(node.value):
            return True
        if isinstance(node, ast.Attribute) and node.attr in _SITE_PACKAGES_MEMBERS and (
            node.attr != "origin" or _may_be_a_spec(node.value, reader, depth=depth)
        ):
            return True
        if isinstance(node, ast.Call) and reader.module_member(node.func) in _SITE_PACKAGES_CALLS:
            return True
        if isinstance(node, ast.Name) and any(reader.read_bindings(
            "site packages", node, depth=depth,
            compute=lambda b, at: b.how == "value" and _writes_site_packages(b.value, reader, depth=at),
        )):
            return True
    return False


def _may_be_a_spec(value: ast.AST, reader: _Reader, *, depth: int = 0) -> bool:
    """Whether `value`, what an `origin` lookup reads, may be a module's
    spec: a call of `_SITE_PACKAGES_CALLS` (`find_spec` among them), a
    module's `__spec__`, or a name bound to either (`_Reader.read_bindings`), up to
    `_PARAM_DEPTH` names. `origin` is a common field of data too, as a
    flight's row has, so a lookup on any other value reads no package's
    place."""
    if depth > _PARAM_DEPTH:
        return False
    if isinstance(value, ast.Call):
        return reader.module_member(value.func) in _SITE_PACKAGES_CALLS
    if isinstance(value, ast.Attribute):
        return value.attr == "__spec__"
    if isinstance(value, ast.Name):
        return any(reader.read_bindings(
            "a spec", value, depth=depth,
            compute=lambda b, at: b.how == "value" and _may_be_a_spec(b.value, reader, depth=at),
        ))
    return False


def _copied_names(call: ast.Call, reader: _Reader, cell: _Cell) -> list[str] | _Unknown:
    """The basenames of the files `call`, a copy or a move of a file into a
    folder (`_COPY_INTO_FOLDER`), reads from, which each keeps under the
    folder it lands in, so `shutil.copy("a/erp.py", "lib")` writes
    `lib/erp.py`. Empty for any other call; unknown for a source the check
    can't place."""
    if reader.dotted(call.func) not in _COPY_INTO_FOLDER:
        return []
    source = _target_arg(call, keywords=("src",))
    got = reader.paths(source, cell=cell) if source is not None else _UNKNOWN
    if isinstance(got, _Unknown):
        return _UNKNOWN
    return [posixpath.basename(p.replace("\\", "/")) for p in got]


def _copied_module_stems(call: ast.Call, reader: _Reader, cell: _Cell) -> set[str]:
    """The basenames of the Python files `call`, a copy or a move of a file
    into a folder, reads from (`_copied_names`). Empty for any other call,
    or a source the check can't place."""
    names = _copied_names(call, reader, cell)
    return set() if isinstance(names, _Unknown) else {name for name in names if _module_stem(name) is not None}


def _copied_tree_into(words: list[str]) -> str | None:
    """The folder a copy command, by its words, copies files into
    (`_TREE_COPY_PROGRAMS`), keeping their names: the last operand, when it
    is a folder (it ends in `/`, or more than one source feeds it) or a
    source is a folder or a listing's wildcard, so a module may land there.
    None for any other command, or a single file copied to a file, which the
    `.py` words the walk already reads cover."""
    bare, _ = _command_words(words)
    if len(bare) < 3 or _program_name(bare[0]) not in _TREE_COPY_PROGRAMS:
        return None
    operands = [w for w in bare[1:] if not w.startswith("-")]
    if len(operands) < 2:
        return None
    dest, sources = operands[-1], operands[:-1]
    folder = dest.endswith("/") or len(sources) > 1
    tree = any(s.endswith("/") or "*" in s or "?" in s or "[" in s for s in sources)
    return dest if folder or tree else None


def _copied_contents_into(words: list[str]) -> str | None:
    """The folder a copy command, by its words (`_TREE_COPY_PROGRAMS`), puts
    a folder's own files into, under names the check can't read: its last
    operand, where a source is a listing's wildcard or ends with `/` or `/.`,
    as `cp hooks/* lib/`, `cp -r hooks/. lib/` and `rsync -a hooks/ lib/`
    do. None for any other command."""
    bare, _ = _command_words(words)
    if len(bare) < 3 or _program_name(bare[0]) not in _TREE_COPY_PROGRAMS:
        return None
    operands = [w for w in bare[1:] if not w.startswith("-")]
    if len(operands) < 2:
        return None
    contents = any(s.endswith(("/", "/.")) or s == "." or "*" in s or "?" in s or "[" in s for s in operands[:-1])
    return operands[-1] if contents else None


def _cloned_into(words: list[str]) -> str | None:
    """The folder a `git clone` command, by its words, clones into: the
    word after the repository, or the repository's own name; None for any
    other command."""
    bare, _ = _command_words(words)
    if len(bare) < 3 or bare[0].rsplit("/", 1)[-1] != "git" or "clone" not in bare:
        return None
    operands = [w for w in bare[bare.index("clone") + 1:] if not w.startswith("-")]
    if not operands:
        return None
    if len(operands) > 1:
        return operands[1]
    return posixpath.basename(operands[0].rstrip("/")).removesuffix(".git")


def _exists_on_disk(path: Path, *, folder: bool = False) -> bool:
    """Whether `path` is a file, or a folder with `folder`, on disk now, as
    an import would find a module there. False for a name too long to be a
    file; True where the system can't say otherwise, as when it refuses to
    look, since a cell may have written it."""
    try:
        return path.is_dir() if folder else path.is_file()
    except OSError as exc:
        return exc.errno != errno.ENAMETOOLONG


def _literal_non_module_args(call: ast.Call) -> bool:
    """Whether each argument `call` passes is a literal that names no
    Python file, so a write it makes to the file an argument names writes
    no module."""
    given = [*call.args, *(k.value for k in call.keywords)]
    return all(
        isinstance(a, ast.Constant) and not (isinstance(a.value, str) and _module_stem(a.value) is not None)
        for a in given
    )


class _WrittenName(NamedTuple):
    """A name an import may reach a file a cell writes by, and whether a
    module or a package is there by that name, rather than a portion of a
    namespace package: the file's own stem, or a folder above it with an
    `__init__.py`."""

    name: str
    package: bool


@dataclass(frozen=True)
class _UnreadModuleEntry:
    """A module a cell stores in `sys.modules` from code the check can't
    read (`_stored_modules`): the name it stores it under, None for a name
    the check can't read, and the place of that cell (`_Cell.order`)."""

    name: str | None
    order: tuple[int, ...]


@dataclass(frozen=True)
class _StoredReach:
    """The earliest place (`_Cell.order`) a module stored in `sys.modules`
    reaches each name an import may ask for (`_Modules.unread`): the name
    it was stored under (`exact`), each package above it (`above`), and any
    name at all for one stored under a name the check can't read
    (`anywhere`)."""

    exact: Mapping[str, tuple[int, ...]]
    above: Mapping[str, tuple[int, ...]]
    anywhere: tuple[int, ...] | None

    @classmethod
    def of(cls, stored: Sequence[_UnreadModuleEntry]) -> _StoredReach:
        exact: dict[str, tuple[int, ...]] = {}
        above: dict[str, tuple[int, ...]] = {}
        anywhere: tuple[int, ...] | None = None
        for entry in stored:
            if entry.name is None:
                anywhere = entry.order if anywhere is None else min(anywhere, entry.order)
                continue
            exact[entry.name] = min(exact.get(entry.name, entry.order), entry.order)
            parts = entry.name.split(".")
            for i in range(1, len(parts)):
                package = ".".join(parts[:i])
                above[package] = min(above.get(package, entry.order), entry.order)
        return cls(exact=exact, above=above, anywhere=anywhere)

    def reaches(self, module: str, order: tuple[int, ...]) -> bool:
        """Whether a store at or before `order` reaches `module`: one under
        its name, a package above it, or a module below it."""
        if self.anywhere is not None and self.anywhere <= order:
            return True
        parts = module.split(".")
        names = (".".join(parts[:i]) for i in range(1, len(parts) + 1))
        return any((at := self.exact.get(name)) is not None and at <= order for name in names) or (
            (below := self.above.get(module)) is not None and below <= order
        )


class _Modules:
    """The modules an import in the pad may load that the check can't
    read: one a cell writes, one a Python file that is in the project now
    but wasn't when the turn began, and, once a cell changes `sys.path`,
    writes where the check can't place, or in code `python -c` runs, one in
    the project or shared folder that a folder on `sys.path` reaches
    (`path_entries`). The pad runs `python -P`, so neither folder is on
    `sys.path` by itself. The writes are read the first time a check asks,
    outside any call it reads, so a write in a function counts with what
    each call passes it (`_Reader.call_sites`); the folders the first time
    an import asks, since most cells import only installed packages.

    A folder with no `__init__.py` is a portion of a namespace package,
    which an import reaches only past every folder on `sys.path`, so never
    ahead of an installed module of that name: one Python ships
    (`sys.stdlib_module_names`) is always installed."""

    def __init__(self, reader: _Reader, snapshots: Sequence[ProjectSnapshot]) -> None:
        self._reader = reader
        self._snapshots = snapshots
        self._new: dict[str, bool] = {}
        # The modules cells store in `sys.modules` from code the check can't
        # read, which `_build_defined` sets once it reads what such code
        # filled (`_stored_modules`).
        self.stored: tuple[_UnreadModuleEntry, ...] = ()
        # The earliest place a store in `stored` reaches each name, kept for
        # the `stored` it was built from (`_stored_reach`).
        self._reach: tuple[tuple[_UnreadModuleEntry, ...], _StoredReach] | None = None
        # How many times `unread` was asked, so a read can tell whether what
        # it found may change as `stored` grows (`_names_tainted`).
        self.asked = 0

    @cached_property
    def _naming(self) -> list[_Cell]:
        """The cells whose writes may name a module: every cell, once one
        names a Python file, as a name one cell binds may name the file
        another writes; else those that copy, unpack or clone a tree of
        files."""
        cells = self._reader.cells
        if any(c.signs.names_a_py_file for c in cells):
            return list(cells)
        if self.path_changed:
            # A sys.path change may put a folder a cell wrote modules into on
            # the import path, even when no cell names a `.py` file.
            return [c for c in cells if c.writer_calls]
        return [c for c in cells if any(
            _last_name(call.func) in _TREE_WRITERS or _last_name(call.func) in _LINK_NAMES
            or ("clone" in c.signs.spelled or c.signs.names_unpack) and _last_name(call.func) in _LAUNCH_NAMES
            for call in c.writer_calls
        )]

    @cached_property
    def _written(self) -> _WrittenModules:
        """What the cells whose writes may name a module write (`_written_modules`)."""
        return _written_modules(self._reader, self._naming)

    @cached_property
    def _anywhere(self) -> bool:
        """Whether a cell writes where the check can't place, read from the
        rest of the cells only once an import asks about a project module."""
        return self._written.anywhere or self._rest_written.anywhere

    @cached_property
    def _rest_written(self) -> _WrittenModules:
        """What the cells `_naming` leaves out write (`_written_modules`),
        read only once a check asks about one."""
        naming = {id(c) for c in self._naming}
        return _written_modules(self._reader, [c for c in self._reader.cells if id(c) not in naming])

    @cached_property
    def project(self) -> ProjectSnapshot | None:
        """The snapshot of the project folder; None without one."""
        return next((s for s in self._snapshots if s.label == "project"), None)

    @cached_property
    def written(self) -> frozenset[str]:
        """The names an import reaches a module a cell writes by: each
        written file's stem, each folder above one that is a package, with
        an `__init__.py` written or in the snapshot, and the names a write
        the check can't place may reach (`_WrittenModules.names`)."""
        return frozenset(self._written.names | {w.name for w in self._written_folders if w.package})

    @cached_property
    def _portions(self) -> frozenset[str]:
        """The other folders above a file a cell writes: portions of a
        namespace package."""
        return frozenset(w.name for w in self._written_folders if not w.package) - self.written

    @cached_property
    def _written_folders(self) -> list[_WrittenName]:
        """Each name an import may reach a written file by: the stem, then
        each folder (`_WrittenName`)."""
        written = self._written.paths
        project = self.project
        out: list[_WrittenName] = []
        for path in written:
            parts = path.split("/")
            out.append(_WrittenName(_module_stem(path) or "", package=True))
            for i, folder in enumerate(parts[:-1]):
                init = posixpath.normpath("/".join([*parts[: i + 1], "__init__.py"]))
                rel = _relative(init, project) if project is not None else None
                out.append(_WrittenName(folder, package=init in written or rel is not None and rel in project.files))
        return [w for w in out if w.name.isidentifier()]

    def _stored_reach(self) -> _StoredReach:
        """The places `stored` reaches each name from (`_StoredReach`), built
        again only once `stored` grows."""
        if self._reach is None or self._reach[0] is not self.stored:
            self._reach = (self.stored, _StoredReach.of(self.stored))
        return self._reach[1]

    def unread(self, module: str, *, cell: _Cell) -> bool:
        """Whether importing `module` in `cell` may run code the check can't
        read. A relative name counts, since the check can't place its
        package. Code `python -c` runs (`_Cell.launched`) has its working
        folder on `sys.path`, as if a cell had put it there. So does a
        module a cell up to this one stored in `sys.modules` from such code
        (`stored`), which an import gets before it looks anywhere else, or a
        package or a module under one stored so; any module, where the
        check can't read the name it was stored under."""
        self.asked += 1
        if module.startswith("."):
            return True
        if self.stored and self._stored_reach().reaches(module, cell.order):
            return True
        top = module.partition(".")[0]
        if not top:
            return False
        if top in self.written or self._new_on_disk(top):
            return True
        installed = not _loaded_first(top) and top not in sys.stdlib_module_names
        if self.wrote_site_packages and installed:
            # A cell wrote a module into the installed packages, where an
            # import finds it beside them: any installed package, not one
            # Python ships or loads first, which an import gets ahead of it.
            return True
        if self.archive_on_path and installed:
            # A cell put an archive on `sys.path`, which zipimport reads any
            # module out of: any installed package, read behind what Python
            # ships and loads first.
            return True
        changed = self.path_changed or cell.launched
        if changed and (top in self._portions and top not in sys.stdlib_module_names
                        or not self._found_first(top) and self._unpacked_on_path(cell)):
            return True
        if not (changed or self._anywhere):
            return False
        entries = self.path_entries
        if entries is None or self._anywhere:
            return self._in_folders(top)
        return self._reachable(top, entries | ({""} if cell.launched else set()))

    def any_unread(self, cells: Sequence[_Cell]) -> bool:
        """Whether an import in one of `cells`, by a statement or a literal
        importer (`_Cell.looked_up`), loads a module the check can't read."""
        return any(
            any(self.unread(a.name, cell=cell) for a in node.names) if isinstance(node, ast.Import)
            else not node.level and bool(node.module) and self.unread(node.module, cell=cell)
            for cell in cells for node in cell.signs.imports
        ) or any(self.unread(node.id, cell=cell) for cell in cells for node in cell.looked_up)

    def walks_unread(self, folders: Sequence[str] | None, *, cell: _Cell) -> bool:
        """Whether walking `folders` imports a module the check can't read:
        the folders absent or unreadable (None) holds, since the walk may
        reach any; else a folder holds when a module or package directly
        under it is one `unread` can't read (`_names_under`)."""
        if folders is None:
            return True
        return any(self.unread(name, cell=cell) for folder in folders for name in self._names_under(folder))

    @cached_property
    def wrote_site_packages(self) -> bool:
        """Whether a cell writes a module into the installed packages
        (`_writes_site_packages`), where an import finds it: a file write to
        such a path, or a copy, move, rename or link into one, that may make
        a module or a package (`_may_write_a_module`). Read once per check,
        where an import asks."""
        reader = self._reader
        for cell in reader.cells:
            for call in cell.writer_calls:
                if _last_name(call.func) not in _FILE_WRITE_NAMES:
                    continue
                target = _written_target(call, reader, cell)
                if target is not None and _may_write_a_module(call, target, reader, cell) and _writes_site_packages(
                    target, reader,
                ):
                    return True
        return False

    @cached_property
    def _hook_writes(self) -> list[_HookWrite]:
        """Where each call of the cells may write a startup hook
        (`_hook_write`), read only in the cells that name one."""
        reader = self._reader
        return [
            write for cell in reader.cells if cell.signs.names_a_startup_hook for call in cell.calls
            for write in [_hook_write(call, reader, cell)] if write.any
        ]

    @cached_property
    def hooks_written(self) -> bool:
        """Whether a cell writes a startup hook that each Python process
        started after it runs as it starts (`_HookWrite.runs_at_start`), or
        unpacks or copies a folder's files into the installed packages,
        where one may land (`_WrittenModules.trees_in_packages`); the cells
        that do are among those `_naming` reads."""
        return self._written.trees_in_packages or any(write.runs_at_start for write in self._hook_writes)

    @cached_property
    def hooks_may_run(self) -> bool:
        """Whether a new Python process may run a startup hook a cell wrote:
        one a cell writes (`hooks_written`), or one a write the check can't
        place, or an unpack, may write (`_WrittenModules.hooks_anywhere`)."""
        return self.hooks_written or self._written.hooks_anywhere or self._rest_written.hooks_anywhere

    def may_hold_pth(self, folder: str) -> bool:
        """Whether `folder`, relative to the project, may hold a `.pth` file
        whose import lines `site.addsitedir` runs: one directly in it in the
        snapshot, one a cell writes there (`_HookWrite.paths`), or one on
        disk now, as an earlier cell may have written; or any once a cell
        writes a startup hook a later Python process runs, one where the
        check can't place it among them (`hooks_may_run`), or unpacks a
        tree of files."""
        written = self._written
        project = self.project
        if project is None or self.hooks_may_run or written.unpacked or written.unpacked_anywhere:
            return True
        prefix = f"{folder}/" if folder else ""
        cell_written = (rel for write in self._hook_writes for path in write.paths
                        for rel in [_relative(path, project)] if rel is not None)
        if any(path.startswith(prefix) and path.endswith(".pth") and "/" not in path[len(prefix):]
               for path in (*project.files, *cell_written)):
            return True
        try:
            return any((project.root / folder).glob("*.pth"))
        except OSError:
            return True

    def _names_under(self, folder: str) -> frozenset[str]:
        """The names an import reaches a module or package directly inside
        `folder` by: a cell-written or snapshot module file there, or a
        package's `__init__` (`_module_place`)."""
        base = folder.rstrip("/")
        prefix = "" if base in ("", ".") else f"{base}/"
        places = {place for path in self._written.paths for place in [_module_place(path)] if place is not None}
        for snap in self._snapshots:
            places |= snap.module_places
        return frozenset(
            name for place in places if place.startswith(prefix)
            for name in [place[len(prefix):]] if "/" not in name and name.isidentifier()
        )

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

    @cached_property
    def archive_on_path(self) -> bool:
        """Whether a cell puts an archive on `sys.path` (`_ARCHIVE_PATH_ENDS`),
        a `.zip`, `.whl`, `.egg` or `.pyz` it copies, downloads or finds there,
        which zipimport reads any module out of."""
        entries = self.path_entries
        return entries is not None and any(entry.lower().endswith(_ARCHIVE_PATH_ENDS) for entry in entries)

    @cached_property
    def path_entries(self) -> frozenset[str] | None:
        """The folders under the project the cells put on `sys.path`, each
        relative to it, "" for the project itself, as `_Reader.paths` reads
        what each change adds; a folder outside it reaches none of its
        files. None when a change adds one the check can't place, or one
        above the project, or the pad's working folder may move."""
        reader = self._reader
        project = self.project
        out: set[str] = set()
        aliases = {name for cell in reader.cells for name in cell.signs.path_aliases}
        for cell in reader.cells:
            signs = cell.signs
            for node in (*signs.path_changes, *(n for name, n in signs.list_changes if name in aliases)):
                if not (reader.dotted(node) in _SYS_PATH_CHANGES or reader.module_member(node) in _SYS_PATH_CHANGES):
                    continue
                added = _path_added(node, cell)
                if added is None or reader.cwds != ["."] or project is None:
                    return None
                got = reader.paths(added, cell=cell)
                if isinstance(got, _Unknown) or any(_TEMP_PATH in p or _unreadable_text(p) for p in got):
                    return None
                for path in got:
                    rel = _relative(path, project)
                    if rel is not None and (rel == ".." or rel.startswith("../")):
                        return None
                    if rel is not None:
                        out.add(rel)
        return frozenset(out)

    def _found_first(self, top: str) -> bool:
        """Whether an import of the top-level module `top` gets one no
        unpack wrote: one built into Python, frozen in it or that the boot
        loads (`_loaded_first`), or one a cell imported before the first
        unpack or clone ran, which Python keeps as loaded (`_imported_early`)."""
        return _loaded_first(top) or top in self._imported_early

    @cached_property
    def _imported_early(self) -> frozenset[str]:
        """The top-level modules a cell surely imports, at its top level,
        before the first unpack or clone runs, in a cell that didn't stop on
        an error."""
        first = self._written.first_unpack
        reader = self._reader
        if first is None:
            return frozenset()
        return frozenset(
            name.partition(".")[0]
            for cell in reader.cells if cell.host not in reader.stopped_early
            for stmt in cell.signs.imports
            if cell.scope_of(stmt) is None and cell.runs_surely(stmt) and reader.position(stmt) < first
            for name in ([a.name for a in stmt.names] if isinstance(stmt, ast.Import) else
                         [stmt.module] if stmt.module and not stmt.level else [])
        )

    def _unpacked_on_path(self, cell: _Cell) -> bool:
        """Whether a folder a cell unpacks an archive or clones a repository
        into may be, or hold, a folder on `sys.path` (`path_entries`), the
        working folder too for code `python -c` runs: what lands there is
        on disk only once the cell runs, so any import but of a module no
        folder there can stand in for (`_Modules._found_first`) may load
        it, one Python ships among them: a folder put ahead of the standard
        library's shadows it."""
        written = self._written
        if not written.unpacked and not written.unpacked_anywhere:
            return False
        entries = self.path_entries
        if written.unpacked_anywhere or entries is None:
            return True
        project = self.project
        folders = {_relative(f, project) if project is not None else None for f in written.unpacked}
        if None in folders:
            # A folder outside the project, on `sys.path` by its own path.
            return True
        # A folder is on `sys.path`, under one, or above one there.
        return any(
            not folder or not entry or entry == folder
            or entry.startswith(f"{folder}/") or folder.startswith(f"{entry}/")
            for folder in folders if folder is not None for entry in (*entries, *([""] if cell.launched else []))
        )

    def _reachable(self, top: str, entries: AbstractSet[str]) -> bool:
        """Whether an import reaches a Python file in the project by the name
        `top` through one of `entries`, folders on `sys.path`: `top.py`, a
        sourceless `top.pyc`, a compiled extension (`_module_place`) or a
        package `top/` there, or a namespace portion `top/` but for a module
        Python ships. Where the walk stopped short, the folders are read on
        disk (`_MODULE_FILE_ENDS`)."""
        snap = self.project
        if snap is None:
            return False
        for entry in entries:
            base = f"{entry}/" if entry else ""
            if f"{base}{top}" in snap.module_places:
                return True
            if f"{base}{top}" in snap.dirs and top not in sys.stdlib_module_names:
                return True
            if (snap.failed or not snap.complete) and (
                any(_exists_on_disk(snap.root / base / f"{top}{end}") for end in _MODULE_FILE_ENDS)
                or _exists_on_disk(snap.root / base / top, folder=True)
            ):
                return True
        return False

    def _new_on_disk(self, top: str) -> bool:
        """Whether the project or shared folder, or a folder under the
        project the cells put on `sys.path` (`path_entries`), holds a module
        file or package by the name `top` now (`_MODULE_FILE_ENDS`), as
        `top.py` or `top/__init__.py`, though the walk when the turn began
        didn't: a cell this turn wrote it, however the check read its write."""
        if top not in self._new:
            entries = self.path_entries or frozenset()
            self._new[top] = any(
                rel not in snap.files and _exists_on_disk(snap.root / rel)
                for snap in self._snapshots if snap.complete and not snap.failed
                for base in ("", *(f"{e}/" for e in entries if e and snap.label == "project"))
                for rel in (f"{base}{top}{end}" for end in _MODULE_FILE_ENDS)
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
            _exists_on_disk(snap.root / f"{top}{end}")
            for snap in self._snapshots if snap.failed or not snap.complete for end in _MODULE_FILE_ENDS
        )


def _loaded_first(top: str) -> bool:
    """Whether an import of the top-level module `top` finds it before any
    folder on `sys.path` is searched: a module built into Python or frozen
    in it, or one the pad's boot imports before any cell runs
    (`BOOT_IMPORTS`), which the import gets as loaded."""
    return (top in sys.builtin_module_names or top in BOOT_IMPORTS
            or importlib.machinery.FrozenImporter.find_spec(top) is not None)


def _path_added(node: ast.AST, cell: _Cell) -> ast.AST | None:
    """The code that gives the folders a change of `sys.path`, one of
    `_CodeSigns.path_changes` or `list_changes` in `cell`, adds: the folder
    `insert`, `append` or `addsitedir` is handed, the list `extend` or `+=`
    is, or the value an assignment to the list or an item of it stores.
    None when the check can't find it."""
    parent = cell.parents.get(node)
    if isinstance(parent, ast.Call) and parent.func is node:
        args = [a for a in parent.args if not isinstance(a, ast.Starred)]
        if len(args) != len(parent.args) or not args:
            return None
        return args[-1] if _last_name(node) in ("insert", "__setitem__") else args[0]
    while parent is not None and not isinstance(parent, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        parent = cell.parents.get(parent)
    return parent.value if parent is not None else None


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
    entry read under a name the check resolves, as `vars(runpy)[k]` is, or
    a `getattr` by a key it reads as literal texts
    (`_Reader.looked_up_member`). `_Reader.module_member` says which of
    them is one."""
    last = _member_name(node)
    if last in _RUNNER_NAMES or last.startswith(_PYAPI_PREFIXES) or last in reader.runner_aliases:
        return True
    if isinstance(node, ast.Call):
        # `getattr(timeit, n)` with `n` bound to "Timer", read as `timeit.Timer`.
        return reader.looked_up_member(node) is not None
    return (isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Constant)
            and reader.module_member(node) is not None)


def _looks_up_a_runner(call: ast.Call, reader: _Reader, *, called: bool) -> bool:
    """Whether `call` looks up a member that may run code, by a key read
    with `_literal_texts`. A `getattr` holds on a loader, on a module that
    holds a runner, a launcher, a runner of text or a class whose instances
    run code (`_MAKER_MODULES`), or on what an instance runner made
    (`_runner_methods`), where the key can't be read or a text names a
    member there that runs code (`_runs_on_holder`): `getattr(runpy, name)`
    may be `runpy.run_path`, while `getattr(types, n)` over a list of
    harmless names holds nothing. Where the lookup is `called` right away
    and one text alone names a runner of text or a class whose instances
    run code (`_READ_AS_MEMBER`), the call is read as the literal key's
    where the lookup reads as that member, with no default or a constant
    one (`_Reader.looked_up_member`): `getattr(timeit, n)("pass")` with `n`
    bound to "Timer" runs the literal it is handed, and `getattr(cProfile,
    n)()` makes a profiler whose `run` holds where it is called. An
    `operator.methodcaller` or `operator.attrgetter` holds where a key
    can't be read or a part of one is named as a member that runs code
    (`_RUNNING_MEMBERS`): the holder it is later called on may be any."""
    last = _last_name(call.func)
    if last in _MEMBER_GETTER_NAMES and reader.module_member(call.func) in _MEMBER_GETTERS:
        keys = call.args[:1] if last == "methodcaller" else call.args
        found = [_literal_texts(key, reader) for key in keys]
        return any(texts is None or any(_member_runs_code(text) for text in texts) for texts in found)
    if not (isinstance(call.func, ast.Name) and call.func.id == "getattr" and len(call.args) >= 2):
        return False
    key = call.args[1]
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return False
    held = call.args[0]
    texts = _literal_texts(key, reader)
    if _last_name(held) == "loader":
        return texts is None or any(text in _LOADER_METHODS for text in texts)
    module = reader.module_member(held)
    if module in _RUNNER_MODULES or module in _LAUNCHER_MODULES or module in _MAKER_MODULES:
        if texts is None:
            return True
        running = [text for text in dict.fromkeys(texts) if _runs_on_holder(module, text)]
        member = f"{module}.{running[0]}" if len(running) == 1 else None
        return bool(running) and not (
            called and member in _READ_AS_MEMBER and reader.looked_up_member(call) == member
        )
    methods = _runner_methods(held, reader)
    return bool(methods) and (texts is None or any(text in methods for text in texts))


# The members a `getattr` called right away by a key bound to literal texts
# is read as (`_looks_up_a_runner`): a runner of text, whose literal code is
# read, and a class whose instances run code where their methods are called.
_READ_AS_MEMBER = frozenset({*_STRING_RUNNERS, *_INSTANCE_RUNNERS})


def _runs_on_holder(module: str, text: str) -> bool:
    """Whether the member `text` of `module`, a module that holds a runner
    or a launcher, runs code or holds what does: one of its runners or
    launchers (`_RUNNERS_BY_MODULE`), a class or call whose result runs
    code (`_INSTANCE_RUNNERS`), as `cProfile.Profile` makes a profiler
    whose `run` does, or a module below it that holds one, as
    `importlib.util` does."""
    full = f"{module}.{text}"
    return text in _RUNNERS_BY_MODULE.get(module, {}) or full in _INSTANCE_RUNNERS or full in _RUNNER_HOLDERS


def _member_runs_code(text: str) -> bool:
    """Whether `text`, a member's name as a getter of members takes it, a
    dotted path for `attrgetter`, names in any part a member that runs code
    on what holds it (`_RUNNING_MEMBERS`) or a PyRun_ function."""
    return any(part in _RUNNING_MEMBERS or part.startswith(_PYAPI_PREFIXES) for part in text.split("."))


def _imports_unread(call: ast.Call, reader: _Reader, cell: _Cell, *, modules: _Modules) -> bool:
    """Whether `call`, an importer's, imports a module `modules` can't read,
    or one by a name the check can't read (`_literal_texts`). A
    `pkgutil.resolve_name` or `pydoc.locate` target names the module before
    its colon. A name read as no text at all imports nothing only where it
    is a parameter of a function nothing calls (`_never_given`), and so
    does a relative name given no package. A call given no name but a
    `**` mapping takes its name from that mapping, which the check doesn't
    read."""
    member = reader.module_member(call.func)
    if member == "pkgutil.walk_packages":
        return _walk_packages_unread(call, reader, cell, modules=modules)
    target = _target_arg(call, keywords=("name", "path"))
    if target is None:
        return any(k.arg is None for k in call.keywords)
    texts = _literal_texts(target, reader)
    if texts is None and member in _STRING_IMPORTERS:
        # A thing, not its name, as `help(df.merge)` is, imports nothing;
        # text the check can't read, as a file read gives, may be a name.
        return _may_be_text(target, reader, unknown=False)
    if texts == []:
        return not _never_given(target, reader)
    if texts and all(text.startswith(".") for text in texts) and len(call.args) < 2 and not call.keywords:
        # A relative name with no package to read it in: the import fails
        # before it loads anything.
        return False
    return texts is None or any(modules.unread(text.partition(":")[0], cell=cell) for text in texts)


def _walk_packages_unread(call: ast.Call, reader: _Reader, cell: _Cell, *, modules: _Modules) -> bool:
    """Whether `pkgutil.walk_packages` imports a module `modules` can't read:
    it imports every package it finds under the folders it is given, so it
    is held when the folders are absent, when the check can't read them, or
    when one holds a module a cell wrote or a project module a `sys.path`
    change reaches (`_Modules.walks_unread`)."""
    target = _target_arg(call, keywords=("path",))
    folders = _literal_texts(target, reader) if target is not None else None
    return modules.walks_unread(folders, cell=cell)


def _runs_unread(call: ast.Call, reader: _Reader, cell: _Cell, *, read: _ExecRead, modules: _Modules) -> bool:
    """Whether `call`, code of `cell`, runs code in this process that the
    check can't read (`_unread_run_at`). A runner of text handed a lambda
    or a function the check reads runs read code."""
    func = call.func
    last = _member_name(func)
    if isinstance(func, ast.Attribute) and last in _LOADER_METHODS:
        return True
    if isinstance(func, ast.Call) and _looks_up_a_runner(func, reader, called=True):
        return True
    if isinstance(func, ast.Name) and func.id in reader.runner_aliases and _holds_a_running_member(func, reader):
        # A name bound to a loader's or a runner's method, as `go = pr.run`,
        # or to a lookup of one, as `go = operator.methodcaller("run_path")`.
        return True
    if reader.module_member(func) in _APP_RUNNERS:
        return _app_runner_unread(call, reader)
    name = reader.module_member(func) if _runner_named(func, reader) else None
    if name == _SITE_DIR_ADDER:
        return _adds_unread_site(call, reader, cell, modules=modules)
    if name in _CODE_RUNNERS or name is not None and name.startswith(_PYAPI_FUNCS):
        return True
    if name in _UNITTEST_MAINS:
        return _unittest_main_unread(call, reader)
    if name in _PIP_MAINS:
        return _pip_main_unread(call, reader)
    if name in _IMPORTERS:
        return _imports_unread(call, reader, cell, modules=modules)
    if name in _STRING_RUNNERS or name == _TIMEIT_MAIN:
        code = _runner_code(call, reader)
        if code is None:
            return False
        literal = [s for s in code.sources if _literal_source(s) is not None]
        return code.spread or any(
            _literal_source(s) is None and not _names_read_code(s, reader) for s in code.sources
        ) or bool(literal) and id(call) not in read.tried
    return isinstance(func, ast.Attribute) and last in _RUNNER_METHODS and last in _runner_methods(func.value, reader)


def _app_runner_unread(call: ast.Call, reader: _Reader) -> bool:
    """Whether an app runner (`_APP_RUNNERS`) imports and runs code the check
    can't read: a string target names a module it imports from the working
    folder (a literal, or a name bound to one), so it runs that module's
    code, and so may a target the check can't read, unless it is an app
    object read code makes (`_may_be_text`), or one a `*` or `**` spread
    gives; an app object, which is read code, runs nothing new."""
    target = call.args[0] if call.args else _keyword(call, "app")
    if target is None or isinstance(target, ast.Starred):
        return bool(call.args) or any(k.arg is None for k in call.keywords)
    texts = _literal_texts(target, reader)
    return _may_be_text(target, reader, unknown=True) if texts is None else bool(texts)


def _may_be_text(node: ast.AST, reader: _Reader, *, unknown: bool, depth: int = 0) -> bool:
    """Whether `node`, a value the check can't read as literal text, may
    still be text, such as a name to import: a call that may give text
    (`_TEXT_SOURCES`), as a file read, `input()` or a decode does, an
    f-string, a `+` or `%`, an entry of a mapping or a list but one of an
    object a call made (`_made_object`), as a frame's column is, or a name
    a value binds to one. A name an import, a def or a class binds, and any
    other call, as `FastAPI()` or `create_app()`, give an object. A
    parameter holds what each call of its function passes for it
    (`_Reader.args_for`), and none for a function nothing calls; an
    attribute holds what a cell stores in it, as `self.app = FastAPI()`
    does, and one of an instance of a class the cells define what their
    code stores on `self` (`_made_by_a_class`), else an object where it is
    one of an object a call made, as `demo.app` is. Any other code reads as
    `unknown`, and so does a name no value binds, or past `_PARAM_DEPTH`
    names."""
    if isinstance(node, ast.Call):
        return _last_name(node.func) in _TEXT_SOURCES
    if isinstance(node, ast.Subscript) and _made_object(node.value, reader, depth=depth):
        return False
    if isinstance(node, (ast.JoinedStr, ast.BinOp, ast.Subscript)):
        return True
    if depth > _PARAM_DEPTH:
        return unknown
    # Read once per check and depth (`_Reader.recalled`): each call of a
    # parameter's function hands on what its own callers pass, so a chain of
    # helpers would read each caller's argument once per path to it.
    return reader.recalled(
        f"may be text, else {unknown}", node, depth=depth,
        compute=lambda: _may_be_text_afresh(node, reader, unknown=unknown, depth=depth),
    )


def _may_be_text_afresh(node: ast.AST, reader: _Reader, *, unknown: bool, depth: int) -> bool:
    """`_may_be_text` of an attribute, a name or other code, read afresh."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        stored = [v for cell in reader.cells for v in cell.member_values.get((node.value.id, node.attr), ())]
        if not stored and _made_by_a_class(node.value, reader):
            stored = [v for cell in reader.cells for v in cell.member_values.get(("self", node.attr), ())]
        if stored:
            return any(_may_be_text(v, reader, unknown=unknown, depth=depth + 1) for v in stored)
        return False if _made_object(node.value, reader, depth=depth) else unknown
    if not isinstance(node, ast.Name):
        return unknown
    if reader.import_of(node) is not None or reader.function_named(node) is not None or node.id in reader.classes:
        return False
    found = reader.bindings_of(node.id, node=node)
    if not found:
        return unknown
    for b in found:
        if b.how == "param":
            passed = reader.args_for(b.value, node.id)
            if not passed and not reader.uncalled(b.value) or any(a is None for a in passed):
                return unknown
            if any(_may_be_text(a.node, reader, unknown=unknown, depth=depth + 1) for a in passed if a is not None):
                return True
        elif b.how != "value":
            return unknown
        elif _may_be_text(b.value, reader, unknown=unknown, depth=depth + 1):
            return True
    return False


# The last names of the calls that give data which may hold texts under its
# keys or attributes, as a config `json.load` reads, a copy of the
# environment or the settings `dotenv_values` reads do (`_made_object`).
_DATA_LOADS = frozenset({"load", "safe_load", "full_load", "copy", "deepcopy", "to_dict", "dotenv_values"})


def _made_object(node: ast.AST, reader: _Reader, *, depth: int) -> bool:
    """Whether `node`, what a subscript or an attribute is read on, is a name
    that holds only what a member of an installed package or of Python's
    own modules made (`_Reader.module_member`) that gives an object, not
    text (`_may_be_text`) nor data that may hold texts (`_DATA_LOADS`), as
    `df = pd.read_csv(...)` or `demo = gr.Interface(...)` do. What another
    call gives may hold texts, as a response's `.json()`, `parse_args()`
    or `dict(os.environ)` does."""
    if not isinstance(node, ast.Name) or depth > _PARAM_DEPTH:
        return False
    found = reader.bindings_of(node.id, node=node)
    return bool(found) and all(
        b.how == "value" and isinstance(b.value, ast.Call) and reader.module_member(b.value.func) is not None
        and _last_name(b.value.func) not in _DATA_LOADS
        and not _may_be_text(b.value, reader, unknown=True, depth=depth + 1) for b in found
    )


def _made_by_a_class(node: ast.Name, reader: _Reader) -> bool:
    """Whether the name `node` holds only instances of a class the cells
    define (`_Reader.classes`), made by calling it by name."""
    found = reader.bindings_of(node.id, node=node)
    return bool(found) and all(
        b.how == "value" and isinstance(b.value, ast.Call) and isinstance(b.value.func, ast.Name)
        and b.value.func.id in reader.classes for b in found
    )


def _adds_unread_site(call: ast.Call, reader: _Reader, cell: _Cell, *, modules: _Modules) -> bool:
    """Whether `site.addsitedir` runs code the check can't read: the import
    lines of the `.pth` files in the folder it adds, which it runs as it
    adds it. Held where the check can't place the folder, where it lies
    outside the project or the pad's working folder may move, or where the
    folder may hold such a file (`_Modules.may_hold_pth`). An import from it
    afterwards is read as one after any change of `sys.path`."""
    target = _target_arg(call, keywords=("sitedir",))
    project = modules.project
    got = reader.paths(target, cell=cell) if target is not None else _UNKNOWN
    if isinstance(got, _Unknown) or project is None or reader.cwds != ["."]:
        return True
    for path in got:
        rel = None if _TEMP_PATH in path or _unreadable_text(path) else _relative(path, project)
        if rel is None or rel == ".." or rel.startswith("../") or modules.may_hold_pth(rel):
            return True
    return False


def _unittest_main_unread(call: ast.Call, reader: _Reader) -> bool:
    """Whether `unittest.main` or `unittest.TestProgram` loads code the check
    can't read: a `module=` that names anything but `__main__`, or that the
    check can't read, loads that module, and a `None` there, which may also
    come through a name, discovers the tests under the working folder and
    imports them; an `argv` the check can't read, or one whose words load
    tests (`_loads_tests`), and a `defaultTest` given, which names the
    tests to load, unless each of its names is the cells' own
    (`_names_own_test`). Each is read by keyword or in the order
    `TestProgram` takes them. With no `argv`, it reads `sys.argv`, which a
    cell may set (`_sets_test_argv`). With none of these it runs the cell's
    own TestCase from `__main__`, which read code holds already."""
    positional = list(itertools.takewhile(lambda a: not isinstance(a, ast.Starred), call.args))
    if len(positional) != len(call.args) or any(k.arg is None for k in call.keywords):
        return True
    module = _given_at(call, positional, "module", 0)
    if module is not None:
        texts = _literal_texts(module, reader)
        if not texts or any(text != "__main__" for text in texts):
            return True
    default = _given_at(call, positional, "defaultTest", 1)
    if default is not None and not (isinstance(default, ast.Constant) and default.value is None):
        named = ([default] if isinstance(default, ast.Constant) else reader.items_of(node=default, key=None, depth=0))
        names = [_literal_source(item) for item in named or ()]
        if not names or any(name is None or not _names_own_test(name, reader) for name in names):
            return True
    argv = _given_at(call, positional, "argv", 2)
    if argv is None or isinstance(argv, ast.Constant) and argv.value is None:
        return _sets_test_argv(reader)
    items = reader.items_of(node=argv, key=None, depth=0)
    return items is None or _loads_tests(items, reader)


# The options unittest's command line takes, and those that take a value.
_UNITTEST_FLAGS = frozenset({
    "-h", "--help", "-v", "--verbose", "-q", "--quiet", "--locals", "-f", "--failfast", "-c", "--catch", "-b",
    "--buffer",
})
_UNITTEST_VALUE_OPTIONS = frozenset({"-k", "--durations"})


def _loads_tests(items: Sequence[ast.AST], reader: _Reader) -> bool:
    """Whether a unittest argv, by its `items`, loads tests from disk: a
    word past the program word that the check can't read or that is
    `discover`; else a test name that isn't the cells' own
    (`_names_own_test`), unless an option unittest doesn't take
    (`_UNITTEST_FLAGS`, `_UNITTEST_VALUE_OPTIONS`, a long one by a prefix,
    short flags run together) stops its run before it loads any."""
    words = [_literal_source(item) for item in items[1:]]
    if any(word is None or word.lower() == "discover" for word in words):
        return True
    loads = False
    rest = iter(word for word in words if word is not None)
    for word in rest:
        option = word.partition("=")[0]
        if option in _UNITTEST_VALUE_OPTIONS:
            if "=" not in word:
                next(rest, None)
        elif word.startswith("-k"):
            continue
        elif word.startswith("--"):
            if not any(flag.startswith(option) for flag in (*_UNITTEST_FLAGS, *_UNITTEST_VALUE_OPTIONS)
                       if flag.startswith("--")):
                return False
        elif word.startswith("-") and len(word) > 1:
            if not all(f"-{letter}" in _UNITTEST_FLAGS for letter in word[1:]):
                return False
        else:
            loads = loads or not _names_own_test(word, reader)
    return loads


def _names_own_test(name: str, reader: _Reader) -> bool:
    """Whether the test name `name` names a class the cells define
    (`_Reader.classes`), or a member of one, as `TestTotals.test_ok` does,
    which unittest looks up in `__main__` and loads nothing for."""
    return name.partition(".")[0] in reader.classes


def _sets_test_argv(reader: _Reader) -> bool:
    """Whether a cell sets `sys.argv`, which `unittest.main` reads when it
    is given no argv, to words that load tests from disk: a list the check
    can't read, or one that does (`_loads_tests`); any change in place, as
    an append or an entry store, counts."""
    for cell in reader.cells:
        for node in cell.stores:
            for target in node.targets if isinstance(node, ast.Assign) else ():
                if isinstance(target, ast.Subscript) and reader.dotted(target.value) == "sys.argv":
                    return True
                if reader.dotted(target) != "sys.argv":
                    continue
                items = reader.items_of(node=node.value, key=None, depth=0)
                if items is None or _loads_tests(items, reader):
                    return True
        for merge in cell.merges:
            if reader.dotted(merge.target) == "sys.argv":
                return True
        if any(isinstance(c.func, ast.Attribute) and c.func.attr in _LIST_CHANGES
               and reader.dotted(c.func.value) == "sys.argv" for c in cell.calls):
            return True
    return False


def _pip_main_unread(call: ast.Call, reader: _Reader) -> bool:
    """Whether pip's in-process main, given an argv list, installs from a
    source the check can't read (`_pip_unread`), as `-m pip` does. An argv
    the check can't read, or a word in it that isn't a literal, is held."""
    argv = call.args[0] if call.args else _keyword(call, "args")
    items = reader.items_of(node=argv, key=None, depth=0) if argv is not None else None
    if items is None:
        return True
    words = [_literal_source(item) for item in items]
    return any(word is None for word in words) or _pip_unread([word for word in words if word is not None])


def _names_read_code(node: ast.AST, reader: _Reader, *, depth: int = 0) -> bool:
    """Whether `node` is a lambda, names a function or lambda a cell
    defines that no other value of the name can stand for, in the code's own
    namespace first (`_Reader.function_named`), is `functools.partial` of
    one, or is a parameter that the call being read, else each call of its
    function, passes such code for: code the check reads where it is written
    or passed on. A function nothing calls passes nothing. Past
    `_PARAM_DEPTH` parameters, no."""
    if isinstance(node, ast.Lambda):
        return True
    if isinstance(node, ast.Call) and node.args and reader.dotted(node.func) == "functools.partial":
        return _names_read_code(node.args[0], reader, depth=depth + 1)
    if not isinstance(node, ast.Name) or depth > _PARAM_DEPTH:
        return False
    # Read once per check and depth, as `_may_be_text` is.
    return reader.recalled(
        "names read code", node, depth=depth, compute=lambda: _names_read_code_afresh(node, reader, depth=depth),
    )


def _names_read_code_afresh(node: ast.Name, reader: _Reader, *, depth: int) -> bool:
    """`_names_read_code` of a name, read afresh."""
    found = reader.bindings_of(node.id, node=node)
    if found and all(b.how == "param" for b in found):
        for b in found:
            passed = reader.args_for(b.value, node.id)
            if not passed and not reader.uncalled(b.value):
                return False
            if any(arg is None or not _names_read_code(arg.node, reader, depth=depth + 1) for arg in passed):
                return False
        return True
    if node.id not in reader.function_names and reader.function_named(node) is None:
        return False
    return all(b.how == "value" and isinstance(b.value, ast.Lambda) for b in found)


def _launches_unread(call: ast.Call, reader: _Reader, *, read: _ExecRead, modules: _Modules) -> bool:
    """Whether `call` launches Python that runs code the check can't read
    (`_python_launch`), installs a project that builds (`_Launch.syncs`),
    hands `python -c` literal code `read` left unread at `_EXEC_LIMIT`, or
    starts a Python interpreter once a cell wrote a startup hook it runs
    (`_Modules.hooks_written`)."""
    launch = _python_launch(call, reader)
    return launch is not None and (
        launch.unread or launch.syncs and reader.project_builds or bool(launch.sources) and id(call) not in read.tried
        or launch.starts_python and modules.hooks_written
    )


class _TargetCaller(NamedTuple):
    """Where a caller that runs a function later takes it and what it hands
    it, by keyword and by position: a thread's `target`, a timer's
    `function`, and the `args` and `kwargs` each passes it."""

    target: str
    at: int
    args_at: int
    kwargs_at: int


# The callers that run the function given as their `target` or `function`
# when they start, with the `args` and `kwargs` they are given, by their last
# name (`_value_call_args`).
_TARGET_CALLERS = {
    "Thread": _TargetCaller(target="target", at=1, args_at=3, kwargs_at=4),
    "Process": _TargetCaller(target="target", at=1, args_at=3, kwargs_at=4),
    "Timer": _TargetCaller(target="function", at=1, args_at=2, kwargs_at=3),
}
# The callers that run the function given first with the arguments after it,
# keywords too (`submit`, `atexit.register`, `Context.run`), and those whose
# keywords are their own, not the function's (an event loop's `call_soon`).
_CALLERS_FORWARDING_ALL = frozenset({"submit", "to_thread", "register", "run"})
_CALLERS_FORWARDING_ARGS = frozenset({"call_soon", "call_soon_threadsafe"})
# The callers that run the function given second with the arguments after
# it: an executor's loop method, and an event loop's timed calls.
_CALLERS_AFTER_A_FIRST = frozenset({"run_in_executor", "call_later", "call_at"})
# A pool's calls that run the function given first with an `args` tuple and a
# `kwds` dict, given by keyword or in that order.
_POOL_APPLIES = frozenset({"apply", "apply_async"})
# The callers that run the function on each element of what they are given,
# which the check doesn't read as one call.
_MAPPING_CALLERS = frozenset({"map", "starmap", "imap", "imap_unordered", "map_async", "starmap_async"})
# The calls that only look at a function value, by their full name and by
# their last name.
_VALUE_LOOKS = frozenset({
    "functools.wraps", "functools.update_wrapper", "inspect.signature", "inspect.getsource", "inspect.getdoc",
    "inspect.getfile", "inspect.getmodule", "typing.cast",
})
_VALUE_LOOK_NAMES = frozenset({
    "isinstance", "issubclass", "callable", "id", "type", "repr", "str", "hasattr", "getattr", "dir", "vars",
    "help", *_HARMLESS_CALLS,
})
# The calls that wrap a function so a call of what they give calls it, as
# `delayed(f)(x)` calls `f(x)` once its batch runs (`_value_call_args`).
_DELAYING_CALLS = frozenset({"joblib.delayed", "dask.delayed"})


def _passed_calls(node: ast.AST, cell: _Cell, reader: _Reader) -> list[ast.Call] | _Unknown:
    """The calls a caller that runs the function value `node`, code of
    `cell`, later makes of it (`_value_call_args`), each as its own call
    with the arguments and keywords it is handed, so
    `ex.submit(subprocess.run, ["rm", "x"])` reads as `subprocess.run(["rm",
    "x"])`. A `functools.partial` of it, written out or held in a name
    (`_Reader.partials_of`), is read where the partial is handed on, with
    the partial's own arguments first (`_partial_call`), each partial a
    name may hold in turn. Unknown where the check can't read what the
    caller hands it; none where `node` sits in no call that may run it."""
    bound = reader.partials_of(node=node) if isinstance(node, ast.Name) else []
    partials: list[ast.Call] = []
    holder = _value_holder(node, cell)
    while (isinstance(holder, ast.Call) and holder.args and holder.args[0] is node
           and reader.dotted(holder.func) == "functools.partial"):
        partials.append(holder)
        node, holder = holder, _value_holder(holder, cell)
    passed = _value_call_args(node, holder, reader)
    if isinstance(passed, _Unknown):
        return passed
    out: list[ast.Call] = []
    for given in passed:
        call = ast.copy_location(ast.Call(func=node, args=given.args, keywords=given.keywords), node)
        for partial in reversed(partials):
            call = _partial_call(partial=partial, call=call)
        out += [_partial_call(partial=partial, call=call) for partial in bound] or [call]
    return out


def _value_holder(node: ast.AST, cell: _Cell) -> ast.AST | None:
    """What holds the value `node`, code of `cell`: the call it is passed to,
    by position, by keyword or spread, or whatever else it sits in."""
    holder = cell.parents.get(node)
    if isinstance(holder, (ast.keyword, ast.Starred)):
        holder = cell.parents.get(holder)
    return holder


def _partial_launcher(node: ast.AST, reader: _Reader) -> bool:
    """Whether `node` is a name a cell binds to a `functools.partial` of a
    launcher (`_LAUNCHERS`), as `job = partial(subprocess.run, cmd)` binds
    `job`; such a name is one of `_Reader.runner_aliases`."""
    if not (isinstance(node, ast.Name) and node.id in reader.runner_aliases):
        return False
    return any(reader.module_member(partial.args[0]) in _LAUNCHERS for partial in reader.partials_of(node=node))


def _passed_launch(node: ast.AST, cell: _Cell, reader: _Reader, *, read: _ExecRead, modules: _Modules) -> bool:
    """Whether a launcher `node`, handed to a caller that runs it later,
    launches Python the check can't read: each call the caller makes of it
    (`_passed_calls`), read as its own, so
    `ex.submit(subprocess.run, [sys.executable, "etl.py"])` holds while
    `ex.submit(subprocess.run, ["git", "status"])` does not. A caller the
    check can't read the arguments of, as `map` over each element, or one
    it doesn't know, holds."""
    calls = _passed_calls(node, cell, reader)
    return isinstance(calls, _Unknown) or any(
        _launches_unread(call, reader, read=read, modules=modules) for call in calls
    )


class _PassedArgs(NamedTuple):
    """The arguments and keywords a caller hands a function value it runs
    later (`_value_call_args`)."""

    args: list[ast.expr]
    keywords: list[ast.keyword]


def _value_call_args(node: ast.AST, wrapper: ast.AST | None, reader: _Reader) -> list[_PassedArgs] | _Unknown:
    """The arguments a caller `wrapper` hands the function value `node`
    each time it runs it: a thread's, a process's or a timer's `args` and
    `kwargs` (`_TARGET_CALLERS`), given by keyword or by position; those
    after it for `submit`, `asyncio.to_thread`, `atexit.register`,
    `Context.run` and an event loop's calls, with the keywords of those
    that forward them; a pool's `args` and `kwds`; those a call of what a
    delaying call gives is handed (`_DELAYING_CALLS`); those each call of
    its parameter in a function the pad defines makes (`_helper_calls`);
    and each item of the one sequence a caller that runs it on each
    element walks (`_MAPPING_CALLERS`, `_mapped_args`). Unknown for
    arguments the check can't read, `node` handed on in another place than
    the function's, or a caller the check doesn't know; none where
    `wrapper` is no call, or one that only looks at the value
    (`_VALUE_LOOKS`), a logger's among them (`_is_harmless_call`)."""
    if not isinstance(wrapper, ast.Call) or wrapper.func is node:
        return []
    dotted, last = reader.dotted(wrapper.func), _last_name(wrapper.func)
    if dotted in _VALUE_LOOKS or last in _VALUE_LOOK_NAMES or _is_harmless_call(wrapper, reader):
        return []
    if dotted in _DELAYING_CALLS or reader.module_member(wrapper.func) in _DELAYING_CALLS:
        later = reader.cell_of(wrapper).parents.get(wrapper)
        if not (isinstance(later, ast.Call) and later.func is wrapper and wrapper.args[:1] == [node]):
            return _UNKNOWN
        return [_PassedArgs(args=list(later.args), keywords=list(later.keywords))]
    helper = reader.function_named(wrapper.func) if isinstance(wrapper.func, ast.Name) else None
    if helper is not None:
        return _helper_calls(wrapper, node, helper, reader)
    positional = list(itertools.takewhile(lambda a: not isinstance(a, ast.Starred), wrapper.args))
    shape = _TARGET_CALLERS.get(last)
    if shape is not None:
        if _given_at(wrapper, positional, shape.target, shape.at) is not node:
            return _UNKNOWN
        return _passed_lists(_given_at(wrapper, positional, "args", shape.args_at),
                             _given_at(wrapper, positional, "kwargs", shape.kwargs_at))
    if last in _POOL_APPLIES:
        if positional[:1] != [node]:
            return _UNKNOWN
        return _passed_lists(_given_at(wrapper, positional, "args", 1), _given_at(wrapper, positional, "kwds", 2))
    if last in _MAPPING_CALLERS:
        if positional[:1] != [node] or len(positional) != len(wrapper.args) or len(positional) != 2:
            return _UNKNOWN
        return _mapped_args(positional[1], reader, spread=last.startswith("starmap"))
    first = 1 if last in _CALLERS_AFTER_A_FIRST else 0
    if last in _CALLERS_FORWARDING_ALL | _CALLERS_FORWARDING_ARGS | _CALLERS_AFTER_A_FIRST:
        if len(positional) <= first or positional[first] is not node or len(positional) != len(wrapper.args):
            return _UNKNOWN
        keywords = list(wrapper.keywords) if last in _CALLERS_FORWARDING_ALL else []
        if any(k.arg is None for k in keywords):
            return _UNKNOWN
        return [_PassedArgs(args=list(wrapper.args[first + 1:]), keywords=keywords)]
    return _UNKNOWN


def _mapped_args(walked: ast.AST, reader: _Reader, *, spread: bool) -> list[_PassedArgs] | _Unknown:
    """The arguments a caller that runs a function on each item of `walked`
    (`_MAPPING_CALLERS`) hands it: each item (`_Reader.items_of`), or the
    element a comprehension makes, as the one argument, or, with `spread`,
    as `starmap` does, the elements of each item, a list or tuple written
    out. Unknown where the check can't read the items."""
    made = (ast.ListComp, ast.GeneratorExp)
    found = reader.bindings_of(walked.id, node=walked) if isinstance(walked, ast.Name) else []
    values = [b.value for b in found if b.how == "value"]
    items: list[ast.AST] | None
    if isinstance(walked, made):
        items = [walked.elt]
    elif found and len(values) == len(found) and all(isinstance(v, made) for v in values):
        items = [v.elt for v in values if isinstance(v, made)]
    else:
        items = reader.items_of(node=walked, key=None, depth=0)
    if items is None:
        return _UNKNOWN
    out: list[_PassedArgs] = []
    for item in items:
        if not spread:
            out.append(_PassedArgs(args=[item], keywords=[]))
        elif isinstance(item, (ast.List, ast.Tuple)) and not any(isinstance(e, ast.Starred) for e in item.elts):
            out.append(_PassedArgs(args=list(item.elts), keywords=[]))
        else:
            return _UNKNOWN
    return out


def _helper_calls(wrapper: ast.Call, node: ast.AST, func: Function, reader: _Reader) -> list[_PassedArgs] | _Unknown:
    """The arguments `func`, a function the pad defines that `wrapper`
    calls, hands the function value `node` it is passed: those of each call
    of that parameter in its body, a `*args` spread there read as the
    items `wrapper` passes for it (`_call_arguments`). Unknown where `node`
    goes to no parameter the check places, or the body hands that
    parameter on, stores or returns it."""
    home = reader.cell_of(func)
    passed = _call_arguments(func, wrapper, call_cell=reader.cell_of(wrapper), func_cell=home)
    spread = func.args.vararg.arg if func.args.vararg is not None else None
    params = [name for name, arg in passed.items() if arg is not None and arg.node is node and name != spread]
    if not params:
        return _UNKNOWN
    items = passed.get(spread) if spread is not None else None
    out: list[_PassedArgs] = []
    for use in (use for name in params for use in _loads_in(func.body, name)):
        call = home.parents.get(use)
        if not (isinstance(call, ast.Call) and call.func is use):
            return _UNKNOWN
        args: list[ast.expr] = []
        for arg in call.args:
            if (isinstance(arg, ast.Starred) and isinstance(arg.value, ast.Name) and arg.value.id == spread
                    and items is not None and isinstance(items.node, ast.Tuple)):
                args += items.node.elts
            else:
                args.append(arg)
        out.append(_PassedArgs(args=args, keywords=list(call.keywords)))
    return out


def _given_at(call: ast.Call, positional: list[ast.expr], name: str, at: int) -> ast.AST | None:
    """The argument `call` gives the parameter `name` at place `at`: by
    keyword, else in `positional`, its arguments before any spread."""
    given = _keyword(call, name)
    if given is None and at < len(positional):
        given = positional[at]
    return given


def _passed_lists(args: ast.AST | None, kwargs: ast.AST | None) -> list[_PassedArgs] | _Unknown:
    """The arguments a literal `args` tuple or list and a literal `kwargs`
    dict of string keys hand a function, either absent or None; unknown
    where the check can't read one."""
    out = _PassedArgs(args=[], keywords=[])
    if args is not None and not (isinstance(args, ast.Constant) and args.value is None):
        if not isinstance(args, (ast.List, ast.Tuple)) or any(isinstance(e, ast.Starred) for e in args.elts):
            return _UNKNOWN
        out.args.extend(args.elts)
    if kwargs is not None and not (isinstance(kwargs, ast.Constant) and kwargs.value is None):
        if not isinstance(kwargs, ast.Dict) or not all(
            isinstance(k, ast.Constant) and isinstance(k.value, str) for k in kwargs.keys
        ):
            return _UNKNOWN
        out.keywords.extend(
            ast.keyword(arg=k.value, value=v) for k, v in zip(kwargs.keys, kwargs.values) if isinstance(k, ast.Constant)
        )
    return [out]


def _unread_run_at(node: ast.AST, reader: _Reader, cell: _Cell, *, read: _ExecRead, modules: _Modules) -> bool:
    """Whether `node`, code of `cell`, runs code in this process, or in a
    Python process it launches, that the check can't read (`_unread_code`):
    a call that runs a script, a module, a notebook, a code object, or text
    a console, debugger or test runner gets (`_CODE_RUNNERS`,
    `_PYAPI_FUNCS`, `_INSTANCE_RUNNERS`, and any `.exec_module()` or
    `.load_module()`); a runner of text (`_STRING_RUNNERS`) handed code that
    isn't a literal, or a literal `read` left unread at `_EXEC_LIMIT`; an
    import by a name that isn't a literal; an import of a module `modules`
    can't read, whether by an import statement or a literal
    `importlib.import_module` or `__import__` (`_Cell.looked_up`); a runner,
    an importer, or a loader's or a runner's own method, passed on or
    stored (`_holds_a_running_member`), and a lookup of a member that may
    run code (`_looks_up_a_runner`); a function given code the check can't
    read (`_swaps_in_unread_code`); a program it runs that runs Python the
    check can't read (`_launches_unread`); and a write of a startup hook a
    later Python process runs as it starts (`_hook_write`,
    `_HookWrite.runs_at_start`). Each name a runner, importer
    or launcher is called by resolves through the imports
    (`_Reader.module_member`)."""
    if isinstance(node, ast.Import):
        return any(modules.unread(a.name, cell=cell) for a in node.names)
    if isinstance(node, ast.ImportFrom):
        # A relative import fails in the pad's cells: they are no package's.
        return not node.level and bool(node.module) and modules.unread(node.module, cell=cell)
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        if node in cell.looked_up and modules.unread(node.id, cell=cell):
            return True
        if _swaps_in_unread_code(node, cell):
            return True
        # A runner or importer passed on or stored, as `map(runpy.run_path, paths)`,
        # and a loader's or a runner's method, as `go = pr.run`.
        call = cell.parents.get(node)
        if not (isinstance(node.ctx, ast.Load) and not (isinstance(call, ast.Call) and call.func is node)):
            return False
        if (isinstance(node, ast.Attribute) or isinstance(node, ast.Name) and node.id in reader.runner_aliases) and (
            _holds_a_running_member(node, reader)
        ):
            return True
        if reader.module_member(node) in _LAUNCHERS or _partial_launcher(node, reader):
            # A launcher, or a partial of one, handed to a caller that runs
            # it later, as `Thread(target=subprocess.run, args=([...],))`.
            return _passed_launch(node, cell, reader, read=read, modules=modules)
        if not _runner_named(node, reader):
            return False
        member = reader.module_member(node)
        return member in _RUNNER_VALUES and not (member in _RUNNER_TYPES and _type_runner_aside(node, cell))
    if not isinstance(node, ast.Call):
        return False
    parent = cell.parents.get(node)
    called = isinstance(parent, ast.Call) and parent.func is node
    if _looks_up_a_runner(node, reader, called=called) or _hook_write(node, reader, cell).runs_at_start:
        return True
    if _sets_code(node) and _unread_code_object(node.args[2], cell):
        # A function given code the check can't read, as `ok = setattr(f, "__code__", co)`.
        return True
    return _runs_unread(node, reader, cell, read=read, modules=modules) or _launches_unread(
        node, reader, read=read, modules=modules,
    )


def _type_runner_aside(node: ast.AST, cell: _Cell) -> bool:
    """Whether `node`, code of `cell` that loads a runner that is a type
    (`_RUNNER_TYPES`), sits where nothing calls it: the class, or one of a
    tuple of them, `isinstance` or `issubclass` checks against, a side of a
    comparison, a dict key or an index, or an annotation."""
    parent = cell.parents.get(node)
    while isinstance(parent, ast.Tuple):
        node, parent = parent, cell.parents.get(parent)
    if isinstance(parent, ast.Call):
        return isinstance(parent.func, ast.Name) and parent.func.id in ("isinstance", "issubclass") and (
            node in parent.args[1:2]
        )
    if isinstance(parent, ast.Dict):
        return node in parent.keys
    if isinstance(parent, ast.Subscript):
        return parent.slice is node
    if isinstance(parent, (ast.AnnAssign, ast.arg)):
        return parent.annotation is node
    if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return parent.returns is node
    return isinstance(parent, ast.Compare)


class _HookWrite(NamedTuple):
    """Where a call may write a startup hook (`_hook_write`): the path of
    each one the check places, read against each folder the write may run
    in (`paths`); and whether it may write one where the check can't place
    it, or into a folder of the installed packages a lookup names
    (`unplaced`)."""

    paths: frozenset[str] = frozenset()
    unplaced: bool = False

    @property
    def any(self) -> bool:
        """Whether the call may write a startup hook at all."""
        return self.unplaced or bool(self.paths)

    @property
    def runs_at_start(self) -> bool:
        """Whether a Python process started later may run a hook the call
        writes as it starts: one the check can't place, or one at a path
        `site` runs it from (`_runs_at_start`)."""
        return self.unplaced or any(_runs_at_start(path) for path in self.paths)

    def joined(self, other: _HookWrite) -> _HookWrite:
        """The hooks this write and `other` may write, together."""
        return _HookWrite(paths=self.paths | other.paths, unplaced=self.unplaced or other.unplaced)


_NO_HOOK = _HookWrite()
_UNPLACED_HOOK = _HookWrite(unplaced=True)


def _hook_write(call: ast.Call, reader: _Reader, cell: _Cell) -> _HookWrite:
    """Where `call`, in a cell that names a startup hook
    (`_CodeSigns.names_a_startup_hook`), may write one (`_is_startup_hook`),
    which `site` may run as a Python starts (`_HookWrite`): a write of a
    file to such a path or one the check can't place, an open whose mode
    the check can't read among them, or a program it runs that may write
    one (`_command_hook_write`), as `cp hook.txt site-packages/hook.pth` or
    an `echo` into one does. A copy or move into a folder writes each file
    it copies there under the file's own name (`_copied_names`), and a
    call that writes the file an argument names (`_made_file`), as
    `urlretrieve` does, writes that file. Each path is read against the
    folders the write runs in (`_placed_hooks`): the pad's
    (`_Reader.cwds`), and those a program's `cwd=` names (`_run_folders`).
    A data writer's file (`_WRITER_METHODS`), as `torch.save` and `to_csv`
    write, holds a frame, an array, a chart or a model, not a module's code
    or the import lines a `.pth` file runs, so it counts only inside the
    installed packages (`_writes_site_packages`, where an `origin` counts
    only on a spec, `_may_be_a_spec`); and a literal text with
    no such line (`_runs_no_pth_line`) writes no hook into a `.pth` file,
    one named so in a folder the check can't place among them
    (`_written_names`)."""
    if not cell.signs.names_a_startup_hook:
        return _NO_HOOK
    last = _last_name(call.func)
    if last in _FILE_WRITE_NAMES or last in _FILE_MAKER_NAMES:
        target = _written_target(call, reader, cell) if last in _FILE_WRITE_NAMES else None
        if target is None:
            target = _made_file(call, reader)
        if target is None:
            # A read, as `open("x.pth")` is, writes nothing; an open by a
            # mode the check can't read may write the file it opens.
            target = _opened_by_unread_mode(call, reader)
        if target is None:
            return _NO_HOOK
        site = _writes_site_packages(target, reader)
        if last in _WRITER_METHODS and not site:
            return _NO_HOOK
        got = reader.paths(target, cell=cell)
        if isinstance(got, _Unknown):
            # A `.pth` file by its literal name, in a folder the check can't
            # place, holds a hook only where it may hold an import line.
            written = _written_names(target, reader, cell)
            pth = written is not None and bool(written) and all(
                name.endswith(".pth") and _RUN_TIME_PART not in name for name in written
            )
            return _NO_HOOK if pth and _runs_no_pth_line(call, cell) else _UNPLACED_HOOK
        hooks = [path for path in got if _is_startup_hook(path)]
        # A copy or move into a folder lands each source there under its own
        # name, as `cp` does (`_copy_landings`); in the installed packages,
        # one the check can't name may land as a hook.
        names = _copied_names(call, reader, cell)
        if site and (isinstance(names, _Unknown) or any(_may_name_a_hook(name) for name in names)):
            return _UNPLACED_HOOK
        if not hooks and not isinstance(names, _Unknown):
            hooks = [path for folder in got for name in names
                     for path in [f"{folder.rstrip('/')}/{name}"] if _is_startup_hook(path)]
        if not hooks or all(path.endswith(".pth") for path in hooks) and _runs_no_pth_line(call, cell):
            return _NO_HOOK
        return _UNPLACED_HOOK if site else _placed_hooks(hooks, reader.cwds)
    name = (reader.module_member(call.func) or reader.dotted(call.func)) if last in _LAUNCH_NAMES else None
    if name not in _LAUNCHERS:
        return _NO_HOOK
    cwds = _run_folders(call, reader, cell)
    out = _NO_HOOK
    for command in _shell_commands(call, name, reader):
        out = out.joined(
            _command_hook_write(command, cwds=cwds, depth=0) if isinstance(command, list)
            else _script_hook_write(command, cwds=cwds, depth=0)
        )
    return out


def _opened_by_unread_mode(call: ast.Call, reader: _Reader) -> ast.AST | None:
    """The file `call` opens with a mode that isn't a literal, so it may
    open it for writing: an `open`'s first argument, or the path an `open`
    method is called on; None for any other call."""
    if reader.dotted(call.func) in ("open", "io.open"):
        mode = call.args[1] if len(call.args) > 1 else _keyword(call, "mode")
        opened = call.args[0] if call.args else _keyword(call, "file")
    elif isinstance(call.func, ast.Attribute) and call.func.attr == "open":
        mode = _target_arg(call, keywords=("mode",))
        opened = call.func.value
    else:
        return None
    literal = mode is None or isinstance(mode, ast.Constant) and isinstance(mode.value, str)
    return None if literal else opened


def _placed_hooks(paths: Iterable[str], cwds: Sequence[str] | _Unknown) -> _HookWrite:
    """Where the startup hooks a write names by `paths` land, each read
    against the folders `cwds` it may run in, as the pad's working folder or
    a program's `cwd=`: unplaced where a folder holds a part picked at run
    time (`_unreadable_text`), or where a path is relative and the check
    can't place the folder it runs in."""
    out: set[str] = set()
    for path in paths:
        path = path.replace("\\", "/")
        if _unreadable_text(posixpath.dirname(path)):
            return _UNPLACED_HOOK
        if posixpath.isabs(path) or path.startswith("~"):
            out.add(posixpath.normpath(path))
        elif isinstance(cwds, _Unknown):
            return _UNPLACED_HOOK
        else:
            out.update(posixpath.normpath(_join(cwd, path)) for cwd in cwds)
    return _HookWrite(paths=frozenset(out))


def _runs_at_start(path: str) -> bool:
    """Whether a Python process runs the startup hook at `path`, a placed
    one (`_placed_hooks`), as it starts: a `sitecustomize` or
    `usercustomize` module in any folder, since the process may put that
    folder on `PYTHONPATH`; a `.pth` file only in a folder `site` reads
    them from as Python starts, the installed packages or the user site,
    whose paths have a `site-packages` or `dist-packages` part
    (`_names_package_folder`), and on Windows also the environment's own
    folder and its base Python's (`sys.prefix`, `sys.base_prefix`), which a
    placed path outside the project may be (`_outside_project`). A `.pth`
    file anywhere else runs only once `site.addsitedir` adds its folder
    (`_Modules.may_hold_pth`)."""
    return _names_hook_module(path) or _names_package_folder(path) or (
        sys.platform == "win32" and path.endswith(".pth") and _outside_project(path)
    )


def _outside_project(path: str) -> bool:
    """Whether `path`, a placed path with `/` separators (`_placed_hooks`),
    may lie outside the project: an absolute, home or drive-letter path, one
    that climbs out with `..`, or one in the pad's own environments under
    `.anton/`."""
    return (posixpath.isabs(path) or path.startswith("~") or re.match(r"[A-Za-z]:/", path) is not None
            or path == ".." or path.startswith("../") or path.startswith(".anton/"))


# Programs that write the files their operands name, so an operand may name
# a startup hook (`_command_hook_write`); and those among them that copy,
# move or link a file to their last operand, or into it as a folder
# (`_copy_landings`).
_COPYING_PROGRAMS = frozenset({*_TREE_COPY_PROGRAMS, "ln"})
_FILE_WRITING_PROGRAMS = frozenset({*_COPYING_PROGRAMS, "tee", "touch", "curl", "wget", "dd"})
# Programs that download what a URL names, by default into the working
# folder under the URL's last part, with the flags that name another folder
# for it (`_download_landings`).
_DOWNLOAD_FOLDER_FLAGS: dict[str, frozenset[str]] = {
    "wget": frozenset({"-P", "--directory-prefix"}),
    "curl": frozenset({"--output-dir"}),
}


def _script_hook_write(script: str, *, cwds: Sequence[str] | _Unknown, depth: int) -> _HookWrite:
    """Where a shell script may write a startup hook, command by command
    (`_script_walk`, `_command_hook_write`), each in the folders the `cd`s
    before it reach (`_shell_cd`)."""
    out = _NO_HOOK
    for command in _script_walk(script):
        out = out.joined(_command_hook_write(command.words, cwds=cwds, depth=depth))
        if command.cd is not None:
            cwds = _shell_cd(cwds, command.cd)
    return out


def _command_hook_write(words: list[str], *, cwds: Sequence[str] | _Unknown, depth: int) -> _HookWrite:
    """Where a command, by its words, run in the folders `cwds`, may write a
    startup hook (`_is_startup_hook`): a file a redirection writes, an
    operand of a program that writes files (`_FILE_WRITING_PROGRAMS`), a
    place a download lands a file (`_download_landings`), or a place a
    copy, move or link lands one (`_copy_landings`), that names one
    (`_placed_hooks`); any, where one of them is a word the check
    can't read, or the program is; what a command `find` runs on each
    match writes (`_find_exec_commands`), in the match's own folder for
    `-execdir` and `-okdir`; and what a shell's script writes in turn
    (`_script_hook_write`), up to `_SHELL_DEPTH` shells."""
    bare, _ = _command_words(words)
    if not bare:
        return _NO_HOOK
    if _unreadable_word(bare[0]):
        return _UNPLACED_HOOK
    verb = _program_name(bare[0])
    if verb in _SCRIPT_RUNNERS:
        script = _inner_script(verb, bare[1:], None)
        if isinstance(script, _Unknown) or depth >= _SHELL_DEPTH:
            return _UNPLACED_HOOK
        return _NO_HOOK if script is None else _script_hook_write(script, cwds=cwds, depth=depth + 1)
    out = _NO_HOOK
    if verb == "find":
        for command in _find_exec_commands(bare[1:]):
            folders = _UNKNOWN if command.flag.endswith("dir") else cwds
            out = out.joined(_command_hook_write(command.words, cwds=folders, depth=depth))
    targets = (_copy_landings(bare[1:]) if verb in _COPYING_PROGRAMS
               else _download_landings(verb, bare[1:]) if verb in _DOWNLOAD_FOLDER_FLAGS
               else bare[1:] if verb in _FILE_WRITING_PROGRAMS else [])
    for at, word in enumerate(words):
        redirect = _REDIRECT.match(word)
        if redirect and ">" in redirect.group():
            targets.append(word[redirect.end():] or (words[at + 1] if at + 1 < len(words) else ""))
    if any(_unreadable_word(target) for target in targets):
        return _UNPLACED_HOOK
    return out.joined(_placed_hooks([target for target in targets if _is_startup_hook(target)], cwds))


def _copy_landings(args: list[str]) -> list[str]:
    """Where a copy, move or link command (`_COPYING_PROGRAMS`), by its
    words past the program, may land a file: its destination, the folder
    its `-t` or `--target-directory` names or else its last operand; and,
    as that may be a folder, each source under it by its own name, unless
    the destination names a startup hook (`_is_startup_hook`), as a file
    does. A source the check can't read stays as it is, since it may land
    under any name."""
    operands: list[str] = []
    into: str | None = None
    rest = iter(args)
    for word in rest:
        if word in ("-t", "--target-directory"):
            into = next(rest, "")
        elif word.startswith("--target-directory="):
            into = word.partition("=")[2]
        elif not word.startswith("-"):
            operands.append(word)
    if into is None:
        if len(operands) < 2:
            return operands
        *operands, into = operands
        if _is_startup_hook(into) and not _unreadable_word(into):
            return [into]
    return [into, *(
        word if _unreadable_word(word) else f"{into.rstrip('/')}/{ntpath.basename(word)}" for word in operands
    )]


def _download_landings(verb: str, args: list[str]) -> list[str]:
    """Where a download command (`_DOWNLOAD_FOLDER_FLAGS`), by its words past
    the program, may write a file: each word, as the file an output flag
    names, and the last part of each URL it gets, in the folder its folder
    flag names, or else the working folder, as `wget -P site-packages` and
    `curl -O --output-dir site-packages` save one."""
    flags = _DOWNLOAD_FOLDER_FLAGS[verb]
    folder: str | None = None
    rest = iter(args)
    for word in rest:
        flag, equals, value = word.partition("=")
        if flag in flags:
            folder = value if equals else next(rest, "")
        elif verb == "wget" and word.startswith("-P") and len(word) > 2:
            folder = word[2:]
    names = [posixpath.basename(re.split(r"[?#]", word, maxsplit=1)[0]) for word in args if "://" in word]
    return [*args, *(name if folder is None else f"{folder.rstrip('/')}/{name}" for name in names if name)]


def _is_startup_hook(path: str) -> bool:
    """Whether `path` names a startup hook by its file's name: a `.pth`
    file, or a `sitecustomize` or `usercustomize` module
    (`_names_hook_module`)."""
    return ntpath.basename(path).endswith(".pth") or _names_hook_module(path)


def _names_hook_module(path: str) -> bool:
    """Whether `path` names a `sitecustomize` or `usercustomize` module, as
    its file (`_STARTUP_HOOK_MODULES`) or as a file inside its package's
    folder (`_STARTUP_HOOK_PACKAGES`), which an import of the module runs."""
    *folders, name = re.split(r"[/\\]", path)
    return name in _STARTUP_HOOK_MODULES or not _STARTUP_HOOK_PACKAGES.isdisjoint(folders)


def _may_name_a_hook(name: str) -> bool:
    """Whether a file's name, `name`, with each part picked at run time
    `_RUN_TIME_PART`, may be a startup hook's (`_is_startup_hook`): one
    that may end in `.pth` past its last such part, or match a hook
    module's name; a listing's wildcard reads as such a part. A name with a
    part the check can't read at all, as a temporary file's (`_TEMP_PATH`),
    may be any."""
    if _NOT_LITERAL in name:
        return True
    name = re.sub(r"[*?]|\[[^]]*\]", _RUN_TIME_PART, name)
    if _RUN_TIME_PART not in name:
        return _is_startup_hook(name)
    tail = name.rsplit(_RUN_TIME_PART, 1)[1]
    pattern = "*".join(re.sub(r"([*?\[])", r"[\1]", part) for part in name.split(_RUN_TIME_PART))
    return tail.endswith(".pth") or ".pth".endswith(tail) or any(
        fnmatch.fnmatchcase(hook, pattern) for hook in _STARTUP_HOOK_MODULES
    )


def _written_names(target: ast.AST, reader: _Reader, cell: _Cell, *, depth: int = 0) -> list[str] | None:
    """The names the file a write to `target`, code of `cell`, may have, a
    part picked at run time `_RUN_TIME_PART` (`_Reader.paths`). Where the
    check can't place the folder, or it is a temporary one, which the path
    stands for whole (`_TEMP_PATH`), they are read from the end of the
    path: the right side of a `/`, or of a `+` after a part the check can't
    read, the last argument of a join, or what an f-string ends with; from
    a temporary file's suffix (`_temp_file_names`); or, for a name, or a
    temporary file's `.name`, from each value it is bound to
    (`_Reader.read_bindings`), up to `_PARAM_DEPTH` names. None where the
    check can't read them."""
    got = reader.paths(target, cell=cell)
    if not isinstance(got, _Unknown) and not any(_TEMP_PATH in p for p in got):
        return [posixpath.basename(p.replace("\\", "/")) for p in got]
    made = _temp_file_names(target, reader)
    if made is not None:
        return made
    holder = target.value if isinstance(target, ast.Attribute) and target.attr == "name" else target
    if isinstance(holder, ast.Name):
        if depth > _PARAM_DEPTH:
            return None
        found = reader.read_bindings("written names", holder, depth=depth, compute=lambda b, at: (
            _written_names(b.value, reader, b.cell, depth=at) if b.how == "value" and holder is target
            else _temp_file_names(b.value, reader)
        ))
        if not found or any(names is None for names in found):
            return None
        return [name for names in found if names is not None for name in names]
    node, after = target, False
    while True:
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
            node, after = node.right, isinstance(node.op, ast.Add)
        elif (isinstance(node, ast.Call) and _last_name(node.func) in ("join", "joinpath") and node.args
              and not isinstance(node.args[-1], ast.Starred)):
            node, after = node.args[-1], False
        elif isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[-1], ast.Constant):
            node, after = node.values[-1], len(node.values) > 1
        else:
            break
    if node is target:
        return None
    texts = reader.paths(node, cell=cell)
    if isinstance(texts, _Unknown):
        return None
    return [posixpath.basename(t) if "/" in t else _RUN_TIME_PART + t if after else t
            for t in (text.replace("\\", "/") for text in texts)]


# The makers of a temporary file that has a name on disk, with where each
# takes the suffix that name ends with.
_TEMP_FILE_SUFFIXES = {
    "tempfile.mkstemp": _ArgPlace(0, "suffix"), "tempfile.mktemp": _ArgPlace(0, "suffix"),
    "tempfile.NamedTemporaryFile": _ArgPlace(4, "suffix"),
}


def _temp_file_names(node: ast.AST, reader: _Reader) -> list[str] | None:
    """The names a temporary file a call (`_TEMP_FILE_SUFFIXES`) makes may
    have: its prefix, random letters, digits and underscores, then its
    suffix, so only the suffix can end it in `.pth`, and none is a hook
    module's. Each is read as "tmp" and a suffix `_literal_texts` reads,
    no suffix or a `None` as an empty one. None for any other node, or a
    suffix the check can't read."""
    if not isinstance(node, ast.Call):
        return None
    where = _TEMP_FILE_SUFFIXES.get(reader.module_member(node.func) or reader.dotted(node.func))
    if where is None:
        return None
    suffix = where.given(node)
    texts = [] if suffix is None else _literal_texts(suffix, reader)
    return None if texts is None else [f"tmp{text}" for text in texts or [""]]


def _unplaced_hook(call: ast.Call, target: ast.AST, reader: _Reader, cell: _Cell) -> bool:
    """Whether `call`, code of `cell` that writes to `target`, a path the
    check can't place, may write a startup hook a new Python process runs:
    a data writer's file (`_WRITER_METHODS`) only inside the installed
    packages, as `_hook_write` reads it (an `origin` only on a spec,
    `_may_be_a_spec`); any other where the file's name may
    be a hook's (`_written_names`, `_may_name_a_hook`), as a CSV saved into
    a temporary folder's isn't."""
    if _last_name(call.func) in _WRITER_METHODS:
        return _writes_site_packages(target, reader)
    names = _written_names(target, reader, cell)
    return names is None or any(_may_name_a_hook(name) for name in names)


# The start of a line of a `.pth` file that `site` runs as code.
_PTH_CODE_LINE = ("import ", "import\t")


def _runs_no_pth_line(call: ast.Call, cell: _Cell) -> bool:
    """Whether what `call`, code of `cell`, writes is a literal text or
    bytes with no line `site` runs from a `.pth` file (`_PTH_CODE_LINE`):
    the text a `write_text` or `write_bytes` is given, or that the file an
    `open` gives is handed by its `write` right away, as
    `open("weights.pth", "wb").write(b"x")` is. The text is read as `site`
    reads it: bytes decoded as UTF-8 past a byte order mark, else as
    Latin-1, and a byte order mark at the start of a text dropped."""
    if isinstance(call.func, ast.Attribute) and call.func.attr in ("write_text", "write_bytes"):
        given = call.args[0] if call.args else None
    else:
        member = cell.parents.get(call)
        write = cell.parents.get(member)
        given = (write.args[0] if isinstance(member, ast.Attribute) and member.attr == "write"
                 and isinstance(write, ast.Call) and write.func is member and write.args else None)
    if not (isinstance(given, ast.Constant) and isinstance(given.value, (str, bytes))):
        return False
    if isinstance(given.value, bytes):
        try:
            text = given.value.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = given.value.decode("latin-1")
    else:
        text = given.value.removeprefix("\ufeff")
    return not any(line.startswith(_PTH_CODE_LINE) for line in text.splitlines())


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
        if key in _BUILTINS_MODULE_NAMES or reader.exec_builtin(node) is not None:
            return True
        # An entry of a module's dict, as `vars(runpy)[k]` is `runpy.run_path`.
        name = reader.module_member(node)
        return name is not None and (name in _RUN_CALLS or name.startswith(_PYAPI_FUNCS))
    last = _last_name(node)
    if last in _LOADER_METHODS or last in _BUILTINS_MODULE_NAMES or reader.dotted(node) in _GLOBALS_CALLS:
        return True
    if _stores_code(node) or _sets_code(node):
        return True
    if last in _EXEC_CALLS:
        return reader.exec_builtin(node) is not None
    name = reader.module_member(node)
    if name is not None and (name in _RUN_CALLS or name.startswith(_PYAPI_FUNCS)):
        return True
    return isinstance(node, ast.Attribute) and last in _RUNNER_METHODS and last in _runner_methods(node.value, reader)


def _runs_code(cell: _Cell, reader: _Reader) -> bool:
    """Whether a name in `cell` names something that may run code the check
    can't read (`_names_a_runner`), so a call that reaches what such code
    defined may hold the cell, and a literal exec in it is read for what it
    defines."""
    return any(_names_a_runner(node, reader) for node in cell.signs.hints)


# The nodes `_unread_exec_at` or `_unread_run_at` can find a place in.
_UNREAD_SITES = (ast.Name, ast.Attribute, ast.Subscript, ast.Call, ast.Import, ast.ImportFrom)


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
    # Only these can be such a place (`_unread_exec_at`, `_unread_run_at`).
    nodes: Iterable[ast.AST] = (node for node in ast.walk(root) if isinstance(node, _UNREAD_SITES))
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
    """A star import of a module the check can read that surely runs when
    its cell runs (`_star_runs_surely`), as `from math import *`: the cell
    it is in, and the import."""

    cell: _Cell
    node: ast.ImportFrom


@dataclass(frozen=True)
class _Taint:
    """One place code the check can't read filled a name: the cell, and the
    line there; None where the check can't tell when it runs, as for a
    function's `global` assignment, which runs whenever the function does."""

    cell: _Cell
    line: int | None


# The names code the check can't read filled, each with the places it did.
_Tainted = Mapping[str, AbstractSet[_Taint]]


@dataclass(frozen=True)
class _Defined:
    """What code the check couldn't read may have defined before this cell.

    `tainted` names hold a namespace, module or value such code filled, so a
    call through one runs code the check never read, until read code gives
    the name a value of its own (`_filled_again`). `namespaces` are the
    cells of literal code an earlier exec ran into a namespace a name holds
    (`_Cell.namespace`), in the order they run, so a call through one
    resolves to the function those that ran before it defined there, the
    latest def of a name winning (`_namespace_at`). `values` are the names
    an assignment binds to a function a literal exec defined (`go =
    ns["go"]`) or to a lambda a literal eval made (`push = eval("lambda:
    ...")`), so a call to one reads that function. `bound` names a statement
    gives beyond what `_Reader.binds` knows (`_CodeSigns.bound_aside`), and
    `stars` holds the star imports read code ran, which may bind any name
    after them (`_star_binds`).
    """

    tainted: _Tainted
    namespaces: tuple[_Cell, ...]
    values: dict[str, Function]
    bound: frozenset[str]
    stars: tuple[_StarImport, ...]


@dataclass(frozen=True)
class _DefinedUses:
    """What code reaches of what code the check couldn't read may have
    defined (`_defined_uses`): the places that call or run such a name or
    value, which hold the cell when such code ran (`sites`); and each call
    through a namespace a literal exec in an earlier cell filled, read at
    that call so its request shows on the card (`callees`)."""

    sites: tuple[ast.AST, ...] = ()
    callees: tuple[_Callee, ...] = ()


def _star_binds(reader: _Reader, node: ast.Name, stars: Sequence[_StarImport]) -> bool:
    """Whether one of `stars` ran before the code `node` is in reads it: in a
    cell whose names land in the cell namespace that runs before it, or in
    its own cell before every place that runs that code (`_runs_from`):
    in a top-level statement before it, or before it in a block that holds
    it (`_Cell.runs_before`). The places are read in the order they sit in
    the cell, so once one comes after the star import's statement, every
    place after it does too."""
    cell = reader.cell_of(node)
    for star in stars:
        if star.cell is cell:
            runs = sorted(_runs_from(node, cell, reader), key=lambda n: (n.lineno, n.col_offset))
            for run in runs:
                if star.node.lineno < _top_line(run, cell):
                    return True
                if not cell.runs_before(star.node, run):
                    break
            else:
                return True
        elif star.cell.shared and star.cell.order < cell.order:
            return True
    return False


def _star_runs_surely(node: ast.ImportFrom, cell: _Cell) -> bool:
    """Whether the star import `node`, code of `cell`, surely runs whenever
    its scope does: no if, loop or try holds it (`_Cell.runs_surely`) but
    one that surely runs it (`_runs_its_body`), or a try that starts with
    it, of a module Python loads before it searches any folder
    (`_loaded_first`), which can't fail to import."""
    child, current = node, cell.parents.get(node)
    scope = cell.scope_of(node)
    while current is not None and current is not scope:
        if isinstance(current, _BRANCHES) and not _runs_its_body(current, child) and not (
            isinstance(current, (ast.Try, ast.TryStar)) and current.body[0] is node and node.module is not None
            and "." not in node.module and _loaded_first(node.module)
        ):
            return False
        child, current = current, cell.parents.get(current)
    return True


def _runs_its_body(branch: ast.AST, child: ast.AST) -> bool:
    """Whether `branch`, an if, loop or try, surely runs `child`, a
    statement of its body, whenever it runs: an if or a loop that surely
    starts its body (`_runs_body_surely`) with no break or continue before
    `child` (`_before_a_jump`), or a try whose handlers each end in a
    raise, so the body ran to its end once it is past."""
    body = getattr(branch, "body", ())
    if child not in body:
        return False
    if isinstance(branch, (ast.Try, ast.TryStar)):
        return all(h.body and isinstance(h.body[-1], ast.Raise) for h in branch.handlers)
    return _runs_body_surely(branch) and child in _before_a_jump(body)


def _outer_function(node: ast.AST, cell: _Cell) -> Function | None:
    """The outermost def or lambda of `cell` that holds `node`; None where
    none does."""
    outer: Function | None = None
    scope = cell.scope_of(node)
    while scope is not None:
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            outer = scope
        scope = cell.scope_of(scope)
    return outer


def _function_names(func: Function, reader: _Reader) -> list[str]:
    """The names `func` is bound to: a def's own, or each one a lambda is
    assigned to, none for a lambda no name holds (`_Reader.lambda_names`)."""
    if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return [func.name]
    return reader.lambda_names.get(id(func), [])


class _PlacedCalls(NamedTuple):
    """A cell's calls outside every def and lambda, in the order they sit in
    its code (`calls`), and the line and column of each (`places`), so a
    search finds those after a place (`_calls_after`)."""

    places: tuple[tuple[int, int], ...]
    calls: tuple[ast.Call, ...]


def _runs_from(node: ast.AST, cell: _Cell, reader: _Reader) -> list[ast.AST]:
    """The code of `cell` outside every def and named lambda that runs the
    code `node` is in: `node` itself, or, inside a def or a named lambda,
    each load outside it of a name the outermost such function is bound
    to (`_outer_function`, `_function_names`), which runs it when called,
    or hands it on to be called. A method, or a function a decorator is
    handed, runs through what holds it, which the check doesn't follow, so
    each call after it outside every def may run it, its class's body
    among them (`_calls_after`), and so does a decorator that calls it as
    it is handed it (`_decorator_runs`). Empty for a function nothing in
    the cell runs: a later cell calls it."""
    outer = _outer_function(node, cell)
    if outer is not None and _runs_through_a_holder(outer, cell):
        return [*_decorator_runs(outer, cell, reader), *_calls_after(outer, cell)]
    names = [] if outer is None else _function_names(outer, reader)
    if outer is None or not names:
        return [node]
    return [load for name in names for load in cell.loads_outside(outer, name)]


def _runs_through_a_holder(func: Function, cell: _Cell) -> bool:
    """Whether `func`, a def or lambda of `cell`, runs through what holds it
    rather than by its name: a method, which a class's instance calls, or
    a def a decorator is handed, which keeps it where it chooses."""
    return isinstance(cell.scope_of(func), ast.ClassDef) or bool(getattr(func, "decorator_list", ()))


def _calls_after(node: ast.AST, cell: _Cell) -> Sequence[ast.AST]:
    """The calls of `cell` outside every def and lambda
    (`_Cell.module_calls`) that may run `node`, a def or lambda of the
    cell, through what holds it: those after the top-level statement
    holding it, any of which may run what that statement made, and, for a
    method or a lambda a class's body binds, those later in that body that
    call it by a name the body binds to it, as `x = helper()` or
    `helper.__func__()` after `def helper()` there, and `sm.__func__()`
    after `sm = staticmethod(helper)`.
    All of them for the cell's tree, as for the checked cell calling into
    an earlier one's."""
    placed = cell.module_calls
    if node is cell.tree:
        return placed.calls
    top = node
    while (parent := cell.parents.get(top)) is not None and not isinstance(parent, ast.Module):
        top = parent
    top_end = bisect.bisect_left(placed.places, _end_of(top))
    after = placed.calls[top_end:]
    scope = cell.scope_of(node)
    if not isinstance(scope, ast.ClassDef):
        return after
    held: ast.AST = node
    while (parent := cell.parents.get(held)) is not None and parent is not scope:
        held = parent
    names = ({node.name} if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             else {name for target in held.targets for name in _stored_names(target)}
             if isinstance(held, ast.Assign) else set())
    for stmt in scope.body:
        if isinstance(stmt, ast.Assign) and _end_of(stmt) > _end_of(node) and any(
            isinstance(n, ast.Name) and n.id in names for n in ast.walk(stmt.value)
        ):
            names.update(name for target in stmt.targets for name in _stored_names(target))
    inside = placed.calls[bisect.bisect_left(placed.places, _end_of(node)):top_end]
    called = [call for call in inside if cell.scope_of(call) is scope
              and isinstance(root := _root_of(call.func), ast.Name) and root.id in names]
    return [*called, *after]


def _end_of(node: ast.AST) -> tuple[int, int]:
    """Where `node` ends in its code: its last line and column."""
    return getattr(node, "end_lineno", 0) or 0, getattr(node, "end_col_offset", 0) or 0


# The decorators that call the function they are handed as they are handed
# it, as ipywidgets' `interact` does with its defaults (`_decorator_runs`).
_RUNNING_DECORATORS = frozenset({"ipywidgets.interact", "ipywidgets.widgets.interact"})


def _decorator_runs(func: Function, cell: _Cell, reader: _Reader) -> list[ast.AST]:
    """The decorators of `func`, a def of `cell`, that call it as they are
    handed it, so it runs where its def does: one of
    `_RUNNING_DECORATORS`, called or not, or a function the pad defines
    that calls its first parameter outside every def inside it, as `def
    now(f): f(); return f` does. Any other decorator keeps it to call
    later, as `atexit.register` does."""
    out: list[ast.AST] = []
    for decorator in getattr(func, "decorator_list", ()):
        head = decorator.func if isinstance(decorator, ast.Call) else decorator
        if reader.module_member(head) in _RUNNING_DECORATORS or reader.dotted(head) in _RUNNING_DECORATORS:
            out.append(decorator)
            continue
        made = reader.function_named(head) if isinstance(head, ast.Name) and head is decorator else None
        params = [] if made is None else [*made.args.posonlyargs, *made.args.args][:1]
        if params and any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == params[0].arg
            and reader.cell_of(made).scope_of(n) is made for n in ast.walk(made)
        ):
            out.append(decorator)
    return out


def _module_runs(starts: Iterable[ast.AST], cell: _Cell, reader: _Reader) -> list[ast.AST]:
    """The places of `cell` outside every def and named lambda that run the
    code each of `starts`, code of `cell`, is in (`_runs_from`), through any
    chain of the cell's functions that run one another: a start itself
    outside them, else each place that runs the function holding it, read
    the same way in turn. Empty where nothing in the cell runs them."""
    out: list[ast.AST] = []
    seen: set[int] = set()
    todo = list(starts)
    while todo:
        current = todo.pop()
        runs = _runs_from(current, cell, reader)
        if len(runs) == 1 and runs[0] is current:
            out.append(current)
            continue
        for run in runs:
            if id(run) not in seen:
                seen.add(id(run))
                todo.append(run)
    return out


def _bound_where_run(reader: _Reader, node: ast.Name, cell: _Cell) -> bool:
    """Whether read code surely gave the module's name `node` reads, from
    inside a def, lambda, class or comprehension of `cell`, a
    conversation's cell, a value wherever that code runs
    (`_Reader.binds_surely`): at module level before the outermost of
    them, so on every run after it; else before each place at module level
    that runs it (`_module_runs`), in its own cell, or, for a function of
    an earlier cell, in the current one, where a cell before it that bound
    the name on every path (`_Reader.bound_before`) gave it one too. A
    method or a decorated def of an earlier cell may run at any call of the
    current one (`_calls_after`). Where nothing in its own cell runs it,
    only a later cell can, so a binding anywhere in that cell that surely
    runs counts (`_Cell.binds_whole`). A binding an if, loop or try may
    skip gives no value. A def that declares the name `global` gives it
    one where its own body surely binds it before the read.
    """
    name = node.id
    declaring = next((scope for scope in cell.scopes_around(node) if cell.declares_outer(scope, name)), None)
    if declaring is not None and cell.bound_on_the_way(
        node, name, setters=reader.global_setters(name), swallowers=reader.swallowers, within=declaring,
    ):
        return True
    outer: ast.AST = node
    while (scope := cell.scope_of(outer)) is not None:
        outer = scope
    if reader.binds_surely(outer, name):
        return True
    current = reader.current
    if cell is current:
        runs = _module_runs([node], cell, reader)
        if not runs:
            # Nothing in the cell runs it, so only a later cell can, once
            # this one ran to its end.
            return cell.binds_whole(name, setters=reader.global_setters(name), swallowers=reader.swallowers)
    elif reader.bound_before(name, current):
        return True
    else:
        func = _outer_function(node, cell)
        if func is not None and _runs_through_a_holder(func, cell):
            starts: Sequence[ast.AST] = _calls_after(current.tree, current)
        else:
            names = [] if func is None else _function_names(func, reader)
            starts = [load for n in names for load in current.loads.get(n, ())]
        runs = _module_runs(starts, current, reader)
    return bool(runs) and all(reader.binds_surely(run, name) for run in runs)


def _first_run_at(node: ast.AST, cell: _Cell, reader: _Reader) -> tuple[int, int]:
    """Where the code `node`, code of `cell`, is in first runs as the cell
    runs, by line and column (`_runs_from`): its own place outside every
    def, or the first call of the function it is in; past the cell's last
    line for a function the cell doesn't name, which only a later cell
    can call."""
    return min(((n.lineno, n.col_offset) for n in _runs_from(node, cell, reader)), default=(len(cell.lines) + 1, 0))


def _top_line(node: ast.AST, cell: _Cell) -> int:
    """The line of the top-level statement of `cell` that holds `node`."""
    while not isinstance(cell.parents.get(node), (ast.Module, type(None))):
        node = cell.parents[node]
    return getattr(node, "lineno", 0)


def _unvouched(reader: _Reader, node: ast.Name, defined: _Defined, tainted: _Tainted) -> bool:
    """Whether the name `node` reads may come from code the check can't read:
    a `tainted` name that read code hasn't given a value of its own since
    (`_filled_again`), or one no read code gives a value (a builtin or a
    name the pad binds, a name `defined.bound` holds, or what
    `_Reader.binds` covers), unless a star import of a readable module ran
    before it. In a conversation's cell, a read of a module's name, at
    module level or from inside a def, lambda, class or comprehension that
    doesn't bind it in its own scope (`_Cell.binds_in_scope`), gets a value
    only from a binding that surely ran on every path to the read
    (`_Reader.binds_surely`) or to each place that runs that code
    (`_bound_where_run`): where an if, loop or try may skip it, the name
    holds what such code put there. A module-level read of literal code
    that runs in the cell namespace is read the same way.
    """
    if node.id in tainted and not _filled_again(reader, node, tainted):
        return True
    cell = reader.cell_of(node)
    if reader.bound_before(node.id, cell):
        return False
    if node.id in _BUILTIN_NAMES or node in cell.looked_up or _star_binds(reader, node, defined.stars):
        return False
    if cell.index == cell.host:
        if not cell.scopes_around(node):
            return not reader.binds_surely(node, node.id)
        if not cell.binds_in_scope(node, node.id, swallowers=reader.swallowers):
            return not _bound_where_run(reader, node, cell)
    elif cell.shared and not cell.scopes_around(node):
        # Literal code run in the cell namespace, read as its host's cells are.
        return not reader.binds_surely(node, node.id)
    return not (node.id in defined.bound or reader.binds(node, surely=True))


def _filled_again(reader: _Reader, node: ast.Name, tainted: _Tainted) -> bool:
    """Whether read code surely gave the name `node` reads, at module level,
    a value of its own after each place `tainted` holds for it that ran
    before the read: a sure binding that reaches the read
    (`_Reader.bindings_of`) and reads no tainted name, or a def, class or
    import of the name before the read that no if, loop or try holds,
    comes after them all. Literal code counts only where the call that
    runs it surely runs (`_Cell.surely_run`), and a cell that stopped on an
    error only for what surely ran before it stopped
    (`_Reader.ran_before_stop`). A value that reads a tainted name may hold
    what it holds, however it gets there, as `[push][0]` does."""
    cell = reader.cell_of(node)
    if cell.scope_of(node) is not None:
        return False
    read_at = reader.position(node)
    before = [t for t in tainted[node.id] if t.line is None or (*t.cell.order, t.line) < read_at]
    if not before:
        return True
    if any(t.line is None for t in before):
        return False
    latest = max((*t.cell.order, t.line or 0) for t in before)
    stopped = reader.stopped_early

    def ran(cell: _Cell, made: ast.AST, *, head: bool = False) -> bool:
        if cell.host in stopped:
            return reader.ran_before_stop(cell, made, head=head)
        return cell.surely_run and cell.runs_to_its_end(cell.statement_of(made), swallowers=reader.swallowers) and (
            not head or cell.binds_loop_target(made)
        )

    places = [(*b.cell.order, b.line) for b in reader.bindings_of(node.id, node=node)
              if b.scope is None and ran(b.cell, b.value, head=b.how in ("element", "context"))
              and tainted.keys().isdisjoint(b.cell.names_within(b.value))]
    at = reader.position(node)
    places += [(*maker.cell.order, maker.statement.lineno) for maker in reader.makers.get(node.id, ())
               if (maker.cell is cell or maker.cell.shared and maker.cell.order < at)
               and maker.cell.scope_of(maker.statement) is None and ran(maker.cell, maker.statement)
               and reader.defined_ahead(maker.cell, maker.statement, node)]
    return any(place > latest for place in places)


def _root_of(node: ast.AST) -> ast.AST:
    """What a chain of members and entries starts from: `ns` for
    `ns["f"].run`, the call for `runpy.run_path("job.py")["main"]`."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node


@dataclass(frozen=True)
class _Unread:
    """What `_from_unread` reads a value against: the cell it is in, the
    modules an import may load unread, what such code may have defined, the
    names known to hold what it filled so far, and the functions whose
    return values are being read, by id, so a recursive one is read once."""

    reader: _Reader
    cell: _Cell
    modules: _Modules
    defined: _Defined
    tainted: _Tainted
    returning: frozenset[int] = frozenset()


def _from_unread(value: ast.AST, at: _Unread) -> bool:
    """Whether `value` holds what code the check can't read made: a tainted
    name or one no read code binds (`_unvouched`), a member or entry of one,
    or what calling one returns; a module an import loads unread, or an
    entry of `sys.modules`, or a member of one, under a name the check
    can't read or of such a module (`_module_entry_unread`); what exec(), eval()
    or compile() makes of a source that isn't a literal; the namespace a
    runner returns (`_NAMESPACE_RUNNERS`), and what any other runner makes
    (`_CODE_RUNNERS`), as a spec or a loader; a module an importer loads
    unread or by a name the check can't read; what a call handed one of
    those returns, but for a call that hands nothing back
    (`_is_harmless_call`); what a lambda or a function read code defines
    returns (`_returns_unread`); and a tuple, list, set, dict,
    comprehension, branch or `or` with one of those in it."""
    reader = at.reader
    while isinstance(value, (ast.NamedExpr, ast.Await, ast.Starred)):
        value = value.value
    if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
        return any(_from_unread(e, at) for e in value.elts)
    if isinstance(value, ast.Dict):
        return any(_from_unread(v, at) for v in (*value.keys, *value.values) if v is not None)
    if isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        parts = [value.key, value.value] if isinstance(value, ast.DictComp) else [value.elt]
        return any(_from_unread(p, at) for p in (*parts, *(g.iter for g in value.generators)))
    if isinstance(value, ast.IfExp):
        return _from_unread(value.body, at) or _from_unread(value.orelse, at)
    if isinstance(value, ast.BoolOp):
        return any(_from_unread(v, at) for v in value.values)
    if isinstance(value, (ast.Attribute, ast.Subscript)):
        entry = _sys_modules_entry(value, reader)
        if entry is not None:
            return _module_entry_unread(entry, at)
        return _from_unread(_root_of(value), at)
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
    name = reader.module_member(func) if _runner_named(func, reader) else None
    if name in _NAMESPACE_RUNNERS or name in _CODE_RUNNERS:
        return True
    if name in _IMPORTERS:
        return _imports_unread(value, reader, at.cell, modules=at.modules)
    if not _is_harmless_call(value, reader) and any(
        _from_unread(arg, at) for arg in (*value.args, *(k.value for k in value.keywords))
    ):
        # Handed on, as `functools.partial(push)`, `getattr(job, "main")` or `dict(ns)` are.
        return True
    return _returns_unread(value, at) or _from_unread(_root_of(func), at)


def _sys_modules_entry(node: ast.AST, reader: _Reader) -> ast.Subscript | None:
    """The entry of `sys.modules`, as the imports read it, that a chain of
    members and entries starts from: `sys.modules["job"]` for
    `sys.modules["job"].main`; None for any other chain."""
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        if isinstance(node, ast.Subscript) and reader.dotted(node.value) == "sys.modules":
            return node
        node = node.value
    return None


def _module_entry_unread(entry: ast.Subscript, at: _Unread) -> bool:
    """Whether `entry`, an entry of `sys.modules`, may hold a module code
    the check can't read made: one under a name the check can't read, or
    one an import of that name loads unread (`_Modules.unread`)."""
    key = entry.slice.value if isinstance(entry.slice, ast.Constant) else None
    return not isinstance(key, str) or at.modules.unread(key, cell=at.cell)


def _returns_unread(call: ast.Call, at: _Unread) -> bool:
    """Whether `call` calls a lambda, or a name read code binds to a lambda or
    a def, whose return value is what code the check can't read made
    (`_from_unread`), as `make()` does after `def make(): return eval(src)`.
    A function the chain of such calls reads already counts as not."""
    func = call.func
    reader = at.reader
    found: list[Function] = []
    if isinstance(func, ast.Lambda):
        found.append(func)
    elif isinstance(func, ast.Name) and func.id in reader.function_names:
        found += [b.value for b in reader.bindings_of(func.id, node=func)
                  if b.how == "value" and isinstance(b.value, ast.Lambda)]
        named = reader.function_named(func)
        if named is not None:
            found.append(named)
    for target in found:
        if id(target) in at.returning:
            continue
        cell = reader.cell_of(target)
        returned = [target.body] if isinstance(target, ast.Lambda) else cell.returns_run_by(target)
        inner = replace(at, cell=cell, returning=at.returning | {id(target)})
        if any(_from_unread(v, inner) for v in returned):
            return True
    return False


class _Store(NamedTuple):
    """The names one binding gives, and the value it gives them."""

    names: list[str]
    value: ast.AST


def _store_heads(node: ast.AST) -> list[ast.AST]:
    """The code that gives the names `node` binds their values: a loop's
    iterable, each context a with enters, and any other statement whole."""
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return [node.iter]
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return [item.context_expr for item in node.items]
    return [node]


def _stored_names(target: ast.AST) -> list[str]:
    """The names an assignment's, loop's or with's target binds, nested and
    starred parts included; a subscript's or attribute's holder binds none."""
    return [n.id for n in ast.walk(target) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]


def _stores(node: ast.AST) -> Iterator[_Store]:
    """The names `node` binds, each with the value it gives them: an
    assignment's (a tuple assigned a tuple of the same length pairs them up,
    any other unpacking gives every name the whole value, and a store into
    the cell's own namespace by a literal key gives that key, as
    `globals()["push"] = f` does), a walrus's, a loop's whole iterable, the
    context a with enters, and the defaults of a function's or lambda's
    parameters."""
    if isinstance(node, (ast.For, ast.AsyncFor)):
        yield _Store(_stored_names(node.target), node.iter)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            if item.optional_vars is not None:
                yield _Store(_stored_names(item.optional_vars), item.context_expr)
    elif isinstance(node, ast.Assign) or isinstance(node, ast.AnnAssign) and node.value is not None:
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
            elif (key := _namespace_key(target)) is not None:
                yield _Store([key], value)
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


def _tainted_in(
    reader: _Reader, nodes: Iterable[ast.AST], cell: _Cell, defined: _Defined, modules: _Modules,
) -> _Unread:
    """What `_from_unread` reads code of `cell` against, with each name one
    of `nodes` binds, in any scope, to what code the check can't read made
    tainted beside `defined.tainted`, in the order of `nodes`, so a name
    bound to one tainted before it is tainted too. The names it taints sit
    over `defined.tainted` (`ChainMap`), which isn't copied for each cell."""
    own: dict[str, set[_Taint]] = {}
    at = _Unread(reader=reader, cell=cell, modules=modules, defined=defined, tainted=ChainMap(own, defined.tainted))
    for node in nodes:
        for names, value in _stores(node):
            if names and _from_unread(value, at):
                for name in names:
                    own.setdefault(name, set(defined.tainted.get(name, ()))).add(_taint_at(node, cell))
    return at


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
    (`ns["f"]()`, `m.f()`). A with that enters such a name, or a member of
    one, as `with guard:` does, calls its enter and exit methods, so it is a
    site too; a context that is a call is read as the call. Nothing when no
    code the check can't read may run (`defined` is None)."""
    if defined is None:
        return _DefinedUses()
    # A whole cell lists its stores, defs, classes and calls already.
    nodes: Sequence[ast.AST] = (
        (*cell.stores, *cell.definitions, *cell.calls) if root is cell.tree else list(ast.walk(root))
    )
    at = _tainted_in(reader, nodes, cell, defined, modules)
    tainted = at.tainted
    sites: list[ast.AST] = []
    callees: list[_Callee] = []

    def unvouched(node: ast.AST) -> bool:
        entry = _sys_modules_entry(node, reader)
        if entry is not None:
            # `sys.modules["job"].main`, for a module unread code stored there.
            return _module_entry_unread(entry, at)
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
        filled = (_namespace_at(defined.namespaces, holder.id, at=reader.position(node))
                  if isinstance(holder, ast.Name) else {})
        if isinstance(member, str) and member in filled:
            callees.append(_Callee(func=filled[member], call=node, use=func))
        head = _root_of(func)
        if (unvouched(func) and not (member in _DICT_METHODS and _holds_a_dict(holder, reader))) or (
            not isinstance(head, ast.Name) and _from_unread(head, at)
        ):
            # A call through such a name, or of what a call made of one
            # returns, as `g.get("main")()` calls the entry it reads.
            sites.append(node)
        elif not _is_harmless_call(node, reader) and not (
            isinstance(func, ast.Name) and func.id in _INSPECTS and len(node.args) == 1 and not reader.binds(func)
        ):
            # A walk over a dict, as `list(ns)` or `sorted(ns)`, reads its keys alone.
            walks = reader.dotted(func) in _ITER_WRAPPERS
            for arg in (*node.args, *(k.value for k in node.keywords)):
                named = arg.value if isinstance(arg, ast.Starred) else arg
                if isinstance(named, (ast.Name, ast.Attribute, ast.Subscript)) and unvouched(named) and not (
                    walks and _holds_a_dict(named, reader)
                ):
                    sites.append(named)
    withs = cell.signs.withs if root is cell.tree else [n for n in nodes if isinstance(n, (ast.With, ast.AsyncWith))]
    sites += [context for stmt in withs for item in stmt.items for context in [item.context_expr]
              if isinstance(context, (ast.Name, ast.Attribute, ast.Subscript)) and unvouched(context)]
    return _DefinedUses(sites=tuple(sites), callees=tuple(callees))


# The methods of a dict that read or take its entries and run none of
# them: a call of one on a namespace code the check can't read filled
# calls nothing that code made, though what it returns may be such code.
_DICT_METHODS = _DICT_READS | _ENTRY_READS
# Builtins that read the one value they're handed without running any of
# its code, as `type(v)` does: what they return may still be such code.
_INSPECTS = frozenset({"type", "id", "callable"})


def _holds_a_dict(node: ast.AST | None, reader: _Reader) -> bool:
    """Whether `node` is a name read code surely bound where it is read
    (`_Reader.binds`), and each binding that reaches it gives a dict: a dict
    display or comprehension, a `dict(...)` call, or the namespace a runner
    returns (`_NAMESPACE_RUNNERS`), which code the check can't read may fill
    but can't make anything else."""
    if not isinstance(node, ast.Name) or not reader.binds(node, surely=True):
        return False
    found = reader.bindings_of(node.id, node=node)
    return bool(found) and all(
        b.how == "value" and (
            isinstance(b.value, (ast.Dict, ast.DictComp))
            or isinstance(b.value, ast.Call) and (
                reader.dotted(b.value.func) == "dict" or reader.module_member(b.value.func) in _NAMESPACE_RUNNERS
            )
        )
        for b in found
    )


def _plainly_read(node: ast.AST, reads: _FillReads, cell: _Cell, reader: _Reader, tainted: _Tainted) -> bool:
    """Whether `node`, a store of `cell` reading `reads`, surely stores
    nothing code the check can't read made, as `_from_unread` reads it: it
    calls only methods by names that run no code (`_is_hint`,
    `_Reader.runner_aliases`), each name it loads is untainted, read as
    itself, imports no `sys` (whose `sys.modules` holds every module), and
    is bound by read code at module level of a cell that runs before this
    one (`_Reader.bound_before`), or of this one on an earlier line by an
    assignment no if, loop or try holds; and it
    gives no function a code object (`_stores_code`). A cheaper read of
    most values a long conversation stores."""
    if reads.methods is None or any(_is_hint(m) or m in reader.runner_aliases for m in reads.methods):
        return False
    if isinstance(node, ast.Assign) and any(_stores_code(t) for t in node.targets) or _sets_code(node):
        return False
    if any(reader.imports.get(name) in _SYS_MODULE_NAMES for name in reads.names):
        return False
    looked_up = {n.id for n in cell.looked_up}
    line = getattr(node, "lineno", 0)
    own = cell.host not in reader.stopped_early and cell.scope_of(node) is None
    return all(
        name not in tainted and name not in looked_up and (
            reader.bound_before(name, cell)
            or own and any(b.scope is None and b.line < line and not b.conditional and b.how != "element"
                           for b in cell.bindings.get(name, ()))
        )
        for name in reads.names
    )


def _taint_at(node: ast.AST, cell: _Cell) -> _Taint:
    """The place `node`, code of `cell` that fills a name, fills it: its
    line at module level, and nowhere the check can tell inside a function
    or class, which may run at any time."""
    return _Taint(cell=cell, line=getattr(node, "lineno", None) if cell.scope_of(node) is None else None)


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
    (`_from_unread`), a function an assignment or a `setattr` gives code
    the check can't read (`_swaps_in_unread_code`, `_sets_code`), and the
    namespace a non-literal exec() or eval() is handed."""
    if isinstance(node, ast.Import):
        return {a.asname or a.name.partition(".")[0] for a in node.names if at.modules.unread(a.name, cell=at.cell)}
    if isinstance(node, ast.ImportFrom):
        if node.level or not node.module or not at.modules.unread(node.module, cell=at.cell):
            return set()
        return set(_import_names(node))
    if isinstance(node, ast.Call):
        if _sets_code(node):
            target = node.args[0]
            return {target.id} if isinstance(target, ast.Name) and _unread_code_object(node.args[2], at.cell) else set()
        if at.reader.exec_builtin(node.func) not in ("exec", "eval"):
            return set()
        source = _exec_source(node, "exec")
        namespace = node.args[1] if len(node.args) > 1 else _keyword(node, "globals")
        root = _namespace_root(namespace)
        return {root} if source is not None and _literal_source(source) is None and root is not None else set()
    # A function given code the check can't read, as `run.__code__ = co`.
    swapped = {t.value.id for t in node.targets if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
               and _swaps_in_unread_code(t, at.cell)} if isinstance(node, ast.Assign) else set()
    return swapped | {name for names, value in _stores(node) if _from_unread(value, at) for name in names}


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
    names read code binds beyond `_Reader.binds`; and each star import of a
    readable module that surely runs (`_star_runs_surely`). An errored cell
    may have stopped before its star import. Names are tainted from
    `taint_from` on, the first cell where such code ran (`_first_unread`):
    before it a name no read code bound held nothing yet. With no such
    cell, nothing is tainted. The modules such code stored in `sys.modules`
    (`_stored_modules`) go to `modules` as the same worklist reads them,
    since an import of one fills what it binds (`_names_tainted`)."""
    namespaces = tuple(sorted(
        (c for c in literal.cells if c.namespace is not None and c.host < reader.current.index), key=lambda c: c.order,
    ))
    values = _literal_exec_values(reader, cells, literal, namespaces, errored=errored)
    start = None if taint_from is None else taint_from.order
    tainting = [c for c in cells if start is not None and c.order >= start]
    bound = frozenset(name for cell in cells if cell.host not in errored for name in cell.signs.bound_aside)
    surely = [_StarImport(cell=cell, node=n) for cell in cells if cell.host not in errored for n in cell.signs.imports
              if isinstance(n, ast.ImportFrom) and n.module and any(a.name == "*" for a in n.names)
              and _star_runs_surely(n, cell)]

    def base() -> _Defined:
        """What is known before the taints, with the star imports of the
        modules `modules` reads now (`_Modules.unread`)."""
        stars = tuple(star for star in surely if not modules.unread(star.node.module or "", cell=star.cell))
        return _Defined(tainted={}, namespaces=namespaces, values=values, stars=stars, bound=bound)

    return _names_tainted(reader, tainting, cells, modules, base, start=start)


class _Fill(NamedTuple):
    """One place a cell may fill names with what code the check can't read
    made (`_names_tainted`): `node`, a store of `_Cell.fills_that_run` that
    reads `reads`, or, with `name`, the value a function sets that name,
    which it declares `global`, to; and the names it loads (`loads`)."""

    cell: _Cell
    node: ast.AST
    loads: frozenset[str]
    reads: _FillReads | None = None
    name: str | None = None


class _ModuleFill(NamedTuple):
    """One cell's stores of a module in `sys.modules` (`_sys_module_stores`),
    read in the worklist of `_names_tainted` for the modules code the check
    can't read made (`_stored_modules`), and the names the cell's stores
    load (`loads`), which `_tainted_in` reads them against."""

    cell: _Cell
    stores: tuple[_ModuleStoreSite, ...]
    loads: frozenset[str]


def _import_tops(node: ast.AST) -> list[str]:
    """The top-level packages an import statement loads; none for a relative
    import or anything else."""
    if isinstance(node, ast.Import):
        return [a.name.partition(".")[0] for a in node.names]
    if isinstance(node, ast.ImportFrom) and node.module and not node.level:
        return [node.module.partition(".")[0]]
    return []


def _names_tainted(
    reader: _Reader, tainting: Sequence[_Cell], cells: Sequence[_Cell], modules: _Modules,
    base: Callable[[], _Defined], *, start: tuple[int, ...] | None,
) -> _Defined:
    """What code the check couldn't read filled in `tainting`, the cells
    from the first where such code ran on (`_build_defined`), over what
    `base` gives, read in one worklist with the modules `cells` store in
    `sys.modules` from such code (`_ModuleFill`): each fill read in order,
    then each that reads a name a read tainted more read again, until no
    more turn up. Each is listed under the names it loads, so a chain of
    names each built on the one before it costs one read again per link,
    not a pass over every fill. A module stored so goes to `modules`, and
    then the imports of its top-level package are read again, with each
    other read that asked `modules` (`_Modules.asked`); where the star
    imports that vouch change with it, every fill is."""
    fills: list[_Fill | _ModuleFill] = []
    for cell in tainting:
        fills += [_Fill(cell=cell, node=node, loads=reads.names, reads=reads)
                  for node, reads in zip(cell.fills_that_run, cell.fill_reads)]
        fills += [_Fill(cell=cell, node=value, name=name, loads=frozenset(
            n.id for n in ast.walk(value) if isinstance(n, ast.Name)
        )) for name, value in _global_fills(cell)]
    for cell in cells:
        stores = _sys_module_stores(reader, cell)
        if stores:
            loads = frozenset(name for node in cell.stores for name in cell.names_within(node))
            fills.append(_ModuleFill(cell=cell, stores=stores, loads=loads))
    readers: dict[str, list[int]] = {}
    imports: dict[str, list[int]] = {}
    for i, fill in enumerate(fills):
        for name in fill.loads:
            readers.setdefault(name, []).append(i)
        for top in _import_tops(fill.node) if isinstance(fill, _Fill) else ():
            imports.setdefault(top, []).append(i)
    defined = base()
    tainted: dict[str, set[_Taint]] = {}
    known = set(modules.stored)
    # The reads other than an import's that asked `modules` the last time.
    asking: set[int] = set()
    pending = deque(range(len(fills)))
    queued = [True] * len(fills)

    def again(indices: Iterable[int]) -> None:
        for j in indices:
            if not queued[j]:
                queued[j] = True
                pending.append(j)

    while pending:
        i = pending.popleft()
        queued[i] = False
        fill = fills[i]
        cell = fill.cell
        asked = modules.asked
        if isinstance(fill, _ModuleFill):
            stored = _stored_modules(reader, cell, fill.stores, modules, replace(defined, tainted=tainted),
                                     tainting=start is not None and cell.order >= start)
            more = [entry for entry in dict.fromkeys(stored) if entry not in known]
            if modules.asked != asked:
                asking.add(i)
            if more:
                known.update(more)
                modules.stored = (*modules.stored, *more)
                tops = {entry.name.partition(".")[0] if entry.name is not None else None for entry in more}
                again(asking)
                again(j for top in tops for j in (
                    (k for found in imports.values() for k in found) if top is None else imports.get(top, ())
                ))
                fresh = base()
                if fresh.stars != defined.stars:
                    defined = fresh
                    again(range(len(fills)))
            continue
        at = _Unread(reader=reader, cell=cell, modules=modules, defined=defined, tainted=tainted)
        if fill.reads is not None:
            found = [] if _plainly_read(fill.node, fill.reads, cell, reader, tainted) else [
                (name, _taint_at(fill.node, cell)) for name in _filled_names(fill.node, at)
            ]
        else:
            found = [(fill.name, _Taint(cell=cell, line=None))] if fill.name and _from_unread(fill.node, at) else []
        if modules.asked != asked and not isinstance(fill.node, (ast.Import, ast.ImportFrom)):
            asking.add(i)
        for name, taint in found:
            if taint not in tainted.setdefault(name, set()):
                tainted[name].add(taint)
                again(readers.get(name, ()))
    return replace(defined, tainted={name: frozenset(taints) for name, taints in tainted.items()})


# Calls that make a module, or a module's spec, under the name their first
# argument gives, which its `.name` or `.__name__` reads back; and the call
# that makes a module from a spec, under the spec's name.
_NAMED_MODULE_MAKERS = frozenset({
    "importlib.util.spec_from_file_location", "importlib.util.spec_from_loader", "importlib.util.find_spec",
    "importlib.machinery.PathFinder.find_spec", "importlib.machinery.ModuleSpec", "types.ModuleType",
})
_MODULE_FROM_SPEC = "importlib.util.module_from_spec"


def _module_key_texts(key: ast.AST, reader: _Reader) -> list[str] | None:
    """The names `key`, the key a module is stored under in `sys.modules`,
    may hold (`_literal_texts`); for `spec.name` or `module.__name__`, the
    name the spec or module was made under (`_made_under`). None where the
    check can't read it."""
    texts = _literal_texts(key, reader)
    if texts is None and isinstance(key, ast.Attribute) and key.attr in ("name", "__name__"):
        return _made_under(key.value, reader)
    return texts


def _made_under(node: ast.AST, reader: _Reader, *, depth: int = 0) -> list[str] | None:
    """The names the module or module spec `node` holds was made under: the
    first argument of a call that makes one (`_NAMED_MODULE_MAKERS`), read
    with `_literal_texts`, a spec's for a module made from it, there or
    through each value a name is bound to (`_Reader.read_bindings`). None
    where any of them can't be read, or past `_PARAM_DEPTH` names."""
    if isinstance(node, ast.Name) and depth < _PARAM_DEPTH:
        found = reader.read_bindings(
            "made under", node, depth=depth,
            compute=lambda b, at: _made_under(b.value, reader, depth=at) if b.how == "value" else None,
        )
        if not found or any(f is None for f in found):
            return None
        return [text for f in found if f is not None for text in f]
    if not (isinstance(node, ast.Call) and node.args):
        return None
    maker = reader.module_member(node.func)
    if maker == _MODULE_FROM_SPEC:
        return _made_under(node.args[0], reader, depth=depth + 1)
    return _literal_texts(node.args[0], reader) if maker in _NAMED_MODULE_MAKERS else None


def _sys_module_stores(reader: _Reader, cell: _Cell) -> tuple[_ModuleStoreSite, ...]:
    """The stores of `cell` into `sys.modules`, as the imports read it
    (`_Cell.module_stores`), there or through a bare name bound to it
    (`_Reader.module_table_names`)."""
    return tuple(s for s in cell.module_stores if (
        not isinstance(s.holder, ast.Name) or s.holder.id in reader.module_table_names
    ) and "sys.modules" in (reader.dotted(s.holder), reader.module_member(s.holder)))


def _stored_modules(
    reader: _Reader, cell: _Cell, stores: Sequence[_ModuleStoreSite], modules: _Modules, defined: _Defined,
    *, tainting: bool,
) -> list[_UnreadModuleEntry]:
    """The modules `cell` stores in `sys.modules` at `stores`
    (`_sys_module_stores`) from code the check can't read: with
    `tainting`, for a cell from the first place such code ran on, a module
    `_from_unread` reads as such code's, with each name the cell binds to
    one, in any scope, tainted too (`_tainted_in`); in any cell, a module a
    literal exec filled (`_Defined.namespaces`), whose functions an import
    of it doesn't reach. Each under every name its key may hold
    (`_module_key_texts`), or under None where the check can't read the key."""
    out: list[_UnreadModuleEntry] = []
    at = _tainted_in(reader, cell.stores, cell, defined, modules) if tainting else None
    for store in stores:
        filled = _namespace_root(store.value)
        literal = filled is not None and any(c.namespace == filled for c in defined.namespaces)
        if not (literal or at is not None and _from_unread(store.value, at)):
            continue
        texts = _module_key_texts(store.key, reader) if store.key is not None else None
        out += [_UnreadModuleEntry(name=None, order=cell.order)] if texts is None else [
            _UnreadModuleEntry(name=text, order=cell.order) for text in texts
        ]
    return out


class _NameValue(NamedTuple):
    """A name and a value bound to it."""

    name: str
    value: ast.AST


def _partial_head(value: ast.AST) -> ast.AST | None:
    """The function a `functools.partial(f, ...)` call `value` wraps, through
    a partial of a partial written out, as `subprocess.run` for
    `partial(partial(subprocess.run, cmd), check=True)`; None for any other
    value."""
    head: ast.AST | None = None
    while isinstance(value, ast.Call) and _last_name(value.func) == "partial" and value.args:
        head = value = value.args[0]
    return head


def _literal_exec_values(
    reader: _Reader, cells: Sequence[_Cell], literal: _ExecRead, namespaces: Sequence[_Cell],
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
    if not lambdas and not namespaces:
        return values
    for cell in cells:
        if not cell.shared or cell.host in errored or cell.host >= reader.current.index:
            continue
        for node in cell.fills_that_run:
            for names, value in _stores(node):
                func = _namespace_function(value, namespaces, at=(*cell.order, value.lineno)) or (
                    lambdas.get((cell.host, value.lineno)) if _is_literal_eval(value, reader) else None
                )
                if func is not None:
                    for name in names:
                        values[name] = func
    return values


def _namespace_function(value: ast.AST, namespaces: Sequence[_Cell], *, at: tuple[int, ...]) -> Function | None:
    """The function a literal exec defined that `value`, an entry of a
    namespace it filled (`ns["go"]`) read at `at` (`_Reader.position`),
    reads (`_namespace_at`); None otherwise."""
    if (isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name)
            and isinstance(value.slice, ast.Constant) and isinstance(value.slice.value, str)):
        return _namespace_at(namespaces, value.value.id, at=at).get(value.slice.value)
    return None


def _namespace_at(namespaces: Sequence[_Cell], name: str, *, at: tuple[int, ...]) -> dict[str, Function]:
    """The functions `namespaces`, cells of literal code in the order they
    run, defined in the namespace `name` holds before `at`
    (`_Reader.position`), by name: a namespace keeps what each of them put
    there, the latest one's def of a name over an earlier one's. Empty when
    none filled it."""
    out: dict[str, Function] = {}
    for cell in namespaces:
        if cell.namespace == name and cell.order < at:
            out.update(cell.functions)
    return out


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
    handed `globals()` and a source that isn't a literal, or a compile() of
    one: a literal's names are read with the cell's (`_exec_place`).
    `vars(obj)` is the object's own, not the cell's."""
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
        builtin = reader.exec_builtin(func)
        if builtin in ("exec", "eval"):
            args = (*node.args[1:], *(k.value for k in node.keywords if k.arg in ("globals", "locals")))
            source = _exec_source(node, builtin)
            if isinstance(source, ast.Call) and reader.exec_builtin(source.func) == "compile":
                source = _exec_source(source, "compile")
            literal = source is not None and _literal_source(source) is not None
            return not literal and any(_cell_namespace(a, reader) for a in args)
    return False


def _hands_on_namespace(node: ast.Call, parent: ast.AST | None) -> bool:
    """Whether `node`, a call that gives the cell's namespace, hands it on
    whole where `parent` holds it: bound to a name, spread into a call or
    passed to one, as `SimpleNamespace(**globals())` and
    `obj.__dict__.update(globals())` do, so an object's members may reach
    the functions in it. A lookup in it, a test of what it holds, exec() or
    eval() given it as their namespace, and a call that only shows it
    (`_HARMLESS_CALLS`), don't."""
    if parent is None or isinstance(parent, (ast.Subscript, ast.Attribute, ast.Compare, ast.Expr)):
        return False
    if isinstance(parent, ast.Call) and node in parent.args:
        return _last_name(parent.func) not in _EXEC_CALLS | _HARMLESS_CALLS
    return True


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
    for cell in sorted(cells, key=lambda c: c.order):
        if reader.earlier(cell.tree) and _unread_code(reader, cell.tree, cell, read=read, modules=modules,
                                                      spelled_only=True):
            return cell
        signs = cell.signs
        if (signs.names_builtins or any(_last_name(h) in _GLOBALS_CALLS for h in signs.hints)) and (
            _namespace_poisoned(cell, reader)
        ):
            return cell
    return None


def _namespace_poisoned(cell: _Cell, reader: _Reader) -> bool:
    """Whether a node of `cell` fills the cell's own namespace with what the
    check can't trace (`_poisons_namespace`). Each such fill holds a call
    that gives that namespace (`_cell_namespace`): as the holder an
    assignment's target subscripts, the holder of `.update`, or an argument
    of exec() or eval(), so only the node above each such call, or the one
    above that, can fill it."""
    for call in cell.calls:
        if not _cell_namespace(call, reader):
            continue
        parent = cell.parents.get(call)
        above = cell.parents.get(parent) if isinstance(parent, (ast.Subscript, ast.Attribute, ast.keyword)) else None
        if any(node is not None and _poisons_namespace(node, reader) for node in (parent, above)):
            return True
    return False


def _earlier_function_uses(
    reader: _Reader, root: ast.AST, cell: _Cell, *, ran_only: bool = False,
) -> Iterator[_Callee]:
    """Each place in `root` that calls a function or lambda an earlier cell
    defined, or passes one on as a value: by its name anywhere, or through
    whatever else `_Reader.callees` reads where it is called or passed to a
    call. With `ran_only`, only the calls in code that runs when `root` runs
    (`_nodes_that_run`): a function passed on as a value may never run. One
    a depth limit stopped `callees` short of comes with them, read on past
    the limit (`_Callee.past`)."""
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
                # A function the depth limit stopped short of, beside any it
                # reached, read for whether it may write alone (`_Callee.past`).
                ids = {id(c.func) for c in found}
                more = reader.read_on(mark, lambda: [
                    c for c in reader.callees(node=node, call=call, use=node) if id(c.func) not in ids
                ])
                found = [*found, *(replace(c, past=True) for c in more or ())]
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


@dataclass(frozen=True)
class _ReadInputs:
    """What every read of the checked cell (`_CellRead`) reads it with: the
    cell and the literal code it runs at its top level (`ran`); all the
    literal code exec() and its kin run (`literal_code`); what code the
    check couldn't read may have defined (`defined`, None when none can
    run); the modules an import may load unread; the snapshots; the files
    the agent's own cells wrote (`owned`); and the HTTP rules."""

    current: _Cell
    ran: tuple[_Cell, ...]
    literal_code: _ExecRead
    defined: _Defined | None
    modules: _Modules
    snapshots: tuple[ProjectSnapshot, ...]
    owned: _OwnedFiles
    http_rules: tuple[HttpWriteRule, ...]


class _Scan(NamedTuple):
    """What a scan of a function past a limit found (`_CellRead.scan`):
    whether it writes or runs code the check can't read, and each place it
    calls an earlier function, unread."""

    writes: bool
    calls: tuple[_Callee, ...]


class _CellRead:
    """One read of the checked cell, and of each earlier function it calls,
    with `reader` (`find_cell_writes`): what it would change, into `files`,
    and, unless `files_only`, `sql`, `calls` and `http`; where it runs code
    the check can't read (`unread`); where it reaches what such code may
    have defined (`reaches_defined`), which holds it only when such code
    ran; and, with `files_only`, each function past a limit in which it
    finds a held file change (`past_cap`)."""

    def __init__(self, reader: _Reader, *, inputs: _ReadInputs, files_only: bool) -> None:
        self.reader = reader
        self.inputs = inputs
        self.files_only = files_only
        self.files: list[FileChange] = []
        self.sql: list[SqlWrite] = []
        self.calls: list[CallWrite] = []
        self.http: list[HttpWrite] = []
        self.unread: list[ast.AST] = []
        self.reaches_defined: list[ast.AST] = []
        self.past_cap: list[ast.AST] = []
        # The shortest chain each function was read at for one set of
        # arguments, so a self-recursive function is read once per set, not
        # once per call that reaches it (`_Visit`); and how many times each
        # function the chain already read was read again for other literals.
        self._read_at: dict[_Visit, int] = {}
        self._reread: dict[int, int] = {}
        # The same shortest chains for reads under a guess (`_Callee.guessed`),
        # which hold the cell on less, so they don't stand in for a read
        # without one.
        self._guessed_at: dict[_Visit, int] = {}
        # What each scan past a limit found (`past_the_limit`): by the
        # function's id where the scan asked nothing of the call, else by
        # the visit, so a body is scanned once per set of arguments at most.
        self._scanned: dict[int, _Scan] = {}
        self._scanned_at: dict[_Visit, _Scan] = {}

    def run(self) -> None:
        """Reads the checked cell and the literal code it runs, then each
        earlier function it calls, and each function a literal exec in an
        earlier cell defined, at the call (`read_function`)."""
        reader, inputs = self.reader, self.inputs
        for cell in (inputs.current, *inputs.ran):
            uses = _defined_uses(reader, cell.tree, cell, inputs.defined, inputs.modules)
            self.files.extend(_file_findings(reader, cell, inputs.snapshots, owned=inputs.owned))
            if not self.files_only:
                self.sql.extend(_sql_findings(reader, cell))
                self.calls.extend(_call_findings(reader, cell))
                self.http.extend(_http_findings(reader, cell, rules=inputs.http_rules))
                self.unread.extend(_unread_code(reader, cell.tree, cell, read=inputs.literal_code,
                                                modules=inputs.modules))
                self.reaches_defined.extend(uses.sites)
            for callee in (*_earlier_function_uses(reader, cell.tree, cell), *uses.callees):
                use = callee.use
                report = _Report(line=cell.host_line or use.lineno, call=scrub_output(cell.source(use)))
                self.read_function(callee, report=report, chain=())

    def skips(self, visit: _Visit, chain: tuple[_Visit, ...], *, guessed: bool = False) -> bool:
        """Whether `read_function` leaves `visit` unread at `chain`: the
        chain reads its function already, with the same literals or none;
        or it was read with these arguments at a chain no longer than this,
        under a guess only when this read is under one too (`guessed`)."""
        if any(v.func == visit.func for v in chain) and (
            not visit.literals or any(v.func == visit.func and v.literals == visit.literals for v in chain)
        ):
            return True
        reads = (self._read_at, self._guessed_at) if guessed else (self._read_at,)
        return any((before := read.get(visit)) is not None and before <= len(chain) for read in reads)

    def read_function(
        self, callee: _Callee, *, report: _Report, chain: tuple[_Visit, ...], guessed: bool = False,
    ) -> None:
        """Reads `callee` at its call, then each earlier function it calls
        in turn, unless it `skips` the call. Past `_CALL_DEPTH` calls, or
        past `_REREAD_LIMIT` reads again of a function the chain reads for
        other literals, it is read only for whether it may write or call on
        (`past_the_limit`), which leaves the cell unchecked; so is one found
        only past a depth limit (`_Callee.past`), unless the chain reads it
        already."""
        func = callee.func
        reader, inputs = self.reader, self.inputs
        visit = _Visit.of(callee, reader)
        guessed = guessed or callee.guessed
        again = any(v.func == visit.func for v in chain)
        if self.skips(visit, chain, guessed=guessed) or callee.past and again:
            return
        (self._guessed_at if guessed else self._read_at)[visit] = len(chain)
        if len(chain) >= _CALL_DEPTH or callee.past or self.reread_out(visit, chain):
            if self.files_only:
                if self.past_the_limit(callee, report=report, chain=chain, guessed=guessed):
                    self.past_cap.append(func)
            elif not self.unread and self.past_the_limit(callee, report=report, chain=chain, guessed=guessed):
                self.unread.append(func)
            return
        if again:
            self._reread[visit.func] = self._reread.get(visit.func, 0) + 1
        home = reader.cell_of(func)
        with reader.calling(func, callee.call, bound=callee.bound):
            # The function's own code, then each cell of literal code it
            # passes to exec(), read whole.
            for code in inputs.literal_code.with_ran(home, func):
                cell, root = code.cell, func if code.cell is home else None
                uses = _defined_uses(reader, code.root, cell, inputs.defined, inputs.modules)
                self.files.extend(_file_findings(reader, cell, inputs.snapshots, root=root, report=report,
                                                 owned=inputs.owned))
                if not self.files_only:
                    self.sql.extend(_sql_findings(reader, cell, root=root, report=report))
                    self.calls.extend(_call_findings(reader, cell, root=root, report=report))
                    self.http.extend(_http_findings(reader, cell, rules=inputs.http_rules, root=root, report=report))
                    self.unread.extend(_unread_code(reader, code.root, cell, read=inputs.literal_code,
                                                    modules=inputs.modules))
                    self.reaches_defined.extend(uses.sites)
                for inner in (*_earlier_function_uses(reader, code.root, cell), *uses.callees):
                    self.read_function(inner, report=report, chain=(*chain, visit), guessed=guessed)

    def past_the_limit(self, callee: _Callee, *, report: _Report, chain: tuple[_Visit, ...], guessed: bool) -> bool:
        """Whether `callee`, a function past a limit (`read_function`),
        writes, runs code the check can't read, or calls another earlier
        function the chain hasn't read: it is read for that alone, with no
        call followed, and nothing it finds goes on the card. With
        `files_only`, only a held file change counts. A chain that started
        at a method of every class that defines its name, on a receiver the
        check can't place (`_Callee.guessed`), holds only on what it finds:
        such a chain may be none the cell runs."""
        func = callee.func
        reader = self.reader
        visit = _Visit.of(callee, reader)
        # The chain an inner call would be read at.
        inner_chain = (*chain, visit)
        with reader.calling(func, callee.call, bound=callee.bound):
            scan = self._scanned.get(id(func)) or self._scanned_at.get(visit)
            if scan is None:
                asks = reader.call_asks
                with reader.afresh():
                    scan = self.scan(callee, report=report)
                # A scan that asked nothing of the call reads alike at any call.
                if reader.call_asks == asks:
                    self._scanned[id(func)] = scan
                else:
                    self._scanned_at[visit] = scan
            if scan.writes:
                return True
            return not guessed and any(
                c.func is not func and not self.skips(inner := _Visit.of(c, reader), inner_chain)
                and not self.reread_out(inner, inner_chain) for c in scan.calls
            )

    def scan(self, callee: _Callee, *, report: _Report) -> _Scan:
        """What `past_the_limit` reads of `callee` at its call: whether it
        writes or runs code the check can't read, and the earlier functions
        it calls, none of them followed. With `files_only`, only a held file
        change counts, and no call does."""
        func = callee.func
        reader, inputs = self.reader, self.inputs
        home = reader.cell_of(func)
        calls: list[_Callee] = []
        for code in inputs.literal_code.with_ran(home, func):
            cell, root = code.cell, func if code.cell is home else None
            changes = _file_findings(reader, cell, inputs.snapshots, root=root, report=report, owned=inputs.owned)
            if any(change.held for change in changes):
                return _Scan(writes=True, calls=())
            if self.files_only:
                continue
            if (_sql_findings(reader, cell, root=root, report=report)
                    or _call_findings(reader, cell, root=root, report=report)
                    or _http_findings(reader, cell, rules=inputs.http_rules, root=root, report=report)
                    or _unread_code(reader, code.root, cell, read=inputs.literal_code, modules=inputs.modules)):
                return _Scan(writes=True, calls=())
            uses = _defined_uses(reader, code.root, cell, inputs.defined, inputs.modules)
            self.reaches_defined.extend(uses.sites)
            calls += [*_earlier_function_uses(reader, code.root, cell), *uses.callees]
        return _Scan(writes=False, calls=tuple(calls))

    def reread_out(self, visit: _Visit, chain: tuple[_Visit, ...]) -> bool:
        """Whether the chain reads `visit`'s function already and has read
        it again for other literals `_REREAD_LIMIT` times, as one of a pair
        of functions that call each other may be: a call to it is read only
        for whether it may write where the chain reaches it
        (`read_function`), so it is no deeper function."""
        return any(v.func == visit.func for v in chain) and self._reread.get(visit.func, 0) >= _REREAD_LIMIT


def find_cell_writes(
    *,
    code: str,
    earlier: Sequence[EarlierCell],
    snapshots: Sequence[ProjectSnapshot],
    http_rules: Sequence[HttpWriteRule] = (),
) -> CellWrites:
    """What `code` would change, read against the conversation's `earlier` cells.

    A function or lambda an earlier cell defined is read again at each place
    this cell calls it or passes it on, with its parameters read from that
    call, and what it would change is reported at that place. A request that
    matches one of `http_rules` counts as a write, whatever its method.
    An earlier cell that stopped on an error (`EarlierCell.errored`) makes
    no file the agent's own with its writes (`_OwnedFiles`), and a path read
    through a name it binds may hold any value it gives the name, or the
    one before it: each file those name is held too (`_added_holds`), up to
    `_PATH_LIMIT` paths for one value. The names it surely bound before the
    line it stopped at (`EarlierCell.stopped_at`) are names read code binds
    (`_Reader.bound_before_stop`). One that doesn't compile ran none of its
    code (`_Cell.compiles`), as one stopped at its first line.

    Literal code the cell, or an earlier function it calls, hands exec(),
    a runner of text or `python -c` is read as one more cell, and so is a
    literal exec in an earlier cell, so a call to what it defined is read
    where this cell makes it (`_exec_cells`). The result is `unchecked` when
    the cell runs code the check can't read (`_unread_code`); when it calls
    a name, namespace or module that such code may have filled and such
    code ran in this cell or an earlier one (`_defined_uses`,
    `_first_unread`); when the last of `earlier` ended its process
    (`EarlierCell.ended_process`), so the pad starts a new one before this
    cell, once a cell wrote a startup hook a new process runs
    (`_Modules.hooks_may_run`); or when a depth limit stopped a read where a
    write may lie past it, a function past `_CALL_DEPTH` calls among them
    (`_Reader.stopped`). Whatever else it read stays on the card. A reset
    of the pad clears its cells, so a hook a cell wrote before one is
    outside what this reads.

    A cell that doesn't parse returns nothing to hold: the runner compiles it
    the same way and refuses it before any of it runs. After a cell that
    ended its process, the pad starts a new one before it compiles the
    cell, so it is read as a cell that runs no code of its own.
    """
    try:
        tree = _parse(code)
    except (SyntaxError, ValueError):
        if earlier and earlier[-1].ended_process:
            return find_cell_writes(code="", earlier=earlier, snapshots=snapshots, http_rules=http_rules)
        return CellWrites()
    cells: list[_Cell] = []
    errored: set[int] = set()
    stopped_at: dict[int, int] = {}
    for given in earlier:
        parsed = _EARLIER_CELLS.get(code=given.code, index=len(cells))
        if parsed is None:
            continue
        if given.errored:
            errored.add(parsed.index)
            line = given.stopped_at
            if line is None and not parsed.compiles:
                # Python refused it before any of it ran: it bound nothing.
                line = NOTHING_KEPT
            if line is not None:
                stopped_at[parsed.index] = line
        cells.append(parsed)
    current = _Cell(tree, code, len(cells))
    cells.append(current)
    project = next((s for s in snapshots if s.label == "project"), None)
    project_root = project.root if project is not None else None
    reader = _Reader(cells, current=current, project_root=project_root, stopped_early=frozenset(errored),
                     stopped_at=stopped_at)
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
        _exec_cells(reader, with_literal, ran_only=True) if with_literal else _ExecRead.empty()
    )

    def with_cells(old: _Reader, read: _ExecRead, *, tainted: frozenset[str] = frozenset()) -> _Reader:
        """A reader of `old`'s cells and what `read` read, with the names
        `tainted` beside those `old` reads (`_Reader.tainted_contexts`),
        stopped where `old` stopped and with what `old` read the cells make;
        `old` itself when neither adds anything."""
        if not read.cells and not tainted:
            return old
        new = _Reader([*old.cells, *read.cells], current=current, project_root=project_root,
                      stopped_early=frozenset(errored), stopped_at=stopped_at,
                      tainted_contexts=old.tainted_contexts | tainted)
        new.stopped = old.stopped
        new.made = old.made
        return new

    reader = with_cells(reader, earlier_exec)
    # Literal code an errored cell runs may not have run either: it stops
    # where its host stops.
    errored.update(c.index for c in earlier_exec.cells if c.host in errored)
    exec_read = _exec_cells(reader, [current])
    reader = with_cells(reader, exec_read)
    literal_code = _ExecRead.joined([earlier_exec, exec_read])
    ran = exec_read.of(current.tree)
    if any(not s.complete for s in snapshots):
        reader.made = _made(reader=reader)
    owned = _OwnedFiles(earlier_cells, snapshots, errored=frozenset(errored))
    conversation = [*earlier_cells, current]
    everything = [*conversation, *literal_code.cells]
    while True:
        modules = _Modules(reader, snapshots)
        # When the conversation may run code the check can't read, what such
        # code could have defined, so a call that reaches it holds the cell;
        # None when it can't.
        may_run_unread = bool(with_code) or _runs_code(current, reader) or modules.any_unread(conversation)
        # Whether such code ran in this cell or an earlier one: only then does
        # a call that reaches what it may have defined hold the cell, and only
        # then is what it tainted read.
        first_unread = _first_unread(reader, everything, modules, literal_code) if may_run_unread else None
        defined = (
            _build_defined(reader, everything, modules, literal_code, errored=frozenset(errored),
                           taint_from=first_unread)
            if may_run_unread else None
        )
        if modules.stored:
            # A module a literal exec filled, stored in `sys.modules`: an
            # import of it ran code the check didn't read through that
            # import, which may come before the first unread code found
            # without the stores.
            stored_first = _first_unread(reader, everything, modules, literal_code)
            if stored_first is not None and (first_unread is None or stored_first.order < first_unread.order):
                first_unread = stored_first
                defined = _build_defined(reader, everything, modules, literal_code, errored=frozenset(errored),
                                         taint_from=first_unread)
        # A with whose context holds what such code made may swallow an error
        # its body raises: read again with it as one that may.
        more = _tainted_contexts(reader, defined)
        if not more:
            break
        reader = with_cells(reader, _ExecRead.empty(), tainted=more)
    inputs = _ReadInputs(
        current=current, ran=tuple(ran), literal_code=literal_code, defined=defined, modules=modules,
        snapshots=tuple(snapshots), owned=owned, http_rules=tuple(http_rules),
    )
    read = _CellRead(reader, inputs=inputs, files_only=False)
    read.run()
    files = read.files
    # Where the cell runs code the check can't read (`_unread_code`).
    unread = read.unread
    if _errored_names_read(reader.cells, errored=frozenset(errored)):
        # Read again with each errored cell as maybe stopped anywhere, for
        # the files only that reading holds. Nothing else is taken from it:
        # a wider set of paths may hold less somewhere, never more. Past
        # `_PATH_LIMIT` paths for one value it gives up, and the first read
        # stands alone.
        wide = _Reader(reader.cells, current=current, project_root=project_root, errored=frozenset(errored),
                       tainted_contexts=reader.tainted_contexts)
        wide.made = reader.made
        maybe = _CellRead(wide, inputs=inputs, files_only=True)
        maybe.run()
        if not wide.past_limit:
            files += _added_holds(found=files, maybe=maybe.files)
            unread += maybe.past_cap
    # A call that reaches what code the check couldn't read may have defined
    # holds the cell only when such code ran.
    if first_unread is not None:
        unread += read.reaches_defined
    if earlier and earlier[-1].ended_process and modules.hooks_may_run:
        # The pad starts a new process before this cell runs, and that start
        # runs the startup hooks a cell wrote, which the check doesn't read.
        unread.append(current.tree)
    return CellWrites(
        sql=tuple(sorted(read.sql, key=lambda w: w.line)),
        calls=tuple(sorted(read.calls, key=lambda c: c.line)),
        files=tuple(files),
        http=tuple(sorted(read.http, key=lambda w: w.line)),
        # A depth limit that may have stopped a read short of a write
        # (`_Reader.stopped`) leaves the cell unchecked too.
        unchecked=bool(unread) or reader.stopped,
    )
