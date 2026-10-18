"""Find the steps in a scratchpad cell that change connected data or existing files.

The host holds a cell that has any of them until the user approves it
(`hold_unapproved_write` in `anton.core.utils.scratchpad`). This module only
reads the cell's code; it never runs it.

What it finds:

- **SQL writes**, in a conversation with a connected data source: a statement
  whose first word is a write verb and whose clause shape matches it (UPDATE ...
  SET, DELETE FROM, INSERT INTO, CREATE TABLE, and so on), written as a string
  literal, an f-string or a ``+`` chain, that reaches something that could run
  it. A statement only printed, logged or written to a file doesn't count, and
  neither does one run on an in-process DuckDB or SQLite handle. ``to_sql``
  into a database that isn't in-process counts too.
- **Deletes, renames and moves** of a file or folder that was in the project
  folder (artifacts included) or the shared folder when the turn began. A
  target the code picks when it runs, from a root the check can't place, counts
  as well, because the check can't rule it out.

What it can't find: a statement or path with no literal behind it, a write
with no SQL text other than ``to_sql``, a write through an HTTP or other
non-SQL API, and code in a module or script the cell imports or runs. An
overwrite of an existing file's contents is reported as ``overwrite`` and isn't
held.
"""

from __future__ import annotations

import ast
import fnmatch
import os
import posixpath
import re
import shlex
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from anton.core.backends.wire import heal_surrogate_source
from anton.utils.datasources import scrub_output

__all__ = [
    "CellWrites",
    "FileChange",
    "ProjectSnapshot",
    "SqlWrite",
    "find_cell_writes",
    "names_connected_data",
    "snapshot_project_files",
]

_STATEMENT_CHARS = 600
_CALL_CHARS = 200

# Folders recorded as entries but not walked: big, and never the agent's
# answer. A path under one still counts as one that existed.
_UNWALKED = frozenset({".git", "node_modules", ".venv", "venv", "__pycache__"})
_WORKSPACE_ENV = "ANTON_SCRATCHPAD_WORKSPACE_PATH"


class SqlWrite(BaseModel):
    """One statement that changes connected data."""

    model_config = ConfigDict(frozen=True)

    statement: str
    values: str = ""
    line: int


class FileChange(BaseModel):
    """One delete, rename, move or overwrite of an existing file."""

    model_config = ConfigDict(frozen=True)

    action: Literal["delete", "rename", "move", "overwrite"]
    # Snapshot paths, the shared folder's prefixed with "shared folder: ".
    # Empty when the code picks the path when it runs.
    paths: tuple[str, ...] = ()
    whole_project: bool = False
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


class ProjectSnapshot(BaseModel):
    """The paths under one root when the turn began, relative and POSIX."""

    model_config = ConfigDict(frozen=True)

    root: Path
    label: Literal["project", "shared"]
    files: frozenset[str] = frozenset()
    dirs: frozenset[str] = frozenset()
    unwalked: frozenset[str] = frozenset()
    taken_at: float
    # Past the entry cap the walk stops, and a path counts as existing when its
    # modification time is older than the turn.
    complete: bool = True
    # The walk couldn't read the root or a folder under it: every path under
    # the root counts as existing.
    failed: bool = False

    def is_empty(self) -> bool:
        return not (self.files or self.dirs) and self.complete and not self.failed

    def holds(self, *, rel: str) -> bool:
        if self.failed or rel in self.files or rel in self.dirs:
            return True
        if any(rel == u or rel.startswith(u + "/") for u in self.unwalked):
            return True
        if not self.complete:
            try:
                return (self.root / rel).lstat().st_mtime < self.taken_at
            except OSError:
                return False
        return False

    def is_file(self, *, rel: str) -> bool:
        return rel in self.files

    def matching(self, *, pattern: str) -> list[str]:
        """Snapshot paths a glob-style pattern names, `*` within one folder and
        `**` across folders, as glob reads them."""
        if self.failed:
            return [pattern]
        return sorted(p for p in (*self.files, *self.dirs) if glob_match(p, pattern))


def glob_match(path: str, pattern: str) -> bool:
    return _match_parts(path.split("/"), pattern.split("/"))


def _match_parts(parts: list[str], patterns: list[str]) -> bool:
    if not patterns:
        return not parts
    head, rest = patterns[0], patterns[1:]
    if head == "**":
        return any(_match_parts(parts[i:], rest) for i in range(len(parts) + 1))
    return bool(parts) and fnmatch.fnmatchcase(parts[0], head) and _match_parts(parts[1:], rest)


def snapshot_project_files(
    *,
    root: Path,
    label: Literal["project", "shared"],
    taken_at: float,
    cap: int = 20_000,
) -> ProjectSnapshot:
    """Record the files and folders under `root`.

    In the project, ``.anton`` holds the host's own folders (scratchpad venvs,
    memory, saved state), so only ``.anton/artifacts``, where artifacts live, is
    walked. A folder the walk can't read marks the snapshot failed, which makes
    every path under the root count as existing: an unread folder must not let
    a delete through.
    """
    real = Path(os.path.realpath(root))
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
            if label == "project" and rel_dir == ".anton":
                dirnames[:] = [d for d in dirnames if d == "artifacts"]
                filenames = []
            walk = []
            for name in dirnames:
                rel = posixpath.join(rel_dir, name) if rel_dir else name
                dirs.add(rel)
                if name in _UNWALKED:
                    unwalked.add(rel)
                else:
                    walk.append(name)
            dirnames[:] = walk
            for name in filenames:
                files.add(posixpath.join(rel_dir, name) if rel_dir else name)
            if len(files) + len(dirs) >= cap:
                complete = False
                break
    return ProjectSnapshot(
        root=real,
        label=label,
        files=frozenset(files),
        dirs=frozenset(dirs),
        unwalked=frozenset(unwalked),
        taken_at=taken_at,
        complete=complete,
        failed=failed,
    )


# ── SQL ───────────────────────────────────────────────────────────────────

_TOKEN = re.compile(
    r"""(--[^\n]*)|(/\*.*?\*/)|('(?:[^']|'')*')|("(?:[^"]|"")*")|(`[^`]*`)"""
    r"""|(\$(\w*)\$.*?\$\7\$)|(;)|([A-Za-z_][A-Za-z_0-9]*)|([()])|(\S)""",
    re.S,
)
_OBJECTS = frozenset({
    "TABLE", "VIEW", "INDEX", "SCHEMA", "DATABASE", "FUNCTION", "PROCEDURE",
    "SEQUENCE", "TRIGGER", "TYPE", "EXTENSION", "ROLE", "USER",
})
_CREATE_MODIFIERS = frozenset({
    "OR", "REPLACE", "UNIQUE", "MATERIALIZED", "EXTERNAL", "TRANSIENT", "SECURE",
    "RECURSIVE", "UNLOGGED", "VIRTUAL", "LOCAL", "GLOBAL", "CLUSTERED", "NONCLUSTERED",
})
_TEMP = frozenset({"TEMP", "TEMPORARY", "VOLATILE"})
_DML = frozenset({"INSERT", "UPDATE", "DELETE", "MERGE"})
_TRUNCATE_TAIL = frozenset({"RESTART", "CONTINUE", "IDENTITY", "CASCADE", "RESTRICT"})
# Stands in for an f-string placeholder or a name in a `+` chain.
_HOLE = "_x_"


def _statements(text: str) -> Iterator[tuple[list[str], int, int]]:
    """Each statement's words (upper case, comments and quoted text dropped,
    parentheses kept), with its start and end offsets in `text`."""
    words: list[str] = []
    start = end = None
    for m in _TOKEN.finditer(text):
        if m.group(8):
            if words:
                yield words, start, end
            words, start, end = [], None, None
            continue
        token = m.group(9) or m.group(10)
        if token is None:
            if m.group(1) or m.group(2):
                continue
            # A quoted literal or any other character: part of the statement,
            # but not a word.
            if start is None:
                start = m.start()
            end = m.end()
            continue
        if start is None:
            start = m.start()
        end = m.end()
        words.append(token.upper())
    if words:
        yield words, start, end


def _top_level(words: list[str]) -> list[str]:
    depth, top = 0, []
    for w in words:
        if w == "(":
            depth += 1
        elif w == ")":
            depth = max(depth - 1, 0)
        elif depth == 0:
            top.append(w)
    return top


def _statement_writes(words: list[str]) -> bool:
    top = _top_level(words)
    if not top:
        return False
    head, rest = top[0], top[1:]
    if head == "UPDATE":
        return "SET" in rest
    if head == "DELETE":
        return "FROM" in rest[:2] or "WHERE" in rest
    if head == "INSERT":
        return any(w in rest for w in ("INTO", "VALUES", "SELECT"))
    if head == "MERGE":
        return "INTO" in rest[:1] or "USING" in rest
    if head in ("REPLACE", "UPSERT"):
        return "INTO" in rest[:1]
    if head == "TRUNCATE":
        names = [w for w in rest if w not in ("TABLE", "ONLY")]
        return len(names) >= 1 and all(w in _TRUNCATE_TAIL for w in names[1:])
    if head == "CREATE":
        i = 0
        while i < len(rest) and rest[i] in _CREATE_MODIFIERS:
            i += 1
        if i < len(rest) and rest[i] in _TEMP:
            return False
        if i + 1 < len(rest) and rest[i] in ("LOCAL", "GLOBAL") and rest[i + 1] in _TEMP:
            return False
        return i < len(rest) and rest[i] in _OBJECTS
    if head in ("ALTER", "DROP"):
        i = 1 if rest[:1] == ["MATERIALIZED"] else 0
        return i < len(rest) and rest[i] in _OBJECTS
    if head == "GRANT":
        return "ON" in rest or "TO" in rest
    if head == "REVOKE":
        return "ON" in rest or "FROM" in rest
    if head == "WITH":
        bodies = any(a == "(" and b in _DML for a, b in zip(words, words[1:]))
        main = next((w for w in rest if w in _DML or w == "SELECT"), None)
        return bodies or main in _DML
    if head == "SELECT":
        if "INTO" not in rest:
            return False
        into = rest.index("INTO")
        target = rest[into + 1] if into + 1 < len(rest) else ""
        before_from = "FROM" not in rest or into < rest.index("FROM")
        return before_from and target not in _TEMP
    if head == "COPY":
        return "FROM" in rest
    return False


def write_statement(text: str) -> str | None:
    """The first statement in `text` that changes data, as written, or None."""
    for words, start, end in _statements(text):
        if _statement_writes(words):
            return text[start:end]
    return None


_DS_TEXT = re.compile(r"\bDS_")
_ATTACH = re.compile(r"""(?:^|[;'"])\s*ATTACH\s""", re.I | re.M)


def names_connected_data(code: str) -> bool:
    """Whether a cell names a connection's DS_ credentials, by name or prefix."""
    return bool(_DS_TEXT.search(code))


# ── The reader ────────────────────────────────────────────────────────────

_EXECUTE = frozenset({
    "execute", "executemany", "executescript", "exec_driver_sql", "sql", "query",
})
_READ_SQL = frozenset({"read_sql", "read_sql_query"})
_HARMLESS_CALLS = frozenset({"print", "pprint", "display", "repr", "len", "input"})
_HARMLESS_METHODS = frozenset({
    "write", "write_text", "writelines", "debug", "info", "warning", "warn",
    "error", "exception", "critical", "log", "startswith", "endswith", "find",
    "count", "index", "split", "splitlines", "lower", "upper",
})
_LOGGER_NAMES = frozenset({"logger", "log", "logging", "_log", "_logger"})
_SQL_WRAPPERS = frozenset({
    "text", "sqlalchemy.text", "SQL", "sql.SQL", "psycopg2.sql.SQL", "psycopg.sql.SQL",
    "dedent", "textwrap.dedent", "str",
})
_STR_METHODS = frozenset({"format", "strip", "lstrip", "rstrip", "replace", "upper", "lower"})

_DELETE_FUNCS = frozenset({"os.remove", "os.unlink", "os.rmdir", "os.removedirs", "shutil.rmtree"})
_MOVE_FUNCS = {"os.rename": "rename", "os.renames": "rename", "os.replace": "rename", "shutil.move": "move"}
_COPY_FUNCS = frozenset({"shutil.copy", "shutil.copy2", "shutil.copyfile"})
_SHELL_FUNCS = frozenset({
    "subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output",
    "subprocess.Popen", "os.system", "os.popen",
})
_SHELL_DELETE = frozenset({"rm", "unlink", "rmdir", "shred"})
_WRITER_METHODS = frozenset({
    "write_text", "write_bytes", "to_csv", "to_excel", "to_parquet", "to_json", "savefig",
})
_PATH_TYPES = frozenset({"pathlib.Path", "pathlib.PurePath", "pathlib.PosixPath", "pathlib.PurePosixPath"})
_PATH_MAKERS = frozenset({"glob", "rglob", "iterdir", "resolve", "absolute", "with_suffix",
                          "with_name", "with_stem", "joinpath", "expanduser"})
_ITER_WRAPPERS = frozenset({"next", "sorted", "list", "max", "min", "reversed", "iter", "tuple"})


class _Unknown:
    """A path the code picks when it runs, from a root the check can't place."""


_UNKNOWN = _Unknown()
_TEMP_PATH = "\x00temp"
# Stands in for a literal-command operand that isn't a literal.
_NOT_LITERAL = "\x00"
_PARAM_DEPTH = 5


class _Cell:
    """One parsed cell: its tree, parents, bindings and source lines."""

    def __init__(self, tree: ast.Module, code: str, index: int) -> None:
        self.tree = tree
        self.index = index
        self.lines = code.splitlines()
        self.parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node
        # name -> [(line, value node, how)], `how` saying what the value is:
        # "value" (the name holds it), "element" (an element of it),
        # "context" (a with-statement's resource), "param" (a parameter of
        # the function in `value`).
        self.bindings: dict[str, list[tuple[int, ast.AST, str]]] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    self._bind(t, node.value, node.lineno, "value")
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                self._bind(node.target, node.iter, node.lineno, "element")
            elif isinstance(node, ast.comprehension):
                self._bind(node.target, node.iter, getattr(node.iter, "lineno", 0), "element")
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                self._bind(node.optional_vars, node.context_expr, node.context_expr.lineno, "context")
            elif isinstance(node, ast.NamedExpr):
                self._bind(node.target, node.value, node.lineno, "value")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                    self.bindings.setdefault(a.arg, []).append((node.lineno, node, "param"))

    def _bind(self, target: ast.AST, value: ast.AST, line: int, how: str) -> None:
        if isinstance(target, ast.Name):
            self.bindings.setdefault(target.id, []).append((line, value, how))
        elif isinstance(target, (ast.Tuple, ast.List)):
            # An unpacked tuple: each name gets a part the check can't place,
            # except a two-name unpack of enumerate(), whose second is the element.
            for elt in target.elts:
                inner = elt.value if isinstance(elt, ast.Starred) else elt
                if isinstance(inner, ast.Name):
                    self.bindings.setdefault(inner.id, []).append((line, value, "unpacked"))

    def source(self, node: ast.AST) -> str:
        line = getattr(node, "lineno", 0)
        text = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        return text[:_CALL_CHARS]


class _Reader:
    """Reads one cell against the conversation's earlier cells."""

    def __init__(self, cells: list[_Cell]) -> None:
        self.cells = cells
        self.current = cells[-1]
        self.imports: dict[str, str] = {}
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.chdir = False
        for cell in cells:
            for node in ast.walk(cell.tree):
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
        for cell in cells:
            for node in ast.walk(cell.tree):
                if isinstance(node, ast.Call) and self.dotted(node.func) == "os.chdir":
                    self.chdir = True
        self.attaches = any(
            isinstance(n, ast.Constant) and isinstance(n.value, str) and _ATTACH.search(n.value)
            for cell in cells for n in ast.walk(cell.tree)
        )

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

    def bindings_of(self, name: str, *, cell: _Cell, line: int) -> list[tuple[ast.AST, str, _Cell]]:
        """What `name` holds at `line`: this cell's bindings before it, else the
        latest earlier cell's."""
        here = [(v, how, cell) for (ln, v, how) in cell.bindings.get(name, ()) if ln <= line]
        if here:
            return here
        for earlier in reversed(self.cells[: cell.index]):
            found = earlier.bindings.get(name)
            if found:
                return [(v, how, earlier) for (_ln, v, how) in found]
        return []

    def call_sites(self, func: ast.FunctionDef | ast.AsyncFunctionDef, param: str) -> list[tuple[ast.AST, _Cell]]:
        """The argument passed as `param` at each call of `func` in the pad's cells."""
        names = [a.arg for a in (*func.args.posonlyargs, *func.args.args)]
        out = []
        for cell in self.cells:
            for node in ast.walk(cell.tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == func.name):
                    continue
                arg = next((k.value for k in node.keywords if k.arg == param), None)
                if arg is None and param in names and names.index(param) < len(node.args):
                    arg = node.args[names.index(param)]
                if arg is not None:
                    out.append((arg, cell))
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
        found = self.bindings_of(node.id, cell=cell, line=node.lineno)
        if not found:
            return _UNKNOWN
        out: list[str] = []
        for value, how, where in found:
            if how == "param":
                # One level only: a parameter is read from the function's call
                # sites, and a parameter met there is unknown.
                sites = self.call_sites(value, node.id) if depth < _PARAM_DEPTH else []
                if not sites:
                    return _UNKNOWN
                for arg, site_cell in sites:
                    got = self.paths(arg, cell=site_cell, depth=_PARAM_DEPTH)
                    if isinstance(got, _Unknown):
                        return got
                    out += got
                continue
            if how == "unpacked":
                got = self._unpacked_paths(value, cell=where, depth=depth)
            else:
                got = self.paths(value, cell=where, depth=depth + 1)
            if isinstance(got, _Unknown):
                return got
            out += got
        return out

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
            return any(
                how in ("value", "element") and self.is_path(v, cell=where, depth=depth + 1)
                for v, how, where in self.bindings_of(node.id, cell=cell, line=node.lineno)
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

    def is_local_handle(self, node: ast.AST, *, cell: _Cell, depth: int = 0) -> bool:
        """Whether `node` is an in-process DuckDB or SQLite handle, which
        changes no connected data unless the pad ATTACHes another database."""
        if self.attaches or depth > 6:
            return False
        if isinstance(node, ast.Name):
            if self.imports.get(node.id, node.id) == "duckdb":
                return True
            found = self.bindings_of(node.id, cell=cell, line=node.lineno)
            return bool(found) and all(
                how in ("value", "context") and self.is_local_handle(v, cell=where, depth=depth + 1)
                for v, how, where in found
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
                return not target.value.lower().startswith(("md:", "motherduck:"))
            if name in ("sqlalchemy.create_engine", "create_engine") and node.args:
                url = node.args[0]
                return (
                    isinstance(url, ast.Constant) and isinstance(url.value, str)
                    and url.value.startswith(("sqlite:", "duckdb:"))
                )
            if isinstance(node.func, ast.Attribute) and node.func.attr in (
                "cursor", "connect", "begin", "raw_connection",
            ):
                return self.is_local_handle(node.func.value, cell=cell, depth=depth + 1)
            if isinstance(node.func, ast.Name) and node.func.id in self.functions:
                returns = [n.value for n in ast.walk(self.functions[node.func.id])
                           if isinstance(n, ast.Return) and n.value is not None]
                return bool(returns) and all(
                    self.is_local_handle(r, cell=cell, depth=depth + 1) for r in returns
                )
        return False

    def runs_locally(self, func: ast.FunctionDef | ast.AsyncFunctionDef, *, cell: _Cell) -> bool:
        """Whether a function the pad defines runs every statement it gets on a
        local handle."""
        calls = [n for n in ast.walk(func) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr in _EXECUTE]
        return bool(calls) and all(
            self.is_local_handle(c.func.value, cell=cell) for c in calls
        )


def _join(a: str, b: str) -> str:
    if b.startswith("/"):
        return b
    if a in ("", "."):
        return b
    return a.rstrip("/") + "/" + b


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


# ── SQL findings ──────────────────────────────────────────────────────────

def _sql_text(node: ast.AST, reader: _Reader, cell: _Cell, depth: int = 0) -> str | None:
    """The text a string-building expression makes, a placeholder for each
    part chosen at run time; None when it isn't one."""
    if depth > 40:
        return None
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        return "".join(
            str(v.value) if isinstance(v, ast.Constant) else _HOLE for v in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        # Flattened with a stack, so a long `+` chain can't exhaust recursion.
        parts: list[str] = []
        stack: list[ast.AST] = [node]
        while stack:
            item = stack.pop()
            if isinstance(item, ast.BinOp) and isinstance(item.op, ast.Add):
                stack.append(item.right)
                stack.append(item.left)
            elif isinstance(item, ast.BinOp) and isinstance(item.op, ast.Mod):
                left = _sql_text(item.left, reader, cell, depth + 1)
                parts.append(left if left is not None else _HOLE)
            else:
                text = _sql_text(item, reader, cell, depth + 1) if not isinstance(item, ast.Name) else None
                parts.append(text if text is not None else _HOLE)
        joined = "".join(parts)
        return joined if joined.replace(_HOLE, "").strip() else None
    if isinstance(node, ast.Call):
        name = reader.dotted(node.func)
        if name in _SQL_WRAPPERS and node.args:
            return _sql_text(node.args[0], reader, cell, depth + 1)
        if isinstance(node.func, ast.Attribute) and node.func.attr in _STR_METHODS:
            return _sql_text(node.func.value, reader, cell, depth + 1)
    return None


def _is_harmless_call(call: ast.Call, reader: _Reader) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in _HARMLESS_CALLS
    if isinstance(func, ast.Attribute):
        if func.attr in _HARMLESS_METHODS:
            return True
        root = func.value
        while isinstance(root, ast.Attribute):
            root = root.value
        return isinstance(root, ast.Name) and (
            root.id in _LOGGER_NAMES or reader.imports.get(root.id) == "logging"
        )
    return False


def _executes_on_local(call: ast.Call, reader: _Reader, cell: _Cell) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in _EXECUTE:
        return reader.is_local_handle(func.value, cell=cell)
    if isinstance(func, ast.Attribute) and func.attr in _READ_SQL:
        con = call.args[1] if len(call.args) > 1 else next(
            (k.value for k in call.keywords if k.arg == "con"), None
        )
        return con is not None and reader.is_local_handle(con, cell=cell)
    if isinstance(func, ast.Name) and func.id in reader.functions:
        return reader.runs_locally(reader.functions[func.id], cell=cell)
    return False


def _reaches_a_runner(node: ast.AST, reader: _Reader, cell: _Cell, depth: int = 0) -> bool:
    """Whether a statement's value can reach code that runs it.

    False only when every use is visibly harmless: a bare expression (a
    docstring), a print, log or file write, a comparison, a name nothing in
    this cell reads, or an execute on a local handle. Anything else counts,
    because the check can't follow it.
    """
    if depth > 6:
        return True
    parent = cell.parents.get(node)
    if parent is None or isinstance(parent, ast.Expr):
        return False
    if isinstance(parent, (ast.JoinedStr, ast.FormattedValue, ast.BinOp, ast.Starred)):
        return _reaches_a_runner(parent, reader, cell, depth + 1)
    if isinstance(parent, ast.Attribute) and parent.attr in _STR_METHODS:
        call = cell.parents.get(parent)
        return _reaches_a_runner(call if call is not None else parent, reader, cell, depth + 1)
    if isinstance(parent, ast.Call):
        if parent.func is node:
            return True
        if reader.dotted(parent.func) in _SQL_WRAPPERS:
            return _reaches_a_runner(parent, reader, cell, depth + 1)
        if _is_harmless_call(parent, reader) or _executes_on_local(parent, reader, cell):
            return False
        return True
    if isinstance(parent, ast.keyword):
        call = cell.parents.get(parent)
        if isinstance(call, ast.Call) and (
            _is_harmless_call(call, reader) or _executes_on_local(call, reader, cell)
        ):
            return False
        return True
    if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
        targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
        if any(not isinstance(t, ast.Name) for t in targets):
            return True
        names = {t.id for t in targets}
        uses = [n for n in ast.walk(cell.tree) if isinstance(n, ast.Name)
                and n.id in names and isinstance(n.ctx, ast.Load)]
        return any(_reaches_a_runner(u, reader, cell, depth + 1) for u in uses)
    if isinstance(parent, (ast.List, ast.Tuple, ast.Set)):
        holder = cell.parents.get(parent)
        if isinstance(holder, (ast.For, ast.AsyncFor)) and holder.iter is parent:
            if not isinstance(holder.target, ast.Name):
                return True
            uses = [n for stmt in holder.body for n in ast.walk(stmt)
                    if isinstance(n, ast.Name) and n.id == holder.target.id
                    and isinstance(n.ctx, ast.Load)]
            return any(_reaches_a_runner(u, reader, cell, depth + 1) for u in uses)
        return _reaches_a_runner(parent, reader, cell, depth + 1)
    if isinstance(parent, (ast.Compare, ast.BoolOp, ast.UnaryOp, ast.If, ast.While, ast.Assert)):
        return False
    return True


def _values_text(call: ast.Call) -> str:
    if len(call.args) > 1 and isinstance(call.args[1], (ast.Tuple, ast.List, ast.Dict)):
        try:
            return scrub_output(ast.unparse(call.args[1]))[:_CALL_CHARS]
        except Exception:
            return ""
    return ""


def _statement(text: str) -> str:
    return scrub_output(" ".join(text.split()))[:_STATEMENT_CHARS]


def _sql_findings(reader: _Reader) -> list[SqlWrite]:
    cell = reader.current
    out: list[SqlWrite] = []
    seen_nodes: set[int] = set()
    for node in ast.walk(cell.tree):
        if id(node) in seen_nodes:
            continue
        parent = cell.parents.get(node)
        # The outermost string-building node only: a part of a `+` chain or an
        # f-string is read with its whole.
        if isinstance(parent, (ast.JoinedStr, ast.FormattedValue)) or (
            isinstance(parent, ast.BinOp) and isinstance(parent.op, (ast.Add, ast.Mod))
            and _sql_text(parent, reader, cell) is not None
        ):
            continue
        if not isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp)):
            continue
        text = _sql_text(node, reader, cell)
        if text is None:
            continue
        statement = write_statement(text)
        if statement is None or not _reaches_a_runner(node, reader, cell):
            continue
        for inner in ast.walk(node):
            seen_nodes.add(id(inner))
        call = parent if isinstance(parent, ast.Call) else None
        out.append(SqlWrite(
            statement=_statement(statement),
            values=_values_text(call) if call is not None else "",
            line=getattr(node, "lineno", 0),
        ))
    # A name an earlier cell bound to a write, run here.
    for node in ast.walk(cell.tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in _EXECUTE and node.args
                and isinstance(node.args[0], ast.Name)):
            continue
        if _executes_on_local(node, reader, cell):
            continue
        for value, how, where in reader.bindings_of(node.args[0].id, cell=cell, line=node.lineno):
            if where is cell or how != "value":
                continue
            text = _sql_text(value, reader, where)
            statement = write_statement(text) if text is not None else None
            if statement is not None:
                out.append(SqlWrite(statement=_statement(statement),
                                    values=_values_text(node), line=node.lineno))
                break
    # DataFrame.to_sql into a database that isn't in-process.
    for node in ast.walk(cell.tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "to_sql"):
            continue
        con = node.args[1] if len(node.args) > 1 else next(
            (k.value for k in node.keywords if k.arg == "con"), None
        )
        if con is not None and reader.is_local_handle(con, cell=cell):
            continue
        table = node.args[0] if node.args else next(
            (k.value for k in node.keywords if k.arg == "name"), None
        )
        label = table.value if isinstance(table, ast.Constant) and isinstance(table.value, str) else "a table"
        out.append(SqlWrite(statement=_statement(f"to_sql into {label}"), line=node.lineno))
    return out


# ── File findings ─────────────────────────────────────────────────────────

class _Targets(BaseModel):
    """Where a path pattern lands: snapshot paths, the whole project, or a
    root the check can't place."""

    model_config = ConfigDict(frozen=True)

    paths: tuple[str, ...] = ()
    whole_project: bool = False
    unknown: bool = False


def _locate(patterns: list[str] | _Unknown, snapshots: Sequence[ProjectSnapshot], *, chdir: bool) -> _Targets:
    if isinstance(patterns, _Unknown):
        return _Targets(unknown=True)
    hits: list[str] = []
    whole = False
    for raw in patterns:
        if raw == _TEMP_PATH:
            continue
        pattern = raw.replace("\\", "/")
        for snap in snapshots:
            rel = _relative(pattern, snap)
            if rel is None:
                continue
            if rel == "" or rel == ".." or rel.startswith("../") and _above(rel):
                if snap.label == "project":
                    whole = True
                continue
            if rel.startswith("../"):
                continue
            prefix = "" if snap.label == "project" else "shared folder: "
            if chdir and not os.path.isabs(pattern) and snap.label == "project":
                # After a chdir, a relative path can be under any folder.
                found = snap.matching(pattern="**/" + rel)
            elif any(ch in rel for ch in "*?["):
                found = snap.matching(pattern=rel)
            else:
                found = [rel] if snap.holds(rel=rel) else []
            hits += [prefix + p for p in found]
    return _Targets(paths=tuple(dict.fromkeys(hits)), whole_project=whole)


def _above(rel: str) -> bool:
    """Whether a relative path made only of `..` parts names the root or above."""
    return all(part in ("..", "") for part in rel.split("/"))


def _relative(pattern: str, snap: ProjectSnapshot) -> str | None:
    """`pattern` relative to the snapshot's root: "" for the root itself, a
    "../" path for above it, None when it lies outside."""
    if os.path.isabs(pattern):
        roots = {str(snap.root), os.path.realpath(snap.root)}
        for root in roots:
            if pattern.rstrip("/") == root:
                return ""
            if pattern.startswith(root + "/"):
                return posixpath.normpath(pattern[len(root) + 1:])
            if root.startswith(pattern.rstrip("/") + "/"):
                return ".."
        return None
    if snap.label != "project":
        return None
    rel = posixpath.normpath(pattern)
    return "" if rel == "." else rel


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


def _shell_operands(call: ast.Call, reader: _Reader) -> list[tuple[str, list[str]]]:
    """(verb, operands) for each rm or mv a literal command runs."""
    arg = call.args[0] if call.args else None
    if isinstance(arg, (ast.List, ast.Tuple)):
        words = [e.value if isinstance(e, ast.Constant) and isinstance(e.value, str) else _NOT_LITERAL
                 for e in arg.elts]
        segments = [words]
    elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        try:
            tokens = shlex.split(arg.value.replace(";", " ; ").replace("\n", " ; "))
        except ValueError:
            tokens = arg.value.split()
        segments, current = [], []
        for t in tokens:
            if t in (";", "&&", "||", "|"):
                segments.append(current)
                current = []
            else:
                current.append(t)
        segments.append(current)
    else:
        return []
    out = []
    for seg in segments:
        if seg[:1] == ["git"] and len(seg) > 1:
            seg = seg[1:]
        if not seg:
            continue
        verb = seg[0].rsplit("/", 1)[-1]
        operands = [t for t in seg[1:] if not t.startswith("-")]
        if verb in _SHELL_DELETE:
            out.append(("delete", operands))
        elif verb == "mv" and len(operands) > 1:
            out.append(("move", operands[:-1]))
    return out


def _file_findings(reader: _Reader, snapshots: Sequence[ProjectSnapshot]) -> list[FileChange]:
    cell = reader.current
    if not snapshots or all(s.is_empty() for s in snapshots):
        return []
    out: list[FileChange] = []
    written_later: set[str] = set()

    def locate(node: ast.AST) -> _Targets:
        return _locate(reader.paths(node, cell=cell), snapshots, chdir=reader.chdir)

    def add(action: str, targets: _Targets, call: ast.AST) -> None:
        if targets.paths or targets.whole_project or targets.unknown:
            out.append(FileChange(
                action=action, paths=targets.paths,
                whole_project=targets.whole_project,
                call=scrub_output(cell.source(call)), line=getattr(call, "lineno", 0),
            ))

    # A literal path this cell writes after deleting it is an overwrite, not a delete.
    for node in ast.walk(cell.tree):
        if isinstance(node, ast.Call):
            target = _overwrite_target(node, reader)
            if target is not None:
                got = reader.paths(target, cell=cell)
                if not isinstance(got, _Unknown):
                    written_later.update(got)

    for node in ast.walk(cell.tree):
        if isinstance(node, ast.Attribute) and reader.dotted(node) in (*_DELETE_FUNCS, *_MOVE_FUNCS):
            parent = cell.parents.get(node)
            if not (isinstance(parent, ast.Call) and parent.func is node):
                # Passed as a value, as in map(os.remove, ...): the target is
                # picked when it runs.
                action = "delete" if reader.dotted(node) in _DELETE_FUNCS else _MOVE_FUNCS[reader.dotted(node)]
                add(action, _Targets(unknown=True), node)
            continue
        if not isinstance(node, ast.Call):
            continue
        name = reader.dotted(node.func)
        method = node.func.attr if isinstance(node.func, ast.Attribute) else ""
        if name in _DELETE_FUNCS:
            target = _target_arg(node, keywords=("path",))
            if target is None:
                continue
            got = reader.paths(target, cell=cell)
            if not isinstance(got, _Unknown) and got and all(p in written_later for p in got):
                add("overwrite", locate(target), node)
            else:
                add("delete", locate(target), node)
        elif name in _MOVE_FUNCS:
            source = _target_arg(node, keywords=("src",))
            if source is not None:
                add(_MOVE_FUNCS[name], locate(source), node)
            _note_overwrite(node, reader, snapshots, out, cell, dest_index=1)
        elif name in _COPY_FUNCS:
            _note_overwrite(node, reader, snapshots, out, cell, dest_index=1)
        elif name in _SHELL_FUNCS:
            for verb, operands in _shell_operands(node, reader):
                if _NOT_LITERAL in operands:
                    add(verb, _Targets(unknown=True), node)
                else:
                    add(verb, _locate(operands, snapshots, chdir=True), node)
        elif method == "unlink" and not node.args:
            if isinstance(node.func.value, (ast.Constant, ast.JoinedStr)):
                continue
            got = reader.paths(node.func.value, cell=cell)
            if not isinstance(got, _Unknown) and got and all(p in written_later for p in got):
                add("overwrite", locate(node.func.value), node)
            else:
                add("delete", locate(node.func.value), node)
        elif method == "rmdir" and reader.is_path(node.func.value, cell=cell):
            add("delete", locate(node.func.value), node)
        elif method in ("rename", "replace") and _is_path_rename(node, reader, cell):
            add("rename", locate(node.func.value), node)
            dest = node.args[0] if node.args else next(
                (k.value for k in node.keywords if k.arg == "target"), None
            )
            if dest is not None:
                _note_overwrite_target(dest, reader, snapshots, out, cell, node)
        else:
            target = _overwrite_target(node, reader)
            if target is not None:
                _note_overwrite_target(target, reader, snapshots, out, cell, node)
    return out


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


def _open_mode_writes_method(call: ast.Call) -> bool:
    mode = call.args[0] if call.args else next(
        (k.value for k in call.keywords if k.arg == "mode"), None
    )
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and any(
        c in mode.value for c in "wax"
    )


def _note_overwrite(call, reader, snapshots, out, cell, *, dest_index: int) -> None:
    dest = call.args[dest_index] if len(call.args) > dest_index else next(
        (k.value for k in call.keywords if k.arg in ("dst", "dest")), None
    )
    if dest is None:
        return
    source = call.args[0] if call.args else None
    got = reader.paths(dest, cell=cell)
    if isinstance(got, _Unknown):
        return
    # A folder destination takes the source's name.
    names = reader.paths(source, cell=cell) if source is not None else _UNKNOWN
    expanded = []
    for d in got:
        for snap in snapshots:
            rel = _relative(d, snap)
            if rel is not None and rel in snap.dirs and not isinstance(names, _Unknown):
                expanded += [_join(d, posixpath.basename(n)) for n in names]
                break
        else:
            expanded.append(d)
    _record_overwrite(expanded, snapshots, out, cell, call)


def _note_overwrite_target(node, reader, snapshots, out, cell, call) -> None:
    got = reader.paths(node, cell=cell)
    if not isinstance(got, _Unknown):
        _record_overwrite(got, snapshots, out, cell, call)


def _record_overwrite(patterns, snapshots, out, cell, call) -> None:
    hits = []
    for pattern in patterns:
        for snap in snapshots:
            rel = _relative(pattern, snap)
            if rel and not rel.startswith("..") and snap.is_file(rel=rel):
                hits.append(rel if snap.label == "project" else "shared folder: " + rel)
    if hits:
        out.append(FileChange(
            action="overwrite", paths=tuple(dict.fromkeys(hits)),
            call=scrub_output(cell.source(call)), line=getattr(call, "lineno", 0),
        ))


# ── Entry point ───────────────────────────────────────────────────────────

def _parse(code: str) -> ast.Module:
    """The cell's tree, from the same healed text the runner compiles.

    Quiet: the agent's own invalid escape sequences would otherwise print a
    SyntaxWarning in the host's log for every check; the runner reports them
    where they belong, in the cell's output.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(heal_surrogate_source(code))


def find_cell_writes(
    *,
    code: str,
    earlier_code: Sequence[str],
    snapshots: Sequence[ProjectSnapshot],
    connected: bool,
) -> CellWrites:
    """What `code` would change, read against the conversation's earlier cells.

    `connected` says the session has a data connection; a cell that names DS_
    credentials, here or earlier, counts as connected too. A cell that doesn't
    parse returns nothing to hold: the runner compiles it the same way and
    refuses it before any of it runs.
    """
    try:
        tree = _parse(code)
    except (SyntaxError, ValueError):
        return CellWrites()
    cells: list[_Cell] = []
    for text in earlier_code:
        try:
            cells.append(_Cell(_parse(text), text, len(cells)))
        except (SyntaxError, ValueError):
            continue
    cells.append(_Cell(tree, code, len(cells)))
    reader = _Reader(cells)
    is_connected = connected or any(names_connected_data(t) for t in (*earlier_code, code))
    sql = _sql_findings(reader) if is_connected else []
    files = _file_findings(reader, snapshots)
    return CellWrites(sql=tuple(sql), files=tuple(files))
