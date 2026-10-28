"""Data files an html-app page is built from.

A gathering or data-loop cell writes the data the page needs to
`<artifact>/.anton-data/<id>.json`. The pipeline, not the model, puts each
file into the page's `<script type="application/json" data-anton-data="<id>">`
block after the page is written, so the page shows the data that was read
instead of rows a model retyped or made up.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anton.core.artifacts.internal_files import DATA_DIRNAME
from anton.core.artifacts.render_check import MAX_PAGE_BYTES
from anton.core.artifacts.snapshot import DirSnapshot, diff_snapshots
from anton.core.backends.sandbox import cell_sandbox_of
from anton.utils.nofollow import opened_dir_nofollow, read_bytes_nofollow, unlink_matching_nofollow

from .discovery.notes import source_key, string_list
from .state import settled_artifact_type

if TYPE_CHECKING:
    from anton.core.backends.sandbox import CellSandbox

ID_PATTERN = re.compile(r"^[1-9][0-9]{0,5}\Z")
_FILE_NAME = re.compile(r"^([1-9][0-9]{0,5})\.json\Z")
# JSON's escape for "<": inside a JSON string it reads back as "<", and outside
# strings "<" cannot occur, so replacing every "<" keeps the JSON exact while
# a "</script>" in the data can no longer close the block.
_LT_ESCAPE = chr(92) + "u003c"
_WRITE_HINT = (
    'Write it with df.to_json(path, orient="records", date_format="iso") or '
    "json.dump(obj, f, default=str, allow_nan=False)."
)
DATA_NOT_SAVED_WARNING = (
    "data read by code in the gathering step was not saved to a data file, "
    "so the page may not show all of it"
)


@dataclass
class DataFile:
    id: str
    sources: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "sources": list(self.sources), "description": self.description}

    @classmethod
    def from_dict(cls, raw: object) -> "DataFile | None":
        if not isinstance(raw, dict):
            return None
        file_id = str(raw.get("id") or "").strip()
        if not ID_PATTERN.match(file_id):
            return None
        return cls(
            id=file_id,
            sources=string_list(raw.get("sources")),
            description=str(raw.get("description") or "").strip(),
        )


def sandbox_of(session: object) -> "CellSandbox | None":
    return cell_sandbox_of(getattr(session, "_scratchpads", None))


def rel_name(file_id: str) -> str:
    return f"{DATA_DIRNAME}/{file_id}.json"


def _path(artifact_path: Path, file_id: str) -> Path:
    return Path(artifact_path) / rel_name(file_id)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _read(artifact_path: Path, file_id: str, cell_sandbox: "CellSandbox | None") -> bytes:
    path = _path(artifact_path, file_id)
    if cell_sandbox is not None and not cell_sandbox.allows_read(path):
        raise OSError(errno.EACCES, "outside the folders agent code may read", str(path))
    return read_bytes_nofollow(artifact_path, path, limit=MAX_PAGE_BYTES)


def _parse(raw: bytes) -> dict | list:
    value = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    if not isinstance(value, (dict, list)):
        raise ValueError("the top level must be a JSON object or array")
    return value


def load(artifact_path: Path, file_id: str, *, cell_sandbox: "CellSandbox | None" = None) -> dict | list:
    """The parsed file. Raises OSError when it cannot be read and ValueError
    when it is not a JSON object or array."""
    return _parse(_read(artifact_path, file_id, cell_sandbox))


def serialize(value: object) -> str:
    """The form the page carries: compact JSON with every "<" escaped."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return text.replace("<", _LT_ESCAPE)


@dataclass(frozen=True)
class _Content:
    size: int  # UTF-8 bytes of the form the page carries
    structure: str  # `_describe` lines, cut where no summary can show more


_INSPECTED: "OrderedDict[bytes, _Content]" = OrderedDict()
_INSPECTED_MAX = 32
_INSPECTED_LOCK = threading.Lock()


def _inspect(artifact_path: Path, file_id: str, cell_sandbox: "CellSandbox | None") -> _Content:
    """What the checks and summaries need from a file, kept by its content:
    a run asks about the same files at every step, and parsing megabytes of
    JSON each time holds up the event loop. Raises as `load` does, and with
    UnicodeEncodeError for a string the page cannot carry."""
    raw = _read(artifact_path, file_id, cell_sandbox)
    key = hashlib.blake2b(raw, digest_size=16).digest()
    with _INSPECTED_LOCK:
        known = _INSPECTED.get(key)
        if known is not None:
            _INSPECTED.move_to_end(key)
            return known
    value = _parse(raw)
    content = _Content(
        size=len(serialize(value).encode("utf-8")),
        structure="\n".join(_describe(value, "data", 0))[:SUMMARY_CHARS + 1],
    )
    with _INSPECTED_LOCK:
        _INSPECTED[key] = content
        while len(_INSPECTED) > _INSPECTED_MAX:
            _INSPECTED.popitem(last=False)
    return content


def _stat(artifact_path: Path, file_id: str) -> tuple[int, int] | None:
    try:
        st = os.lstat(_path(artifact_path, file_id))
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def validate(
    artifact_path: Path,
    file_id: str,
    *,
    cell_sandbox: "CellSandbox | None" = None,
    protected: DirSnapshot | None = None,
) -> str | None:
    """Why the file cannot be used, or None. `protected` is `snapshot` taken
    when the run started: a file it lists must still be as it was then."""
    if not ID_PATTERN.match(file_id):
        return f"`{file_id}` is not a data file id: use a positive number such as 1."
    name = rel_name(file_id)
    current = _stat(artifact_path, file_id)
    # A missing file is reported as missing below, not as changed.
    if protected and file_id in protected and current is not None and current != protected[file_id]:
        return (
            f"{name} holds data saved before this call and was changed. Save new data "
            f"under id {next_free_id(artifact_path)} and up instead."
        )
    try:
        # Encoding too: a lone surrogate parses but cannot be written to the page.
        _inspect(artifact_path, file_id, cell_sandbox)
    except FileNotFoundError:
        return f"{name} does not exist. Write it in a scratchpad cell first."
    except OSError as exc:
        if exc.errno == errno.EFBIG:
            return (
                f"{name} is larger than the {MAX_PAGE_BYTES // (1024 * 1024)} MB page limit: "
                "save fewer rows or fields."
            )
        return f"{name} could not be read: {exc.strerror or exc}."
    except (ValueError, RecursionError) as exc:
        return f"{name} is not valid JSON for a page ({exc}). {_WRITE_HINT}"
    return None


def check_total_size(
    artifact_path: Path, files: list[DataFile], *, cell_sandbox: "CellSandbox | None" = None
) -> str | None:
    """An early stop for data that cannot fit a page. Passing it does not
    promise the page fits: markup and embedded attachments come on top."""
    total = 0
    for item in files:
        try:
            total += _inspect(artifact_path, item.id, cell_sandbox).size
        except (OSError, ValueError):
            continue
    if total > MAX_PAGE_BYTES:
        return (
            f"The data files hold {total / (1024 * 1024):.1f} MB, over the "
            f"{MAX_PAGE_BYTES // (1024 * 1024)} MB page limit: save fewer rows or fields, "
            "or aggregates."
        )
    return None


def parse_declared(raw: object) -> list[DataFile]:
    """The `data_files` the model declared. The schema is a hint, so an id may
    arrive as a number and `sources` as one string. An id listed twice is one
    file: its entries merge, so it is checked and counted once."""
    if not isinstance(raw, list):
        return []
    parsed: dict[str, DataFile] = {}
    for entry in raw:
        if not isinstance(entry, dict) or entry.get("id") is None:
            continue
        file_id = str(entry.get("id")).strip()
        sources = string_list(entry.get("sources"))
        description = str(entry.get("description") or "").strip()
        known = parsed.get(file_id)
        if known is None:
            parsed[file_id] = DataFile(id=file_id, sources=sources, description=description)
            continue
        known.sources += [s for s in sources if s not in known.sources]
        known.description = known.description or description
    return list(parsed.values())


def check_declared(
    artifact_path: Path,
    declared: list[DataFile],
    data_sources: list[str],
    *,
    cell_sandbox: "CellSandbox | None" = None,
    protected: DirSnapshot | None = None,
) -> tuple[list[DataFile], list[str]]:
    """The declared files that can be used, and one error line per file that
    cannot. A file must name the `data_sources` it covers."""
    keys = {source_key(s) for s in data_sources}
    valid: list[DataFile] = []
    errors: list[str] = []
    for item in declared:
        error = validate(artifact_path, item.id, cell_sandbox=cell_sandbox, protected=protected)
        if error is None and not item.sources:
            error = "`sources` is empty: name the `data_sources` this file covers."
        if error is None:
            unknown = [s for s in item.sources if source_key(s) not in keys]
            if unknown:
                error = (
                    f"{', '.join(unknown)} not in `data_sources`: name each source exactly as "
                    "listed there."
                )
        if error is None:
            valid.append(item)
        else:
            errors.append(f"id {item.id}: {error}")
    if valid:
        size_error = check_total_size(artifact_path, valid, cell_sandbox=cell_sandbox)
        if size_error:
            return [], [size_error]
    return valid, errors


def state_check_declared(
    state: object, declared: list[DataFile], data_sources: list[str]
) -> tuple[list[DataFile], list[str]]:
    """`check_declared` for a run: its artifact, its sandbox and the files the
    call found on disk."""
    if not declared:
        return [], []
    return check_declared(
        state.artifact_path, declared, data_sources,
        cell_sandbox=sandbox_of(state.session), protected=state.data_snapshot,
    )


def covered_sources(files: list[DataFile]) -> set[str]:
    return {source_key(s) for item in files for s in item.sources}


def merge(current: list[DataFile], new: list[DataFile], declared: list[str]) -> list[DataFile]:
    """`new` over `current`: a listed id takes its new record, and a file in
    hand stays while one of its sources is still declared, so a call that
    lists no files keeps the ones saved before it."""
    keys = {source_key(s) for s in declared}
    new_ids = {f.id for f in new}
    return [
        f for f in current
        if f.id not in new_ids and any(source_key(s) in keys for s in f.sources)
    ] + new


def mark_data_not_saved(state: object) -> None:
    state.data_not_saved = True
    if DATA_NOT_SAVED_WARNING not in state.verify_warnings:
        state.verify_warnings.append(DATA_NOT_SAVED_WARNING)


def clear_data_not_saved(state: object) -> None:
    """Drop the "data not saved" mark and its warning once the data files
    cover every declared source: the page is then built from saved data."""
    if not getattr(state, "data_not_saved", False) or not state.data_files:
        return
    covered = covered_sources(state.data_files)
    if all(source_key(s) in covered for s in state.declared_sources):
        state.data_not_saved = False
        while DATA_NOT_SAVED_WARNING in state.verify_warnings:
            state.verify_warnings.remove(DATA_NOT_SAVED_WARNING)


def records(raw: list | None) -> list[DataFile]:
    """The data file records `discovery.json` holds, malformed ones left out."""
    return [f for f in (DataFile.from_dict(r) for r in raw or []) if f is not None]


def restorable(
    artifact_path: Path, files: list[DataFile], *, cell_sandbox: "CellSandbox | None" = None
) -> list[DataFile]:
    """The records whose files are still there and valid."""
    return [f for f in files if validate(artifact_path, f.id, cell_sandbox=cell_sandbox) is None]


def snapshot(artifact_path: Path) -> DirSnapshot:
    """id -> (mtime_ns, size) of each regular data file. A folder that is a
    symlink counts as empty. The folder is read through its descriptor, so a
    link swapped in after the check is not followed either."""
    folder = Path(artifact_path) / DATA_DIRNAME
    found: DirSnapshot = {}
    try:
        with opened_dir_nofollow(artifact_path, (DATA_DIRNAME,)) as dir_fd:
            with os.scandir(folder if dir_fd is None else dir_fd) as entries:
                for entry in entries:
                    match = _FILE_NAME.match(entry.name)
                    if match is None:
                        continue
                    try:
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    found[match.group(1)] = (st.st_mtime_ns, st.st_size)
    except OSError:
        pass
    return found


def written_since(artifact_path: Path, before: DirSnapshot) -> list[str]:
    return sorted(diff_snapshots(before, snapshot(artifact_path)), key=int)


def next_free_id(artifact_path: Path) -> str:
    ids = [int(i) for i in snapshot(artifact_path)]
    return str(max(ids, default=0) + 1)


def prune(artifact_path: Path, keep: set[str]) -> None:
    drop = tuple(f"{i}.json" for i in snapshot(artifact_path) if i not in keep)
    if not drop:
        return
    try:
        unlink_matching_nofollow(artifact_path, Path(artifact_path) / DATA_DIRNAME, patterns=drop)
    except OSError:
        pass


def save_hint(artifact_path: Path) -> str:
    folder = Path(artifact_path) / DATA_DIRNAME
    return (
        f"Data files folder: {folder}/ — save data as <id>.json there; "
        f"the first free id is {next_free_id(artifact_path)}.\n"
    )


BLOCK_ATTR = "data-anton-data"
READ_CALL = """JSON.parse(document.querySelector('script[data-anton-data="<id>"]').textContent)"""
DATA_FILES_HEADER = "## Data files"
SAVED_DATA_FILES_HEADER = "## Saved data files"
SUMMARY_CHARS = 2000
_SAMPLE_RECORDS = 3
_MAX_DISTINCT = 20
_MAX_DEPTH = 3
_MAX_KEYS = 30

_DATA_FILES_LEAD = (
    "The pipeline puts each file into the page's "
    '`<script type="application/json" data-anton-data="<id>"></script>` block; read it with '
    f"`{READ_CALL}`. This summary is complete: do not print the files in a cell or read them."
)
_SAVED_DATA_FILES_LEAD = (
    "Data saved by the previous build of this artifact. To use a file again, list its id in "
    "`data_files` without writing it; a file nobody lists is deleted once this build succeeds."
)


def _type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _describe(value: object, path: str, depth: int) -> list[str]:
    indent = "  " * depth
    if isinstance(value, dict):
        keys = list(value)
        lines = [f"{indent}- {path}: object with keys {json.dumps(keys[:_MAX_KEYS], ensure_ascii=False)}"]
        if depth + 1 < _MAX_DEPTH:
            for key in keys[:_MAX_KEYS]:
                if isinstance(value[key], (dict, list)):
                    child = f"{path}[{json.dumps(key, ensure_ascii=False)}]"
                    lines.extend(_describe(value[key], child, depth + 1))
        return lines
    if isinstance(value, list):
        records = [v for v in value if isinstance(v, dict)]
        if value and len(records) == len(value):
            fields = list(dict.fromkeys(k for r in records for k in r))
            typed = []
            for key in fields:
                names = sorted({_type_name(r[key]) for r in records if key in r})
                typed.append(f"{key} ({'/'.join(names)})")
            lines = [
                f"{indent}- {path}: array of {len(value)} records",
                f"{indent}  fields: {', '.join(typed)}",
            ]
            for key in fields:
                values = [r.get(key) for r in records if key in r]
                if all(isinstance(v, (str, bool)) or v is None for v in values):
                    distinct = list(dict.fromkeys(values))
                    # Only fields whose values repeat: a status, not an id.
                    if len(distinct) <= _MAX_DISTINCT and len(distinct) < len(values):
                        lines.append(f"{indent}  {key} values: {json.dumps(distinct, ensure_ascii=False)}")
            for record in records[:_SAMPLE_RECORDS]:
                lines.append(f"{indent}  sample: {json.dumps(record, ensure_ascii=False)}")
            return lines
        first = json.dumps(value[:_SAMPLE_RECORDS], ensure_ascii=False)
        return [f"{indent}- {path}: array of {len(value)} items, first: {first}"]
    return [f"{indent}- {path} = {json.dumps(value, ensure_ascii=False)}"]


def summarize(
    artifact_path: Path, files: list[DataFile], *, cell_sandbox: "CellSandbox | None" = None
) -> str:
    """One bounded entry per file: what it covers and the shape of its data,
    with real sample records, so later steps never need the file itself."""
    entries: list[str] = []
    for item in files:
        head = [
            f"### id {item.id} — {rel_name(item.id)}",
            f"Sources: {', '.join(item.sources) or '(not named)'}",
        ]
        if item.description:
            head.append(f"Description: {item.description}")
        try:
            content = _inspect(artifact_path, item.id, cell_sandbox)
        except (OSError, ValueError) as exc:
            entries.append("\n".join(head + [f"(unreadable: {exc})"]))
            continue
        head.append(f"Size: {content.size:,} bytes")
        body = "\n".join(head + ["Structure:", content.structure])
        if len(body) > SUMMARY_CHARS:
            body = body[:SUMMARY_CHARS] + " …"
        entries.append(body)
    return "\n\n".join(entries)


def files_section(
    artifact_path: Path,
    files: list[DataFile],
    *,
    cell_sandbox: "CellSandbox | None" = None,
    saved: bool = False,
) -> str:
    """The data files this build uses, or with `saved` the previous build's
    files a new request may reuse."""
    if not files:
        return ""
    header, lead = (
        (SAVED_DATA_FILES_HEADER, _SAVED_DATA_FILES_LEAD) if saved
        else (DATA_FILES_HEADER, _DATA_FILES_LEAD)
    )
    return f"{header}\n{lead}\n\n" + summarize(artifact_path, files, cell_sandbox=cell_sandbox)


# A run that moved from html-app to a fullstack type keeps its files, but its
# generators have no data blocks to fill.
def state_files_section(state: object, *, saved: bool = False) -> str:
    files = getattr(state, "previous_data_files" if saved else "data_files", None) or []
    if not files or settled_artifact_type(state) != "html-app":
        return ""
    return files_section(
        state.artifact_path, files, cell_sandbox=sandbox_of(state.session), saved=saved
    )


_SCRIPT_CLOSE = re.compile(r"</script\s*>", re.IGNORECASE)
_SCRIPT_ELEMENT = re.compile(r"<script\b[^>]*>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class Block:
    file_id: str
    attrs: dict[str, str | None]
    tag_start: int
    body_start: int
    body_end: int
    tag_end: int


class _BlockFinder(HTMLParser):
    """Opening `<script data-anton-data>` tags, by offset. A parser rather than
    a regex: a block written inside a comment or an attribute value is text,
    not a block."""

    def __init__(self, html: str) -> None:
        super().__init__(convert_charrefs=False)
        self._line_starts = [0] + [m.end() for m in re.finditer("\n", html)]
        self.opens: list[tuple[dict, int, int]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "script":
            return
        found = {name.lower(): value for name, value in attrs}
        if BLOCK_ATTR not in found:
            return
        line, column = self.getpos()
        start = self._line_starts[line - 1] + column
        self.opens.append((found, start, start + len(self.get_starttag_text() or "")))


_BLOCK_ATTR_TEXT = re.compile(re.escape(BLOCK_ATTR), re.IGNORECASE)


def find_blocks(html: str) -> list[Block]:
    """The page's data blocks in document order."""
    # Most pages have none, and an embedded asset can make one tens of MB.
    if not _BLOCK_ATTR_TEXT.search(html):
        return []
    finder = _BlockFinder(html)
    finder.feed(html)
    finder.close()
    blocks: list[Block] = []
    for attrs, tag_start, body_start in finder.opens:
        close = _SCRIPT_CLOSE.search(html, body_start)
        if close is None:
            continue
        blocks.append(Block(
            file_id=str(attrs.get(BLOCK_ATTR) or "").strip(),
            attrs=attrs,
            tag_start=tag_start,
            body_start=body_start,
            body_end=close.start(),
            tag_end=close.end(),
        ))
    return blocks


def _splice(html: str, spans: list[tuple[int, int, str]]) -> str:
    """`html` with each (start, end) span, in document order, replaced."""
    parts, cursor = [], 0
    for start, end, text in spans:
        parts += [html[cursor:start], text]
        cursor = end
    parts.append(html[cursor:])
    return "".join(parts)


def without_data(html: str) -> str:
    """`html` with every data block emptied, for checks that read the page's
    own code as text: data is not code, and a retry cannot change it."""
    blocks = find_blocks(html)
    if not blocks:
        return html
    return _splice(html, [(b.body_start, b.body_end, "") for b in blocks])


def reads_blocks(html: str, blocks: list[Block]) -> bool:
    """Whether a script other than the blocks names the block attribute."""
    rest = _splice(html, [(b.tag_start, b.tag_end, "") for b in blocks])
    return any(BLOCK_ATTR in m.group(1) for m in _SCRIPT_ELEMENT.finditer(rest))


@dataclass(frozen=True)
class FillResult:
    html: str
    fatal: str | None = None


def fill_page(
    html: str,
    artifact_path: Path,
    files: list[DataFile],
    *,
    cell_sandbox: "CellSandbox | None" = None,
) -> FillResult:
    """`html` with each known block's body replaced by its file, on a line of
    its own. Blocks with an unknown id, and a repeat of an id after its first
    block, stay as they are for the verifier: a copy the page must drop anyway
    does not count against the size limit."""
    known = {item.id for item in files}
    blocks: list[Block] = []
    for block in find_blocks(html):
        if block.file_id in known:
            known.discard(block.file_id)
            blocks.append(block)
    if not blocks:
        return FillResult(html)
    payload: dict[str, str] = {}
    for block in blocks:
        try:
            payload[block.file_id] = serialize(load(artifact_path, block.file_id, cell_sandbox=cell_sandbox))
            # A lone surrogate parses but cannot be encoded into the page.
            payload[block.file_id].encode("utf-8")
        except (OSError, ValueError, RecursionError) as exc:
            return FillResult(html, fatal=(
                f"The data file {rel_name(block.file_id)} could not be read ({exc}), so the page "
                "cannot show its data. Call generate_artifact again so the data is saved anew."
            ))
    filled = _splice(html, [(b.body_start, b.body_end, "\n" + payload[b.file_id] + "\n") for b in blocks])
    size = len(filled.encode("utf-8"))
    if size > MAX_PAGE_BYTES:
        return FillResult(html, fatal=(
            f"The page with its data is {size / (1024 * 1024):.1f} MB, over the "
            f"{MAX_PAGE_BYTES // (1024 * 1024)} MB page limit. Save fewer rows or fields, or "
            "aggregates, and call generate_artifact again."
        ))
    return FillResult(filled)
