"""Data files an html-app page is built from.

A gathering or data-loop cell writes the data the page needs to
`<artifact>/.anton-data/<id>.json`. The pipeline, not the model, puts each
file into the page's `<script type="application/json" data-anton-data="<id>">`
block after the page is written, so the page shows the data that was read
instead of rows a model retyped or made up.
"""

from __future__ import annotations

import errno
import json
import os
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anton.core.artifacts.internal_files import DATA_DIRNAME
from anton.core.artifacts.render_check import MAX_PAGE_BYTES
from anton.core.backends.sandbox import cell_sandbox_of
from anton.utils.nofollow import opened_dir_nofollow, read_bytes_nofollow, unlink_matching_nofollow

from .discovery.notes import source_key, string_list

if TYPE_CHECKING:
    from anton.core.backends.sandbox import CellSandbox

ID_PATTERN = re.compile(r"^[1-9][0-9]{0,5}$")
_FILE_NAME = re.compile(r"^([1-9][0-9]{0,5})\.json$")
# JSON's escape for "<": inside a JSON string it reads back as "<", and outside
# strings "<" cannot occur, so replacing every "<" keeps the JSON exact while
# a "</script>" in the data can no longer close the block.
_LT_ESCAPE = chr(92) + "u003c"
_WRITE_HINT = (
    'Write it with df.to_json(path, orient="records", date_format="iso") or '
    "json.dump(obj, f, default=str, allow_nan=False)."
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
    return Path(artifact_path) / DATA_DIRNAME / f"{file_id}.json"


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def load(artifact_path: Path, file_id: str, *, cell_sandbox: "CellSandbox | None" = None) -> dict | list:
    """The parsed file. Raises OSError when it cannot be read and ValueError
    when it is not a JSON object or array."""
    path = _path(artifact_path, file_id)
    if cell_sandbox is not None and not cell_sandbox.allows_read(path):
        raise OSError(errno.EACCES, "outside the folders agent code may read", str(path))
    raw = read_bytes_nofollow(artifact_path, path, limit=MAX_PAGE_BYTES)
    value = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    if not isinstance(value, (dict, list)):
        raise ValueError("the top level must be a JSON object or array")
    return value


def serialize(value: object) -> str:
    """The form the page carries: compact JSON with every "<" escaped."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return text.replace("<", _LT_ESCAPE)


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
    protected: dict[str, tuple[int, int]] | None = None,
) -> str | None:
    """Why the file cannot be used, or None. `protected` is `snapshot` taken
    when the run started: a file it lists must still be as it was then."""
    if not ID_PATTERN.match(file_id):
        return f"`{file_id}` is not a data file id: use a positive number such as 1."
    name = rel_name(file_id)
    if protected and file_id in protected and _stat(artifact_path, file_id) != protected[file_id]:
        return (
            f"{name} belongs to data saved by an earlier run and was changed. Save new data "
            f"under id {next_free_id(artifact_path)} and up instead."
        )
    try:
        # Encoding too: a lone surrogate parses but cannot be written to the page.
        serialize(load(artifact_path, file_id, cell_sandbox=cell_sandbox)).encode("utf-8")
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
            total += len(serialize(load(artifact_path, item.id, cell_sandbox=cell_sandbox)).encode("utf-8"))
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
    arrive as a number and `sources` as one string."""
    if not isinstance(raw, list):
        return []
    parsed: list[DataFile] = []
    for entry in raw:
        if not isinstance(entry, dict) or entry.get("id") is None:
            continue
        parsed.append(DataFile(
            id=str(entry.get("id")).strip(),
            sources=string_list(entry.get("sources")),
            description=str(entry.get("description") or "").strip(),
        ))
    return parsed


def check_declared(
    artifact_path: Path,
    declared: list[DataFile],
    data_sources: list[str],
    *,
    cell_sandbox: "CellSandbox | None" = None,
    protected: dict[str, tuple[int, int]] | None = None,
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


def covered_sources(files: list[DataFile]) -> set[str]:
    return {source_key(s) for item in files for s in item.sources}


def restorable(
    artifact_path: Path, raw: list, *, cell_sandbox: "CellSandbox | None" = None
) -> list[DataFile]:
    """Saved records whose files are still there and valid."""
    files = [f for f in (DataFile.from_dict(r) for r in raw or []) if f is not None]
    return [f for f in files if validate(artifact_path, f.id, cell_sandbox=cell_sandbox) is None]


def snapshot(artifact_path: Path) -> dict[str, tuple[int, int]]:
    """id -> (mtime_ns, size) of each regular data file. A folder that is a
    symlink counts as empty. The folder is read through its descriptor, so a
    link swapped in after the check is not followed either."""
    folder = Path(artifact_path) / DATA_DIRNAME
    found: dict[str, tuple[int, int]] = {}
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
        return found
    return found


def written_since(artifact_path: Path, before: dict[str, tuple[int, int]]) -> list[str]:
    now = snapshot(artifact_path)
    return sorted((i for i, sig in now.items() if before.get(i) != sig), key=int)


def next_free_id(artifact_path: Path) -> str:
    ids = [int(i) for i in snapshot(artifact_path)]
    return str(max(ids, default=0) + 1)


def prune(artifact_path: Path, keep: set[str]) -> None:
    folder = Path(artifact_path) / DATA_DIRNAME
    for file_id in snapshot(artifact_path):
        if file_id not in keep:
            try:
                unlink_matching_nofollow(artifact_path, folder, patterns=(f"{file_id}.json",))
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
            value = load(artifact_path, item.id, cell_sandbox=cell_sandbox)
            size = len(serialize(value).encode('utf-8'))
        except (OSError, ValueError) as exc:
            entries.append("\n".join(head + [f"(unreadable: {exc})"]))
            continue
        head.append(f"Size: {size:,} bytes")
        body = "\n".join(head + ["Structure:"] + _describe(value, "data", 0))
        if len(body) > SUMMARY_CHARS:
            body = body[:SUMMARY_CHARS] + " …"
        entries.append(body)
    return "\n\n".join(entries)


def files_section(
    artifact_path: Path, files: list[DataFile], *, cell_sandbox: "CellSandbox | None" = None
) -> str:
    if not files:
        return ""
    return (
        f"{DATA_FILES_HEADER}\n{_DATA_FILES_LEAD}\n\n"
        + summarize(artifact_path, files, cell_sandbox=cell_sandbox)
    )


def saved_files_section(
    artifact_path: Path, files: list[DataFile], *, cell_sandbox: "CellSandbox | None" = None
) -> str:
    if not files:
        return ""
    return (
        f"{SAVED_DATA_FILES_HEADER}\n{_SAVED_DATA_FILES_LEAD}\n\n"
        + summarize(artifact_path, files, cell_sandbox=cell_sandbox)
    )


def state_files_section(state: object) -> str:
    files = getattr(state, "data_files", None) or []
    if not files:
        return ""
    return files_section(state.artifact_path, files, cell_sandbox=sandbox_of(state.session))


def state_saved_files_section(state: object) -> str:
    files = getattr(state, "previous_data_files", None) or []
    if not files:
        return ""
    return saved_files_section(state.artifact_path, files, cell_sandbox=sandbox_of(state.session))


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


def find_blocks(html: str) -> list[Block]:
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


def reads_blocks(html: str, blocks: list[Block]) -> bool:
    """Whether a script other than the blocks names the block attribute."""
    rest, cursor = [], 0
    for block in sorted(blocks, key=lambda b: b.tag_start):
        rest.append(html[cursor:block.tag_start])
        cursor = block.tag_end
    rest.append(html[cursor:])
    return any(BLOCK_ATTR in m.group(1) for m in _SCRIPT_ELEMENT.finditer("".join(rest)))


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
    its own. Blocks with an unknown id stay as they are for the verifier."""
    known = {item.id for item in files}
    blocks = [b for b in find_blocks(html) if b.file_id in known]
    if not blocks:
        return FillResult(html)
    payload: dict[str, str] = {}
    for file_id in {b.file_id for b in blocks}:
        try:
            payload[file_id] = serialize(load(artifact_path, file_id, cell_sandbox=cell_sandbox))
            # A lone surrogate parses but cannot be encoded into the page.
            payload[file_id].encode("utf-8")
        except (OSError, ValueError, RecursionError) as exc:
            return FillResult(html, fatal=(
                f"The data file {rel_name(file_id)} could not be read ({exc}), so the page "
                "cannot show its data. Call generate_artifact again so the data is saved anew."
            ))
    parts, cursor = [], 0
    for block in sorted(blocks, key=lambda b: b.body_start):
        parts.append(html[cursor:block.body_start])
        parts.append("\n" + payload[block.file_id] + "\n")
        cursor = block.body_end
    parts.append(html[cursor:])
    filled = "".join(parts)
    size = len(filled.encode("utf-8"))
    if size > MAX_PAGE_BYTES:
        return FillResult(html, fatal=(
            f"The page with its data is {size / (1024 * 1024):.1f} MB, over the "
            f"{MAX_PAGE_BYTES // (1024 * 1024)} MB page limit. Save fewer rows or fields, or "
            "aggregates, and call generate_artifact again."
        ))
    return FillResult(filled)
