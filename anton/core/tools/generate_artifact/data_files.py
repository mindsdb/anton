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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anton.core.artifacts.internal_files import DATA_DIRNAME
from anton.core.artifacts.render_check import MAX_PAGE_BYTES
from anton.core.backends.sandbox import cell_sandbox_of
from anton.utils.nofollow import read_bytes_nofollow

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
        load(artifact_path, file_id, cell_sandbox=cell_sandbox)
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
    """id -> (mtime_ns, size) of each regular data file."""
    folder = Path(artifact_path) / DATA_DIRNAME
    found: dict[str, tuple[int, int]] = {}
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return found
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
    return found


def written_since(artifact_path: Path, before: dict[str, tuple[int, int]]) -> list[str]:
    now = snapshot(artifact_path)
    return sorted((i for i, sig in now.items() if before.get(i) != sig), key=int)


def next_free_id(artifact_path: Path) -> str:
    ids = [int(i) for i in snapshot(artifact_path)]
    return str(max(ids, default=0) + 1)


def prune(artifact_path: Path, keep: set[str]) -> None:
    for file_id in snapshot(artifact_path):
        if file_id not in keep:
            try:
                _path(artifact_path, file_id).unlink()
            except OSError:
                pass


def save_hint(artifact_path: Path) -> str:
    folder = Path(artifact_path) / DATA_DIRNAME
    return (
        f"Data files folder: {folder}/ — save data as <id>.json there; "
        f"the first free id is {next_free_id(artifact_path)}.\n"
    )
