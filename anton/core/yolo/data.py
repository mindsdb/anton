"""Generated data files and the schemas that describe them.

A data file is one line:

    window.ANTON_DATA_prices = [{"date": "2026-08-31", "price": 41.22}];

That shape is chosen so it can be two things at once. A browser loads it
with a plain `<script>` tag, which is the only thing that works both
locally (an artifact opened from disk cannot `fetch()` a sibling file —
browsers treat `file://` as cross-origin) and published over HTTP. And
because the value is a JSON literal behind a fixed prefix, anything can
read it back by stripping the wrapper — no JavaScript engine required.

That second property is what makes the schema sidecar trustworthy.
Rather than asking whoever wrote the data to also describe it — and
hoping the description stays true — the schema is **derived from the
bytes on disk**, and can be re-derived at any time. A sidecar cannot
drift from its data, because nothing is ever asked to keep them in step.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from anton.core.yolo.workspace import (
    DATA_SUFFIX,
    Workspace,
    WorkspaceError,
    is_generated_data,
    is_schema,
    schema_for,
)

__all__ = [
    "DataError",
    "derive_schema",
    "global_name",
    "read_data",
    "reconcile",
    "write_data",
]

# Every generated global is prefixed, so a page can tell data apart from
# whatever else it has hung on window.
GLOBAL_PREFIX = "ANTON_DATA_"

# The wrapper, and the pattern that undoes it.
_ASSIGNMENT = re.compile(
    r"^\s*window\.(" + GLOBAL_PREFIX + r"\w+)\s*=\s*(.*?);?\s*$", re.DOTALL
)

# How many rows to inspect when working out the shape. Enough to catch a
# column that is null in the first row and a number in the second;
# cheap enough to run on every reconcile.
SAMPLE_ROWS = 200


class DataError(Exception):
    """A data file that could not be read back."""


def global_name(name: str) -> str:
    """The global a data file defines, from its base name."""
    return GLOBAL_PREFIX + re.sub(r"\W", "_", name)


def write_data(
    workspace: Workspace, name: str, rows: Any, notes: str = ""
) -> tuple[str, str]:
    """Write `<name>.data.js` and its derived `<name>.schema.json`.

    Returns both paths. `notes` is the one thing that cannot be derived —
    units, gaps, timezones — so it is the only thing asked for.
    """
    data_path = f"{name}{DATA_SUFFIX}"
    payload = json.dumps(rows, ensure_ascii=False, default=str)
    workspace.write(data_path, f"window.{global_name(name)} = {payload};\n")
    schema_path = _write_schema(workspace, data_path, rows, notes)
    return data_path, schema_path


def read_data(workspace: Workspace, data_path: str) -> Any:
    """Read a data file back, without running any JavaScript."""
    try:
        text = workspace.read(data_path)
    except WorkspaceError as error:
        raise DataError(str(error)) from error
    match = _ASSIGNMENT.match(text)
    if not match:
        raise DataError(
            f"{data_path} is not in the expected "
            f"`window.{GLOBAL_PREFIX}<name> = <json>;` shape"
        )
    try:
        return json.loads(match.group(2))
    except json.JSONDecodeError as error:
        raise DataError(f"{data_path} does not hold valid JSON: {error}") from error


def derive_schema(name: str, rows: Any, notes: str = "") -> dict:
    """Work out what is in the data by looking at it.

    Never by asking. A hand-written `{"price": "number"}` over rows that
    actually hold `"1,234.00"` strings makes whoever reads the schema
    write code against a lie — and that failure renders as a broken chart
    rather than an error.
    """
    schema: dict[str, Any] = {
        "global": global_name(name),
        "file": f"{name}{DATA_SUFFIX}",
        "shape": _shape(rows),
        "generated": {"at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    }
    if isinstance(rows, list):
        schema["rows"] = len(rows)
        sample = rows[:SAMPLE_ROWS]
        if sample and all(isinstance(row, dict) for row in sample):
            schema["fields"] = _fields(sample)
        elif sample:
            schema["items"] = {"type": _type_of(sample[0])}
    elif isinstance(rows, dict):
        schema["fields"] = _fields([rows])
    if notes.strip():
        schema["notes"] = notes.strip()
    return schema


def reconcile(workspace: Workspace) -> list[str]:
    """Make every data file's sidecar match the data, and report what changed.

    This is the whole answer to sidecar drift: rather than requiring one
    blessed way to produce data and hoping nothing else ever writes one,
    the sidecars are recomputed from what is actually on disk. Run it
    after anything that may have touched the folder.

    A `notes` line already in a sidecar is preserved — it is the one field
    nobody can derive, and regenerating over it would throw away the only
    part a human wrote.
    """
    report: list[str] = []
    data_files = [info.path for info in workspace.files() if is_generated_data(info.path)]

    for data_path in data_files:
        name = data_path[: -len(DATA_SUFFIX)]
        sidecar = schema_for(data_path)
        try:
            rows = read_data(workspace, data_path)
        except DataError as error:
            report.append(f"{data_path}: cannot read it back — {error}")
            continue

        existing = _load_schema(workspace, sidecar)
        fresh = derive_schema(name, rows, notes=str(existing.get("notes", "")))
        if not existing:
            _save(workspace, sidecar, fresh)
            report.append(f"{sidecar}: written (it was missing)")
        elif _differs(existing, fresh):
            _save(workspace, sidecar, fresh)
            report.append(f"{sidecar}: refreshed (it no longer matched the data)")

    for info in workspace.files():
        if is_schema(info.path) and not workspace.exists(_data_for(info.path)):
            report.append(f"{info.path}: describes a data file that is not there")
    return report


# ── internals ───────────────────────────────────────────────────────────


def _write_schema(workspace: Workspace, data_path: str, rows: Any, notes: str) -> str:
    schema_path = schema_for(data_path)
    _save(workspace, schema_path, derive_schema(data_path[: -len(DATA_SUFFIX)], rows, notes))
    return schema_path


def _save(workspace: Workspace, path: str, schema: dict) -> None:
    workspace.write(path, json.dumps(schema, indent=2, ensure_ascii=False) + "\n")


def _load_schema(workspace: Workspace, path: str) -> dict:
    if not workspace.exists(path):
        return {}
    try:
        loaded = json.loads(workspace.read(path))
    except (WorkspaceError, json.JSONDecodeError):
        return {}  # unreadable is as good as missing: it gets rewritten
    return loaded if isinstance(loaded, dict) else {}


def _differs(existing: dict, fresh: dict) -> bool:
    """Compare what was derived, ignoring what was not.

    `generated` is a timestamp and `notes` is human text; neither says
    anything about whether the schema still describes the data.
    """
    keys = {"global", "file", "shape", "rows", "fields", "items"}
    return {key: existing.get(key) for key in keys} != {
        key: fresh.get(key) for key in keys
    }


def _data_for(schema_path: str) -> str:
    return schema_path[: -len(".schema.json")] + DATA_SUFFIX


def _shape(rows: Any) -> str:
    if isinstance(rows, list):
        if rows and all(isinstance(row, dict) for row in rows[:SAMPLE_ROWS]):
            return "array<object>"
        return "array"
    if isinstance(rows, dict):
        return "object"
    return _type_of(rows)


def _fields(sample: list[dict]) -> dict:
    """Describe each column from the values actually present."""
    fields: dict[str, Any] = {}
    for key in _ordered_keys(sample):
        values = [row.get(key) for row in sample]
        present = [value for value in values if value is not None]
        types = sorted({_type_of(value) for value in present})
        field: dict[str, Any] = {
            "type": types[0] if len(types) == 1 else " | ".join(types) or "null"
        }
        # Missing entirely and present-but-null are the same thing to
        # whoever writes the code that reads it.
        if len(present) != len(values):
            field["nullable"] = True
        if present:
            field["example"] = _example(present[0])
        fields[key] = field
    return fields


def _ordered_keys(sample: list[dict]) -> list[str]:
    """Every key any row has, in the order first seen."""
    keys: list[str] = []
    for row in sample:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def _type_of(value: Any) -> str:
    # bool before int: in Python, True is an int, and calling it a number
    # would send someone looking for arithmetic on a flag.
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


def _example(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 80:
        return value[:80] + "…"
    if isinstance(value, (list, dict)):
        return f"<{_type_of(value)}>"
    return value
