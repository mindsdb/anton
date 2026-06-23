"""Sub-tools exposed to the inner generation LLM.

Only three are needed for stage 1:

  - ``write_file(path, content)``  — produce one file inside the artifact folder.
  - ``read_file(path)``             — read a file the sub-agent previously wrote
    (useful for iterative refinement when a single write doesn't cut it).
  - ``finish(summary)``             — terminal tool; signals the loop to stop.

Each handler accepts the artifact ``root`` plus the sub-agent's input dict and
returns a string the engine forwards back to the LLM via a ``tool_result``
block. The path sandbox lives here so the engine doesn't have to repeat the
``relative_to`` check at every call site.
"""

from __future__ import annotations

from pathlib import Path


WRITE_FILE_SCHEMA: dict = {
    "name": "write_file",
    "description": (
        "Write a UTF-8 text file at the given path inside the artifact folder. "
        "Path is relative to the artifact root (e.g. \"dashboard.html\", "
        "\"static/index.html\", \"backend.py\"). Parent directories are "
        "created automatically. Overwrites any existing file at the same path."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path inside the artifact folder.",
            },
            "content": {
                "type": "string",
                "description": "Full UTF-8 contents to write.",
            },
        },
        "required": ["path", "content"],
    },
}


READ_FILE_SCHEMA: dict = {
    "name": "read_file",
    "description": (
        "Read a file you previously wrote into the artifact folder. Use this "
        "to check or amend earlier output. Path is relative to the artifact root."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path inside the artifact folder.",
            },
        },
        "required": ["path"],
    },
}


FINISH_SCHEMA: dict = {
    "name": "finish",
    "description": (
        "Terminate the generation. Call this after every file has been written. "
        "Pass a one-line `summary` describing what you produced."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "One-line summary of the generated artifact.",
            },
        },
        "required": ["summary"],
    },
}


def _scratchpad_schema() -> dict:
    # Reuse the exact schema + description the main agent sees, so the
    # sub-generator drives scratchpads with the same contract. Imported
    # lazily to avoid a tool_defs <-> generate_artifact import cycle.
    from anton.core.tools.tool_defs import SCRATCHPAD_TOOL

    return {
        "name": SCRATCHPAD_TOOL.name,
        "description": SCRATCHPAD_TOOL.description,
        "input_schema": SCRATCHPAD_TOOL.input_schema,
    }


def tool_schemas() -> list[dict]:
    return [WRITE_FILE_SCHEMA, READ_FILE_SCHEMA, FINISH_SCHEMA, _scratchpad_schema()]


def _sandboxed_path(root: Path, rel: str) -> Path | None:
    """Resolve ``rel`` against ``root`` and reject anything escaping it.

    Returns ``None`` for paths that traverse outside the artifact folder
    (via ``..`` or absolute prefixes). The engine surfaces a clear error
    to the sub-agent so it can retry with a corrected path.
    """
    if not rel or not isinstance(rel, str):
        return None
    rel = rel.strip().lstrip("/")
    if not rel:
        return None
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def write_file(root: Path, rel_path: str, content: str) -> dict:
    """Write ``content`` into ``<root>/<rel_path>``.

    Returns ``{"ok", "message", "written"?}`` where ``written`` is the
    relative path (string) when the write succeeded.
    """
    target = _sandboxed_path(root, rel_path)
    if target is None:
        return {
            "ok": False,
            "message": (
                "Error: `path` must be inside the artifact folder "
                "and non-empty (received: "
                f"{rel_path!r})."
            ),
        }
    if not isinstance(content, str):
        return {"ok": False, "message": "Error: `content` must be a string."}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    rel_written = str(target.relative_to(root.resolve()))
    return {
        "ok": True,
        "written": rel_written,
        "message": f"Wrote {rel_written} ({len(content)} bytes).",
    }


def read_file(root: Path, rel_path: str) -> dict:
    """Read ``<root>/<rel_path>`` and return its contents as text."""
    target = _sandboxed_path(root, rel_path)
    if target is None:
        return {
            "ok": False,
            "message": (
                "Error: `path` must be inside the artifact folder "
                f"(received: {rel_path!r})."
            ),
        }
    if not target.is_file():
        return {"ok": False, "message": f"Error: file not found: {rel_path}"}
    try:
        return {"ok": True, "message": target.read_text(encoding="utf-8")}
    except OSError as exc:
        return {"ok": False, "message": f"Error reading {rel_path}: {exc}"}
