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

# Chunk size the write discipline asks for (see prompts._WRITE_DISCIPLINE).
# Soft: an oversized chunk that made it through IS written — the warning in the
# result is about the next call, which may not be as lucky against the output cap.
CHUNK_SOFT_LIMIT = 6_000

# Tail returned by read_file when `full` is not requested.
READ_TAIL_CHARS = 500


WRITE_FILE_SCHEMA: dict = {
    "name": "write_file",
    "description": (
        "Write a UTF-8 text file at the given path inside the artifact folder. "
        "Path is relative to the artifact root (e.g. \"dashboard.html\", "
        "\"static/index.html\", \"backend.py\"). Parent directories are "
        "created automatically.\n\n"
        "`mode=\"w\"` (default) creates or overwrites the file. `mode=\"a\"` "
        "appends to it, creating it first if needed — use append to build a "
        "large file in several small calls instead of one huge one. A single "
        "call whose `content` is too large gets cut off by the output limit and "
        "is rejected, so keep each call's `content` at most 6,000 characters."
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
                "description": "Full UTF-8 contents to write (or the chunk to append).",
            },
            "mode": {
                "type": "string",
                "enum": ["w", "a"],
                "description": "\"w\" overwrite (default), \"a\" append.",
            },
        },
        "required": ["path", "content"],
    },
}


READ_FILE_SCHEMA: dict = {
    "name": "read_file",
    "description": (
        "Check a file you previously wrote into the artifact folder. By default "
        "returns the file's size and its tail — enough to see what landed and "
        "whether the file is closed. Pass `full=true` ONLY when you must "
        "re-read the entire content (expensive: the whole file enters your "
        "context) — never to verify finished work, the pipeline verifier does "
        "that after `finish`. Path is relative to the artifact root."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path inside the artifact folder.",
            },
            "full": {
                "type": "boolean",
                "description": (
                    "Return the entire file content instead of size + tail. "
                    "Default false."
                ),
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


def write_file(root: Path, rel_path: str, content: str, *, mode: str = "w") -> dict:
    """Write ``content`` into ``<root>/<rel_path>``.

    ``mode="a"`` appends (creating the file when absent) so the sub-generator can
    build a large file in several small calls — a single call carrying a whole
    dashboard gets truncated by the output-token limit (see the design spec, 3.1).

    Returns ``{"ok", "message", "written"?}`` where ``written`` is the
    relative path (string) when the write succeeded.
    """
    if mode not in ("w", "a"):
        return {"ok": False, "message": f"Error: `mode` must be \"w\" or \"a\" (received: {mode!r})."}
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
    with open(target, mode, encoding="utf-8") as f:
        f.write(content)
    rel_written = str(target.relative_to(root.resolve()))
    size = target.stat().st_size
    verb = "Appended to" if mode == "a" else "Wrote"
    message = f"{verb} {rel_written} (+{len(content)} bytes, file now {size} bytes)."
    if len(content) > CHUNK_SOFT_LIMIT:
        # The write itself succeeded — this call fit under the output cap. The
        # warning is about the NEXT one: a model that got away with an
        # oversized chunk keeps growing them until one is cut off mid-argument
        # and the whole round is wasted (measured in both attempts of the
        # 2026-08-27 live run).
        message += (
            f" WARNING: this chunk was {len(content)} characters — over the "
            f"{CHUNK_SOFT_LIMIT:,}-character chunk limit. Keep every following "
            "chunk under the limit or it will be cut off by the output cap "
            "and rejected."
        )
    return {
        "ok": True,
        "written": rel_written,
        "message": message,
    }


def read_file(root: Path, rel_path: str, *, full: bool = False) -> dict:
    """Read ``<root>/<rel_path>``.

    By default returns the size plus the tail of the file, not the whole
    content: the loop's own prompt tells the model to use this call to check
    what landed, and for that the tail is sufficient. Returning the full text
    by default meant re-reading a whole 48 KB page into the context (and
    through the prompt-cache prefix) just to confirm it ends with ``</html>``
    — measured 2026-08-27 at ~19k input tokens per check. ``full=True``
    returns the entire content for genuine re-reads.
    """
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
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "message": f"Error reading {rel_path}: {exc}"}
    if full or len(text) <= READ_TAIL_CHARS:
        return {"ok": True, "message": text}
    return {
        "ok": True,
        "message": (
            f"{rel_path} is {len(text)} characters, "
            f"{text.count(chr(10)) + 1} lines. Last {READ_TAIL_CHARS} "
            f"characters:\n…{text[-READ_TAIL_CHARS:]}\n\n"
            "(pass `full=true` to read the entire file)"
        ),
    }
