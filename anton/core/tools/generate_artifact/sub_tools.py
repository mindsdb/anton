"""Sub-tools exposed to the inner generation LLM.

The write loop offers four:

  - ``write_file(path, mode)`` — produce one file inside the artifact folder;
    the body travels as plain text in the reply (see the file-body protocol).
  - ``read_text_file``         — the main agent's tool; the engine reads paths
    relative to the artifact folder and never returns line numbers.
  - ``finish(summary)``        — terminal tool; signals the loop to stop.
  - ``scratchpad``             — the main agent's scratchpad, unchanged.

Each handler accepts the artifact ``root`` plus the sub-agent's input dict and
returns a string the engine forwards back to the LLM via a ``tool_result``
block. The path sandbox lives here so the engine doesn't have to repeat the
``relative_to`` check at every call site.
"""

from __future__ import annotations

from pathlib import Path

from anton.core.tools.text_file import (
    DEFAULT_LINE_COUNT,
    MAX_LINE_CHARS,
    count_lines,
    next_line_number,
)


# ── Tool-call protocol helpers shared by both loops ─────────────────────────
#
# The generation loop (engine.py) and the gathering loop (discovery/engine.py)
# speak the same Anthropic-style tool_use / tool_result block protocol. These
# helpers are the one place that shape is written down.


def tool_schema(tool, description: str | None = None) -> dict:
    """The LLM-facing schema of a `ToolDef`, optionally with another description.

    Reusing the main agent's `ToolDef` keeps a sub-tool's contract identical to
    the top-level tool it forwards to; the description is the one thing a
    pipeline may want to phrase for its own context.
    """
    return {
        "name": tool.name,
        "description": tool.description if description is None else description,
        "input_schema": tool.input_schema,
    }


def tool_result(tool_use_id: str, content) -> dict:
    """One `tool_result` block answering the tool call `tool_use_id`."""
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


def assistant_blocks(response) -> list[dict]:
    """The assistant turn `response` becomes in the history: its text, then
    one `tool_use` block per tool call."""
    blocks: list[dict] = []
    if response.content:
        blocks.append({"type": "text", "text": response.content})
    for tc in response.tool_calls:
        blocks.append(
            {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
        )
    return blocks


def malformed_input_result(tc) -> dict:
    """The `tool_result` for a call whose JSON input did not parse."""
    return tool_result(
        tc.id,
        "Error: malformed tool input — re-emit with valid JSON. "
        f"({tc.parse_error})",
    )


def unwrap_outcome(result):
    """Flatten a handler result down to `tool_result`-ready content.

    Some handlers return a `ToolOutcome` (content + the
    handler's own ok/reason verdict) instead of a bare string —
    `handle_scratchpad`'s exec path is the one both loops hit. The verdict
    drives the outer agent's error streak, which the sub-loops do not
    participate in, so only `content` is meaningful here; passing the
    dataclass straight into a tool_result block would ship its repr to the
    model.
    """
    from anton.core.tools.registry import ToolOutcome

    return result.content if isinstance(result, ToolOutcome) else result


# ── The file-body protocol ──────────────────────────────────────────────────
#
# The body of a generated file travels as PLAIN TEXT in the assistant's reply,
# between these two markers, and `write_file` carries only `path` and `mode`.
#
# Why, in one line: the Anthropic API buffers and validates a tool parameter in
# full before streaming it, so a body sent as a tool argument means a silent
# connection for the whole generation (measured 2026-08-28: 112s of dead air
# for a 59 000-character argument, reproduced against api.anthropic.com). Plain
# text streams evenly through the same channel. The API's
# `eager_input_streaming` flag would lift this, but the gateway does not
# forward it yet.
#
# Both values are quoted to the model in several places; every one of them
# reads these constants, because a literal on any surface drifts silently.
FILE_BEGIN_MARKER = "<<<ANTON_FILE_BEGIN>>>"
FILE_END_MARKER = "<<<ANTON_FILE_END>>>"

#: Prose around the body. NOT a failure, and deliberately not named as one: the
#: body was complete, the file was written, nothing was retried. A preamble
#: before the body is the model's ordinary habit, and the interleaving of text
#: and tool_use blocks is lost by the time a response reaches us
#: (`LLMResponse.content` is one joined string), so a plain "Done." after the
#: tool call is indistinguishable from prose after the end marker.
#:
#: Recorded anyway, and only for this: it is how a CHANGE in the model's
#: formatting becomes visible. It must NOT count toward the protocol's
#: acceptance thresholds — those are about rounds lost, and this costs none.
NOTE_TEXT_BEFORE = "text before the begin marker"
NOTE_TEXT_AFTER = "text after the end marker"


def extract_file_body(
    text: str, *, looks_truncated: bool = False
) -> tuple[str | None, str | None, list[str]]:
    """Pull one file body out of the assistant's text.

    Returns ``(body, error, notes)``: exactly one of ``body``/``error`` is not
    None. ``notes`` carries the formatting remarks above and is only ever
    non-empty alongside a body — observations, not failures.

    Strictness is deliberately asymmetric. A duplicated marker or an empty body
    could make us write the WRONG bytes, so those refuse; stray prose cannot,
    so it is recorded and the body is used. Silently writing a truncated file
    is the failure this whole protocol exists to make impossible.
    """
    text = text or ""
    begins = text.count(FILE_BEGIN_MARKER)
    ends = text.count(FILE_END_MARKER)

    if begins > 1 or ends > 1:
        # Never "take the first pair": if the content itself contains a marker,
        # every guess about which pair is real is a guess about where the file
        # ends, and guessing wrong truncates it without a sign.
        return None, (
            f"Error: the body markers must appear exactly once each, but this "
            f"reply has {begins} `{FILE_BEGIN_MARKER}` and {ends} "
            f"`{FILE_END_MARKER}`. Nothing was written. Re-send the body with "
            f"exactly one of each."
        ), []

    if begins == 0:
        return None, (
            f"Error: no file body found in your reply. Put the complete file "
            f"content between a line `{FILE_BEGIN_MARKER}` and a line "
            f"`{FILE_END_MARKER}`, then call `write_file` with the path. "
            f"Nothing was written."
        ), []

    if ends == 0:
        if looks_truncated:
            return None, (
                "Error: your reply was cut off before the end marker, so the "
                "body is incomplete and nothing was written.\n\n"
                "In your NEXT REPLY send a SHORTER first part: the content "
                f"itself between `{FILE_BEGIN_MARKER}` and `{FILE_END_MARKER}`, "
                "plus the `write_file` call for it. Both in that one reply. "
                "Append the remainder with `mode=\"a\"` in the reply after "
                "that.\n\n"
                "Do NOT announce the part instead of sending it: a reply whose "
                "text has no body between the markers writes nothing, however "
                "it is introduced."
            ), []
        return None, (
            f"Error: the body has no `{FILE_END_MARKER}` line, so nothing was "
            f"written. Your reply was not cut off, so re-send the body with "
            f"the closing marker in place."
        ), []

    start_at = text.index(FILE_BEGIN_MARKER)
    end_at = text.index(FILE_END_MARKER)
    if end_at < start_at:
        return None, (
            f"Error: `{FILE_END_MARKER}` appears before `{FILE_BEGIN_MARKER}`. "
            f"Nothing was written."
        ), []

    body = text[start_at + len(FILE_BEGIN_MARKER):end_at]
    # Exactly one newline on each side — the markers sit on their own lines, so
    # those two newlines belong to the protocol, not to the file. Everything
    # else is kept byte for byte: trailing blank lines can be significant.
    for nl in ("\r\n", "\n"):
        if body.startswith(nl):
            body = body[len(nl):]
            break
    for nl in ("\r\n", "\n"):
        if body.endswith(nl):
            body = body[: -len(nl)]
            break

    if not body.strip():
        return None, (
            "Error: the body markers are empty. Put the file content between "
            "them. Nothing was written."
        ), []

    notes: list[str] = []
    if text[:start_at].strip():
        notes.append(NOTE_TEXT_BEFORE)
    if text[end_at + len(FILE_END_MARKER):].strip():
        notes.append(NOTE_TEXT_AFTER)
    return body, None, notes


WRITE_FILE_SCHEMA: dict = {
    "name": "write_file",
    "description": (
        "Write a UTF-8 text file inside the artifact folder.\n\n"
        "The file's CONTENT is NOT an argument of this call. Put it in your "
        "reply as plain text, between a line "
        f"`{FILE_BEGIN_MARKER}` and a line `{FILE_END_MARKER}`, and then make "
        "this call with just the path. Everything between those two lines is "
        "written verbatim.\n\n"
        "Path is relative to the artifact root (e.g. \"index.html\", "
        "\"static/index.html\", \"backend.py\"). Parent directories are "
        "created automatically.\n\n"
        "`mode=\"w\"` (default) creates or overwrites the file; `mode=\"a\"` "
        "appends to it, creating it first if needed. ONE call per reply: a "
        "reply carries one body, so one file (or one appended part) at a time."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path inside the artifact folder.",
            },
            "mode": {
                "type": "string",
                "enum": ["w", "a"],
                "description": "\"w\" overwrite (default), \"a\" append.",
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

    return tool_schema(SCRATCHPAD_TOOL)


GEN_READ_TEXT_FILE_DESCRIPTION = (
    "Read a file you are writing, only when you must see it to keep writing; "
    "never to check finished work. `path` is relative to the artifact folder. "
    "To see where your last part ended, read the end: `start_line=-20`. To see "
    "one section, pass its line range; `write_file` reports the lines each part "
    f"occupies. Without a range you get the first {DEFAULT_LINE_COUNT} lines; `end_line=-1` "
    "returns the whole file. Whatever you read stays in your context for every "
    "remaining round. If the lines you asked for do not fit into one call, a line "
    f"longer than {MAX_LINE_CHARS:,} characters is shortened to its "
    f"first and last {MAX_LINE_CHARS // 2:,} characters with a mark in between; never copy the "
    "mark into what you write."
)


def _read_text_file_schema() -> dict:
    # The main agent's contract with two changes for the write loop: paths are
    # relative to the artifact folder, and there are no line numbers to copy
    # into a file. Deep-copied because `tool_schema` hands out the shared dict.
    import copy

    from anton.core.tools.tool_defs import READ_TEXT_FILE_TOOL

    schema = tool_schema(READ_TEXT_FILE_TOOL, description=GEN_READ_TEXT_FILE_DESCRIPTION)
    input_schema = copy.deepcopy(schema["input_schema"])
    input_schema["properties"]["path"]["description"] = (
        "Path relative to the artifact folder, or absolute."
    )
    input_schema["properties"].pop("line_numbers", None)
    return {**schema, "input_schema": input_schema}


def tool_schemas() -> list[dict]:
    return [WRITE_FILE_SCHEMA, _read_text_file_schema(), FINISH_SCHEMA, _scratchpad_schema()]


def is_host_path(path: str) -> bool:
    """`path` starts at a directory that exists on this machine.

    A model writes "/index.html" or "/static/app.js" meaning the artifact
    folder's root; "/home/u/project/data.csv" names a real file elsewhere.
    """
    if not path.startswith("/"):
        return False
    parts = Path(path).parts
    if len(parts) < 2:
        return False
    try:
        return Path(parts[0], parts[1]).is_dir()
    except OSError:
        return False


def _sandboxed_path(root: Path, rel: str) -> Path | None:
    """Resolve ``rel`` against ``root`` and reject anything escaping it.

    Returns ``None`` for paths that traverse outside the artifact folder
    (via ``..``, or an absolute path to another place on this machine). The
    engine surfaces a clear error to the sub-agent so it can retry with a
    corrected path.
    """
    if not rel or not isinstance(rel, str):
        return None
    rel = rel.strip()
    root_resolved = root.resolve()
    if rel.startswith("/"):
        absolute = Path(rel).resolve()
        if absolute != root_resolved and absolute.is_relative_to(root_resolved):
            return absolute
        # Re-rooting a real path would bury a copy of it inside the artifact,
        # where nothing reads it and the publish ships it.
        if is_host_path(rel):
            return None
    rel = rel.lstrip("/")
    if not rel:
        return None
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root_resolved)
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
    before = ""
    if mode == "a":
        try:
            before = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            before = ""
    with open(target, mode, encoding="utf-8") as f:
        f.write(content)
    rel_written = str(target.relative_to(root.resolve()))
    verb = "Appended to" if mode == "a" else "Wrote"
    # The model's next question after a part lands is "where did it land, and
    # is the file closed"; answering it here saves the round it would spend
    # reading (measured 2026-09-14). The line range is counted the way
    # `read_text_file` counts, so it can read exactly that range back.
    #
    # Sizes are CHARACTERS, the unit the model writes in (REPLY_BODY_CHARS): a
    # byte count from stat() once disagreed with the character delta by the
    # UTF-8 overhead alone, implying text had been there before.
    try:
        whole = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # The write succeeded; only the read-back is unavailable. Reporting the
        # write as failed here would be a lie about what is on disk.
        tail = ""
    else:
        total = count_lines(whole)
        first = next_line_number(before)
        span = f"; this part is lines {first}-{total}" if content and first <= total else ""
        tail = f"{span}; file now {len(whole)} characters / {total} lines"
    message = f"{verb} {rel_written} (+{len(content)} characters{tail})."
    # A reminder about the NEXT part, delivered at the only moment it is needed.
    #
    # Measured twice, 2026-09-15: after a successful first part the model
    # replied "Now I'll append the JavaScript logic:" and called `write_file`
    # with no body — costing a round each time. It is not a misread rule. In
    # the tool-use format the normal shape of a reply is a short preamble
    # followed by the call, with the payload inside the call; this protocol
    # puts the payload outside it, and at the start of a continuation round the
    # model is answering a tool result, which is exactly when a preamble feels
    # natural. The system prompt is thousands of tokens behind by then. This
    # line is not.
    message += (
        f" If this file needs another part, put that part's content between "
        f"`{FILE_BEGIN_MARKER}` and `{FILE_END_MARKER}` in the SAME reply as "
        f"its `write_file` call — a reply that only announces the next part "
        f"writes nothing."
    )
    return {
        "ok": True,
        "written": rel_written,
        "message": message,
    }
