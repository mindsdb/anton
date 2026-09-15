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
# result is about the next call.
#
# The bound is DURATION, not the output-token budget, and that is the whole
# reason it is not simply `GEN_WRITE_MAX_TOKENS` worth of text. Measured
# 2026-08-28: a large `write_file` argument is not streamed incrementally —
# the connection carries nothing for the entire generation and everything
# arrives in one burst at the end (112s of silence for a 59 000-character
# argument, reproduced identically against api.anthropic.com, so this is not
# the gateway's doing). Whether such a call survives is a race against the
# proxy's idle timeout: silences of 112-115s came back, 131-143s were dropped.
#
# So: coding model ~170 output tokens/s, ~2.59 characters per token on Cyrillic
# prose (the token-hungriest content we generate) → ~440 chars/s → 16 000
# characters is roughly 37s of silence, a ~3x margin against the shortest
# observed drop. Raising this trades that margin for fewer rounds; re-measure
# the drop threshold before doing so.
CHUNK_SOFT_LIMIT = 16_000

# Tail returned by read_file when `full` is not requested.
READ_TAIL_CHARS = 500


# ── The file-body protocol ──────────────────────────────────────────────────
#
# The body of a generated file travels as PLAIN TEXT in the assistant's reply,
# between these two markers, and `write_file` carries only `path` and `mode`.
#
# Why, in one line: the Anthropic API buffers and validates a tool parameter in
# full before streaming it, so a body sent as a tool argument means a silent
# connection for the whole generation (measured 2026-08-28: 112s of dead air
# for a 59 000-character argument, reproduced against api.anthropic.com). Plain
# text streams evenly through the same channel. The full analysis, including
# the supported `eager_input_streaming` flag the gateway does not forward yet,
# is in docs/artifact-generation-tools/2026-09-15-tool-call-argument-not-streamed.md.
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
        "Path is relative to the artifact root (e.g. \"dashboard.html\", "
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
    verb = "Appended to" if mode == "a" else "Wrote"
    # Lines beside the size: the model's next question after a chunk lands is
    # "where did it land, and is the file closed" — a size alone answers
    # neither, and the round it spends re-learning what this message already
    # reported is pure loss (measured 2026-09-14). With the chunk's line count
    # and the file's, an append's span is the last N lines, which is the map a
    # targeted re-read needs instead of pulling the whole file back through the
    # context.
    #
    # Both figures are CHARACTERS, and that is a correction: the delta used to
    # be len(content) labelled "bytes" while the total came from stat(), so a
    # `mode="w"` write of Cyrillic reported "+28114 bytes ... file now 29004
    # bytes" — the two disagreeing by the UTF-8 overhead alone, implying 890
    # bytes had been there before. The model is told to reason about sizes in
    # characters (REPLY_BODY_CHARS), `read_file` answers in characters, and it
    # counts what it writes in characters; a second unit here bought nothing
    # and contradicted the rest.
    chunk_lines = content.count("\n") + 1 if content else 0
    try:
        whole = target.read_text(encoding="utf-8")
    except OSError:
        # The write succeeded; only the read-back is unavailable. Reporting the
        # write as failed here would be a lie about what is on disk.
        tail = ""
    else:
        tail = (
            f", file now {len(whole)} characters"
            f" / {whole.count(chr(10)) + 1} lines"
        )
    message = (
        f"{verb} {rel_written} "
        f"(+{len(content)} characters / {chunk_lines} lines{tail})."
    )
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
