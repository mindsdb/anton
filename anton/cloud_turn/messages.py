"""Explicit conversion: Anton internal messages → V1 wire messages.

Anton stores Anthropic-style dicts (``{"role", "content"}`` where content is a
string or a list of ``text`` / ``tool_use`` / ``tool_result`` blocks). We never
put raw internal dicts on the wire — everything goes through :class:`MessageV1`
validation here, so the wire shape is enforced and any unsupported block type
fails loudly rather than being emitted unchecked.

V1 limitation: only ``text`` / ``tool_use`` / ``tool_result`` blocks are
representable (the cloud tool surface — scratchpad + artifacts — produces only
these). An ``image`` or other block type raises :class:`MessageConversionError`
instead of being silently dropped.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from anton.cloud_turn.protocol import ContentBlockV1, MessageV1


class MessageConversionError(Exception):
    """An Anton message could not be represented in the V1 wire shape."""


# ── wire → Anton (input direction) ──────────────────────────────────────────
# The request's typed models must become the Anthropic-style dicts Anton's
# ChatSession/turn_stream consume. model_dump() yields exactly that shape
# ({"type": ...} blocks, string content preserved).


def to_anton_input(value: "str | list[ContentBlockV1]") -> "str | list[dict]":
    """Convert the request's ``input`` to what ``turn_stream`` expects."""
    if isinstance(value, str):
        return value
    return [block.model_dump() for block in value]


def to_anton_history(history: list[MessageV1]) -> list[dict[str, Any]]:
    """Convert the request's typed input history to Anton history dicts."""
    return [msg.model_dump() for msg in history]


def _to_wire_message(msg: dict) -> MessageV1:
    try:
        return MessageV1.model_validate(msg)
    except ValidationError as exc:
        raise MessageConversionError(
            f"message not representable in V1: {exc}"
        ) from exc


def turn_output_messages(
    history: list[dict], pre_turn_messages: list[dict]
) -> list[MessageV1]:
    """Return only the messages GENERATED during the current turn.

    NOT length-based slicing: ``ChatSession.turn_stream`` can rewrite history
    mid-turn — ``_summarize_history`` reassigns the list, collapsing the old
    prefix into a *new* summary message (plus an "Understood" separator), and a
    tool-result "seal" can ``insert`` a message. A stored index would then point
    at the wrong place.

    Instead we anchor on object identity. ``pre_turn_messages`` is the list of
    message objects captured BEFORE the turn (``list(session.history)``). The
    caller MUST keep it alive across the turn: holding the references guarantees
    those objects can't be garbage-collected and have their ``id()`` recycled by
    a message created during the turn (compaction frees the collapsed prefix —
    without live references its ids get reused, causing false matches).

    The current-turn region is the suffix of the final history after the LAST
    message whose identity was present pre-turn (the most recent surviving input
    message). Compaction keeps a suffix of the input turns as the anchor and
    prepends the summary/separator BEFORE them, so those artifacts fall outside
    the region; a seal-inserted tool_result lands after the current assistant
    and stays inside it. We then drop the leading user echo (this turn's input,
    which cowork-server already owns) and start at the first assistant message.

    Assumption (documented): the input history is well-formed — a compaction
    summary only ever precedes surviving input messages, and the current user
    input does not merge into a prior history entry (input ends with an
    assistant message or is empty). A message compacted away mid-turn is, by
    design, absent from the returned record.
    """
    pre_turn_ids = {id(m) for m in pre_turn_messages}
    # Anchor: index just past the last surviving pre-turn (input) message.
    boundary = 0
    for i in range(len(history) - 1, -1, -1):
        if id(history[i]) in pre_turn_ids:
            boundary = i + 1
            break
    generated = history[boundary:]
    # Drop the leading input echo: start at the first assistant message. Tool
    # results (also user role) are interior — they follow an assistant tool_use.
    start = None
    for i, m in enumerate(generated):
        if m.get("role") == "assistant":
            start = i
            break
    if start is None:
        return []  # no assistant output produced this turn
    return [_to_wire_message(m) for m in generated[start:]]


def final_assistant_text(messages: list[MessageV1]) -> str:
    """Concatenated text of the last assistant message (empty if none)."""
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        if isinstance(msg.content, str):
            return msg.content
        return "".join(
            block.text for block in msg.content if getattr(block, "type", None) == "text"
        )
    return ""
