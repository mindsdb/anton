"""Extract one turn's tool block-rows from a raw anton history slice.

Shared by both consumers of that shape, which is why it lives in anton rather
than in either caller:

  * the pod entrypoint (`anton.cloud_turn.__main__`) builds the wire payload
    cowork-server persists for a remote turn;
  * cowork-server's in-process harness builds the same rows on the desktop path.

Keeping one implementation matters because the two paths must produce
byte-identical rows: a conversation can move between them (desktop and SaaS read
the same `messages` table), and a divergence would only surface as an invalid
tool_use/tool_result sequence many turns later.
"""

from __future__ import annotations

REPLAY_IMAGE_PLACEHOLDER = "[an image was returned here; omitted from replayed history]"


def sanitize_tool_result(block: dict) -> dict:
    """Replace image parts inside a tool_result with a text marker.

    Base64 image payloads would bloat the JSON column and add nothing to the
    replayed history. The marker states the removal happened at replay time
    (not that the tool returned it) so the model doesn't misread it.
    """
    content = block.get("content")
    if not isinstance(content, list):
        return block
    scrubbed = [
        {"type": "text", "text": REPLAY_IMAGE_PLACEHOLDER}
        if isinstance(part, dict) and part.get("type") == "image"
        else part
        for part in content
    ]
    return {**block, "content": scrubbed}


def split_turn_into_rows(history_slice: list) -> list[dict]:
    """Extract the tool block-rows of one turn from anton's raw history slice.

    Keeps only `tool_use` (assistant) / `tool_result` (user) blocks as
    `{role, content}` rows. Text blocks are dropped — the assistant's visible
    text is persisted once in the display row — so a pure-text message yields
    no row. Images inside tool_result are replaced with a replay marker.
    """
    rows: list[dict] = []
    for msg in history_slice:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        keep_type = "tool_use" if role == "assistant" else "tool_result"
        blocks = [
            sanitize_tool_result(b) if keep_type == "tool_result" else b
            for b in content
            if isinstance(b, dict) and b.get("type") == keep_type
        ]
        if blocks:
            rows.append({"role": role, "content": blocks})
    return rows
