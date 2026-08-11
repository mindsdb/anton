"""Shared serialization of LLM messages/responses to plain JSON-able dicts.

Used by the optional debug loggers (`anton._llm_history_hook` and the
`generate_artifact` step trace). Kept provider-agnostic and dependency-free
so any logger can import it without pulling in the provider classes.
"""

from __future__ import annotations

from typing import Any


def serialize_content(content: Any) -> Any:
    """Normalize a message `content` field to JSON-able form.

    Non-list content (plain strings) passes through untouched. List content
    is mapped block-by-block; unknown block shapes pass through unchanged so
    nothing is silently dropped.
    """
    if not isinstance(content, list):
        return content
    blocks: list[Any] = []
    for block in content:
        if not isinstance(block, dict):
            blocks.append(block)
            continue
        t = block.get("type")
        if t == "text":
            blocks.append({"type": "text", "text": block.get("text", "")})
        elif t == "tool_use":
            blocks.append({
                "type": "tool_use",
                "id": block.get("id"),
                "name": block.get("name"),
                "input": block.get("input"),
            })
        elif t == "tool_result":
            blocks.append({
                "type": "tool_result",
                "tool_use_id": block.get("tool_use_id"),
                "content": block.get("content"),
            })
        else:
            blocks.append(block)
    return blocks


def serialize_messages(messages: list[dict]) -> list[dict]:
    """Normalize a list of chat messages to `[{role, content}]`."""
    return [
        {"role": m.get("role", "?"), "content": serialize_content(m.get("content", ""))}
        for m in messages
    ]


def serialize_response(response) -> dict:
    """Normalize an `LLMResponse` to a JSON-able dict."""
    return {
        "content": response.content or "",
        "tool_calls": [
            {"name": tc.name, "input": tc.input} for tc in response.tool_calls
        ],
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "stop_reason": response.stop_reason,
    }
