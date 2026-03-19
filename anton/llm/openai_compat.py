from __future__ import annotations

import json


def translate_tools(tools: list[dict]) -> list[dict]:
    """Anthropic tool format -> OpenAI/Ollama function-calling format."""
    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return result


def translate_tool_choice(tool_choice: dict) -> dict | str:
    """Anthropic tool_choice -> OpenAI tool_choice."""
    tc_type = tool_choice.get("type")
    if tc_type == "tool":
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    if tc_type == "any":
        return "required"
    if tc_type == "auto":
        return "auto"
    return "auto"


def translate_messages(system: str, messages: list[dict]) -> list[dict]:
    """Convert Anthropic-style messages to OpenAI chat format."""
    result: list[dict] = []
    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        role = msg["role"]
        content = msg.get("content")

        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            if role == "assistant":
                result.extend(_translate_assistant_blocks(content))
            elif role == "user":
                result.extend(_translate_user_blocks(content))
            else:
                text = " ".join(
                    block.get("text", "")
                    for block in content
                    if block.get("type") == "text"
                )
                result.append({"role": role, "content": text or ""})
            continue

        result.append({"role": role, "content": str(content) if content else ""})

    return result


def _translate_assistant_blocks(blocks: list[dict]) -> list[dict]:
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in blocks:
        if block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    msg: dict = {"role": "assistant"}
    msg["content"] = "\n".join(text_parts) if text_parts else None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return [msg]


def _translate_user_blocks(blocks: list[dict]) -> list[dict]:
    result: list[dict] = []
    content_parts: list[dict] = []

    for block in blocks:
        if block.get("type") == "tool_result":
            if content_parts:
                result.append({"role": "user", "content": content_parts})
                content_parts = []
            content = block.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    item.get("text", "")
                    for item in content
                    if item.get("type") == "text"
                )
            result.append({
                "role": "tool",
                "tool_call_id": block["tool_use_id"],
                "content": str(content),
            })
        elif block.get("type") == "text":
            content_parts.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data}"},
                })

    if content_parts:
        if all(part.get("type") == "text" for part in content_parts):
            result.append({
                "role": "user",
                "content": "\n".join(part["text"] for part in content_parts),
            })
        else:
            result.append({"role": "user", "content": content_parts})

    return result


_translate_assistant_blocks = _translate_assistant_blocks
_translate_messages = translate_messages
_translate_tool_choice = translate_tool_choice
_translate_tools = translate_tools
_translate_user_blocks = _translate_user_blocks
