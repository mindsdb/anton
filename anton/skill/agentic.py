"""Agentic workflow helper for skills.

When a skill needs to run an LLM tool-call loop (e.g., a multi-step reasoning
task with custom tools), it can use ``agentic_loop`` instead of hand-rolling
the loop.

Quick start::

    from anton.skill.agentic import agentic_loop
    from anton.skill.base import SkillResult, skill

    @skill("classify_batch", "Classify a batch of items using LLM with tools")
    async def classify_batch(items: str) -> SkillResult:
        tools = [{
            "name": "record_classification",
            "description": "Record the classification of an item",
            "input_schema": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                },
                "required": ["item", "label"],
            },
        }]

        classifications = {}

        async def handle_tool(name: str, inputs: dict) -> str:
            if name == "record_classification":
                classifications[inputs["item"]] = inputs["label"]
                return f"Recorded: {inputs['item']} -> {inputs['label']}"
            return "Unknown tool"

        await agentic_loop(
            system="Classify each item. Use the record_classification tool for each one.",
            user_message=f"Classify these items: {items}",
            tools=tools,
            handle_tool=handle_tool,
        )

        return SkillResult(output=classifications)

For simple single-call LLM usage (no tools), just use ``get_llm()`` directly::

    from anton.skill.context import get_llm

    llm = get_llm()
    response = await llm.complete(
        system="Summarize the following text.",
        messages=[{"role": "user", "content": text}],
    )
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from anton.skill.context import get_llm


async def agentic_loop(
    *,
    system: str,
    user_message: str,
    tools: list[dict[str, Any]],
    handle_tool: Callable[[str, dict[str, Any]], Awaitable[str]],
    max_turns: int = 10,
    max_tokens: int = 4096,
) -> str:
    """Run an LLM tool-call loop until the model stops calling tools.

    This is the building block for agentic workflows inside skills. The LLM
    is provided automatically by Anton's runtime — no credentials needed.

    Args:
        system: System prompt for the LLM.
        user_message: Initial user message to start the conversation.
        tools: Tool definitions (Anthropic tool schema format).
        handle_tool: Async callback ``(tool_name, tool_input) -> result_string``.
            Called for each tool the LLM invokes.
        max_turns: Safety limit on LLM round-trips (default 10).
        max_tokens: Max tokens per LLM call.

    Returns:
        The final text response from the LLM after it stops calling tools.
    """
    llm = get_llm()
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

    for _ in range(max_turns):
        response = await llm.complete(
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )

        if not response.tool_calls:
            return response.content

        # Build the assistant message with both text and tool_use blocks
        assistant_content: list[dict[str, Any]] = []
        if response.content:
            assistant_content.append({"type": "text", "text": response.content})
        for tc in response.tool_calls:
            assistant_content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.input,
            })
        messages.append({"role": "assistant", "content": assistant_content})

        # Execute each tool and collect results
        tool_results: list[dict[str, Any]] = []
        for tc in response.tool_calls:
            try:
                result = await handle_tool(tc.name, tc.input)
            except Exception as exc:
                result = f"Error: {exc}"
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

    # Hit max_turns — return whatever we have
    return response.content
