"""Structured output extraction via forced tool calling."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from .provider import LLMProvider

T = TypeVar("T", bound=BaseModel)


async def generate_object(
    schema_class: type[T],
    *,
    llm_provider: LLMProvider,
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int = 256,
) -> T:
    """Extract a Pydantic object from the LLM using forced tool calling."""
    schema = schema_class.model_json_schema()
    tool_name = schema_class.__name__
    tool = {
        "name": tool_name,
        "description": f"Generate structured output matching the {tool_name} schema.",
        "input_schema": schema,
    }
    response = await llm_provider.complete(
        model=model,
        system=system,
        messages=messages,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool_name},
        max_tokens=max_tokens,
    )
    if not response.tool_calls:
        raise ValueError("LLM did not return structured output.")
    return schema_class.model_validate(response.tool_calls[0].input)
