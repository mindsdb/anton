"""Shared schema-building / response-unwrapping for structured LLM output.

Two pure helper functions that turn a Pydantic model (or `list[Model]`)
into the inputs needed for a forced tool-call, and validate the LLM's
response back into a typed Python instance.

Used by:

  - `LLMClient.generate_object` — async, planning-LLM, main process
  - `_ScratchpadLLM.generate_object` — sync, scratchpad subprocess bridge

The two call sites differ only in *how* they invoke the provider (async
vs sync, different model/credential resolution). The schema-derivation
and Pydantic validation logic is identical and lives here exactly once.

Why a separate module
=====================

The original implementation was duplicated across `client.py` (added
for the cerebellum) and `scratchpad_boot.py` (the existing scratchpad
bridge). The two halves can't share a class because they live in
different runtime contexts (main process async vs subprocess sync), but
they CAN share pure helper functions — which is what this module
provides. Importing this module from either side is cheap and safe;
the subprocess already imports from `anton.core.*` at boot.
"""

from __future__ import annotations

from typing import Any


def build_structured_tool(schema_class) -> tuple[dict, type, bool]:
    """Build a forced tool-call definition from a Pydantic schema.

    Args:
        schema_class: A Pydantic ``BaseModel`` subclass, OR a
            ``list[Model]`` annotation for a homogeneous list. The
            list-of-model case is supported by wrapping the inner
            type in a synthetic ``_ArrayWrapper`` model with an
            ``items`` field — many providers refuse top-level
            arrays in tool input schemas, so the wrapper is required.

    Returns:
        A 3-tuple of:

        - **tool_dict**: ready to pass as ``tools=[tool_dict]`` to
          ``provider.complete()``. The caller should also pass
          ``tool_choice={"type": "tool", "name": tool_dict["name"]}``
          to force the LLM to call this specific tool.
        - **validator_class**: the Pydantic class to call
          ``model_validate()`` on (the wrapper for the list case,
          the original class otherwise).
        - **is_list**: True iff the original input was a ``list[Model]``
          annotation. The caller uses this to decide whether to unwrap
          the wrapper's ``items`` field after validation.

    Note:
        Pydantic is imported lazily so this module can be imported
        without forcing pydantic to be available at import time. The
        only operations on this module that REQUIRE pydantic are the
        actual function calls — at which point any caller doing
        structured output already needs pydantic anyway.
    """
    from pydantic import BaseModel

    is_list = (
        hasattr(schema_class, "__origin__")
        and schema_class.__origin__ is list
    )
    if is_list:
        inner_class = schema_class.__args__[0]

        class _ArrayWrapper(BaseModel):
            items: list[inner_class]  # type: ignore[valid-type]

        schema = _ArrayWrapper.model_json_schema()
        tool_name = f"{inner_class.__name__}_array"
        validator_class: type = _ArrayWrapper
    else:
        schema = schema_class.model_json_schema()
        tool_name = schema_class.__name__
        validator_class = schema_class

    tool = {
        "name": tool_name,
        "description": (
            f"Generate structured output matching the {tool_name} schema."
        ),
        "input_schema": schema,
    }
    return tool, validator_class, is_list


def unwrap_structured_response(
    tool_call_input: dict[str, Any],
    validator_class: type,
    is_list: bool,
):
    """Validate a forced tool-call's input via Pydantic and unwrap.

    Args:
        tool_call_input: The ``.input`` dict from the LLM's ``ToolCall``.
        validator_class: The validator class returned from
            ``build_structured_tool``.
        is_list: The ``is_list`` flag returned from
            ``build_structured_tool``.

    Returns:
        A validated Pydantic instance, or a ``list[Model]`` if the
        original schema was a list annotation.

    Raises:
        pydantic.ValidationError: If ``tool_call_input`` doesn't match
            the schema. With forced tool_choice this is rare — the
            provider usually rejects misshapen tool calls server-side
            — but the validation step is the safety net.
    """
    validated = validator_class.model_validate(tool_call_input)
    if is_list:
        return validated.items  # type: ignore[attr-defined]
    return validated


__all__ = [
    "build_structured_tool",
    "unwrap_structured_response",
]
