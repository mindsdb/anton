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

from typing import Any, NoReturn

from .provider import StructuredOutputError


def looks_truncated(response, budget: int) -> bool:
    """True if `response` was cut off by the `max_tokens` budget.

    Token count first, because the MindsHub gateway reports
    ``finish_reason: "stop"`` at the cap for most aliases (ENG-1082) — the
    standard ``"length"`` check can't be relied on there. Both dialects are
    honoured when reported: OpenAI says ``"length"``, Anthropic ``"max_tokens"``.
    No usage information → ``False``; without evidence we don't buy a retry.
    """
    usage = getattr(response, "usage", None)
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    stop_reason = getattr(response, "stop_reason", None)
    return stop_reason in ("length", "max_tokens") or (
        budget > 0 and output_tokens >= budget
    )


def raise_unusable_tool_call(response, *, tool_name: str, budget: int) -> NoReturn:
    """Raise `StructuredOutputError` explaining *why* the tool call is unusable.

    Covers both shapes of the same underlying problem:

    - **No tool call at all** — the model narrated in plain ``content`` until
      the budget ran out.
    - **A damaged tool call** — the budget ran out *inside* the call's JSON
      arguments, so ``safe_parse_tool_input`` salvaged a partial dict and set
      ``parse_error``, and validation then fails. Without this, that case
      surfaces as a bare ``ValidationError``, which reads like a schema bug and
      never gets the retry a truncation deserves (ENG-1081).

    Lives here, next to `build_structured_tool`/`unwrap_structured_response`,
    because both structured-output paths must classify the failure the same
    way — the async `LLMClient._generate_object_with` and the sync
    `_ScratchpadLLM.generate_object` in the scratchpad subprocess. Keeping it
    in one of the two callers is how they drift (ENG-1081).

    A forced tool call can come back empty for two very different reasons:

    - **Truncated** — the model narrated in plain ``content`` and ran out of
      ``budget`` before reaching the call. Retrying with more room usually
      works. Models served through MindsHub's Fireworks aliases
      (``mindshub_air``/``kimi``, ``deepseek``) narrate before acting, so a
      tight budget fails them.
    - **Anything else** — the provider errored, refused, or returned nothing.
      A bigger budget won't help.

    See `looks_truncated` for how truncation is detected.

    Args:
        response: The provider's ``LLMResponse``.
        tool_name: Name of the forced tool, for the message.
        budget: The ``max_tokens`` the call was given.

    Raises:
        StructuredOutputError: Always. ``.truncated`` carries the verdict.
    """
    usage = getattr(response, "usage", None)
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    stop_reason = getattr(response, "stop_reason", None)
    truncated = looks_truncated(response, budget)
    what = (
        "returned an unusable tool call for"
        if getattr(response, "tool_calls", None)
        else "did not return a tool call for"
    )
    detail = (
        f" (truncated: {output_tokens}/{budget} output tokens spent before the "
        "call was complete)."
        if truncated
        else "."
    )
    raise StructuredOutputError(
        f"LLM {what} forced schema {tool_name}{detail}",
        truncated=truncated,
        output_tokens=output_tokens,
        max_tokens=budget,
        stop_reason=stop_reason,
    )


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
