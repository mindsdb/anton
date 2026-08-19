"""Shared schema-building / response-unwrapping for structured LLM output.

Pure helper functions that turn a Pydantic model (or `list[Model]`) into
the inputs needed for a forced tool-call and validate the LLM's response
back into a typed Python instance — plus the shared budget ladder
(`generate_with_truncation_retry`) every forced-schema call site runs
through so narrating models get room to reach the call (ENG-1084).

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

import logging
from typing import Any, Awaitable, Callable, NoReturn

from .provider import StructuredOutputError


# Default output-budget ladder for forced-schema calls: first attempt, then
# one retry used only when the first came back truncated. Sized by the same
# measurement as the completion verifier's (ENG-1081): narrating models
# (`mindshub_air`/kimi, `deepseek`) spend 245–1,654+ tokens on prose before
# reaching the forced tool call — and the narration scales with the input, so
# a consolidation pass over a whole scratchpad session was observed filling
# 2,048 exactly (ENG-1084). Non-narrating models answer these calls in tens
# of tokens and never pay for the headroom.
DEFAULT_STRUCTURED_BUDGETS: tuple[int, ...] = (2048, 4096)


def no_preamble_instruction(schema_class) -> str:
    """System-prompt suffix that asks for the forced tool call first.

    Shortens the preamble on narrating models but is NOT sufficient alone
    (measured 0/3 at a 256 budget even with it — ENG-1081), so it pairs with
    the budget ladder rather than replacing it. Tool name comes off the
    schema class so it can't go stale.
    """
    name = getattr(schema_class, "__name__", str(schema_class))
    return (
        f"\n\nCall the {name} tool immediately as your first action. Do not "
        "think out loud, restate the conversation, or explain your reasoning "
        "before calling it — put any brief justification in the tool's fields."
    )


def truncation_verdict(exc: StructuredOutputError) -> str:
    """The log verdict for a failed forced-schema call, naming its cure.

    Three outcomes, because the two truncated ones need opposite fixes and a
    single ``TRUNCATED`` could not tell them apart — which is what made
    ENG-1523 take a round-trip to diagnose. Shared by every log site so they
    cannot drift into reporting the same failure differently.
    """
    if not exc.truncated:
        return "NO_TOOL_CALL"
    return "TRUNCATED_INSIDE_CALL" if exc.reached_tool_call else "TRUNCATED_BEFORE_CALL"


async def generate_with_truncation_retry(
    generate: Callable[..., Awaitable[Any]],
    schema_class,
    *,
    system: str,
    messages: list[dict],
    budgets: tuple[int, ...] = DEFAULT_STRUCTURED_BUDGETS,
    log: logging.Logger | None = None,
    subsystem: str = "structured-output",
):
    """Run a forced-schema call through the ENG-1081 budget ladder.

    ``generate`` is a bound ``LLMClient.generate_object`` /
    ``generate_object_code``. Each budget in ``budgets`` is tried in order;
    a retry is bought ONLY for a truncated attempt
    (``StructuredOutputError.truncated`` — the model narrated past the budget
    before reaching the tool call). Any other failure — provider error,
    schema mismatch, a model that declines the call outright — is re-raised
    immediately: a bigger budget cannot fix it, so don't pay for one
    (measured: fable returns an unusable ``{}`` call at 8 output tokens
    regardless of budget, ENG-1095).

    Exists because ENG-1084 found seven `generate_object*` call sites that
    each hand-rolled a single tight budget with no retry — 512/600 sat
    *inside* the measured narration range, so identity extraction, the
    cerebellum diff pass and session consolidation silently returned nothing
    for every narrating-model user. One ladder, shared, instead of an eighth
    hand-rolled copy.

    Logging stays content-safe: budgets, token counts and the truncation
    verdict only — never the exception message, which can quote model output
    derived from the user's conversation.
    """
    logger = log or logging.getLogger(__name__)
    last_exc: StructuredOutputError | None = None
    for attempt, budget in enumerate(budgets):
        try:
            return await generate(
                schema_class, system=system, messages=messages, max_tokens=budget
            )
        except StructuredOutputError as exc:
            retrying = exc.truncated and attempt + 1 < len(budgets)
            logger.warning(
                "%s: forced %s call failed (%s, budget=%d, output_tokens=%s) "
                "retrying=%s",
                subsystem,
                getattr(schema_class, "__name__", schema_class),
                truncation_verdict(exc),
                budget,
                exc.output_tokens,
                retrying,
            )
            if not retrying:
                raise
            last_exc = exc
    # Only reachable with an empty `budgets`; treat as a caller bug but keep
    # the contract (raise, never return None silently).
    raise last_exc or StructuredOutputError(
        "generate_with_truncation_retry called with no budgets",
        truncated=False,
        output_tokens=0,
        max_tokens=0,
        stop_reason=None,
    )


def looks_truncated(response, budget: int) -> bool:
    """True if `response` was cut off by the `max_tokens` budget.

    Token count first: it is provider-agnostic and needs no dialect mapping.
    The clause exists because the MindsHub gateway once reported
    ``finish_reason: "stop"`` at the cap for most aliases (ENG-1082) — measured
    2026-08-11, it now reports ``"length"`` on all 19 chat aliases, streaming
    and non-streaming, so both gates fire. Keep both: the token count is the
    one that cannot silently regress. Both dialects are honoured when
    reported: OpenAI says ``"length"``, Anthropic ``"max_tokens"``.
    No usage information → ``False``; without evidence we don't buy a retry.
    """
    usage = getattr(response, "usage", None)
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    stop_reason = getattr(response, "stop_reason", None)
    return stop_reason in ("length", "max_tokens") or (
        budget > 0 and output_tokens >= budget
    )


def usable_tool_call(response) -> bool:
    """True when `response` carries tool calls and every one of them is intact.

    A tool call is what lets a round that hit the output cap still be worth
    using — but only if nothing it emitted was cut:

    - ``repaired`` marks arguments the repair pass patched back into valid JSON,
      which means the model's own body ended mid-value. It closes an open string
      or brace; it cannot invent the argument that never arrived.
    - ``parse_error`` marks a body it could not salvage at all.

    One damaged call makes the whole round unfinished, even alongside intact
    ones: using the intact half acts on part of what the model was still in the
    middle of asking for.

    Lives here rather than in the session because the reason it exists is
    shared: a repaired dict whose missing field happens to be optional
    validates cleanly, so no amount of schema validation downstream can catch
    it. The structured-output paths (`LLMClient.generate_object` and its sync
    twin) test `repaired` alone — a `parse_error` there is left to the
    validation branch, which classifies it only when the budget ran out.
    """
    calls = getattr(response, "tool_calls", None)
    return bool(calls) and not any(tc.parse_error or tc.repaired for tc in calls)


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
        StructuredOutputError: Always. ``.truncated`` says whether a retry can
            help and ``.reached_tool_call`` which of the two truncations it was
            — the message itself cannot be logged, it can quote model output.
    """
    usage = getattr(response, "usage", None)
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    stop_reason = getattr(response, "stop_reason", None)
    truncated = looks_truncated(response, budget)
    reached_tool_call = bool(getattr(response, "tool_calls", None))
    what = (
        "returned an unusable tool call for"
        if reached_tool_call
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
        reached_tool_call=reached_tool_call,
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
