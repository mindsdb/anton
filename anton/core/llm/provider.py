from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]
    # Populated only when the streamed tool-call arguments couldn't be
    # parsed as JSON (truncation mid-string, missing comma, etc.). The
    # session dispatcher reads this *before* invoking the handler — if
    # set, it short-circuits with a synthetic tool_result that asks the
    # LLM to re-emit the call with a complete body, instead of letting
    # the handler run with `input={}` and produce a confusing
    # "missing required field" trail. See `safe_parse_tool_input`.
    parse_error: str | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    context_pressure: float = 0.0


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None


@dataclass
class StreamTextDelta:
    text: str


@dataclass
class StreamToolUseStart:
    id: str
    name: str


@dataclass
class StreamToolUseDelta:
    id: str
    json_delta: str


@dataclass
class StreamToolUseEnd:
    id: str


@dataclass
class StreamComplete:
    response: LLMResponse


@dataclass
class StreamTaskProgress:
    """Progress event from agent task execution (planning, building, executing)."""

    phase: str
    message: str
    eta_seconds: float | None = None


@dataclass
class StreamToolResult:
    """Tool result that should be displayed to the user (e.g. scratchpad dump)."""
    name: str
    content: str
    action: str | None = None  # Relevant only for scratchpad tool calls.


@dataclass
class StreamContextCompacted:
    """Notification that context was compacted to free up space."""

    message: str


StreamEvent = (
    StreamTextDelta
    | StreamToolUseStart
    | StreamToolUseDelta
    | StreamToolUseEnd
    | StreamComplete
    | StreamTaskProgress
    | StreamToolResult
    | StreamContextCompacted
)


def _try_repair_tool_json(raw: str):
    """Permissive recovery pass for malformed streamed tool-call JSON.

    Many failures we see in practice are simple truncations: the model
    was cut off mid-call by a token cap and we ended up with a
    missing closing bracket / quote / comma. Brute-forcing a clean
    parse covers the easy cases without dragging in a heavyweight
    repair library:

      • Trim trailing junk after the last balanced point.
      • Close any unterminated string with a `"`.
      • Append `]` / `}` to balance open `[` / `{`.

    Returns the parsed dict on success, or None if even the repaired
    string is unparseable. Never raises.
    """
    if not raw:
        return None
    import json as _json

    s = raw.strip()
    # Track the bracket / brace stack and whether we're inside a
    # quoted string. The stack is only `{` and `[`. Backslash escapes
    # inside strings are honoured so `"\""` doesn't fool us.
    stack: list[str] = []
    in_string = False
    escape = False
    last_safe = 0  # index of the last '}' or ']' that closed back to depth 0
    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append(ch)
            continue
        if ch in "}]":
            if stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
                stack.pop()
                if not stack:
                    last_safe = i + 1
            else:
                # Mismatched closer — bail out, can't recover.
                return None

    # Try the simplest repair first: close the open string + open
    # containers in reverse order, drop a stray trailing comma.
    repaired = s
    if in_string:
        repaired += '"'
    # Strip trailing comma just before the synthetic closers, which is
    # the most common shape of "model was cut off after a comma".
    repaired = repaired.rstrip().rstrip(",")
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"

    try:
        parsed = _json.loads(repaired)
        return parsed if isinstance(parsed, dict) else None
    except _json.JSONDecodeError:
        pass

    # Fall back to "everything up to the last fully-balanced close" —
    # works when the model emitted a complete top-level object plus
    # garbage. Only useful when last_safe > 0.
    if last_safe > 0:
        try:
            parsed = _json.loads(s[:last_safe])
            return parsed if isinstance(parsed, dict) else None
        except _json.JSONDecodeError:
            pass
    return None


def safe_parse_tool_input(raw_json: str) -> tuple[dict, str | None]:
    """Parse the JSON body of a streamed `tool_use` call without
    crashing the turn when the assembled body is malformed.

    Anthropic + OpenAI both stream a tool call's input as a sequence
    of `input_json_delta` / `function.arguments` chunks that the
    provider client concatenates and `json.loads` at the end of the
    block. In long conversations, the model can be cut off mid-JSON
    (token cap, context overflow, network drop) so the assembled
    string is truncated — `}` missing, a string left unclosed,
    `[` with nothing after it. Calling `json.loads` then raises
    `JSONDecodeError`, the streaming pipeline tears down, and the
    user sees an opaque "JSON delimiter error" from the tool layer.

    Recovery cascades through three steps:

      1. Strict `json.loads`. Almost every well-formed call lands here.
      2. Permissive repair pass (`_try_repair_tool_json`) — closes
         unterminated strings, balances brackets, drops trailing
         commas. Catches the common "cut off mid-token" shape.
      3. Empty dict + `parse_error` populated.

    Returns ``(parsed_dict, parse_error_or_None)``. The session
    dispatcher reads ``parse_error`` to decide whether to invoke the
    tool handler (parse_error is None) or short-circuit with a
    structured tool_result that asks the LLM to re-emit the call
    (parse_error is set). Either way this function never raises.
    """
    if not raw_json:
        return {}, None
    import json as _json
    import logging as _logging

    try:
        parsed = _json.loads(raw_json)
    except _json.JSONDecodeError as exc:
        # Try the repair pass before giving up entirely.
        repaired = _try_repair_tool_json(raw_json)
        if repaired is not None:
            _logging.getLogger(__name__).info(
                "Tool-use input JSON was malformed (%s) but repaired "
                "successfully. Raw bytes: %d.",
                exc, len(raw_json),
            )
            return repaired, None
        _logging.getLogger(__name__).warning(
            "Tool-use input JSON was malformed and unrecoverable (%s). "
            "Raw bytes: %d, head: %r",
            exc, len(raw_json), raw_json[:160],
        )
        return {}, str(exc)
    # Anthropic occasionally emits a top-level scalar (e.g. a string
    # for a single-arg tool); coerce to a dict so callers always see
    # the same shape. Treat as a parse error so the dispatcher asks
    # for a re-emit instead of running the handler with an empty dict.
    if not isinstance(parsed, dict):
        return {}, f"tool input was not a JSON object (got {type(parsed).__name__})"
    return parsed, None


_CONTEXT_WINDOWS: list[tuple[str, int]] = [
    # Anton defaults (exact model IDs first)
    ("claude-sonnet-4-6", 200_000),
    ("claude-haiku-4-5-20251001", 200_000),
    # Claude families
    ("claude-opus-4", 200_000),
    ("claude-sonnet-4", 200_000),
    ("claude-haiku-4", 200_000),
    ("claude-3", 200_000),
    ("claude-", 200_000),
    # OpenAI families
    ("gpt-5", 400_000),
    ("gpt-4.1", 1_000_000),
    ("gpt-4o", 128_000),
    ("gpt-4", 128_000),
    ("o3", 200_000),
    ("o1", 200_000),
]
_DEFAULT_CONTEXT_WINDOW = 128_000


def compute_context_pressure(model: str, input_tokens: int) -> float:
    """Return input_tokens / context_window as a 0.0–1.0 float."""
    window = _DEFAULT_CONTEXT_WINDOW
    for prefix, size in _CONTEXT_WINDOWS:
        if model.startswith(prefix):
            window = size
            break
    return min(input_tokens / window, 1.0)


class ContextOverflowError(Exception):
    """Raised when the LLM rejects a request due to context length exceeded."""

    def __init__(self, message: str, input_tokens: int = 0, limit: int = 0):
        super().__init__(message)
        self.input_tokens = input_tokens
        self.limit = limit


class TokenLimitExceeded(Exception):
    """Raised when the LLM returns 429 due to billing/token limits."""


@dataclass
class ProviderConnectionInfo:
    """Serializable provider connection details.

    `api_key` is marked repr=False to reduce accidental leakage via logs/debugging.
    """

    provider: str
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    ssl_verify: bool | None = None
    api_version: str | None = None  # Azure api-version query param


class LLMProvider(ABC):
    # Human-readable provider id (e.g. "anthropic", "openai-compatible").
    name: str = ""

    @abstractmethod
    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...

    def export_connection_info(self) -> ProviderConnectionInfo:
        """Return provider connection details for other runtimes (e.g. scratchpad).

        Providers should override this to expose the minimal needed configuration
        without relying on SDK client internals.
        """
        return ProviderConnectionInfo(provider=self.name)

    async def stream(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamEvent]:
        """Stream LLM responses. Default falls back to complete()."""
        response = await self.complete(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
        )
        if response.content:
            yield StreamTextDelta(text=response.content)
        yield StreamComplete(response=response)
