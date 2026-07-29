from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anton.core.interaction.elicit import AskAnswer, AskRequest


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
    """Progress event from agent task execution (planning, building, executing).

    `id` carries the originating tool_use id when this progress event
    is a scratchpad phase marker (e.g. `scratchpad_start` /
    `scratchpad_done`). The frontend correlates the event to the
    specific step it advances; without it, multi-cell turns where the
    LLM queued several tool calls before execution would patch the
    wrong step (always the last one in the array).
    """

    phase: str
    message: str
    eta_seconds: float | None = None
    id: str | None = None


@dataclass
class StreamToolResult:
    """Tool result that should be displayed to the user (e.g. scratchpad dump).

    `id` is the originating tool_use id. Required so the frontend can
    correlate this result to the specific step that emitted the call —
    when a turn has multiple tool calls, all start/end events fire
    upfront (during the LLM stream) and only THEN do results arrive
    sequentially. Patching by "the last scratchpad step" silently
    misattributes the output of cell A to cell B.
    """
    name: str
    content: str
    action: str | None = None  # Relevant only for scratchpad tool calls.
    id: str | None = None


@dataclass
class StreamContextCompacted:
    """Notification that context was compacted to free up space."""

    message: str


@dataclass
class StreamAskUser:
    """A question the user must answer before the turn can continue.

    ``id`` is the question id the host echoes back with the answer — the
    originating ``tool_use.id`` for a tool question, a prefixed uuid for one
    raised by a sub-agent.
    """

    id: str
    request: "AskRequest"


@dataclass
class StreamAskUserAnswered:
    """Retires a previously published question.

    Emitted so a client replaying the buffer from the start does not show
    live buttons on a question that was already answered.
    """

    id: str
    answer: "AskAnswer"


StreamEvent = (
    StreamTextDelta
    | StreamToolUseStart
    | StreamToolUseDelta
    | StreamToolUseEnd
    | StreamComplete
    | StreamTaskProgress
    | StreamToolResult
    | StreamContextCompacted
    | StreamAskUser
    | StreamAskUserAnswered
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


def compute_context_pressure(model: str, input_tokens: int | None) -> float:
    """Return input_tokens / context_window as a 0.0–1.0 float.

    ``input_tokens`` may be ``None``: some providers omit a usage count for
    tool-augmented responses (e.g. the MindsHub passthrough returns
    ``usage.input_tokens = None`` on web-search responses). Treat a missing
    count as no measurable pressure rather than crashing on ``None / int``.
    """
    if not input_tokens:  # None or 0 → no measurable pressure
        return 0.0
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


class StructuredOutputError(ValueError):
    """Raised when a forced-tool-call structured-output call yields no usable call.

    ``truncated`` says whether a retry can help: ``True`` means the model spent
    the ``max_tokens`` budget narrating before it reached the tool call (the
    narrating aliases — ``mindshub_air``/``kimi``, ``deepseek``, ``qwen`` — do
    this deterministically under a tight budget, ENG-1081); ``False`` means the
    provider errored, refused, or returned nothing, and a bigger budget won't
    fix it. See `structured.looks_truncated` for how that's decided.

    Subclasses ``ValueError`` so call sites catching the documented ``ValueError``
    from ``generate_object``/``generate_object_code`` keep working.
    """

    def __init__(
        self,
        message: str,
        *,
        truncated: bool = False,
        output_tokens: int = 0,
        max_tokens: int = 0,
        stop_reason: str | None = None,
    ):
        super().__init__(message)
        self.truncated = truncated
        self.output_tokens = output_tokens
        self.max_tokens = max_tokens
        self.stop_reason = stop_reason


class TransientProviderError(ConnectionError):
    """Raised when the provider fails in a way that a retry might fix.

    Covers provider/infra-side hiccups — an ``overloaded_error``/``api_error``
    event (including the mid-stream case that arrives inside an HTTP-200 stream,
    see ENG-673), a 5xx, a plain 429 (no quota detail), a dropped connection, or
    a truncated stream. The session loop backs off and retries these within a
    bounded budget; distinct from auth/quota/model-gate failures, which are
    permanent for the identical request and must fail fast.

    Subclasses ConnectionError so legacy call sites that only know the
    ConnectionError mapping keep working unchanged.
    """

    def __init__(
        self, message: str, *, provider: str = "", code: str | None = None,
        retry_after: float | None = None, session_backoff: bool = True,
        model: str = "",
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.retry_after = retry_after
        # The model that was in flight when this failed — so a downstream
        # ProviderOverloadedError names the ACTUAL model (planning OR coding),
        # not whatever the session defaults to (ENG-673).
        self.model = model
        # Whether the SESSION should spend its backoff budget on this (ENG-673).
        # True for failures that had NO prior retry — a mid-stream error (arrives
        # inside an HTTP-200 stream), a dropped connection, or a truncated stream.
        # False for request-time HTTP errors (real 4xx/5xx): the SDK already
        # retried those with backoff, so the session fails fast (still with the
        # honest typed message) instead of stacking another 30s on top.
        self.session_backoff = session_backoff


class ProviderOverloadedError(ConnectionError):
    """Terminal form of a transient failure: the retry budget was exhausted.

    Carries the failing ``model`` + ``provider`` so cowork-server can render the
    ``provider_overloaded`` card (the MindsHub cross-provider-failover nudge).
    Typed consumers read ``code``/``model``/``provider``; it subclasses
    ConnectionError for legacy call sites.
    """

    def __init__(
        self, message: str, *, provider: str = "", model: str = "",
        code: str = "provider_overloaded",
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.code = code


# Body ``error.type``/``error.code`` values that mean "provider is momentarily
# failing" — retryable. The mid-stream overload arrives with one of these inside
# an HTTP-200 stream (ENG-673), so classification must read the body, not status.
_TRANSIENT_ERROR_TYPES = frozenset(
    {"overloaded_error", "overloaded", "api_error", "server_error", "service_unavailable"}
)


def classify_transient(
    status_code: int | None, body: Any, *, provider: str = "", model: str = ""
) -> "TransientProviderError | None":
    """Arm A of the transient classifier (see ENG-673): inspect an
    ``APIStatusError``'s status + body and return a ``TransientProviderError`` if
    it's a retryable provider/infra failure, else ``None``.

    Shared by both providers so the mapping can't drift. Call this only AFTER the
    permanent classifications (401 / 429-quota / 403 model-gate) have been ruled
    out — this decides between "retryable transient" and "generic unavailable".
    """
    b = body if isinstance(body, dict) else {}
    # Two body dialects: Anthropic nests the error under `error` ({"error":
    # {"type": ...}}); the OpenAI SDK unwraps its envelope (`body.get("error",
    # body)`) so the type sits at the TOP level. Read the nested object when
    # present, otherwise treat the body itself as the error object — so the
    # mid-stream case classifies on BOTH providers (ENG-673, Sam's review).
    err = b.get("error") if isinstance(b.get("error"), dict) else b
    etype = err.get("type") or err.get("code")
    # A mid-stream failure has no real HTTP error status (it's smuggled into an
    # already-sent 200, or there's no status at all), so the SDK never retried it
    # → the session must. A request-time error carries a real 4xx/5xx → the SDK
    # already retried → fail fast.
    session_backoff = status_code is None or status_code == 200
    if isinstance(etype, str) and etype in _TRANSIENT_ERROR_TYPES:
        # Explicit overload/api_error — the body is authoritative (status may be
        # a misleading 200 mid-stream, or a real 529 at request time).
        return TransientProviderError(
            f"{provider or 'The model provider'} is momentarily overloaded.",
            provider=provider, code=etype, session_backoff=session_backoff, model=model,
        )
    if isinstance(status_code, int) and 500 <= status_code < 600:
        return TransientProviderError(
            f"{provider or 'The model provider'} returned {status_code}.",
            provider=provider, code=f"http_{status_code}", session_backoff=False, model=model,
        )
    if status_code == 429 and not b.get("detail"):
        # Plain rate-limit ("slow down"), NOT an out-of-quota 429 (that carries a
        # `detail` and is mapped to TokenLimitExceeded upstream of this call).
        return TransientProviderError(
            f"{provider or 'The model provider'} is rate-limiting requests.",
            provider=provider, code="rate_limited", session_backoff=False, model=model,
        )
    return None


class ModelUnavailableError(ConnectionError):
    """Raised when the gateway rejects the requested model with a structured 403.

    The MindsHub gateway distinguishes two cases via ``error.code``:

    - ``model_access_denied`` — the caller's plan tier doesn't include the
      model (an upgrade fixes it).
    - ``model_disabled`` — an admin kill switch (an upgrade does NOT fix it).

    Subclasses ConnectionError so call sites that only know the legacy
    ConnectionError mapping keep working unchanged; typed consumers
    (cowork-server's turn-error mapping) read ``code``/``model`` to pick the
    right user-facing remedy instead of string-matching.
    """

    def __init__(self, message: str, *, code: str, model: str) -> None:
        super().__init__(message)
        self.code = code
        self.model = model


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

    def native_web_tools(self) -> set[str]:
        """Subset of {"web_search", "web_fetch"} this provider executes server-side.

        When a tool is declared here, the provider is responsible for translating
        the capability into its own native tool spec (e.g. Anthropic's
        ``web_search_*`` server-tool, OpenAI's Responses API ``web_search``,
        mdb.ai's ``{"type": "web_search"}`` passthrough). Server-side execution
        means the model's response already incorporates the search/fetch
        results — Anton's tool-dispatch loop never sees a ``tool_use`` for
        these names.

        Providers without native support return an empty set, and the session
        falls back to handler-dispatched ``ToolDef``s for any enabled web tools.
        """
        return set()

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
        native_web_tools: set[str] | None = None,
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
        native_web_tools: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream LLM responses. Default falls back to complete()."""
        response = await self.complete(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            native_web_tools=native_web_tools,
        )
        if response.content:
            yield StreamTextDelta(text=response.content)
        yield StreamComplete(response=response)
