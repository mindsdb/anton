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
    # True when the arguments JSON was malformed but the repair pass in
    # `safe_parse_tool_input` salvaged a parseable dict, so `parse_error` is
    # None and the handler *would* run. The dict is syntactically valid and
    # semantically unfinished: the repair closes an open string or brace, it
    # cannot invent the argument the model never emitted. Read in two places —
    # `usable_tool_call`, to retry a round the output cap cut off, and
    # `damaged_tool_call_result`, which refuses the call outright.
    repaired: bool = False


@dataclass
class Usage:
    """Token usage for one LLM call, normalized across providers (ENG-1288).

    Component semantics are UNIFORM regardless of provider:
    - ``input_tokens``: fresh (non-cached) prompt tokens only. Anthropic's
      ``input_tokens`` already excludes cache activity; OpenAI's
      ``prompt_tokens`` INCLUDES cached tokens, so the OpenAI provider
      subtracts ``cached_tokens`` out — without that, the same call would
      report different components depending on the wire format.
    - ``cache_read_tokens`` / ``cache_creation_tokens``: prompt tokens served
      from / written to the provider prompt cache. Both are populated on the
      OpenAI dialect too — our gateway publishes
      ``prompt_tokens_details.{cached_tokens, cache_write_tokens}`` — and are
      subtracted out of ``prompt_tokens`` by ``_split_cached_input``. A
      third-party endpoint that omits either field reports 0 for it.
    - Total context for a call = input + cache_read + cache_creation
      (cache tokens ARE context; dropping them understates a warm-cache
      call by ~10x).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    context_pressure: float = 0.0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def context_tokens(self) -> int:
        """Total prompt-side tokens the call carried (all three components)."""
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens


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

    `id` carries the originating tool_use id when this progress event is a
    scratchpad phase marker (e.g. `scratchpad_start` / `scratchpad_done`), a
    generic streaming tool's `tool_progress` marker, or that same tool's
    closing `tool_done` marker (see `ToolRegistry.dispatch_tool_stream`,
    `anton/core/tools/registry.py`). The frontend correlates the event to the
    specific step it advances; without it, multi-cell turns where the LLM
    queued several tool calls before execution would patch the wrong step
    (always the last one in the array).

    `ok` carries the tool's success/failure verdict on a `tool_done` marker
    for a generic tool — same tri-state as `ToolOutcome.ok` (`anton/core/
    tools/registry.py`): `None` when the handler hasn't declared a verdict
    (legacy/unclassified), `True`/`False` otherwise. A handler exception
    always forces `False` regardless of what it would have otherwise been.
    Without this, every consumer that renders `tool_done` as "done" (CLI
    activity line, cowork's step UI) has no way to tell a failed tool call
    from a successful one — the CLI printed a green checkmark for a tool
    that raised, because `tool_done` firing was the only signal it had, and
    it always fires now (ENG-763's own reliability fix made this reachable —
    see PR #304 review). Unset (`None`) on every other phase.
    """

    phase: str
    message: str
    eta_seconds: float | None = None
    id: str | None = None
    ok: bool | None = None


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

    ``id`` is the question id the host echoes back with the answer, and an
    opaque correlation/dedup key as far as the host is concerned: a minted
    uuid, prefixed by origin (``ask:``, ``path:``). It is deliberately NOT the
    originating ``tool_use.id``, which a tool handler cannot see —
    ``dispatch_tool`` passes only the tool name and input.
    """

    id: str
    request: AskRequest


@dataclass
class StreamAskUserAnswered:
    """Retires a previously published question.

    Emitted so a client replaying the buffer from the start does not show
    live buttons on a question that was already answered.
    """

    id: str
    answer: AskAnswer


@dataclass
class StreamReasoningDelta:
    """A chunk of the model's own extended-thinking/reasoning text.

    NOT part of the final answer — Anthropic's `thinking_delta` content
    blocks (surfaced via `output_config.effort`'s adaptive thinking) and
    OpenAI's `response.reasoning_summary_text.delta` Responses-API events
    both map to this. Kept distinct from `StreamTextDelta` so the harness
    layer can route it to a separate "current thought" channel instead of
    the persisted answer body.
    """

    text: str


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
    | StreamReasoningDelta
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

    Returns ``(parsed_dict, was_truncated)`` on success, or None if even
    the repaired string is unparseable. Never raises. The two recovery
    branches mean different things to the caller:

    - synthetic closers → the body ended mid-value, so an argument the
      model meant to send is missing or cut short (``was_truncated``).
    - trailing junk after a balanced top-level object → every argument
      arrived; only the tail is garbage.
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
    # containers in reverse order, drop a stray trailing comma. Only
    # attempt it when something was actually left open — with nothing to
    # close, the body is a complete value plus junk, which belongs to the
    # `last_safe` branch below and is not a truncation.
    if in_string or stack:
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
            return (parsed, True) if isinstance(parsed, dict) else None
        except _json.JSONDecodeError:
            pass

    # Fall back to "everything up to the last fully-balanced close" —
    # works when the model emitted a complete top-level object plus
    # garbage. Only useful when last_safe > 0.
    if last_safe > 0:
        try:
            parsed = _json.loads(s[:last_safe])
            return (parsed, False) if isinstance(parsed, dict) else None
        except _json.JSONDecodeError:
            pass
    return None


def safe_parse_tool_input(raw_json: str) -> tuple[dict, str | None, bool]:
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

    Returns ``(parsed_dict, parse_error_or_None, was_repaired)``. The
    session dispatcher reads ``parse_error`` to decide whether to
    invoke the tool handler (parse_error is None) or short-circuit
    with a structured tool_result that asks the LLM to re-emit the
    call (parse_error is set). ``was_repaired`` marks the truncating
    half of step 2 — a body that ended mid-value, as opposed to a
    complete object with junk after it, which parses to every argument
    the model sent and stays dispatchable. The dict is then parseable
    but built from an incomplete body, so a caller that
    knows *why* the body was cut — the session, which can see the
    round hit its output cap — can retry the round instead of running
    a handler on arguments the model never finished. Either way this
    function never raises.
    """
    if not raw_json:
        return {}, None, False
    import json as _json
    import logging as _logging

    try:
        parsed = _json.loads(raw_json)
    except _json.JSONDecodeError as exc:
        # Try the repair pass before giving up entirely.
        repair = _try_repair_tool_json(raw_json)
        if repair is not None:
            repaired, truncated = repair
            _logging.getLogger(__name__).info(
                "Tool-use input JSON was malformed (%s) but repaired "
                "successfully. Raw bytes: %d, truncated: %s.",
                exc, len(raw_json), truncated,
            )
            return repaired, None, truncated
        _logging.getLogger(__name__).warning(
            "Tool-use input JSON was malformed and unrecoverable (%s). "
            "Raw bytes: %d, head: %r",
            exc, len(raw_json), raw_json[:160],
        )
        return {}, str(exc), False
    # Anthropic occasionally emits a top-level scalar (e.g. a string
    # for a single-arg tool); coerce to a dict so callers always see
    # the same shape. Treat as a parse error so the dispatcher asks
    # for a re-emit instead of running the handler with an empty dict.
    if not isinstance(parsed, dict):
        return {}, f"tool input was not a JSON object (got {type(parsed).__name__})", False
    return parsed, None, False


def damaged_tool_call_result(tc: ToolCall) -> dict | None:
    """The `tool_result` to answer an unfinished tool call with, or None.

    None means the call is intact and may be dispatched. Two shapes are not:

    - ``parse_error`` — the arguments couldn't be parsed at all (a cut
      mid-string, a missing comma).
    - ``repaired`` — they parsed only after the repair pass closed an open
      string or brace, so the dict is valid JSON built from a body the model
      never finished. It cannot contain the argument that never arrived, and
      running a handler on it acts on half a request.

    Answering with a `tool_result` keeps the recovery inside the tool_use /
    tool_result protocol the model already understands, so no caller needs a
    retry of its own. Lives next to `safe_parse_tool_input`, which produces the
    two flags, and is shared by every tool loop — the session's streaming and
    non-streaming ones, and `agentic_loop` in the scratchpad subprocess — so a
    damaged call is refused identically whichever one runs.
    """
    if not (tc.parse_error or tc.repaired):
        return None
    reason = (
        f"failed to parse: {tc.parse_error}"
        if tc.parse_error
        else "arrived incomplete — the body ended mid-value and was only closed "
             "off by a repair pass, so at least one argument is missing or cut short"
    )
    return {
        "type": "tool_result",
        "tool_use_id": tc.id,
        "content": (
            f"Tool call arguments {reason}. This is most often a token-cap "
            "truncation mid-call. Re-emit this call with a complete, valid JSON "
            "body; if an argument was large, make it smaller or split the work "
            "across several calls."
        ),
        "is_error": True,
    }


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

    ``truncated`` says whether a retry can help: ``True`` means the
    ``max_tokens`` budget ran out before a usable call existed; ``False`` means
    the provider errored, refused, or returned nothing, and a bigger budget
    won't fix it. See `structured.looks_truncated` for how that's decided.

    ``reached_tool_call`` splits the truncated case into the two cures, which
    otherwise look identical in a log and pull in opposite directions:

    - ``False`` — the model narrated in plain ``content`` and never got to the
      call (the narrating aliases — ``mindshub_air``/``kimi``, ``deepseek``,
      ``qwen`` — do this deterministically under a tight budget, ENG-1081).
      The cure is a bigger budget.
    - ``True`` — the call started and the budget ran out inside its JSON
      arguments. A bigger budget only helps until the payload grows again; the
      cure is a smaller response (ENG-1523).

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
        reached_tool_call: bool = False,
    ):
        super().__init__(message)
        self.truncated = truncated
        self.output_tokens = output_tokens
        self.max_tokens = max_tokens
        self.stop_reason = stop_reason
        self.reached_tool_call = reached_tool_call


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
        code: str = "provider_overloaded", retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.code = code
        # Seconds the server said to wait, when it said so (ENG-1537). Set only
        # on the rate-limit exhaustion path, where the wait was skipped or ran
        # out and the card needs to name the interval rather than say
        # "a moment".
        self.retry_after = retry_after


# Body ``error.type``/``error.code`` values that mean "provider is momentarily
# failing" — retryable. The mid-stream overload arrives with one of these inside
# an HTTP-200 stream (ENG-673), so classification must read the body, not status.
_TRANSIENT_ERROR_TYPES = frozenset(
    {"overloaded_error", "overloaded", "api_error", "server_error", "service_unavailable"}
)

# The MindsHub M3 authorization gate's out-of-credits deny codes (ENG-1169):
# ``wallet_empty`` rides a 402, ``included_allowance_exhausted`` a 429 (with NO
# FastAPI ``detail``, so the legacy 429-quota branch never sees it). Both are
# permanent for the identical request — they belong on the out-of-credits card,
# never in the retry loop. The gate's velocity 429 (``rate_limited``) is NOT
# here on purpose: that one means "slow down", and stays transient.
_WALLET_DENIAL_CODES = frozenset({"wallet_empty", "included_allowance_exhausted"})


# Hosts that ARE the MindsHub gateway. Only a response from one of these may
# select a billing verdict — see :func:`origin_is_known_third_party` (ENG-1693).
#
# VERIFIED COMPLETE for every host MindsHub ITSELF serves (review question on
# #363, checked 2026-08-18 against the terraform host inventory, which is the
# source of truth for zones and certs). Deliberately not the broader claim
# "every host that can serve a gateway billing denial" — see the relay note
# below, where that is false by design. Two apexes cover the served set because
# every environment and vanity host is a subdomain of one of them:
# prod `api.mindshub.ai`, `api.staging.mindshub.ai`, `api.dev.mindshub.ai`,
# per-PR envs `api-pr<N>.dev.mindshub.ai`, and the white-label surfaces
# `llm.mdb.ai`, `llm.staging.mdb.ai`, `writer.mdb.ai`, `terabase.dev.mdb.ai`,
# `view.mindshub.ai`. Terraform declares no other apex zone, and the wildcards
# `*.mindshub.ai` / `*.mdb.ai` cover anything added later under them. The
# positive cases are pinned in tests/test_status_error_mapper.py so this
# paragraph cannot quietly go stale.
#
# A RELAY in front of MindsHub is the known, accepted exception: a corporate
# proxy that forwards our denials resolves its own host, so a GENUINE
# out-of-credits denial arriving through one loses the credits card and
# ENG-1169's symptom returns for that shape. That is a decision, not an
# oversight — a spoofed billing card asks the user for money, a spoofed wait
# does not, which is why `_velocity` is ungated and this is not. Recorded on
# ENG-1693 under "Accepted tradeoff". If relayed MindsHub ever becomes a
# supported deployment the remedy is a CONFIGURABLE trusted-host list, never a
# looser match here.
#
# `4nton.ai` is a real production zone and is DELIBERATELY absent: it serves
# agent provisioning and per-instance hosts (`sp_<hash>.4nton.ai`,
# `cw-<id>.4nton.ai`) plus artifact publishing, never LLM inference. If an
# inference endpoint is ever routed onto it, it MUST be added here — otherwise a
# genuine out-of-credits denial from that host silently degrades to generic copy
# and reopens ENG-1169's user-visible bug. Failing closed is the right default;
# this note exists so the cost of that default is not discovered in production.
_MINDSHUB_HOSTS = ("mindshub.ai", "mdb.ai")


def is_mindshub_host(host: str | None) -> bool:
    """Whether ``host`` is the MindsHub gateway or one of its subdomains.

    Deliberately NOT the ``"mindshub.ai" in base_url`` substring test used
    elsewhere in this package for flavour detection and trace-header opt-in.
    That form is fine for choosing an API dialect and unfit for a trust
    decision: ``mindshub.ai.evil.com`` satisfies it. This matches the exact
    domain or a dot-delimited subdomain of it, so an attacker cannot buy a
    name that merely contains ours.
    """
    if not host:
        return False
    h = str(host).strip().lower().rstrip(".")
    return any(h == d or h.endswith("." + d) for d in _MINDSHUB_HOSTS)


def response_origin_host(exc: BaseException) -> str | None:
    """Hostname the failing request was sent to, or ``None`` if unknowable.

    ``httpx.Response.url`` is a property that RAISES when no request is
    attached, so this cannot be a bare ``getattr`` chain.

    Falls back to the exception's OWN ``request``, which is what makes this
    work on the mid-stream lane. A mid-stream failure surfaces as a bare
    ``openai.APIError`` — constructed as ``APIError(message, request, body=…)``,
    so it has **no** ``.response`` but does carry ``.request`` with the real
    URL. Reading only ``.response`` made the gate silently inert exactly where
    an attacker has the freest hand: answering 200 and smuggling the wallet
    code into an SSE frame is the remote's choice, not a quirk of our plumbing.
    """
    resp = getattr(exc, "response", None)
    try:
        url = None
        if resp is not None:
            # BOTH `httpx.Response.url` and `httpx.Response.request` are
            # properties that RAISE RuntimeError when no request is attached, and
            # `getattr(resp, name, None)` does NOT rescue that — getattr's
            # default only covers AttributeError. Either one reaching the outer
            # handler returned None, i.e. "origin unknown", a state this gate
            # deliberately TRUSTS — so a request-less response shadowed a
            # foreign host sitting on `exc.request`. Each read is contained
            # individually so the next fallback stays reachable.
            #
            # The review nit on #363 named `.url`; `.request` has the identical
            # trap, and guarding only `.url` still failed the test below.
            try:
                url = resp.url
            except Exception:
                url = None
            if url is None:
                try:
                    url = resp.request.url
                except Exception:
                    url = None
        if url is None:
            url = getattr(getattr(exc, "request", None), "url", None)
        if url is None:
            return None
        host = getattr(url, "host", None)
        if not host:
            from urllib.parse import urlparse

            host = urlparse(str(url)).hostname
    except Exception:
        return None
    return str(host).lower() if host else None


def origin_is_known_third_party(exc: BaseException) -> bool:
    """Whether this failure provably came from somewhere that is NOT our gateway.

    Three-valued on purpose, mirroring cowork-server's gate (ENG-1686): it
    answers "do we KNOW it was someone else", so an unknown origin stays
    trusted rather than being treated as hostile. A real SDK error always
    carries its request, so every genuine HTTP response resolves a host;
    unknown origin means a synthetic or mid-stream error, which no remote
    server can choose.

    Used to stop a BYOK endpoint selecting a MindsHub billing verdict by
    echoing our private ``X-MindsHub-Reason`` header or wallet ``code``
    (ENG-1693). NOT used in :func:`classify_transient`'s wallet check — see
    the comment there.
    """
    return origin_is_known_third_party_host(response_origin_host(exc))


def origin_is_known_third_party_host(host: str | None) -> bool:
    """Host-level form of :func:`origin_is_known_third_party`."""
    return host is not None and not is_mindshub_host(host)


def wallet_denial_code(body: Any) -> str | None:
    """The M3 gate's out-of-credits code carried in an error body, if any.

    Reads ``code`` from both dialects — the SDK-unwrapped top level (OpenAI
    SDK peels the ``error`` envelope, ENG-747) and the wire envelope
    (Anthropic SDK / proxies that deliver it unmodified). Detection is
    code-exact on purpose: BYOK 402s (e.g. OpenRouter's insufficient-credits
    402) carry no such code and must stay generic — the remedy there is the
    user's own provider billing, not MindsHub credits.
    """
    b = body if isinstance(body, dict) else {}
    err = b.get("error") if isinstance(b.get("error"), dict) else {}
    code = b.get("code") or err.get("code")
    # isinstance first: `in` on a frozenset HASHES the value, so a hostile/buggy
    # endpoint sending a list `code` would otherwise TypeError the classifier
    # (every other wire-value membership check in these mappers uses tuples).
    return code if isinstance(code, str) and code in _WALLET_DENIAL_CODES else None


def retry_after_seconds(exc: BaseException) -> float | None:
    """Seconds from a response's ``Retry-After`` header, or ``None`` (ENG-1537).

    The MindsHub gateway sends this on the velocity 429 (``rate_limited``,
    `minds/inference/errors.py`) as integer seconds — the only form we act on.
    The HTTP-date form is legal but nothing in use emits it, and mis-reading a
    date as a number would produce an absurd delay, so an unparseable value is
    treated as absent: the caller then falls back to its own backoff curve.

    Negative and non-finite values are dropped for the same reason. Zero is
    meaningful ("retry now") and is preserved by the caller's ``> 0`` checks
    behaving as "no hint", which is the same outcome.
    """
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        secs = float(str(raw).strip())
    except (TypeError, ValueError):
        return None  # HTTP-date form, or junk
    if secs != secs or secs in (float("inf"), float("-inf")) or secs < 0:
        return None
    return secs


def classify_transient(
    status_code: int | None,
    body: Any,
    *,
    provider: str = "",
    model: str = "",
    retry_after: float | None = None,
    velocity_confirmed: bool = False,
) -> "TransientProviderError | None":
    """Arm A of the transient classifier (see ENG-673): inspect an
    ``APIStatusError``'s status + body and return a ``TransientProviderError`` if
    it's a retryable provider/infra failure, else ``None``.

    Shared by both providers so the mapping can't drift. Call this only AFTER the
    permanent classifications (401 / 429-quota / 403 model-gate) have been ruled
    out — this decides between "retryable transient" and "generic unavailable".

    ``retry_after`` is the parsed ``Retry-After`` hint (see
    :func:`retry_after_seconds`); it is attached to the velocity-429 result so
    the session waits the interval the server actually named (ENG-1537).

    ``velocity_confirmed`` says the caller POSITIVELY identified a velocity
    limit — our gateway's ``rate_limited`` reason header or body code. Only then
    does the 429 earn a session wait. The absence of billing carriers is not
    evidence of transience: both fail-fast guards below are string-exact, so a
    provider whose quota denial uses a different dialect (Gemini sends an
    INTEGER ``code`` with ``status: RESOURCE_EXHAUSTED``) slips past them and
    would otherwise spend the whole budget waiting out a daily quota that resets
    at midnight — then be told it is not a credits problem. Unconfirmed 429s
    keep the pre-ENG-1537 behaviour: typed, honest, and failed fast.
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
        # Plain rate-limit ("slow down"), NOT an out-of-quota 429. Quota 429s are
        # mapped upstream (gateway dialect carries a `detail`, OpenAI's carries
        # ``insufficient_quota``, the M3 gate's allowance 429 carries a wallet
        # code); the guards here are defense for direct callers — a billing
        # failure is permanent and must never enter the retry loop (ENG-1169).
        if etype == "insufficient_quota":
            return None
        if wallet_denial_code(b):
            # Deliberately NOT origin-gated (ENG-1693), for a plainer reason
            # than an earlier version of this comment claimed. It said gating
            # would make a hostile wallet code "retryable"; that is false —
            # both this branch and the fallthrough reach the same count-based
            # retry, so retryability is unchanged either way. The real reasons
            # are that this call site cannot see the origin (it receives only
            # status + body, never the exception), and that suppressing a
            # retry is conservative regardless of who sent the code: retrying
            # a third party's quota denial cannot succeed either.
            return None
        # session_backoff=True, unlike every other request-time status here
        # (ENG-1537). The flag means "should the SESSION spend its budget on
        # this?", and the SDK's own 2 retries fire seconds apart — the right
        # answer for a 5xx that recovers instantly, and useless against a
        # per-minute token ceiling. Leaving it False sent this down the
        # count-based path, which re-issued the request TWICE with no delay and
        # a recovery note appended each time: told "too many tokens per
        # minute", we immediately sent more. This is the one failure class
        # where waiting is both necessary and sufficient, so it waits — for the
        # interval the server named, when it named one.
        return TransientProviderError(
            f"{provider or 'The model provider'} is rate-limiting requests.",
            provider=provider, code="rate_limited",
            session_backoff=velocity_confirmed,
            retry_after=retry_after if velocity_confirmed else None,
            model=model,
        )
    return None


def raise_on_empty_response(
    *, content: str, tool_calls: list, stop_reason: str | None,
    provider: str = "", model: str = "",
) -> None:
    """Fail loud on an empty 200: no content, no tool calls, no stop reason.

    The non-streaming mirror of the streaming truncated-response guard (ENG-673).
    An empty-from-start 200 is:

    - a weak incident signal (real mid-incident silence surfaces as a dropped
      connection / read timeout, which back off above), and
    - a strong broken/misconfigured-endpoint signal.

    So it fails fast (``session_backoff=False``) rather than looping the retry
    budget — and, critically, raises instead of handing back an empty
    ``LLMResponse`` the agent would misdiagnose as a backend outage.
    """
    # Empty-string stop_reason ("" — no real provider sends it) is treated as
    # absent, same as None: a truthiness check keeps the guard from being fooled.
    if content or tool_calls or stop_reason:
        return
    raise TransientProviderError(
        f"{provider or 'The model provider'} returned an empty response — try again in a moment.",
        provider=provider or "The model provider", code="empty_response",
        session_backoff=False, model=model,
    )


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


def classify_404(
    model: str,
    *,
    message: str | None,
    code: str | None = None,
    status: str | None = None,
    error_type: str | None = None,
) -> "ModelUnavailableError | EndpointConfigurationError":
    """Classify a bare 404 as model-not-found vs. endpoint-misconfiguration.

    Shared by the OpenAI-compatible and Anthropic status-error mappers
    (ENG-1139) so the heuristic — and the exact non-duplicated wording —
    can't drift between them. A bare 404 does NOT by itself mean the model
    is missing: a wrong base URL, a missing ``/v1``, a reverse-proxy route,
    or an unsupported API path all 404 too, and there "switch models" is
    the wrong remedy (ENG-1145). Only treat it as model-not-found when the
    structured body actually points at the model: OpenAI's
    ``code="model_not_found"``, Anthropic's ``error.type="not_found_error"``,
    or a model-oriented message (Gemini's ``status="NOT_FOUND"`` with
    "models/<id> is not found / no longer available"). Everything else is
    surfaced as an endpoint/configuration failure carrying the provider's
    own words.
    """
    msg_l = message.lower() if isinstance(message, str) else ""
    status_str = (status or "").upper()
    model_specific = (
        code == "model_not_found"
        or error_type == "not_found_error"
        or (
            "model" in msg_l
            and (
                status_str == "NOT_FOUND"
                or "not found" in msg_l
                or "not available" in msg_l
                or "no longer available" in msg_l
                or "does not exist" in msg_l
            )
        )
    )
    # Provider message as a leading-space fragment with a normalized
    # terminator, so the appended copy reads cleanly whether or not the
    # provider punctuated its own message (Gemini's ends in a period; a raw
    # proxy/FastAPI detail may not).
    clean = message.strip() if isinstance(message, str) else ""
    if clean and clean[-1] not in ".!?":
        clean += "."
    suffix = f" {clean}" if clean else ""
    if model_specific:
        reason = f":{suffix}" if suffix else "."
        return ModelUnavailableError(
            f"The model '{model}' isn't available{reason} Switch models in Settings.",
            code="model_not_found", model=model,
        )
    # Not model-specific → almost always a misrouted/misconfigured endpoint
    # (bad base URL, missing /v1, proxy route). Permanent for this request,
    # but the remedy is the endpoint config, not the model — a distinct type
    # so the CLI defaults it to `setup` (fix provider/endpoint), not `retry`,
    # and never to "switch models" (ENG-1145 review).
    return EndpointConfigurationError(
        f"The model endpoint returned 404 — check the endpoint URL and model "
        f"configuration.{suffix}"
    )


class EndpointConfigurationError(ConnectionError):
    """Raised when a request fails in a way that points at the endpoint
    configuration — a wrong base URL, a missing ``/v1``, a reverse-proxy route,
    or an unsupported API path — rather than at the model or a transient outage.

    Permanent for the identical request (a retry re-sends it to the same broken
    route), and the remedy is the provider *setup* flow (fix the base URL /
    route), NOT switching models and NOT waiting. Subclasses ConnectionError so
    legacy call sites that only know the ConnectionError mapping keep working;
    the interactive CLI reads the type to default such a failure to ``setup``
    rather than ``retry`` (ENG-1145 review).
    """


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
