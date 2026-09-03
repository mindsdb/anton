"""Per-turn token cost accounting (ENG-1288).

The turn is the measured unit: one ``turn_stream()``/``turn()`` invocation
end to end, including planning and tool-loop calls, the completion
verifier's verdict calls, verifier-forced continuations, hand-back
diagnosis streams, and in-turn history compaction. The session installs
``TurnCost.add`` as the LLMClient's ``usage_listener`` at turn start, so
every call that flows through the client is counted at one narrow waist
instead of at each call site.

Consumers, in order:
- observability: one structured log line + one ``turn_completed`` analytics
  event per turn (see ``ChatSession._emit_turn_cost``). The event relays into
  PostHog project 355390, where each field lands as a queryable property —
  one event per turn also makes turn VOLUME countable, not just cost. The
  field names are therefore a published schema: renaming one breaks queries
  downstream. Identity there is per-install (``aid``), so per-user cost needs
  the ``conversation_id`` -> Langfuse hop;
- ENG-1286's per-turn spend ceiling: reads ``total_tokens`` mid-turn.

Component semantics follow ``Usage`` (normalized across providers): input
is fresh prompt tokens, cache reads/writes are separate, and total context
for a call is their sum. Kept raw and separate deliberately — consumers
weight them differently (cache reads are ~10x cheaper for dollars but count
full-weight against the gateway TPM limiter).

Stated gaps (by design, documented on ENG-1288):
- usage on errored/aborted calls may be absent from the provider — the turn
  undercounts rather than crashes;
- LLM use inside scratchpad user code (``get_llm``) happens in a separate
  process and is never seen here;
- post-turn background work (cerebellum flush, identity extraction) runs
  after the turn's books close and is not attributed to any turn;
- retried calls count every attempt — retries cost real money.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

from anton.core.llm.provider import Usage


# The roles the event reports — a closed set, which is what lets the event carry
# per-role figures as flat properties. `TurnCost.add` folds any role outside it
# (empty, None, novel, or a case/whitespace variant) into `unknown`, so the
# per-role token sum always equals the turn total.
#
# `unknown` is EXPECTED to stay at zero, not guaranteed to: a non-zero value is
# the alarm that some caller invented a role, and the fold is what keeps that
# alarm's tokens from disappearing instead. The folded role's name is lost, but
# the log line keeps the model (`unknown=haiku:320/1`), which is usually enough
# to find the caller.
UNKNOWN_ROLE = "unknown"
EVENT_ROLES = ("planning", "coding", "router", UNKNOWN_ROLE)


@dataclass
class RoleCost:
    """One role's slice of a turn: which model ran it and what it spent.

    Roles are a closed set (planning / coding / router), which is why the turn
    event can carry this as flat properties instead of a nested blob.

    ``model`` is the alias anton REQUESTED. The gateway may resolve or fail
    over to a different model server-side; it does echo the resolved id on the
    response (``LLMResponse.model``, ENG-1638 — verified live: ``mindshub_air``
    comes back as ``gpt-5.6-luna``), but this field deliberately keeps the
    requested alias so the per-model dollar math stays keyed on what the user
    picked and can be joined to the catalog. Langfuse holds the served id.
    Normally one model per role per turn; if a role somehow sees more, they are
    joined with ``|`` so the ambiguity is visible rather than silently dropping
    one (a joined value means per-model dollar math for that role is unsafe).
    """

    model: str = ""
    tokens: int = 0
    calls: int = 0

    def observe(self, model: str, tokens: int) -> None:
        if model and model not in self.model.split("|"):
            self.model = f"{self.model}|{model}" if self.model else model
        self.tokens += tokens
        self.calls += 1


@dataclass
class TurnCost:
    """Running token totals for one turn. Mutated only via ``add`` plus the
    session's explicit shape/outcome writes (rounds, continuations, ended_by).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    llm_calls: int = 0
    # Largest single-call context (input + cache_read + cache_creation) seen
    # this turn. Together with ``rounds`` this classifies expensive turns by
    # shape: few rounds x huge context (bloat) vs many rounds x modest
    # context (thrash) — cost = rounds x context (ENG-578).
    peak_context_tokens: int = 0
    rounds: int = 0
    continuations: int = 0
    # Terminal path of the turn. Default holds unless the session marks a
    # specific exit; ``_emit_turn_cost`` overrides with cancelled/error when
    # the turn didn't end on its own terms.
    ended_by: str = "completed"
    # Exception class that ended the turn. `ended_by` alone cannot say WHY:
    # `error` is a catch-all and `retry_exhausted` discards the exception it
    # already formats into chat history, so together ~12% of real turns booked
    # a failure with no diagnosis (ENG-1689). Empty for every terminal that is
    # not an exception — `completed`, `cancelled`, `spend_ceiling`, and
    # `handback_verifier_failure`, which is a verdict rather than a raise.
    error_type: str = ""
    # The completion verifier was applicable but produced no verdict this turn
    # (denied by billing/model access, or suppressed by the verifier latch —
    # ENG-1632). Such turns deliberately book ended_by="completed" (the work
    # succeeded; a priced-out check is not a broken turn), so WITHOUT this flag
    # they are byte-identical in analytics to turns that were verified and
    # passed — silently joining the honest-stop denominator, concentrated in
    # exactly the wallet-locked cohort a measurement would want to isolate.
    # Stamped at the skip/deny sites, never read from live session state at
    # emit (late finalizers would read a later turn's latch state — see the
    # note below on stamped-at-open fields). Handback terminals don't need it:
    # their ended_by already distinguishes them.
    verification_skipped: bool = False
    # One-time grace(s) granted this turn: "", "round", "ceiling", or
    # "ceiling,round" — alphabetical, not firing order (see `_record_grace`).
    # Without this a turn that used a grace and finished is
    # byte-identical in `turn_completed` to one that never neared a limit —
    # `ended_by=round_cap`/`spend_ceiling` counts drop while actual spend
    # rises, unmeasurably.
    grace_granted: str = ""
    # Size of the spend-ceiling grace actually granted, 0 if none. Queryable
    # alongside `grace_granted` so the size of the concession is visible.
    grace_tokens: int = 0
    # WHY the completion verifier produced no verdict this turn (ENG-1858).
    # `ended_by="handback_verifier_failure"` and `verification_skipped=True`
    # both say only THAT it failed; 343 such turns / 14 days (5.3% of real
    # turns, 3x the tokens of a completed one) had no groupable cause. Values
    # are the classification the verdict loop already draws for the latch —
    # `truncated` / `transient` / `hard` / `denied` — plus `latched_hard` /
    # `latched_truncated` / `latched_mixed` / `latched_denied` for turns that
    # never made the call because an earlier one latched. `latched_mixed` means
    # the counted failures were not all the same class, so neither names the
    # cause on its own. Empty when a verdict was produced or the verifier was
    # not applicable. Stamped at the loop's exits, never read from latch
    # state at emit (late finalizers would see a later turn's latch).
    verifier_failure: str = ""
    # Exception TYPE of the last verdict attempt — `_safe_error_type`, i.e.
    # the class name only, never the message (which can quote model output
    # derived from the user's conversation). `StructuredOutputError` carries
    # a `:no_call` / `:unusable_call` suffix: the model never produced the
    # forced call vs produced one the schema rejected — the split ENG-1095
    # needs. Empty when no exception ended the verdict (verdict produced, or
    # latched skip with no call made).
    verifier_error_type: str = ""
    # WHY the retry flow terminated, when a turn ended after retrying (ENG-1361).
    # Named for TERMINATION, not exhaustion: `rate_limit_wait_too_long` is a
    # terminal where nothing ran out — the server named an interval past our cap
    # and we declined to wait — so an "exhaustion" name would be a lie on a
    # quarter of its own values.
    #
    # It exists because ENG-1361 makes the request-time and mid-stream terminals
    # raise the SAME ProviderOverloadedError, and `code` (which splits the
    # rate-limit case) is a card contract that never reaches analytics. Without
    # this, three distinct outcomes become one indistinguishable bucket. Note
    # the retry count itself is a local in `turn_stream` and reaches nothing —
    # there is no other telemetry saying a turn retried at all.
    #   "request_attempt_limit"     — the count-based attempt budget ran out
    #   "provider_recovery_timeout" — the mid-stream incident budget ran out
    #   "rate_limit_wait_limit"     — the rate-limit wait allowance ran out
    #   "rate_limit_wait_too_long"  — Retry-After exceeded our cap; we carded
    #                                 immediately rather than stalling the turn
    # Empty for every terminal that did not retry.
    retry_terminal_reason: str = ""
    # WHAT kind of provider failure ended the turn — a closed vocabulary
    # (`PROVIDER_FAILURE_KINDS`), NOT the raw exception `code` (ENG-1361).
    # `code` selects the client's card and mixes cardinalities (`http_503`
    # beside `connection_error`), so it is unfit for a groupable analytics
    # dimension. Empty when the failure was not a provider failure.
    provider_failure_kind: str = ""
    # The HTTP status the failure was classified from, when there was one.
    # `int | None`, never "" — a mid-stream failure (status 200) and a
    # connection error (no response) are genuinely ABSENT, and an empty string
    # beside integers is the shape that makes an analytics column unqueryable.
    # Omitted from the event entirely when None, rather than sent as a
    # placeholder.
    provider_http_status: int | None = None
    started_monotonic: float = field(default_factory=time.monotonic)
    # Set when these books have been reported. Replaces "the shared slot is
    # None" as the double-emit guard, because a late finalizer now emits the
    # books it was handed even when a newer turn already owns the slot
    # (#309 review).
    emitted: bool = False
    # Per-turn facts stamped at books-OPEN, not read at emit. A late finalizer
    # emits books whose turn ended long ago, and reading live session state
    # there gave the abandoned turn a LATER, unrelated turn's index — the
    # cost→Langfuse hop then pointed at the wrong turn (#309 review follow-up).
    turn_index: int = 0
    # Unique per turn EXECUTION. `turn_index` is only a POSITION in the history
    # — `_turn_count` is seeded by counting the user messages the session was
    # handed — and cowork-server rebuilds the session every turn, so a retried
    # or cancelled attempt arrives with the same history and stamps the same
    # `turn_index` (ENG-2243). Measured on prod 2026-08-28..09-01: that collided
    # on 14.5% of desktop turn keys (worst: 16 rows on one key, spanning 34
    # hours) and left 18.5% of `tool_completed` rows joining to more than one
    # `turn_completed` row — the join ENG-1486 stamped the pair to enable.
    #
    # Random, not a counter, deliberately: ANY session-local counter resets on
    # that same rebuild, so nothing derived from session state can be unique
    # across attempts. `token_hex(8)` is 16 hex chars of 64 real bits —
    # collision-free at our volume (the birthday bound is ~5e9 attempts), short
    # enough to read in a log line.
    #
    # NOT `uuid.uuid4().hex[:16]`, which this field shipped as in review and
    # which is 60 bits, not 64: hex position 12 is uuid4's version nibble, so
    # it is the literal `4` in every id ever generated (measured 2000/2000).
    # The width claim was wrong and so was the comparison to the `aid` install
    # fingerprint: `get_installation_id()` in `anton/analytics.py` takes the
    # `sha256(str(node)).hexdigest()[:16]` branch whenever the machine has a
    # real MAC, which is genuinely 64 bits; its `uuid4().hex[:16]` branch is
    # the fallback for a host with no MAC to read (Docker with stripped
    # networking), and carries the same 60-bit shortfall described above.
    #
    # A `default_factory` rather than a value passed in at each construction
    # site: there are two today (the streaming and non-streaming turns in
    # `session.py`) and a third that forgot would silently reintroduce the
    # collision this field exists to remove. Also makes it a stamped-at-open
    # fact like `turn_index` above, so a late finalizer reports the id of the
    # turn whose books it holds rather than whichever turn owns the slot now.
    attempt_id: str = field(default_factory=lambda: secrets.token_hex(8))
    # When the turn ended. None until resolved. A late finalizer cannot know the
    # real end, so it falls back to `last_activity_monotonic` (the last LLM
    # call) — understating but bounded, rather than measuring up to whenever
    # asyncio happened to run the finalizer, which inflates the field a runaway
    # query sorts on. Known residual: a turn abandoned BEFORE its first LLM call
    # has no watermark, so its duration still runs to finalizer time. Accepted —
    # it has `llm_calls == 0` and `tokens_total == 0`, so it is trivially
    # excluded from any cost or runaway query, which is the only consumer that
    # reads duration.
    ended_monotonic: float | None = None
    last_activity_monotonic: float | None = None
    # Per-role attribution. A turn routinely mixes an expensive planning model
    # with a cheap coding model (the completion verifier runs on the latter),
    # so a single blended token total cannot be priced at any one rate —
    # dollars are only computable from (model, tokens) pairs, which this
    # provides. Also the only place the router model is recorded at all.
    by_role: dict[str, RoleCost] = field(default_factory=dict)

    def add(self, role: str, model: str, usage: Usage) -> None:
        """Count one LLM call. Installed as ``LLMClient.usage_listener``.

        Records both the turn total and the per-``role`` slice, so the event
        can answer "which model was this user actually on" and "what did each
        model cost" rather than only a blended sum.
        """
        self.llm_calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_tokens
        self.cache_creation_tokens += usage.cache_creation_tokens
        self.peak_context_tokens = max(self.peak_context_tokens, usage.context_tokens)
        self.last_activity_monotonic = time.monotonic()
        # Any role the event doesn't emit — empty OR novel (a future caller
        # passing e.g. "verifier") — is folded into `unknown`. Keying on the
        # raw role instead would put those tokens in a bucket nothing reads:
        # they'd vanish from the per-role breakdown AND leave `unknown` at 0,
        # so the reconciliation would break with no alarm (#309 review).
        # The role's *name* is lost when folded; its tokens are not, and that
        # is the invariant that has to hold.
        key = role if role in EVENT_ROLES else UNKNOWN_ROLE
        slice_ = self.by_role.setdefault(key, RoleCost())
        slice_.observe(
            model,
            usage.input_tokens
            + usage.output_tokens
            + usage.cache_read_tokens
            + usage.cache_creation_tokens,
        )

    @property
    def total_tokens(self) -> int:
        """All four components summed — ENG-1286's ceiling reads this."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_creation_tokens
        )

    @property
    def duration_ms(self) -> int:
        end = (
            self.ended_monotonic
            if self.ended_monotonic is not None
            else time.monotonic()
        )
        return int((end - self.started_monotonic) * 1000)
