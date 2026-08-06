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

import time
from dataclasses import dataclass, field

from anton.core.llm.provider import Usage


# Roles the event reports. `unknown` catches an empty/unexpected role so the
# per-role token sum always equals the turn total; it should stay empty.
UNKNOWN_ROLE = "unknown"
EVENT_ROLES = ("planning", "coding", "router", UNKNOWN_ROLE)


@dataclass
class RoleCost:
    """One role's slice of a turn: which model ran it and what it spent.

    Roles are a closed set (planning / coding / router), which is why the turn
    event can carry this as flat properties instead of a nested blob.

    ``model`` is the alias anton REQUESTED. The gateway may resolve or fail
    over to a different model server-side and does not report that back on the
    response (``LLMResponse`` carries no model field), so this is the client's
    intent, not proof of what served the call — Langfuse holds the latter.
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
