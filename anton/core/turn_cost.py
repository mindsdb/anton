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

    def add(self, role: str, model: str, usage: Usage) -> None:
        """Count one LLM call. Installed as ``LLMClient.usage_listener``.

        ``role``/``model`` are accepted (they're the listener contract and
        what a future per-role breakdown would key on) but not yet stored
        per-role — the turn event carries the client's planning/coding model
        names instead.
        """
        self.llm_calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_tokens
        self.cache_creation_tokens += usage.cache_creation_tokens
        self.peak_context_tokens = max(self.peak_context_tokens, usage.context_tokens)

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
        return int((time.monotonic() - self.started_monotonic) * 1000)
