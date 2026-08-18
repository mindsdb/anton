from pydantic_settings import BaseSettings


#
class CoreSettings(BaseSettings):
    model_config = {"env_prefix": "ANTON_", "extra": "ignore"}

    # Router — cheap front-model gating (ENG-648). When enabled, every text
    # turn first hits the router model, which either answers trivial/from-
    # context requests directly or delegates to the planning model (optionally
    # preloading skills). The mechanism is the "thalamus" (see
    # anton/core/llm/thalamus.py); the user-facing knobs stay "router". Off by
    # default until evaluated; flip with ANTON_ROUTER_ENABLED=true.
    router_enabled: bool = False
    # Output budget for the gating call. Deliberately small: a direct
    # answer that doesn't fit here is evidence the turn wasn't trivial,
    # and the router treats truncation as "delegate".
    router_max_tokens: int = 1024

    # Output-token budget per planning/coding LLM call. Was a hardcoded
    # fallback in LLMClient.from_settings (always 8192, because this field
    # didn't exist — ENG-1042 Fix 4); now configurable via ANTON_MAX_TOKENS
    # or a host overlay (cowork-server). Reasoning models spend internal
    # thinking from this same budget, so raising it is the blunt mitigation
    # for answers that die at the cap; the session's truncation recovery
    # retries one-off at double this value.
    max_tokens: int = 8192

    # Session orchestration tuning
    max_tool_rounds: int = 25
    max_continuations: int = 3
    # Skip the completion verifier when a turn used fewer than this many tool
    # rounds. Default 1 preserves today's behavior (only pure Q&A, tool_round==0,
    # is skipped). Raise to 2 to also skip trivial single-tool-round turns once
    # verdict logs confirm they're rarely INCOMPLETE (ENG-716).
    verify_min_tool_rounds: int = 1
    # Per-turn spend ceiling in RAW tokens — input + output + cache_read +
    # cache_creation, i.e. `TurnCost.total_tokens` (ENG-1286). 0 disables it.
    #
    # RAW is deliberate, and the instinct to discount cache reads here is wrong:
    # they bill at ~a tenth, but they draw the user's included-token allowance at
    # full 1:1 weight (`auth/entitlements/services/included_usage.py` sums
    # input + output + cached_input). This ceiling protects the user's allowance,
    # so it counts the unit that allowance drains in. The TPM limiter's identical
    # weighting IS a defect (ENG-1132) because it guards money — same arithmetic,
    # opposite verdict, because they protect different things.
    #
    # 1.25M measured against 30 days of production (2026-08-12): trips ~15% of
    # external turns at a median of 16 LLM calls in. Set above Kiranam's worst
    # turn (1,480,766) and you stop catching the case this exists for — 1.5M
    # misses it by 20k tokens.
    max_turn_tokens: int = 1_250_000
    context_pressure_threshold: float = 0.7
    max_consecutive_errors: int = 5
    resilience_nudge_at: int = 2
    token_status_cache_ttl: float = 60.0

    # Scratchpad execution tuning
    cell_timeout_default: int = 120  # Total timeout when no estimate given (s)
    cell_inactivity_timeout: int = 30  # Max silence between output lines (s)
    cell_inactivity_after_progress: int = 60  # Grace window after progress() call (s)
    cell_inactivity_max: int = 60  # Ceiling on the silence window even when a large estimate scales it up (s)
    # Absolute ceiling on total cell runtime (s); 0 = off. Non-zero since the
    # liveness heartbeat (ENG-578): a userland deadlock or infinite loop beats
    # like a working cell, so this is the only bound that ends it. 1h is
    # generous enough for a throttled batch campaign (50 sends x 30s ≈ 25min)
    # while capping how long an agent-supplied estimate can extend a hang.
    cell_total_max: int = 3600
    cell_install_timeout: int = 120  # pip/uv install timeout (s)
    cell_keep_recent: int = 5  # Recent cells preserved during compaction

