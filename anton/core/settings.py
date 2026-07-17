from pydantic_settings import BaseSettings


#
class CoreSettings(BaseSettings):
    model_config = {"env_prefix": "ANTON_", "extra": "ignore"}

    # Session orchestration tuning
    max_tool_rounds: int = 25
    max_continuations: int = 3
    # Skip the completion verifier when a turn used fewer than this many tool
    # rounds. Default 1 preserves today's behavior (only pure Q&A, tool_round==0,
    # is skipped). Raise to 2 to also skip trivial single-tool-round turns once
    # verdict logs confirm they're rarely INCOMPLETE (ENG-716).
    verify_min_tool_rounds: int = 1
    context_pressure_threshold: float = 0.7
    max_consecutive_errors: int = 5
    resilience_nudge_at: int = 2
    token_status_cache_ttl: float = 60.0

    # Scratchpad execution tuning
    cell_timeout_default: int = 120  # Total timeout when no estimate given (s)
    cell_inactivity_timeout: int = 30  # Max silence between output lines (s)
    cell_inactivity_after_progress: int = 60  # Grace window after progress() call (s)
    cell_inactivity_max: int = 60  # Ceiling on the silence window even when a large estimate scales it up (s)
    cell_total_max: int = 0  # Optional absolute ceiling on total cell runtime (s); 0 = off (let it scale)
    cell_install_timeout: int = 120  # pip/uv install timeout (s)
    cell_keep_recent: int = 5  # Recent cells preserved during compaction

