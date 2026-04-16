from pydantic_settings import BaseSettings


#
class CoreSettings(BaseSettings):
    model_config = {"env_prefix": "ANTON_", "extra": "ignore"}

    # Session orchestration tuning
    max_tool_rounds: int = 25
    max_continuations: int = 3
    context_pressure_threshold: float = 0.7
    max_consecutive_errors: int = 5
    resilience_nudge_at: int = 2
    token_status_cache_ttl: float = 60.0

    # Scratchpad execution tuning
    cell_timeout_default: int = 120  # Total timeout when no estimate given (s)
    cell_inactivity_timeout: int = 30  # Max silence between output lines (s)
    cell_inactivity_after_progress: int = 60  # Grace window after progress() call (s)
    cell_install_timeout: int = 120  # pip/uv install timeout (s)
    cell_keep_recent: int = 5  # Recent cells preserved during compaction
