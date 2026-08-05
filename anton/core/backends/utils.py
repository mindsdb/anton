from anton.core.settings import CoreSettings


def compute_timeouts(estimated_seconds: int) -> tuple[float, float]:
    """Compute (total_timeout, inactivity_timeout) from an estimated run time.

    Reads defaults from CoreSettings so they're tunable via env vars.
    """
    s = CoreSettings()
    if estimated_seconds <= 0:
        total = float(s.cell_timeout_default)
        inactivity = float(s.cell_inactivity_timeout)
    else:
        total = float(max(estimated_seconds * 2, estimated_seconds + 30))
        inactivity = float(max(estimated_seconds * 0.5, 30))
    # Clamp the silence window: a large estimate must not buy minutes of
    # undetected silence. Liveness is signalled by the worker's own heartbeat
    # thread (scratchpad_boot), NOT by userland stdout — cell prints go to a
    # per-cell StringIO and never reach this timer. A cell whose worker sends
    # no heartbeat/progress line for cell_inactivity_max seconds is killed as
    # dead or wedged regardless of its estimate.
    inactivity = min(inactivity, float(s.cell_inactivity_max))
    # The total is deliberately left scaling so long-but-active cells run to
    # completion. cell_total_max (default 0 = off) is an optional absolute
    # backstop for a runaway that keeps producing output forever (which the
    # inactivity cap can't catch); set it only when that risk outweighs
    # clipping a genuinely long batch job.
    if s.cell_total_max > 0:
        total = min(total, float(s.cell_total_max))
    return total, inactivity