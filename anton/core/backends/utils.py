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
    # undetected silence (an est=600 cell would otherwise allow 300s of no
    # output before being killed). A cell quiet for cell_inactivity_max
    # seconds is killed regardless of its estimate. stdout/progress() reset
    # this window, so legitimate long-but-active cells — e.g. a batch loop
    # pinging progress() — are unaffected; only genuinely stuck cells die.
    inactivity = min(inactivity, float(s.cell_inactivity_max))
    # The total is deliberately left scaling so long-but-active cells run to
    # completion. cell_total_max (default 0 = off) is an optional absolute
    # backstop for a runaway that keeps producing output forever (which the
    # inactivity cap can't catch); set it only when that risk outweighs
    # clipping a genuinely long batch job.
    if s.cell_total_max > 0:
        total = min(total, float(s.cell_total_max))
    return total, inactivity