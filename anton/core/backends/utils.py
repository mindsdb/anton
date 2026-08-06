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
    # The total scales with the estimate so long-but-active cells run to
    # completion, but cell_total_max (default 3600) caps it: since the
    # liveness heartbeat keeps deliberately-quiet cells alive, a userland
    # deadlock or infinite loop beats like a working cell, and this ceiling
    # is the only thing that ends it. It also bounds how far an agent-chosen
    # estimate can extend that exposure.
    if s.cell_total_max > 0:
        total = min(total, float(s.cell_total_max))
    return total, inactivity