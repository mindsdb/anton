from anton.core.settings import CoreSettings


def compute_timeouts(estimated_seconds: int) -> tuple[float, float]:
    """Compute (total_timeout, inactivity_timeout) from an estimated run time.

    Reads defaults from CoreSettings so they're tunable via env vars.
    """
    s = CoreSettings()
    if estimated_seconds <= 0:
        return float(s.cell_timeout_default), float(s.cell_inactivity_timeout)
    total = max(estimated_seconds * 2, estimated_seconds + 30)
    inactivity = max(estimated_seconds * 0.5, 30)
    return float(total), float(inactivity)