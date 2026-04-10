from anton.core.settings import CoreSettings


def _compute_timeouts(core_settings: CoreSettings, estimated_seconds: int) -> tuple[float, float]:
    """Compute (total_timeout, inactivity_timeout) from an estimated run time.

    Reads defaults from CoreSettings so they're tunable via env vars.
    """
    if estimated_seconds <= 0:
        return float(core_settings.cell_timeout_default), float(core_settings.cell_inactivity_timeout)
    total = max(estimated_seconds * 2, estimated_seconds + 30)
    inactivity = max(estimated_seconds * 0.5, 30)
    return float(total), float(inactivity)