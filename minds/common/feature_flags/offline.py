from __future__ import annotations

from statsig_python_core import Statsig

from minds.common.logger import setup_logging
from minds.common.settings.app_settings import AppSettings, get_app_settings

logger = setup_logging()


def apply_offline_overrides(statsig: Statsig, settings: AppSettings | None = None) -> None:
    """
    Apply offline overrides to the Statsig client.

    Args:
        statsig: The Statsig client.
        settings: The app settings.

    Returns:
        None
    """
    # Get the app settings
    settings = settings or get_app_settings()

    # If the network is not disabled, return
    if not settings.statsig.disable_network:
        logger.debug("Skipping offline overrides because network is not disabled")
        return

    # Override the feature flag
    logger.debug(f"Overriding feature flag {settings.feature_flag_enable_langfuse.name} to True")
    statsig.override_gate(settings.feature_flag_enable_langfuse.name, True)

    # Add other feature flags to override here
    logger.debug("Offline overrides applied")
