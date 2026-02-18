from __future__ import annotations

import atexit
import threading

from statsig_python_core import Statsig, StatsigOptions

from minds.common.logger import setup_logging
from minds.common.settings.app_settings import AppSettings, get_app_settings
from minds.common.statsig.offline import apply_offline_overrides

_lock = threading.Lock()
_client: Statsig | None = None

logger = setup_logging()


def build_statsig_options(settings: AppSettings) -> StatsigOptions:
    """
    Build the Statsig options.

    Args:
        settings: The app settings.

    Returns:
        The Statsig options.
    """
    logger.debug(f"Building Statsig options for environment: {settings.statsig.environment}")
    logger.debug(f"Building Statsig options for disable network: {settings.statsig.disable_network}")
    logger.debug(f"Building Statsig options for disable all logging: {settings.statsig.disable_all_logging}")

    options = StatsigOptions(
        environment=settings.statsig.environment,
        disable_network=settings.statsig.disable_network,
        disable_all_logging=settings.statsig.disable_all_logging,
    )
    logger.debug(f"Statsig options: {options}")
    return options


def init_statsig(settings: AppSettings | None = None) -> Statsig:
    """
    Initialize the Statsig client.

    Args:
        settings: The app settings.

    Returns:
        The Statsig client.
    """
    global _client
    settings = settings or get_app_settings()

    # If the client is already initialized, return it
    if _client is not None:
        return _client

    with _lock:
        # If the client is already initializing, return it
        if _client is not None:
            return _client

        # Build the Statsig options
        options = build_statsig_options(settings)
        statsig = Statsig(sdk_key=settings.statsig.sdk_key, options=options)

        # Apply offline overrides
        apply_offline_overrides(statsig=statsig, settings=settings)

        # Initialize the Statsig client
        statsig.initialize().wait()

        # Set the client
        _client = statsig

        # Register the shutdown function
        atexit.register(shutdown_statsig)
        return _client


def get_statsig(settings: AppSettings | None = None) -> Statsig:
    """
    Get the Statsig client.

    Args:
        settings: The app settings.

    Returns:
        The Statsig client.
    """
    settings = settings or get_app_settings()

    if _client is None:
        init_statsig(settings=settings)
    assert _client is not None
    return _client


def shutdown_statsig() -> None:
    """
    Shutdown the Statsig client.

    Returns:
        None
    """
    global _client
    if _client is None:
        return
    try:
        _client.shutdown().wait()
    finally:
        _client = None
