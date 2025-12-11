import ldclient

from minds.common.logger import setup_logging
from minds.common.settings.app_settings import get_app_settings

# Set up logging
logger = setup_logging()

# Initialize LaunchDarkly SDK
_is_initialized = False

if not _is_initialized:
    _is_initialized = True
    settings = get_app_settings()
    logger.info("Initializing LaunchDarkly Client")
    logger.info(f"LaunchDarkly SDK Key: {settings.launchdarkly.sdk_key}")
    logger.info(f"LaunchDarkly Offline Mode: {settings.launchdarkly.offline_mode}")

    ldclient.set_config(
        config=ldclient.Config(
            sdk_key=settings.launchdarkly.sdk_key,
            offline=settings.launchdarkly.offline_mode,
        )
    )

    if not ldclient.get().is_initialized():
        logger.error("*** LaunchDarkly failed to initialize. Please check your internet connection and SDK credential.")
