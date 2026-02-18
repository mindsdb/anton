from minds.common.feature_flags.client import get_statsig
from minds.common.feature_flags.users import build_statsig_user
from minds.common.logger import setup_logging
from minds.common.settings.app_settings import AppSettings, get_app_settings
from minds.requests.context import Context

# Set up logging
logger = setup_logging()


def is_langfuse_enabled(context: Context, settings: AppSettings | None = None) -> bool:
    """
    The function checks if Langfuse is enabled.

    Args:
        context (Context): The context.

    Returns:
        bool: Whether Langfuse is enabled.
    """
    settings = settings or get_app_settings()

    logger.debug(f"Checking if Langfuse is enabled: {context.user_email}")
    logger.debug(f"Feature flag name: {settings.feature_flag_enable_langfuse.name}")
    logger.debug(f"Feature flag default value: {settings.feature_flag_enable_langfuse.default_value}")

    # Get the feature flag settings
    gate = settings.feature_flag_enable_langfuse

    # Get the Statsig client
    statsig = get_statsig(settings=settings)

    # Build the user context
    user = build_statsig_user(context=context, settings=settings)

    # Check if the gate is enabled
    enabled = statsig.check_gate(user=user, name=gate.name)
    logger.debug(f"Is Langfuse enabled?: {enabled}")

    return enabled
