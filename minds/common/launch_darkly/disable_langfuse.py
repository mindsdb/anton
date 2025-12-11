from minds.common.launch_darkly import ldclient
from minds.common.logger import setup_logging
from minds.common.settings.app_settings import get_app_settings
from minds.requests.context import Context

# Set up logging
logger = setup_logging()
settings = get_app_settings()


def is_langfuse_disabled(context: Context) -> bool:
    """
    The function checks if Langfuse is disabled.

    Args:
        context (Context): The context.

    Returns:
        bool: Whether Langfuse is disabled.
    """
    logger.debug(f"Checking if Langfuse is disabled: {context.user_email}")
    logger.debug(f"Feature flag name: {settings.feature_flag_disable_langfuse.name}")
    logger.debug(f"Feature flag default value: {settings.feature_flag_disable_langfuse.default_value}")

    context = (
        ldclient.Context.builder(str(context.user_email))
        .kind("user")
        .name(context.user_email)
        .set("email", context.user_email)
        .build()
    )

    return ldclient.get().variation(
        settings.feature_flag_disable_langfuse.name, context, settings.feature_flag_disable_langfuse.default_value
    )
