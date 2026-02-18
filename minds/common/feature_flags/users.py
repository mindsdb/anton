from __future__ import annotations

from statsig_python_core import StatsigUser

from minds.common.logger import setup_logging
from minds.common.settings.app_settings import AppSettings, get_app_settings
from minds.requests.context import Context

logger = setup_logging()


def build_statsig_user(context: Context, settings: AppSettings | None = None) -> StatsigUser:
    """
    Build a Statsig user from the context.

    Args:
      context: The context.
      settings: The app settings.

    Returns:
      The Statsig user.
    """
    # Get the app settings
    settings = settings or get_app_settings()

    logger.debug(f"Building Statsig user for context: {context}")
    logger.debug(f"Statsig user ID: {context.user_id}")
    logger.debug(f"Statsig user email: {context.user_email}")
    logger.debug(f"Statsig user organization ID: {context.organization_id}")
    logger.debug(f"Statsig user environment: {settings.statsig.environment}")

    # Build the Statsig user
    user = StatsigUser(
        user_id=str(context.user_id),
        email=context.user_email,
        custom={
            "organization_id": str(context.organization_id),
            "user_id": str(context.user_id),
            "env": settings.env,
        },
    )
    logger.debug(f"Statsig user: {user}")
    return user
