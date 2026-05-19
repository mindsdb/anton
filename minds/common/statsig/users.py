from __future__ import annotations

from statsig_python_core import StatsigUser

from minds.common.constants import (
    CONTEXT_FIELD_ENV,
    CONTEXT_FIELD_ORGANIZATION_ID,
    CONTEXT_FIELD_USER_ID,
    CONTEXT_FIELD_USER_ROLES,
)
from minds.common.logger import get_logger
from minds.common.settings.app_settings import AppSettings, get_app_settings
from minds.requests.context import Context

logger = get_logger(__name__)


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
        custom_ids={
            CONTEXT_FIELD_ORGANIZATION_ID: str(context.organization_id),
        },
        custom={
            CONTEXT_FIELD_ORGANIZATION_ID: str(context.organization_id),
            CONTEXT_FIELD_USER_ID: str(context.user_id),
            CONTEXT_FIELD_ENV: settings.env,
            CONTEXT_FIELD_USER_ROLES: context.user_roles,
        },
    )
    logger.debug(f"Statsig user: {user}")
    return user
