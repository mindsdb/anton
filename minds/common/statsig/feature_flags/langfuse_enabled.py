from minds.common.logger import get_logger
from minds.common.settings.app_settings import AppSettings, DeploymentMode, get_app_settings
from minds.common.statsig.client import get_statsig
from minds.common.statsig.users import build_statsig_user
from minds.requests.context import Context

logger = get_logger(__name__)


def is_langfuse_enabled(context: Context, settings: AppSettings | None = None) -> bool:
    """
    Check if Langfuse is enabled for the given context.

    - Self-hosted: returns the configured default value.
    - Cloud: checks Statsig gate; falls back to default if Statsig is unavailable.

    Args:
        context: The request context.
        settings: Optional app settings override.

    Returns:
        Whether Langfuse is enabled.
    """
    settings = settings or get_app_settings()
    gate = settings.feature_flag_enable_langfuse

    logger.debug(
        f"Checking if Langfuse is enabled for user '{context.user_id}' in organization '{context.organization_id}'"
    )

    # Self-hosted: skip Statsig, use configured default
    if settings.deployment_mode == DeploymentMode.SELF_HOSTED:
        logger.debug(f"Self-hosted deployment: using default value {gate.default_value}")
        return gate.default_value

    # Cloud: check Statsig gate, fall back to default if unavailable
    try:
        statsig = get_statsig(settings=settings)
        user = build_statsig_user(context=context, settings=settings)
        enabled = statsig.check_gate(user=user, name=gate.name)
        logger.debug(f"Langfuse enabled (Statsig): {enabled}")
        return enabled
    except Exception as e:
        logger.warning(f"Failed to check Statsig gate '{gate.name}', falling back to default {gate.default_value}: {e}")
        return gate.default_value
