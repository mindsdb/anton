"""
Mind limits config from Statsig dynamic config.
"""

from minds.common.constants import DYNAMIC_CONFIG_MIND_USAGE_LIMITS
from minds.common.logger import get_logger
from minds.common.settings.app_settings import AppSettings, DeploymentMode, get_app_settings
from minds.common.statsig.client import get_statsig
from minds.common.statsig.users import build_statsig_user
from minds.requests.context import Context
from minds.schemas.limits import MindLimitsConfig

logger = get_logger(__name__)


def get_mind_limits_config(context: Context, settings: AppSettings | None = None) -> MindLimitsConfig:
    """
    Get the mind limits config for the given context from Statsig dynamic config.

    - Self-hosted: returns unlimited (all defaults are -1).
    - Cloud: fetches from Statsig; falls back to unlimited if Statsig is unavailable.
    """
    settings = settings or get_app_settings()

    logger.debug(f"Getting mind limits config for user '{context.user_id}' in organization '{context.organization_id}'")

    # Self-hosted: skip Statsig entirely, return unlimited
    if settings.deployment_mode == DeploymentMode.SELF_HOSTED:
        logger.debug("Self-hosted deployment: returning unlimited limits")
        return MindLimitsConfig()

    # Cloud: fetch from Statsig, fail open if unavailable
    try:
        statsig = get_statsig(settings=settings)
        user = build_statsig_user(context=context, settings=settings)
        dynamic_config = statsig.get_dynamic_config(user=user, name=DYNAMIC_CONFIG_MIND_USAGE_LIMITS)
        config = MindLimitsConfig(**dynamic_config.value)
        logger.debug(f"Mind limits config: {config.model_dump_json()}")
        return config
    except Exception as e:
        logger.warning(f"Failed to fetch limits from Statsig, falling back to unlimited: {e}")
        return MindLimitsConfig()
