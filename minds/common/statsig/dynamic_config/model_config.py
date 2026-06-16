"""
Passthrough routing policy from Statsig dynamic config.

Per-user ``alias_overrides`` (repoint ``latest:opus`` etc.), the allow-list of
callable aliases, and the web-search provider / kill switch. Mirrors
``mind_limits.py``: self-hosted skips Statsig entirely and cloud fails open, so
a Statsig outage or self-hosted deployment degrades to today's env-backed
behavior (no overrides, all aliases allowed, env-configured search) rather than
locking users out.
"""

from minds.common.constants import DYNAMIC_CONFIG_PASSTHROUGH_MODELS
from minds.common.logger import get_logger
from minds.common.search import supported_search_providers
from minds.common.settings.app_settings import AppSettings, DeploymentMode, get_app_settings
from minds.common.statsig.client import get_statsig
from minds.common.statsig.users import build_statsig_user
from minds.requests.context import Context
from minds.schemas.passthrough import PassthroughModelStatsigConfig

logger = get_logger(__name__)


def get_passthrough_model_config(
    context: Context, settings: AppSettings | None = None
) -> PassthroughModelStatsigConfig:
    """
    Get the passthrough routing policy for the given context from Statsig.

    - Self-hosted: returns an empty policy (no overrides, all allowed, env search).
    - Cloud: fetches from Statsig; falls back to an empty policy if unavailable.

    A search-provider name that this build can't construct is dropped to None
    (with a warning) so a Statsig typo can't take down search for the user; the
    env-configured registry path stays strict.
    """
    settings = settings or get_app_settings()

    logger.debug(
        f"Getting passthrough model config for user '{context.user_id}' in organization '{context.organization_id}'"
    )

    # Self-hosted: skip Statsig entirely, return the empty (no-policy) config.
    if settings.deployment_mode == DeploymentMode.SELF_HOSTED:
        logger.debug("Self-hosted deployment: returning empty passthrough config")
        return PassthroughModelStatsigConfig()

    # Cloud: fetch from Statsig, fail open if unavailable.
    try:
        statsig = get_statsig(settings=settings)
        user = build_statsig_user(context=context, settings=settings)
        dynamic_config = statsig.get_dynamic_config(user=user, name=DYNAMIC_CONFIG_PASSTHROUGH_MODELS)
        config = PassthroughModelStatsigConfig(**dynamic_config.value)
    except Exception as e:
        logger.warning(f"Failed to fetch passthrough config from Statsig, falling back to empty: {e}")
        return PassthroughModelStatsigConfig()

    # Validate the search-provider override against what this build can build;
    # an unknown name (e.g. a not-yet-implemented provider) drops to None so we
    # fall back to the env-configured provider rather than raising mid-request.
    if config.search_provider is not None and config.search_provider not in supported_search_providers():
        logger.warning(
            f"Statsig search_provider {config.search_provider!r} is not supported "
            f"({supported_search_providers()}); falling back to the configured default"
        )
        config.search_provider = None

    logger.debug(f"Passthrough model config: {config.model_dump_json()}")
    return config
