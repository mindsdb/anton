"""
Limits service — token quota enforcement.

For the inference-only implementation, this tracks token usage against Statsig-configured limits.
Minds and datasources limits are deprecated (always unlimited).
"""

from minds.common.logger import get_logger
from minds.common.settings.app_settings import AppSettings, get_app_settings
from minds.common.statsig.dynamic_config.mind_limits import get_mind_limits_config
from minds.requests.context import Context
from minds.schemas.limits import MindLimitsConfig, UsageConfig
from minds.services.usage import UsageService

logger = get_logger(__name__)


class LimitsService:
    """Enforces token usage limits from Statsig configuration."""

    def __init__(
        self,
        usage_service: UsageService,
        context: Context,
        settings: AppSettings | None = None,
    ):
        """
        Initialize the limits service.

        Args:
            usage_service: Service for token counting.
            context: Request context (user_id, organization_id, billing dates).
            settings: App settings (defaults to global if not provided).
        """
        settings = settings or get_app_settings()
        self.usage_service = usage_service
        self.context = context
        self.settings = settings
        logger.debug(f"LimitsService initialized for user {context.user_id} in org {context.organization_id}")

    async def get_mind_limits(self) -> MindLimitsConfig:
        """
        Get token limits and current usage.

        Returns token usage against Statsig-configured limits.
        Minds and datasources are deprecated (set to unlimited usage).
        """
        try:
            limits = get_mind_limits_config(context=self.context, settings=self.settings)
            logger.debug(f"Limits config from Statsig: {limits}")

            billing_cycle_start = self.context.billing_cycle_start
            billing_cycle_end = self.context.billing_cycle_end

            # Token counts (the only resource we track)
            tokens_lifetime = await self.usage_service.count_tokens()
            tokens_cycle = await self.usage_service.count_tokens(
                since=billing_cycle_start, until=billing_cycle_end
            )

            # Populate token usage
            limits.tokens.usage = UsageConfig(lifetime=tokens_lifetime, billing_cycle=tokens_cycle)

            # Minds and datasources are deprecated — zero usage
            limits.minds.usage = UsageConfig(lifetime=0, billing_cycle=0)
            limits.datasources.usage = UsageConfig(lifetime=0, billing_cycle=0)

            logger.debug(f"Token usage for user {self.context.user_id}: {limits.tokens.usage}")
            return limits
        except Exception as e:
            logger.error(f"Error getting limits for user {self.context.user_id}: {e}")
            raise
