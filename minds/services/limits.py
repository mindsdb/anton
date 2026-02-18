"""
Limits service layer for business logic and data operations.

This module contains the LimitsService class that handles all business logic
related to limits management, including feature flag limits management.

It fetches limit thresholds from Statsig (via dynamic config) and then
populates actual usage counts by delegating to MindsService,
DatasourcesService, and UsageService.
"""

from minds.common.logger import setup_logging
from minds.common.settings.app_settings import AppSettings, get_app_settings
from minds.common.statsig.dynamic_config.mind_limits import get_mind_limits_config
from minds.requests.context import Context
from minds.schemas.limits import MindLimitsConfig, UsageConfig
from minds.services.datasources import DatasourcesService
from minds.services.minds import MindsService
from minds.services.usage import UsageService

logger = setup_logging()


class LimitsService:
    """
    Service class for limits management.

    Combines limit thresholds (from Statsig) with real-time usage counts
    (from the database) to produce a complete MindLimitsConfig.
    """

    def __init__(
        self,
        minds_service: MindsService,
        datasources_service: DatasourcesService,
        usage_service: UsageService,
        context: Context,
        settings: AppSettings | None = None,
    ):
        """
        Initialize the limits service.

        Args:
            minds_service: Service for mind-related operations (provides count_minds).
            datasources_service: Service for datasource operations (provides count_datasources).
            usage_service: Service for token/question aggregation queries.
            context: Request context containing user_id and organization_id.
            settings: App settings (defaults to global settings if not provided).
        """
        settings = settings or get_app_settings()

        self.minds_service: MindsService = minds_service
        self.datasources_service: DatasourcesService = datasources_service
        self.usage_service: UsageService = usage_service
        self.context = context
        self.settings = settings

        logger.debug(f"LimitsService initialized for context: {context}")

    async def get_mind_limits(self) -> MindLimitsConfig:
        """
        Get the mind limits and current usage for the current user.

        1. Fetches limit thresholds from Statsig dynamic config.
        2. Populates lifetime usage counts (no date filter).
        3. Populates billing-cycle usage counts (filtered by billing_period_start
           from context, if available).

        Returns:
            MindLimitsConfig: Limits and usage for tokens, minds, datasources,
                              and questions.
        """
        try:
            # Step 1: Get limit thresholds from Statsig (or unlimited for self-hosted)
            limits = get_mind_limits_config(context=self.context, settings=self.settings)
            logger.debug(f"Mind limits config from Statsig: {limits}")

            # Step 2: Populate actual usage counts
            logger.debug(
                f"Fetching usage counts for user {self.context.user_id} in organization {self.context.organization_id}"
            )

            billing_start = self.context.billing_period_start

            # Lifetime counts (no date filter)
            minds_lifetime = await self.minds_service.count_minds(is_sample=False)
            datasources_lifetime = await self.datasources_service.count_datasources(is_sample=False)
            tokens_lifetime = await self.usage_service.count_tokens()
            questions_lifetime = await self.usage_service.count_questions()

            # Billing-cycle counts (filtered by billing_period_start when available)
            minds_cycle = await self.minds_service.count_minds(is_sample=False, since=billing_start)
            datasources_cycle = await self.datasources_service.count_datasources(is_sample=False, since=billing_start)
            tokens_cycle = await self.usage_service.count_tokens(since=billing_start)
            questions_cycle = await self.usage_service.count_questions(since=billing_start)

            limits.minds.usage = UsageConfig(lifetime=minds_lifetime, billing_cycle=minds_cycle)
            limits.datasources.usage = UsageConfig(lifetime=datasources_lifetime, billing_cycle=datasources_cycle)
            limits.tokens.usage = UsageConfig(lifetime=tokens_lifetime, billing_cycle=tokens_cycle)
            limits.questions.usage = UsageConfig(lifetime=questions_lifetime, billing_cycle=questions_cycle)

            logger.debug(
                f"Usage counts for user {self.context.user_id} in organization {self.context.organization_id}: "
                f"minds={limits.minds.usage}, datasources={limits.datasources.usage}, "
                f"tokens={limits.tokens.usage}, questions={limits.questions.usage}"
            )

            return limits
        except Exception as e:
            logger.error(f"Error getting mind limits for user {self.context.user_id}: {e}")
            raise
