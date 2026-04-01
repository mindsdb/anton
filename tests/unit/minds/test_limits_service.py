"""
Unit tests for LimitsService.

Tests the limits service layer that combines Statsig thresholds
with real-time usage counts from MindsService, DatasourcesService,
and UsageService.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from minds.requests.context import Context
from minds.schemas.limits import LimitsConfig, MindLimitsConfig, ResourceUsageConfig
from minds.services.datasources import DatasourcesService
from minds.services.limits import LimitsService
from minds.services.minds import MindsService
from minds.services.usage import UsageService


class TestLimitsService:
    """Test suite for LimitsService."""

    @pytest.fixture
    def mock_minds_service(self):
        """Mock MindsService instance."""
        service = Mock(spec=MindsService)
        service.count_minds = AsyncMock(return_value=5)
        return service

    @pytest.fixture
    def mock_datasources_service(self):
        """Mock DatasourcesService instance."""
        service = Mock(spec=DatasourcesService)
        service.count_datasources = AsyncMock(return_value=3)
        return service

    @pytest.fixture
    def mock_usage_service(self):
        """Mock UsageService instance."""
        service = Mock(spec=UsageService)
        service.count_tokens = AsyncMock(return_value=10000)
        return service

    @pytest.fixture
    def mock_context(self):
        """Mock request context."""
        return Context(
            user_id=uuid4(),
            organization_id=uuid4(),
            user_email="test@test.com",
            user_roles=["user"],
            billing_cycle_start=datetime(2026, 2, 1, tzinfo=timezone.utc),
            billing_cycle_end=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )

    @pytest.fixture
    def mock_settings(self):
        """Mock AppSettings."""
        settings = Mock()
        settings.deployment_mode = "cloud"
        return settings

    @pytest.fixture
    def limits_service(
        self,
        mock_minds_service,
        mock_datasources_service,
        mock_usage_service,
        mock_context,
        mock_settings,
    ):
        """Create LimitsService with mocked dependencies."""
        return LimitsService(
            minds_service=mock_minds_service,
            datasources_service=mock_datasources_service,
            usage_service=mock_usage_service,
            context=mock_context,
            settings=mock_settings,
        )

    def test_service_initialization(
        self,
        mock_minds_service,
        mock_datasources_service,
        mock_usage_service,
        mock_context,
        mock_settings,
    ):
        """Test LimitsService initialization."""
        service = LimitsService(
            minds_service=mock_minds_service,
            datasources_service=mock_datasources_service,
            usage_service=mock_usage_service,
            context=mock_context,
            settings=mock_settings,
        )

        assert service.minds_service == mock_minds_service
        assert service.datasources_service == mock_datasources_service
        assert service.usage_service == mock_usage_service
        assert service.context == mock_context
        assert service.settings == mock_settings

    @pytest.mark.asyncio
    async def test_get_mind_limits_success(self, limits_service, mock_context):
        """Test successful mind limits retrieval with usage populated."""
        mock_limits = MindLimitsConfig(
            tokens=ResourceUsageConfig(
                limit=LimitsConfig(lifetime=100000, monthly=50000),
            ),
            minds=ResourceUsageConfig(
                limit=LimitsConfig(lifetime=10, monthly=-1),
            ),
            datasources=ResourceUsageConfig(
                limit=LimitsConfig(lifetime=20, monthly=-1),
            ),
        )

        with patch("minds.services.limits.get_mind_limits_config", return_value=mock_limits):
            result = await limits_service.get_mind_limits()

        assert isinstance(result, MindLimitsConfig)
        # Verify lifetime usage counts were populated from the sub-services
        assert result.minds.usage.lifetime == 5
        assert result.datasources.usage.lifetime == 3
        assert result.tokens.usage.lifetime == 10000

        # Verify billing_cycle usage counts were populated
        assert result.minds.usage.billing_cycle == 5
        assert result.datasources.usage.billing_cycle == 3
        assert result.tokens.usage.billing_cycle == 10000

        # Verify the limit thresholds were preserved
        assert result.tokens.limit.lifetime == 100000
        assert result.tokens.limit.monthly == 50000
        assert result.minds.limit.lifetime == 10
        assert result.datasources.limit.lifetime == 20

    @pytest.mark.asyncio
    async def test_get_mind_limits_default_unlimited(self, limits_service):
        """Test that default config returns unlimited limits with usage populated."""
        default_limits = MindLimitsConfig()

        with patch("minds.services.limits.get_mind_limits_config", return_value=default_limits):
            result = await limits_service.get_mind_limits()

        assert isinstance(result, MindLimitsConfig)
        # Usage should be populated
        assert result.minds.usage.lifetime == 5
        assert result.datasources.usage.lifetime == 3
        assert result.tokens.usage.lifetime == 10000
        # Limits should be unlimited (default -1)
        assert result.tokens.limit.lifetime == -1
        assert result.minds.limit.lifetime == -1

    @pytest.mark.asyncio
    async def test_get_mind_limits_calls_count_with_is_sample_false(
        self,
        limits_service,
        mock_minds_service,
        mock_datasources_service,
        mock_context,
    ):
        """Test that count methods are called with is_sample=False for both lifetime and billing cycle."""
        default_limits = MindLimitsConfig()

        with patch("minds.services.limits.get_mind_limits_config", return_value=default_limits):
            await limits_service.get_mind_limits()

        # Called twice: once for lifetime (since=None, until=None), once for billing cycle
        assert mock_minds_service.count_minds.call_count == 2
        mock_minds_service.count_minds.assert_any_call(is_sample=False)
        mock_minds_service.count_minds.assert_any_call(
            is_sample=False,
            since=mock_context.billing_cycle_start,
            until=mock_context.billing_cycle_end,
        )

        assert mock_datasources_service.count_datasources.call_count == 2
        mock_datasources_service.count_datasources.assert_any_call(is_sample=False)
        mock_datasources_service.count_datasources.assert_any_call(
            is_sample=False,
            since=mock_context.billing_cycle_start,
            until=mock_context.billing_cycle_end,
        )

    @pytest.mark.asyncio
    async def test_get_mind_limits_calls_usage_service(
        self,
        limits_service,
        mock_usage_service,
        mock_context,
    ):
        """Test that usage service methods are called for both lifetime and billing cycle."""
        default_limits = MindLimitsConfig()

        with patch("minds.services.limits.get_mind_limits_config", return_value=default_limits):
            await limits_service.get_mind_limits()

        # Called twice: once for lifetime, once for billing cycle
        assert mock_usage_service.count_tokens.call_count == 2
        mock_usage_service.count_tokens.assert_any_call()
        mock_usage_service.count_tokens.assert_any_call(
            since=mock_context.billing_cycle_start,
            until=mock_context.billing_cycle_end,
        )

    @pytest.mark.asyncio
    async def test_get_mind_limits_zero_usage(
        self,
        mock_minds_service,
        mock_datasources_service,
        mock_usage_service,
        mock_context,
        mock_settings,
    ):
        """Test mind limits when all usage counts are zero."""
        mock_minds_service.count_minds = AsyncMock(return_value=0)
        mock_datasources_service.count_datasources = AsyncMock(return_value=0)
        mock_usage_service.count_tokens = AsyncMock(return_value=0)

        service = LimitsService(
            minds_service=mock_minds_service,
            datasources_service=mock_datasources_service,
            usage_service=mock_usage_service,
            context=mock_context,
            settings=mock_settings,
        )

        default_limits = MindLimitsConfig()

        with patch("minds.services.limits.get_mind_limits_config", return_value=default_limits):
            result = await service.get_mind_limits()

        assert result.minds.usage.lifetime == 0
        assert result.minds.usage.billing_cycle == 0
        assert result.datasources.usage.lifetime == 0
        assert result.datasources.usage.billing_cycle == 0
        assert result.tokens.usage.lifetime == 0
        assert result.tokens.usage.billing_cycle == 0

    @pytest.mark.asyncio
    async def test_get_mind_limits_no_billing_period(
        self,
        mock_minds_service,
        mock_datasources_service,
        mock_usage_service,
        mock_settings,
    ):
        """Test mind limits when billing_cycle_start and billing_cycle_end are None."""
        context_no_billing = Context(
            user_id=uuid4(),
            organization_id=uuid4(),
            user_email="test@test.com",
            user_roles=["user"],
            billing_cycle_start=None,
            billing_cycle_end=None,
        )

        service = LimitsService(
            minds_service=mock_minds_service,
            datasources_service=mock_datasources_service,
            usage_service=mock_usage_service,
            context=context_no_billing,
            settings=mock_settings,
        )

        default_limits = MindLimitsConfig()

        with patch("minds.services.limits.get_mind_limits_config", return_value=default_limits):
            result = await service.get_mind_limits()

        # Both lifetime and billing_cycle should be populated (billing cycle uses since=None, until=None)
        assert result.minds.usage.lifetime == 5
        assert result.minds.usage.billing_cycle == 5
        mock_minds_service.count_minds.assert_any_call(is_sample=False, since=None, until=None)

    @pytest.mark.asyncio
    async def test_get_mind_limits_different_lifetime_and_cycle(
        self,
        mock_datasources_service,
        mock_usage_service,
        mock_context,
        mock_settings,
    ):
        """Test that lifetime and billing cycle can return different counts."""
        minds_service = Mock(spec=MindsService)
        # First call (lifetime) returns 10, second call (billing cycle) returns 3
        minds_service.count_minds = AsyncMock(side_effect=[10, 3])

        service = LimitsService(
            minds_service=minds_service,
            datasources_service=mock_datasources_service,
            usage_service=mock_usage_service,
            context=mock_context,
            settings=mock_settings,
        )

        default_limits = MindLimitsConfig()

        with patch("minds.services.limits.get_mind_limits_config", return_value=default_limits):
            result = await service.get_mind_limits()

        assert result.minds.usage.lifetime == 10
        assert result.minds.usage.billing_cycle == 3

    @pytest.mark.asyncio
    async def test_get_mind_limits_propagates_exception(
        self,
        mock_minds_service,
        mock_datasources_service,
        mock_usage_service,
        mock_context,
        mock_settings,
    ):
        """Test that exceptions from sub-services propagate."""
        mock_minds_service.count_minds = AsyncMock(side_effect=Exception("DB error"))

        service = LimitsService(
            minds_service=mock_minds_service,
            datasources_service=mock_datasources_service,
            usage_service=mock_usage_service,
            context=mock_context,
            settings=mock_settings,
        )

        default_limits = MindLimitsConfig()

        with (
            patch("minds.services.limits.get_mind_limits_config", return_value=default_limits),
            pytest.raises(Exception, match="DB error"),
        ):
            await service.get_mind_limits()

    @pytest.mark.asyncio
    async def test_get_mind_limits_statsig_error_propagates(
        self,
        limits_service,
    ):
        """Test that Statsig config errors propagate."""
        with (
            patch("minds.services.limits.get_mind_limits_config", side_effect=Exception("Statsig error")),
            pytest.raises(Exception, match="Statsig error"),
        ):
            await limits_service.get_mind_limits()

    @pytest.mark.asyncio
    async def test_get_mind_limits_only_billing_start_no_end(
        self,
        mock_minds_service,
        mock_datasources_service,
        mock_usage_service,
        mock_settings,
    ):
        """Test mind limits when only billing_cycle_start is set (no end)."""
        context = Context(
            user_id=uuid4(),
            organization_id=uuid4(),
            user_email="test@test.com",
            user_roles=["user"],
            billing_cycle_start=datetime(2026, 2, 1, tzinfo=timezone.utc),
            billing_cycle_end=None,
        )

        service = LimitsService(
            minds_service=mock_minds_service,
            datasources_service=mock_datasources_service,
            usage_service=mock_usage_service,
            context=context,
            settings=mock_settings,
        )

        default_limits = MindLimitsConfig()

        with patch("minds.services.limits.get_mind_limits_config", return_value=default_limits):
            result = await service.get_mind_limits()

        assert result.minds.usage.lifetime == 5
        assert result.minds.usage.billing_cycle == 5
        mock_minds_service.count_minds.assert_any_call(
            is_sample=False,
            since=context.billing_cycle_start,
            until=None,
        )
        mock_usage_service.count_tokens.assert_any_call(
            since=context.billing_cycle_start,
            until=None,
        )
