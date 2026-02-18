"""
Unit tests for the usage guard module.

Tests the ``require_usage_available`` function that enforces resource
consumption limits at the endpoint level.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from minds.common.guards.usage import (
    ResourceType,
    UsageLimitExceededError,
    _is_limit_exceeded,
    require_usage_available,
)
from minds.schemas.limits import (
    UNLIMITED,
    LimitsConfig,
    MindLimitsConfig,
    ResourceUsageConfig,
    UsageConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_limits_service(limits: MindLimitsConfig) -> Mock:
    """Build a mock LimitsService that returns the given config."""
    service = Mock()
    service.get_mind_limits = AsyncMock(return_value=limits)
    return service


def _resource_config(
    lifetime_limit: int = UNLIMITED,
    monthly_limit: int = UNLIMITED,
    lifetime_usage: int = 0,
    billing_cycle_usage: int = 0,
) -> ResourceUsageConfig:
    return ResourceUsageConfig(
        limit=LimitsConfig(lifetime=lifetime_limit, monthly=monthly_limit),
        usage=UsageConfig(lifetime=lifetime_usage, billing_cycle=billing_cycle_usage),
    )


# ---------------------------------------------------------------------------
# _is_limit_exceeded
# ---------------------------------------------------------------------------


class TestIsLimitExceeded:
    """Tests for the internal ``_is_limit_exceeded`` helper."""

    def test_unlimited_is_never_exceeded(self):
        assert _is_limit_exceeded(usage=999_999, limit=UNLIMITED) is False

    def test_zero_usage_with_positive_limit(self):
        assert _is_limit_exceeded(usage=0, limit=10) is False

    def test_under_limit(self):
        assert _is_limit_exceeded(usage=5, limit=10) is False

    def test_at_limit(self):
        assert _is_limit_exceeded(usage=10, limit=10) is True

    def test_over_limit(self):
        assert _is_limit_exceeded(usage=15, limit=10) is True

    def test_zero_limit_always_exceeded(self):
        assert _is_limit_exceeded(usage=0, limit=0) is True


# ---------------------------------------------------------------------------
# ResourceType enum
# ---------------------------------------------------------------------------


class TestResourceType:
    """Ensure the enum covers all expected resource types."""

    def test_values(self):
        assert ResourceType.MINDS.value == "minds"
        assert ResourceType.DATASOURCES.value == "datasources"
        assert ResourceType.TOKENS.value == "tokens"
        assert ResourceType.QUESTIONS.value == "questions"

    def test_is_str_enum(self):
        assert isinstance(ResourceType.MINDS, str)


# ---------------------------------------------------------------------------
# UsageLimitExceededError
# ---------------------------------------------------------------------------


class TestUsageLimitExceededError:
    """Tests for the custom HTTP exception."""

    def test_default_detail(self):
        err = UsageLimitExceededError(ResourceType.MINDS)
        assert err.status_code == 429
        assert "minds" in err.detail

    def test_custom_detail(self):
        err = UsageLimitExceededError(ResourceType.TOKENS, detail="custom msg")
        assert err.status_code == 429
        assert err.detail == "custom msg"


# ---------------------------------------------------------------------------
# require_usage_available – happy paths
# ---------------------------------------------------------------------------


class TestRequireUsageAvailableAllowed:
    """Cases where the guard should pass without raising."""

    @pytest.mark.asyncio
    async def test_unlimited_always_passes(self):
        """Both lifetime and monthly unlimited -> no error."""
        limits = MindLimitsConfig(
            minds=_resource_config(
                lifetime_limit=UNLIMITED,
                monthly_limit=UNLIMITED,
                lifetime_usage=9999,
                billing_cycle_usage=9999,
            ),
        )
        service = _make_limits_service(limits)
        await require_usage_available(service, ResourceType.MINDS)
        service.get_mind_limits.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_under_both_limits(self):
        limits = MindLimitsConfig(
            datasources=_resource_config(
                lifetime_limit=100,
                monthly_limit=50,
                lifetime_usage=10,
                billing_cycle_usage=5,
            ),
        )
        service = _make_limits_service(limits)
        await require_usage_available(service, ResourceType.DATASOURCES)

    @pytest.mark.asyncio
    async def test_zero_usage(self):
        limits = MindLimitsConfig(
            questions=_resource_config(
                lifetime_limit=100,
                monthly_limit=50,
                lifetime_usage=0,
                billing_cycle_usage=0,
            ),
        )
        service = _make_limits_service(limits)
        await require_usage_available(service, ResourceType.QUESTIONS)

    @pytest.mark.asyncio
    async def test_lifetime_unlimited_monthly_under(self):
        limits = MindLimitsConfig(
            tokens=_resource_config(
                lifetime_limit=UNLIMITED,
                monthly_limit=1000,
                lifetime_usage=999999,
                billing_cycle_usage=500,
            ),
        )
        service = _make_limits_service(limits)
        await require_usage_available(service, ResourceType.TOKENS)


# ---------------------------------------------------------------------------
# require_usage_available – rejection paths
# ---------------------------------------------------------------------------


class TestRequireUsageAvailableRejected:
    """Cases where the guard should raise UsageLimitExceededError (429)."""

    @pytest.mark.asyncio
    async def test_monthly_limit_exceeded(self):
        limits = MindLimitsConfig(
            questions=_resource_config(
                lifetime_limit=UNLIMITED,
                monthly_limit=100,
                lifetime_usage=50,
                billing_cycle_usage=100,
            ),
        )
        service = _make_limits_service(limits)
        with pytest.raises(UsageLimitExceededError) as exc_info:
            await require_usage_available(service, ResourceType.QUESTIONS)
        assert exc_info.value.status_code == 429
        assert "questions" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_lifetime_limit_exceeded(self):
        limits = MindLimitsConfig(
            minds=_resource_config(
                lifetime_limit=10,
                monthly_limit=UNLIMITED,
                lifetime_usage=10,
                billing_cycle_usage=2,
            ),
        )
        service = _make_limits_service(limits)
        with pytest.raises(UsageLimitExceededError) as exc_info:
            await require_usage_available(service, ResourceType.MINDS)
        assert exc_info.value.status_code == 429
        assert "minds" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_both_limits_exceeded_monthly_reported_first(self):
        """When both limits exceeded, the monthly check fires first."""
        limits = MindLimitsConfig(
            datasources=_resource_config(
                lifetime_limit=10,
                monthly_limit=5,
                lifetime_usage=15,
                billing_cycle_usage=8,
            ),
        )
        service = _make_limits_service(limits)
        with pytest.raises(UsageLimitExceededError) as exc_info:
            await require_usage_available(service, ResourceType.DATASOURCES)
        assert "Monthly" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_monthly_over_but_lifetime_ok(self):
        limits = MindLimitsConfig(
            tokens=_resource_config(
                lifetime_limit=100000,
                monthly_limit=1000,
                lifetime_usage=500,
                billing_cycle_usage=1001,
            ),
        )
        service = _make_limits_service(limits)
        with pytest.raises(UsageLimitExceededError):
            await require_usage_available(service, ResourceType.TOKENS)


# ---------------------------------------------------------------------------
# require_usage_available – each ResourceType maps correctly
# ---------------------------------------------------------------------------


class TestResourceTypeMapping:
    """Ensure each ResourceType resolves to the correct attribute."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("resource", list(ResourceType))
    async def test_each_resource_type_is_checked(self, resource: ResourceType):
        """Guard reads the matching field from MindLimitsConfig."""
        config = _resource_config(lifetime_limit=1, monthly_limit=1, lifetime_usage=0, billing_cycle_usage=0)
        kwargs = {resource.value: config}
        limits = MindLimitsConfig(**kwargs)
        service = _make_limits_service(limits)
        # Should pass without error when under limit
        await require_usage_available(service, resource)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("resource", list(ResourceType))
    async def test_each_resource_type_rejects_at_limit(self, resource: ResourceType):
        """Guard raises for each resource type when at limit."""
        config = _resource_config(lifetime_limit=5, monthly_limit=5, lifetime_usage=5, billing_cycle_usage=5)
        kwargs = {resource.value: config}
        limits = MindLimitsConfig(**kwargs)
        service = _make_limits_service(limits)
        with pytest.raises(UsageLimitExceededError):
            await require_usage_available(service, resource)
