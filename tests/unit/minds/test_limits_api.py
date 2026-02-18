"""
Unit tests for limits API endpoints.
"""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from minds.api.v1.endpoints.limits import get_limits
from minds.requests.context import Context
from minds.schemas.limits import MindLimitsConfig
from minds.services.limits import LimitsService


class TestLimitsAPI:
    @pytest.fixture
    def mock_limits_service(self):
        service = Mock(spec=LimitsService)
        service.get_mind_limits = AsyncMock(return_value=MindLimitsConfig())
        return service

    @pytest.fixture
    def mock_context(self):
        return Context(
            user_id=uuid4(),
            organization_id=uuid4(),
            user_email="test@example.com",
            user_roles=["user"],
        )

    @pytest.mark.asyncio
    async def test_get_limits_success(self, mock_limits_service):
        """Returns limits payload on success."""
        expected = MindLimitsConfig()
        mock_limits_service.get_mind_limits = AsyncMock(return_value=expected)

        result = await get_limits(limits_service=mock_limits_service)

        assert result == expected
        mock_limits_service.get_mind_limits.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_limits_maps_error_to_http_500(self, mock_limits_service):
        """Maps unexpected exceptions to HTTP 500."""
        mock_limits_service.get_mind_limits = AsyncMock(side_effect=Exception("boom"))

        with pytest.raises(HTTPException) as exc_info:
            await get_limits(limits_service=mock_limits_service)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Internal server error"
