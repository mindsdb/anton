"""
Unit tests for UsageService.

Tests the aggregation queries for token counting across
both the Responses API (messages) and Chat Completions API (chat_completions).
"""

from datetime import datetime, timezone
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlmodel import Session

from minds.requests.context import Context
from minds.services.usage import UsageService, UsageServiceError


class TestUsageService:
    """Test suite for UsageService."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        session = Mock(spec=Session)
        session.exec = Mock()
        return session

    @pytest.fixture
    def mock_context(self):
        """Mock request context."""
        return Context(
            user_id=uuid4(),
            organization_id=uuid4(),
            user_email="test@test.com",
            user_roles=["user"],
        )

    @pytest.fixture
    def usage_service(self, mock_session, mock_context):
        """Create UsageService instance with mocked dependencies."""
        return UsageService(session=mock_session, context=mock_context)

    def test_service_initialization(self, mock_session, mock_context):
        """Test UsageService initialization."""
        service = UsageService(session=mock_session, context=mock_context)

        assert service.session == mock_session
        assert service.context == mock_context
        assert service.user_id == mock_context.user_id
        assert service.organization_id == mock_context.organization_id

    @pytest.mark.asyncio
    async def test_count_tokens_success(self, usage_service, mock_session):
        """Test successful token counting across both API surfaces."""
        # First call returns messages token sum, second returns chat_completions token sum
        mock_session.exec.return_value.one.side_effect = [1500, 500]

        result = await usage_service.count_tokens()

        assert result == 2000
        assert mock_session.exec.call_count == 2

    @pytest.mark.asyncio
    async def test_count_tokens_zero(self, usage_service, mock_session):
        """Test token counting when no tokens have been consumed."""
        mock_session.exec.return_value.one.side_effect = [0, 0]

        result = await usage_service.count_tokens()

        assert result == 0

    @pytest.mark.asyncio
    async def test_count_tokens_only_messages(self, usage_service, mock_session):
        """Test token counting when only Responses API has usage."""
        mock_session.exec.return_value.one.side_effect = [3000, 0]

        result = await usage_service.count_tokens()

        assert result == 3000

    @pytest.mark.asyncio
    async def test_count_tokens_only_chat_completions(self, usage_service, mock_session):
        """Test token counting when only Chat Completions API has usage."""
        mock_session.exec.return_value.one.side_effect = [0, 2500]

        result = await usage_service.count_tokens()

        assert result == 2500

    @pytest.mark.asyncio
    async def test_count_tokens_database_error(self, usage_service, mock_session):
        """Test token counting with database error."""
        mock_session.exec.side_effect = Exception("Database error")

        with pytest.raises(UsageServiceError, match="Failed to count tokens"):
            await usage_service.count_tokens()

    @pytest.mark.asyncio
    async def test_count_tokens_with_since(self, usage_service, mock_session):
        """Test token counting with since parameter for billing cycle."""
        mock_session.exec.return_value.one.side_effect = [800, 200]

        since = datetime(2026, 2, 1, tzinfo=timezone.utc)
        result = await usage_service.count_tokens(since=since)

        assert result == 1000
        assert mock_session.exec.call_count == 2

    @pytest.mark.asyncio
    async def test_count_tokens_with_until(self, usage_service, mock_session):
        """Test token counting with until parameter for billing cycle."""
        mock_session.exec.return_value.one.side_effect = [600, 400]

        until = datetime(2026, 3, 1, tzinfo=timezone.utc)
        result = await usage_service.count_tokens(until=until)

        assert result == 1000
        assert mock_session.exec.call_count == 2

    @pytest.mark.asyncio
    async def test_count_tokens_with_since_and_until(self, usage_service, mock_session):
        """Test token counting with both since and until for a bounded billing cycle."""
        mock_session.exec.return_value.one.side_effect = [300, 150]

        since = datetime(2026, 2, 1, tzinfo=timezone.utc)
        until = datetime(2026, 3, 1, tzinfo=timezone.utc)
        result = await usage_service.count_tokens(since=since, until=until)

        assert result == 450
        assert mock_session.exec.call_count == 2

    @pytest.mark.asyncio
    async def test_count_tokens_with_since_and_until_zero(self, usage_service, mock_session):
        """Test token counting returns zero for bounded period with no usage."""
        mock_session.exec.return_value.one.side_effect = [0, 0]

        since = datetime(2026, 2, 1, tzinfo=timezone.utc)
        until = datetime(2026, 3, 1, tzinfo=timezone.utc)
        result = await usage_service.count_tokens(since=since, until=until)

        assert result == 0
