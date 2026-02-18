"""
Unit tests for UsageService.

Tests the aggregation queries for token and question counting across
both the Responses API (messages) and Chat Completions API (chat_completions).
"""

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
    async def test_count_questions_success(self, usage_service, mock_session):
        """Test successful question counting across both API surfaces."""
        # First call returns messages user count, second returns chat_completions count
        mock_session.exec.return_value.one.side_effect = [10, 5]

        result = await usage_service.count_questions()

        assert result == 15
        assert mock_session.exec.call_count == 2

    @pytest.mark.asyncio
    async def test_count_questions_zero(self, usage_service, mock_session):
        """Test question counting when no questions have been asked."""
        mock_session.exec.return_value.one.side_effect = [0, 0]

        result = await usage_service.count_questions()

        assert result == 0

    @pytest.mark.asyncio
    async def test_count_questions_only_messages(self, usage_service, mock_session):
        """Test question counting when only Responses API has usage."""
        mock_session.exec.return_value.one.side_effect = [25, 0]

        result = await usage_service.count_questions()

        assert result == 25

    @pytest.mark.asyncio
    async def test_count_questions_only_chat_completions(self, usage_service, mock_session):
        """Test question counting when only Chat Completions API has usage."""
        mock_session.exec.return_value.one.side_effect = [0, 12]

        result = await usage_service.count_questions()

        assert result == 12

    @pytest.mark.asyncio
    async def test_count_questions_database_error(self, usage_service, mock_session):
        """Test question counting with database error."""
        mock_session.exec.side_effect = Exception("Database error")

        with pytest.raises(UsageServiceError, match="Failed to count questions"):
            await usage_service.count_questions()
