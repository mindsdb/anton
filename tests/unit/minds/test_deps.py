"""
Unit tests for shared FastAPI dependency functions.

Tests the dependency factories in minds.api.v1.deps that construct
service objects from request context.
"""

from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from minds.api.v1.deps import (
    get_context,
    get_conversations_service,
    get_data_catalog_loader,
    get_datasources_service,
    get_limits_service,
    get_minds_service,
    get_mindsdb_client,
    get_tree_service,
    get_usage_service,
)
from minds.requests.context import Context
from minds.services.conversations import ConversationsService
from minds.services.data_catalog.data_catalog_loader import DataCatalogLoader
from minds.services.datasources import DatasourcesService
from minds.services.limits import LimitsService
from minds.services.minds import MindsService
from minds.services.tree import TreeService
from minds.services.usage import UsageService


class TestDeps:
    """Test suite for shared dependency functions."""

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request."""
        return Mock()

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        return Mock()

    @pytest.fixture
    def mock_context(self):
        """Mock Context."""
        return Context(
            user_id=uuid4(),
            organization_id=uuid4(),
            user_email="test@test.com",
            user_roles=["user"],
        )

    @pytest.fixture
    def mock_mindsdb_client(self):
        """Mock MindsDB client."""
        return Mock()

    # -- Low-level building blocks ---------------------------------------------

    def test_get_context(self, mock_request, mock_context):
        """Test get_context extracts context from request."""
        with patch("minds.api.v1.deps.extract_context_from_request", return_value=mock_context) as mock_extract:
            result = get_context(mock_request)

            assert result == mock_context
            mock_extract.assert_called_once_with(mock_request)

    def test_get_mindsdb_client(self, mock_request, mock_context, mock_mindsdb_client):
        """Test get_mindsdb_client uses injected context instead of re-extracting."""
        with patch(
            "minds.api.v1.deps.create_mindsdb_client_from_request", return_value=mock_mindsdb_client
        ) as mock_create:
            result = get_mindsdb_client(mock_request, context=mock_context)

            assert result == mock_mindsdb_client
            mock_create.assert_called_once_with(mock_request, mock_context)

    # -- Service factories -----------------------------------------------------

    def test_get_minds_service(self, mock_session, mock_context, mock_mindsdb_client):
        """Test get_minds_service creates MindsService correctly."""
        result = get_minds_service(context=mock_context, session=mock_session, mindsdb_client=mock_mindsdb_client)

        assert isinstance(result, MindsService)
        assert result.session == mock_session
        assert result.user_id == mock_context.user_id
        assert result.organization_id == mock_context.organization_id
        assert result.mindsdb_client == mock_mindsdb_client

    def test_get_conversations_service(self, mock_session, mock_context, mock_mindsdb_client):
        """Test get_conversations_service creates ConversationsService correctly."""
        result = get_conversations_service(
            context=mock_context, session=mock_session, mindsdb_client=mock_mindsdb_client
        )

        assert isinstance(result, ConversationsService)
        assert result.session == mock_session
        assert result.user_id == mock_context.user_id
        assert result.organization_id == mock_context.organization_id
        assert result.mindsdb_client == mock_mindsdb_client

    def test_get_datasources_service(self, mock_session, mock_context, mock_mindsdb_client):
        """Test get_datasources_service creates DatasourcesService correctly."""
        result = get_datasources_service(context=mock_context, session=mock_session, mindsdb_client=mock_mindsdb_client)

        assert isinstance(result, DatasourcesService)
        assert result.session == mock_session
        assert result.user_id == mock_context.user_id
        assert result.organization_id == mock_context.organization_id
        assert result.mindsdb_client == mock_mindsdb_client

    def test_get_data_catalog_loader(self, mock_session, mock_context):
        """Test get_data_catalog_loader creates DataCatalogLoader correctly."""
        result = get_data_catalog_loader(context=mock_context, session=mock_session)

        assert isinstance(result, DataCatalogLoader)
        assert result.session == mock_session
        assert result.organization_id == mock_context.organization_id
        assert result.user_id == mock_context.user_id

    def test_get_tree_service(self, mock_context, mock_mindsdb_client):
        """Test get_tree_service creates TreeService correctly."""
        result = get_tree_service(context=mock_context, mindsdb_client=mock_mindsdb_client)

        assert isinstance(result, TreeService)
        assert result.mindsdb_client == mock_mindsdb_client
        assert result.user_id == mock_context.user_id

    def test_get_usage_service(self, mock_session, mock_context):
        """Test get_usage_service creates UsageService correctly."""
        result = get_usage_service(context=mock_context, session=mock_session)

        assert isinstance(result, UsageService)
        assert result.session == mock_session
        assert result.user_id == mock_context.user_id
        assert result.organization_id == mock_context.organization_id

    def test_get_limits_service(self, mock_context, mock_mindsdb_client):
        """Test get_limits_service creates LimitsService with all sub-services."""
        mock_minds_service = Mock(spec=MindsService)
        mock_datasources_service = Mock(spec=DatasourcesService)
        mock_usage_service = Mock(spec=UsageService)
        mock_settings = Mock()

        with patch("minds.api.v1.deps.get_app_settings", return_value=mock_settings):
            result = get_limits_service(
                context=mock_context,
                minds_service=mock_minds_service,
                datasources_service=mock_datasources_service,
                usage_service=mock_usage_service,
            )

        assert isinstance(result, LimitsService)
        assert result.minds_service == mock_minds_service
        assert result.datasources_service == mock_datasources_service
        assert result.usage_service == mock_usage_service
        assert result.context == mock_context
        assert result.settings == mock_settings
