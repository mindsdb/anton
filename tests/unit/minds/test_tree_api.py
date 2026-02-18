"""
Unit tests for tree API endpoints.
"""

from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from minds.api.v1.endpoints.tree import get_database_tree, get_databases_tree
from minds.schemas.tree import TreeNodeResponse
from minds.services.tree import TreeService


class TestTreeAPI:
    """Test suite for Tree API endpoints."""

    @pytest.fixture
    def mock_tree_service(self):
        """Mock TreeService instance."""
        return Mock(spec=TreeService)

    @pytest.fixture
    def sample_database_nodes(self):
        """Sample database tree nodes."""
        return [
            TreeNodeResponse(
                name="mindsdb",
                **{"class": "db"},
                type="project",
                engine=None,
                deletable=False,
                visible=True,
            ),
            TreeNodeResponse(
                name="test_postgres",
                **{"class": "db"},
                type="data",
                engine="postgres",
                deletable=True,
                visible=True,
            ),
        ]

    @pytest.fixture
    def sample_table_nodes(self):
        """Sample table tree nodes."""
        return [
            TreeNodeResponse(
                name="users",
                **{"class": "table"},
                type="table",
                engine=None,
                deletable=False,
                visible=True,
            ),
            TreeNodeResponse(
                name="orders",
                **{"class": "table"},
                type="table",
                engine=None,
                deletable=False,
                visible=True,
            ),
        ]

    @pytest.mark.asyncio
    async def test_get_databases_tree_success(self, mock_tree_service, sample_database_nodes):
        """Test successful databases tree retrieval."""
        mock_tree_service.get_databases_tree.return_value = sample_database_nodes

        result = await get_databases_tree(tree_service=mock_tree_service)

        assert len(result) == 2
        assert result[0].name == "mindsdb"
        assert result[1].name == "test_postgres"
        mock_tree_service.get_databases_tree.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_databases_tree_empty(self, mock_tree_service):
        """Test databases tree when no databases exist."""
        mock_tree_service.get_databases_tree.return_value = []

        result = await get_databases_tree(tree_service=mock_tree_service)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_databases_tree_error(self, mock_tree_service):
        """Test databases tree with unexpected error."""
        mock_tree_service.get_databases_tree.side_effect = Exception("Connection failed")

        with pytest.raises(HTTPException) as exc_info:
            await get_databases_tree(tree_service=mock_tree_service)

        assert exc_info.value.status_code == 500
        assert "Failed to get databases tree" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_database_tree_success(self, mock_tree_service, sample_table_nodes):
        """Test successful database tree retrieval."""
        mock_tree_service.get_database_tree.return_value = sample_table_nodes

        result = await get_database_tree(
            database_name="test_postgres",
            with_schemas=False,
            tree_service=mock_tree_service,
        )

        assert len(result) == 2
        assert result[0].name == "users"
        assert result[1].name == "orders"
        mock_tree_service.get_database_tree.assert_called_once_with("test_postgres", with_schemas=False)

    @pytest.mark.asyncio
    async def test_get_database_tree_with_schemas(self, mock_tree_service, sample_table_nodes):
        """Test database tree retrieval with schemas enabled."""
        mock_tree_service.get_database_tree.return_value = sample_table_nodes

        result = await get_database_tree(
            database_name="test_postgres",
            with_schemas=True,
            tree_service=mock_tree_service,
        )

        assert len(result) == 2
        mock_tree_service.get_database_tree.assert_called_once_with("test_postgres", with_schemas=True)

    @pytest.mark.asyncio
    async def test_get_database_tree_not_found(self, mock_tree_service):
        """Test database tree when database doesn't exist."""
        mock_tree_service.get_database_tree.side_effect = ValueError("Database 'missing' not found")

        with pytest.raises(HTTPException) as exc_info:
            await get_database_tree(
                database_name="missing",
                tree_service=mock_tree_service,
            )

        assert exc_info.value.status_code == 404
        assert "Database 'missing' not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_database_tree_error(self, mock_tree_service):
        """Test database tree with unexpected error."""
        mock_tree_service.get_database_tree.side_effect = Exception("Connection failed")

        with pytest.raises(HTTPException) as exc_info:
            await get_database_tree(
                database_name="test_postgres",
                tree_service=mock_tree_service,
            )

        assert exc_info.value.status_code == 500
        assert "Failed to get database tree" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_database_tree_empty(self, mock_tree_service):
        """Test database tree when database has no tables."""
        mock_tree_service.get_database_tree.return_value = []

        result = await get_database_tree(
            database_name="empty_db",
            tree_service=mock_tree_service,
        )

        assert result == []
