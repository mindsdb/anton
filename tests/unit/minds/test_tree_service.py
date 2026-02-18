"""
Unit tests for TreeService.

Tests the service layer for tree operations with the MindsDB SDK,
including database listing and table/schema exploration.
"""

from unittest.mock import Mock

import pytest

from minds.schemas.tree import TreeNodeResponse
from minds.services.tree import TreeService


class TestTreeService:
    """Test suite for TreeService."""

    @pytest.fixture
    def mock_mindsdb_client(self):
        """Mock MindsDB client."""
        return Mock()

    @pytest.fixture
    def tree_service(self, mock_mindsdb_client):
        """Create TreeService instance with mocked dependencies."""
        return TreeService(mindsdb_client=mock_mindsdb_client, user_id="test-user-123")

    def _make_node(
        self, name, class_="db", type_="data", engine=None, deletable=True, visible=True, children=None, schema=None
    ):
        """Helper to create a mock tree node."""
        node = Mock()
        node.name = name
        node.class_ = class_
        node.type = type_
        node.engine = engine
        node.deletable = deletable
        node.visible = visible
        node.children = children
        if schema:
            node.schema = schema
        else:
            # Ensure hasattr(node, 'schema') still works but evaluates to falsy
            del node.schema
        return node

    def test_service_initialization(self, mock_mindsdb_client):
        """Test TreeService initialization."""
        service = TreeService(mindsdb_client=mock_mindsdb_client, user_id="user-123")

        assert service.mindsdb_client == mock_mindsdb_client
        assert service.user_id == "user-123"

    def test_get_databases_tree_success(self, tree_service, mock_mindsdb_client):
        """Test successful databases tree retrieval."""
        node1 = self._make_node("mindsdb", class_="db", type_="project", engine=None, deletable=False)
        node2 = self._make_node("test_postgres", class_="db", type_="data", engine="postgres")

        mock_mindsdb_client.tree.return_value = [node1, node2]

        result = tree_service.get_databases_tree()

        assert len(result) == 2
        assert isinstance(result[0], TreeNodeResponse)
        assert result[0].name == "mindsdb"
        assert result[1].name == "test_postgres"
        assert result[1].engine == "postgres"
        mock_mindsdb_client.tree.assert_called_once()

    def test_get_databases_tree_empty(self, tree_service, mock_mindsdb_client):
        """Test databases tree when no databases exist."""
        mock_mindsdb_client.tree.return_value = []

        result = tree_service.get_databases_tree()

        assert result == []

    def test_get_databases_tree_error(self, tree_service, mock_mindsdb_client):
        """Test databases tree with MindsDB error."""
        mock_mindsdb_client.tree.side_effect = Exception("Connection failed")

        with pytest.raises(RuntimeError, match="Failed to get databases tree"):
            tree_service.get_databases_tree()

    def test_get_database_tree_success(self, tree_service, mock_mindsdb_client):
        """Test successful database tree retrieval."""
        node1 = self._make_node("users", class_="table", type_="table")
        node2 = self._make_node("orders", class_="table", type_="table")

        mock_database = Mock()
        mock_database.tree.return_value = [node1, node2]
        mock_mindsdb_client.databases.get.return_value = mock_database

        result = tree_service.get_database_tree("test_postgres")

        assert len(result) == 2
        assert isinstance(result[0], TreeNodeResponse)
        assert result[0].name == "users"
        assert result[1].name == "orders"
        mock_mindsdb_client.databases.get.assert_called_once_with("test_postgres")
        mock_database.tree.assert_called_once_with(with_schemas=False)

    def test_get_database_tree_with_schemas(self, tree_service, mock_mindsdb_client):
        """Test database tree with schemas parameter."""
        mock_database = Mock()
        mock_database.tree.return_value = []
        mock_mindsdb_client.databases.get.return_value = mock_database

        tree_service.get_database_tree("test_postgres", with_schemas=True)

        mock_database.tree.assert_called_once_with(with_schemas=True)

    def test_get_database_tree_lowercase_name(self, tree_service, mock_mindsdb_client):
        """Test that database name is lowercased."""
        mock_database = Mock()
        mock_database.tree.return_value = []
        mock_mindsdb_client.databases.get.return_value = mock_database

        tree_service.get_database_tree("TEST_POSTGRES")

        mock_mindsdb_client.databases.get.assert_called_once_with("test_postgres")

    def test_get_database_tree_not_found(self, tree_service, mock_mindsdb_client):
        """Test database tree when database doesn't exist."""
        mock_mindsdb_client.databases.get.side_effect = AttributeError("Database doesn't exist")

        with pytest.raises(ValueError, match="Database 'test_postgres' not found"):
            tree_service.get_database_tree("test_postgres")

    def test_get_database_tree_attribute_error_other(self, tree_service, mock_mindsdb_client):
        """Test database tree with non-existence-related AttributeError."""
        mock_mindsdb_client.databases.get.side_effect = AttributeError("Some other attribute error")

        with pytest.raises(RuntimeError, match="Failed to get database tree"):
            tree_service.get_database_tree("test_postgres")

    def test_get_database_tree_error(self, tree_service, mock_mindsdb_client):
        """Test database tree with general error."""
        mock_mindsdb_client.databases.get.side_effect = Exception("Connection failed")

        with pytest.raises(RuntimeError, match="Failed to get database tree"):
            tree_service.get_database_tree("test_postgres")

    def test_get_database_tree_with_children(self, tree_service, mock_mindsdb_client):
        """Test database tree with nested children (schema nodes)."""
        child_node = self._make_node("users", class_="table", type_="table")
        schema_node = self._make_node("public", class_="schema", type_="schema")
        schema_node.children = [child_node]

        mock_database = Mock()
        mock_database.tree.return_value = [schema_node]
        mock_mindsdb_client.databases.get.return_value = mock_database

        result = tree_service.get_database_tree("test_postgres")

        assert len(result) == 1
        assert result[0].name == "public"
        assert result[0].children is not None
        assert len(result[0].children) == 1
        assert result[0].children[0].name == "users"

    def test_get_database_tree_with_schema_attribute(self, tree_service, mock_mindsdb_client):
        """Test database tree where nodes have schema attribute."""
        node = self._make_node("users", class_="table", type_="table", schema="public")

        mock_database = Mock()
        mock_database.tree.return_value = [node]
        mock_mindsdb_client.databases.get.return_value = mock_database

        result = tree_service.get_database_tree("test_postgres")

        assert len(result) == 1
        assert result[0].name == "users"
        assert result[0].schema == "public"

    def test_convert_tree_nodes_to_responses(self, tree_service):
        """Test recursive conversion of tree nodes to response dicts."""
        child = self._make_node("col1", class_="column", type_="column")
        parent = self._make_node("users", class_="table", type_="table")
        parent.children = [child]

        result = tree_service._convert_tree_nodes_to_responses([parent])

        assert len(result) == 1
        assert result[0]["name"] == "users"
        assert result[0]["children"] is not None
        assert len(result[0]["children"]) == 1
        assert result[0]["children"][0]["name"] == "col1"

    def test_convert_tree_nodes_to_responses_no_children(self, tree_service):
        """Test conversion when nodes have no children."""
        node = self._make_node("users", class_="table", type_="table")

        result = tree_service._convert_tree_nodes_to_responses([node])

        assert len(result) == 1
        assert result[0]["name"] == "users"
        assert "children" not in result[0]
