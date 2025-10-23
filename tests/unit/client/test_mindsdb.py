from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException, Request
from mindsdb_sdk.server import Server

from minds.client.mindsdb import (
    create_mindsdb_client,
    create_mindsdb_client_from_request,
)


class TestCreateMindsdbClientFromRequest:
    """Test cases for create_mindsdb_client_from_request function."""

    @patch("minds.client.mindsdb.get_authorization_bearer_token")
    @patch("minds.client.mindsdb.create_mindsdb_client")
    def test_create_mindsdb_client_from_request_success(self, mock_create_client, mock_get_token):
        """Test successful client creation from request."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_api_key = "test-api-key"
        mock_client = Mock(spec=Server)

        mock_get_token.return_value = mock_api_key
        mock_create_client.return_value = mock_client

        # Act
        result = create_mindsdb_client_from_request(mock_request)

        # Assert
        mock_get_token.assert_called_once_with(mock_request)
        mock_create_client.assert_called_once_with(mock_api_key)
        assert result == mock_client

    @patch("minds.client.mindsdb.get_authorization_bearer_token")
    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_from_request_no_token(self, mock_connect, mock_get_token):
        """Test client creation when no token is found connects without auth."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_get_token.return_value = None
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Act
        result = create_mindsdb_client_from_request(mock_request)

        # Assert
        mock_get_token.assert_called_once_with(mock_request)
        mock_connect.assert_called_once()  # Should connect without auth
        assert result == mock_client

    @patch("minds.client.mindsdb.get_authorization_bearer_token")
    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_from_request_empty_token(self, mock_connect, mock_get_token):
        """Test client creation when token is empty string connects without auth."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_get_token.return_value = ""
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Act
        result = create_mindsdb_client_from_request(mock_request)

        # Assert
        mock_get_token.assert_called_once_with(mock_request)
        mock_connect.assert_called_once()  # Should connect without auth
        assert result == mock_client

    @patch("minds.client.mindsdb.get_authorization_bearer_token")
    def test_create_mindsdb_client_from_request_auth_exception(self, mock_get_token):
        """Test client creation when authentication raises HTTPException."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_get_token.side_effect = HTTPException(status_code=401, detail="Invalid token")

        # Act & Assert
        with pytest.raises(HTTPException):
            create_mindsdb_client_from_request(mock_request)

        mock_get_token.assert_called_once_with(mock_request)

    @patch("minds.client.mindsdb.connect")
    @patch("minds.client.mindsdb.MINDSDB_URL", "http://localhost:47334")
    def test_full_flow_success(self, mock_connect):
        """Test the full flow from request to client creation."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "Bearer test-token-123"}
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Act
        result = create_mindsdb_client_from_request(mock_request)

        # Assert
        mock_connect.assert_called_once_with(url="http://localhost:47334", api_key="test-token-123")
        assert result == mock_client

    @patch("minds.client.mindsdb.connect")
    def test_full_flow_invalid_auth_header(self, mock_connect):
        """Test the full flow with invalid authorization header."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "InvalidFormat test-token-123"}

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            create_mindsdb_client_from_request(mock_request)

        assert exc_info.value.status_code == 401
        assert "Authorization header must start with 'Bearer '" in str(exc_info.value.detail)
        mock_connect.assert_not_called()

    @patch("minds.client.mindsdb.connect")
    def test_full_flow_missing_auth_header(self, mock_connect):
        """Test the full flow with missing authorization header connects without auth."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_request.headers = {}
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Act
        result = create_mindsdb_client_from_request(mock_request)

        # Assert - should connect without auth when no header
        mock_connect.assert_called_once()
        assert result == mock_client


class TestCreateMindsdbClient:
    """Test cases for create_mindsdb_client function."""

    @patch("minds.client.mindsdb.connect")
    @patch("minds.client.mindsdb.MINDSDB_URL", "http://test-server:8080")
    def test_create_mindsdb_client_success(self, mock_connect):
        """Test successful client creation with valid API key."""
        # Arrange
        api_key = "valid-api-key"
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Act
        result = create_mindsdb_client(api_key)

        # Assert
        mock_connect.assert_called_once_with(url="http://test-server:8080", api_key=api_key)
        assert result == mock_client

    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_none_api_key(self, mock_connect):
        """Test client creation with None API key connects without auth."""
        # Arrange
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Act
        result = create_mindsdb_client(None)

        # Assert - should connect without auth when no API key
        mock_connect.assert_called_once()
        assert result == mock_client

    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_empty_api_key(self, mock_connect):
        """Test client creation with empty API key connects without auth."""

        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        result = create_mindsdb_client("")

        # Assert - should connect without auth when empty API key
        mock_connect.assert_called_once()
        assert result == mock_client

    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_whitespace_api_key(self, mock_connect):
        """Test client creation with whitespace-only API key connects without auth."""
        # Arrange
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Act
        result = create_mindsdb_client("   ")

        # Assert - should connect without auth when whitespace API key
        mock_connect.assert_called_once()
        assert result == mock_client

    @patch("minds.client.mindsdb.connect")
    @patch("minds.client.mindsdb.MINDSDB_URL", "https://production-server.com")
    def test_create_mindsdb_client_with_production_server(self, mock_connect):
        """Test client creation with production server URL."""
        # Arrange
        api_key = "prod-api-key"
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Act
        result = create_mindsdb_client(api_key)

        # Assert
        mock_connect.assert_called_once_with(url="https://production-server.com", api_key=api_key)
        assert result == mock_client

    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_connection_error(self, mock_connect):
        """Test client creation when connection fails."""
        # Arrange
        api_key = "valid-api-key"
        mock_connect.side_effect = ConnectionError("Unable to connect to MindsDB server")

        # Act & Assert
        with pytest.raises(ConnectionError, match="Unable to connect to MindsDB server"):
            create_mindsdb_client(api_key)

        mock_connect.assert_called_once()

    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_authentication_error(self, mock_connect):
        """Test client creation when authentication fails."""
        # Arrange
        api_key = "invalid-api-key"
        mock_connect.side_effect = Exception("Authentication failed")

        # Act & Assert
        with pytest.raises(Exception, match="Authentication failed"):
            create_mindsdb_client(api_key)

        mock_connect.assert_called_once()
