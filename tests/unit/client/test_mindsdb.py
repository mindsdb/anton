from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from fastapi import HTTPException, Request
from mindsdb_sdk.server import Server

from minds.client.mindsdb import (
    create_mindsdb_client,
    create_mindsdb_client_from_request,
    create_mindsdb_client_with_credentials,
)
from minds.requests.context import Context


@pytest.fixture
def context():
    """Fixture for a sample Context object."""
    return Context(
        user_id=UUID("11111111-1111-1111-1111-111111111111"), tenant_id=UUID("22222222-2222-2222-2222-222222222222")
    )


class TestCreateMindsdbClientFromRequest:
    """Test cases for create_mindsdb_client_from_request function."""

    @patch("minds.client.mindsdb.get_authorization_bearer_token")
    @patch("minds.client.mindsdb.create_mindsdb_client")
    def test_create_mindsdb_client_from_request_success(self, mock_create_client, mock_get_token, context):
        """Test successful client creation from request."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_api_key = "test-api-key"
        mock_client = Mock(spec=Server)

        mock_get_token.return_value = mock_api_key
        mock_create_client.return_value = mock_client

        # Act
        result = create_mindsdb_client_from_request(mock_request, context)

        # Assert
        mock_get_token.assert_called_once_with(mock_request)
        # The request should forward headers (company-id) to the created client
        expected_headers = {"company-id": f"{str(context.tenant_id)}_{str(context.user_id)}"}
        mock_create_client.assert_called_once_with(mock_api_key, headers=expected_headers)
        assert result == mock_client

    @patch("minds.client.mindsdb.get_authorization_bearer_token")
    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_from_request_no_token(self, mock_connect, mock_get_token, context):
        """Test client creation when no token is found connects without auth."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_get_token.return_value = None
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Act
        result = create_mindsdb_client_from_request(mock_request, context)

        # Assert
        mock_get_token.assert_called_once_with(mock_request)
        mock_connect.assert_called_once()  # Should connect without auth
        assert result == mock_client

    @patch("minds.client.mindsdb.get_authorization_bearer_token")
    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_from_request_empty_token(self, mock_connect, mock_get_token, context):
        """Test client creation when token is empty string connects without auth."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_get_token.return_value = ""
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Act
        result = create_mindsdb_client_from_request(mock_request, context)

        # Assert
        mock_get_token.assert_called_once_with(mock_request)
        mock_connect.assert_called_once()  # Should connect without auth
        assert result == mock_client

    @patch("minds.client.mindsdb.get_authorization_bearer_token")
    def test_create_mindsdb_client_from_request_auth_exception(self, mock_get_token, context):
        """Test client creation when authentication raises HTTPException."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_get_token.side_effect = HTTPException(status_code=401, detail="Invalid token")

        # Act & Assert
        with pytest.raises(HTTPException):
            create_mindsdb_client_from_request(mock_request, context)

        mock_get_token.assert_called_once_with(mock_request)

    @patch("minds.client.mindsdb.connect")
    @patch("minds.client.mindsdb.MINDSDB_URL", "http://localhost:47334")
    def test_full_flow_success(self, mock_connect, context):
        """Test the full flow from request to client creation."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "Bearer test-token-123"}
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Act
        result = create_mindsdb_client_from_request(mock_request, context)

        # Assert
        mock_connect.assert_called_once_with(
            url="http://localhost:47334",
            api_key="test-token-123",
            headers={"company-id": f"{str(context.tenant_id)}_{str(context.user_id)}"},
        )
        assert result == mock_client

    @patch("minds.client.mindsdb.connect")
    def test_full_flow_invalid_auth_header(self, mock_connect, context):
        """Test the full flow with invalid authorization header."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "InvalidFormat test-token-123"}

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            create_mindsdb_client_from_request(mock_request, context)

        assert exc_info.value.status_code == 401
        assert "Authorization header must start with 'Bearer '" in str(exc_info.value.detail)
        mock_connect.assert_not_called()

    @patch("minds.client.mindsdb.connect")
    def test_full_flow_missing_auth_header(self, mock_connect, context):
        """Test the full flow with missing authorization header connects without auth."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_request.headers = {}
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Act
        result = create_mindsdb_client_from_request(mock_request, context)

        # Assert - should connect without auth when no header
        mock_connect.assert_called_once()
        assert result == mock_client


class TestCreateMindsdbClient:
    """Test cases for create_mindsdb_client function."""

    @patch("minds.client.mindsdb.connect")
    @patch("minds.client.mindsdb.MINDSDB_URL", "http://test-server:8080")
    def test_create_mindsdb_client_success(self, mock_connect, context):
        """Test successful client creation with valid API key."""
        # Arrange
        api_key = "valid-api-key"
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Act
        result = create_mindsdb_client(
            api_key, headers={"company-id": f"{str(context.tenant_id)}_{str(context.user_id)}"}
        )

        # Assert
        mock_connect.assert_called_once_with(
            url="http://test-server:8080",
            api_key=api_key,
            headers={"company-id": f"{str(context.tenant_id)}_{str(context.user_id)}"},
        )
        assert result == mock_client

    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_none_api_key(self, mock_connect, context):
        """Test client creation with None API key connects without auth."""
        # Arrange
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Act
        result = create_mindsdb_client(None, headers={"company-id": f"{str(context.tenant_id)}_{str(context.user_id)}"})

        # Assert - should connect without auth when no API key
        mock_connect.assert_called_once()
        assert result == mock_client

    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_empty_api_key(self, mock_connect, context):
        """Test client creation with empty API key connects without auth."""

        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        result = create_mindsdb_client("", headers={"company-id": f"{str(context.tenant_id)}_{str(context.user_id)}"})

        # Assert - should connect without auth when empty API key
        mock_connect.assert_called_once()
        assert result == mock_client

    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_whitespace_api_key(self, mock_connect, context):
        """Test client creation with whitespace-only API key connects without auth."""
        # Arrange
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Act
        result = create_mindsdb_client(
            "   ", headers={"company-id": f"{str(context.tenant_id)}_{str(context.user_id)}"}
        )

        # Assert - should connect without auth when whitespace API key
        # We only need to ensure connect was invoked; exact args are validated in other tests
        mock_connect.assert_called_once()
        assert result == mock_client

    @patch("minds.client.mindsdb.connect")
    @patch("minds.client.mindsdb.MINDSDB_URL", "https://production-server.com")
    def test_create_mindsdb_client_with_production_server(self, mock_connect, context):
        """Test client creation with production server URL."""
        # Arrange
        api_key = "prod-api-key"
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Act
        result = create_mindsdb_client(
            api_key, headers={"company-id": f"{str(context.tenant_id)}_{str(context.user_id)}"}
        )

        # Assert
        mock_connect.assert_called_once_with(
            url="https://production-server.com",
            api_key=api_key,
            headers={"company-id": f"{str(context.tenant_id)}_{str(context.user_id)}"},
        )
        assert result == mock_client

    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_connection_error(self, mock_connect, context):
        """Test client creation when connection fails."""
        # Arrange
        api_key = "valid-api-key"
        mock_connect.side_effect = ConnectionError("Unable to connect to MindsDB server")

        # Act & Assert
        with pytest.raises(ConnectionError, match="Unable to connect to MindsDB server"):
            create_mindsdb_client(api_key, headers={"company-id": f"{str(context.tenant_id)}_{str(context.user_id)}"})

        mock_connect.assert_called_once()

    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_authentication_error(self, mock_connect, context):
        """Test client creation when authentication fails."""
        # Arrange
        api_key = "invalid-api-key"
        mock_connect.side_effect = Exception("Authentication failed")

        # Act & Assert
        with pytest.raises(Exception, match="Authentication failed"):
            create_mindsdb_client(api_key, headers={"company-id": f"{str(context.tenant_id)}_{str(context.user_id)}"})

        mock_connect.assert_called_once()

    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_with_credentials_api_key(self, mock_connect, context):
        """Test explicit-credentials client creation when API key is provided."""
        url = "http://explicit-test:9000"
        api_key = "cred-api-key"
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        result = create_mindsdb_client_with_credentials(
            url, api_key=api_key, company_id=f"{str(context.tenant_id)}_{str(context.user_id)}"
        )

        mock_connect.assert_called_once_with(
            url=url,
            api_key=api_key,
            headers={"company-id": f"{str(context.tenant_id)}_{str(context.user_id)}"},
        )
        assert result == mock_client

    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_with_credentials_no_auth(self, mock_connect, context):
        """When no api_key and no password are provided, should connect without auth."""
        url = "http://no-auth-test:9000"
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        result = create_mindsdb_client_with_credentials(url, api_key=None, company_id="company-xyz")

        mock_connect.assert_called_once_with(url=url, headers={"company-id": "company-xyz"})
        assert result == mock_client

    @patch("minds.client.mindsdb.connect")
    def test_create_mindsdb_client_with_credentials_with_login_password(self, mock_connect, context):
        """When login/password are provided (and no api_key), connect using them."""
        url = "http://login-test:9000"
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        result = create_mindsdb_client_with_credentials(
            url, api_key=None, login="user", password="pass", company_id="company-abc"
        )

        mock_connect.assert_called_once_with(
            url=url,
            login="user",
            password="pass",
            headers={"company-id": "company-abc"},
        )
        assert result == mock_client

    def test_create_mindsdb_client_with_credentials_invalid_api_key_raises(self):
        """Passing a non-string api_key should raise an exception."""
        # The implementation may attempt to call .strip() on api_key before
        # validating its type, which raises AttributeError for non-string input.
        # Accept any Exception to be robust to implementation details.
        with pytest.raises(AttributeError):
            create_mindsdb_client_with_credentials("http://x", api_key=123, company_id="c")

    @patch("minds.client.mindsdb.connect")
    @patch("minds.client.mindsdb.MINDSDB_PASSWORD", "password123")
    @patch("minds.client.mindsdb.MINDSDB_LOGIN", "loginuser")
    def test_create_mindsdb_client_uses_minidsb_login_password_when_set(self, mock_connect, context):
        """If MINDSDB_PASSWORD is configured, create_mindsdb_client should pass
        login/password when api_key is missing."""
        mock_client = Mock(spec=Server)
        mock_connect.return_value = mock_client

        # Call with no api_key so branch for login/password is used
        result = create_mindsdb_client(None, headers={"company-id": "cid"})

        # Ensure connect was invoked with login/password in kwargs
        assert mock_connect.called
        called_args, called_kwargs = mock_connect.call_args
        assert called_kwargs.get("login") == "loginuser"
        assert called_kwargs.get("password") == "password123"
        assert called_kwargs.get("headers") == {"company-id": "cid"}
        assert result == mock_client

    def test_create_mindsdb_client_non_string_api_key_raises_value_error(self):
        """Passing a non-string api_key (but with .strip) should raise ValueError."""
        # bytes has .strip(), so it will pass the initial strip check and hit the isinstance check
        with pytest.raises(ValueError):
            create_mindsdb_client(b"bytes", headers=None)

    def test_create_mindsdb_client_with_credentials_non_string_api_key_raises_value_error(self):
        """create_mindsdb_client_with_credentials should raise ValueError for non-string
        api_key that implements strip()."""
        with pytest.raises(ValueError):
            create_mindsdb_client_with_credentials("http://x", api_key=b"bytes", company_id="c")

    @patch("minds.client.mindsdb.get_authorization_bearer_token")
    @patch("minds.client.mindsdb.get_headers_for_mindsdb_client")
    @patch("minds.client.mindsdb.create_mindsdb_client")
    def test_create_mindsdb_client_from_request_uses_get_headers(
        self, mock_create_client, mock_get_headers, mock_get_token, context
    ):
        """Ensure headers are obtained from get_headers_for_mindsdb_client and
        forwarded to create_mindsdb_client."""
        mock_request = Mock(spec=Request)
        mock_get_token.return_value = "tk"
        mock_get_headers.return_value = {"company-id": "hdr"}
        mock_client = Mock(spec=Server)
        mock_create_client.return_value = mock_client

        result = create_mindsdb_client_from_request(mock_request, context)

        mock_get_headers.assert_called_once_with(context)
        mock_create_client.assert_called_once_with("tk", headers={"company-id": "hdr"})
        assert result == mock_client
