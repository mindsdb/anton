import pytest
from unittest.mock import Mock
from fastapi import Request, HTTPException, status

from minds.common.authentication import (
    get_authorization_bearer_token,
    get_api_key_from_request,
    get_api_key_from_headers,
    AuthHeaders,
)


class TestGetAuthorizationBearerToken:
    """Test cases for get_authorization_bearer_token function."""

    def test_get_authorization_bearer_token_from_request_success(self):
        """Test successful token extraction from FastAPI Request object."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "Bearer test-api-key-123"}

        # Act
        result = get_authorization_bearer_token(mock_request)

        # Assert
        assert result == "test-api-key-123"

    def test_get_authorization_bearer_token_from_request_no_header(self):
        """Test token extraction when no authorization header is present."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_request.headers = {}

        # Act
        result = get_authorization_bearer_token(mock_request)

        # Assert
        assert result is None

    def test_get_authorization_bearer_token_from_request_invalid_format(self):
        """Test token extraction with invalid authorization header format."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "Basic invalid-format"}

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            get_authorization_bearer_token(mock_request)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Authorization header must start with 'Bearer '" in str(
            exc_info.value.detail
        )

    def test_get_authorization_bearer_token_from_request_bearer_only(self):
        """Test token extraction when only 'Bearer ' is present without token."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "Bearer "}

        # Act
        result = get_authorization_bearer_token(mock_request)

        # Assert
        assert result is None

    def test_get_authorization_bearer_token_from_dict_lowercase_key(self):
        """Test successful token extraction from dictionary with lowercase authorization key."""
        # Arrange
        headers_dict = {"authorization": "Bearer test-dict-key-456"}

        # Act
        result = get_authorization_bearer_token(headers_dict)

        # Assert
        assert result == "test-dict-key-456"

    def test_get_authorization_bearer_token_from_dict_uppercase_key(self):
        """Test successful token extraction from dictionary with uppercase Authorization key."""
        # Arrange
        headers_dict = {"Authorization": "Bearer test-dict-key-789"}

        # Act
        result = get_authorization_bearer_token(headers_dict)

        # Assert
        assert result == "test-dict-key-789"

    def test_get_authorization_bearer_token_from_dict_both_keys_prefers_lowercase(self):
        """Test token extraction when both authorization keys are present (should prefer lowercase)."""
        # Arrange
        headers_dict = {
            "authorization": "Bearer lowercase-key",
            "Authorization": "Bearer uppercase-key",
        }

        # Act
        result = get_authorization_bearer_token(headers_dict)

        # Assert
        assert result == "lowercase-key"

    def test_get_authorization_bearer_token_from_dict_no_header(self):
        """Test token extraction from dictionary when no authorization header is present."""
        # Arrange
        headers_dict = {"content-type": "application/json"}

        # Act
        result = get_authorization_bearer_token(headers_dict)

        # Assert
        assert result is None

    def test_get_authorization_bearer_token_from_dict_invalid_format(self):
        """Test token extraction from dictionary with invalid format."""
        # Arrange
        headers_dict = {"authorization": "Token invalid-format"}

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            get_authorization_bearer_token(headers_dict)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Authorization header must start with 'Bearer '" in str(
            exc_info.value.detail
        )

    def test_get_authorization_bearer_token_from_auth_headers_model_success(self):
        """Test successful token extraction from AuthHeaders model."""
        # Arrange
        auth_headers = AuthHeaders(authorization="Bearer model-key-123")

        # Act
        result = get_authorization_bearer_token(auth_headers)

        # Assert
        assert result == "model-key-123"

    def test_get_authorization_bearer_token_from_auth_headers_model_none(self):
        """Test token extraction from AuthHeaders model when authorization is None."""
        # Arrange
        auth_headers = AuthHeaders(authorization=None)

        # Act
        result = get_authorization_bearer_token(auth_headers)

        # Assert
        assert result is None

    def test_get_authorization_bearer_token_from_auth_headers_model_invalid_format(
        self,
    ):
        """Test token extraction from AuthHeaders model with invalid format."""
        # Arrange
        auth_headers = AuthHeaders(authorization="ApiKey invalid-format")

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            get_authorization_bearer_token(auth_headers)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Authorization header must start with 'Bearer '" in str(
            exc_info.value.detail
        )

    def test_get_authorization_bearer_token_unsupported_type(self):
        """Test token extraction with unsupported input type."""
        # Arrange
        unsupported_input = ["Bearer", "test-key"]

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            get_authorization_bearer_token(unsupported_input)

        assert "Unsupported type" in str(exc_info.value)
        assert "Expected Request, dict, or AuthHeaders" in str(exc_info.value)

    def test_get_authorization_bearer_token_with_whitespace_in_token(self):
        """Test token extraction with whitespace in the token."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "Bearer   token-with-spaces   "}

        # Act
        result = get_authorization_bearer_token(mock_request)

        # Assert
        assert result == "  token-with-spaces   "

    def test_get_authorization_bearer_token_complex_token(self):
        """Test token extraction with complex token containing special characters."""
        # Arrange
        complex_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWV9"
        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": f"Bearer {complex_token}"}

        # Act
        result = get_authorization_bearer_token(mock_request)

        # Assert
        assert result == complex_token


class TestGetApiKeyFromRequest:
    """Test cases for get_api_key_from_request function."""

    def test_get_api_key_from_request_success(self):
        """Test successful API key extraction from request."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "Bearer request-api-key"}

        # Act
        result = get_api_key_from_request(mock_request)

        # Assert
        assert result == "request-api-key"

    def test_get_api_key_from_request_no_header(self):
        """Test API key extraction when no authorization header is present."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_request.headers = {}

        # Act
        result = get_api_key_from_request(mock_request)

        # Assert
        assert result is None

    def test_get_api_key_from_request_invalid_format(self):
        """Test API key extraction with invalid format."""
        # Arrange
        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "Basic invalid-format"}

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            get_api_key_from_request(mock_request)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Authorization header must start with 'Bearer '" in str(
            exc_info.value.detail
        )


class TestGetApiKeyFromHeaders:
    """Test cases for get_api_key_from_headers function."""

    def test_get_api_key_from_headers_success(self):
        """Test successful API key extraction from headers dictionary."""
        # Arrange
        headers_dict = {"authorization": "Bearer headers-api-key"}

        # Act
        result = get_api_key_from_headers(headers_dict)

        # Assert
        assert result == "headers-api-key"

    def test_get_api_key_from_headers_uppercase_key(self):
        """Test API key extraction from headers with uppercase Authorization key."""
        # Arrange
        headers_dict = {"Authorization": "Bearer uppercase-headers-key"}

        # Act
        result = get_api_key_from_headers(headers_dict)

        # Assert
        assert result == "uppercase-headers-key"

    def test_get_api_key_from_headers_no_header(self):
        """Test API key extraction when no authorization header is present."""
        # Arrange
        headers_dict = {"content-type": "application/json"}

        # Act
        result = get_api_key_from_headers(headers_dict)

        # Assert
        assert result is None

    def test_get_api_key_from_headers_invalid_format(self):
        """Test API key extraction with invalid format."""
        # Arrange
        headers_dict = {"authorization": "Token invalid-format"}

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            get_api_key_from_headers(headers_dict)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Authorization header must start with 'Bearer '" in str(
            exc_info.value.detail
        )


class TestAuthHeaders:
    """Test cases for AuthHeaders model."""

    def test_auth_headers_creation_with_authorization(self):
        """Test AuthHeaders model creation with authorization value."""
        # Act
        auth_headers = AuthHeaders(authorization="Bearer test-token")

        # Assert
        assert auth_headers.authorization == "Bearer test-token"

    def test_auth_headers_creation_without_authorization(self):
        """Test AuthHeaders model creation without authorization (should default to None)."""
        # Act
        auth_headers = AuthHeaders()

        # Assert
        assert auth_headers.authorization is None

    def test_auth_headers_creation_with_none_authorization(self):
        """Test AuthHeaders model creation with explicit None authorization."""
        # Act
        auth_headers = AuthHeaders(authorization=None)

        # Assert
        assert auth_headers.authorization is None
