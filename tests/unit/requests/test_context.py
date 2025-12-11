from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from fastapi import HTTPException, Request

from minds.requests.context import (
    Context,
    LangfuseContext,
    LangfuseContextMetadata,
    create_langfuse_context,
    extract_context_from_request,
)


class TestContext:
    """Test cases for Context model."""

    def test_context_initialization_with_defaults(self):
        """Test Context initialization with default values."""
        context = Context()

        assert context.user_id == UUID("00000000-0000-0000-0000-000000000000")
        assert context.tenant_id == UUID("00000000-0000-0000-0000-000000000000")

    def test_context_initialization_with_values(self):
        """Test Context initialization with provided values."""
        context = Context(
            user_id=UUID("00000000-0000-0000-0000-000000000001"), tenant_id=UUID("00000000-0000-0000-0000-000000000002")
        )

        assert context.user_id == UUID("00000000-0000-0000-0000-000000000001")
        assert context.tenant_id == UUID("00000000-0000-0000-0000-000000000002")

    def test_context_is_basemodel(self):
        """Test that Context inherits from BaseModel."""
        context = Context()

        # Should have BaseModel methods
        assert hasattr(context, "model_dump")
        assert hasattr(context, "model_validate")


class TestExtractContextFromRequest:
    """Test cases for extract_context_from_request function."""

    def test_extract_context_with_all_headers(self):
        """Test extracting context when all headers are present and auth is enabled."""
        with patch("minds.requests.context.settings.auth.disable", False):
            mock_request = Mock(spec=Request)
            mock_request.headers = {
                "x-user-id": "1",
                "x-company-id": "2",
            }

            context = extract_context_from_request(mock_request)

            assert context.user_id == UUID("00000000-0000-0000-0000-000000000001")
            assert context.tenant_id == UUID("00000000-0000-0000-0000-000000000002")

    def test_extract_context_with_auth_disabled(self):
        """Test extracting context when auth is disabled."""
        with patch("minds.requests.context.settings.auth.disable", True):
            mock_request = Mock(spec=Request)
            # Headers should be ignored when auth is disabled
            mock_request.headers = {
                "x-user-id": "1",
                "x-company-id": "2",
            }

            context = extract_context_from_request(mock_request)

            assert context.user_id == UUID("00000000-0000-0000-0000-000000000000")
            assert context.tenant_id == UUID("00000000-0000-0000-0000-000000000000")

    def test_extract_context_with_missing_headers_raises_http_exception(self):
        """Test that missing headers raise HTTPException when auth is enabled."""
        with patch("minds.requests.context.settings.auth.disable", False):
            mock_request = Mock(spec=Request)
            mock_request.headers = {"x-user-id": None, "x-company-id": None}

            with pytest.raises(HTTPException) as exc_info:
                _ = extract_context_from_request(mock_request)

            assert exc_info.value.status_code == 400
            assert exc_info.value.detail == "Missing required authentication"

    def test_extract_context_with_invalid_user_id(self):
        """Test extracting context when user ID is not a valid integer."""
        with patch("minds.requests.context.settings.auth.disable", False):
            mock_request = Mock(spec=Request)
            mock_request.headers = {
                "x-user-id": "non-integer",
                "x-company-id": "2",
            }

            with pytest.raises(ValueError) as e:
                _ = extract_context_from_request(mock_request)

                assert "invalid literal for int() with base 10: 'non-integer'" in str(e.value)

    def test_extract_context_with_invalid_tenant_id(self):
        """Test extracting context when tenant ID is not a valid integer."""
        with patch("minds.requests.context.settings.auth.disable", False):
            mock_request = Mock(spec=Request)
            mock_request.headers = {
                "x-user-id": "1",
                "x-company-id": "non-integer",
            }

            with pytest.raises(ValueError) as e:
                _ = extract_context_from_request(mock_request)

                assert "invalid literal for int() with base 10: 'non-integer'" in str(e.value)


class TestLangfuseContextMetadata:
    """Test cases for LangfuseContextMetadata model."""

    def test_langfuse_context_metadata_defaults(self):
        """Test LangfuseContextMetadata initialization with defaults."""
        metadata = LangfuseContextMetadata()

        assert metadata.user_id == UUID("00000000-0000-0000-0000-000000000000")
        assert metadata.tenant_id == UUID("00000000-0000-0000-0000-000000000000")

    def test_langfuse_context_metadata_with_values(self):
        """Test LangfuseContextMetadata initialization with values."""
        metadata = LangfuseContextMetadata(
            user_id=UUID("00000000-0000-0000-0000-000000000001"), tenant_id=UUID("00000000-0000-0000-0000-000000000002")
        )

        assert metadata.user_id == UUID("00000000-0000-0000-0000-000000000001")
        assert metadata.tenant_id == UUID("00000000-0000-0000-0000-000000000002")


class TestLangfuseContext:
    """Test cases for LangfuseContext model."""

    def test_langfuse_context_defaults(self):
        """Test LangfuseContext initialization with defaults."""
        langfuse_context = LangfuseContext()

        assert langfuse_context.user_id == UUID("00000000-0000-0000-0000-000000000000")
        assert langfuse_context.metadata.user_id == UUID("00000000-0000-0000-0000-000000000000")
        assert langfuse_context.metadata.tenant_id == UUID("00000000-0000-0000-0000-000000000000")
        assert langfuse_context.tags == []  # Empty list because no user or tenant ID is provided
        assert langfuse_context.trace_id is None

    def test_langfuse_context_with_values(self):
        """Test LangfuseContext initialization with values."""
        metadata = LangfuseContextMetadata(
            user_id=UUID("00000000-0000-0000-0000-000000000001"), tenant_id=UUID("00000000-0000-0000-0000-000000000002")
        )

        langfuse_context = LangfuseContext(
            user_id=UUID("00000000-0000-0000-0000-000000000001"),
            metadata=metadata,
            tags=["tag1", "tag2"],
            trace_id="trace-123",
        )

        assert langfuse_context.user_id == UUID("00000000-0000-0000-0000-000000000001")
        assert langfuse_context.metadata == metadata
        assert langfuse_context.tags == ["tag1", "tag2"]
        assert langfuse_context.trace_id == "trace-123"


class TestCreateLangfuseContext:
    """Test cases for create_langfuse_context function."""

    def test_create_langfuse_context_from_context(self):
        """Test creating LangfuseContext from Context."""
        context = Context(
            user_id=UUID("00000000-0000-0000-0000-000000000001"), tenant_id=UUID("00000000-0000-0000-0000-000000000002")
        )

        langfuse_context = create_langfuse_context(context)

        assert langfuse_context.user_id == UUID("00000000-0000-0000-0000-000000000001")
        assert langfuse_context.metadata.user_id == UUID("00000000-0000-0000-0000-000000000001")
        assert langfuse_context.metadata.tenant_id == UUID("00000000-0000-0000-0000-000000000002")
        assert langfuse_context.tags == [
            f"user_id:{UUID('00000000-0000-0000-0000-000000000001')}",
            f"tenant_id:{UUID('00000000-0000-0000-0000-000000000002')}",
            "local",
            f"request_id:{context.request_id}",
            context.user_email,
        ]
        assert langfuse_context.trace_id is None

    def test_create_langfuse_context_with_empty_context(self):
        """Test creating LangfuseContext from empty Context."""
        context = Context()

        langfuse_context = create_langfuse_context(context)

        assert langfuse_context.user_id == UUID("00000000-0000-0000-0000-000000000000")
        assert langfuse_context.metadata.user_id == UUID("00000000-0000-0000-0000-000000000000")
        assert langfuse_context.metadata.tenant_id == UUID("00000000-0000-0000-0000-000000000000")
        assert langfuse_context.tags == [
            f"user_id:{UUID('00000000-0000-0000-0000-000000000000')}",
            f"tenant_id:{UUID('00000000-0000-0000-0000-000000000000')}",
            "local",
            f"request_id:{context.request_id}",
            context.user_email,
        ]

    def test_create_langfuse_context_tags_format(self):
        """Test that tags are created in the correct format."""
        context = Context(
            user_id=UUID("00000000-0000-0000-0000-000000000001"), tenant_id=UUID("00000000-0000-0000-0000-000000000002")
        )

        langfuse_context = create_langfuse_context(context)

        # Tags should be [user_email]
        expected_tags = [
            f"user_id:{UUID('00000000-0000-0000-0000-000000000001')}",
            f"tenant_id:{UUID('00000000-0000-0000-0000-000000000002')}",
            "local",
            f"request_id:{context.request_id}",
            context.user_email,
        ]
        assert langfuse_context.tags == expected_tags

    def test_create_langfuse_context_metadata_consistency(self):
        """Test that metadata fields match the context fields."""
        context = Context(
            user_id=UUID("00000000-0000-0000-0000-000000000001"), tenant_id=UUID("00000000-0000-0000-0000-000000000002")
        )

        langfuse_context = create_langfuse_context(context)

        # All corresponding fields should match
        assert langfuse_context.user_id == context.user_id
        assert langfuse_context.metadata.user_id == context.user_id
        assert langfuse_context.metadata.tenant_id == context.tenant_id
