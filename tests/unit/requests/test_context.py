from unittest.mock import Mock
from uuid import UUID

import pytest
from fastapi import Request

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
        """Test extracting context when all headers are present."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "x-user-id": "1",
            "x-company-id": "2",
        }

        context = extract_context_from_request(mock_request)

        assert context.user_id == UUID("00000000-0000-0000-0000-000000000001")
        assert context.tenant_id == UUID("00000000-0000-0000-0000-000000000002")

    def test_extract_context_with_missing_headers_invalid_literals_empty_string(self):
        """Test extracting context when headers are missing."""
        mock_request = Mock(spec=Request)

        # Create a proper mock headers object with get method
        mock_headers = Mock()
        mock_headers.get = Mock(side_effect=lambda key, default: default)
        mock_request.headers = mock_headers

        with pytest.raises(ValueError) as e:
            _ = extract_context_from_request(mock_request)

            assert "invalid literal for int() with base 10: ''" in str(e.value)

    def test_extract_context_with_missing_headers_invalid_literals_non_integer_user_id(self):
        """Test extracting context when all headers are present but the user ID is not an integer."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "x-user-id": "non-integer",
            "x-company-id": "2",
        }

        with pytest.raises(ValueError) as e:
            _ = extract_context_from_request(mock_request)

            assert "invalid literal for int() with base 10: 'non-integer'" in str(e.value)

    def test_extract_context_with_missing_headers_invalid_literals_non_integer_tenant_id(self):
        """Test extracting context when all headers are present but the tenant ID is not an integer."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "x-user-id": "1",
            "x-company-id": "non-integer",
        }

        with pytest.raises(ValueError) as e:
            _ = extract_context_from_request(mock_request)

            assert "invalid literal for int() with base 10: '2'" in str(e.value)


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
            UUID("00000000-0000-0000-0000-000000000001"),
            UUID("00000000-0000-0000-0000-000000000002"),
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
            UUID("00000000-0000-0000-0000-000000000000"),
            UUID("00000000-0000-0000-0000-000000000000"),
        ]

    def test_create_langfuse_context_tags_format(self):
        """Test that tags are created in the correct format."""
        context = Context(
            user_id=UUID("00000000-0000-0000-0000-000000000001"), tenant_id=UUID("00000000-0000-0000-0000-000000000002")
        )

        langfuse_context = create_langfuse_context(context)

        # Tags should be [user_email]
        expected_tags = [UUID("00000000-0000-0000-0000-000000000001"), UUID("00000000-0000-0000-0000-000000000002")]
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
