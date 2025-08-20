import pytest
from unittest.mock import Mock, MagicMock
from fastapi import Request

from minds.requests.context import (
    Context,
    extract_context_from_request,
    LangfuseContextMetadata,
    LangfuseContext,
    create_langfuse_context
)


class TestContext:
    """Test cases for Context model."""

    def test_context_initialization_with_defaults(self):
        """Test Context initialization with default values."""
        context = Context()
        
        assert context.user_id == 0
        assert context.user_email == ""
        assert context.company_id == 0

    def test_context_initialization_with_values(self):
        """Test Context initialization with provided values."""
        context = Context(
            user_id=123,
            user_email="test@example.com",
            company_id=456
        )
        
        assert context.user_id == 123
        assert context.user_email == "test@example.com"
        assert context.company_id == 456

    def test_context_is_basemodel(self):
        """Test that Context inherits from BaseModel."""
        context = Context()
        
        # Should have BaseModel methods
        assert hasattr(context, 'model_dump')
        assert hasattr(context, 'model_validate')


class TestExtractContextFromRequest:
    """Test cases for extract_context_from_request function."""

    def test_extract_context_with_all_headers(self):
        """Test extracting context when all headers are present."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "x-user-id": "123",
            "x-user-email": "test@example.com",
            "x-company-id": "456"
        }
        
        context = extract_context_from_request(mock_request)
        
        assert context.user_id == 123
        assert context.user_email == "test@example.com"
        assert context.company_id == 456

    def test_extract_context_with_missing_headers(self):
        """Test extracting context when headers are missing."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {}
        
        # Mock get method to return None for missing headers
        mock_request.headers.get = Mock(side_effect=lambda key, default: default)
        
        context = extract_context_from_request(mock_request)
        
        assert context.user_id == 0
        assert context.user_email == ""
        assert context.company_id == 0

    def test_extract_context_with_partial_headers(self):
        """Test extracting context when only some headers are present."""
        mock_request = Mock(spec=Request)
        
        def mock_get(key, default):
            headers = {"x-user-id": "789", "x-user-email": "partial@example.com"}
            return headers.get(key, default)
        
        mock_request.headers.get = mock_get
        
        context = extract_context_from_request(mock_request)
        
        assert context.user_id == 789
        assert context.user_email == "partial@example.com"
        assert context.company_id == 0  # Default value

    def test_extract_context_with_invalid_numeric_headers(self):
        """Test extracting context when numeric headers contain invalid values."""
        mock_request = Mock(spec=Request)
        
        def mock_get(key, default):
            headers = {
                "x-user-id": "invalid",
                "x-user-email": "test@example.com",
                "x-company-id": "also_invalid"
            }
            return headers.get(key, default)
        
        mock_request.headers.get = mock_get
        
        # This should raise ValueError when trying to convert invalid strings to int
        with pytest.raises(ValueError):
            extract_context_from_request(mock_request)


class TestLangfuseContextMetadata:
    """Test cases for LangfuseContextMetadata model."""

    def test_langfuse_context_metadata_defaults(self):
        """Test LangfuseContextMetadata initialization with defaults."""
        metadata = LangfuseContextMetadata()
        
        assert metadata.user_id == 0
        assert metadata.user_email == ""
        assert metadata.company_id == ""

    def test_langfuse_context_metadata_with_values(self):
        """Test LangfuseContextMetadata initialization with values."""
        metadata = LangfuseContextMetadata(
            user_id=123,
            user_email="test@example.com",
            company_id="456"
        )
        
        assert metadata.user_id == 123
        assert metadata.user_email == "test@example.com"
        assert metadata.company_id == "456"


class TestLangfuseContext:
    """Test cases for LangfuseContext model."""

    def test_langfuse_context_defaults(self):
        """Test LangfuseContext initialization with defaults."""
        langfuse_context = LangfuseContext()
        
        assert langfuse_context.user_id == 0
        assert isinstance(langfuse_context.metadata, LangfuseContextMetadata)
        assert langfuse_context.tags == []
        assert langfuse_context.trace_id is None

    def test_langfuse_context_with_values(self):
        """Test LangfuseContext initialization with values."""
        metadata = LangfuseContextMetadata(
            user_id=123,
            user_email="test@example.com",
            company_id="456"
        )
        
        langfuse_context = LangfuseContext(
            user_id=123,
            metadata=metadata,
            tags=["tag1", "tag2"],
            trace_id="trace-123"
        )
        
        assert langfuse_context.user_id == 123
        assert langfuse_context.metadata == metadata
        assert langfuse_context.tags == ["tag1", "tag2"]
        assert langfuse_context.trace_id == "trace-123"


class TestCreateLangfuseContext:
    """Test cases for create_langfuse_context function."""

    def test_create_langfuse_context_from_context(self):
        """Test creating LangfuseContext from Context."""
        context = Context(
            user_id=123,
            user_email="test@example.com",
            company_id=456
        )
        
        langfuse_context = create_langfuse_context(context)
        
        assert langfuse_context.user_id == 123
        assert langfuse_context.metadata.user_id == 123
        assert langfuse_context.metadata.user_email == "test@example.com"
        assert langfuse_context.metadata.company_id == 456
        assert langfuse_context.tags == ["test@example.com", 456]
        assert langfuse_context.trace_id is None

    def test_create_langfuse_context_with_empty_context(self):
        """Test creating LangfuseContext from empty Context."""
        context = Context()
        
        langfuse_context = create_langfuse_context(context)
        
        assert langfuse_context.user_id == 0
        assert langfuse_context.metadata.user_id == 0
        assert langfuse_context.metadata.user_email == ""
        assert langfuse_context.metadata.company_id == 0
        assert langfuse_context.tags == ["", 0]

    def test_create_langfuse_context_tags_format(self):
        """Test that tags are created in the correct format."""
        context = Context(
            user_id=999,
            user_email="tags@example.com",
            company_id=888
        )
        
        langfuse_context = create_langfuse_context(context)
        
        # Tags should be [user_email, company_id]
        expected_tags = ["tags@example.com", 888]
        assert langfuse_context.tags == expected_tags

    def test_create_langfuse_context_metadata_consistency(self):
        """Test that metadata fields match the context fields."""
        context = Context(
            user_id=555,
            user_email="consistency@example.com",
            company_id=666
        )
        
        langfuse_context = create_langfuse_context(context)
        
        # All corresponding fields should match
        assert langfuse_context.user_id == context.user_id
        assert langfuse_context.metadata.user_id == context.user_id
        assert langfuse_context.metadata.user_email == context.user_email
        assert langfuse_context.metadata.company_id == context.company_id
