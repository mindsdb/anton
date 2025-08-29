from unittest.mock import Mock

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

        assert context.user_id == ""
        assert context.user_email == ""

    def test_context_initialization_with_values(self):
        """Test Context initialization with provided values."""
        context = Context(user_id="123", user_email="test@example.com")

        assert context.user_id == "123"
        assert context.user_email == "test@example.com"

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
            "x-user-id": "123",
            "x-user-email": "test@example.com",
        }

        context = extract_context_from_request(mock_request)

        assert context.user_id == "123"
        assert context.user_email == "test@example.com"

    def test_extract_context_with_missing_headers(self):
        """Test extracting context when headers are missing."""
        mock_request = Mock(spec=Request)

        # Create a proper mock headers object with get method
        mock_headers = Mock()
        mock_headers.get = Mock(side_effect=lambda key, default: default)
        mock_request.headers = mock_headers

        context = extract_context_from_request(mock_request)

        assert context.user_id == ""
        assert context.user_email == ""

    def test_extract_context_with_partial_headers(self):
        """Test extracting context when only some headers are present."""
        mock_request = Mock(spec=Request)

        def mock_get(key, default):
            headers = {"x-user-id": "789", "x-user-email": "partial@example.com"}
            return headers.get(key, default)

        mock_request.headers.get = mock_get

        context = extract_context_from_request(mock_request)

        assert context.user_id == "789"
        assert context.user_email == "partial@example.com"


class TestLangfuseContextMetadata:
    """Test cases for LangfuseContextMetadata model."""

    def test_langfuse_context_metadata_defaults(self):
        """Test LangfuseContextMetadata initialization with defaults."""
        metadata = LangfuseContextMetadata()

        assert metadata.user_id == ""
        assert metadata.user_email == ""

    def test_langfuse_context_metadata_with_values(self):
        """Test LangfuseContextMetadata initialization with values."""
        metadata = LangfuseContextMetadata(user_id="123", user_email="test@example.com")

        assert metadata.user_id == "123"
        assert metadata.user_email == "test@example.com"


class TestLangfuseContext:
    """Test cases for LangfuseContext model."""

    def test_langfuse_context_defaults(self):
        """Test LangfuseContext initialization with defaults."""
        langfuse_context = LangfuseContext()

        assert langfuse_context.user_id == ""
        assert isinstance(langfuse_context.metadata, LangfuseContextMetadata)
        assert langfuse_context.tags == []
        assert langfuse_context.trace_id is None

    def test_langfuse_context_with_values(self):
        """Test LangfuseContext initialization with values."""
        metadata = LangfuseContextMetadata(user_id="123", user_email="test@example.com")

        langfuse_context = LangfuseContext(
            user_id="123",
            metadata=metadata,
            tags=["tag1", "tag2"],
            trace_id="trace-123",
        )

        assert langfuse_context.user_id == "123"
        assert langfuse_context.metadata == metadata
        assert langfuse_context.tags == ["tag1", "tag2"]
        assert langfuse_context.trace_id == "trace-123"


class TestCreateLangfuseContext:
    """Test cases for create_langfuse_context function."""

    def test_create_langfuse_context_from_context(self):
        """Test creating LangfuseContext from Context."""
        context = Context(user_id="123", user_email="test@example.com")

        langfuse_context = create_langfuse_context(context)

        assert langfuse_context.user_id == "123"
        assert langfuse_context.metadata.user_id == "123"
        assert langfuse_context.metadata.user_email == "test@example.com"
        assert langfuse_context.tags == ["test@example.com"]
        assert langfuse_context.trace_id is None

    def test_create_langfuse_context_with_empty_context(self):
        """Test creating LangfuseContext from empty Context."""
        context = Context()

        langfuse_context = create_langfuse_context(context)

        assert langfuse_context.user_id == ""
        assert langfuse_context.metadata.user_id == ""
        assert langfuse_context.metadata.user_email == ""
        assert langfuse_context.tags == [""]

    def test_create_langfuse_context_tags_format(self):
        """Test that tags are created in the correct format."""
        context = Context(user_id="999", user_email="tags@example.com")

        langfuse_context = create_langfuse_context(context)

        # Tags should be [user_email]
        expected_tags = ["tags@example.com"]
        assert langfuse_context.tags == expected_tags

    def test_create_langfuse_context_metadata_consistency(self):
        """Test that metadata fields match the context fields."""
        context = Context(user_id="555", user_email="consistency@example.com")

        langfuse_context = create_langfuse_context(context)

        # All corresponding fields should match
        assert langfuse_context.user_id == context.user_id
        assert langfuse_context.metadata.user_id == context.user_id
        assert langfuse_context.metadata.user_email == context.user_email
