import uuid
from unittest.mock import patch

import pytest

from minds.requests.context import Context, LangfuseContext, LangfuseContextMetadata
from minds.requests.utils import setup_langfuse_observation


@pytest.fixture()
def context():
    return Context(user_id="123", user_email="test@example.com", company_id="456")


@pytest.fixture()
def langfuse_context():
    return LangfuseContext(
        user_id="123",
        metadata=LangfuseContextMetadata(user_id="123", user_email="test@example.com", company_id="456"),
        tags=["test@example.com", "456"],
    )


class TestSetupLangfuseObservation:
    """Test cases for setup_langfuse_observation function."""

    @patch("minds.requests.utils.create_langfuse_context")
    @patch("minds.requests.utils.get_client")
    @patch("minds.requests.utils.uuid.uuid4")
    def test_setup_langfuse_observation_success(
        self,
        mock_uuid4,
        mock_get_client,
        mock_create_langfuse_context,
        context,
        langfuse_context,
    ):
        """Test successful setup of Langfuse observation."""
        # Arrange
        test_context = context
        test_langfuse_context = langfuse_context

        expected_trace_id = "12345678-1234-5678-1234-567812345678"
        mock_uuid4.return_value = uuid.UUID(expected_trace_id)
        mock_create_langfuse_context.return_value = test_langfuse_context

        mock_client = type("MockClient", (), {})()
        mock_client.update_current_trace = lambda **kwargs: None
        mock_client.get_current_trace_id = lambda: expected_trace_id
        mock_get_client.return_value = mock_client

        # Act
        result = setup_langfuse_observation(test_context)

        # Assert
        assert result == expected_trace_id
        mock_create_langfuse_context.assert_called_once_with(test_context)
        mock_get_client.assert_called_once()

    @patch("minds.requests.utils.create_langfuse_context")
    @patch("minds.requests.utils.get_client")
    @patch("minds.requests.utils.uuid.uuid4")
    def test_setup_langfuse_observation_no_trace_id(
        self,
        mock_uuid4,
        mock_get_client,
        mock_create_langfuse_context,
        context,
        langfuse_context,
    ):
        """Test setup when get_current_trace_id returns None."""
        # Arrange
        test_context = context
        test_langfuse_context = langfuse_context

        default_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        expected_default_id = str(default_uuid)

        mock_uuid4.return_value = default_uuid
        mock_create_langfuse_context.return_value = test_langfuse_context

        mock_client = type("MockClient", (), {})()
        mock_client.update_current_trace = lambda **kwargs: None
        mock_client.get_current_trace_id = lambda: None
        mock_get_client.return_value = mock_client

        # Act
        result = setup_langfuse_observation(test_context)

        # Assert
        assert result == expected_default_id
        mock_create_langfuse_context.assert_called_once_with(test_context)
        mock_get_client.assert_called_once()

    @patch("minds.requests.utils.create_langfuse_context")
    @patch("minds.requests.utils.get_client")
    @patch("minds.requests.utils.uuid.uuid4")
    @patch("minds.requests.utils.logger")
    def test_setup_langfuse_observation_update_exception(
        self,
        mock_logger,
        mock_uuid4,
        mock_get_client,
        mock_create_langfuse_context,
        context,
        langfuse_context,
    ):
        """Test setup when update_current_observation raises exception."""
        # Arrange
        test_context = context
        test_langfuse_context = langfuse_context

        default_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        expected_default_id = str(default_uuid)
        test_exception = Exception("Langfuse update failed")

        mock_uuid4.return_value = default_uuid
        mock_create_langfuse_context.return_value = test_langfuse_context

        def update_current_trace_with_exception(**kwargs):
            raise test_exception

        mock_client = type("MockClient", (), {})()
        mock_client.update_current_trace = update_current_trace_with_exception
        mock_client.get_current_trace_id = lambda: "some_id"
        mock_get_client.return_value = mock_client

        # Act
        result = setup_langfuse_observation(test_context)

        # Assert
        assert result == expected_default_id
        mock_create_langfuse_context.assert_called_once_with(test_context)
        mock_get_client.assert_called_once()

        # Verify error logging
        mock_logger.error.assert_any_call("Error updating Langfuse observation: Langfuse update failed")
        # Check that traceback was also logged (second call to mock_logger.error)
        assert mock_logger.error.call_count == 2

    @patch("minds.requests.utils.create_langfuse_context")
    @patch("minds.requests.utils.get_client")
    @patch("minds.requests.utils.uuid.uuid4")
    @patch("minds.requests.utils.logger")
    def test_setup_langfuse_observation_get_trace_id_exception(
        self,
        mock_logger,
        mock_uuid4,
        mock_get_client,
        mock_create_langfuse_context,
        context,
        langfuse_context,
    ):
        """Test setup when get_current_trace_id raises exception."""
        # Arrange
        test_context = context
        test_langfuse_context = langfuse_context

        default_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        expected_default_id = str(default_uuid)
        test_exception = Exception("Get trace ID failed")

        mock_uuid4.return_value = default_uuid
        mock_create_langfuse_context.return_value = test_langfuse_context

        def get_current_trace_id_with_exception():
            raise test_exception

        mock_client = type("MockClient", (), {})()
        mock_client.update_current_trace = lambda **kwargs: None
        mock_client.get_current_trace_id = get_current_trace_id_with_exception
        mock_get_client.return_value = mock_client

        # Act
        result = setup_langfuse_observation(test_context)

        # Assert
        assert result == expected_default_id
        mock_create_langfuse_context.assert_called_once_with(test_context)
        mock_get_client.assert_called_once()

        # Verify error logging
        mock_logger.error.assert_any_call("Error updating Langfuse observation: Get trace ID failed")

    @patch("minds.requests.utils.create_langfuse_context")
    @patch("minds.requests.utils.get_client")
    @patch("minds.requests.utils.uuid.uuid4")
    @patch("minds.requests.utils.logger")
    def test_setup_langfuse_observation_create_context_exception(
        self,
        mock_logger,
        mock_uuid4,
        mock_get_client,
        mock_create_langfuse_context,
        context,
    ):
        """Test setup when create_langfuse_context raises exception."""
        # Arrange
        test_context = context
        default_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        expected_default_id = str(default_uuid)
        test_exception = Exception("Create context failed")

        mock_uuid4.return_value = default_uuid
        mock_create_langfuse_context.side_effect = test_exception

        # Act - the function should catch the exception and return default UUID
        with patch("minds.requests.utils.traceback.format_exc", return_value="mocked traceback"):
            result = setup_langfuse_observation(test_context)

        # Assert
        assert result == expected_default_id
        mock_create_langfuse_context.assert_called_once_with(test_context)

        # get_client should NOT be called since context creation fails early
        mock_get_client.assert_not_called()

        # Verify error logging
        mock_logger.error.assert_any_call("Error updating Langfuse observation: Create context failed")

    @patch("minds.requests.utils.create_langfuse_context")
    @patch("minds.requests.utils.get_client")
    @patch("minds.requests.utils.uuid.uuid4")
    @patch("minds.requests.utils.logger")
    def test_setup_langfuse_observation_logs_success(
        self,
        mock_logger,
        mock_uuid4,
        mock_get_client,
        mock_create_langfuse_context,
        context,
        langfuse_context,
    ):
        """Test that successful setup logs debug message."""
        # Arrange
        test_context = context
        test_langfuse_context = langfuse_context

        expected_trace_id = "12345678-1234-5678-1234-567812345678"

        mock_uuid4.return_value = uuid.UUID(expected_trace_id)
        mock_create_langfuse_context.return_value = test_langfuse_context

        mock_client = type("MockClient", (), {})()
        mock_client.update_current_trace = lambda **kwargs: None
        mock_client.get_current_trace_id = lambda: expected_trace_id
        mock_get_client.return_value = mock_client

        # Act
        result = setup_langfuse_observation(test_context)

        # Assert
        assert result == expected_trace_id
        mock_logger.debug.assert_called_once_with(f"Created langfuse context with trace ID: {expected_trace_id}")
        # Should not log any errors
        mock_logger.error.assert_not_called()

    @patch("minds.requests.utils.create_langfuse_context")
    @patch("minds.requests.utils.get_client")
    @patch("minds.requests.utils.uuid.uuid4")
    @patch("minds.requests.utils.logger")
    def test_setup_langfuse_observation_logs_no_trace_id_error(
        self,
        mock_logger,
        mock_uuid4,
        mock_get_client,
        mock_create_langfuse_context,
        context,
        langfuse_context,
    ):
        """Test that missing trace ID logs error message."""
        # Arrange
        test_context = context
        test_langfuse_context = LangfuseContext()
        default_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")

        mock_uuid4.return_value = default_uuid
        mock_create_langfuse_context.return_value = test_langfuse_context

        mock_client = type("MockClient", (), {})()
        mock_client.update_current_trace = lambda **kwargs: None
        mock_client.get_current_trace_id = lambda: None
        mock_get_client.return_value = mock_client

        # Act
        _ = setup_langfuse_observation(test_context)

        # Assert
        mock_logger.error.assert_called_once_with("Failed to retrieve trace ID from Langfuse context.")
        # Should not log debug message since trace ID is None
        mock_logger.debug.assert_not_called()

    def test_setup_langfuse_observation_uuid_format(self):
        """Test that default request ID is a valid UUID string."""
        # We can't easily mock uuid.uuid4() in the actual function call,
        # but we can test that the function returns a string that looks like a UUID
        test_context = Context()

        with patch("minds.requests.utils.create_langfuse_context") as mock_create:
            with patch(
                "minds.requests.utils.traceback.format_exc",
                return_value="mocked traceback",
            ):
                mock_create.side_effect = Exception("Force default UUID")

                result = setup_langfuse_observation(test_context)

                # Result should be a string representation of a UUID
                assert isinstance(result, str)
                assert len(result) == 36  # Standard UUID string length
                assert result.count("-") == 4  # Standard UUID has 4 hyphens

                # Should be able to parse as UUID
                parsed_uuid = uuid.UUID(result)
                assert str(parsed_uuid) == result
