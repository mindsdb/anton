from unittest.mock import patch
from uuid import UUID

import pytest

from minds.requests.context import Context, LangfuseContext, LangfuseContextMetadata
from minds.requests.langfuse_tracing import setup_langfuse_observation


@pytest.fixture()
def context():
    return Context(
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        organization_id=UUID("00000000-0000-0000-0000-000000000002"),
    )


@pytest.fixture()
def langfuse_context():
    return LangfuseContext(
        user_id=UUID("00000000-0000-0000-0000-000000000001"),
        metadata=LangfuseContextMetadata(user_id=UUID("00000000-0000-0000-0000-000000000001")),
        tags=["test"],
    )


class TestSetupLangfuseObservation:
    """Test cases for setup_langfuse_observation function."""

    @patch("minds.requests.langfuse_tracing.create_langfuse_context")
    @patch("minds.requests.langfuse_tracing.get_client")
    def test_setup_langfuse_observation_success(
        self,
        mock_get_client,
        mock_create_langfuse_context,
        context,
        langfuse_context,
    ):
        """Test successful setup of Langfuse observation."""
        # Arrange
        test_context = context
        test_langfuse_context = langfuse_context

        mock_create_langfuse_context.return_value = test_langfuse_context

        mock_client = type("MockClient", (), {})()
        mock_client.update_current_trace = lambda **kwargs: None
        mock_get_client.return_value = mock_client

        # Act
        _ = setup_langfuse_observation(test_context)

        # Assert
        mock_create_langfuse_context.assert_called_once_with(test_context)
        mock_get_client.assert_called_once()

    @patch("minds.requests.langfuse_tracing.create_langfuse_context")
    @patch("minds.requests.langfuse_tracing.get_client")
    def test_setup_langfuse_observation_no_trace_id(
        self,
        mock_get_client,
        mock_create_langfuse_context,
        context,
        langfuse_context,
    ):
        """Test setup when get_current_trace_id returns None."""

        # Arrange
        test_context = context
        test_langfuse_context = langfuse_context

        mock_create_langfuse_context.return_value = test_langfuse_context

        mock_client = type("MockClient", (), {})()
        mock_client.update_current_trace = lambda **kwargs: None
        mock_client.get_current_trace_id = lambda: None
        mock_get_client.return_value = mock_client

        # Act
        _ = setup_langfuse_observation(test_context)

        # Assert
        mock_create_langfuse_context.assert_called_once_with(test_context)
        mock_get_client.assert_called_once()

    @patch("minds.requests.langfuse_tracing.create_langfuse_context")
    @patch("minds.requests.langfuse_tracing.get_client")
    @patch("minds.requests.langfuse_tracing.logger")
    def test_setup_langfuse_observation_update_exception(
        self,
        mock_logger,
        mock_get_client,
        mock_create_langfuse_context,
        context,
        langfuse_context,
    ):
        """Test setup when update_current_observation raises exception."""
        # Arrange
        test_context = context
        test_langfuse_context = langfuse_context
        test_exception = Exception("Langfuse update failed")
        mock_create_langfuse_context.return_value = test_langfuse_context

        def update_current_trace_with_exception(**kwargs):
            raise test_exception

        mock_client = type("MockClient", (), {})()
        mock_client.update_current_trace = update_current_trace_with_exception
        mock_client.get_current_trace_id = lambda: "some_id"
        mock_get_client.return_value = mock_client

        # Act
        _ = setup_langfuse_observation(test_context)

        # Assert
        mock_create_langfuse_context.assert_called_once_with(test_context)
        mock_get_client.assert_called_once()

        # Verify error logging
        mock_logger.error.assert_any_call("Error updating Langfuse observation: Langfuse update failed")
        # Check that traceback was also logged (second call to mock_logger.error)
        assert mock_logger.error.call_count == 2

    @patch("minds.requests.langfuse_tracing.create_langfuse_context")
    @patch("minds.requests.langfuse_tracing.get_client")
    @patch("minds.requests.langfuse_tracing.logger")
    def test_setup_langfuse_observation_get_trace_id_exception(
        self,
        mock_logger,
        mock_get_client,
        mock_create_langfuse_context,
        context,
        langfuse_context,
    ):
        """Test setup when get_current_trace_id raises exception."""
        # Arrange
        test_context = context
        test_langfuse_context = langfuse_context

        test_exception = Exception("Get trace ID failed")

        mock_create_langfuse_context.return_value = test_langfuse_context

        def get_current_trace_id_with_exception():
            raise test_exception

        mock_client = type("MockClient", (), {})()
        mock_client.update_current_trace = lambda **kwargs: None
        mock_client.get_current_trace_id = get_current_trace_id_with_exception
        mock_get_client.return_value = mock_client

        # Act
        _ = setup_langfuse_observation(test_context)

        # Assert
        mock_create_langfuse_context.assert_called_once_with(test_context)
        mock_get_client.assert_called_once()

        # Verify error logging
        mock_logger.error.assert_any_call("Error updating Langfuse observation: Get trace ID failed")

    @patch("minds.requests.langfuse_tracing.create_langfuse_context")
    @patch("minds.requests.langfuse_tracing.get_client")
    @patch("minds.requests.langfuse_tracing.logger")
    @patch("minds.requests.langfuse_tracing.traceback.format_exc", return_value="mocked traceback")
    def test_setup_langfuse_observation_create_context_exception(
        self,
        mock_traceback,
        mock_logger,
        mock_get_client,
        mock_create_langfuse_context,
        context,
    ):
        """Test setup when create_langfuse_context raises exception."""
        # Arrange
        test_context = context
        test_exception = Exception("Create context failed")

        mock_create_langfuse_context.side_effect = test_exception

        # Act - the function should catch the exception and return default UUID
        setup_langfuse_observation(test_context)

        # Assert
        mock_create_langfuse_context.assert_called_once_with(test_context)

        # get_client should NOT be called since context creation fails early
        mock_get_client.assert_not_called()

        # Verify error logging
        mock_logger.error.assert_any_call("Error updating Langfuse observation: Create context failed")

    @patch("minds.requests.langfuse_tracing.create_langfuse_context")
    @patch("minds.requests.langfuse_tracing.get_client")
    def test_setup_langfuse_observation_logs_success(
        self,
        mock_get_client,
        mock_create_langfuse_context,
        context,
        langfuse_context,
    ):
        """Test that successful setup logs debug message."""
        # Arrange
        test_context = context
        test_langfuse_context = langfuse_context

        mock_create_langfuse_context.return_value = test_langfuse_context

        mock_client = type("MockClient", (), {})()
        mock_client.update_current_trace = lambda **kwargs: None
        mock_get_client.return_value = mock_client

        # Act
        result = setup_langfuse_observation(test_context)

        # Assert
        assert result is None

    @patch("minds.requests.langfuse_tracing.create_langfuse_context")
    @patch("minds.requests.langfuse_tracing.get_client")
    @patch("minds.requests.langfuse_tracing.logger")
    def test_setup_langfuse_observation_logs_no_trace_id_error(
        self,
        mock_logger,
        mock_get_client,
        mock_create_langfuse_context,
        context,
    ):
        """Test that missing trace ID logs error message."""
        # Arrange
        test_context = context
        test_langfuse_context = LangfuseContext()

        mock_create_langfuse_context.return_value = test_langfuse_context

        mock_client = type("MockClient", (), {})()
        mock_client.update_current_trace = lambda **kwargs: None
        mock_client.get_current_trace_id = lambda: None
        mock_get_client.return_value = mock_client

        # Act
        _ = setup_langfuse_observation(test_context)

        # Assert
        mock_logger.error.assert_called_once_with("Failed to retrieve trace ID from Langfuse context.")
