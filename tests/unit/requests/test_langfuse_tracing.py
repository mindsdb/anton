from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from minds.requests.context import Context, LangfuseContext, LangfuseContextMetadata
from minds.requests.langfuse_tracing import (
    capture_langfuse_generation_context,
    setup_langfuse_observation,
    update_generation_usage,
)


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


class TestUpdateGenerationUsage:
    """Test cases for update_generation_usage helper."""

    @patch("minds.requests.langfuse_tracing.get_client")
    def test_calls_update_current_generation_when_no_trace_context(self, mock_get_client):
        """In-scope mode: updates the active @observe generation directly."""
        client = MagicMock()
        mock_get_client.return_value = client

        update_generation_usage((11, 7), model="gpt-4o-mini")

        client.update_current_generation.assert_called_once_with(
            model="gpt-4o-mini",
            usage_details={"input": 11, "output": 7, "total": 18},
            input=None,
            output=None,
            metadata=None,
        )
        client.start_observation.assert_not_called()

    @patch("minds.requests.langfuse_tracing.get_client")
    def test_uses_start_observation_when_trace_context_provided(self, mock_get_client):
        """Detached mode: attaches a child generation observation to the captured trace."""
        client = MagicMock()
        observation = MagicMock()
        client.start_observation.return_value = observation
        mock_get_client.return_value = client

        trace_context = {"trace_id": "abc", "parent_span_id": "def"}
        update_generation_usage(
            (5, 3),
            model="claude-sonnet-4-6",
            trace_context=trace_context,
            output="hello",
        )

        client.start_observation.assert_called_once_with(
            trace_context=trace_context,
            name="llm-usage",
            as_type="generation",
            model="claude-sonnet-4-6",
            usage_details={"input": 5, "output": 3, "total": 8},
            input=None,
            output="hello",
            metadata=None,
        )
        observation.end.assert_called_once()
        client.update_current_generation.assert_not_called()

    @patch("minds.requests.langfuse_tracing.get_client")
    def test_uses_custom_name_in_detached_mode(self, mock_get_client):
        """Custom ``name`` argument is forwarded to start_observation in detached mode."""
        client = MagicMock()
        client.start_observation.return_value = MagicMock()
        mock_get_client.return_value = client

        update_generation_usage(
            (1, 1),
            model="m",
            trace_context={"trace_id": "t"},
            name="passthrough-llm-call",
        )

        kwargs = client.start_observation.call_args.kwargs
        assert kwargs["name"] == "passthrough-llm-call"

    @patch("minds.requests.langfuse_tracing.get_client")
    def test_skips_when_usage_is_none(self, mock_get_client):
        """``usage=None`` is a hard no-op (upstream didn't return usage)."""
        client = MagicMock()
        mock_get_client.return_value = client

        update_generation_usage(None, model="gpt-4o-mini")

        client.update_current_generation.assert_not_called()
        client.start_observation.assert_not_called()
        # get_client should not even be called when usage is None — verify
        # we short-circuit before instantiating any client.
        mock_get_client.assert_not_called()

    @patch("minds.requests.langfuse_tracing.logger")
    @patch("minds.requests.langfuse_tracing.get_client")
    def test_swallows_exceptions(self, mock_get_client, mock_logger):
        """Errors from Langfuse must never propagate to the user request."""
        client = MagicMock()
        client.update_current_generation.side_effect = RuntimeError("boom")
        mock_get_client.return_value = client

        # Should not raise.
        update_generation_usage((1, 2), model="m")

        mock_logger.error.assert_any_call("Error updating Langfuse generation usage: boom")

    @patch("minds.requests.langfuse_tracing.get_client")
    def test_forwards_metadata_kwarg_in_scope(self, mock_get_client):
        """Passthrough flows pass alias/provider metadata; the helper must
        forward it to ``update_current_generation`` verbatim."""
        client = MagicMock()
        mock_get_client.return_value = client

        meta = {"passthrough_alias": "sonnet", "provider": "anthropic"}
        update_generation_usage((1, 2), model="claude-sonnet-4-6", metadata=meta)

        client.update_current_generation.assert_called_once_with(
            model="claude-sonnet-4-6",
            usage_details={"input": 1, "output": 2, "total": 3},
            input=None,
            output=None,
            metadata=meta,
        )

    @patch("minds.requests.langfuse_tracing.get_client")
    def test_forwards_metadata_kwarg_detached(self, mock_get_client):
        """Detached mode (streaming) also forwards metadata to ``start_observation``."""
        client = MagicMock()
        client.start_observation.return_value = MagicMock()
        mock_get_client.return_value = client

        meta = {"passthrough_alias": "gpt-5.5-medium", "reasoning_effort": "medium"}
        update_generation_usage(
            (4, 8),
            model="gpt-5.5",
            trace_context={"trace_id": "t"},
            metadata=meta,
        )

        assert client.start_observation.call_args.kwargs["metadata"] == meta

    @patch("minds.requests.langfuse_tracing.get_client")
    def test_handles_zero_token_counts(self, mock_get_client):
        """Zero counts still produce a well-formed usage_details payload."""
        client = MagicMock()
        mock_get_client.return_value = client

        update_generation_usage((0, 0), model="m")

        client.update_current_generation.assert_called_once_with(
            model="m",
            usage_details={"input": 0, "output": 0, "total": 0},
            input=None,
            output=None,
            metadata=None,
        )


class TestCaptureLangfuseGenerationContext:
    """Test cases for capture_langfuse_generation_context helper."""

    @patch("minds.requests.langfuse_tracing.get_client")
    def test_returns_dict_with_trace_and_parent(self, mock_get_client):
        client = MagicMock()
        client.get_current_trace_id.return_value = "trace1"
        client.get_current_observation_id.return_value = "obs1"
        mock_get_client.return_value = client

        ctx = capture_langfuse_generation_context()

        assert ctx == {"trace_id": "trace1", "parent_span_id": "obs1"}

    @patch("minds.requests.langfuse_tracing.get_client")
    def test_omits_parent_span_id_when_observation_disabled(self, mock_get_client):
        client = MagicMock()
        client.get_current_trace_id.return_value = "trace1"
        client.get_current_observation_id.return_value = "disabled"
        mock_get_client.return_value = client

        ctx = capture_langfuse_generation_context()

        assert ctx == {"trace_id": "trace1"}

    @patch("minds.requests.langfuse_tracing.get_client")
    def test_returns_none_when_disabled(self, mock_get_client):
        client = MagicMock()
        client.get_current_trace_id.return_value = "disabled"
        mock_get_client.return_value = client

        ctx = capture_langfuse_generation_context()

        assert ctx is None

    @patch("minds.requests.langfuse_tracing.get_client")
    def test_returns_none_when_trace_id_missing(self, mock_get_client):
        client = MagicMock()
        client.get_current_trace_id.return_value = None
        mock_get_client.return_value = client

        ctx = capture_langfuse_generation_context()

        assert ctx is None

    @patch("minds.requests.langfuse_tracing.logger")
    @patch("minds.requests.langfuse_tracing.get_client")
    def test_returns_none_on_exception(self, mock_get_client, mock_logger):
        client = MagicMock()
        client.get_current_trace_id.side_effect = RuntimeError("kaboom")
        mock_get_client.return_value = client

        ctx = capture_langfuse_generation_context()

        assert ctx is None
        mock_logger.error.assert_any_call("Error capturing Langfuse generation context: kaboom")
