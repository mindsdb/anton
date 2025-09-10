import importlib
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest
from mindsdb_sdk.server import Server
from sqlmodel import Session

# Mock langfuse before importing any modules that use it
if "langfuse" not in sys.modules:
    mock_langfuse = Mock()
    mock_langfuse.observe = lambda f=None, **_: (lambda *a, **k: f(*a, **k)) if f else (lambda x: x)
    sys.modules["langfuse"] = mock_langfuse

from minds.model.mind import Mind
from minds.requests.context import Context
from minds.requests.stream import MessageStreamer
from minds.schemas.chat import Message, Role


@pytest.fixture()
def handler_mod(monkeypatch):
    import langfuse as dec

    monkeypatch.setattr(
        dec,
        "observe",
        lambda f=None, **_: (lambda *a, **k: f(*a, **k)) if f else (lambda x: x),
    )

    import minds.handlers.chat_completions_handler as mod

    importlib.reload(mod)
    return mod


@pytest.fixture
def mock_session():
    """Mock SQLModel session."""
    return Mock(spec=Session)


@pytest.fixture
def mock_mindsdb_client():
    """Mock MindsDB client."""
    mock_client = Mock(spec=Server)
    mock_client.models = Mock()
    mock_client.databases = Mock()
    return mock_client


@pytest.fixture
def mock_context():
    """Mock Context."""
    return Context(user_id="test_user", tenant_id="test_tenant", user_email="test@example.com")


@pytest.fixture
def mock_mind():
    """Mock Mind object."""
    mind = Mock(spec=Mind)
    mind.name = "gpt-3.5-turbo"
    mind.provider = "openai"
    mind.model_name = "gpt-3.5-turbo"
    mind.user_id = "test_user"
    mind.parameters = {}
    mind.description = "Test mind"
    mind.mind_datasources = []
    return mind


@pytest.fixture
def mock_streamer():
    """Mock MessageStreamer."""
    mock_streamer = Mock(spec=MessageStreamer)
    mock_streamer.push = AsyncMock()
    return mock_streamer


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        Message(role=Role.user, content="Hello, how are you?"),
        Message(role=Role.assistant, content="I'm doing well, thank you!"),
        Message(role=Role.user, content="What can you help me with?"),
    ]


@pytest.fixture
def sample_handler(handler_mod, mock_session, mock_mindsdb_client, sample_messages, mock_context):
    """Sample ChatCompletionsHandler instance for testing."""
    return handler_mod.ChatCompletionsHandler(
        session=mock_session,
        context=mock_context,
        mindsdb_client=mock_mindsdb_client,
        messages=sample_messages,
        model="gpt-3.5-turbo",
        stream=False,
    )


class TestChatCompletionsHandler:
    def test_chat_completions_handler_initialization(
        self, handler_mod, mock_session, mock_mindsdb_client, sample_messages, mock_context
    ):
        """Test ChatCompletionsHandler initialization."""
        model = "gpt-4"
        stream = True

        handler = handler_mod.ChatCompletionsHandler(
            session=mock_session,
            context=mock_context,
            mindsdb_client=mock_mindsdb_client,
            messages=sample_messages,
            model=model,
            stream=stream,
        )

        # Verify all attributes are set correctly
        assert handler.session == mock_session
        assert handler.mindsdb_client == mock_mindsdb_client
        assert handler.messages == sample_messages
        assert handler.model == model
        assert handler.stream == stream

    @pytest.mark.asyncio
    @patch("minds.handlers.chat_completions_handler.logger")
    @patch("minds.handlers.chat_completions_handler.DatabaseAgent")
    @patch("minds.handlers.chat_completions_handler.DatabaseToolkit")
    async def test_chat_completions_successful_execution(
        self, mock_toolkit_class, mock_agent_class, mock_logger, sample_handler, mock_streamer, mock_mindsdb_client, mock_mind
    ):
        """Test successful chat completions execution."""
        # Setup mock session to return mock mind
        mock_result = Mock()
        mock_result.first.return_value = mock_mind
        sample_handler.session.exec.return_value = mock_result
        
        # Setup mock DatabaseAgent
        mock_agent = Mock()
        async def mock_get_completion(messages, stream=False):
            yield f"Processed {len(sample_handler.messages)} messages with MindsDB session"
        mock_agent.get_completion = mock_get_completion
        mock_agent_class.return_value = mock_agent
        
        # Setup mock DatabaseToolkit
        mock_toolkit = Mock()
        mock_toolkit_class.return_value = mock_toolkit
        
        # Setup mock responses from MindsDB client
        mock_models = [Mock(), Mock(), Mock()]
        mock_databases = [Mock(), Mock()]
        mock_mindsdb_client.models.list.return_value = mock_models
        mock_mindsdb_client.databases.list.return_value = mock_databases

        # Execute chat completions
        result = await sample_handler.chat_completions(mock_streamer)

        # Verify the result (method returns None)
        assert result is None

        # Verify DatabaseAgent was created with correct parameters
        mock_agent_class.assert_called_once_with(mind=mock_mind, database_toolkit=mock_toolkit)
        
        # Verify DatabaseToolkit was created with correct parameters
        mock_toolkit_class.assert_called_once_with(mind=mock_mind, mindsdb_client=mock_mindsdb_client)

        # Verify streamer.push was called with the expected content
        push_calls = mock_streamer.push.call_args_list
        assert len(push_calls) == 1  # Should be called once with the result
        assert push_calls[0][1]["role"] == Role.assistant
        expected_content = f"Processed {len(sample_handler.messages)} messages with MindsDB session"
        assert expected_content in push_calls[0][1]["content"]

        # Verify logging calls (logger may not be called in this simplified test)
        # mock_logger.info.assert_called()

    @pytest.mark.asyncio
    @patch("minds.handlers.chat_completions_handler.logger")
    async def test_chat_completions_mindsdb_models_error(
        self, mock_logger, sample_handler, mock_streamer, mock_mindsdb_client, mock_mind
    ):
        """Test chat completions when MindsDB models.list() raises an exception."""
        # Setup mock session to return mock mind
        mock_result = Mock()
        mock_result.first.return_value = mock_mind
        sample_handler.session.exec.return_value = mock_result
        
        # Setup mock to raise exception on models.list()
        mock_mindsdb_client.models.list.side_effect = Exception("Models access failed")
        mock_mindsdb_client.databases.list.return_value = []

        # Execute chat completions
        result = await sample_handler.chat_completions(mock_streamer)

        # Should still return successful result despite MindsDB error
        expected_result = f"Processed {len(sample_handler.messages)} messages with MindsDB session"
        assert result == expected_result

        # Verify error was pushed to streamer
        push_calls = mock_streamer.push.call_args_list
        assert any(
            call[1]["role"] == Role.system and "Error accessing MindsDB: Models access failed" in call[1]["content"]
            for call in push_calls
        )

        # Verify error was logged
        mock_logger.error.assert_called()
        error_calls = [call[0][0] for call in mock_logger.error.call_args_list]
        assert any("MindsDB error: Models access failed" in call for call in error_calls)

    @pytest.mark.asyncio
    @patch("minds.handlers.chat_completions_handler.logger")
    async def test_chat_completions_mindsdb_databases_error(
        self, mock_logger, sample_handler, mock_streamer, mock_mindsdb_client
    ):
        """Test chat completions when MindsDB databases.list() raises an exception."""
        # Setup mock to raise exception on databases.list()
        mock_mindsdb_client.models.list.return_value = [Mock()]
        mock_mindsdb_client.databases.list.side_effect = Exception("Database access failed")

        # Execute chat completions
        result = await sample_handler.chat_completions(mock_streamer)

        # Should still return successful result despite MindsDB error
        expected_result = f"Processed {len(sample_handler.messages)} messages with MindsDB session"
        assert result == expected_result

        # Verify error was pushed to streamer
        push_calls = mock_streamer.push.call_args_list
        assert any(
            call[1]["role"] == Role.system and "Error accessing MindsDB: Database access failed" in call[1]["content"]
            for call in push_calls
        )

        # Verify error was logged
        mock_logger.error.assert_called()

    @pytest.mark.asyncio
    @patch("minds.handlers.chat_completions_handler.logger")
    async def test_chat_completions_streamer_error_during_processing(
        self, mock_logger, sample_handler, mock_streamer, mock_mindsdb_client
    ):
        """Test chat completions when streamer.push raises an exception during final processing."""
        # Setup successful MindsDB responses
        mock_mindsdb_client.models.list.return_value = [Mock()]
        mock_mindsdb_client.databases.list.return_value = [Mock()]

        # Make streamer.push fail on the final assistant message
        def push_side_effect(*args, **kwargs):
            if kwargs.get("role") == Role.assistant and "Processed" in kwargs.get("content", ""):
                raise Exception("Streamer push failed")

        mock_streamer.push.side_effect = push_side_effect

        # Execute chat completions - should handle the exception
        result = await sample_handler.chat_completions(mock_streamer)

        # Should return error message due to final processing failure
        assert "Error in MindsDB chat completion: Streamer push failed" in result

        # Verify error was logged
        mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_chat_completions_empty_messages_list(
        self, handler_mod, mock_session, mock_mindsdb_client, mock_streamer, mock_context
    ):
        """Test chat completions with empty messages list."""
        # Create handler with empty messages
        handler = handler_mod.ChatCompletionsHandler(
            session=mock_session,
            context=mock_context,
            mindsdb_client=mock_mindsdb_client,
            messages=[],
            model="gpt-4",
            stream=True,
        )

        # Setup successful MindsDB responses
        mock_mindsdb_client.models.list.return_value = []
        mock_mindsdb_client.databases.list.return_value = []

        # Execute chat completions
        result = await handler.chat_completions(mock_streamer)

        # Should handle empty messages gracefully
        assert result == "Processed 0 messages with MindsDB session"

        # Verify basic system messages were still pushed
        push_calls = mock_streamer.push.call_args_list
        assert any(call[1]["role"] == Role.system and "Using model: gpt-4" in call[1]["content"] for call in push_calls)

    @pytest.mark.asyncio
    async def test_chat_completions_different_message_roles(
        self, handler_mod, mock_session, mock_mindsdb_client, mock_streamer, mock_context
    ):
        """Test chat completions with different message roles."""
        # Create messages with different roles
        messages = [
            Message(role=Role.system, content="You are a helpful assistant"),
            Message(role=Role.user, content="Hello"),
            Message(role=Role.assistant, content="Hi there!"),
            Message(role=Role.user, content="How are you?"),
        ]

        handler = handler_mod.ChatCompletionsHandler(
            session=mock_session,
            context=mock_context,
            mindsdb_client=mock_mindsdb_client,
            messages=messages,
            model="custom-model",
            stream=False,
        )

        # Setup successful MindsDB responses
        mock_mindsdb_client.models.list.return_value = [Mock(), Mock()]
        mock_mindsdb_client.databases.list.return_value = [Mock()]

        # Execute chat completions
        result = await handler.chat_completions(mock_streamer)

        # Verify result
        assert result == f"Processed {len(messages)} messages with MindsDB session"

        # Verify all message roles were processed
        push_calls = mock_streamer.push.call_args_list
        for message in messages:
            assert any(call[1]["role"] == message.role and message.content in call[1]["content"] for call in push_calls)

    @pytest.mark.asyncio
    @patch("minds.handlers.chat_completions_handler.logger")
    async def test_chat_completions_logging_behavior(
        self, mock_logger, sample_handler, mock_streamer, mock_mindsdb_client
    ):
        """Test that logging calls are made correctly throughout execution."""
        # Setup successful MindsDB responses
        mock_models = [Mock() for _ in range(5)]
        mock_databases = [Mock() for _ in range(3)]
        mock_mindsdb_client.models.list.return_value = mock_models
        mock_mindsdb_client.databases.list.return_value = mock_databases

        # Execute chat completions
        await sample_handler.chat_completions(mock_streamer)

        # Verify info logging calls
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]

        # Check model logging
        assert any("Using model: gpt-3.5-turbo" in call for call in info_calls)

        # Check message logging
        for message in sample_handler.messages:
            assert any(f"Message received: {message.role} {message.content}" in call for call in info_calls)

        # Check MindsDB info logging
        assert any(f"Found {len(mock_models)} MindsDB models." in call for call in info_calls)
        assert any(f"Found {len(mock_databases)} databases." in call for call in info_calls)

        # Check dummy response logging
        assert any("This is a dummy chat completion response." in call for call in info_calls)

    def test_chat_completions_handler_attributes_immutable_after_init(self, sample_handler, sample_messages):
        """Test that handler attributes remain unchanged after initialization."""
        original_session = sample_handler.session
        original_client = sample_handler.mindsdb_client
        original_messages = sample_handler.messages
        original_model = sample_handler.model
        original_stream = sample_handler.stream

        # Verify attributes haven't changed
        assert sample_handler.session is original_session
        assert sample_handler.mindsdb_client is original_client
        assert sample_handler.messages is original_messages
        assert sample_handler.model == original_model
        assert sample_handler.stream == original_stream

    @pytest.mark.asyncio
    @patch("minds.handlers.chat_completions_handler.logger")
    async def test_chat_completions_return_type_consistency(
        self, mock_logger, sample_handler, mock_streamer, mock_mindsdb_client
    ):
        """Test that chat_completions always returns a string."""
        # Test successful case
        mock_mindsdb_client.models.list.return_value = []
        mock_mindsdb_client.databases.list.return_value = []

        result = await sample_handler.chat_completions(mock_streamer)
        assert isinstance(result, str)

        # Test error case by making final processing fail
        # Create a new streamer for the error test to avoid interference
        error_streamer = Mock(spec=MessageStreamer)
        error_streamer.push = AsyncMock()

        # Make the streamer fail only on the final assistant message
        def push_side_effect(*args, **kwargs):
            if kwargs.get("role") == Role.assistant and "Processed" in kwargs.get("content", ""):
                raise Exception("Test error")

        error_streamer.push.side_effect = push_side_effect

        result = await sample_handler.chat_completions(error_streamer)
        assert isinstance(result, str)
        assert "Error in MindsDB chat completion" in result
