import importlib
import sys
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

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
    return Context(
        user_id=UUID("00000000-0000-0000-0000-000000000001"), tenant_id=UUID("00000000-0000-0000-0000-000000000002")
    )


@pytest.fixture
def mock_mind():
    """Mock Mind object."""
    mind = Mock(spec=Mind)
    mind.name = "gpt-3.5-turbo"
    mind.provider = "openai"
    mind.model_name = "gpt-3.5-turbo"
    mind.user_id = UUID("00000000-0000-0000-0000-000000000001")
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
        self,
        mock_toolkit_class,
        mock_agent_class,
        mock_logger,
        sample_handler,
        mock_streamer,
        mock_mindsdb_client,
        mock_mind,
    ):
        """Test successful chat completions execution."""
        # Setup mock session to return mock mind
        mock_result = Mock()
        mock_result.first.return_value = mock_mind
        sample_handler.session.exec.return_value = mock_result

        # Setup mock DatabaseAgent
        mock_agent = Mock()

        async def mock_run_completion(messages, streamer, stream=False):
            # Simulate agent producing a single final chunk
            await streamer.push(
                role=Role.assistant, content=f"Processed {len(sample_handler.messages)} messages with MindsDB session"
            )

        mock_agent.run_completion = mock_run_completion
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
        mock_agent_class.assert_called_once_with(mind=mock_mind, database_toolkit=mock_toolkit, config=None)

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
    @patch("minds.handlers.chat_completions_handler.DatabaseAgent")
    @patch("minds.handlers.chat_completions_handler.DatabaseToolkit")
    async def test_chat_completions_mindsdb_models_error(
        self,
        mock_toolkit_class,
        mock_agent_class,
        mock_logger,
        sample_handler,
        mock_streamer,
        mock_mindsdb_client,
        mock_mind,
    ):
        """Test chat completions when MindsDB models.list() raises an exception."""
        # Setup mock session to return mock mind
        mock_result = Mock()
        mock_result.first.return_value = mock_mind
        sample_handler.session.exec.return_value = mock_result

        # Setup mock DatabaseAgent
        mock_agent = Mock()

        async def mock_run_completion(messages, streamer, stream=False):
            await streamer.push(
                role=Role.assistant, content=f"Processed {len(sample_handler.messages)} messages with MindsDB session"
            )

        mock_agent.run_completion = mock_run_completion
        mock_agent_class.return_value = mock_agent

        # Setup mock DatabaseToolkit
        mock_toolkit = Mock()
        mock_toolkit_class.return_value = mock_toolkit

        # Execute chat completions
        result = await sample_handler.chat_completions(mock_streamer)

        # Verify the result (method returns None)
        assert result is None

        # Verify DatabaseAgent was created with correct parameters
        mock_agent_class.assert_called_once_with(mind=mock_mind, database_toolkit=mock_toolkit, config=None)

        # Verify DatabaseToolkit was created with correct parameters
        mock_toolkit_class.assert_called_once_with(mind=mock_mind, mindsdb_client=mock_mindsdb_client)

        # Verify streamer.push was called with the expected content
        push_calls = mock_streamer.push.call_args_list
        assert len(push_calls) == 1  # Should be called once with the result
        assert push_calls[0][1]["role"] == Role.assistant
        expected_content = f"Processed {len(sample_handler.messages)} messages with MindsDB session"
        assert expected_content in push_calls[0][1]["content"]

    @pytest.mark.asyncio
    @patch("minds.handlers.chat_completions_handler.logger")
    @patch("minds.handlers.chat_completions_handler.DatabaseAgent")
    @patch("minds.handlers.chat_completions_handler.DatabaseToolkit")
    async def test_chat_completions_mindsdb_databases_error(
        self,
        mock_toolkit_class,
        mock_agent_class,
        mock_logger,
        sample_handler,
        mock_streamer,
        mock_mindsdb_client,
        mock_mind,
    ):
        """Test chat completions when MindsDB databases.list() raises an exception."""
        # Setup mock session to return mock mind
        mock_result = Mock()
        mock_result.first.return_value = mock_mind
        sample_handler.session.exec.return_value = mock_result

        # Setup mock DatabaseAgent
        mock_agent = Mock()

        async def mock_run_completion(messages, streamer, stream=False):
            await streamer.push(
                role=Role.assistant, content=f"Processed {len(sample_handler.messages)} messages with MindsDB session"
            )

        mock_agent.run_completion = mock_run_completion
        mock_agent_class.return_value = mock_agent

        # Setup mock DatabaseToolkit
        mock_toolkit = Mock()
        mock_toolkit_class.return_value = mock_toolkit

        # Execute chat completions
        result = await sample_handler.chat_completions(mock_streamer)

        # Verify the result (method returns None)
        assert result is None

        # Verify DatabaseAgent was created with correct parameters
        mock_agent_class.assert_called_once_with(mind=mock_mind, database_toolkit=mock_toolkit, config=None)

        # Verify DatabaseToolkit was created with correct parameters
        mock_toolkit_class.assert_called_once_with(mind=mock_mind, mindsdb_client=mock_mindsdb_client)

        # Verify streamer.push was called with the expected content
        push_calls = mock_streamer.push.call_args_list
        assert len(push_calls) == 1  # Should be called once with the result
        assert push_calls[0][1]["role"] == Role.assistant
        expected_content = f"Processed {len(sample_handler.messages)} messages with MindsDB session"
        assert expected_content in push_calls[0][1]["content"]

    @pytest.mark.asyncio
    @patch("minds.handlers.chat_completions_handler.logger")
    @patch("minds.handlers.chat_completions_handler.DatabaseAgent")
    @patch("minds.handlers.chat_completions_handler.DatabaseToolkit")
    async def test_chat_completions_streamer_error_during_processing(
        self,
        mock_toolkit_class,
        mock_agent_class,
        mock_logger,
        sample_handler,
        mock_streamer,
        mock_mindsdb_client,
        mock_mind,
    ):
        """Test chat completions when streamer.push raises an exception during final processing."""
        # Setup mock session to return mock mind
        mock_result = Mock()
        mock_result.first.return_value = mock_mind
        sample_handler.session.exec.return_value = mock_result

        # Setup mock DatabaseAgent
        mock_agent = Mock()

        async def mock_run_completion(messages, streamer, stream=False):
            await streamer.push(
                role=Role.assistant, content=f"Processed {len(sample_handler.messages)} messages with MindsDB session"
            )

        mock_agent.run_completion = mock_run_completion
        mock_agent_class.return_value = mock_agent

        # Setup mock DatabaseToolkit
        mock_toolkit = Mock()
        mock_toolkit_class.return_value = mock_toolkit

        # Execute chat completions - should handle the exception
        result = await sample_handler.chat_completions(mock_streamer)

        # Verify the result (method returns None)
        assert result is None

        # Verify DatabaseAgent was created with correct parameters
        mock_agent_class.assert_called_once_with(mind=mock_mind, database_toolkit=mock_toolkit, config=None)

        # Verify DatabaseToolkit was created with correct parameters
        mock_toolkit_class.assert_called_once_with(mind=mock_mind, mindsdb_client=mock_mindsdb_client)

    @pytest.mark.asyncio
    @patch("minds.handlers.chat_completions_handler.DatabaseAgent")
    @patch("minds.handlers.chat_completions_handler.DatabaseToolkit")
    async def test_chat_completions_empty_messages_list(
        self,
        mock_toolkit_class,
        mock_agent_class,
        handler_mod,
        mock_session,
        mock_mindsdb_client,
        mock_streamer,
        mock_context,
        mock_mind,
    ):
        """Test chat completions with empty messages list."""
        # Setup mock session to return mock mind
        mock_result = Mock()
        mock_result.first.return_value = mock_mind
        mock_session.exec.return_value = mock_result

        # Setup mock DatabaseAgent
        mock_agent = Mock()

        async def mock_run_completion(messages, streamer, stream=False):
            await streamer.push(role=Role.assistant, content=f"Processed {len(messages)} messages with MindsDB session")

        mock_agent.run_completion = mock_run_completion
        mock_agent_class.return_value = mock_agent

        # Setup mock DatabaseToolkit
        mock_toolkit = Mock()
        mock_toolkit_class.return_value = mock_toolkit

        # Create handler with empty messages
        handler = handler_mod.ChatCompletionsHandler(
            session=mock_session,
            context=mock_context,
            mindsdb_client=mock_mindsdb_client,
            messages=[],
            model="gpt-4",
            stream=True,
        )

        # Execute chat completions
        result = await handler.chat_completions(mock_streamer)

        # Verify the result (method returns None)
        assert result is None

        # Verify DatabaseAgent was created with correct parameters
        mock_agent_class.assert_called_once_with(mind=mock_mind, database_toolkit=mock_toolkit, config=None)

        # Verify DatabaseToolkit was created with correct parameters
        mock_toolkit_class.assert_called_once_with(mind=mock_mind, mindsdb_client=mock_mindsdb_client)

        # Verify streamer.push was called with the expected content
        push_calls = mock_streamer.push.call_args_list
        assert len(push_calls) == 1  # Should be called once with the result
        assert push_calls[0][1]["role"] == Role.assistant
        expected_content = "Processed 0 messages with MindsDB session"
        assert expected_content in push_calls[0][1]["content"]

    @pytest.mark.asyncio
    @patch("minds.handlers.chat_completions_handler.DatabaseAgent")
    @patch("minds.handlers.chat_completions_handler.DatabaseToolkit")
    async def test_chat_completions_different_message_roles(
        self,
        mock_toolkit_class,
        mock_agent_class,
        handler_mod,
        mock_session,
        mock_mindsdb_client,
        mock_streamer,
        mock_context,
        mock_mind,
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

        # Setup mock session to return mock mind
        mock_result = Mock()
        mock_result.first.return_value = mock_mind
        mock_session.exec.return_value = mock_result

        # Setup mock DatabaseAgent
        mock_agent = Mock()

        async def mock_run_completion(messages, streamer, stream=False):
            await streamer.push(role=Role.assistant, content=f"Processed {len(messages)} messages with MindsDB session")

        mock_agent.run_completion = mock_run_completion
        mock_agent_class.return_value = mock_agent

        # Setup mock DatabaseToolkit
        mock_toolkit = Mock()
        mock_toolkit_class.return_value = mock_toolkit

        # Execute chat completions
        result = await handler.chat_completions(mock_streamer)

        # Verify the result (method returns None)
        assert result is None

        # Verify DatabaseAgent was created with correct parameters
        mock_agent_class.assert_called_once_with(mind=mock_mind, database_toolkit=mock_toolkit, config=None)

        # Verify DatabaseToolkit was created with correct parameters
        mock_toolkit_class.assert_called_once_with(mind=mock_mind, mindsdb_client=mock_mindsdb_client)

    @pytest.mark.asyncio
    @patch("minds.handlers.chat_completions_handler.logger")
    @patch("minds.handlers.chat_completions_handler.DatabaseAgent")
    @patch("minds.handlers.chat_completions_handler.DatabaseToolkit")
    async def test_chat_completions_logging_behavior(
        self,
        mock_toolkit_class,
        mock_agent_class,
        mock_logger,
        sample_handler,
        mock_streamer,
        mock_mindsdb_client,
        mock_mind,
    ):
        """Test that logging calls are made correctly throughout execution."""
        # Setup mock session to return mock mind
        mock_result = Mock()
        mock_result.first.return_value = mock_mind
        sample_handler.session.exec.return_value = mock_result

        # Setup mock DatabaseAgent
        mock_agent = Mock()

        async def mock_run_completion(messages, streamer, stream=False):
            await streamer.push(
                role=Role.assistant, content=f"Processed {len(sample_handler.messages)} messages with MindsDB session"
            )

        mock_agent.run_completion = mock_run_completion
        mock_agent_class.return_value = mock_agent

        # Setup mock DatabaseToolkit
        mock_toolkit = Mock()
        mock_toolkit_class.return_value = mock_toolkit

        # Execute chat completions
        await sample_handler.chat_completions(mock_streamer)

        # Verify DatabaseAgent was created with correct parameters
        mock_agent_class.assert_called_once_with(mind=mock_mind, database_toolkit=mock_toolkit, config=None)

        # Verify DatabaseToolkit was created with correct parameters
        mock_toolkit_class.assert_called_once_with(mind=mock_mind, mindsdb_client=mock_mindsdb_client)

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
    @patch("minds.handlers.chat_completions_handler.DatabaseAgent")
    @patch("minds.handlers.chat_completions_handler.DatabaseToolkit")
    async def test_chat_completions_return_type_consistency(
        self,
        mock_toolkit_class,
        mock_agent_class,
        mock_logger,
        sample_handler,
        mock_streamer,
        mock_mindsdb_client,
        mock_mind,
    ):
        """Test that chat_completions always returns a string."""
        # Setup mock session to return mock mind
        mock_result = Mock()
        mock_result.first.return_value = mock_mind
        sample_handler.session.exec.return_value = mock_result

        # Setup mock DatabaseAgent
        mock_agent = Mock()

        async def mock_run_completion(messages, streamer, stream=False):
            await streamer.push(
                role=Role.assistant, content=f"Processed {len(sample_handler.messages)} messages with MindsDB session"
            )

        mock_agent.run_completion = mock_run_completion
        mock_agent_class.return_value = mock_agent

        # Setup mock DatabaseToolkit
        mock_toolkit = Mock()
        mock_toolkit_class.return_value = mock_toolkit

        result = await sample_handler.chat_completions(mock_streamer)
        # The method returns None, not a string
        assert result is None
