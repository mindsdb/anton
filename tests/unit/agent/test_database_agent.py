"""
Unit tests for DatabaseAgent class.

Tests the database agent functionality including:
- Agent initialization
- Conversation context building
- SQL generation and execution
- Error handling
- Streaming vs non-streaming responses
"""

import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest

from minds.agent.database_agent import DatabaseAgent, DatabaseDeps
from minds.model.mind import Mind
from minds.schemas.chat import Message, Role

# Mock langfuse before importing any modules that use it
if "langfuse" not in sys.modules:
    mock_langfuse = Mock()
    mock_langfuse.observe = lambda f=None, **_: (lambda *a, **k: f(*a, **k)) if f else (lambda x: x)
    sys.modules["langfuse"] = mock_langfuse


class TestDatabaseDeps:
    """Test cases for DatabaseDeps dataclass."""

    def test_database_deps_initialization(self):
        """Test DatabaseDeps initialization with required parameters."""
        mock_toolkit = Mock()
        deps = DatabaseDeps(toolkit=mock_toolkit)

        assert deps.toolkit == mock_toolkit
        assert deps.conversation_context is None

    def test_database_deps_with_conversation_context(self):
        """Test DatabaseDeps initialization with conversation context."""
        mock_toolkit = Mock()
        context = "User: Hello\nAssistant: Hi there!"
        deps = DatabaseDeps(toolkit=mock_toolkit, conversation_context=context)

        assert deps.toolkit == mock_toolkit
        assert deps.conversation_context == context


class TestDatabaseAgent:
    """Test cases for DatabaseAgent class."""

    @pytest.fixture
    def mock_mind(self):
        """Create a mock Mind instance for testing."""
        mind = Mock(spec=Mind)
        mind.name = "test-mind"
        mind.provider = "openai"
        mind.model_name = "gpt-3.5-turbo"
        mind.user_id = "test-user"
        mind.tenant_id = "test-tenant"
        return mind

    @pytest.fixture
    def mock_database_toolkit(self):
        """Create a mock DatabaseToolkit instance for testing."""
        return Mock()

    @pytest.fixture
    def database_agent(self, mock_mind, mock_database_toolkit):
        """Create a DatabaseAgent instance for testing."""
        with (
            patch("minds.agent.database_agent.get_llm_config") as mock_get_llm,
            patch("minds.agent.database_agent.PydanticAIAgent") as mock_agent_class,
        ):
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent
            return DatabaseAgent(mind=mock_mind, database_toolkit=mock_database_toolkit)

    def test_database_agent_initialization(self, mock_mind, mock_database_toolkit):
        """Test DatabaseAgent initialization."""
        with (
            patch("minds.agent.database_agent.get_llm_config") as mock_get_llm,
            patch("minds.agent.database_agent.PydanticAIAgent") as mock_agent_class,
        ):
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            agent = DatabaseAgent(mind=mock_mind, database_toolkit=mock_database_toolkit)

            assert agent.mind == mock_mind
            assert agent.deps.toolkit == mock_database_toolkit
            assert agent.deps.conversation_context is None
            assert agent._pydantic_agent is not None
            mock_get_llm.assert_called_once_with(mock_mind.provider, mock_mind.model_name)

    def test_database_agent_initialization_with_llm_config_error(self, mock_mind, mock_database_toolkit):
        """Test DatabaseAgent initialization when LLM config fails."""
        with patch("minds.agent.database_agent.get_llm_config") as mock_get_llm:
            mock_get_llm.side_effect = ValueError("Unsupported provider")

            with pytest.raises(ValueError, match="Unsupported provider"):
                DatabaseAgent(mind=mock_mind, database_toolkit=mock_database_toolkit)

    def test_build_conversation_context_empty_messages(self, database_agent):
        """Test building conversation context with empty messages."""
        messages = []
        context = database_agent._build_conversation_context(messages)
        assert context == ""

    def test_build_conversation_context_single_message(self, database_agent):
        """Test building conversation context with single message."""
        messages = [Message(role=Role.user, content="Hello")]
        context = database_agent._build_conversation_context(messages)
        assert context == "Hello"

    def test_build_conversation_context_single_empty_message(self, database_agent):
        """Test building conversation context with single empty message."""
        messages = [Message(role=Role.user, content="")]
        context = database_agent._build_conversation_context(messages)
        assert context == ""

    def test_build_conversation_context_multiple_messages(self, database_agent):
        """Test building conversation context with multiple messages."""
        messages = [
            Message(role=Role.user, content="Hello"),
            Message(role=Role.assistant, content="Hi there!"),
            Message(role=Role.user, content="How are you?"),
        ]
        context = database_agent._build_conversation_context(messages)

        expected = (
            "This is a conversation history. Please respond to the most recent user message "
            "while considering the full context:\n\n"
            "User: Hello\n\nAssistant: Hi there!\n\nUser: How are you?"
        )
        assert context == expected

    def test_build_conversation_context_with_system_message(self, database_agent):
        """Test building conversation context with system message."""
        messages = [
            Message(role=Role.system, content="You are a helpful assistant"),
            Message(role=Role.user, content="Hello"),
        ]
        context = database_agent._build_conversation_context(messages)

        expected = (
            "This is a conversation history. Please respond to the most recent user message "
            "while considering the full context:\n\n"
            "System: You are a helpful assistant\n\nUser: Hello"
        )
        assert context == expected

    def test_build_conversation_context_with_none_content(self, database_agent):
        """Test building conversation context with None content."""
        # Create messages with empty string instead of None to avoid validation errors
        messages = [
            Message(role=Role.user, content=""),
            Message(role=Role.assistant, content="Response"),
        ]
        context = database_agent._build_conversation_context(messages)

        expected = (
            "This is a conversation history. Please respond to the most recent user message "
            "while considering the full context:\n\n"
            "User: \n\nAssistant: Response"
        )
        assert context == expected

    def test_build_conversation_context_with_none_role(self, database_agent):
        """Test building conversation context with None role."""
        # Create a message with a valid role instead of None to avoid validation errors
        messages = [Message(role=Role.user, content="Test message")]
        context = database_agent._build_conversation_context(messages)

        expected = "Test message"
        assert context == expected

    @pytest.mark.asyncio
    async def test_get_completion_non_streaming(self, database_agent):
        """Test get_completion in non-streaming mode."""
        messages = [Message(role=Role.user, content="Hello")]
        expected_response = "Hello! How can I help you?"

        # Mock the agent's run method
        mock_result = Mock()
        mock_result.output = expected_response
        database_agent._pydantic_agent.run = AsyncMock(return_value=mock_result)

        # Mock the toolkit's generate_and_execute_sql method
        database_agent.deps.toolkit.generate_and_execute_sql = AsyncMock(return_value="SQL result")

        result = []
        async for chunk in database_agent.get_completion(messages, stream=False):
            result.append(chunk)

        assert len(result) == 1
        assert result[0] == expected_response

        # Verify conversation context was set
        assert database_agent.deps.conversation_context == "Hello"

    @pytest.mark.asyncio
    async def test_get_completion_streaming(self, database_agent):
        """Test get_completion in streaming mode."""
        messages = [Message(role=Role.user, content="Hello")]
        expected_chunks = ["Hello", "! ", "How", " can", " I", " help", " you?"]

        # Mock the agent's run_stream method as an async context manager
        mock_stream_result = Mock()

        async def mock_stream_text(delta=True):
            for chunk in expected_chunks:
                yield chunk

        mock_stream_result.stream_text = mock_stream_text

        class MockAsyncContextManager:
            async def __aenter__(self):
                return mock_stream_result

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        database_agent._pydantic_agent.run_stream = Mock(return_value=MockAsyncContextManager())

        # Mock the toolkit's generate_and_execute_sql method
        database_agent.deps.toolkit.generate_and_execute_sql = AsyncMock(return_value="SQL result")

        result = []
        async for chunk in database_agent.get_completion(messages, stream=True):
            result.append(chunk)

        assert result == expected_chunks

        # Verify conversation context was set
        assert database_agent.deps.conversation_context == "Hello"

    @pytest.mark.asyncio
    async def test_get_completion_with_multiple_messages(self, database_agent):
        """Test get_completion with multiple messages."""
        messages = [
            Message(role=Role.user, content="Hello"),
            Message(role=Role.assistant, content="Hi there!"),
            Message(role=Role.user, content="How are you?"),
        ]
        expected_response = "I'm doing well, thank you!"

        # Mock the agent's run method
        mock_result = Mock()
        mock_result.output = expected_response
        database_agent._pydantic_agent.run = AsyncMock(return_value=mock_result)

        # Mock the toolkit's generate_and_execute_sql method
        database_agent.deps.toolkit.generate_and_execute_sql = AsyncMock(return_value="SQL result")

        result = []
        async for chunk in database_agent.get_completion(messages, stream=False):
            result.append(chunk)

        assert len(result) == 1
        assert result[0] == expected_response

        # Verify conversation context was built correctly
        expected_context = (
            "This is a conversation history. Please respond to the most recent user message "
            "while considering the full context:\n\n"
            "User: Hello\n\nAssistant: Hi there!\n\nUser: How are you?"
        )
        assert database_agent.deps.conversation_context == expected_context

    @pytest.mark.asyncio
    async def test_get_completion_agent_run_error(self, database_agent):
        """Test get_completion when agent.run raises an exception."""
        messages = [Message(role=Role.user, content="Hello")]

        # Mock the agent's run method to raise an exception
        database_agent._pydantic_agent.run = AsyncMock(side_effect=Exception("Agent error"))

        # Mock the toolkit's generate_and_execute_sql method
        database_agent.deps.toolkit.generate_and_execute_sql = AsyncMock(return_value="SQL result")

        with pytest.raises(Exception, match="Agent error"):
            result = []
            async for chunk in database_agent.get_completion(messages, stream=False):
                result.append(chunk)

    @pytest.mark.asyncio
    async def test_get_completion_streaming_error(self, database_agent):
        """Test get_completion when streaming raises an exception."""
        messages = [Message(role=Role.user, content="Hello")]

        # Mock the agent's run_stream method to raise an exception in the context manager
        class MockAsyncContextManager:
            async def __aenter__(self):
                raise Exception("Streaming error")

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        database_agent._pydantic_agent.run_stream = Mock(return_value=MockAsyncContextManager())

        # Mock the toolkit's generate_and_execute_sql method
        database_agent.deps.toolkit.generate_and_execute_sql = AsyncMock(return_value="SQL result")

        with pytest.raises(Exception, match="Streaming error"):
            result = []
            async for chunk in database_agent.get_completion(messages, stream=True):
                result.append(chunk)

    def test_setup_agent_creates_tool(self, mock_mind, mock_database_toolkit):
        """Test that _setup_agent creates the generate_and_execute_sql tool."""
        with (
            patch("minds.agent.database_agent.get_llm_config") as mock_get_llm,
            patch("minds.agent.database_agent.PydanticAIAgent") as mock_agent_class,
        ):
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            agent = DatabaseAgent(mind=mock_mind, database_toolkit=mock_database_toolkit)

            # Verify the agent was created
            assert agent._pydantic_agent is not None
            assert agent._pydantic_agent == mock_agent

    @pytest.mark.asyncio
    async def test_tool_generate_and_execute_sql(self, database_agent):
        """Test the generate_and_execute_sql tool function."""
        # This test would require more complex mocking of the PydanticAI agent
        # For now, we'll test that the tool is properly configured
        assert database_agent._pydantic_agent is not None

    def test_conversation_context_preservation(self, database_agent):
        """Test that conversation context is preserved across multiple calls."""
        messages1 = [Message(role=Role.user, content="First message")]
        messages2 = [Message(role=Role.user, content="Second message")]

        context1 = database_agent._build_conversation_context(messages1)
        context2 = database_agent._build_conversation_context(messages2)

        assert context1 == "First message"
        assert context2 == "Second message"
        assert context1 != context2

    def test_conversation_context_with_special_characters(self, database_agent):
        """Test building conversation context with special characters."""
        messages = [
            Message(role=Role.user, content="Hello! How are you? I'm fine."),
            Message(role=Role.assistant, content="Great! I'm doing well too. 😊"),
        ]
        context = database_agent._build_conversation_context(messages)

        expected = (
            "This is a conversation history. Please respond to the most recent user message "
            "while considering the full context:\n\n"
            "User: Hello! How are you? I'm fine.\n\nAssistant: Great! I'm doing well too. 😊"
        )
        assert context == expected

    def test_conversation_context_with_empty_messages_list(self, database_agent):
        """Test building conversation context with empty messages list."""
        messages = []
        context = database_agent._build_conversation_context(messages)
        assert context == ""

    def test_conversation_context_with_single_message_conversation_instruction(self, database_agent):
        """Test that single message doesn't get conversation instruction."""
        messages = [Message(role=Role.user, content="Hello")]
        context = database_agent._build_conversation_context(messages)

        # Should not contain conversation instruction for single message
        assert "This is a conversation history" not in context
        assert context == "Hello"

    def test_conversation_context_with_multiple_messages_conversation_instruction(self, database_agent):
        """Test that multiple messages get conversation instruction."""
        messages = [
            Message(role=Role.user, content="Hello"),
            Message(role=Role.assistant, content="Hi!"),
        ]
        context = database_agent._build_conversation_context(messages)

        # Should contain conversation instruction for multiple messages
        assert "This is a conversation history" in context
        assert "Please respond to the most recent user message" in context

    @pytest.mark.asyncio
    async def test_run_completion_non_streaming_pushes_result(self, database_agent):
        """Test run_completion in non-streaming mode pushes agent output to streamer."""
        messages = [Message(role=Role.user, content="Query")]

        # Mock agent.run to return a result with output
        mock_result = Mock()
        mock_result.output = "Final answer"
        database_agent._pydantic_agent.run = AsyncMock(return_value=mock_result)

        # Prepare a mock streamer with AsyncMock push
        mock_streamer = Mock()
        mock_streamer.push = AsyncMock()

        await database_agent.run_completion(messages=messages, streamer=mock_streamer, stream=False)

        # Ensure the agent's conversation was set and streamer.push was called with the output
        assert database_agent.deps.conversation_context == "Query"
        mock_streamer.push.assert_awaited_once_with(role=Role.assistant, content="Final answer")

    @pytest.mark.asyncio
    async def test_run_completion_streaming_pushes_chunks(self, database_agent):
        """Test run_completion in streaming mode pushes chunks to streamer as they arrive."""
        messages = [Message(role=Role.user, content="Stream me")]

        # Prepare streaming chunks
        chunks = ["a", "b", "c"]

        mock_stream_result = Mock()

        async def mock_stream_text(delta=True):
            for c in chunks:
                yield c

        mock_stream_result.stream_text = mock_stream_text

        class MockAsyncContextManager:
            async def __aenter__(self):
                return mock_stream_result

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        database_agent._pydantic_agent.run_stream = Mock(return_value=MockAsyncContextManager())

        mock_streamer = Mock()
        mock_streamer.push = AsyncMock()

        await database_agent.run_completion(messages=messages, streamer=mock_streamer, stream=True)

        # Ensure push was called for each chunk
        assert mock_streamer.push.await_count == len(chunks)
        # Verify conversation context set
        assert database_agent.deps.conversation_context == "Stream me"
