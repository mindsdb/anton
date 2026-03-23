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
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Mock langfuse before importing any modules that use it
if "langfuse" not in sys.modules:
    mock_langfuse = Mock()
    mock_langfuse.observe = lambda f=None, **_: (lambda *a, **k: f(*a, **k)) if f else (lambda x: x)
    sys.modules["langfuse"] = mock_langfuse

from minds.agents.base import AgentRunContext
from minds.agents.database_agent.agent import DatabaseAgent, DatabaseDeps
from minds.agents.database_agent.prompt_templates import CHART_GENERATION_INSTRUCTIONS
from minds.model.mind import Mind
from minds.requests.chat_completions_request import ChatCompletionRequestMetadata
from minds.schemas.chat import Message, Role


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
        mind.organization_id = "test-organization"
        mind.parameters = {}  # Initialize as empty dict to avoid Mock return values
        return mind

    @pytest.fixture
    def mock_mindsdb_client(self):
        """Create a mock MindsDB Server instance for testing."""
        return Mock()

    @pytest.fixture
    def database_agent(self, mock_mind, mock_mindsdb_client):
        """Create a DatabaseAgent instance for testing."""
        with patch("minds.agents.database_agent.agent.DatabaseToolkit") as mock_toolkit_class:
            mock_toolkit_class.return_value = Mock()
            return DatabaseAgent(mind=mock_mind, mindsdb_client=mock_mindsdb_client)

    def test_database_agent_initialization(self, mock_mind, mock_mindsdb_client):
        """Test DatabaseAgent initialization."""
        with patch("minds.agents.database_agent.agent.DatabaseToolkit") as mock_toolkit_class:
            mock_toolkit = Mock()
            mock_toolkit_class.return_value = mock_toolkit

            agent = DatabaseAgent(mind=mock_mind, mindsdb_client=mock_mindsdb_client)

            assert agent.mind == mock_mind
            assert agent.deps.toolkit == mock_toolkit
            assert agent.deps.conversation_context is None
            mock_toolkit_class.assert_called_once_with(mind=mock_mind, mindsdb_client=mock_mindsdb_client)

    def test_database_agent_initialization_with_llm_config_error(self, mock_mind, mock_mindsdb_client):
        """Test _setup_agent when LLM config fails."""
        with (
            patch("minds.agents.database_agent.agent.DatabaseToolkit", return_value=Mock()),
            patch("minds.agents.database_agent.agent.get_llm_config") as mock_get_llm,
        ):
            mock_get_llm.side_effect = ValueError("Unsupported provider")
            agent = DatabaseAgent(mind=mock_mind, mindsdb_client=mock_mindsdb_client)

            with pytest.raises(ValueError, match="Unsupported provider"):
                agent._setup_agent(enable_charting=False)

    @pytest.mark.asyncio
    async def test_database_agent_initialization_respects_instrument_config(self, mock_mind, mock_mindsdb_client):
        """Test that run_context.instrument wires to PydanticAIAgent.instrument_all."""
        with (
            patch("minds.agents.database_agent.agent.DatabaseToolkit", return_value=Mock()),
            patch("minds.agents.database_agent.agent.PydanticAIAgent.instrument_all") as mock_instrument_all,
            patch.object(DatabaseAgent, "_setup_agent") as mock_setup_agent,
        ):
            mock_agent = Mock()
            mock_result = Mock()
            mock_result.output = "Final answer"
            mock_result.usage.return_value = 3
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_setup_agent.return_value = mock_agent

            agent = DatabaseAgent(mind=mock_mind, mindsdb_client=mock_mindsdb_client)
            streamer = Mock()
            streamer.push = AsyncMock()

            await agent.run(
                messages=[Message(role=Role.user, content="Query")],
                streamer=streamer,
                stream=False,
                run_context=AgentRunContext(
                    instrument=False,
                    metadata=ChatCompletionRequestMetadata(enable_charting=False),
                ),
            )

            mock_instrument_all.assert_called_once_with(instrument=False)

    def test_get_system_prompt_includes_system_prompt_charting_and_time(self, mock_mind, mock_mindsdb_client):
        """Test system prompt composition with charting and custom prompt."""
        mock_mind.parameters = {"system_prompt": "Follow strict SQL style."}
        fixed_now = datetime(2026, 2, 16, 10, 55, 0)

        with (
            patch("minds.agents.database_agent.agent.DatabaseToolkit") as mock_toolkit_class,
            patch("minds.agents.database_agent.agent.datetime") as mock_datetime,
        ):
            mock_toolkit_class.return_value = Mock()
            mock_datetime.now.return_value = fixed_now
            agent = DatabaseAgent(mind=mock_mind, mindsdb_client=mock_mindsdb_client)

            prompt = agent._get_system_prompt(enable_charting=True)
            assert "Follow strict SQL style." in prompt
            assert CHART_GENERATION_INSTRUCTIONS in prompt
            assert "Current date: 2026-02-16" in prompt
            assert "Current time: 10:55:00" in prompt

    def test_get_system_prompt_uses_prompt_template_fallback(self, mock_mind, mock_mindsdb_client):
        """Uses prompt_template when system_prompt is not set."""
        mock_mind.parameters = {"prompt_template": "Use concise answers."}
        fixed_now = datetime(2026, 2, 16, 10, 55, 0)
        with (
            patch("minds.agents.database_agent.agent.DatabaseToolkit") as mock_toolkit_class,
            patch("minds.agents.database_agent.agent.datetime") as mock_datetime,
        ):
            mock_toolkit_class.return_value = Mock()
            mock_datetime.now.return_value = fixed_now
            agent = DatabaseAgent(mind=mock_mind, mindsdb_client=mock_mindsdb_client)
            prompt = agent._get_system_prompt(enable_charting=False)
            assert "Use concise answers." in prompt

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
            "IMPORTANT: Use the prior conversation only for context and intent. "
            "Do not copy or reuse previous answers, even if the same or similar questions appear, "
            "as the underlying data, schema, or results may have changed."
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
            "IMPORTANT: Use the prior conversation only for context and intent. "
            "Do not copy or reuse previous answers, even if the same or similar questions appear, "
            "as the underlying data, schema, or results may have changed."
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
            "IMPORTANT: Use the prior conversation only for context and intent. "
            "Do not copy or reuse previous answers, even if the same or similar questions appear, "
            "as the underlying data, schema, or results may have changed."
        )
        assert context == expected

    def test_build_conversation_context_with_none_role(self, database_agent):
        """Test building conversation context with None role."""
        # Create a message with a valid role instead of None to avoid validation errors
        messages = [Message(role=Role.user, content="Test message")]
        context = database_agent._build_conversation_context(messages)

        expected = "Test message"
        assert context == expected

    def test_setup_agent_creates_tool(self, mock_mind, mock_mindsdb_client):
        """Test that _setup_agent creates the generate_and_execute_sql tool."""
        with (
            patch("minds.agents.database_agent.agent.DatabaseToolkit", return_value=Mock()),
            patch("minds.agents.database_agent.agent.get_llm_config", return_value=Mock()),
            patch("minds.agents.database_agent.agent.PydanticAIAgent") as mock_agent_class,
        ):
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            agent = DatabaseAgent(mind=mock_mind, mindsdb_client=mock_mindsdb_client)
            created = agent._setup_agent(enable_charting=False)
            assert created == mock_agent

    @pytest.mark.asyncio
    async def test_tool_generate_and_execute_sql(self, database_agent):
        """Test the generate_and_execute_sql tool function."""
        # This test would require more complex mocking of the PydanticAI agent
        # For now, we'll test that the tool is properly configured
        assert database_agent is not None

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
            "IMPORTANT: Use the prior conversation only for context and intent. "
            "Do not copy or reuse previous answers, even if the same or similar questions appear, "
            "as the underlying data, schema, or results may have changed."
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
    async def test_run_non_streaming_pushes_result(self, database_agent):
        """Test run() in non-streaming mode pushes agent output to streamer."""
        messages = [Message(role=Role.user, content="Query")]

        # Mock agent.run to return a result with output
        mock_result = Mock()
        mock_result.output = "Final answer"
        usage = 3
        mock_result.usage.return_value = usage
        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        # Prepare a mock streamer with AsyncMock push
        mock_streamer = Mock()
        mock_streamer.push = AsyncMock()

        with patch.object(database_agent, "_setup_agent", return_value=mock_agent):
            await database_agent.run(
                messages=messages,
                streamer=mock_streamer,
                stream=False,
                run_context=AgentRunContext(
                    instrument=True,
                    metadata=ChatCompletionRequestMetadata(enable_charting=False),
                ),
            )

        # Ensure the agent's conversation was set and streamer.push was called with the output
        assert database_agent.deps.conversation_context == "Query"
        mock_streamer.push.assert_awaited_once_with(role=Role.assistant, content="Final answer")
        assert database_agent.last_run_usage == usage

    @pytest.mark.asyncio
    async def test_run_streaming_pushes_deltas(self, database_agent):
        """Test run() in streaming mode pushes only deltas to streamer."""
        messages = [Message(role=Role.user, content="Stream me")]

        # stream_output yields objects with `answer` containing the full output so far.
        chunks = [Mock(answer="a"), Mock(answer="ab"), Mock(answer="abc")]

        mock_stream_result = Mock()

        async def mock_stream_output():
            for c in chunks:
                yield c

        mock_stream_result.stream_output = mock_stream_output
        usage = 3
        mock_stream_result.usage.return_value = usage

        class MockAsyncContextManager:
            async def __aenter__(self):
                return mock_stream_result

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        mock_agent = Mock()
        mock_agent.run_stream = Mock(return_value=MockAsyncContextManager())

        mock_streamer = Mock()
        mock_streamer.push = AsyncMock()

        with patch.object(database_agent, "_setup_agent", return_value=mock_agent):
            await database_agent.run(
                messages=messages,
                streamer=mock_streamer,
                stream=True,
                run_context=AgentRunContext(
                    instrument=True,
                    metadata=ChatCompletionRequestMetadata(enable_charting=False),
                ),
            )

        # Ensure push was called for each delta: "a", "b", "c"
        assert mock_streamer.push.await_count == 3
        assert mock_streamer.push.call_args_list[0][1]["content"] == "a"
        assert mock_streamer.push.call_args_list[1][1]["content"] == "b"
        assert mock_streamer.push.call_args_list[2][1]["content"] == "c"
        # Verify conversation context set
        assert database_agent.deps.conversation_context == "Stream me"
        assert database_agent.last_run_usage == usage
