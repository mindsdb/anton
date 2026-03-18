"""Unit tests for the AntonAgent Harbor adapter.

All LLM and environment calls are mocked — no API keys or Docker required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.llm.provider import LLMResponse, ToolCall, Usage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_logs(tmp_path: Path) -> Path:
    logs = tmp_path / "logs"
    logs.mkdir()
    return logs


@pytest.fixture
def mock_environment():
    """A fake BaseEnvironment that records exec calls."""
    env = AsyncMock()
    env.exec = AsyncMock()
    return env


@pytest.fixture
def mock_context():
    """A bare AgentContext-like object."""
    ctx = MagicMock()
    ctx.n_input_tokens = None
    ctx.n_output_tokens = None
    ctx.cost_usd = None
    return ctx


def _make_response(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _make_exec_result(stdout: str = "", stderr: str = "", return_code: int = 0):
    """Create a mock ExecResult."""
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.return_code = return_code
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAntonAgentMetadata:
    """Test agent identity methods."""

    def test_name(self, tmp_logs):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            from evals.terminal_bench.agent import AntonAgent
            agent = AntonAgent(logs_dir=tmp_logs, model_name="anthropic/claude-sonnet-4-6")
            assert agent.name() == "anton"

    def test_version(self, tmp_logs):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            from evals.terminal_bench.agent import AntonAgent
            import anton
            agent = AntonAgent(logs_dir=tmp_logs)
            assert agent.version() == anton.__version__

    def test_model_parsing(self, tmp_logs):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            from evals.terminal_bench.agent import AntonAgent
            agent = AntonAgent(logs_dir=tmp_logs, model_name="anthropic/claude-sonnet-4-6")
            assert agent._planning_model == "claude-sonnet-4-6"

    def test_model_without_provider_prefix(self, tmp_logs):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            from evals.terminal_bench.agent import AntonAgent
            agent = AntonAgent(logs_dir=tmp_logs, model_name="claude-sonnet-4-6")
            assert agent._planning_model == "claude-sonnet-4-6"


class TestSetup:
    """Test agent setup."""

    async def test_setup_creates_llm_client(self, tmp_logs, mock_environment):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            from evals.terminal_bench.agent import AntonAgent
            agent = AntonAgent(logs_dir=tmp_logs, model_name="anthropic/claude-sonnet-4-6")
            await agent.setup(mock_environment)
            assert agent._llm is not None

    async def test_setup_fails_without_api_key(self, tmp_logs, mock_environment):
        with patch.dict("os.environ", {}, clear=True):
            # Also clear any existing ANTHROPIC_API_KEY
            import os
            old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
            try:
                from evals.terminal_bench.agent import AntonAgent
                agent = AntonAgent(logs_dir=tmp_logs)
                with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                    await agent.setup(mock_environment)
            finally:
                if old_key:
                    os.environ["ANTHROPIC_API_KEY"] = old_key


class TestExecuteTool:
    """Test the execute tool handler."""

    async def test_execute_success(self, tmp_logs, mock_environment):
        from evals.terminal_bench.agent import AntonAgent
        agent = AntonAgent(logs_dir=tmp_logs)

        mock_environment.exec.return_value = _make_exec_result(
            stdout="hello world\n", return_code=0
        )

        result = await agent._handle_execute(
            {"command": "echo hello world", "description": "test echo"},
            mock_environment,
        )

        assert "[stdout]" in result
        assert "hello world" in result
        assert "[return_code] 0" in result
        mock_environment.exec.assert_awaited_once()

    async def test_execute_with_stderr(self, tmp_logs, mock_environment):
        from evals.terminal_bench.agent import AntonAgent
        agent = AntonAgent(logs_dir=tmp_logs)

        mock_environment.exec.return_value = _make_exec_result(
            stdout="", stderr="warning: something\n", return_code=0
        )

        result = await agent._handle_execute(
            {"command": "some_cmd", "description": "test"},
            mock_environment,
        )

        assert "[stderr]" in result
        assert "warning: something" in result

    async def test_execute_failure(self, tmp_logs, mock_environment):
        from evals.terminal_bench.agent import AntonAgent
        agent = AntonAgent(logs_dir=tmp_logs)

        mock_environment.exec.return_value = _make_exec_result(
            stderr="command not found\n", return_code=127
        )

        result = await agent._handle_execute(
            {"command": "nonexistent", "description": "test"},
            mock_environment,
        )

        assert "[return_code] 127" in result

    async def test_execute_no_command(self, tmp_logs, mock_environment):
        from evals.terminal_bench.agent import AntonAgent
        agent = AntonAgent(logs_dir=tmp_logs)

        result = await agent._handle_execute(
            {"command": "", "description": "empty"},
            mock_environment,
        )

        assert "Error" in result or "error" in result

    async def test_execute_truncates_long_output(self, tmp_logs, mock_environment):
        from evals.terminal_bench.agent import AntonAgent
        agent = AntonAgent(logs_dir=tmp_logs)

        long_output = "x" * 20_000
        mock_environment.exec.return_value = _make_exec_result(
            stdout=long_output, return_code=0
        )

        result = await agent._handle_execute(
            {"command": "cat bigfile", "description": "test truncation"},
            mock_environment,
        )

        assert "truncated" in result
        assert len(result) < 20_000


class TestWriteFileTool:
    """Test the write_file tool handler."""

    async def test_write_file_success(self, tmp_logs, mock_environment):
        from evals.terminal_bench.agent import AntonAgent
        agent = AntonAgent(logs_dir=tmp_logs)

        mock_environment.exec.return_value = _make_exec_result(return_code=0)

        result = await agent._handle_write_file(
            {"path": "/tmp/test.py", "content": "print('hello')", "description": "test script"},
            mock_environment,
        )

        assert "File written" in result
        assert "/tmp/test.py" in result
        mock_environment.exec.assert_awaited_once()

    async def test_write_file_no_path(self, tmp_logs, mock_environment):
        from evals.terminal_bench.agent import AntonAgent
        agent = AntonAgent(logs_dir=tmp_logs)

        result = await agent._handle_write_file(
            {"path": "", "content": "data", "description": "test"},
            mock_environment,
        )

        assert "Error" in result or "error" in result


class TestAgentLoop:
    """Test the full run() agent loop."""

    async def test_simple_run_no_tools(self, tmp_logs, mock_environment, mock_context):
        """Agent responds without using any tools."""
        from evals.terminal_bench.agent import AntonAgent

        agent = AntonAgent(logs_dir=tmp_logs, model_name="anthropic/claude-sonnet-4-6")

        # Mock LLM client
        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_make_response(
            content="The task is already complete.",
            input_tokens=200,
            output_tokens=100,
        ))
        agent._llm = mock_llm

        await agent.run("Complete this task.", mock_environment, mock_context)

        # LLM called once, no tools
        mock_llm.plan.assert_awaited_once()
        assert mock_context.n_input_tokens == 200
        assert mock_context.n_output_tokens == 100
        assert mock_context.cost_usd > 0

    async def test_run_with_tool_calls(self, tmp_logs, mock_environment, mock_context):
        """Agent uses execute tool then finishes."""
        from evals.terminal_bench.agent import AntonAgent

        agent = AntonAgent(logs_dir=tmp_logs, model_name="anthropic/claude-sonnet-4-6")

        # First LLM call: use the execute tool
        response_1 = _make_response(
            content="Let me check the files.",
            tool_calls=[
                ToolCall(id="tc_1", name="execute", input={
                    "command": "ls -la /",
                    "description": "List root directory",
                }),
            ],
            input_tokens=300,
            output_tokens=150,
        )

        # Second LLM call: done
        response_2 = _make_response(
            content="Task complete. I found the files.",
            input_tokens=500,
            output_tokens=80,
        )

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(side_effect=[response_1, response_2])
        agent._llm = mock_llm

        mock_environment.exec.return_value = _make_exec_result(
            stdout="bin  etc  home  tmp  usr  var\n", return_code=0
        )

        await agent.run("List all files.", mock_environment, mock_context)

        assert mock_llm.plan.await_count == 2
        mock_environment.exec.assert_awaited_once()
        assert mock_context.n_input_tokens == 800  # 300 + 500
        assert mock_context.n_output_tokens == 230  # 150 + 80

    async def test_run_max_rounds(self, tmp_logs, mock_environment, mock_context):
        """Agent hits the max tool rounds limit."""
        from evals.terminal_bench.agent import AntonAgent, _MAX_TOOL_ROUNDS

        agent = AntonAgent(logs_dir=tmp_logs)

        # Always return a tool call — never finish
        infinite_response = _make_response(
            content="Trying again...",
            tool_calls=[
                ToolCall(id="tc_loop", name="execute", input={
                    "command": "echo loop",
                    "description": "Infinite loop",
                }),
            ],
        )

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=infinite_response)
        agent._llm = mock_llm

        mock_environment.exec.return_value = _make_exec_result(stdout="loop\n")

        await agent.run("Do something.", mock_environment, mock_context)

        # Should have called LLM _MAX_TOOL_ROUNDS times (each round = 1 LLM call)
        assert mock_llm.plan.await_count == _MAX_TOOL_ROUNDS

    async def test_conversation_log_saved(self, tmp_logs, mock_environment, mock_context):
        """Verify conversation log is written to disk."""
        from evals.terminal_bench.agent import AntonAgent

        agent = AntonAgent(logs_dir=tmp_logs, model_name="anthropic/claude-sonnet-4-6")

        mock_llm = AsyncMock()
        mock_llm.plan = AsyncMock(return_value=_make_response(content="Done."))
        agent._llm = mock_llm

        await agent.run("Simple task.", mock_environment, mock_context)

        log_path = tmp_logs / "anton_agent_log.json"
        assert log_path.exists()

        log_data = json.loads(log_path.read_text())
        assert log_data["agent"] == "anton"
        assert log_data["model"] == "claude-sonnet-4-6"
        assert len(log_data["messages"]) >= 2  # user + assistant


class TestCostTracking:
    """Test token counting and cost estimation."""

    def test_cost_calculation(self, tmp_logs):
        from evals.terminal_bench.agent import AntonAgent

        agent = AntonAgent(logs_dir=tmp_logs)
        ctx = MagicMock()

        agent._update_context(ctx, input_tokens=1_000_000, output_tokens=100_000)

        # Input: 1M * $3/MTok = $3.00
        # Output: 100K * $15/MTok = $1.50
        # Total = $4.50
        assert ctx.cost_usd == pytest.approx(4.50, abs=0.01)
        assert ctx.n_input_tokens == 1_000_000
        assert ctx.n_output_tokens == 100_000
