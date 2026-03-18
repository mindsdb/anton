"""
AntonAgent — Harbor BaseAgent adapter for Terminal-Bench 2.0.

Bridges Anton's LLM planning loop to Harbor's containerized environment.
Instead of running code in a local scratchpad subprocess, all execution
is routed through ``environment.exec()`` inside the Docker container.

Usage with Harbor CLI::

    harbor run --agent evals.terminal_bench.agent:AntonAgent \\
               --model anthropic/claude-sonnet-4-6 \\
               --task <task_dir>

Or programmatically::

    from evals.terminal_bench.agent import AntonAgent
    agent = AntonAgent(logs_dir=Path("/tmp/logs"), model_name="anthropic/claude-sonnet-4-6")
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig

import anton
from anton.llm.anthropic import AnthropicProvider
from anton.llm.client import LLMClient
from anton.llm.provider import LLMResponse, ToolCall

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_TOOL_ROUNDS = 30  # Hard ceiling on consecutive tool-call rounds
_MAX_OUTPUT_CHARS = 15_000  # Truncate long exec output to save context
_DEFAULT_EXEC_TIMEOUT = 120  # seconds

_SYSTEM_PROMPT = """\
You are Anton — an autonomous AI agent solving a task inside a Linux container.

You have access to the container's shell via the provided tools. Your goal is to
read the task instruction, reason about what needs to be done, and execute shell
commands to complete the task.

APPROACH:
1. Start by understanding the environment — list files, check installed tools,
   read any relevant config or data files.
2. Break the problem into steps. Execute them one at a time so you can inspect
   intermediate results and course-correct.
3. When a command fails, read the error carefully and try a different approach.
   Exhaust at least 2-3 strategies before giving up.
4. When you believe the task is complete, do a quick sanity check (e.g. verify
   output files exist, re-run the test command if you know it).

TOOLS:
- **execute**: Run any shell command in the container. Use this for everything:
  listing files, reading files (cat/head), installing packages, running scripts,
  compiling code, etc.
- **write_file**: Write content to a file in the container. Useful for creating
  scripts, config files, patches, etc. without complex shell escaping.

IMPORTANT RULES:
- Always include a brief description of what each command does.
- Use print-style debugging — echo intermediate values, check return codes.
- If you need to install packages, use the container's package manager (apt, pip, etc.).
- Do NOT attempt to use a web browser, GUI, or anything requiring a display.
- The container may or may not have internet access — check before relying on downloads.
- Stay focused on the task. Do not modify files outside the task scope.
"""

# Tool schemas for the LLM
_EXECUTE_TOOL = {
    "name": "execute",
    "description": (
        "Execute a shell command in the container. Returns stdout, stderr, and "
        "return code. Use this for all shell operations: file listing, reading, "
        "installing packages, running scripts, compiling, etc."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "description": {
                "type": "string",
                "description": "Brief description of what this command does.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": (
                    f"Timeout in seconds (default {_DEFAULT_EXEC_TIMEOUT}). "
                    "Increase for long-running operations."
                ),
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the command (optional).",
            },
        },
        "required": ["command", "description"],
    },
}

_WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": (
        "Write content to a file in the container. Creates parent directories "
        "automatically. Use this instead of complex heredoc/echo escaping in shell."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the file in the container.",
            },
            "content": {
                "type": "string",
                "description": "The file content to write.",
            },
            "description": {
                "type": "string",
                "description": "Brief description of what this file is for.",
            },
        },
        "required": ["path", "content", "description"],
    },
}

_TOOLS = [_EXECUTE_TOOL, _WRITE_FILE_TOOL]


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    """Truncate text with a marker if it exceeds the limit."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n... (truncated, {len(text)} chars total)"


class AntonAgent(BaseAgent):
    """
    Harbor agent adapter that uses Anton's LLM planning loop to solve
    Terminal-Bench tasks inside a containerized environment.

    All code execution is routed through ``environment.exec()`` rather than
    Anton's local scratchpad subprocess.
    """

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            logger=logger,
            mcp_servers=mcp_servers,
            *args,
            **kwargs,
        )
        self._llm: LLMClient | None = None
        self._planning_model: str = "claude-sonnet-4-6"

        # Parse model_name (format: "anthropic/claude-sonnet-4-6" or just "claude-sonnet-4-6")
        if model_name:
            parts = model_name.split("/", maxsplit=1)
            self._planning_model = parts[-1]  # Take the model name part

    @staticmethod
    def name() -> str:
        return "anton"

    def version(self) -> str | None:
        return anton.__version__

    def _build_llm_client(self) -> LLMClient:
        """Create an LLMClient from environment variables."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is required. "
                "Set it before running the agent."
            )

        provider = AnthropicProvider(api_key=api_key)
        return LLMClient(
            planning_provider=provider,
            planning_model=self._planning_model,
            # Use the same provider for coding — not used directly but required by LLMClient
            coding_provider=provider,
            coding_model=self._planning_model,
        )

    async def setup(self, environment: BaseEnvironment) -> None:
        """Initialize the LLM client. No container-side setup needed."""
        self._llm = self._build_llm_client()
        self.logger.info(
            "AntonAgent initialized — model=%s, version=%s",
            self._planning_model,
            self.version(),
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """
        Execute the agent loop: LLM plans → tools execute in container → repeat.

        Populates ``context`` with token usage and cost as the agent runs.
        """
        if self._llm is None:
            raise RuntimeError("Agent not set up — call setup() first.")

        # Initialize tracking
        total_input_tokens = 0
        total_output_tokens = 0
        tool_rounds = 0
        start_time = time.monotonic()

        # Build conversation history
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": instruction},
        ]

        self.logger.info("Starting agent loop for task (instruction: %d chars)", len(instruction))

        while tool_rounds < _MAX_TOOL_ROUNDS:
            # --- LLM call ---
            try:
                response: LLMResponse = await self._llm.plan(
                    system=_SYSTEM_PROMPT,
                    messages=messages,
                    tools=_TOOLS,
                    temperature=0.0,
                )
            except Exception as e:
                self.logger.error("LLM call failed: %s", e)
                # Populate context with what we have so far
                self._update_context(context, total_input_tokens, total_output_tokens)
                raise

            # Track tokens
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            # Update context incrementally (so it's available even if we timeout)
            self._update_context(context, total_input_tokens, total_output_tokens)

            # --- No tool calls → agent is done ---
            if not response.tool_calls:
                # Append final assistant message
                messages.append({"role": "assistant", "content": response.content})
                self.logger.info(
                    "Agent finished after %d tool rounds (%.1fs)",
                    tool_rounds,
                    time.monotonic() - start_time,
                )
                break

            # --- Build assistant message with tool_use blocks ---
            assistant_content: list[dict[str, Any]] = []
            if response.content:
                assistant_content.append({"type": "text", "text": response.content})

            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })

            messages.append({"role": "assistant", "content": assistant_content})

            # --- Execute tools ---
            tool_results: list[dict[str, Any]] = []
            for tc in response.tool_calls:
                result_text = await self._dispatch_tool(tc, environment)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_text,
                })

            messages.append({"role": "user", "content": tool_results})
            tool_rounds += 1

            self.logger.debug(
                "Round %d complete — %d tool calls, %d input tokens, %d output tokens",
                tool_rounds,
                len(response.tool_calls),
                total_input_tokens,
                total_output_tokens,
            )
        else:
            self.logger.warning(
                "Agent hit max tool rounds (%d) — stopping.", _MAX_TOOL_ROUNDS
            )

        # Final context update
        self._update_context(context, total_input_tokens, total_output_tokens)

        elapsed = time.monotonic() - start_time
        self.logger.info(
            "Agent complete — rounds=%d, input_tokens=%d, output_tokens=%d, "
            "cost=$%.4f, elapsed=%.1fs",
            tool_rounds,
            total_input_tokens,
            total_output_tokens,
            context.cost_usd or 0,
            elapsed,
        )

        # Save conversation log
        self._save_log(messages, tool_rounds, elapsed)

    async def _dispatch_tool(
        self, tc: ToolCall, environment: BaseEnvironment
    ) -> str:
        """Route a tool call to the appropriate handler."""
        if tc.name == "execute":
            return await self._handle_execute(tc.input, environment)
        elif tc.name == "write_file":
            return await self._handle_write_file(tc.input, environment)
        else:
            return f"Unknown tool: {tc.name}"

    async def _handle_execute(
        self, tc_input: dict[str, Any], environment: BaseEnvironment
    ) -> str:
        """Execute a shell command in the container via environment.exec()."""
        command = tc_input.get("command", "")
        if not command:
            return "Error: no command provided."

        description = tc_input.get("description", "")
        timeout = tc_input.get("timeout_seconds", _DEFAULT_EXEC_TIMEOUT)
        cwd = tc_input.get("cwd")

        self.logger.info("exec [%s]: %s", description, command[:200])

        try:
            result: ExecResult = await environment.exec(
                command=command,
                cwd=cwd,
                timeout_sec=timeout,
            )
        except Exception as e:
            return f"[error] Command execution failed: {e}"

        # Format output similar to Anton's scratchpad format_cell_result
        parts: list[str] = []
        if result.stdout:
            parts.append(f"[stdout]\n{_truncate(result.stdout)}")
        if result.stderr:
            parts.append(f"[stderr]\n{_truncate(result.stderr)}")
        parts.append(f"[return_code] {result.return_code}")

        if not result.stdout and not result.stderr:
            if result.return_code == 0:
                return "Command executed successfully (no output). [return_code] 0"
            else:
                return f"Command failed with no output. [return_code] {result.return_code}"

        return "\n".join(parts)

    async def _handle_write_file(
        self, tc_input: dict[str, Any], environment: BaseEnvironment
    ) -> str:
        """Write a file in the container using a shell command."""
        path = tc_input.get("path", "")
        content = tc_input.get("content", "")
        description = tc_input.get("description", "")

        if not path:
            return "Error: no file path provided."

        self.logger.info("write_file [%s]: %s (%d bytes)", description, path, len(content))

        # Use python3 to write the file to avoid shell escaping issues
        # Base64-encode the content to handle arbitrary binary/special chars
        import base64
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")

        write_cmd = (
            f"mkdir -p $(dirname {path!r}) && "
            f"echo '{encoded}' | base64 -d > {path!r}"
        )

        try:
            result = await environment.exec(command=write_cmd, timeout_sec=30)
        except Exception as e:
            return f"[error] Failed to write file: {e}"

        if result.return_code != 0:
            stderr = result.stderr or ""
            return f"[error] Failed to write file (rc={result.return_code}): {stderr}"

        return f"File written: {path} ({len(content)} bytes)"

    def _update_context(
        self,
        context: AgentContext,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Update AgentContext with current cumulative token counts and estimated cost."""
        context.n_input_tokens = input_tokens
        context.n_output_tokens = output_tokens

        # Cost estimate based on Claude Sonnet 4 pricing
        # Input: $3/MTok, Output: $15/MTok
        input_cost = input_tokens * 3.0 / 1_000_000
        output_cost = output_tokens * 15.0 / 1_000_000
        context.cost_usd = input_cost + output_cost

    def _save_log(
        self,
        messages: list[dict[str, Any]],
        tool_rounds: int,
        elapsed_seconds: float,
    ) -> None:
        """Save the conversation log to the agent's logs directory."""
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.logs_dir / "anton_agent_log.json"
            log_data = {
                "agent": self.name(),
                "version": self.version(),
                "model": self._planning_model,
                "tool_rounds": tool_rounds,
                "elapsed_seconds": round(elapsed_seconds, 2),
                "messages": messages,
            }
            log_path.write_text(json.dumps(log_data, indent=2, default=str))
            self.logger.info("Conversation log saved to %s", log_path)
        except Exception as e:
            self.logger.warning("Failed to save conversation log: %s", e)
