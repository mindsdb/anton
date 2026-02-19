from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import anthropic

from anton.channel.base import Channel
from anton.events.types import AntonEvent, StatusUpdate, TaskComplete, TaskFailed
from anton.llm.prompts import CHAT_SYSTEM_PROMPT
from anton.llm.provider import (
    StreamComplete,
    StreamEvent,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolUseStart,
)

if TYPE_CHECKING:
    from rich.console import Console

    from anton.config.settings import AntonSettings
    from anton.llm.client import LLMClient

EXECUTE_TASK_TOOL = {
    "name": "execute_task",
    "description": (
        "Execute a coding task autonomously through Anton's agent pipeline. "
        "Call this when you have enough context to act on the user's request."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "A clear, specific description of the task to execute.",
            },
        },
        "required": ["task"],
    },
}


class _ProgressChannel(Channel):
    """Channel that captures agent events into an asyncio.Queue instead of rendering."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[AntonEvent | None] = asyncio.Queue()

    async def emit(self, event: AntonEvent) -> None:
        await self.queue.put(event)

    async def prompt(self, question: str) -> str:
        return ""

    async def close(self) -> None:
        await self.queue.put(None)


class ChatSession:
    """Manages a multi-turn conversation with tool-call delegation."""

    def __init__(self, llm_client: LLMClient, run_task, *, run_task_stream=None) -> None:
        self._llm = llm_client
        self._run_task = run_task
        self._run_task_stream = run_task_stream
        self._history: list[dict] = []

    @property
    def history(self) -> list[dict]:
        return self._history

    async def turn(self, user_input: str) -> str:
        self._history.append({"role": "user", "content": user_input})

        response = await self._llm.plan(
            system=CHAT_SYSTEM_PROMPT,
            messages=self._history,
            tools=[EXECUTE_TASK_TOOL],
        )

        # Handle tool calls (execute_task)
        while response.tool_calls:
            # Build assistant message with content blocks
            assistant_content: list[dict] = []
            if response.content:
                assistant_content.append({"type": "text", "text": response.content})
            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })
            self._history.append({"role": "assistant", "content": assistant_content})

            # Process each tool call
            tool_results: list[dict] = []
            for tc in response.tool_calls:
                if tc.name == "execute_task":
                    task_desc = tc.input.get("task", "")
                    try:
                        await self._run_task(task_desc)
                        result_text = f"Task completed: {task_desc}"
                    except Exception as exc:
                        result_text = f"Task failed: {exc}"
                else:
                    result_text = f"Unknown tool: {tc.name}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_text,
                })

            self._history.append({"role": "user", "content": tool_results})

            # Get follow-up from LLM
            response = await self._llm.plan(
                system=CHAT_SYSTEM_PROMPT,
                messages=self._history,
                tools=[EXECUTE_TASK_TOOL],
            )

        # Text-only response
        reply = response.content or ""
        self._history.append({"role": "assistant", "content": reply})
        return reply

    async def turn_stream(self, user_input: str) -> AsyncIterator[StreamEvent]:
        """Streaming version of turn(). Yields events as they arrive."""
        self._history.append({"role": "user", "content": user_input})

        async for event in self._stream_and_handle_tools():
            yield event

    async def _stream_and_handle_tools(self) -> AsyncIterator[StreamEvent]:
        """Stream one LLM call, handle tool loops, yield all events."""
        response: StreamComplete | None = None

        async for event in self._llm.plan_stream(
            system=CHAT_SYSTEM_PROMPT,
            messages=self._history,
            tools=[EXECUTE_TASK_TOOL],
        ):
            yield event
            if isinstance(event, StreamComplete):
                response = event

        if response is None:
            return

        llm_response = response.response

        # Tool-call loop
        while llm_response.tool_calls:
            # Build assistant message with content blocks
            assistant_content: list[dict] = []
            if llm_response.content:
                assistant_content.append({"type": "text", "text": llm_response.content})
            for tc in llm_response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })
            self._history.append({"role": "assistant", "content": assistant_content})

            # Process each tool call
            tool_results: list[dict] = []
            for tc in llm_response.tool_calls:
                if tc.name == "execute_task":
                    task_desc = tc.input.get("task", "")
                    try:
                        if self._run_task_stream is not None:
                            async for progress in self._run_task_stream(task_desc):
                                yield progress
                        else:
                            await self._run_task(task_desc)
                        result_text = f"Task completed: {task_desc}"
                    except Exception as exc:
                        result_text = f"Task failed: {exc}"
                else:
                    result_text = f"Unknown tool: {tc.name}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_text,
                })

            self._history.append({"role": "user", "content": tool_results})

            # Stream follow-up
            response = None
            async for event in self._llm.plan_stream(
                system=CHAT_SYSTEM_PROMPT,
                messages=self._history,
                tools=[EXECUTE_TASK_TOOL],
            ):
                yield event
                if isinstance(event, StreamComplete):
                    response = event

            if response is None:
                return
            llm_response = response.response

        # Text-only final response — append to history
        reply = llm_response.content or ""
        self._history.append({"role": "assistant", "content": reply})


def run_chat(console: Console, settings: AntonSettings) -> None:
    """Launch the interactive chat REPL."""
    asyncio.run(_chat_loop(console, settings))


async def _chat_loop(console: Console, settings: AntonSettings) -> None:
    from pathlib import Path

    from anton.channel.terminal import CLIChannel
    from anton.core.agent import Agent
    from anton.llm.client import LLMClient
    from anton.skill.registry import SkillRegistry

    # Use a mutable container so closures always see the current client
    state: dict = {"llm_client": LLMClient.from_settings(settings)}
    registry = SkillRegistry()

    builtin = Path(__file__).resolve().parent.parent / settings.skills_dir
    registry.discover(builtin)

    user_dir = Path(settings.user_skills_dir).expanduser()
    registry.discover(user_dir)

    memory = None
    learnings_store = None
    if settings.memory_enabled:
        from anton.memory.learnings import LearningStore
        from anton.memory.store import SessionStore

        memory_dir = Path(settings.memory_dir).expanduser()
        memory = SessionStore(memory_dir)
        learnings_store = LearningStore(memory_dir)

    channel = CLIChannel()

    async def _do_run_task(task: str) -> None:
        agent = Agent(
            channel=channel,
            llm_client=state["llm_client"],
            registry=registry,
            user_skills_dir=user_dir,
            memory=memory,
            learnings=learnings_store,
        )
        await agent.run(task)

    async def _do_run_task_stream(task: str) -> AsyncIterator[StreamTaskProgress]:
        """Run agent task, yielding progress events as StreamTaskProgress."""
        progress_ch = _ProgressChannel()
        agent = Agent(
            channel=progress_ch,
            llm_client=state["llm_client"],
            registry=registry,
            user_skills_dir=user_dir,
            memory=memory,
            learnings=learnings_store,
        )
        agent_task = asyncio.create_task(agent.run(task))

        try:
            while True:
                try:
                    event = await asyncio.wait_for(progress_ch.queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    if agent_task.done():
                        break
                    continue

                if event is None:
                    break

                if isinstance(event, StatusUpdate):
                    yield StreamTaskProgress(
                        phase=event.phase.value,
                        message=event.message,
                        eta_seconds=event.eta_seconds,
                    )
                elif isinstance(event, (TaskComplete, TaskFailed)):
                    break

            # Drain remaining events
            while not progress_ch.queue.empty():
                event = progress_ch.queue.get_nowait()
                if isinstance(event, StatusUpdate):
                    yield StreamTaskProgress(
                        phase=event.phase.value,
                        message=event.message,
                        eta_seconds=event.eta_seconds,
                    )

            # Re-raise any exception from the agent
            await agent_task
        except BaseException:
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except (asyncio.CancelledError, Exception):
                    pass
            raise

    session = ChatSession(
        state["llm_client"], _do_run_task, run_task_stream=_do_run_task_stream
    )

    console.print("[anton.muted]Chat with Anton. Type 'exit' to quit.[/]")
    console.print()

    from anton.chat_ui import StreamDisplay

    display = StreamDisplay(console)

    try:
        while True:
            try:
                user_input = console.input("[bold]you>[/] ")
            except EOFError:
                break

            stripped = user_input.strip()
            if not stripped:
                continue
            if stripped.lower() in ("exit", "quit", "bye"):
                break

            display.start()
            t0 = time.monotonic()
            ttft: float | None = None
            total_input = 0
            total_output = 0

            try:
                async for event in session.turn_stream(stripped):
                    if isinstance(event, StreamTextDelta):
                        if ttft is None:
                            ttft = time.monotonic() - t0
                        display.append_text(event.text)
                    elif isinstance(event, StreamToolUseStart):
                        display.show_tool_execution(event.name)
                    elif isinstance(event, StreamTaskProgress):
                        display.update_progress(
                            event.phase, event.message, event.eta_seconds
                        )
                    elif isinstance(event, StreamComplete):
                        total_input += event.response.usage.input_tokens
                        total_output += event.response.usage.output_tokens

                elapsed = time.monotonic() - t0
                display.finish(total_input, total_output, elapsed, ttft)
            except anthropic.AuthenticationError:
                display.abort()
                console.print()
                console.print(
                    "[anton.error]Invalid API key. Let's set up a new one.[/]"
                )
                settings.anthropic_api_key = None
                from anton.cli import _ensure_api_key
                _ensure_api_key(settings)
                state["llm_client"] = LLMClient.from_settings(settings)
                session = ChatSession(
                    state["llm_client"], _do_run_task,
                    run_task_stream=_do_run_task_stream,
                )
            except KeyboardInterrupt:
                display.abort()
                console.print()
                break
            except Exception as exc:
                display.abort()
                console.print(f"[anton.error]Error: {exc}[/]")
                console.print()
    except KeyboardInterrupt:
        pass

    console.print()
    console.print("[anton.muted]See you.[/]")
    await channel.close()
