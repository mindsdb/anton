from __future__ import annotations

import asyncio
import json as _json
import os
import urllib.error
import re as _re
import sys
import uuid
import yaml as _yaml
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TYPE_CHECKING

import anthropic

from anton.clipboard import (
    cleanup_old_uploads,
    clipboard_unavailable_reason,
    grab_clipboard,
    is_clipboard_supported,
    parse_dropped_paths as _parse_dropped_paths,
    save_clipboard_image,
)
from anton.llm.prompts import CHAT_SYSTEM_PROMPT, build_visualizations_prompt
from anton.llm.provider import (
    ContextOverflowError,
    StreamComplete,
    StreamContextCompacted,
    StreamEvent,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolResult,
    StreamToolUseDelta,
    StreamToolUseEnd,
    StreamToolUseStart,
)
from anton.scratchpad import ScratchpadManager
from anton.tools import (
    CONNECT_DATASOURCE_TOOL,
    MEMORIZE_TOOL,
    RECALL_TOOL,
    SCRATCHPAD_TOOL,
    dispatch_tool,
    format_cell_result,
    prepare_scratchpad_exec,
)
from anton.checks import TokenLimitInfo, TokenLimitStatus, check_minds_token_limits
from anton.prompt_utils import (
    MINDS_KEYS,
    LLM_KEYS,
    SECRET_PATTERNS,
    mask_secret,
    is_secret_key,
    display_value,
    prompt_or_cancel,
    prompt_minds_api_key,
)
from anton.minds_client import (
    normalize_minds_url,
    describe_minds_connection_error,
    list_minds,
    get_mind,
    refresh_knowledge,
    list_datasources,
    test_llm,
)
from anton.data_vault import DataVault, _slug_env_prefix
from anton.datasource_registry import (
    DatasourceEngine,
    DatasourceField,
    DatasourceRegistry,
    _YAML_BLOCK_RE,
)
from anton.llm.openai import build_chat_completion_kwargs

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PTStyle
from rich.prompt import Confirm, Prompt

if TYPE_CHECKING:
    from rich.console import Console

    from anton.config.settings import AntonSettings
    from anton.context.self_awareness import SelfAwarenessContext
    from anton.llm.client import LLMClient
    from anton.memory.cortex import Cortex
    from anton.memory.episodes import EpisodicMemory
    from anton.memory.history_store import HistoryStore
    from anton.workspace import Workspace


_MAX_TOOL_ROUNDS = 25  # Hard limit on consecutive tool-call rounds per turn
_MAX_CONTINUATIONS = 3  # Max times the verification loop can restart the tool loop
_CONTEXT_PRESSURE_THRESHOLD = 0.7  # Trigger compaction when context is 70% full
_MAX_CONSECUTIVE_ERRORS = 5  # Stop if the same tool fails this many times in a row
_RESILIENCE_NUDGE_AT = 2  # Inject resilience nudge after this many consecutive errors
_RESILIENCE_NUDGE = (
    "\n\nSYSTEM: This tool has failed twice in a row. Before retrying the same approach or "
    "asking the user for help, try a creative workaround — different headers/user-agent, "
    "a public API, archive.org, an alternate library, or a completely different data source. "
    "Only involve the user if the problem truly requires something only they can provide."
)

# TODO: Is this enough for now?
TOKEN_STATUS_CACHE_TTL = 60.0

_PROMPT_RECONNECT_CANCEL = "(reconnect/cancel)"


class ChatSession:
    """Manages a multi-turn conversation with tool-call delegation."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        self_awareness: SelfAwarenessContext | None = None,
        cortex: Cortex | None = None,
        episodic: EpisodicMemory | None = None,
        runtime_context: str = "",
        workspace: Workspace | None = None,
        console: Console | None = None,
        coding_provider: str = "anthropic",
        coding_api_key: str = "",
        coding_base_url: str = "",
        initial_history: list[dict] | None = None,
        history_store: HistoryStore | None = None,
        session_id: str | None = None,
        proactive_dashboards: bool = False,
    ) -> None:
        self._llm = llm_client
        self._self_awareness = self_awareness
        self._cortex = cortex
        self._episodic = episodic
        self._runtime_context = runtime_context
        self._proactive_dashboards = proactive_dashboards
        self._workspace = workspace
        self._console = console
        self._history: list[dict] = list(initial_history) if initial_history else []
        self._pending_memory_confirmations: list = []
        self._turn_count = (
            sum(1 for m in self._history if m.get("role") == "user")
            if initial_history
            else 0
        )
        self._history_store = history_store
        self._session_id = session_id
        self._cancel_event = asyncio.Event()
        self._escape_watcher: _EscapeWatcher | None = None
        self._active_datasource: str | None = None
        self._scratchpads = ScratchpadManager(
            coding_provider=coding_provider,
            coding_model=getattr(llm_client, "coding_model", ""),
            coding_api_key=coding_api_key,
            coding_base_url=coding_base_url,
            workspace_path=workspace.base if workspace else None,
        )

    @property
    def history(self) -> list[dict]:
        return self._history

    def repair_history(self) -> None:
        """Fix dangling tool_use blocks left by mid-stream cancellation.

        The Anthropic API requires every tool_use to be followed by a
        tool_result.  If we cancelled mid-turn, the last assistant message
        may contain tool_use blocks with no corresponding tool_result in
        the next message.  Append synthetic tool_results so the
        conversation can continue.
        """
        if not self._history:
            return
        last = self._history[-1]
        if last.get("role") != "assistant":
            return
        content = last.get("content")
        if not isinstance(content, list):
            return
        tool_ids = [
            block["id"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        if not tool_ids:
            return
        self._history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": "Cancelled by user.",
                    }
                    for tid in tool_ids
                ],
            }
        )

    def _persist_history(self) -> None:
        """Save current history to disk if a history store is configured."""
        if self._history_store and self._session_id:
            self._history_store.save(self._session_id, self._history)

    async def _build_system_prompt(self, user_message: str = "") -> str:
        import datetime as _dt
        _now = _dt.datetime.now()
        _current_datetime = _now.strftime("%A, %B %d, %Y at %I:%M %p")

        prompt = CHAT_SYSTEM_PROMPT.format(
            runtime_context=self._runtime_context,
            visualizations_section=build_visualizations_prompt(
                self._proactive_dashboards
            ),
            current_datetime=_current_datetime,
        )
        # Inject memory context (replaces old self_awareness)
        if self._cortex is not None:
            memory_section = await self._cortex.build_memory_context(user_message)
            if memory_section:
                prompt += memory_section
        elif self._self_awareness is not None:
            # Fallback for legacy usage (tests, etc.)
            sa_section = self._self_awareness.build_prompt_section()
            if sa_section:
                prompt += sa_section
        # Inject anton.md project context (user-written takes priority)
        if self._workspace is not None:
            md_context = self._workspace.build_anton_md_context()
            if md_context:
                prompt += md_context
        # Inject connected datasource context without credentials
        ds_ctx = _build_datasource_context(active_only=self._active_datasource)
        if ds_ctx:
            prompt += ds_ctx
        return prompt

    # Packages the LLM is most likely to care about when writing scratchpad code.
    _NOTABLE_PACKAGES: set[str] = {
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "requests",
        "httpx",
        "aiohttp",
        "beautifulsoup4",
        "lxml",
        "pillow",
        "sympy",
        "networkx",
        "sqlalchemy",
        "pydantic",
        "rich",
        "tqdm",
        "click",
        "fastapi",
        "flask",
        "django",
        "openai",
        "anthropic",
        "tiktoken",
        "transformers",
        "torch",
        "polars",
        "pyarrow",
        "openpyxl",
        "xlsxwriter",
        "plotly",
        "bokeh",
        "altair",
        "pytest",
        "hypothesis",
        "yaml",
        "pyyaml",
        "toml",
        "tomli",
        "tomllib",
        "jinja2",
        "markdown",
        "pygments",
        "cryptography",
        "paramiko",
        "boto3",
    }

    def _build_tools(self) -> list[dict]:
        scratchpad_tool = dict(SCRATCHPAD_TOOL)
        pkg_list = self._scratchpads._available_packages
        if pkg_list:
            notable = sorted(p for p in pkg_list if p.lower() in self._NOTABLE_PACKAGES)
            if notable:
                pkg_line = ", ".join(notable)
                extra = f"\n\nInstalled packages ({len(pkg_list)} total, notable: {pkg_line})."
            else:
                extra = f"\n\nInstalled packages: {len(pkg_list)} total (standard library plus dependencies)."
            scratchpad_tool["description"] = SCRATCHPAD_TOOL["description"] + extra

        # Inject scratchpad wisdom from memory (procedural priming)
        if self._cortex is not None:
            wisdom = self._cortex.get_scratchpad_context()
            if wisdom:
                scratchpad_tool[
                    "description"
                ] += f"\n\nLessons from past sessions:\n{wisdom}"

        tools = [scratchpad_tool]
        if self._cortex is not None:
            tools.append(MEMORIZE_TOOL)
        elif self._self_awareness is not None:
            # Legacy fallback
            from anton.tools import MEMORIZE_TOOL as _MT

            tools.append(_MT)
        if self._episodic is not None and self._episodic.enabled:
            tools.append(RECALL_TOOL)
        tools.append(CONNECT_DATASOURCE_TOOL)
        return tools

    async def close(self) -> None:
        """Clean up scratchpads and other resources."""
        await self._scratchpads.close_all()

    async def _summarize_history(self) -> None:
        """Compress old conversation turns into a summary using the coding model.

        Splits history into old (first 60%) and recent (last 40%), keeping at
        least 4 recent turns.  The old portion is summarized by the fast coding
        model and replaced with a single user message.
        """
        if len(self._history) < 6:
            return  # Too short to summarize

        min_recent = 4
        split = max(int(len(self._history) * 0.6), 1)
        # Ensure we keep at least min_recent turns
        split = min(split, len(self._history) - min_recent)
        if split < 2:
            return

        # Walk split backward to avoid breaking tool_use / tool_result pairs.
        # A user message containing tool_result blocks must stay with the
        # preceding assistant message that contains the matching tool_use.
        while split > 1:
            msg = self._history[split]
            if msg.get("role") != "user":
                break
            content = msg.get("content")
            if not isinstance(content, list):
                break
            has_tool_result = any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            )
            if not has_tool_result:
                break
            # This user message has tool_results — keep it (and its paired
            # assistant message) in the recent portion.
            split -= 1
            # Also pull back over the preceding assistant message so the
            # pair stays together.
            if split > 1 and self._history[split].get("role") == "assistant":
                split -= 1

        if split < 2:
            return

        old_turns = self._history[:split]
        recent_turns = self._history[split:]

        # Serialize old turns into text for summarization
        lines: list[str] = []
        for msg in old_turns:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                lines.append(f"[{role}]: {content[:2000]}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            lines.append(f"[{role}]: {block['text'][:1000]}")
                        elif block.get("type") == "tool_use":
                            lines.append(
                                f"[{role}/tool_use]: {block.get('name', '')}({str(block.get('input', ''))[:500]})"
                            )
                        elif block.get("type") == "tool_result":
                            lines.append(
                                f"[tool_result]: {str(block.get('content', ''))[:500]}"
                            )

        old_text = "\n".join(lines)
        # Cap at ~8000 chars to avoid overloading the summarizer
        if len(old_text) > 8000:
            old_text = old_text[:8000] + "\n... (truncated)"

        try:
            summary_response = await self._llm.code(
                system=(
                    "Summarize this conversation history concisely. Preserve:\n"
                    "- Key decisions and conclusions\n"
                    "- Important data/results discovered\n"
                    "- Variable names and values that are still relevant\n"
                    "- Errors encountered and how they were resolved\n"
                    "Keep it under 2000 tokens. Use bullet points."
                ),
                messages=[{"role": "user", "content": old_text}],
                max_tokens=2048,
            )
            summary = summary_response.content or "(summary unavailable)"
        except Exception:
            # If summarization fails, just do a simple truncation
            summary = f"(Earlier conversation with {len(old_turns)} turns — summarization failed)"

        summary_msg = {
            "role": "user",
            "content": f"[Context summary of earlier conversation]\n{summary}",
        }

        # If the recent portion starts with a user message, insert a minimal
        # assistant separator to avoid consecutive user messages (API error).
        if recent_turns and recent_turns[0].get("role") == "user":
            self._history = [
                summary_msg,
                {"role": "assistant", "content": "Understood."},
                *recent_turns,
            ]
        else:
            self._history = [summary_msg] + recent_turns

    def _compact_scratchpads(self) -> bool:
        """Compact all active scratchpads. Returns True if any were compacted."""
        compacted = False
        for pad in self._scratchpads._pads.values():
            if pad._compact_cells():
                compacted = True
        return compacted

    async def turn(self, user_input: str | list[dict]) -> str:
        self._history.append({"role": "user", "content": user_input})

        user_msg_str = user_input if isinstance(user_input, str) else ""
        system = await self._build_system_prompt(user_msg_str)
        tools = self._build_tools()

        try:
            response = await self._llm.plan(
                system=system,
                messages=self._history,
                tools=tools,
            )
        except ContextOverflowError:
            await self._summarize_history()
            self._compact_scratchpads()
            response = await self._llm.plan(
                system=system,
                messages=self._history,
                tools=tools,
            )

        # Proactive compaction
        if response.usage.context_pressure > _CONTEXT_PRESSURE_THRESHOLD:
            await self._summarize_history()
            self._compact_scratchpads()

        # Handle tool calls
        tool_round = 0
        error_streak: dict[str, int] = {}
        resilience_nudged: set[str] = set()

        while response.tool_calls:
            tool_round += 1
            if tool_round > _MAX_TOOL_ROUNDS:
                self._history.append(
                    {"role": "assistant", "content": response.content or ""}
                )
                self._history.append(
                    {
                        "role": "user",
                        "content": (
                            f"SYSTEM: You have used {_MAX_TOOL_ROUNDS} tool-call rounds on this turn. "
                            "Pause here. Summarize what you have accomplished so far and what remains. "
                            "If you believe you are on a good track and can finish the task with more steps, "
                            "tell the user and ask if they'd like you to continue. "
                            "Do NOT retry automatically — wait for the user's response."
                        ),
                    }
                )
                response = await self._llm.plan(
                    system=system,
                    messages=self._history,
                )
                break

            # Build assistant message with content blocks
            assistant_content: list[dict] = []
            if response.content:
                assistant_content.append({"type": "text", "text": response.content})
            for tc in response.tool_calls:
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    }
                )
            self._history.append({"role": "assistant", "content": assistant_content})

            # Process each tool call via registry
            tool_results: list[dict] = []
            for tc in response.tool_calls:
                try:
                    result_text = await dispatch_tool(self, tc.name, tc.input)
                except Exception as exc:
                    result_text = f"Tool '{tc.name}' failed: {exc}"

                result_text = _scrub_credentials(result_text)
                result_text = _apply_error_tracking(
                    result_text,
                    tc.name,
                    error_streak,
                    resilience_nudged,
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result_text,
                    }
                )

            self._history.append({"role": "user", "content": tool_results})

            # Get follow-up from LLM
            try:
                response = await self._llm.plan(
                    system=system,
                    messages=self._history,
                    tools=tools,
                )
            except ContextOverflowError:
                await self._summarize_history()
                self._compact_scratchpads()
                response = await self._llm.plan(
                    system=system,
                    messages=self._history,
                    tools=tools,
                )

            # Proactive compaction during tool loop
            if response.usage.context_pressure > _CONTEXT_PRESSURE_THRESHOLD:
                await self._summarize_history()
                self._compact_scratchpads()

        # Text-only response
        reply = response.content or ""
        self._history.append({"role": "assistant", "content": reply})

        # Periodic memory vacuum (Systems Consolidation)
        if self._cortex is not None and self._cortex.mode != "off":
            self._cortex.maybe_vacuum()

        return reply

    async def turn_stream(
        self, user_input: str | list[dict]
    ) -> AsyncIterator[StreamEvent]:
        """Streaming version of turn(). Yields events as they arrive."""
        self._history.append({"role": "user", "content": user_input})

        # Log user input to episodic memory
        if self._episodic is not None:
            content = (
                user_input if isinstance(user_input, str) else str(user_input)[:2000]
            )
            self._episodic.log_turn(self._turn_count + 1, "user", content)

        user_msg_str = user_input if isinstance(user_input, str) else ""
        assistant_text_parts: list[str] = []
        _max_auto_retries = 2
        _retry_count = 0

        while True:
            try:
                async for event in self._stream_and_handle_tools(user_msg_str):
                    if isinstance(event, StreamTextDelta):
                        assistant_text_parts.append(event.text)
                    yield event
                break  # completed successfully
            except Exception as _agent_exc:
                _retry_count += 1
                if _retry_count <= _max_auto_retries:
                    # Inject the error into history and let the LLM try to recover
                    self._history.append(
                        {
                            "role": "user",
                            "content": (
                                f"SYSTEM: An error interrupted execution: {_agent_exc}\n\n"
                                "If you can diagnose and fix the issue, continue working on the task. "
                                "Adjust your approach to avoid the same error. "
                                "If this is unrecoverable, summarize what you accomplished and suggest next steps."
                            ),
                        }
                    )
                    # Continue the while loop — _stream_and_handle_tools will be called
                    # again with the error context now in history
                    continue
                else:
                    # Exhausted retries — stop and summarize for the user
                    self._history.append(
                        {
                            "role": "user",
                            "content": (
                                f"SYSTEM: The task has failed {_retry_count} times. Latest error: {_agent_exc}\n\n"
                                "Stop retrying. Please:\n"
                                "1. Summarize what you accomplished so far.\n"
                                "2. Explain what went wrong in plain language.\n"
                                "3. Suggest next steps — what the user can try (e.g. rephrase, "
                                "simplify the request, or ask you to continue from where you left off).\n"
                                "Be concise and helpful."
                            ),
                        }
                    )
                    try:
                        async for event in self._llm.plan_stream(
                            system=await self._build_system_prompt(user_msg_str),
                            messages=self._history,
                        ):
                            if isinstance(event, StreamTextDelta):
                                assistant_text_parts.append(event.text)
                            yield event
                    except Exception:
                        fallback = f"An unexpected error occurred: {_agent_exc}. Please try again or rephrase your request."
                        assistant_text_parts.append(fallback)
                        yield StreamTextDelta(text=fallback)
                    break

        # Log assistant response to episodic memory
        if self._episodic is not None and assistant_text_parts:
            self._episodic.log_turn(
                self._turn_count + 1,
                "assistant",
                "".join(assistant_text_parts)[:2000],
            )

        # Identity extraction (Default Mode Network — every 5 turns)
        self._turn_count += 1
        self._persist_history()
        if self._cortex is not None and self._cortex.mode != "off":
            if self._turn_count % 5 == 0 and isinstance(user_input, str):
                asyncio.create_task(self._cortex.maybe_update_identity(user_input))
            # Periodic memory vacuum (Systems Consolidation)
            self._cortex.maybe_vacuum()

    async def _stream_and_handle_tools(
        self, user_message: str = ""
    ) -> AsyncIterator[StreamEvent]:
        """Stream one LLM call, handle tool loops, yield all events."""
        system = await self._build_system_prompt(user_message)
        tools = self._build_tools()

        # Guard against summarizing an already-summarized history within the same
        # turn (e.g. ContextOverflowError on first call + pressure > threshold on
        # the tool-loop follow-up would previously produce a summary of a summary).
        _compacted_this_turn = False

        response: StreamComplete | None = None

        try:
            async for event in self._llm.plan_stream(
                system=system,
                messages=self._history,
                tools=tools,
            ):
                yield event
                if isinstance(event, StreamComplete):
                    response = event
        except ContextOverflowError:
            await self._summarize_history()
            self._compact_scratchpads()
            _compacted_this_turn = True
            yield StreamContextCompacted(
                message="Context was getting long — older history has been summarized."
            )
            async for event in self._llm.plan_stream(
                system=system,
                messages=self._history,
                tools=tools,
            ):
                yield event
                if isinstance(event, StreamComplete):
                    response = event

        if response is None:
            return

        llm_response = response.response

        # Detect max_tokens truncation — the LLM was cut off mid-response.
        # Inject a continuation prompt so it can finish what it was doing.
        if llm_response.stop_reason in ("max_tokens", "length") and not llm_response.tool_calls:
            self._history.append(
                {"role": "assistant", "content": llm_response.content or ""}
            )
            self._history.append(
                {
                    "role": "user",
                    "content": (
                        "SYSTEM: Your response was truncated because it exceeded the output token limit. "
                        "Continue exactly where you left off. If you were about to call a tool, "
                        "call it now. If the code you were writing was too long, split it into smaller parts."
                    ),
                }
            )
            response = None
            try:
                async for event in self._llm.plan_stream(
                    system=system,
                    messages=self._history,
                    tools=tools,
                ):
                    yield event
                    if isinstance(event, StreamComplete):
                        response = event
            except ContextOverflowError:
                if not _compacted_this_turn:
                    await self._summarize_history()
                    self._compact_scratchpads()
                    _compacted_this_turn = True
                yield StreamContextCompacted(
                    message="Context was getting long — older history has been summarized."
                )
                async for event in self._llm.plan_stream(
                    system=system,
                    messages=self._history,
                    tools=tools,
                ):
                    yield event
                    if isinstance(event, StreamComplete):
                        response = event

            if response is None:
                return
            llm_response = response.response

        # Proactive compaction
        if (
            not _compacted_this_turn
            and llm_response.usage.context_pressure > _CONTEXT_PRESSURE_THRESHOLD
        ):
            await self._summarize_history()
            self._compact_scratchpads()
            _compacted_this_turn = True
            yield StreamContextCompacted(
                message="Context was getting long — older history has been summarized."
            )

        # Tool-call loop with circuit breaker, wrapped in a completion
        # verification outer loop that can restart the tool loop if the
        # task isn't actually done yet.
        continuation = 0
        _max_rounds_hit = False

        while True:  # Completion verification loop
            tool_round = 0
            error_streak: dict[str, int] = {}
            resilience_nudged: set[str] = set()

            while llm_response.tool_calls:
                tool_round += 1
                if tool_round > _MAX_TOOL_ROUNDS:
                    _max_rounds_hit = True
                    self._history.append(
                        {"role": "assistant", "content": llm_response.content or ""}
                    )
                    self._history.append(
                        {
                            "role": "user",
                            "content": (
                                f"SYSTEM: You have used {_MAX_TOOL_ROUNDS} tool-call rounds on this turn. "
                                "Pause here. Summarize what you have accomplished so far and what remains. "
                                "If you believe you are on a good track and can finish the task with more steps, "
                                "tell the user and ask if they'd like you to continue. "
                                "Do NOT retry automatically — wait for the user's response."
                            ),
                        }
                    )
                    async for event in self._llm.plan_stream(
                        system=system,
                        messages=self._history,
                    ):
                        yield event
                    break

                # Build assistant message with content blocks
                assistant_content: list[dict] = []
                if llm_response.content:
                    assistant_content.append(
                        {"type": "text", "text": llm_response.content}
                    )
                for tc in llm_response.tool_calls:
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.input,
                        }
                    )
                self._history.append(
                    {"role": "assistant", "content": assistant_content}
                )

                # Process each tool call
                tool_results: list[dict] = []
                for tc in llm_response.tool_calls:
                    if self._episodic is not None:
                        self._episodic.log_turn(
                            self._turn_count + 1,
                            "tool_call",
                            str(tc.input)[:2000],
                            tool=tc.name,
                        )

                    try:
                        if tc.name == "scratchpad" and tc.input.get("action") == "exec":
                            # Inline streaming exec — yields progress events
                            prep = await prepare_scratchpad_exec(self, tc.input)
                            if isinstance(prep, str):
                                result_text = prep
                            else:
                                (
                                    pad,
                                    code,
                                    description,
                                    estimated_time,
                                    estimated_seconds,
                                ) = prep
                                yield StreamTaskProgress(
                                    phase="scratchpad_start",
                                    message=description or "Running code",
                                    eta_seconds=estimated_seconds,
                                )
                                import time as _time

                                _sp_t0 = _time.monotonic()
                                from anton.scratchpad import Cell

                                cell = None
                                async for item in pad.execute_streaming(
                                    code,
                                    description=description,
                                    estimated_time=estimated_time,
                                    estimated_seconds=estimated_seconds,
                                    cancel_event=self._cancel_event,
                                ):
                                    if isinstance(item, str):
                                        yield StreamTaskProgress(
                                            phase="scratchpad", message=item
                                        )
                                    elif isinstance(item, Cell):
                                        cell = item
                                _sp_elapsed = _time.monotonic() - _sp_t0
                                yield StreamTaskProgress(
                                    phase="scratchpad_done",
                                    message=description or "Done",
                                    eta_seconds=_sp_elapsed,
                                )
                                result_text = (
                                    format_cell_result(cell)
                                    if cell
                                    else "No result produced."
                                )
                                if self._episodic is not None and cell is not None:
                                    self._episodic.log_turn(
                                        self._turn_count + 1,
                                        "scratchpad",
                                        (cell.stdout or "")[:2000],
                                        description=description,
                                    )
                        elif tc.name == "connect_new_datasource":
                            # Interactive tool — pause spinner AND escape watcher
                            yield StreamTaskProgress(
                                phase="connect_datasource",
                                message="Connecting datasource...",
                            )
                            if self._escape_watcher:
                                self._escape_watcher.pause()
                            result_text = await dispatch_tool(self, tc.name, tc.input)
                            if self._escape_watcher:
                                self._escape_watcher.resume()
                            yield StreamTaskProgress(
                                phase="analyzing",
                                message="Analyzing results...",
                            )
                        else:
                            result_text = await dispatch_tool(self, tc.name, tc.input)
                            if (
                                tc.name == "scratchpad"
                                and tc.input.get("action") == "dump"
                            ):
                                yield StreamToolResult(content=result_text)
                                result_text = (
                                    "The full notebook has been displayed to the user above. "
                                    "Do not repeat it. Here is the content for your reference:\n\n"
                                    + result_text
                                )
                    except Exception as exc:
                        result_text = f"Tool '{tc.name}' failed: {exc}"

                    if self._episodic is not None:
                        self._episodic.log_turn(
                            self._turn_count + 1,
                            "tool_result",
                            result_text[:2000],
                            tool=tc.name,
                        )
                    result_text = _scrub_credentials(result_text)
                    result_text = _apply_error_tracking(
                        result_text, tc.name, error_streak, resilience_nudged
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": result_text,
                        }
                    )

                self._history.append({"role": "user", "content": tool_results})

                # Signal that tools are done and LLM is now analyzing
                yield StreamTaskProgress(
                    phase="analyzing", message="Analyzing results..."
                )

                # Stream follow-up
                response = None
                try:
                    async for event in self._llm.plan_stream(
                        system=system,
                        messages=self._history,
                        tools=tools,
                    ):
                        yield event
                        if isinstance(event, StreamComplete):
                            response = event
                except ContextOverflowError:
                    if not _compacted_this_turn:
                        await self._summarize_history()
                        self._compact_scratchpads()
                        _compacted_this_turn = True
                    yield StreamContextCompacted(
                        message="Context was getting long — older history has been summarized."
                    )
                    async for event in self._llm.plan_stream(
                        system=system,
                        messages=self._history,
                        tools=tools,
                    ):
                        yield event
                        if isinstance(event, StreamComplete):
                            response = event

                if response is None:
                    return
                llm_response = response.response

                # Detect max_tokens truncation inside tool loop
                if llm_response.stop_reason in ("max_tokens", "length") and not llm_response.tool_calls:
                    self._history.append(
                        {"role": "assistant", "content": llm_response.content or ""}
                    )
                    self._history.append(
                        {
                            "role": "user",
                            "content": (
                                "SYSTEM: Your response was truncated because it exceeded the output token limit. "
                                "Continue exactly where you left off. If you were about to call a tool, "
                                "call it now. If the code you were writing was too long, split it into smaller parts."
                            ),
                        }
                    )
                    response = None
                    try:
                        async for event in self._llm.plan_stream(
                            system=system,
                            messages=self._history,
                            tools=tools,
                        ):
                            yield event
                            if isinstance(event, StreamComplete):
                                response = event
                    except ContextOverflowError:
                        if not _compacted_this_turn:
                            await self._summarize_history()
                            self._compact_scratchpads()
                            _compacted_this_turn = True
                        yield StreamContextCompacted(
                            message="Context was getting long — older history has been summarized."
                        )
                        async for event in self._llm.plan_stream(
                            system=system,
                            messages=self._history,
                            tools=tools,
                        ):
                            yield event
                            if isinstance(event, StreamComplete):
                                response = event

                    if response is None:
                        return
                    llm_response = response.response

                # Proactive compaction during tool loop
                if (
                    not _compacted_this_turn
                    and llm_response.usage.context_pressure
                    > _CONTEXT_PRESSURE_THRESHOLD
                ):
                    await self._summarize_history()
                    self._compact_scratchpads()
                    _compacted_this_turn = True
                    yield StreamContextCompacted(
                        message="Context was getting long — older history has been summarized."
                    )

            # --- Completion verification ---
            # Only verify when tools were actually used (not for simple Q&A)
            # and we haven't hit the max-rounds hard stop.
            if tool_round == 0 or _max_rounds_hit:
                break

            # Append the assistant's final text so the verifier can see it
            reply = llm_response.content or ""
            self._history.append({"role": "assistant", "content": reply})

            if continuation >= _MAX_CONTINUATIONS:
                # Budget exhausted — ask LLM to diagnose and present to user
                self._history.append(
                    {
                        "role": "user",
                        "content": (
                            "SYSTEM: You have attempted to complete this task multiple times "
                            "but verification indicates it is still not done. Do NOT try again. "
                            "Instead:\n"
                            "1. Summarize exactly what was accomplished so far.\n"
                            "2. Identify the specific blocker or failure preventing completion.\n"
                            "3. Suggest concrete next steps the user can take to unblock this.\n"
                            "Be honest and specific — do not be vague about what went wrong."
                        ),
                    }
                )
                yield StreamTaskProgress(
                    phase="analyzing", message="Diagnosing incomplete task..."
                )
                async for event in self._llm.plan_stream(
                    system=system,
                    messages=self._history,
                ):
                    yield event
                # Consolidation still runs after diagnosis
                break

            # Ask the LLM to self-assess completion.
            # Use a copy of history with a trailing user message so models
            # that don't support assistant-prefill won't reject the request.
            verify_messages = list(self._history) + [
                {
                    "role": "user",
                    "content": (
                        "SYSTEM: Evaluate whether the task the user originally requested "
                        "has been fully completed based on the conversation above."
                    ),
                }
            ]
            verification = await self._llm.plan(
                system=(
                    "You are a task-completion verifier. Given the conversation, determine "
                    "whether the user's original request has been fully completed.\n\n"
                    "Respond with EXACTLY one of these lines, followed by a brief reason:\n"
                    "STATUS: COMPLETE — <reason>\n"
                    "STATUS: INCOMPLETE — <reason>\n"
                    "STATUS: STUCK — <reason>\n\n"
                    "COMPLETE = the task is done or the response fully answers the question.\n"
                    "INCOMPLETE = more work can be done to finish the task.\n"
                    "STUCK = a blocker prevents completion (missing info, permissions, etc).\n\n"
                    "Be strict: if the user asked for X and only part of X was delivered, "
                    "that is INCOMPLETE, not COMPLETE. But if the user asked a question "
                    "and the assistant answered it, that is COMPLETE even without tool use."
                ),
                messages=verify_messages,
                max_tokens=256,
            )

            status_text = (verification.content or "").strip().upper()
            if "STATUS: COMPLETE" in status_text:
                break
            if "STATUS: STUCK" in status_text:
                # Stuck — inject diagnosis request and let the LLM explain
                reason = (verification.content or "").strip()
                self._history.append(
                    {
                        "role": "user",
                        "content": (
                            f"SYSTEM: Task verification determined this task is stuck.\n"
                            f"Verifier assessment: {reason}\n\n"
                            "Explain to the user what went wrong, what you tried, and "
                            "suggest specific next steps they can take to unblock this."
                        ),
                    }
                )
                yield StreamTaskProgress(
                    phase="analyzing", message="Diagnosing blocked task..."
                )
                async for event in self._llm.plan_stream(
                    system=system,
                    messages=self._history,
                ):
                    yield event
                break

            # INCOMPLETE — continue working
            continuation += 1
            reason = (verification.content or "").strip()
            self._history.append(
                {
                    "role": "user",
                    "content": (
                        f"SYSTEM: Task verification determined this task is not yet complete "
                        f"(attempt {continuation}/{_MAX_CONTINUATIONS}).\n"
                        f"Verifier assessment: {reason}\n\n"
                        "Continue working on the original request. Pick up where you left off "
                        "and finish the remaining work. Do not repeat work already done."
                    ),
                }
            )
            yield StreamTaskProgress(
                phase="analyzing",
                message=f"Task incomplete — continuing ({continuation}/{_MAX_CONTINUATIONS})...",
            )

            # Re-enter tool loop: get next LLM response with tools available
            response = None
            async for event in self._llm.plan_stream(
                system=system,
                messages=self._history,
                tools=tools,
            ):
                yield event
                if isinstance(event, StreamComplete):
                    response = event
            if response is None:
                return
            llm_response = response.response
            # Loop back to the top of the completion verification loop

        # Text-only final response — append to history (if not already appended
        # by the verification block above).
        if not self._history or self._history[-1].get("role") != "assistant":
            reply = llm_response.content or ""
            self._history.append({"role": "assistant", "content": reply})

        # Consolidation: replay scratchpad sessions to extract lessons
        if self._cortex is not None and self._cortex.mode != "off":
            self._maybe_consolidate_scratchpads()

    def _maybe_consolidate_scratchpads(self) -> None:
        """Check if any scratchpad sessions warrant consolidation and fire it off."""
        from anton.memory.consolidator import Consolidator

        consolidator = Consolidator()
        for pad in self._scratchpads._pads.values():
            cells = list(pad.cells)
            if consolidator.should_replay(cells):
                asyncio.create_task(self._consolidate(cells))

    async def _consolidate(self, cells: list) -> None:
        """Run offline consolidation on a completed scratchpad session."""
        from anton.memory.consolidator import Consolidator

        consolidator = Consolidator()
        engrams = await consolidator.replay_and_extract(cells, self._llm)
        if not engrams or self._cortex is None:
            return

        auto_encode = [e for e in engrams if not self._cortex.encoding_gate(e)]
        needs_confirm = [e for e in engrams if self._cortex.encoding_gate(e)]

        if auto_encode:
            await self._cortex.encode(auto_encode)

        if needs_confirm:
            self._pending_memory_confirmations.extend(needs_confirm)


def _apply_error_tracking(
    result_text: str,
    tool_name: str,
    error_streak: dict[str, int],
    resilience_nudged: set[str],
) -> str:
    """Track consecutive errors per tool and append nudge/circuit-breaker messages."""
    is_error = any(
        marker in result_text
        for marker in ("[error]", "Task failed:", "failed", "timed out", "Rejected:")
    )
    if is_error:
        error_streak[tool_name] = error_streak.get(tool_name, 0) + 1
    else:
        error_streak[tool_name] = 0
        resilience_nudged.discard(tool_name)

    streak = error_streak.get(tool_name, 0)
    if streak >= _RESILIENCE_NUDGE_AT and tool_name not in resilience_nudged:
        result_text += _RESILIENCE_NUDGE
        resilience_nudged.add(tool_name)

    if streak >= _MAX_CONSECUTIVE_ERRORS:
        result_text += (
            f"\n\nSYSTEM: The '{tool_name}' tool has failed {_MAX_CONSECUTIVE_ERRORS} times "
            "in a row. Stop retrying this approach. Either try a completely different "
            "strategy or tell the user what's going wrong so they can help."
        )

    return result_text


# DS_* var names whose values are known to be secret (passwords, tokens, keys).
# Populated at startup and after each successful connect.
_DS_SECRET_VARS: set[str] = set()

# DS_* var names for **ALL** fields of registered engines.
_DS_KNOWN_VARS: set[str] = set()


def _reset_registered_ds_vars() -> None:
    """Clear the DS_* var registries so they can be rebuilt from current vault state."""
    _DS_SECRET_VARS.clear()
    _DS_KNOWN_VARS.clear()


def parse_connection_slug(
    slug: str,
    known_engines: list[str],
    *,
    vault: DataVault | None = None,
) -> tuple[str, str] | None:
    """Split a connection slug into (engine, name) using longest-prefix matching.

    First tries each known registry engine longest-first so that 'sql-server-prod-db' is
    correctly parsed as engine='sql-server', name='prod-db' rather than
    engine='sql', name='server-prod-db'.

    If nothing matches and a vault is supplied, falls back to scanning vault
    connections for an exact slug match — handles custom/unregistered engines.

    Returns None if no match found or name part is empty.
    """
    for engine in sorted(known_engines, key=len, reverse=True):
        prefix = engine + "-"
        if slug.startswith(prefix) and len(slug) > len(prefix):
            return (engine, slug[len(prefix):])

    if vault is not None:
        for conn in vault.list_connections():
            if f"{conn['engine']}-{conn['name']}" == slug:
                return (conn["engine"], conn["name"])

    return None


def _register_secret_vars(
    engine_def: "DatasourceEngine", *, engine: str = "", name: str = ""
) -> None:
    """Record which DS_* var names correspond to known/secret fields for engine_def.

    If engine and name are given, registers namespaced vars (DS_ENGINE_NAME__FIELD).
    Otherwise registers flat vars (DS_FIELD) — for temporary test_snippet execution.
    """
    all_fields = list(engine_def.fields)
    for am in engine_def.auth_methods or []:
        all_fields.extend(am.fields)
    for f in all_fields:
        if engine and name:
            prefix = _slug_env_prefix(engine, name)
            key = f"{prefix}__{f.name.upper()}"
        else:
            key = f"DS_{f.name.upper()}"
        _DS_KNOWN_VARS.add(key)
        if f.secret:
            _DS_SECRET_VARS.add(key)


def _scrub_credentials(text: str) -> str:
    """Remove secret DS_* values from scratchpad output before it reaches the LLM.

    Only redacts vars registered as secret via _register_secret_vars (driven by
    DatasourceField.secret=true in datasources.md).  Non-secret fields of known
    engines (DS_HOST, DS_PORT, DS_BASE_URL, …) are left readable so the LLM can
    reason about connection errors.  For truly unknown DS_* vars (custom engines
    not yet in the registry) the fallback scrubs any long value — conservative
    but safe.
    """
    for key in _DS_SECRET_VARS:
        value = os.environ.get(key, "")
        if not value:
            continue
        text = text.replace(value, f"[{key}]")
    for key, value in os.environ.items():
        if not key.startswith("DS_") or key in _DS_KNOWN_VARS:
            continue
        # Length guard only for unknown DS_* vars (not registered secrets).
        # Unknown vars are matched heuristically — a short value like "on"
        # or "true" in a DS_ENABLE_X var should not be scrubbed.
        # Registered secret vars bypass this check entirely.
        if not value or len(value) <= 8:
            continue
        text = text.replace(value, f"[{key}]")
    return text


def _build_datasource_context(active_only: str | None = None) -> str:
    """Build a system-prompt section listing available DS_* env vars by name.

    Shows the LLM what data sources are connected and which environment
    variable names to use — without exposing any credential values.

    If active_only is set, only the matching slug is included.
    """
    try:
        vault = DataVault()
        conns = vault.list_connections()
    except Exception:
        return ""
    if not conns:
        return ""
    lines = ["\n\n## Connected Data Sources"]
    lines.append(
        "Credentials are pre-injected as namespaced DS_<ENGINE_NAME>__<FIELD> "
        "environment variables. Use them directly in scratchpad code "
        "(e.g. DS_POSTGRES_PROD_DB__HOST). "
        "Never read ~/.anton/data_vault/ files directly.\n"
    )
    for c in conns:
        slug = f"{c['engine']}-{c['name']}"
        if active_only and slug != active_only:
            continue
        fields = vault.load(c["engine"], c["name"]) or {}
        prefix = _slug_env_prefix(c["engine"], c["name"])
        var_names = ", ".join(f"{prefix}__{k.upper()}" for k in fields)
        lines.append(f"- `{slug}` ({c['engine']}) → {var_names}")
    return "\n".join(lines)


def _restore_namespaced_env(vault: DataVault) -> None:
    """Clear all DS_* vars, then reinject every saved connection as namespaced."""
    from anton.datasource_registry import DatasourceRegistry

    _reset_registered_ds_vars()
    vault.clear_ds_env()
    dreg = DatasourceRegistry()
    for conn in vault.list_connections():
        vault.inject_env(conn["engine"], conn["name"])  # flat=False by default
        edef = dreg.get(conn["engine"])
        if edef is not None:
            _register_secret_vars(edef, engine=conn["engine"], name=conn["name"])


def _build_runtime_context(settings: AntonSettings) -> str:
    """Build runtime context string including Minds datasource info if configured."""
    ctx = (
        f"- Provider: {settings.planning_provider}\n"
        f"- Planning model: {settings.planning_model}\n"
        f"- Coding model: {settings.coding_model}\n"
        f"- Workspace: {settings.workspace_path}\n"
        f"- Memory mode: {settings.memory_mode}"
    )
    if settings.minds_api_key and (
        settings.minds_mind_name or settings.minds_datasource
    ):
        engine = settings.minds_datasource_engine or "unknown"
        ctx += f"\n\n**CONNECTED MIND (Minds):**\n"
        if settings.minds_mind_name:
            ctx += f"- Mind: {settings.minds_mind_name}\n"
        if settings.minds_datasource:
            ctx += (
                f"- Datasource: {settings.minds_datasource}\n" f"- Engine: {engine}\n"
            )
        ctx += (
            f"- Minds URL: {settings.minds_url}\n"
            f"- To query data, use the scratchpad with the built-in `query_minds_data()` function.\n"
            f"  It is pre-loaded in the scratchpad namespace — DO NOT import it. Just call it directly.\n"
            f'  Example: result = query_minds_data("SELECT * FROM users LIMIT 5")\n'
            f"  Returns dict with 'type', 'data' (list of rows), 'column_names', 'error_message'.\n"
            f'  Optional: query_minds_data("SELECT ...", datasource="other_ds")\n'
        )
        if settings.minds_datasource:
            ctx += f"- Write SQL appropriate for the {engine} engine.\n"
    return ctx


def _rebuild_session(
    *,
    settings: AntonSettings,
    state: dict,
    self_awareness,
    cortex,
    workspace,
    console: Console,
    episodic: EpisodicMemory | None = None,
    history_store: HistoryStore | None = None,
    session_id: str | None = None,
) -> ChatSession:
    """Rebuild LLMClient + ChatSession after settings change."""
    from anton.llm.client import LLMClient

    state["llm_client"] = LLMClient.from_settings(settings)

    # Update cortex with new LLM client and memory mode
    if cortex is not None:
        cortex._llm = state["llm_client"]
        cortex.mode = settings.memory_mode

    # Refresh mind knowledge from remote server
    refresh_knowledge(settings, cortex)

    runtime_context = _build_runtime_context(settings)
    api_key = (
        settings.anthropic_api_key
        if settings.coding_provider == "anthropic"
        else settings.openai_api_key
    ) or ""
    return ChatSession(
        state["llm_client"],
        self_awareness=self_awareness,
        cortex=cortex,
        episodic=episodic,
        runtime_context=runtime_context,
        workspace=workspace,
        console=console,
        coding_provider=settings.coding_provider,
        coding_api_key=api_key,
        coding_base_url=settings.openai_base_url or "",
        history_store=history_store,
        session_id=session_id,
        proactive_dashboards=settings.proactive_dashboards,
    )


def _handle_memory(
    console: Console,
    settings: AntonSettings,
    cortex,
    episodic: EpisodicMemory | None = None,
) -> None:
    """Show memory status — read-only dashboard."""
    console.print()
    console.print("[anton.cyan]Memory Status[/]")
    console.print()

    mode_labels = {
        "autopilot": "Autopilot — Anton decides what to remember",
        "copilot": "Co-pilot — save obvious, confirm ambiguous",
        "off": "Off — never save (still reads existing)",
    }
    mode_label = mode_labels.get(settings.memory_mode, settings.memory_mode)
    console.print(f"  Mode:  [bold]{mode_label}[/]")
    console.print()

    if cortex is None:
        console.print("  [anton.warning]Memory system not initialized.[/]")
        console.print()
        return

    def _show_scope(label: str, hc) -> int:
        identity = hc.recall_identity()
        rules = hc.recall_rules()
        lessons_raw = hc._read_full_lessons()
        rule_count = (
            sum(1 for ln in rules.splitlines() if ln.strip().startswith("- "))
            if rules
            else 0
        )
        lesson_count = (
            sum(1 for ln in lessons_raw.splitlines() if ln.strip().startswith("- "))
            if lessons_raw
            else 0
        )
        topics: list[str] = []
        if hc._topics_dir.is_dir():
            topics = [
                p.stem for p in sorted(hc._topics_dir.iterdir()) if p.suffix == ".md"
            ]

        console.print(f"  [anton.cyan]{label}[/] [dim]({hc._dir})[/]")
        if identity:
            entries = [
                ln.strip()[2:]
                for ln in identity.splitlines()
                if ln.strip().startswith("- ")
            ]
            if entries:
                console.print(
                    f"    Identity:  {', '.join(entries[:3])}"
                    + (" ..." if len(entries) > 3 else "")
                )
            else:
                console.print("    Identity:  [dim](set)[/]")
        else:
            console.print("    Identity:  [dim](empty)[/]")
        console.print(f"    Rules:     {rule_count}")
        console.print(f"    Lessons:   {lesson_count}")
        if topics:
            console.print(f"    Topics:    {', '.join(topics)}")
        else:
            console.print("    Topics:    [dim](none)[/]")
        console.print()
        return rule_count + lesson_count

    global_total = _show_scope("Global Memory", cortex.global_hc)

    project_total = _show_scope("Project Memory", cortex.project_hc)

    total = global_total + project_total
    console.print(f"  Total entries: [bold]{total}[/]")
    if cortex.needs_compaction():
        console.print("  [anton.warning]Compaction needed (>50 entries in a scope)[/]")
    console.print()

    if episodic is not None:
        status = "[bold]ON[/]" if episodic.enabled else "[dim]OFF[/]"
        sessions = episodic.session_count()
        console.print(f"  [anton.cyan]Episodic Memory[/]")
        console.print(f"    Status:    {status}")
        console.print(f"    Sessions:  {sessions}")
        console.print()

    console.print("[dim]  Use /setup > Memory to change configuration.[/]")
    console.print()


async def _handle_resume(
    console: Console,
    settings: AntonSettings,
    state: dict,
    self_awareness,
    cortex,
    workspace,
    session: ChatSession,
    episodic: EpisodicMemory | None = None,
    history_store: HistoryStore | None = None,
) -> tuple[ChatSession, str | None]:
    """Show session picker and resume a previous chat session.

    Returns (new_session, resumed_session_id) or (original_session, None).
    """
    from rich.table import Table

    if history_store is None:
        console.print("[anton.warning]History store not available.[/]")
        console.print()
        return session, None

    sessions = history_store.list_sessions(limit=10)
    if not sessions:
        console.print()
        console.print("[anton.warning]No previous sessions to resume.[/]")
        console.print()
        return session, None

    console.print()
    console.print("[anton.cyan]Recent sessions:[/]")
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="bold", width=3)
    table.add_column("Date", style="anton.cyan")
    table.add_column("Turns", justify="right")
    table.add_column("Preview")

    for i, s in enumerate(sessions, 1):
        table.add_row(str(i), s["date"], str(s["turns"]), s["preview"])

    console.print(table)
    console.print()

    choices = [str(i) for i in range(1, len(sessions) + 1)] + ["q"]
    choice = await prompt_or_cancel("(anton) Select session (or q to cancel)", choices=choices, default="q")
    if choice is None or choice == "q":
        console.print()
        return session, None

    idx = int(choice) - 1
    selected = sessions[idx]
    sid = selected["session_id"]

    history = history_store.load(sid)
    if history is None:
        console.print("[anton.error]Failed to load session history.[/]")
        console.print()
        return session, None

    # Resume episodic memory for this session
    if episodic is not None and episodic.enabled:
        episodic.resume_session(sid)

    # Close old scratchpads
    if session._scratchpads.list_pads():
        await session._scratchpads.close_all()

    # Build new session with restored history
    new_session = _rebuild_session(
        settings=settings,
        state=state,
        self_awareness=self_awareness,
        cortex=cortex,
        workspace=workspace,
        console=console,
        episodic=episodic,
        history_store=history_store,
        session_id=sid,
    )
    new_session._history = list(history)
    new_session._turn_count = sum(1 for m in history if m.get("role") == "user")

    console.print()
    console.print(
        f"[anton.success]Resumed session from {selected['date']} ({selected['turns']} turns)[/]"
    )
    console.print()

    return new_session, sid


async def _handle_setup(
    console: Console,
    settings: AntonSettings,
    workspace: Workspace,
    state: dict,
    self_awareness,
    cortex,
    session: ChatSession,
    episodic: EpisodicMemory | None = None,
    history_store: HistoryStore | None = None,
    session_id: str | None = None,
) -> ChatSession:
    """Interactive setup wizard with sub-menu: Models, Memory, or Minds."""
    console.print()
    console.print("[anton.cyan]/setup[/]")
    console.print()
    console.print("  What do you want to configure?")
    console.print("    [bold]1[/]  LLM — provider, API key, and models")
    console.print("    [bold]2[/]  Memory — memory mode and episodic memory")
    console.print("    [bold]q[/]  Back")
    console.print()

    top_choice = await prompt_or_cancel("(anton) Select", choices=["1", "2", "q"], default="q")
    if top_choice is None:
        console.print()
        return session

    if top_choice == "q":
        console.print()
        return session
    elif top_choice == "1":
        return await _handle_setup_models(
            console,
            settings,
            workspace,
            state,
            self_awareness,
            cortex,
            session,
            episodic=episodic,
            history_store=history_store,
            session_id=session_id,
        )
    else:
        await _handle_setup_memory(console, settings, workspace, cortex, episodic=episodic)
        return session


async def _handle_setup_models(
    console: Console,
    settings: AntonSettings,
    workspace: Workspace,
    state: dict,
    self_awareness,
    cortex,
    session: ChatSession,
    episodic: EpisodicMemory | None = None,
    history_store: HistoryStore | None = None,
    session_id: str | None = None,
) -> ChatSession:
    """Setup sub-menu: provider, API key, and models."""
    from anton.workspace import Workspace as _Workspace
    from anton.cli import _SetupRetry, _setup_minds, _setup_other_provider

    # Always persist API keys and model settings to global ~/.anton/.env
    global_ws = _Workspace(Path.home())

    def _provider_label(provider: str) -> str:
        if provider == "openai-compatible":
            if settings.minds_url and "mdb.ai" in settings.minds_url:
                return "Minds-Enterprise-Cloud"
            return "Minds-Enterprise"
        return provider.capitalize()

    def _model_label(model: str, role: str) -> str:
        if model in ("_reason_", "_code_"):
            return f"smart_router({role})"
        return model

    provider_display = _provider_label(settings.planning_provider)
    planning_display = _model_label(settings.planning_model, "planning")
    coding_display = _model_label(settings.coding_model, "coding")

    console.print()
    console.print("[anton.cyan]Current configuration:[/]")
    console.print(f"  Provider: [bold]{provider_display}[/]")
    if planning_display == coding_display:
        console.print(f"  Model:    [bold]{planning_display}[/]")
    else:
        console.print(f"  Planning: [bold]{planning_display}[/]")
        console.print(f"  Coding:   [bold]{coding_display}[/]")
    console.print()

    def _print_choices():
        console.print("  [bold]1[/]  [link=https://mdb.ai][anton.cyan]Minds-Enterprise-Cloud[/][/link] [anton.success](recommended)[/]")
        console.print("  [bold]2[/]  [anton.cyan]Minds-Enterprise-Server[/] [anton.muted]self-hosted[/]")
        console.print("  [bold]3[/]  [anton.cyan]Bring your own key[/] [anton.muted]Anthropic / OpenAI / Gemini[/]")
        console.print("  [bold]q[/]  [anton.muted]Back[/]")
        console.print()

    _print_choices()

    while True:
        choice = await prompt_or_cancel("(anton) Choose LLM Provider",
                                         choices=["1", "2", "3", "q"],
                                         default="1")
        if choice is None or choice == "q":
            return session

        try:
            if choice == "1":
                _setup_minds(settings, global_ws)
            elif choice == "2":
                _setup_minds(settings, global_ws, default_url=None)
            elif choice == "3":
                _setup_other_provider(settings, global_ws)
            break
        except _SetupRetry:
            console.print()
            _print_choices()
            continue

    global_ws.apply_env_to_process()

    console.print()
    console.print("[anton.success]Configuration updated.[/]")
    console.print()

    return _rebuild_session(
        settings=settings,
        state=state,
        self_awareness=self_awareness,
        cortex=cortex,
        workspace=workspace,
        console=console,
        episodic=episodic,
        history_store=history_store,
        session_id=session_id,
    )


async def _handle_setup_memory(
    console: Console,
    settings: AntonSettings,
    workspace: Workspace,
    cortex,
    episodic: EpisodicMemory | None = None,
) -> None:
    """Setup sub-menu: memory mode and episodic memory toggle."""
    console.print()
    console.print("[anton.cyan]Memory configuration[/]")
    console.print()

    console.print("  Memory mode:")
    console.print(
        r"    [bold]1[/]  Autopilot — Anton decides what to remember       [dim]\[recommended][/]"
    )
    console.print(
        r"    [bold]2[/]  Co-pilot — save obvious, confirm ambiguous        [dim]\[selective][/]"
    )
    console.print(
        r"    [bold]3[/]  Off — never save memory (still reads existing)    [dim]\[suppressed][/]"
    )
    console.print()

    mode_map = {"1": "autopilot", "2": "copilot", "3": "off"}
    current_mode_num = {"autopilot": "1", "copilot": "2", "off": "3"}.get(
        settings.memory_mode, "1"
    )
    mode_choice = await prompt_or_cancel("(anton) Memory mode", choices=["1", "2", "3"], default=current_mode_num)
    if mode_choice is None:
        console.print()
        return
    memory_mode = mode_map[mode_choice]
    settings.memory_mode = memory_mode
    workspace.set_secret("ANTON_MEMORY_MODE", memory_mode)
    if cortex is not None:
        cortex.mode = memory_mode

    if episodic is not None:
        console.print()
        ep_status = "ON" if episodic.enabled else "OFF"
        console.print(
            f"  Episodic memory (conversation archive): Currently [bold]{ep_status}[/]"
        )
        toggle = await prompt_or_cancel("(anton) Toggle episodic memory?", choices=["y", "n"], default="n")
        if toggle is None:
            toggle = "n"
        if toggle == "y":
            new_state = not episodic.enabled
            episodic.enabled = new_state
            settings.episodic_memory = new_state
            workspace.set_secret(
                "ANTON_EPISODIC_MEMORY", "true" if new_state else "false"
            )
            console.print(f"  Episodic memory: [bold]{'ON' if new_state else 'OFF'}[/]")

    console.print()
    console.print("[anton.success]Configuration updated.[/]")
    console.print()




#TODO: The /data-connections menu is deprecated and will be removed in a future release.
async def _handle_data_connections(
    console: Console,
    settings: AntonSettings,
    workspace: Workspace,
    session: ChatSession,
) -> ChatSession:
    """View and manage stored keys and connections across global and project vaults."""
    from anton.workspace import Workspace as _Workspace

    global_ws = _Workspace(Path.home())

    global_env = global_ws.load_env()
    project_env = workspace.load_env()

    # Merge with source tags: project keys override global for display,
    # but we track where each lives for writes/removals.
    all_keys: dict[str, tuple[str, str, str]] = (
        {}
    )  # key -> (value, source, scope_label)
    for k, v in global_env.items():
        all_keys[k] = (v, "global", "~/.anton/.env")
    for k, v in project_env.items():
        all_keys[k] = (v, "project", f"{workspace.base}/.anton/.env")

    console.print()

    if not all_keys:
        console.print("[anton.warning]No connections or secrets configured.[/]")
        console.print(
            "[anton.muted]Use /connect to set up a Minds connection, or ask Anton to store a key.[/]"
        )
        console.print()
        return session

    def _print_table() -> list[tuple[str, str, str, str]]:
        """Print grouped key table and return flat list for menu selection."""
        minds = {k: all_keys[k] for k in sorted(all_keys) if k in MINDS_KEYS}
        llm = {k: all_keys[k] for k in sorted(all_keys) if k in LLM_KEYS}
        other = {
            k: all_keys[k]
            for k in sorted(all_keys)
            if k not in MINDS_KEYS and k not in LLM_KEYS
        }

        flat: list[tuple[str, str, str, str]] = []  # (key, value, source, scope_label)
        idx = 1

        if minds:
            console.print("[anton.cyan]Minds Connection[/]")
            for k, (v, src, lbl) in minds.items():
                console.print(
                    f"    [bold]{idx}[/]  {k} = {display_value(k, v)}  [dim]({lbl})[/]"
                )
                flat.append((k, v, src, lbl))
                idx += 1
            console.print()

        if llm:
            console.print("[anton.cyan]LLM Configuration[/]")
            for k, (v, src, lbl) in llm.items():
                console.print(
                    f"    [bold]{idx}[/]  {k} = {display_value(k, v)}  [dim]({lbl})[/]"
                )
                flat.append((k, v, src, lbl))
                idx += 1
            console.print()

        if other:
            console.print("[anton.cyan]Other Integrations[/]")
            for k, (v, src, lbl) in other.items():
                console.print(
                    f"    [bold]{idx}[/]  {k} = {display_value(k, v)}  [dim]({lbl})[/]"
                )
                flat.append((k, v, src, lbl))
                idx += 1
            console.print()

        return flat

    while True:
        console.print("[anton.cyan]/data-connections[/]")
        console.print()
        flat = _print_table()

        console.print("  [bold]1[/]  Edit a key")
        console.print("  [bold]2[/]  Remove a key")
        console.print("  [bold]3[/]  Add a new key")
        console.print("  [bold]q[/]  Back")
        console.print()

        action = await prompt_or_cancel("(anton) Select", choices=["1", "2", "3", "q"], default="q")
        if action is None or action == "q":
            console.print()
            return session

        if action == "1":
            # --- Edit ---
            console.print()
            pick = await prompt_or_cancel(f"(anton) Key number to edit (1-{len(flat)})")
            if pick is None:
                continue
            try:
                pick_idx = int(pick) - 1
                if not 0 <= pick_idx < len(flat):
                    raise ValueError
            except ValueError:
                console.print("[anton.warning]Invalid selection.[/]")
                console.print()
                continue

            key, old_val, src, lbl = flat[pick_idx]
            use_password = is_secret_key(key)
            new_val = (await prompt_or_cancel(
                f"(anton) New value for {key}",
                default="" if use_password else old_val,
                password=use_password,
            ) or "").strip()
            if new_val is None or not new_val:
                console.print("[anton.muted]Value unchanged.[/]")
                console.print()
                continue

            target_ws = global_ws if src == "global" else workspace
            target_ws.set_secret(key, new_val)
            target_ws.apply_env_to_process()
            all_keys[key] = (new_val, src, lbl)
            console.print(f"[anton.success]Updated {key}.[/]")
            console.print()

        elif action == "2":
            # --- Remove ---
            console.print()
            pick = await prompt_or_cancel(f"(anton) Key number to remove (1-{len(flat)})")
            if pick is None:
                continue
            try:
                pick_idx = int(pick) - 1
                if not 0 <= pick_idx < len(flat):
                    raise ValueError
            except ValueError:
                console.print("[anton.warning]Invalid selection.[/]")
                console.print()
                continue

            key, _, src, lbl = flat[pick_idx]
            if not Confirm.ask(
                f"Remove {key} from {lbl}?", default=False, console=console
            ):
                console.print("[anton.muted]Cancelled.[/]")
                console.print()
                continue

            target_ws = global_ws if src == "global" else workspace
            target_ws.remove_secret(key)
            del all_keys[key]
            console.print(f"[anton.success]Removed {key}.[/]")
            console.print()

        elif action == "3":
            # --- Add ---
            console.print()
            new_key = (await prompt_or_cancel("(anton) Key name (e.g. HUBSPOT_API_KEY)") or "").strip()
            if not new_key:
                console.print("[anton.warning]Key name cannot be empty.[/]")
                console.print()
                continue

            if new_key in all_keys:
                if not Confirm.ask(
                    f"{new_key} already exists. Overwrite?",
                    default=False,
                    console=console,
                ):
                    console.print("[anton.muted]Cancelled.[/]")
                    console.print()
                    continue

            use_password = is_secret_key(new_key)
            new_val = (await prompt_or_cancel(
                f"(anton) Value for {new_key}",
                password=use_password,
            ) or "").strip()
            if not new_val:
                console.print("[anton.warning]Value cannot be empty.[/]")
                console.print()
                continue

            scope = Prompt.ask(
                "Store in",
                choices=["global", "project"],
                default="global",
                console=console,
            )
            target_ws = global_ws if scope == "global" else workspace
            scope_label = (
                "~/.anton/.env"
                if scope == "global"
                else f"{workspace.base}/.anton/.env"
            )
            target_ws.set_secret(new_key, new_val)
            target_ws.apply_env_to_process()
            all_keys[new_key] = (new_val, scope, scope_label)
            console.print(f"[anton.success]Saved {new_key}.[/]")
            console.print()


async def _handle_connect(
    console: Console,
    settings: AntonSettings,
    workspace: Workspace,
    state: dict,
    self_awareness,
    cortex,
    session: ChatSession,
    episodic: EpisodicMemory | None = None,
) -> ChatSession:
    """Connect to a Minds server: select a Mind, then optionally a datasource."""
    from anton.workspace import Workspace as _Workspace

    global_ws = _Workspace(Path.home())

    console.print()

    # --- Prompt for URL and API key (use saved values as defaults) ---
    saved_url = normalize_minds_url(settings.minds_url)
    minds_url = await prompt_or_cancel("(anton) Minds server URL", default=saved_url)
    if minds_url is None:
        return session
    minds_url = normalize_minds_url(minds_url)

    saved_key = settings.minds_api_key or ""
    api_key = await prompt_minds_api_key(
        console,
        current_key=saved_key,
        allow_empty_keep=True,
    )
    if not api_key:
        console.print("[anton.error]API key is required.[/]")
        console.print()
        return session

    ssl_verify = settings.minds_ssl_verify

    # --- Try to connect ---
    minds = None
    while minds is None:
        console.print()
        console.print(f"[anton.muted]Connecting to {minds_url}...[/]")
        try:
            minds = list_minds(minds_url, api_key, verify=ssl_verify)
            break
        except (urllib.error.URLError, urllib.error.HTTPError) as err:
            headline, advice = describe_minds_connection_error(err)
            console.print(f"[anton.error]{headline}[/]")
            console.print(f"[anton.muted]{advice}[/]")
        except Exception as err:
            headline, advice = describe_minds_connection_error(err)
            console.print(f"[anton.error]{headline}[/]")
            console.print(f"[anton.muted]{advice}[/]")

        console.print()
        console.print("  Recovery options:")
        console.print("    [bold]1[/]  Reconfigure API key")
        console.print("    [bold]2[/]  Retry without SSL verification")
        console.print("    [bold]q[/]  Back")
        console.print()

        action = await prompt_or_cancel("(anton) Select", choices=["1", "2", "q"], default="q")
        if action is None or action == "q":
            console.print("[anton.muted]Aborted.[/]")
            console.print()
            return session
        if action == "1":
            new_key = await prompt_minds_api_key(
                console,
                current_key=api_key,
                allow_empty_keep=False,
            )
            if new_key is None:
                console.print("[anton.muted]API key unchanged.[/]")
                continue
            api_key = new_key
            ssl_verify = settings.minds_ssl_verify
            continue

        ssl_verify = False

    if not minds:
        console.print("[anton.warning]No minds found on this server.[/]")
        console.print()
        return session

    # --- Select a Mind ---
    console.print()
    console.print("[anton.cyan]Available minds:[/]")
    for i, mind in enumerate(minds, 1):
        name = mind.get("name", "?")
        ds_list = mind.get("datasources", [])
        ds_count = len(ds_list)
        ds_label = (
            f"{ds_count} datasource{'s' if ds_count != 1 else ''}"
            if ds_count
            else "no datasources"
        )
        console.print(f"    [bold]{i}[/]  {name} [dim]({ds_label})[/]")
    console.print()

    choices = [str(i) for i in range(1, len(minds) + 1)]
    pick = await prompt_or_cancel("(anton) Select mind", choices=choices)
    if pick is None:
        return session
    selected_mind = minds[int(pick) - 1]
    mind_name = selected_mind.get("name", "")

    # --- Datasource selection within the mind ---
    mind_datasources = selected_mind.get("datasources", [])
    ds_name = ""
    ds_engine = ""

    if len(mind_datasources) > 1:
        console.print()
        console.print(f"[anton.cyan]Datasources in mind '{mind_name}':[/]")
        for i, ds_ref in enumerate(mind_datasources, 1):
            # datasource refs may be strings or dicts
            ref_name = ds_ref if isinstance(ds_ref, str) else ds_ref.get("name", "?")
            console.print(f"    [bold]{i}[/]  {ref_name}")
        console.print()
        ds_choices = [str(i) for i in range(1, len(mind_datasources) + 1)]
        ds_pick = await prompt_or_cancel("(anton) Select datasource", choices=ds_choices)
        if ds_pick is None:
            return session
        picked_ds = mind_datasources[int(ds_pick) - 1]
        ds_name = picked_ds if isinstance(picked_ds, str) else picked_ds.get("name", "")
    elif len(mind_datasources) == 1:
        picked_ds = mind_datasources[0]
        ds_name = picked_ds if isinstance(picked_ds, str) else picked_ds.get("name", "")
        console.print(f"[anton.muted]Auto-selected datasource: {ds_name}[/]")

    if ds_name:
        try:
            all_datasources = list_datasources(
                minds_url, api_key, verify=ssl_verify
            )
            for ds in all_datasources:
                if ds.get("name") == ds_name:
                    ds_engine = ds.get("engine", "unknown")
                    break
        except Exception:
            ds_engine = "unknown"

    # --- Persist to global ~/.anton/.env ---
    global_ws.set_secret("ANTON_MINDS_API_KEY", api_key)
    global_ws.set_secret("ANTON_MINDS_URL", minds_url)
    global_ws.set_secret("ANTON_MINDS_MIND_NAME", mind_name)
    global_ws.set_secret("ANTON_MINDS_DATASOURCE", ds_name)
    global_ws.set_secret("ANTON_MINDS_DATASOURCE_ENGINE", ds_engine)
    global_ws.set_secret("ANTON_MINDS_SSL_VERIFY", "true" if ssl_verify else "false")

    settings.minds_api_key = api_key
    settings.minds_url = minds_url
    settings.minds_mind_name = mind_name
    settings.minds_datasource = ds_name
    settings.minds_datasource_engine = ds_engine
    settings.minds_ssl_verify = ssl_verify

    console.print()
    status = f"[anton.success]Selected mind: {mind_name}[/]"
    if ds_name:
        status += f" [anton.success]| datasource: {ds_name} ({ds_engine})[/]"
    console.print(status)

    # --- Test if the Minds server also supports LLM endpoints ---
    # (silenced: was printing "Testing LLM endpoints..." and "not available" messages)
    llm_ok = test_llm(minds_url, api_key, verify=ssl_verify)

    if llm_ok:
        console.print(
            "[anton.success]LLM endpoints available — using Minds server as LLM provider.[/]"
        )
        base_url = f"{minds_url.rstrip('/')}/api/v1"
        settings.openai_api_key = api_key
        settings.openai_base_url = base_url
        settings.planning_provider = "openai-compatible"
        settings.coding_provider = "openai-compatible"
        settings.planning_model = "_reason_"
        settings.coding_model = "_code_"
        global_ws.set_secret("ANTON_OPENAI_API_KEY", api_key)
        global_ws.set_secret("ANTON_OPENAI_BASE_URL", base_url)
        global_ws.set_secret("ANTON_PLANNING_PROVIDER", "openai-compatible")
        global_ws.set_secret("ANTON_CODING_PROVIDER", "openai-compatible")
        global_ws.set_secret("ANTON_PLANNING_MODEL", "_reason_")
        global_ws.set_secret("ANTON_CODING_MODEL", "_code_")
    else:
        # Check if Anthropic key is already configured
        has_anthropic = settings.anthropic_api_key or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        if not has_anthropic:
            anthropic_key = Prompt.ask("Anthropic API key (for LLM)", console=console)
            if anthropic_key.strip():
                anthropic_key = anthropic_key.strip()
                settings.anthropic_api_key = anthropic_key
                settings.planning_provider = "anthropic"
                settings.coding_provider = "anthropic"
                settings.planning_model = "claude-sonnet-4-6"
                settings.coding_model = "claude-haiku-4-5-20251001"
                global_ws.set_secret("ANTON_ANTHROPIC_API_KEY", anthropic_key)
                global_ws.set_secret("ANTON_PLANNING_PROVIDER", "anthropic")
                global_ws.set_secret("ANTON_CODING_PROVIDER", "anthropic")
                global_ws.set_secret("ANTON_PLANNING_MODEL", "claude-sonnet-4-6")
                global_ws.set_secret("ANTON_CODING_MODEL", "claude-haiku-4-5-20251001")
                console.print("[anton.success]Anthropic API key saved.[/]")
            else:
                console.print(
                    "[anton.warning]No API key provided — LLM calls will not work.[/]"
                )

    global_ws.apply_env_to_process()
    console.print()

    return _rebuild_session(
        settings=settings,
        state=state,
        self_awareness=self_awareness,
        cortex=cortex,
        workspace=workspace,
        console=console,
        episodic=episodic,
    )


def _format_file_message(text: str, paths: list[Path], console: Console) -> str:
    """Rewrite user input to include file contents for detected paths."""
    parts: list[str] = []

    # Determine what the user typed besides the paths
    remaining = text
    for p in paths:
        # Remove various representations of the path from the text
        for representation in (str(p), f"'{p}'", f'"{p}"', str(p).replace(" ", "\\ ")):
            remaining = remaining.replace(representation, "")
    remaining = remaining.strip()

    # Build the instruction
    if remaining:
        parts.append(remaining)
    else:
        if len(paths) == 1:
            parts.append(f"Analyze this file: {paths[0].name}")
        else:
            names = ", ".join(p.name for p in paths)
            parts.append(f"Analyze these files: {names}")

    # Attach each file
    for p in paths:
        suffix = p.suffix.lower()
        size = p.stat().st_size

        # Show what we're picking up
        console.print(f"  [anton.muted]attached: {p.name} ({_human_size(size)})[/]")

        # Skip very large files (>500KB) — just reference them
        if size > 512_000:
            parts.append(
                f'\n<file path="{p}">\n(File too large to inline — {_human_size(size)}. '
                f"Use the scratchpad to read it.)\n</file>"
            )
            continue

        # Skip binary-looking files
        if suffix in (
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".ico",
            ".webp",
            ".pdf",
            ".zip",
            ".tar",
            ".gz",
            ".exe",
            ".dll",
            ".so",
            ".pyc",
            ".pyo",
            ".whl",
            ".egg",
            ".db",
            ".sqlite",
        ):
            parts.append(
                f'\n<file path="{p}">\n(Binary file — {_human_size(size)}. '
                f"Use the scratchpad to process it.)\n</file>"
            )
            continue

        try:
            content = p.read_text(errors="replace")
        except Exception:
            parts.append(f'\n<file path="{p}">\n(Could not read file.)\n</file>')
            continue

        parts.append(f'\n<file path="{p}">\n{content}\n</file>')

    return "\n".join(parts)


def _format_clipboard_image_message(
    uploaded: object, user_text: str = ""
) -> list[dict]:
    """Build a multimodal LLM message for a clipboard image upload.

    Returns a list of content blocks (image + text) so the LLM can see
    the image directly. The file path is included so the LLM can pass
    it to the scratchpad if deeper processing is needed.
    """
    import base64

    text = (
        user_text.strip()
        if user_text
        else "I've pasted an image from my clipboard. Analyze it."
    )
    text += (
        f"\n\nThe image is also saved at: {uploaded.path}\n"
        f"({uploaded.width}x{uploaded.height}, {_human_size(uploaded.size_bytes)}). "
        f"If you need to process it programmatically, use that path in the scratchpad."
    )

    # Read and base64-encode the saved PNG
    image_data = Path(uploaded.path).read_bytes()
    b64 = base64.standard_b64encode(image_data).decode("ascii")

    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            },
        },
        {
            "type": "text",
            "text": text,
        },
    ]


async def _ensure_clipboard(console: Console) -> bool:
    """Check clipboard support; offer to install Pillow if missing.

    Returns True if clipboard is ready to use, False otherwise.
    """
    reason = clipboard_unavailable_reason()
    if reason is None:
        return True
    if reason == "unsupported_platform":
        console.print("[anton.warning]Clipboard is not supported on this platform.[/]")
        return False
    # reason == "missing_pillow"
    console.print("[anton.muted]Clipboard image support requires Pillow.[/]")
    answer = console.input("[bold]Install Pillow now? (y/n):[/] ").strip().lower()
    if answer not in ("y", "yes"):
        console.print("[anton.muted]Skipped.[/]")
        return False
    console.print("[anton.muted]Installing Pillow...[/]")
    import subprocess

    proc = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: subprocess.run(
            ["uv", "pip", "install", "--python", sys.executable, "Pillow"],
            capture_output=True,
            timeout=120,
        ),
    )
    if proc.returncode == 0:
        console.print("[anton.success]Pillow installed. Clipboard is now available.[/]")
        return True
    else:
        # Fallback: try pip directly
        proc = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                [sys.executable, "-m", "pip", "install", "Pillow"],
                capture_output=True,
                timeout=120,
            ),
        )
        if proc.returncode == 0:
            console.print(
                "[anton.success]Pillow installed. Clipboard is now available.[/]"
            )
            return True
        console.print("[anton.error]Failed to install Pillow.[/]")
        return False


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.0f}{unit}" if unit == "B" else f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def _remove_engine_block(text: str, slug: str) -> str:
    """Return *text* with any YAML datasource block for *slug* removed."""
    cleaned = []
    prev = 0
    for m in _YAML_BLOCK_RE.finditer(text):
        try:
            data = _yaml.safe_load(m.group(3))
            is_dup = isinstance(data, dict) and str(data.get("engine", "")) == slug
        except Exception:
            is_dup = False
        if is_dup:
            pre = text[prev : m.start()].rstrip()
            pre = _re.sub(r"\n---\s*$", "", pre)
            cleaned.append(pre)
        else:
            cleaned.append(text[prev : m.end()])
        prev = m.end()
    cleaned.append(text[prev:])
    return "".join(cleaned)


async def _handle_add_custom_datasource(
    console: Console,
    name: str,
    registry,
    session: "ChatSession",
    *,
    known_service: bool = False,
):
    """Ask for the tool name, use the LLM to identify required fields, then collect credentials."""

    console.print()
    if name:
        tool_name = name
    else:
        tool_name = await prompt_or_cancel(
            "(anton) What is the name of the tool or service?",
        )
        if not tool_name or not tool_name.strip():
            return None
        tool_name = tool_name.strip()

    if known_service:
        # LLM already recognised this service — skip the auth question
        user_answer = ""
        console.print("[anton.muted]        Working out the connection details…[/]")
    else:
        user_answer = await prompt_or_cancel(
            f"(anton) How do you authenticate with {tool_name}? "
            "Describe what credentials you have (don't paste actual values)",
        )
        if not user_answer or not user_answer.strip():
            return None
        console.print()
        console.print("[anton.muted]    Got it — working out the connection details…[/]")

    llm_prompt = f"The user wants to connect to {repr(tool_name)}."
    if user_answer:
        llm_prompt += f" They said: {user_answer}"
    else:
        llm_prompt += " Determine the standard authentication fields for this service."
    llm_prompt += (
        "\n\nReturn ONLY valid JSON (no markdown fences, no commentary):\n"
        '{"display_name":"Human-readable name","pip":"pip-package or empty string",'
        '"test_snippet":"python code that tests the connection using os.environ vars DS_FIELDNAME (uppercase field name with DS_ prefix) and prints ok on success, or empty string if untestable",'
        '"fields":[{"name":"snake_case_name","value":"value if given inline else empty",'
        '"secret":true or false,"required":true or false,"description":"what it is"}]}'
    )

    try:
        response = await session._llm.plan(
            system="You are a data source connection expert.",
            messages=[
                {
                    "role": "user",
                    "content": llm_prompt,
                }
            ],
            max_tokens=1024,
        )
        text = response.content.strip()
        # Keep 
        text = _re.sub(r"^```[^\n]*\n|```\s*$", "", text, flags=_re.MULTILINE).strip()
        data = _json.loads(text)
    except Exception:
        console.print(
            "[anton.warning]        Couldn't identify connection details. Try again.[/]"
        )
        console.print()
        return None

    test_snippet = str(data.get("test_snippet", "")).strip()
    raw_fields = data.get("fields") or []
    fields: list[DatasourceField] = []
    for f in raw_fields:
        if not isinstance(f, dict) or not f.get("name"):
            continue
        fields.append(
            DatasourceField(
                name=f["name"],
                required=bool(f.get("required", True)),
                secret=bool(f.get("secret", False)),
                description=str(f.get("description", "")),
            )
        )

    if not fields:
        console.print("[anton.warning]    Couldn't identify any connection fields.[/]")
        console.print()
        return None

    display_name = str(data.get("display_name", name))
    pip_pkg = str(data.get("pip", ""))

    # Show summary
    console.print()
    console.print("      [bold]── What I'll save ──────────────────────────[/]")
    credentials: dict[str, str] = {}
    for f, raw in zip(fields, raw_fields):
        inline_value = str(raw.get("value", "")).strip()
        if f.secret and inline_value:
            console.print(
                f"        • [bold]{f.name:<14}[/] (secret — provided, stored securely)"
            )
            credentials[f.name] = inline_value
        elif f.secret:
            console.print(
                f"        • [bold]{f.name:<14}[/] (secret — I'll ask for this)"
            )
        else:
            val_display = inline_value or "[anton.muted]<to be collected>[/]"
            console.print(f"        • [bold]{f.name:<14}[/] {val_display}")
            if inline_value:
                credentials[f.name] = inline_value
    console.print()

    # Offer help before collecting credentials
    help_answer = await prompt_or_cancel(
        "(anton) Do you need instructions on how to obtain these credentials?",
        choices=["y", "n"], default="n",
    )
    if help_answer is None:
        return None
    if help_answer.strip().lower() == "y":
        await _show_credential_help(
            console, session, display_name, None, fields,
        )

    # Prompt for any secret fields not provided inline
    for f, raw in zip(fields, raw_fields):
        if not f.secret:
            continue
        if str(raw.get("value", "")).strip():
            continue
        value = await prompt_or_cancel(f"(anton) {f.name}", password=True)
        if value is None:
            return None
        if value:
            credentials[f.name] = value

    # Prompt for any required non-secret fields not provided inline
    for f, raw in zip(fields, raw_fields):
        if f.secret:
            continue
        if not f.required:
            continue
        if f.name in credentials:
            continue
        value = await prompt_or_cancel(f"(anton) {f.name}")
        if value is None:
            return None
        if value:
            credentials[f.name] = value

    # Offer to collect optional non-secret fields
    for f, raw in zip(fields, raw_fields):
        if f.secret or f.required or f.name in credentials:
            continue
        value = await prompt_or_cancel(f"(anton) {f.name} (optional — press Enter to skip)")
        if value is None:
            return None
        if value:
            credentials[f.name] = value

    if not credentials:
        console.print("[anton.warning]        No credentials collected. Aborting.[/]")
        console.print()
        return None

    # Build engine slug and write definition to ~/.anton/datasources.md
    slug = _re.sub(r"[^\w]", "_", display_name.lower()).strip("_")
    field_lines = "\n".join(
        f"  - {{ name: {f.name}, required: {str(f.required).lower()}, "
        f'secret: {str(f.secret).lower()}, description: "{f.description}" }}'
        for f in fields
    )
    test_snippet_yaml = ""
    if test_snippet:
        indented = "\n".join(f"  {line}" for line in test_snippet.splitlines())
        test_snippet_yaml = f"test_snippet: |\n{indented}\n"

    yaml_block = (
        f"\n---\n\n## {display_name}\n"
        "```yaml\n"
        f"engine: {slug}\n"
        f"display_name: {display_name}\n"
        + (f"pip: {pip_pkg}\n" if pip_pkg else "")
        + f"fields:\n{field_lines}\n"
        + test_snippet_yaml
        + "```\n"
    )
    user_ds_path = Path("~/.anton/datasources.md").expanduser()
    tmp_path = user_ds_path.with_suffix(".tmp")

    # Write to temp, validate it parses, then rename atomically
    existing = (
        user_ds_path.read_text(encoding="utf-8") if user_ds_path.is_file() else ""
    )

    existing = _remove_engine_block(existing, slug)

    tmp_path.write_text(existing + yaml_block, encoding="utf-8")

    parsed = registry.validate_file(tmp_path)
    if slug in parsed:
        import shutil

        shutil.move(str(tmp_path), str(user_ds_path))
    else:
        tmp_path.unlink(missing_ok=True)
        console.print(
            "[anton.warning]Could not validate engine definition — "
            "credentials saved but engine not written to datasources.md.[/]"
        )

    registry.reload()
    engine_def = registry.get(slug)
    if engine_def is None:
        # Fallback: construct inline so the flow can continue even if parse failed
        engine_def = DatasourceEngine(
            engine=slug,
            display_name=display_name,
            pip=pip_pkg,
            fields=fields,
            test_snippet=test_snippet,
        )

    # All required fields must be present before the caller saves credentials
    missing_required = [f.name for f in fields if f.required and f.name not in credentials]
    if missing_required:
        console.print(
            "[anton.warning]    Cannot save — missing required fields: "
            f"{', '.join(missing_required)}. Aborting.[/]"
        )
        console.print()
        return None

    return engine_def, credentials


async def _run_connection_test(
    console: "Console",
    scratchpads: "ScratchpadManager",
    vault: "DataVault",
    engine_def: "DatasourceEngine",
    credentials: dict[str, str],
    retry_fields: "list[DatasourceField]",
) -> bool:
    """Inject flat DS_* vars, run engine_def.test_snippet, restore env.

    Returns True on success, False if the user declines retry after failure.
    Mutates credentials in-place when the user re-enters secrets on retry.
    """
    import os as _os

    while True:
        console.print()
        console.print("[anton.cyan](anton)[/] Got it. Testing connection…")

        vault.clear_ds_env()
        for key, value in credentials.items():
            _os.environ[f"DS_{key.upper()}"] = value
        _register_secret_vars(engine_def)  # flat mode, for scrubbing during test

        try:
            pad = await scratchpads.get_or_create("__datasource_test__")
            await pad.reset()
            if engine_def.pip:
                if isinstance(engine_def.pip, list):
                    pip_pkgs = engine_def.pip
                else:
                    # Split space-separated package strings into individual packages
                    pip_pkgs = engine_def.pip.split()
                install_result = await pad.install_packages(pip_pkgs)
                if "failed" in (install_result or "").lower():
                    console.print()
                    console.print(f"[anton.warning](anton)[/] Package install issue: {install_result[:200]}")

            # Run the test, retry up to 2 times on ModuleNotFoundError
            cell = None
            for _attempt in range(3):
                cell = await pad.execute(engine_def.test_snippet)
                if cell.error and "ModuleNotFoundError" in cell.error:
                    # Extract the missing module and try to install it
                    import re as _re
                    _match = _re.search(r"No module named '([^']+)'", cell.error)
                    if _match:
                        _missing = _match.group(1).split(".")[0]
                        await pad.install_packages([_missing])
                        continue
                break
        finally:
            _restore_namespaced_env(vault)

        if cell.error or (cell.stdout.strip() != "ok" and cell.stderr.strip()):
            error_text = cell.error or cell.stderr.strip() or cell.stdout.strip()
            last_line = next(
                (ln for ln in reversed(error_text.splitlines()) if ln.strip()), error_text
            )
            console.print()
            console.print("[anton.warning](anton)[/] ✗ Connection failed.")
            console.print()
            console.print(f"        Error: {last_line}")
            console.print()
            retry = await prompt_or_cancel(
                "(anton) Would you like to re-enter your credentials?",
                choices=["y", "n"], default="n",
            )
            if retry is None or retry.strip().lower() != "y":
                return False
            console.print()
            for f in retry_fields:
                if not f.secret:
                    continue
                value = await prompt_or_cancel(f"(anton) {f.name}", password=True)
                if value is None:
                    return False
                if value:
                    credentials[f.name] = value
            continue

        console.print("[anton.success]        ✓ Connected successfully![/]")
        return True


async def _show_credential_help(
    console: Console,
    session: "ChatSession",
    service_name: str,
    current_field,
    all_fields: list,
) -> None:
    """Use the LLM to explain how to obtain credentials."""
    field_descriptions = ", ".join(
        f"{f.name} ({f.description})" for f in all_fields
    )
    storage_note = (
        "The credentials will be stored securely in Anton's Local Vault — "
        "do NOT suggest storage tips, password managers, or safe-keeping advice."
    )
    if current_field is not None:
        prompt = (
            f"I'm connecting to {service_name} and need to provide: {field_descriptions}\n\n"
            f"I need help with the '{current_field.name}' field"
            f" ({current_field.description}).\n\n"
            "Give me a brief step-by-step guide on where and how to get this credential. "
            f"Be concise — numbered steps, no fluff. {storage_note}"
        )
        heading = f"[anton.cyan](anton)[/] How to get [bold]{current_field.name}[/]:"
    else:
        prompt = (
            f"I'm connecting to {service_name} and need these credentials: {field_descriptions}\n\n"
            "Give me a brief step-by-step guide on where and how to obtain each of these. "
            f"Be concise — numbered steps, no fluff. {storage_note}"
        )
        heading = f"[anton.cyan](anton)[/] How to get credentials for [bold]{service_name}[/]:"

    console.print()
    console.print("[anton.muted]        Looking up instructions…[/]")

    try:
        resp = await session._llm.plan(
            system="You are a helpful assistant that guides users through obtaining credentials for services.",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=512,
        )
        help_text = (resp.content or "").strip()
    except Exception:
        help_text = "Sorry, couldn't fetch help right now. Try checking the service's documentation."

    from rich.markdown import Markdown as _Markdown
    from rich.padding import Padding

    console.print()
    console.print(heading)
    console.print()
    console.print(Padding(_Markdown(help_text), (0, 0, 0, 8)))
    console.print()


async def _handle_connect_datasource(
    console: Console,
    scratchpads: ScratchpadManager,
    session: "ChatSession",
    datasource_name: str | None = None,
    prefill: str | None = None,
) -> "ChatSession":
    """
    Connect a data source by entering credentials, either for a new name or re-entering for an existing one.
    """

    vault = DataVault()
    registry = DatasourceRegistry()
    
    if datasource_name is not None:
        _parsed = parse_connection_slug(
            datasource_name, [e.engine for e in registry.all_engines()], vault=vault
        )
        if _parsed is None:
            console.print(
                f"[anton.warning]Invalid slug '{datasource_name}'. "
                "Expected format: engine-name.[/]"
            )
            console.print()
            return session
        edit_engine, edit_name = _parsed
        existing = vault.load(edit_engine, edit_name)
        if existing is None:
            console.print(
                f"[anton.warning]No connection '{datasource_name}' found in Local Vault.[/]"
            )
            console.print()
            return session
        engine_def = registry.get(edit_engine)
        if engine_def is None:
            console.print(
                f"[anton.warning]Unknown engine '{edit_engine}'. "
                "Cannot update credentials.[/]"
            )
            console.print()
            return session

        console.print()
        console.print(
            f"[anton.cyan](anton)[/] Editing [bold]\"{datasource_name}\"[/bold]"
            f" ({engine_def.display_name})."
        )
        console.print("[anton.muted]        Press Enter to keep the current value.[/]")
        console.print()

        # Detect which fields to present (handle auth_method=choice)
        active_fields = engine_def.fields
        if engine_def.auth_method == "choice" and engine_def.auth_methods:
            for am in engine_def.auth_methods:
                am_field_names = {af.name for af in am.fields}
                if any(k in am_field_names for k in existing):
                    active_fields = am.fields
                    break
            if not active_fields:
                active_fields = engine_def.auth_methods[0].fields

        # Start from existing values; let user update field-by-field
        credentials: dict[str, str] = dict(existing)
        for f in active_fields:
            current = existing.get(f.name, "")
            field_label = f"(anton) {f.name}"
            if not f.required:
                field_label += " (optional)"

            if f.secret:
                masked = "••••••••" if current else ""
                label = f"{field_label} [{masked}]" if masked else field_label
                value = await prompt_or_cancel(label, password=True)
                if value is None:
                    return session
                if value:
                    credentials[f.name] = value
                # else: keep existing (already in credentials)
            elif current:
                value = await prompt_or_cancel(
                    f"{field_label}",
                    default=current,
                )
                if value is None:
                    return session
                credentials[f.name] = value if value else current
            elif f.default:
                value = await prompt_or_cancel(
                    f"{field_label}",
                    default=f.default,
                )
                if value is None:
                    return session
                if value:
                    credentials[f.name] = value
            else:
                value = await prompt_or_cancel(field_label)
                if value is None:
                    return session
                if value:
                    credentials[f.name] = value

        if engine_def.test_snippet:
            if not await _run_connection_test(
                console, scratchpads, vault, engine_def, credentials, active_fields
            ):
                return session

        vault.save(edit_engine, edit_name, credentials)
        _restore_namespaced_env(vault)
        _register_secret_vars(engine_def, engine=edit_engine, name=edit_name)
        console.print()
        console.print(
            f'        Credentials updated for [bold]"{datasource_name}"[/bold].'
        )
        console.print()
        console.print(
            "[anton.muted]        You can now ask me questions about your data.[/]"
        )
        console.print()
        session._history.append(
            {
                "role": "assistant",
                "content": (
                    f"I've updated the credentials for the {engine_def.display_name} connection "
                    f'"{datasource_name}" in the Local Vault.'
                ),
            }
        )
        return session

    console.print()
    all_engines = registry.all_engines()
    popular_engines = [e for e in all_engines if e.popular and not e.custom]
    other_engines = [e for e in all_engines if not e.popular and not e.custom]
    custom_engines = [e for e in all_engines if e.custom]
    display_engines = popular_engines + other_engines + custom_engines

    saved_connections = vault.list_connections()
    # Build deduplicated list of saved connection display entries
    saved_entries: list[tuple[str, str]] = []  # (slug, display_name)
    for c in saved_connections:
        slug = f"{c['engine']}-{c['name']}"
        engine = registry.get(c["engine"])
        label = engine.display_name if engine else c["engine"]
        saved_entries.append((slug, label))

    def _print_sections() -> None:
        console.print(
            "[anton.cyan](anton)[/] Choose a data source:\n"
        )
        console.print("       [bold]  Primary")
        console.print(
            "         [bold]  0.[/bold] Custom datasource"
            " (connect anything via API, SQL, or MCP)\n"
        )
        if popular_engines:
            console.print("       [bold]  Most popular")
            for i, e in enumerate(popular_engines, 1):
                console.print(f"          [bold]{i:>2}.[/bold] {e.display_name}")
            console.print()
        if saved_entries:
            start = len(popular_engines) + 1
            console.print("       [bold]  Recent connections")
            for i, (slug, label) in enumerate(saved_entries, start):
                console.print(f"          [bold]{i:>2}.[/bold] {label}")
            console.print()

    def _print_all() -> None:
        console.print(
            "[anton.cyan](anton)[/] All data sources (★ = popular):\n"
        )
        console.print("       [bold]  Primary")
        console.print(
            "         [bold]  0.[/bold] Custom datasource"
            " (connect anything via API, SQL, or MCP)\n"
        )
        for i, e in enumerate(display_engines, 1):
            star = " ★" if e.popular else ""
            console.print(f"          [bold]{i:>2}.[/bold] {e.display_name}{star}")
        console.print()

    if prefill:
        answer = prefill
    else:
        _print_sections()
        console.print(
            "       [anton.muted]Don't see yours? Type a datasource name (e.g., GitHub, Gmail, Jira, ...)\n"
            "       It can be virtually any datasource — we'll figure out the details together.[/]"
        )
        console.print()
        answer = await prompt_or_cancel(
            "(anton) Enter a number or type a datasource name",
        )
        if answer is None:
            return session
        if answer.strip().lower() == "all":
            console.print()
            _print_all()
            answer = await prompt_or_cancel(
                "(anton) Enter a number or type a name",
            )
            if answer is None:
                return session

    stripped_answer = answer.strip()
    known_slugs = {
        f"{c['engine']}-{c['name']}": c for c in vault.list_connections()
    }
    if stripped_answer in known_slugs:
        conn = known_slugs[stripped_answer]
        _restore_namespaced_env(vault)
        session._active_datasource = stripped_answer
        recon_engine_def = registry.get(conn["engine"])
        if recon_engine_def:
            _register_secret_vars(recon_engine_def, engine=conn["engine"], name=conn["name"])
            engine_label = recon_engine_def.display_name
        else:
            engine_label = conn["engine"]
        console.print()
        console.print(
            f'[anton.success]        ✓ Reconnected to [bold]"{stripped_answer}"[/bold].[/]'
        )
        console.print()
        session._history.append(
            {
                "role": "assistant",
                "content": (
                    f'I\'ve reconnected to the {engine_label} connection "{stripped_answer}" '
                    f"in the Local Vault. I can now query this data source when needed."
                ),
            }
        )
        return session

    engine_def: DatasourceEngine | None = None
    custom_source = False
    llm_recognised = False
    # Saved connections are numbered after popular engines
    saved_start = len(popular_engines) + 1
    max_num = len(popular_engines) + len(saved_entries)

    if stripped_answer.isdigit() or (stripped_answer.lstrip("-").isdigit()):
        pick_num = int(stripped_answer)
        if pick_num == 0:
            custom_source = True
        elif 1 <= pick_num <= len(popular_engines):
            engine_def = popular_engines[pick_num - 1]
        elif saved_entries and saved_start <= pick_num <= max_num:
            # User picked a recent connection type — start a new connection of that engine
            picked_slug, picked_label = saved_entries[pick_num - saved_start]
            picked_engine = picked_slug.split("-", 1)[0]
            engine_def = registry.get(picked_engine)
            if engine_def is None:
                custom_source = True
        else:
            console.print(
                f"[anton.warning](anton)[/] '{stripped_answer}' is out of range. "
                f"Please enter 0\u2013{max_num}.[/]"
            )
            console.print()
            return session

    if engine_def is None and not custom_source:
        engine_def = registry.find_by_name(stripped_answer)
        # if exact match not found, try substring match against display and engine names
        if engine_def is None:
            needle = stripped_answer.lower()
            candidates = [
                e
                for e in all_engines
                if needle in e.display_name.lower() or needle in e.engine.lower()
            ]
            if len(candidates) == 1:
                engine_def = candidates[0]
            elif len(candidates) > 1:
                console.print()
                console.print(
                    f"[anton.warning](anton)[/] '{stripped_answer}' matches multiple engines — "
                    "which one did you mean?"
                )
                console.print()
                for i, e in enumerate(candidates, 1):
                    console.print(f"        {i}. {e.display_name}")
                console.print()
                pick = await prompt_or_cancel("(anton) Enter a number")
                if pick is None:
                    return session
                pick = (pick or "").strip()
                try:
                    engine_def = candidates[int(pick) - 1]
                except (ValueError, IndexError):
                    console.print("[anton.warning]Invalid choice. Aborting.[/]")
                    console.print()
                    return session
        # Ask the LLM to identify the datasource
        if engine_def is None:
            engine_names = [e.display_name for e in all_engines]
            try:
                console.print()
                console.print("[anton.muted]        Looking up datasource…[/]")
                llm_resp = await session._llm.plan(
                    system="You are a datasource identification assistant.",
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"The user typed: {stripped_answer!r}\n"
                                f"Known datasources: {engine_names!r}\n\n"
                                "If the user input clearly matches one of the known datasources, "
                                "reply with EXACTLY: MATCH:<display_name>\n"
                                "If it does NOT match any known datasource but you recognise it "
                                "as a real service/tool, reply with EXACTLY: CUSTOM\n"
                                "If you don't recognise it at all, reply with EXACTLY: UNKNOWN\n"
                                "Reply with only one of those three forms, nothing else."
                            ),
                        }
                    ],
                    max_tokens=64,
                )
                llm_text = (llm_resp.content or "").strip()
            except Exception:
                llm_text = "UNKNOWN"

            llm_recognised = llm_text == "CUSTOM" or llm_text.startswith("MATCH:")

            if llm_text.startswith("MATCH:"):
                matched_name = llm_text[len("MATCH:"):].strip()
                matched_engine = next(
                    (e for e in all_engines if e.display_name == matched_name), None
                )
                if matched_engine is not None:
                    if matched_name.lower() != stripped_answer.lower():
                        confirm = await prompt_or_cancel(
                            f'(anton) Did you mean "{matched_name}"?',
                            choices=["y", "n"], default="y",
                        )
                        if confirm is not None and confirm.strip().lower() == "y":
                            engine_def = matched_engine
                    else:
                        engine_def = matched_engine

            if engine_def is None:
                custom_source = True

    if custom_source:
        result = await _handle_add_custom_datasource(
            console, stripped_answer if not stripped_answer.isdigit() else "", registry, session,
            known_service=llm_recognised,
        )
        if result is None:
            return session
        engine_def, credentials = result
        if engine_def.test_snippet:
            if not await _run_connection_test(
                console, scratchpads, vault, engine_def, credentials, engine_def.fields
            ):
                return session
        conn_name = uuid.uuid4().hex[:8]
        vault.save(engine_def.engine, conn_name, credentials)
        slug = f"{engine_def.engine}-{conn_name}"
        _restore_namespaced_env(vault)
        session._active_datasource = slug
        _register_secret_vars(engine_def, engine=engine_def.engine, name=conn_name)
        console.print(
            f'        Credentials saved to Local Vault as [bold]"{slug}"[/bold].'
        )
        console.print()
        console.print(
            "[anton.muted]        You can now ask me questions about your data.[/]"
        )
        console.print()
        session._history.append(
            {
                "role": "assistant",
                "content": (
                    f'I\'ve saved a {engine_def.display_name} connection named "{slug}" '
                    f"to the Local Vault. I can now query this data source when needed."
                ),
            }
        )
        return session

    assert engine_def is not None  # custom_source path always returns before this line
    active_fields = engine_def.fields
    if engine_def.auth_method == "choice" and engine_def.auth_methods:
        console.print()
        console.print(
            f"[anton.cyan](anton)[/] How would you like to authenticate with "
            f"[bold]{engine_def.display_name}[/]?"
        )
        console.print()
        for i, am in enumerate(engine_def.auth_methods, 1):
            console.print(f"        {i}. {am.display}")
        console.print()
        choice_str = await prompt_or_cancel("(anton) Enter a number")
        if choice_str is None:
            return session
        choice_str = (choice_str or "").strip()
        try:
            choice_idx = int(choice_str) - 1
            chosen_method = engine_def.auth_methods[choice_idx]
        except (ValueError, IndexError):
            console.print("[anton.warning]Invalid choice. Aborting.[/]")
            console.print()
            return session
        active_fields = chosen_method.fields

    required_fields = [f for f in active_fields if f.required]
    optional_fields = [f for f in active_fields if not f.required]

    console.print()
    console.print(
        f"[anton.cyan](anton)[/] To connect [bold]{engine_def.display_name}[/], "
        "I'll need the following:"
    )
    console.print()

    if required_fields:
        console.print("        [bold]Required[/]      " + "─" * 39)
        for f in required_fields:
            console.print(
                f"        • [bold]{f.name:<12}[/] [anton.muted]— {f.description}[/]"
            )

    if optional_fields:
        console.print()
        console.print("        [bold]Optional[/]      " + "─" * 39)
        for f in optional_fields:
            console.print(
                f"        • [bold]{f.name:<12}[/] [anton.muted]— {f.description}[/]"
            )

    console.print()

    help_answer = await prompt_or_cancel(
        "(anton) Do you need instructions on how to obtain these credentials?",
        choices=["y", "n"], default="n",
    )
    if help_answer is None:
        return session
    if help_answer.strip().lower() == "y":
        await _show_credential_help(
            console, session, engine_def.display_name, None, active_fields,
        )

    field_name_set = {f.name.lower() for f in active_fields}

    while True:
        mode_answer = await prompt_or_cancel(
            "(anton) Do you have these available?",
            choices_display="y/n/list params", default="y",
        )
        if mode_answer is None:
            return session
        mode_answer = mode_answer.strip().lower()

        if mode_answer in ("y", "n"):
            break

        # Check if user gave valid comma-separated param names
        requested = {n.strip().lower() for n in mode_answer.split(",")}
        matched = [f for f in active_fields if f.name.lower() in requested]
        if matched:
            break

        console.print(
            "[anton.warning]        Please enter y, n, or a comma-separated list of parameter names "
            f"({', '.join(f.name for f in active_fields)}).[/]"
        )
        console.print()

    if mode_answer == "n":
        console.print()
        console.print(
            "[anton.cyan](anton)[/] No problem. Which parameters do you have? "
            "I'll save a partial connection now, and you can fill in the rest later "
            "with [bold]/edit[/]."
        )
        console.print()
        console.print("       Provide what you have (press enter to skip any field):")
        console.print()
        fields_to_collect = active_fields
        partial = True
    elif mode_answer == "y":
        fields_to_collect = active_fields
        partial = False
    else:
        fields_to_collect = matched
        partial = False

    console.print()
    credentials: dict[str, str] = {}

    for f in fields_to_collect:
        if f.secret:
            value = await prompt_or_cancel(f"(anton) {f.name}", password=True)
        elif f.default:
            value = await prompt_or_cancel(f"(anton) {f.name}", default=f.default)
        else:
            value = await prompt_or_cancel(f"(anton) {f.name}")
        if value is None:
            return session
        if value:
            credentials[f.name] = value

    if partial:
        auto_name = uuid.uuid4().hex[:8]
        vault.save(engine_def.engine, auto_name, credentials)
        slug = f"{engine_def.engine}-{auto_name}"
        console.print()
        console.print(
            f"[anton.muted]Partial connection saved to Local Vault as "
            f'[bold]"{slug}"[/bold]. '
            f"Run [bold]/edit {slug}[/bold] to complete it when you're ready.[/]"
        )
        console.print()
        return session
    
    if engine_def.test_snippet:
        if not await _run_connection_test(
            console, scratchpads, vault, engine_def, credentials, active_fields
        ):
            return session

    conn_name = registry.derive_name(engine_def, credentials)
    if not conn_name:
        conn_name = uuid.uuid4().hex[:8]

    slug = f"{engine_def.engine}-{conn_name}"

    if vault.load(engine_def.engine, conn_name) is not None:
        console.print()
        console.print(
            f'[anton.warning](anton)[/] A connection [bold]"{slug}"[/bold] already exists.'
        )
        console.print()
        choice = await prompt_or_cancel(
            f"(anton) {_PROMPT_RECONNECT_CANCEL}",
        )
        if choice is None or choice.strip().lower() != "reconnect":
            console.print("[anton.muted]Cancelled.[/]")
            console.print()
            return session
        _restore_namespaced_env(vault)
        _register_secret_vars(engine_def, engine=engine_def.engine, name=conn_name)
        console.print()
        console.print(
            f'[anton.success]        ✓ Reconnected to [bold]"{slug}"[/bold].[/]'
        )
        console.print()
        session._history.append(
            {
                "role": "assistant",
                "content": (
                    f'I\'ve reconnected to the {engine_def.display_name} connection "{slug}" '
                    f"in the Local Vault. I can now query this data source when needed."
                ),
            }
        )
        return session

    vault.save(engine_def.engine, conn_name, credentials)
    _restore_namespaced_env(vault)
    session._active_datasource = slug
    _register_secret_vars(engine_def, engine=engine_def.engine, name=conn_name)
    console.print(f'        Credentials saved to Local Vault as [bold]"{slug}"[/bold].')

    console.print()
    console.print(
        "[anton.muted]        You can now ask me questions about your data.[/]"
    )
    console.print()

    # Inject a brief assistant message so the LLM is aware of the new connection
    session._history.append(
        {
            "role": "assistant",
            "content": (
                f'I\'ve saved a {engine_def.display_name} connection named "{slug}" '
                f"to the Local Vault. I can now query this data source when needed."
            ),
        }
    )
    return session


def _handle_list_data_sources(console: Console) -> None:
    """Print all saved Local Vault connections in a table with status."""
    from rich.table import Table

    vault = DataVault()
    registry = DatasourceRegistry()
    conns = vault.list_connections()
    console.print()
    if not conns:
        console.print("[anton.muted]No data sources connected yet.[/]")
        console.print("[anton.muted]Use /connect to add one.[/]")
        console.print()
        return

    table = Table(title="Local Vault — Saved Connections", show_lines=False)
    table.add_column("Name", style="bold")
    table.add_column("Source")
    table.add_column("Status")

    for c in conns:
        slug = f"{c['engine']}-{c['name']}"
        engine_def = registry.get(c["engine"])
        source = engine_def.display_name if engine_def else c["engine"]
        fields = vault.load(c["engine"], c["name"]) or {}

        if not fields:
            status = "[yellow]incomplete[/]"
        elif engine_def and engine_def.auth_method != "choice":
            required = [f.name for f in engine_def.fields if f.required]
            missing = [name for name in required if name not in fields]
            status = "[yellow]incomplete[/]" if missing else "[green]saved[/]"
        else:
            # choice-auth engine or unknown engine: presence of any field = saved
            status = "[green]saved[/]"

        table.add_row(slug, source, status)

    console.print(table)
    console.print()


async def _handle_remove_data_source(console: Console, slug: str) -> None:
    """Delete a connection from the Local Vault by slug (engine-name)."""
    vault = DataVault()
    registry = DatasourceRegistry()

    if not slug:
        connections = vault.list_connections()
        if not connections:
            console.print("[anton.muted]No saved connections to remove.[/]")
            console.print()
            return
        console.print()
        console.print("[anton.cyan](anton)[/] Which connection do you want to remove?\n")
        for i, c in enumerate(connections, 1):
            conn_slug = f"{c['engine']}-{c['name']}"
            engine_def = registry.get(c["engine"])
            label = engine_def.display_name if engine_def else c["engine"]
            console.print(f"          [bold]{i:>2}.[/bold] {conn_slug} [dim]({label})[/]")
        console.print()
        choices = [str(i) for i in range(1, len(connections) + 1)]
        pick = await prompt_or_cancel("(anton) Enter a number", choices=choices)
        if pick is None:
            console.print("[anton.muted]Cancelled.[/]")
            console.print()
            return
        picked = connections[int(pick) - 1]
        slug = f"{picked['engine']}-{picked['name']}"

    _parsed = parse_connection_slug(slug, [e.engine for e in registry.all_engines()], vault=vault)
    if _parsed is None:
        console.print(
            f"[anton.warning]Invalid name '{slug}'. Use engine-name format.[/]"
        )
        console.print()
        return
    engine, name = _parsed
    if vault.load(engine, name) is None:
        console.print(f"[anton.warning]No connection '{slug}' found.[/]")
        console.print()
        return

    confirm = await prompt_or_cancel(
        f"(anton) Remove '{slug}' from Local Vault?",
        choices=["y", "n"], default="n",
    )
    if confirm is not None and confirm.strip().lower() == "y":
        vault.delete(engine, name)
        _restore_namespaced_env(vault)
        engine_def = registry.get(engine)
        if engine_def is not None and engine_def.custom:
            remaining = [
                c for c in vault.list_connections() if c["engine"] == engine
            ]
            if not remaining:
                user_path = DatasourceRegistry._USER_PATH
                if user_path.is_file():
                    updated = _remove_engine_block(
                        user_path.read_text(encoding="utf-8"), engine
                    )
                    user_path.write_text(updated, encoding="utf-8")
                    registry.reload()
        console.print(f"[anton.success]Removed {slug}.[/]")
    else:
        console.print("[anton.muted]Cancelled.[/]")
    console.print()


async def _handle_test_datasource(
    console: Console,
    scratchpads: ScratchpadManager,
    slug: str,
) -> None:
    """Test an existing Local Vault connection by running its test_snippet."""
    if not slug:
        console.print(
            "[anton.warning]Usage: /test <engine-name>[/]"
        )
        console.print()
        return

    vault = DataVault()
    registry = DatasourceRegistry()
    _parsed = parse_connection_slug(slug, [e.engine for e in registry.all_engines()], vault=vault)
    if _parsed is None:
        console.print(
            f"[anton.warning]Invalid name '{slug}'. Use engine-name format.[/]"
        )
        console.print()
        return
    engine, name = _parsed
    fields = vault.load(engine, name)
    if fields is None:
        console.print(
            f"[anton.warning]No connection '{slug}' found in Local Vault.[/]"
        )
        console.print()
        return

    engine_def = registry.get(engine)
    if engine_def is None:
        console.print(
            f"[anton.warning]Unknown engine '{engine}'. Cannot test.[/]"
        )
        console.print()
        return

    if not engine_def.test_snippet:
        console.print(
            f"[anton.warning]No test snippet defined for '{engine}'. Cannot test.[/]"
        )
        console.print()
        return

    console.print()
    console.print(
        f"[anton.cyan](anton)[/] Testing connection [bold]{slug}[/bold]…"
    )

    vault.clear_ds_env()
    vault.inject_env(engine, name, flat=True)
    _register_secret_vars(engine_def)  # flat names for scrubbing during test

    cell = None
    try:
        pad = await scratchpads.get_or_create("__datasource_test__")
        await pad.reset()
        if engine_def.pip:
            await pad.install_packages([engine_def.pip])
        cell = await pad.execute(engine_def.test_snippet)
    finally:
        _restore_namespaced_env(vault)

    if cell is None or cell.error or (
        cell.stdout.strip() != "ok" and cell.stderr.strip()
    ):
        error_text = ""
        if cell is not None:
            error_text = cell.error or cell.stderr.strip() or cell.stdout.strip()
        first_line = (
            next((ln for ln in error_text.splitlines() if ln.strip()), error_text)
            if error_text
            else "unknown error"
        )
        console.print()
        console.print(
            f"[anton.warning](anton)[/] ✗ Connection test failed for"
            f" [bold]{slug}[/bold]."
        )
        console.print()
        console.print(f"        Error: {first_line}")
    else:
        console.print(
            f"[anton.success]        ✓ Connection test passed for"
            f" [bold]{slug}[/bold]![/]"
        )
    console.print()


def _handle_theme(console: Console, arg: str) -> None:
    """Switch the color theme (light/dark)."""
    import os
    from anton.channel.theme import detect_color_mode, build_rich_theme

    current = detect_color_mode()

    if not arg:
        new_mode = "light" if current == "dark" else "dark"
    elif arg in ("light", "dark"):
        new_mode = arg
    else:
        console.print(f"[anton.warning]Unknown theme '{arg}'. Use: /theme light | /theme dark[/]")
        console.print()
        return

    os.environ["ANTON_THEME"] = new_mode
    # Re-apply the theme to the console
    console._theme_stack.push_theme(build_rich_theme(new_mode))
    console.print(f"[anton.success]Theme set to {new_mode}.[/]")
    console.print()


def _print_slash_help(console: Console) -> None:
    """Print available slash commands."""
    console.print()

    console.print("[anton.cyan]Available commands:[/]")

    console.print("\n[bold]LLM Provider[/]")
    console.print("  [bold]/llm[/]      — Change LLM provider or API key")

    console.print("\n[bold]Data Connections[/]")
    console.print("  [bold]/connect[/]   — Connect a database or API to your Local Vault")
    console.print("  [bold]/list[/]      — List all saved connections")
    console.print("  [bold]/edit[/]      — Edit credentials for an existing connection")
    console.print("  [bold]/remove[/]    — Remove a saved connection")
    console.print("  [bold]/test[/]      — Test a saved connection")
    
    console.print("\n[bold]Workspace[/]")
    console.print("  [bold]/setup[/]     — Configure models and memory settings")
    console.print("  [bold]/memory[/]    — View memory status and usage")
    console.print("  [bold]/theme[/]     — Switch theme (light/dark)")

    console.print("\n[bold]Chat Tools[/]")
    console.print("  [bold]/paste[/]     — Attach an image from your clipboard")
    console.print("  [bold]/resume[/]    — Continue a previous session")
    
    console.print("\n[bold]General[/]")
    console.print("  [bold]/help[/]      — Show this help menu")
    console.print("  [bold]exit[/]       — Exit the chat")
    
    console.print()


class _EscapeWatcher:
    """Detect Escape keypress during streaming via cbreak terminal mode."""

    def __init__(self, on_cancel: Callable[[], None] | None = None) -> None:
        self.cancelled = asyncio.Event()
        self._on_cancel = on_cancel
        self._task: asyncio.Task | None = None
        self._old_settings: list | None = None
        self._stop = False
        self._paused = False

    async def __aenter__(self) -> _EscapeWatcher:
        if sys.platform != "win32" and sys.stdin.isatty():
            self._task = asyncio.create_task(self._watch())
        return self

    def pause(self) -> None:
        """Temporarily restore normal terminal mode for interactive prompts."""
        if self._paused or self._old_settings is None:
            return
        import termios
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSADRAIN, self._old_settings)
        self._paused = True

    def resume(self) -> None:
        """Re-enter cbreak mode after interactive prompts."""
        if not self._paused:
            return
        import tty
        fd = sys.stdin.fileno()
        tty.setcbreak(fd)
        self._paused = False

    async def __aexit__(self, *exc: object) -> None:
        self._stop = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            # Drain any leftover bytes (e.g. partial CPR responses) so they
            # don't leak into the next prompt_toolkit input session.
            # Only needed on Unix where _watch() was running (fcntl/termios
            # are not available on Windows).
            self._drain_stdin()

    @staticmethod
    def _drain_stdin() -> None:
        if sys.platform == "win32":
            return
        import fcntl

        fd = sys.stdin.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            while True:
                try:
                    if not os.read(fd, 1024):
                        break
                except BlockingIOError:
                    break
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)

    async def _watch(self) -> None:
        if sys.platform == "win32":
            return
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            loop = asyncio.get_running_loop()
            while not self._stop:
                # Skip reading while paused (interactive tool running)
                if self._paused:
                    await asyncio.sleep(0.1)
                    continue
                # Use select with a short timeout so the executor thread
                # can check the stop flag and exit cleanly — a bare
                # os.read() blocks forever and survives task cancellation,
                # which causes it to steal bytes from the next prompt.
                ready = await loop.run_in_executor(
                    None, lambda: select.select([fd], [], [], 0.1)[0]
                )
                if not ready:
                    continue
                ch = os.read(fd, 1)
                if ch == b"\x1b":
                    # Arrow keys and other special keys send escape
                    # sequences starting with \x1b (e.g. \x1b[A for
                    # up-arrow).  Wait briefly to see if more bytes
                    # follow — if they do, this is a multi-byte
                    # sequence, not a standalone Escape press.
                    followup = await loop.run_in_executor(
                        None, lambda: select.select([fd], [], [], 0.05)[0]
                    )
                    if followup:
                        # Consume the rest of the escape sequence and
                        # ignore it (not a bare ESC key).
                        os.read(fd, 32)
                        continue
                    if self._on_cancel is not None:
                        self._on_cancel()
                    self.cancelled.set()
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_settings)


class _ClosingSpinner:
    """Animated spinner shown while scratchpad processes are being killed."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._live: object | None = None

    def start(self) -> None:
        from rich.live import Live
        from rich.spinner import Spinner
        from rich.text import Text

        spinner = Spinner(
            "dots", text=Text(" Closing scratchpad processes…", style="anton.muted")
        )
        self._live = Live(
            spinner, console=self._console, refresh_per_second=6, transient=True
        )
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None


async def _agent_zero(console: Console, session: "ChatSession", settings) -> str | None:
    """First-run staged demo. Runs the backup script in a real scratchpad cell.

    Returns "_AGENT_ZERO_DONE" if demo ran, None if skipped/failed.
    """
    import os as _os
    import time as _time

    script_path = Path(__file__).resolve().parent / "demo_data" / "nvda_btc_scratchpad_backup.py"
    if not script_path.is_file():
        return None

    # Clear screen
    _os.system("cls" if sys.platform == "win32" else "clear")

    console.print()
    _line1 = "All set! To test things out, I\u2019ll pull NVIDIA vs Bitcoin data from"
    _line2 = "the web and build you a 5-year investment comparison dashboard."
    console.print("[anton.prompt]anton>[/] ", end="")
    for ch in _line1:
        console.file.write(ch)
        console.file.flush()
        _time.sleep(0.02)
    console.print()
    console.print("       ", end="")
    for ch in _line2:
        console.file.write(ch)
        console.file.flush()
        _time.sleep(0.02)
    console.print()
    console.print()
    console.print()

    answer = await prompt_or_cancel(
        "(anton) Run analysis, or skip straight to chatting?",
        choices_display="run/skip",
        default="run",
        allow_cancel=True,
    )
    if answer is None:
        return None

    answer_text = (answer or "").strip().lower()

    # Classify: does the user want to run it?
    _skip_words = {"no", "n", "skip", "nah", "pass", "nope", "later", "chat", "straight"}
    _go_words = {"yes", "y", "ok", "sure", "go", "yeah", "yep", "run", "do it", "let's go", "lets go", "go for it"}

    wants_demo = None
    for w in _go_words:
        if w in answer_text:
            wants_demo = True
            break
    if wants_demo is None:
        for w in _skip_words:
            if w in answer_text:
                wants_demo = False
                break
    if wants_demo is None:
        # Default to yes if ambiguous
        wants_demo = True if not answer_text else True

    if not wants_demo:
        console.print()
        console.print("  [anton.muted]All good! Ask me anything \u2014 data questions, dashboards, analysis, you name it.[/]")
        console.print()
        return None

    # Typed message with ellipsis animation
    console.print()
    from anton.channel.theme import get_palette as _gp3
    _c = _gp3().cyan
    _r, _g, _b = int(_c[1:3], 16), int(_c[3:5], 16), int(_c[5:7], 16)
    _ac = f"\033[1;38;2;{_r};{_g};{_b}m"
    _ar = "\033[0m"

    _prefix = f"{_ac}anton>{_ar} "
    _typed_msg = "Perfect! Fetching live data, crunching numbers, and building the dashboard"
    console.file.write(_prefix)
    console.file.flush()
    for ch in _typed_msg:
        console.file.write(ch)
        console.file.flush()
        _time.sleep(0.02)

    # Ellipsis + spinner for ~10 seconds
    console.file.write("...\n")
    console.file.flush()

    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text

    with Live(
        Spinner("dots", text=Text("", style="anton.muted"), style="anton.cyan"),
        console=console,
        refresh_per_second=10,
        transient=True,
    ):
        await asyncio.sleep(10)
    console.print()

    # Read the script and patch for scratchpad execution.
    # 1. __file__ doesn't exist inside exec() — set it so os.path.dirname works
    # 2. Override OUTPUT_PATH to write to .anton/output/ instead of demo_data/
    code = script_path.read_text()
    output_dir = str(Path(settings.workspace_path) / ".anton" / "output")
    output_html = str(Path(output_dir) / "nvda_btc_dashboard.html")
    code = (
        f"import os as _os; _os.makedirs({output_dir!r}, exist_ok=True)\n"
        f"__file__ = {str(script_path)!r}\n"
        + code
    )
    # Replace the OUTPUT_PATH line so the dashboard goes to .anton/output/
    code = code.replace(
        'OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvda_btc_dashboard.html")',
        f'OUTPUT_PATH = {output_html!r}',
    )

    from anton.scratchpad import Cell
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text

    pad = await session._scratchpads.get_or_create("main")

    # Pre-install dependencies so the main script doesn't fail mid-run
    install_spinner = Text("  Installing dependencies (yfinance, pandas, numpy)...", style="anton.muted")
    with Live(
        Spinner("dots", text=install_spinner, style="anton.cyan"),
        console=console,
        refresh_per_second=10,
        transient=True,
    ):
        await pad.install_packages(["yfinance", "pandas", "numpy"])
    console.print(f"  [anton.success]\u2714[/] [anton.muted]Dependencies ready[/]")

    spinner_text = Text("  Scratchpad(Building NVDA vs BTC dashboard...)", style="anton.muted")
    cell = None
    with Live(
        Spinner("dots", text=spinner_text, style="anton.cyan"),
        console=console,
        refresh_per_second=10,
        transient=True,
    ):
        async for item in pad.execute_streaming(
            code,
            description="Build NVDA vs BTC investment dashboard",
            estimated_time="~2 min",
            estimated_seconds=120,
        ):
            if isinstance(item, str):
                # Progress message from the script — update spinner
                spinner_text = Text(f"  Scratchpad({item})", style="anton.muted")
            elif isinstance(item, Cell):
                cell = item

    if cell is None or cell.error:
        err = cell.error if cell else "No result"
        console.print()
        err_line = err.strip().split("\n")[-1] if err else err
        console.print(f"[anton.error]  Demo encountered an issue: {err_line}[/]")
        console.print("[anton.muted]  You can still use Anton normally.[/]")
        console.print()
        return None

    console.print(f"  [anton.success]\u2714[/] [anton.muted]Dashboard built successfully[/]")

    # Inject context into session history so the LLM knows data is live
    _demo_stdout = (cell.stdout or "")[:3000]
    session._history.append({
        "role": "assistant",
        "content": (
            "I built an interactive NVIDIA vs Bitcoin 5-year investment dashboard. "
            "The dashboard HTML is at: " + output_html + "\n\n"
            "The scratchpad 'main' is still running with all data loaded in memory:\n"
            "- prices DataFrame (monthly OHLCV, returns, cumulative, drawdowns)\n"
            "- risk DataFrame (annual stats, Sharpe, Sortino, Calmar, win rate)\n"
            "- annual DataFrame (year-by-year breakdown)\n"
            "- mc DataFrame (1,000-path Monte Carlo, 60 months)\n"
            "- scorecard DataFrame (12-metric head-to-head comparison)\n\n"
            "All variables are live in the 'main' scratchpad — the user can ask "
            "follow-up questions and I can use the existing data without re-fetching.\n\n"
            f"Script output:\n{_demo_stdout}"
        ),
    })

    # Show findings — typed out like the intro message
    console.print()
    _lines = [
        "Everything worked! I pulled 5 years of data from Yahoo Finance,",
        "ran the numbers on NVIDIA vs Bitcoin, and built you a full",
        "interactive dashboard \u2014 it\u2019s open in your browser.",
        "",
        "6 tabs to explore: Performance \u00b7 Risk \u00b7 Monte Carlo \u00b7 Annual \u00b7",
        "Scorecard \u00b7 Decision.",
        "",
        "My take? If I had money to put down, NVIDIA wins this one.",
        "",
    ]
    from anton.channel.theme import get_palette as _gp2
    _cyan = _gp2().cyan
    # Convert hex color to ANSI 24-bit escape
    _r, _g, _b = int(_cyan[1:3], 16), int(_cyan[3:5], 16), int(_cyan[5:7], 16)
    _ansi_cyan = f"\033[1;38;2;{_r};{_g};{_b}m"
    _ansi_reset = "\033[0m"

    for li, line in enumerate(_lines):
        console.file.write("  ")
        for ch in line:
            console.file.write(ch)
            console.file.flush()
            _time.sleep(0.015)
        console.file.write("\n")
        console.file.flush()
    console.print()
    console.print("[anton.muted] Ask me follow-ups, a completely different question, or connect your own data (using the /connect command).[/]")
    console.print("[anton.muted] What\u2019s next, boss?[/]")
    console.print()

    return "_AGENT_ZERO_DONE"


def _persist_first_run_done(settings) -> None:
    """Write ANTON_FIRST_RUN_DONE=true to ~/.anton/.env."""
    from pathlib import Path

    env_path = Path.home() / ".anton" / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    existing = env_path.read_text() if env_path.is_file() else ""
    if "ANTON_FIRST_RUN_DONE" not in existing:
        with env_path.open("a") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write("ANTON_FIRST_RUN_DONE=true\n")
    settings.first_run_done = True


_GREETING_EXAMPLES = [
    (
        "Go through my inbox, find every subscription I never read,\n"
        "       and build me a dashboard with unsubscribe links right there."
    ),
    (
        "Classify my last 200 emails \u2014 what actually needs my\n"
        "       attention vs what\u2019s noise? Show me a breakdown."
    ),
    (
        "Show me all my meetings next month \u2014 who\u2019s taking most\n"
        "       of my time? Build me a dashboard."
    ),
    (
        "Find all recurring meetings I haven\u2019t attended in 3+ months \u2014\n"
        "       should I drop them? Give me a report."
    ),
    (
        "Compare AAPL, NVDA, and TSLA over the last year \u2014\n"
        "       full interactive investment dashboard."
    ),
    (
        "What\u2019s the latest tech news today? Pull the headlines\n"
        "       and summarize what actually matters."
    ),
    (
        "I have a spreadsheet with sales data \u2014 analyze it and\n"
        "       build me an interactive dashboard with the key insights."
    ),
    (
        "Help me plan a trip to Tokyo \u2014 flights, hotels, budget,\n"
        "       all in one dashboard."
    ),
]


def _desktop_greeting(console: Console, settings) -> None:
    """First-time greeting for desktop app users. Types out a welcome + example."""
    import random
    import time as _time

    from anton.channel.theme import get_palette as _gp

    _c = _gp().cyan
    _r, _g, _b = int(_c[1:3], 16), int(_c[3:5], 16), int(_c[5:7], 16)
    _ac = f"\033[1;38;2;{_r};{_g};{_b}m"
    _ar = "\033[0m"

    example = random.choice(_GREETING_EXAMPLES)  # noqa: S311

    console.print()

    # Line 1: "Hi Boss! I'm Anton — here to help with anything."
    _line1 = "Hi Boss! I\u2019m Anton \u2014 here to help with anything."
    console.file.write(f"{_ac}anton>{_ar} ")
    for ch in _line1:
        console.file.write(ch)
        console.file.flush()
        _time.sleep(0.02)
    console.file.write("\n")
    console.file.flush()

    _time.sleep(0.3)

    # Line 2: blank
    console.file.write("\n")

    # Line 3: "For example, try something like:"
    _line2 = "For example, try something like:"
    console.file.write("       ")
    for ch in _line2:
        console.file.write(ch)
        console.file.flush()
        _time.sleep(0.02)
    console.file.write("\n")
    console.file.flush()

    _time.sleep(0.2)

    # Line 4: blank
    console.file.write("\n")

    # Line 5+: the example (quoted, italic feel)
    console.file.write("       \u201c")
    for ch in example:
        console.file.write(ch)
        console.file.flush()
        _time.sleep(0.015)
    console.file.write("\u201d\n")
    console.file.flush()

    console.print()

    _persist_first_run_done(settings)


def run_chat(
    console: Console, settings: AntonSettings, *, resume: bool = False, first_run: bool = False, desktop_first_run: bool = False
) -> None:
    """Launch the interactive chat REPL."""
    asyncio.run(_chat_loop(console, settings, resume=resume, first_run=first_run, desktop_first_run=desktop_first_run))


async def _chat_loop(
    console: Console, settings: AntonSettings, *, resume: bool = False, first_run: bool = False, desktop_first_run: bool = False
) -> None:
    from anton.context.self_awareness import SelfAwarenessContext
    from anton.llm.client import LLMClient
    from anton.memory.cortex import Cortex
    from anton.workspace import Workspace

    # Use a mutable container so closures always see the current client
    state: dict = {"llm_client": LLMClient.from_settings(settings)}

    # Self-awareness context (legacy, kept for backward compatibility)
    self_awareness = SelfAwarenessContext(Path(settings.context_dir))

    # Workspace for anton.md and secret vault
    workspace = Workspace(settings.workspace_path)
    workspace.apply_env_to_process()

    # Inject all Local Vault connections as namespaced DS_* env vars so every
    # scratchpad subprocess inherits them. Must happen before any ChatSession is created.
    _dv = DataVault()
    _dreg = DatasourceRegistry()
    for _conn in _dv.list_connections():
        _dv.inject_env(_conn["engine"], _conn["name"])  # flat=False by default
        _edef = _dreg.get(_conn["engine"])
        if _edef is not None:
            _register_secret_vars(_edef, engine=_conn["engine"], name=_conn["name"])
    del _dv, _dreg

    global_memory_dir = Path.home() / ".anton" / "memory"
    project_memory_dir = settings.workspace_path / ".anton" / "memory"

    cortex = Cortex(
        global_dir=global_memory_dir,
        project_dir=project_memory_dir,
        mode=settings.memory_mode,
        llm_client=state["llm_client"],
    )

    # Reconsolidation: migrate legacy memory formats on first run
    from anton.memory.reconsolidator import needs_reconsolidation, reconsolidate

    project_anton_dir = settings.workspace_path / ".anton"
    if needs_reconsolidation(project_anton_dir):
        actions = reconsolidate(project_anton_dir)
        if actions:
            console.print(f"[anton.muted]  Memory migration: {actions[0]}[/]")

    # Background compaction if needed
    if cortex.needs_compaction():
        asyncio.create_task(cortex.compact_all())

    from anton.memory.episodes import EpisodicMemory

    episodes_dir = settings.workspace_path / ".anton" / "episodes"
    episodic = EpisodicMemory(episodes_dir, enabled=settings.episodic_memory)
    if episodic.enabled:
        episodic.start_session()

    from anton.memory.history_store import HistoryStore

    history_store = HistoryStore(episodes_dir)
    current_session_id = episodic._session_id if episodic.enabled else None

    # Clean up old clipboard uploads
    uploads_dir = Path(settings.workspace_path) / ".anton" / "uploads"
    cleanup_old_uploads(uploads_dir)

    # Build runtime context so the LLM knows what it's running on
    runtime_context = _build_runtime_context(settings)

    coding_api_key = (
        settings.anthropic_api_key
        if settings.coding_provider == "anthropic"
        else settings.openai_api_key
    ) or ""
    session = ChatSession(
        state["llm_client"],
        self_awareness=self_awareness,
        cortex=cortex,
        episodic=episodic,
        runtime_context=runtime_context,
        workspace=workspace,
        console=console,
        coding_provider=settings.coding_provider,
        coding_api_key=coding_api_key,
        coding_base_url=settings.openai_base_url or "",
        history_store=history_store,
        session_id=current_session_id,
        proactive_dashboards=settings.proactive_dashboards,
    )

    # Handle --resume flag at startup
    if resume:
        session, resumed_id = await _handle_resume(
            console,
            settings,
            state,
            self_awareness,
            cortex,
            workspace,
            session,
            episodic=episodic,
            history_store=history_store,
        )
        if resumed_id:
            current_session_id = resumed_id

    if desktop_first_run and not settings.first_run_done:
        try:
            _desktop_greeting(console, settings)
        except Exception:
            pass

    _agent_zero_query: str | None = None
    if first_run and not settings.first_run_done:
        try:
            _agent_zero_result = await _agent_zero(console, session, settings)
            if _agent_zero_result == "_AGENT_ZERO_DONE":
                _agent_zero_query = None
            else:
                _agent_zero_query = _agent_zero_result
        except Exception:
            pass
        _persist_first_run_done(settings)

    if not first_run and not desktop_first_run:
        console.print(f"[anton.cyan_dim] {'━' * 40}[/]")
    console.print("[anton.muted] type '/help' for commands or 'exit' to quit.[/]")
    console.print()

    from anton.analytics import send_event
    _query_count = 0
    _total_questions = 0  # tracks first 10 questions for time estimates

    from anton.chat_ui import StreamDisplay

    toolbar = {"stats": "", "status": ""}
    display = StreamDisplay(console, toolbar=toolbar)
    last_token_status: TokenLimitInfo | None = None
    last_token_status_checked_at: float | None = None

    def _bottom_toolbar():
        stats = toolbar["stats"]
        status = toolbar["status"]
        if not stats and not status:
            return ""
        try:
            width = os.get_terminal_size().columns
        except OSError:
            width = 80
        gap = width - len(status) - len(stats)
        if gap < 1:
            gap = 1
        line = status + " " * gap + stats
        return HTML(f"\n<style fg='#555570'>{line}</style>")

    pt_style = PTStyle.from_dict(
        {
            "bottom-toolbar": "noreverse nounderline bg:default",
        }
    )

    prompt_session: PromptSession[str] = PromptSession(
        mouse_support=False,
        bottom_toolbar=_bottom_toolbar,
        style=pt_style,
    )

    try:
        while True:
            # Memory confirmation UX — show pending lessons before prompt
            if session._pending_memory_confirmations:
                pending = session._pending_memory_confirmations
                console.print("[anton.muted]Lessons learned from this session:[/]")
                for i, engram in enumerate(pending, 1):
                    console.print(f"  [bold]{i}.[/] [{engram.kind}] {engram.text}")
                console.print()
                confirm = (
                    console.input("[bold]Save to memory? (y/n/pick numbers):[/] ")
                    .strip()
                    .lower()
                )
                if confirm in ("y", "yes"):
                    if cortex is not None:
                        await cortex.encode(pending)
                    console.print("[anton.muted]Saved.[/]")
                elif confirm in ("n", "no"):
                    console.print("[anton.muted]Discarded.[/]")
                else:
                    # Parse number selections like "1 3" or "1,3"
                    try:
                        nums = [
                            int(x.strip())
                            for x in confirm.replace(",", " ").split()
                            if x.strip().isdigit()
                        ]
                        selected = [
                            pending[n - 1] for n in nums if 1 <= n <= len(pending)
                        ]
                        if selected and cortex is not None:
                            await cortex.encode(selected)
                            console.print(
                                f"[anton.muted]Saved {len(selected)} entries.[/]"
                            )
                        else:
                            console.print("[anton.muted]Discarded.[/]")
                    except (ValueError, IndexError):
                        console.print("[anton.muted]Discarded.[/]")
                session._pending_memory_confirmations = []
                console.print()

            try:
                from anton.channel.theme import get_palette as _gp
                _you_color = _gp().user_prompt
                user_input = await prompt_session.prompt_async(
                    [(f"bold fg:{_you_color}", "you>"), ("", " ")]
                )
            except EOFError:
                break

            stripped = user_input.strip()
            # message_content holds what we send to the LLM — may be str or
            # list[dict] (multimodal content blocks for images).
            message_content: str | list[dict] | None = None

            # Empty input → check clipboard for an image
            if not stripped:
                if is_clipboard_supported():
                    clip = grab_clipboard()
                    if clip.image:
                        uploaded = save_clipboard_image(clip.image.image, uploads_dir)
                        console.print(
                            f"  [anton.muted]attached: clipboard image "
                            f"({uploaded.width}x{uploaded.height}, "
                            f"{_human_size(uploaded.size_bytes)})[/]"
                        )
                        message_content = _format_clipboard_image_message(uploaded)
                    elif clip.file_paths:
                        stripped = _format_file_message("", clip.file_paths, console)
                if not stripped and message_content is None:
                    continue

            if message_content is None and stripped.lower() in ("exit", "quit", "bye"):
                break

            # Detect dragged file paths early — a dragged absolute path like
            # "/Users/foo/bar.txt" starts with "/" and would otherwise be
            # mistaken for a slash command.
            if message_content is None and stripped.startswith("/"):
                dropped_early = _parse_dropped_paths(stripped)
                if dropped_early:
                    stripped = _format_file_message(stripped, dropped_early, console)
                    message_content = stripped

            # Slash command dispatch
            if message_content is None and stripped.startswith("/"):
                parts = stripped.split(maxsplit=1)
                cmd = parts[0].lower()
                if cmd == "/llm":
                    session = await _handle_setup_models(
                        console,
                        settings,
                        workspace,
                        state,
                        self_awareness,
                        cortex,
                        session,
                        episodic=episodic,
                        history_store=history_store,
                        session_id=current_session_id,
                    )
                    continue
                elif cmd == "/minds":
                    session = await _handle_connect(
                        console,
                        settings,
                        workspace,
                        state,
                        self_awareness,
                        cortex,
                        session,
                        episodic=episodic,
                    )
                    continue
                elif cmd == "/setup":
                    session = await _handle_setup(
                        console,
                        settings,
                        workspace,
                        state,
                        self_awareness,
                        cortex,
                        session,
                        episodic=episodic,
                        history_store=history_store,
                        session_id=current_session_id,
                    )
                    continue
                elif cmd == "/memory":
                    _handle_memory(console, settings, cortex, episodic=episodic)
                    continue
                elif cmd == "/connect":
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    session = await _handle_connect_datasource(
                        console,
                        session._scratchpads,
                        session,
                        prefill=arg or None,
                    )
                    continue
                elif cmd == "/list":
                    _handle_list_data_sources(console)
                    continue
                elif cmd == "/remove":
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    await _handle_remove_data_source(console, arg)
                    continue
                elif cmd == "/edit":
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    if not arg:
                        console.print(
                            "[anton.warning]Usage: /edit <engine-name>[/]"
                        )
                        console.print()
                    else:
                        session = await _handle_connect_datasource(
                            console,
                            session._scratchpads,
                            session,
                            datasource_name=arg,
                        )
                    continue
                elif cmd == "/test":
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    await _handle_test_datasource(
                        console, session._scratchpads, arg
                    )
                    continue
                elif cmd == "/resume":
                    session, resumed_id = await _handle_resume(
                        console,
                        settings,
                        state,
                        self_awareness,
                        cortex,
                        workspace,
                        session,
                        episodic=episodic,
                        history_store=history_store,
                    )
                    if resumed_id:
                        current_session_id = resumed_id
                    continue
                elif cmd == "/theme":
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    _handle_theme(console, arg)
                    continue
                elif cmd == "/help":
                    _print_slash_help(console)
                    continue
                elif cmd == "/paste":
                    if not await _ensure_clipboard(console):
                        continue
                    clip = grab_clipboard()
                    if clip.image:
                        uploaded = save_clipboard_image(clip.image.image, uploads_dir)
                        console.print(
                            f"  [anton.muted]attached: clipboard image "
                            f"({uploaded.width}x{uploaded.height}, "
                            f"{_human_size(uploaded.size_bytes)})[/]"
                        )
                        user_text = parts[1] if len(parts) > 1 else ""
                        message_content = _format_clipboard_image_message(
                            uploaded, user_text
                        )
                        # Fall through to turn_stream (don't continue)
                    else:
                        console.print("[anton.warning]No image found on clipboard.[/]")
                        continue
                else:
                    console.print(f"[anton.warning]Unknown command: {cmd}[/]")
                    continue

            # Detect dragged file paths and reformat the message
            if message_content is None:
                dropped = _parse_dropped_paths(stripped)
                if dropped:
                    stripped = _format_file_message(stripped, dropped, console)

            # Use multimodal content if set, otherwise the text string
            if message_content is None:
                message_content = stripped

            _query_count += 1
            _total_questions += 1
            if _query_count == 1:
                send_event(settings, "anton_first_query")
            else:
                send_event(settings, "anton_query")

            display.start()
            t0 = time.monotonic()
            ttft: float | None = None
            total_input = 0
            total_output = 0
            session._cancel_event.clear()

            try:
                async with _EscapeWatcher(on_cancel=display.show_cancelling) as esc:
                    session._escape_watcher = esc
                    async for event in session.turn_stream(message_content):
                        if esc.cancelled.is_set():
                            session._cancel_event.set()
                            raise KeyboardInterrupt
                        if isinstance(event, StreamTextDelta):
                            if ttft is None:
                                ttft = time.monotonic() - t0
                            display.append_text(event.text)
                        elif isinstance(event, StreamToolResult):
                            display.show_tool_result(event.content)
                        elif isinstance(event, StreamToolUseStart):
                            display.on_tool_use_start(event.id, event.name)
                        elif isinstance(event, StreamToolUseDelta):
                            display.on_tool_use_delta(event.id, event.json_delta)
                        elif isinstance(event, StreamToolUseEnd):
                            display.on_tool_use_end(event.id)
                        elif isinstance(event, StreamTaskProgress):
                            display.update_progress(
                                event.phase, event.message, event.eta_seconds
                            )
                        elif isinstance(event, StreamContextCompacted):
                            display.show_context_compacted(event.message)
                        elif isinstance(event, StreamComplete):
                            total_input += event.response.usage.input_tokens
                            total_output += event.response.usage.output_tokens

                elapsed = time.monotonic() - t0
                parts = []

                if settings.minds_api_key and settings.minds_url:
                    #TODO: Lets check if this is best solution
                    now = time.monotonic()
                    if last_token_status_checked_at is None or (now - last_token_status_checked_at) >= TOKEN_STATUS_CACHE_TTL:
                        last_token_status = check_minds_token_limits(
                            settings.minds_url.rstrip("/"),
                            settings.minds_api_key,
                            verify=settings.minds_ssl_verify,
                        )
                        last_token_status_checked_at = now
                    if last_token_status.billing_cycle_limit > 0:
                        _pct = last_token_status.billing_cycle_used * 100 // last_token_status.billing_cycle_limit
                        parts.append(f"{last_token_status.billing_cycle_used:,} / {last_token_status.billing_cycle_limit:,} ({_pct}%)")

                parts.append(f"{elapsed:.1f}s")
                if not settings.minds_api_key and not settings.minds_url:
                    parts.append(f"{total_input} in / {total_output} out")
                if ttft is not None:
                    parts.append(f"TTFT {int(ttft * 1000)}ms")
                toolbar["stats"] = "  ".join(parts)
                toolbar["status"] = ""
                display.finish()
                if settings.minds_api_key and settings.minds_url and last_token_status is not None and last_token_status.status is TokenLimitStatus.WARNING:
                    pct = int(last_token_status.used / last_token_status.limit * 100) if last_token_status.limit else 80
                    console.print(
                        f"[anton.warning]Approaching token limit: {last_token_status.used:,} / "
                        f"{last_token_status.limit:,} tokens used ({pct}%). "
                        "Visit mdb.ai to upgrade your plan or top up your tokens.[/]"
                    )
                    console.print()
                if _query_count == 1:
                    send_event(settings, "anton_first_answer")
            except anthropic.AuthenticationError:
                display.abort()
                console.print()
                console.print(
                    "[anton.error]Invalid API key. Let's set up a new one.[/]"
                )
                settings.anthropic_api_key = None
                from anton.cli import _ensure_api_key

                _ensure_api_key(settings)
                session = _rebuild_session(
                    settings=settings,
                    state=state,
                    self_awareness=self_awareness,
                    cortex=cortex,
                    workspace=workspace,
                    console=console,
                    episodic=episodic,
                    history_store=history_store,
                    session_id=current_session_id,
                )
            except KeyboardInterrupt:
                display.abort()
                session.repair_history()
                # Kill any running scratchpad processes (they may have
                # spawned subprocesses that would otherwise be orphaned).
                if session._scratchpads.list_pads():
                    console.print()
                    _closing = _ClosingSpinner(console)
                    _closing.start()
                    try:
                        await session._scratchpads.close_all()
                    finally:
                        _closing.stop()
                else:
                    console.print()
                console.print("[anton.muted]Cancelled.[/]")
                console.print()
                # Cancel the turn but stay in the chat loop
                continue
            except Exception as exc:
                display.abort()
                console.print(f"[anton.error]Error: {exc}[/]")
                console.print()
                err_msg = str(exc)
                if "401" in err_msg or "403" in err_msg or "Authentication" in err_msg:
                    if Confirm.ask(
                        "  Would you like to set up new LLM credentials?",
                        default=True,
                        console=console,
                    ):
                        session = await _handle_setup_models(
                            console,
                            settings,
                            workspace,
                            state,
                            self_awareness,
                            cortex,
                            session,
                            episodic=episodic,
                            history_store=history_store,
                            session_id=current_session_id,
                        )
                    console.print()
    except KeyboardInterrupt:
        pass

    console.print()
    console.print("[anton.muted]See you.[/]")
    await session.close()
