from __future__ import annotations

import asyncio
import json as _json
import os
import urllib.error
import re as _re
import sys
import uuid
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
    TokenLimitExceeded,
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
    PUBLISH_TOOL,
    RECALL_TOOL,
    SCRATCHPAD_TOOL,
    dispatch_tool,
    format_cell_result,
    prepare_scratchpad_exec,
)
from anton.checks import TokenLimitInfo, TokenLimitStatus, check_minds_token_limits
from anton.commands.setup import (
    handle_memory,
    handle_setup,
    handle_setup_memory,
    handle_setup_models,
)
from anton.commands.ui import handle_theme, print_slash_help
from anton.utils.clipboard import (
    ensure_clipboard,
    format_clipboard_image_message,
    format_file_message,
    human_size,
)
from anton.commands.bug_report import handle_report_bug
from anton.chat_session import build_runtime_context, rebuild_session
from anton.commands.session import handle_resume
from anton.commands.datasource import (
    handle_list_data_sources,
    handle_remove_data_source,
    handle_connect_datasource,
    handle_test_datasource,
)
from anton.utils.prompt import (
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
from anton.data_vault import DataVault
from anton.utils.datasources import (
    build_datasource_context,
    register_secret_vars,
    restore_namespaced_env,
    remove_engine_block,
    scrub_credentials,
    parse_connection_slug,
)
from anton.datasource_registry import (
    DatasourceEngine,
    DatasourceField,
    DatasourceRegistry,
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
        self._escape_watcher: "EscapeWatcher | None" = None
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
        ds_ctx = build_datasource_context(active_only=self._active_datasource)
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
                scratchpad_tool["description"] += (
                    f"\n\nLessons from past sessions:\n{wisdom}"
                )

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
        tools.append(PUBLISH_TOOL)
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

                result_text = scrub_credentials(result_text)
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
                # Token/billing limit — don't retry, let the chat loop handle it
                if isinstance(_agent_exc, TokenLimitExceeded):
                    raise
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
        if (
            llm_response.stop_reason in ("max_tokens", "length")
            and not llm_response.tool_calls
        ):
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
                        elif tc.name == "connect_new_datasource" or (
                            tc.name == "publish_or_preview"
                            and tc.input.get("action") == "publish"
                        ):
                            # Interactive tool — pause spinner AND escape watcher
                            yield StreamTaskProgress(
                                phase="interactive",
                                message="",
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
                    result_text = scrub_credentials(result_text)
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
                if (
                    llm_response.stop_reason in ("max_tokens", "length")
                    and not llm_response.tool_calls
                ):
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

        action = await prompt_or_cancel(
            "(anton) Select", choices=["1", "2", "q"], default="q"
        )
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
        ds_pick = await prompt_or_cancel(
            "(anton) Select datasource", choices=ds_choices
        )
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
            all_datasources = list_datasources(minds_url, api_key, verify=ssl_verify)
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
        settings.planning_provider = "openai-compatible"
        settings.coding_provider = "openai-compatible"
        settings.planning_model = "_reason_"
        settings.coding_model = "_code_"
        # openai_api_key and openai_base_url are derived at runtime from
        # minds_api_key and minds_url via model_post_init — no need to persist them.
        settings.model_post_init(None)
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

    return rebuild_session(
        settings=settings,
        state=state,
        self_awareness=self_awareness,
        cortex=cortex,
        workspace=workspace,
        console=console,
        episodic=episodic,
    )


def _extract_html_title(path, re_module) -> str:
    """Extract <title> content from an HTML file. Returns '' if not found."""
    try:
        # Read only the first 4KB — title is always near the top
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(4096)
        m = re_module.search(
            r"<title[^>]*>(.*?)</title>", head, re_module.IGNORECASE | re_module.DOTALL
        )
        return m.group(1).strip() if m else ""
    except Exception:
        return ""


async def _handle_publish(
    console: Console,
    settings,
    workspace,
    file_arg: str = "",
) -> None:
    """Handle /publish command — publish an HTML report to the web."""
    import webbrowser
    from pathlib import Path

    from anton.publisher import publish

    console.print()

    # 1. Ensure Minds API key is available
    if not settings.minds_api_key:
        console.print(
            "  [anton.muted]To publish dashboards you need a free Minds account.[/]"
        )
        console.print()
        has_key = await prompt_or_cancel(
            "  Do you have an mdb.ai API key?",
            choices=["y", "n"],
            choices_display="y/n",
            default="y",
        )
        if has_key is None:
            console.print()
            return
        if has_key.lower() == "n":
            webbrowser.open(
                "https://mdb.ai/auth/realms/mindsdb/protocol/openid-connect/registrations"
                "?client_id=public-client&response_type=code&scope=openid"
                "&redirect_uri=https%3A%2F%2Fmdb.ai"
            )
            console.print()

        api_key = await prompt_or_cancel("  API key", password=True)
        if api_key is None or not api_key.strip():
            console.print()
            return
        api_key = api_key.strip()
        settings.minds_api_key = api_key
        if workspace:
            workspace.set_secret("ANTON_MINDS_API_KEY", api_key)
        console.print()

    # 2. Find the HTML file to publish
    import re

    output_dir = Path(settings.workspace_path) / ".anton" / "output"

    if file_arg:
        target = Path(file_arg)
        if not target.is_absolute():
            target = Path(settings.workspace_path) / file_arg
    else:
        # List HTML files sorted by modification time (most recent first)
        html_files = (
            sorted(
                output_dir.glob("*.html"), key=lambda f: f.stat().st_mtime, reverse=True
            )
            if output_dir.is_dir()
            else []
        )
        if not html_files:
            console.print("  [anton.warning]No HTML files found in .anton/output/[/]")
            console.print()
            return

        PAGE_SIZE = 10
        offset = 0

        while True:
            page = html_files[offset : offset + PAGE_SIZE]
            has_more = offset + PAGE_SIZE < len(html_files)

            console.print("  [anton.cyan]Available reports:[/]")
            console.print()
            for i, f in enumerate(page, offset + 1):
                title = _extract_html_title(f, re)
                label = title or f.name
                console.print(f"  [bold]{i}[/]  {label}  [anton.muted]{f.name}[/]")

            if has_more:
                console.print(
                    f"\n  [anton.muted]m  Show more ({len(html_files) - offset - PAGE_SIZE} remaining)[/]"
                )

            console.print()
            choice = await prompt_or_cancel("  Select", default="1")
            if choice is None:
                console.print()
                return

            if choice.strip().lower() == "m" and has_more:
                offset += PAGE_SIZE
                console.print()
                continue

            try:
                idx = int(choice) - 1
                if idx < 0 or idx >= len(html_files):
                    raise ValueError
                target = html_files[idx]
                break
            except (ValueError, IndexError):
                console.print("  [anton.warning]Invalid choice.[/]")
                console.print()
                return

    if not target.exists():
        console.print(f"  [anton.warning]File not found: {target}[/]")
        console.print()
        return

    # 3. Publish
    from rich.live import Live
    from rich.spinner import Spinner

    with Live(
        Spinner("dots", text="  Publishing...", style="anton.cyan"),
        console=console,
        transient=True,
    ):
        try:
            result = publish(
                target,
                api_key=settings.minds_api_key,
                publish_url=settings.publish_url,
                ssl_verify=settings.minds_ssl_verify,
            )
        except Exception as e:
            console.print(f"  [anton.error]Publish failed: {e}[/]")
            console.print()
            return

    view_url = result.get("view_url", "")
    console.print(f"  [anton.success]Published![/]")
    console.print(f"  [link={view_url}]{view_url}[/link]")
    console.print()

    if view_url:
        webbrowser.open(view_url)


async def _handle_unpublish(
    console: Console,
    settings,
    workspace,
) -> None:
    """Handle /unpublish command — list published reports and delete one."""
    from anton.publisher import list_published, unpublish

    console.print()

    # 1. Ensure Minds API key is available
    if not settings.minds_api_key:
        console.print(
            "  [anton.warning]No Minds API key configured. Run /publish first.[/]"
        )
        console.print()
        return

    # 2. Fetch published reports
    from rich.live import Live
    from rich.spinner import Spinner

    reports = []
    with Live(
        Spinner("dots", text="  Loading published reports...", style="anton.cyan"),
        console=console,
        transient=True,
    ):
        try:
            reports = list_published(
                api_key=settings.minds_api_key,
                publish_url=settings.publish_url,
                ssl_verify=settings.minds_ssl_verify,
            )
        except Exception as e:
            console.print(f"  [anton.error]Failed to list reports: {e}[/]")
            console.print()
            return

    if not reports:
        console.print("  [anton.muted]No published reports found.[/]")
        console.print()
        return

    # 3. Display paginated list
    PAGE_SIZE = 10
    offset = 0

    while True:
        page = reports[offset : offset + PAGE_SIZE]
        has_more = offset + PAGE_SIZE < len(reports)

        console.print("  [anton.cyan]Published reports:[/]")
        console.print()
        for i, r in enumerate(page, offset + 1):
            title = r.get("title", "Untitled")
            url = r.get("view_url", "")
            console.print(f"  [bold]{i}[/]  {title}  [anton.muted]{url}[/]")

        if has_more:
            console.print(
                f"\n  [anton.muted]m  Show more ({len(reports) - offset - PAGE_SIZE} remaining)[/]"
            )

        console.print()
        choice = await prompt_or_cancel("  Select report to unpublish")
        if choice is None:
            console.print()
            return

        if choice.strip().lower() == "m" and has_more:
            offset += PAGE_SIZE
            console.print()
            continue

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(reports):
                raise ValueError
            selected = reports[idx]
            break
        except (ValueError, IndexError):
            console.print("  [anton.warning]Invalid choice.[/]")
            console.print()
            return

    # 4. Confirm
    title = selected.get("title", "Untitled")
    console.print(f"  [anton.warning]This will remove:[/] {title}")
    confirm = await prompt_or_cancel(
        "  Are you sure?",
        choices=["y", "n"],
        choices_display="y/n",
        default="n",
    )
    if confirm is None or confirm != "y":
        console.print()
        return

    # 5. Delete
    with Live(
        Spinner("dots", text="  Removing...", style="anton.cyan"),
        console=console,
        transient=True,
    ):
        try:
            unpublish(
                selected["md5"],
                api_key=settings.minds_api_key,
                publish_url=settings.publish_url,
                ssl_verify=settings.minds_ssl_verify,
            )
        except Exception as e:
            console.print(f"  [anton.error]Failed to remove: {e}[/]")
            console.print()
            return

    console.print(f"  [anton.success]Removed:[/] {title}")
    console.print()


async def _agent_zero(console: Console, session: "ChatSession", settings) -> str | None:
    """First-run staged demo. Runs the backup script in a real scratchpad cell.

    Returns "_AGENT_ZERO_DONE" if demo ran, None if skipped/failed.
    """
    import os as _os
    import time as _time

    script_path = (
        Path(__file__).resolve().parent / "demo_data" / "nvda_btc_scratchpad_backup.py"
    )
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
    _skip_words = {
        "no",
        "n",
        "skip",
        "nah",
        "pass",
        "nope",
        "later",
        "chat",
        "straight",
    }
    _go_words = {
        "yes",
        "y",
        "ok",
        "sure",
        "go",
        "yeah",
        "yep",
        "run",
        "do it",
        "let's go",
        "lets go",
        "go for it",
    }

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
        console.print(
            "  [anton.muted]All good! Ask me anything \u2014 data questions, dashboards, analysis, you name it.[/]"
        )
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
    _typed_msg = (
        "Perfect! Fetching live data, crunching numbers, and building the dashboard"
    )
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
        f"__file__ = {str(script_path)!r}\n" + code
    )
    # Replace the OUTPUT_PATH line so the dashboard goes to .anton/output/
    code = code.replace(
        'OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvda_btc_dashboard.html")',
        f"OUTPUT_PATH = {output_html!r}",
    )

    from anton.scratchpad import Cell
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text

    pad = await session._scratchpads.get_or_create("main")

    # Pre-install dependencies so the main script doesn't fail mid-run
    install_spinner = Text(
        "  Installing dependencies (yfinance, pandas, numpy)...", style="anton.muted"
    )
    with Live(
        Spinner("dots", text=install_spinner, style="anton.cyan"),
        console=console,
        refresh_per_second=10,
        transient=True,
    ):
        await pad.install_packages(["yfinance", "pandas", "numpy"])
    console.print(f"  [anton.success]\u2714[/] [anton.muted]Dependencies ready[/]")

    spinner_text = Text(
        "  Scratchpad(Building NVDA vs BTC dashboard...)", style="anton.muted"
    )
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

    console.print(
        f"  [anton.success]\u2714[/] [anton.muted]Dashboard built successfully[/]"
    )

    # Inject context into session history so the LLM knows data is live
    _demo_stdout = (cell.stdout or "")[:3000]
    session._history.append(
        {
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
        }
    )

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
    console.print(
        "[anton.muted] Ask me follow-ups, a completely different question, or connect your own data (using the /connect command).[/]"
    )
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
    console: Console,
    settings: AntonSettings,
    *,
    resume: bool = False,
    first_run: bool = False,
    desktop_first_run: bool = False,
) -> None:
    """Launch the interactive chat REPL."""
    asyncio.run(
        _chat_loop(
            console,
            settings,
            resume=resume,
            first_run=first_run,
            desktop_first_run=desktop_first_run,
        )
    )


async def _chat_loop(
    console: Console,
    settings: AntonSettings,
    *,
    resume: bool = False,
    first_run: bool = False,
    desktop_first_run: bool = False,
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
    dv = DataVault()
    dreg = DatasourceRegistry()
    for conn in dv.list_connections():
        dv.inject_env(conn["engine"], conn["name"])  # flat=False by default
        edef = dreg.get(conn["engine"])
        if edef is not None:
            register_secret_vars(edef, engine=conn["engine"], name=conn["name"])
    del dv, dreg

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
    runtime_context = build_runtime_context(settings)

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
        session, resumed_id = await handle_resume(
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

    from anton.chat_ui import StreamDisplay, EscapeWatcher, ClosingSpinner

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
                            f"{human_size(uploaded.size_bytes)})[/]"
                        )
                        message_content = format_clipboard_image_message(uploaded)
                    elif clip.file_paths:
                        stripped = format_file_message("", clip.file_paths, console)
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
                    stripped = format_file_message(stripped, dropped_early, console)
                    message_content = stripped

            # Slash command dispatch
            if message_content is None and stripped.startswith("/"):
                parts = stripped.split(maxsplit=1)
                cmd = parts[0].lower()
                if cmd == "/llm":
                    session = await handle_setup_models(
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
                    session = await handle_setup(
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
                    handle_memory(console, settings, cortex, episodic=episodic)
                    continue
                elif cmd == "/connect":
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    session = await handle_connect_datasource(
                        console,
                        session._scratchpads,
                        session,
                        prefill=arg or None,
                    )
                    continue
                elif cmd == "/list":
                    handle_list_data_sources(console)
                    continue
                elif cmd == "/remove":
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    await handle_remove_data_source(console, arg)
                    continue
                elif cmd == "/edit":
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    if not arg:
                        console.print("[anton.warning]Usage: /edit <engine-name>[/]")
                        console.print()
                    else:
                        session = await handle_connect_datasource(
                            console,
                            session._scratchpads,
                            session,
                            datasource_name=arg,
                        )
                    continue
                elif cmd == "/test":
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    await handle_test_datasource(console, session._scratchpads, arg)
                    continue
                elif cmd == "/resume":
                    session, resumed_id = await handle_resume(
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
                    handle_theme(console, arg)
                    continue
                elif cmd == "/publish":
                    arg = parts[1].strip() if len(parts) > 1 else ""
                    await _handle_publish(console, settings, workspace, arg)
                    continue
                elif cmd == "/unpublish":
                    await _handle_unpublish(console, settings, workspace)
                    continue
                elif cmd == "/report-bug":
                    await handle_report_bug(
                        console, settings, workspace, session, cortex
                    )
                    continue
                elif cmd == "/help":
                    print_slash_help(console)
                    continue
                elif cmd == "/paste":
                    if not await ensure_clipboard(console):
                        continue
                    clip = grab_clipboard()
                    if clip.image:
                        uploaded = save_clipboard_image(clip.image.image, uploads_dir)
                        console.print(
                            f"  [anton.muted]attached: clipboard image "
                            f"({uploaded.width}x{uploaded.height}, "
                            f"{human_size(uploaded.size_bytes)})[/]"
                        )
                        user_text = parts[1] if len(parts) > 1 else ""
                        message_content = format_clipboard_image_message(
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
                    stripped = format_file_message(stripped, dropped, console)

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
                async with EscapeWatcher(on_cancel=display.show_cancelling) as esc:
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
                    # TODO: Lets check if this is best solution
                    now = time.monotonic()
                    if (
                        last_token_status_checked_at is None
                        or (now - last_token_status_checked_at)
                        >= TOKEN_STATUS_CACHE_TTL
                    ):
                        last_token_status = check_minds_token_limits(
                            settings.minds_url.rstrip("/"),
                            settings.minds_api_key,
                            verify=settings.minds_ssl_verify,
                        )
                        last_token_status_checked_at = now
                    if last_token_status.billing_cycle_limit > 0:
                        _pct = (
                            last_token_status.billing_cycle_used
                            * 100
                            // last_token_status.billing_cycle_limit
                        )
                        parts.append(
                            f"{last_token_status.billing_cycle_used:,} / {last_token_status.billing_cycle_limit:,} ({_pct}%)"
                        )

                parts.append(f"{elapsed:.1f}s")
                if not settings.minds_api_key and not settings.minds_url:
                    parts.append(f"{total_input} in / {total_output} out")
                if ttft is not None:
                    parts.append(f"TTFT {int(ttft * 1000)}ms")
                toolbar["stats"] = "  ".join(parts)
                toolbar["status"] = ""
                display.finish()
                if (
                    settings.minds_api_key
                    and settings.minds_url
                    and last_token_status is not None
                    and last_token_status.status is TokenLimitStatus.WARNING
                ):
                    pct = (
                        int(last_token_status.used / last_token_status.limit * 100)
                        if last_token_status.limit
                        else 80
                    )
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
                session = rebuild_session(
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
                    _closing = ClosingSpinner(console)
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
            except (TokenLimitExceeded, ConnectionError) as exc:
                display.abort()
                console.print()
                console.print(f"[anton.warning]{exc}[/]")
                console.print()
                choice = await prompt_or_cancel(
                    "  (anton) Switch LLM provider, update API key, or retry?",
                    choices=["setup", "retry", "s", "r"],
                    choices_display="setup/retry",
                    default="retry" if isinstance(exc, ConnectionError) else "setup",
                )
                if choice in ("setup", "s"):
                    session = await handle_setup_models(
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
                # retry or after setup — loop continues and re-sends the last message
                continue
            except Exception as exc:
                display.abort()
                console.print()
                console.print(f"[anton.error]{exc}[/]")
                console.print()
                choice = await prompt_or_cancel(
                    "  (anton) Switch LLM provider, or retry?",
                    choices=["setup", "retry", "s", "r"],
                    choices_display="setup/retry",
                    default="retry",
                )
                if choice in ("setup", "s"):
                    session = await handle_setup_models(
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
    except KeyboardInterrupt:
        pass

    console.print()
    console.print("[anton.muted]See you.[/]")
    await session.close()
