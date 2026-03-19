from __future__ import annotations

import asyncio
import os
import re
import sys
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
    MEMORIZE_TOOL,
    RECALL_TOOL,
    SCRATCHPAD_TOOL,
    dispatch_tool,
    format_cell_result,
    prepare_scratchpad_exec,
)

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

_VERIFIER_STATUS_PREFIXES = (
    "STATUS: COMPLETE",
    "STATUS: INCOMPLETE",
    "STATUS: STUCK",
)


def _strip_ollama_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)


def _is_parseable_verifier_status(text: str) -> bool:
    upper = text.upper()
    return any(prefix in upper for prefix in _VERIFIER_STATUS_PREFIXES)


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
        self._turn_count = sum(1 for m in self._history if m.get("role") == "user") if initial_history else 0
        self._history_store = history_store
        self._session_id = session_id
        self._cancel_event = asyncio.Event()
        self._scratchpads = ScratchpadManager(
            coding_provider=coding_provider,
            coding_model=getattr(llm_client, "coding_model", ""),
            coding_api_key=coding_api_key,
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
        self._history.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": "Cancelled by user.",
                }
                for tid in tool_ids
            ],
        })

    def _persist_history(self) -> None:
        """Save current history to disk if a history store is configured."""
        if self._history_store and self._session_id:
            self._history_store.save(self._session_id, self._history)

    async def _build_system_prompt(self, user_message: str = "") -> str:
        prompt = CHAT_SYSTEM_PROMPT.format(
            runtime_context=self._runtime_context,
            visualizations_section=build_visualizations_prompt(self._proactive_dashboards),
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
        return prompt

    # Packages the LLM is most likely to care about when writing scratchpad code.
    _NOTABLE_PACKAGES: set[str] = {
        "numpy", "pandas", "matplotlib", "seaborn", "scipy", "scikit-learn",
        "requests", "httpx", "aiohttp", "beautifulsoup4", "lxml",
        "pillow", "sympy", "networkx", "sqlalchemy", "pydantic",
        "rich", "tqdm", "click", "fastapi", "flask", "django",
        "openai", "anthropic", "tiktoken", "transformers", "torch",
        "polars", "pyarrow", "openpyxl", "xlsxwriter",
        "plotly", "bokeh", "altair",
        "pytest", "hypothesis",
        "yaml", "pyyaml", "toml", "tomli", "tomllib",
        "jinja2", "markdown", "pygments",
        "cryptography", "paramiko", "boto3",
    }

    def _build_tools(self) -> list[dict]:
        scratchpad_tool = dict(SCRATCHPAD_TOOL)
        pkg_list = self._scratchpads._available_packages
        if pkg_list:
            notable = sorted(
                p for p in pkg_list
                if p.lower() in self._NOTABLE_PACKAGES
            )
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
                scratchpad_tool["description"] += f"\n\nLessons from past sessions:\n{wisdom}"

        tools = [scratchpad_tool]
        if self._cortex is not None:
            tools.append(MEMORIZE_TOOL)
        elif self._self_awareness is not None:
            # Legacy fallback
            from anton.tools import MEMORIZE_TOOL as _MT
            tools.append(_MT)
        if self._episodic is not None and self._episodic.enabled:
            tools.append(RECALL_TOOL)
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
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in content
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
                            lines.append(f"[{role}/tool_use]: {block.get('name', '')}({str(block.get('input', ''))[:500]})")
                        elif block.get("type") == "tool_result":
                            lines.append(f"[tool_result]: {str(block.get('content', ''))[:500]}")

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
            self._history = [summary_msg, {"role": "assistant", "content": "Understood."}, *recent_turns]
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
                self._history.append({"role": "assistant", "content": response.content or ""})
                self._history.append({
                    "role": "user",
                    "content": (
                        f"SYSTEM: You have used {_MAX_TOOL_ROUNDS} tool-call rounds on this turn. "
                        "Stop retrying. Summarize what you accomplished and what failed, "
                        "then tell the user what they can do to unblock the issue."
                    ),
                })
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
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })
            self._history.append({"role": "assistant", "content": assistant_content})

            # Process each tool call via registry
            tool_results: list[dict] = []
            for tc in response.tool_calls:
                try:
                    result_text = await dispatch_tool(self, tc.name, tc.input)
                except Exception as exc:
                    result_text = f"Tool '{tc.name}' failed: {exc}"

                result_text = _apply_error_tracking(
                    result_text, tc.name, error_streak, resilience_nudged,
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_text,
                })

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

    async def turn_stream(self, user_input: str | list[dict]) -> AsyncIterator[StreamEvent]:
        """Streaming version of turn(). Yields events as they arrive."""
        self._history.append({"role": "user", "content": user_input})

        # Log user input to episodic memory
        if self._episodic is not None:
            content = user_input if isinstance(user_input, str) else str(user_input)[:2000]
            self._episodic.log_turn(self._turn_count + 1, "user", content)

        user_msg_str = user_input if isinstance(user_input, str) else ""
        assistant_text_parts: list[str] = []
        async for event in self._stream_and_handle_tools(user_msg_str):
            if isinstance(event, StreamTextDelta):
                assistant_text_parts.append(event.text)
            yield event

        # Log assistant response to episodic memory
        if self._episodic is not None and assistant_text_parts:
            self._episodic.log_turn(
                self._turn_count + 1, "assistant", "".join(assistant_text_parts)[:2000],
            )

        # Identity extraction (Default Mode Network — every 5 turns)
        self._turn_count += 1
        self._persist_history()
        if self._cortex is not None and self._cortex.mode != "off":
            if self._turn_count % 5 == 0 and isinstance(user_input, str):
                asyncio.create_task(self._cortex.maybe_update_identity(user_input))
            # Periodic memory vacuum (Systems Consolidation)
            self._cortex.maybe_vacuum()

    async def _stream_and_handle_tools(self, user_message: str = "") -> AsyncIterator[StreamEvent]:
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

        # Proactive compaction
        if not _compacted_this_turn and llm_response.usage.context_pressure > _CONTEXT_PRESSURE_THRESHOLD:
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
                    self._history.append({"role": "assistant", "content": llm_response.content or ""})
                    self._history.append({
                        "role": "user",
                        "content": (
                            f"SYSTEM: You have used {_MAX_TOOL_ROUNDS} tool-call rounds on this turn. "
                            "Stop retrying. Summarize what you accomplished and what failed, "
                            "then tell the user what they can do to unblock the issue."
                        ),
                    })
                    async for event in self._llm.plan_stream(
                        system=system,
                        messages=self._history,
                    ):
                        yield event
                    break

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
                    # Log tool call to episodic memory
                    if self._episodic is not None:
                        tc_desc = str(tc.input)[:2000]
                        self._episodic.log_turn(
                            self._turn_count + 1, "tool_call", tc_desc,
                            tool=tc.name,
                        )

                    try:
                        if tc.name == "scratchpad" and tc.input.get("action") == "exec":
                            # Inline streaming exec — yields progress events
                            prep = await prepare_scratchpad_exec(self, tc.input)
                            if isinstance(prep, str):
                                result_text = prep
                            else:
                                pad, code, description, estimated_time, estimated_seconds = prep
                                # Signal intent + ETA before execution begins
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
                                result_text = format_cell_result(cell) if cell else "No result produced."

                                # Log scratchpad cell to episodic memory
                                if self._episodic is not None and cell is not None:
                                    self._episodic.log_turn(
                                        self._turn_count + 1, "scratchpad",
                                        (cell.stdout or "")[:2000],
                                        description=description,
                                    )
                        else:
                            result_text = await dispatch_tool(self, tc.name, tc.input)
                            if tc.name == "scratchpad" and tc.input.get("action") == "dump":
                                yield StreamToolResult(content=result_text)
                                result_text = (
                                    "The full notebook has been displayed to the user above. "
                                    "Do not repeat it. Here is the content for your reference:\n\n"
                                    + result_text
                                )
                    except Exception as exc:
                        result_text = f"Tool '{tc.name}' failed: {exc}"

                    # Log tool result to episodic memory
                    if self._episodic is not None:
                        self._episodic.log_turn(
                            self._turn_count + 1, "tool_result", result_text[:2000],
                            tool=tc.name,
                        )

                    result_text = _apply_error_tracking(
                        result_text, tc.name, error_streak, resilience_nudged,
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result_text,
                    })

                self._history.append({"role": "user", "content": tool_results})

                # Signal that tools are done and LLM is now analyzing
                yield StreamTaskProgress(phase="analyzing", message="Analyzing results...")

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

                # Proactive compaction during tool loop
                if not _compacted_this_turn and llm_response.usage.context_pressure > _CONTEXT_PRESSURE_THRESHOLD:
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
                self._history.append({
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
                })
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
            verify_messages = list(self._history) + [{
                "role": "user",
                "content": (
                    "SYSTEM: Evaluate whether the task the user originally requested "
                    "has been fully completed based on the conversation above."
                ),
            }]
            verifier_request_options = (
                {"think": False}
                if self._llm.planning_provider_name == "ollama"
                else None
            )
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
                request_options=verifier_request_options,
            )

            verification_text = (verification.content or "").strip()
            if self._llm.planning_provider_name == "ollama":
                verification_text = _strip_ollama_think_tags(verification_text).strip()
                if not verification_text or not _is_parseable_verifier_status(verification_text):
                    break

            status_text = verification_text.upper()
            if "STATUS: COMPLETE" in status_text:
                break
            if "STATUS: STUCK" in status_text:
                # Stuck — inject diagnosis request and let the LLM explain
                reason = verification_text
                self._history.append({
                    "role": "user",
                    "content": (
                        f"SYSTEM: Task verification determined this task is stuck.\n"
                        f"Verifier assessment: {reason}\n\n"
                        "Explain to the user what went wrong, what you tried, and "
                        "suggest specific next steps they can take to unblock this."
                    ),
                })
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
            reason = verification_text
            self._history.append({
                "role": "user",
                "content": (
                    f"SYSTEM: Task verification determined this task is not yet complete "
                    f"(attempt {continuation}/{_MAX_CONTINUATIONS}).\n"
                    f"Verifier assessment: {reason}\n\n"
                    "Continue working on the original request. Pick up where you left off "
                    "and finish the remaining work. Do not repeat work already done."
                ),
            })
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


def _build_runtime_context(settings: AntonSettings) -> str:
    """Build runtime context string including Minds datasource info if configured."""
    ctx = (
        f"- Provider: {settings.planning_provider}\n"
        f"- Planning model: {settings.planning_model}\n"
        f"- Coding model: {settings.coding_model}\n"
        f"- Workspace: {settings.workspace_path}\n"
        f"- Memory mode: {settings.memory_mode}"
    )
    if settings.minds_api_key and (settings.minds_mind_name or settings.minds_datasource):
        engine = settings.minds_datasource_engine or "unknown"
        ctx += f"\n\n**CONNECTED MIND (Minds):**\n"
        if settings.minds_mind_name:
            ctx += f"- Mind: {settings.minds_mind_name}\n"
        if settings.minds_datasource:
            ctx += (
                f"- Datasource: {settings.minds_datasource}\n"
                f"- Engine: {engine}\n"
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
    _minds_refresh_knowledge(settings, cortex)

    runtime_context = _build_runtime_context(settings)
    api_key = (
        settings.anthropic_api_key if settings.coding_provider == "anthropic"
        else settings.openai_api_key if settings.coding_provider in {"openai", "openai-compatible"} else ""
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

    # --- Current mode ---
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

    # --- Helper to display a hippocampus scope ---
    def _show_scope(label: str, hc) -> int:
        identity = hc.recall_identity()
        rules = hc.recall_rules()
        lessons_raw = hc._read_full_lessons()
        rule_count = sum(1 for ln in rules.splitlines() if ln.strip().startswith("- ")) if rules else 0
        lesson_count = sum(1 for ln in lessons_raw.splitlines() if ln.strip().startswith("- ")) if lessons_raw else 0
        topics: list[str] = []
        if hc._topics_dir.is_dir():
            topics = [p.stem for p in sorted(hc._topics_dir.iterdir()) if p.suffix == ".md"]

        console.print(f"  [anton.cyan]{label}[/] [dim]({hc._dir})[/]")
        if identity:
            entries = [ln.strip()[2:] for ln in identity.splitlines() if ln.strip().startswith("- ")]
            if entries:
                console.print(f"    Identity:  {', '.join(entries[:3])}" + (" ..." if len(entries) > 3 else ""))
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

    # --- Global scope ---
    global_total = _show_scope("Global Memory", cortex.global_hc)

    # --- Project scope ---
    project_total = _show_scope("Project Memory", cortex.project_hc)

    total = global_total + project_total
    console.print(f"  Total entries: [bold]{total}[/]")
    if cortex.needs_compaction():
        console.print("  [anton.warning]Compaction needed (>50 entries in a scope)[/]")
    console.print()

    # --- Episodic memory stats ---
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
    from rich.prompt import Prompt
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
    choice = Prompt.ask(
        "Select session (or q to cancel)",
        choices=choices,
        default="q",
        console=console,
    )

    if choice == "q":
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
        await session._scratchpads.cancel_all_running()

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
    console.print(f"[anton.success]Resumed session from {selected['date']} ({selected['turns']} turns)[/]")
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
    from rich.prompt import Prompt

    console.print()
    console.print("[anton.cyan]/setup[/]")
    console.print()
    console.print("  What do you want to configure?")
    console.print("    [bold]1[/]  LLM — provider, API key, and models")
    console.print("    [bold]2[/]  Memory — memory mode and episodic memory")
    console.print("    [bold]q[/]  Back")
    console.print()

    top_choice = Prompt.ask(
        "Select",
        choices=["1", "2", "q"],
        default="q",
        console=console,
    )

    if top_choice == "q":
        console.print()
        return session
    elif top_choice == "1":
        return await _handle_setup_models(
            console, settings, workspace, state,
            self_awareness, cortex, session, episodic=episodic,
            history_store=history_store, session_id=session_id,
        )
    else:
        _handle_setup_memory(console, settings, workspace, cortex, episodic=episodic)
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
    from anton.llm.setup import configure_llm_settings
    from anton.workspace import Workspace as _Workspace

    # Always persist API keys and model settings to global ~/.anton/.env
    global_ws = _Workspace(Path.home())

    applied = configure_llm_settings(
        console,
        settings,
        global_ws,
        show_current_config=True,
    )
    if not applied:
        return session

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


def _handle_setup_memory(
    console: Console,
    settings: AntonSettings,
    workspace: Workspace,
    cortex,
    episodic: EpisodicMemory | None = None,
) -> None:
    """Setup sub-menu: memory mode and episodic memory toggle."""
    from rich.prompt import Prompt

    console.print()
    console.print("[anton.cyan]Memory configuration[/]")
    console.print()

    # --- Memory mode ---
    console.print("  Memory mode:")
    console.print(r"    [bold]1[/]  Autopilot — Anton decides what to remember       [dim]\[recommended][/]")
    console.print(r"    [bold]2[/]  Co-pilot — save obvious, confirm ambiguous        [dim]\[selective][/]")
    console.print(r"    [bold]3[/]  Off — never save memory (still reads existing)    [dim]\[suppressed][/]")
    console.print()

    mode_map = {"1": "autopilot", "2": "copilot", "3": "off"}
    current_mode_num = {"autopilot": "1", "copilot": "2", "off": "3"}.get(
        settings.memory_mode, "1"
    )
    mode_choice = Prompt.ask(
        "  Memory mode",
        choices=["1", "2", "3"],
        default=current_mode_num,
        console=console,
    )
    memory_mode = mode_map[mode_choice]
    settings.memory_mode = memory_mode
    workspace.set_secret("ANTON_MEMORY_MODE", memory_mode)
    if cortex is not None:
        cortex.mode = memory_mode

    # --- Episodic memory toggle ---
    if episodic is not None:
        console.print()
        ep_status = "ON" if episodic.enabled else "OFF"
        console.print(f"  Episodic memory (conversation archive): Currently [bold]{ep_status}[/]")
        toggle = Prompt.ask(
            "  Toggle episodic memory? (y/n)",
            choices=["y", "n"],
            default="n",
            console=console,
        )
        if toggle == "y":
            new_state = not episodic.enabled
            episodic.enabled = new_state
            settings.episodic_memory = new_state
            workspace.set_secret("ANTON_EPISODIC_MEMORY", "true" if new_state else "false")
            console.print(f"  Episodic memory: [bold]{'ON' if new_state else 'OFF'}[/]")

    console.print()
    console.print("[anton.success]Configuration updated.[/]")
    console.print()


def _normalize_minds_url(url: str) -> str:
    """Add https:// if no scheme present, strip trailing slash."""
    url = url.strip()
    if url and not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    return url.rstrip("/")


def _mask_secret(value: str, *, keep: int = 4) -> str:
    if len(value) <= keep * 2:
        return "*" * max(len(value), 3)
    return f"{value[:keep]}...{value[-keep:]}"


def _prompt_minds_api_key(
    console: Console,
    *,
    current_key: str,
    allow_empty_keep: bool,
) -> str | None:
    from rich.prompt import Prompt

    prompt = "API key"
    if current_key:
        masked = _mask_secret(current_key)
        if allow_empty_keep:
            prompt += f" [dim](Enter to keep {masked})[/]"
        else:
            prompt += f" [dim](current: {masked}; Enter to cancel)[/]"

    if current_key:
        api_key = Prompt.ask(prompt, default="", console=console, password=True).strip()
    else:
        api_key = Prompt.ask(prompt, console=console, password=True).strip()
    if api_key:
        return api_key
    if current_key and allow_empty_keep:
        return current_key
    return None


def _describe_minds_connection_error(err: Exception) -> tuple[str, str]:
    import socket
    import ssl
    import urllib.error

    if isinstance(err, urllib.error.HTTPError):
        reason = err.reason or "HTTP error"
        if err.code in (401, 403):
            return (
                f"Connection failed (HTTP {err.code}: {reason}). The server rejected the request.",
                "Common reasons: invalid or expired credentials, insufficient access, or the wrong server/endpoint.",
            )
        if 400 <= err.code < 500:
            return (
                f"Connection failed (HTTP {err.code}: {reason}). The server rejected the request.",
                "Common reasons: wrong URL, malformed request, or access restrictions on that endpoint.",
            )
        if err.code >= 500:
            return (
                f"Connection failed (HTTP {err.code}: {reason}). The server returned an error.",
                "Common reasons: server-side failure or a temporary outage.",
            )
        return (
            f"Connection failed (HTTP {err.code}: {reason}).",
            "Common reasons: a server response Anton could not use or a transient connectivity problem.",
        )

    if isinstance(err, urllib.error.URLError):
        reason = getattr(err, "reason", None)
        if isinstance(reason, ssl.SSLCertVerificationError):
            return (
                "Connection failed during TLS certificate verification.",
                "Common reasons: a self-signed, expired, or otherwise untrusted certificate.",
            )
        if isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason).lower():
            return (
                "Connection failed because the request timed out.",
                "Common reasons: the server is slow or unavailable, the URL is wrong, or there is a network path issue.",
            )
        return (
            f"Connection failed ({err}).",
            "Common reasons: network connectivity problems, DNS issues, or a server Anton could not reach.",
        )

    if "timed out" in str(err).lower():
        return (
            "Connection failed because the request timed out.",
            "Common reasons: the server is slow or unavailable, the URL is wrong, or there is a network path issue.",
        )

    return (
        f"Connection failed ({err}).",
        "Common reasons: network connectivity problems, authentication issues, or a server-side failure.",
    )


def _minds_request(
    url: str,
    api_key: str,
    *,
    method: str = "GET",
    payload: bytes | None = None,
    verify: bool = True,
    timeout: int = 30,
) -> bytes:
    """Shared HTTP helper for all Minds API calls.

    Sets headers that pass through Cloudflare bot detection.
    """
    import ssl
    import urllib.request

    req = urllib.request.Request(url, data=payload, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    # Browser-like headers to avoid Cloudflare bot detection
    req.add_header("User-Agent", "Mozilla/5.0 (compatible; Anton/1.0; +https://github.com/mindsdb/anton)")
    req.add_header("Accept-Language", "en-US,en;q=0.9")
    req.add_header("Accept-Encoding", "identity")
    req.add_header("Connection", "keep-alive")

    ctx = None
    if not verify:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        return resp.read()


def _minds_list_minds(base_url: str, api_key: str, verify: bool = True) -> list[dict]:
    """Fetch minds list from a Minds server using stdlib urllib."""
    import json as _json

    url = f"{base_url}/api/v1/minds/"  # trailing slash required
    raw = _minds_request(url, api_key, verify=verify)
    data = _json.loads(raw.decode())

    if isinstance(data, list):
        return data
    return data.get("minds", data if isinstance(data, list) else [])


def _minds_get_mind(base_url: str, api_key: str, mind_name: str, verify: bool = True) -> dict | None:
    """Fetch a single mind's details from a Minds server."""
    import json as _json

    url = f"{base_url}/api/v1/minds/{mind_name}"
    try:
        raw = _minds_request(url, api_key, verify=verify, timeout=15)
        return _json.loads(raw.decode())
    except Exception:
        return None


def _minds_refresh_knowledge(settings: AntonSettings, cortex) -> None:
    """Fetch the configured mind's parameters and update the memory topic file."""
    if not settings.minds_api_key or not settings.minds_mind_name or cortex is None:
        return

    mind = _minds_get_mind(
        _normalize_minds_url(settings.minds_url),
        settings.minds_api_key,
        settings.minds_mind_name,
        verify=settings.minds_ssl_verify,
    )
    if not mind:
        return

    params = mind.get("parameters", {}) or {}
    parts = []
    if params.get("system_prompt"):
        parts.append(params["system_prompt"])
    if params.get("prompt_template"):
        parts.append(params["prompt_template"])

    if not parts:
        return

    knowledge = "\n\n".join(parts)
    topic_content = f"# Minds — {settings.minds_mind_name}\n\n{knowledge}\n"
    topic_path = cortex.project_hc._topics_dir / "minds-datasource.md"
    cortex.project_hc._topics_dir.mkdir(parents=True, exist_ok=True)
    cortex.project_hc._encode_with_lock(topic_path, topic_content, mode="write")


def _minds_list_datasources(base_url: str, api_key: str, verify: bool = True) -> list[dict]:
    """Fetch datasource list from a Minds server using stdlib urllib."""
    import json as _json

    url = f"{base_url}/api/v1/datasources"
    raw = _minds_request(url, api_key, verify=verify)
    data = _json.loads(raw.decode())

    # Response may be a list or a dict with a "datasources" key
    if isinstance(data, list):
        return data
    return data.get("datasources", data if isinstance(data, list) else [])


def _minds_test_llm(base_url: str, api_key: str, verify: bool = True) -> bool:
    """Test if the Minds server supports LLM endpoints (_code_/_reason_ models)."""
    import json as _json

    url = f"{base_url}/api/v1/chat/completions"
    payload = _json.dumps({
        "model": "_code_",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }).encode()

    try:
        _minds_request(url, api_key, method="POST", payload=payload, verify=verify)
        return True
    except Exception:
        return False


_MINDS_KEYS = {
    "ANTON_MINDS_API_KEY", "ANTON_MINDS_URL", "ANTON_MINDS_MIND_NAME",
    "ANTON_MINDS_DATASOURCE", "ANTON_MINDS_DATASOURCE_ENGINE", "ANTON_MINDS_SSL_VERIFY",
}

_LLM_KEYS = {
    "ANTON_PLANNING_PROVIDER", "ANTON_CODING_PROVIDER",
    "ANTON_PLANNING_MODEL", "ANTON_CODING_MODEL",
    "ANTON_ANTHROPIC_API_KEY", "ANTON_OPENAI_API_KEY", "ANTON_OPENAI_BASE_URL",
    "ANTON_OLLAMA_BASE_URL",
}

_SECRET_PATTERNS = ("KEY", "TOKEN", "SECRET", "PAT", "PASSWORD")


def _is_secret_key(key: str) -> bool:
    upper = key.upper()
    return any(p in upper for p in _SECRET_PATTERNS)


def _display_value(key: str, value: str) -> str:
    if _is_secret_key(key) and value:
        return _mask_secret(value)
    return value or "[dim]<empty>[/]"


async def _handle_data_connections(
    console: Console,
    settings: AntonSettings,
    workspace: Workspace,
    session: ChatSession,
) -> ChatSession:
    """View and manage stored keys and connections across global and project vaults."""
    from rich.prompt import Confirm, Prompt

    from anton.workspace import Workspace as _Workspace

    global_ws = _Workspace(Path.home())

    global_env = global_ws.load_env()
    project_env = workspace.load_env()

    # Merge with source tags: project keys override global for display,
    # but we track where each lives for writes/removals.
    all_keys: dict[str, tuple[str, str, str]] = {}  # key -> (value, source, scope_label)
    for k, v in global_env.items():
        all_keys[k] = (v, "global", "~/.anton/.env")
    for k, v in project_env.items():
        all_keys[k] = (v, "project", f"{workspace.base}/.anton/.env")

    console.print()

    if not all_keys:
        console.print("[anton.warning]No connections or secrets configured.[/]")
        console.print("[anton.muted]Use /connect to set up a Minds connection, or ask Anton to store a key.[/]")
        console.print()
        return session

    def _print_table() -> list[tuple[str, str, str, str]]:
        """Print grouped key table and return flat list for menu selection."""
        minds = {k: all_keys[k] for k in sorted(all_keys) if k in _MINDS_KEYS}
        llm = {k: all_keys[k] for k in sorted(all_keys) if k in _LLM_KEYS}
        other = {k: all_keys[k] for k in sorted(all_keys) if k not in _MINDS_KEYS and k not in _LLM_KEYS}

        flat: list[tuple[str, str, str, str]] = []  # (key, value, source, scope_label)
        idx = 1

        if minds:
            console.print("[anton.cyan]Minds Connection[/]")
            for k, (v, src, lbl) in minds.items():
                console.print(f"    [bold]{idx}[/]  {k} = {_display_value(k, v)}  [dim]({lbl})[/]")
                flat.append((k, v, src, lbl))
                idx += 1
            console.print()

        if llm:
            console.print("[anton.cyan]LLM Configuration[/]")
            for k, (v, src, lbl) in llm.items():
                console.print(f"    [bold]{idx}[/]  {k} = {_display_value(k, v)}  [dim]({lbl})[/]")
                flat.append((k, v, src, lbl))
                idx += 1
            console.print()

        if other:
            console.print("[anton.cyan]Other Integrations[/]")
            for k, (v, src, lbl) in other.items():
                console.print(f"    [bold]{idx}[/]  {k} = {_display_value(k, v)}  [dim]({lbl})[/]")
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

        action = Prompt.ask("Select", choices=["1", "2", "3", "q"], default="q", console=console)

        if action == "q":
            console.print()
            return session

        if action == "1":
            # --- Edit ---
            console.print()
            pick = Prompt.ask(
                f"Key number to edit (1-{len(flat)})",
                console=console,
            )
            try:
                pick_idx = int(pick) - 1
                if not 0 <= pick_idx < len(flat):
                    raise ValueError
            except ValueError:
                console.print("[anton.warning]Invalid selection.[/]")
                console.print()
                continue

            key, old_val, src, lbl = flat[pick_idx]
            use_password = _is_secret_key(key)
            new_val = Prompt.ask(
                f"New value for {key}",
                default="" if use_password else old_val,
                console=console,
                password=use_password,
            ).strip()
            if not new_val:
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
            pick = Prompt.ask(
                f"Key number to remove (1-{len(flat)})",
                console=console,
            )
            try:
                pick_idx = int(pick) - 1
                if not 0 <= pick_idx < len(flat):
                    raise ValueError
            except ValueError:
                console.print("[anton.warning]Invalid selection.[/]")
                console.print()
                continue

            key, _, src, lbl = flat[pick_idx]
            if not Confirm.ask(f"Remove {key} from {lbl}?", default=False, console=console):
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
            new_key = Prompt.ask("Key name (e.g. HUBSPOT_API_KEY)", console=console).strip()
            if not new_key:
                console.print("[anton.warning]Key name cannot be empty.[/]")
                console.print()
                continue

            if new_key in all_keys:
                if not Confirm.ask(f"{new_key} already exists. Overwrite?", default=False, console=console):
                    console.print("[anton.muted]Cancelled.[/]")
                    console.print()
                    continue

            use_password = _is_secret_key(new_key)
            new_val = Prompt.ask(
                f"Value for {new_key}",
                console=console,
                password=use_password,
            ).strip()
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
            scope_label = "~/.anton/.env" if scope == "global" else f"{workspace.base}/.anton/.env"
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
    import urllib.error

    from rich.prompt import Prompt

    from anton.workspace import Workspace as _Workspace

    global_ws = _Workspace(Path.home())

    console.print()

    # --- Prompt for URL and API key (use saved values as defaults) ---
    saved_url = _normalize_minds_url(settings.minds_url)
    minds_url = Prompt.ask("Minds server URL", default=saved_url, console=console)
    minds_url = _normalize_minds_url(minds_url)

    saved_key = settings.minds_api_key or ""
    api_key = _prompt_minds_api_key(
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
            minds = _minds_list_minds(minds_url, api_key, verify=ssl_verify)
            break
        except (urllib.error.URLError, urllib.error.HTTPError) as err:
            headline, advice = _describe_minds_connection_error(err)
            console.print(f"[anton.error]{headline}[/]")
            console.print(f"[anton.muted]{advice}[/]")
        except Exception as err:
            headline, advice = _describe_minds_connection_error(err)
            console.print(f"[anton.error]{headline}[/]")
            console.print(f"[anton.muted]{advice}[/]")

        console.print()
        console.print("  Recovery options:")
        console.print("    [bold]1[/]  Reconfigure API key")
        console.print("    [bold]2[/]  Retry without SSL verification")
        console.print("    [bold]q[/]  Back")
        console.print()

        action = Prompt.ask(
            "Select",
            choices=["1", "2", "q"],
            default="q",
            console=console,
        )
        if action == "q":
            console.print("[anton.muted]Aborted.[/]")
            console.print()
            return session
        if action == "1":
            new_key = _prompt_minds_api_key(
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
        ds_label = f"{ds_count} datasource{'s' if ds_count != 1 else ''}" if ds_count else "no datasources"
        console.print(f"    [bold]{i}[/]  {name} [dim]({ds_label})[/]")
    console.print()

    choices = [str(i) for i in range(1, len(minds) + 1)]
    pick = Prompt.ask("Select mind", choices=choices, console=console)
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
        ds_pick = Prompt.ask("Select datasource", choices=ds_choices, console=console)
        picked_ds = mind_datasources[int(ds_pick) - 1]
        ds_name = picked_ds if isinstance(picked_ds, str) else picked_ds.get("name", "")
    elif len(mind_datasources) == 1:
        picked_ds = mind_datasources[0]
        ds_name = picked_ds if isinstance(picked_ds, str) else picked_ds.get("name", "")
        console.print(f"[anton.muted]Auto-selected datasource: {ds_name}[/]")

    # --- Resolve engine type from datasources list ---
    if ds_name:
        try:
            all_datasources = _minds_list_datasources(minds_url, api_key, verify=ssl_verify)
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
    llm_ok = _minds_test_llm(minds_url, api_key, verify=ssl_verify)

    if llm_ok:
        console.print("[anton.success]LLM endpoints available — using Minds server as LLM provider.[/]")
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
        has_anthropic = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
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
                console.print("[anton.warning]No API key provided — LLM calls will not work.[/]")

    global_ws.apply_env_to_process()
    console.print()

    return _rebuild_session(
        settings=settings, state=state, self_awareness=self_awareness,
        cortex=cortex, workspace=workspace, console=console, episodic=episodic,
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
            parts.append(f"\n<file path=\"{p}\">\n(File too large to inline — {_human_size(size)}. "
                         f"Use the scratchpad to read it.)\n</file>")
            continue

        # Skip binary-looking files
        if suffix in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
                       ".pdf", ".zip", ".tar", ".gz", ".exe", ".dll", ".so",
                       ".pyc", ".pyo", ".whl", ".egg", ".db", ".sqlite"):
            parts.append(f"\n<file path=\"{p}\">\n(Binary file — {_human_size(size)}. "
                         f"Use the scratchpad to process it.)\n</file>")
            continue

        try:
            content = p.read_text(errors="replace")
        except Exception:
            parts.append(f"\n<file path=\"{p}\">\n(Could not read file.)\n</file>")
            continue

        parts.append(f"\n<file path=\"{p}\">\n{content}\n</file>")

    return "\n".join(parts)


def _format_clipboard_image_message(uploaded: object, user_text: str = "") -> list[dict]:
    """Build a multimodal LLM message for a clipboard image upload.

    Returns a list of content blocks (image + text) so the LLM can see
    the image directly. The file path is included so the LLM can pass
    it to the scratchpad if deeper processing is needed.
    """
    import base64

    text = user_text.strip() if user_text else "I've pasted an image from my clipboard. Analyze it."
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
            console.print("[anton.success]Pillow installed. Clipboard is now available.[/]")
            return True
        console.print("[anton.error]Failed to install Pillow.[/]")
        return False


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.0f}{unit}" if unit == "B" else f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}TB"


def _print_slash_help(console: Console) -> None:
    """Print available slash commands."""
    console.print()
    console.print("[anton.cyan]Available commands:[/]")
    console.print("  [bold]/connect[/]           — Connect to a Minds server and select a mind")
    console.print("  [bold]/data-connections[/]  — View and manage stored keys and connections")
    console.print("  [bold]/setup[/]             — Configure models or memory settings")
    console.print("  [bold]/memory[/]            — Show memory status dashboard")
    console.print("  [bold]/paste[/]             — Attach clipboard image to your message")
    console.print("  [bold]/resume[/]            — Resume a previous chat session")
    console.print("  [bold]/help[/]              — Show this help message")
    console.print("  [bold]exit[/]               — Quit the chat")
    console.print()


class _EscapeWatcher:
    """Detect Escape keypress during streaming via cbreak terminal mode."""

    def __init__(self, on_cancel: Callable[[], None] | None = None) -> None:
        self.cancelled = asyncio.Event()
        self._on_cancel = on_cancel
        self._task: asyncio.Task | None = None
        self._old_settings: list | None = None
        self._stop = False

    async def __aenter__(self) -> _EscapeWatcher:
        if sys.platform != "win32" and sys.stdin.isatty():
            self._task = asyncio.create_task(self._watch())
        return self

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

        spinner = Spinner("dots", text=Text(" Closing scratchpad processes…", style="anton.muted"))
        self._live = Live(spinner, console=self._console, refresh_per_second=6, transient=True)
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None


def run_chat(console: Console, settings: AntonSettings, *, resume: bool = False) -> None:
    """Launch the interactive chat REPL."""
    asyncio.run(_chat_loop(console, settings, resume=resume))


async def _chat_loop(console: Console, settings: AntonSettings, *, resume: bool = False) -> None:
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

    # --- Memory system (brain-inspired architecture) ---
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

    # --- Episodic memory ---
    from anton.memory.episodes import EpisodicMemory

    episodes_dir = Path.home() / ".anton" / "episodes"
    episodic = EpisodicMemory(episodes_dir, enabled=settings.episodic_memory)
    if episodic.enabled:
        episodic.start_session()

    # --- History store (for /resume) ---
    from anton.memory.history_store import HistoryStore

    history_store = HistoryStore(episodes_dir)
    current_session_id = episodic._session_id if episodic.enabled else None

    # Clean up old clipboard uploads
    uploads_dir = Path(settings.workspace_path) / ".anton" / "uploads"
    cleanup_old_uploads(uploads_dir)

    # Build runtime context so the LLM knows what it's running on
    runtime_context = _build_runtime_context(settings)

    coding_api_key = (
        settings.anthropic_api_key if settings.coding_provider == "anthropic"
        else settings.openai_api_key if settings.coding_provider in {"openai", "openai-compatible"} else ""
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
        history_store=history_store,
        session_id=current_session_id,
        proactive_dashboards=settings.proactive_dashboards,
    )

    # Handle --resume flag at startup
    if resume:
        session, resumed_id = await _handle_resume(
            console, settings, state, self_awareness, cortex,
            workspace, session, episodic=episodic,
            history_store=history_store,
        )
        if resumed_id:
            current_session_id = resumed_id


    console.print("[anton.muted] Chat with Anton. Type '/help' for commands or 'exit' to quit.[/]")
    console.print(f"[anton.cyan_dim] {'━' * 40}[/]")
    console.print()

    from anton.chat_ui import StreamDisplay

    toolbar = {"stats": "", "status": ""}
    display = StreamDisplay(console, toolbar=toolbar)

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.styles import Style as PTStyle

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

    pt_style = PTStyle.from_dict({
        "bottom-toolbar": "noreverse nounderline bg:default",
    })

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
                confirm = console.input("[bold]Save to memory? (y/n/pick numbers):[/] ").strip().lower()
                if confirm in ("y", "yes"):
                    if cortex is not None:
                        await cortex.encode(pending)
                    console.print("[anton.muted]Saved.[/]")
                elif confirm in ("n", "no"):
                    console.print("[anton.muted]Discarded.[/]")
                else:
                    # Parse number selections like "1 3" or "1,3"
                    try:
                        nums = [int(x.strip()) for x in confirm.replace(",", " ").split() if x.strip().isdigit()]
                        selected = [pending[n - 1] for n in nums if 1 <= n <= len(pending)]
                        if selected and cortex is not None:
                            await cortex.encode(selected)
                            console.print(f"[anton.muted]Saved {len(selected)} entries.[/]")
                        else:
                            console.print("[anton.muted]Discarded.[/]")
                    except (ValueError, IndexError):
                        console.print("[anton.muted]Discarded.[/]")
                session._pending_memory_confirmations = []
                console.print()

            try:
                user_input = await prompt_session.prompt_async(
                    [("bold fg:#00ff9f", "you>"), ("", " ")]
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
                if cmd == "/connect":
                    session = await _handle_connect(
                        console, settings, workspace, state,
                        self_awareness, cortex, session,
                        episodic=episodic,
                    )
                    continue
                elif cmd == "/setup":
                    session = await _handle_setup(
                        console, settings, workspace, state,
                        self_awareness, cortex, session,
                        episodic=episodic,
                        history_store=history_store,
                        session_id=current_session_id,
                    )
                    continue
                elif cmd == "/memory":
                    _handle_memory(console, settings, cortex, episodic=episodic)
                    continue
                elif cmd == "/data-connections":
                    session = await _handle_data_connections(
                        console, settings, workspace, session,
                    )
                    continue
                elif cmd == "/resume":
                    session, resumed_id = await _handle_resume(
                        console, settings, state, self_awareness, cortex,
                        workspace, session, episodic=episodic,
                        history_store=history_store,
                    )
                    if resumed_id:
                        current_session_id = resumed_id
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
                        message_content = _format_clipboard_image_message(uploaded, user_text)
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

            display.start()
            t0 = time.monotonic()
            ttft: float | None = None
            total_input = 0
            total_output = 0
            session._cancel_event.clear()

            try:
                async with _EscapeWatcher(on_cancel=display.show_cancelling) as esc:
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
                parts = [f"{elapsed:.1f}s", f"{total_input} in / {total_output} out"]
                if ttft is not None:
                    parts.append(f"TTFT {int(ttft * 1000)}ms")
                toolbar["stats"] = "  ".join(parts)
                toolbar["status"] = ""
                display.finish()
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
                        await session._scratchpads.cancel_all_running()
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
    except KeyboardInterrupt:
        pass

    console.print()
    console.print("[anton.muted]See you.[/]")
    await session.close()
