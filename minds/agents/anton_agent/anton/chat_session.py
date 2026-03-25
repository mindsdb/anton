"""ChatSession — multi-turn conversation handler with tool-call delegation.

Ported from anton/chat.py for server-side use. Removed all CLI dependencies
(Console, Rich, clipboard, Workspace, SelfAwarenessContext).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from minds.common.logger import get_logger
from minds.services.memory import MemoryBlock, MemoryService

from .constants import (
    CONTEXT_PRESSURE_THRESHOLD,
    MAX_CONSECUTIVE_ERRORS,
    MAX_TOOL_ROUNDS,
    NOTABLE_PACKAGES,
    RESILIENCE_NUDGE_AT,
)
from .llm.provider import (
    ContextOverflowError,
    StreamComplete,
    StreamContextCompacted,
    StreamEvent,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolResult,
)
from .prompts import (
    CHAT_SYSTEM_PROMPT,
    MAX_CONSECUTIVE_ERRORS_PROMPT,
    MAX_TOOL_ROUNDS_PROMPT,
    RESILIENCE_NUDGE_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
)
from .scratchpad_manager import ScratchpadManager
from .verification import MAX_VERIFICATION_CONTINUATIONS, verify_task
from .tools import (
    MEMORIZE_TOOL,
    RECALL_TOOL,
    SCRATCHPAD_TOOL,
    dispatch_tool,
    format_cell_result,
    prepare_scratchpad_exec,
)

if TYPE_CHECKING:
    from .llm.client import LLMClient
    from .memory.cortex import Cortex
    from .memory.episodes import EpisodicMemory

logger = get_logger(__name__)


class ChatSession:
    """Manages a multi-turn conversation with tool-call delegation."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        cortex: Cortex,
        episodic: EpisodicMemory,
        backend: str,
        coding_provider: str,
        coding_api_key: str,
        coding_model: str,
        workspace_path: Path,
        runtime_context: str = "",
        extra_env: dict[str, str] | None = None,
        shared_memory: MemoryService | None = None,
        events: list[dict] = None,
        classification=None,
    ) -> None:
        self._llm = llm_client
        self._cortex = cortex
        self._episodic = episodic
        self._runtime_context = runtime_context
        self._history: list[dict] = []
        self._pending_memory_confirmations: list = []
        self._turn_count = 0
        self._classification = classification
        self._last_tool_round = 0
        self._shared_memory = shared_memory
        self._shared_memory_block: MemoryBlock | None = None
        self._shared_memory_load_failed = False
        self._scratchpads = ScratchpadManager(
            backend=backend,
            coding_provider=coding_provider,
            coding_model=coding_model,
            coding_api_key=coding_api_key,
            workspace_path=workspace_path,
            extra_env=extra_env,
            events=events,
        )

    @property
    def history(self) -> list[dict]:
        return self._history

    def load_history(self, history: list[dict]) -> None:
        """Load previous conversation history."""
        self._history = list(history)

    def repair_history(self) -> None:
        """Fix dangling tool_use blocks left by mid-stream cancellation."""
        if not self._history:
            return
        last = self._history[-1]
        if last.get("role") != "assistant":
            return
        content = last.get("content")
        if not isinstance(content, list):
            return
        tool_ids = [block["id"] for block in content if isinstance(block, dict) and block.get("type") == "tool_use"]
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

    def _init_shared_memory(self, query: str) -> None:
        """Load mind memory once for this session. Failures are non-fatal."""
        if self._shared_memory is None:
            return
        try:
            # NOTE: We load topics per initial question, not for every question. Worth improving
            self._shared_memory_block = self._shared_memory.load_for_session(query)
        except Exception:
            logger.exception("Failed to load mind memory for session — continuing without it")
            self._shared_memory_block = MemoryBlock()
            self._shared_memory_load_failed = True

    def _build_system_prompt(self) -> str:
        prompt = CHAT_SYSTEM_PROMPT.format(
            runtime_context=self._runtime_context,
        )
        if self._shared_memory_block is not None and not self._shared_memory_block.is_empty:
            memory_section = MemoryService.format_block(self._shared_memory_block)
            if memory_section:
                prompt = memory_section + "\n\n" + prompt
        if self._cortex is not None:
            memory_section = self._cortex.build_memory_context()
            if memory_section:
                prompt += memory_section
        return prompt

    def _build_tools(self) -> list[dict]:
        scratchpad_tool = dict(SCRATCHPAD_TOOL)
        pkg_list = self._scratchpads._available_packages
        if pkg_list:
            notable = sorted(p for p in pkg_list if p.lower() in NOTABLE_PACKAGES)
            if notable:
                pkg_line = ", ".join(notable)
                extra = f"\n\nInstalled packages ({len(pkg_list)} total, notable: {pkg_line})."
            else:
                extra = f"\n\nInstalled packages: {len(pkg_list)} total (standard library plus dependencies)."
            scratchpad_tool["description"] = SCRATCHPAD_TOOL["description"] + extra

        if self._cortex is not None:
            wisdom = self._cortex.get_scratchpad_context()
            if wisdom:
                scratchpad_tool["description"] += f"\n\nLessons from past sessions:\n{wisdom}"

        tools = [scratchpad_tool]

        if self._cortex is not None:
            tools.append(MEMORIZE_TOOL)
        if self._episodic is not None and self._episodic.enabled:
            tools.append(RECALL_TOOL)
        return tools

    async def close(self) -> None:
        """Clean up scratchpads and other resources."""
        await self._scratchpads.close_all()

    async def _summarize_history(self) -> None:
        """Compress old conversation turns into a summary using the coding model."""
        if len(self._history) < 6:
            return

        min_recent = 4
        split = max(int(len(self._history) * 0.6), 1)
        split = min(split, len(self._history) - min_recent)
        if split < 2:
            return

        old_turns = self._history[:split]
        recent_turns = self._history[split:]

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
                            lines.append(f"[tool_result]: {str(block.get('content', ''))[:500]}")

        old_text = "\n".join(lines)
        if len(old_text) > 8000:
            old_text = old_text[:8000] + "\n... (truncated)"

        try:
            summary_response = await self._llm.code(
                system=SUMMARIZE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": old_text}],
                max_tokens=2048,
            )
            summary = summary_response.content or "(summary unavailable)"
        except Exception:
            summary = f"(Earlier conversation with {len(old_turns)} turns — summarization failed)"

        summary_msg = {
            "role": "user",
            "content": f"[Context summary of earlier conversation]\n{summary}",
        }
        self._history = [summary_msg] + recent_turns

    def _compact_scratchpads(self) -> bool:
        """Compact all active scratchpads. Returns True if any were compacted."""
        compacted = False
        if self._scratchpads._pad is not None and self._scratchpads._pad._compact_cells():
            compacted = True
        return compacted

    async def turn_stream(self, user_input: str | list[dict]) -> AsyncIterator[StreamEvent]:
        """Streaming version of turn(). Yields events as they arrive."""
        self._history.append({"role": "user", "content": user_input})

        # Load mind memory once on the first turn using the user's query for scoring.
        if self._shared_memory is not None and self._shared_memory_block is None:
            query = user_input if isinstance(user_input, str) else ""
            self._init_shared_memory(query)
            if self._shared_memory_load_failed:
                yield StreamContextCompacted(message="Note: Mind memory could not be loaded for this session.")

        if self._episodic is not None:
            content = user_input if isinstance(user_input, str) else str(user_input)[:2000]
            self._episodic.log_turn(self._turn_count + 1, "user", content)

        assistant_text_parts: list[str] = []
        async for event in self._stream_and_handle_tools():
            if isinstance(event, StreamTextDelta):
                assistant_text_parts.append(event.text)
            yield event

        if self._episodic is not None and assistant_text_parts:
            self._episodic.log_turn(
                self._turn_count + 1,
                "assistant",
                "".join(assistant_text_parts)[:2000],
            )

        self._turn_count += 1
        if (
            self._turn_count % 5 == 0
            and self._cortex is not None
            and self._cortex.mode != "off"
            and isinstance(user_input, str)
        ):
            asyncio.create_task(self._cortex.maybe_update_identity(user_input))

    async def _stream_and_handle_tools(self) -> AsyncIterator[StreamEvent]:
        """Stream tool loop with completion verification wrapper."""
        continuation = 0

        while True:
            # Run the core tool loop
            async for event in self._run_tool_loop():
                yield event

            # Decide whether to verify completion
            should_verify = (
                self._classification is not None
                and self._classification.is_multi_step
                and self._last_tool_round > 0
                and continuation < MAX_VERIFICATION_CONTINUATIONS
            )

            if not should_verify:
                break

            # Build classification context for the verifier
            cls = self._classification
            classification_context = (
                f"TASK CLASSIFICATION:\n"
                f"- Summary: {cls.task_summary}\n"
                f"- Success criteria: {', '.join(cls.success_criteria)}\n"
                f"- Expected artifacts: {', '.join(cls.expected_artifacts)}\n"
                f"- Requires data query: {cls.requires_data_query}\n"
            )

            try:
                verification = await verify_task(
                    llm_provider=self._llm.coding_provider,
                    model=self._llm.coding_model,
                    classification_context=classification_context,
                    history=self._history,
                )
            except Exception:
                logger.warning("Task verification failed — treating as complete", exc_info=True)
                break

            if verification.status == "complete":
                break

            if verification.status == "stuck":
                self._history.append({
                    "role": "user",
                    "content": (
                        f"[System: Task verification detected a blocker: {verification.blocker or 'unknown'}. "
                        f"Reason: {verification.reason}. "
                        f"Explain to the user what went wrong and suggest specific next steps they can take.]"
                    ),
                })
                yield StreamTaskProgress(phase="verification", message="Diagnosing blocked task...")
                system = self._build_system_prompt()
                async for event in self._llm.plan_stream(
                    system=system,
                    messages=self._history,
                ):
                    yield event
                    if isinstance(event, StreamComplete):
                        resp = event.response
                        self._history.append({"role": "assistant", "content": resp.content or ""})
                break

            # status == "incomplete"
            continuation += 1
            if continuation >= MAX_VERIFICATION_CONTINUATIONS:
                self._history.append({
                    "role": "user",
                    "content": (
                        f"[System: After {continuation} continuation attempts, the task is still incomplete. "
                        f"Remaining: {', '.join(verification.remaining_work)}. "
                        f"Provide your best answer with what you have and explain what could not be completed.]"
                    ),
                })
                yield StreamTaskProgress(phase="verification", message="Task incomplete — providing best answer...")
                system = self._build_system_prompt()
                async for event in self._llm.plan_stream(
                    system=system,
                    messages=self._history,
                ):
                    yield event
                    if isinstance(event, StreamComplete):
                        resp = event.response
                        self._history.append({"role": "assistant", "content": resp.content or ""})
                break

            # Continue working
            self._history.append({
                "role": "user",
                "content": (
                    f"[System: Task verification found this incomplete "
                    f"(attempt {continuation}/{MAX_VERIFICATION_CONTINUATIONS}). "
                    f"Remaining work: {', '.join(verification.remaining_work)}. "
                    f"Reason: {verification.reason}. "
                    f"Continue working on the remaining items. Do not repeat work already done.]"
                ),
            })
            yield StreamTaskProgress(
                phase="verification",
                message=f"Task incomplete — continuing ({continuation}/{MAX_VERIFICATION_CONTINUATIONS})...",
            )
            # Loop back to _run_tool_loop

        # Consolidation: replay scratchpad sessions to extract lessons
        if self._cortex is not None and self._cortex.mode != "off":
            self._maybe_consolidate_scratchpads()

    async def _run_tool_loop(self) -> AsyncIterator[StreamEvent]:
        """Stream one LLM call, handle tool loops, yield all events."""
        system = self._build_system_prompt()
        tools = self._build_tools()

        response: StreamComplete | None = None
        _compacted_this_turn = False

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
            yield StreamContextCompacted(message="Context was getting long — older history has been summarized.")
            async for event in self._llm.plan_stream(
                system=system,
                messages=self._history,
                tools=tools,
            ):
                yield event
                if isinstance(event, StreamComplete):
                    response = event

        if response is None:
            self._last_tool_round = 0
            return

        llm_response = response.response

        if not _compacted_this_turn and llm_response.usage.context_pressure > CONTEXT_PRESSURE_THRESHOLD:
            await self._summarize_history()
            self._compact_scratchpads()
            _compacted_this_turn = True
            yield StreamContextCompacted(message="Context was getting long — older history has been summarized.")

        # Tool-call loop with circuit breaker
        tool_round = 0
        error_streak: dict[str, int] = {}
        resilience_nudged: set[str] = set()

        while llm_response.tool_calls:
            tool_round += 1
            if tool_round > MAX_TOOL_ROUNDS:
                self._history.append({"role": "assistant", "content": llm_response.content or ""})
                self._history.append(
                    {
                        "role": "user",
                        "content": (MAX_TOOL_ROUNDS_PROMPT.format(max_tool_rounds=MAX_TOOL_ROUNDS)),
                    }
                )
                async for event in self._llm.plan_stream(
                    system=system,
                    messages=self._history,
                ):
                    yield event
                self._last_tool_round = tool_round
                return

            # Build assistant message with content blocks
            assistant_content: list[dict] = []
            if llm_response.content:
                assistant_content.append({"type": "text", "text": llm_response.content})
            for tc in llm_response.tool_calls:
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    }
                )
            self._history.append({"role": "assistant", "content": assistant_content})

            # Process each tool call
            tool_results: list[dict] = []
            for tc in llm_response.tool_calls:
                if self._episodic is not None:
                    tc_desc = str(tc.input)[:2000]
                    self._episodic.log_turn(
                        self._turn_count + 1,
                        "tool_call",
                        tc_desc,
                        tool=tc.name,
                    )

                try:
                    if tc.name == "scratchpad" and tc.input.get("action") == "exec":
                        prep = await prepare_scratchpad_exec(self, tc.input)
                        if isinstance(prep, str):
                            result_text = prep
                        else:
                            pad, code, description, estimated_time, estimated_seconds = prep
                            from .backends.base import Cell

                            cell = None
                            async for item in pad.execute_streaming(
                                code,
                                description=description,
                                estimated_time=estimated_time,
                                estimated_seconds=estimated_seconds,
                            ):
                                if isinstance(item, str):
                                    yield StreamTaskProgress(phase="scratchpad", message=item)
                                elif isinstance(item, Cell):
                                    cell = item
                            result_text = format_cell_result(cell) if cell else "No result produced."

                            yield StreamToolResult(content=json.dumps(asdict(cell)))

                            if self._episodic is not None and cell is not None:
                                self._episodic.log_turn(
                                    self._turn_count + 1,
                                    "scratchpad",
                                    (cell.stdout or "")[:2000],
                                    description=description,
                                )
                    else:
                        result_text = await dispatch_tool(self, tc.name, tc.input)
                        if tc.name == "scratchpad" and tc.input.get("action") == "dump":
                            yield StreamToolResult(content=result_text)
                            result_text = (
                                "The full notebook has been displayed to the user above. "
                                "Do not repeat it. Here is the content for your reference:\n\n" + result_text
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
                yield StreamContextCompacted(message="Context history has been summarized to free up space.")
                async for event in self._llm.plan_stream(
                    system=system,
                    messages=self._history,
                    tools=tools,
                ):
                    yield event
                    if isinstance(event, StreamComplete):
                        response = event

            if response is None:
                self._last_tool_round = tool_round
                return
            llm_response = response.response

            if not _compacted_this_turn and llm_response.usage.context_pressure > CONTEXT_PRESSURE_THRESHOLD:
                await self._summarize_history()
                self._compact_scratchpads()
                _compacted_this_turn = True
                yield StreamContextCompacted(message="Context was getting long — older history has been summarized.")

        self._last_tool_round = tool_round

        # Text-only final response — append to history
        reply = llm_response.content or ""
        self._history.append({"role": "assistant", "content": reply})

    def _maybe_consolidate_scratchpads(self) -> None:
        """Check if any scratchpad sessions warrant consolidation and fire it off."""
        from .memory.consolidator import Consolidator

        consolidator = Consolidator()
        if self._scratchpads._pad is not None:
            cells = list(self._scratchpads._pad.cells)
            if consolidator.should_replay(cells):
                asyncio.create_task(self._consolidate(cells))

    async def _consolidate(self, cells: list) -> None:
        """Run offline consolidation on a completed scratchpad session."""
        from .memory.consolidator import Consolidator

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
    is_error = any(marker in result_text for marker in ("[error]", "Task failed:", "failed", "timed out", "Rejected:"))
    if is_error:
        error_streak[tool_name] = error_streak.get(tool_name, 0) + 1
    else:
        error_streak[tool_name] = 0
        resilience_nudged.discard(tool_name)

    streak = error_streak.get(tool_name, 0)
    if streak >= RESILIENCE_NUDGE_AT and tool_name not in resilience_nudged:
        result_text += RESILIENCE_NUDGE_PROMPT
        resilience_nudged.add(tool_name)

    if streak >= MAX_CONSECUTIVE_ERRORS:
        result_text += MAX_CONSECUTIVE_ERRORS_PROMPT.format(
            tool_name=tool_name,
            max_consecutive_errors=MAX_CONSECUTIVE_ERRORS,
        )

    return result_text
