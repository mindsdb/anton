from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from anton.core.backends.base import Cell, ScratchpadRuntimeFactory
from anton.core.backends.local import local_scratchpad_runtime_factory
from anton.core.datasources.data_vault import DataVault
from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext
from anton.core.memory.cerebellum import Cerebellum
from anton.core.memory.skills import SkillStore
from anton.core.tools.recall_skill import RECALL_SKILL_TOOL
from anton.core.llm.prompts import RESILIENCE_NUDGE
from anton.core.llm.provider import (
    ContextOverflowError,
    StreamComplete,
    StreamContextCompacted,
    StreamEvent,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolResult,
    TokenLimitExceeded,
)
from anton.core.backends.manager import ScratchpadManager
from anton.core.tools.registry import ToolRegistry
from anton.core.tools.tool_defs import (
    SCRATCHPAD_TOOL,
    MEMORIZE_TOOL,
    RECALL_TOOL,
    ToolDef,
)
from anton.core.utils.scratchpad import prepare_scratchpad_exec, format_cell_result

from anton.explainability import ExplainabilityCollector, ExplainabilityStore

from anton.utils.datasources import (
    build_datasource_context,
    scrub_credentials,
)
from anton.core.settings import CoreSettings


if TYPE_CHECKING:
    from rich.console import Console
    from anton.context.self_awareness import SelfAwarenessContext
    from anton.chat_ui import EscapeWatcher
    from anton.core.llm.client import LLMClient
    from anton.core.memory.cortex import Cortex
    from anton.core.memory.episodes import EpisodicMemory
    from anton.memory.history_store import HistoryStore
    from anton.workspace import Workspace


@dataclass
class ChatSessionConfig:
    """All construction parameters for a ChatSession.

    Separates configuration assembly (the host app's job) from session
    orchestration (the core's job). Hosts build this object and pass it
    to ChatSession — the session never needs to know where values came from.
    """

    llm_client: LLMClient
    runtime_factory: ScratchpadRuntimeFactory = field(default=local_scratchpad_runtime_factory)
    cells: list[Cell] | None = None
    settings: CoreSettings | None = None
    self_awareness: SelfAwarenessContext | None = None
    cortex: Cortex | None = None
    episodic: EpisodicMemory | None = None
    system_prompt_context: SystemPromptContext = field(default_factory=SystemPromptContext)
    workspace: Workspace | None = None
    data_vault: DataVault | None = None
    console: Console | None = None
    initial_history: list[dict] | None = None
    history_store: HistoryStore | None = None
    session_id: str | None = None
    proactive_dashboards: bool = False
    tools: list[ToolDef] = field(default_factory=list)


class ChatSession:
    """Manages a multi-turn conversation with tool-call delegation."""

    def __init__(self, config: ChatSessionConfig) -> None:
        s = config.settings or CoreSettings()
        self._max_tool_rounds = s.max_tool_rounds
        self._max_continuations = s.max_continuations
        self._context_pressure_threshold = s.context_pressure_threshold
        self._max_consecutive_errors = s.max_consecutive_errors
        self._resilience_nudge_at = s.resilience_nudge_at
        self._token_status_cache_ttl = s.token_status_cache_ttl
        self._llm = config.llm_client
        self._self_awareness = config.self_awareness
        self._cortex = config.cortex
        self._episodic = config.episodic
        self._system_prompt_context = config.system_prompt_context
        self._proactive_dashboards = config.proactive_dashboards
        self._extra_tools = config.tools
        self._workspace = config.workspace
        self._data_vault = config.data_vault
        self._console = config.console
        self._history: list[dict] = (
            list(config.initial_history) if config.initial_history else []
        )
        self._pending_memory_confirmations: list = []
        self._turn_count = (
            sum(1 for m in self._history if m.get("role") == "user")
            if config.initial_history
            else 0
        )
        self._history_store = config.history_store
        self._session_id = config.session_id
        self._cancel_event = asyncio.Event()
        self._escape_watcher: EscapeWatcher | None = None
        self._active_datasource: str | None = None

        coding_provider = config.llm_client.coding_provider
        coding_conn = coding_provider.export_connection_info()
        self._scratchpads = ScratchpadManager(
            runtime_factory=config.runtime_factory,
            coding_provider=coding_conn.provider,
            coding_model=config.llm_client.coding_model,
            coding_api_key=coding_conn.api_key or "",
            coding_base_url=coding_conn.base_url or "",
            workspace_path=config.workspace.base if config.workspace else None,
        )

        self.tool_registry = ToolRegistry()
        # Procedural memory: brain-inspired skills (Stage 1 = declarative).
        # Lives at ~/.anton/skills/<label>/. The recall_skill tool retrieves
        # entries on demand and increments per-stage usage counters.
        self._skill_store = SkillStore()
        # Cerebellum: supervised error learning over scratchpad cells.
        # Buffers errored/warning cells across the turn, runs one diff
        # call at end-of-turn, and encodes lessons via cortex.encode().
        # Wired into the dispatcher's observer list below.
        self._cerebellum = Cerebellum(
            cortex=self._cortex,
            llm=self._llm,
        )
        # Scratchpad observers — list of objects with on_pre_execute /
        # on_post_execute. Fired by handle_scratchpad around pad.execute.
        # The runtime never sees this list; observation lives at the
        # dispatcher layer to keep local/remote runtimes interchangeable.
        self._scratchpad_observers: list = [self._cerebellum]
        self._explainability_store = (
            ExplainabilityStore(config.workspace.base) if config.workspace is not None else None
        )
        self._active_explainability: ExplainabilityCollector | None = None

    @property
    def history(self) -> list[dict]:
        return self._history

    def _apply_error_tracking(
        self,
        result_text: str,
        tool_name: str,
        error_streak: dict[str, int],
        resilience_nudged: set[str],
    ) -> str:
        """Track consecutive errors per tool and append nudge/circuit-breaker messages."""
        is_error = any(
            marker in result_text
            for marker in (
                "[error]",
                "Task failed:",
                "failed",
                "timed out",
                "Rejected:",
            )
        )
        if is_error:
            error_streak[tool_name] = error_streak.get(tool_name, 0) + 1
        else:
            error_streak[tool_name] = 0
            resilience_nudged.discard(tool_name)

        streak = error_streak.get(tool_name, 0)
        if streak >= self._resilience_nudge_at and tool_name not in resilience_nudged:
            result_text += RESILIENCE_NUDGE
            resilience_nudged.add(tool_name)

        if streak >= self._max_consecutive_errors:
            result_text += (
                f"\n\nSYSTEM: The '{tool_name}' tool has failed {self._max_consecutive_errors} times "
                "in a row. Stop retrying this approach. Either try a completely different "
                "strategy or tell the user what's going wrong so they can help."
            )

        return result_text

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

    def _record_cell_explainability(
        self, *, pad_name: str, description: str, cell
    ) -> None:
        if self._active_explainability is None:
            return
        if description:
            self._active_explainability.add_scratchpad_step(description)
        elif pad_name:
            self._active_explainability.add_scratchpad_step(
                f"work in scratchpad {pad_name}"
            )
        for query in getattr(cell, "explainability_queries", []) or []:
            if not isinstance(query, dict):
                continue
            self._active_explainability.add_query(
                datasource=str(query.get("datasource", "")),
                sql=str(query.get("sql", "")),
                engine=(
                    str(query.get("engine"))
                    if query.get("engine") is not None
                    else None
                ),
                status=str(query.get("status", "ok")),
                error_message=(
                    str(query.get("error_message"))
                    if query.get("error_message") is not None
                    else None
                ),
            )
        self._active_explainability.add_sources_from_text(
            getattr(cell, "code", ""),
            getattr(cell, "stdout", ""),
            getattr(cell, "logs", ""),
        )
        self._active_explainability.add_inferred_queries_from_code(
            getattr(cell, "code", "")
        )

    async def _build_system_prompt(self, user_message: str = "") -> str:
        import datetime as _dt

        _now = _dt.datetime.now()
        _current_datetime = _now.strftime("%A, %B %d, %Y at %I:%M %p")

        # Inject memory context (replaces old self_awareness)
        memory_section = ""
        if self._cortex is not None:
            memory_section = await self._cortex.build_memory_context(user_message)

        sa_section = ""
        if self._self_awareness is not None and self._cortex is None:
            # Fallback for legacy usage (tests, etc.)
            sa_section = self._self_awareness.build_prompt_section()

        # Inject anton.md project context (user-written takes priority)
        md_context = ""
        if self._workspace is not None:
            md_context = self._workspace.build_anton_md_context()

        # Inject connected datasource context without credentials
        ds_ctx = build_datasource_context(self._data_vault, active_only=self._active_datasource)

        # Ensure the registry is populated before we extract tool prompts.
        self._build_tools()

        prompt_builder = ChatSystemPromptBuilder()
        prompt = prompt_builder.build(
            current_datetime=_current_datetime,
            system_prompt_context=self._system_prompt_context,
            proactive_dashboards=self._proactive_dashboards,
            tool_defs=self.tool_registry.get_tool_defs(),
            memory_context=memory_section,
            project_context=md_context,
            self_awareness_context=sa_section,
            datasource_context=ds_ctx,
            skill_store=self._skill_store,
        )

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
        if not self.tool_registry:
            self._build_core_tools()
            for tool in self._extra_tools:
                self.tool_registry.register_tool(tool)
        return self.tool_registry.dump()

    def _build_core_tools(self) -> None:
        scratchpad_tool = SCRATCHPAD_TOOL
        pkg_list = self._scratchpads.available_packages
        if pkg_list:
            notable = sorted(p for p in pkg_list if p.lower() in self._NOTABLE_PACKAGES)
            if notable:
                pkg_line = ", ".join(notable)
                extra = f"\n\nInstalled packages ({len(pkg_list)} total, notable: {pkg_line})."
            else:
                extra = f"\n\nInstalled packages: {len(pkg_list)} total (standard library plus dependencies)."
            scratchpad_tool.description = scratchpad_tool.description + extra

        # Inject scratchpad wisdom from memory (procedural priming)
        if self._cortex is not None:
            wisdom = self._cortex.get_scratchpad_context()
            if wisdom:
                scratchpad_tool.description += (
                    f"\n\nLessons from past sessions:\n{wisdom}"
                )

        self.tool_registry.register_tool(scratchpad_tool)

        if self._cortex is not None or self._self_awareness is not None:
            self.tool_registry.register_tool(MEMORIZE_TOOL)

        if self._episodic is not None and self._episodic.enabled:
            self.tool_registry.register_tool(RECALL_TOOL)

        # Procedural memory retrieval — always available, no-op if no skills.
        self.tool_registry.register_tool(RECALL_SKILL_TOOL)

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
        for pad in self._scratchpads.pads.values():
            if pad._compact_cells():
                compacted = True
        return compacted

    def _schedule_cerebellum_flush(self) -> None:
        """Fire the cerebellum's batched diff pass without blocking the turn.

        The cerebellum buffered any errored / warning cells across the
        turn via its observer hooks. Now we kick off the (at most one)
        LLM diff call as a background task — the user gets their reply
        immediately, and any extracted lessons get encoded into the
        existing wisdom store before the next turn typically begins.

        Best-effort: if there's no buffered work or no event loop, this
        is a no-op. Exceptions in the background task are swallowed
        because they're already logged inside cerebellum.flush().
        """
        cb = getattr(self, "_cerebellum", None)
        if cb is None:
            return
        if cb.buffered_count == 0:
            return
        try:
            asyncio.create_task(cb.flush())
        except RuntimeError:
            # No running loop (e.g. called from a sync context in tests).
            # Cerebellum learning is best-effort, so just drop the buffer.
            cb.reset()

    async def turn(self, user_input: str | list[dict]) -> str:
        self._history.append({"role": "user", "content": user_input})

        user_msg_str = user_input if isinstance(user_input, str) else ""
        tools = self._build_tools()
        system = await self._build_system_prompt(user_msg_str)

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
        if response.usage.context_pressure > self._context_pressure_threshold:
            await self._summarize_history()
            self._compact_scratchpads()

        # Handle tool calls
        tool_round = 0
        error_streak: dict[str, int] = {}
        resilience_nudged: set[str] = set()

        while response.tool_calls:
            tool_round += 1
            if tool_round > self._max_tool_rounds:
                self._history.append(
                    {"role": "assistant", "content": response.content or ""}
                )
                self._history.append(
                    {
                        "role": "user",
                        "content": (
                            f"SYSTEM: You have used {self._max_tool_rounds} tool-call rounds on this turn. "
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
                    result_text = await self.tool_registry.dispatch_tool(
                        self, tc.name, tc.input
                    )
                except Exception as exc:
                    result_text = f"Tool '{tc.name}' failed: {exc}"

                result_text = scrub_credentials(result_text)
                result_text = self._apply_error_tracking(
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
            if response.usage.context_pressure > self._context_pressure_threshold:
                await self._summarize_history()
                self._compact_scratchpads()

        # Text-only response
        reply = response.content or ""
        self._history.append({"role": "assistant", "content": reply})

        # Periodic memory vacuum (Systems Consolidation)
        if self._cortex is not None and self._cortex.mode != "off":
            self._cortex.maybe_vacuum()

        # Cerebellar consolidation — fire-and-forget so the user gets
        # their reply immediately while supervised error learning runs
        # in the background. Brain analogue: cerebellar plasticity
        # operates in parallel with continued action, not blocking it.
        self._schedule_cerebellum_flush()

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
        self._active_explainability = ExplainabilityCollector(
            self._explainability_store,
            turn=self._turn_count + 1,
            user_message=user_msg_str,
        )

        try:
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
        finally:
            if self._active_explainability is not None:
                self._active_explainability.finalize(
                    "".join(assistant_text_parts)[:2000]
                )

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

        # Cerebellar consolidation — same fire-and-forget contract as
        # the non-streaming turn. Lets the user-facing stream finish
        # immediately while supervised error learning runs in the background.
        self._schedule_cerebellum_flush()

    async def _stream_and_handle_tools(
        self, user_message: str = ""
    ) -> AsyncIterator[StreamEvent]:
        """Stream one LLM call, handle tool loops, yield all events."""
        tools = self._build_tools()
        system = await self._build_system_prompt(user_message)

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
            and llm_response.usage.context_pressure > self._context_pressure_threshold
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
                if tool_round > self._max_tool_rounds:
                    _max_rounds_hit = True
                    self._history.append(
                        {"role": "assistant", "content": llm_response.content or ""}
                    )
                    self._history.append(
                        {
                            "role": "user",
                            "content": (
                                f"SYSTEM: You have used {self._max_tool_rounds} tool-call rounds on this turn. "
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
                import time as _time

                tool_results: list[dict] = []
                for tc in llm_response.tool_calls:
                    if self._episodic is not None:
                        self._episodic.log_turn(
                            self._turn_count + 1,
                            "tool_call",
                            str(tc.input)[:2000],
                            tool=tc.name,
                        )

                    _tool_t0 = _time.monotonic()

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

                                _sp_t0 = _time.monotonic()
                                from anton.core.backends.base import Cell

                                cell = None
                                async for item in pad.execute_streaming(
                                    code,
                                    description=description,
                                    estimated_time=estimated_time,
                                    estimated_seconds=estimated_seconds,
                                ):
                                    if self._cancel_event.is_set():
                                        await pad.cancel()
                                        break
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
                                if cell is not None:
                                    self._record_cell_explainability(
                                        pad_name=tc.input.get("name", ""),
                                        description=description,
                                        cell=cell,
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
                            result_text = await self.tool_registry.dispatch_tool(
                                self, tc.name, tc.input
                            )
                            if self._escape_watcher:
                                self._escape_watcher.resume()
                            yield StreamTaskProgress(
                                phase="analyzing",
                                message="Analyzing results...",
                            )
                        else:
                            # Non-scratchpad, non-interactive tool — track elapsed
                            yield StreamTaskProgress(
                                phase="tool_start",
                                message=tc.name,
                            )
                            result_text = await self.tool_registry.dispatch_tool(
                                self, tc.name, tc.input
                            )
                            _tool_elapsed = _time.monotonic() - _tool_t0
                            yield StreamTaskProgress(
                                phase="tool_done",
                                message=tc.name,
                                eta_seconds=_tool_elapsed,
                            )
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
                    result_text = self._apply_error_tracking(
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

                # Signal that tools are done and LLM is now reasoning
                _reasoning_t0 = _time.monotonic()
                yield StreamTaskProgress(
                    phase="reasoning_start", message="Thinking..."
                )

                # Stream follow-up
                response = None
                try:
                    async for event in self._llm.plan_stream(
                        system=system,
                        messages=self._history,
                        tools=tools,
                    ):
                        # Capture reasoning elapsed on first text or tool event
                        if _reasoning_t0 and isinstance(
                            event, (StreamTextDelta, StreamComplete)
                        ):
                            _reasoning_elapsed = _time.monotonic() - _reasoning_t0
                            _reasoning_t0 = 0  # only fire once
                            yield StreamTaskProgress(
                                phase="reasoning_done",
                                message="",
                                eta_seconds=_reasoning_elapsed,
                            )
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
                    > self._context_pressure_threshold
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

            if continuation >= self._max_continuations:
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
                        f"(attempt {continuation}/{self._max_continuations}).\n"
                        f"Verifier assessment: {reason}\n\n"
                        "Continue working on the original request. Pick up where you left off "
                        "and finish the remaining work. Do not repeat work already done."
                    ),
                }
            )
            yield StreamTaskProgress(
                phase="analyzing",
                message=f"Task incomplete — continuing ({continuation}/{self._max_continuations})...",
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
        from anton.core.memory.consolidator import Consolidator

        consolidator = Consolidator()
        for pad in self._scratchpads.pads.values():
            cells = list(pad.cells)
            if consolidator.should_replay(cells):
                asyncio.create_task(self._consolidate(cells))

    async def _consolidate(self, cells: list) -> None:
        """Run offline consolidation on a completed scratchpad session."""
        from anton.core.memory.consolidator import Consolidator

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
