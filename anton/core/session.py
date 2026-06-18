from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass, field
import json
import re
from typing import TYPE_CHECKING, List
import os

from anton.core.backends.base import Cell, ScratchpadRuntimeFactory
from anton.core.backends.local import local_scratchpad_runtime_factory
from anton.core.datasources.data_vault import DataVault
from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext
from anton.core.memory.acc import AnteriorCingulate
from anton.core.memory.base import Engram
from anton.core.memory.cerebellum import Cerebellum
from anton.core.memory.skills import SkillStore
from anton.core.tools.recall_skill import RECALL_SKILL_TOOL
from anton.core.llm.prompts import (
    RESILIENCE_NUDGE,
    SCRATCHPAD_SIZE_NUDGE,
    SCRATCHPAD_TIMEOUT_NUDGE,
)
from anton.core.llm.provider import (
    ContextOverflowError,
    StreamComplete,
    StreamContextCompacted,
    StreamEvent,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolResult,
    TokenLimitExceeded,
    ToolCall,
)
from anton.core.llm.tracing import (
    TraceContext,
    reset_trace_context,
    set_trace_context,
)
from anton.core.backends.manager import ScratchpadManager
from anton.core.tools.registry import ToolRegistry
from anton.core.tools.tool_defs import (
    CREATE_ARTIFACT_TOOL,
    LAUNCH_BACKEND_TOOL,
    LIST_ARTIFACTS_TOOL,
    MEMORIZE_TOOL,
    OPEN_ARTIFACT_TOOL,
    READ_IMAGE_TOOL,
    RECALL_TOOL,
    SCRATCHPAD_TOOL,
    UPDATE_ARTIFACT_METADATA_TOOL,
    ToolDef,
)
from anton.core.utils.scratchpad import (
    prepare_scratchpad_exec,
    format_cell_result,
    observe_scratchpad_cell,
)

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


def _extract_datasources(tool_call: ToolCall) -> List[str]:
    """Return unique datasource slugs referenced in scratchpad code via DS_*__ env vars."""
    if tool_call.name != "scratchpad":
        return []
    code = tool_call.input.get("code", "") if isinstance(tool_call.input, dict) else ""
    if not code:
        return []
    seen = set()

    for m in re.compile(r"\bDS_([A-Z0-9_]+?)__").finditer(code):
        seen.add(m.group(1).lower())
    return list(seen)

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
    # Identifier for the host harness driving this session (e.g. "cowork",
    # "cli"). Surfaced on telemetry / langfuse traces so the harness that
    # produced a given trace is filterable in the dashboard. None means the
    # host didn't identify itself.
    harness: str | None = None
    proactive_dashboards: bool = False
    tools: list[ToolDef] = field(default_factory=list)
    output_dir: str = ".anton/output"
    # Web tools — on by default. Each is independently resolved at session
    # construction into either a native provider capability (passed to the LLM
    # via ``native_web_tools``) or a handler-dispatched fallback ToolDef
    # (registered on the tool registry). See ChatSession.__init__.
    web_search_enabled: bool = True
    web_fetch_enabled: bool = True


class ChatSession:
    """Manages a multi-turn conversation with tool-call delegation."""

    def __init__(self, config: ChatSessionConfig) -> None:
        s = config.settings or CoreSettings()
        # Stash the full settings object (may be AntonSettings, CoreSettings,
        # or None). Tool handlers read host-only fields like
        # ``external_search_provider`` / ``exa_api_key`` via getattr so the
        # session stays decoupled from the host's settings shape.
        self._settings = config.settings
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
        self._output_dir = config.output_dir
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
        self._harness = config.harness
        # Set per-turn by `turn_stream` so any LLM call made during that
        # turn can read the current turn identifier (used by telemetry /
        # langfuse propagation in the provider layer).
        self._current_turn_id: int | None = None
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
            cells=config.cells,
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
        # Anterior Cingulate Cortex: turn-level pattern detection.
        # Where the cerebellum looks at one cell and asks "did this
        # cell do what it claimed", the ACC looks at the whole turn
        # and asks "is the same failure pattern firing more than
        # once". Emit points are scattered (scratchpad dispatcher,
        # tool dispatch, history-repair, round-cap) rather than
        # routed through the scratchpad observer list, because most
        # of what the ACC watches isn't scratchpad-scoped. The
        # session holds the reference; emit sites call
        # `session._acc.observe(kind, detail, ...)` directly.
        #
        # has_similar_lesson: cheap substring check against the
        # current rules.md content. Avoids re-encoding the same
        # rule every turn. Embedding similarity is a v2 upgrade.
        def _acc_has_similar(rule: str) -> bool:
            cortex = getattr(self, "_cortex", None)
            hc = getattr(cortex, "global_hc", None) if cortex else None
            if hc is None:
                return False
            try:
                existing = hc.recall_rules() or ""
            except Exception:
                return False
            probe = (rule or "")[:60].lower()
            return bool(probe) and probe in existing.lower()

        self._acc = AnteriorCingulate(has_similar_lesson=_acc_has_similar)
        # ANTON_ACC_MODE controls how aggressively ACC affects the
        # turn. Mirrors ANTON_MEMORY_MODE for shape consistency:
        #   "off"     — ACC observes nothing (skipped at every emit site).
        #   "passive" — Layer 1: lessons drain to memory at end-of-turn,
        #               next turn's system prompt picks them up. SAFE
        #               DEFAULT — adds no surface-area to the turn loop.
        #   "active"  — Layer 2: ALSO inject lessons inline as text
        #               blocks in tool_results so the LLM sees them on
        #               the very next round. Stronger learning signal,
        #               but more invasive — the LLM has to handle the
        #               nudge gracefully without confusing it for a
        #               user instruction.
        _mode_raw = os.environ.get("ANTON_ACC_MODE", "passive").strip().lower()
        self._acc_mode = _mode_raw if _mode_raw in ("off", "passive", "active") else "passive"
        # Scratchpad observers — list of objects with on_pre_execute /
        # on_post_execute. Fired by handle_scratchpad around pad.execute.
        # The runtime never sees this list; observation lives at the
        # dispatcher layer to keep local/remote runtimes interchangeable.
        # ACC is intentionally NOT in this list — its emit footprint
        # is broader than scratchpad cells (it also needs to see tool
        # calls, history repairs, the round cap), so it's wired via
        # direct `session._acc.observe(...)` at each emit site.
        self._scratchpad_observers: list = [self._cerebellum]
        self._explainability_store = (
            ExplainabilityStore(config.workspace.base) if config.workspace is not None else None
        )
        self._active_explainability: ExplainabilityCollector | None = None
        # Per-turn guard: set to True by the recovery helpers or the
        # proactive pressure check after they summarize history; reset
        # at the start of each turn. Prevents double-summarization when
        # the post-recovery response still reports high pressure.
        self._compacted_this_turn = False
        # Backends launched via the launch_backend tool. Keyed by
        # artifact slug; each entry holds the asyncio.subprocess.Process
        # plus its port. Reaped in close() so backend processes don't
        # outlive the chat session.
        self._tracked_backends: dict[str, dict] = {}

        # Resolve web tool routing once per session. ``_native_web_tools`` is
        # the set the planning provider will execute server-side (passed
        # through every ``plan*`` call); ``_fallback_web_tools`` is the set
        # we run ourselves via handler-dispatched ToolDefs (registered in
        # ``_build_core_tools``). The two sets are disjoint by construction.
        desired_web: set[str] = set()
        if config.web_search_enabled:
            desired_web.add("web_search")
        if config.web_fetch_enabled:
            desired_web.add("web_fetch")
        provider_native = self._llm.planning_provider.native_web_tools()
        self._native_web_tools: set[str] = desired_web & provider_native
        self._fallback_web_tools: set[str] = desired_web - provider_native

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
            nudge = self._select_resilience_nudge(tool_name, result_text)
            if nudge:
                result_text += nudge
                resilience_nudged.add(tool_name)

        if streak >= self._max_consecutive_errors:
            result_text += (
                f"\n\nSYSTEM: The '{tool_name}' tool has failed {self._max_consecutive_errors} times "
                "in a row. Stop retrying this approach. Either try a completely different "
                "strategy or tell the user what's going wrong so they can help."
            )

        return result_text

    @staticmethod
    def _select_resilience_nudge(tool_name: str, result_text: str) -> str:
        """Pick the right soft-nudge for a repeated failure.

        The generic RESILIENCE_NUDGE is scrape/fetch advice ("try a public
        API / archive.org / different headers"). That actively misdirects a
        scratchpad failure: a cell that's too big or too slow doesn't need a
        different data source, it needs to be chunked or scoped down. Route
        scratchpad failures to size/timeout-specific guidance by inspecting
        the error text; everything else keeps the generic nudge. Returns ""
        when no useful nudge applies (e.g. a generic scratchpad runtime error
        like NameError, where scraping advice would be nonsense).
        """
        if tool_name != "scratchpad":
            return RESILIENCE_NUDGE
        low = result_text.lower()
        if "timed out" in low or "inactivity" in low:
            return SCRATCHPAD_TIMEOUT_NUDGE
        # Match the empty-code dispatcher message specifically — generic
        # phrases like "too large"/"truncated" appear in unrelated errors
        # (e.g. a MySQL "Data truncated for column" warning) and would
        # misfire the chunking advice.
        if "argument was empty" in low:
            return SCRATCHPAD_SIZE_NUDGE
        return ""

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
        self._append_history(
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

    # ── History append helpers ─────────────────────────────────────────
    #
    # Most chat APIs require `messages` to alternate user / assistant
    # roles strictly:
    #
    #   • Anthropic — rejects two same-role messages back-to-back
    #     with a 400.
    #   • Mistral, Groq, and most "OpenAI-compatible" relays (mdb.ai,
    #     Together.ai, Fireworks, llama.cpp servers) — same.
    #   • OpenAI proper — technically tolerates non-alternating, but
    #     model output quality drops when fed consecutive same-role
    #     turns; the model tends to fold them together or treat the
    #     second as an interruption.
    #
    # Anton appends to history from a dozen places — tool_results,
    # SYSTEM-recovery prompts, intermediate assistant text, etc. —
    # and the auto-retry path used to be able to slip two user
    # messages in a row (a synthetic tool_result append + a
    # SYSTEM-recovery append back-to-back), which any strict
    # provider rejects.
    #
    # Centralising every append through `_append_history` enforces
    # the alternation invariant at the source — *before* any provider
    # sees the messages — so clean output is portable across every
    # provider we support today and any we add tomorrow. When the
    # new message has the same role as the previous one, the helper
    # merges them rather than pushing a new entry. The merge is
    # content-shape-aware: list-of-blocks + list-of-blocks →
    # concatenated list, string + string → list-of-text-blocks,
    # mixed shapes get normalised to a list-of-blocks (the form
    # every chat API accepts for both roles).

    @staticmethod
    def _coerce_to_block_list(content) -> list[dict]:
        """Normalise a message's content into a list of blocks.

        Strings become a single ``{"type": "text", "text": ...}``.
        Existing block lists pass through unchanged. Anything else
        (None, dicts) is wrapped sensibly.
        """
        if isinstance(content, list):
            return list(content)
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, dict):
            return [content]
        return []

    def _append_history(self, msg: dict) -> None:
        """Append `msg` to history, preserving role alternation.

        If the previous message has the same role, merge the new
        content INTO the previous message instead of pushing a fresh
        entry. The merged form always uses a list-of-blocks so the
        Anthropic API accepts it whether the originals were strings
        or already block lists.

        Direct ``self._append_history(...)`` calls inside this class
        should be avoided — every append site routes through here
        so the invariant is impossible to violate accidentally.
        """
        if not isinstance(msg, dict):
            return
        role = msg.get("role")
        if role not in ("user", "assistant"):
            # System-role messages aren't expected in `history`
            # (system goes via the `system` argument on the
            # provider), but if anything ever drops one in, just
            # accept it without merging.
            self._history.append(msg)
            return
        # Empty-content append → no-op (would just create a phantom
        # turn that the API may reject).
        content = msg.get("content")
        if content in (None, "", []):
            return
        if not self._history:
            self._history.append(msg)
            return
        prev = self._history[-1]
        if prev.get("role") != role:
            self._history.append(msg)
            return
        # Same-role back-to-back. Merge by concatenating block lists.
        merged_blocks = (
            self._coerce_to_block_list(prev.get("content"))
            + self._coerce_to_block_list(content)
        )
        self._history[-1] = {**prev, "role": role, "content": merged_blocks}
        import logging as _logging
        _logging.getLogger(__name__).info(
            "Merged consecutive %s messages in history (would have violated "
            "Anthropic role alternation). Combined block count: %d.",
            role, len(merged_blocks),
        )

    def _validate_history_for_provider(self, messages: list[dict]) -> None:
        """Defensive pre-flight: warn (don't raise) if the messages
        list still violates the chat-API structural invariants.

        Provider-agnostic. The two assertions below are what every
        major chat API expects — Anthropic and most OpenAI-compatible
        relays enforce them strictly; even providers that technically
        tolerate non-alternating messages produce better output when
        the rules hold.

        With `_append_history` at every append site this should never
        fire; treating it as a paranoia check that surfaces in logs
        if a future code path forgets to use the helper. We don't
        raise — sending the request and letting the provider return
        its own 400 is more useful for debugging than crashing here.
        """
        import logging as _logging
        log = _logging.getLogger(__name__)
        if not messages:
            return
        if messages[0].get("role") != "user":
            log.warning(
                "History pre-flight: first message has role %r, expected 'user'. "
                "The provider call is likely to 400.",
                messages[0].get("role"),
            )
        for i in range(1, len(messages)):
            prev_role = messages[i - 1].get("role")
            curr_role = messages[i].get("role")
            if prev_role == curr_role and prev_role in ("user", "assistant"):
                log.warning(
                    "History pre-flight: consecutive %s messages at indices "
                    "%d and %d. Most providers will reject this; OpenAI may "
                    "accept it but produce worse output. Some append site "
                    "isn't routing through _append_history.",
                    prev_role, i - 1, i,
                )
                # Only flag the first violation per call; the noise
                # of a longer broken stretch isn't useful.
                return

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
            output_dir=self._output_dir,
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
        self.tool_registry.register_tool(READ_IMAGE_TOOL)

        if self._cortex is not None or self._self_awareness is not None:
            self.tool_registry.register_tool(MEMORIZE_TOOL)

        if self._episodic is not None and self._episodic.enabled:
            self.tool_registry.register_tool(RECALL_TOOL)

        # Procedural memory retrieval — always available, no-op if no skills.
        self.tool_registry.register_tool(RECALL_SKILL_TOOL)

        # Handler-dispatched web tools — registered only when the LLM provider
        # does NOT execute them natively. On Anthropic / OpenAI BYOK / mdb.ai
        # passthrough, ``_fallback_web_tools`` is empty and these tools never
        # appear in the registry; the model uses the provider's server-side
        # web tools instead and Anton's dispatch loop never sees a ``tool_use``
        # for them. See ``anton/core/tools/web_tools.py`` for the handlers.
        if "web_search" in self._fallback_web_tools:
            from anton.core.tools.web_tools import WEB_SEARCH_FALLBACK_TOOL
            self.tool_registry.register_tool(WEB_SEARCH_FALLBACK_TOOL)
        if "web_fetch" in self._fallback_web_tools:
            from anton.core.tools.web_tools import WEB_FETCH_FALLBACK_TOOL
            self.tool_registry.register_tool(WEB_FETCH_FALLBACK_TOOL)

        # Artifacts — only register when a workspace is bound to the
        # session. Bare-cwd CLI sessions without `resolve_workspace`
        # have nowhere to write artifacts to, and the tool handlers
        # would just return error strings — better to hide the tools
        # entirely so the LLM doesn't try to use them.
        if self._workspace is not None:
            self.tool_registry.register_tool(CREATE_ARTIFACT_TOOL)
            self.tool_registry.register_tool(LIST_ARTIFACTS_TOOL)
            self.tool_registry.register_tool(OPEN_ARTIFACT_TOOL)
            self.tool_registry.register_tool(UPDATE_ARTIFACT_METADATA_TOOL)
            self.tool_registry.register_tool(LAUNCH_BACKEND_TOOL)

    async def close(self) -> None:
        """Clean up scratchpads and other resources."""
        await self._reap_tracked_backends()
        await self._scratchpads.close_all()

    async def _reap_tracked_backends(self) -> None:
        """Terminate every backend launched via launch_backend.

        SIGTERM first, then SIGKILL after a short grace period. Errors
        are swallowed — close() must not raise on shutdown.
        """
        for slug, info in list(self._tracked_backends.items()):
            proc = info.get("proc")
            if proc is None or proc.returncode is not None:
                continue
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            except (ProcessLookupError, OSError):
                pass
        self._tracked_backends.clear()

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

    def _seal_dangling_tool_uses(self, reason: str = "interrupted") -> int:
        """Append synthetic `tool_result` blocks for any unmatched
        `tool_use` blocks in the last assistant message.

        Anthropic's API requires every assistant `tool_use` to be
        followed by a user message containing a `tool_result` for the
        same id. If `_stream_and_handle_tools` raised after the
        tool_use was committed to history but before the dispatcher
        appended its tool_result (e.g. an HTTP failure inside the LLM
        call, an exception in a tool handler), the next API request
        sees an orphan tool_use and returns a 400.

        Call this BEFORE appending any non-tool-result user message
        on an error path. It walks back to the last assistant turn
        with tool_use blocks and inserts a user message carrying
        synthetic `is_error: true` results for whichever ids didn't
        get acknowledged in the immediately following message.

        Returns the number of synthetic results inserted (0 if the
        history is already clean).
        """
        if not self._history:
            return 0
        # Find the last assistant message with tool_use blocks.
        last_assistant_idx = None
        for j in range(len(self._history) - 1, -1, -1):
            msg = self._history[j]
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_use"
                    for b in content
                ):
                    last_assistant_idx = j
                break
        if last_assistant_idx is None:
            return 0
        assistant = self._history[last_assistant_idx]
        tool_use_ids = [
            b.get("id") for b in assistant["content"]
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
        ]
        if not tool_use_ids:
            return 0
        # Gather the ids ALREADY acknowledged by the next message
        # (if any). The seal only adds what's missing.
        ack_ids: set = set()
        next_msg = (
            self._history[last_assistant_idx + 1]
            if last_assistant_idx + 1 < len(self._history)
            else None
        )
        if isinstance(next_msg, dict) and next_msg.get("role") == "user":
            nc = next_msg.get("content")
            if isinstance(nc, list):
                for b in nc:
                    if (
                        isinstance(b, dict)
                        and b.get("type") == "tool_result"
                        and b.get("tool_use_id")
                    ):
                        ack_ids.add(b["tool_use_id"])
        missing = [tid for tid in tool_use_ids if tid not in ack_ids]
        if not missing:
            return 0
        synth_blocks = [
            {
                "type": "tool_result",
                "tool_use_id": tid,
                "content": f"[{reason} — tool call did not complete]",
                "is_error": True,
            }
            for tid in missing
        ]
        if (
            isinstance(next_msg, dict)
            and next_msg.get("role") == "user"
            and isinstance(next_msg.get("content"), list)
        ):
            # Splice into the existing user message.
            next_msg["content"] = synth_blocks + next_msg["content"]
        else:
            # Insert a fresh user message right after the assistant.
            self._history.insert(
                last_assistant_idx + 1,
                {"role": "user", "content": synth_blocks},
            )
        # ACC: emit history_repair so detect_repair_churn can fire
        # when the LLM is generating malformed tool_use/result pairs
        # repeatedly. One repair is a hiccup; three in a turn is the
        # conversation derailing.
        self._acc_observe(
            "history_repair",
            {"reason": reason, "sealed_count": len(missing)},
            severity=5,
        )
        return len(missing)

    def hard_truncate_history(self, keep: int = 4) -> None:
        """Last-resort history truncation for persistent context overflow.

        Summarize-and-compact can fall flat when a single message is huge,
        or when the system prompt plus tools already exhaust context. This
        throws away everything except the last `keep` messages, preserving
        tool_use/tool_result pairing and the API rule that the first
        message must be from the user.
        """
        if len(self._history) <= keep:
            return
        tail = list(self._history[-keep:])

        # Strip leading messages that would leave tail in an invalid state:
        # - assistant at head (API requires user first)
        # - user whose only blocks are tool_result references (their
        #   matching tool_use is in the dropped prefix, so they're orphaned)
        # Repeat because dropping one can expose another. A user message
        # with mixed content keeps its non-tool_result blocks.
        while tail:
            head = tail[0]
            role = head.get("role")
            if role == "assistant":
                tail.pop(0)
                continue
            if role == "user":
                content = head.get("content")
                if isinstance(content, list):
                    filtered = [
                        b for b in content
                        if not (isinstance(b, dict) and b.get("type") == "tool_result")
                    ]
                    if not filtered:
                        tail.pop(0)
                        continue
                    if len(filtered) != len(content):
                        tail[0] = {**head, "content": filtered}
            break

        placeholder = {
            "role": "user",
            "content": "[Earlier conversation was truncated due to persistent context overflow.]",
        }
        separator = {"role": "assistant", "content": "Understood."}
        # If the tail starts with assistant, the separator above would
        # land us with assistant→assistant. Drop the separator in that
        # case — the tail's first assistant message can directly
        # respond to the placeholder user message.
        if tail and tail[0].get("role") == "assistant":
            self._history = [placeholder, *tail]
        else:
            self._history = [placeholder, separator, *tail]

    async def plan_with_recovery(
        self,
        *,
        system: str,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        messages_factory: Callable[[], list[dict]] | None = None,
    ):
        """Call _llm.plan with three-tier ContextOverflowError recovery.

        Attempts, in order: normal → summarize+compact → hard-truncate.
        A fourth overflow propagates to the caller.

        `messages_factory` is re-invoked before each attempt so callers
        that build synthetic message lists (e.g. verification with an
        appended prompt) see the latest post-compaction history.
        """
        factory = messages_factory if messages_factory is not None else (lambda: self._history)
        # Defensive pre-flight — log a warning if the message list
        # would violate the role-alternation invariant that every
        # major chat API expects (strict on Anthropic / Mistral /
        # most OpenAI-compatible relays; soft-required on OpenAI for
        # output quality). Should never fire now that every append
        # routes through `_append_history`; catches future code paths
        # that forget the helper.
        def factory_validated():
            msgs = factory()
            self._validate_history_for_provider(msgs)
            return msgs

        kwargs: dict = {"system": system}
        if tools is not None:
            kwargs["tools"] = tools
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        # Native web tools are a per-session capability — forward to every
        # planning call automatically so callers don't have to remember.
        if self._native_web_tools:
            kwargs["native_web_tools"] = self._native_web_tools

        try:
            return await self._llm.plan(messages=factory_validated(), **kwargs)
        except ContextOverflowError:
            pass

        await self._summarize_history()
        self._compact_scratchpads()
        self._compacted_this_turn = True
        try:
            return await self._llm.plan(messages=factory_validated(), **kwargs)
        except ContextOverflowError:
            pass

        self.hard_truncate_history()
        return await self._llm.plan(messages=factory_validated(), **kwargs)

    async def plan_stream_with_recovery(
        self,
        *,
        system: str,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        messages_factory: Callable[[], list[dict]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming analogue of plan_with_recovery.

        Yields all events from the underlying plan_stream call. On
        ContextOverflowError, yields StreamContextCompacted, shrinks
        history (summarize+compact, then hard-truncate on a repeat
        overflow), and restarts the stream. A fourth overflow propagates.
        """
        factory = messages_factory if messages_factory is not None else (lambda: self._history)
        # Same defensive pre-flight as plan_with_recovery — see the
        # comment there for the why.
        def factory_validated():
            msgs = factory()
            self._validate_history_for_provider(msgs)
            return msgs

        kwargs: dict = {"system": system}
        if tools is not None:
            kwargs["tools"] = tools
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if self._native_web_tools:
            kwargs["native_web_tools"] = self._native_web_tools

        try:
            async for event in self._llm.plan_stream(messages=factory_validated(), **kwargs):
                yield event
            return
        except ContextOverflowError:
            pass

        await self._summarize_history()
        self._compact_scratchpads()
        self._compacted_this_turn = True
        yield StreamContextCompacted(
            message="Context was getting long — older history has been summarized."
        )
        try:
            async for event in self._llm.plan_stream(messages=factory_validated(), **kwargs):
                yield event
            return
        except ContextOverflowError:
            pass

        self.hard_truncate_history()
        yield StreamContextCompacted(
            message="Context still exceeded limits — older history was hard-truncated."
        )
        async for event in self._llm.plan_stream(messages=factory_validated(), **kwargs):
            yield event

    def _acc_observe(
        self,
        kind: str,
        detail: dict | None = None,
        *,
        severity: int = 1,
        round_idx: int = 0,
    ) -> None:
        """Safe-emit wrapper for ACC events.

        Returns silently when:
          - the ACC isn't attached (defensive — should always be set),
          - the cortex is disabled (`mode == "off"`), so observation
            without persistence is pointless,
          - `observe()` raises (e.g. unknown kind from a stale call site).

        Emit sites call this rather than touching `self._acc` directly
        so that adding/renaming kinds, or turning the ACC off via a
        future env var, lives in one place.
        """
        acc = getattr(self, "_acc", None)
        if acc is None:
            return
        if getattr(self, "_acc_mode", "passive") == "off":
            return
        cortex = getattr(self, "_cortex", None)
        if cortex is not None and getattr(cortex, "mode", "") == "off":
            return
        try:
            acc.observe(kind, detail or {}, severity=severity, round_idx=round_idx)
        except ValueError:
            # Unknown event kind from a stale emit site — log via the
            # cerebellum's logger contract once we have one; for now,
            # swallow so observation drift never breaks a turn.
            pass

    def _acc_maybe_nudge(self, tool_results: list[dict]) -> int:
        """Layer 2 — mid-turn nudging.

        If `ANTON_ACC_MODE == "active"`, run the ACC's per-round
        detection pass and append any newly-fired lessons as text
        blocks INSIDE the `tool_results` content list. They piggy-back
        on the user-role message that's about to be appended to
        history, so the LLM sees them on its very next round.

        Why text blocks alongside tool_result blocks (vs. a separate
        user message)? Anthropic's API allows a user message to mix
        types in its content array. Reusing the same message keeps the
        nudge tightly bound to the round that produced it and avoids
        introducing a new consecutive-user-message edge case that the
        history validator would have to learn about.

        Returns the number of nudges appended (mostly for tests /
        observability). Zero in passive mode, zero when no detectors
        newly fired.
        """
        if getattr(self, "_acc_mode", "passive") != "active":
            return 0
        acc = getattr(self, "_acc", None)
        if acc is None:
            return 0
        try:
            lessons = acc.at_round_n()
        except Exception:
            # Defensive: a buggy detector should never crash the turn.
            # Layer 1 still drains at end-of-turn so we lose nothing.
            return 0
        if not lessons:
            return 0
        for lesson in lessons:
            tool_results.append({
                "type": "text",
                "text": (
                    f"[Anton self-check — {lesson.detector}] {lesson.rule} "
                    "(This is an automatic mid-turn observation from your own "
                    "monitoring layer, not a user message.)"
                ),
            })
        return len(lessons)

    def _schedule_acc_flush(self) -> None:
        """Drain the ACC's turn buffer into Engrams and clear it.

        Parallel to `_schedule_cerebellum_flush()`: same fire-and-
        forget contract, same end-of-turn slot. The ACC's detectors
        are pure functions (no LLM call), so running them is cheap;
        the only async work is `cortex.encode()`, which writes the
        lessons to disk. We still wrap it in `asyncio.create_task`
        so the user-facing reply isn't blocked on file I/O.

        Best-effort: if there's no event loop (sync test, edge case),
        we drop the buffer rather than raise.
        """
        acc = getattr(self, "_acc", None)
        if acc is None:
            return
        cortex = getattr(self, "_cortex", None)
        if cortex is None or getattr(cortex, "mode", "") == "off":
            acc.clear()
            return

        lessons = acc.at_end_of_turn()
        if not lessons:
            acc.clear()
            return

        engrams = [
            Engram(
                text=l.rule,
                kind=l.kind,         # always / never / when from the detector
                scope="global",      # ACC lessons are cross-project
                confidence="high",   # detectors only fire on confirmed patterns
                source="consolidation",
            )
            for l in lessons
        ]

        # Check for a running event loop first so we don't construct a
        # coroutine object only to drop it (which triggers an unawaited-
        # coroutine warning). ACC learning is best-effort, same as
        # cerebellum learning — if there's no loop we drop the buffer.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            acc.clear()
            return

        async def _drain() -> None:
            try:
                await cortex.encode(engrams)
            finally:
                acc.clear()

        asyncio.create_task(_drain())

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
        self._append_history({"role": "user", "content": user_input})

        user_msg_str = (
            user_input
            if isinstance(user_input, str)
            else next((b["text"] for b in user_input if b.get("type") == "text"), "")
        )
        tools = self._build_tools()
        system = await self._build_system_prompt(user_msg_str)
        self._compacted_this_turn = False

        response = await self.plan_with_recovery(system=system, tools=tools)

        # Proactive compaction — gated so we never double-summarize within
        # a single turn (the recovery helper may already have compacted).
        if (
            not self._compacted_this_turn
            and response.usage.context_pressure > self._context_pressure_threshold
        ):
            await self._summarize_history()
            self._compact_scratchpads()
            self._compacted_this_turn = True

        # Handle tool calls
        tool_round = 0
        error_streak: dict[str, int] = {}
        resilience_nudged: set[str] = set()

        while response.tool_calls:
            tool_round += 1
            if tool_round > self._max_tool_rounds:
                self._append_history(
                    {"role": "assistant", "content": response.content or ""}
                )
                self._append_history(
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
                response = await self.plan_with_recovery(system=system)
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
            self._append_history({"role": "assistant", "content": assistant_content})

            # Process each tool call via registry
            tool_results: list[dict] = []
            for tc in response.tool_calls:
                try:
                    result = await self.tool_registry.dispatch_tool(
                        self, tc.name, tc.input
                    )
                except Exception as exc:
                    result = f"Tool '{tc.name}' failed: {exc}"

                if isinstance(result, list):
                    # Multimodal tool result — scrub credentials from text
                    # blocks; image-block payloads are raw bytes and have
                    # nothing to scrub. A list result signals success, so
                    # mirror the success branch of `_apply_error_tracking`
                    # and reset the streak instead of running the full
                    # string-only nudge logic.
                    content: "str | list[dict]" = [
                        {**b, "text": scrub_credentials(b.get("text", ""))}
                        if b.get("type") == "text"
                        else b
                        for b in result
                    ]
                    error_streak[tc.name] = 0
                    resilience_nudged.discard(tc.name)
                else:
                    result = scrub_credentials(result)
                    result = self._apply_error_tracking(
                        result,
                        tc.name,
                        error_streak,
                        resilience_nudged,
                    )
                    content = result

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": content,
                    }
                )

            self._append_history({"role": "user", "content": tool_results})

            # Get follow-up from LLM
            response = await self.plan_with_recovery(system=system, tools=tools)

            # Proactive compaction during tool loop — gated to at most
            # once per turn.
            if (
                not self._compacted_this_turn
                and response.usage.context_pressure > self._context_pressure_threshold
            ):
                await self._summarize_history()
                self._compact_scratchpads()
                self._compacted_this_turn = True

        # Text-only response
        reply = response.content or ""
        self._append_history({"role": "assistant", "content": reply})

        # Periodic memory vacuum (Systems Consolidation)
        if self._cortex is not None and self._cortex.mode != "off":
            self._cortex.maybe_vacuum()

        # Cerebellar consolidation — fire-and-forget so the user gets
        # their reply immediately while supervised error learning runs
        # in the background. Brain analogue: cerebellar plasticity
        # operates in parallel with continued action, not blocking it.
        self._schedule_cerebellum_flush()
        self._schedule_acc_flush()

        return reply

    async def turn_stream(
        self,
        user_input: str | list[dict],
        *,
        turn_id: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming version of turn(). Yields events as they arrive.

        `turn_id` lets the host (cowork, CLI, …) tag the turn with its
        own identifier so downstream telemetry can correlate the LLM
        calls + tool spans made during this turn. Stored on
        `self._current_turn_id` so the provider layer can read it
        without threading the arg through every internal call.
        """
        self._current_turn_id = turn_id
        self._append_history({"role": "user", "content": user_input})

        # Log user input to episodic memory
        if self._episodic is not None:
            content = (
                user_input if isinstance(user_input, str) else str(user_input)[:2000]
            )
            self._episodic.log_turn(self._turn_count + 1, "user", content)

        user_msg_str = (
            user_input
            if isinstance(user_input, str)
            else next((b["text"] for b in user_input if b.get("type") == "text"), "")
        )
        assistant_text_parts: list[str] = []
        _max_auto_retries = 2
        _retry_count = 0
        self._active_explainability = ExplainabilityCollector(
            self._explainability_store,
            turn=self._turn_count + 1,
            user_message=user_msg_str,
        )

        # Per-turn trace identity. The OpenAI provider reads this when
        # talking to MindsHub and attaches langfuse-style headers so the
        # router can attribute every LLM call (and any spans nested
        # inside this turn via tools / scratchpad) to the right session.
        # ContextVar propagation also covers `asyncio.create_task` spawns
        # — the cerebellum flush + identity extraction tasks scheduled
        # below inherit a copy of this context.
        _trace_token = set_trace_context(
            TraceContext(
                session_id=self._session_id,
                turn_id=turn_id if turn_id is not None else self._turn_count + 1,
                harness=self._harness,
            )
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
                    # Anthropic's API rejects any history where the
                    # message after a `tool_use` lacks matching
                    # `tool_result` blocks. If `_stream_and_handle_tools`
                    # raised AFTER the assistant's tool_use was
                    # appended but BEFORE the dispatcher could add the
                    # tool_result (e.g. an HTTP error inside the LLM
                    # call), the next history entry MUST start with
                    # tool_result blocks for those orphan ids — otherwise
                    # the auto-retry below sends a malformed history
                    # and we get the same 400 forever.
                    self._seal_dangling_tool_uses("interrupted by error")
                    if _retry_count <= _max_auto_retries:
                        # Inject the error into history and let the LLM try to recover
                        self._append_history(
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
                        self._append_history(
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
                            self._validate_history_for_provider(self._history)
                            async for event in self._llm.plan_stream(
                                system=await self._build_system_prompt(user_msg_str),
                                messages=self._history,
                            ):
                                if isinstance(event, StreamTextDelta):
                                    assistant_text_parts.append(event.text)
                                yield event
                        except Exception as e:
                            fallback = f"An unexpected error occurred: {e}. Please try again or rephrase your request."
                            assistant_text_parts.append(fallback)
                            yield StreamTextDelta(text=fallback)
                        break
        finally:
            if self._active_explainability is not None:
                self._active_explainability.finalize(
                    "".join(assistant_text_parts)[:2000]
                )
            reset_trace_context(_trace_token)

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
                if self._episodic:
                    user_messages =[
                        ep.content
                        for ep in self._episodic.get_conversation()
                        if ep.role == "user"
                    ]
                    messages_str = "\n\n".join(user_messages[-5:])
                else:
                    messages_str = user_input

                asyncio.create_task(self._cortex.maybe_update_identity(messages_str))
            # Periodic memory vacuum (Systems Consolidation)
            self._cortex.maybe_vacuum()

        # Cerebellar consolidation — same fire-and-forget contract as
        # the non-streaming turn. Lets the user-facing stream finish
        # immediately while supervised error learning runs in the background.
        self._schedule_cerebellum_flush()
        self._schedule_acc_flush()

    async def _stream_and_handle_tools(
        self, user_message: str = ""
    ) -> AsyncIterator[StreamEvent]:
        """Stream one LLM call, handle tool loops, yield all events."""
        tools = self._build_tools()
        system = await self._build_system_prompt(user_message)
        self._compacted_this_turn = False

        response: StreamComplete | None = None

        async for event in self.plan_stream_with_recovery(system=system, tools=tools):
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
            self._append_history(
                {"role": "assistant", "content": llm_response.content or ""}
            )
            self._append_history(
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
            async for event in self.plan_stream_with_recovery(system=system, tools=tools):
                yield event
                if isinstance(event, StreamComplete):
                    response = event

            if response is None:
                return
            llm_response = response.response

        # Proactive compaction — gated via _compacted_this_turn so we
        # never double-summarize within a single turn.
        if (
            not self._compacted_this_turn
            and llm_response.usage.context_pressure > self._context_pressure_threshold
        ):
            await self._summarize_history()
            self._compact_scratchpads()
            self._compacted_this_turn = True
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
                    self._acc_observe(
                        "cap_exhausted",
                        {"cap": self._max_tool_rounds},
                        severity=9,
                        round_idx=tool_round,
                    )
                    self._append_history(
                        {"role": "assistant", "content": llm_response.content or ""}
                    )
                    self._append_history(
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
                    async for event in self.plan_stream_with_recovery(system=system):
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
                self._append_history(
                    {"role": "assistant", "content": assistant_content}
                )

                # Process each tool call
                import time as _time

                tool_results: list[dict] = []
                for tc in llm_response.tool_calls:
                    # ACC: tool_call emit. Args_summary is intentionally
                    # truncated — the ACC vocabulary documents it as a
                    # summary string, not a full payload. Detectors
                    # don't read args today; this is reserved for a
                    # future `detect_orphaned_tool_call`.
                    self._acc_observe(
                        "tool_call",
                        {"name": tc.name, "args_summary": str(tc.input)[:120]},
                        severity=1,
                        round_idx=tool_round,
                    )
                    if self._episodic is not None:
                        self._episodic.log_turn(
                            self._turn_count + 1,
                            "tool_call",
                            str(tc.input),
                            tool=tc.name,
                            datasources=_extract_datasources(tc)
                        )

                    # If the streamed tool-call arguments couldn't be
                    # parsed (truncation mid-string, missing comma,
                    # etc.), short-circuit before invoking the
                    # handler. We synthesise a tool_result asking the
                    # LLM to re-emit the call with valid JSON. This
                    # keeps the recovery inside the tool_use /
                    # tool_result protocol — no session-level retry,
                    # no SYSTEM message clutter in history. The next
                    # turn the LLM sees the explanation and re-emits
                    # cleanly.
                    if tc.parse_error:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": (
                                f"Tool call arguments failed to parse: {tc.parse_error}. "
                                "The streamed JSON was malformed (most often a token-cap "
                                "truncation mid-call). Re-emit this call with a complete, "
                                "valid JSON body."
                            ),
                            "is_error": True,
                        })
                        continue

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
                                    id=tc.id,
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
                                            phase="scratchpad", message=item, id=tc.id,
                                        )
                                    elif isinstance(item, Cell):
                                        cell = item
                                _sp_elapsed = _time.monotonic() - _sp_t0
                                yield StreamTaskProgress(
                                    phase="scratchpad_done",
                                    message=description or "Done",
                                    eta_seconds=_sp_elapsed,
                                    id=tc.id,
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
                                    # Same post-execute ACC event as the CLI
                                    # path (handle_scratchpad) — this inline
                                    # streaming exec bypasses that handler, so
                                    # without this scratchpad_killed/result
                                    # would never fire here and detect_kill_loop
                                    # would be blind in the streaming product.
                                    observe_scratchpad_cell(
                                        self, tc.input.get("name", ""), cell
                                    )
                                    yield StreamToolResult(
                                        name=tc.name,
                                        action="exec",
                                        content=json.dumps(asdict(cell)),
                                        id=tc.id,
                                    )
                                if self._episodic is not None and cell is not None:
                                    self._episodic.log_turn(
                                        self._turn_count + 1,
                                        "scratchpad",
                                        (cell.stdout or ""),
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
                                yield StreamToolResult(name=tc.name, action="dump", content=result_text, id=tc.id)
                                result_text = (
                                    "The full notebook has been displayed to the user above. "
                                    "Do not repeat it. Here is the content for your reference:\n\n"
                                    + result_text
                                )
                    except Exception as exc:
                        result_text = f"Tool '{tc.name}' failed: {exc}"

                    if isinstance(result_text, list):
                        # Multimodal tool result — scrub credentials from text
                        # blocks (image payloads carry no secrets). A list
                        # result signals success, so mirror the success
                        # branch of `_apply_error_tracking` and reset the
                        # streak instead of running the full string-only
                        # nudge logic.
                        scrubbed_blocks = [
                            {**b, "text": scrub_credentials(b.get("text", ""))}
                            if b.get("type") == "text"
                            else b
                            for b in result_text
                        ]
                        error_streak[tc.name] = 0
                        resilience_nudged.discard(tc.name)
                        if self._episodic is not None:
                            self._episodic.log_turn(
                                self._turn_count + 1,
                                "tool_result",
                                f"[{tc.name} → multimodal result]",
                                tool=tc.name,
                            )
                        self._acc_observe(
                            "tool_result",
                            {"name": tc.name, "success": True, "error": ""},
                            severity=1,
                            round_idx=tool_round,
                        )
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tc.id,
                                "content": scrubbed_blocks,
                            }
                        )
                        continue

                    if self._episodic is not None:
                        self._episodic.log_turn(
                            self._turn_count + 1,
                            "tool_result",
                            result_text,
                            tool=tc.name,
                        )
                    result_text = scrub_credentials(result_text)
                    result_text = self._apply_error_tracking(
                        result_text, tc.name, error_streak, resilience_nudged
                    )
                    # ACC: tool_result emit. Heuristic success-detection
                    # from the result text — anton-core does not have a
                    # structured success/error envelope at this layer,
                    # so we look for the conventional "Tool 'X' failed"
                    # prefix that the exception branch above sets, plus
                    # any handler that prefixed its return with "Error:"
                    # or the dispatcher's own error-tracking markers.
                    _failed = (
                        f"Tool '{tc.name}' failed:" in result_text
                        or result_text.startswith("Error:")
                        or "ERROR:" in result_text[:200].upper()
                    )
                    self._acc_observe(
                        "tool_result",
                        {
                            "name": tc.name,
                            "success": not _failed,
                            "error": result_text[:300] if _failed else "",
                        },
                        severity=5 if _failed else 1,
                        round_idx=tool_round,
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": result_text,
                        }
                    )

                # ACC Layer 2 — mid-turn nudge. No-op when mode != "active"
                # or when no new patterns fired this round. When it does
                # fire, the lesson text appears inline alongside tool_results
                # so the LLM sees the alarm before its next decision.
                self._acc_maybe_nudge(tool_results)

                self._append_history({"role": "user", "content": tool_results})

                # Signal that tools are done and LLM is now reasoning
                _reasoning_t0 = _time.monotonic()
                yield StreamTaskProgress(
                    phase="reasoning_start", message="Thinking..."
                )

                # Stream follow-up
                response = None
                async for event in self.plan_stream_with_recovery(
                    system=system, tools=tools
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

                if response is None:
                    return
                llm_response = response.response

                # Detect max_tokens truncation inside tool loop
                if (
                    llm_response.stop_reason in ("max_tokens", "length")
                    and not llm_response.tool_calls
                ):
                    self._append_history(
                        {"role": "assistant", "content": llm_response.content or ""}
                    )
                    self._append_history(
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
                    async for event in self.plan_stream_with_recovery(
                        system=system, tools=tools
                    ):
                        yield event
                        if isinstance(event, StreamComplete):
                            response = event

                    if response is None:
                        return
                    llm_response = response.response

                # Proactive compaction during tool loop — gated to at
                # most once per turn.
                if (
                    not self._compacted_this_turn
                    and llm_response.usage.context_pressure
                    > self._context_pressure_threshold
                ):
                    await self._summarize_history()
                    self._compact_scratchpads()
                    self._compacted_this_turn = True
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
            self._append_history({"role": "assistant", "content": reply})

            if continuation >= self._max_continuations:
                # Budget exhausted — ask LLM to diagnose and present to user
                self._append_history(
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
                async for event in self.plan_stream_with_recovery(system=system):
                    yield event
                # Consolidation still runs after diagnosis
                break

            # Ask the LLM to self-assess completion.
            # Use a copy of history with a trailing user message so models
            # that don't support assistant-prefill won't reject the request.
            # Factory is re-invoked on each recovery attempt so the verifier
            # sees the latest post-compaction history.
            def build_verify_messages() -> list[dict]:
                return list(self._history) + [
                    {
                        "role": "user",
                        "content": (
                            "SYSTEM: Evaluate whether the task the user originally requested "
                            "has been fully completed based on the conversation above."
                        ),
                    }
                ]
            verifier_system = (
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
            )
            verification = await self.plan_with_recovery(
                system=verifier_system,
                max_tokens=256,
                messages_factory=build_verify_messages,
            )

            status_text = (verification.content or "").strip().upper()
            if "STATUS: COMPLETE" in status_text:
                break
            if "STATUS: STUCK" in status_text:
                # Stuck — inject diagnosis request and let the LLM explain
                reason = (verification.content or "").strip()
                self._append_history(
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
                async for event in self.plan_stream_with_recovery(system=system):
                    yield event
                break

            # INCOMPLETE — continue working
            continuation += 1
            reason = (verification.content or "").strip()
            self._append_history(
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
            async for event in self.plan_stream_with_recovery(
                system=system, tools=tools
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
            self._append_history({"role": "assistant", "content": reply})

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
