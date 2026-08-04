from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import json
import logging
import re
from typing import TYPE_CHECKING, List, Literal
import os

from pydantic import BaseModel, Field

from anton.core.backends.base import Cell, ScratchpadRuntimeFactory
from anton.core.backends.local import local_scratchpad_runtime_factory
from anton.core.datasources.data_vault import DataVault
from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext
from anton.core.memory.acc import AnteriorCingulate
from anton.core.memory.base import Engram
from anton.core.memory.cerebellum import Cerebellum
from anton.core.memory.skills import SkillStore
from anton.core.tools.recall_skill import RECALL_SKILL_TOOL
from anton.memory.history_store import is_user_turn
from anton.core.llm.prompts import (
    RESILIENCE_NUDGE,
    SCRATCHPAD_SIZE_NUDGE,
    SCRATCHPAD_TIMEOUT_NUDGE,
)
from anton.core.llm.provider import (
    ContextOverflowError,
    LLMResponse,
    ModelUnavailableError,
    ProviderOverloadedError,
    StreamComplete,
    StreamContextCompacted,
    StreamEvent,
    StreamReasoningDelta,
    StreamTaskProgress,
    StreamTextDelta,
    StreamToolResult,
    StructuredOutputError,
    TokenLimitExceeded,
    ToolCall,
    TransientProviderError,
)
from anton.core.llm.structured import looks_truncated
from anton.core.llm.thalamus import (
    ACTION_RESPOND,
    ThalamicDecision,
    gate_turn,
)
from anton.core.llm.tracing import (
    TraceContext,
    reset_trace_context,
    set_trace_context,
)
from anton.core.backends.manager import ScratchpadManager
from anton.core.tools.progress import ToolProgress
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
    SELECT_PATH_TOOL,
    UPDATE_ARTIFACT_METADATA_TOOL,
    ToolDef,
)
from anton.core.interaction.selection import SelectionElicitor
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


# Sentinel prefixing a compacted-history summary so later compactions can
# recognize and update it in place rather than summarize a summary.
_COMPACTED_MARKER = "[COMPACTED CONTEXT — REFERENCE ONLY]"

# Truncation-recovery nudges (ENG-1042). Two variants of the same failure —
# the response burned its whole output budget without producing a tool call:
#
# - Partial text arrived → the answer was cut mid-flight; ask the model to
#   pick up where it stopped (the pre-existing recovery message).
# - Nothing visible arrived → the whole budget went to internal reasoning
#   (reasoning models share one max_tokens between thinking and answer);
#   "continue where you left off" is meaningless when nothing was emitted,
#   so ask for the answer up front instead.
_TRUNCATED_CONTINUE_NUDGE = (
    "SYSTEM: Your response was truncated because it exceeded the output token limit. "
    "Continue exactly where you left off. If you were about to call a tool, "
    "call it now. If the code you were writing was too long, split it into smaller parts."
)
_TRUNCATED_SILENT_NUDGE = (
    "SYSTEM: Your previous response spent its entire output-token budget before "
    "producing any visible text or tool call — the user saw nothing. Respond again, "
    "and lead with the answer or the tool call immediately; keep deliberation brief. "
    "If the output you are building is large, produce it in smaller parts."
)
# Shown to the user when the retry ALSO burns its (doubled) budget with
# nothing visible. The one outcome this ticket forbids is the turn ending
# silently.
_TRUNCATION_FAILURE_NOTICE = (
    "I ran out of output-token budget twice in a row before completing a "
    "response, so I could not finish this step. Try splitting the request "
    "into smaller steps; if this keeps happening, lowering the reasoning "
    "effort in Settings also helps."
)

logger = logging.getLogger(__name__)


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


def _scrub_user_input(user_input: str | list[dict]) -> str | list[dict]:
    """Scrub credential values from an inbound user message.

    Applied at the `turn`/`turn_stream` entry, before the first
    `_append_history`, so a secret pasted into chat never reaches model
    context, episodic memory, or the trace sinks downstream of the LLM
    gateway (Langfuse). Only text blocks are scrubbed; image and file
    blocks carry no scrubbable text.
    """
    if isinstance(user_input, str):
        return scrub_credentials(user_input)
    return [
        {**b, "text": scrub_credentials(b.get("text", ""))}
        if b.get("type") == "text"
        else b
        for b in user_input
    ]


class _VerifierVerdict(BaseModel):
    """Structured verdict from the completion verifier (runs on the cheap
    coding model). The field descriptions below double as the verifier's
    instructions — see LLMClient.generate_object_code (ENG-716)."""

    status: Literal["COMPLETE", "WAITING", "INCOMPLETE", "STUCK"] = Field(
        description=(
            "Classify the assistant's latest message against the user's request. "
            "Judge the FINAL delivered outcome, not every intermediate step taken "
            "to reach it:\n"
            "- COMPLETE: the requested task is done and the assistant delivered a "
            "usable answer. A tool that errored or returned nothing but that the "
            "assistant RECOVERED from — it got the needed data another way, or the "
            "failed step wasn't essential to the answer — is still COMPLETE; do NOT "
            "mark a turn incomplete just because an earlier tool call failed. A "
            "finished task followed by an optional 'want me to…?' offer is still "
            "COMPLETE.\n"
            "- WAITING: the assistant's latest message asks the user a question it "
            "genuinely needs answered to proceed with the requested task, or is a "
            "reasoned refusal. This is a valid stopping point — do NOT treat it as "
            "unfinished; the correct action is to wait for the user's reply.\n"
            "- INCOMPLETE: the assistant stopped partway through the requested task "
            "WITHOUT asking the user anything, and could keep going on its own — "
            "including when it implied success but the data its answer actually "
            "depends on errored or came back empty and was never recovered.\n"
            "- STUCK: a hard blocker prevents completion (missing credentials, an "
            "unavailable service, or a permission the assistant does not have)."
        )
    )
    reason: str = Field(description="One brief sentence explaining the verdict.")


# Output budgets for the verdict call: first attempt, then the retry used when
# it comes back truncated (ENG-1081).
#
# Models that narrate before acting (`mindshub_air`/`kimi`, `deepseek`, `qwen`)
# spend the budget on prose and never reach the forced tool call. The original
# 256 truncated them on essentially every call — 98.6% of `mindshub_air` verdicts
# in prod returned no tool call, which the fail-safe below turned into a silent
# "task complete".
#
# 2048 is sized from a measured distribution, not one sample: 16 identical calls
# spanned 245–1654 output tokens (median ~290). That 6.7x per-call spread is also
# why there is no "this model can't do it" latch — one truncation is a tail
# sample, not proof about the next turn — and why the 4096 retry exists rather
# than a single bigger budget. 1024 was measurably too small. Nothing pays for
# headroom it doesn't use; first-party models answer in 43–115 tokens either way.
_VERIFIER_TOKEN_BUDGETS = (2048, 4096)

# Appended to the verifier system prompt. Shortens the preamble on narrating
# models but is not sufficient alone (0/3 at 256 with it), so it pairs with the
# budgets above. Tool name comes off the schema class so it can't go stale.
_VERIFIER_NO_PREAMBLE = (
    f"Call the {_VerifierVerdict.__name__} tool immediately as your first "
    "action. Do not think out loud, restate the conversation, or explain your "
    "reasoning before calling it — put your one-sentence justification in the "
    "tool's `reason` field."
)

# Judgment rubric appended to the per-turn verify message. Keys the "errored
# tool → not COMPLETE" logic off the FINAL answer's dependency on the failed
# data, not the mere presence of an errored tool result in the transcript —
# without this, a turn where the model tried a tool, it failed, and the model
# recovered another way and answered correctly was judged INCOMPLETE and forced
# into a redundant continuation that re-streamed the whole answer (ENG-1134).
# The hallucinated-success safeguard is preserved: implying success while the
# data the answer relies on errored/came back empty is still INCOMPLETE.
_VERIFIER_JUDGMENT_RUBRIC = (
    "Which status applies? Judge the FINAL outcome, not every intermediate step. "
    "An errored or empty tool result only makes the task INCOMPLETE or STUCK when "
    "the assistant's final answer depends on data that never arrived and was not "
    "recovered another way. A tool that failed but the assistant worked around — "
    "getting what it needed elsewhere, or the failed step being inessential — and "
    "then answered from is COMPLETE. Conversely, an assistant that implied success "
    "while the data its answer relies on errored or came back empty is INCOMPLETE, "
    "not COMPLETE."
)

def _safe_error_detail(exc: BaseException) -> str:
    """Describe an exception for logs without copying model or user content.

    Neither ``str(exc)`` nor ``repr(exc)`` is safe here: a Pydantic
    ``ValidationError`` embeds the rejected ``input_value`` — for the verifier
    that's model-generated text derived from the user's conversation — and
    provider exceptions can carry response bodies. Emits the exception type,
    plus for validation errors the field locations and error codes only, which
    is what actually identifies the failure.
    """
    # This runs *inside* an `except` handler, so it must not raise: an exception
    # escaping here would turn a gracefully-handled verifier failure into a dead
    # turn. Everything below is therefore wrapped, including the attribute reads
    # (a custom exception can expose `status_code`/`errors` as a property that
    # raises).
    try:
        name = type(exc).__name__
    except Exception:  # pragma: no cover — defensive
        return "unavailable"
    try:
        status = getattr(exc, "status_code", None)
        if status is not None:
            return f"{name}(status={status})"
        errors = getattr(exc, "errors", None)
        if callable(errors):
            try:
                details = errors(
                    include_input=False, include_url=False, include_context=False
                )
            except TypeError:  # pydantic v1 has no include_* kwargs
                details = errors()
            # Only `loc` and `type` are ever read — `msg`/`input`/`ctx` can
            # quote the rejected value.
            fields = ",".join(
                f"{'.'.join(str(p) for p in e.get('loc') or ())}:{e.get('type', '?')}"
                for e in details
            )
            if fields:
                return f"{name}({fields})"
    except Exception:
        # Anything odd about this exception object — a property that raises, an
        # `errors()` that misbehaves — degrades to the type name, never a crash
        # and never the message.
        pass
    return name


# Shared closing instruction for every path that hands control back to the
# user (STUCK, budget-exhausted, verifier-call failure): a plain self-
# assessment of solvability, not just a status dump. Without this, a
# diagnosis can read as "here's what happened, good luck" instead of an
# actual recommendation the user can act on.
_SOLVABILITY_CLAUSE = (
    "State plainly whether you believe this is still solvable on your own "
    "(and how you'd approach it differently if so) or whether it genuinely "
    "needs the user's input, a decision, or credentials you don't have."
)


def _render_tool_result_content(content, cap: int) -> str:
    """Render a tool_result's content as bounded plain text.

    Never serializes raw payloads: a multimodal result (e.g. read_image) can
    carry megabytes of base64, so we keep only text blocks and mark images with
    a placeholder rather than ``json.dumps``-ing the whole thing (ENG-716).
    """
    if isinstance(content, str):
        return content[:cap] or "(empty result)"
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append((block.get("text") or "").strip())
            elif block.get("type") in ("image", "image_url"):
                parts.append("[image]")
        return (" ".join(p for p in parts if p)[:cap]) or "[non-text result]"
    return str(content)[:cap]


def _render_verify_transcript(
    history: list[dict],
    *,
    max_convo: int = 10,
    max_tool: int = 12,
    tool_cap: int = 400,
    text_cap: int = 2000,
) -> str:
    """Render a compact, text-only view of the recent conversation for the
    completion verifier.

    Budgets the conversational thread and the tool activity *separately* so a
    voluminous tool loop can't crowd out the user/assistant turns the verifier
    needs to resolve referential requests ("do the same for the other file"):
    the most recent ``max_convo`` user/assistant text turns plus the most recent
    ``max_tool`` tool events, merged back into chronological order. Speaker is
    taken from the message role (list-based *user* content — images/files — is
    labelled USER, not ASSISTANT); multimodal blocks are rendered block-by-block
    (text kept, images as a placeholder, never raw base64); internal ``SYSTEM:``
    injections are dropped before budgeting so they don't consume slots. Keeps
    the call cheap and free of tool_use/tool_result pairing constraints
    (ENG-716).
    """
    convo: list[tuple[int, str]] = []   # (orig_index, line) — user/assistant text
    tools: list[tuple[int, str]] = []   # (orig_index, line) — tool activity

    for i, msg in enumerate(history):
        role = msg.get("role")
        content = msg.get("content")
        speaker = "USER" if role == "user" else "ASSISTANT"
        if isinstance(content, str):
            text = content.strip()
            if not text or (role == "user" and text.startswith("SYSTEM:")):
                continue
            convo.append((i, f"{speaker}: {text[:text_cap]}"))
        elif isinstance(content, list):
            # Assistant text emitted alongside a tool call is preamble/narration
            # ("Processing step 4"), not a conversational turn — route it to the
            # tool budget so a long tool loop can't evict real requests/replies
            # from the conversation budget.
            preamble = role == "assistant" and any(
                isinstance(b, dict) and b.get("type") == "tool_use" for b in content
            )
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text = (block.get("text") or "").strip()
                    if not text:
                        continue
                    bucket = tools if preamble else convo
                    bucket.append((i, f"{speaker}: {text[:text_cap]}"))
                elif btype in ("image", "image_url"):
                    convo.append((i, f"{speaker}: [image]"))
                elif btype == "tool_use":
                    tools.append((i, f"ASSISTANT called tool: {block.get('name')}"))
                elif btype == "tool_result":
                    rendered = _render_tool_result_content(block.get("content"), tool_cap)
                    tools.append((i, f"TOOL RESULT: {rendered}"))

    kept = convo[-max_convo:] + tools[-max_tool:]
    kept.sort(key=lambda entry: entry[0])
    return "\n".join(line for _, line in kept) or "(no conversation)"


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
    # When True (default), Anton acts on reasonable defaults and surfaces its
    # assumptions inline instead of stopping to ask ("do first, ask later").
    # When False, it falls back to the cautious ask-first discipline.
    act_first: bool = True
    tools: list[ToolDef] = field(default_factory=list)
    output_dir: str = ".anton/output"
    # Web tools — on by default. Each is independently resolved at session
    # construction into either a native provider capability (passed to the LLM
    # via ``native_web_tools``) or a handler-dispatched fallback ToolDef
    # (registered on the tool registry). See ChatSession.__init__.
    web_search_enabled: bool = True
    web_fetch_enabled: bool = True
    # When the task (conversation) was created. Rendered as a fixed
    # "Conversation started: …" line in the cache-stable prompt prefix — it
    # never changes across turns, so it doesn't bust the prefix cache. The
    # LIVE current time goes in the volatile tail instead (see _build_system_prompt),
    # so resuming a conversation days later still reports the real "now".
    # None → fall back to today.
    started_at: datetime | None = None
    selection_elicitor: SelectionElicitor | None = None
    # Cheap front-model routing (ENG-648). None (default) defers to the
    # settings' `router_enabled` (ANTON_ROUTER_ENABLED); hosts pass an
    # explicit bool to override per session.
    router_enabled: bool | None = None


class ChatSession:
    """Manages a multi-turn conversation with tool-call delegation."""

    # ENG-673: wall-clock budget for backing off + retrying transient provider
    # failures within a single turn. Class attribute so it's tunable (a future
    # per-surface / config knob) and injectable in tests.
    _transient_budget_s: float = 30.0

    def __init__(self, config: ChatSessionConfig) -> None:
        s = config.settings or CoreSettings()
        # Stash the full settings object (may be AntonSettings, CoreSettings,
        # or None). Tool handlers read host-only fields like
        # ``external_search_provider`` / ``exa_api_key`` via getattr so the
        # session stays decoupled from the host's settings shape.
        self._settings = config.settings
        self._max_tool_rounds = s.max_tool_rounds
        self._max_continuations = s.max_continuations
        self._verify_min_tool_rounds = s.verify_min_tool_rounds
        self._context_pressure_threshold = s.context_pressure_threshold
        self._max_consecutive_errors = s.max_consecutive_errors
        self._resilience_nudge_at = s.resilience_nudge_at
        self._token_status_cache_ttl = s.token_status_cache_ttl
        self._llm = config.llm_client
        # Router (ENG-648): explicit host override wins; otherwise the
        # settings flag (ANTON_ROUTER_ENABLED). getattr-guarded because
        # tests pass bare CoreSettings-shaped objects.
        self._router_enabled = (
            config.router_enabled
            if config.router_enabled is not None
            else bool(getattr(s, "router_enabled", False))
        )
        self._router_max_tokens = int(getattr(s, "router_max_tokens", 1024))
        # Monotonic counter for thalamus-preloaded tool_use ids. Deliberately
        # separate from `_turn_count`, which only increments in turn_stream()
        # — using it here would emit the same id on every call made through
        # the non-streaming turn() API.
        self._thalamus_recall_counter = 0
        self._self_awareness = config.self_awareness
        self._cortex = config.cortex
        self._episodic = config.episodic
        self._system_prompt_context = config.system_prompt_context
        self._output_dir = config.output_dir
        self._proactive_dashboards = config.proactive_dashboards
        self._act_first = config.act_first
        self._started_at = config.started_at
        self._extra_tools = config.tools
        self._workspace = config.workspace
        self._data_vault = config.data_vault
        self._console = config.console
        self._history: list[dict] = (
            list(config.initial_history) if config.initial_history else []
        )
        self._pending_memory_confirmations: list = []
        self._turn_count = (
            sum(1 for m in self._history if is_user_turn(m))
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
        # Strategy for mid-turn file/folder disambiguation (the `select_path`
        # tool). Hosts inject a concrete elicitor — a streaming GUI picker in
        # cowork-server, a terminal picker on the CLI. None falls back to the
        # console picker (CLI) or a graceful no-op (headless).
        self.selection_elicitor: SelectionElicitor | None = config.selection_elicitor

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
            session_id=config.session_id,
        )

        self.tool_registry = ToolRegistry()
        # Procedural memory: brain-inspired skills (Stage 1 = declarative).
        # Lives at ~/.anton/skills/<label>/. The recall_skill tool retrieves
        # entries on demand and increments per-stage usage counters.
        self._skill_store = SkillStore(root=getattr(s, "skills_root", None))
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
        #               next turn's system prompt picks them up. No
        #               surface-area on the turn loop.
        #   "active"  — Layer 2 (DEFAULT): ALSO inject lessons inline as
        #               text blocks in tool_results so the LLM sees them on
        #               the very next round and can self-correct mid-task.
        #               Stronger signal; the nudge is clearly labelled as an
        #               automatic self-check (not a user instruction). Set
        #               ANTON_ACC_MODE=passive to revert to learn-next-turn,
        #               or =off to disable, if it ever causes trouble.
        _mode_raw = os.environ.get("ANTON_ACC_MODE", "active").strip().lower()
        self._acc_mode = _mode_raw if _mode_raw in ("off", "passive", "active") else "active"
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
        the error text; a generic scratchpad error (e.g. a SyntaxError) and
        every non-scratchpad tool keep the generic nudge.
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
        # Other scratchpad failures (syntax/runtime errors): the generic
        # "you've failed twice, change approach" nudge still applies — only
        # the size/timeout cases get specialised advice.
        return RESILIENCE_NUDGE

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

        # Two stamps, deliberately split for cache-stability AND correctness:
        #  • conversation_started — the task's creation time (self._started_at),
        #    a FIXED fact rendered in the cache-stable prefix; identical every
        #    turn so it never busts the prefix cache.
        #  • current_datetime — the real wall clock, rendered in the VOLATILE
        #    tail (after the cached prefix) so it's always accurate even when a
        #    conversation is resumed days/weeks later, without touching the cache.
        _started = self._started_at or _dt.datetime.now()
        _conversation_started = _started.strftime("%A, %B %d, %Y")
        _current_datetime = _dt.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

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
            conversation_started=_conversation_started,
            current_datetime=_current_datetime,
            system_prompt_context=self._system_prompt_context,
            proactive_dashboards=self._proactive_dashboards,
            act_first=self._act_first,
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
        # Copy — SCRATCHPAD_TOOL is a module-level singleton; mutating its
        # .description in place would leak across every session/user sharing
        # this process instead of resetting per session.
        scratchpad_tool = replace(SCRATCHPAD_TOOL)
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
        # Interactive file/folder disambiguation — always available; degrades
        # to a plain-text prompt when no elicitor/console is present.
        self.tool_registry.register_tool(SELECT_PATH_TOOL)

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

        # Minimal stub tool exercising a multi-step, own-context structure
        # (see anton/core/tools/test_tool.py). Delete that file plus this
        # block to remove it.
        from anton.core.tools.test_tool import TEST_TOOL
        self.tool_registry.register_tool(TEST_TOOL)

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
        """Compress old conversation turns into a summary.

        Splits history into old (first 60%) and recent (last 40%), keeping at
        least 4 recent turns. The old portion is summarized by the routing/
        summarization model (the router role, which falls back to the coding
        model when no distinct one is configured) and replaced with a single
        user message.
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

        # Serialize old turns. Pull out any prior compacted summary so we
        # UPDATE it in place rather than summarize a summary (which compounds
        # loss every compaction).
        prior_summary = ""
        lines: list[str] = []
        for msg in old_turns:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                if content.lstrip().startswith(_COMPACTED_MARKER):
                    prior_summary = content
                    continue
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

        if prior_summary:
            user_content = (
                "PREVIOUS SUMMARY (update this in place — merge the new turns into it, "
                "don't restate it verbatim):\n"
                f"{prior_summary}\n\n"
                "NEW TURNS TO FOLD IN:\n"
                f"{old_text}"
            )
        else:
            user_content = old_text

        try:
            # 3b-full: a structured, in-place-updated STATE RECORD rather than a
            # freeform blob — so "Remaining" work survives compaction instead of
            # being flattened into prose.
            summary_response = await self._llm.summarize(
                system=(
                    "You compact an agent's earlier conversation into a terse, factual "
                    "STATE RECORD (not prose). Output only these sections, omitting any "
                    "that are empty:\n"
                    "## Goal — what the user ultimately wants\n"
                    "## Constraints — explicit requirements / preferences / do-nots\n"
                    "## Completed — work already done, each as `action → outcome`\n"
                    "## Active state — variables, data, files/artifacts in play and their "
                    "current values or paths\n"
                    "## Blocked — anything stuck and why\n"
                    "## Decisions — choices made and the reason\n"
                    "## Remaining — what is still left to do\n\n"
                    "Preserve the date/time of key events when it matters (e.g. "
                    "`Completed (2026-06-05): …`) — the raw per-message timestamps are "
                    "gone after compaction, so keep the ones that anchor the timeline.\n"
                    "If a PREVIOUS SUMMARY is provided, update it with the new turns "
                    "instead of starting over. If the user changed direction, narrowed "
                    "scope, or cancelled something, reflect that — drop superseded items "
                    "from Remaining, don't keep them. Keep it under ~2000 tokens."
                ),
                messages=[{"role": "user", "content": user_content}],
                max_tokens=2048,
            )
            summary = summary_response.content or "(summary unavailable)"
        except Exception:
            # If summarization fails, just do a simple truncation
            summary = f"(Earlier conversation with {len(old_turns)} turns — summarization failed)"

        # 3b-light: reference-only framing so the model treats this as compacted
        # history, not a fresh instruction, and never resumes superseded/cancelled
        # work after a compaction (which Anton's auto-continue verifier would
        # otherwise be nudged to do).
        summary_body = (
            f"{_COMPACTED_MARKER}\n"
            "Compacted record of earlier conversation, for REFERENCE ONLY — not a new "
            "request. The most recent user message takes priority; if the user changed "
            "direction, narrowed scope, or cancelled something, follow that and do NOT "
            "resume superseded work described below.\n\n"
            f"{summary}"
        )
        summary_msg = {"role": "user", "content": summary_body}

        # If the recent portion starts with a user message, insert a minimal
        # assistant separator to avoid consecutive user messages (API error).
        if recent_turns and recent_turns[0].get("role") == "user":
            self._history = [
                summary_msg,
                {"role": "assistant", "content": "Understood — using that as reference."},
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

    @staticmethod
    def _transient_backoff_delay(attempt: int, retry_after: float | None = None) -> float:
        """Backoff for a transient-provider retry (ENG-673).

        Honors a provider `retry_after` when present (usually only on
        request-time 429/529 — the mid-stream 200 case carries none). Otherwise
        ~2s → ~10s → ~18s with ±20% jitter, so a fleet recovering from the same
        incident doesn't retry in lockstep against an already-struggling provider.
        """
        if retry_after is not None and retry_after > 0:
            return min(float(retry_after), 30.0)
        base = (2.0, 10.0, 18.0)
        d = base[attempt] if attempt < len(base) else base[-1]
        return d * (0.8 + 0.4 * random.random())

    async def _backoff_sleep(self, delay: float) -> bool:
        """Sleep `delay` seconds, waking early if the turn is cancelled.

        Returns True if cancelled during the wait, False if the full delay
        elapsed — so a user hitting stop during backoff aborts immediately
        instead of waiting out the incident (ENG-673).
        """
        try:
            await asyncio.wait_for(self._cancel_event.wait(), timeout=delay)
            return True  # _cancel_event fired
        except asyncio.TimeoutError:
            return False  # slept the full delay

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

    async def _gate_turn(self) -> ThalamicDecision | None:
        """Run the cheap routing call for the turn just appended to history.

        Returns None — meaning "proceed to the planning model as if no
        thalamus existed" — on any thalamus failure. Routing must never be
        able to break a turn; it can only save one.
        """
        try:
            summaries = (
                self._skill_store.list_summaries()
                if self._skill_store is not None
                else []
            )
        except Exception:
            summaries = []
        try:
            return await gate_turn(
                self._llm,
                history=self._history,
                skill_summaries=summaries,
                max_tokens=self._router_max_tokens,
            )
        except Exception as exc:
            logger.warning(
                "Router call failed (%s) — falling through to the planning model.",
                exc,
            )
            return None

    def _inject_recalled_skills(self, labels: list[str]) -> None:
        """Preload thalamus-named skills as a synthetic recall_skill exchange.

        Appends an assistant `tool_use` + user `tool_result` pair to
        history, byte-identical in payload to what the planning model
        would have gotten by calling `recall_skill` itself — but without
        spending a full-context planning round on the fetch. Labels must
        match exactly (no fuzzy fallback: a wrong preload is worse than
        none), unknown labels are dropped silently, and at most 3 skills
        load per turn. Built-ins and user skills resolve through the same
        SkillStore that `recall_skill` uses.
        """
        if not labels:
            return
        store = self._skill_store
        if store is None:
            return
        from anton.core.tools.recall_skill import (
            _already_in_history,
            _format_skill_response,
        )

        self._thalamus_recall_counter += 1
        tool_uses: list[dict] = []
        results: list[dict] = []
        seen: set[str] = set()
        for label in labels:
            label = (label or "").strip()
            if not label or label in seen:
                continue
            seen.add(label)
            if len(tool_uses) >= 3:
                break
            try:
                skill = store.load(label)
            except Exception:
                skill = None
            if skill is None:
                continue
            # Skip skills whose full body is already in context — mirrors
            # handle_recall_skill's stub path so a preload can't duplicate a
            # procedure the planning model already has (wasted tokens).
            if _already_in_history(self, skill.label):
                continue
            content = _format_skill_response(skill)
            try:
                store.increment_recommended(skill.label, stage=1)
            except Exception:
                pass
            tu_id = f"thalamus_recall_{self._thalamus_recall_counter}_{len(tool_uses)}"
            tool_uses.append(
                {
                    "type": "tool_use",
                    "id": tu_id,
                    "name": "recall_skill",
                    "input": {"label": skill.label},
                }
            )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": content,
                }
            )
        if tool_uses:
            self._append_history({"role": "assistant", "content": tool_uses})
            self._append_history({"role": "user", "content": results})

    async def turn(self, user_input: str | list[dict]) -> str:
        user_input = _scrub_user_input(user_input)
        self._append_history({"role": "user", "content": user_input})

        user_msg_str = (
            user_input
            if isinstance(user_input, str)
            else next((b["text"] for b in user_input if b.get("type") == "text"), "")
        )

        # Cheap front-model routing (ENG-648). Text-only turns first hit
        # the thalamus model, which either answers trivial/from-context
        # requests directly (skipping the full prompt + tools + planning
        # model entirely) or delegates, optionally preloading skills.
        # Image turns skip the thalamus — attachments imply real work.
        if self._router_enabled and isinstance(user_input, str):
            decision = await self._gate_turn()
            if decision is not None and decision.action == ACTION_RESPOND:
                self._append_history(
                    {"role": "assistant", "content": decision.text}
                )
                if self._cortex is not None and self._cortex.mode != "off":
                    self._cortex.maybe_vacuum()
                self._schedule_cerebellum_flush()
                self._schedule_acc_flush()
                return decision.text
            if decision is not None and decision.skills:
                self._inject_recalled_skills(decision.skills)

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
        trace_tags: list[str] | None = None,
        trace_metadata: dict[str, str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming version of turn(). Yields events as they arrive.

        `turn_id` lets the host (cowork, CLI, …) tag the turn with its
        own identifier so downstream telemetry can correlate the LLM
        calls + tool spans made during this turn. Stored on
        `self._current_turn_id` so the provider layer can read it
        without threading the arg through every internal call.

        `trace_tags` / `trace_metadata` are optional, opaque annotations the
        host can attach to this turn's trace (forwarded to the MindsHub
        langfuse headers — see the provider's `_build_trace_headers`). They
        are deliberately generic: hosts can add arbitrary correlation data
        (e.g. an eval-run id) without any change to Anton.
        """
        self._current_turn_id = turn_id
        user_input = _scrub_user_input(user_input)
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
        # ENG-673: a mid-stream provider failure that had NO prior retry (an
        # overload smuggled into an HTTP-200 stream) gets budget-bounded
        # backoff-and-retry — separate from the instant, count-bounded recovery
        # used for genuine errors. The clock starts on the first such error and
        # is measured across the turn. Request-time transients (real 5xx/429,
        # connection errors) carry session_backoff=False and skip this path.
        _transient_deadline: float | None = None
        _transient_attempt = 0
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
                tags=tuple(trace_tags or ()),
                metadata=trace_metadata or None,
            )
        )

        try:
            # Cheap front-model routing (ENG-648). Text-only turns first
            # hit the thalamus model, which either answers trivial/from-
            # context requests directly (skipping the full prompt + tool
            # schemas + planning model entirely) or delegates, optionally
            # preloading skills into history so the planning model doesn't
            # spend a round on recall_skill. Image turns skip the thalamus —
            # attachments imply real work. The thalamus buffers rather than
            # streams: direct answers are short by construction
            # (router_max_tokens), and a delegate decision must never leak
            # preamble text to the user.
            routed_direct = False
            if self._router_enabled and isinstance(user_input, str):
                decision = await self._gate_turn()
                if decision is not None and decision.action == ACTION_RESPOND:
                    self._append_history(
                        {"role": "assistant", "content": decision.text}
                    )
                    assistant_text_parts.append(decision.text)
                    yield StreamTextDelta(text=decision.text)
                    yield StreamComplete(
                        response=decision.response
                        or LLMResponse(content=decision.text)
                    )
                    routed_direct = True
                elif decision is not None:
                    # Delegating: surface the gate call's usage as its own
                    # StreamComplete so token accounting counts it, exactly
                    # like a planning round (the loop below emits more). The
                    # gate hits every turn, so dropping it would under-report.
                    if decision.response is not None:
                        yield StreamComplete(response=decision.response)
                    if decision.skills:
                        self._inject_recalled_skills(decision.skills)

            while not routed_direct:
                try:
                    async for event in self._stream_and_handle_tools(user_msg_str):
                        if isinstance(event, StreamTextDelta):
                            assistant_text_parts.append(event.text)
                        yield event
                    break  # completed successfully
                except Exception as _agent_exc:
                    # Token/billing limits and model-gate 403s are
                    # deterministic — the auto-retry below would just re-send
                    # the same doomed request (and burn its budget) before
                    # failing anyway. Don't retry; let the chat loop / server
                    # map them to their cards.
                    if isinstance(_agent_exc, (TokenLimitExceeded, ModelUnavailableError)):
                        raise

                    # ENG-673: a mid-stream transient failure that had NO prior
                    # retry (overload smuggled into a 200, or a truncated stream).
                    # Back off and retry the SAME step within a per-turn budget —
                    # do NOT inject a "you errored, change approach" note (it was
                    # a provider blip, not the model's fault) and do NOT spend the
                    # count-based retry budget reserved for genuine errors.
                    # Completed tool_results already sit in history and are never
                    # re-executed on retry (idempotency); we only seal any
                    # tool_use the interrupted stream left dangling so the retried
                    # request is valid. Request-time transients (real 5xx/429,
                    # connection errors) set session_backoff=False — the SDK
                    # already retried them, so they fall through to the fast
                    # count-based path below with their honest typed message.
                    if isinstance(_agent_exc, TransientProviderError) and _agent_exc.session_backoff:
                        now = asyncio.get_running_loop().time()
                        if _transient_deadline is None:
                            _transient_deadline = now + self._transient_budget_s
                        self._seal_dangling_tool_uses("interrupted by a transient provider error")
                        remaining = _transient_deadline - now
                        if remaining > 0.5:
                            delay = min(
                                self._transient_backoff_delay(
                                    _transient_attempt, _agent_exc.retry_after
                                ),
                                remaining,
                            )
                            _transient_attempt += 1
                            if await self._backoff_sleep(delay):
                                # User cancelled during backoff — stop cleanly
                                # (like a normal stop), not with an error card.
                                break
                            continue
                        # Budget exhausted — fail with an actionable, honest error
                        # the server renders as the provider_overloaded card.
                        # Name the model that actually failed (planning OR coding),
                        # falling back to the session's planning model.
                        raise ProviderOverloadedError(
                            f"{_agent_exc.provider or 'The model provider'} is experiencing an "
                            "incident and didn't recover in time.",
                            provider=getattr(_agent_exc, "provider", "") or "",
                            model=(getattr(_agent_exc, "model", "") or "")
                            or getattr(self._llm, "planning_model", "") or "",
                        ) from _agent_exc

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
                        # Inject the error into history and let the LLM try to
                        # recover. A TransientProviderError reaching here is a
                        # request-time provider blip (5xx / rate-limit / dropped
                        # connection) — NOT the model's fault, so don't tell it to
                        # "adjust your approach" (that misattributes the failure
                        # and can degrade the next attempt during an incident);
                        # just note it was transient and continue as planned (ENG-673).
                        if isinstance(_agent_exc, TransientProviderError):
                            recovery_note = (
                                f"SYSTEM: A temporary provider error interrupted execution: {_agent_exc}\n\n"
                                "This was a transient service issue, not a problem with your approach — "
                                "continue the task as planned."
                            )
                        else:
                            recovery_note = (
                                f"SYSTEM: An error interrupted execution: {_agent_exc}\n\n"
                                "If you can diagnose and fix the issue, continue working on the task. "
                                "Adjust your approach to avoid the same error. "
                                "If this is unrecoverable, summarize what you accomplished and suggest next steps."
                            )
                        self._append_history({"role": "user", "content": recovery_note})
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
                        except (TokenLimitExceeded, ModelUnavailableError):
                            # Curated provider failures must FAIL the turn, not
                            # get wrapped into assistant prose: the server maps
                            # them to actionable error cards (token_limit /
                            # model-unavailable), which can only fire when the
                            # exception propagates. Wrapping them as text is
                            # how "Server returned 403" ended up mid-chat with
                            # "please rephrase your request" advice.
                            raise
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

    def _turn_max_tokens(self) -> int:
        """Output-token budget the turn's planning calls actually run with.

        `looks_truncated` compares output tokens against the budget the
        call was given; the main loop never overrides ``max_tokens``, so
        that is the client default. Falls back to LLMClient's own default
        when the client doesn't expose one (mocks, exotic hosts).
        """
        budget = getattr(self._llm, "max_tokens", None)
        return budget if isinstance(budget, int) and budget > 0 else 8192

    async def _recover_truncated_stream(
        self,
        llm_response: LLMResponse,
        *,
        system: str,
        tools: list[dict] | None,
    ) -> AsyncIterator[StreamEvent]:
        """Retry a response that burned its output budget without finishing.

        ``llm_response`` hit ``max_tokens`` before producing a tool call —
        detected by token count (`looks_truncated`), NOT ``stop_reason``:
        the MindsHub gateway reports a normal stop at the cap (ENG-1082),
        which is what kept this recovery dead for every hosted user
        (ENG-1042).

        The retry always CHANGES the call — an identical re-issue dies
        identically (measured: three unchanged retries 14 minutes apart,
        all silent):

        - the output budget is doubled for this one call, and
        - a corrective nudge is injected into history — "continue where
          you left off" when partial text arrived, "answer now, deliberate
          less" when the whole budget went to internal reasoning.

        If the retry also comes back truncated with nothing visible, a
        failure notice is shown to the user (and recorded in history) —
        the turn must never end silently. The retry's ``StreamComplete``
        is withheld until after that decision so the notice text lands
        inside the message, then re-yielded for the caller to capture.
        """
        budget = self._turn_max_tokens()
        silent = not (llm_response.content or "").strip()
        # Empty-content appends are no-ops inside _append_history, so the
        # silent variant only records the nudge.
        self._append_history(
            {"role": "assistant", "content": llm_response.content or ""}
        )
        self._append_history(
            {
                "role": "user",
                "content": (
                    _TRUNCATED_SILENT_NUDGE if silent else _TRUNCATED_CONTINUE_NUDGE
                ),
            }
        )
        retry_budget = budget * 2
        logger.warning(
            "Response truncated at the output budget with no tool call "
            "(output_tokens=%s, budget=%s, stop_reason=%s, silent=%s) — "
            "retrying once with max_tokens=%s",
            llm_response.usage.output_tokens,
            budget,
            llm_response.stop_reason,
            silent,
            retry_budget,
        )

        retry: StreamComplete | None = None
        async for event in self.plan_stream_with_recovery(
            system=system, tools=tools, max_tokens=retry_budget
        ):
            if isinstance(event, StreamComplete):
                retry = event
                continue  # withheld — re-yielded below, after the failure check
            yield event

        if retry is None:
            return

        retried = retry.response
        if (
            not retried.tool_calls
            and not (retried.content or "").strip()
            and looks_truncated(retried, retry_budget)
        ):
            logger.error(
                "Truncation retry also burned its whole budget with no "
                "visible output (output_tokens=%s, retry_budget=%s) — "
                "surfacing failure to the user",
                retried.usage.output_tokens,
                retry_budget,
            )
            self._append_history(
                {"role": "assistant", "content": _TRUNCATION_FAILURE_NOTICE}
            )
            yield StreamTextDelta(text=_TRUNCATION_FAILURE_NOTICE)
        yield retry

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
        # By token count, not stop_reason: the gateway reports a normal stop
        # at the cap (ENG-1082), which made a stop_reason gate dead code for
        # every MindsHub-routed user (ENG-1042).
        if not llm_response.tool_calls and looks_truncated(
            llm_response, self._turn_max_tokens()
        ):
            response = None
            async for event in self._recover_truncated_stream(
                llm_response, system=system, tools=tools
            ):
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
        import logging as _logging
        _verifier_log = _logging.getLogger(__name__)

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
                        elif (
                            tc.name == "connect_new_datasource"
                            or tc.name == "select_path"
                            or (
                                tc.name == "publish_or_preview"
                                and tc.input.get("action") == "publish"
                            )
                        ):
                            # Interactive tool — pause spinner AND escape watcher
                            yield StreamTaskProgress(
                                phase="interactive",
                                message="",
                            )
                            if self._escape_watcher:
                                self._escape_watcher.pause()
                            try:
                                result_text = await self.tool_registry.dispatch_tool(
                                    self, tc.name, tc.input
                                )
                            finally:
                                if self._escape_watcher:
                                    self._escape_watcher.resume()
                            yield StreamTaskProgress(
                                phase="analyzing",
                                message="Analyzing results...",
                            )
                        else:
                            # Non-scratchpad, non-interactive tool — track elapsed.
                            # dispatch_tool_stream() forwards ToolProgress markers
                            # from a streaming handler as they arrive; a plain
                            # (non-streaming) handler yields exactly one item (its
                            # result), so this same loop works for both kinds.
                            yield StreamTaskProgress(
                                phase="tool_start",
                                message=tc.name,
                            )
                            result_text = None
                            cancelled = False
                            _tool_error: Exception | None = None
                            try:
                                async with aclosing(
                                    self.tool_registry.dispatch_tool_stream(
                                        self, tc.name, tc.input
                                    )
                                ) as stream:
                                    async for item in stream:
                                        if isinstance(item, ToolProgress):
                                            yield StreamTaskProgress(
                                                phase="tool_progress",
                                                message=item.text,
                                                id=tc.id,
                                            )
                                        else:
                                            result_text = item
                                        # Checked AFTER handling item, not before —
                                        # otherwise an already-arrived final result
                                        # could be discarded on the same iteration
                                        # the flag flips. NOTE: on today's one real
                                        # consumer (CLI, anton/chat.py:1858-1866) this
                                        # flag is set and a KeyboardInterrupt is
                                        # raised in the SAME step, so this branch is
                                        # rarely what actually stops execution — the
                                        # aclosing() above is: it closes this
                                        # generator (and the handler's own generator
                                        # underneath it) via GeneratorExit as soon as
                                        # the outer turn_stream() is torn down. This
                                        # check stays for symmetry with the
                                        # scratchpad branch above and as a safety net
                                        # for any future consumer that sets the flag
                                        # without immediately raising.
                                        if self._cancel_event.is_set():
                                            cancelled = True
                                            break
                            except Exception as exc:
                                # Caught locally ONLY so the tool_done marker
                                # below can still be yielded from ordinary
                                # (non-unwinding) execution; re-raised right
                                # after, so the outer handler a few lines below
                                # still builds "Tool 'x' failed: ...".
                                # GeneratorExit is a BaseException, not an
                                # Exception, so cancellation via the consumer
                                # closing this generator is NOT caught here —
                                # it propagates straight through and the
                                # generator closes without a tool_done, same as
                                # any other stream abort. NOT a try/finally
                                # with a yield inside: yielding while the
                                # generator is being closed raises "async
                                # generator ignored GeneratorExit".
                                _tool_error = exc

                            _tool_elapsed = _time.monotonic() - _tool_t0
                            yield StreamTaskProgress(
                                phase="tool_done",
                                message=tc.name,
                                eta_seconds=_tool_elapsed,
                                id=tc.id,
                            )
                            if _tool_error is not None:
                                raise _tool_error
                            if cancelled and result_text is None:
                                # Deliberately NOT prefixed "Error:"/"failed:" —
                                # the _failed heuristic a few lines below this
                                # branch would flag it as a tool failure, but a
                                # user-initiated cancellation isn't one. Same
                                # precedent as the scratchpad branch's own
                                # "No result produced." (also not flagged as an
                                # error) a few dozen lines above.
                                result_text = (
                                    f"Tool '{tc.name}' was cancelled by the user "
                                    "before producing a result."
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
                    # Capture reasoning elapsed on first text, reasoning, or
                    # tool event — a StreamReasoningDelta means the model has
                    # already started reasoning, same signal as the first
                    # text token.
                    if _reasoning_t0 and isinstance(
                        event, (StreamTextDelta, StreamReasoningDelta, StreamComplete)
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

                # Detect max_tokens truncation inside tool loop — same
                # token-count evidence as the pre-loop gate (ENG-1042).
                if not llm_response.tool_calls and looks_truncated(
                    llm_response, self._turn_max_tokens()
                ):
                    response = None
                    async for event in self._recover_truncated_stream(
                        llm_response, system=system, tools=tools
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
            # Skip when too few tool rounds were used (pure Q&A always skips at
            # tool_round==0; raising verify_min_tool_rounds also skips trivial
            # single-round turns) or when we hit the max-rounds hard stop.
            if tool_round < self._verify_min_tool_rounds or _max_rounds_hit:
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
                            f"4. {_SOLVABILITY_CLAUSE}\n"
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

            # Ask the cheap coding model to self-assess completion over a compact,
            # text-rendered view of the recent conversation: enough context for
            # referential follow-ups plus truncated tool-result evidence to
            # cross-check success claims, but far smaller than the raw transcript
            # and free of tool_use/tool_result pairing constraints (ENG-716). The
            # assistant's latest reply is already in history (appended above).
            transcript = _render_verify_transcript(self._history)
            # Always state the current request explicitly: a long tool-heavy turn
            # can push the turn's opening user message out of the transcript window,
            # and the request is the anchor for the whole judgment (ENG-716).
            request = (user_message or "").strip()
            request_header = f"USER'S CURRENT REQUEST: {request}\n\n" if request else ""
            verify_messages = [
                {
                    "role": "user",
                    "content": (
                        "Assess the conversation below (tool results are truncated) and "
                        "decide the status of the USER's most recent request.\n\n"
                        f"{request_header}{transcript}\n\n"
                        + _VERIFIER_JUDGMENT_RUBRIC
                    ),
                },
            ]
            verifier_system = (
                "You are a task-completion verifier. Decide whether the user's "
                "request is complete, the assistant is waiting on the user, the work "
                "is unfinished, or the assistant is blocked. Follow the status "
                "definitions exactly.\n\n"
                # Models that narrate before acting spend the whole budget on
                # prose and never reach the tool call (ENG-1081). Asking for the
                # call first shortens the preamble; it does not eliminate it,
                # which is why the budget below is generous as well.
                + _VERIFIER_NO_PREAMBLE
            )
            verdict = None
            for attempt, budget in enumerate(_VERIFIER_TOKEN_BUDGETS):
                try:
                    verdict = await self._llm.generate_object_code(
                        _VerifierVerdict,
                        system=verifier_system,
                        messages=verify_messages,
                        max_tokens=budget,
                    )
                    break
                except StructuredOutputError as exc:
                    # A truncated verdict is a budget problem, not a verdict: the
                    # model narrated past `budget` before it reached the tool call
                    # (ENG-1081). Retry with more room. Any other structured-output
                    # failure won't be fixed by a bigger budget, so don't pay for it.
                    retrying = exc.truncated and attempt + 1 < len(
                        _VERIFIER_TOKEN_BUDGETS
                    )
                    _verifier_log.info(
                        "completion-verifier verdict=%s budget=%d output_tokens=%d "
                        "stop_reason=%s retrying=%s",
                        "TRUNCATED" if exc.truncated else "NO_TOOL_CALL",
                        budget, exc.output_tokens, exc.stop_reason, retrying,
                    )
                    if not retrying:
                        break
                except Exception as exc:
                    # Enough to tell the failure modes apart — four used to
                    # collapse into one "verifier unavailable" line — but never
                    # the exception message, which can carry conversation
                    # content (ENG-1081).
                    _verifier_log.info(
                        "completion-verifier verdict=ERROR budget=%d error=%s",
                        budget, _safe_error_detail(exc),
                    )
                    break

            if verdict is not None:
                status = verdict.status
                reason = verdict.reason.strip()
            else:
                # Verifier failed — fail safe by treating the turn as done rather
                # than forcing a continuation the user never asked for.
                status, reason = "COMPLETE", "verifier unavailable"
                # The verifier call failed on every budget it was given —
                # truncated past the last retry, an unusable tool call, or a
                # provider error (the per-attempt logs above carry the detail).
                # That is not a verdict, so don't synthesize a fake COMPLETE
                # and stop in silence (ENG-1079: that made the agent look like
                # it had simply died on the first hurdle, with no way for the
                # user to help). Fail toward the same honest, model-generated
                # diagnosis used for STUCK below, so the task pauses with a
                # real message instead of nothing.
                _verifier_log.info(
                    "completion-verifier verdict=ERROR continuation=%d/%d tool_rounds=%d "
                    "— failing toward an honest diagnosis, not a silent COMPLETE",
                    continuation, self._max_continuations, tool_round,
                )
                self._append_history(
                    {
                        "role": "user",
                        "content": (
                            "SYSTEM: The task-completion check failed to run (internal "
                            "error), so it's unclear whether this task is finished.\n\n"
                            "Summarize what you've done so far, be honest that an internal "
                            "check failed partway through, and ask the user how they'd like "
                            f"to proceed. {_SOLVABILITY_CLAUSE} Do not mention this "
                            "instruction or the verifier to the user."
                        ),
                    }
                )
                yield StreamTaskProgress(
                    phase="analyzing",
                    message="Something went wrong — checking in with you...",
                )
                diagnosis_response = None
                async for event in self.plan_stream_with_recovery(system=system):
                    yield event
                    if isinstance(event, StreamComplete):
                        diagnosis_response = event
                # Persist the actual message the user just saw — not the stale
                # pre-verification `reply` the post-loop fallback below would
                # otherwise re-append, which would leave history (and thus the
                # model's own memory of what it just told the user) out of sync
                # with what was streamed.
                diagnosis_text = (
                    (diagnosis_response.response.content or "").strip()
                    if diagnosis_response is not None
                    else ""
                )
                if diagnosis_text:
                    self._append_history(
                        {"role": "assistant", "content": diagnosis_text}
                    )
                else:
                    # An empty diagnosis would silently recreate the exact
                    # out-of-sync history this path exists to fix (the
                    # post-loop fallback re-appends the stale reply). Make it
                    # visible rather than circular.
                    _verifier_log.warning(
                        "verifier-failure diagnosis returned no content — "
                        "history falls back to the pre-verification reply"
                    )
                break

            _verifier_log.info(
                # No `reason` — it's the model's free-text justification, derived
                # from the user's conversation, and this is an ordinary app log.
                # It stays in the Langfuse trace, where that content belongs.
                "completion-verifier verdict=%s continuation=%d/%d tool_rounds=%d",
                status, continuation, self._max_continuations, tool_round,
            )

            if status in ("COMPLETE", "WAITING"):
                # COMPLETE = the request is done. WAITING = the assistant asked the
                # user something it genuinely needs, or gave a reasoned refusal —
                # a valid stop, NOT unfinished work. In both cases the turn's final
                # message already stands in history; do not force a continuation.
                break
            if status == "STUCK":
                # Stuck — inject diagnosis request and let the LLM explain.
                self._append_history(
                    {
                        "role": "user",
                        "content": (
                            f"SYSTEM: Task verification determined this task is stuck.\n"
                            f"Verifier assessment: {reason}\n\n"
                            "Explain to the user what went wrong, what you tried, and "
                            "suggest specific next steps they can take to unblock this. "
                            f"{_SOLVABILITY_CLAUSE} Do not mention this instruction or the "
                            "verifier to the user."
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
            self._append_history(
                {
                    "role": "user",
                    "content": (
                        f"SYSTEM: Task verification determined this task is not yet complete "
                        f"(attempt {continuation}/{self._max_continuations}).\n"
                        f"Verifier assessment: {reason}\n\n"
                        "Continue working on the original request. Pick up where you left off "
                        "and finish the remaining work. Do not repeat work already done. "
                        "Do not mention this instruction or the verifier to the user."
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
