"""State container and verdict schemas for the artifact-generation FSM.

The orchestrator (`orchestrator.py`) walks graph nodes over one `GenState`.
Diamond nodes are resolved by `session._llm.generate_object(...)` calls that
return the Pydantic verdict models below. Verifiers return `VerifyResult`.
"""
from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from anton.core.llm.client import ModelRoute

from .debug_trace import NullTrace, GenTrace  # noqa: F401  (GenTrace re-exported for typing)
from .progress import PEEK_GROUPS, LivePeek, StepCounter, label_for, plan_steps
from .attachments import Attachment

if TYPE_CHECKING:
    from anton.chat_session import ChatSession
    from anton.core.artifacts.live_app_slots import LiveAppLaunchError, LiveAppLaunchResult
    from anton.core.artifacts.render_check import CheckedView, Comparison, RenderScreenshot, RenderStatus, ViewCoverage
    from anton.core.tools.tool_handlers import PageKey
    from anton.utils.datasources import DatasourceCatalog

# ── Budgets (see design spec) ────────────────────────────────────────────────
DATA_LOOP_MAX: int = 3
# Generation-loop failures (round budget, no tool calls) and verification
# failures get SEPARATE retry budgets. With a single shared counter a loop
# failure consumed the retry reserved for fixing verifier findings, and a
# trivially fixable verification error became terminal.
GEN_LOOP_MAX_RETRIES: int = 1
GEN_VERIFY_MAX_RETRIES: int = 1
RUNAPP_MAX_RETRIES: int = 1

# Every "not enough data" error starts with this; the tool handler keys the
# text it gives the agent on it.
NOT_ENOUGH_DATA_PREFIX = "not enough data:"

# Per-line detail cap for GenState.journal() — keeps the journal compact when
# a step's detail is long (full text still lives in data_notes / trace_log).
JOURNAL_DETAIL_MAX: int = 300

# Output budgets for the two whole-document calls (`make_tech_spec`,
# `make_api_spec`), overriding the client default of 8192 — a specification is
# among the longest single answers anton asks for, and reasoning models spend
# their internal thinking from the same budget (a 25 842-character spec was
# measured dying at 8192 output tokens).
#
# The gateway used to answer HTTP 500 above 20480 output tokens for a
# non-streamed call; it now assembles such calls from an internal stream, so
# that limit is gone. These calls are streamed anyway. Re-measure before
# raising them. A top effort's larger streamed budget replaces both
# (engine._budget_for_effort).
SPEC_MAX_TOKENS: int = 16384
SPEC_MAX_TOKENS_RETRY: int = 20480

# Output budget for the WRITE rounds of `_run_loop` (rounds > 0, coding model),
# overriding the client default of 8192. Measured 2026-08-28 against
# `api.mindshub.ai/v1`: 20480 is accepted on both aliases (the earlier note
# above claiming 16384 for `haiku` was wrong — its ceiling is the same 20480),
# and one `write_file` call at that budget delivered 50 402 characters of
# Russian HTML in 15 754 tokens. The live artifact was 41 570 characters /
# 16 063 tokens, i.e. the default 8192 was a quarter of what the gateway holds
# and forced the file into ~8 chunks.
#
# Applied to EVERY round, including round 0. It was withheld there while the
# file body rode in a tool argument: round 0 runs on the planning model (~2.3x
# slower per token, 75 vs 170 measured), a tool argument is not streamed, and
# at this budget such a call stayed silent long enough to be dropped — 4
# failures out of 4 at 131-143s, measured 2026-08-28. The client default of
# 8192 turned that into a plain truncation, which the loop survives.
#
# The body is streamed text since the text-then-tool protocol, so there is no
# silence left to survive and the low cap only cost rounds: measured
# 2026-09-15, round 0 hit exactly 8192 output tokens in the middle of a body.
# The planning model is also less token-efficient on this content (1.57
# characters per token against the coding model's 2.62), so 8192 bought it
# ~12 800 characters — under the size of an average artifact.
#
# A top effort's larger streamed budget replaces it (engine._budget_for_effort).
GEN_WRITE_MAX_TOKENS: int = 20480

# The same budget restated in CHARACTERS, because that is the unit the model
# can actually compare against what it is about to write. It has no view of its
# remaining tokens, so "write the whole file when it fits in your output budget"
# is a condition it cannot evaluate — and measured 2026-09-15, it answered that
# condition by following the concrete splitting recipe underneath it instead,
# cutting a 33 687-character file that would have fit.
#
# Derived from the ratios measured on real bodies: the planning model, which
# writes round 0, ran 1.57 / 1.78 / 1.99 characters per token across three runs;
# the coding model 2.62 and 2.91. Taken at the planning model's worst,
# 20480 * 1.57 is about 32 100, rounded down for the preamble and the tool call.
# Deliberately conservative: overshooting costs the whole part (a body with no
# end marker writes nothing), undershooting costs one extra round.
REPLY_BODY_CHARS: int = 30_000

# The builder's own model and effort, which the host passes in the process
# environment.
ARTIFACT_MODEL_ENV_VAR = "ANTON_ARTIFACT_MODEL"
ARTIFACT_EFFORT_ENV_VAR = "ANTON_ARTIFACT_REASONING_EFFORT"


def artifact_model_route() -> ModelRoute | None:
    """The model and effort every call of the pipeline asks for, or None to
    build on the agent's own, as with the variable unset or blank.

    Read from the process environment alone, as the render check's address is
    (`configured_url` in anton/core/artifacts/render_check.py). A blank effort
    sends none, so the model runs its own default.
    """
    model = os.environ.get(ARTIFACT_MODEL_ENV_VAR, "").strip()
    if not model:
        return None
    return ModelRoute(model=model, effort=os.environ.get(ARTIFACT_EFFORT_ENV_VAR, "").strip() or None)

# Reserved out of MAX_QUESTIONS_PER_TURN for the brief phase: one
# `show_and_confirm` call plus up to two "revise brief, show again" cycles.
# Not a separate hard cap — the shared budget itself is what eventually stops
# the revise loop (elicit() returns "limit"); this only decides how many of
# the turn's questions the gathering phase may spend before that.
PHASE2_RESERVED_QUESTIONS = 3


def gathering_question_budget(session: "ChatSession | Any", *, act_first: bool = False) -> int:
    """How many `ask_user` calls the gathering phase may make this time.

    Recomputed on every call rather than cached on the state, because
    `session.question_count` keeps changing as questions are asked. In
    act-first mode the brief is not confirmed, so nothing is reserved for it.
    """
    from anton.core.interaction.elicit import MAX_QUESTIONS_PER_TURN

    remaining = MAX_QUESTIONS_PER_TURN - getattr(session, "question_count", 0)
    reserved = 0 if act_first else PHASE2_RESERVED_QUESTIONS
    return max(0, remaining - reserved)


# ── Verdict schemas for diamond nodes (generate_object) ──────────────────────
class RequiredDataItem(BaseModel):
    name: str  # what the datum is, e.g. "list of orders"
    where: str  # where it conceptually lives, e.g. "postgres `orders` table"
    why: str  # why the task needs it


class RequiredData(BaseModel):
    """`define_required_data`: what data is needed and where to get it."""

    items: list[RequiredDataItem] = Field(
        description="Data still missing for the task, one entry per item. "
        "Empty when nothing is missing."
    )
    reasoning: str


class FetchVerdict(BaseModel):
    """`is_possible_to_fetch`: can the required data actually be obtained?"""

    possible: bool
    reasoning: str


# ── Verifier result ──────────────────────────────────────────────────────────
@dataclass
class VerifyResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class BrowserCheck:
    """One load of the page in a browser, or why the page was not loaded.

    `status` is `passed` or `failed` when a browser loaded the page (`verdict`
    holds its findings as verifier messages), `unverified` when none did or
    the check could not vouch for it (`reason` says why; the verdict is
    empty). `findings_listed` says whether `reason` also lists what an
    unverified check still found. `screenshot` is the render check's
    desktop view, when it sent one. `coverage` names the views a browser
    that loaded the page measured and the tabs and toggles it left
    unopened; None when no browser loaded it. `views` are those views as
    the render check listed them, empty from the local browser, which
    presses nothing and lists none. `comparison` is the render check's
    comparison with the page as the turn found it, when the check was sent
    one.
    """

    status: "RenderStatus"
    verdict: VerifyResult = field(default_factory=VerifyResult)
    reason: str = ""
    findings_listed: bool = False
    screenshot: "RenderScreenshot | None" = None
    coverage: "ViewCoverage | None" = None
    views: "tuple[CheckedView, ...]" = ()
    comparison: "Comparison | None" = None


@dataclass(frozen=True)
class BrowserCheckFailed:
    """How a run ends when its last build passed the static checks and
    failed its browser check. The page that build wrote stays on disk, and
    the agent repairs it in a scratchpad cell instead of the run building
    it again.

    `page` is the file the check loaded, relative to the artifact folder.
    `findings` are the check's findings as the build's verdict worded them,
    `views` the views it measured, `screenshot` its view of the page when it
    sent one. `error` is the run's terminal message, for the trace. All of
    it comes from the last build's own check: an earlier build's browser
    fields on `GenState` can name a page a later build deleted.
    `comparison` is that check's comparison with the page as the turn found
    it, when there was one.
    """

    page: str
    findings: tuple[str, ...]
    views: "tuple[CheckedView, ...]"
    screenshot: "RenderScreenshot | None"
    error: str
    comparison: "Comparison | None" = None


# ── FSM state ────────────────────────────────────────────────────────────────
@dataclass
class StepResult:
    node: str
    outcome: str
    detail: str = ""


# Language for a direct caller that names none (bench runs, tests); the tool
# handler requires `user_language`. A concrete name, never "the language of
# the user request": that wording made the model pick an unrelated language.
DEFAULT_USER_LANGUAGE = "English"
# Longest `user_language` the tool accepts. A language name fits; a sentence
# about the user would be quoted into every step header and change the
# call fingerprint each time it is reworded.
MAX_USER_LANGUAGE_CHARS = 40


def language_name(user_language: str | None) -> str:
    """The language every step writes in, as the prompts name it."""
    return " ".join((user_language or "").split()) or DEFAULT_USER_LANGUAGE


def settled_artifact_type(state) -> str:
    """The type gathering settled on, or the registered one before it ran."""
    return state.final_artifact_type or state.artifact_type


def known_data_usable(state) -> bool:
    """Whether `known_data` can stand in for executed code.

    Only for an html-app, whose page embeds a snapshot. A fullstack backend
    reads its source at run time and needs working access code in
    `data_notes`.
    """
    return settled_artifact_type(state) == "html-app" and bool((state.known_data or "").strip())


@dataclass
class GenState:
    session: "ChatSession | Any"
    artifact_type: str
    artifact_path: Path
    slug: str
    # The brief shown to the user. Empty until phase B has drafted one —
    # which is most of a run's life now that the pipeline starts at gathering
    # rather than at a brief handed in by the caller.
    brief: str = ""
    # Derived from `artifact_type` when not given (see `__post_init__`).
    # Callers may still pass it explicitly; leaving it out is the safer
    # default, because a state whose type and this flag disagree would send
    # an html-app down the fullstack branch with nothing reporting it.
    is_fullstack: bool | None = None
    # Entry-file name from the artifact metadata. May be None — `create_artifact`
    # allows omitting it; HTML_APP_DEFAULT_PRIMARY then applies.
    primary: str | None = None
    # Body of `prd.md` when a previous run left one in the artifact folder.
    # This — not `brief` — is the requirements source on the normal path: it
    # is the document written from the latest brief, while `brief`
    # is assembled by the calling agent. Empty when there is no PRD, and every
    # reader treats empty as "fall back to brief".
    prd: str = ""
    data_notes: str = ""
    data_iterations: int = 0
    # The data loop's one fetch against a "not possible" verdict was spent.
    # Per call, never persisted: a repeat call gets its own attempt.
    forced_fetch_used: bool = False
    api_spec: str | None = None
    files_written: list[str] = field(default_factory=list)
    # Generation inputs (spec.md, openapi.json) rather than user-facing output:
    # reported in a separate field so the agent does not present them as artifacts.
    internal_files: list[str] = field(default_factory=list)
    # What the verifiers flagged on the ACCEPTED attempt without failing it,
    # and which checks never ran at all (no headless browser; a fullstack page
    # that needs its backend). Both go into the result: the calling agent is
    # told not to re-verify the files, so this is the one place the user can
    # learn that a check was skipped rather than passed (S-03).
    verify_warnings: list[str] = field(default_factory=list)
    checks_skipped: list[str] = field(default_factory=list)
    # The latest browser load of the page this run: `passed`, `failed`, or
    # `unverified` when no browser loaded it or the check could not vouch for
    # it. None until a browser step runs. The result reports whether it
    # passed, and the session records it for the end-of-turn line.
    browser_status: "RenderStatus | None" = None
    # Whether the latest load was unverified and its `checks_skipped` line
    # lists what the check still found. The session then lists those
    # findings no second time this turn.
    browser_findings_listed: bool = False
    # The render check's desktop view of the latest load, when it sent one.
    # Reaches the calling agent as an image beside the result.
    browser_screenshot: "RenderScreenshot | None" = None
    # The views the latest load measured and the tabs and toggles it left
    # unopened, or None when no browser loaded the page. Beside a pass, the
    # result names both, so the agent calls only those views verified.
    browser_coverage: "ViewCoverage | None" = None
    # The page the latest load checked, by its path inside the artifact
    # folder (`index.html`); it captions the screenshot. The generator may
    # have named it differently from the primary the run started with.
    browser_page: str = ""
    # The artifact's main page as it was when the turn began, and its key:
    # each build's browser check is compared with it. None for a new
    # artifact, a fullstack app, or without the render-check service.
    previous_page: "tuple[PageKey, str] | None" = None
    # The latest browser check's comparison with `previous_page`.
    comparison: "Comparison | None" = None
    # Held while `_embed_attachments` rewrites the page. The tool handler sets
    # `abandoned` under it before it puts the page back for a run the turn
    # gave up on: the embedding thread outlives the cancelled task and must
    # not write over the restored page.
    page_lock: threading.Lock = field(default_factory=threading.Lock)
    abandoned: threading.Event = field(default_factory=threading.Event)
    # Files the user attached to the conversation, as the tool call named
    # them (S-01). Data files are read by the gathering step; assets are
    # copied into the artifact by `orchestrator._stage_attachments`.
    attachments: list["Attachment"] = field(default_factory=list)
    # Where the launched backend answers (fullstack only, set by `run_app`).
    # Reported as the app's entry point: `static/index.html` opened from disk
    # cannot reach its `/api/*`, so the file path is not one.
    app_url: str | None = None
    app_port: int | None = None
    # A fullstack app launched into an app slot (`live_app_slots`): its
    # address, slot and the declared connections it didn't get. No port.
    live_app: "LiveAppLaunchResult | None" = None
    # Why a live app's launch didn't happen, when no new backend could fix it.
    launch_failure: "LiveAppLaunchError | None" = None
    # Where an app the run gave up on may still run, when its stop failed:
    # the run's final text carries it. Empty when the stop went through.
    live_app_stop_note: str = ""
    trace: list[StepResult] = field(default_factory=list)
    error: str | None = None
    trace_log: "GenTrace | NullTrace" = field(default_factory=NullTrace)
    # Progress channel to the tool handler. None when nobody is
    # listening — the non-streaming path, `bench_generate.py`, most tests — so
    # every call site stays unconditional. Must be UNBOUNDED: `step_started`
    # is called from synchronous FSM code that cannot await a full queue, and
    # `QueueFull` there would abort a generation over a progress line.
    progress: "asyncio.Queue[str | None] | None" = None
    # Steps that already showed an "attempt 2" line for a truncation retry.
    # Later retries of the same step show only on the live status line, which
    # every retry sets, so a gathering run with several cut rounds does not
    # repeat the same row.
    retry_announced: set[str] = field(default_factory=set)
    # Where this run entered the pipeline (`discovery.checkpoint.ENTRY_*`,
    # set by `orchestrator.run`): a resumed run has fewer steps ahead of it,
    # and the `N of M` on every progress line counts only those.
    entry: str = "full"
    # Built on the first counted step, not up front: the artifact type that
    # decides the plan is settled by `finish_gathering`, after the run began.
    step_counter: StepCounter | None = None
    # Live tail of the streaming LLM calls, on the same channel (see
    # `progress.LivePeek`). Built on first use, only when someone listens.
    peek: LivePeek | None = None

    # ── Discovery phases (A-C) ───────────────────────────────────────────
    # The tool's own inputs. `brief` above holds the brief markdown
    # once phase B has run; before that it is empty.
    user_request: str = ""
    agent_understanding: str = ""
    known_data: str = ""
    user_preferences: str = ""
    # English name of the language the user writes in, as the calling agent
    # stated it. Read through `language_name`.
    user_language: str = ""
    # The session's connections as data (slugs and DS_* names, no values), or
    # None when nothing is connected. Filled by the entry point; rendered whole
    # as `datasource_context`, and for only the declared sources by the
    # backend prompt.
    datasource_catalog: "DatasourceCatalog | None" = None
    # The project's anton.md as the `## Project rules` section
    # (`prompts.project_rules_section`), or "" when it has none. Read once per
    # call by the entry point and placed in every step's system prompt. Not
    # persisted, so an edit to anton.md applies from the next call.
    project_rules: str = ""
    # The skills the host trusts for artifact builds
    # (`SkillStore.list_for_artifacts`) as the `## Artifact skills` section
    # (`prompts.artifact_skills_section`), or "". Read with `project_rules`;
    # the page writers, the backend writer and the fetch step get it.
    artifact_skills: str = ""
    # The `## Scratchpads already in this session` section of the gathering
    # kickoff (`discovery.prompts.render_scratchpads_context`): the pads the
    # calling agent already ran, so the step reuses them instead of guessing
    # a new name and being refused by the single-scratchpad guard.
    scratchpads_context: str = ""
    # Body of the built-in public-data-sources skill, read from disk on the
    # data loop's first fetch and reused by its later iterations. None until
    # then; "" when the skill is unavailable.
    public_sources: str | None = None
    # THE shared message list for phases A-D. Dropped at the spec boundary:
    # generation nodes build their context from the fields on this state, not
    # from this list. One list, because phase B relies on seeing what phase
    # A's scratchpad calls returned.
    messages: list[dict] = field(default_factory=list)
    qa_log: list[str] = field(default_factory=list)
    gathering_notes: str = ""
    # Structured halves of the `finish_gathering` call, kept apart from the
    # rendered `gathering_notes` because the brief presents them differently:
    # assumptions as proposals to confirm, open points as questions to the
    # user. Both survive the process via `discovery.json`.
    assumptions: list[str] = field(default_factory=list)
    open_points: list[str] = field(default_factory=list)
    # Set by `finish_gathering`. Empty means it was never called and the
    # originally registered `artifact_type` stands.
    final_artifact_type: str = ""
    # Set by `finish_gathering`. False means the loop ran out of rounds
    # instead — one of the two conditions that opens the emergency data loop.
    gathering_complete: bool = False
    declared_sources: list[str] = field(default_factory=list)
    # Declared sources with nothing executed against them. Tracked explicitly
    # rather than inferred from `data_notes` being empty: after a user
    # correction the notes are full of the PREVIOUS gathering's cells, and an
    # emptiness check would read that as "everything is covered".
    unverified_sources: list[str] = field(default_factory=list)
    # Declared sources that only the `known_data` mark verified. A new
    # iteration that brings no `known_data` puts them back in
    # `unverified_sources`: nothing else vouches for them.
    known_data_verified: list[str] = field(default_factory=list)
    # Raw material for the deterministic renderers in discovery/notes.py.
    scratchpad_execs: list[dict] = field(default_factory=list)
    web_calls: list[dict] = field(default_factory=list)
    web_notes: str = ""
    # True when a repeat call arrived with changed soft fields, i.e. the user
    # asked for something. Set by the entry point from the stored
    # `call_fingerprint`; decides whether the brief is redrawn or reused
    # verbatim. An optimization, never a confirmation signal.
    call_changed: bool = False
    # The session's "act first, ask later" setting, read once by the entry
    # point. True: ask only what blocks the build and show the brief without
    # waiting for confirmation. False — also for a session double without the
    # attribute — keeps the confirm flow.
    act_first: bool = False
    # Whether the user has seen the brief without being asked to confirm it:
    # `announce_brief` sent it in this call, or (acting first) an earlier turn
    # showed it — set by `orchestrator.run` from the entry. Not persisted.
    brief_shown: bool = False
    # A gathering question timed out. The user is away, so the brief is
    # written but not shown for confirmation: the run hands back
    # `needs_confirmation`, and their late answer to the still-open card
    # arrives as a repeat call carrying it. Not persisted.
    question_unanswered: bool = False
    # Installed by the entry point. None on the bench harness and in unit
    # tests that construct a state directly, so every read goes through
    # `winding_down()`.
    spend: "Any | None" = None
    # Built on first read and never rebuilt. Together with `messages` they
    # are the cached prefix of every phase A-D call, so anything that would
    # change them mid-run has to live in a step message instead.
    _pipeline_system: str = ""
    _pipeline_tools: "list[dict] | None" = None
    _pipeline_kickoff_evidence: str = ""

    @property
    def datasource_context(self) -> str:
        """The whole `## Connected Data Sources` section, as the chat prompt
        renders it, or "" when nothing is connected. The call kickoff and the
        fetch node carry it so they know what can be queried."""
        return self.datasource_catalog.render() if self.datasource_catalog else ""

    @property
    def pipeline_system(self) -> str:
        if not self._pipeline_system:
            from .discovery.prompts import build_pipeline_system_prompt

            self._pipeline_system = build_pipeline_system_prompt(self)
        return self._pipeline_system

    @property
    def pipeline_tools(self) -> list[dict]:
        if self._pipeline_tools is None:
            from .discovery.sub_tools import pipeline_tool_schemas

            self._pipeline_tools = pipeline_tool_schemas(self.buildable_types)
        return self._pipeline_tools

    @property
    def buildable_types(self) -> tuple[str, ...]:
        """The types this run may settle on, in schema order: html-app alone
        where the session offers no fullstack app, because nothing there
        starts an app backend (`live_app_slots.fullstack_offered`, read once
        by the session into `_fullstack_offered` for its tools and prompt)."""
        from anton.core.artifacts.models import GENERATOR_ARTIFACT_TYPES_ORDERED

        if getattr(self.session, "_fullstack_offered", True):
            return GENERATOR_ARTIFACT_TYPES_ORDERED
        return ("html-app",)

    def __post_init__(self) -> None:
        if self.is_fullstack is None:
            self.is_fullstack = self.artifact_type != "html-app"

    def settle_artifact_type(self, final_type: str) -> None:
        """Make `final_type` the type the rest of the run builds for.

        The gathering step may choose a type other than the one the artifact
        was registered with. Writing that choice into metadata and the
        checkpoint is not enough: `is_fullstack`, the `stateless` switches in
        the spec and kickoff prompts and `verify_backend` all read THIS
        object, and a run whose state still says `html-app` skips the API
        spec and the backend of the fullstack app it just decided to build
        (I-40). The step plan is rebuilt too — the counter was created at the
        first `gathering` line, before the type was known — while the
        numbers already handed out stay as they are.
        """
        if not final_type or final_type == self.artifact_type:
            return
        self.artifact_type = final_type
        self.is_fullstack = final_type != "html-app"
        if self.step_counter is not None:
            self.step_counter.plan = plan_steps(
                is_fullstack=self.is_fullstack, entry=self.entry
            )

    def record_qa(self, question: str, answer_summary: str) -> None:
        self.qa_log.append(f"- **Q:** {question}\n  **A:** {answer_summary}")

    def qa_log_markdown(self) -> str:
        return "\n".join(self.qa_log) if self.qa_log else "(no questions were asked)"

    def winding_down(self) -> bool:
        """The one place that tolerates a missing guard.

        `spend` is None for the bench harness and for tests that build a
        state by hand, and neither should acquire budget behaviour just by
        existing. Every phase asks through here rather than reaching into
        `spend` directly, so that None-check lives once.
        """
        return self.spend is not None and self.spend.should_wind_down()

    def record(self, node: str, outcome: str, detail: str = "") -> None:
        self.trace.append(StepResult(node=node, outcome=outcome, detail=detail))
        self.trace_log.node(node, outcome, detail)

    def step_started(self, node: str, *, attempt: int = 0) -> None:
        """Announce that `node` is STARTING, for the user's benefit.

        Deliberately separate from `record`, which fires when a node is
        already done: the two longest nodes (`make_tech_spec`, a single
        minute-plus LLM call, and the generation loops) would otherwise
        report only in hindsight, leaving exactly the silence this is meant
        to remove. `record` stays the sole source of the journal and the
        trace — nothing here feeds a prompt.
        """
        if self.progress is None:
            return
        if self.step_counter is None:
            self.step_counter = StepCounter(
                plan_steps(is_fullstack=bool(self.is_fullstack), entry=self.entry)
            )
        text = label_for(
            node, is_fullstack=self.is_fullstack, attempt=attempt,
            position=self.step_counter.position(node),
        )
        if text is not None:
            self.progress.put_nowait(text)

    def peek_for(self, node: str):
        """The `on_text` callback for `node`'s streaming LLM call, or None.

        None when there is no progress channel, so the engine's stream
        consumer stays a plain drain there. The node's group name is what a
        second live stream is told apart by (`progress.PEEK_GROUPS`).
        """
        if self.progress is None:
            return None
        if self.peek is None:
            self.peek = LivePeek(self.progress)
        group = PEEK_GROUPS.get(node, node)
        peek = self.peek
        return lambda delta: peek.feed(group, delta)

    def peek_done(self, node: str) -> None:
        """Drop `node`'s tail: its call is over, the footer must not keep
        showing the end of a file as if it were still being written."""
        if self.peek is not None:
            self.peek.clear(PEEK_GROUPS.get(node, node))

    def journal(self) -> str:
        """Compact one-line-per-step log of everything the FSM did so far.

        Injected into later steps' prompts (prompts._brief_and_notes,
        orchestrator._spec_context) so every node sees the run's history —
        including failed attempts — without sharing full transcripts.
        """
        lines: list[str] = []
        for s in self.trace:
            detail = " ".join(s.detail.split())
            if len(detail) > JOURNAL_DETAIL_MAX:
                detail = detail[:JOURNAL_DETAIL_MAX] + "…"
            lines.append(
                f"- {s.node}: {s.outcome}" + (f" — {detail}" if detail else "")
            )
        return "\n".join(lines)
