"""State container and verdict schemas for the artifact-generation FSM.

The orchestrator (`orchestrator.py`) walks graph nodes over one `GenState`.
Diamond nodes are resolved by `session._llm.generate_object(...)` calls that
return the Pydantic verdict models below. Verifiers return `VerifyResult`.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .debug_trace import NullTrace, GenTrace  # noqa: F401  (GenTrace re-exported for typing)
from .progress import PEEK_GROUPS, LivePeek, StepCounter, label_for, plan_steps
from .attachments import Attachment

if TYPE_CHECKING:
    from anton.chat_session import ChatSession

# ── Budgets (see design spec) ────────────────────────────────────────────────
DATA_LOOP_MAX: int = 3
# Generation-loop failures (round budget, no tool calls) and verification
# failures get SEPARATE retry budgets. With a single shared counter a loop
# failure consumed the retry reserved for fixing verifier findings, and a
# trivially fixable verification error became terminal.
GEN_LOOP_MAX_RETRIES: int = 1
GEN_VERIFY_MAX_RETRIES: int = 1
RUNAPP_MAX_RETRIES: int = 1

# Per-line detail cap for GenState.journal() — keeps the journal compact when
# a step's detail is long (full text still lives in data_notes / trace_log).
JOURNAL_DETAIL_MAX: int = 300

# Output budgets for the two whole-document calls (`make_tech_spec`,
# `make_api_spec`), overriding the client default of 8192 — a specification is
# among the longest single answers anton asks for, and reasoning models spend
# their internal thinking from the same budget (a 25 842-character spec was
# measured dying at 8192 output tokens).
#
# The two values are what the MindsHub gateway actually accepts, not round
# numbers: measured 2026-08-24 against `api.mindshub.ai/v1`, alias `opus` —
# 8192/16384/20480 answer normally, 24576 and above return HTTP 500. That 500
# is classified as a transient provider error, so an over-large budget does not
# fail fast; it burns the retry ladder first. Raise both together, and re-measure
# before doing so.
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

# Reserved out of MAX_QUESTIONS_PER_TURN for the brief phase: one
# `show_and_confirm` call plus up to two "revise brief, show again" cycles.
# Not a separate hard cap — the shared budget itself is what eventually stops
# the revise loop (elicit() returns "limit"); this only decides how many of
# the turn's questions the gathering phase may spend before that.
PHASE2_RESERVED_QUESTIONS = 3


def gathering_question_budget(session: "ChatSession | Any") -> int:
    """How many `ask_user` calls the gathering phase may make this time.

    Recomputed on every call rather than cached on the state, because
    `session.question_count` keeps changing as questions are asked.
    """
    from anton.core.interaction.elicit import MAX_QUESTIONS_PER_TURN

    remaining = MAX_QUESTIONS_PER_TURN - getattr(session, "question_count", 0)
    return max(0, remaining - PHASE2_RESERVED_QUESTIONS)


# ── Verdict schemas for diamond nodes (generate_object) ──────────────────────
class RequiredDataItem(BaseModel):
    name: str  # what the datum is, e.g. "list of orders"
    where: str  # where it conceptually lives, e.g. "postgres `orders` table"
    why: str  # why the task needs it


class RequiredData(BaseModel):
    """`define_required_data`: what data is needed and where to get it."""

    items: list[RequiredDataItem]
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


# ── FSM state ────────────────────────────────────────────────────────────────
@dataclass
class StepResult:
    node: str
    outcome: str
    detail: str = ""


@dataclass
class GenState:
    session: "ChatSession | Any"
    artifact_type: str
    artifact_path: Path
    slug: str
    # The brief the user agreed to. Empty until phase B has drafted one —
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
    # is the document the user actually reviewed and accepted, while `brief`
    # is assembled by the calling agent. Empty when there is no PRD, and every
    # reader treats empty as "fall back to brief".
    prd: str = ""
    data_notes: str = ""
    data_iterations: int = 0
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
    # Files the user attached to the conversation, as the tool call named
    # them (S-01). Data files are read by the gathering step; assets are
    # copied into the artifact by `orchestrator._stage_attachments`.
    attachments: list["Attachment"] = field(default_factory=list)
    # Where the launched backend answers (fullstack only, set by `run_app`).
    # Reported as the app's entry point: `static/index.html` opened from disk
    # cannot reach its `/api/*`, so the file path is not one.
    app_url: str | None = None
    app_port: int | None = None
    trace: list[StepResult] = field(default_factory=list)
    error: str | None = None
    trace_log: "GenTrace | NullTrace" = field(default_factory=NullTrace)
    # Progress channel to the tool handler. None when nobody is
    # listening — the non-streaming path, `bench_generate.py`, most tests — so
    # every call site stays unconditional. Must be UNBOUNDED: `step_started`
    # is called from synchronous FSM code that cannot await a full queue, and
    # `QueueFull` there would abort a generation over a progress line.
    progress: "asyncio.Queue[str | None] | None" = None
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
    # The tool's own inputs. `brief` above holds the confirmed brief markdown
    # once phase B has run; before that it is empty.
    user_request: str = ""
    agent_understanding: str = ""
    known_data: str = ""
    user_preferences: str = ""
    # The `## Connected Data Sources` section as `build_datasource_context`
    # renders it (slugs and DS_* names, no values), or "" when nothing is
    # connected. Filled by the entry point; rendered into the call kickoff so
    # the gathering step knows what it can query.
    datasource_context: str = ""
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
    # Raw material for the deterministic renderers in discovery/notes.py.
    scratchpad_execs: list[dict] = field(default_factory=list)
    web_calls: list[dict] = field(default_factory=list)
    web_notes: str = ""
    # True when a repeat call arrived with changed soft fields, i.e. the user
    # asked for something. Set by the entry point from the stored
    # `call_fingerprint`; decides whether the brief is redrawn or reused
    # verbatim. An optimization, never a confirmation signal.
    call_changed: bool = False
    # Installed by the entry point. None on the bench harness and in unit
    # tests that construct a state directly, so every read goes through
    # `winding_down()`.
    spend: "Any | None" = None
    # Built on first read and never rebuilt. Together with `messages` they
    # are the cached prefix of every phase A-D call, so anything that would
    # change them mid-run has to live in a step message instead.
    _pipeline_system: str = ""
    _pipeline_tools: "list[dict] | None" = None

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

            self._pipeline_tools = pipeline_tool_schemas()
        return self._pipeline_tools

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
        spec and the backend of the fullstack app it just agreed to build
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
