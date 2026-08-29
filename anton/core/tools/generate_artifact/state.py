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
from .progress import label_for

if TYPE_CHECKING:
    from anton.chat_session import ChatSession

# ── Budgets (see design spec) ────────────────────────────────────────────────
DATA_LOOP_MAX: int = 3
# Generation-loop failures (round budget, no tool calls) and verification
# failures get SEPARATE retry budgets. With a single shared counter a loop
# failure consumed the retry reserved for fixing verifier findings, and a
# trivially fixable verification error became terminal (live run 2026-08-27).
GEN_LOOP_MAX_RETRIES: int = 1
GEN_VERIFY_MAX_RETRIES: int = 1
RUNAPP_MAX_RETRIES: int = 1

# Per-line detail cap for GenState.journal() — keeps the journal compact when
# a step's detail is long (full text still lives in data_notes / trace_log).
JOURNAL_DETAIL_MAX: int = 300

# Output budgets for the two whole-document calls (`make_tech_spec`,
# `make_api_spec`), overriding the client default of 8192 — a specification is
# among the longest single answers anton asks for, and reasoning models spend
# their internal thinking from the same budget (stage-1a measured a 25 842-char
# spec dying at 8192 output tokens, ENG-1116).
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
# NOT applied to round 0. That round runs on the planning model, which is
# ~2.3x slower per token (75 vs 170 tok/s measured), and a long generation is
# silent on the wire for its whole duration (see sub_tools.CHUNK_SOFT_LIMIT).
# Round 0 therefore keeps the client default: at 8192 an over-long write is
# merely truncated, which the loop recovers from, whereas at 20480 the same
# call runs long enough to have its connection dropped — measured 4 failures
# out of 4 at 131-143s.
GEN_WRITE_MAX_TOKENS: int = 20480

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
class DataVerdict(BaseModel):
    """`is_data_enough`: is there enough data to solve the task?"""

    enough: bool
    reasoning: str


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
    # Body of `prd.md` when `generate_prd` left one in the artifact folder
    # (ENG-969 → ENG-968). This — not `brief` — is the requirements source on
    # the normal path: it is the document the user actually reviewed and
    # accepted, while `brief` is assembled by the calling agent. Empty when
    # there is no PRD, and every reader treats empty as "fall back to brief".
    prd: str = ""
    data_notes: str = ""
    data_iterations: int = 0
    api_spec: str | None = None
    files_written: list[str] = field(default_factory=list)
    # Generation inputs (spec.md, openapi.json) rather than user-facing output:
    # reported in a separate field so the agent does not present them as artifacts.
    internal_files: list[str] = field(default_factory=list)
    trace: list[StepResult] = field(default_factory=list)
    error: str | None = None
    trace_log: "GenTrace | NullTrace" = field(default_factory=NullTrace)
    # Progress channel to the tool handler (ENG-970). None when nobody is
    # listening — the non-streaming path, `bench_generate.py`, most tests — so
    # every call site stays unconditional. Must be UNBOUNDED: `step_started`
    # is called from synchronous FSM code that cannot await a full queue, and
    # `QueueFull` there would abort a generation over a progress line.
    progress: "asyncio.Queue[str | None] | None" = None

    # ── Discovery phases (A-C) ───────────────────────────────────────────
    # The tool's own inputs. `brief` above holds the confirmed brief markdown
    # once phase B has run; before that it is empty.
    user_request: str = ""
    agent_understanding: str = ""
    known_data: str = ""
    user_preferences: str = ""
    # THE shared message list for phases A-D. Dropped at the spec boundary:
    # generation nodes build their context from the fields on this state, not
    # from this list. One list, because phase B relies on seeing what phase
    # A's scratchpad calls returned.
    messages: list[dict] = field(default_factory=list)
    qa_log: list[str] = field(default_factory=list)
    gathering_notes: str = ""
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

    def __post_init__(self) -> None:
        if self.is_fullstack is None:
            self.is_fullstack = self.artifact_type != "html-app"

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
        text = label_for(node, is_fullstack=self.is_fullstack, attempt=attempt)
        if text is not None:
            self.progress.put_nowait(text)

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
