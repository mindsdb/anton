"""State container and verdict schemas for the artifact-generation FSM.

The orchestrator (`orchestrator.py`) walks graph nodes over one `GenState`.
Diamond nodes are resolved by `session._llm.generate_object(...)` calls that
return the Pydantic verdict models below. Verifiers return `VerifyResult`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .debug_trace import NullTrace, GenTrace  # noqa: F401  (GenTrace re-exported for typing)

if TYPE_CHECKING:
    from anton.chat_session import ChatSession

# ── Budgets (see design spec) ────────────────────────────────────────────────
DATA_LOOP_MAX: int = 3
GEN_VERIFY_MAX_RETRIES: int = 1
RUNAPP_MAX_RETRIES: int = 1

# Per-line detail cap for GenState.journal() — keeps the journal compact when
# a step's detail is long (full text still lives in data_notes / trace_log).
JOURNAL_DETAIL_MAX: int = 300


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
    brief: str
    is_fullstack: bool
    # Entry-file name from the artifact metadata. May be None — `create_artifact`
    # allows omitting it; HTML_APP_DEFAULT_PRIMARY then applies.
    primary: str | None = None
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

    def record(self, node: str, outcome: str, detail: str = "") -> None:
        self.trace.append(StepResult(node=node, outcome=outcome, detail=detail))
        self.trace_log.node(node, outcome, detail)

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
