"""Core data types for the eval harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Benchmark case specification (loaded from YAML)
# ---------------------------------------------------------------------------


@dataclass
class TurnSpec:
    """One turn of user input in a multi-turn eval case."""

    user_input: str
    expect_tool_calls: list[str] | None = None  # optional per-turn tool assertions


@dataclass
class Scorer:
    """A single scoring criterion applied to eval results."""

    type: Literal[
        "contains",
        "regex",
        "not_contains",
        "tool_called",
        "tool_not_called",
        "tool_call_count",
        "code_output",
        "llm_judge",
    ]
    value: str  # substring, regex pattern, tool name, or rubric text
    weight: float = 1.0
    expected_count: int | None = None  # only used by tool_call_count


@dataclass
class EvalCase:
    """A single benchmark case, loaded from YAML."""

    id: str  # unique slug, e.g. "fibonacci-20"
    category: str  # simple_qa, code_generation, multi_step, memory, error_recovery, context_pressure
    description: str
    turns: list[TurnSpec]
    scorers: list[Scorer]
    tags: list[str] = field(default_factory=list)
    estimated_cost_usd: float = 0.0
    temperature: float = 0.0  # default to deterministic


# ---------------------------------------------------------------------------
# Execution results (produced by the runner)
# ---------------------------------------------------------------------------


@dataclass
class ToolCallRecord:
    """A single tool invocation captured during a turn."""

    name: str
    input: dict[str, Any]
    result_text: str


@dataclass
class TurnRecord:
    """Captured data for a single turn execution."""

    user_input: str
    response_text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    tool_rounds: int = 0  # how many LLM round-trips (tool loops) in this turn


@dataclass
class ScoreResult:
    """Result of applying one Scorer to eval output."""

    scorer_type: str
    value: str
    passed: bool
    weight: float
    detail: str = ""  # explanation, especially for llm_judge


@dataclass
class EvalResult:
    """Complete result for one EvalCase."""

    case_id: str
    category: str
    description: str
    turns: list[TurnRecord]
    scores: list[ScoreResult] = field(default_factory=list)
    overall_score: float = 0.0  # weighted pass rate: 0.0 to 1.0
    total_tokens: int = 0
    total_latency_seconds: float = 0.0
    error: str | None = None  # set if the case crashed entirely
