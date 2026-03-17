"""Scorer implementations — apply grading criteria to eval results."""

from __future__ import annotations

import re

from evals.types import EvalResult, ScoreResult, Scorer, TurnRecord


def _all_response_text(turns: list[TurnRecord]) -> str:
    """Concatenate all assistant response text across turns."""
    return "\n".join(t.response_text for t in turns)


def _all_tool_calls(turns: list[TurnRecord]) -> list[dict]:
    """Flatten all tool call records across turns into dicts."""
    return [
        {"name": tc.name, "input": tc.input, "result_text": tc.result_text}
        for t in turns
        for tc in t.tool_calls
    ]


def _all_tool_result_text(turns: list[TurnRecord]) -> str:
    """Concatenate all tool result text (e.g. scratchpad stdout)."""
    return "\n".join(
        tc.result_text
        for t in turns
        for tc in t.tool_calls
    )


def apply_scorer(scorer: Scorer, turns: list[TurnRecord]) -> ScoreResult:
    """Apply a single deterministic scorer to collected turn data.

    For ``llm_judge`` scorers, use :func:`apply_scorer_async` instead.

    Raises:
        ValueError: If the scorer type is ``llm_judge`` (must use async variant).
    """
    response_text = _all_response_text(turns)
    tool_calls = _all_tool_calls(turns)
    tool_result_text = _all_tool_result_text(turns)

    match scorer.type:
        case "contains":
            passed = scorer.value.lower() in response_text.lower()
            detail = "substring found" if passed else "substring not found"

        case "regex":
            match_obj = re.search(scorer.value, response_text, re.IGNORECASE)
            passed = match_obj is not None
            detail = f"matched: {match_obj.group()}" if passed else "no regex match"

        case "not_contains":
            passed = scorer.value.lower() not in response_text.lower()
            detail = "substring absent (good)" if passed else "substring found (bad)"

        case "tool_called":
            passed = any(tc["name"] == scorer.value for tc in tool_calls)
            detail = f"tool '{scorer.value}' was called" if passed else f"tool '{scorer.value}' was NOT called"

        case "tool_not_called":
            passed = not any(tc["name"] == scorer.value for tc in tool_calls)
            detail = f"tool '{scorer.value}' was not called (good)" if passed else f"tool '{scorer.value}' was called (bad)"

        case "tool_call_count":
            count = sum(1 for tc in tool_calls if tc["name"] == scorer.value)
            expected = scorer.expected_count or 0
            passed = count == expected
            detail = f"tool '{scorer.value}' called {count} times (expected {expected})"

        case "code_output":
            passed = scorer.value in tool_result_text
            detail = "output found in tool results" if passed else "output NOT found in tool results"

        case "llm_judge":
            raise ValueError(
                "llm_judge scorers must be applied via apply_scorer_async()"
            )

        case _:
            raise ValueError(f"Unknown scorer type: {scorer.type}")

    return ScoreResult(
        scorer_type=scorer.type,
        value=scorer.value,
        passed=passed,
        weight=scorer.weight,
        detail=detail,
    )


async def apply_scorer_async(
    scorer: Scorer,
    turns: list[TurnRecord],
    *,
    judge_fn=None,
) -> ScoreResult:
    """Apply a scorer, including async ``llm_judge`` types.

    Args:
        scorer: The scorer to apply.
        turns: Collected turn records.
        judge_fn: Async callable for LLM-as-judge. Signature:
                  ``async (rubric: str, response_text: str, tool_outputs: str) -> tuple[bool, str]``
                  If None and scorer is llm_judge, returns a warning result.
    """
    if scorer.type != "llm_judge":
        return apply_scorer(scorer, turns)

    response_text = _all_response_text(turns)
    tool_result_text = _all_tool_result_text(turns)

    if judge_fn is None:
        return ScoreResult(
            scorer_type="llm_judge",
            value=scorer.value,
            passed=True,
            weight=scorer.weight,
            detail="WARN: no judge_fn provided — defaulting to pass",
        )

    passed, detail = await judge_fn(scorer.value, response_text, tool_result_text)
    return ScoreResult(
        scorer_type="llm_judge",
        value=scorer.value,
        passed=passed,
        weight=scorer.weight,
        detail=detail,
    )


def compute_overall_score(scores: list[ScoreResult]) -> float:
    """Compute weighted pass rate: sum(weight * passed) / sum(weight)."""
    total_weight = sum(s.weight for s in scores)
    if total_weight == 0:
        return 0.0
    earned = sum(s.weight for s in scores if s.passed)
    return earned / total_weight
