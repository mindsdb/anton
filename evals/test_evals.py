"""Parametrized eval tests — one pytest test per YAML benchmark case.

Usage:
    pytest evals/ -v                          # run all cases
    pytest evals/ -v -k fibonacci-20          # run one case
    pytest evals/ -v -k code_generation       # run one category
    pytest evals/ -v -k "fast"                # run cases tagged 'fast'

Requires ANTHROPIC_API_KEY (or ANTON_ANTHROPIC_API_KEY) in the environment.
"""

from __future__ import annotations

import pytest

from evals.loader import load_cases
from evals.report import build_report, format_summary, save_report
from evals.runner import EvalRunner
from evals.types import EvalCase

_ALL_CASES = load_cases()


@pytest.mark.parametrize("case", _ALL_CASES, ids=[c.id for c in _ALL_CASES])
async def test_eval_case(eval_runner: EvalRunner, case: EvalCase) -> None:
    """Run a single benchmark case and assert it passes."""
    result = await eval_runner.run_case(case)

    # Print details for debugging failures
    if result.error:
        pytest.fail(f"Case {case.id} crashed: {result.error}")

    failed_scorers = [s for s in result.scores if not s.passed]
    assert result.overall_score >= 0.5, (
        f"Case {case.id} scored {result.overall_score:.2f}. "
        f"Failed scorers: {[(s.scorer_type, s.detail) for s in failed_scorers]}"
    )


async def test_full_suite_report(eval_runner: EvalRunner) -> None:
    """Run the full suite and produce a report (smoke test for reporting)."""
    # Only run this if explicitly requested (it's expensive)
    cases = load_cases(filter_tags=["fast", "cheap"])
    if not cases:
        pytest.skip("No fast/cheap cases found")

    results = await eval_runner.run_suite(cases, budget_usd=0.10)
    report = build_report(
        results,
        planning_model="claude-sonnet-4-6",
        coding_model="claude-haiku-4-5-20251001",
    )

    # Verify report structure
    assert "summary" in report
    assert "by_category" in report
    assert "cases" in report
    assert report["summary"]["total_cases"] == len(results)

    # Print summary for visibility
    print("\n" + format_summary(report))

    # Save report
    path = save_report(report)
    assert path.exists()
