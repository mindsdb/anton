"""Generate JSON and human-readable reports from eval results."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.types import EvalResult

# Approximate per-token costs (USD) — Anthropic, March 2026
_COST_PER_INPUT_TOKEN = 3.0 / 1_000_000  # Sonnet input
_COST_PER_OUTPUT_TOKEN = 15.0 / 1_000_000  # Sonnet output

_RESULTS_DIR = Path(__file__).parent / "results"


def _estimate_cost(result: EvalResult) -> float:
    """Rough cost estimate for an eval result based on token counts."""
    input_tok = sum(t.input_tokens for t in result.turns)
    output_tok = sum(t.output_tokens for t in result.turns)
    return input_tok * _COST_PER_INPUT_TOKEN + output_tok * _COST_PER_OUTPUT_TOKEN


def build_report(
    results: list[EvalResult],
    *,
    planning_model: str = "",
    coding_model: str = "",
) -> dict[str, Any]:
    """Build a structured report dict from eval results.

    Returns:
        A dict suitable for JSON serialization.
    """
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%d_%H%M%S")

    # Per-category aggregation
    by_cat: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "passed": 0, "score_sum": 0.0, "tokens": 0, "cost": 0.0}
    )

    total_tokens = 0
    total_cost = 0.0
    total_passed = 0

    case_reports: list[dict[str, Any]] = []
    for r in results:
        cost = _estimate_cost(r)
        passed = r.overall_score >= 0.5 and r.error is None

        cat = by_cat[r.category]
        cat["total"] += 1
        cat["passed"] += int(passed)
        cat["score_sum"] += r.overall_score
        cat["tokens"] += r.total_tokens
        cat["cost"] += cost

        total_tokens += r.total_tokens
        total_cost += cost
        total_passed += int(passed)

        case_reports.append({
            "case_id": r.case_id,
            "category": r.category,
            "description": r.description,
            "overall_score": r.overall_score,
            "passed": passed,
            "total_tokens": r.total_tokens,
            "total_latency_seconds": r.total_latency_seconds,
            "error": r.error,
            "scores": [asdict(s) for s in r.scores],
            "turns": [
                {
                    "user_input": t.user_input[:200],
                    "response_text": t.response_text[:500],
                    "tool_calls": [
                        {"name": tc.name, "result_preview": tc.result_text[:200]}
                        for tc in t.tool_calls
                    ],
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "latency_seconds": t.latency_seconds,
                    "tool_rounds": t.tool_rounds,
                }
                for t in r.turns
            ],
        })

    # Finalize category scores
    by_category_final = {}
    for cat_name, cat in by_cat.items():
        avg_score = cat["score_sum"] / cat["total"] if cat["total"] else 0
        by_category_final[cat_name] = {
            "total": cat["total"],
            "passed": cat["passed"],
            "score": round(avg_score, 4),
            "tokens": cat["tokens"],
            "cost_usd": round(cat["cost"], 4),
        }

    total_cases = len(results)
    overall_score = sum(r.overall_score for r in results) / total_cases if total_cases else 0

    return {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "config": {
            "planning_model": planning_model,
            "coding_model": coding_model,
        },
        "summary": {
            "total_cases": total_cases,
            "passed": total_passed,
            "failed": total_cases - total_passed,
            "overall_score": round(overall_score, 4),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "total_latency_seconds": round(
                sum(r.total_latency_seconds for r in results), 2
            ),
        },
        "by_category": by_category_final,
        "cases": case_reports,
    }


def save_report(report: dict[str, Any], *, output_dir: Path | None = None) -> Path:
    """Save a JSON report to disk. Returns the file path."""
    out_dir = output_dir or _RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{report['run_id']}.json"
    path = out_dir / filename

    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    return path


def format_summary(report: dict[str, Any]) -> str:
    """Format a human-readable summary table from a report dict."""
    lines: list[str] = []

    config = report["config"]
    summary = report["summary"]

    lines.append("=== Anton Eval Results ===")
    lines.append(
        f"Run: {report['run_id']} | "
        f"Models: {config.get('planning_model', '?')} / {config.get('coding_model', '?')}"
    )
    lines.append("")

    # Header
    lines.append(
        f"{'Category':<22} {'Pass':>6} {'Total':>6} {'Score':>7} {'Tokens':>8} {'Cost':>8}"
    )
    lines.append("-" * 60)

    # Per-category rows
    for cat_name, cat in sorted(report["by_category"].items()):
        pass_str = f"{cat['passed']}/{cat['total']}"
        lines.append(
            f"{cat_name:<22} {pass_str:>6} {cat['total']:>6} "
            f"{cat['score']:>7.2f} {cat['tokens']:>8} ${cat['cost_usd']:>7.4f}"
        )

    lines.append("-" * 60)

    # Totals row
    pass_str = f"{summary['passed']}/{summary['total_cases']}"
    lines.append(
        f"{'TOTAL':<22} {pass_str:>6} {summary['total_cases']:>6} "
        f"{summary['overall_score']:>7.2f} {summary['total_tokens']:>8} "
        f"${summary['total_cost_usd']:>7.4f}"
    )
    lines.append("")

    # Failed cases
    failed = [c for c in report["cases"] if not c["passed"]]
    if failed:
        lines.append("Failed cases:")
        for c in failed:
            reason = c.get("error") or ""
            if not reason:
                # Find first failing scorer
                for s in c.get("scores", []):
                    if not s.get("passed"):
                        reason = f"{s['scorer_type']}: {s.get('detail', '')}"
                        break
            lines.append(
                f"  [FAIL] {c['category']}/{c['case_id']} "
                f"({c['overall_score']:.2f}) — {reason[:80]}"
            )
        lines.append("")

    lines.append(
        f"Time: {summary['total_latency_seconds']:.1f}s | "
        f"Avg: {summary['total_latency_seconds'] / max(summary['total_cases'], 1):.1f}s/case"
    )

    return "\n".join(lines)
