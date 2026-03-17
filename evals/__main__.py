"""Standalone eval runner — ``python -m evals`` from the project root.

Usage:
    python -m evals                              # run all cases
    python -m evals --case fibonacci-20          # run one case
    python -m evals --category code_generation   # run one category
    python -m evals --tag fast                   # run cases with tag
    python -m evals --budget 0.50                # stop at $0.50
    python -m evals --list                       # list cases with cost estimates
    python -m evals --output results.json        # save report to custom path
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``import anton`` works
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from anton.llm.anthropic import AnthropicProvider  # noqa: E402
from anton.llm.client import LLMClient  # noqa: E402

from evals.loader import estimate_suite_cost, load_cases  # noqa: E402
from evals.report import build_report, format_summary, save_report  # noqa: E402
from evals.runner import EvalRunner  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Anton eval harness — run benchmark cases against live LLMs."
    )
    p.add_argument("--case", type=str, help="Run a single case by id")
    p.add_argument("--category", type=str, help="Run all cases in a category")
    p.add_argument("--tag", type=str, help="Run cases with this tag")
    p.add_argument("--budget", type=float, help="Budget cap in USD")
    p.add_argument("--output", type=str, help="Custom output path for JSON report")
    p.add_argument(
        "--list", action="store_true", dest="list_cases", help="List all cases and exit"
    )
    p.add_argument(
        "--planning-model",
        default="claude-sonnet-4-6",
        help="Planning model (default: claude-sonnet-4-6)",
    )
    p.add_argument(
        "--coding-model",
        default="claude-haiku-4-5-20251001",
        help="Coding model (default: claude-haiku-4-5-20251001)",
    )
    return p.parse_args()


def _list_cases(cases: list) -> None:
    """Print all cases with cost estimates."""
    print(f"\n{'ID':<35} {'Category':<22} {'Turns':>5} {'Est.$':>7}  Description")
    print("-" * 100)
    for c in cases:
        print(
            f"{c.id:<35} {c.category:<22} {len(c.turns):>5} "
            f"${c.estimated_cost_usd:>6.3f}  {c.description[:40]}"
        )
    total = estimate_suite_cost(cases)
    print("-" * 100)
    print(f"{'TOTAL':<35} {'':<22} {sum(len(c.turns) for c in cases):>5} ${total:>6.3f}")
    print(f"\n{len(cases)} cases, estimated cost: ${total:.3f}\n")


async def _run(args: argparse.Namespace) -> int:
    # Resolve API key
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
        "ANTON_ANTHROPIC_API_KEY"
    )
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY or ANTON_ANTHROPIC_API_KEY", file=sys.stderr)
        return 1

    # Load cases with filters
    filter_ids = [args.case] if args.case else None
    filter_categories = [args.category] if args.category else None
    filter_tags = [args.tag] if args.tag else None

    cases = load_cases(
        filter_ids=filter_ids,
        filter_categories=filter_categories,
        filter_tags=filter_tags,
    )

    if not cases:
        print("No matching cases found.", file=sys.stderr)
        return 1

    if args.list_cases:
        _list_cases(cases)
        return 0

    # Build LLM client
    provider = AnthropicProvider(api_key=api_key)
    llm_client = LLMClient(
        planning_provider=provider,
        planning_model=args.planning_model,
        coding_provider=provider,
        coding_model=args.coding_model,
    )

    runner = EvalRunner(
        llm_client,
        coding_provider="anthropic",
        coding_api_key=api_key,
    )

    # Show pre-run summary
    total_cost = estimate_suite_cost(cases)
    print(f"\nRunning {len(cases)} case(s), estimated cost: ${total_cost:.3f}")
    if args.budget:
        print(f"Budget cap: ${args.budget:.2f}")
    print()

    # Execute
    results = await runner.run_suite(cases, budget_usd=args.budget)

    # Build and display report
    report = build_report(
        results,
        planning_model=args.planning_model,
        coding_model=args.coding_model,
    )

    print(format_summary(report))

    # Save report
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {out_path}")
    else:
        path = save_report(report)
        print(f"\nReport saved to: {path}")

    # Exit code: 0 if all passed, 1 if any failed
    failed = report["summary"]["failed"]
    return 1 if failed > 0 else 0


def main() -> None:
    args = _parse_args()

    if args.list_cases:
        cases = load_cases(
            filter_ids=[args.case] if args.case else None,
            filter_categories=[args.category] if args.category else None,
            filter_tags=[args.tag] if args.tag else None,
        )
        _list_cases(cases)
        return

    exit_code = asyncio.run(_run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
