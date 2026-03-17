"""Load and validate benchmark cases from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from evals.types import EvalCase, Scorer, TurnSpec

_CASES_DIR = Path(__file__).parent / "cases"


def _parse_case(raw: dict) -> EvalCase:
    """Parse a single raw YAML dict into an EvalCase."""
    turns = [
        TurnSpec(
            user_input=t["user_input"],
            expect_tool_calls=t.get("expect_tool_calls"),
        )
        for t in raw["turns"]
    ]
    scorers = [
        Scorer(
            type=s["type"],
            value=s["value"],
            weight=s.get("weight", 1.0),
            expected_count=s.get("expected_count"),
        )
        for s in raw["scorers"]
    ]
    return EvalCase(
        id=raw["id"],
        category=raw["category"],
        description=raw["description"],
        turns=turns,
        scorers=scorers,
        tags=raw.get("tags", []),
        estimated_cost_usd=raw.get("estimated_cost_usd", 0.0),
        temperature=raw.get("temperature", 0.0),
    )


def load_cases(
    *,
    cases_dir: Path | None = None,
    filter_ids: list[str] | None = None,
    filter_categories: list[str] | None = None,
    filter_tags: list[str] | None = None,
) -> list[EvalCase]:
    """Load all YAML case files, optionally filtered.

    Args:
        cases_dir: Directory containing YAML files. Defaults to evals/cases/.
        filter_ids: If set, only return cases whose id is in this list.
        filter_categories: If set, only return cases whose category is in this list.
        filter_tags: If set, only return cases that have at least one matching tag.

    Returns:
        List of EvalCase objects, sorted by category then id.
    """
    root = cases_dir or _CASES_DIR
    all_cases: list[EvalCase] = []

    for yaml_file in sorted(root.glob("*.yaml")):
        with open(yaml_file) as f:
            raw_list = yaml.safe_load(f)
        if not isinstance(raw_list, list):
            continue
        for item in raw_list:
            try:
                case = _parse_case(item)
                all_cases.append(case)
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"Invalid case in {yaml_file.name}: {exc}"
                ) from exc

    # Apply filters
    if filter_ids:
        ids = set(filter_ids)
        all_cases = [c for c in all_cases if c.id in ids]
    if filter_categories:
        cats = set(filter_categories)
        all_cases = [c for c in all_cases if c.category in cats]
    if filter_tags:
        tags = set(filter_tags)
        all_cases = [c for c in all_cases if tags & set(c.tags)]

    # Validate uniqueness
    seen_ids: set[str] = set()
    for case in all_cases:
        if case.id in seen_ids:
            raise ValueError(f"Duplicate case id: {case.id}")
        seen_ids.add(case.id)

    return sorted(all_cases, key=lambda c: (c.category, c.id))


def estimate_suite_cost(cases: list[EvalCase]) -> float:
    """Return the total estimated cost in USD for a list of cases."""
    return sum(c.estimated_cost_usd for c in cases)
