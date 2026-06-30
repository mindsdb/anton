"""Eval-case schema + loader (ENG-381).

A case is a self-contained YAML record describing one analytical task: what
Anton is given, the ground truth, and how to score the answer. The loader keeps
``reference`` and ``scoring`` as raw dicts on purpose — the scorers own their
own field contracts, so cases stay forgiving as we add dimensions.

See ``cases/reasoning-sales-dip-01.yaml`` for the canonical example and
``README.md`` for the field reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EvalCase:
    id: str
    title: str
    tier: int
    capabilities: list[str]
    dimensions: list[str]
    prompt: str
    reference: dict[str, Any]
    scoring: dict[str, Any]
    fixtures: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    overall_pass: str = "all declared dimensions pass"
    # Directory the case file lives in — fixtures resolve relative to it.
    source_dir: Path = field(default_factory=Path)

    @property
    def fixture_paths(self) -> list[Path]:
        return [(self.source_dir / "fixtures" / f) for f in self.fixtures]


_REQUIRED = ("id", "title", "tier", "prompt", "dimensions", "scoring")


def load_case(path: str | Path) -> EvalCase:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: case file must be a YAML mapping")
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"{path}: case missing required keys: {missing}")
    return EvalCase(
        id=str(raw["id"]),
        title=str(raw["title"]),
        tier=int(raw["tier"]),
        capabilities=list(raw.get("capabilities", [])),
        dimensions=list(raw["dimensions"]),
        prompt=str(raw["prompt"]),
        reference=dict(raw.get("reference", {})),
        scoring=dict(raw["scoring"]),
        fixtures=list(raw.get("fixtures", [])),
        environment=dict(raw.get("environment", {})),
        overall_pass=str(raw.get("overall_pass", "all declared dimensions pass")),
        source_dir=path.parent,
    )


def discover_cases(cases_dir: str | Path) -> list[Path]:
    """All ``*.yaml`` case files under ``cases_dir`` (non-recursive)."""
    return sorted(Path(cases_dir).glob("*.yaml"))
