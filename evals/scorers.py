"""Scorers (ENG-381): hybrid grading of an Anton answer.

Two methods, one per scoring dimension declared in a case:

- ``fact_match`` (deterministic): each ``key_fact`` is a case-insensitive regex
  that MUST appear in the answer. Catches wrong/absent numbers and entities —
  cheap and perfectly repeatable.
- ``llm_judge`` (model-as-judge): an LLM scores the answer 1–5 against the
  case's ``ideal_reasoning`` anchors. Catches shallow-but-plausible reasoning,
  which is exactly what ENG-380 is about and what a substring check can't see.

A dimension passes per its ``pass_bar``. Each scorer returns a uniform dict so
the runner can aggregate without special-casing.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .models import ModelConfig, chat_completion


def score_fact_match(answer: str, key_facts: list[str]) -> dict[str, Any]:
    """Every pattern in ``key_facts`` must match (case-insensitive regex)."""
    text = answer or ""
    checks = []
    for pat in key_facts:
        try:
            hit = re.search(pat, text, re.IGNORECASE | re.DOTALL) is not None
        except re.error:
            # A malformed pattern degrades to a plain substring search rather
            # than crashing the whole eval.
            hit = pat.lower() in text.lower()
        checks.append({"pattern": pat, "matched": hit})
    matched = sum(c["matched"] for c in checks)
    return {
        "method": "fact_match",
        "passed": matched == len(checks) and checks != [],
        "detail": f"{matched}/{len(checks)} facts present",
        "checks": checks,
    }


_JUDGE_SYSTEM = (
    "You are a strict evaluator of an AI data analyst's answer. You are given the "
    "user's task, a list of reasoning qualities an EXCELLENT answer would show, "
    "and the answer to grade. Score how well the answer demonstrates those "
    "qualities on a 1-5 integer scale:\n"
    "  5 = shows all the listed qualities; insightful and correct.\n"
    "  4 = shows most; minor gaps.\n"
    "  3 = partial; misses an important quality.\n"
    "  2 = shallow; addresses the surface only.\n"
    "  1 = wrong or generic; misses the point.\n"
    "Judge ONLY reasoning quality against the listed qualities — do not reward "
    "length or formatting. Respond with STRICT JSON only, no prose, no code "
    'fences: {"score": <1-5>, "rationale": "<one or two sentences>"}.'
)


def _extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object from a model response (handles stray
    fences / prose around it)."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


def score_llm_judge(
    *,
    task: str,
    answer: str,
    ideal_reasoning: list[str],
    pass_bar_min: int,
    cfg: ModelConfig,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    """LLM judge scores 1–5 against ``ideal_reasoning``; passes at ``>= pass_bar_min``."""
    anchors = "\n".join(f"  - {a}" for a in ideal_reasoning)
    user = (
        f"## User task\n{task}\n\n"
        f"## Qualities an excellent answer shows\n{anchors}\n\n"
        f"## Answer to grade\n{answer}\n"
    )
    try:
        raw, resolved = chat_completion(
            base_url,
            api_key,
            cfg.judge_model,
            system=_JUDGE_SYSTEM,
            user=user,
            max_tokens=512,
            effort=cfg.judge_effort,
        )
    except Exception as exc:  # noqa: BLE001 — a judge error fails the dim, not the run
        return {
            "method": "llm_judge",
            "passed": False,
            "score": None,
            "pass_bar_min": pass_bar_min,
            "rationale": "",
            "detail": f"judge call failed: {type(exc).__name__}: {exc}",
        }
    parsed = _extract_json(raw)
    score = parsed.get("score")
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = None
    passed = score is not None and score >= pass_bar_min
    return {
        "method": "llm_judge",
        "passed": passed,
        "score": score,
        "pass_bar_min": pass_bar_min,
        "rationale": parsed.get("rationale", ""),
        "judge_model": cfg.judge_model,
        "judge_model_resolved": resolved,
        "detail": f"score={score} (need >= {pass_bar_min})",
        "raw": raw if score is None else None,  # keep raw only when parse failed
    }
