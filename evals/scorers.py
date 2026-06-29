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
from pathlib import Path
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


# Chart-ish markers — a revenue dashboard that claims a chart should carry one
# of these. Deliberately broad (any common charting approach counts).
_CHART_MARKERS = re.compile(
    r"<canvas|<svg|chart\.js|new\s+Chart\s*\(|plotly|echarts|\bd3\b|"
    r"highcharts|apexcharts|<script[^>]*chart",
    re.IGNORECASE,
)


def _resolve_primary_html(folder: Path, primary: str | None) -> Path | None:
    """Find the artifact's entry HTML on DISK (not via metadata.files[], which is
    reconciled only on store read and can be stale — ENG-372). Mirrors the
    renderer heuristic: declared primary → index.html → newest .html."""
    if primary:
        p = folder / primary
        if p.is_file() and p.suffix.lower() in (".html", ".htm"):
            return p
    htmls = sorted(folder.rglob("*.html")) + sorted(folder.rglob("*.htm"))
    if not htmls:
        return None
    for h in htmls:
        if h.name.lower() == "index.html":
            return h
    return max(htmls, key=lambda h: h.stat().st_mtime)


def score_artifact_check(artifacts_dir: Path | None, spec: dict[str, Any]) -> dict[str, Any]:
    """Grade a built artifact ON DISK — never the chat text — so a case can't pass
    by *claiming* it built something (the C12 failure mode).

    ``artifacts_dir`` is the captured ``.anton/artifacts`` directory (the runner
    copies it out before tearing down the workspace). ``spec`` (case
    ``reference.artifact``) declares: ``type`` (default ``html-app``),
    ``require_chart`` (bool), and ``must_contain`` (regexes that must appear in
    the entry HTML — the figures/labels the dashboard should render).

    Offline only: inspects files, does NOT publish to 4nton.ai.
    """
    want_type = spec.get("type", "html-app")
    must_contain = spec.get("must_contain", [])
    require_chart = bool(spec.get("require_chart", False))
    checks: list[dict[str, Any]] = []

    def _fail(detail: str) -> dict[str, Any]:
        return {"method": "artifact_check", "passed": False, "detail": detail, "checks": checks}

    if not artifacts_dir or not Path(artifacts_dir).is_dir():
        return _fail("no artifacts produced (no .anton/artifacts directory)")
    artifacts_dir = Path(artifacts_dir)

    # Find an artifact folder of the wanted type (read metadata.json per folder).
    match_folder: Path | None = None
    match_primary: str | None = None
    seen_types: list[str] = []
    for meta_path in sorted(artifacts_dir.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        seen_types.append(meta.get("type", "?"))
        if meta.get("type") == want_type:
            match_folder = meta_path.parent
            match_primary = meta.get("primary")
            break

    checks.append({"check": f"artifact type {want_type!r} exists",
                   "passed": match_folder is not None,
                   "detail": f"types found: {seen_types or 'none'}"})
    if match_folder is None:
        return _fail(f"no {want_type!r} artifact (found: {seen_types or 'none'})")

    html = _resolve_primary_html(match_folder, match_primary)
    checks.append({"check": "entry HTML file on disk", "passed": html is not None,
                   "detail": (str(html.relative_to(artifacts_dir)) if html else "none")})
    if html is None:
        return _fail("html-app artifact has no entry .html file on disk")

    content = html.read_text(encoding="utf-8", errors="replace")
    low = content.lower()

    complete = ("<html" in low or "<!doctype" in low) and "</html>" in low and len(content) >= 200
    checks.append({"check": "self-contained HTML document", "passed": complete,
                   "detail": f"{len(content)} bytes, has <html>/</html>"})

    if require_chart:
        has_chart = _CHART_MARKERS.search(content) is not None
        checks.append({"check": "renders a chart", "passed": has_chart,
                       "detail": "chart marker present" if has_chart else "no chart markup found"})

    for pat in must_contain:
        try:
            hit = re.search(pat, content, re.IGNORECASE | re.DOTALL) is not None
        except re.error:
            hit = pat.lower() in low
        checks.append({"check": f"contains /{pat}/", "passed": hit, "detail": ""})

    passed = all(c["passed"] for c in checks)
    npass = sum(c["passed"] for c in checks)
    return {
        "method": "artifact_check",
        "passed": passed,
        "detail": f"{npass}/{len(checks)} checks passed — {match_folder.name}/{html.name}",
        "artifact": match_folder.name,
        "primary": html.name,
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
