"""Eval runner (ENG-381).

Drives Anton end-to-end on one (or all) eval case(s) and scores the answer.

    uv run python -m evals.runner evals/cases/reasoning-sales-dip-01.yaml
    uv run python -m evals.runner --all
    uv run python -m evals.runner <case> --baseline   # also write results/baseline/

For each case it: builds a minds-cloud LLMClient pinned to ModelConfig, records
the *resolved* snapshot (drift detection — minds-cloud can't be pinned, see
models.py), stages the fixtures into a scratch workspace, runs
``ChatSession.turn()``, scores every declared dimension, and writes a JSON
result. Real LLM calls — this is an offline/manual quality suite, not CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    ModelConfig,
    UsageMeter,
    build_llm_client,
    load_minds_credentials,
    resolve_model,
)
from .scorers import (
    score_artifact_check,
    score_efficiency,
    score_fact_match,
    score_llm_judge,
)
from .spec import EvalCase, discover_cases, load_case

EVALS_DIR = Path(__file__).resolve().parent
CASES_DIR = EVALS_DIR / "cases"
RESULTS_DIR = EVALS_DIR / "results"


async def _run_turn(case: EvalCase, prompt: str, workspace_dir: Path, cfg: ModelConfig,
                    api_key: str, base_url: str, meter: UsageMeter | None = None) -> str:
    """Build a minimal ChatSession scoped to ``workspace_dir`` and run one turn."""
    from anton.config.settings import AntonSettings
    from anton.core.datasources.data_vault import LocalDataVault
    from anton.core.session import ChatSession, ChatSessionConfig
    from anton.workspace import Workspace

    settings = AntonSettings()
    settings.resolve_workspace(str(workspace_dir))
    workspace = Workspace(workspace_dir)
    workspace.initialize()
    workspace.apply_env_to_process()

    # Hermetic isolation: hand the session an EMPTY vault scoped to this
    # workspace. Without it, build_datasource_context() falls back to
    # `LocalDataVault()` = the operator's real ~/.anton/data_vault, which leaks
    # whoever-ran-it's connected datasources into the prompt — making cases
    # non-reproducible and tripping scope/honesty grading (a real `mysql-*` conn
    # reads as a fabrication to the environment-blind judge). See CAPABILITIES.md.
    iso_vault = LocalDataVault(vault_dir=workspace_dir / ".anton" / "data_vault")

    llm_client = build_llm_client(cfg, api_key, base_url, meter=meter)
    env = case.environment or {}
    web = bool(env.get("web", False))

    config = ChatSessionConfig(
        llm_client=llm_client,
        settings=settings,
        workspace=workspace,
        data_vault=iso_vault,
        web_search_enabled=web,
        web_fetch_enabled=web,
        session_id=f"eval-{case.id}",
        harness="eval",
    )
    session = ChatSession(config)
    return await session.turn(prompt)


def _score_case(case: EvalCase, answer: str, cfg: ModelConfig,
                api_key: str, base_url: str, artifacts_dir: Path | None = None,
                metrics: dict | None = None) -> dict:
    """Score every declared dimension via its method, pulling ground truth from
    ``case.reference``. ``artifacts_dir`` is the captured ``.anton/artifacts``
    directory (build cases grade the produced file, not the chat text);
    ``metrics`` is the measured turn cost (for the efficiency dimension)."""
    ref = case.reference
    dims: dict[str, dict] = {}
    for dim in case.dimensions:
        spec = case.scoring.get(dim, {})
        method = spec.get("method")
        if method == "fact_match":
            dims[dim] = score_fact_match(answer, ref.get("key_facts", []))
        elif method == "efficiency":
            dims[dim] = score_efficiency(metrics or {}, ref.get("efficiency", {}))
        elif method == "llm_judge":
            dims[dim] = score_llm_judge(
                task=case.prompt,
                answer=answer,
                ideal_reasoning=ref.get("ideal_reasoning", []),
                pass_bar_min=int(spec.get("pass_bar_min", 4)),
                cfg=cfg,
                base_url=base_url,
                api_key=api_key,
            )
        elif method == "artifact_check":
            dims[dim] = score_artifact_check(artifacts_dir, ref.get("artifact", {}))
        else:
            dims[dim] = {"method": method, "passed": False,
                         "detail": f"unknown scoring method: {method!r}"}
    return dims


def run_case(case_path: Path, cfg: ModelConfig, api_key: str, base_url: str) -> dict:
    case = load_case(case_path)
    print(f"\n=== {case.id} — {case.title}  (tier {case.tier}) ===")

    # Drift detection: snapshot what the moving aliases resolve to right now.
    resolved = {
        "planning": resolve_model(base_url, api_key, cfg.planning_model),
        "coding": resolve_model(base_url, api_key, cfg.coding_model),
        "judge": resolve_model(base_url, api_key, cfg.judge_model),
    }
    print(f"  models: planning={cfg.planning_model}→{resolved['planning']} "
          f"(effort={cfg.planning_effort})  judge={cfg.judge_model}→{resolved['judge']}")

    # Stage fixtures into an isolated scratch workspace.
    workspace_dir = Path(tempfile.mkdtemp(prefix=f"eval-{case.id}-"))
    staged = []
    for fp in case.fixture_paths:
        if not fp.exists():
            raise FileNotFoundError(f"fixture not found: {fp}")
        shutil.copy(fp, workspace_dir / fp.name)
        staged.append(workspace_dir / fp.name)

    # Hand the agent the absolute fixture paths (mirrors the cowork harness) and
    # pin the scratchpad cwd to the workspace.
    prompt = case.prompt
    if staged:
        listing = "\n".join(f"  - {p}" for p in staged)
        prompt = f"{prompt}\n\nFiles available in your working directory:\n{listing}"
    prev_cwd = Path.cwd()
    os.chdir(workspace_dir)

    started = time.time()
    error = None
    answer = ""
    artifacts_capture: Path | None = None
    meter = UsageMeter()
    try:
        answer = asyncio.run(_run_turn(case, prompt, workspace_dir, cfg, api_key, base_url, meter))
    except Exception as exc:  # noqa: BLE001 — record, don't crash the suite
        error = f"{type(exc).__name__}: {exc}"
        print(f"  !! turn failed: {error}")
    finally:
        # Capture any produced artifacts BEFORE tearing down the workspace —
        # build cases grade the file on disk, and the workspace is about to go.
        artifacts_src = workspace_dir / ".anton" / "artifacts"
        if artifacts_src.is_dir():
            artifacts_capture = Path(tempfile.mkdtemp(prefix=f"eval-art-{case.id}-")) / "artifacts"
            shutil.copytree(artifacts_src, artifacts_capture)
        os.chdir(prev_cwd)
        shutil.rmtree(workspace_dir, ignore_errors=True)
    elapsed = round(time.time() - started, 1)
    metrics = {**meter.as_dict(), "elapsed_seconds": elapsed}

    try:
        dims = {} if error else _score_case(
            case, answer, cfg, api_key, base_url,
            artifacts_dir=artifacts_capture, metrics=metrics)
    finally:
        if artifacts_capture is not None:
            shutil.rmtree(artifacts_capture.parent, ignore_errors=True)
    overall = bool(dims) and all(d.get("passed") for d in dims.values())

    for name, d in dims.items():
        mark = "PASS" if d.get("passed") else "FAIL"
        print(f"  [{mark}] {name}: {d.get('detail', '')}"
              + (f"  — {d['rationale']}" if d.get("rationale") else ""))
    print(f"  cost: {metrics['total_tokens']} tok / {metrics['llm_calls']} calls / {elapsed}s")
    print(f"  overall: {'PASS' if overall else 'FAIL'}")

    return {
        "case_id": case.id,
        "title": case.title,
        "tier": case.tier,
        "capabilities": case.capabilities,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_config": {
            "provider": cfg.provider,
            "base_url": cfg.base_url,
            "planning_model": cfg.planning_model,
            "coding_model": cfg.coding_model,
            "judge_model": cfg.judge_model,
            "planning_effort": cfg.planning_effort,
        },
        "resolved_models": resolved,  # drift detection
        "elapsed_seconds": elapsed,
        "efficiency": metrics,  # C11: measured turn cost (tokens/calls/seconds), recorded every run
        "error": error,
        "answer": answer,
        "dimensions": dims,
        "overall_pass": overall,
    }


def _write_result(result: dict, baseline: bool) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = result["timestamp"].replace(":", "").replace("-", "")[:15]
    out = RESULTS_DIR / f"{result['case_id']}__{stamp}.json"
    out.write_text(json.dumps(result, indent=2))
    if baseline:
        bdir = RESULTS_DIR / "baseline"
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / f"{result['case_id']}.json").write_text(json.dumps(result, indent=2))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Anton analytical evals (ENG-381)")
    ap.add_argument("case", nargs="?", help="path to a case .yaml (omit with --all)")
    ap.add_argument("--all", action="store_true", help="run every case under cases/")
    ap.add_argument("--baseline", action="store_true",
                    help="also write results/baseline/<case>.json")
    ap.add_argument("--effort", choices=["low", "medium", "high", "max"],
                    help="override the SUBJECT's planning effort (judge stays fixed "
                         "so grading is a constant yardstick). Default: ModelConfig (high).")
    args = ap.parse_args(argv)

    if not args.case and not args.all:
        ap.error("provide a case path or --all")

    base_url, api_key = load_minds_credentials()
    cfg = ModelConfig()
    if args.effort:
        # Vary only the subject's reasoning effort — the judge keeps its own
        # (high) effort so the measuring stick doesn't move between runs.
        cfg.planning_effort = args.effort

    case_paths = discover_cases(CASES_DIR) if args.all else [Path(args.case)]
    results = [run_case(p, cfg, api_key, base_url) for p in case_paths]
    for r in results:
        out = _write_result(r, args.baseline)
        print(f"  wrote {out.relative_to(EVALS_DIR.parent)}")

    passed = sum(r["overall_pass"] for r in results)
    print(f"\n==== {passed}/{len(results)} cases passed ====")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
