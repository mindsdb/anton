"""Benchmark: is changing an artifact faster with the artifact editor (ENG-4207)?

Runs the edit tasks in ``tests/fixtures/artifact_edits/tasks.json`` against a
fresh copy of their artifact, in three arms, with the same model:

* ``A`` (baseline): a real ChatSession turn with ``ANTON_ARTIFACT_EDITOR=off``,
  so the agent edits as staging does (scratchpad surgery, or a
  generate_artifact rebuild).
* ``B`` (new): a real ChatSession turn with ``ANTON_ARTIFACT_EDITOR=on``, so
  the agent can route the change to ``edit_artifact``.
* ``C`` (lower bound): :class:`ArtifactEditor` called directly with the task.

Each run is its own process (fresh settings, no shared state) with HOME and the
workspace in a throwaway folder, so nothing touches the real ``~/.anton``.

Per run: wall time, LLM calls (all roles), input / output / cached tokens, the
task's assertions on the resulting files, the check status the run reported,
an independent render check of the resulting page (same service for every
arm), and content files changed that the task does not name.

Credentials come from the environment (``ANTON_MINDS_API_KEY``) and are never
written anywhere. Usage::

    ANTON_MINDS_API_KEY=... python scripts/bench_artifact_edit.py \\
        --arms A,B,C --models mindshub_air --runs 3 --out docs/benchmarks/raw \\
        --render-check-url http://127.0.0.1:18765

    python scripts/bench_artifact_edit.py --summarize docs/benchmarks/raw/results.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "artifact_edits"
# Housekeeping and generation inputs: never what the user means by "the artifact".
IGNORED = {"metadata.json", "README.md", "prd.md", "spec.md", "openapi.json", "discovery.json",
           "observed-data.json", "backend.log"}


# ── one run (child process) ───────────────────────────────────────────


def _content_files(folder: Path) -> dict[str, str]:
    out = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(folder).as_posix()
        if rel.split("/")[0] in IGNORED or rel.startswith(".") or "/." in rel or "__pycache__" in rel:
            continue
        out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _holds(text: str | None, a: dict) -> bool:
    kind, value = a["type"], a["value"]
    if text is None:
        return kind in ("not_contains", "not_regex")
    if kind == "contains":
        return value in text
    if kind == "not_contains":
        return value not in text
    if kind == "regex":
        return re.search(value, text, re.S) is not None
    if kind == "not_regex":
        return re.search(value, text, re.S) is None
    raise ValueError(kind)


def _page_of(folder: Path) -> Path:
    return folder / "index.html" if (folder / "index.html").exists() else folder / "static" / "index.html"


def _usage_hook(llm, calls: list[dict], started: list[float]):
    original = llm._notify_usage

    def notify(role, model, usage, listener=None):
        calls.append({"role": role, "model": model, "input": usage.input_tokens, "output": usage.output_tokens,
                      "cache_read": usage.cache_read_tokens, "cache_creation": usage.cache_creation_tokens,
                      "t": round(time.monotonic() - started[0], 2)})
        return original(role, model, usage, listener)

    llm._notify_usage = notify


async def _session_turn(workspace: Path, prompt: str, *, session_id: str) -> dict:
    from anton.chat_session import build_runtime_context
    from anton.config.settings import AntonSettings
    from anton.core.llm.client import LLMClient
    from anton.core.llm.provider import StreamTaskProgress, StreamTextDelta, StreamToolUseStart
    from anton.core.session import ChatSession, ChatSessionConfig, SystemPromptContext
    from anton.workspace import Workspace

    settings = AntonSettings()
    settings.resolve_workspace(str(workspace))
    ws = Workspace(workspace)
    ws.initialize()
    llm = LLMClient.from_settings(settings)
    session = ChatSession(ChatSessionConfig(
        llm_client=llm, settings=settings,
        system_prompt_context=SystemPromptContext(runtime_context=build_runtime_context(settings)),
        output_dir=str(settings.artifacts_dir), workspace=ws, session_id=session_id, harness="anton",
        act_first=True, proactive_dashboards=settings.proactive_dashboards, background_memory=False,
    ))
    calls: list[dict] = []
    started = [time.monotonic()]
    _usage_hook(llm, calls, started)
    tools, text, progress, error = [], [], [], None
    try:
        async def consume():
            async for event in session.turn_stream(prompt):
                if isinstance(event, StreamToolUseStart):
                    tools.append(event.name)
                elif isinstance(event, StreamTextDelta):
                    text.append(event.text)
                elif isinstance(event, StreamTaskProgress) and event.message and len(progress) < 300:
                    progress.append(f"{event.phase}: {event.message[:160]}")

        await asyncio.wait_for(consume(), timeout=1500)
    except Exception as exc:  # noqa: BLE001 — the run's error, recorded
        error = f"{type(exc).__name__}: {exc}"[:1000]
    wall = time.monotonic() - started[0]
    checks = {f"{k.slug}/{k.page}": v for k, v in (getattr(session, "_page_checks", {}) or {}).items()}
    try:
        await session.close()
    except Exception:  # noqa: BLE001
        pass
    return {"wall_s": wall, "calls": calls, "tools": tools, "reply": "".join(text)[-3000:], "progress": progress,
            "error": error, "page_checks": checks, "models": {"planning": settings.planning_model,
                                                              "coding": settings.coding_model}}


async def _direct_edit(workspace: Path, folder: Path, task: str, original_page: str | None) -> dict:
    from anton.config.settings import AntonSettings
    from anton.core.artifact_editor.checkers import select_checker
    from anton.core.artifact_editor.editor import ArtifactEditor
    from anton.core.artifact_editor.workspace import ArtifactWorkspace
    from anton.core.llm.client import LLMClient

    settings = AntonSettings()
    settings.resolve_workspace(str(workspace))
    llm = LLMClient.from_settings(settings)
    meta = json.loads((folder / "metadata.json").read_text())
    kind = meta.get("type", "html-app")
    primary = meta.get("primary") if meta.get("primary") and (folder / meta["primary"]).is_file() else None
    ws = ArtifactWorkspace(folder, kind=kind, primary=primary)
    checker = select_checker(kind, slug=folder.name, previous_html=original_page)
    calls: list[dict] = []
    started = [time.monotonic()]
    _usage_hook(llm, calls, started)

    class Trail:
        lines: list[str] = []

        def status(self, text):
            pass

        def log(self, text):
            self.lines.append(text)

    trail = Trail()
    role = os.environ.get("ANTON_ARTIFACT_EDITOR_ROLE", "planning")
    editor = ArtifactEditor(ws, llm, checker, trail, role=role, slug=folder.name, name=meta.get("name", ""))
    error = None
    try:
        outcome = await editor.edit(task)
    except Exception as exc:  # noqa: BLE001
        outcome, error = None, f"{type(exc).__name__}: {exc}"
    wall = time.monotonic() - started[0]
    await llm.aclose()
    return {
        "wall_s": wall, "calls": calls, "tools": ["edit_artifact(direct)"], "progress": trail.lines[:200],
        "reply": outcome.reply if outcome else "", "error": error or (outcome.error if outcome else None),
        "editor_status": outcome.status if outcome else None, "rewrote": outcome.rewrote if outcome else None,
        "attempts": outcome.attempts if outcome else None,
        "page_checks": {f"{folder.name}/{ws.primary}": outcome.check.status} if outcome and outcome.check else {},
        "models": {"planning": settings.planning_model, "coding": settings.coding_model},
    }


def _final_render(folder: Path, kind: str) -> str:
    """An independent render check of the resulting page, the same for every arm."""
    if kind != "html-app" or not os.environ.get("ANTON_RENDER_CHECK_URL"):
        return "n/a"
    from anton.core.artifacts import render_check

    page = _page_of(folder)
    if not page.exists():
        return "missing"
    html = page.read_text(encoding="utf-8", errors="replace")
    for _ in range(3):
        outcome = render_check.check_html(html, screenshot=False, timeout=60)
        if outcome.status != "unverified" or "busy" not in outcome.unverified_reason:
            break
        time.sleep(5)
    return outcome.status


def run_job(job: dict) -> dict:
    """One task, one arm, one model, one fresh copy of the artifact."""
    task = job["task"]
    root = Path(job["root"])
    workspace = root / "ws"
    artifacts = workspace / ".anton" / "artifacts"
    folder = artifacts / task["slug"]
    shutil.copytree(FIXTURES / task["slug"], folder)
    for seed in task.get("seed") or []:
        path = folder / seed["file"]
        text = path.read_text(encoding="utf-8")
        assert text.count(seed["find"]) == 1, (task["id"], seed["file"])
        path.write_text(text.replace(seed["find"], seed["replace"]), encoding="utf-8")
    meta = json.loads((folder / "metadata.json").read_text())
    kind = meta.get("type", "html-app")
    before = _content_files(folder)
    page = _page_of(folder)
    original_page = page.read_text(encoding="utf-8") if page.exists() else None

    if job["arm"] in ("A", "B"):
        result = asyncio.run(_session_turn(workspace, task["task"], session_id=f"bench-{job['id']}"))
    else:
        result = asyncio.run(_direct_edit(workspace, folder, task["task"], original_page))

    after = _content_files(folder) if folder.exists() else {}
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    named = {a["file"] for a in task["assert"]} | {s["file"] for s in task.get("seed") or []}
    if kind == "html-app":
        named.add("presentation-program.js")  # kept in step with a component page's program
    outside = [p for p in changed if p not in named]
    texts = {}
    passed = []
    for a in task["assert"]:
        path = folder / a["file"]
        if a["file"] not in texts:
            texts[a["file"]] = path.read_text(encoding="utf-8", errors="replace") if path.exists() else None
        passed.append(_holds(texts[a["file"]], a))
    calls = result.pop("calls")
    statuses = list(result.get("page_checks", {}).values())
    worst = ("failed" if "failed" in statuses else "unverified" if "unverified" in statuses
             else "passed" if statuses else "none")
    return {
        **{k: job[k] for k in ("id", "arm", "model", "run")}, "task_id": task["id"], "category": task["category"],
        "kind": kind, "wall_s": round(result.pop("wall_s"), 2),
        "llm_calls": len(calls), "llm_calls_by_role": _count(c["role"] for c in calls),
        "input_tokens": sum(c["input"] for c in calls), "output_tokens": sum(c["output"] for c in calls),
        "cache_read_tokens": sum(c["cache_read"] for c in calls),
        "cache_creation_tokens": sum(c["cache_creation"] for c in calls),
        "assert_pass": all(passed), "asserts": passed, "check_status": worst,
        "final_render": _final_render(folder, kind), "changed": changed, "changed_outside": outside,
        "edit_artifact_used": "edit_artifact" in " ".join(result.get("tools", [])),
        **result, "call_log": calls,
    }


def _count(items) -> dict:
    out: dict = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


# ── the matrix (parent process) ───────────────────────────────────────


def _jobs(args) -> list[dict]:
    tasks = json.loads((FIXTURES / "tasks.json").read_text())
    if args.tasks:
        wanted = re.compile(args.tasks)
        tasks = [t for t in tasks if wanted.search(t["id"])]
    jobs = []
    # Interleaved (run, task, arm), so the provider's load at any moment falls on every arm alike.
    for model in args.models.split(","):
        for run in range(1, args.runs + 1):
            for task in tasks:
                for arm in args.arms.split(","):
                    jobs.append({"id": f"{model}.{arm}.{task['id']}.{run}", "arm": arm, "model": model, "run": run,
                                 "task": task})
    return jobs


async def _run_matrix(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            row = json.loads(line)
            if not row.get("harness_error"):
                done.add(row["id"])
    jobs = [j for j in _jobs(args) if j["id"] not in done]
    print(f"{len(jobs)} runs to do ({len(done)} already done)", flush=True)
    scratch = Path(args.scratch or tempfile.mkdtemp(prefix="bench-artifact-edit-"))
    home = scratch / "home"
    home.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    # One render-check service per running job: the service checks one page at a
    # time, and a queue behind it would turn checks into "unverified" by timeout.
    services: asyncio.Queue = asyncio.Queue()
    for url in (args.render_check_url or "").split(","):
        if url.strip():
            services.put_nowait(url.strip())

    async def one(job: dict) -> None:
        async with semaphore:
            service = await services.get() if args.render_check_url else None
            try:
                await run_one(job, service)
            finally:
                if service is not None:
                    services.put_nowait(service)

    async def run_one(job: dict, service: str | None) -> None:
        if True:
            root = scratch / job["id"]
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True)
            env = {k: v for k, v in os.environ.items() if not k.startswith("MOCHI_")}
            env.update({
                "HOME": str(home), "ANTON_ANALYTICS_ENABLED": "false",
                "ANTON_MINDS_URL": env.get("ANTON_MINDS_URL", "https://api.mindshub.ai"),
                "ANTON_PLANNING_PROVIDER": "minds-cloud", "ANTON_CODING_PROVIDER": "minds-cloud",
                "ANTON_PLANNING_MODEL": job["model"], "ANTON_CODING_MODEL": job["model"],
                "ANTON_ARTIFACT_EDITOR": "off" if job["arm"] == "A" else "on",
                "ANTON_ARTIFACT_EDITOR_LOG": str(root / "editor.jsonl"),
            })
            if service:
                env["ANTON_RENDER_CHECK_URL"] = service
            spec = root / "job.json"
            spec.write_text(json.dumps({**job, "root": str(root)}))
            started = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(Path(__file__).resolve()), "--job", str(spec), cwd=str(root), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            text = stdout.decode("utf-8", "replace")
            (root / "run.log").write_text(text)
            marker = text.rfind("RESULT:")
            if marker == -1:
                row = {"id": job["id"], "arm": job["arm"], "model": job["model"], "run": job["run"],
                       "task_id": job["task"]["id"], "harness_error": text[-1500:]}
            else:
                row = json.loads(text[marker + len("RESULT:"):].strip().splitlines()[0])
            async with lock:
                with open(results, "a") as handle:
                    handle.write(json.dumps(row) + "\n")
            flag = "ok" if row.get("assert_pass") else "FAIL" if "assert_pass" in row else "ERROR"
            print(f"{job['id']}: {flag} {row.get('wall_s', '?')}s calls={row.get('llm_calls', '?')} "
                  f"check={row.get('check_status')} render={row.get('final_render')} "
                  f"({time.monotonic() - started:.0f}s)", flush=True)

    await asyncio.gather(*(one(job) for job in jobs))
    summarize(results, out)


# ── the summary ───────────────────────────────────────────────────────


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


ARM_NAMES = {"A": "A baseline (flag off)", "B": "B edit_artifact (flag on)", "C": "C editor direct"}


def summarize(results: Path, out: Path) -> str:
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    good = [r for r in rows if "assert_pass" in r]
    errors = [r for r in rows if "harness_error" in r]
    lines = ["| Model | Arm | N | Median time (s) | p90 time (s) | Median LLM calls | Median input tok (incl. cached) "
             "| Median output tok | Assertions pass | Page renders after | Checked in browser | Edited other files |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    table = {}
    for model in sorted({r["model"] for r in good}):
        for arm in ("A", "B", "C"):
            group = [r for r in good if r["model"] == model and r["arm"] == arm]
            if not group:
                continue
            walls = [r["wall_s"] for r in group]
            calls = [r["llm_calls"] for r in group]
            prompt = [r["input_tokens"] + r["cache_read_tokens"] + r["cache_creation_tokens"] for r in group]
            output = [r["output_tokens"] for r in group]
            html = [r for r in group if r["final_render"] in ("passed", "failed", "unverified")]
            renders = sum(r["final_render"] == "passed" for r in html)
            browser = sum(r["check_status"] in ("passed", "failed") for r in group)
            outside = sum(bool(r["changed_outside"]) for r in group)
            stats = {
                "n": len(group), "median_s": statistics.median(walls), "p90_s": _pct(walls, 0.9),
                "median_calls": statistics.median(calls), "median_prompt_tokens": statistics.median(prompt),
                "median_output_tokens": statistics.median(output),
                "assert_pass": sum(r["assert_pass"] for r in group), "renders": renders, "render_n": len(html),
                "browser_checked": browser, "changed_outside": outside,
            }
            table[f"{model}/{arm}"] = stats
            lines.append(
                f"| {model} | {ARM_NAMES[arm]} | {stats['n']} | {stats['median_s']:.1f} | {stats['p90_s']:.1f} | "
                f"{stats['median_calls']:g} | {stats['median_prompt_tokens']:,.0f} | {stats['median_output_tokens']:,.0f} "
                f"| {stats['assert_pass']}/{stats['n']} | {renders}/{len(html)} | {browser}/{stats['n']} "
                f"| {outside}/{stats['n']} |")
    by_category = ["", "| Model | Arm | Category | N | Median time (s) | Median calls | Assertions pass |",
                   "|---|---|---|---|---|---|---|"]
    for model in sorted({r["model"] for r in good}):
        for arm in ("A", "B", "C"):
            for category in ("cosmetic", "content", "behaviour", "bugfix"):
                group = [r for r in good if r["model"] == model and r["arm"] == arm and r["category"] == category]
                if group:
                    by_category.append(
                        f"| {model} | {arm} | {category} | {len(group)} | "
                        f"{statistics.median(r['wall_s'] for r in group):.1f} | "
                        f"{statistics.median(r['llm_calls'] for r in group):g} | "
                        f"{sum(r['assert_pass'] for r in group)}/{len(group)} |")
    paths = ["", "What the baseline did (tools the agent called, arm A):", ""]
    for model in sorted({r["model"] for r in good}):
        group = [r for r in good if r["model"] == model and r["arm"] == "A"]
        used = _count("generate_artifact" if "generate_artifact" in r.get("tools", []) else
                      "scratchpad" if "scratchpad" in r.get("tools", []) else "other" for r in group)
        paths.append(f"- {model}: {used}")
        group_b = [r for r in good if r["model"] == model and r["arm"] == "B"]
        if group_b:
            paths.append(f"- {model}, arm B: edit_artifact used in {sum(r['edit_artifact_used'] for r in group_b)}"
                         f"/{len(group_b)} runs")
    text = "\n".join(lines + by_category + paths)
    if errors:
        text += f"\n\n{len(errors)} runs failed in the harness and are excluded."
    (out / "summary.md").write_text(text + "\n")
    # One row per run, without transcripts: the numbers the tables come from.
    columns = ["model", "arm", "task_id", "category", "run", "wall_s", "llm_calls", "input_tokens", "output_tokens",
               "cache_read_tokens", "cache_creation_tokens", "assert_pass", "check_status", "final_render",
               "edit_artifact_used", "path"]
    csv_lines = [",".join(columns)]
    for r in sorted(good, key=lambda r: (r["model"], r["task_id"], r["arm"], r["run"])):
        path = ("edit_artifact" if r["edit_artifact_used"] else "generate_artifact"
                if "generate_artifact" in r.get("tools", []) else "scratchpad" if "scratchpad" in r.get("tools", [])
                else "other")
        csv_lines.append(",".join(str({**r, "path": path}[c]) for c in columns))
    (out / "runs.csv").write_text("\n".join(csv_lines) + "\n")
    (out / "summary.json").write_text(json.dumps(table, indent=2))
    print(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", help=argparse.SUPPRESS)
    parser.add_argument("--arms", default="A,B,C")
    parser.add_argument("--models", default="mindshub_air")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tasks", help="regex over task ids")
    parser.add_argument("--out", default=str(REPO / "docs" / "benchmarks" / "raw"))
    parser.add_argument("--scratch", help="where the throwaway workspaces go")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--render-check-url", help="one or more render-check services, comma separated")
    parser.add_argument("--summarize", help="a results.jsonl to summarize")
    args = parser.parse_args()
    if args.job:
        sys.path.insert(0, str(REPO))
        job = json.loads(Path(args.job).read_text())
        print("RESULT:" + json.dumps(run_job(job)), flush=True)
        return
    if args.summarize:
        path = Path(args.summarize)
        summarize(path, path.parent)
        return
    if not os.environ.get("ANTON_MINDS_API_KEY"):
        raise SystemExit("set ANTON_MINDS_API_KEY")
    asyncio.run(_run_matrix(args))


if __name__ == "__main__":
    main()
