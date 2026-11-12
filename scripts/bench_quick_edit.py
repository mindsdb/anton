"""Benchmark: the artifact viewer's quick-edit bar (ENG-4227).

Two parts, both live against MindsHub:

* ``--routing``: :func:`route_request` on every message of
  ``tests/fixtures/artifact_edits/routing.json`` (labelled ``edit`` or
  ``agent``), in two variants: ``contract`` (message, files and kind only, as
  the shared contract has it) and ``context`` (plus the artifact's name and
  the shape of its embedded data, from :func:`describe_artifact`, and the bar's
  earlier messages). Accuracy, the confusion matrix, latency.
* ``--e2e``: the bar end to end on the routing set's messages that are the
  #790 benchmark's 28 edit tasks: ``route_request`` then, when routed to edit,
  ``run_quick_edit`` with the same checker the #790 benchmark's arm C used
  (static + the render-check service), on a fresh copy of the artifact. Each
  run is its own process (as in ``bench_artifact_edit.py``), with HOME and the
  workspace in a throwaway folder and its own render-check service.
* ``--escalation``: the safety net behind the router. ``run_quick_edit``
  directly (no routing, as if the router had said edit) on every message
  labelled ``agent``: does the editor hand it back (``escalate``) or answer
  (``noop``) rather than change the page?

Credentials come from the environment (``ANTON_MINDS_API_KEY``) and are never
written anywhere. Usage::

    ANTON_MINDS_API_KEY=... python scripts/bench_quick_edit.py --routing --runs 2 --out docs/benchmarks/raw-q
    ANTON_MINDS_API_KEY=... python scripts/bench_quick_edit.py --e2e --runs 2 --out docs/benchmarks/raw-q \\
        --render-check-url http://127.0.0.1:18801,http://127.0.0.1:18802
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from bench_artifact_edit import (  # noqa: E402 — the #790 harness's helpers, reused as is
    FIXTURES,
    _content_files,
    _final_render,
    _holds,
    _page_of,
    _pct,
    _usage_hook,
)

ROUTING = FIXTURES / "routing.json"
JEV_MODEL = "jev-1.13.0"
# The #790 benchmark (docs/benchmarks/artifact-edit-2026-11.md), mindshub_air, the same 28 tasks.
PR790 = {
    "A baseline turn (flag off)": (34.3, 50.5, "82/84"),
    "B agent turn + edit_artifact": (18.0, 27.9, "82/84"),
    "C editor direct": (9.8, 19.1, "83/84"),
}


def _base_url() -> str:
    return os.environ.get("ANTON_MINDS_URL", "https://api.mindshub.ai").rstrip("/") + "/v1"


def _meta(folder: Path) -> dict:
    return json.loads((folder / "metadata.json").read_text())


def _files(folder: Path, kind: str) -> list[str]:
    from anton.core.artifact_editor.workspace import ArtifactWorkspace

    return [entry.path for entry in ArtifactWorkspace(folder, kind=kind).map()]


async def _route(item: dict, folder: Path, *, variant: str, http=None):
    from anton.core.artifact_editor.router import describe_artifact, route_request

    kind = _meta(folder).get("type", "html-app")
    extra = {}
    if variant == "context":
        described = describe_artifact(folder)
        extra = {"name": described["name"], "data": described["data"], "history": item.get("history")}
    return await route_request(
        item["message"], files=_files(folder, kind), kind=kind, base_url=_base_url(),
        api_key=os.environ["ANTON_MINDS_API_KEY"], model=JEV_MODEL, http=http, **extra,
    )


# ── routing ───────────────────────────────────────────────────────────


async def run_routing(args) -> None:
    import httpx2 as httpx

    items = json.loads(ROUTING.read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    async with httpx.AsyncClient(timeout=10) as http:
        for run in range(1, args.runs + 1):
            for variant in args.variants.split(","):
                for item in items:
                    decision = await _route(item, FIXTURES / item["slug"], variant=variant, http=http)
                    row = {"id": item["id"], "label": item["label"], "variant": variant, "run": run,
                           "route": decision.route, "reason": decision.reason,
                           "confidence": decision.confidence, "ms": decision.ms}
                    rows.append(row)
                    flag = "ok" if row["route"] == row["label"] else "MISS"
                    print(f"{variant} r{run} {flag} {item['label']:>5}->{decision.route:<5} "
                          f"p={decision.confidence:.2f} {decision.reason} {decision.ms}ms  {item['message'][:70]}",
                          flush=True)
    (out / "routing.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(summarize_routing(rows, items, out))


def summarize_routing(rows: list[dict], items: list[dict], out: Path) -> str:
    messages = {i["id"]: i["message"] for i in items}
    lines = ["| Variant | N | Accuracy | edit→edit | edit→agent | agent→agent | agent→edit | Uncertain/error "
             "| Median ms | p90 ms |", "|---|---|---|---|---|---|---|---|---|---|"]
    misses: list[str] = []
    for variant in dict.fromkeys(r["variant"] for r in rows):
        group = [r for r in rows if r["variant"] == variant]
        right = sum(r["route"] == r["label"] for r in group)
        cell = {(a, b): sum(r["label"] == a and r["route"] == b for r in group)
                for a in ("edit", "agent") for b in ("edit", "agent")}
        soft = sum(r["reason"] != "jev" for r in group)
        ms = [r["ms"] for r in group]
        lines.append(
            f"| {variant} | {len(group)} | {right}/{len(group)} ({100 * right / len(group):.0f}%) | "
            f"{cell['edit', 'edit']} | {cell['edit', 'agent']} | {cell['agent', 'agent']} | {cell['agent', 'edit']} "
            f"| {soft} | {statistics.median(ms):.0f} | {_pct(ms, 0.9):.0f} |")
        for r in group:
            if r["route"] != r["label"]:
                misses.append(f"- {variant} run {r['run']}: \"{messages[r['id']]}\" labelled {r['label']}, "
                              f"routed {r['route']} ({r['reason']}, p={r['confidence']:.2f})")
    text = "\n".join(lines + ["", "Misrouted:", ""] + (misses or ["- none"]))
    (out / "routing-summary.md").write_text(text + "\n")
    return text


# ── end to end (one run per child process) ────────────────────────────


async def _e2e_job(job: dict) -> dict:
    from anton.config.settings import AntonSettings
    from anton.core.artifact_editor.checkers import select_checker
    from anton.core.artifact_editor.quick import quick_edit_llm_from_settings, run_quick_edit

    item, task = job["item"], job.get("task") or {"id": job["item"]["id"], "category": "agent", "assert": []}
    root = Path(job["root"])
    workspace = root / "ws"
    folder = workspace / ".anton" / "artifacts" / item["slug"]
    shutil.copytree(FIXTURES / item["slug"], folder)
    for seed in task.get("seed") or []:
        path = folder / seed["file"]
        text = path.read_text(encoding="utf-8")
        assert text.count(seed["find"]) == 1, (task["id"], seed["file"])
        path.write_text(text.replace(seed["find"], seed["replace"]), encoding="utf-8")
    meta = _meta(folder)
    kind = meta.get("type", "html-app")
    before = _content_files(folder)
    page = _page_of(folder)
    original_page = page.read_text(encoding="utf-8") if page.exists() else None

    started = time.monotonic()
    if job.get("force_edit"):
        row: dict = {"route": "edit", "route_reason": "forced", "route_p": 1.0, "route_ms": 0}
    else:
        decision = await _route(item, folder, variant=job["variant"])
        row = {"route": decision.route, "route_reason": decision.reason, "route_p": decision.confidence,
               "route_ms": decision.ms}
    if row["route"] == "edit":
        settings = AntonSettings()
        settings.resolve_workspace(str(workspace))
        llm = quick_edit_llm_from_settings(settings)
        calls: list[dict] = []
        _usage_hook(llm, calls, [time.monotonic()])
        stages: list[str] = []
        checker = select_checker(kind, slug=folder.name, previous_html=original_page)
        result = await run_quick_edit(folder, item["message"], kind=kind, llm=llm, checker=checker,
                                      history=item.get("history"), on_progress=lambda e: stages.append(e["stage"]))
        await llm.aclose()
        row.update({
            "status": result.status, "summary": result.summary, "escalate_reason": result.escalate_reason,
            "edit_ms": result.ms, "rounds": result.rounds, "llm_calls": result.llm_calls, "stages": stages,
            "check_status": result.check.status if result.check else "none", "error": result.error,
            "input_tokens": sum(c["input"] + c["cache_read"] + c["cache_creation"] for c in calls),
            "output_tokens": sum(c["output"] for c in calls),
        })
    else:
        row.update({"status": "agent", "edit_ms": 0, "rounds": 0, "llm_calls": 0, "check_status": "none"})
    row["total_ms"] = int((time.monotonic() - started) * 1000)
    after = _content_files(folder)
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    named = {a["file"] for a in task["assert"]} | {s["file"] for s in task.get("seed") or []}
    if kind == "html-app":
        named.add("presentation-program.js")
    texts: dict = {}
    passed = []
    for a in task["assert"]:
        path = folder / a["file"]
        if a["file"] not in texts:
            texts[a["file"]] = path.read_text(encoding="utf-8", errors="replace") if path.exists() else None
        passed.append(_holds(texts[a["file"]], a))
    row.update({"id": job["id"], "item_id": item["id"], "task_id": task["id"], "category": task["category"],
                "kind": kind, "run": job["run"], "variant": job["variant"], "assert_pass": all(passed),
                "changed": changed, "changed_outside": [p for p in changed if p not in named],
                "final_render": _final_render(folder, kind) if row["status"] == "applied" else "n/a"})
    return row


async def run_e2e(args, *, escalation: bool = False) -> None:
    if escalation:
        items = [i for i in json.loads(ROUTING.read_text()) if i["label"] == "agent"]
    else:
        items = [i for i in json.loads(ROUTING.read_text()) if i.get("task_id")]
    tasks = {t["id"]: t for t in json.loads((FIXTURES / "tasks.json").read_text())}
    if args.tasks:
        import re

        items = [i for i in items if re.search(args.tasks, i.get("task_id") or i["id"])]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / ("escalation.jsonl" if escalation else "e2e.jsonl")
    done = set()
    if results.exists():
        done = {json.loads(line)["id"] for line in results.read_text().splitlines()
                if "harness_error" not in json.loads(line)}
    jobs = [{"id": f"{i.get('task_id') or i['id']}.{run}", "item": i, "task": tasks.get(i.get("task_id")),
             "run": run, "variant": args.variant, "force_edit": escalation}
            for run in range(1, args.runs + 1) for i in items]
    jobs = [j for j in jobs if j["id"] not in done]
    print(f"{len(jobs)} runs to do ({len(done)} already done)", flush=True)
    scratch = Path(args.scratch or tempfile.mkdtemp(prefix="bench-quick-edit-"))
    home = scratch / "home"
    home.mkdir(parents=True, exist_ok=True)
    services: asyncio.Queue = asyncio.Queue()
    urls = [u.strip() for u in (args.render_check_url or "").split(",") if u.strip()]
    for url in urls:
        services.put_nowait(url)
    semaphore = asyncio.Semaphore(min(args.concurrency, len(urls)) if urls else args.concurrency)
    lock = asyncio.Lock()

    async def one(job: dict) -> None:
        async with semaphore:
            service = await services.get() if urls else None
            try:
                root = scratch / job["id"]
                if root.exists():
                    shutil.rmtree(root)
                root.mkdir(parents=True)
                env = {k: v for k, v in os.environ.items() if not k.startswith("MOCHI_")}
                env.update({
                    "HOME": str(home), "ANTON_ANALYTICS_ENABLED": "false",
                    "ANTON_MINDS_URL": env.get("ANTON_MINDS_URL", "https://api.mindshub.ai"),
                    "ANTON_PLANNING_PROVIDER": "minds-cloud", "ANTON_CODING_PROVIDER": "minds-cloud",
                    "ANTON_PLANNING_MODEL": args.model, "ANTON_CODING_MODEL": args.model,
                    "ANTON_ARTIFACT_EDITOR_LOG": str(root / "editor.jsonl"),
                })
                if service:
                    env["ANTON_RENDER_CHECK_URL"] = service
                spec = root / "job.json"
                spec.write_text(json.dumps({**job, "root": str(root)}))
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, str(Path(__file__).resolve()), "--job", str(spec), cwd=str(root), env=env,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                )
                stdout, _ = await proc.communicate()
            finally:
                if service is not None:
                    services.put_nowait(service)
            text = stdout.decode("utf-8", "replace")
            (root / "run.log").write_text(text)
            marker = text.rfind("RESULT:")
            if marker == -1:
                row = {"id": job["id"], "harness_error": text[-1500:]}
            else:
                row = json.loads(text[marker + len("RESULT:"):].strip().splitlines()[0])
            row["model"] = args.model
            async with lock:
                with open(results, "a") as handle:
                    handle.write(json.dumps(row) + "\n")
            print(f"{job['id']}: {row.get('route')}/{row.get('status')} total={row.get('total_ms')}ms "
                  f"route={row.get('route_ms')}ms calls={row.get('llm_calls')} rounds={row.get('rounds')} "
                  f"check={row.get('check_status')} render={row.get('final_render')} "
                  f"assert={row.get('assert_pass')}" + (" HARNESS ERROR" if "harness_error" in row else ""),
                  flush=True)

    await asyncio.gather(*(one(job) for job in jobs))
    print(summarize_escalation(results, out) if escalation else summarize_e2e(results, out))


def summarize_escalation(results: Path, out: Path) -> str:
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    good = [r for r in rows if "harness_error" not in r]
    messages = {i["id"]: i["message"] for i in json.loads(ROUTING.read_text())}
    counts = {s: sum(r["status"] == s for r in good) for s in ("escalate", "noop", "applied", "failed")}
    ms = [r["edit_ms"] / 1000 for r in good]
    lines = [f"Agent-labelled messages run straight through the editor: {len(good)} runs "
             f"({len(rows) - len(good)} harness errors excluded). " + ", ".join(f"{k} {v}" for k, v in counts.items())
             + f". Median {statistics.median(ms):.1f} s, p90 {_pct(ms, 0.9):.1f} s; "
             f"folders left unchanged when not applied: "
             f"{sum(not r['changed'] for r in good if r['status'] != 'applied')}/"
             f"{sum(r['status'] != 'applied' for r in good)}.", ""]
    lines += [f"- {r['status']}: \"{messages[r['item_id']]}\" → {(r.get('escalate_reason') or r.get('summary') or '')[:160]}"
              for r in sorted(good, key=lambda r: (r["status"], r["item_id"]))]
    text = "\n".join(lines)
    (out / "escalation-summary.md").write_text(text + "\n")
    return text


def summarize_e2e(results: Path, out: Path) -> str:
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    good = [r for r in rows if "harness_error" not in r]
    errors = len(rows) - len(good)
    applied = [r for r in good if r["status"] == "applied"]
    edited = [r for r in good if r["route"] == "edit"]
    total = [r["total_ms"] / 1000 for r in edited]
    route = [r["route_ms"] for r in good]
    edit = [r["edit_ms"] / 1000 for r in edited]
    statuses = {s: sum(r["status"] == s for r in good) for s in ("applied", "noop", "escalate", "failed", "agent")}
    html = [r for r in applied if r["final_render"] in ("passed", "failed", "unverified")]
    lines = [
        f"Runs: {len(good)} ({errors} harness errors excluded). Routed to edit: {len(edited)}/{len(good)}; "
        f"outcomes: " + ", ".join(f"{k} {v}" for k, v in statuses.items()) + ".",
        "",
        "| Path | N | Median (s) | p90 (s) | Assertions pass |",
        "|---|---|---|---|---|",
    ]
    for name, (median, p90, asserts) in PR790.items():
        lines.append(f"| #790 {name} | 84 | {median} | {p90} | {asserts} |")
    if edited:
        lines.append(f"| **Quick-edit bar: route + edit + check** | {len(edited)} | **{statistics.median(total):.1f}** | "
                     f"**{_pct(total, 0.9):.1f}** | {sum(r['assert_pass'] for r in edited)}/{len(edited)} |")
        lines += [
            "",
            f"Router: median {statistics.median(route):.0f} ms, p90 {_pct(route, 0.9):.0f} ms. "
            f"Editor + check: median {statistics.median(edit):.1f} s, p90 {_pct(edit, 0.9):.1f} s. "
            f"Median LLM calls {statistics.median(r['llm_calls'] for r in edited):g}; "
            f"fix rounds used: {sum(r['rounds'] > 0 for r in edited)}/{len(edited)} runs; "
            f"page renders after (html-app, applied): {sum(r['final_render'] == 'passed' for r in html)}/{len(html)}; "
            f"checked in a browser: {sum(r['check_status'] in ('passed', 'failed') for r in edited)}/{len(edited)}; "
            f"edited other files: {sum(bool(r['changed_outside']) for r in good)}/{len(good)}.",
        ]
        by = ["", "| Category | N | Median total (s) | Assertions pass |", "|---|---|---|---|"]
        for category in ("cosmetic", "content", "behaviour", "bugfix"):
            group = [r for r in edited if r["category"] == category]
            if group:
                by.append(f"| {category} | {len(group)} | {statistics.median(r['total_ms'] / 1000 for r in group):.1f}"
                          f" | {sum(r['assert_pass'] for r in group)}/{len(group)} |")
        lines += by
    odd = [r for r in good if r["status"] != "applied" or not r["assert_pass"]]
    if odd:
        lines += ["", "Not applied, or applied with a failed assertion:", ""]
        lines += [f"- {r['id']}: {r['status']} (route {r['route']}/{r['route_reason']}, assert {r['assert_pass']})"
                  + (f": {r.get('escalate_reason') or r.get('error') or r.get('summary') or ''}"[:200]
                     if r["status"] != "applied" else "") for r in odd]
    text = "\n".join(lines)
    (out / "e2e-summary.md").write_text(text + "\n")
    columns = ["task_id", "category", "run", "route", "route_reason", "route_p", "route_ms", "status", "edit_ms",
               "total_ms", "llm_calls", "rounds", "check_status", "final_render", "assert_pass"]
    csv = [",".join(columns)] + [",".join(str(r.get(c, "")) for c in columns)
                                 for r in sorted(good, key=lambda r: (r["task_id"], r["run"]))]
    (out / "e2e-runs.csv").write_text("\n".join(csv) + "\n")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", help=argparse.SUPPRESS)
    parser.add_argument("--routing", action="store_true")
    parser.add_argument("--e2e", action="store_true")
    parser.add_argument("--escalation", action="store_true")
    parser.add_argument("--summarize", action="store_true", help="re-summarize what is in --out")
    parser.add_argument("--variants", default="contract,context")
    parser.add_argument("--variant", default="context", help="the router variant the e2e runs use")
    parser.add_argument("--model", default="mindshub_air")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--tasks", help="regex over task ids (e2e)")
    parser.add_argument("--out", default=str(REPO / "docs" / "benchmarks" / "raw-quick-edit"))
    parser.add_argument("--scratch")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--render-check-url")
    args = parser.parse_args()
    if args.job:
        job = json.loads(Path(args.job).read_text())
        print("RESULT:" + json.dumps(asyncio.run(_e2e_job(job))), flush=True)
        return
    out = Path(args.out)
    if args.summarize:
        if (out / "routing.jsonl").exists():
            rows = [json.loads(line) for line in (out / "routing.jsonl").read_text().splitlines()]
            print(summarize_routing(rows, json.loads(ROUTING.read_text()), out))
        if (out / "e2e.jsonl").exists():
            print(summarize_e2e(out / "e2e.jsonl", out))
        if (out / "escalation.jsonl").exists():
            print(summarize_escalation(out / "escalation.jsonl", out))
        return
    if not os.environ.get("ANTON_MINDS_API_KEY"):
        raise SystemExit("set ANTON_MINDS_API_KEY")
    if args.routing:
        asyncio.run(run_routing(args))
    if args.e2e:
        asyncio.run(run_e2e(args))
    if args.escalation:
        asyncio.run(run_e2e(args, escalation=True))


if __name__ == "__main__":
    main()
