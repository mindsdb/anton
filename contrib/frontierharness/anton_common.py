"""Anton adapter logic shared by the Harbor and Pier agent classes; run-task.sh sets its env."""

from __future__ import annotations

import json
import os
import shlex
import uuid
from pathlib import Path

# The benchmark fixes the model; the MindsHub alias serves Fireworks Kimi K3.
BENCHMARK_MODEL = "kimi-k3"

EVENTS_FILE = "anton-events.jsonl"
STDERR_FILE = "anton-stderr.log"
USAGE_PREFIX = "llm_usage "


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set; run-task.sh sets it")
    return value


def anton_ref() -> str:
    return _required_env("ANTON_FH_REF")


def base_url() -> str:
    return _required_env("ANTON_FH_BASE_URL")


def install_script(ref: str) -> str:
    # The GitHub tarball needs no git in the container, and uv brings its own
    # Python, so the task image's Python version does not matter.
    source = f"anton-agent @ https://github.com/mindsdb/anton/archive/{ref}.tar.gz"
    return (
        "set -euo pipefail; "
        'export PATH="$HOME/.local/bin:$PATH"; '
        "command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh; "
        f"uv tool install --force --python 3.12 {shlex.quote(source)}; "
        '"$(uv tool dir)/anton-agent/bin/python" -c "import anton.cloud_turn"'
    )


def anton_env(api_key: str | None) -> dict[str, str]:
    if not api_key:
        raise RuntimeError("ANTON_FH_API_KEY is not set; run-task.sh sets it")
    # Fireworks directly names it accounts/fireworks/models/kimi-k3.
    model = os.environ.get("ANTON_FH_MODEL", "").strip() or BENCHMARK_MODEL
    return {
        "ANTON_PLANNING_PROVIDER": "openai-compatible",
        "ANTON_CODING_PROVIDER": "openai-compatible",
        "ANTON_PLANNING_MODEL": model,
        "ANTON_CODING_MODEL": model,
        "ANTON_OPENAI_BASE_URL": base_url(),
        "ANTON_OPENAI_API_KEY": api_key,
        "ANTON_USAGE_LOG": "1",
        # Artifacts would otherwise land under the workspace and in the graded diff.
        "ANTON_CLOUD_ARTIFACTS_ROOT": "/tmp/anton-artifacts",
        # Every model call must use the benchmark model, so no Jev calls.
        "ANTON_VERIFIER_JEV": "off",
        "ANTON_DECISIONS_MODEL": "off",
        "ANTON_BROWSER_JEV": "off",
        "ANTON_ANALYTICS_ENABLED": "false",
        "ANTON_POSTHOG_KEY": "",
        "ANTON_DISABLE_AUTOUPDATES": "true",
    }


def run_script(instruction: str, logs_dir: str) -> str:
    request = json.dumps({
        "protocol_version": 1,
        "conversation_id": f"fh-{uuid.uuid4().hex}",
        "input": instruction,
        # On api.mindshub.ai anton would otherwise ask Jev, a non-benchmark model.
        "jev_decisions": False,
    })
    events = f"{logs_dir}/{EVENTS_FILE}"
    stderr = f"{logs_dir}/{STDERR_FILE}"
    # Anton keeps state and scratchpad venvs in <workspace>/.anton. Git-exclude it
    # while the agent works (tasks ask it to commit everything), then remove it
    # before the verifier looks at the workspace.
    return f"""set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
ws="$(pwd)"
mkdir -p {shlex.quote(logs_dir)}
printf '%s\\n' {shlex.quote(request)} > /tmp/anton-request.json
excl="$ws/.git/info/exclude"
if [ -d "$ws/.git" ]; then
  mkdir -p "$ws/.git/info"
  [ -f "$excl" ] && cp "$excl" /tmp/anton-exclude.bak
  printf '%s\\n' '.anton/' >> "$excl"
fi
echo "anton workspace: $ws" > {shlex.quote(stderr)}
ANTON_CLOUD_WORKSPACE_PATH="$ws" "$(uv tool dir)/anton-agent/bin/python" -m anton.cloud_turn \\
  < /tmp/anton-request.json > {shlex.quote(events)} 2>> {shlex.quote(stderr)}
rc=$?
rm -rf "$ws/.anton"
if [ -d "$ws/.git" ]; then
  if [ -f /tmp/anton-exclude.bak ]; then mv /tmp/anton-exclude.bak "$excl"; else rm -f "$excl"; fi
fi
tail -n 1 {shlex.quote(events)}
# cloud_turn exits 0 even when the turn fails; a failed turn is a harness error,
# so the runner treats it like the baselines' crashes and retries.
if [ "$rc" -eq 0 ] && ! tail -n 1 {shlex.quote(events)} | grep -q '"kind": *"turn_completed"'; then
  rc=1
fi
exit $rc
"""


def usage_records(text: str) -> list[dict]:
    records = []
    for line in text.splitlines():
        if not line.startswith(USAGE_PREFIX):
            continue
        try:
            record = json.loads(line[len(USAGE_PREFIX):])
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def usage_summary(logs_dir: Path) -> dict | None:
    path = logs_dir / STDERR_FILE
    if not path.exists():
        return None
    records = usage_records(path.read_text(errors="replace"))
    if not records:
        return None

    def total(key: str) -> int:
        return sum(int(r.get(key) or 0) for r in records)

    fresh, cache_read, cache_write = total("input"), total("cache_read"), total("cache_creation")
    served = [str(r.get("served_model") or "unknown") for r in records]
    return {
        # Anton's `input` is fresh tokens only; the benchmark counts all prompt tokens.
        "n_input_tokens": fresh + cache_read + cache_write,
        "n_cache_tokens": cache_read,
        "n_output_tokens": total("output"),
        "calls": len(records),
        "cache_write_tokens": cache_write,
        "served_models": sorted(set(served)),
        # A call with no served model is counted too: the gateway may have fallen back.
        "non_k3_calls": sum(BENCHMARK_MODEL not in model for model in served),
        "terminal": terminal_kind(logs_dir),
    }


def fill_context(context, logs_dir: Path) -> None:
    """Copy anton's token totals onto a Harbor or Pier AgentContext."""
    summary = usage_summary(logs_dir)
    if summary is None:
        return
    context.n_input_tokens = summary.pop("n_input_tokens")
    context.n_cache_tokens = summary.pop("n_cache_tokens")
    context.n_output_tokens = summary.pop("n_output_tokens")
    if hasattr(context, "n_agent_steps"):
        context.n_agent_steps = summary["calls"]
    context.metadata = {**(context.metadata or {}), "anton": summary}


def terminal_kind(logs_dir: Path) -> str | None:
    path = logs_dir / EVENTS_FILE
    if not path.exists():
        return None
    kind = None
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("kind") in ("turn_completed", "turn_failed"):
            kind = event["kind"]
    return kind
