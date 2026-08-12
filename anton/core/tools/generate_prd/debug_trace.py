"""Optional, generation-scoped step trace for the `generate_prd` tool.

Activated by the same `ANTON_DEBUG_ARTIFACT_GENERATE_TOOL` env var
`generate_artifact` uses (see `generate_artifact/debug_trace.py`) —
deliberately the same variable, not a `generate_prd`-specific one: the two
tools are the two steps of one end-to-end artifact-generation pipeline (PRD,
then code). Both loggers append (`"a"` mode) rather than truncate, so running
them back to back within one process — the normal `create_artifact` ->
`generate_prd` -> `generate_artifact` flow — writes one combined,
chronologically ordered log; open it once in `artifact_trace_viewer.html` to
see the whole run.

When the env var is unset, `make_trace()` returns a `NullTrace` whose methods
are all no-ops, so call sites never branch. Record shape (`ts`, `event`, plus
per-event fields) matches `generate_artifact.debug_trace.GenTrace` for the
event kinds both tools share (`run_start`, `node`, `llm_call`, `verdict`,
`scratchpad`, `run_result`) so the viewer renders either tool's entries with
no changes — it groups consecutive records by their shared `node` field
regardless of which tool wrote them.

Every write is best-effort — the logger must never raise into generation code.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any


class NullTrace:
    """No-op trace: identical surface to `PrdTrace`, does nothing."""

    def run_start(self, **_: Any) -> None: ...
    def node(self, *_: Any, **__: Any) -> None: ...
    def llm_call(self, **_: Any) -> None: ...
    def verdict(self, **_: Any) -> None: ...
    def scratchpad(self, **_: Any) -> None: ...
    def run_result(self, **_: Any) -> None: ...


class PrdTrace:
    """Append-only JSON-lines writer for one process's `generate_prd` runs."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def _emit(self, event: str, payload: dict) -> None:
        try:
            rec = {
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "event": event,
                **payload,
            }
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:
            # Best-effort: never let the logger break generation.
            pass

    def run_start(
        self, *, slug, artifact_type, user_request, agent_understanding,
        known_data="", user_preferences="",
    ) -> None:
        parts = [
            f"User request: {user_request}",
            f"Agent understanding: {agent_understanding}",
        ]
        if known_data:
            parts.append(f"Known data: {known_data}")
        if user_preferences:
            parts.append(f"User preferences: {user_preferences}")
        self._emit("run_start", {
            "slug": slug,
            "artifact_type": artifact_type,
            "brief": "\n\n".join(parts),
        })

    def node(self, node, outcome, detail="") -> None:
        self._emit("node", {"node": node, "outcome": outcome, "detail": detail})

    def llm_call(self, *, node, method, system, messages,
                 response=None, value=None, round=None) -> None:
        from anton.core.llm.serialize import serialize_messages, serialize_response

        if response is not None:
            resp = serialize_response(response)
        else:
            resp = {"structured": value}
        self._emit("llm_call", {
            "node": node,
            "method": method,
            "round": round,
            "system": system,
            "messages": serialize_messages(messages),
            "response": resp,
        })

    def verdict(self, *, node, schema, value) -> None:
        self._emit("verdict", {"node": node, "schema": schema, "value": value})

    def scratchpad(self, *, node, input, output) -> None:
        self._emit("scratchpad", {"node": node, "input": input, "output": output})

    def run_result(self, *, ok, result=None, error=None) -> None:
        self._emit("run_result", {"ok": ok, "result": result, "error": error})


def make_trace() -> "PrdTrace | NullTrace":
    """Return a `PrdTrace` when the debug env var names a file, else `NullTrace`."""
    path = os.environ.get("ANTON_DEBUG_ARTIFACT_GENERATE_TOOL", "").strip()
    return PrdTrace(path) if path else NullTrace()
