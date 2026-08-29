"""Optional, generation-scoped step trace for the `generate_artifact` tool.

Activated by the `ANTON_DEBUG_ARTIFACT_GENERATE_TOOL` env var: when its value
is a non-empty path, `make_trace()` returns a `GenTrace` that appends one
JSON-lines event per step to that file; otherwise a `NullTrace` whose methods
are all no-ops, so call sites never branch.

The trace is stored on `GenState` and every FSM node logs with an explicit
`node` label. That explicitness is deliberate: the backend and frontend
sub-loops run under `asyncio.gather`, so an ambient "current node" would race.

Every write is best-effort — the logger must never raise into generation code.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any


class NullTrace:
    """No-op trace: identical surface to `GenTrace`, does nothing."""

    def run_start(self, **_: Any) -> None: ...
    def node(self, *_: Any, **__: Any) -> None: ...
    def llm_call(self, **_: Any) -> None: ...
    def verdict(self, **_: Any) -> None: ...
    def scratchpad(self, **_: Any) -> None: ...
    def file_written(self, **_: Any) -> None: ...
    def verifier(self, **_: Any) -> None: ...
    def run_result(self, **_: Any) -> None: ...


class GenTrace:
    """Append-only JSON-lines writer for one process's generation runs."""

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

    def run_start(self, **payload: Any) -> None:
        """One event for the whole pipeline.

        Accepts both shapes of caller: the generation FSM passes
        `artifact_path` and `is_fullstack`, the discovery phases pass
        `user_request` and `agent_understanding`. Open-ended on purpose — a
        merged tool with two historical entry points would otherwise need two
        near-identical events, and the viewer keys off `event`, not the field
        set.
        """
        self._emit("run_start", {
            k: str(v) if isinstance(v, Path) else v for k, v in payload.items()
        })

    def node(self, node, outcome, detail="") -> None:
        self._emit("node", {"node": node, "outcome": outcome, "detail": detail})

    def llm_call(self, *, node, method, system, messages,
                 response=None, value=None, attempt=None, round=None) -> None:
        from anton.core.llm.serialize import serialize_messages, serialize_response

        if response is not None:
            resp = serialize_response(response)
        else:
            resp = {"structured": value}
        self._emit("llm_call", {
            "node": node,
            "method": method,
            "attempt": attempt,
            "round": round,
            "system": system,
            "messages": serialize_messages(messages),
            "response": resp,
        })

    def verdict(self, *, node, schema, value) -> None:
        self._emit("verdict", {"node": node, "schema": schema, "value": value})

    def scratchpad(self, *, node, input, output) -> None:
        self._emit("scratchpad", {"node": node, "input": input, "output": output})

    def file_written(self, *, node, path) -> None:
        self._emit("file_written", {"node": node, "path": path})

    def verifier(self, *, node, ok, errors, warnings) -> None:
        self._emit("verifier", {
            "node": node, "ok": ok, "errors": errors, "warnings": warnings,
        })

    def run_result(self, *, ok, result=None, error=None) -> None:
        self._emit("run_result", {"ok": ok, "result": result, "error": error})


def make_trace() -> GenTrace | NullTrace:
    """Return a `GenTrace` when the debug env var names a file, else `NullTrace`."""
    path = os.environ.get("ANTON_DEBUG_ARTIFACT_GENERATE_TOOL", "").strip()
    return GenTrace(path) if path else NullTrace()
