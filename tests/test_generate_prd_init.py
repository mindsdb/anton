"""generate_prd.generate(): wires PrdState's trace_log to the file the
ANTON_DEBUG_ARTIFACT_GENERATE_TOOL env var names, and logs run_start/
run_result around orchestrator.run — mirrors generate_artifact/engine.py's
generate() wrapping around its own orchestrator.run."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from anton.core.tools import generate_prd


def _kwargs(**over) -> dict:
    base = dict(
        session=SimpleNamespace(),
        slug="s",
        artifact_path=Path("/tmp/s"),
        artifact_type="html-app",
        user_request="build a clock",
        agent_understanding="an analog clock",
        known_data="",
        user_preferences="",
    )
    base.update(over)
    return base


async def test_generate_passes_the_trace_log_into_prd_state(monkeypatch):
    trace = MagicMock()
    monkeypatch.setattr("anton.core.tools.generate_prd.debug_trace.make_trace", lambda: trace)

    seen_state = {}

    async def fake_run(state):
        seen_state["state"] = state
        return {"status": "prd_written"}

    monkeypatch.setattr(generate_prd, "run", fake_run)
    await generate_prd.generate(**_kwargs())
    assert seen_state["state"].trace_log is trace


async def test_generate_logs_run_start_before_run_and_run_result_after(monkeypatch):
    trace = MagicMock()
    monkeypatch.setattr("anton.core.tools.generate_prd.debug_trace.make_trace", lambda: trace)

    calls = []

    async def fake_run(state):
        calls.append("run")
        return {"status": "prd_written"}

    monkeypatch.setattr(generate_prd, "run", fake_run)
    result = await generate_prd.generate(**_kwargs())

    trace.run_start.assert_called_once_with(
        slug="s", artifact_type="html-app",
        user_request="build a clock", agent_understanding="an analog clock",
        known_data="", user_preferences="",
    )
    assert calls == ["run"]
    trace.run_result.assert_called_once_with(ok=True, result=result)


async def test_generate_logs_a_failed_run_result_and_still_raises(monkeypatch):
    trace = MagicMock()
    monkeypatch.setattr("anton.core.tools.generate_prd.debug_trace.make_trace", lambda: trace)

    async def fake_run(state):
        raise RuntimeError("draft_brief: the model replied with no text")

    monkeypatch.setattr(generate_prd, "run", fake_run)
    with pytest.raises(RuntimeError, match="no text"):
        await generate_prd.generate(**_kwargs())
    trace.run_result.assert_called_once_with(ok=False, error="draft_brief: the model replied with no text")


async def test_generate_writes_a_real_log_file_when_the_env_var_is_set(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ANTON_DEBUG_ARTIFACT_GENERATE_TOOL", str(path))

    async def fake_run(state):
        return {"status": "prd_written"}

    monkeypatch.setattr(generate_prd, "run", fake_run)
    await generate_prd.generate(**_kwargs())

    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["event"] for r in records] == ["run_start", "run_result"]
