"""handle_generate_artifact: FSM failures must come back wrapped with an
instruction to report to the user (never DIY the artifact); input-validation
errors stay unwrapped so the agent fixes its call instead."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anton.core.tools.generate_artifact as gen_pkg
from anton.core.artifacts import ArtifactStore
from anton.core.tools.tool_handlers import handle_generate_artifact


def _session(tmp_path: Path):
    return SimpleNamespace(
        _workspace=SimpleNamespace(artifacts_dir=tmp_path / "artifacts")
    )


def _make_artifact(tmp_path: Path) -> str:
    store = ArtifactStore(tmp_path / "artifacts")
    return store.create(
        name="Clock", description="d", type="fullstack-stateless-app"
    ).slug


async def test_fsm_failure_is_wrapped_with_report_instruction(tmp_path: Path, monkeypatch):
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        return "Backend verification failed after retry: boom"

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    out = await handle_generate_artifact(
        _session(tmp_path), {"slug": slug, "context": "brief"}
    )
    assert "artifact generation failed" in out
    assert "Backend verification failed after retry: boom" in out
    assert "do NOT build or repair the artifact yourself" in out
    assert "Report this failure to the user" in out


async def test_generator_crash_is_wrapped(tmp_path: Path, monkeypatch):
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        raise RuntimeError("kaput")

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    out = await handle_generate_artifact(
        _session(tmp_path), {"slug": slug, "context": "brief"}
    )
    assert "artifact generation failed" in out
    assert "kaput" in out


async def test_input_validation_errors_are_not_wrapped(tmp_path: Path):
    out = await handle_generate_artifact(
        _session(tmp_path), {"slug": "nope", "context": "b"}
    )
    assert out.startswith("Error: no artifact found")
    assert "generation failed" not in out


async def test_success_returns_json_unchanged(tmp_path: Path, monkeypatch):
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        return {"files_written": ["backend.py"], "summary": "ok", "trace": []}

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    out = await handle_generate_artifact(
        _session(tmp_path), {"slug": slug, "context": "brief"}
    )
    assert '"files_written"' in out
    assert "generation failed" not in out
