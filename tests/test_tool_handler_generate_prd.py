"""handle_generate_prd: input validation stays unwrapped; pipeline outcomes
(cancelled / prd_written_unconfirmed / crash) come back with an explicit
instruction not to DIY the next step."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import json

import anton.core.tools.generate_prd as gen_pkg
from anton.core.artifacts import ArtifactStore
from anton.core.tools.tool_handlers import handle_generate_prd


def _session(tmp_path: Path):
    return SimpleNamespace(_workspace=SimpleNamespace(artifacts_dir=tmp_path / "artifacts"))


def _make_artifact(tmp_path: Path) -> str:
    store = ArtifactStore(tmp_path / "artifacts")
    return store.create(name="Clock", description="d", type="html-app").slug


async def test_missing_slug_is_not_wrapped(tmp_path):
    out = await handle_generate_prd(_session(tmp_path), {"user_request": "x", "agent_understanding": "y"})
    assert out.startswith("Error: `slug` is required")
    assert "PRD generation failed" not in out


async def test_unknown_slug_is_not_wrapped(tmp_path):
    out = await handle_generate_prd(
        _session(tmp_path), {"slug": "nope", "user_request": "x", "agent_understanding": "y"}
    )
    assert out.startswith("Error: no artifact found")
    assert "PRD generation failed" not in out


async def test_missing_user_request_is_not_wrapped(tmp_path):
    slug = _make_artifact(tmp_path)
    out = await handle_generate_prd(_session(tmp_path), {"slug": slug, "agent_understanding": "y"})
    assert out.startswith("Error: `user_request` is required")


async def test_missing_agent_understanding_is_not_wrapped(tmp_path):
    slug = _make_artifact(tmp_path)
    out = await handle_generate_prd(_session(tmp_path), {"slug": slug, "user_request": "x"})
    assert out.startswith("Error: `agent_understanding` is required")


async def test_success_returns_json_with_prd_written_status(tmp_path, monkeypatch):
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        return {"status": "prd_written", "prd_path": "x/prd.md", "artifact_type": "html-app", "brief_summary": "b", "qa_log": "q"}

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    out = await handle_generate_prd(
        _session(tmp_path), {"slug": slug, "user_request": "x", "agent_understanding": "y"}
    )
    payload = json.loads(out)
    assert payload["status"] == "prd_written"
    assert "instruction" not in payload


async def test_cancelled_carries_a_do_not_proceed_instruction(tmp_path, monkeypatch):
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        return {"status": "cancelled", "reason": "user declined the PRD brief", "qa_log": "q"}

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    out = await handle_generate_prd(
        _session(tmp_path), {"slug": slug, "user_request": "x", "agent_understanding": "y"}
    )
    payload = json.loads(out)
    assert payload["status"] == "cancelled"
    assert "do NOT" in payload["instruction"]
    assert "generate_artifact" not in payload["instruction"]  # not merged into this branch yet


async def test_unconfirmed_carries_a_confirmation_instruction(tmp_path, monkeypatch):
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        return {"status": "prd_written_unconfirmed", "prd_path": "x/prd.md", "artifact_type": "html-app", "brief_summary": "b", "qa_log": "q"}

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    out = await handle_generate_prd(
        _session(tmp_path), {"slug": slug, "user_request": "x", "agent_understanding": "y"}
    )
    payload = json.loads(out)
    assert payload["status"] == "prd_written_unconfirmed"
    assert "confirm" in payload["instruction"].lower()


async def test_generator_crash_is_wrapped(tmp_path, monkeypatch):
    slug = _make_artifact(tmp_path)

    async def fake_generate(**kw):
        raise RuntimeError("kaput")

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    out = await handle_generate_prd(
        _session(tmp_path), {"slug": slug, "user_request": "x", "agent_understanding": "y"}
    )
    assert "PRD generation failed" in out
    assert "kaput" in out
    assert "do NOT write prd.md yourself" in out


async def test_optional_fields_default_to_empty_string(tmp_path, monkeypatch):
    slug = _make_artifact(tmp_path)
    captured = {}

    async def fake_generate(**kw):
        captured.update(kw)
        return {"status": "prd_written", "prd_path": "x", "artifact_type": "html-app", "brief_summary": "", "qa_log": ""}

    monkeypatch.setattr(gen_pkg, "generate", fake_generate)
    await handle_generate_prd(_session(tmp_path), {"slug": slug, "user_request": "x", "agent_understanding": "y"})
    assert captured["known_data"] == ""
    assert captured["user_preferences"] == ""
