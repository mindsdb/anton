"""Desktop regressions for the cloud memory work.

The cloud pod reports memory writes instead of performing them, and disables the
automatic end-of-turn passes. Neither may leak into the desktop default: there,
`memorize` still writes straight to local files and every automatic pass runs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import anton.core.session as session_mod
from anton.core.memory.cortex import Cortex
from anton.core.memory.hippocampus import Hippocampus
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.tool_handlers import handle_memorize
from anton.workspace import Workspace

from tests.test_cloud_turn_session import _mock_llm


@pytest.fixture()
def memory_dirs(tmp_path):
    global_dir, project_dir = tmp_path / "global", tmp_path / "project"
    global_dir.mkdir()
    project_dir.mkdir()
    return global_dir, project_dir


def _desktop_session(tmp_path, memory_dirs, **overrides) -> ChatSession:
    """A session built the way the desktop host builds one: real Cortex over real
    directories, and no `background_memory` override."""
    global_dir, project_dir = memory_dirs
    ws = Workspace(tmp_path / "ws")
    ws.initialize()
    session = ChatSession(ChatSessionConfig(
        llm_client=_mock_llm(),
        workspace=ws,
        cortex=Cortex(
            global_hc=Hippocampus(global_dir),
            project_hc=Hippocampus(project_dir),
            mode="autopilot",
            llm_client=_mock_llm(),
        ),
        **overrides,
    ))
    session._scratchpads = MagicMock(available_packages=[], pads={})
    return session


# ── memorize still writes to local files ─────────────────────────────────────

async def test_desktop_memorize_writes_straight_to_disk(tmp_path, memory_dirs):
    global_dir, _ = memory_dirs
    session = _desktop_session(tmp_path, memory_dirs)

    result = await handle_memorize(session, {"entries": [
        {"text": "Use httpx instead of requests", "kind": "always", "scope": "global"},
        {"text": "Name: Zoran", "kind": "profile", "scope": "global"},
        {"text": "CoinGecko rate-limits at 50/min", "kind": "lesson", "scope": "global"},
    ]})
    assert "Memory updated" in result
    await session.settle_memory_writes()

    assert "Use httpx instead of requests" in (global_dir / "rules.md").read_text()
    assert "Name: Zoran" in (global_dir / "profile.md").read_text()
    assert "CoinGecko rate-limits at 50/min" in (global_dir / "lessons.md").read_text()
    # Desktop persists; it does not queue engrams for a host to apply later.
    assert not hasattr(session._cortex, "pending_memory")


async def test_desktop_project_scope_writes_to_the_project_dir(tmp_path, memory_dirs):
    global_dir, project_dir = memory_dirs
    session = _desktop_session(tmp_path, memory_dirs)

    await handle_memorize(session, {"entries": [
        {"text": "Deploy on green only", "kind": "always", "scope": "project"},
    ]})
    await session.settle_memory_writes()

    assert "Deploy on green only" in (project_dir / "rules.md").read_text()
    assert not (global_dir / "rules.md").exists()


async def test_desktop_memory_reaches_the_prompt(tmp_path, memory_dirs):
    session = _desktop_session(tmp_path, memory_dirs)
    await handle_memorize(session, {"entries": [
        {"text": "Answer in Spanish", "kind": "always", "scope": "global"},
    ]})
    await session.settle_memory_writes()

    context = await session._cortex.build_memory_context("hola")
    assert "Answer in Spanish" in context


# ── the automatic passes still run by default ────────────────────────────────

def test_background_memory_defaults_on(tmp_path, memory_dirs):
    """The desktop host passes no override, so the flag must default to on."""
    assert ChatSessionConfig(llm_client=_mock_llm()).background_memory is True
    assert _desktop_session(tmp_path, memory_dirs)._background_memory is True


def test_the_shared_gate_truth_table(tmp_path, memory_dirs):
    """`_background_memory_active` is the single gate on all four automatic call
    sites (vacuum x2, identity extraction, scratchpad consolidation), so its
    truth table is their contract. Only cloud's override should close it."""
    assert _desktop_session(tmp_path, memory_dirs)._background_memory_active is True

    off = _desktop_session(tmp_path, memory_dirs, background_memory=False)
    assert off._background_memory_active is False          # what cloud sets

    no_cortex = _desktop_session(tmp_path, memory_dirs)
    no_cortex._cortex = None
    assert no_cortex._background_memory_active is False

    encoding_off = _desktop_session(tmp_path, memory_dirs)
    encoding_off._cortex.mode = "off"
    assert encoding_off._background_memory_active is False


def test_cerebellum_and_acc_flushes_fire_by_default(tmp_path, memory_dirs, monkeypatch):
    """Both guard on cortex.mode as well, so they need the desktop default proved
    separately from the four `_background_memory_active` sites."""
    spawned = []
    monkeypatch.setattr(session_mod.asyncio, "create_task",
                        lambda coro, **kw: (spawned.append(coro), coro.close())[0])

    session = _desktop_session(tmp_path, memory_dirs)
    session._cerebellum = MagicMock(buffered_count=3)
    session._acc = MagicMock(at_end_of_turn=MagicMock(return_value=[]))

    session._schedule_cerebellum_flush()
    assert len(spawned) == 1                       # the LLM diff pass ran
    assert not session._cerebellum.reset.called    # buffer not discarded

    session._schedule_acc_flush()
    assert session._acc.at_end_of_turn.called      # detectors ran


async def test_identity_extraction_still_writes_on_desktop(tmp_path, memory_dirs, monkeypatch):
    """Cloud disables this pass because its writes land after the pod exits.
    Desktop must still run it end to end — it's how the agent learns who it's
    talking to. Only the LLM call is stubbed; the write path is real.
    """
    import anton.core.memory.cortex as cortex_mod

    global_dir, _ = memory_dirs
    facts = cortex_mod._IdentityFacts(facts=["Name: Zoran", "Timezone: CET"])

    async def fake_extract(*args, **kwargs):
        return facts

    monkeypatch.setattr(cortex_mod, "generate_with_truncation_retry", fake_extract)

    session = _desktop_session(tmp_path, memory_dirs)
    await session._cortex.maybe_update_identity("my name is Zoran, I'm in CET")

    stored = (global_dir / "profile.md").read_text()
    assert "Name: Zoran" in stored and "Timezone: CET" in stored
