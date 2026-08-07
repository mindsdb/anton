"""Safety tests for the cloud-safe ChatSession builder.

Two layers:
* Config-level (fast, offline): the ChatSessionConfig handed to ChatSession has
  every cross-tenant hazard off and the right allowlist/factory.
* Enforcement-level (real code paths): drive the real lazy _build_tools() and a
  real sanitized scratchpad subprocess so a claimed property is proven, not just
  configured.
"""

from __future__ import annotations

import pytest

import anton.core.llm.client as llm_client_mod
import anton.core.session as session_mod
from anton.cloud_turn.contract import TurnRequestV1
from anton.cloud_turn.session import (
    CLOUD_TOOL_ALLOWLIST,
    _WORKSPACE_PATH_ENV,
    build_cloud_chat_session,
    resolve_trusted_workspace_path,
)
from anton.core.backends.local import local_scratchpad_runtime_factory


class _FakeSession:
    """Placeholder return value for the mocked ChatSession - config-level tests
    only inspect the captured ChatSessionConfig, never the session."""


def _build(tmp_path, monkeypatch, **req_overrides):
    captured: dict = {}

    def fake_chat_session(config):
        captured["config"] = config
        return _FakeSession()

    monkeypatch.setattr(session_mod, "ChatSession", fake_chat_session)
    monkeypatch.setattr(
        llm_client_mod.LLMClient, "from_settings",
        classmethod(lambda cls, settings: object()),
    )
    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))

    body = dict(protocol_version=1, conversation_id="conv_1", input="hello")
    body.update(req_overrides)
    session = build_cloud_chat_session(TurnRequestV1(**body))
    return session, captured["config"]


# ── config-level safety ──────────────────────────────────────────────────────

def test_all_cross_tenant_hazards_off(tmp_path, monkeypatch):
    _, cfg = _build(tmp_path, monkeypatch)
    assert cfg.episodic is None
    assert cfg.self_awareness is None
    assert cfg.data_vault is None        # connectors / vault OFF
    assert cfg.history_store is None     # disk history OFF
    assert cfg.console is None           # headless
    assert cfg.tools == []               # no host connector/publish tools
    assert cfg.web_search_enabled is False
    assert cfg.web_fetch_enabled is False


def test_scratchpad_uses_local_factory_and_is_workspace_bound(tmp_path, monkeypatch):
    _, cfg = _build(tmp_path, monkeypatch)
    assert cfg.runtime_factory is local_scratchpad_runtime_factory
    assert cfg.workspace is not None
    assert cfg.harness == "cloud"
    assert cfg.session_id == "conv_1"


def test_db_history_is_seeded_not_loaded(tmp_path, monkeypatch):
    _, cfg = _build(
        tmp_path, monkeypatch,
        history=[{"role": "user", "content": "prior turn"}],
    )
    assert cfg.initial_history == [{"role": "user", "content": "prior turn"}]


def test_config_uses_explicit_tool_allowlist(tmp_path, monkeypatch):
    _, cfg = _build(tmp_path, monkeypatch)
    assert cfg.tool_allowlist == CLOUD_TOOL_ALLOWLIST
    assert "launch_backend" not in cfg.tool_allowlist
    assert "scratchpad" in cfg.tool_allowlist


def test_model_override_applied(tmp_path, monkeypatch):
    _, cfg = _build(tmp_path, monkeypatch, model="claude-opus-4-8")
    assert cfg.settings.planning_model == "claude-opus-4-8"


def test_cloud_session_uses_turn_key_from_request(tmp_path, monkeypatch):
    # No minds env set; the credential must come from the request's llm block.
    monkeypatch.delenv("ANTON_MINDS_API_KEY", raising=False)
    monkeypatch.delenv("ANTON_PLANNING_PROVIDER", raising=False)
    _, cfg = _build(
        tmp_path, monkeypatch,
        llm={"provider": "minds-cloud", "api_key": "mdb_turnkey",
             "base_url": "https://api.mindshub.ai/v1"},
    )
    s = cfg.settings
    assert s.planning_provider == "openai-compatible"  # minds-cloud maps to this
    assert s.coding_provider == "openai-compatible"
    assert s.openai_api_key == "mdb_turnkey"
    assert "mindshub.ai" in (s.openai_base_url or "")


def test_llm_block_coding_model_applied(tmp_path, monkeypatch):
    # Without this the pod keeps the built-in (paid) coding default and every
    # coding-model call 402s on an unfunded wallet.
    _, cfg = _build(
        tmp_path, monkeypatch,
        llm={"provider": "minds-cloud", "api_key": "mdb_turnkey",
             "base_url": "https://api.mindshub.ai/v1", "coding_model": "mindshub_air"},
    )
    assert cfg.settings.coding_model == "mindshub_air"


def test_llm_block_without_coding_model_keeps_default(tmp_path, monkeypatch):
    _, cfg = _build(
        tmp_path, monkeypatch,
        llm={"provider": "minds-cloud", "api_key": "mdb_turnkey",
             "base_url": "https://api.mindshub.ai/v1"},
    )
    assert cfg.settings.coding_model  # built-in default survives


def test_cloud_session_without_llm_block_falls_back_to_env(tmp_path, monkeypatch):
    # Back-compat: no llm block on the request means env-based settings, same
    # as before this request field existed. Hermetic env: earlier tests that
    # exercise the connect flow export ANTON_* into this process via
    # Workspace.set_secret (deliberate for the CLI), which would leak in here.
    import os
    for k in list(os.environ):
        if k.startswith("ANTON_"):
            monkeypatch.delenv(k, raising=False)
    _, cfg = _build(tmp_path, monkeypatch)
    assert cfg.settings.planning_provider == "anthropic"


# ── trusted workspace path (never from the wire) ─────────────────────────────

def test_request_workspace_path_is_ignored_trusted_mount_used(tmp_path, monkeypatch):
    # A request may carry workspace_path (cowork sends "/workspace"), but the pod
    # uses its OWN trusted mount, not the wire value.
    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))
    _, cfg = _build(tmp_path, monkeypatch, workspace_path="/etc/evil")
    assert cfg.workspace.base == tmp_path.resolve()


def test_resolver_uses_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))
    assert resolve_trusted_workspace_path() == tmp_path.resolve()


def test_resolver_rejects_relative_path(monkeypatch):
    monkeypatch.setenv(_WORKSPACE_PATH_ENV, "not/absolute")
    with pytest.raises(ValueError, match="absolute"):
        resolve_trusted_workspace_path()


def test_resolver_rejects_parent_traversal(monkeypatch):
    monkeypatch.setenv(_WORKSPACE_PATH_ENV, "/workspace/../etc")
    with pytest.raises(ValueError, match=r"\.\."):
        resolve_trusted_workspace_path()


def test_artifact_tools_cannot_escape_workspace(tmp_path):
    from anton.core.artifacts.store import ArtifactStore

    store = ArtifactStore(tmp_path / "artifacts")
    for bad in ("../../etc", "/etc/passwd", "a/../../b"):
        with pytest.raises(ValueError, match="escapes"):
            store.folder_for(bad)
    assert store.open("../../../etc/passwd") is None
    assert store.folder_for("my-report").parent == (tmp_path / "artifacts")


# ── dotenv never loaded ──────────────────────────────────────────────────────

def test_apply_env_to_process_never_called(tmp_path, monkeypatch):
    import anton.workspace as ws_mod
    calls = {"apply_env": 0, "init": 0}
    real_init = ws_mod.Workspace.initialize
    real_apply = ws_mod.Workspace.apply_env_to_process

    monkeypatch.setattr(ws_mod.Workspace, "initialize",
                        lambda self: (calls.__setitem__("init", calls["init"] + 1), real_init(self))[1])
    monkeypatch.setattr(ws_mod.Workspace, "apply_env_to_process",
                        lambda self: (calls.__setitem__("apply_env", calls["apply_env"] + 1), real_apply(self))[1])

    _build(tmp_path, monkeypatch)
    assert calls["init"] == 1
    assert calls["apply_env"] == 0       # .env NOT loaded into process env


def test_cloud_settings_ignore_dotenv_files(tmp_path, monkeypatch):
    from anton.config.settings import AntonSettings

    monkeypatch.delenv("ANTON_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    envf = tmp_path / "sentinel.env"
    envf.write_text("ANTON_ANTHROPIC_API_KEY=DOTENV_SENTINEL\n")
    assert AntonSettings(_env_file=str(envf)).anthropic_api_key == "DOTENV_SENTINEL"

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("ANTON_ANTHROPIC_API_KEY=DOTENV_SENTINEL\n")
    _, cfg = _build(tmp_path, monkeypatch)
    assert cfg.settings.anthropic_api_key != "DOTENV_SENTINEL"


# ── real enforcement paths ───────────────────────────────────────────────────

def _mock_llm():
    from unittest.mock import AsyncMock, MagicMock

    from anton.core.llm.provider import ProviderConnectionInfo

    llm = AsyncMock()
    llm.coding_provider = MagicMock()
    llm.coding_provider.export_connection_info = MagicMock(
        return_value=ProviderConnectionInfo(provider="anthropic", api_key="test")
    )
    llm.coding_model = "claude-sonnet-4-6"
    llm.planning_provider = MagicMock()
    llm.planning_provider.native_web_tools = MagicMock(return_value=set())
    return llm


def _real_session_with_workspace(tmp_path, **cfg_overrides):
    from unittest.mock import MagicMock

    from anton.core.session import ChatSession, ChatSessionConfig
    from anton.workspace import Workspace

    ws = Workspace(tmp_path)
    ws.initialize()
    session = ChatSession(
        ChatSessionConfig(llm_client=_mock_llm(), workspace=ws, **cfg_overrides)
    )
    session._scratchpads = MagicMock(available_packages=[])
    return session


def test_final_tool_set_equals_allowlist_after_real_build(tmp_path, monkeypatch):
    """The registry is built LAZILY at turn time, so the allowlist must be
    enforced by the real _build_tools() - assert the EXACT final tool set."""
    from unittest.mock import MagicMock

    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))
    monkeypatch.setattr(
        llm_client_mod.LLMClient, "from_settings",
        classmethod(lambda cls, settings: _mock_llm()),
    )
    req = TurnRequestV1(protocol_version=1, conversation_id="c", input="hi")
    session = build_cloud_chat_session(req)  # REAL ChatSession
    session._scratchpads = MagicMock(available_packages=[])

    session._build_tools()
    names = {t.name for t in session.tool_registry.get_tool_defs()}
    assert names == set(CLOUD_TOOL_ALLOWLIST), (
        f"cloud tool set drifted from the allowlist: {sorted(names)}"
    )


def test_tool_allowlist_none_preserves_desktop_tools(tmp_path):
    session = _real_session_with_workspace(tmp_path)  # allowlist defaults to None
    assert session._tool_allowlist is None
    session._build_tools()
    names = {t.name for t in session.tool_registry.get_tool_defs()}
    assert {"launch_backend", "select_path", "scratchpad", "create_artifact"} <= names


async def test_scratchpad_inherits_parent_env(tmp_path, monkeypatch):
    """The scratchpad subprocess inherits the parent env (no sanitizing)."""
    monkeypatch.setenv("MY_DESKTOP_SENTINEL", "DESKTOP-VISIBLE")
    pad = local_scratchpad_runtime_factory(
        name="desktop-env", coding_provider="anthropic", coding_model="",
        coding_api_key="", coding_base_url="", cells=None, workspace_path=tmp_path,
    )
    await pad.start()
    try:
        cell = await pad.execute(
            "import os; print(os.environ.get('MY_DESKTOP_SENTINEL', 'MISSING'))"
        )
        assert cell.error is None, cell.error
        assert cell.stdout.strip() == "DESKTOP-VISIBLE"
    finally:
        await pad.close()


# ── org memory (server-supplied, read-only) ──────────────────────────────────

_MEMORY = {
    "global": {"profile": "Name: Zoran", "rules": "## Always\n- Reply in Spanish"},
    "project": {"lessons": "- The staging DB is read-only"},
}


@pytest.fixture()
def memory_dir(tmp_path_factory, monkeypatch):
    """Redirect the pod-local memory scratch so tests never touch the real one.

    Deliberately outside `tmp_path` — that IS the workspace mount here, and the
    real scratch lives outside the mount (see the isolation test below).
    """
    import anton.cloud_turn.session as cloud_session

    path = tmp_path_factory.mktemp("mem-scratch")
    monkeypatch.setattr(cloud_session, "_MEMORY_DIR", path)
    return path


def test_empty_memory_still_allows_a_first_memory(tmp_path, monkeypatch, memory_dir):
    """A fresh org sends no slots, but must still be able to remember something —
    core only registers `memorize` when a cortex exists."""
    _, cfg = _build(tmp_path, monkeypatch)
    assert cfg.cortex is not None
    assert "memorize" in cfg.tool_allowlist
    # staged dirs exist but hold no slots, so no memory section is injected
    assert list((memory_dir / "global").iterdir()) == []


def test_memory_block_builds_cortex_without_background_passes(tmp_path, monkeypatch, memory_dir):
    _, cfg = _build(tmp_path, monkeypatch, memory=_MEMORY)
    assert cfg.cortex is not None
    # Not "off" — that would make the memorize tool refuse. The passes it unlocks
    # (identity extraction, vacuum, consolidation) are disabled separately.
    assert cfg.cortex.mode == "autopilot"
    assert cfg.background_memory is False
    assert (memory_dir / "global" / "profile.md").read_text() == "Name: Zoran"
    assert (memory_dir / "project" / "lessons.md").is_file()


async def test_memory_reaches_the_prompt(tmp_path, monkeypatch, memory_dir):
    """The point of the feature: the tenant's memory lands in the system prompt."""
    _, cfg = _build(tmp_path, monkeypatch, memory=_MEMORY)
    context = await cfg.cortex.build_memory_context("hola")
    assert "Name: Zoran" in context
    assert "Reply in Spanish" in context
    assert "staging DB is read-only" in context


def test_unknown_slots_ignored(tmp_path, monkeypatch, memory_dir):
    _build(tmp_path, monkeypatch, memory={"global": {"../escape": "x", "secrets": "y"}})
    assert [p.name for p in (memory_dir / "global").iterdir()] == []


def test_stale_slot_cleared_between_turns(tmp_path, monkeypatch, memory_dir):
    # A long-lived pod reuses the dir, so a slot deleted server-side must vanish.
    _build(tmp_path, monkeypatch, memory=_MEMORY)
    assert (memory_dir / "global" / "rules.md").is_file()

    _build(tmp_path, monkeypatch, memory={"global": {"profile": "Name: Zoran"}})
    assert not (memory_dir / "global" / "rules.md").exists()
    assert (memory_dir / "global" / "profile.md").is_file()


def test_memory_never_lands_in_the_tenant_workspace(tmp_path, monkeypatch, memory_dir):
    # Memory is server-owned: it must not persist in the PVC or be readable by
    # scratchpad code running against the mount.
    _, cfg = _build(tmp_path, monkeypatch, memory=_MEMORY)
    for slot_file in ("profile.md", "rules.md", "lessons.md"):
        assert list(cfg.workspace.base.rglob(slot_file)) == []


def test_memory_scratch_is_outside_the_mount_by_default():
    """The shipped path, not the test override: scratch must not be in the PVC."""
    import anton.cloud_turn.session as cloud_session

    assert not cloud_session._MEMORY_DIR.is_relative_to(
        cloud_session.DEFAULT_CLOUD_WORKSPACE_PATH
    )


def _real_cloud_session(tmp_path, monkeypatch, **req_overrides):
    from unittest.mock import MagicMock

    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))
    monkeypatch.setattr(
        llm_client_mod.LLMClient, "from_settings",
        classmethod(lambda cls, settings: _mock_llm()),
    )
    body = dict(protocol_version=1, conversation_id="c", input="hi")
    body.update(req_overrides)
    session = build_cloud_chat_session(TurnRequestV1(**body))  # REAL ChatSession
    session._scratchpads = MagicMock(available_packages=[])
    return session


def test_memorize_is_offered_with_or_without_existing_memory(tmp_path, monkeypatch, memory_dir):
    """The allowlist names `memorize` unconditionally, and an allowlist name that
    matches no built tool is a hard error — so a cortex must always exist. Both
    directions are checked because an empty store is the case that used to break.
    """
    for overrides in ({}, {"memory": _MEMORY}):
        session = _real_cloud_session(tmp_path, monkeypatch, **overrides)
        session._build_tools()
        names = {t.name for t in session.tool_registry.get_tool_defs()}
        assert names == set(CLOUD_TOOL_ALLOWLIST)
        assert "memorize" in names


# ── memory write path (pod reports, never persists) ──────────────────────────

def _engram(**kw):
    from anton.core.memory.base import Engram

    return Engram(**{"text": "Reply in Spanish", "kind": "always", "scope": "global", **kw})


async def test_encode_captures_instead_of_writing(tmp_path, monkeypatch, memory_dir):
    from anton.cloud_turn.session import drain_pending_memory

    session, cfg = _build(tmp_path, monkeypatch, memory=_MEMORY)
    before = (memory_dir / "global" / "rules.md").read_text()

    actions = await cfg.cortex.encode([_engram()])
    assert actions == ["Encoded always: Reply in Spanish"]
    # nothing persisted pod-side — the staged file is untouched
    assert (memory_dir / "global" / "rules.md").read_text() == before

    assert cfg.cortex.pending_memory == [{
        "text": "Reply in Spanish", "kind": "always", "scope": "global",
        "topic": "", "confidence": "medium", "source": "llm",
    }]


async def test_encode_normalizes_and_drops_blanks(tmp_path, monkeypatch, memory_dir):
    """The pod normalizes; cowork validates. Kinds/scopes are deliberately NOT
    checked here — this runs inside the sandbox the check would be guarding
    against, so it buys nothing and a second allowlist could only drift."""
    _, cfg = _build(tmp_path, monkeypatch, memory=_MEMORY)
    await cfg.cortex.encode([_engram(scope=None), _engram(text="   ")])

    assert [(e["scope"], e["confidence"]) for e in cfg.cortex.pending_memory] == [
        ("global", "medium"),                # scope defaulted, blank dropped
    ]


async def test_drain_is_idempotent(tmp_path, monkeypatch, memory_dir):
    from anton.cloud_turn.session import drain_pending_memory

    session, cfg = _build(tmp_path, monkeypatch, memory=_MEMORY)
    session._cortex = cfg.cortex  # the fake session stands in for the real one
    await cfg.cortex.encode([_engram()])

    assert len(drain_pending_memory(session)) == 1
    assert drain_pending_memory(session) == []   # cleared: no double-apply


async def test_memorize_registers_its_write_for_settling(tmp_path, monkeypatch, memory_dir):
    """`handle_memorize` fires encoding as a task; the pod exits at end of turn, so
    the task must be registered for `settle_memory_writes` to await it."""
    from anton.core.tools.tool_handlers import handle_memorize
    from anton.cloud_turn.session import drain_pending_memory

    session = _real_cloud_session(tmp_path, monkeypatch, memory=_MEMORY)
    result = await handle_memorize(session, {"entries": [
        {"text": "Answer in Spanish", "kind": "always", "scope": "global"},
    ]})
    assert "Memory updated" in result
    assert len(session._memory_writes) == 1        # registered, not yet awaited

    await session.settle_memory_writes()
    assert session._memory_writes == set()         # done-callback cleared it
    assert [e["text"] for e in drain_pending_memory(session)] == ["Answer in Spanish"]


async def test_settling_is_a_noop_without_writes(tmp_path, monkeypatch, memory_dir):
    from anton.cloud_turn.session import drain_pending_memory

    session = _real_cloud_session(tmp_path, monkeypatch, memory=_MEMORY)
    await session.settle_memory_writes()           # must not hang or raise
    assert drain_pending_memory(session) == []


def test_background_memory_passes_are_off_in_cloud(tmp_path, monkeypatch, memory_dir):
    """One turn per pod: identity/vacuum/consolidation would spend LLM calls on
    writes that are discarded when the process exits."""
    _, cfg = _build(tmp_path, monkeypatch, memory=_MEMORY)
    assert cfg.background_memory is False
    assert cfg.cortex.mode == "autopilot"   # required for `memorize` to encode at all


def test_cerebellum_and_acc_do_not_fire_without_background_memory(tmp_path, monkeypatch):
    """Both flushes guard on cortex.mode, which cloud must set to "autopilot" —
    so background_memory is the only thing keeping their LLM passes off.
    """
    from unittest.mock import MagicMock

    spawned = []
    monkeypatch.setattr(session_mod.asyncio, "create_task",
                        lambda coro, **kw: (spawned.append(coro), coro.close())[0])

    for background, tasks in ((False, 0), (True, 1)):
        session = _real_session_with_workspace(
            tmp_path, cortex=MagicMock(mode="autopilot"), background_memory=background,
        )
        session._cerebellum = MagicMock(buffered_count=3)
        session._acc = MagicMock(at_end_of_turn=MagicMock(return_value=[]))
        spawned.clear()

        session._schedule_cerebellum_flush()
        assert len(spawned) == tasks             # the LLM diff pass
        assert session._cerebellum.reset.called is (not background)

        session._schedule_acc_flush()
        assert session._acc.at_end_of_turn.called is background
