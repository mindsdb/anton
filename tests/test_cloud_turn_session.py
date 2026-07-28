"""Safety tests for the cloud-safe ChatSession builder.

Two layers:
* Config-level (fast, offline): assert the ChatSessionConfig handed to
  ChatSession has every cross-tenant hazard off and the right allowlist/factory.
* Enforcement-level (real code paths): drive the real lazy _build_tools() and a
  real sanitized scratchpad subprocess so a claimed property is actually proven,
  not just configured.
"""

from __future__ import annotations

import pytest

import anton.core.llm.client as llm_client_mod
import anton.core.session as session_mod
from anton.cloud_turn.protocol import CapabilitiesV1, TurnRequestV1
from anton.cloud_turn.session import (
    CLOUD_TOOL_ALLOWLIST,
    _MODEL_ALLOWLIST_ENV,
    _WORKSPACE_PATH_ENV,
    build_cloud_chat_session,
    resolve_trusted_workspace_path,
)
from anton.core.backends.local import sanitized_scratchpad_runtime_factory


class _FakeRegistry:
    def __init__(self) -> None:
        self.removed: list[str] = []

    def unregister_tool(self, name: str) -> None:
        self.removed.append(name)

    def get_tool_defs(self):
        return []


class _FakeSession:
    def __init__(self) -> None:
        self.tool_registry = _FakeRegistry()


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
    # The mount path is trusted pod config, supplied via env — never the wire.
    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))

    body = dict(
        run_id="run_1", attempt_id="att_1", conversation_id="conv_1",
        input="hello",
    )
    body.update(req_overrides)
    session = build_cloud_chat_session(TurnRequestV1(**body))
    return session, captured["config"]


def test_all_cross_tenant_hazards_off(tmp_path, monkeypatch):
    _, cfg = _build(tmp_path, monkeypatch)
    assert cfg.cortex is None            # personal memory OFF
    assert cfg.episodic is None
    assert cfg.self_awareness is None
    assert cfg.data_vault is None        # connectors / vault OFF
    assert cfg.history_store is None     # local-file history OFF
    assert cfg.console is None           # headless
    assert cfg.tools == []               # no host connector/publish tools
    assert cfg.web_search_enabled is False
    assert cfg.web_fetch_enabled is False


def test_scratchpad_uses_sanitized_factory_and_is_workspace_bound(tmp_path, monkeypatch):
    _, cfg = _build(tmp_path, monkeypatch)
    # Scratchpad stays ON but through the sanitized factory (no secret env).
    assert cfg.runtime_factory is sanitized_scratchpad_runtime_factory
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
    # An explicit allowlist (not a denylist) drives the lazy _build_tools().
    assert cfg.tool_allowlist == CLOUD_TOOL_ALLOWLIST
    assert "launch_backend" not in cfg.tool_allowlist
    assert "select_path" not in cfg.tool_allowlist
    assert "scratchpad" in cfg.tool_allowlist


# ── model-selection policy (item 6) ─────────────────────────────────────────

def test_model_override_rejected_by_default(tmp_path, monkeypatch):
    # Default: no trusted allowlist → a request-selected model is rejected.
    from anton.cloud_turn.errors import UnsupportedModelError

    monkeypatch.delenv(_MODEL_ALLOWLIST_ENV, raising=False)
    with pytest.raises(UnsupportedModelError, match="claude-opus-4-8"):
        _build(tmp_path, monkeypatch, model="claude-opus-4-8")


def test_model_override_permitted_when_allowlisted(tmp_path, monkeypatch):
    monkeypatch.setenv(_MODEL_ALLOWLIST_ENV, "claude-opus-4-8, other-model")
    _, cfg = _build(tmp_path, monkeypatch, model="claude-opus-4-8")
    assert cfg.settings.planning_model == "claude-opus-4-8"


def test_no_model_uses_trusted_settings_default(tmp_path, monkeypatch):
    # No override → whatever the trusted settings resolved to (unchanged by us).
    monkeypatch.delenv(_MODEL_ALLOWLIST_ENV, raising=False)
    _, cfg = _build(tmp_path, monkeypatch)  # no model in request
    assert cfg.settings.planning_model  # a default is present, not forced by wire


# ── #1 trusted workspace path ────────────────────────────────────────────────

def test_request_cannot_carry_workspace_path():
    # The wire model forbids it (extra="forbid"); a request can't steer the path.
    assert "workspace_path" not in TurnRequestV1.model_fields


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
    """The only enabled file tools (artifacts) resolve strictly under
    <workspace>/artifacts/. A slug with traversal must be rejected, and a
    read-style lookup for an escaping slug must fail closed (no artifact)."""
    from anton.core.artifacts.store import ArtifactStore

    store = ArtifactStore(tmp_path / "artifacts")
    for bad in ("../../etc", "/etc/passwd", "a/../../b", "foo/../../../bar"):
        with pytest.raises(ValueError, match="escapes"):
            store.folder_for(bad)
    # open() on an escaping slug returns None rather than reading outside.
    assert store.open("../../../etc/passwd") is None
    # A normal slug still resolves inside the root.
    assert store.folder_for("my-report").parent == (tmp_path / "artifacts")


def test_artifact_containment_is_canonical_not_lexical(tmp_path):
    """Containment is enforced by real-path resolution, so a nested slug that
    stays INSIDE the root is allowed (create never emits these — _sanitize_slug
    collapses '/'—but open/update must not reject a legitimately-nested path),
    while any slug that resolves outside is rejected."""
    from anton.core.artifacts.store import ArtifactStore

    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    # Nested-but-contained: allowed, and stays under the root.
    nested = store.folder_for("reports/summary")
    assert root.resolve() in nested.resolve().parents
    # A ".." that stays inside is fine; one that escapes is not.
    assert store.folder_for("reports/../summary").resolve() == (root / "summary").resolve()
    with pytest.raises(ValueError, match="escapes"):
        store.folder_for("reports/../../summary")


# ── #2 dotenv is never loaded ────────────────────────────────────────────────

def test_apply_env_to_process_never_called(tmp_path, monkeypatch):
    import anton.workspace as ws_mod
    calls = {"apply_env": 0, "init": 0}
    real_init = ws_mod.Workspace.initialize
    real_apply = ws_mod.Workspace.apply_env_to_process

    def spy_init(self):
        calls["init"] += 1
        return real_init(self)

    def spy_apply(self):
        calls["apply_env"] += 1
        return real_apply(self)

    monkeypatch.setattr(ws_mod.Workspace, "initialize", spy_init)
    monkeypatch.setattr(ws_mod.Workspace, "apply_env_to_process", spy_apply)

    _build(tmp_path, monkeypatch)
    assert calls["init"] == 1            # structure created
    assert calls["apply_env"] == 0       # but .env NOT loaded into process env


def test_cloud_settings_ignore_dotenv_files(tmp_path, monkeypatch):
    """A .env in the AntonSettings chain must NOT seed the cloud session's
    settings — dotenv loading is disabled at the source, not merely unused."""
    from anton.config.settings import AntonSettings

    # Neutralise ambient sources for this key so the assertions turn only on
    # whether a dotenv file is read (env vars would otherwise win over dotenv).
    monkeypatch.delenv("ANTON_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # Positive control: an explicit dotenv IS honoured by AntonSettings.
    envf = tmp_path / "sentinel.env"
    envf.write_text("ANTON_ANTHROPIC_API_KEY=DOTENV_SENTINEL\n")
    assert AntonSettings(_env_file=str(envf)).anthropic_api_key == "DOTENV_SENTINEL"

    # A .env sitting in the workspace/cwd must NOT leak into the cloud session,
    # which builds AntonSettings with the whole dotenv chain disabled.
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("ANTON_ANTHROPIC_API_KEY=DOTENV_SENTINEL\n")
    _, cfg = _build(tmp_path, monkeypatch)
    assert cfg.settings.anthropic_api_key != "DOTENV_SENTINEL"


# ── #3/#4 real enforcement paths ─────────────────────────────────────────────

def _mock_llm():
    """Minimal LLM client sufficient for ChatSession.__init__ + _build_tools
    (mirrors tests/test_tools.py::_make_session)."""
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


def test_final_tool_set_equals_allowlist_after_real_build(tmp_path, monkeypatch):
    """#4 regression: the registry is built LAZILY at turn time, so the allowlist
    must be enforced by the real _build_tools() — not merely configured. Assert
    the EXACT final tool set, so a newly-registered core tool can't slip in."""
    from unittest.mock import MagicMock

    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))
    monkeypatch.setattr(
        llm_client_mod.LLMClient, "from_settings",
        classmethod(lambda cls, settings: _mock_llm()),
    )
    req = TurnRequestV1(run_id="r", attempt_id="a", conversation_id="c", input="hi")
    session = build_cloud_chat_session(req)  # REAL ChatSession (not mocked)
    # Don't spawn a real scratchpad subprocess when the tools get built.
    session._scratchpads = MagicMock(available_packages=[])

    session._build_tools()
    names = {t.name for t in session.tool_registry.get_tool_defs()}

    assert names == set(CLOUD_TOOL_ALLOWLIST), (
        f"cloud tool set drifted from the allowlist: {sorted(names)}"
    )


def test_unsupported_capability_fails_loud(tmp_path, monkeypatch):
    from anton.cloud_turn.errors import UnsupportedCapabilityError

    for field in CapabilitiesV1.model_fields:
        with pytest.raises(UnsupportedCapabilityError, match=field):
            _build(tmp_path, monkeypatch, capabilities={field: True})


async def test_sanitized_scratchpad_cannot_read_secret_env(tmp_path, monkeypatch):
    """#3 enforcement (real subprocess): with the sanitized factory, secret
    sentinels present in the parent env must be UNREADABLE from cell code."""
    # Plant secrets of every category the sanitizer must withhold.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SENTINEL")
    monkeypatch.setenv("ANTON_ANTHROPIC_API_KEY", "ANTON-KEY-SENTINEL")
    monkeypatch.setenv("ANTON_GATEWAY_TOKEN", "GATEWAY-SENTINEL")
    monkeypatch.setenv("DS_POSTGRES_MAIN__PASSWORD", "DATASOURCE-SENTINEL")
    monkeypatch.setenv("MY_UNRELATED_SECRET", "UNRELATED-SENTINEL")

    pad = sanitized_scratchpad_runtime_factory(
        name="cloud-sanitize",
        coding_provider="anthropic",
        coding_model="",
        coding_api_key="",
        coding_base_url="",
        cells=None,
        workspace_path=tmp_path,
    )
    await pad.start()
    try:
        cell = await pad.execute(
            "import os, json\n"
            "keys = ['ANTHROPIC_API_KEY','ANTON_ANTHROPIC_API_KEY',"
            "'ANTON_GATEWAY_TOKEN','DS_POSTGRES_MAIN__PASSWORD','MY_UNRELATED_SECRET']\n"
            "print(json.dumps({k: os.environ.get(k) for k in keys}))"
        )
        assert cell.error is None, cell.error
        import json

        seen = json.loads(cell.stdout.strip())
        assert all(v is None for v in seen.values()), f"secret leaked into cell env: {seen}"
    finally:
        await pad.close()


def _real_session_with_workspace(tmp_path, **cfg_overrides):
    """Build a REAL ChatSession bound to a workspace (so artifact +
    launch_backend tools register), with the scratchpad stubbed so
    _build_tools() doesn't spawn a subprocess."""
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


def test_unknown_allowlist_entry_fails_loud(tmp_path):
    """#1 (item): an allowlist name that matches no built tool must raise, not be
    silently dropped — otherwise the host believes a tool is enabled when it can
    never appear."""
    session = _real_session_with_workspace(
        tmp_path, tool_allowlist=frozenset({"scratchpad", "totally_made_up_tool"})
    )
    with pytest.raises(ValueError, match="totally_made_up_tool"):
        session._build_tools()


def test_tool_allowlist_none_preserves_desktop_tools(tmp_path):
    """Regression: tool_allowlist=None (the desktop default) applies NO filtering
    — the full core set, including launch_backend and select_path, is present."""
    session = _real_session_with_workspace(tmp_path)  # tool_allowlist defaults to None
    assert session._tool_allowlist is None
    session._build_tools()
    names = {t.name for t in session.tool_registry.get_tool_defs()}
    # Tools the cloud allowlist strips must still be here on desktop.
    assert {"launch_backend", "select_path", "scratchpad", "create_artifact"} <= names


def test_sanitized_env_contract(monkeypatch):
    """#4 (item): the sanitized parent env passes through ONLY the documented
    non-secret allowlist (paths / temp dirs / locale / TLS roots), and never a
    credential — and only when the parent actually defines the name."""
    from anton.core.backends.local import (
        _SCRATCHPAD_ENV_ALLOWLIST,
        _sanitized_parent_env,
    )

    # Documented minimums the contract must permit through when present.
    required = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/anton",
        "TMPDIR": "/tmp/x",
        "LANG": "en_US.UTF-8",
        "SSL_CERT_FILE": "/etc/ssl/cert.pem",
    }
    for k, v in required.items():
        assert k in _SCRATCHPAD_ENV_ALLOWLIST, f"{k} must be in the contract"
        monkeypatch.setenv(k, v)

    # Secrets of every category the sanitizer must withhold.
    secrets = {
        "ANTHROPIC_API_KEY": "sk-ant-x",
        "ANTON_MINDS_API_KEY": "anton-x",
        "ANTON_GATEWAY_TOKEN": "gw-x",
        "DS_POSTGRES_MAIN__PASSWORD": "ds-x",
        "OPENAI_API_KEY": "sk-oai-x",
        "SOME_RANDOM_SECRET": "nope",
    }
    for k, v in secrets.items():
        monkeypatch.setenv(k, v)

    env = _sanitized_parent_env()

    for k, v in required.items():
        assert env.get(k) == v, f"required env {k} was dropped"
    for k in secrets:
        assert k not in env, f"secret {k} leaked through the allowlist"
    # Nothing outside the allowlist survives.
    assert set(env) <= set(_SCRATCHPAD_ENV_ALLOWLIST)


async def test_desktop_scratchpad_env_unchanged(tmp_path, monkeypatch):
    """Regression (real subprocess): sanitize_env=False (desktop default via
    local_scratchpad_runtime_factory) STILL inherits the parent env — a planted
    secret remains readable, i.e. the sanitizer is strictly opt-in."""
    from anton.core.backends.local import local_scratchpad_runtime_factory

    monkeypatch.setenv("MY_DESKTOP_SENTINEL", "DESKTOP-VISIBLE")
    pad = local_scratchpad_runtime_factory(
        name="desktop-env",
        coding_provider="anthropic",
        coding_model="",
        coding_api_key="",
        coding_base_url="",
        cells=None,
        workspace_path=tmp_path,
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


# ── item 2: history-delta slicing against the REAL mutation paths ────────────

async def test_output_survives_real_compaction_and_preserves_request(tmp_path, monkeypatch):
    """Drive the REAL _append_history + _summarize_history (the exact history
    mutations turn_stream uses — compaction reassigns the list and collapses the
    old prefix) and prove current-turn collection is still correct."""
    from anton.cloud_turn.messages import final_assistant_text, turn_output_messages

    monkeypatch.setenv(_WORKSPACE_PATH_ENV, str(tmp_path))
    monkeypatch.setattr(
        llm_client_mod.LLMClient, "from_settings",
        classmethod(lambda cls, settings: _mock_llm()),
    )
    # Input history long enough to trigger real compaction (>= 6 messages).
    req_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(8)
    ]
    req = TurnRequestV1(
        run_id="r", attempt_id="a", conversation_id="c",
        input="CURRENT INPUT", history=req_history,
    )
    request_history_before = [m.model_copy(deep=True) for m in req.history]

    session = build_cloud_chat_session(req)          # REAL ChatSession
    pre_turn_messages = list(session.history)        # hold refs → stable ids

    # Replay what turn_stream does: echo user input, compact mid-turn, then emit.
    session._append_history({"role": "user", "content": "CURRENT INPUT"})
    await session._summarize_history()               # REAL compaction (rewrites list)
    session._append_history({"role": "assistant", "content": [
        {"type": "text", "text": "working"},
        {"type": "tool_use", "id": "t1", "name": "scratchpad", "input": {}},
    ]})
    session._append_history({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
    ]})
    session._append_history({"role": "assistant", "content": "FINAL"})

    out = turn_output_messages(session.history, pre_turn_messages)

    # Only current-turn generated messages, in order; echo + summary excluded.
    assert [m.role for m in out] == ["assistant", "user", "assistant"]
    assert out[0].content[1].type == "tool_use"
    assert out[1].content[0].type == "tool_result"
    assert final_assistant_text(out) == "FINAL"
    assert all(
        not (isinstance(m.content, str) and m.content == "CURRENT INPUT") for m in out
    )
    # Generated begin exactly after the boundary → the current-turn tail of the
    # final session history, in the same order.
    tail = session.history[-3:]
    assert [m["role"] for m in tail] == ["assistant", "user", "assistant"]
    assert tail[-1]["content"] == "FINAL"
    # The request's own history object is untouched by the turn's mutations.
    assert req.history == request_history_before
