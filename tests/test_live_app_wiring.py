"""Where live apps meet the rest of anton: the in-process launcher's refusal,
the `launch_backend` tool, the generator's run step and checks, and the
relaunch after a cell. With live apps off, each answers as before."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.artifacts import backend_launcher
from anton.core.artifacts import live_app_slots as las
from anton.core.artifacts.live_app_slots import LiveAppLaunchError, LiveAppLaunchResult, MissingConnection, ProbeResult
from anton.core.backends.sandbox import BACKENDS_OFF_MESSAGE, CellSandbox
from anton.core.tools import tool_handlers
from anton.core.tools.generate_artifact import orchestrator, prompts, verifiers
from anton.core.tools.generate_artifact.state import GenState

_SLOTS = "http://live-app-1:8001,http://live-app-2:8001"


@pytest.fixture
def policy(tmp_path: Path) -> CellSandbox:
    project = tmp_path / "project"
    for folder in (project, tmp_path / "home", tmp_path / "tmp"):
        folder.mkdir()
    return CellSandbox(project_dir=project, home_dir=tmp_path / "home", tmp_dir=tmp_path / "tmp", user_name="bob")


@pytest.fixture
def live_apps_on(monkeypatch):
    monkeypatch.setenv(las.SLOTS_ENV_VAR, _SLOTS)


@pytest.fixture
def live_apps_off(monkeypatch):
    monkeypatch.delenv(las.SLOTS_ENV_VAR, raising=False)


class _Pool:
    def __init__(self, policy: CellSandbox | None):
        self.cell_sandbox = policy
        self.calls: list[str] = []

    async def venv_python(self, name):
        self.calls.append("venv_python")
        return "/venv/bin/python"

    async def get_or_create(self, name):
        self.calls.append("get_or_create")
        return SimpleNamespace()


def _result(**overrides) -> LiveAppLaunchResult:
    fields = {"slot": 1, "address": "http://app1.view.localhost:8080/", "slug": "app"}
    fields.update(overrides)
    return LiveAppLaunchResult(**fields)


# ── The in-process launcher never starts a backend on a sandboxed host ───────


async def test_with_live_apps_on_the_in_process_launcher_points_at_the_slot(tmp_path, policy, live_apps_on):
    folder = tmp_path / "app"
    folder.mkdir()
    (folder / "backend.py").write_text("", encoding="utf-8")
    pool = _Pool(policy)
    result = await backend_launcher.launch_artifact_backend(
        slug="app", artifact_folder=folder, scratchpad_pool=pool, tracked_backends={},
    )
    assert result == backend_launcher.SLOT_ONLY_MESSAGE
    assert pool.calls == []


async def test_with_live_apps_off_the_in_process_launcher_refuses_as_before(tmp_path, policy, live_apps_off):
    folder = tmp_path / "app"
    folder.mkdir()
    (folder / "backend.py").write_text("", encoding="utf-8")
    result = await backend_launcher.launch_artifact_backend(
        slug="app", artifact_folder=folder, scratchpad_pool=_Pool(policy), tracked_backends={},
    )
    assert result == BACKENDS_OFF_MESSAGE


# ── The launch_backend tool ──────────────────────────────────────────────────


def _tool_session(tmp_path, policy, monkeypatch):
    folder = tmp_path / "artifacts" / "app"
    folder.mkdir(parents=True)
    updates: list[dict] = []
    store = SimpleNamespace(
        open=lambda slug: SimpleNamespace(slug=slug, datasources=["declared"]),
        folder_for=lambda slug: folder,
        update=lambda slug, **kw: updates.append(kw),
    )
    monkeypatch.setattr(tool_handlers, "resolve_artifact_store", lambda session: store)
    session = SimpleNamespace(_scratchpads=_Pool(policy), _data_vault=None)
    return session, folder, updates


async def test_the_tool_launches_into_a_slot_and_returns_the_address(tmp_path, policy, live_apps_on, monkeypatch):
    session, folder, updates = _tool_session(tmp_path, policy, monkeypatch)
    calls: list[dict] = []

    async def launch(session_arg, **kwargs):
        calls.append(kwargs)
        return _result(missing=(MissingConnection(engine="snowflake", name="sales", reason="not_approved"),))

    monkeypatch.setattr(las, "launch_live_app", launch)
    outcome = await tool_handlers.handle_launch_backend(session, {"slug": "app"})
    assert outcome.ok is True
    assert "`http://app1.view.localhost:8080/`" in outcome.content
    assert "`snowflake/sales` (not approved for live apps by the operator)" in outcome.content
    assert "127.0.0.1" not in outcome.content and "pid" not in outcome.content
    assert calls == [{"slug": "app", "folder": folder, "datasources": ["declared"]}]
    assert updates == []  # no port goes into metadata.json
    assert session._scratchpads.calls == []


async def test_the_tool_relays_a_refused_card_as_the_persons_decision(tmp_path, policy, live_apps_on, monkeypatch):
    session, _, _ = _tool_session(tmp_path, policy, monkeypatch)

    async def launch(session_arg, **kwargs):
        raise LiveAppLaunchError(las.NOT_APPROVED_MESSAGE, cause="launch_not_approved")

    monkeypatch.setattr(las, "launch_live_app", launch)
    outcome = await tool_handlers.handle_launch_backend(session, {"slug": "app"})
    assert outcome.ok is True
    assert outcome.content == las.NOT_APPROVED_MESSAGE


async def test_the_tool_relays_a_slot_refusal_as_a_failure(tmp_path, policy, live_apps_on, monkeypatch):
    session, _, _ = _tool_session(tmp_path, policy, monkeypatch)

    async def launch(session_arg, **kwargs):
        raise LiveAppLaunchError(las.NO_SLOT_MESSAGE, cause="no_slot")

    monkeypatch.setattr(las, "launch_live_app", launch)
    outcome = await tool_handlers.handle_launch_backend(session, {"slug": "app"})
    assert outcome.ok is False
    assert outcome.reason == "live_app_unavailable"
    assert las.NO_SLOT_MESSAGE in outcome.content


def test_every_launch_failure_has_a_tiered_tool_reason():
    from typing import get_args

    from anton.core.root_cause import _SENTINEL_REASONS

    assert set(tool_handlers._LIVE_APP_FAILURE_REASONS) == set(get_args(las.LaunchFailure))
    assert set(tool_handlers._LIVE_APP_FAILURE_REASONS.values()) <= set(_SENTINEL_REASONS)


@pytest.mark.parametrize("tc_input", [{"path": "server.py"}, {"extra_args": ["--debug"]}])
async def test_the_tool_refuses_what_a_slot_wont_run(tmp_path, policy, live_apps_on, monkeypatch, tc_input):
    session, _, _ = _tool_session(tmp_path, policy, monkeypatch)

    async def launch(session_arg, **kwargs):
        raise AssertionError("must not launch")

    monkeypatch.setattr(las, "launch_live_app", launch)
    outcome = await tool_handlers.handle_launch_backend(session, {"slug": "app", **tc_input})
    assert outcome.ok is False
    assert "`path` and `extra_args`" in outcome.content


async def test_with_live_apps_off_the_tool_refuses_as_before(tmp_path, policy, live_apps_off, monkeypatch):
    session, _, _ = _tool_session(tmp_path, policy, monkeypatch)
    outcome = await tool_handlers.handle_launch_backend(session, {"slug": "app"})
    assert BACKENDS_OFF_MESSAGE in outcome.content
    assert outcome.ok is False


# ── The generator ────────────────────────────────────────────────────────────


def _gen_state(tmp_path, policy, **fields) -> GenState:
    session = SimpleNamespace(_scratchpads=SimpleNamespace(cell_sandbox=policy), _data_vault=None)
    return GenState(session=session, artifact_type="fullstack-stateless-app", artifact_path=tmp_path,
                    slug="app", brief="b", is_fullstack=True, **fields)


@pytest.fixture
def gen_store(monkeypatch):
    store = SimpleNamespace(open=lambda slug: SimpleNamespace(datasources=["declared"]))
    monkeypatch.setattr(tool_handlers, "resolve_artifact_store", lambda session: store)
    return store


async def test_with_live_apps_on_a_fullstack_run_gets_past_the_sandbox_refusal(tmp_path, policy, live_apps_on,
                                                                             monkeypatch):
    from anton.core.tools.generate_artifact.discovery import checkpoint as cp

    reached: list[str] = []

    async def stop_here(state, **_):
        reached.append("data")
        return "stopped for the test"

    monkeypatch.setattr(orchestrator, "_data_phase", stop_here)
    state = _gen_state(tmp_path, policy)
    assert await orchestrator.run(state, entry=cp.ENTRY_SPEC) == "stopped for the test"
    assert reached == ["data"]


async def test_the_run_step_launches_in_a_slot_and_probes_through_it(tmp_path, policy, live_apps_on, monkeypatch,
                                                                     gen_store):
    calls: list[dict] = []
    api = '{"paths": {"/api/health": {"get": {}}, "/api/items": {"get": {}}, "/api/item/{id}": {"get": {}}}}'

    async def launch(session, **kwargs):
        calls.append(kwargs)
        return _result(probe=(ProbeResult(path="/api/health", status=200), ProbeResult(path="/api/items", status=200)))

    monkeypatch.setattr(orchestrator, "_launch_live_app", launch)
    state = _gen_state(tmp_path, policy, api_spec=api)
    assert await orchestrator._run_and_verify_app(state) is None
    assert calls[0]["probe_paths"] == ["/api/health", "/api/health", "/api/items"]
    assert calls[0]["datasources"] == ["declared"]
    assert state.app_url == "http://app1.view.localhost:8080/" and state.app_port is None
    assert state.browser_status == "unverified"
    assert any("opens only in its owner's browser" in note for note in state.checks_skipped)
    shell = orchestrator._result_shell(state)
    assert shell["url"] == "http://app1.view.localhost:8080/"
    assert "port" not in shell
    assert "`http://app1.view.localhost:8080/`" in shell["live_app"]


async def test_a_failed_probe_regenerates_the_backend_and_launches_again(tmp_path, policy, live_apps_on, monkeypatch,
                                                                         gen_store):
    statuses = iter([500, 200])
    regenerated: list[str] = []

    async def launch(session, **kwargs):
        return _result(probe=(ProbeResult(path="/api/health", status=next(statuses)),))

    async def regenerate(state, extra_context=""):
        regenerated.append(extra_context)
        return None

    monkeypatch.setattr(orchestrator, "_launch_live_app", launch)
    monkeypatch.setattr(orchestrator, "_gen_verify_backend", regenerate)
    state = _gen_state(tmp_path, policy)
    assert await orchestrator._run_and_verify_app(state) is None
    assert len(regenerated) == 1
    assert "health check GET /api/health returned 500" in regenerated[0]


@pytest.mark.parametrize("reason", ["launch_not_approved", "no_slot", "slots_full", "config", "launch_files"])
async def test_a_launch_a_new_backend_cant_fix_ends_the_run(tmp_path, policy, live_apps_on, monkeypatch, gen_store,
                                                            reason):
    async def launch(session, **kwargs):
        raise LiveAppLaunchError(f"Not launched: {reason}.", cause=reason)

    async def must_not_regenerate(state, extra_context=""):
        raise AssertionError("must not regenerate")

    monkeypatch.setattr(orchestrator, "_launch_live_app", launch)
    monkeypatch.setattr(orchestrator, "_gen_verify_backend", must_not_regenerate)
    state = _gen_state(tmp_path, policy)
    assert await orchestrator._run_and_verify_app(state) == f"Not launched: {reason}."
    assert state.error == f"Not launched: {reason}."


def test_with_live_apps_on_the_verifier_fails_a_script_from_another_host():
    page = (
        '<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1.0">'
        '<meta name="api-base" content="">'
        '<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>'
        '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>'
        '<script src="//cdn.jsdelivr.net/npm/d3@7"></script>'
        '<script type="module">import x from "https://esm.sh/x";</script>'
        "</head><body></body></html>"
    )
    errors = verifiers.verify_frontend(page, is_fullstack=True, live_app=True).errors
    (blocked,) = [e for e in errors if e.startswith("Script from a host a live app does not load: ")]
    assert "unpkg.com/leaflet" in blocked and "//cdn.jsdelivr.net/npm/d3@7" in blocked and "esm.sh" in blocked
    assert "echarts" not in blocked
    off = verifiers.verify_frontend(page, is_fullstack=True).errors
    assert not any(e.startswith("Script from a host a live app") for e in off)


def test_a_page_with_only_jsdelivr_scripts_passes_the_live_app_rule():
    page = (
        '<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1.0">'
        '<meta name="api-base" content="">'
        '<script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.3.3"></script>'
        '<script src="/static/app.js"></script>'
        "</head><body></body></html>"
    )
    errors = verifiers.verify_frontend(page, is_fullstack=True, live_app=True).errors
    assert not any(e.startswith("Script from a host a live app") for e in errors)


async def test_the_generator_holds_a_live_apps_page_to_the_script_rule(tmp_path, policy, live_apps_on):
    assert orchestrator._runs_live_apps(_gen_state(tmp_path, policy)) is True


async def test_without_a_sandbox_live_apps_dont_apply(tmp_path, live_apps_on):
    state = _gen_state(tmp_path, None)
    assert orchestrator._runs_live_apps(state) is False


def test_prompts_and_tool_definitions_dont_depend_on_the_setting(tmp_path, monkeypatch):
    from anton.core.tools import tool_defs

    def snapshot() -> list:
        built = [
            prompts.build_subagent_system_prompt(tmp_path),
            prompts.build_frontend_system_prompt(tmp_path),
            prompts.build_backend_system_prompt(tmp_path, stateless=True),
            prompts.build_backend_system_prompt(tmp_path, stateless=False),
        ]
        tools = [
            (t.name, t.description, repr(t.input_schema))
            for t in vars(tool_defs).values() if type(t).__name__ == "ToolDef"
        ]
        return [built, sorted(tools)]

    monkeypatch.delenv(las.SLOTS_ENV_VAR, raising=False)
    off = snapshot()
    monkeypatch.setenv(las.SLOTS_ENV_VAR, _SLOTS)
    assert snapshot() == off


# ── After a cell ─────────────────────────────────────────────────────────────


async def test_a_cell_that_edits_a_running_app_relaunches_it(tmp_path, policy, live_apps_on, monkeypatch):
    root = tmp_path / "artifacts"
    folder = root / "app"
    (folder / "static").mkdir(parents=True)
    (folder / "metadata.json").write_text("{}", encoding="utf-8")
    (folder / "static" / "index.html").write_text("v1", encoding="utf-8")
    (folder / "prd.md").write_text("p", encoding="utf-8")
    store = SimpleNamespace(root=root)
    before = tool_handlers.snapshot_existing_artifacts(store)
    (folder / "static" / "index.html").write_text("version two", encoding="utf-8")
    asked: list[list[str]] = []

    async def relaunch(session, store_arg, slugs):
        asked.append(list(slugs))
        return ["Relaunched live app `app` at `http://app1.view.localhost:8080/` with this cell's changes."]

    monkeypatch.setattr(las, "relaunch_changed_apps", relaunch)
    session = SimpleNamespace(_scratchpads=SimpleNamespace(cell_sandbox=policy))
    lines = await tool_handlers._relaunch_live_apps(session, store, before)
    assert asked == [["app"]]
    text = tool_handlers.cell_content_with_lint("[output]\nok", tool_handlers.ArtifactLint(live_app_lines=lines),
                                                cell_ok=True)
    assert text.endswith("\n\n[live app]\nRelaunched live app `app` at `http://app1.view.localhost:8080/` "
                         "with this cell's changes.")


async def test_a_cell_that_edits_only_the_prd_relaunches_nothing(tmp_path, policy, live_apps_on, monkeypatch):
    root = tmp_path / "artifacts"
    folder = root / "app"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text("{}", encoding="utf-8")
    (folder / "prd.md").write_text("p", encoding="utf-8")
    store = SimpleNamespace(root=root)
    before = tool_handlers.snapshot_existing_artifacts(store)
    (folder / "prd.md").write_text("a longer prd", encoding="utf-8")
    asked: list[list[str]] = []

    async def relaunch(session, store_arg, slugs):
        asked.append(list(slugs))
        return []

    monkeypatch.setattr(las, "relaunch_changed_apps", relaunch)
    session = SimpleNamespace(_scratchpads=SimpleNamespace(cell_sandbox=policy))
    assert await tool_handlers._relaunch_live_apps(session, store, before) == []
    assert asked == [[]]


async def test_with_live_apps_off_a_cell_relaunches_nothing(tmp_path, policy, live_apps_off, monkeypatch):
    async def must_not_run(*args):
        raise AssertionError("must not relaunch")

    monkeypatch.setattr(las, "relaunch_changed_apps", must_not_run)
    session = SimpleNamespace(_scratchpads=SimpleNamespace(cell_sandbox=policy))
    assert await tool_handlers._relaunch_live_apps(session, SimpleNamespace(root=tmp_path), {}) == []
    text = tool_handlers.cell_content_with_lint("[output]\nok", tool_handlers.ArtifactLint(), cell_ok=True)
    assert text == "[output]\nok"


def test_the_session_keeps_the_hosts_live_app_object():
    from anton.core.session import ChatSession, ChatSessionConfig
    from tests.conftest import make_mock_llm

    host = SimpleNamespace(launcher=None)
    assert ChatSession(ChatSessionConfig(llm_client=make_mock_llm(), live_app_host=host))._live_app_host is host
    assert ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))._live_app_host is None
