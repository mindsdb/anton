"""Where live apps meet the rest of anton: the in-process launcher's refusal,
the `launch_backend` tool, the generator's run step and checks, and the
relaunch after a cell. With live apps off, each answers as before."""
from __future__ import annotations

import json
import os
import subprocess
import sys
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
        return _result(missing=(MissingConnection(engine="snowflake", name="sales", reason="not_approved"),),
                       probe=(ProbeResult(path="/api/health", status=200),))

    monkeypatch.setattr(las, "launch_live_app", launch)
    outcome = await tool_handlers.handle_launch_backend(session, {"slug": "app", "health_path": "api/health"})
    assert outcome.ok is True
    assert "`http://app1.view.localhost:8080/`" in outcome.content
    assert "`snowflake/sales` (not approved for live apps by the operator)" in outcome.content
    assert "127.0.0.1" not in outcome.content and "pid" not in outcome.content
    assert calls == [{"slug": "app", "folder": folder, "datasources": ["declared"], "probe_paths": ["/api/health"]}]
    assert updates == []  # no port goes into metadata.json
    assert session._scratchpads.calls == []
    # The details the joints fix: no port, no pid, and no value in `missing`.
    assert json.loads(outcome.content)["details"] == {
        "slug": "app", "slot": 1, "address": "http://app1.view.localhost:8080/",
        "missing": [{"engine": "snowflake", "name": "sales", "reason": "not_approved"}],
    }


@pytest.mark.parametrize("status, ok, said", [
    (0, False, "didn't answer GET /"),
    (500, True, "Its health check, GET /, answered 500: read backend.log."),
    (404, True, "is running at"),
])
async def test_the_tool_reads_its_health_probe_as_the_local_launcher_does(tmp_path, policy, live_apps_on,
                                                                          monkeypatch, status, ok, said):
    session, _, _ = _tool_session(tmp_path, policy, monkeypatch)

    async def launch(session_arg, **kwargs):
        return _result(probe=(ProbeResult(path="/", status=status),), log_tail="boot")

    monkeypatch.setattr(las, "launch_live_app", launch)
    outcome = await tool_handlers.handle_launch_backend(session, {"slug": "app"})
    assert outcome.ok is ok
    assert said in outcome.content
    # The app runs in its slot either way, so the result says where.
    assert "`http://app1.view.localhost:8080/`" in outcome.content


async def test_a_probe_the_slot_couldnt_run_is_named_not_blamed_on_the_app(tmp_path, policy, live_apps_on,
                                                                          monkeypatch):
    session, _, _ = _tool_session(tmp_path, policy, monkeypatch)

    async def launch(session_arg, **kwargs):
        return _result(probe=(ProbeResult(path="/", status=0),), probe_error="the slot gave no usable answer (X)")

    monkeypatch.setattr(las, "launch_live_app", launch)
    outcome = await tool_handlers.handle_launch_backend(session, {"slug": "app"})
    assert outcome.ok is False
    assert "its health check couldn't run: the slot gave no usable answer (X)" in outcome.content


@pytest.mark.parametrize("health_path", ["//evil.example.com/", "/a b", "/a#b"])
async def test_a_health_path_the_slot_wont_probe_is_refused_before_the_launch(tmp_path, policy, live_apps_on,
                                                                             monkeypatch, health_path):
    session, _, _ = _tool_session(tmp_path, policy, monkeypatch)

    async def launch(session_arg, **kwargs):
        raise AssertionError("must not launch")

    monkeypatch.setattr(las, "launch_live_app", launch)
    outcome = await tool_handlers.handle_launch_backend(session, {"slug": "app", "health_path": health_path})
    assert outcome.ok is False and outcome.reason == "live_app_invalid_launch"


async def test_a_health_timeout_says_it_doesnt_apply_in_a_slot(tmp_path, policy, live_apps_on, monkeypatch):
    session, _, _ = _tool_session(tmp_path, policy, monkeypatch)

    async def launch(session_arg, **kwargs):
        return _result(probe=(ProbeResult(path="/", status=200),))

    monkeypatch.setattr(las, "launch_live_app", launch)
    outcome = await tool_handlers.handle_launch_backend(session, {"slug": "app", "health_timeout": 60})
    assert "`health_timeout` doesn't apply in an app slot" in outcome.content
    plain = await tool_handlers.handle_launch_backend(session, {"slug": "app"})
    assert "health_timeout" not in plain.content


async def test_the_tool_relays_a_refused_card_as_the_persons_decision(tmp_path, policy, live_apps_on, monkeypatch):
    session, _, _ = _tool_session(tmp_path, policy, monkeypatch)
    refused = las.not_approved_message("cancelled", running=True)

    async def launch(session_arg, **kwargs):
        raise LiveAppLaunchError(refused, cause="launch_not_approved")

    monkeypatch.setattr(las, "launch_live_app", launch)
    outcome = await tool_handlers.handle_launch_backend(session, {"slug": "app"})
    assert outcome.ok is True
    assert outcome.content == refused


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
    connections = tmp_path / "connections.json"
    connections.write_text('{"version": 1, "connections": []}', encoding="utf-8")
    monkeypatch.setenv(las.CONNECTIONS_FILE_ENV_VAR, str(connections))
    state = _gen_state(tmp_path, policy)
    state.session._live_app_host = SimpleNamespace(launcher=None, launch_token="t" * 64)
    assert await orchestrator.run(state, entry=cp.ENTRY_SPEC) == "stopped for the test"
    assert reached == ["data"]


async def test_the_run_step_launches_in_a_slot_and_probes_through_it(tmp_path, policy, live_apps_on, monkeypatch,
                                                                     gen_store):
    calls: list[dict] = []
    api = '{"paths": {"/api/health": {"get": {}}, "/api/items": {"get": {}}, "/api/item/{id}": {"get": {}}}}'

    async def launch(session, **kwargs):
        calls.append(kwargs)
        return _result(probe=(ProbeResult(path="/api/health", status=200), ProbeResult(path="/api/items", status=200)))

    monkeypatch.setattr(las, "launch_live_app", launch)
    state = _gen_state(tmp_path, policy, api_spec=api)
    assert await orchestrator._run_and_verify_app(state) is None
    assert calls[0]["probe_paths"] == ["/api/health", "/api/items"]
    assert calls[0]["datasources"] == ["declared"]
    assert state.app_url == "http://app1.view.localhost:8080/" and state.app_port is None
    assert state.browser_status == "unverified"
    assert any("opens only in its owner's browser" in note for note in state.checks_skipped)
    shell = orchestrator._result_shell(state)
    assert shell["url"] == "http://app1.view.localhost:8080/"
    assert "port" not in shell
    assert "`http://app1.view.localhost:8080/`" in shell["live_app"]


async def test_the_run_step_probes_no_more_paths_than_a_slot_takes(tmp_path, policy, live_apps_on, monkeypatch,
                                                                   gen_store):
    calls: list[dict] = []
    api = '{"paths": {' + ", ".join(f'"/api/r{i}": {{"get": {{}}}}' for i in range(40)) + "}}"

    async def launch(session, **kwargs):
        calls.append(kwargs)
        return _result(probe=tuple(ProbeResult(path=p, status=200) for p in kwargs["probe_paths"]))

    monkeypatch.setattr(las, "launch_live_app", launch)
    assert await orchestrator._run_and_verify_app(_gen_state(tmp_path, policy, api_spec=api)) is None
    sent = calls[0]["probe_paths"]
    assert len(sent) == las.MAX_PROBE_PATHS and sent[0] == "/api/health"


async def test_a_failed_probe_regenerates_the_backend_and_launches_again(tmp_path, policy, live_apps_on, monkeypatch,
                                                                         gen_store):
    statuses = iter([500, 200])
    regenerated: list[str] = []

    async def launch(session, **kwargs):
        return _result(probe=(ProbeResult(path="/api/health", status=next(statuses)),))

    async def regenerate(state, extra_context=""):
        regenerated.append(extra_context)
        return None

    monkeypatch.setattr(las, "launch_live_app", launch)
    monkeypatch.setattr(orchestrator, "_gen_verify_backend", regenerate)
    state = _gen_state(tmp_path, policy)
    assert await orchestrator._run_and_verify_app(state) is None
    assert len(regenerated) == 1
    assert "health check GET /api/health returned 500" in regenerated[0]


@pytest.mark.parametrize("cause", ["app_not_ready", "missing_package"])
async def test_a_launch_a_new_backend_can_fix_regenerates_it(tmp_path, policy, live_apps_on, monkeypatch, gen_store,
                                                             cause):
    failures = iter([LiveAppLaunchError(f"Not launched: {cause}.", cause=cause, log_tail="boom")])
    regenerated: list[str] = []

    async def launch(session, **kwargs):
        failure = next(failures, None)
        if failure is not None:
            raise failure
        return _result(probe=(ProbeResult(path="/api/health", status=200),))

    async def regenerate(state, extra_context=""):
        regenerated.append(extra_context)
        return None

    monkeypatch.setattr(las, "launch_live_app", launch)
    monkeypatch.setattr(orchestrator, "_gen_verify_backend", regenerate)
    assert await orchestrator._run_and_verify_app(_gen_state(tmp_path, policy)) is None
    (context,) = regenerated
    assert f"Not launched: {cause}." in context and context.count("boom") == 1


@pytest.mark.parametrize("reason", ["launch_not_approved", "no_slot", "slots_full", "config", "launch_files"])
async def test_a_launch_a_new_backend_cant_fix_ends_the_run_as_not_launched(tmp_path, policy, live_apps_on,
                                                                            monkeypatch, gen_store, reason):
    async def launch(session, **kwargs):
        raise LiveAppLaunchError(f"Not launched: {reason}.", cause=reason)

    async def must_not_regenerate(state, extra_context=""):
        raise AssertionError("must not regenerate")

    monkeypatch.setattr(las, "launch_live_app", launch)
    monkeypatch.setattr(orchestrator, "_gen_verify_backend", must_not_regenerate)
    state = _gen_state(tmp_path, policy)
    assert await orchestrator._run_and_verify_app(state) is orchestrator.NOT_LAUNCHED
    assert state.error is None and state.launch_failure.cause == reason
    result = orchestrator._not_launched(state)
    assert result["status"] == "not_launched" and result["cause"] == reason
    assert result["detail"] == f"Not launched: {reason}."
    assert "files_written" in result and "port" not in result


@pytest.mark.parametrize("cause, ok, reason", [
    ("launch_not_approved", True, "live_app_not_approved"),
    ("slots_full", False, "live_app_slots_full"),
    ("no_slot", False, "live_app_unavailable"),
])
def test_a_generation_that_didnt_launch_reads_as_launch_backend_does(cause, ok, reason):
    assert tool_handlers._not_launched_verdict({"cause": cause}) == (ok, reason)
    assert "launch_backend" in tool_handlers._STATUS_INSTRUCTIONS["not_launched"]


async def test_a_run_that_gives_up_stops_the_app_it_launched(tmp_path, policy, live_apps_on, monkeypatch, gen_store):
    stopped: list[Path] = []

    async def launch(session, **kwargs):
        return _result(probe=(ProbeResult(path="/api/health", status=500),), log_tail="Traceback: boom")

    async def regenerate(state, extra_context=""):
        return None

    async def stop(session, *, folder):
        stopped.append(folder)

    monkeypatch.setattr(las, "launch_live_app", launch)
    monkeypatch.setattr(las, "stop_live_app", stop)
    monkeypatch.setattr(orchestrator, "_gen_verify_backend", regenerate)
    state = _gen_state(tmp_path, policy)
    error = await orchestrator._run_and_verify_app(state)
    assert error.startswith("Application failed to launch: health check GET /api/health returned 500")
    assert error.count("Traceback: boom") == 1
    assert stopped == [tmp_path]
    assert state.app_url is None and state.live_app is None


@pytest.mark.parametrize("stop", ["Error: the backend could not be rebuilt.", orchestrator.OVER_BUDGET],
                         ids=["error", "over-budget"])
async def test_a_regeneration_that_ends_the_run_stops_the_app_it_launched(tmp_path, policy, live_apps_on,
                                                                          monkeypatch, gen_store, stop):
    stopped: list[Path] = []

    async def launch(session, **kwargs):
        return _result(probe=(ProbeResult(path="/api/health", status=500),))

    async def regenerate(state, extra_context=""):
        return stop

    async def record_stop(session, *, folder):
        stopped.append(folder)

    monkeypatch.setattr(las, "launch_live_app", launch)
    monkeypatch.setattr(las, "stop_live_app", record_stop)
    monkeypatch.setattr(orchestrator, "_gen_verify_backend", regenerate)
    state = _gen_state(tmp_path, policy)
    assert await orchestrator._run_and_verify_app(state) == stop
    assert stopped == [tmp_path]
    result = orchestrator._stopped_over_budget(state, "the backend could not be rebuilt in time")
    assert "url" not in result and "live_app" not in result


@pytest.mark.parametrize("cause, still_runs", [("launch_not_approved", True), ("slot_unavailable", False)],
                         ids=["card-refused", "stopped-after-claim"])
async def test_a_relaunch_that_fails_after_a_launched_attempt_sends_no_address(tmp_path, policy, live_apps_on,
                                                                               monkeypatch, gen_store, cause,
                                                                               still_runs):
    """The first attempt's app failed its probe; the run says so, and never
    hands its address over as a working app's."""
    address = "http://app1.view.localhost:8080/"
    attempts = iter([
        _result(probe=(ProbeResult(path="/api/health", status=500),)),
        LiveAppLaunchError(f"Not launched: {cause}.", cause=cause),
    ])

    async def launch(session, **kwargs):
        step = next(attempts)
        if isinstance(step, Exception):
            raise step
        return step

    async def regenerate(state, extra_context=""):
        return None

    async def running_address(session, *, folder):
        return address if still_runs else None

    async def must_not_stop(session, *, folder):
        raise AssertionError("the launch failure leaves the app as it was")

    monkeypatch.setattr(las, "launch_live_app", launch)
    monkeypatch.setattr(las, "running_app_address", running_address)
    monkeypatch.setattr(las, "stop_live_app", must_not_stop)
    monkeypatch.setattr(orchestrator, "_gen_verify_backend", regenerate)
    state = _gen_state(tmp_path, policy)
    assert await orchestrator._run_and_verify_app(state) is orchestrator.NOT_LAUNCHED
    result = orchestrator._not_launched(state)
    assert result["status"] == "not_launched" and result["cause"] == cause
    assert "url" not in result and "live_app" not in result
    assert result["detail"].startswith(f"Not launched: {cause}. The app this run launched earlier ")
    assert result["detail"].endswith("failed its check: health check GET /api/health returned 500 (expected 200).")
    assert (f"still runs at `{address}`" in result["detail"]) is still_runs


async def test_a_probe_the_slot_couldnt_run_ends_the_run_without_a_new_backend(tmp_path, policy, live_apps_on,
                                                                              monkeypatch, gen_store):
    async def launch(session, **kwargs):
        return _result(probe=(ProbeResult(path="/api/health", status=0),), probe_error="the slot gave no usable answer")

    async def must_not_regenerate(state, extra_context=""):
        raise AssertionError("must not regenerate")

    async def stop(session, *, folder):
        pass

    monkeypatch.setattr(las, "launch_live_app", launch)
    monkeypatch.setattr(las, "stop_live_app", stop)
    monkeypatch.setattr(orchestrator, "_gen_verify_backend", must_not_regenerate)
    error = await orchestrator._run_and_verify_app(_gen_state(tmp_path, policy))
    assert "the app slot couldn't probe the app: the slot gave no usable answer" in error
    assert "health check GET /api/health returned" not in error


async def test_the_run_step_probes_only_paths_the_slot_takes(tmp_path, policy, live_apps_on, monkeypatch, gen_store):
    calls: list[dict] = []
    api = '{"paths": {"/api/items": {"get": {}}, "/api/a b": {"get": {}}, "//x": {"get": {}}}}'

    async def launch(session, **kwargs):
        calls.append(kwargs)
        return _result(probe=tuple(ProbeResult(path=p, status=200) for p in kwargs["probe_paths"]))

    monkeypatch.setattr(las, "launch_live_app", launch)
    state = _gen_state(tmp_path, policy, api_spec=api)
    assert await orchestrator._run_and_verify_app(state) is None
    assert calls[0]["probe_paths"] == ["/api/health", "/api/items"]
    assert any("an app slot probes only plain paths" in note for note in state.checks_skipped)


async def test_with_no_host_a_fullstack_run_is_refused_before_anything_runs(tmp_path, policy, live_apps_on,
                                                                           monkeypatch):
    async def must_not_run(*a, **kw):
        raise AssertionError("the pipeline must not start")

    monkeypatch.setattr(orchestrator, "run_discovery", must_not_run)
    state = _gen_state(tmp_path, policy)
    assert await orchestrator.run(state) == las.NO_HOST_MESSAGE


async def test_a_run_keeps_the_after_cell_relaunch_off_its_app(tmp_path, policy, live_apps_on, monkeypatch):
    seen: list[set] = []

    async def body(state, *, entry):
        seen.append(set(state.session._live_apps_building))
        return "done"

    monkeypatch.setattr(orchestrator, "_run", body)
    state = _gen_state(tmp_path, policy)
    state.session._live_apps_building = set()
    assert await orchestrator.run(state) == "done"
    assert seen == [{"app"}] and state.session._live_apps_building == set()


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


@pytest.mark.parametrize("markup, blocked", [
    ('<script type="importmap">{"imports": {"a": "https://unpkg.com/a.js", "b": "./b.js"}}</script>',
     "https://unpkg.com/a.js"),
    ('<script type="module">export { x } from "https://esm.sh/x";</script>', "https://esm.sh/x"),
    ("<script type=module>import(`https://esm.sh/y`)</script>", "https://esm.sh/y"),
    ('<script src="https://cdn.jsdelivr.net.evil.com/x.js"></script>', "cdn.jsdelivr.net.evil.com"),
    ('<script src="https://cdn.jsdelivr.net@evil.com/x.js"></script>', "evil.com"),
])
def test_the_live_app_rule_reads_every_way_a_page_loads_a_script(markup, blocked):
    page = f'<!doctype html><html><head><meta name="viewport" content="width=device-width">{markup}</head></html>'
    errors = verifiers.verify_frontend(page, is_fullstack=True, live_app=True).errors
    (error,) = [e for e in errors if e.startswith("Script from a host a live app does not load: ")]
    assert blocked in error


@pytest.mark.parametrize("markup", [
    '<script src="https://cdn.jsdelivr.net:443/npm/x@1/x.js"></script>',
    '<!-- <script src="https://unpkg.com/old.js"></script> -->',
])
def test_the_live_app_rule_passes_what_the_policy_loads(markup):
    page = f'<!doctype html><html><head><meta name="viewport" content="width=device-width">{markup}</head></html>'
    errors = verifiers.verify_frontend(page, is_fullstack=True, live_app=True).errors
    assert not any(e.startswith("Script from a host a live app") for e in errors)


def test_a_live_apps_page_is_warned_of_its_own_policy_not_the_previews():
    page = (
        '<!doctype html><html><head><meta name="viewport" content="width=device-width">'
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/x@1/x.css">'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter">'
        '<img src="https://example.com/logo.png"></head></html>'
    )
    warnings = verifiers.verify_frontend(page, is_fullstack=True, live_app=True).warnings
    assert not any("app preview" in w for w in warnings)
    (warning,) = [w for w in warnings if w.startswith("External resource a live app does not load: ")]
    assert "fonts.googleapis.com" in warning and "example.com/logo.png" in warning
    assert "cdn.jsdelivr.net/npm/x@1/x.css" not in warning


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


async def _verify_frontend_live_app(tmp_path, policy, monkeypatch, seen: dict | None = None) -> bool:
    """Run the generator's frontend step and return the `live_app` the
    verifier got; `seen` also gets the system prompt the step wrote with."""
    from anton.core.tools.generate_artifact.state import VerifyResult

    seen = {} if seen is None else seen

    async def fake_loop(**kw):
        seen["system"] = kw["system"]
        (tmp_path / "static").mkdir(exist_ok=True)
        (tmp_path / "static" / "index.html").write_text(
            '<html><head><meta name="viewport" content="width=device-width"></head><body><div id="a"></div></body>'
            "</html>", encoding="utf-8",
        )
        return {"files_written": ["static/index.html"], "rounds_used": 1, "summary": "s"}

    def fake_verify(html, *, is_fullstack, api_paths=None, live_app=False):
        seen["live_app"] = live_app
        return VerifyResult(errors=[])

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend", fake_verify)
    await orchestrator._gen_verify_frontend(_gen_state(tmp_path, policy))
    return seen["live_app"]


async def test_the_generator_holds_a_live_apps_page_to_the_script_rule(tmp_path, policy, live_apps_on, monkeypatch):
    seen: dict = {}
    assert await _verify_frontend_live_app(tmp_path, policy, monkeypatch, seen) is True
    assert "## Live app scripts" in seen["system"], "the page is written to the rule it is checked against"


async def test_with_live_apps_off_the_generators_page_isnt_held_to_it(tmp_path, policy, live_apps_off, monkeypatch):
    seen: dict = {}
    assert await _verify_frontend_live_app(tmp_path, policy, monkeypatch, seen) is False
    assert "## Live app scripts" not in seen["system"]


@pytest.mark.parametrize("on", [True, False])
async def test_the_generators_backend_check_installs_nothing_only_for_a_live_app(tmp_path, policy, monkeypatch, on):
    from anton.core.tools.generate_artifact.state import VerifyResult

    if on:
        monkeypatch.setenv(las.SLOTS_ENV_VAR, _SLOTS)
    else:
        monkeypatch.delenv(las.SLOTS_ENV_VAR, raising=False)
    seen: dict = {}

    async def fake_loop(**kw):
        return {"files_written": ["backend.py"], "rounds_used": 1, "summary": "s"}

    async def fake_verify(**kw):
        seen.update(kw)
        return VerifyResult(errors=[]), []

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_backend", fake_verify)
    monkeypatch.setattr(orchestrator, "_map_datasources", lambda s, k: ([], []))
    await orchestrator._gen_verify_backend(_gen_state(tmp_path, policy))
    assert seen.get("live_app", False) is on


def test_only_a_live_apps_frontend_prompt_carries_its_script_rule(tmp_path):
    plain = prompts.build_frontend_system_prompt(tmp_path)
    live = prompts.build_frontend_system_prompt(tmp_path, live_app=True)
    assert "## Live app scripts" in live and "## Live app scripts" not in plain
    assert live.replace("\n\n" + prompts._LIVE_APP_SCRIPTS, "") == plain


async def test_a_live_apps_backend_installs_nothing_and_names_what_it_lacks(tmp_path, policy):
    (tmp_path / "backend.py").write_text("app = None\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pydantic\nsurely-not-installed-pkg-29>=1\n", encoding="utf-8")

    class _NoInstallPool(_Pool):
        async def get_or_create(self, name):
            async def install_packages(pkgs):
                raise AssertionError("a live app installs nothing")
            return SimpleNamespace(install_packages=install_packages)

    verdict, _ = await verifiers.verify_backend(
        scratchpad_pool=_NoInstallPool(policy), slug="app", artifact_path=tmp_path, live_app=True,
    )
    (error,) = verdict.errors
    assert "`surely-not-installed-pkg-29`" in error and "pydantic" not in error


async def test_without_a_sandbox_live_apps_dont_apply(tmp_path, live_apps_on):
    state = _gen_state(tmp_path, None)
    assert orchestrator._runs_live_apps(state) is False


_SNAPSHOT = """
import json, sys
from pathlib import Path
from anton.core.tools import tool_defs
from anton.core.tools.generate_artifact import prompts
d = Path(sys.argv[1])
built = [prompts.build_subagent_system_prompt(d), prompts.build_frontend_system_prompt(d),
         prompts.build_backend_system_prompt(d, stateless=True),
         prompts.build_backend_system_prompt(d, stateless=False)]
tools = sorted([t.name, t.description, repr(t.input_schema), t.prompt, t.unlock_skill]
               for t in vars(tool_defs).values() if type(t).__name__ == "ToolDef")
print(json.dumps([built, tools]))
"""


def test_prompts_and_tool_definitions_dont_depend_on_the_setting(tmp_path):
    """Each snapshot in a fresh interpreter: a definition computed from the
    setting at import would differ here, where one process can't see it."""
    def snapshot(slots: str | None) -> str:
        env = {k: v for k, v in os.environ.items() if k != las.SLOTS_ENV_VAR}
        if slots is not None:
            env[las.SLOTS_ENV_VAR] = slots
        out = subprocess.run([sys.executable, "-B", "-c", _SNAPSHOT, str(tmp_path)], env=env, check=True,
                             capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        return out.stdout

    assert snapshot(_SLOTS) == snapshot(None)


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
    asked: list[list[str]] = []

    async def relaunch(session, store_arg, slugs):
        asked.append(list(slugs))
        return []

    monkeypatch.setattr(las, "relaunch_changed_apps", relaunch)
    session = SimpleNamespace(_scratchpads=SimpleNamespace(cell_sandbox=policy))
    assert await tool_handlers._relaunch_live_apps(session, SimpleNamespace(root=tmp_path), {}) == []
    assert asked == []
    text = tool_handlers.cell_content_with_lint("[output]\nok", tool_handlers.ArtifactLint(), cell_ok=True)
    assert text == "[output]\nok"


def _running_app(root: Path) -> tuple[str, Path]:
    """A fullstack artifact made through the store, as the tools make one: its slug and folder."""
    from anton.core.artifacts.store import ArtifactStore

    store = ArtifactStore(root)
    artifact = store.create(name="App", description="d", type="fullstack-stateless-app")
    folder = store.folder_for(artifact.slug)
    (folder / "static").mkdir(parents=True, exist_ok=True)
    (folder / "static" / "index.html").write_text("v1", encoding="utf-8")
    return artifact.slug, folder


async def test_a_cells_relaunch_reaches_its_result_through_the_check(tmp_path, policy, live_apps_on, monkeypatch):
    from anton.core.artifacts.store import ArtifactStore

    root = tmp_path / "artifacts"
    slug, folder = _running_app(root)
    store = ArtifactStore(root)
    before = tool_handlers.snapshot_existing_artifacts(store)
    (folder / "static" / "index.html").write_text("version two", encoding="utf-8")

    async def relaunch(session, store_arg, slugs):
        return [f"Relaunched live app `{s}`." for s in slugs]

    monkeypatch.setattr(las, "relaunch_changed_apps", relaunch)
    monkeypatch.delenv("ANTON_RENDER_CHECK_URL", raising=False)
    session = SimpleNamespace(_scratchpads=SimpleNamespace(cell_sandbox=policy), _artifacts_touched=set())
    lint = await tool_handlers.check_cell_artifacts(session, store=store, before=before)
    assert lint.live_app_lines == [f"Relaunched live app `{slug}`."]


async def test_a_cell_inside_a_generation_leaves_its_app_alone(tmp_path, policy, live_apps_on, monkeypatch):
    root = tmp_path / "artifacts"
    slug, folder = _running_app(root)
    store = SimpleNamespace(root=root)
    before = tool_handlers.snapshot_existing_artifacts(store)
    (folder / "static" / "index.html").write_text("half written", encoding="utf-8")
    asked: list[list[str]] = []

    async def relaunch(session, store_arg, slugs):
        asked.append(list(slugs))
        return []

    monkeypatch.setattr(las, "relaunch_changed_apps", relaunch)
    session = SimpleNamespace(_scratchpads=SimpleNamespace(cell_sandbox=policy), _live_apps_building={slug})
    await tool_handlers._relaunch_live_apps(session, store, before)
    assert asked == [[]]


async def test_new_connections_on_a_running_app_relaunch_it(tmp_path, policy, live_apps_on, monkeypatch):
    from anton.core.artifacts.store import ArtifactStore

    root = tmp_path / "artifacts"
    slug, _ = _running_app(root)
    store = ArtifactStore(root)
    monkeypatch.setattr(tool_handlers, "resolve_artifact_store", lambda session: store)
    vault = SimpleNamespace(list_connections=lambda: [{"engine": "postgres", "name": "demo"}])
    asked: list[list[str]] = []

    async def relaunch(session, store_arg, slugs):
        asked.append(list(slugs))
        return ["Relaunched live app `app` at `http://app1.view.localhost:8080/` with this cell's changes."]

    monkeypatch.setattr(las, "relaunch_changed_apps", relaunch)
    session = SimpleNamespace(_scratchpads=SimpleNamespace(cell_sandbox=policy), _data_vault=vault,
                              _artifacts_touched=set())
    outcome = await tool_handlers.handle_update_artifact_metadata(
        session, {"slug": slug, "datasources": ["postgres-demo"]},
    )
    assert asked == [[slug]]
    assert "[live app]\nRelaunched live app `app`" in json.loads(outcome.content)["message"]
    again = await tool_handlers.handle_update_artifact_metadata(
        session, {"slug": slug, "datasources": ["postgres-demo"]},
    )
    assert asked == [[slug]] and "[live app]" not in json.loads(again.content)["message"]


def test_a_live_apps_unchecked_page_is_named_as_a_live_app(tmp_path, live_apps_on):
    from anton.core.artifacts.store import ArtifactStore
    from anton.core.tools.tool_handlers import PageKey

    store = ArtifactStore(tmp_path / "artifacts")
    slug, _ = _running_app(tmp_path / "artifacts")
    (line,) = tool_handlers.page_check_lines(store, checks={PageKey(slug, "static/index.html"): "unverified"})
    assert line == "“App” was not checked in a browser: it is a live app, so open it at its address to check it."


def test_a_generated_live_app_gets_its_own_instruction():
    instruction = tool_handlers._STATUS_INSTRUCTIONS["generated_live_app"]
    assert "never in the preview" in instruction and "preview;" not in instruction


def test_the_session_keeps_the_hosts_live_app_object():
    from anton.core.session import ChatSession, ChatSessionConfig
    from tests.conftest import make_mock_llm

    host = SimpleNamespace(launcher=None)
    assert ChatSession(ChatSessionConfig(llm_client=make_mock_llm(), live_app_host=host))._live_app_host is host
    assert ChatSession(ChatSessionConfig(llm_client=make_mock_llm()))._live_app_host is None
