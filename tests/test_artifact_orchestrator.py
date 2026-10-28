from __future__ import annotations

import json
import threading
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from anton.core.artifacts.internal_files import DATA_DIRNAME
from anton.core.datasources.data_vault import LocalDataVault
from anton.core.llm.provider import StreamComplete
from anton.core.tools.generate_artifact import engine, orchestrator, prompts
from anton.core.tools.generate_artifact.data_files import DataFile
from anton.core.tools.generate_artifact.discovery import checkpoint as cp
from anton.core.tools.generate_artifact.state import (
    BrowserCheck, BrowserCheckFailed, FetchVerdict, GenState, RequiredData, RequiredDataItem,
    VerifyResult,
)
from anton.utils.datasources import DATASOURCES_HEADER


def _no_browser(entry, *, cell_sandbox, screenshot=True, previous_html=None):
    """`verify_frontend_live` where no browser can load the page."""
    return BrowserCheck(
        status="unverified",
        reason=orchestrator.verifiers.browser_check_skip_reason(sandboxed=cell_sandbox is not None),
    )


def _loaded(verdict: VerifyResult) -> BrowserCheck:
    """`verify_frontend_live` where a browser loaded the page and found `verdict`."""
    return BrowserCheck(status="passed" if verdict.ok else "failed", verdict=verdict)


async def _one_event_stream(response):
    yield StreamComplete(response=response)


def _stream_mock(response):
    """`plan_stream` fake: every call returns a fresh one-event stream of the
    same response (mirrors `AsyncMock(return_value=...)`)."""
    return Mock(side_effect=lambda **kw: _one_event_stream(response))


def _session() -> AsyncMock:
    """A mock chat session whose scratchpad pool carries no sandbox policy."""
    session = AsyncMock()
    session._scratchpads.cell_sandbox = None
    return session


def _state(tmp_path, **kw):
    base = dict(
        session=_session(), artifact_type="html-app", artifact_path=tmp_path,
        slug="a", brief="Show current time", is_fullstack=False,
    )
    base.update(kw)
    return GenState(**base)


# ── Task 7: data phase ───────────────────────────────────────────────────────

async def test_data_phase_costs_nothing_on_the_normal_path(tmp_path: Path):
    """No LLM call at all. `is_data_enough` used to sit here asking a second
    model to re-derive the verdict the gathering phase already gave by
    calling `finish_gathering`."""
    st = _state(tmp_path)
    st.gathering_complete = True
    st.session._llm.generate_object = AsyncMock(
        side_effect=AssertionError("the data phase must not call the model here")
    )
    err = await orchestrator._data_phase(st)
    assert err is None
    assert st.data_iterations == 0


async def test_data_phase_terminates_when_impossible(tmp_path: Path):
    st = _state(tmp_path)
    st.gathering_complete = True
    st.declared_sources = ["orders"]
    st.unverified_sources = ["orders"]
    seq = [
        RequiredData(items=[RequiredDataItem(name="orders", where="nowhere", why="chart")], reasoning="r"),
        FetchVerdict(possible=False, reasoning="no orders source connected"),
    ]
    st.session._llm.generate_object = AsyncMock(side_effect=seq)
    err = await orchestrator._data_phase(st)
    assert err is not None
    assert "not enough data" in err.lower()
    assert "no orders source connected" in err  # reasoning carried through


async def test_data_phase_exits_after_one_successful_fetch(tmp_path: Path, monkeypatch):
    """A gathering that ended without `finish_gathering` leaves
    `gathering_complete=False`, and nothing else ever set it — so the loop
    used to run its three iterations and fail with "not enough data" after
    three GOOD fetches (I-41). The fetch is what vouches for the data."""
    st = _state(tmp_path)
    st.gathering_complete = False  # never finished: the loop has to run

    def gen_obj(schema_class, **kw):
        if schema_class is RequiredData:
            return RequiredData(items=[RequiredDataItem(name="x", where="db", why="y")], reasoning="r")
        return FetchVerdict(possible=True, reasoning="yes")

    st.session._llm.generate_object = AsyncMock(
        side_effect=lambda schema_class, **kw: gen_obj(schema_class, **kw)
    )

    async def fake_fetch(state, required_text=""):
        return "pulled a sample", True
    monkeypatch.setattr(orchestrator, "_fetch_data_sample", fake_fetch)

    err = await orchestrator._data_phase(st)
    assert err is None
    assert st.data_iterations == 1
    assert st.gathering_complete is True
    assert "pulled a sample" in st.data_notes
    assert orchestrator._needs_data_loop(st) is False


# ── fetch_data_sample: exec-code record + journal handoff ────────────────────

async def test_fetch_data_sample_appends_exec_code(tmp_path: Path, monkeypatch):
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)

    async def fake_run_loop(**kw):
        return {
            "files_written": [], "rounds_used": 2, "summary": "pulled 100 rows",
            "scratchpad_execs": [
                {"name": "pad", "code": "df = q('select 1')", "output": "ok  100 rows"}
            ],
        }

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_run_loop)
    notes, ok = await orchestrator._fetch_data_sample(st)
    assert notes.startswith("pulled 100 rows")
    assert "### Code executed while fetching" in notes
    assert "df = q('select 1')" in notes
    assert "Output: ok 100 rows" in notes
    assert ok is True


async def test_fetch_for_an_html_app_counts_only_a_saved_data_file(tmp_path: Path, monkeypatch):
    st = _state(tmp_path)
    st.unverified_sources = ["orders"]

    async def fake_run_loop(**kw):
        assert "data files folder" in kw["system"].lower()
        (tmp_path / DATA_DIRNAME).mkdir()
        (tmp_path / DATA_DIRNAME / "1.json").write_text(json.dumps([{"v": 1}]))
        (tmp_path / DATA_DIRNAME / "2.json").write_text("not json")
        return {"files_written": [], "rounds_used": 2, "summary": "saved .anton-data/1.json",
                "scratchpad_execs": [{"name": "p", "code": "save()", "output": "ok"}]}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_run_loop)
    notes, ok = await orchestrator._fetch_data_sample(st)
    assert ok is True
    assert st.data_files == [DataFile("1", ["orders"], "saved .anton-data/1.json")]
    assert st.loop_data_ids == ["1"]


async def test_fetch_for_an_html_app_without_a_file_fails(tmp_path: Path, monkeypatch):
    st = _state(tmp_path)
    st.unverified_sources = ["orders"]

    async def fake_run_loop(**kw):
        return {"files_written": [], "rounds_used": 1, "summary": "looked",
                "scratchpad_execs": [{"name": "p", "code": "print(1)", "output": "1"}]}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_run_loop)
    notes, ok = await orchestrator._fetch_data_sample(st)
    assert ok is False and "no valid data file" in notes


async def test_a_file_only_a_cell_mentions_is_not_taken_as_data(tmp_path: Path, monkeypatch):
    """The summary names 1.json; a valid 2.json that only a cell's code mentions
    is a scratch file, not data for the page."""
    st = _state(tmp_path)
    st.unverified_sources = ["orders"]

    async def fake_run_loop(**kw):
        (tmp_path / DATA_DIRNAME).mkdir()
        (tmp_path / DATA_DIRNAME / "1.json").write_text(json.dumps([{"v": 1}]))
        (tmp_path / DATA_DIRNAME / "2.json").write_text(json.dumps([{"tmp": 1}]))
        return {"files_written": [], "rounds_used": 2, "summary": "saved .anton-data/1.json",
                "scratchpad_execs": [{"name": "p", "output": "ok",
                                      "code": "json.dump(rows, open('.anton-data/2.json', 'w'))"}]}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_run_loop)
    notes, ok = await orchestrator._fetch_data_sample(st)
    assert ok is True
    assert [f.id for f in st.data_files] == ["1"]
    assert st.loop_data_ids == ["1"]


async def test_a_summary_that_names_no_file_takes_the_new_valid_one(tmp_path: Path, monkeypatch):
    st = _state(tmp_path)
    st.unverified_sources = ["orders"]

    async def fake_run_loop(**kw):
        (tmp_path / DATA_DIRNAME).mkdir()
        (tmp_path / DATA_DIRNAME / "1.json").write_text(json.dumps([{"v": 1}]))
        return {"files_written": [], "rounds_used": 1, "summary": "saved the orders",
                "scratchpad_execs": [{"name": "p", "code": "save()", "output": "ok"}]}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_run_loop)
    notes, ok = await orchestrator._fetch_data_sample(st)
    assert ok is True
    assert [f.id for f in st.data_files] == ["1"]
    assert st.loop_data_ids == ["1"]


async def test_fetch_counts_the_size_of_all_data_files(tmp_path: Path, monkeypatch):
    """Gathering's file and the fetched one each fit; together they do not."""
    (tmp_path / DATA_DIRNAME).mkdir()
    (tmp_path / DATA_DIRNAME / "1.json").write_text(json.dumps(["x" * 300]))
    st = _state(tmp_path)
    st.unverified_sources = ["orders"]
    st.data_files = [DataFile("1", ["regions"])]
    monkeypatch.setattr(orchestrator.data_files, "MAX_PAGE_BYTES", 500)

    async def fake_run_loop(**kw):
        (tmp_path / DATA_DIRNAME / "2.json").write_text(json.dumps(["y" * 300]))
        return {"files_written": [], "rounds_used": 1, "summary": "saved .anton-data/2.json",
                "scratchpad_execs": [{"name": "p", "code": "save()", "output": "ok"}]}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_run_loop)
    notes, ok = await orchestrator._fetch_data_sample(st)
    assert ok is False and "page limit" in notes
    assert [f.id for f in st.data_files] == ["1"]


async def test_a_saved_fetch_clears_the_data_not_saved_warning(tmp_path: Path, monkeypatch):
    """An earlier pass did not save the data; this fetch did, so neither the
    result nor the next checkpoint repeats the warning."""
    from anton.core.tools.generate_artifact.data_files import DATA_NOT_SAVED_WARNING

    st = _state(tmp_path)
    st.declared_sources = ["orders"]
    st.unverified_sources = ["orders"]
    st.data_not_saved = True
    st.verify_warnings.append(DATA_NOT_SAVED_WARNING)

    async def fake_run_loop(**kw):
        (tmp_path / DATA_DIRNAME).mkdir()
        (tmp_path / DATA_DIRNAME / "1.json").write_text(json.dumps([{"v": 1}]))
        return {"files_written": [], "rounds_used": 1, "summary": "saved .anton-data/1.json",
                "scratchpad_execs": [{"name": "p", "code": "save()", "output": "ok"}]}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_run_loop)
    notes, ok = await orchestrator._fetch_data_sample(st)
    assert ok is True and st.data_not_saved is False
    result = orchestrator._finish(st)
    assert DATA_NOT_SAVED_WARNING not in result["warnings"]
    assert cp.load(tmp_path).data_not_saved is False


def test_render_exec_notes_caps_and_drops_oldest():
    assert orchestrator._render_exec_notes([]) == ""
    # Cells with no code are skipped entirely.
    assert orchestrator._render_exec_notes([{"name": "p", "code": " ", "output": "o"}]) == ""

    execs = [
        {"name": f"p{i}", "code": "x" * 3000, "output": "y" * 1000} for i in range(10)
    ]
    notes = orchestrator._render_exec_notes(execs)
    assert "# … truncated …" in notes  # per-cell code cap applied
    assert "p9" in notes and "p0" not in notes  # oldest cells dropped first
    assert "omitted for size" in notes
    assert len(notes) < orchestrator.EXEC_NOTES_MAX + 500


def test_spec_context_includes_journal(tmp_path: Path):
    st = _state(tmp_path)
    st.data_notes = "sample data"
    st.record("is_data_enough", "yes", "ok")
    st.record("make_tech_spec", "done", "wrote spec.md")
    ctx = orchestrator._spec_context(st)
    assert "## Progress journal" in ctx
    assert "- is_data_enough: yes — ok" in ctx
    assert "- make_tech_spec: done — wrote spec.md" in ctx


# ── Task 8: tech spec, api spec, gen+verify+retry, declare_datasources ────────

async def test_gen_verify_backend_retries_once_then_succeeds(tmp_path: Path, monkeypatch):
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.api_spec = "{}"

    calls = {"gen": 0, "verify": 0}
    async def fake_loop(**kw):
        calls["gen"] += 1
        (tmp_path / "backend.py").write_text("x")
        return {"files_written": ["backend.py"], "rounds_used": 1, "summary": "s"}
    async def fake_verify(**kw):
        calls["verify"] += 1
        if calls["verify"] == 1:
            return VerifyResult(errors=["missing /api/health"]), []
        return VerifyResult(errors=[]), ["DS_PG_MAIN__PASSWORD"]
    declared = {}
    async def fake_declare(state, refs):
        declared["refs"] = refs
    def fake_map(session, ds_keys):
        # (refs, unmapped) — pretend the key maps cleanly.
        return (["REF"], [])
    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_backend", fake_verify)
    monkeypatch.setattr(orchestrator, "_map_datasources", fake_map)
    monkeypatch.setattr(orchestrator, "_declare_datasources", fake_declare)
    # "REF" stands in for a connection reference, so the verify env, built on
    # every host, can't be resolved from it.
    monkeypatch.setattr(orchestrator, "_verify_datasource_env", lambda state: {})

    err = await orchestrator._gen_verify_backend(st)
    assert err is None
    assert calls["gen"] == 2  # one retry
    assert declared["refs"] == ["REF"]


async def test_gen_verify_backend_hands_the_import_its_own_connections_without_a_sandbox(
    tmp_path: Path, monkeypatch
):
    """The import check gets only the connections backend.py reads on every
    host, as the launch does, not only under a sandbox."""
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.api_spec = "{}"
    seen = {}

    async def fake_loop(**kw):
        (tmp_path / "backend.py").write_text("x")
        return {"files_written": ["backend.py"], "rounds_used": 1, "summary": "s"}

    async def fake_verify(**kw):
        seen.update(kw)
        return VerifyResult(errors=[]), []

    async def fake_declare(state, refs):
        pass

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_backend", fake_verify)
    monkeypatch.setattr(orchestrator, "_cell_sandbox", lambda state: None)
    monkeypatch.setattr(orchestrator, "_map_datasources", lambda session, keys: ([], []))
    monkeypatch.setattr(orchestrator, "_declare_datasources", fake_declare)
    monkeypatch.setattr(
        orchestrator, "_verify_datasource_env", lambda state: {"DS_PG_MAIN__PASSWORD": "pw"}
    )

    assert await orchestrator._gen_verify_backend(st) is None
    assert seen["ds_env"] == {"DS_PG_MAIN__PASSWORD": "pw"}


async def test_declare_datasources_writes_an_empty_list_to_clear_the_previous_one(
    tmp_path: Path, monkeypatch
):
    """The early return on an empty mapping left the previous generation's
    connection in metadata."""
    from anton.core.tools import tool_handlers

    store = Mock()
    monkeypatch.setattr(tool_handlers, "resolve_artifact_store", lambda session: store)
    st = _state(tmp_path)
    await orchestrator._declare_datasources(st, [])
    store.update.assert_called_once_with("a", datasources=[])
    rec = [r for r in st.trace if r.node == "declare_datasources"][0]
    assert rec.outcome == "done" and "cleared" in rec.detail


def _stub_backend_run(monkeypatch, tmp_path: Path) -> dict:
    """Stub the backend generation step's loop, verifier and datasource
    mapping; returns where the loop's and the verifier's kwargs land."""
    captured: dict = {}

    async def fake_loop(**kw):
        captured["loop"] = kw
        (tmp_path / "backend.py").write_text("x")
        return {"files_written": ["backend.py"], "rounds_used": 1, "summary": "s"}

    async def fake_verify(**kw):
        captured["verify"] = kw
        return VerifyResult(errors=[]), []

    async def fake_declare(state, refs):
        pass

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_backend", fake_verify)
    monkeypatch.setattr(orchestrator, "_map_datasources", lambda s, k: ([], []))
    monkeypatch.setattr(orchestrator, "_declare_datasources", fake_declare)
    return captured


async def test_gen_verify_backend_stateful_wiring(tmp_path: Path, monkeypatch):
    """Stateful: the manifest step is injected and the type reaches the verifier."""
    st = _state(tmp_path, artifact_type="fullstack-stateful-app", is_fullstack=True)
    st.api_spec = "{}"
    captured = _stub_backend_run(monkeypatch, tmp_path)
    err = await orchestrator._gen_verify_backend(st)
    assert err is None
    assert [t for t, _ in captured["loop"]["step_injections"]] == ["backend.py", "state_manifest.json"]
    assert captured["verify"].get("artifact_type") == "fullstack-stateful-app"
    # The system prompt carries the STATE contract, not the old sqlite model.
    assert "state_manifest.json" in captured["loop"]["system"]
    assert "sqlite file (or similar) IS allowed" not in captured["loop"]["system"]


async def test_gen_verify_backend_stateless_wiring(tmp_path: Path, monkeypatch):
    """Stateless: two files only — no manifest step; the type reaches the verifier."""
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.api_spec = "{}"
    captured = _stub_backend_run(monkeypatch, tmp_path)
    err = await orchestrator._gen_verify_backend(st)
    assert err is None
    assert [t for t, _ in captured["loop"]["step_injections"]] == ["backend.py"]
    assert captured["verify"].get("artifact_type") == "fullstack-stateless-app"


async def test_backend_prompt_gets_only_the_declared_connection_and_its_notes(tmp_path: Path, monkeypatch):
    vault = LocalDataVault(tmp_path / "vault")
    vault.save("langfuse", "prod", {"public_key": "pk", "secret_key": "sk"})
    vault.save("postgres", "db", {"host": "h", "password": "p"})
    session = SimpleNamespace(
        _data_vault=vault,
        _connector_usage_notes={"langfuse": "LF-NOTE", "postgres": "PG-NOTE"},
    )
    st = _state(
        tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True,
        datasource_catalog=orchestrator._datasource_catalog(session),
        declared_sources=["daily metrics from langfuse"],
    )
    st.api_spec = "{}"
    captured = _stub_backend_run(monkeypatch, tmp_path)

    assert await orchestrator._gen_verify_backend(st) is None
    system = captured["loop"]["system"]
    assert "DS_LANGFUSE_PROD__PUBLIC_KEY" in system
    assert "LF-NOTE" in system
    assert "DS_POSTGRES_DB__HOST" not in system
    assert "PG-NOTE" not in system


async def test_gen_verify_backend_terminal_after_second_failure(tmp_path: Path, monkeypatch):
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.api_spec = "{}"
    async def fake_loop(**kw):
        (tmp_path / "backend.py").write_text("x")
        return {"files_written": ["backend.py"], "rounds_used": 1, "summary": "s"}
    async def fake_verify(**kw):
        return VerifyResult(errors=["still broken"]), []
    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_backend", fake_verify)
    err = await orchestrator._gen_verify_backend(st)
    assert err is not None and "verification failed" in err.lower()


async def test_write_tech_spec_writes_spec_md(tmp_path: Path):
    st = _state(tmp_path)
    st.session._llm.plan_stream = _stream_mock(type("R", (), {"content": "# Spec\nbody"})())
    err = await orchestrator._write_tech_spec(st)
    assert err is None
    assert (tmp_path / "spec.md").read_text().startswith("# Spec")


# ── Task 9: run_app, verify_fullstack, run() ─────────────────────────────────

async def test_run_html_app_happy_path(tmp_path: Path, monkeypatch):
    # ENTRY_SPEC is the resume-from-PRD path: discovery is restored from disk
    # and the run starts at the spec node — which is exactly what `run(state)`
    # did before the merge, so these FSM tests keep their shape.
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    st.gathering_complete = True
    st.session._llm.plan_stream = _stream_mock(type("R", (), {"content": "# Spec"})())

    async def fake_front(state):
        (tmp_path / "dashboard.html").write_text("<body><div id=x></div></body>")
        state.files_written.append("dashboard.html")
        state.record("verify_frontend", "ok")
        return None
    monkeypatch.setattr(orchestrator, "_gen_verify_frontend", fake_front)

    out = await orchestrator.run(st, entry=cp.ENTRY_SPEC)
    assert isinstance(out, dict)
    assert out["status"] == "generated"
    assert "dashboard.html" in out["files_written"]
    # html-app must NOT run backend / run_app.
    assert not any(s["node"] == "run_app" for s in out["trace"])


async def test_run_data_terminal_returns_error(tmp_path: Path):
    st = _state(tmp_path)
    st.gathering_complete = True
    st.declared_sources = ["x"]
    st.unverified_sources = ["x"]
    st.session._llm.generate_object = AsyncMock(side_effect=[
        RequiredData(items=[], reasoning="need x from nowhere"),
        FetchVerdict(possible=False, reasoning="no source"),
    ])
    out = await orchestrator.run(st, entry=cp.ENTRY_SPEC)
    assert isinstance(out, str)
    assert "not enough data" in out.lower()


async def test_run_fullstack_launches_and_verifies(tmp_path: Path, monkeypatch):
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.gathering_complete = True
    st.session._workspace = None  # resolve_artifact_store returns None → port update skipped
    st.session._llm.generate_object = AsyncMock(
        side_effect=AssertionError("the data phase must not call the model here")
    )
    st.session._llm.plan_stream = _stream_mock(type("R", (), {"content": "# Spec"})())

    async def fake_api(state):
        state.api_spec = '{"paths": {"/api/items": {"get": {"parameters": []}}}}'
        return None
    async def fake_back(state):
        state.record("verify_backend", "ok"); return None
    async def fake_front(state, extra_context=""):
        state.record("verify_frontend", "ok"); return None
    launched = {}
    async def fake_launch(**kw):
        launched.update(kw); return {"port": 5555, "pid": 1, "url": "http://127.0.0.1:5555", "log_path": "x"}
    async def fake_probe(state, port):
        return None
    monkeypatch.setattr(orchestrator, "_make_api_spec", fake_api)
    monkeypatch.setattr(orchestrator, "_gen_verify_backend", fake_back)
    monkeypatch.setattr(orchestrator, "_gen_verify_frontend", fake_front)
    monkeypatch.setattr(orchestrator, "_launch_backend", fake_launch)
    monkeypatch.setattr(orchestrator, "_probe_app", fake_probe)
    # The served page loads cleanly (S-02); the browser itself is not needed here.
    monkeypatch.setattr(orchestrator.verifiers, "verify_app_live", lambda url, *, cell_sandbox: VerifyResult(errors=[]))

    out = await orchestrator.run(st, entry=cp.ENTRY_SPEC)
    assert isinstance(out, dict)
    assert launched["health_path"] == "/api/health"
    assert any(s["node"] == "verify_fullstack" and s["outcome"] == "ok" for s in out["trace"])
    # The launch URL is the app's entry point and rides the result at the top
    # level — a live run had it only as "port 42303" inside the
    # trace, and the agent pointed the user at static/index.html on disk.
    assert out["url"] == "http://127.0.0.1:5555"
    assert out["port"] == 5555


def test_result_carries_no_url_until_an_app_was_launched(tmp_path: Path):
    """html-app runs, and fullstack runs that stop before `run_app`, have no
    running backend — the key is absent rather than null, so the agent's
    "when the result carries `url`" branch is unambiguous."""
    st = _state(tmp_path, artifact_type="html-app")
    st.files_written = ["index.html"]
    shell = orchestrator._result_shell(st)
    assert "url" not in shell and "port" not in shell
    st.app_url, st.app_port = "http://127.0.0.1:4242", 4242
    shell = orchestrator._result_shell(st)
    assert shell["url"] == "http://127.0.0.1:4242" and shell["port"] == 4242


def test_agent_facing_texts_name_the_url_as_the_fullstack_entry_point():
    """Three surfaces the agent reads before writing its final message must
    agree: the `generated` instruction in the tool result, and the ARTIFACTS
    section of the system prompt."""
    from anton.core.llm import prompts as llm_prompts
    from anton.core.tools.tool_handlers import _STATUS_INSTRUCTIONS

    generated = _STATUS_INSTRUCTIONS["generated"]
    assert "`url`" in generated and "entry point" in generated
    assert "static/index.html" in generated
    assert "prefer" not in llm_prompts.ARTIFACTS_PROMPT.split("AFTER FINISHING")[1]
    assert "never present its path" in llm_prompts.ARTIFACTS_PROMPT
    assert "cannot reach the API" in llm_prompts.ARTIFACTS_PROMPT


# ── Task 10: public generate() delegate ──────────────────────────────────────

async def test_public_generate_delegates_to_run(tmp_path: Path, monkeypatch):
    from anton.core.tools.generate_artifact import generate as public_generate

    session = AsyncMock()
    session._scratchpads.cell_sandbox = None
    captured = {}
    async def fake_run(state, *, entry):
        captured["type"] = state.artifact_type
        captured["slug"] = state.slug
        captured["is_fullstack"] = state.is_fullstack
        captured["entry"] = entry
        captured["user_request"] = state.user_request
        return {"files_written": [], "summary": "ok", "trace": []}
    monkeypatch.setattr("anton.core.tools.generate_artifact.orchestrator.run", fake_run)

    out = await public_generate(
        session=session, artifact_type="fullstack-stateless-app",
        artifact_path=tmp_path, slug="a",
        user_request="build an orders app",
        agent_understanding="a small orders dashboard",
    )
    assert out["summary"] == "ok"
    assert captured["type"] == "fullstack-stateless-app"
    assert captured["slug"] == "a"
    assert captured["is_fullstack"] is True
    assert captured["user_request"] == "build an orders app"
    # Nothing on disk for this request, so the run starts at the beginning.
    assert captured["entry"] == cp.ENTRY_FULL


async def test_gen_verify_frontend_reports_missing_html_file(tmp_path: Path, monkeypatch):
    """The generator wrote nothing → the terminal error names the cause.

    In this branch verdict used to stay None and the caller got "generation did not
    produce a verifiable frontend" — a message that gives no hint that the file was
    simply never created.
    """
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)

    async def fake_loop(**kw):
        # the loop "succeeds", but there is no .html on disk
        return {"files_written": [], "rounds_used": 1, "summary": "s"}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)

    err = await orchestrator._gen_verify_frontend(st)

    assert err is not None
    assert "No HTML entry file was written" in err


def test_datasource_context_is_defensive_against_mock_session():
    """In tests the session is an AsyncMock: every attribute truthy, every call a coroutine.

    We assert not only the empty result but also the absence of a RuntimeWarning:
    an un-closed coroutine surfaces at GC time and pollutes the output of any test
    with a mock session (i.e. nearly every test here).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert orchestrator._datasource_catalog(AsyncMock()) is None
        assert orchestrator._known_connection_hints(AsyncMock()) == []


def test_list_connections_tolerates_junk():
    """A non-vault input must not break generation."""
    class _Raises:
        def list_connections(self):
            raise RuntimeError("vault unavailable")

    class _NotIterable:
        def list_connections(self):
            return 42

    assert orchestrator._list_connections(_Raises()) == []
    assert orchestrator._list_connections(_NotIterable()) == []
    assert orchestrator._list_connections(object()) == []


def test_known_connection_hints_give_ready_env_prefixes():
    """The generator needs the env prefix, not the slug: the conversion sanitises
    special characters (`prod-db.eu` -> `PROD_DB_EU`), and asking the model to
    repeat it means asking it to repeat the mistake that got it here."""
    class _Vault:
        def list_connections(self):
            return [
                {"engine": "postgres", "name": "prod-db.eu"},
                {"engine": "hubspot", "name": "main"},
                {"engine": "", "name": "broken"},
            ]

    session = AsyncMock()
    session._scratchpads.cell_sandbox = None
    session._data_vault = _Vault()
    assert orchestrator._known_connection_hints(session) == [
        "hubspot-main → DS_HUBSPOT_MAIN__<FIELD>",
        "postgres-prod-db.eu → DS_POSTGRES_PROD_DB_EU__<FIELD>",
    ]


async def test_unmapped_ds_error_lists_available_connections(tmp_path: Path, monkeypatch):
    """The message must give the generator valid names, not only the wrong ones."""
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.api_spec = "{}"

    class _Vault:
        def list_connections(self):
            return [{"engine": "postgres", "name": "prod_db"}]

    st.session._data_vault = _Vault()

    async def fake_loop(**kw):
        (tmp_path / "backend.py").write_text("x")
        return {"files_written": ["backend.py"], "rounds_used": 1, "summary": "s"}

    async def fake_verify(**kw):
        return VerifyResult(errors=[]), ["DS_TYPO_DB__PASSWORD"]

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_backend", fake_verify)
    monkeypatch.setattr(orchestrator, "_map_datasources",
                        lambda session, keys: ([], ["DS_TYPO_DB__PASSWORD"]))

    err = await orchestrator._gen_verify_backend(st)

    assert err is not None
    assert "DS_TYPO_DB__PASSWORD" in err          # what was wrong
    assert "postgres-prod_db" in err              # which connections exist
    assert "DS_POSTGRES_PROD_DB__<FIELD>" in err  # and the ready prefix, no deriving


def test_html_app_default_primary_is_shared():
    """The cleanup step and the prompt must rely on one shared default."""
    assert orchestrator.HTML_APP_DEFAULT_PRIMARY == "index.html"


def test_read_frontend_html_prefers_primary_among_written(tmp_path: Path):
    """Among what was written, primary takes priority."""
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    st.primary = "report.html"
    (tmp_path / "extra.html").write_text("<body>extra</body>")
    (tmp_path / "report.html").write_text("<body>target</body>")
    html = orchestrator._read_frontend_html(st, ["extra.html", "report.html"])
    assert html == "<body>target</body>"


def test_read_frontend_html_ignores_files_this_run_did_not_write(tmp_path: Path):
    """primary is the expectation, written is the fact; the fact must be verified.

    An easy regression to introduce: make primary (or its default) candidate #1
    unconditionally. Then, with no primary set, an `index.html` left over from a
    previous generation would shadow the fresh `report.html`.
    """
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    st.primary = None  # the agent set no primary — create_artifact allows that
    (tmp_path / "index.html").write_text("<body>stale from a previous run</body>")
    (tmp_path / "report.html").write_text("<body>written now</body>")
    html = orchestrator._read_frontend_html(st, ["report.html"])
    assert html == "<body>written now</body>"


def test_fullstack_entry_ignores_a_static_index_this_run_did_not_write(tmp_path: Path):
    """Same discipline as the html-app branch (I-43): a `static/index.html`
    left over from a previous generation must not pass the verifier on behalf
    of an attempt that wrote only `static/app.js`."""
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "index.html").write_text("<body>stale</body>")
    (tmp_path / "static" / "app.js").write_text("// fresh")

    assert orchestrator._frontend_entry(st, ["static/app.js"]) is None
    assert orchestrator._read_frontend_html(st, ["static/index.html"]) == "<body>stale</body>"


def test_fullstack_entry_is_none_when_the_written_file_is_gone(tmp_path: Path):
    """Listed as written but not on disk: no entry, not an exception."""
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    assert orchestrator._frontend_entry(st, ["static/index.html"]) is None


async def test_frontend_terminal_error_carries_the_loop_reason(tmp_path: Path, monkeypatch):
    """The generic "did not produce a verifiable frontend" lost the recorded cause."""
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)

    async def fake_loop(**kw):
        return "generator exceeded round budget (20) after writing 0 file(s): []."

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)

    err = await orchestrator._gen_verify_frontend(st)

    assert err is not None
    assert "round budget" in err
    assert "did not produce a verifiable frontend" not in err


async def test_backend_terminal_error_carries_the_loop_reason(tmp_path: Path, monkeypatch):
    """The same gap in the twin method — it must not be skipped."""
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.api_spec = "{}"

    async def fake_loop(**kw):
        return "generator exceeded round budget (20) after writing 1 file(s): ['backend.py']."

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)

    err = await orchestrator._gen_verify_backend(st)

    assert err is not None
    assert "round budget" in err
    assert "did not produce a verifiable backend" not in err


async def test_frontend_retry_starts_from_a_clean_entry_file(tmp_path: Path, monkeypatch):
    """The first attempt's truncated remains must not survive into the second (append!)."""
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    st.primary = "dashboard.html"
    calls = {"n": 0}
    seen_before_second: dict = {}

    async def fake_loop(**kw):
        calls["n"] += 1
        target = tmp_path / "dashboard.html"
        if calls["n"] == 1:
            target.write_text("<html><body><div")  # truncated, no </body>
            return {"files_written": ["dashboard.html"], "rounds_used": 1, "summary": "s"}
        seen_before_second["existed"] = target.exists()
        target.write_text(
            '<html><head><meta name="viewport" content="width=device-width">'
            '</head><body><div id="a"></div></body></html>'
        )
        return {"files_written": ["dashboard.html"], "rounds_used": 1, "summary": "s"}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)

    err = await orchestrator._gen_verify_frontend(st)

    assert calls["n"] == 2, "the first attempt should have failed verification"
    assert seen_before_second["existed"] is False, "the entry file was not removed before the second attempt"
    assert err is None


_VALID_PAGE = (
    '<html><head><meta name="viewport" content="width=device-width">'
    '</head><body><div id="a"></div></body></html>'
)


async def test_frontend_retry_removes_what_the_failed_attempt_wrote(tmp_path: Path, monkeypatch):
    """With no `primary`, the cleanup used to delete the expected `index.html`
    (absent) and leave the failed attempt's `report.html` on disk for
    `mode="a"` to extend (I-44). What the attempt reported writing goes too."""
    from anton.core.tools import tool_handlers

    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    st.primary = None
    monkeypatch.setattr(tool_handlers, "resolve_artifact_store", lambda session: None)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend_live", _no_browser)
    calls = {"n": 0}
    seen_before_second: dict = {}

    async def fake_loop(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            (tmp_path / "report.html").write_text("<html><body><div")  # truncated
            return {"files_written": ["report.html"], "rounds_used": 1, "summary": "s"}
        seen_before_second["report_exists"] = (tmp_path / "report.html").exists()
        (tmp_path / "index.html").write_text(_VALID_PAGE)
        return {"files_written": ["index.html"], "rounds_used": 1, "summary": "s"}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)

    err = await orchestrator._gen_verify_frontend(st)

    assert calls["n"] == 2, "the first attempt should have failed verification"
    assert seen_before_second["report_exists"] is False
    assert err is None
    assert st.files_written == ["index.html"], "a deleted file must not be reported as written"


async def test_primary_follows_the_verified_entry_not_the_first_html(tmp_path: Path, monkeypatch):
    """`partials.html` written before `report.html`: the verifier opened
    `report.html` (it is `primary`), so metadata must keep saying so — the
    first `.html` in the list is a file nobody checked (I-45)."""
    from unittest.mock import Mock

    from anton.core.tools import tool_handlers

    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    st.primary = "report.html"
    store = Mock()
    monkeypatch.setattr(tool_handlers, "resolve_artifact_store", lambda session: store)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend_live", _no_browser)

    async def fake_loop(**kw):
        (tmp_path / "partials.html").write_text("<div>fragment</div>")
        (tmp_path / "report.html").write_text(_VALID_PAGE)
        return {
            "files_written": ["partials.html", "report.html"], "rounds_used": 1, "summary": "s",
        }

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)

    err = await orchestrator._gen_verify_frontend(st)

    assert err is None
    assert st.primary == "report.html"
    store.update.assert_not_called()


async def test_frontend_first_attempt_does_not_delete_anything(tmp_path: Path, monkeypatch):
    """Nothing is deleted before the FIRST attempt: an instant failure would wipe a working version."""
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    st.primary = "dashboard.html"
    previous = tmp_path / "dashboard.html"
    previous.write_text("<html><body>previous good version</body></html>")

    calls = {"n": 0}

    async def fake_loop(**kw):
        calls["n"] += 1
        # Assert ONLY on the first call: GEN_VERIFY_MAX_RETRIES = 1, so a second
        # attempt follows and the cleanup before it deletes the file — which is the
        # correct behaviour. Without this guard the assert would fail on a correct
        # implementation and look like an implementation bug.
        if calls["n"] == 1:
            assert previous.exists(), "the previous version was deleted before the first attempt"
        return "boom"

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    await orchestrator._gen_verify_frontend(st)
    assert calls["n"] == 2


async def test_spec_files_are_reported_separately(tmp_path: Path, monkeypatch):
    """spec.md and openapi.json are generation inputs; the agent must not call them artifacts."""
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    st.gathering_complete = True
    st.session._llm.generate_object = AsyncMock(
        side_effect=AssertionError("the data phase must not call the model here")
    )
    st.session._llm.plan_stream = _stream_mock(type("R", (), {"content": "# Spec"})())

    async def fake_front(state):
        (tmp_path / "dashboard.html").write_text("<body><div id=x></div></body>")
        state.files_written.append("dashboard.html")
        state.record("verify_frontend", "ok")
        return None

    monkeypatch.setattr(orchestrator, "_gen_verify_frontend", fake_front)

    out = await orchestrator.run(st, entry=cp.ENTRY_SPEC)

    assert isinstance(out, dict)
    assert out["files_written"] == ["dashboard.html"]
    assert "spec.md" not in out["files_written"]
    assert "spec.md" in out["internal_files"]
    # The file physically stays in the folder — the change is cosmetic.
    assert (tmp_path / "spec.md").is_file()


# ── The emergency data loop ─────────────────────────────────────────────────

async def test_data_phase_still_fetches_when_a_source_is_unverified(
    tmp_path: Path, monkeypatch
):
    """The loop is emergency-only now, but it is not gone.

    `is_data_enough` was removed because the gathering phase already answered
    that question by calling `finish_gathering`. What survives is the fetch
    path, entered when a declared source has nothing executed against it.
    """
    st = _state(tmp_path)
    st.gathering_complete = True
    st.declared_sources = ["orders table"]
    st.unverified_sources = ["orders table"]

    seq = [
        RequiredData(items=[RequiredDataItem(name="totals", where="db", why="chart")], reasoning="r"),
        FetchVerdict(possible=True, reasoning="yes"),
    ]
    st.session._llm.generate_object = AsyncMock(side_effect=seq)
    fetched = {"n": 0}

    async def fake_fetch(state, required_text=""):
        fetched["n"] += 1
        return "pulled totals", True

    monkeypatch.setattr(orchestrator, "_fetch_data_sample", fake_fetch)

    err = await orchestrator._data_phase(st)

    assert err is None
    assert fetched["n"] == 1, "the fetch loop did not run"
    assert st.unverified_sources == [], "the fetch did not clear the backlog"


async def test_a_correction_that_names_a_new_source_opens_the_loop(tmp_path: Path):
    """The case the whole `unverified_sources` list exists for.

    After a correction the notes are full — they hold the PREVIOUS
    gathering's cells — so an "are the notes empty" check would read that as
    "everything is covered" and the new source would never be fetched.
    """
    st = _state(tmp_path)
    st.gathering_complete = True
    st.data_notes = "Scratchpad `old`:\n```python\nrows = q()\n```"
    st.declared_sources = ["orders table", "returns table"]
    st.unverified_sources = ["returns table"]

    assert orchestrator._needs_data_loop(st) is True


async def test_the_loop_is_skipped_when_gathering_covered_the_data(tmp_path: Path):
    st = _state(tmp_path)
    st.gathering_complete = True
    st.declared_sources = ["orders table"]
    st.unverified_sources = []

    assert orchestrator._needs_data_loop(st) is False
    assert await orchestrator._data_phase(st) is None
    assert any(s.node == "data_check" for s in st.trace)


async def test_an_unfinished_gathering_phase_opens_the_loop(tmp_path: Path):
    """Condition 1: the round budget ran out without `finish_gathering`, so
    nothing vouches for the data. Computable on a cold start too — the flag
    comes back from `discovery.json`."""
    st = _state(tmp_path)
    st.gathering_complete = False

    assert orchestrator._needs_data_loop(st) is True


def test_public_data_sources_skill_is_loaded_from_the_store():
    class _Skill:
        declarative_md = "PUBLIC DATA AND WORLD EVENTS:\n- Google News RSS: ..."

    class _Store:
        def __init__(self):
            self.asked = []

        def load_for_artifacts(self, label):
            self.asked.append(label)
            return _Skill()

    session = AsyncMock()
    session._scratchpads.cell_sandbox = None
    store = _Store()
    session._skill_store = store

    out = orchestrator._public_data_sources_skill(session)

    assert "Google News RSS" in out
    assert store.asked == ["public-data-sources"]


def test_public_data_sources_skill_is_defensive():
    """AsyncMock session: store.load_for_artifacts() returns a coroutine that must be closed.

    We also assert the absence of a RuntimeWarning: without an explicit close() the
    abandoned coroutine surfaces at GC time in the output of an arbitrary test. The
    invariant "a mock session produces no warnings" would otherwise be declared but
    unguarded.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert orchestrator._public_data_sources_skill(AsyncMock()) == ""

    class _MissingStore:
        def load_for_artifacts(self, label):
            return None

    session = AsyncMock()
    session._scratchpads.cell_sandbox = None
    session._skill_store = _MissingStore()
    assert orchestrator._public_data_sources_skill(session) == ""


# ── Separate retry budgets: loop failures vs verification failures ───────────

_VALID_HTML = (
    '<html><head><meta name="viewport" content="width=device-width">'
    '</head><body><div id="a"></div></body></html>'
)


async def test_frontend_loop_failure_does_not_consume_the_verify_retry(
    tmp_path: Path, monkeypatch
):
    """Seen live: attempt 0 died on the round budget, attempt 1
    failed verification — and the trivially fixable CSS was terminal, because
    the loop failure had already burned the only retry."""
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    st.primary = "dashboard.html"
    calls = {"n": 0}

    async def fake_loop(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return "generator exceeded round budget (20) without writing any files."
        target = tmp_path / "dashboard.html"
        if calls["n"] == 2:  # verifiable but invalid
            target.write_text("<html><body>no viewport</body></html>")
        else:  # fixed after the verifier's feedback
            target.write_text(_VALID_HTML)
        return {"files_written": ["dashboard.html"], "rounds_used": 1, "summary": "s"}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)

    err = await orchestrator._gen_verify_frontend(st)

    assert err is None
    assert calls["n"] == 3  # loop retry + verify retry, each from its own budget


async def test_frontend_two_loop_failures_are_terminal_and_named_as_generation(
    tmp_path: Path, monkeypatch
):
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)

    async def fake_loop(**kw):
        return "generator stopped without writing files (round 1/20). Last output: ''"

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)

    err = await orchestrator._gen_verify_frontend(st)

    assert err is not None
    assert err.startswith("Frontend generation failed")
    assert "verification failed" not in err.lower()


async def test_frontend_two_verify_failures_are_terminal_with_honest_count(
    tmp_path: Path, monkeypatch
):
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    st.primary = "dashboard.html"

    async def fake_loop(**kw):
        (tmp_path / "dashboard.html").write_text("<html><body>no viewport</body></html>")
        return {"files_written": ["dashboard.html"], "rounds_used": 1, "summary": "s"}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)

    err = await orchestrator._gen_verify_frontend(st)

    assert err is not None
    assert "Frontend verification failed after 2 attempt(s)" in err


_BROWSER_ERROR = (
    "Loaded in a headless browser, the page logged a console error: "
    "console.error: Route network failed: ROUTES is not defined (desktop)"
)
_SHOT = "iVBORw0KGgo="


def _failed_in_a_browser(entry, *, cell_sandbox, screenshot=True, previous_html=None):
    """`verify_frontend_live` where a browser loaded the page, measured it as
    it opens and behind "Routes", and failed it."""
    from anton.core.artifacts.render_check import CheckedView, RenderScreenshot

    return BrowserCheck(
        status="failed", verdict=VerifyResult(errors=[_BROWSER_ERROR]),
        screenshot=RenderScreenshot(media_type="image/png", data=_SHOT),
        views=(CheckedView(pressed=None, viewports=["desktop"]), CheckedView(pressed="Routes", viewports=["desktop"])),
    )


@pytest.mark.parametrize("page", ["index.html", "dashboard.html"])
async def test_a_last_build_that_fails_only_its_browser_check_comes_back_to_repair(
    tmp_path: Path, monkeypatch, page
):
    """Both builds pass the static checks and fail in a browser. The run
    ends with the page the last build wrote, still on disk, the check's
    findings, the views it measured and its screenshot, and the primary
    names that page, whatever the build called it."""
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)

    async def fake_loop(**kw):
        (tmp_path / page).write_text(_VALID_HTML)
        return {"files_written": [page], "rounds_used": 1, "summary": "s"}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend_live", _failed_in_a_browser)

    result = await orchestrator._gen_verify_frontend(st)

    assert isinstance(result, BrowserCheckFailed)
    assert (result.page, result.findings) == (page, (_BROWSER_ERROR,))
    assert [view.pressed for view in result.views] == [None, "Routes"]
    assert result.screenshot is not None and result.screenshot.data == _SHOT
    assert result.error.startswith("Frontend verification failed after 2 attempt(s)")
    assert (tmp_path / page).is_file()
    assert (st.primary or "index.html") == page


async def test_a_browser_failure_then_a_static_failure_ends_with_the_old_string(tmp_path: Path, monkeypatch):
    """The first build fails in a browser, the second fails the static
    checks, so no browser loaded the page left on disk. The run must not
    offer that page for repair on the strength of the first build's check,
    which `GenState`'s browser fields still hold."""
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    builds = iter([_VALID_HTML, "<html><body>no viewport</body></html>"])

    async def fake_loop(**kw):
        (tmp_path / "index.html").write_text(next(builds))
        return {"files_written": ["index.html"], "rounds_used": 1, "summary": "s"}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend_live", _failed_in_a_browser)

    result = await orchestrator._gen_verify_frontend(st)

    assert isinstance(result, str)
    assert result.startswith("Frontend verification failed after 2 attempt(s)")
    assert st.browser_status == "failed"


async def test_unfinished_loop_output_is_still_verified(tmp_path: Path, monkeypatch):
    """A dict with finished=False (round budget exhausted, files on disk) must
    reach the verifier instead of being treated as a failed attempt."""
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    st.primary = "dashboard.html"
    calls = {"n": 0}

    async def fake_loop(**kw):
        calls["n"] += 1
        (tmp_path / "dashboard.html").write_text(_VALID_HTML)
        return {
            "files_written": ["dashboard.html"], "rounds_used": 20,
            "summary": "(round budget 20 exhausted before finish was called)",
            "finished": False,
        }

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)

    err = await orchestrator._gen_verify_frontend(st)

    assert err is None
    assert calls["n"] == 1  # verified on the spot, no regeneration


async def test_backend_loop_failure_does_not_consume_the_verify_retry(
    tmp_path: Path, monkeypatch
):
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.api_spec = "{}"
    calls = {"gen": 0, "verify": 0}

    async def fake_loop(**kw):
        calls["gen"] += 1
        if calls["gen"] == 1:
            return "generator exceeded round budget (20) without writing any files."
        (tmp_path / "backend.py").write_text("x")
        return {"files_written": ["backend.py"], "rounds_used": 1, "summary": "s"}

    async def fake_verify(**kw):
        calls["verify"] += 1
        if calls["verify"] == 1:
            return VerifyResult(errors=["missing /api/health"]), []
        return VerifyResult(errors=[]), []

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_backend", fake_verify)
    monkeypatch.setattr(orchestrator, "_map_datasources", lambda s, k: ([], []))

    err = await orchestrator._gen_verify_backend(st)

    assert err is None
    assert calls["gen"] == 3


# ── verify_frontend: the headless-browser gate ───────────────────────────────

async def test_frontend_browser_findings_trigger_a_regeneration_with_the_message(
    tmp_path: Path, monkeypatch
):
    """A page that passes every text check but throws on load used to ship.
    Now the browser's own message rides the same retry channel as the static
    rules, so the second attempt knows the exact line to fix."""
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    kickoffs: list[str] = []

    async def fake_loop(**kw):
        kickoffs.append(kw["kickoff"])
        (tmp_path / "index.html").write_text(_VALID_HTML)
        return {"files_written": ["index.html"], "rounds_used": 1, "summary": "s"}

    live_verdicts = iter([
        VerifyResult(errors=[
            "Loaded in a headless browser, the page logged a console error: "
            "Uncaught ReferenceError: state is not defined (line 4)"
        ]),
        VerifyResult(),
    ])
    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(
        orchestrator.verifiers, "verify_frontend_live", lambda entry, *, cell_sandbox, screenshot=True, previous_html=None: _loaded(next(live_verdicts))
    )

    err = await orchestrator._gen_verify_frontend(st)

    assert err is None
    assert len(kickoffs) == 2
    assert "## Verification failed" in kickoffs[1]
    assert "ReferenceError: state is not defined (line 4)" in kickoffs[1]
    assert [r.outcome for r in st.trace if r.node == "verify_frontend"] == ["fail", "ok"]


async def test_frontend_browser_skip_leaves_the_static_verdict_and_is_traced(
    tmp_path: Path, monkeypatch
):
    """No browser (the cloud, a bare CLI): the step rests on the text checks,
    and the trace says the live check did not run — otherwise a clean trace
    row could not be told from a page that was actually loaded."""
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    st.trace_log = Mock()
    monkeypatch.delenv("ANTON_HTML_LINT_BROWSER", raising=False)

    async def fake_loop(**kw):
        (tmp_path / "index.html").write_text(_VALID_HTML)
        return {"files_written": ["index.html"], "rounds_used": 1, "summary": "s"}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend_live", _no_browser)

    err = await orchestrator._gen_verify_frontend(st)

    assert err is None
    st.trace_log.node.assert_any_call(
        "verify_frontend", "browser_skipped",
        "no headless browser configured (ANTON_HTML_LINT_BROWSER unset)",
    )


async def test_frontend_browser_check_is_not_run_for_fullstack(tmp_path: Path, monkeypatch):
    """Under file:// the fullstack page's `/api/*` fetches fail with
    `TypeError: Failed to fetch` — a harness artifact, not a page bug."""
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.api_spec = "{}"

    async def fake_loop(**kw):
        (tmp_path / "static").mkdir(exist_ok=True)
        (tmp_path / "static" / "index.html").write_text(
            _VALID_HTML.replace("</head>", '<meta name="api-base" content=""></head>')
        )
        return {"files_written": ["static/index.html"], "rounds_used": 1, "summary": "s"}

    def must_not_run(entry, *, cell_sandbox, screenshot=True, previous_html=None):
        raise AssertionError("the browser check must not run for a fullstack frontend")

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend_live", must_not_run)

    assert await orchestrator._gen_verify_frontend(st) is None


async def test_frontend_browser_check_waits_for_the_static_checks_to_pass(
    tmp_path: Path, monkeypatch
):
    """A page the text checks already reject is not worth a browser launch:
    the retry gets the static findings first, and the browser sees attempt 2."""
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    calls = {"loop": 0, "live": 0}

    async def fake_loop(**kw):
        calls["loop"] += 1
        html = "<html><body>no viewport</body></html>" if calls["loop"] == 1 else _VALID_HTML
        (tmp_path / "index.html").write_text(html)
        return {"files_written": ["index.html"], "rounds_used": 1, "summary": "s"}

    def fake_live(entry, *, cell_sandbox, screenshot=True, previous_html=None):
        calls["live"] += 1
        return _loaded(VerifyResult())

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend_live", fake_live)

    assert await orchestrator._gen_verify_frontend(st) is None
    assert calls == {"loop": 2, "live": 1}


@pytest.mark.parametrize("refused", [False, True])
async def test_the_browser_check_asks_for_no_screenshot_once_the_model_refused_one(
    tmp_path: Path, monkeypatch, refused
):
    """After the model refused a request carrying a render screenshot, the
    agent gets none, so the service is not asked to take one."""
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    st.session._render_screenshots_refused = refused
    asked: list[bool] = []

    async def fake_loop(**kw):
        (tmp_path / "index.html").write_text(_VALID_HTML)
        return {"files_written": ["index.html"], "rounds_used": 1, "summary": "s"}

    def fake_live(entry, *, cell_sandbox, screenshot=True, previous_html=None):
        asked.append(screenshot)
        return _loaded(VerifyResult())

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend_live", fake_live)

    assert await orchestrator._gen_verify_frontend(st) is None
    assert asked == [not refused]


async def test_relaunch_rebuilds_the_datasource_env_after_the_backend_is_regenerated(
    tmp_path: Path, monkeypatch,
):
    """The retry regenerates `backend.py`, and `_gen_verify_backend` declares
    the datasources the new code reads. An env built once before the loop
    would relaunch without the `DS_*` a rewrite introduced (I-42)."""
    from anton.core.tools import tool_handlers

    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.session._scratchpads = None
    monkeypatch.setattr(tool_handlers, "resolve_artifact_store", lambda session: None)

    envs = iter([{"DS_PG_OLD__HOST": "a"}, {"DS_PG_NEW__HOST": "b"}])
    monkeypatch.setattr(orchestrator, "_launch_datasource_env", lambda state: next(envs))

    seen: list[dict] = []

    async def fake_launch(**kw):
        seen.append(dict(kw["ds_env"]))
        return "install failed" if len(seen) == 1 else {"port": 5000}
    monkeypatch.setattr(orchestrator, "_launch_backend", fake_launch)

    async def fake_regen(state, **kw):
        return None
    monkeypatch.setattr(orchestrator, "_gen_verify_backend", fake_regen)

    async def fake_tail(state, limit=2000):
        return ""
    monkeypatch.setattr(orchestrator, "_tail_log", fake_tail)

    async def fake_probe(state, port):
        return None
    monkeypatch.setattr(orchestrator, "_probe_app", fake_probe)
    monkeypatch.setattr(orchestrator.verifiers, "verify_app_live", lambda url, *, cell_sandbox: None)

    err = await orchestrator._run_and_verify_app(st)

    assert err is None
    assert seen == [{"DS_PG_OLD__HOST": "a"}, {"DS_PG_NEW__HOST": "b"}]
    assert st.app_port == 5000


def test_map_datasources_survives_a_vault_that_raises(tmp_path: Path):
    """After the backend is written and verified, a vault error must be an
    unmapped-keys verdict for the loop, not a crash of the whole run (I-47)."""
    from types import SimpleNamespace

    class _BrokenVault:
        def list_connections(self):
            raise RuntimeError("registry unreadable")

    session = SimpleNamespace(_data_vault=_BrokenVault())
    refs, unmapped = orchestrator._map_datasources(session, ["DS_PG_MAIN__HOST"])
    assert refs == []
    assert unmapped == ["DS_PG_MAIN__HOST"]


_CONTRACT = '{"paths": {"/api/items": {"get": {"responses": {"200": {}}}}}}'


async def test_gen_verify_backend_hands_the_contract_to_the_verifier(tmp_path: Path, monkeypatch):
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.api_spec = _CONTRACT
    seen: dict = {}

    async def fake_loop(**kw):
        return {"files_written": ["backend.py"], "rounds_used": 1, "summary": "s"}

    async def fake_verify(**kw):
        seen.update(kw)
        return VerifyResult(errors=[]), []

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_backend", fake_verify)
    monkeypatch.setattr(orchestrator, "_map_datasources", lambda s, k: ([], []))

    assert await orchestrator._gen_verify_backend(st) is None
    assert seen["api_operations"] == {"GET /api/items"}


async def test_gen_verify_frontend_hands_the_contract_paths_to_the_verifier(tmp_path: Path, monkeypatch):
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.api_spec = _CONTRACT
    seen: dict = {}

    async def fake_loop(**kw):
        (tmp_path / "static").mkdir(exist_ok=True)
        (tmp_path / "static" / "index.html").write_text(_VALID_PAGE)
        return {"files_written": ["static/index.html"], "rounds_used": 1, "summary": "s"}

    def fake_verify(html, *, is_fullstack, api_paths=None, live_app=False, data_ids=None):
        seen["api_paths"] = api_paths
        seen["live_app"] = live_app
        seen["data_ids"] = data_ids
        return VerifyResult(errors=[])

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend", fake_verify)

    assert await orchestrator._gen_verify_frontend(st) is None
    assert seen["api_paths"] == {"/api/items"}
    # No sandbox, no live apps: the page isn't held to a live app's script rule.
    assert seen["live_app"] is False
    # The fullstack page keeps its data behind the API: no block checks.
    assert seen["data_ids"] is None


# ── S-03: skipped checks and accepted warnings reach the result ─────────────

def test_result_shell_always_carries_warnings_and_checks_skipped(tmp_path: Path):
    """Empty lists, not absent keys: the agent's instruction branches on
    them, so a run with nothing to report must still show the fields."""
    st = _state(tmp_path, artifact_type="html-app")
    shell = orchestrator._result_shell(st)
    assert shell["warnings"] == [] and shell["checks_skipped"] == []


async def test_browser_skip_is_reported_as_a_skipped_check(tmp_path: Path, monkeypatch):
    """The trace row alone is invisible to the user; the result has to say
    the page was never loaded in a browser."""
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    monkeypatch.delenv("ANTON_HTML_LINT_BROWSER", raising=False)

    async def fake_loop(**kw):
        (tmp_path / "index.html").write_text(_VALID_HTML)
        return {"files_written": ["index.html"], "rounds_used": 1, "summary": "s"}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend_live", _no_browser)

    assert await orchestrator._gen_verify_frontend(st) is None
    assert st.checks_skipped == [
        "browser load of the page: no headless browser configured "
        "(ANTON_HTML_LINT_BROWSER unset)"
    ]
    assert orchestrator._result_shell(st)["checks_skipped"] == st.checks_skipped


def test_a_skipped_check_is_noted_once_per_reason(tmp_path: Path):
    st = _state(tmp_path)
    orchestrator._note_skipped(st, "browser load of the page: no browser")
    orchestrator._note_skipped(st, "browser load of the page: no browser")
    assert st.checks_skipped == ["browser load of the page: no browser"]


async def test_warnings_of_the_accepted_frontend_attempt_reach_the_result(tmp_path: Path, monkeypatch):
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)

    async def fake_loop(**kw):
        (tmp_path / "index.html").write_text(_VALID_HTML)
        return {"files_written": ["index.html"], "rounds_used": 1, "summary": "s"}

    def fake_verify(html, *, is_fullstack, api_paths=None, live_app=False, data_ids=None):
        return VerifyResult(errors=[], warnings=["Significant blocks have no stable `id` attributes."])

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend", fake_verify)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend_live", lambda entry, *, cell_sandbox, screenshot=True, previous_html=None: _loaded(VerifyResult()))

    assert await orchestrator._gen_verify_frontend(st) is None
    assert st.verify_warnings == ["frontend: Significant blocks have no stable `id` attributes."]
    assert orchestrator._result_shell(st)["warnings"] == st.verify_warnings


# ── S-02: the served fullstack page is loaded in a browser after run_app ────

def _served_state(tmp_path: Path):
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.app_url, st.app_port = "http://127.0.0.1:5555", 5555
    st.trace_log = Mock()
    return st


async def test_served_page_check_notes_the_skip_when_no_browser_runs(tmp_path: Path, monkeypatch):
    st = _served_state(tmp_path)
    monkeypatch.delenv("ANTON_HTML_LINT_BROWSER", raising=False)
    monkeypatch.setattr(orchestrator.verifiers, "verify_app_live", lambda url, *, cell_sandbox: None)

    assert await orchestrator._verify_app_in_browser(st) is None
    assert st.checks_skipped == [
        "browser load of static/index.html via the running backend: "
        "no headless browser configured (ANTON_HTML_LINT_BROWSER unset)"
    ]
    st.trace_log.node.assert_any_call(
        "verify_fullstack", "browser_skipped",
        "no headless browser configured (ANTON_HTML_LINT_BROWSER unset)",
    )


async def test_served_page_check_loads_the_backend_url_and_keeps_warnings(tmp_path: Path, monkeypatch):
    st = _served_state(tmp_path)
    seen: list[str] = []

    def fake_live(url, *, cell_sandbox):
        seen.append(url)
        return VerifyResult(errors=[], warnings=["Loaded in a headless browser, the page rendered no visible text or elements."])
    monkeypatch.setattr(orchestrator.verifiers, "verify_app_live", fake_live)

    assert await orchestrator._verify_app_in_browser(st) is None
    assert seen == ["http://127.0.0.1:5555"]
    assert st.verify_warnings == [
        "frontend (served): Loaded in a headless browser, the page rendered no visible text or elements."
    ]
    assert st.checks_skipped == []


async def test_a_served_page_that_passed_vouches_for_the_page_as_it_opens_only(tmp_path: Path, monkeypatch):
    """The local browser presses no tab or toggle, so the result names the
    page as it opens as the one view checked."""
    st = _served_state(tmp_path)
    monkeypatch.setattr(orchestrator.verifiers, "verify_app_live", lambda url, *, cell_sandbox: VerifyResult(errors=[]))

    assert await orchestrator._verify_app_in_browser(st) is None

    shell = orchestrator._result_shell(st)
    assert shell["verified_in_browser"] is True
    assert shell["browser_checked"] == "as it opens"
    assert shell["browser_unchecked"] == []


async def test_served_page_findings_regenerate_the_frontend_once_then_pass(tmp_path: Path, monkeypatch):
    """The findings are the page's own (its /api/* resolve when served), so
    they go back to the FRONTEND generator as kickoff context; the backend
    keeps running and serves the rewritten file, no relaunch."""
    st = _served_state(tmp_path)
    verdicts = iter([
        VerifyResult(errors=["Loaded in a headless browser from the running backend, the page requested a URL that failed: http://127.0.0.1:5555/logo.png — HTTP 404"]),
        VerifyResult(errors=[]),
    ])
    monkeypatch.setattr(orchestrator.verifiers, "verify_app_live", lambda url, *, cell_sandbox: next(verdicts))
    regen: list[dict] = []

    async def fake_front(state, extra_context=""):
        regen.append({"extra_context": extra_context})
        return None
    monkeypatch.setattr(orchestrator, "_gen_verify_frontend", fake_front)

    assert await orchestrator._verify_app_in_browser(st) is None
    assert len(regen) == 1
    assert "## Loaded in a browser from the running backend — fix these" in regen[0]["extra_context"]
    assert "logo.png — HTTP 404" in regen[0]["extra_context"]
    assert [r.outcome for r in st.trace if r.node == "verify_fullstack"] == ["fail"]


async def test_served_page_that_stays_broken_fails_the_run_with_the_browser_message(tmp_path: Path, monkeypatch):
    st = _served_state(tmp_path)
    broken = VerifyResult(errors=["Loaded in a headless browser, the page logged a console error: TypeError: x is undefined (line 9)"])
    monkeypatch.setattr(orchestrator.verifiers, "verify_app_live", lambda url, *, cell_sandbox: broken)

    async def fake_front(state, extra_context=""):
        return None
    monkeypatch.setattr(orchestrator, "_gen_verify_frontend", fake_front)

    err = await orchestrator._verify_app_in_browser(st)
    assert isinstance(err, str)
    assert err.startswith("The page served at http://127.0.0.1:5555 does not load cleanly: ")
    assert "TypeError: x is undefined (line 9)" in err


async def test_served_page_regeneration_over_budget_propagates_the_sentinel(tmp_path: Path, monkeypatch):
    st = _served_state(tmp_path)
    broken = VerifyResult(errors=["Loaded in a headless browser, the page crashed the renderer process."])
    monkeypatch.setattr(orchestrator.verifiers, "verify_app_live", lambda url, *, cell_sandbox: broken)

    async def fake_front(state, extra_context=""):
        return orchestrator.OVER_BUDGET
    monkeypatch.setattr(orchestrator, "_gen_verify_frontend", fake_front)

    assert await orchestrator._verify_app_in_browser(st) is orchestrator.OVER_BUDGET


async def test_run_app_runs_the_browser_gate_after_the_probe(tmp_path: Path, monkeypatch):
    """Order: launch → probe → browser. A page that stays broken is the run's
    error, with the browser's own message, and the trace shows the fail."""
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.session._workspace = None
    st.api_spec = "{}"
    order: list[str] = []

    async def fake_launch(**kw):
        order.append("launch")
        return {"port": 5000, "url": "http://127.0.0.1:5000"}
    async def fake_probe(state, port):
        order.append("probe")
        return None
    def fake_live(url, *, cell_sandbox):
        order.append(f"browser:{url}")
        return VerifyResult(errors=["Loaded in a headless browser, the page logged a console error: boom (line 1)"])
    async def fake_front(state, extra_context=""):
        order.append("regen_frontend")
        return None
    async def fake_tail(state, limit=2000):
        return ""
    monkeypatch.setattr(orchestrator, "_launch_backend", fake_launch)
    monkeypatch.setattr(orchestrator, "_probe_app", fake_probe)
    monkeypatch.setattr(orchestrator.verifiers, "verify_app_live", fake_live)
    monkeypatch.setattr(orchestrator, "_gen_verify_frontend", fake_front)
    monkeypatch.setattr(orchestrator, "_tail_log", fake_tail)

    err = await orchestrator._run_and_verify_app(st)
    assert err is not None and "does not load cleanly" in err
    assert st.error == err
    assert order == ["launch", "probe", "browser:http://127.0.0.1:5000", "regen_frontend", "browser:http://127.0.0.1:5000"]
    # The backend was launched once: the frontend rewrite needs no relaunch.
    assert order.count("launch") == 1


async def test_run_app_passes_when_the_served_page_is_clean(tmp_path: Path, monkeypatch):
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.session._workspace = None
    st.api_spec = "{}"

    async def fake_launch(**kw):
        return {"port": 5000, "url": "http://127.0.0.1:5000"}
    async def fake_probe(state, port):
        return None
    monkeypatch.setattr(orchestrator, "_launch_backend", fake_launch)
    monkeypatch.setattr(orchestrator, "_probe_app", fake_probe)
    monkeypatch.setattr(orchestrator.verifiers, "verify_app_live", lambda url, *, cell_sandbox: VerifyResult(errors=[]))

    assert await orchestrator._run_and_verify_app(st) is None
    assert [r.outcome for r in st.trace if r.node == "verify_fullstack"] == ["ok"]


async def test_tail_log_redacts_a_logged_token(tmp_path: Path):
    """The tail goes to the backend retry prompt and into the returned error,
    so a token the backend logged must not reach either."""
    token = "FAKE0tok3nZr1xK2mN4pL7vT9wY3sB6dF"
    (tmp_path / "backend.log").write_text(f'INFO got {{"access_token": "{token}"}}\n')

    tail = await orchestrator._tail_log(_state(tmp_path))

    assert "access_token" in tail
    assert token not in tail


async def test_tail_log_cut_between_key_and_token_still_redacts_it(tmp_path: Path):
    """A cut that lands just after the key must not leave the token bare."""
    token = "FAKE0tok3nZr1xK2mN4pL7vT9wY3sB6dF"
    filler = "x" * (2000 - len(token) - len('"}'))
    (tmp_path / "backend.log").write_text(f'INFO got {{"access_token": "{token}"}}{filler}')

    tail = await orchestrator._tail_log(_state(tmp_path))

    assert len(tail) <= 2000
    assert token not in tail


async def test_tail_log_is_scrubbed_off_the_event_loop(tmp_path: Path, monkeypatch):
    """The whole log is scrubbed, so the work must not block the event loop."""
    loop_thread = threading.get_ident()
    scrub_threads: list[int] = []

    def recording_scrub(text):
        scrub_threads.append(threading.get_ident())
        return text

    monkeypatch.setattr(orchestrator, "scrub_output", recording_scrub)
    (tmp_path / "backend.log").write_text("INFO started\n")

    await orchestrator._tail_log(_state(tmp_path))

    assert scrub_threads and loop_thread not in scrub_threads


# ── known_data in the node and generator contexts ───────────────────────────

_SITES = "site,lat,lon\nToronto,43.7,-79.4\nAustin,30.3,-97.7"


def test_the_generators_see_the_known_data(tmp_path: Path):
    st = _state(tmp_path, known_data=_SITES)
    ctx = orchestrator._spec_context(st)
    assert f"{prompts.KNOWN_DATA_HEADER}\n{_SITES}" in ctx


def test_long_known_data_reaches_every_context_whole(tmp_path: Path):
    """A header line over one long JSON line used to be cut down to the
    header alone; nothing is cut now."""
    rows = "Sites (as JSON):\n[" + ",".join(f'{{"site":"s{i}","lat":{i}.5}}' for i in range(600)) + "]"
    st = _state(tmp_path, known_data=rows)
    for context in (
        orchestrator._spec_context(st),
        prompts.build_fetch_data_kickoff(st),
        prompts.build_required_data_prompt(st)[1],
        prompts.build_can_fetch_prompt(st, "- sites")[1],
    ):
        assert f"{prompts.KNOWN_DATA_HEADER}\n{rows}" in context
        assert "(truncated" not in context


def test_without_known_data_no_section_is_added(tmp_path: Path):
    st = _state(tmp_path)
    assert prompts.KNOWN_DATA_HEADER not in orchestrator._spec_context(st)
    assert prompts.KNOWN_DATA_HEADER not in prompts._brief_and_notes(st)


# ── the data loop: "nothing missing" exit and the list the fetch gets ───────


def _slug_catalog(*slugs: str, labels: dict[str, str] | None = None) -> SimpleNamespace:
    """A `DatasourceCatalog` stand-in: the connections' names and the rendered section."""
    labels = labels or {}
    rendered = DATASOURCES_HEADER + "\n" + "\n".join(f"- {s} (REST API)" for s in slugs)
    return SimpleNamespace(
        connections=tuple(
            SimpleNamespace(slug=s, label=labels.get(s, ""), engine="", engine_name="")
            for s in slugs
        ),
        render=lambda slugs=None: rendered,
    )


def _unverified(st, *sources):
    st.gathering_complete = True
    st.declared_sources = list(sources)
    st.unverified_sources = list(sources)
    return st


def _outcomes(st, node):
    return [(s.outcome, s.detail) for s in st.trace if s.node == node]


async def test_rows_in_known_data_need_no_fetch_when_gathering_did_not_mark_them(
    tmp_path: Path, monkeypatch
):
    st = _unverified(_state(tmp_path, known_data=_SITES, datasource_catalog=_slug_catalog("rest-gw")), "sites")
    st.public_sources = "## Public data sources\n- open-meteo"
    seen = []

    async def gen_obj(schema_class, *, system, messages):
        seen.append((schema_class, messages[0]["content"]))
        return RequiredData(items=[], reasoning="the rows are in the known data")

    st.session._llm.generate_object = AsyncMock(side_effect=gen_obj)

    async def no_fetch(state, required_text=""):
        raise AssertionError("nothing is missing, so nothing is fetched")

    monkeypatch.setattr(orchestrator, "_fetch_data_sample", no_fetch)

    assert await orchestrator._data_phase(st) is None
    assert st.unverified_sources == [] and st.gathering_complete is True
    assert [s for s, _ in seen] == [RequiredData]
    user = seen[0][1]
    assert "- rest-gw (REST API)" in user and "open-meteo" in user and "Toronto,43.7,-79.4" in user
    assert _outcomes(st, "define_required_data")[0][0] == "nothing_missing"


@pytest.mark.parametrize("kw, why", [
    (dict(artifact_type="fullstack-stateless-app", is_fullstack=True, known_data=_SITES),
     "the backend reads it at run time"),
    (dict(known_data=""), "no executed code read it and the known data does not hold its values"),
])
async def test_an_empty_list_is_rejected_when_nothing_vouches_for_the_sources(
    tmp_path: Path, monkeypatch, kw, why
):
    st = _unverified(_state(tmp_path, **kw), "sites")
    users = {}

    async def gen_obj(schema_class, *, system, messages):
        users[schema_class] = messages[0]["content"]
        if schema_class is RequiredData:
            return RequiredData(items=[], reasoning="nothing is missing")
        return FetchVerdict(possible=True, reasoning="the pad has it")

    st.session._llm.generate_object = AsyncMock(side_effect=gen_obj)
    got = {}

    async def fake_fetch(state, required_text=""):
        got["text"] = required_text
        return "pulled sites", True

    monkeypatch.setattr(orchestrator, "_fetch_data_sample", fake_fetch)

    assert await orchestrator._data_phase(st) is None
    assert _outcomes(st, "define_required_data") == [("nothing_missing_rejected", "sites")]
    expected_line = f"- sites — needs executed code that reads it: {why}"
    assert expected_line in users[FetchVerdict]
    assert expected_line in got["text"]
    assert "Run code that reads these sources even if their values appear under the known data" in got["text"]


async def test_a_page_without_external_data_needs_no_fetch(tmp_path: Path, monkeypatch):
    """Gathering ended without `finish_gathering` and declared nothing: an
    empty list is accepted, as a successful fetch used to be: a gathering
    without finish_gathering must still be able to leave the loop."""
    st = _state(tmp_path)
    st.gathering_complete = False
    st.session._llm.generate_object = AsyncMock(
        return_value=RequiredData(items=[], reasoning="the page needs no external data")
    )

    async def no_fetch(state, required_text=""):
        raise AssertionError("nothing to fetch")

    monkeypatch.setattr(orchestrator, "_fetch_data_sample", no_fetch)

    assert await orchestrator._data_phase(st) is None
    assert st.gathering_complete is True
    assert st.session._llm.generate_object.await_count == 1


@pytest.mark.parametrize("artifact_type, fetches", [
    ("html-app", 0),
    ("fullstack-stateless-app", 1),
])
async def test_a_repeat_call_builds_from_the_known_data_it_brings(
    tmp_path: Path, monkeypatch, artifact_type, fetches
):
    """The call `not enough data` recommends: same request, data in
    `known_data`. It enters at the spec phase, so gathering never runs."""
    art = tmp_path / "art"
    art.mkdir()
    fullstack = artifact_type != "html-app"
    first = _unverified(
        _state(art, artifact_type=artifact_type, is_fullstack=fullstack, user_request="map the sites"),
        "sites",
    )
    orchestrator._save_checkpoint(first, cp.STAGE_PRD_WRITTEN)
    stored = cp.load(art)
    assert cp.decide_entry(stored, request_fp=cp.request_fingerprint("map the sites")) == cp.ENTRY_SPEC

    again = _state(
        art, artifact_type=artifact_type, is_fullstack=fullstack,
        user_request="map the sites", known_data=_SITES,
    )
    engine._restore(again, stored)
    assert again.unverified_sources == ["sites"]

    again.session._llm.generate_object = AsyncMock(
        side_effect=lambda schema_class, **kw: (
            RequiredData(items=[], reasoning="the rows are in the known data")
            if schema_class is RequiredData
            else FetchVerdict(possible=True, reasoning="the pad has it")
        )
    )
    fetched = []

    async def fake_fetch(state, required_text=""):
        fetched.append(required_text)
        return "pulled sites", True

    monkeypatch.setattr(orchestrator, "_fetch_data_sample", fake_fetch)

    assert await orchestrator._data_phase(again) is None
    assert len(fetched) == fetches
    if fetches:
        assert "the backend reads it at run time" in fetched[0]
        assert "Run code that reads these sources" in fetched[0]


async def test_the_fetch_gets_the_full_list_define_wrote(tmp_path: Path, monkeypatch):
    long_why = "needed for the chart " * 30
    st = _unverified(_state(tmp_path), "orders")
    st.session._llm.generate_object = AsyncMock(side_effect=[
        RequiredData(items=[RequiredDataItem(name="orders", where="rest-gw /orders", why=long_why)], reasoning="r"),
        FetchVerdict(possible=True, reasoning="yes"),
    ])
    got = {}

    async def fake_fetch(state, required_text=""):
        got["text"] = required_text
        return "pulled orders", True

    monkeypatch.setattr(orchestrator, "_fetch_data_sample", fake_fetch)

    await orchestrator._data_phase(st)
    expected = f"- orders — from rest-gw /orders ({long_why})"
    assert got["text"] == expected
    assert _outcomes(st, "define_required_data") == [("done", expected)]


async def test_the_fetch_loop_gets_the_list_the_pads_and_the_language(tmp_path: Path, monkeypatch):
    st = _state(
        tmp_path, user_language="Spanish",
        scratchpads_context="## Scratchpads already in this session\n- main (3 cells)",
    )
    seen = {}

    async def fake_run_loop(**kw):
        seen.update(kw)
        return {"files_written": [], "rounds_used": 1, "summary": "s", "scratchpad_execs": []}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_run_loop)
    await orchestrator._fetch_data_sample(st, "- sites — from rest-gw (map)")
    assert "Write `reasoning` and any summary in Spanish." in seen["system"]
    assert "## Data to fetch in this step\n- sites — from rest-gw (map)" in seen["kickoff"]
    assert "- main (3 cells)" in seen["kickoff"]


def test_a_repeat_call_without_known_data_builds_from_the_saved_copy(tmp_path: Path):
    art = tmp_path / "art"
    art.mkdir()
    orchestrator._save_checkpoint(_state(art, user_request="map", known_data=_SITES), cp.STAGE_PRD_WRITTEN)
    stored = cp.load(art)
    assert stored.known_data == _SITES

    again = _state(art, user_request="map")
    engine._restore(again, stored, entry=cp.ENTRY_SPEC)
    assert again.known_data == _SITES
    assert again.call_changed is False

    other = _state(art, user_request="map", known_data="site,lat\nOslo,59.9")
    engine._restore(other, stored, entry=cp.ENTRY_SPEC)
    assert other.known_data == "site,lat\nOslo,59.9"
    assert other.call_changed is True


def test_a_new_iteration_without_known_data_reads_the_sources_its_rows_verified(tmp_path: Path):
    """A built page, then a new request that brings no rows: the old paste
    may be what the user asked to drop, so it is not inherited, and the
    sources only it verified go back to the data loop."""
    art = tmp_path / "art"
    art.mkdir()
    first = _state(art, user_request="map", known_data=_SITES)
    first.declared_sources = ["sites", "regions"]
    first.known_data_verified = ["sites"]
    orchestrator._save_checkpoint(first, cp.STAGE_GENERATED)
    stored = cp.load(art)
    assert stored.known_data_verified == ["sites"]

    again = _state(art, user_request="map")
    engine._restore(again, stored, entry=cp.ENTRY_NEW_ITERATION)
    assert again.known_data == ""
    assert again.unverified_sources == ["sites"]
    assert again.known_data_verified == []
    assert orchestrator._needs_data_loop(again)

    brought = _state(art, user_request="map", known_data="site,lat\nOslo,59.9")
    engine._restore(brought, stored, entry=cp.ENTRY_NEW_ITERATION)
    assert brought.known_data == "site,lat\nOslo,59.9"
    assert brought.unverified_sources == []


# ── the fetch verdict and the forced fetch ──────────────────────────────────

_SITES_ITEM = RequiredDataItem(name="sites", where="rest-gw /sites endpoint", why="map")


@pytest.mark.parametrize("loop_result", [
    "generator exceeded round budget (20) without writing any files.",
    {"files_written": [], "rounds_used": 1, "summary": "nothing", "scratchpad_execs": []},
    {"files_written": [], "rounds_used": 1, "summary": "tried",
     "scratchpad_execs": [{"name": "pad", "code": "get('/sites')", "output": "401", "ok": False}]},
])
async def test_a_fetch_without_a_cell_that_succeeded_is_not_data(tmp_path: Path, monkeypatch, loop_result):
    async def fake_run_loop(**kw):
        return loop_result

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_run_loop)
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    notes, ok = await orchestrator._fetch_data_sample(st)
    assert ok is False
    if isinstance(loop_result, dict):
        assert "(the fetch step ran no cell that succeeded)" in notes


async def test_one_cell_that_succeeded_makes_the_fetch_count(tmp_path: Path, monkeypatch):
    async def fake_run_loop(**kw):
        return {
            "files_written": [], "rounds_used": 2, "summary": "read the sites",
            "scratchpad_execs": [
                {"name": "pad", "code": "get('/site')", "output": "404", "ok": False},
                {"name": "pad", "code": "rows = get('/sites')", "output": "13 rows", "ok": True},
            ],
        }

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_run_loop)
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    notes, ok = await orchestrator._fetch_data_sample(st)
    assert ok is True
    assert "Scratchpad `pad` (failed):" in notes
    assert "(the fetch step ran no cell that succeeded)" not in notes


async def test_failed_fetches_keep_the_sources_unverified_until_the_loop_gives_up(
    tmp_path: Path, monkeypatch
):
    st = _state(tmp_path)
    st.gathering_complete = False
    st.declared_sources = ["sites"]
    st.unverified_sources = ["sites"]
    st.session._llm.generate_object = AsyncMock(
        side_effect=lambda schema_class, **kw: (
            RequiredData(items=[_SITES_ITEM], reasoning="sites are missing")
            if schema_class is RequiredData
            else FetchVerdict(possible=True, reasoning="rest-gw has them")
        )
    )
    fetched = []

    async def fake_fetch(state, required_text=""):
        fetched.append(state.data_notes)
        return "(data fetch step reported: 401)", False

    monkeypatch.setattr(orchestrator, "_fetch_data_sample", fake_fetch)

    err = await orchestrator._data_phase(st)
    assert len(fetched) == orchestrator.DATA_LOOP_MAX
    assert "(data fetch step reported: 401)" in fetched[1], "the next iteration did not see the failure"
    assert err.startswith("not enough data: the data loop did not converge")
    assert st.unverified_sources == ["sites"]
    assert st.gathering_complete is False
    assert [o for o, _ in _outcomes(st, "fetch_data_sample")] == ["fail"] * orchestrator.DATA_LOOP_MAX


async def test_a_no_with_a_connection_gets_one_forced_fetch_then_names_where_it_looked(
    tmp_path: Path, monkeypatch
):
    st = _unverified(_state(tmp_path, datasource_catalog=_slug_catalog("rest-gw", "crm")), "sites")
    st.session._llm.generate_object = AsyncMock(
        side_effect=lambda schema_class, **kw: (
            RequiredData(items=[_SITES_ITEM], reasoning="sites are missing")
            if schema_class is RequiredData
            else FetchVerdict(possible=False, reasoning="no source holds the sites")
        )
    )
    fetched = []

    async def fake_fetch(state, required_text=""):
        fetched.append(required_text)
        return "listed the endpoints, no sites", True

    monkeypatch.setattr(orchestrator, "_fetch_data_sample", fake_fetch)

    err = await orchestrator._data_phase(st)
    assert len(fetched) == 1
    assert err == "not enough data: no source holds the sites. Looked in: rest-gw, rest-gw /sites endpoint."
    assert "crm" not in err
    assert [o for o, _ in _outcomes(st, "is_possible_to_fetch")] == ["no_but_trying", "no"]


async def test_a_forced_fetch_that_finds_the_data_lets_the_build_go_on(tmp_path: Path, monkeypatch):
    st = _unverified(
        _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True,
               datasource_catalog=_slug_catalog("rest-gw")),
        "sites",
    )
    st.session._llm.generate_object = AsyncMock(side_effect=[
        RequiredData(items=[_SITES_ITEM], reasoning="sites are missing"),
        FetchVerdict(possible=False, reasoning="not sure any source has them"),
        RequiredData(items=[], reasoning="the fetch read the sites"),
    ])

    async def fake_fetch(state, required_text=""):
        return "read 13 sites from rest-gw", True

    monkeypatch.setattr(orchestrator, "_fetch_data_sample", fake_fetch)

    assert await orchestrator._data_phase(st) is None
    assert st.unverified_sources == [] and st.gathering_complete is True
    assert _outcomes(st, "define_required_data")[-1][0] == "nothing_missing"


async def test_a_no_without_connections_fails_without_a_fetch(tmp_path: Path, monkeypatch):
    st = _unverified(_state(tmp_path), "sites")
    st.session._llm.generate_object = AsyncMock(side_effect=[
        RequiredData(items=[_SITES_ITEM], reasoning="sites are missing"),
        FetchVerdict(possible=False, reasoning="nothing is connected"),
    ])

    async def no_fetch(state, required_text=""):
        raise AssertionError("no connection, no forced fetch")

    monkeypatch.setattr(orchestrator, "_fetch_data_sample", no_fetch)

    err = await orchestrator._data_phase(st)
    assert err == "not enough data: nothing is connected"


def test_the_error_names_the_first_connections_when_none_is_mentioned(tmp_path: Path):
    st = _state(tmp_path, datasource_catalog=_slug_catalog("a", "b", "c", "d", "e", "f"))
    st.forced_fetch_used = True
    err = orchestrator._not_enough_data(st, "gave up", None)
    assert err == "not enough data: gave up. Looked in: a, b, c, d, e."
    st2 = _state(tmp_path, datasource_catalog=_slug_catalog("a"))
    st2.forced_fetch_used = True
    assert orchestrator._not_enough_data(st2, "gave up.", None) == "not enough data: gave up. Looked in: a."
    assert orchestrator._not_enough_data(st2, "why?", None) == "not enough data: why? Looked in: a."


def test_the_error_names_the_connections_the_required_data_mentions(tmp_path: Path):
    """The backend prompt's rule: slug or label, 3+ characters, so `db`
    does not match inside "mongodb"."""
    st = _state(tmp_path, datasource_catalog=_slug_catalog("db", "crm", "wh1", labels={"wh1": "Warehouse"}))
    st.forced_fetch_used = True
    required = RequiredData(
        items=[
            RequiredDataItem(name="orders", where="mongodb collection on CRM", why="chart"),
            RequiredDataItem(name="stock", where="the warehouse tables", why="chart"),
        ],
        reasoning="r",
    )
    err = orchestrator._not_enough_data(st, "no orders ", required)
    assert err == (
        "not enough data: no orders. Looked in: crm, wh1, "
        "mongodb collection on CRM, the warehouse tables."
    )


async def test_no_forced_fetch_when_no_iteration_is_left_to_judge_it(tmp_path: Path, monkeypatch):
    """A forced fetch runs after can-fetch said no; only the next define can
    tell whether it found anything. On the last iteration there is no next
    define, so the "no" stands."""
    st = _unverified(_state(tmp_path, datasource_catalog=_slug_catalog("rest-gw")), "sites")
    st.session._llm.generate_object = AsyncMock(side_effect=[
        RequiredData(items=[_SITES_ITEM], reasoning="sites are missing"),
        FetchVerdict(possible=True, reasoning="rest-gw has them"),
        RequiredData(items=[_SITES_ITEM], reasoning="sites are missing"),
        FetchVerdict(possible=True, reasoning="rest-gw has them"),
        RequiredData(items=[_SITES_ITEM], reasoning="sites are missing"),
        FetchVerdict(possible=False, reasoning="the endpoint keeps failing"),
    ])
    fetched = []

    async def fake_fetch(state, required_text=""):
        fetched.append(required_text)
        return "(data fetch step reported: 500)", False

    monkeypatch.setattr(orchestrator, "_fetch_data_sample", fake_fetch)

    err = await orchestrator._data_phase(st)
    assert len(fetched) == 2
    assert err == "not enough data: the endpoint keeps failing"
    assert st.forced_fetch_used is False
    assert [o for o, _ in _outcomes(st, "is_possible_to_fetch")] == ["yes", "yes", "no"]


async def test_a_rejected_empty_list_does_not_become_the_users_reason(tmp_path: Path, monkeypatch):
    """Define said the rows are there; the loop rejected that and every fetch
    failed. The error must not tell the user the data was there."""
    st = _unverified(_state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True), "sites")
    st.session._llm.generate_object = AsyncMock(
        side_effect=lambda schema_class, **kw: (
            RequiredData(items=[], reasoning="the rows are already in the known data")
            if schema_class is RequiredData
            else FetchVerdict(possible=True, reasoning="the pad has it")
        )
    )

    async def fake_fetch(state, required_text=""):
        return "(data fetch step reported: 500)", False

    monkeypatch.setattr(orchestrator, "_fetch_data_sample", fake_fetch)

    err = await orchestrator._data_phase(st)
    assert err.startswith("not enough data: the data loop did not converge")
    assert err.endswith("Last assessment: no executed code read sites")
    assert "already in the known data" not in err


def test_a_backend_heading_in_known_data_does_not_replace_the_specs(tmp_path: Path):
    from anton.core.tools.generate_artifact import engine as gen_engine

    (tmp_path / "spec.md").write_text("## Backend\n- `GET /api/sites` — the sites\n", encoding="utf-8")
    st = _state(
        tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True,
        known_data="## Backend\n- GET /api/regions\n\nsite,lat\nToronto,43.7",
    )
    ctx = orchestrator._spec_context(st)
    assert set(gen_engine._declared_api_paths(ctx)) == {"/api/sites"}
    assert ctx.index("## Technical specification") < ctx.index(prompts.KNOWN_DATA_HEADER)
    assert ctx.rstrip().endswith(prompts.KNOWN_DATA_FOOTER) or prompts.KNOWN_DATA_FOOTER in ctx


async def test_known_data_is_scrubbed_before_any_step_sees_it(tmp_path: Path, monkeypatch):
    seen = {}

    async def fake_run(state, *, entry):
        seen["known_data"] = state.known_data
        return {"status": "generated", "files_written": [], "trace": []}

    monkeypatch.setattr(orchestrator, "run", fake_run)
    secret = "sk-ant-api03-" + "A" * 40
    session = _session()
    session._scratchpads.agent_pads = Mock(return_value=set())
    await engine.generate(
        session=session, artifact_type="html-app", artifact_path=tmp_path, slug="a",
        user_request="map the sites", agent_understanding="a map",
        known_data=f"site,lat\nToronto,43.7\nfetched with key {secret}",
    )
    assert "Toronto,43.7" in seen["known_data"]
    assert secret not in seen["known_data"]


# ── data files ──────────────────────────────────────────────────────────────


def test_a_finished_run_keeps_only_its_data_files(tmp_path: Path):
    folder = tmp_path / DATA_DIRNAME
    folder.mkdir()
    for name in ("1.json", "2.json"):
        (folder / name).write_text("[]")
    st = _state(tmp_path)
    st.data_files = [DataFile("2", ["orders"])]
    orchestrator._finish(st)
    assert sorted(p.name for p in folder.iterdir()) == ["2.json"]
    assert cp.load(tmp_path).data_files == [{"id": "2", "sources": ["orders"], "description": ""}]


def test_spec_context_carries_the_data_files_after_known_data(tmp_path: Path):
    (tmp_path / DATA_DIRNAME).mkdir()
    (tmp_path / DATA_DIRNAME / "1.json").write_text(json.dumps([{"Id": "SO-5001"}]))
    st = _state(tmp_path, known_data="scratchpad `p`, variable `df` (DataFrame)")
    st.data_files = [DataFile("1", ["orders"])]
    text = orchestrator._spec_context(st)
    assert text.index("## Known data") < text.index("## Data files")
    assert "SO-5001" in text
    assert "## Data files" in prompts._brief_and_notes(st)


async def test_hot_path_spec_sees_the_files_the_data_loop_saved(tmp_path: Path, monkeypatch):
    """The data loop runs after the PRD; on the hot path the spec step reads
    only the gathering history, so its files are shown to it separately."""
    (tmp_path / DATA_DIRNAME).mkdir()
    (tmp_path / DATA_DIRNAME / "2.json").write_text(json.dumps([{"Id": "SO-5001"}]))
    st = _state(tmp_path)
    st.messages = [{"role": "user", "content": "gathering"}, {"role": "assistant", "content": "ok"}]
    st.data_files = [DataFile("2", ["orders"])]
    st.loop_data_ids = ["2"]
    seen = {}

    async def fake_plan(session, **kw):
        seen["user"] = kw["user"]
        return "## Insights\n- x", None

    monkeypatch.setattr(orchestrator.engine, "_plan_whole_document", fake_plan)
    assert await orchestrator._write_tech_spec(st) is None
    assert "## Data files" in seen["user"] and "SO-5001" in seen["user"]


_GOOD_PAGE = (
    '<html><head><meta name="viewport" content="width=device-width"></head><body><div id="a"></div>'
    '<script type="application/json" data-anton-data="1"></script>'
    "<script>const d=JSON.parse(document.querySelector('script[data-anton-data=\"1\"]').textContent);</script>"
    "</body></html>"
)


async def test_frontend_fills_the_data_blocks_before_the_checks(tmp_path: Path, monkeypatch):
    (tmp_path / DATA_DIRNAME).mkdir()
    (tmp_path / DATA_DIRNAME / "1.json").write_text(json.dumps([{"Id": "SO-5001"}]))
    st = _state(tmp_path)
    st.data_files = [DataFile("1", ["orders"])]

    async def fake_loop(**kw):
        (tmp_path / "index.html").write_text(_GOOD_PAGE)
        return {"files_written": ["index.html"], "rounds_used": 1, "summary": "s"}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend_live", _no_browser)
    assert await orchestrator._gen_verify_frontend(st) is None
    assert "SO-5001" in (tmp_path / "index.html").read_text()


async def test_an_oversized_page_stops_without_a_retry(tmp_path: Path, monkeypatch):
    (tmp_path / DATA_DIRNAME).mkdir()
    # The file fits the limit it is read with; the page with it does not.
    (tmp_path / DATA_DIRNAME / "1.json").write_text(json.dumps(["x" * 400]))
    st = _state(tmp_path)
    st.data_files = [DataFile("1", ["orders"])]
    calls = {"n": 0}

    async def fake_loop(**kw):
        calls["n"] += 1
        (tmp_path / "index.html").write_text(_GOOD_PAGE)
        return {"files_written": ["index.html"], "rounds_used": 1, "summary": "s"}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.data_files, "MAX_PAGE_BYTES", len(_GOOD_PAGE) + 300)
    err = await orchestrator._gen_verify_frontend(st)
    assert calls["n"] == 1 and err is not None and "page limit" in err


async def test_an_unknown_block_id_is_retried(tmp_path: Path, monkeypatch):
    (tmp_path / DATA_DIRNAME).mkdir()
    (tmp_path / DATA_DIRNAME / "1.json").write_text("[1]")
    st = _state(tmp_path)
    st.data_files = [DataFile("1", ["orders"])]
    pages = iter([_GOOD_PAGE.replace('data-anton-data="1"></script>', 'data-anton-data="9"></script>'), _GOOD_PAGE])

    async def fake_loop(**kw):
        (tmp_path / "index.html").write_text(next(pages))
        return {"files_written": ["index.html"], "rounds_used": 1, "summary": "s"}

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_frontend_live", _no_browser)
    assert await orchestrator._gen_verify_frontend(st) is None


def test_filling_the_data_blocks_writes_nothing_once_the_run_is_abandoned(tmp_path: Path):
    (tmp_path / DATA_DIRNAME).mkdir()
    (tmp_path / DATA_DIRNAME / "1.json").write_text(json.dumps([{"Id": "SO-5001"}]))
    st = _state(tmp_path)
    st.data_files = [DataFile("1", ["orders"])]
    page = tmp_path / "index.html"
    page.write_text(_GOOD_PAGE)

    st.abandoned.set()
    assert orchestrator._fill_data_blocks(st, page) is None
    assert page.read_text() == _GOOD_PAGE

    st.abandoned.clear()
    assert orchestrator._fill_data_blocks(st, page) is None
    assert "SO-5001" in page.read_text()


def test_filling_the_data_blocks_leaves_an_unreadable_page_to_the_checks(tmp_path: Path):
    st = _state(tmp_path)
    st.data_files = [DataFile("1", ["orders"])]
    assert orchestrator._fill_data_blocks(st, tmp_path / "missing.html") is None
