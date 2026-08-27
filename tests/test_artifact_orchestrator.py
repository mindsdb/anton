from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from anton.core.llm.provider import StreamComplete
from anton.core.tools.generate_artifact import orchestrator
from anton.core.tools.generate_artifact.state import (
    DataVerdict, FetchVerdict, GenState, RequiredData, RequiredDataItem, VerifyResult,
)


async def _one_event_stream(response):
    yield StreamComplete(response=response)


def _stream_mock(response):
    """`plan_stream` fake: every call returns a fresh one-event stream of the
    same response (mirrors `AsyncMock(return_value=...)`)."""
    return Mock(side_effect=lambda **kw: _one_event_stream(response))


def _state(tmp_path, **kw):
    base = dict(
        session=AsyncMock(), artifact_type="html-app", artifact_path=tmp_path,
        slug="a", brief="Show current time", is_fullstack=False,
    )
    base.update(kw)
    return GenState(**base)


# ── Task 7: data phase ───────────────────────────────────────────────────────

async def test_data_phase_short_circuits_when_enough(tmp_path: Path):
    st = _state(tmp_path)
    st.session._llm.generate_object = AsyncMock(
        return_value=DataVerdict(enough=True, reasoning="no data needed")
    )
    err = await orchestrator._data_phase(st)
    assert err is None
    assert st.data_iterations == 0
    assert any(s.node == "is_data_enough" and s.outcome == "yes" for s in st.trace)


async def test_data_phase_terminates_when_impossible(tmp_path: Path):
    st = _state(tmp_path)
    seq = [
        DataVerdict(enough=False, reasoning="need orders"),
        RequiredData(items=[RequiredDataItem(name="orders", where="nowhere", why="chart")], reasoning="r"),
        FetchVerdict(possible=False, reasoning="no orders source connected"),
    ]
    st.session._llm.generate_object = AsyncMock(side_effect=seq)
    err = await orchestrator._data_phase(st)
    assert err is not None
    assert "not enough data" in err.lower()
    assert "no orders source connected" in err  # reasoning carried through


async def test_data_phase_budget_exhausted(tmp_path: Path, monkeypatch):
    st = _state(tmp_path)

    # Always: not enough → required → possible → fetch (writes notes) → repeat.
    def gen_obj(schema_class, **kw):
        if schema_class is DataVerdict:
            return DataVerdict(enough=False, reasoning="still missing")
        if schema_class is RequiredData:
            return RequiredData(items=[RequiredDataItem(name="x", where="db", why="y")], reasoning="r")
        return FetchVerdict(possible=True, reasoning="yes")

    st.session._llm.generate_object = AsyncMock(
        side_effect=lambda schema_class, **kw: gen_obj(schema_class, **kw)
    )

    async def fake_fetch(state):
        return "pulled a sample"
    monkeypatch.setattr(orchestrator, "_fetch_data_sample", fake_fetch)

    err = await orchestrator._data_phase(st)
    assert err is not None
    assert "not enough data" in err.lower()
    assert st.data_iterations == 3


# ── fetch_data_sample: exec-code record + journal handoff ────────────────────

async def test_fetch_data_sample_appends_exec_code(tmp_path: Path, monkeypatch):
    st = _state(tmp_path)

    async def fake_run_loop(**kw):
        return {
            "files_written": [], "rounds_used": 2, "summary": "pulled 100 rows",
            "scratchpad_execs": [
                {"name": "pad", "code": "df = q('select 1')", "output": "ok  100 rows"}
            ],
        }

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_run_loop)
    notes = await orchestrator._fetch_data_sample(st)
    assert notes.startswith("pulled 100 rows")
    assert "### Code executed while fetching" in notes
    assert "df = q('select 1')" in notes
    assert "Output: ok 100 rows" in notes


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

    err = await orchestrator._gen_verify_backend(st)
    assert err is None
    assert calls["gen"] == 2  # one retry
    assert declared["refs"] == ["REF"]


async def test_gen_verify_backend_stateful_wiring(tmp_path: Path, monkeypatch):
    """Stateful: the manifest step is injected and the type reaches the verifier."""
    st = _state(tmp_path, artifact_type="fullstack-stateful-app", is_fullstack=True)
    st.api_spec = "{}"
    captured = {}

    async def fake_loop(**kw):
        captured["injections"] = kw["step_injections"]
        captured["system"] = kw["system"]
        (tmp_path / "backend.py").write_text("x")
        return {"files_written": ["backend.py"], "rounds_used": 1, "summary": "s"}

    async def fake_verify(**kw):
        captured["artifact_type"] = kw.get("artifact_type")
        return VerifyResult(errors=[]), []

    async def fake_declare(state, refs):
        pass

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_backend", fake_verify)
    monkeypatch.setattr(orchestrator, "_map_datasources", lambda s, k: ([], []))
    monkeypatch.setattr(orchestrator, "_declare_datasources", fake_declare)
    monkeypatch.setattr(orchestrator, "_datasource_context", lambda s: "")
    err = await orchestrator._gen_verify_backend(st)
    assert err is None
    assert [t for t, _ in captured["injections"]] == ["backend.py", "state_manifest.json"]
    assert captured["artifact_type"] == "fullstack-stateful-app"
    # The system prompt carries the STATE contract, not the old sqlite model.
    assert "state_manifest.json" in captured["system"]
    assert "sqlite file (or similar) IS allowed" not in captured["system"]


async def test_gen_verify_backend_stateless_wiring(tmp_path: Path, monkeypatch):
    """Stateless: two files only — no manifest step; the type reaches the verifier."""
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.api_spec = "{}"
    captured = {}

    async def fake_loop(**kw):
        captured["injections"] = kw["step_injections"]
        (tmp_path / "backend.py").write_text("x")
        return {"files_written": ["backend.py"], "rounds_used": 1, "summary": "s"}

    async def fake_verify(**kw):
        captured["artifact_type"] = kw.get("artifact_type")
        return VerifyResult(errors=[]), []

    async def fake_declare(state, refs):
        pass

    monkeypatch.setattr(orchestrator.engine, "_run_loop", fake_loop)
    monkeypatch.setattr(orchestrator.verifiers, "verify_backend", fake_verify)
    monkeypatch.setattr(orchestrator, "_map_datasources", lambda s, k: ([], []))
    monkeypatch.setattr(orchestrator, "_declare_datasources", fake_declare)
    monkeypatch.setattr(orchestrator, "_datasource_context", lambda s: "")
    err = await orchestrator._gen_verify_backend(st)
    assert err is None
    assert [t for t, _ in captured["injections"]] == ["backend.py"]
    assert captured["artifact_type"] == "fullstack-stateless-app"


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
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    st.session._llm.generate_object = AsyncMock(
        return_value=DataVerdict(enough=True, reasoning="no data needed")
    )
    st.session._llm.plan_stream = _stream_mock(type("R", (), {"content": "# Spec"})())

    async def fake_front(state):
        (tmp_path / "dashboard.html").write_text("<body><div id=x></div></body>")
        state.files_written.append("dashboard.html")
        state.record("verify_frontend", "ok")
        return None
    monkeypatch.setattr(orchestrator, "_gen_verify_frontend", fake_front)

    out = await orchestrator.run(st)
    assert isinstance(out, dict)
    assert "dashboard.html" in out["files_written"]
    # html-app must NOT run backend / run_app.
    assert not any(s["node"] == "run_app" for s in out["trace"])


async def test_run_data_terminal_returns_error(tmp_path: Path):
    st = _state(tmp_path)
    st.session._llm.generate_object = AsyncMock(side_effect=[
        DataVerdict(enough=False, reasoning="need x"),
        RequiredData(items=[], reasoning="need x from nowhere"),
        FetchVerdict(possible=False, reasoning="no source"),
    ])
    out = await orchestrator.run(st)
    assert isinstance(out, str)
    assert "not enough data" in out.lower()


async def test_run_fullstack_launches_and_verifies(tmp_path: Path, monkeypatch):
    st = _state(tmp_path, artifact_type="fullstack-stateless-app", is_fullstack=True)
    st.session._workspace = None  # _artifact_store returns None → port update skipped
    st.session._llm.generate_object = AsyncMock(
        return_value=DataVerdict(enough=True, reasoning="have data")
    )
    st.session._llm.plan_stream = _stream_mock(type("R", (), {"content": "# Spec"})())

    async def fake_api(state):
        state.api_spec = '{"paths": {"/api/items": {"get": {"parameters": []}}}}'
        return None
    async def fake_back(state):
        state.record("verify_backend", "ok"); return None
    async def fake_front(state):
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

    out = await orchestrator.run(st)
    assert isinstance(out, dict)
    assert launched["health_path"] == "/api/health"
    assert any(s["node"] == "verify_fullstack" and s["outcome"] == "ok" for s in out["trace"])


# ── Task 10: public generate() delegate ──────────────────────────────────────

async def test_public_generate_delegates_to_run(tmp_path: Path, monkeypatch):
    from anton.core.tools.generate_artifact import generate as public_generate

    session = AsyncMock()
    captured = {}
    async def fake_run(state):
        captured["type"] = state.artifact_type
        captured["slug"] = state.slug
        captured["is_fullstack"] = state.is_fullstack
        return {"files_written": [], "summary": "ok", "trace": []}
    monkeypatch.setattr("anton.core.tools.generate_artifact.orchestrator.run", fake_run)

    out = await public_generate(
        session=session, artifact_type="fullstack-stateless-app",
        artifact_path=tmp_path, context="brief", slug="a",
    )
    assert out["summary"] == "ok"
    assert captured["type"] == "fullstack-stateless-app"
    assert captured["slug"] == "a"
    assert captured["is_fullstack"] is True


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
        assert orchestrator._datasource_context(AsyncMock()) == ""
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
    assert orchestrator.HTML_APP_DEFAULT_PRIMARY == "dashboard.html"


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
    unconditionally. Then, with no primary set, a `dashboard.html` left over from a
    previous generation would shadow the fresh `report.html`.
    """
    st = _state(tmp_path, artifact_type="html-app", is_fullstack=False)
    st.primary = None  # the agent set no primary — create_artifact allows that
    (tmp_path / "dashboard.html").write_text("<body>stale from a previous run</body>")
    (tmp_path / "report.html").write_text("<body>written now</body>")
    html = orchestrator._read_frontend_html(st, ["report.html"])
    assert html == "<body>written now</body>"


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
    st.session._llm.generate_object = AsyncMock(
        return_value=DataVerdict(enough=True, reasoning="no data needed")
    )
    st.session._llm.plan_stream = _stream_mock(type("R", (), {"content": "# Spec"})())

    async def fake_front(state):
        (tmp_path / "dashboard.html").write_text("<body><div id=x></div></body>")
        state.files_written.append("dashboard.html")
        state.record("verify_frontend", "ok")
        return None

    monkeypatch.setattr(orchestrator, "_gen_verify_frontend", fake_front)

    out = await orchestrator.run(st)

    assert isinstance(out, dict)
    assert out["files_written"] == ["dashboard.html"]
    assert "spec.md" not in out["files_written"]
    assert "spec.md" in out["internal_files"]
    # The file physically stays in the folder — the change is cosmetic.
    assert (tmp_path / "spec.md").is_file()


# ── Scratchpad inspection ────────────────────────────────────────────────────

class _FakeCell:
    def __init__(self, code, stdout="", error=None):
        self.code = code
        self.stdout = stdout
        self.stderr = ""
        self.error = error
        self.description = ""
        self.logs = ""


class _FakePad:
    def __init__(self, cells):
        self.cells = cells


def _session_with_pads(pads: dict):
    session = AsyncMock()
    session._scratchpads.pads = pads
    return session


def test_pads_named_in_brief_matches_on_word_boundary():
    """A pad named `dash` must not match the word "dashboard"."""
    session = _session_with_pads({"dash": _FakePad([]), "orders": _FakePad([])})
    brief = "Build a dashboard from the `orders` pad."
    assert orchestrator._pads_named_in_brief(session, brief) == ["orders"]


def test_pads_named_in_brief_is_defensive_against_mock_session():
    """session is an AsyncMock: pads is a mock, not a dict.

    We also assert the absence of a RuntimeWarning: rejecting the mock by calling
    `.keys()` inside try/except instead of by type would return an abandoned
    coroutine, which surfaces at GC time in the output of EVERY test with a mock
    session. Elsewhere such a coroutine has to be closed by hand — here it must
    never appear.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert orchestrator._pads_named_in_brief(AsyncMock(), "any brief") == []
        assert orchestrator._live_pad_names(AsyncMock()) == []
        assert orchestrator._pads_dict(AsyncMock()) == {}


def test_inspect_puts_cells_into_data_notes(tmp_path: Path):
    session = _session_with_pads(
        {"orders": _FakePad([
            _FakeCell("df = q('select * from orders')", stdout="1000 rows"),
            _FakeCell("   ", stdout="ignored"),  # empty code is skipped
        ])}
    )
    st = _state(tmp_path, session=session, brief="Use the `orders` scratchpad.")

    orchestrator._inspect_named_scratchpads(st)

    assert "select * from orders" in st.data_notes
    assert "1000 rows" in st.data_notes
    assert "orders" in st.data_notes
    assert any(s.node == "inspect_scratchpads" and s.outcome == "done" for s in st.trace)


def test_inspect_reports_cell_errors_not_just_stdout(tmp_path: Path):
    """Without this a failed cell lands in the notes as code with no result —
    indistinguishable from a successful one that printed nothing."""
    session = _session_with_pads(
        {"orders": _FakePad([
            _FakeCell("import nope", stdout="", error="ModuleNotFoundError: nope")
        ])}
    )
    st = _state(tmp_path, session=session, brief="See the `orders` pad.")

    orchestrator._inspect_named_scratchpads(st)

    assert "ModuleNotFoundError" in st.data_notes


def test_inspect_records_skip_when_nothing_matched(tmp_path: Path):
    """A silent no-op here has already hidden a wrong premise twice."""
    session = _session_with_pads({"unrelated": _FakePad([_FakeCell("x = 1")])})
    st = _state(tmp_path, session=session, brief="No pad is named here.")

    orchestrator._inspect_named_scratchpads(st)

    assert st.data_notes == ""
    step = next(s for s in st.trace if s.node == "inspect_scratchpads")
    assert step.outcome == "skipped"
    assert "unrelated" in step.detail  # the live pads are listed so the skip is visible


def test_inspect_records_skip_when_no_live_pads(tmp_path: Path):
    session = _session_with_pads({})
    st = _state(tmp_path, session=session, brief="Use the `orders` pad.")

    orchestrator._inspect_named_scratchpads(st)

    step = next(s for s in st.trace if s.node == "inspect_scratchpads")
    assert step.outcome == "skipped"
    assert "no live scratchpads" in step.detail


def test_inspect_records_empty_when_pad_has_no_cells(tmp_path: Path):
    session = _session_with_pads({"orders": _FakePad([])})
    st = _state(tmp_path, session=session, brief="Use the `orders` pad.")

    orchestrator._inspect_named_scratchpads(st)

    assert st.data_notes == ""
    step = next(s for s in st.trace if s.node == "inspect_scratchpads")
    assert step.outcome == "empty"


def test_inspect_applies_the_existing_cap(tmp_path: Path):
    """The cap already lives in _render_exec_notes — reuse it, do not write a second one."""
    session = _session_with_pads(
        {"big": _FakePad([_FakeCell("x" * 3000, stdout="y" * 1000) for _ in range(10)])}
    )
    st = _state(tmp_path, session=session, brief="Use the `big` pad.")

    orchestrator._inspect_named_scratchpads(st)

    assert len(st.data_notes) < orchestrator.EXEC_NOTES_MAX + 1000
    assert "omitted for size" in st.data_notes


async def test_data_phase_sees_inspected_cells(tmp_path: Path):
    """The inspection runs BEFORE is_data_enough, or it is pointless."""
    session = _session_with_pads(
        {"orders": _FakePad([_FakeCell("df = q('select 1')", stdout="1 row")])}
    )
    st = _state(tmp_path, session=session, brief="Chart the `orders` pad.")
    seen = {}

    async def fake_generate_object(schema, *, system, messages):
        seen["user"] = messages[0]["content"]
        return DataVerdict(enough=True, reasoning="already have it")

    st.session._llm.generate_object = fake_generate_object

    err = await orchestrator._data_phase(st)

    assert err is None
    assert "select 1" in seen["user"], "is_data_enough did not see the inspected cells"


async def test_data_phase_still_fetches_when_not_enough(tmp_path: Path, monkeypatch):
    """Regression guard: the inspection does NOT replace the data phase.

    Design decision 3: the define_required_data -> is_possible_to_fetch ->
    fetch_data_sample loop stays in full.
    """
    session = _session_with_pads({"orders": _FakePad([_FakeCell("x = 1", stdout="ok")])})
    st = _state(tmp_path, session=session, brief="Chart the `orders` pad.")

    seq = [
        DataVerdict(enough=False, reasoning="need more"),
        RequiredData(items=[RequiredDataItem(name="totals", where="db", why="chart")], reasoning="r"),
        FetchVerdict(possible=True, reasoning="yes"),
        DataVerdict(enough=True, reasoning="now fine"),
    ]
    st.session._llm.generate_object = AsyncMock(side_effect=seq)
    fetched = {"n": 0}

    async def fake_fetch(state):
        fetched["n"] += 1
        return "pulled totals"

    monkeypatch.setattr(orchestrator, "_fetch_data_sample", fake_fetch)

    err = await orchestrator._data_phase(st)

    assert err is None
    assert fetched["n"] == 1, "the fetch loop did not run"


def test_public_data_sources_skill_is_loaded_from_the_store():
    class _Skill:
        declarative_md = "PUBLIC DATA AND WORLD EVENTS:\n- Google News RSS: ..."

    class _Store:
        def __init__(self):
            self.asked = []

        def load(self, label):
            self.asked.append(label)
            return _Skill()

    session = AsyncMock()
    store = _Store()
    session._skill_store = store

    out = orchestrator._public_data_sources_skill(session)

    assert "Google News RSS" in out
    assert store.asked == ["public-data-sources"]


def test_public_data_sources_skill_is_defensive():
    """AsyncMock session: store.load() returns a coroutine that must be closed.

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
        def load(self, label):
            return None

    session = AsyncMock()
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
    """Live run 2026-08-27: attempt 0 died on the round budget, attempt 1
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
