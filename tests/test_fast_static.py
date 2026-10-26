from types import SimpleNamespace

import pytest

from anton.core.llm.provider import LLMResponse, StreamComplete
from anton.core.tools.generate_artifact import fast_static
from anton.core.tools.generate_artifact.state import BrowserCheck, GenState


def test_observed_rows_survive_delegation_without_the_old_output_excerpt_limit():
    observed = "observed-record:" + "r" * 2500 + ":final-value=42"
    session = SimpleNamespace(history=[
        {"content": [{"type": "tool_use", "id": "read", "name": "scratchpad"}]},
        {"content": [{"type": "tool_result", "tool_use_id": "read", "content": observed}]},
    ])
    assert observed in fast_static.source_evidence(session)


async def test_source_contract_reaches_the_delegate_and_live_apps_keep_the_normal_pipeline(tmp_path, monkeypatch):
    session = SimpleNamespace(history=[
        {"content": [{"type": "tool_use", "id": "contract", "name": "recall_skill"}]},
        {"content": [{"type": "tool_result", "tool_use_id": "contract", "content": "Documented API contract"}]},
    ])
    state = GenState(session=session, artifact_type="fullstack-stateful-app", artifact_path=tmp_path,
                     slug="notes", user_request="Keep notes between visits", act_first=True)

    async def gathering(received):
        assert "Documented API contract" in received.pipeline_system
        assert all(t["name"] in {"scratchpad", "web_fetch", "web_search", "ask_user", "finish_gathering"}
                   for t in received.pipeline_tools)
        received.final_artifact_type = "fullstack-stateful-app"

    from anton.core.tools.generate_artifact.discovery import engine
    monkeypatch.setattr(engine, "run_gathering_loop", gathering)
    assert await fast_static.prepare(state) is False
    assert state.artifact_type == "fullstack-stateful-app"
    assert state.is_fullstack
    assert state._pipeline_tools is None


async def test_profile_stage_caches_records_and_keeps_analysis_completion_separate(tmp_path, monkeypatch):
    from anton.core.tools.generate_artifact.discovery import engine
    monkeypatch.setenv("ANTON_STATIC_PROFILE_FIRST", "on")
    monkeypatch.setenv("ANTON_STATIC_BATCH_GATHER", "on")
    state = GenState(session=SimpleNamespace(history=[]), artifact_type="html-app",
                     artifact_path=tmp_path, slug="appointments", act_first=True)

    async def gathering(received):
        system = received.pipeline_system
        assert str(tmp_path / "source-records.json") in system
        assert "source_profile" in system
        assert "inspect the returned profile" in system
        assert "Do not finish collection" in system
        assert "fetching the same rows again" in system
        assert "one scratchpad execution when practical" not in system
        received.gathering_complete = True
        received.final_artifact_type = "html-app"

    monkeypatch.setattr(engine, "run_gathering_loop", gathering)
    assert await fast_static.prepare(state)


def test_explicit_cowork_home_does_not_inherit_daily_app_dotenv(tmp_path, monkeypatch):
    from anton.config.settings import _build_env_files
    (tmp_path / ".env").write_text("ACT_FIRST=true\n")
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    assert _build_env_files() == [".env", str(tmp_path / ".env")]


@pytest.mark.parametrize("body", ["Here is HTML", "&lt;html&gt;x&lt;/html&gt;", "<html><body>partial"])
def test_incomplete_or_printed_source_cannot_replace_a_page(body):
    with pytest.raises(ValueError):
        fast_static.document(body)


async def test_direct_generation_uses_one_coding_call_and_the_existing_verifiers(tmp_path, monkeypatch):
    html = '<!doctype html><html lang="en"><head><meta name="viewport" content="width=device-width"></head><body>Revenue: 42</body></html>'
    calls = []

    async def stream(**kwargs):
        calls.append(kwargs)
        yield StreamComplete(LLMResponse(content=html))

    monkeypatch.setattr(fast_static.verifiers, "verify_frontend_live",
                        lambda *args, **kwargs: BrowserCheck(status="passed"))
    session = SimpleNamespace(_llm=SimpleNamespace(code_stream=stream), history=[])
    state = GenState(session=session, artifact_type="html-app", artifact_path=tmp_path,
                     slug="revenue", user_request="Show revenue", user_language="English", act_first=True)
    result = await fast_static.run(state, mode="direct")
    assert result["status"] == "generated"
    assert result["verified_in_browser"] is True
    assert len(calls) == 1
    assert (tmp_path / "index.html").read_text() == html


async def test_primary_cannot_read_or_write_outside_the_artifact(tmp_path):
    session = SimpleNamespace(history=[])
    state = GenState(session=session, artifact_type="html-app", artifact_path=tmp_path,
                     slug="revenue", primary="../outside.html", act_first=True)
    assert "inside the artifact" in await fast_static.run(state, mode="direct")


async def test_direct_writer_respects_the_existing_spend_guard(tmp_path):
    class Stop:
        def winding_down(self):
            return True

    # The generic state's guard drives the same budget result as the normal pipeline.
    state = GenState(session=SimpleNamespace(), artifact_type="html-app", artifact_path=tmp_path,
                     slug="budget", act_first=True)
    state.winding_down = Stop().winding_down
    result = await fast_static.run(state, mode="direct")
    assert result["status"] == "stopped_over_budget"
    assert not (tmp_path / "index.html").exists()


async def test_canonical_channel_avoids_conflicting_console_datasets_and_keeps_product_rules(tmp_path, monkeypatch):
    import json
    from anton.core.tools.generate_artifact import observed_data

    monkeypatch.setenv("ANTON_STATIC_DATA_BINDING", "on")
    monkeypatch.setenv("ANTON_STATIC_CANONICAL_ONLY", "on")
    monkeypatch.setenv("ANTON_STATIC_STANDARD_DESIGN", "on")
    monkeypatch.setenv("ANTON_STATIC_PAGE_CHARS", "9000")
    monkeypatch.setenv("ANTON_STATIC_WRITER_TOKENS", "4000")
    canonical = {"records": [{"region": "west", "amount": 42}]}
    old = [{"amount": 999}]
    (tmp_path / "observed-data.json").write_text(json.dumps(canonical))
    html = '<html lang="en"><head><meta name="viewport" content="width=device-width"></head><body>42</body></html>'
    calls = []

    async def stream(**kwargs):
        calls.append(kwargs)
        yield StreamComplete(LLMResponse(content=html))

    session = SimpleNamespace(_llm=SimpleNamespace(code_stream=stream), history=[
        {"content": [{"type": "tool_use", "id": "source", "name": "scratchpad"}]},
        {"content": [{"type": "tool_result", "tool_use_id": "source", "content": json.dumps(old)}]},
    ])
    monkeypatch.setattr(fast_static.verifiers, "verify_frontend_live", lambda *args, **kwargs: BrowserCheck(status="passed"))
    state = GenState(session=session, artifact_type="html-app", artifact_path=tmp_path, slug="totals", act_first=True)
    result = await fast_static.run(state, mode="direct")
    assert result["status"] == "generated"
    data = observed_data.previous_data((tmp_path / "index.html").read_text())
    assert list(data.values()) == [canonical]
    assert observed_data.identity(old) not in calls[0]["messages"][0]["content"]
    assert "9000 characters" in calls[0]["system"]
    assert "ALWAYS use Apache ECharts" in calls[0]["system"]
    assert calls[0]["max_tokens"] == 4000
async def test_coherent_gather_changes_only_its_copied_control_tool(tmp_path, monkeypatch):
    from anton.core.tools.generate_artifact.discovery import engine, prompts
    monkeypatch.setenv("ANTON_STATIC_COHERENT_GATHER", "on")
    monkeypatch.setenv("ANTON_STATIC_FAST_PATH", "direct")
    state = GenState(session=SimpleNamespace(history=[]), artifact_type="html-app", artifact_path=tmp_path,
                     slug="report", act_first=True)
    original = state.pipeline_tools
    original_finish = next(t for t in original if t["name"] == "finish_gathering")
    before = original_finish["description"]
    async def gather(received):
        finish = next(t for t in received.pipeline_tools if t["name"] == "finish_gathering")
        assert "SAME response" in finish["description"]
        assert finish is not original_finish
        assert finish["input_schema"] == original_finish["input_schema"]
        assert "print only a short manifest" in prompts._gathering_instruction(True)
        received.gathering_complete = True
        received.final_artifact_type = "html-app"
    monkeypatch.setattr(engine, "run_gathering_loop", gather)
    assert await fast_static.prepare(state)
    assert original_finish["description"] == before


@pytest.mark.parametrize("installed", [True, False])
async def test_geocoding_is_offered_only_when_the_optional_gazetteer_is_installed(tmp_path, monkeypatch, installed):
    import importlib.util
    from anton.core.tools.generate_artifact.discovery import engine
    monkeypatch.setenv("ANTON_STATIC_GATHER_GEOCODING", "on")
    monkeypatch.setenv("ANTON_STATIC_GEO_CAPABILITY", "on")
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: (object() if installed else None) if name == "geonamescache" else find_spec(name, *a))
    state = GenState(session=SimpleNamespace(history=[]), artifact_type="html-app", artifact_path=tmp_path,
                     slug="map", act_first=True)

    async def gather(received):
        assert ("resolve_place" in received.pipeline_system) is installed
        assert ("geonamescache" in received.pipeline_system) is installed
        received.gathering_complete = True
        received.final_artifact_type = "html-app"

    monkeypatch.setattr(engine, "run_gathering_loop", gather)
    assert await fast_static.prepare(state)
