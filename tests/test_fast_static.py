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


def test_connected_procedures_follow_saved_connection_names_without_loading_other_skills():
    loaded = []

    def load(label):
        loaded.append(label)
        return SimpleNamespace(declarative_md="Query /documented-route with the observed schema.")

    session = SimpleNamespace(
        _skill_store=SimpleNamespace(load=load),
        _data_vault=SimpleNamespace(list_connections=lambda: [{"name": "inventory-prod"}]),
    )
    result = fast_static._connection_procedures(session)
    assert loaded == ["inventory-prod"]
    assert "/documented-route" in result


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
