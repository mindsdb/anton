import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.streaming_llm import StreamingLLM

from anton.core.llm.provider import LLMResponse, ToolCall
from anton.core.tools.generate_artifact import static_gathering
from anton.core.tools.generate_artifact.discovery import engine
from anton.core.tools.generate_artifact.state import GenState
from anton.core.tools.registry import ToolOutcome


@pytest.fixture
def atomic(monkeypatch):
    for key in ("ATOMIC_GATHER", "DATA_BINDING", "CANONICAL_ONLY"):
        monkeypatch.setenv("ANTON_STATIC_" + key, "on")


def test_identical_fresh_source_data_is_allowed_but_a_stale_file_is_not(tmp_path):
    path = tmp_path / "observed-data.json"
    path.write_text('{"revenue":42}')
    before = static_gathering.fingerprint(tmp_path)
    with pytest.raises(ValueError, match="fresh"):
        static_gathering.validate_handoff(tmp_path, before)
    path.write_text('{"revenue":42}')
    static_gathering.validate_handoff(tmp_path, before)


@pytest.mark.parametrize("body", ['invalid JSON', '42', '{"access_token":"do-not-embed"}'])
def test_invalid_or_credential_bearing_data_cannot_complete_collection(tmp_path, body):
    (tmp_path / "observed-data.json").write_text(body)
    with pytest.raises(ValueError):
        static_gathering.validate_handoff(tmp_path, None)


def test_handoff_cannot_follow_a_symlink_outside_the_artifact(tmp_path):
    folder = tmp_path / "artifact"
    folder.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"amount":42}')
    (folder / "observed-data.json").symlink_to(outside)
    with pytest.raises((OSError, ValueError)):
        static_gathering.validate_handoff(folder, None)


@pytest.mark.parametrize("failure", [None, "failed-exec", "stale", "invalid", "refused"])
async def test_collection_completes_in_one_call_only_after_successful_fresh_data(tmp_path, monkeypatch, atomic, failure):
    import anton.core.tools.tool_handlers as handlers

    if failure == "stale":
        (tmp_path / "observed-data.json").write_text('{"amount":42}')
    async def execute(session, inp):
        assert "complete_static_artifact" not in inp
        if failure not in {"stale", "failed-exec", "refused"}:
            (tmp_path / "observed-data.json").write_text('not JSON' if failure == "invalid" else '{"amount":42}')
        return ToolOutcome("source rows", ok=failure != "failed-exec",
                           reason="new_scratchpad_challenged" if failure == "refused" else "")
    monkeypatch.setattr(handlers, "handle_scratchpad", execute)
    responses = iter([
        LLMResponse(content="", tool_calls=[ToolCall(id="exec", name="scratchpad", input={
            "action": "exec", "name": "data", "code": "fetch_and_validate()",
            "complete_static_artifact": {"summary": "Scoped revenue", "data_sources": ["warehouse"]}})]),
        LLMResponse(content="", tool_calls=[ToolCall(id="finish", name="finish_gathering", input={
            "summary": "normal fallback", "artifact_type": "html-app"})]),
    ])
    async def response(**kwargs):
        return next(responses)
    session = SimpleNamespace(_llm=StreamingLLM(plan=AsyncMock(side_effect=response), code=AsyncMock(side_effect=response)),
                              question_count=0, elicitor=None, emit=AsyncMock())
    state = GenState(session=session, artifact_type="html-app", artifact_path=tmp_path, slug="revenue", act_first=True)
    await engine.run_gathering_loop(state)
    assert state.gathering_complete
    if failure is None:
        assert session._llm.code.call_count == 0
        assert state.declared_sources == ["warehouse"]
        assert not state.unverified_sources
    else:
        assert session._llm.code.call_count == 1
        assert state.gathering_notes.startswith("normal fallback")
        if failure in {"refused", "failed-exec"}:
            assert not state.scratchpad_execs


def test_ask_first_and_live_apps_never_offer_automatic_static_completion(atomic, tmp_path):
    for artifact_type, act_first in [("html-app", False), ("fullstack-stateful-app", True)]:
        state = GenState(session=SimpleNamespace(), artifact_type=artifact_type, artifact_path=tmp_path,
                         slug="page", act_first=act_first)
        assert not static_gathering.enabled(state)
