"""sub_tools: tool schemas for generate_prd's phase 1 gathering loop, and the
direct-elicit() path for ask_user (bypassing handle_ask_user so `limit` and
`unavailable` stay distinguishable from a real user decision)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from anton.core.interaction.elicit import AskAnswer, AskOption, AskRequest
from anton.core.tools.generate_artifact.discovery import sub_tools


def test_the_pipeline_array_always_carries_all_five_tools():
    """There is no conditional array any more. Availability per step is
    decided by `rejection_for`, because the array is the cached prefix for
    every call in phases A-D and cannot be edited mid-run."""
    names = {t["name"] for t in sub_tools.pipeline_tool_schemas()}
    assert names == {
        "scratchpad", "web_search", "web_fetch", "finish_gathering", "ask_user",
    }


def test_finish_gathering_schema_requires_summary_and_artifact_type():
    schema = next(
        t for t in sub_tools.pipeline_tool_schemas()
        if t["name"] == "finish_gathering"
    )
    assert set(schema["input_schema"]["required"]) == {"summary", "artifact_type"}


def test_finish_gathering_schema_constrains_artifact_type_to_the_closed_enum():
    """Without this, the model is free to invent a type string that later
    crashes write_prd's `ArtifactStore.update(type=...)` call — see
    store.py's ValueError on anything outside ARTIFACT_TYPES."""
    from anton.core.artifacts.models import ARTIFACT_TYPES

    schema = next(
        t for t in sub_tools.pipeline_tool_schemas()
        if t["name"] == "finish_gathering"
    )
    assert schema["input_schema"]["properties"]["artifact_type"]["enum"] == list(ARTIFACT_TYPES)


class _FakeElicitor:
    supported_kinds = ("choice",)
    answer_hint = "hint"
    timeout_s = None


def _tc_input(**over) -> dict:
    base = {
        "question": "Which theme?",
        "options": [{"value": "light", "label": "Light"}, {"value": "dark", "label": "Dark"}],
    }
    base.update(over)
    return base


async def test_ask_via_elicit_returns_error_status_instead_of_raising(monkeypatch):
    async def _raising_elicit(session, question_id, request):
        raise RuntimeError("boom")

    monkeypatch.setattr("anton.core.interaction.elicit.elicit", _raising_elicit)
    answer = await sub_tools.ask_via_elicit(SimpleNamespace(), AskRequest(prompt="x", options=(AskOption(value="a", label="a"), AskOption(value="b", label="b"))))
    assert answer.status == "error"


async def test_ask_via_elicit_passes_through_a_real_answer(monkeypatch):
    async def _fake_elicit(session, question_id, request):
        return AskAnswer(status="answered", values=("a",))

    monkeypatch.setattr("anton.core.interaction.elicit.elicit", _fake_elicit)
    answer = await sub_tools.ask_via_elicit(SimpleNamespace(), AskRequest(prompt="x", options=(AskOption(value="a", label="a"), AskOption(value="b", label="b"))))
    assert answer == AskAnswer(status="answered", values=("a",))


async def test_dispatch_ask_user_keeps_limit_distinct_from_unavailable(monkeypatch):
    async def _fake_elicit(session, question_id, request):
        return AskAnswer(status="limit")

    monkeypatch.setattr("anton.core.interaction.elicit.elicit", _fake_elicit)
    out = await sub_tools.dispatch_ask_user(SimpleNamespace(elicitor=_FakeElicitor()), _tc_input())
    assert out["status"] == "limit"
    assert "assumption" in out["tool_result"]


async def test_dispatch_ask_user_returns_answered_summary(monkeypatch):
    async def _fake_elicit(session, question_id, request):
        return AskAnswer(status="answered", values=("dark",))

    monkeypatch.setattr("anton.core.interaction.elicit.elicit", _fake_elicit)
    out = await sub_tools.dispatch_ask_user(SimpleNamespace(elicitor=_FakeElicitor()), _tc_input())
    assert out["status"] == "answered"
    assert out["answer_summary"] == "dark"
    assert out["question"] == "Which theme?"


async def test_dispatch_ask_user_rejects_malformed_input_without_calling_elicit(monkeypatch):
    called = False

    async def _fake_elicit(session, question_id, request):
        nonlocal called
        called = True
        return AskAnswer(status="answered")

    monkeypatch.setattr("anton.core.interaction.elicit.elicit", _fake_elicit)
    out = await sub_tools.dispatch_ask_user(SimpleNamespace(elicitor=_FakeElicitor()), {"question": "q"})
    assert out["status"] == "unavailable"
    assert called is False
