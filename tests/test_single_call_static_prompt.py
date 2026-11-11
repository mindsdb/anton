"""The single-call artifact instructions offer no fullstack app where nothing starts one.

With `ANTON_STATIC_SINGLE_CALL=on`, the chat prompt swaps in the single-call
artifact instructions. Where live apps are off (`fullstack_offered` False), those
instructions must name no fullstack type, no `launch_backend` and no backend URL,
as the static tool definitions already do."""
from types import SimpleNamespace

import pytest

from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext
from anton.core.llm.static_artifact_prompt import ARTIFACTS, STATIC_ARTIFACTS

_FULLSTACK_TEXT = ("fullstack-stateless-app", "fullstack-stateful-app", "launch_backend", "running backend URL")


def build(*, fullstack_offered):
    tool = SimpleNamespace(name="generate_artifact", prompt=None, input_schema={"properties": {"name": {}}})
    return ChatSystemPromptBuilder().build(
        conversation_started="2026-11-10", system_prompt_context=SystemPromptContext(),
        proactive_dashboards=False, output_dir="/tmp/unused", act_first=True,
        tool_defs=[tool], fullstack_offered=fullstack_offered,
    )


@pytest.fixture
def single_call(monkeypatch):
    monkeypatch.setenv("ANTON_STATIC_SINGLE_CALL", "on")


def test_without_live_apps_the_single_call_prompt_offers_no_fullstack_app(single_call):
    prompt = build(fullstack_offered=False)
    assert STATIC_ARTIFACTS in prompt
    assert "This one\noperation registers the folder" in prompt
    assert "Write non-web artifacts directly in the registered folder." in prompt
    assert not [text for text in _FULLSTACK_TEXT if text in prompt]


def test_with_live_apps_the_single_call_prompt_is_unchanged(single_call):
    prompt = build(fullstack_offered=True)
    assert ARTIFACTS in prompt
    assert "fullstack-stateless-app" in prompt
