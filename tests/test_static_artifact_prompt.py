from types import SimpleNamespace

import pytest

from anton.core.llm import prompts, static_artifact_prompt

from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext
from anton.core.llm.prompts import CONVERSATION_DISCIPLINE_ASK_FIRST


def build(*, has_name=True, proactive=False, act_first=True, **extra):
    tool = SimpleNamespace(name="generate_artifact", prompt=None,
                           input_schema={"properties": {"name": {}} if has_name else {"slug": {}}})
    return ChatSystemPromptBuilder().build(
        conversation_started="2026-10-24", system_prompt_context=SystemPromptContext(),
        proactive_dashboards=proactive, output_dir="/tmp/unused", act_first=act_first,
        tool_defs=[tool], **extra,
    )


def test_delegated_artifacts_have_no_competing_registration_or_data_fetch_instruction():
    prompt = build()
    assert "This one\noperation registers the folder" in prompt
    assert "REGISTER FIRST, always" not in prompt
    assert "DIRECT REPORTS:" not in prompt
    assert "FETCH DATA FIRST" not in prompt
    assert "For inline analysis without an artifact" in prompt
    assert "dataset for CSV/JSON/parquet" in prompt
    assert "Never create a duplicate artifact" in prompt
    assert "tell the user it is missing" in " ".join(prompt.split())


def test_a_generate_tool_without_name_keeps_the_register_first_workflow():
    assert "REGISTER FIRST, always" in build(has_name=False)
    assert "REGISTER FIRST, always" not in build()


def test_proactive_preference_and_ask_first_are_preserved():
    assert "Do not proactively create HTML" in build(proactive=False)
    assert "benefits from a saved visual" in build(proactive=True)
    prompt = build(act_first=False)
    assert "configured ask-first/act-first behavior" in prompt
    assert CONVERSATION_DISCIPLINE_ASK_FIRST in prompt


@pytest.mark.parametrize("text", [
    static_artifact_prompt.ARTIFACTS, static_artifact_prompt.STATIC_ARTIFACTS,
    prompts.ARTIFACTS_PROMPT, prompts.STATIC_ARTIFACTS_PROMPT,
    prompts.NO_DIRECT_REPORTS_ARTIFACTS_PROMPT, prompts.NO_DIRECT_REPORTS_STATIC_ARTIFACTS_PROMPT,
])
def test_a_file_the_user_names_goes_to_the_project_not_an_artifact(text):
    assert "write it into the project folder under that name" in " ".join(text.split())


_SINGLE_CALL_URL = (" For a fullstack app, use the running backend URL, never the standalone\n"
                    "static HTML path, which cannot reach the app's API.")
_REGISTER_FIRST_URL = "the entry point is the `url`"


def test_published_apps_drop_only_the_backend_url_sentence_from_the_single_call_section():
    assert static_artifact_prompt.ARTIFACTS_PUBLISHED_APPS == static_artifact_prompt.ARTIFACTS.replace(_SINGLE_CALL_URL, "")
    prompt = build(app_backends_reachable=False)
    assert "use the running backend URL" not in prompt
    assert "Reference the artifact by name and slug" in prompt


def test_published_apps_drop_the_backend_url_from_the_register_first_sections():
    for name in ("ARTIFACTS_PROMPT", "NO_DIRECT_REPORTS_ARTIFACTS_PROMPT"):
        assert _REGISTER_FIRST_URL in getattr(prompts, name), name
        assert _REGISTER_FIRST_URL not in getattr(prompts, f"{name}_PUBLISHED_APPS"), name
    # Without direct reports the generate tool has no `name`, so the builder takes this section.
    assert _REGISTER_FIRST_URL in build(has_name=False, direct_reports=False)
    assert _REGISTER_FIRST_URL not in build(has_name=False, direct_reports=False, app_backends_reachable=False)


def test_reachable_backends_keep_the_prompt_byte_identical():
    assert build() == build(app_backends_reachable=True)
    assert build(has_name=False) == build(has_name=False, app_backends_reachable=True)
