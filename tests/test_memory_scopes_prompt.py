"""The base system prompt describes memory scopes without storage paths.

The same prompt serves the CLI, the desktop, self-hosted and org-mode turns,
and each of them keeps memory in different folders, so no path is true for all.
"""

import pytest

from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext
from anton.core.llm.prompts import CHAT_SYSTEM_PROMPT, NO_INTERNET_CHAT_SYSTEM_PROMPT


def _build(*, internet_research: bool) -> str:
    return ChatSystemPromptBuilder().build(
        conversation_started="2026-11-06",
        system_prompt_context=SystemPromptContext(),
        proactive_dashboards=False,
        output_dir="/tmp/artifacts",
        internet_research=internet_research,
    )


@pytest.mark.parametrize(
    "template", [CHAT_SYSTEM_PROMPT, NO_INTERNET_CHAT_SYSTEM_PROMPT], ids=["internet", "no_internet"]
)
def test_templates_name_no_memory_paths(template):
    assert ".anton/memory" not in template
    assert "Your memory is local to this workspace" not in template


@pytest.mark.parametrize("internet_research", [True, False])
def test_assembled_prompt_describes_both_scopes(internet_research):
    system = _build(internet_research=internet_research)

    assert ".anton/memory" not in system
    assert (
        "Memories persist across sessions at two scopes: global (applies across all "
        "projects) and project (applies only to the current project)."
    ) in system
