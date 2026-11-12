"""Every ARTIFACTS prompt variant tells the agent how to handle an artifact
that disappeared from the workspace."""
import pytest

from anton.core.llm import prompts

RULE_KEY = "tell the user it is missing"


def _flat(text: str) -> str:
    """Collapse line breaks and runs of spaces, so a wrapped phrase still matches."""
    return " ".join(text.split())


@pytest.mark.parametrize(
    "name",
    [
        "ARTIFACTS_PROMPT",
        "STATIC_ARTIFACTS_PROMPT",
        "NO_DIRECT_REPORTS_ARTIFACTS_PROMPT",
        "NO_DIRECT_REPORTS_STATIC_ARTIFACTS_PROMPT",
    ],
)
def test_artifacts_prompt_variant_has_missing_artifact_rule(name):
    text = _flat(getattr(prompts, name))
    assert RULE_KEY in text
    assert "ask whether to create it again as a new artifact" in text
    assert "end your turn" in text


@pytest.mark.parametrize("name", ["ARTIFACTS", "STATIC_ARTIFACTS"])
def test_single_call_prompt_variant_has_missing_artifact_rule(name):
    from anton.core.llm import static_artifact_prompt

    text = _flat(getattr(static_artifact_prompt, name))
    assert RULE_KEY in text
    assert "ask whether to create it again as a new artifact" in text
    assert "end your turn" in text
