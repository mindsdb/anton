"""Agent-facing artifact texts: CREATE_ARTIFACT_TOOL's description drives the
agent into `generate_artifact` for a web artifact, and none of the artifact
texts sends the agent to a project-root `artifacts/` folder that does not
exist."""
from __future__ import annotations

import pytest

from anton.core.llm.prompts import ARTIFACTS_PROMPT
from anton.core.tools.tool_defs import CREATE_ARTIFACT_TOOL
from anton.tools import PUBLISH_TOOL


def test_description_points_web_artifact_types_at_the_generator():
    """One tool now. A description still naming a PRD step would send the
    agent to call something that does not exist."""
    assert "generate_artifact" in CREATE_ARTIFACT_TOOL.description
    assert "generate_prd" not in CREATE_ARTIFACT_TOOL.description


def test_description_still_tells_non_web_types_to_write_files_directly():
    """The AFTER REGISTERING paragraph must say document/dataset/image/mixed
    have no generator and should be written by hand; otherwise the agent
    might assume `generate_artifact` applies to them too."""
    assert "write the files yourself" in CREATE_ARTIFACT_TOOL.description


_PUBLISH_FILE_PATH = PUBLISH_TOOL.input_schema["properties"]["file_path"]["description"]


@pytest.mark.parametrize(
    "text",
    [_PUBLISH_FILE_PATH, ARTIFACTS_PROMPT, CREATE_ARTIFACT_TOOL.description],
    ids=["publish-file-path", "artifacts-prompt", "create-artifact"],
)
def test_artifact_texts_do_not_place_artifacts_at_the_project_root(text):
    """Artifacts live under `.anton/artifacts/`. A text naming a project-root
    `artifacts/` folder sends the agent to a path that does not exist."""
    for stale in ("artifacts/<artifact-id>", "e.g. artifacts/<slug>", "<workspace>/artifacts/"):
        assert stale not in text


def test_publish_file_path_says_what_to_pass():
    assert "create_artifact" in _PUBLISH_FILE_PATH
    assert "slug" in _PUBLISH_FILE_PATH
