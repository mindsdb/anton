"""CREATE_ARTIFACT_TOOL's description drives the agent into
`generate_artifact` for a web artifact — without this text the pipeline is
never entered and the agent writes the files by hand."""
from __future__ import annotations

from anton.core.tools.tool_defs import CREATE_ARTIFACT_TOOL


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
