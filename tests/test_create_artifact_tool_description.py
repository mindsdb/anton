"""CREATE_ARTIFACT_TOOL's description drives the agent to call generate_prd
before writing any code for a web artifact — without this text the new tool
is never used (see prd-design.md, "Prompts and wiring into the main flow")."""
from __future__ import annotations

from anton.core.tools.tool_defs import CREATE_ARTIFACT_TOOL


def test_description_points_to_generate_prd_for_web_artifact_types():
    assert "generate_prd" in CREATE_ARTIFACT_TOOL.description


def test_description_still_tells_non_web_types_to_write_files_directly():
    """The new AFTER REGISTERING paragraph (added by Step 3 below — this
    phrase does not exist in the description before this task) must say
    document/dataset/image/mixed have no PRD step and should be written by
    hand; otherwise the agent might assume generate_prd applies to them too."""
    assert "write the files yourself" in CREATE_ARTIFACT_TOOL.description
