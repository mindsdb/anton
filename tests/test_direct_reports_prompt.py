"""The DIRECT REPORTS guidance stays consistent with the code it documents."""
import inspect
import re

from anton.core.artifacts import report_tools
from anton.core.llm.prompts import ARTIFACTS_PROMPT
from anton.core.tools.tool_defs import CREATE_ARTIFACT_TOOL, GENERATE_ARTIFACT_TOOL


def _direct_reports_block() -> str:
    return ARTIFACTS_PROMPT.split("DIRECT REPORTS:", 1)[1].split("WORKFLOW:", 1)[0]


def test_every_documented_helper_exists_with_the_documented_parameters():
    block = _direct_reports_block()
    assert "from anton.core.artifacts import report_tools as rt" in block
    documented = dict(re.findall(r"`rt\.(\w+)\(([^`]*)\)`", block))
    assert documented
    for name, args in documented.items():
        assert name in report_tools.__all__, name
        params = inspect.signature(getattr(report_tools, name)).parameters
        for arg in re.findall(r"(\w+)=", args):
            assert arg in params, f"rt.{name} has no parameter {arg}"


def test_routing_is_stated_once_with_the_criterion_first():
    """Each routing text names the work it is for; no 'Exception:' afterthoughts."""
    for text in (ARTIFACTS_PROMPT, CREATE_ARTIFACT_TOOL.description, GENERATE_ARTIFACT_TOOL.description,
                 GENERATE_ARTIFACT_TOOL.prompt):
        assert "fully specified" in text
        assert "Exception:" not in text


def test_the_block_is_short():
    assert len(_direct_reports_block()) < 2600
