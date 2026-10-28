"""The actual parent session must reach the replacement, including edits."""
import json

import pytest

from anton.core.artifacts import live_app_slots
from anton.core.backends.sandbox import CellSandbox
from anton.core.llm.prompt_builder import ChatSystemPromptBuilder, SystemPromptContext
from tests.test_fullstack_offered import _first_request


@pytest.mark.parametrize('fullstack',[False,True])
async def test_v2_session_has_one_static_authoring_route(tmp_path,monkeypatch,fullstack):
    monkeypatch.setenv('ANTON_STATIC_ARTIFACTS','v2')
    monkeypatch.delenv(live_app_slots.SLOTS_ENV_VAR,raising=False)
    policy=None
    if not fullstack:
        for name in ['project','home','tmp']:(tmp_path/name).mkdir()
        policy=CellSandbox(project_dir=tmp_path/'project',home_dir=tmp_path/'home',tmp_dir=tmp_path/'tmp',user_name='test')
    system,tools=await _first_request(tmp_path,policy)
    assert 'STATIC WEB ARTIFACTS:' in system
    assert 'DIRECT REPORTS' not in system
    assert 'report_tools' not in system
    assert 'current change request' in system
    assert 'source while building a report also needs no backend' in system
    assert 'Do not write the page yourself' in tools['generate_artifact']['description']
    assert 'fully specified you build yourself' not in json.dumps(tools)
    assert ('launch_backend' in tools)==fullstack
    offered=tools['create_artifact']['input_schema']['properties']['type']['enum']
    assert ('fullstack-stateless-app' in offered)==fullstack
    if not fullstack:
        assert 'fullstack-stateless-app' not in system


@pytest.mark.parametrize('proactive',[False,True])
def test_v2_visualization_guidance_does_not_restore_the_old_route(monkeypatch,proactive):
    monkeypatch.setenv('ANTON_STATIC_ARTIFACTS','v2')
    prompt=ChatSystemPromptBuilder().build(conversation_started='2026-10-01',system_prompt_context=SystemPromptContext(),
        proactive_dashboards=proactive,output_dir='',act_first=True)
    assert 'DIRECT REPORTS' not in prompt and 'report_tools' not in prompt
    assert 'Use the static web artifact workflow' in prompt


def test_confirmation_first_keeps_its_existing_workflow(monkeypatch):
    monkeypatch.setenv('ANTON_STATIC_ARTIFACTS','v2')
    prompt=ChatSystemPromptBuilder().build(conversation_started='2026-10-01',system_prompt_context=SystemPromptContext(),
        proactive_dashboards=False,output_dir='',act_first=False)
    assert 'DIRECT REPORTS' in prompt
    assert 'STATIC WEB ARTIFACTS:' not in prompt
