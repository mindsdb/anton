"""Where nothing starts an app backend, the agent is offered html-apps only.

Agent code that runs under a sandbox policy, with live apps off, can't start
an app backend. Its agent is offered none: no `launch_backend`, no fullstack
type in `create_artifact` or in the generator's gathering step, no backend
section, no backend skill and no backend URL in the prompt. Without a policy
(desktop, cloud), or with live apps on, the offer is what it was."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.artifacts import live_app_slots as las
from anton.core.backends.sandbox import CellSandbox
from anton.core.llm.prompts import ARTIFACTS_PROMPT, BACKEND_GENERATION_PROMPT, STATIC_ARTIFACTS_PROMPT
from anton.core.memory import skills as skills_mod
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.generate_artifact.state import GenState
from anton.core.tools.tool_handlers import handle_create_artifact
from anton.workspace import Workspace
from tests.conftest import make_mock_llm, run_turn

_FULLSTACK_TYPES = ("fullstack-stateless-app", "fullstack-stateful-app")


@pytest.fixture
def policy(tmp_path: Path) -> CellSandbox:
    project = tmp_path / "project"
    for folder in (project, tmp_path / "home", tmp_path / "tmp"):
        folder.mkdir()
    return CellSandbox(project_dir=project, home_dir=tmp_path / "home", tmp_dir=tmp_path / "tmp", user_name="bob")


@pytest.fixture
def live_apps_off(monkeypatch):
    monkeypatch.delenv(las.SLOTS_ENV_VAR, raising=False)


@pytest.fixture
def live_apps_on(monkeypatch):
    monkeypatch.setenv(las.SLOTS_ENV_VAR, "http://live-app-1:8001")


@pytest.fixture
def builtin_skills(monkeypatch):
    """The real built-in skills, which the suite hides (conftest's
    `_no_builtin_skills`): a real turn's prompt lists each one's description,
    and `build-fullstack-backend`'s offers the fullstack types."""
    monkeypatch.setattr(skills_mod, "_BUILTIN_SKILLS_ROOT", Path(skills_mod.__file__).parent / "builtin_skills")


def test_fullstack_is_offered_without_a_policy_or_with_live_apps(policy, monkeypatch):
    monkeypatch.delenv(las.SLOTS_ENV_VAR, raising=False)
    assert las.fullstack_offered(None) is True
    assert las.fullstack_offered(policy) is False
    monkeypatch.setenv(las.SLOTS_ENV_VAR, "http://live-app-1:8001")
    assert las.fullstack_offered(policy) is True


async def _first_request(tmp_path: Path, cell_sandbox: CellSandbox | None) -> tuple[str, dict[str, dict]]:
    """The system prompt and tools, by name, of a turn's first model request."""
    workspace = Workspace(tmp_path / "workspace")
    workspace.initialize()
    llm = make_mock_llm()
    session = ChatSession(ChatSessionConfig(llm_client=llm, workspace=workspace, cell_sandbox=cell_sandbox))
    try:
        await run_turn(session, "Build me an interactive demo with a knowledge graph and a map.")
    finally:
        await session.close()
    kwargs = llm.plan.call_args_list[0].kwargs
    return kwargs["system"], {tool["name"]: tool for tool in kwargs["tools"]}


async def test_a_sandboxed_turn_without_live_apps_offers_no_fullstack_app(
    tmp_path, policy, live_apps_off, builtin_skills,
):
    system, tools = await _first_request(tmp_path, policy)

    assert "launch_backend" not in tools
    assert tools["create_artifact"]["input_schema"]["properties"]["type"]["enum"] == [
        "html-app", "document", "dataset", "image", "mixed",
    ]
    assert "port" not in tools["update_artifact"]["input_schema"]["properties"]
    every_tool = json.dumps(list(tools.values()))
    assert not any(t in every_tool for t in _FULLSTACK_TYPES)
    assert "launch_backend" not in every_tool
    assert BACKEND_GENERATION_PROMPT not in system
    assert STATIC_ARTIFACTS_PROMPT in system
    assert "the entry point is the `url`" not in system
    assert "launch_backend" not in system
    assert "build-fullstack-backend" not in system
    assert not any(t in system for t in _FULLSTACK_TYPES)


@pytest.mark.parametrize("sandboxed", [False, True], ids=["no-policy", "policy-and-live-apps"])
async def test_without_a_policy_or_with_live_apps_the_offer_is_unchanged(
    tmp_path, policy, monkeypatch, builtin_skills, sandboxed,
):
    if sandboxed:
        monkeypatch.setenv(las.SLOTS_ENV_VAR, "http://live-app-1:8001")
    else:
        monkeypatch.delenv(las.SLOTS_ENV_VAR, raising=False)
    system, tools = await _first_request(tmp_path, policy if sandboxed else None)

    assert "launch_backend" in tools
    assert set(_FULLSTACK_TYPES) <= set(tools["create_artifact"]["input_schema"]["properties"]["type"]["enum"])
    assert "port" in tools["update_artifact"]["input_schema"]["properties"]
    assert ARTIFACTS_PROMPT in system
    assert BACKEND_GENERATION_PROMPT in system
    assert "- `build-fullstack-backend` — " in system


def _registering_session(tmp_path: Path, *, fullstack_offered: bool) -> SimpleNamespace:
    return SimpleNamespace(
        _workspace=SimpleNamespace(artifacts_dir=tmp_path), _data_vault=None, _fullstack_offered=fullstack_offered,
    )


async def test_create_artifact_refuses_a_fullstack_type_where_none_is_offered(tmp_path):
    """The schema leaves the fullstack types out, but not every model keeps
    to an enum: the handler refuses one too, before any folder exists."""
    session = _registering_session(tmp_path, fullstack_offered=False)
    outcome = await handle_create_artifact(
        session, {"name": "Demo", "description": "d", "type": "fullstack-stateless-app"},
    )

    assert outcome.ok is False
    assert "register an html-app instead" in outcome.content
    assert not any(tmp_path.iterdir())


async def test_create_artifact_registers_a_fullstack_type_where_one_is_offered(tmp_path):
    session = _registering_session(tmp_path, fullstack_offered=True)
    outcome = await handle_create_artifact(
        session, {"name": "Demo", "description": "d", "type": "fullstack-stateless-app"},
    )

    assert outcome.ok is True


def _gen_state(tmp_path: Path, *, fullstack_offered: bool) -> GenState:
    return GenState(
        session=SimpleNamespace(_fullstack_offered=fullstack_offered), artifact_type="html-app",
        artifact_path=tmp_path, slug="demo", brief="b",
    )


def test_the_gathering_step_offers_html_app_only_where_the_session_offers_no_fullstack(tmp_path):
    state = _gen_state(tmp_path, fullstack_offered=False)
    finish = next(tool for tool in state.pipeline_tools if tool["name"] == "finish_gathering")

    assert state.buildable_types == ("html-app",)
    assert finish["input_schema"]["properties"]["artifact_type"]["enum"] == ["html-app"]
    assert "Valid artifact types: `html-app`\n" in state.pipeline_system
    assert "fullstack" not in state.pipeline_system


def test_the_gathering_step_offers_every_buildable_type_where_the_session_offers_fullstack(tmp_path):
    state = _gen_state(tmp_path, fullstack_offered=True)
    finish = next(tool for tool in state.pipeline_tools if tool["name"] == "finish_gathering")

    assert finish["input_schema"]["properties"]["artifact_type"]["enum"] == ["html-app", *_FULLSTACK_TYPES]
    assert "Choosing between the two fullstack types" in state.pipeline_system
