"""Where nothing starts an app backend, the agent is offered html-apps only.

Agent code that runs under a sandbox policy, with live apps off, can't start
an app backend. Its agent is offered none: no `launch_backend`, no fullstack
type in `create_artifact` or in the generator's gathering step, no backend
section and no backend URL in the prompt. Without a policy (desktop, cloud),
or with live apps on, the offer is what it was."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from anton.core.artifacts import live_app_slots as las
from anton.core.backends.sandbox import CellSandbox
from anton.core.llm.prompts import ARTIFACTS_PROMPT, BACKEND_GENERATION_PROMPT, STATIC_ARTIFACTS_PROMPT
from anton.core.session import ChatSession, ChatSessionConfig
from anton.core.tools.generate_artifact.state import GenState
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


async def test_a_sandboxed_turn_without_live_apps_offers_no_fullstack_app(tmp_path, policy, live_apps_off):
    system, tools = await _first_request(tmp_path, policy)

    assert "launch_backend" not in tools
    assert tools["create_artifact"]["input_schema"]["properties"]["type"]["enum"] == [
        "html-app", "document", "dataset", "image", "mixed",
    ]
    for name in ("create_artifact", "generate_artifact"):
        assert not any(t in tools[name]["description"] for t in _FULLSTACK_TYPES)
        assert "launch_backend" not in tools[name]["description"]
    assert BACKEND_GENERATION_PROMPT not in system
    assert STATIC_ARTIFACTS_PROMPT in system
    assert "the entry point is the `url`" not in system
    assert "launch_backend" not in system
    assert not any(t in system for t in _FULLSTACK_TYPES)


@pytest.mark.parametrize("sandboxed", [False, True], ids=["no-policy", "policy-and-live-apps"])
async def test_without_a_policy_or_with_live_apps_the_offer_is_unchanged(tmp_path, policy, monkeypatch, sandboxed):
    if sandboxed:
        monkeypatch.setenv(las.SLOTS_ENV_VAR, "http://live-app-1:8001")
    else:
        monkeypatch.delenv(las.SLOTS_ENV_VAR, raising=False)
    system, tools = await _first_request(tmp_path, policy if sandboxed else None)

    assert "launch_backend" in tools
    assert set(_FULLSTACK_TYPES) <= set(tools["create_artifact"]["input_schema"]["properties"]["type"]["enum"])
    assert ARTIFACTS_PROMPT in system
    assert BACKEND_GENERATION_PROMPT in system


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
