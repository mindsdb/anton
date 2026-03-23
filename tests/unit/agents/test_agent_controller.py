from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from minds.agents.agent_controller import AgentController


def test_agent_controller_get_agent_instantiates_agent():
    # The repo can contain partially implemented agent directories during refactors.
    # Avoid importing/discovering all agents here; only validate `get_agent` behavior.
    controller = AgentController.__new__(AgentController)
    controller.agents = {}

    class DummyAgent:
        def __init__(self, mind, mindsdb_client):
            self.mind = mind
            self.mindsdb_client = mindsdb_client

    controller.agents = {"dummy_agent": DummyAgent}

    mind = Mock()
    client = Mock()

    agent = controller.get_agent("dummy_agent", mind=mind, mindsdb_client=client)

    assert agent.mind is mind
    assert agent.mindsdb_client is client


def _make_agent_module(name: str, *, class_names: list[str]) -> tuple[ModuleType, list[type]]:
    from minds.agents.base import BaseAgent

    mod = ModuleType(name)
    created: list[type] = []
    for cls_name in class_names:

        async def _run(*_a, **_k):  # noqa: ANN001
            return None

        async def _usage(*_a, **_k):  # noqa: ANN001
            return None

        cls = type(
            cls_name,
            (BaseAgent,),
            {
                "__module__": mod.__name__,
                "_run": _run,
                "get_last_run_usage": _usage,
            },
        )
        created.append(cls)
        setattr(mod, cls_name, cls)
    return mod, created


def test_agent_controller_find_agent_class_single_concrete():
    controller = AgentController.__new__(AgentController)
    mod, created = _make_agent_module("minds.agents.fake_agent.agent", class_names=["FakeAgent"])

    cls = controller._find_agent_class(mod)
    assert cls is created[0]


def test_agent_controller_find_agent_class_raises_on_multiple_candidates():
    controller = AgentController.__new__(AgentController)
    mod, _created = _make_agent_module("minds.agents.fake_agent.agent", class_names=["A1", "A2"])

    with pytest.raises(ValueError, match="Expected exactly 1 BaseAgent type"):
        controller._find_agent_class(mod)


def test_agent_controller_find_agent_class_ignores_imported_and_base_agent():
    from minds.agents.base import BaseAgent

    controller = AgentController.__new__(AgentController)
    mod, created = _make_agent_module("minds.agents.fake_agent.agent", class_names=["RealAgent"])

    imported_mod, imported_created = _make_agent_module("minds.agents.other.agent", class_names=["ImportedAgent"])
    mod.ImportedAgent = imported_created[0]  # wrong __module__ -> should be ignored
    mod.BaseAgent = BaseAgent  # cls is BaseAgent -> should be ignored

    cls = controller._find_agent_class(mod)
    assert cls is created[0]


def test_agent_controller_find_agent_class_skips_literal_base_agent_branch():
    # This specifically covers the `if cls is BaseAgent: continue` branch.
    from minds.agents.base import BaseAgent

    controller = AgentController.__new__(AgentController)

    mod = ModuleType(BaseAgent.__module__)

    async def _run(*_a, **_k):
        return None

    async def _usage(*_a, **_k):
        return None

    RealAgent = type(
        "RealAgent",
        (BaseAgent,),
        {"__module__": mod.__name__, "_run": _run, "get_last_run_usage": _usage},
    )
    mod.RealAgent = RealAgent
    mod.BaseAgent = BaseAgent

    cls = controller._find_agent_class(mod)
    assert cls is RealAgent


def test_agent_controller_discover_agents_filters_and_registers(monkeypatch):
    # Patch the filesystem scan to deterministic fake directories
    fake_dirs = [
        SimpleNamespace(name="_private_agent", is_dir=lambda: True),
        SimpleNamespace(name="not_an_agent_dir", is_dir=lambda: True),
        SimpleNamespace(name="cool_agent", is_dir=lambda: True),
        SimpleNamespace(name="a_file.txt", is_dir=lambda: False),
    ]
    monkeypatch.setattr("minds.agents.agent_controller.Path.iterdir", lambda _self: iter(fake_dirs))

    fake_mod, created = _make_agent_module("minds.agents.cool_agent.agent", class_names=["CoolAgent"])
    monkeypatch.setattr(
        "minds.agents.agent_controller.importlib.import_module",
        lambda name: fake_mod if name == "minds.agents.cool_agent.agent" else ModuleType(name),
    )

    controller = AgentController()
    assert "cool_agent" in controller.agents
    assert controller.agents["cool_agent"] is created[0]
    assert "_private_agent" not in controller.agents
    assert "not_an_agent_dir" not in controller.agents


def test_agent_controller_discover_agents_raises_when_no_agent_class(monkeypatch):
    fake_dirs = [SimpleNamespace(name="empty_agent", is_dir=lambda: True)]
    monkeypatch.setattr("minds.agents.agent_controller.Path.iterdir", lambda _self: iter(fake_dirs))
    monkeypatch.setattr(importlib, "import_module", lambda _name: ModuleType("minds.agents.empty_agent.agent"))

    # Force the None-branch in _discover_agents.
    monkeypatch.setattr(AgentController, "_find_agent_class", lambda _self, _mod: None)

    with pytest.raises(ValueError, match="No agent class found"):
        AgentController()


def test_agent_controller_get_agent_unknown_key_raises_key_error():
    controller = AgentController.__new__(AgentController)
    controller.agents = {}
    with pytest.raises(KeyError):
        controller.get_agent("missing", mind=Mock(), mindsdb_client=Mock())
