from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import Mock

from minds.agents.agent_controller import AgentController


def test_agent_controller_get_agent_builds_config_and_instantiates():
    # The repo can contain partially implemented agent directories during refactors.
    # Avoid importing/discovering all agents here; only validate `get_agent` behavior.
    controller = AgentController.__new__(AgentController)
    controller.agents = {}

    @dataclass
    class DummyRunContext:
        instrument: bool = True
        metadata: object | None = None

    class DummyAgent:
        @classmethod
        def build_config(cls, run_context):
            return {"instrument": getattr(run_context, "instrument", None)}

        def __init__(self, mind, mindsdb_client, config=None):
            self.mind = mind
            self.mindsdb_client = mindsdb_client
            self.config = config

    controller.agents = {"dummy_agent": DummyAgent}

    mind = Mock()
    client = Mock()
    run_context = DummyRunContext(instrument=False)

    agent = controller.get_agent("dummy_agent", mind=mind, mindsdb_client=client, run_context=run_context)

    assert agent.mind is mind
    assert agent.mindsdb_client is client
    assert agent.config == {"instrument": False}
