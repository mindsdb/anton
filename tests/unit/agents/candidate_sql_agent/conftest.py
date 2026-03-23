import sys
from types import ModuleType
from unittest.mock import Mock

import pytest


def _ensure_module(name: str) -> ModuleType:
    mod = sys.modules.get(name)
    if isinstance(mod, ModuleType):
        return mod
    mod = ModuleType(name)
    sys.modules[name] = mod
    return mod


# Some environments running unit tests may not have optional deps installed.
# Stub them so that importing agent modules is possible without installing extras.

# - langfuse is used for instrumentation in some code paths
langfuse = _ensure_module("langfuse")
if not hasattr(langfuse, "observe"):
    langfuse.observe = lambda f=None, **_: (lambda *a, **k: f(*a, **k)) if f else (lambda x: x)
if not hasattr(langfuse, "get_client"):
    langfuse.get_client = lambda *args, **kwargs: Mock()

# - mind_castle is used by SQLAlchemy type helpers in models
mc = _ensure_module("mind_castle")
mc_sqlalchemy_type = _ensure_module("mind_castle.sqlalchemy_type")
if not hasattr(mc_sqlalchemy_type, "SecretData"):
    from sqlalchemy.types import String, TypeDecorator

    class SecretData(TypeDecorator):  # noqa: N801 (keep name to match expected symbol)
        """Minimal stand-in for mind_castle SecretData SQLAlchemy type."""

        impl = String
        cache_ok = True

        def __init__(self, *args, **kwargs):
            super().__init__()

        def process_bind_param(self, value, dialect):
            return value

        def process_result_value(self, value, dialect):
            return value

    mc_sqlalchemy_type.SecretData = SecretData


@pytest.fixture(autouse=True)
def _patch_helpers_for_candidate_sql_agent_tests(monkeypatch):
    """Patch `minds.agents.helpers` to avoid heavy config/model stack.

    We patch the real module instead of replacing it in `sys.modules` so we don't
    affect unrelated unit tests that import `minds.agents.helpers`.
    """
    import minds.agents.helpers as helpers

    fixed_now = "The current date and time is 2000-01-01 00:00:00"

    def fixed_model(_mind):
        return Mock()

    def fixed_mind_layer(_mind):
        return ""

    def fixed_chart_layer():
        return ""

    def fixed_native_enabled(_mind, _settings):
        return False

    monkeypatch.setattr(helpers, "current_date_time_layer", lambda: "The current date and time is 2000-01-01 00:00:00")
    monkeypatch.setattr(helpers, "mind_layer", fixed_mind_layer)
    monkeypatch.setattr(helpers, "charting_layer", fixed_chart_layer)
    monkeypatch.setattr(helpers, "model_for", fixed_model)
    monkeypatch.setattr(helpers, "is_native_query_mode_enabled", fixed_native_enabled)

    # Patch modules that imported helpers symbols directly.
    import minds.agents.candidate_sql_agent.agent as candidate_sql_agent
    import minds.agents.candidate_sql_agent.controller_agents.agents as controller_agents
    import minds.agents.candidate_sql_agent.linker_agent.agent as linker_agent
    import minds.agents.candidate_sql_agent.selection_agent.agent as selection_agent
    import minds.agents.candidate_sql_agent.text_to_sql_agents.agents as t2s_agents

    monkeypatch.setattr(controller_agents, "current_date_time_layer", lambda: fixed_now)
    monkeypatch.setattr(controller_agents, "mind_layer", fixed_mind_layer)

    monkeypatch.setattr(linker_agent, "current_date_time_layer", lambda: fixed_now)
    monkeypatch.setattr(linker_agent, "mind_layer", fixed_mind_layer)
    monkeypatch.setattr(linker_agent, "model_for", fixed_model)

    monkeypatch.setattr(selection_agent, "model_for", fixed_model)

    monkeypatch.setattr(t2s_agents, "current_date_time_layer", lambda: fixed_now)
    monkeypatch.setattr(t2s_agents, "mind_layer", fixed_mind_layer)
    monkeypatch.setattr(t2s_agents, "charting_layer", fixed_chart_layer)
    monkeypatch.setattr(t2s_agents, "model_for", fixed_model)

    monkeypatch.setattr(candidate_sql_agent, "is_native_query_mode_enabled", fixed_native_enabled)
    monkeypatch.setattr(candidate_sql_agent, "model_for", fixed_model)
