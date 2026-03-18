"""Conftest for terminal_bench tests — mocks Harbor imports.

Harbor is an optional dependency (pip install harbor). These mocks allow
the agent unit tests to run without Harbor installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest


def _install_harbor_mocks() -> None:
    """Install fake harbor modules into sys.modules so agent.py can import."""
    if "harbor" in sys.modules:
        return  # Harbor is actually installed — don't mock

    # Create mock module hierarchy
    harbor = ModuleType("harbor")
    harbor_agents = ModuleType("harbor.agents")
    harbor_agents_base = ModuleType("harbor.agents.base")
    harbor_environments = ModuleType("harbor.environments")
    harbor_environments_base = ModuleType("harbor.environments.base")
    harbor_models = ModuleType("harbor.models")
    harbor_models_agent = ModuleType("harbor.models.agent")
    harbor_models_agent_context = ModuleType("harbor.models.agent.context")
    harbor_models_task = ModuleType("harbor.models.task")
    harbor_models_task_config = ModuleType("harbor.models.task.config")
    harbor_models_trial = ModuleType("harbor.models.trial")
    harbor_models_trial_result = ModuleType("harbor.models.trial.result")
    harbor_utils = ModuleType("harbor.utils")
    harbor_utils_logger = ModuleType("harbor.utils.logger")

    # BaseAgent — an ABC-like class the agent can subclass
    class _MockBaseAgent:
        SUPPORTS_ATIF = False

        def __init__(self, logs_dir, model_name=None, logger=None, mcp_servers=None, *args, **kwargs):
            import logging
            self.logs_dir = logs_dir
            self.model_name = model_name
            self.logger = logger or logging.getLogger("harbor.mock")
            self.mcp_servers = mcp_servers or []
            self._parsed_model_provider = None
            self._parsed_model_name = None
            if model_name and "/" in model_name:
                self._parsed_model_provider, self._parsed_model_name = model_name.split("/", 1)
            elif model_name:
                self._parsed_model_name = model_name

    harbor_agents_base.BaseAgent = _MockBaseAgent

    # ExecResult
    class _MockExecResult:
        def __init__(self, stdout=None, stderr=None, return_code=0):
            self.stdout = stdout
            self.stderr = stderr
            self.return_code = return_code

    harbor_environments_base.BaseEnvironment = object  # Just needs to exist for type hints
    harbor_environments_base.ExecResult = _MockExecResult

    # AgentContext
    class _MockAgentContext:
        def __init__(self):
            self.n_input_tokens = None
            self.n_cache_tokens = None
            self.n_output_tokens = None
            self.cost_usd = None
            self.rollout_details = None
            self.metadata = None

    harbor_models_agent_context.AgentContext = _MockAgentContext

    # MCPServerConfig
    harbor_models_task_config.MCPServerConfig = MagicMock

    # AgentInfo / ModelInfo (used by BaseAgent.to_agent_info)
    harbor_models_trial_result.AgentInfo = MagicMock
    harbor_models_trial_result.ModelInfo = MagicMock

    # Logger
    harbor_utils_logger.logger = MagicMock()

    # Wire up the module tree
    sys.modules["harbor"] = harbor
    sys.modules["harbor.agents"] = harbor_agents
    sys.modules["harbor.agents.base"] = harbor_agents_base
    sys.modules["harbor.environments"] = harbor_environments
    sys.modules["harbor.environments.base"] = harbor_environments_base
    sys.modules["harbor.models"] = harbor_models
    sys.modules["harbor.models.agent"] = harbor_models_agent
    sys.modules["harbor.models.agent.context"] = harbor_models_agent_context
    sys.modules["harbor.models.task"] = harbor_models_task
    sys.modules["harbor.models.task.config"] = harbor_models_task_config
    sys.modules["harbor.models.trial"] = harbor_models_trial
    sys.modules["harbor.models.trial.result"] = harbor_models_trial_result
    sys.modules["harbor.utils"] = harbor_utils
    sys.modules["harbor.utils.logger"] = harbor_utils_logger


# Install mocks before any test collection happens
_install_harbor_mocks()
