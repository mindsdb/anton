import importlib
import inspect
from pathlib import Path
from types import ModuleType

from mindsdb_sdk.server import Server

from minds.agents.base import BaseAgent
from minds.common.logger import get_logger
from minds.model.mind import Mind
from minds.requests.context import Context

logger = get_logger(__name__)


class AgentController:
    """
    Controller for the various implementation of agents.
    """

    def __init__(self):
        self.agents = {}
        self._discover_agents()

    def get_agent(
        self,
        agent_name: str,
        mind: Mind,
        mindsdb_client: Server,
        context: Context | None = None,
    ) -> BaseAgent:
        """
        Get the agent instance for the given agent name.

        Args:
            agent_name: The name of the agent.
            mind: The mind to use for the agent.
            mindsdb_client: The MindsDB client to use for the agent.
            context: The request context.

        Returns:
            The agent instance.
        """
        logger.debug(f"Getting agent: {agent_name}")
        agent_class = self.agents[agent_name]
        logger.debug(f"Found agent class: {agent_class}")

        return agent_class(
            mind=mind,
            mindsdb_client=mindsdb_client,
            context=context,
        )

    def _discover_agents(self):
        """
        Discover the agents in the `agents` directory.
        """
        agents_dir = Path(__file__).parent
        for agent_dir in agents_dir.iterdir():
            if agent_dir.is_dir():
                agent_name = agent_dir.name
                logger.debug(f"Discovered agent: {agent_name}")

                if agent_name.startswith("_"):
                    continue
                if not agent_dir.name.endswith("_agent"):
                    continue
                # passthrough_agent is not a standard BaseAgent — skip it
                if agent_name == "passthrough_agent":
                    continue

                agent_module = importlib.import_module(f"minds.agents.{agent_name}.agent")
                agent_class = self._find_agent_class(agent_module)
                logger.debug(f"Found agent class: {agent_class}")

                if agent_class is None:
                    logger.warning(f"No agent class found in {agent_module.__name__}")
                    raise ValueError(f"No agent class found in {agent_module.__name__}")
                self.agents[agent_name] = agent_class

    def _find_agent_class(self, mod: ModuleType) -> type[BaseAgent]:
        """
        Find the agent class in the given module.

        Args:
            mod: The module to find the agent class in.

        Returns:
            The agent type.
        """
        candidates: list[type[BaseAgent]] = []
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ != mod.__name__:
                continue
            if cls is BaseAgent:
                continue
            if issubclass(cls, BaseAgent) and not inspect.isabstract(cls):
                candidates.append(cls)

        if len(candidates) != 1:
            raise ValueError(f"Expected exactly 1 BaseAgent type in {mod.__name__}, found {candidates}")
        return candidates[0]
