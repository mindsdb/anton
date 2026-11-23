"""Anton as a Harbor installed agent: `harbor run --agent anton_harbor:AntonHarbor`."""

from __future__ import annotations

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

import anton_common


class AntonHarbor(BaseInstalledAgent):
    @staticmethod
    def name() -> str:
        return "anton"

    def version(self) -> str | None:
        return anton_common.anton_ref(self._get_env)

    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("curl",))
        await self.exec_as_agent(
            environment, command=anton_common.install_script(anton_common.anton_ref(self._get_env))
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await self.exec_as_agent(
            environment,
            command=anton_common.run_script(instruction, str(self.environment_logs_dir)),
            env=anton_common.anton_env(self._get_env, self.model_name),
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        anton_common.fill_context(context, self.logs_dir)
