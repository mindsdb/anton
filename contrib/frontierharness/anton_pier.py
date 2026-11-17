"""Anton as a Pier installed agent: `pier run --agent-import-path anton_pier:AntonPier`."""

from __future__ import annotations

from pier.agents.installed.base import BaseInstalledAgent, with_prompt_template
from pier.agents.network import allowlist_from_urls
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist
from pier.models.trial.paths import EnvironmentPaths

import anton_common

_ROOT_DEPS = (
    "if command -v apt-get >/dev/null 2>&1; then apt-get update && apt-get install -y curl ca-certificates;"
    " elif command -v apk >/dev/null 2>&1; then apk add --no-cache curl bash ca-certificates;"
    " elif command -v dnf >/dev/null 2>&1; then dnf install -y curl;"
    " elif command -v yum >/dev/null 2>&1; then yum install -y curl;"
    " fi"
)


class AntonPier(BaseInstalledAgent):
    @staticmethod
    def name() -> str:
        return "anton"

    def install_spec(self) -> AgentInstallSpec:
        ref = anton_common.anton_ref()
        return AgentInstallSpec(
            agent_name=self.name(),
            version=ref,
            steps=[
                InstallStep(user="root", env={"DEBIAN_FRONTEND": "noninteractive"}, run=_ROOT_DEPS),
                InstallStep(user="agent", run=anton_common.install_script(ref)),
            ],
        )

    def network_allowlist(self) -> NetworkAllowlist:
        # DeepSWE agents run with no network; the model gateway is the one exception.
        return allowlist_from_urls([anton_common.base_url()])

    def version(self) -> str | None:
        return anton_common.anton_ref()

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await self.exec_as_agent(
            environment,
            command=anton_common.run_script(instruction, str(EnvironmentPaths.agent_dir)),
            env=self.build_process_env(anton_common.anton_env(self._get_env("ANTON_FH_API_KEY"))),
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        anton_common.fill_context(context, self.logs_dir)
