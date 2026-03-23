"""Anton — public API class that orchestrates a chat session.

Thin orchestrator that wires LLM providers, memory, and the ChatSession together.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from minds.common.settings.app_settings import get_app_settings
from minds.services.memory import MemoryService

from .chat_session import ChatSession
from .llm.anthropic import AnthropicProvider
from .llm.client import LLMClient
from .llm.openai import OpenAIProvider
from .llm.provider import StreamEvent
from .memory.cortex import Cortex
from .memory.episodes import EpisodicMemory

app_settings = get_app_settings()


class Anton:
    def __init__(
        self,
        workspace_dir: str,
        runtime_context: str,
        history: list[dict],
        backend: str,
        planning_provider: str,
        planning_model: str,
        planning_api_key: str,
        coding_provider: str,
        coding_model: str,
        coding_api_key: str,
        extra_env: dict[str, str] | None = None,
        shared_memory: MemoryService | None = None,
    ):
        workspace = Path(workspace_dir)
        workspace.mkdir(parents=True, exist_ok=True)

        # 1. Build LLM providers
        if planning_provider == "openai":
            planning_provider = OpenAIProvider(api_key=planning_api_key)
        else:
            planning_provider = AnthropicProvider(api_key=planning_api_key)

        if coding_provider == "openai":
            coding_provider = OpenAIProvider(api_key=coding_api_key)
        else:
            coding_provider = AnthropicProvider(api_key=coding_api_key)

        # 2. Create LLMClient
        llm_client = LLMClient(
            planning_provider=planning_provider,
            planning_model=planning_model,
            coding_provider=coding_provider,
            coding_model=coding_model,
        )

        # DISCLAIMER: These directories are created in the file system of the Minds server.
        # Anton will not interact with these directories directly.

        # 3. Set up memory: Cortex with global_dir + project_dir
        global_memory_dir = workspace / "memory"
        project_memory_dir = workspace / "memory"  # same scope for server (per-user already)
        cortex = Cortex(
            global_dir=global_memory_dir,
            project_dir=project_memory_dir,
            mode="autopilot",
            llm_client=llm_client,
        )

        # 4. Set up EpisodicMemory
        episodes_dir = workspace / "episodes"
        episodic = EpisodicMemory(episodes_dir, enabled=True)
        episodic.start_session()

        # 5. Create ChatSession with all dependencies
        self.session = ChatSession(
            llm_client,
            runtime_context=runtime_context,
            backend=backend,
            cortex=cortex,
            episodic=episodic,
            coding_provider=coding_provider,
            coding_api_key=coding_api_key,
            coding_model=coding_model,
            workspace_path=workspace,
            extra_env=extra_env,
            shared_memory=shared_memory,
        )

        # 6. Load history into session if provided
        if history:
            self.session.load_history(history)
            self.session.repair_history()

        self._usage = (0, 0)

    async def chat_stream(self, prompt: str) -> AsyncIterator[StreamEvent]:
        """Stream a single turn. Yields events as they arrive."""
        async for event in self.session.turn_stream(prompt):
            yield event

    async def close(self):
        """Clean up resources."""
        await self.session.close()
