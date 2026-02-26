"""
The client for running Anton via Python.
"""

from collections.abc import AsyncIterator
from pathlib import Path
from rich.console import Console

from anton.chat import ChatSession
from anton.config.settings import AntonSettings
from anton.context.self_awareness import SelfAwarenessContext
from anton.llm.client import LLMClient
from anton.llm.provider import StreamEvent
from anton.scratchpad import ScratchpadManager
from anton.workspace import Workspace


class Anton:
    def __init__(
        self,
        workspace_dir: str | None = None,
        runtime_context: str = "",
        history: list[dict] = [],
        scratchpad_manager: ScratchpadManager | None = None,
    ):
        """
        Initialize the Anton client.

        Args:
            workspace_dir: The directory to use for the workspace. Defaults to the current directory.
            runtime_context: The runtime context to use for the chat session. Defaults to an empty string.
            history: The history of the chat session. Defaults to an empty list.
            scratchpad_manager: The scratchpad manager to use for the chat session. Defaults to a new ScratchpadManager.
        """
        # Initialize settings
        settings = AntonSettings()
        settings.resolve_workspace(workspace_dir)
        
        # Create console and workspace
        console = Console()
        workspace = Workspace(settings.workspace_path)
        workspace.apply_env_to_process()
        
        # Initialize self-awareness context
        self_awareness = SelfAwarenessContext(Path(settings.context_dir))
        
        # Create LLM client
        llm_client = LLMClient.from_settings(settings)

        # Create chat session
        self._chat_session = ChatSession(
            llm_client=llm_client,
            self_awareness=self_awareness,
            runtime_context=runtime_context,
            workspace=workspace,
            console=console,
            history=history,
            scratchpad_manager=scratchpad_manager,
        )

    async def chat(self, user_input: str | list[dict]) -> str:
        """
        Chat with Anton.
        """
        return await self._chat_session.turn(user_input)

    async def chat_stream(self, user_input: str | list[dict]) -> AsyncIterator[StreamEvent]:
        """
        Chat with Anton.
        """
        async for event in self._chat_session.turn_stream(user_input):
            yield event
