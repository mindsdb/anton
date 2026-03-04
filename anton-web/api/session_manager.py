from __future__ import annotations

import os
from pathlib import Path


class SessionManager:
    """Creates and manages ChatSession instances for the web API."""

    def __init__(self, workspace_path: Path) -> None:
        self._workspace_path = workspace_path
        self._sessions: dict = {}
        self._settings = None
        self._llm_client = None
        self._workspace = None
        self._cortex = None
        self._history_store = None

    async def initialize(self) -> None:
        from anton.config.settings import AntonSettings
        from anton.llm.client import LLMClient
        from anton.memory.cortex import Cortex
        from anton.memory.history_store import HistoryStore
        from anton.workspace import Workspace

        self._workspace_path.mkdir(parents=True, exist_ok=True)

        ws = Workspace(self._workspace_path)
        if not ws.is_initialized():
            ws.initialize()
        ws.apply_env_to_process()

        global_ws = Workspace(Path.home())
        if global_ws.is_initialized():
            global_ws.apply_env_to_process()

        self._workspace = ws

        self._settings = AntonSettings()
        self._settings.resolve_workspace(str(self._workspace_path))

        self._llm_client = LLMClient.from_settings(self._settings)

        global_memory_dir = Path.home() / ".anton" / "memory"
        project_memory_dir = self._workspace_path / ".anton" / "memory"

        self._cortex = Cortex(
            global_dir=global_memory_dir,
            project_dir=project_memory_dir,
            mode=self._settings.memory_mode,
            llm_client=self._llm_client,
        )

        episodes_dir = self._workspace_path / ".anton" / "episodes"
        self._history_store = HistoryStore(episodes_dir)

        # Scratchpad subprocesses inherit CWD for file output
        os.chdir(self._workspace_path)

    async def create_session(self):
        from anton.chat import ChatSession
        from anton.memory.episodes import EpisodicMemory

        s = self._settings
        episodes_dir = self._workspace_path / ".anton" / "episodes"
        episodic = EpisodicMemory(episodes_dir, enabled=s.episodic_memory)
        session_id = episodic.start_session()

        coding_api_key = (
            s.anthropic_api_key if s.coding_provider == "anthropic" else s.openai_api_key
        ) or ""

        runtime_context = (
            f"- Provider: {s.planning_provider}\n"
            f"- Planning model: {s.planning_model}\n"
            f"- Coding model: {s.coding_model}\n"
            f"- Workspace: {self._workspace_path}\n"
            f"- Memory mode: {s.memory_mode}\n"
            f"- Interface: web"
        )

        session = ChatSession(
            self._llm_client,
            cortex=self._cortex,
            episodic=episodic,
            runtime_context=runtime_context,
            workspace=self._workspace,
            coding_provider=s.coding_provider,
            coding_api_key=coding_api_key,
            history_store=self._history_store,
            session_id=session_id,
        )

        self._sessions[session_id] = session
        return session_id, session

    def get_session(self, session_id: str):
        return self._sessions.get(session_id)

    async def resume_session(self, session_id: str):
        from anton.chat import ChatSession
        from anton.memory.episodes import EpisodicMemory

        history = self._history_store.load(session_id)
        if history is None:
            return None

        s = self._settings
        episodes_dir = self._workspace_path / ".anton" / "episodes"
        episodic = EpisodicMemory(episodes_dir, enabled=s.episodic_memory)
        episodic.resume_session(session_id)

        coding_api_key = (
            s.anthropic_api_key if s.coding_provider == "anthropic" else s.openai_api_key
        ) or ""

        runtime_context = (
            f"- Provider: {s.planning_provider}\n"
            f"- Planning model: {s.planning_model}\n"
            f"- Coding model: {s.coding_model}\n"
            f"- Workspace: {self._workspace_path}\n"
            f"- Memory mode: {s.memory_mode}\n"
            f"- Interface: web"
        )

        session = ChatSession(
            self._llm_client,
            cortex=self._cortex,
            episodic=episodic,
            runtime_context=runtime_context,
            workspace=self._workspace,
            coding_provider=s.coding_provider,
            coding_api_key=coding_api_key,
            history_store=self._history_store,
            session_id=session_id,
            initial_history=history,
        )

        self._sessions[session_id] = session
        return session_id, session

    def list_sessions(self) -> list[dict]:
        return self._history_store.list_sessions()

    async def shutdown(self) -> None:
        self._sessions.clear()
