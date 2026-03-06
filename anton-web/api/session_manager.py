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

        os.chdir(self._workspace_path)

    def _runtime_context(self) -> str:
        from anton.chat import _build_runtime_context

        ctx = _build_runtime_context(self._settings)
        ctx += "\n- Interface: web"
        return ctx

    def _coding_api_key(self) -> str:
        s = self._settings
        if s.coding_provider == "anthropic":
            return s.anthropic_api_key or ""
        return s.openai_api_key or ""

    async def create_session(self):
        from anton.chat import ChatSession
        from anton.memory.episodes import EpisodicMemory

        episodes_dir = self._workspace_path / ".anton" / "episodes"
        episodic = EpisodicMemory(episodes_dir, enabled=self._settings.episodic_memory)
        session_id = episodic.start_session()

        session = ChatSession(
            self._llm_client,
            cortex=self._cortex,
            episodic=episodic,
            runtime_context=self._runtime_context(),
            workspace=self._workspace,
            coding_provider=self._settings.coding_provider,
            coding_api_key=self._coding_api_key(),
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

        episodes_dir = self._workspace_path / ".anton" / "episodes"
        episodic = EpisodicMemory(episodes_dir, enabled=self._settings.episodic_memory)
        episodic.resume_session(session_id)

        session = ChatSession(
            self._llm_client,
            cortex=self._cortex,
            episodic=episodic,
            runtime_context=self._runtime_context(),
            workspace=self._workspace,
            coding_provider=self._settings.coding_provider,
            coding_api_key=self._coding_api_key(),
            history_store=self._history_store,
            session_id=session_id,
            initial_history=history,
        )

        self._sessions[session_id] = session
        return session_id, session

    def list_sessions(self) -> list[dict]:
        return self._history_store.list_sessions()

    def cancel_session(self, session_id: str) -> bool:
        """Signal a running session to cancel its current turn."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session._cancel_event.set()
        return True

    async def shutdown(self) -> None:
        self._sessions.clear()
