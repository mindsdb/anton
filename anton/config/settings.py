from __future__ import annotations

from pathlib import Path

from pydantic import PrivateAttr
from pydantic_settings import BaseSettings


def _build_env_files() -> list[str]:
    """Build .env loading chain: cwd/.env -> .anton/.env -> ~/.anton/.env"""
    files: list[str] = [".env"]
    local_env = Path.cwd() / ".anton" / ".env"
    if local_env.is_file():
        files.append(str(local_env))
    user_env = Path("~/.anton/.env").expanduser()
    if user_env.is_file():
        files.append(str(user_env))
    return files


_ENV_FILES = _build_env_files()


class AntonSettings(BaseSettings):
    model_config = {"env_prefix": "ANTON_", "env_file": _ENV_FILES, "env_file_encoding": "utf-8", "extra": "ignore"}

    planning_provider: str = "anthropic"
    planning_model: str = "claude-sonnet-4-6"
    coding_provider: str = "anthropic"
    coding_model: str = "claude-opus-4-6"

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    skills_dir: str = "skills"
    user_skills_dir: str = ".anton/skills"

    memory_enabled: bool = True
    memory_dir: str = ".anton"

    context_dir: str = ".anton/context"

    theme: str = "auto"

    _workspace: Path = PrivateAttr(default=None)

    @property
    def workspace_path(self) -> Path:
        """Return the resolved workspace root (parent of .anton/)."""
        if self._workspace is not None:
            return self._workspace
        return Path.cwd()

    def resolve_workspace(self, folder: str | None = None) -> None:
        """Resolve all relative paths against the workspace base directory.

        Args:
            folder: Optional explicit folder path. Defaults to cwd.
        """
        base = Path(folder).resolve() if folder else Path.cwd()
        self._workspace = base

        # Convert relative paths to absolute under base
        if not Path(self.memory_dir).is_absolute():
            self.memory_dir = str(base / self.memory_dir)
        if not Path(self.user_skills_dir).is_absolute():
            self.user_skills_dir = str(base / self.user_skills_dir)
        if not Path(self.context_dir).is_absolute():
            self.context_dir = str(base / self.context_dir)

        # Ensure .anton/ directory exists
        anton_dir = base / ".anton"
        anton_dir.mkdir(parents=True, exist_ok=True)
