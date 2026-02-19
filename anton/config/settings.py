from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

_ENV_FILES = [".env"]
_user_env = Path("~/.anton/.env").expanduser()
if _user_env.is_file():
    _ENV_FILES.append(str(_user_env))


class AntonSettings(BaseSettings):
    model_config = {"env_prefix": "ANTON_", "env_file": _ENV_FILES, "env_file_encoding": "utf-8"}

    planning_provider: str = "anthropic"
    planning_model: str = "claude-sonnet-4-6"
    coding_provider: str = "anthropic"
    coding_model: str = "claude-opus-4-6"

    anthropic_api_key: str | None = None

    skills_dir: str = "skills"
    user_skills_dir: str = "~/.anton/skills"

    memory_enabled: bool = True
    memory_dir: str = "~/.anton"

    theme: str = "auto"
