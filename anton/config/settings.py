from __future__ import annotations

from pathlib import Path

from pydantic import PrivateAttr, field_validator
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
    model_config = {
        "env_prefix": "ANTON_",
        "env_file": _ENV_FILES,
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    planning_provider: str = "anthropic"
    planning_model: str = "claude-sonnet-4-6"
    coding_provider: str = "anthropic"
    coding_model: str = "claude-haiku-4-5-20251001"

    max_tokens: int = 8192  # max output tokens per LLM call

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    memory_enabled: bool = True
    memory_dir: str = ".anton"

    context_dir: str = ".anton/context"

    memory_mode: str = "autopilot"  # autopilot | copilot | off

    episodic_memory: bool = True  # episodic memory archive — on by default

    proactive_dashboards: bool = (
        False  # when True, build HTML dashboards; when False, CLI output only
    )

    theme: str = "auto"

    disable_autoupdates: bool = False

    terms_consent: bool = False
    first_run_done: bool = False

    # Analytics — anonymous usage events (set ANTON_ANALYTICS_ENABLED=false to opt out)
    analytics_enabled: bool = True
    analytics_url: str = "https://x6nik28qi6.execute-api.us-east-2.amazonaws.com/default/zoomInfoCollector"

    # Minds datasource integration
    minds_enabled: bool = True  # use Minds server as LLM provider
    minds_api_key: str | None = None
    minds_url: str = "https://mdb.ai"
    minds_mind_name: str | None = None
    minds_datasource: str | None = None
    minds_datasource_engine: str | None = None
    minds_ssl_verify: bool = True

    # Publish service (anton-services API Gateway)
    publish_url: str = "https://4nton.ai"
    bug_report_url: str | None = None  # Defaults to publish_url if not set

    @field_validator("minds_ssl_verify", mode="before")
    @classmethod
    def _parse_minds_ssl_verify(cls, v):
        if isinstance(v, str) and v.strip() == "":
            return True
        return v

    def model_post_init(self, __context) -> None:
        """Derive openai vars from minds credentials when appropriate."""
        if (
            self.minds_api_key
            and not self.openai_api_key
            and (
                self.planning_provider == "openai-compatible"
                or self.coding_provider == "openai-compatible"
            )
        ):
            self.openai_api_key = self.minds_api_key
            if not self.openai_base_url:
                self.openai_base_url = f"{self.minds_url.rstrip('/')}/api/v1"

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
        if not Path(self.context_dir).is_absolute():
            self.context_dir = str(base / self.context_dir)
