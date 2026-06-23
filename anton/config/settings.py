from __future__ import annotations

from pathlib import Path

from pydantic import PrivateAttr, ValidationInfo, field_validator, model_validator

from anton.core.settings import CoreSettings


def _build_env_files() -> list[str]:
    """Build .env loading chain: cwd/.env -> .anton/.env -> ~/.anton/.env
    -> ~/.cowork/.env. Later files win, so the consolidated ~/.cowork/.env
    takes precedence; ~/.anton/.env stays as a fallback for installs that
    haven't migrated yet."""
    files: list[str] = [".env"]
    local_env = Path.cwd() / ".anton" / ".env"
    if local_env.is_file():
        files.append(str(local_env))
    user_env = Path("~/.anton/.env").expanduser()
    if user_env.is_file():
        files.append(str(user_env))
    cowork_env = Path("~/.cowork/.env").expanduser()
    if cowork_env.is_file():
        files.append(str(cowork_env))
    return files


_ENV_FILES = _build_env_files()


class AntonSettings(CoreSettings):
    model_config = {"env_prefix": "ANTON_", "env_file": _ENV_FILES, "env_file_encoding": "utf-8", "extra": "ignore"}

    planning_provider: str = "anthropic"
    planning_model: str = "claude-sonnet-4-6"
    coding_provider: str = "anthropic"
    coding_model: str = "claude-haiku-4-5-20251001"

    # Opaque reasoning-effort level (e.g. "low" | "medium" | "high" | "xhigh" |
    # "max"), forwarded to the provider in its native shape when set. None means
    # the provider's own default. The value is validated by the upstream
    # router/provider, not here.
    #
    # NOTE: effort only applies to reasoning-capable models. Anthropic returns a
    # 400 for `effort` on Haiku 4.5 / Sonnet 4.5 — including the default
    # `coding_model` above — so it needs an Opus-4.5+/Sonnet-4.6-class model; the
    # OpenAI `reasoning_effort` likewise needs a reasoning model (o-series / GPT-5
    # class). Set these only when the corresponding model supports it.
    planning_reasoning_effort: str | None = None
    coding_reasoning_effort: str | None = None

    max_tokens: int = 8192  # max output tokens per LLM call

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_api_version: str | None = None  # Azure api-version query param

    # Web tools — on by default. For LLM providers that ship native server-side
    # web search/fetch (Anthropic, OpenAI, mdb.ai passthrough), the tools execute
    # inside the provider on the user's existing key. For generic
    # openai-compatible endpoints, web_search needs an external provider key
    # (Exa or Brave); web_fetch always falls back to stdlib HTTP.
    web_search_enabled: bool = True
    web_fetch_enabled: bool = True

    # Case 3 fallback — only consulted when the LLM provider lacks native web
    # search and the user is on a generic OpenAI-compatible endpoint.
    external_search_provider: str | None = None  # "exa" | "brave" | None
    exa_api_key: str | None = None
    brave_api_key: str | None = None

    skills_root: Path | None = None

    memory_enabled: bool = True
    # TODO: Calling this memory_dir is a bit misleading, because there are other directories that live here
    memory_dir: str = ".anton"

    context_dir: str = ".anton/context"

    # Project-visible directory where user-facing artifacts (HTML
    # apps, docs, datasets, etc.) live. One subfolder per artifact,
    # each carrying a `metadata.json` and a `README.md` rendered
    # from it. Replaces the legacy `output_dir = ".anton/output"`
    # setting — anton-core no longer auto-migrates; users move
    # their old `.anton/output/` files manually if they want them
    # tracked under the new model.
    artifacts_dir: str = "artifacts"

    memory_mode: str = "autopilot"  # autopilot | copilot | off

    episodic_memory: bool = True  # episodic memory archive — on by default

    proactive_dashboards: bool = False  # when True, build HTML dashboards; when False, CLI output only

    # "Do first, ask later": act on reasonable defaults and surface assumptions
    # inline instead of stopping to ask. False = cautious ask-first discipline.
    act_first: bool = True

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

    # Publish service
    publish_url: str = "https://4nton.ai"

    backend: str = "local"  # local | remote

    @field_validator("backend", mode="after")
    @classmethod
    def _validate_backend(cls, v: str, info: ValidationInfo) -> str:
        if v == "remote" and (not info.data.get("minds_url") or not info.data.get("minds_api_key")):
            raise ValueError("Minds URL and API key are required for remote backend")
        return v

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
            and (self.planning_provider == "openai-compatible" or self.coding_provider == "openai-compatible")
        ):
            self.openai_api_key = self.minds_api_key
            if not self.openai_base_url:
                # Host-aware base URL: api.mindshub.ai serves the
                # OpenAI-compatible API at /v1, the legacy mdb.ai host at
                # /api/v1. The previous hardcoded /api/v1 was correct only
                # for mdb.ai and produced a wrong endpoint for mindshub
                # (ENG-436). Mirrors cowork-server's minds_chat_base_url.
                base = self.minds_url.rstrip("/")
                if base.endswith("/v1"):
                    self.openai_base_url = base
                elif "mdb.ai" in base:
                    self.openai_base_url = f"{base}/api/v1"
                else:
                    self.openai_base_url = f"{base}/v1"

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
            memory_root = base / self.memory_dir
            self.memory_dir = str(memory_root)
        else:
            memory_root = Path(self.memory_dir)

        if not Path(self.context_dir).is_absolute():
            self.context_dir = str(base / self.context_dir)
        if not Path(self.artifacts_dir).is_absolute():
            self.artifacts_dir = str(memory_root / self.artifacts_dir)
