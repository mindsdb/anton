from enum import Enum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeploymentMode(str, Enum):
    SELF_HOSTED = "self-hosted"
    CLOUD = "cloud"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )


class DatabaseSettings(Settings):
    uri: str = Field(
        default="postgresql://minds:minds@localhost:35432/minds", description="The database connection URI"
    )  # DATABASE__URI

    # Connection pool configurations. Sized for async FastAPI where each worker's
    # event loop multiplexes many requests but only a handful are mid-query at once.
    # Total Postgres connections = replicas * workers * (pool_size + max_overflow).
    # At HPA max (8 replicas * 6 workers * 15) = 720 connections worst case.
    max_overflow: int = Field(
        default=10, description="The maximum overflow size of the database connection pool"
    )  # DATABASE__MAX_OVERFLOW
    pool_pre_ping: bool = Field(default=True, description="Whether to enable pool pre-ping")  # DATABASE__POOL_PRE_PING
    pool_recycle: int = Field(default=300, description="The pool recycle time in seconds")  # DATABASE__POOL_RECYCLE
    pool_size: int = Field(default=5, description="The size of the database connection pool")  # DATABASE__POOL_SIZE
    # Lower than the legacy 5-minute default: if a worker waits >30s for a connection the pool
    # is exhausted (likely a leaked session). Fail fast so we see the issue instead of hiding it.
    pool_timeout: int = Field(default=30, description="The pool timeout in seconds")  # DATABASE__POOL_TIMEOUT

    # Query timeout configurations
    query_timeout: int = Field(default=300, description="The query timeout in seconds")  # DATABASE__QUERY_TIMEOUT
    statement_timeout: int = Field(
        default=300000, description="The statement timeout in milliseconds"
    )  # DATABASE__STATEMENT_TIMEOUT


class OpenAISettings(Settings):
    api_url: str = Field(
        default="https://api.openai.com/v1", description="The URL of the OpenAI API"
    )  # OPENAI__API_URL
    api_key: str = Field(default="not set", description="The OpenAI API key")  # OPENAI__API_KEY
    max_tokens: int = Field(
        default=400000, description="The maximum number of tokens for OpenAI API"
    )  # OPENAI__MAX_TOKENS
    # TODO: Should the models be extracted programmatically?
    supported_models: list[str] = Field(
        default=["gpt-4o", "gpt-4.1", "gpt-5.2"], description="The supported OpenAI models"
    )  # OPENAI__SUPPORTED_MODELS
    supported_coding_models: list[str] = Field(
        default=["gpt-4.1", "gpt-5.3-codex", "gpt-5.3-instant"], description="The supported OpenAI coding models"
    )  # OPENAI__SUPPORTED_CODING_MODELS

    # Passthrough-agent alias → model mappings. Aliases drop the version
    # from the request-facing name (``latest:gpt``, ``latest:gpt-mini`` etc.)
    # so client code doesn't churn when ops bump a major upstream version;
    # the actual model identifier lives here and can be overridden via env.
    passthrough_gpt_model: str = Field(
        default="gpt-5.5",
        description=(
            "OpenAI model used for the `latest:gpt` / `latest:gpt-low|medium|high` "
            "aliases. The alias selects reasoning_effort; this setting selects "
            "the upstream model."
        ),
    )  # OPENAI__PASSTHROUGH_GPT_MODEL
    passthrough_gpt_codex_model: str = Field(
        default="gpt-5.3-codex",
        description="OpenAI model used for the `latest:gpt-codex` alias.",
    )  # OPENAI__PASSTHROUGH_GPT_CODEX_MODEL
    passthrough_gpt_mini_model: str = Field(
        default="gpt-5.4-mini",
        description="OpenAI model used for the `latest:gpt-mini` alias.",
    )  # OPENAI__PASSTHROUGH_GPT_MINI_MODEL
    passthrough_gpt_nano_model: str = Field(
        default="gpt-5.4-nano",
        description="OpenAI model used for the `latest:gpt-nano` alias.",
    )  # OPENAI__PASSTHROUGH_GPT_NANO_MODEL

    @field_validator("supported_models", "supported_coding_models", mode="before")
    @classmethod
    def split_supported_openai_models(cls, v: list[str] | str) -> list[str]:
        if isinstance(v, str):
            return [model.strip() for model in v.split(",")]
        return [model.strip() for model in v]


class AnthropicSettings(Settings):
    api_key: str = Field(default="", description="The Anthropic API key")  # ANTHROPIC__API_KEY
    # TODO: Should the models be extracted programmatically?
    supported_models: list[str] = Field(
        default=["claude-sonnet-4-5", "claude-opus-4-5", "claude-sonnet-4-6", "claude-opus-4-6"],
        description="The supported Anthropic models",
    )  # ANTHROPIC__SUPPORTED_MODELS
    supported_coding_models: list[str] = Field(
        default=["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"],
        description="The supported Anthropic coding models",
    )  # ANTHROPIC__SUPPORTED_CODING_MODELS

    # Native web-tool versions used by the passthrough agent. These are
    # Anthropic-versioned API contracts (e.g. "web_search_20250305") that
    # change in a regular cadence.
    web_search_tool_type: str = Field(
        default="web_search_20250305",
        description="Anthropic native web_search tool type (versioned). Example: 'web_search_20250305'.",
    )  # ANTHROPIC__WEB_SEARCH_TOOL_TYPE
    web_fetch_tool_type: str = Field(
        default="web_fetch_20250910",
        description="Anthropic native web_fetch tool type (versioned). Example: 'web_fetch_20250910'.",
    )  # ANTHROPIC__WEB_FETCH_TOOL_TYPE
    web_fetch_beta_header: str = Field(
        default="web-fetch-2025-09-10",
        description="Value for the 'anthropic-beta' header required by the "
        "web_fetch tool. Example: 'web-fetch-2025-09-10'.",
    )  # ANTHROPIC__WEB_FETCH_BETA_HEADER

    # Passthrough-agent explicit-model aliases. Override via env to pin
    # or upgrade independently of code releases — the alias surface stays
    # stable while the upstream model can move.
    passthrough_sonnet_model: str = Field(
        default="claude-sonnet-4-6",
        description="Anthropic model used for the `latest:sonnet` alias.",
    )  # ANTHROPIC__PASSTHROUGH_SONNET_MODEL
    passthrough_opus_model: str = Field(
        default="claude-opus-4-7",
        description="Anthropic model used for the `latest:opus` alias.",
    )  # ANTHROPIC__PASSTHROUGH_OPUS_MODEL
    passthrough_haiku_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Anthropic model used for the `latest:haiku` alias.",
    )  # ANTHROPIC__PASSTHROUGH_HAIKU_MODEL
    passthrough_fable_model: str = Field(
        default="claude-fable-5",
        description="Anthropic model used for the `latest:fable` alias.",
    )  # ANTHROPIC__PASSTHROUGH_FABLE_MODEL

    @field_validator("supported_models", "supported_coding_models", mode="before")
    @classmethod
    def split_supported_anthropic_models(cls, v: list[str] | str) -> list[str]:
        if isinstance(v, str):
            return [model.strip() for model in v.split(",")]
        return [model.strip() for model in v]


class FireworksSettings(Settings):
    api_key: str = Field(default="", description="The Fireworks.ai API key")  # FIREWORKS__API_KEY
    anthropic_base_url: str = Field(
        default="https://api.fireworks.ai/inference",
        description="Anthropic-compatible base URL for Fireworks (SDK appends /v1/messages)",
    )  # FIREWORKS__ANTHROPIC_BASE_URL

    # Passthrough-agent alias → Fireworks-hosted model name. Aliases are
    # versionless (``latest:kimi``); the version-pinned identifier lives
    # here so a Kimi/DeepSeek/Qwen point-release is a one-env-var change.
    passthrough_kimi_model: str = Field(
        default="accounts/fireworks/models/kimi-k2p6",
        description="Fireworks model used for the `latest:kimi` alias.",
    )  # FIREWORKS__PASSTHROUGH_KIMI_MODEL
    passthrough_deepseek_model: str = Field(
        default="accounts/fireworks/models/deepseek-v4-pro",
        description="Fireworks model used for the `latest:deepseek` alias.",
    )  # FIREWORKS__PASSTHROUGH_DEEPSEEK_MODEL
    passthrough_qwen_model: str = Field(
        default="accounts/fireworks/models/qwen3p6-plus",
        description="Fireworks model used for the `latest:qwen` alias.",
    )  # FIREWORKS__PASSTHROUGH_QWEN_MODEL


class GeminiSettings(Settings):
    api_key: str = Field(default="", description="The Google Gemini API key")  # GEMINI__API_KEY

    # Passthrough-agent alias → Gemini model. Override via env to pin or
    # upgrade independently of code releases.
    passthrough_gemini_model: str = Field(
        default="gemini-3.1-pro-preview",
        description="Gemini model used for the `latest:gemini` alias.",
    )  # GEMINI__PASSTHROUGH_GEMINI_MODEL
    passthrough_gemini_flash_model: str = Field(
        default="gemini-3.5-flash",
        description=(
            "Gemini Flash model used for the `latest:gemini-flash` alias — cheaper, faster sibling of the Pro line."
        ),
    )  # GEMINI__PASSTHROUGH_GEMINI_FLASH_MODEL


class MindsDBSettings(Settings):
    url: str = Field(default="http://localhost:47334", description="The URL of the MindsDB instance")  # MINDSDB__URL
    api_key: str = Field(default="", description="The MindsDB API key")  # MINDSDB__API_KEY
    login: str = Field(default="mindsdb", description="The MindsDB login username")  # MINDSDB__LOGIN
    password: str = Field(default="", description="The MindsDB password")  # MINDSDB__PASSWORD


class DefaultModelsSettings(Settings):
    default_provider: str = Field(
        default="anthropic", description="The default provider"
    )  # DEFAULT_MODELS__DEFAULT_PROVIDER
    openai_model: str = Field(default="gpt-4o", description="The default OpenAI model")  # DEFAULT_MODELS__OPENAI_MODEL
    anthropic_model: str = Field(
        default="claude-opus-4-6", description="The default Anthropic model"
    )  # DEFAULT_MODELS__ANTHROPIC_MODEL

    default_coding_provider: str = Field(
        default="anthropic", description="The default coding provider"
    )  # DEFAULT_MODELS__DEFAULT_CODING_PROVIDER
    openai_coding_model: str = Field(
        default="gpt-5.3-codex", description="The default OpenAI coding model"
    )  # DEFAULT_MODELS__OPENAI_CODING_MODEL
    anthropic_coding_model: str = Field(
        default="claude-haiku-4-5-20251001", description="The default Anthropic coding model"
    )  # DEFAULT_MODELS__ANTHROPIC_CODING_MODEL


class RedisSettings(Settings):
    db: int = Field(default=0, description="The Redis database number")  # REDIS__DB
    host: str = Field(default="localhost", description="The Redis host")  # REDIS__HOST
    port: int = Field(default=6379, description="The Redis port")  # REDIS__PORT
    cache_ttl: int = Field(default=3600, description="The Redis cache TTL in seconds")  # REDIS__CACHE_TTL


class MindsSettings(Settings):
    max_display_rows: int = Field(
        default=100, description="Maximum number of rows to display"
    )  # MINDS__MAX_DISPLAY_ROWS
    max_column_width: int = Field(default=300, description="Maximum width of columns")  # MINDS__MAX_COLUMN_WIDTH
    max_sql_retries: int = Field(default=4, description="Maximum number of SQL retries")  # MINDS__MAX_SQL_RETRIES

    enable_model_selection: bool = Field(
        default=False, description="Whether to enable model selection"
    )  # MINDS__ENABLE_MODEL_SELECTION


class MindCastleSettings(Settings):
    encryption_type: str = Field(
        default="localencryption", description="The encryption type for MindCastle"
    )  # MIND_CASTLE__ENCRYPTION_TYPE


class FeatureFlagSettings(Settings):
    name: str = Field(default="", description="The feature flag name")  # FEATURE_FLAG__NAME
    default_value: bool = Field(
        default=False, description="The default value for feature flag"
    )  # FEATURE_FLAG__DEFAULT_VALUE


class StatsigSettings(Settings):
    sdk_key: str = Field(default="", description="The Statsig SDK key")  # STATSIG__SDK_KEY
    environment: str = Field(default="", description="The Statsig environment")  # STATSIG__ENVIRONMENT
    disable_network: bool = Field(default=True, description="Whether to disable network")  # STATSIG__DISABLE_NETWORK
    disable_all_logging: bool = Field(
        default=True, description="Whether to disable all logging"
    )  # STATSIG__DISABLE_ALL_LOGGING


class AppSettings(Settings):
    env: str = Field(default="local", description="The environment (local, dev, prod, etc.)")  # ENV

    log_level: str = Field(default="WARNING", description="The logging level")  # LOG_LEVEL

    deployment_mode: DeploymentMode = Field(
        default=DeploymentMode.SELF_HOSTED, description="The deployment mode"
    )  # DEPLOYMENT_MODE

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)  # DATABASE__*
    openai: OpenAISettings = Field(default_factory=OpenAISettings)  # OPENAI__*
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)  # ANTHROPIC__*
    fireworks: FireworksSettings = Field(default_factory=FireworksSettings)  # FIREWORKS__*
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)  # GEMINI__*
    mindsdb: MindsDBSettings = Field(default_factory=MindsDBSettings)  # MINDSDB__*
    default_models: DefaultModelsSettings = Field(default_factory=DefaultModelsSettings)  # DEFAULT_MODELS__*
    mind_castle: MindCastleSettings = Field(default_factory=MindCastleSettings)  # MIND_CASTLE__*
    redis: RedisSettings = Field(default_factory=RedisSettings)  # REDIS__*

    statsig: StatsigSettings = Field(default_factory=StatsigSettings)  # STATSIG__*

    feature_flag_enable_langfuse: FeatureFlagSettings = Field(
        default=FeatureFlagSettings(name="enable-langfuse", default_value=True)
    )  # FEATURE_FLAG__ENABLE_LANGFUSE

    feature_flag_enable_model_selection: FeatureFlagSettings = Field(
        default=FeatureFlagSettings(name="enable-model-selection", default_value=True)
    )  # FEATURE_FLAG__ENABLE_MODEL_SELECTION


@lru_cache
def get_app_settings() -> AppSettings:
    """Get cached application settings."""
    return AppSettings()
