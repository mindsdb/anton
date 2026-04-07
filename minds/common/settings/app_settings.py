from enum import Enum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeploymentMode(str, Enum):
    SELF_HOSTED = "self-hosted"
    CLOUD = "cloud"


class Agent(str, Enum):
    ANTON = "anton_agent"
    TEXT_TO_SQL = "candidate_sql_agent"


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

    # Connection pool configurations
    max_overflow: int = Field(
        default=20, description="The maximum overflow size of the database connection pool"
    )  # DATABASE__MAX_OVERFLOW
    pool_pre_ping: bool = Field(default=True, description="Whether to enable pool pre-ping")  # DATABASE__POOL_PRE_PING
    pool_recycle: int = Field(default=300, description="The pool recycle time in seconds")  # DATABASE__POOL_RECYCLE
    pool_size: int = Field(default=20, description="The size of the database connection pool")  # DATABASE__POOL_SIZE
    pool_timeout: int = Field(default=300, description="The pool timeout in seconds")  # DATABASE__POOL_TIMEOUT

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

    @field_validator("supported_models", "supported_coding_models", mode="before")
    @classmethod
    def split_supported_anthropic_models(cls, v: list[str] | str) -> list[str]:
        if isinstance(v, str):
            return [model.strip() for model in v.split(",")]
        return [model.strip() for model in v]


class MindsDBSettings(Settings):
    url: str = Field(default="http://localhost:47334", description="The URL of the MindsDB instance")  # MINDSDB__URL
    api_key: str = Field(default="", description="The MindsDB API key")  # MINDSDB__API_KEY
    login: str = Field(default="mindsdb", description="The MindsDB login username")  # MINDSDB__LOGIN
    password: str = Field(default="", description="The MindsDB password")  # MINDSDB__PASSWORD


class DataCatalogSettings(Settings):
    execution_mode: str = Field(
        default="asynchronous", description="The execution mode for data catalog operations"
    )  # DATA_CATALOG__EXECUTION_MODE
    job_name: str = Field(
        default="load-data-catalog", description="The name of the data catalog job"
    )  # DATA_CATALOG__JOB_NAME
    job_deployment_name: str = Field(
        default="local--data-catalog-loader", description="The deployment name for the data catalog job"
    )  # DATA_CATALOG__JOB_DEPLOYMENT_NAME
    cache_type: str = Field(
        default="in_memory", description="The type of cache to use for data catalog"
    )  # DATA_CATALOG__CACHE_TYPE
    cache_max_size: int = Field(
        default=100, description="The maximum size of the data catalog cache"
    )  # DATA_CATALOG__CACHE_MAX_SIZE


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


class ChartCompilerSettings(Settings):
    max_rows_to_process: int = Field(
        default=1000, description="Maximum number of rows to process for chart generation"
    )  # CHART_COMPILER__MAX_ROWS_TO_PROCESS
    max_series: int = Field(
        default=12, description="Maximum number of series to render in a chart"
    )  # CHART_COMPILER__MAX_SERIES


class ChartRendererSettings(Settings):
    image_width: int = Field(
        default=1600, ge=1, description="Default PNG width in pixels for server-rendered charts"
    )  # CHART_RENDERER__IMAGE_WIDTH
    image_height: int = Field(
        default=800, ge=1, description="Default PNG height in pixels for server-rendered charts"
    )  # CHART_RENDERER__IMAGE_HEIGHT
    image_dpi: int = Field(
        default=100, ge=1, description="DPI for Matplotlib figure when rendering chart PNGs"
    )  # CHART_RENDERER__IMAGE_DPI


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


class AgentsSettings(Settings):
    default_agent: Agent = Field(
        default=Agent.TEXT_TO_SQL, description="The default agent to use"
    )  # AGENTS__DEFAULT_AGENT

    @field_validator("default_agent", mode="before")
    @classmethod
    def validate_default_agent(cls, v: Agent | str) -> Agent:
        if isinstance(v, str):
            v = Agent(v)
        return v


class AppSettings(Settings):
    env: str = Field(default="local", description="The environment (local, dev, prod, etc.)")  # ENV

    log_level: str = Field(default="WARNING", description="The logging level")  # LOG_LEVEL

    deployment_mode: DeploymentMode = Field(
        default=DeploymentMode.SELF_HOSTED, description="The deployment mode"
    )  # DEPLOYMENT_MODE

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)  # DATABASE__*
    openai: OpenAISettings = Field(default_factory=OpenAISettings)  # OPENAI__*
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)  # ANTHROPIC__*
    mindsdb: MindsDBSettings = Field(default_factory=MindsDBSettings)  # MINDSDB__*
    data_catalog: DataCatalogSettings = Field(default_factory=DataCatalogSettings)  # DATA_CATALOG__*
    default_models: DefaultModelsSettings = Field(default_factory=DefaultModelsSettings)  # DEFAULT_MODELS__*
    minds: MindsSettings = Field(default_factory=MindsSettings)  # MINDS__*
    mind_castle: MindCastleSettings = Field(default_factory=MindCastleSettings)  # MIND_CASTLE__*
    redis: RedisSettings = Field(default_factory=RedisSettings)  # REDIS__*
    agents: AgentsSettings = Field(default_factory=AgentsSettings)  # AGENTS__*
    chart_compiler: ChartCompilerSettings = Field(default_factory=ChartCompilerSettings)  # CHART_COMPILER__*
    chart_renderer: ChartRendererSettings = Field(default_factory=ChartRendererSettings)  # CHART_RENDERER__*

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
