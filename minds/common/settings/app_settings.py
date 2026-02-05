from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    model_name: str = Field(default="gpt-4o", description="The OpenAI model name")  # OPENAI__MODEL_NAME


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
    mind_model: str = Field(default="gpt-4o", description="The default model for minds")  # DEFAULT_MODELS__MIND_MODEL
    google_model: str = Field(
        default="gemini-2.5-pro", description="The default Google model"
    )  # DEFAULT_MODELS__GOOGLE_MODEL


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


class AuthSettings(Settings):
    disable: bool = Field(default=False, description="Whether to disable authentication")  # AUTH__DISABLE


class LaunchDarklySettings(Settings):
    sdk_key: str = Field(default="not set", description="The LaunchDarkly SDK key")  # LAUNCHDARKLY__SDK_KEY
    offline_mode: bool = Field(
        default=True, description="Whether LaunchDarkly is in offline mode"
    )  # LAUNCHDARKLY__OFFLINE_MODE

    # Optional Relay Proxy (or custom endpoint) configuration. When set, the SDK
    # will connect to these URLs instead of LaunchDarkly-hosted endpoints.
    #
    # For LaunchDarkly Relay Proxy, these typically all point to the proxy URL.
    base_uri: str | None = Field(
        default="", description="Override LaunchDarkly base URI (relay proxy URL)"
    )  # LAUNCHDARKLY__BASE_URI
    stream_uri: str | None = Field(
        default="", description="Override LaunchDarkly stream URI (relay proxy URL)"
    )  # LAUNCHDARKLY__STREAM_URI
    events_uri: str | None = Field(
        default="", description="Override LaunchDarkly events URI (relay proxy URL)"
    )  # LAUNCHDARKLY__EVENTS_URI

    # HTTP timeouts for SDK -> relay (or SDK -> LaunchDarkly if not using relay).
    # These map to ldclient.config.HTTPConfig.
    http_connect_timeout: float = Field(
        default=20.0, description="HTTP connect timeout (seconds) for LaunchDarkly SDK"
    )  # LAUNCHDARKLY__HTTP_CONNECT_TIMEOUT
    http_read_timeout: float = Field(
        default=30.0, description="HTTP read timeout (seconds) for LaunchDarkly SDK"
    )  # LAUNCHDARKLY__HTTP_READ_TIMEOUT

    # If your cluster blocks egress from the relay to LaunchDarkly's events service,
    # you'll see repeated HTTP 503s when the SDK tries to post diagnostics/analytics.
    # These do not affect flag evaluation; they only affect telemetry.
    send_events: bool = Field(
        default=False, description="Whether to send analytics/diagnostic events"
    )  # LAUNCHDARKLY__SEND_EVENTS
    diagnostic_opt_out: bool = Field(
        default=False, description="Disable diagnostic events reporting"
    )  # LAUNCHDARKLY__DIAGNOSTIC_OPT_OUT


class FeatureFlagSettings(Settings):
    name: str = Field(default="", description="The feature flag name")  # FEATURE_FLAG__NAME
    default_value: bool = Field(
        default=False, description="The default value for feature flag"
    )  # FEATURE_FLAG__DEFAULT_VALUE


class AppSettings(Settings):
    env: str = Field(default="local", description="The environment (local, dev, prod, etc.)")  # ENV

    log_level: str = Field(default="DEBUG", description="The logging level")  # LOG_LEVEL

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)  # DATABASE__*
    openai: OpenAISettings = Field(default_factory=OpenAISettings)  # OPENAI__*
    mindsdb: MindsDBSettings = Field(default_factory=MindsDBSettings)  # MINDSDB__*
    data_catalog: DataCatalogSettings = Field(default_factory=DataCatalogSettings)  # DATA_CATALOG__*
    default_models: DefaultModelsSettings = Field(default_factory=DefaultModelsSettings)  # DEFAULT_MODELS__*
    minds: MindsSettings = Field(default_factory=MindsSettings)  # MINDS__*
    auth: AuthSettings = Field(default_factory=AuthSettings)  # AUTH__*
    redis: RedisSettings = Field(default_factory=RedisSettings)  # REDIS__*

    launchdarkly: LaunchDarklySettings = Field(default_factory=LaunchDarklySettings)  # LAUNCHDARKLY__*
    feature_flag_disable_langfuse: FeatureFlagSettings = Field(
        default=FeatureFlagSettings(name="disable-langfuse", default_value=False)
    )  # FEATURE_FLAG__DISABLE_LANGFUSE


@lru_cache
def get_app_settings() -> AppSettings:
    """Get cached application settings."""
    return AppSettings()
