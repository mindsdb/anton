from pydantic import Field

from minds.common.settings.app_settings import Settings


class CandidateSQLAgentSettings(Settings):
    max_display_rows_to_agent: int = Field(
        default=100, description="Maximum number of rows to display"
    )  # CANDIDATE_SQL_AGENT__MAX_DISPLAY_ROWS_TO_AGENT
    max_display_rows_to_user: int = Field(
        default=20, description="Maximum number of rows to display to the user"
    )  # CANDIDATE_SQL_AGENT__MAX_DISPLAY_ROWS_TO_USER
    max_request_limit: int = Field(default=100, description="Maximum number of requests")  # MINDS__MAX_REQUEST_LIMIT
    max_column_width: int = Field(default=300, description="Maximum width of columns")  # MINDS__MAX_COLUMN_WIDTH
    max_planning_retries: int = Field(
        default=3, description="Maximum number of planning retries"
    )  # CANDIDATE_SQL_AGENT__MAX_PLANNING_RETRIES
    max_candidate_retries: int = Field(
        default=2, description="Maximum number of candidate generation retries (for 500s only)"
    )  # CANDIDATE_SQL_AGENT__MAX_CANDIDATE_RETRIES
    max_sql_retries: int = Field(default=4, description="Maximum number of SQL retries")  # MINDS__MAX_SQL_RETRIES
    enable_model_selection: bool = Field(
        default=False, description="Whether to enable model selection"
    )  # CANDIDATE_SQL_AGENT__ENABLE_MODEL_SELECTION
    max_query_plan_steps: int = Field(
        default=10, description="Maximum number of steps allowed in a query plan"
    )  # CANDIDATE_SQL_AGENT__MAX_QUERY_PLAN_STEPS

    # Token budget settings for data catalog context
    max_catalog_tokens_pipeline: int = Field(
        default=80000, description="Token budget for text-to-sql pipeline data catalog"
    )  # CANDIDATE_SQL_AGENT__MAX_CATALOG_TOKENS_PIPELINE
    max_catalog_tokens_orchestrator: int = Field(
        default=100000, description="Token budget for orchestrator agent data catalog"
    )  # CANDIDATE_SQL_AGENT__MAX_CATALOG_TOKENS_ORCHESTRATOR
    max_catalog_tokens_pruned: int = Field(
        default=20000, description="Token budget for pruned data catalogs"
    )  # CANDIDATE_SQL_AGENT__MAX_CATALOG_TOKENS_PRUNED
    large_catalog_table_threshold: int = Field(
        default=100, description="Number of tables to trigger summary mode in pipeline"
    )  # CANDIDATE_SQL_AGENT__LARGE_CATALOG_TABLE_THRESHOLD
    large_catalog_table_threshold_orchestrator: int = Field(
        default=80, description="Number of tables to trigger summary mode in orchestrator"
    )  # CANDIDATE_SQL_AGENT__LARGE_CATALOG_TABLE_THRESHOLD_ORCHESTRATOR
    large_catalog_token_threshold: int = Field(
        default=60000, description="Estimated tokens to trigger summary mode"
    )  # CANDIDATE_SQL_AGENT__LARGE_CATALOG_TOKEN_THRESHOLD
    avg_columns_per_table_estimate: int = Field(
        default=15, description="Conservative estimate for token calculation"
    )  # CANDIDATE_SQL_AGENT__AVG_COLUMNS_PER_TABLE_ESTIMATE
    chars_per_column_line_estimate: int = Field(
        default=100, description="Approximate characters per column line"
    )  # CANDIDATE_SQL_AGENT__CHARS_PER_COLUMN_LINE_ESTIMATE

    # Schema linking settings
    enable_schema_linking: bool = Field(
        default=True, description="Enable schema linking to filter relevant tables/columns before planning"
    )  # CANDIDATE_SQL_AGENT__ENABLE_SCHEMA_LINKING
    # Multi-path generation settings
    enable_multi_path_generation: bool = Field(
        default=True, description="Enable multi-path SQL candidate generation with selection"
    )  # CANDIDATE_SQL_AGENT__ENABLE_MULTI_PATH_GENERATION

    use_native_query_mode: bool = Field(
        default=True, description="Whether to use native query mode"
    )  # CANDIDATE_SQL_AGENT__USE_NATIVE_QUERY_MODE
    native_query_mode_supported_engines: list[str] = Field(
        default=["snowflake"], description="The engines that support native query mode"
    )  # CANDIDATE_SQL_AGENT__NATIVE_QUERY_MODE_SUPPORTED_ENGINES
