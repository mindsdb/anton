from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AntonAgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        env_prefix="ANTON_AGENT__",
    )

    backend: str = Field(default="docker")  # ANTON_AGENT__BACKEND
    root_workspace_dir: str = Field(default="/tmp/anton")  # ANTON_AGENT__ROOT_WORKSPACE_DIR
    output_dir: str = Field(default="output")  # ANTON_AGENT__OUTPUT_DIR
    output_file_name: str = Field(default="report.html")  # ANTON_AGENT__OUTPUT_FILE_NAME
    scratchpad_timeout: int = Field(default=120)  # ANTON_AGENT__SCRATCHPAD_TIMEOUT
    max_tool_rounds: int = Field(default=25)  # ANTON_AGENT__MAX_TOOL_ROUNDS

    minds_internal_url: str = Field(default="http://host.docker.internal:8000")  # ANTON_AGENT__MINDS_INTERNAL_URL

    # Shared memory settings
    shared_memory_token_budget: int = Field(default=3000)  # ANTON_AGENT__SHARED_MEMORY_TOKEN_BUDGET
    shared_memory_max_topics: int = Field(default=5)  # ANTON_AGENT__SHARED_MEMORY_MAX_TOPICS

    # Scratchpad runtime settings
    docker_image: str = Field(default="anton-scratchpad:py312")  # ANTON_AGENT__DOCKER_IMAGE
    docker_network: str | None = Field(default=None)  # ANTON_AGENT__DOCKER_NETWORK
    docker_api_version: str = Field(default="1.44")  # ANTON_AGENT__DOCKER_API_VERSION
