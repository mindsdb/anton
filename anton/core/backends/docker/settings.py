from pydantic import Field
from pydantic_settings import BaseSettings


class DockerBackendSettings(BaseSettings):
    model_config = {"env_prefix": "ANTON_BACKEND_", "extra": "ignore"}

    docker_image: str = Field(default="anton-scratchpad:py312")
    docker_network: str | None = Field(default=None)
    docker_api_version: str = Field(default="1.44") 