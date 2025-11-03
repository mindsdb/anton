import json
import os
import socket
from functools import lru_cache
from urllib.parse import urlparse, urlunparse

from prefect.blocks.system import Secret
from pydantic import BaseModel, Field

_SETTINGS_PREFIX = "prefect-flow-settings"

_ENV = os.environ.get("ENV", "local").lower()


def _block_name(env: str) -> str:
    return f"{env}--{_SETTINGS_PREFIX}"


# TODO: Is this resoltuion really needed?
def _resolve_service_url(url: str) -> str:
    """
    Resolve service names in URLs to their actual IP addresses.
    For example: http://mindsdb:80 -> http://10.96.123.45:80
    """
    try:
        parsed = urlparse(url)
        if parsed.hostname and not parsed.hostname.replace(".", "").isdigit():
            # Only resolve if hostname is not already an IP address
            resolved_ip = socket.gethostbyname(parsed.hostname)
            resolved_url = urlunparse(
                (
                    parsed.scheme,
                    f"{resolved_ip}:{parsed.port}" if parsed.port else resolved_ip,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                )
            )
            return resolved_url
    except (socket.gaierror, socket.herror):
        # If resolution fails, return original URL
        pass
    return url


class PrefectSettings(BaseModel):
    database_uri: str = Field(
        default_factory=lambda: os.environ.get("DATABASE_URI", "postgresql://minds:minds@localhost:35432/minds")
    )
    mindsdb_url: str = Field(
        default_factory=lambda: _resolve_service_url(os.environ.get("MINDSDB_URL", "http://localhost:47334"))
    )
    mindsdb_api_key: str = Field(default_factory=lambda: os.environ.get("MINDSDB_API_KEY", ""))
    mindsdb_login: str = Field(default_factory=lambda: os.environ.get("MINDSDB_LOGIN", "mindsdb"))
    mindsdb_password: str = Field(default_factory=lambda: os.environ.get("MINDSDB_PASSWORD", ""))


def create_prefect_settings() -> None:
    """
    Create/update the Prefect Secret block from variables in the environment.
    Safe to call multiple times.
    """
    prefect_settings = PrefectSettings()
    Secret(value=json.dumps(prefect_settings.model_dump())).save(_block_name(_ENV), overwrite=True)


@lru_cache
def get_prefect_settings() -> PrefectSettings:
    """
    Load the Prefect Secret block; if missing, return default PrefectSettings.
    This allows decorators to work with defaults while still loading from secrets at runtime.
    """
    name = _block_name(_ENV)
    try:
        block = Secret.load(name)
        data = block.get()

        return PrefectSettings(**data)
    except Exception as e:
        # Return default settings if secret doesn't exist (useful during deployment)
        print(f"Warning: Could not load Prefect Secret block '{name}': {e}")
        return PrefectSettings()
