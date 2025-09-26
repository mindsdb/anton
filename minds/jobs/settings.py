from functools import lru_cache
import os
import json

from prefect.blocks.system import Secret
from pydantic import BaseModel

from minds.common.vars import DATABASE_URI


_SETTINGS_PREFIX = "prefect-flow-settings"

_ENV = os.environ.get("ENV", "local").lower()


def _block_name(env: str) -> str:
	return f"{env}--{_SETTINGS_PREFIX}"


class PrefectSettings(BaseModel):
    database_uri: str = DATABASE_URI


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
