"""
OpenAI-compatible models listing endpoint for API v1.

Backs the OpenAI Python SDK's ``client.models.list()``, which issues
``GET /v1/models`` and expects an OpenAI ``ListObject[Model]`` payload.

Only passthrough aliases whose upstream provider is actually configured
are advertised — matches the 400-on-missing-provider contract of
``resolve_passthrough_model``. Deprecated bare spellings (``_reason_``,
``_code_``) are intentionally hidden from this listing.
"""

from fastapi import APIRouter, Depends

from minds.api.v1.deps import get_context
from minds.common.logger import get_logger
from minds.common.settings.app_settings import get_app_settings
from minds.inference.model_resolver import ModelResolver
from minds.requests.context import Context

logger = get_logger(__name__)

router = APIRouter()


@router.get("/")
async def list_models(context: Context = Depends(get_context)) -> dict:
    """
    List currently-callable passthrough models in OpenAI format.

    Returns:
        dict: ``{"object": "list", "data": [{"id": ..., "object": "model",
        "created": 0, "owned_by": ...}, ...]}`` — the shape the OpenAI SDK
        deserializes into ``ListObject[Model]``.
    """
    logger.debug(f"List models requested for user {context.user_id}")
    resolver = ModelResolver(get_app_settings())
    configs = resolver.list_available()
    return {
        "object": "list",
        "data": [
            {
                "id": f"latest:{cfg.alias}",
                "object": "model",
                "created": 0,
                "owned_by": cfg.label or cfg.api_kind.value,
            }
            for cfg in configs
        ],
    }
