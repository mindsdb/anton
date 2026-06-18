"""
OpenAI-compatible models listing endpoint for API v1.

Backs the OpenAI Python SDK's ``client.models.list()``, which issues
``GET /v1/models`` and expects an OpenAI ``ListObject[Model]`` payload.

Only passthrough aliases whose upstream provider is actually configured
are advertised — matches the 400-on-missing-provider contract of
``ModelResolver.resolve``. Per-user Statsig policy (alias overrides /
allow-list) is threaded in via ``get_passthrough_model_config``. Deprecated
bare spellings (``_reason_``, ``_code_``) are intentionally hidden here.
"""

from fastapi import APIRouter, Depends

from minds.api.v1.deps import get_context
from minds.common.logger import get_logger
from minds.common.settings.app_settings import get_app_settings
from minds.common.statsig.dynamic_config.model_config import get_passthrough_model_config
from minds.inference.effort import get_effort_capability
from minds.inference.model_resolver import ModelResolver
from minds.requests.context import Context

logger = get_logger(__name__)

router = APIRouter()


# Registered for both ``/v1/models`` and ``/v1/models/``. The OpenAI SDK's
# ``client.models.list()`` issues ``GET /v1/models`` with no trailing slash;
# serving both paths directly avoids FastAPI's 307 redirect, which can drop
# auth headers behind some proxies/ingresses.
@router.get("")
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
    policy = get_passthrough_model_config(context)
    resolver = ModelResolver(get_app_settings())
    configs = resolver.list_available(policy=policy)
    effort_overrides = policy.effort_capabilities()

    data = []
    for cfg in configs:
        entry: dict = {
            "id": f"latest:{cfg.alias}",
            "object": "model",
            "created": 0,
            "owned_by": cfg.label or cfg.api_kind.value,
        }
        # Non-standard extension fields (OpenAI clients ignore unknown keys):
        # the discrete reasoning-effort levels this model accepts, in display
        # order, so a UI can render the right picker per model. Keyed off the
        # *concrete* model id, so per-user alias overrides are reflected.
        # Omitted entirely for models without effort support — a UI shows the
        # picker iff ``reasoning_efforts`` is present.
        capability = get_effort_capability(cfg.model_name, effort_overrides)
        if capability is not None and capability.supported:
            entry["reasoning_efforts"] = list(capability.levels)
            # Alias-pinned effort (gpt-low/medium/high) is what a request
            # without an explicit reasoning_effort actually gets, so it wins
            # over the model's provider-side default.
            default = cfg.reasoning_effort or capability.default
            if default:
                entry["default_reasoning_effort"] = default
        data.append(entry)

    return {"object": "list", "data": data}
