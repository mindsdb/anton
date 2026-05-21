import re
from dataclasses import dataclass

from minds.common.settings.app_settings import get_app_settings

_PASSTHROUGH_PATTERN = re.compile(r"^_([a-zA-Z0-9]+)_$")

settings = get_app_settings()


@dataclass(frozen=True)
class PassthroughModelConfig:
    provider: str  # "openai" or "anthropic"
    model_name: str


def is_passthrough_model(model: str) -> bool:
    """Return True if *model* matches the ``_xxx_`` passthrough pattern."""
    return _PASSTHROUGH_PATTERN.match(model) is not None


def resolve_passthrough_model(model: str) -> PassthroughModelConfig:
    """Resolve a ``_xxx_`` model name to a concrete provider + model.

    Mapping logic:
    - If an Anthropic API key is configured:
        ``_reason_`` → claude-sonnet-4-6
        ``_code_``   → claude-haiku-4-5-20251001
    - Otherwise (fall back to OpenAI):
        ``_reason_`` → gpt-5.2
        ``_code_``   → gpt-5.1-codex
    - Unknown ``_{anything}_`` falls back to the ``_reason_`` mapping.
    """
    m = _PASSTHROUGH_PATTERN.match(model)
    if not m:
        raise ValueError(f"{model!r} is not a valid passthrough model name")

    alias = m.group(1).lower()

    has_anthropic = bool(settings.anthropic.api_key and settings.anthropic.api_key != "")

    if has_anthropic:
        mapping = {
            "reason": PassthroughModelConfig(provider="anthropic", model_name="claude-sonnet-4-6"),
            "code": PassthroughModelConfig(provider="anthropic", model_name="claude-haiku-4-5-20251001"),
        }
        default = mapping["reason"]
    else:
        mapping = {
            "reason": PassthroughModelConfig(provider="openai", model_name="gpt-5.2"),
            "code": PassthroughModelConfig(provider="openai", model_name="gpt-5.1-codex"),
        }
        default = mapping["reason"]

    return mapping.get(alias, default)
