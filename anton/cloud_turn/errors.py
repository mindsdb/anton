"""Typed cloud-turn errors + central exception → :class:`TurnErrorV1` mapping.

One place decides the wire ``code``/``retryable`` for every failure, so the
mapping can't drift. Wire messages are short and credential-scrubbed; full
tracebacks stay on stderr (the runner logs them via ``logger.exception``).
"""

from __future__ import annotations

from anton.cloud_turn.protocol import ErrorCodeV1, TurnErrorV1
from anton.utils.datasources import scrub_credentials

#: Wire error messages are truncated to keep them short and log-safe.
MAX_ERROR_MESSAGE_CHARS = 300


class CloudTurnError(Exception):
    """Base for failures the runner raises itself. Each subclass pins a stable
    wire code + retryable flag."""

    code: ErrorCodeV1 = ErrorCodeV1.INTERNAL_TURN_FAILURE
    retryable: bool = False


class InvalidRequestError(CloudTurnError):
    code = ErrorCodeV1.INVALID_REQUEST
    retryable = False


class UnsupportedProtocolVersionError(CloudTurnError):
    code = ErrorCodeV1.UNSUPPORTED_PROTOCOL_VERSION
    retryable = False


class UnsupportedCapabilityError(CloudTurnError):
    code = ErrorCodeV1.UNSUPPORTED_CAPABILITY
    retryable = False


class UnsupportedModelError(CloudTurnError):
    code = ErrorCodeV1.UNSUPPORTED_MODEL
    retryable = False


class DeadlineExceededError(CloudTurnError):
    code = ErrorCodeV1.DEADLINE_EXCEEDED
    retryable = True


def _clean(exc: Exception) -> str:
    text = scrub_credentials(f"{type(exc).__name__}: {exc}")
    if len(text) > MAX_ERROR_MESSAGE_CHARS:
        text = text[: MAX_ERROR_MESSAGE_CHARS - 1] + "…"
    return text


def _looks_like_auth(message: str) -> bool:
    low = message.lower()
    return any(s in low for s in ("api key", "unauthorized", "authentication", "401"))


def classify_error(exc: Exception) -> TurnErrorV1:
    """Map any exception onto a structured :class:`TurnErrorV1`.

    Order matters: several Anton provider errors subclass ``ConnectionError``,
    so the specific types are checked before the generic one.
    """
    # Errors the runner raises itself already carry code + retryable.
    if isinstance(exc, CloudTurnError):
        return TurnErrorV1(code=exc.code, message=_clean(exc), retryable=exc.retryable)

    # Anton provider/LLM exceptions (imported lazily so importing this module
    # never drags in the provider stack).
    from anton.core.llm.provider import (
        ContextOverflowError,
        ModelUnavailableError,
        ProviderOverloadedError,
        TokenLimitExceeded,
        TransientProviderError,
    )

    if isinstance(exc, ModelUnavailableError):
        # Gateway rejected the model (plan tier / kill switch).
        return TurnErrorV1(
            code=ErrorCodeV1.UNSUPPORTED_MODEL, message=_clean(exc), retryable=False
        )
    if isinstance(exc, (TransientProviderError, ProviderOverloadedError)):
        return TurnErrorV1(
            code=ErrorCodeV1.MODEL_PROVIDER_FAILURE, message=_clean(exc), retryable=True
        )
    if isinstance(exc, (TokenLimitExceeded, ContextOverflowError)):
        # Quota / context-length — a bare retry of the identical request won't fix it.
        return TurnErrorV1(
            code=ErrorCodeV1.MODEL_PROVIDER_FAILURE, message=_clean(exc), retryable=False
        )
    if isinstance(exc, ConnectionError):
        if _looks_like_auth(str(exc)):
            return TurnErrorV1(
                code=ErrorCodeV1.MODEL_AUTH_FAILURE, message=_clean(exc), retryable=False
            )
        return TurnErrorV1(
            code=ErrorCodeV1.MODEL_PROVIDER_FAILURE, message=_clean(exc), retryable=True
        )

    return TurnErrorV1(
        code=ErrorCodeV1.INTERNAL_TURN_FAILURE, message=_clean(exc), retryable=False
    )
