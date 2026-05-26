import json
import uuid
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from minds.common.constants import (
    CONTEXT_FIELD_ORGANIZATION_ID,
    CONTEXT_FIELD_REQUEST_ID,
    CONTEXT_FIELD_USER_ID,
    HEADER_BILLING_PERIOD_END,
    HEADER_BILLING_PERIOD_START,
    HEADER_LANGFUSE_METADATA,
    HEADER_LANGFUSE_SESSION_ID,
    HEADER_LANGFUSE_TAGS,
    HEADER_ORGANIZATION_ID,
    HEADER_USER_EMAIL,
    HEADER_USER_ID,
    HEADER_USER_ROLES,
)
from minds.common.logger import get_logger
from minds.common.settings.app_settings import get_app_settings

# Set up logging
logger = get_logger(__name__)
settings = get_app_settings()


class Context(BaseModel):
    """
    Context for the application.
    """

    request_id: UUID = Field(default=UUID("00000000-0000-0000-0000-000000000000"), description="The request ID")
    user_id: UUID = Field(default=UUID("00000000-0000-0000-0000-000000000000"), description="The user ID")
    organization_id: UUID = Field(
        default=UUID("00000000-0000-0000-0000-000000000000"), description="The organization ID"
    )
    user_email: str = Field(default="", description="The user email")
    user_roles: list[str] = Field(default=[], description="The user roles")
    billing_cycle_start: datetime | None = Field(
        default=None, description="Start of the current billing period for monthly usage calculations"
    )
    billing_cycle_end: datetime | None = Field(
        default=None, description="End of the current billing period for monthly usage calculations"
    )
    # Langfuse-proxy convention headers (Langfuse-Session-Id / Langfuse-Tags /
    # Langfuse-Metadata) emitted by upstream harnesses (e.g. anton-core/cowork)
    # to group multi-step requests under a single Langfuse Session and to name
    # traces after the originating harness + turn. All three are optional and
    # default to empty values when absent.
    langfuse_session_id: str | None = Field(
        default=None,
        description="Conversation id, used as Langfuse session_id to group multi-step traces.",
    )
    langfuse_tags: list[str] = Field(
        default_factory=list,
        description="Comma-split tags from the Langfuse-Tags header, merged into the trace's tags.",
    )
    langfuse_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed Langfuse-Metadata JSON object. Typically carries turn_id and harness.",
    )


def extract_context_from_request(request: Request) -> Context:
    """
    Extract the context from the request headers.
    """

    if request.headers.get(HEADER_USER_ID) is None or request.headers.get(HEADER_ORGANIZATION_ID) is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    logger.debug(f"Request headers: {request.headers}")

    x_user_id = str(request.headers.get(HEADER_USER_ID))
    x_organization_id = str(request.headers.get(HEADER_ORGANIZATION_ID))
    x_user_email = str(request.headers.get(HEADER_USER_EMAIL))
    x_user_roles = str(request.headers.get(HEADER_USER_ROLES))

    user_id = UUID(x_user_id)
    organization_id = UUID(x_organization_id)
    user_email = x_user_email
    user_roles = x_user_roles.split(",") if x_user_roles else []

    billing_cycle_start: datetime | None = None
    billing_cycle_end: datetime | None = None
    x_billing_period_start = str(request.headers.get(HEADER_BILLING_PERIOD_START))
    x_billing_period_end = str(request.headers.get(HEADER_BILLING_PERIOD_END))

    logger.debug(f"Billing period start: {x_billing_period_start}")
    logger.debug(f"Billing period end: {x_billing_period_end}")

    if x_billing_period_start:
        try:
            billing_cycle_start = datetime.fromisoformat(x_billing_period_start.strip())
            logger.debug(f"Billing cycle start: {billing_cycle_start}")
        except (ValueError, TypeError):
            logger.warning(f"Invalid {HEADER_BILLING_PERIOD_START} header value: {x_billing_period_start!r}")

    if x_billing_period_end:
        try:
            billing_cycle_end = datetime.fromisoformat(x_billing_period_end.strip())
            logger.debug(f"Billing cycle end: {billing_cycle_end}")
        except (ValueError, TypeError):
            logger.warning(f"Invalid {HEADER_BILLING_PERIOD_END} header value: {x_billing_period_end!r}")

    request_id = uuid.uuid4()

    langfuse_session_id, langfuse_tags, langfuse_metadata = _extract_langfuse_headers(request)

    logger.debug(
        f"Extracted context from request: request_id={request_id}, user_id={user_id}, "
        f"organization_id={organization_id}, user_email={user_email}, user_roles={user_roles}, "
        f"billing_cycle_start={billing_cycle_start}, billing_cycle_end={billing_cycle_end}, "
        f"langfuse_session_id={langfuse_session_id}, langfuse_tags={langfuse_tags}, "
        f"langfuse_metadata_keys={list(langfuse_metadata.keys())}"
    )

    return Context(
        user_id=user_id,
        organization_id=organization_id,
        request_id=request_id,
        user_email=user_email,
        user_roles=user_roles,
        billing_cycle_start=billing_cycle_start,
        billing_cycle_end=billing_cycle_end,
        langfuse_session_id=langfuse_session_id,
        langfuse_tags=langfuse_tags,
        langfuse_metadata=langfuse_metadata,
    )


def _extract_langfuse_headers(request: Request) -> tuple[str | None, list[str], dict[str, Any]]:
    """Parse the three Langfuse-proxy convention headers off ``request``.

    Returns ``(session_id, tags, metadata)``. Any of them is optional:
    - ``Langfuse-Session-Id`` absent or empty → ``None``.
    - ``Langfuse-Tags`` absent → ``[]``; otherwise comma-split, whitespace-stripped,
      with empty entries dropped.
    - ``Langfuse-Metadata`` absent → ``{}``; otherwise JSON-parsed. Non-dict or
      malformed JSON is logged at warning level and treated as ``{}`` so a
      typo in the client doesn't take down the trace pipeline.
    """
    raw_session = request.headers.get(HEADER_LANGFUSE_SESSION_ID)
    session_id = raw_session.strip() if isinstance(raw_session, str) and raw_session.strip() else None

    raw_tags = request.headers.get(HEADER_LANGFUSE_TAGS) or ""
    tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

    raw_metadata = request.headers.get(HEADER_LANGFUSE_METADATA) or ""
    metadata: dict[str, Any] = {}
    if raw_metadata.strip():
        try:
            parsed = json.loads(raw_metadata)
            if isinstance(parsed, dict):
                metadata = parsed
            else:
                logger.warning(
                    f"{HEADER_LANGFUSE_METADATA} header was not a JSON object (got {type(parsed).__name__}); ignoring"
                )
        except (TypeError, ValueError) as exc:
            logger.warning(f"Malformed {HEADER_LANGFUSE_METADATA} header, ignoring: {exc}")

    return session_id, tags, metadata


class LangfuseContextMetadata(Context):
    """
    Metadata for the Langfuse context.
    Attributes:
        request_id: UUID
        user_id: UUID
        organization_id: UUID
    """

    pass


class LangfuseContext(BaseModel):
    """
    Context for Langfuse, including user and metadata.
    Attributes:
        user_id: str
        metadata: LangfuseContextMetadata
        tags: list[str]
        session_id: optional Langfuse session id (groups multi-turn traces).
        trace_name: optional override for the trace's display name
            (e.g. ``"cowork:turn-3"`` when the client passes ``harness``+``turn_id``).
        extra_metadata: free-form metadata blob from the ``Langfuse-Metadata``
            header, merged into the trace's metadata alongside ``metadata``.
    """

    user_id: UUID = UUID("00000000-0000-0000-0000-000000000000")
    metadata: LangfuseContextMetadata = LangfuseContextMetadata()
    tags: list[Any] = []
    trace_id: str | None = None
    session_id: str | None = None
    trace_name: str | None = None
    extra_metadata: dict[str, Any] = {}


def create_langfuse_context(context: Context) -> LangfuseContext:
    """
    Create a Langfuse context from request context.

    Per-request identity tags (user / org / env / request_id / email) are
    always emitted. Client-supplied ``Langfuse-Tags`` are appended so a
    harness like cowork ends up filterable as ``tag = cowork`` alongside
    our internal identity tags. ``Langfuse-Metadata`` is preserved verbatim
    in ``extra_metadata`` so the trace setup can merge it into the trace's
    metadata; we additionally derive a ``trace_name`` from ``harness`` +
    ``turn_id`` when both are present, since that's the dashboard-scanning
    affordance documented in the integration guide.
    """
    identity_tags = [
        f"{CONTEXT_FIELD_USER_ID}:{context.user_id}",
        f"{CONTEXT_FIELD_ORGANIZATION_ID}:{context.organization_id}",
        settings.env,
        f"{CONTEXT_FIELD_REQUEST_ID}:{context.request_id}",
        context.user_email,
    ]
    # De-dupe while preserving order so a client retransmitting a tag we
    # already set doesn't double it.
    merged_tags: list[str] = []
    seen_tags: set[str] = set()
    for t in (*identity_tags, *context.langfuse_tags):
        if t in seen_tags:
            continue
        seen_tags.add(t)
        merged_tags.append(t)

    harness = context.langfuse_metadata.get("harness") if isinstance(context.langfuse_metadata, dict) else None
    turn_id = context.langfuse_metadata.get("turn_id") if isinstance(context.langfuse_metadata, dict) else None
    trace_name: str | None = None
    if harness and turn_id is not None:
        trace_name = f"{harness}:turn-{turn_id}"

    return LangfuseContext(
        user_id=context.user_id,
        metadata=LangfuseContextMetadata(
            user_id=context.user_id,
            organization_id=context.organization_id,
            request_id=context.request_id,
            user_email=context.user_email,
            user_roles=context.user_roles,
        ),
        tags=merged_tags,
        session_id=context.langfuse_session_id,
        trace_name=trace_name,
        extra_metadata=dict(context.langfuse_metadata) if isinstance(context.langfuse_metadata, dict) else {},
    )
