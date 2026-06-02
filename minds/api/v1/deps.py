"""
Shared FastAPI dependency functions for API v1.

Provides factories for request-scoped dependencies:
- Context extraction (multi-tenancy headers)
- Database session
- Services (Usage, Limits, Conversations — inference-only)

All factories use Depends(...) so FastAPI caches a single Context per request.
"""

from fastapi import Depends, Request
from sqlmodel import Session

from minds.common.settings.app_settings import get_app_settings
from minds.db.pg_session import get_session
from minds.requests.context import Context, extract_context_from_request
from minds.services.conversations import ConversationsService
from minds.services.limits import LimitsService
from minds.services.usage import UsageService


def get_context(request: Request) -> Context:
    """Extract the authenticated user/org context from the incoming request."""
    return extract_context_from_request(request)


def get_usage_service(
    context: Context = Depends(get_context),
    session: Session = Depends(get_session),
) -> UsageService:
    """Create UsageService with user context."""
    return UsageService(session=session, context=context)


def get_conversations_service(
    context: Context = Depends(get_context),
    session: Session = Depends(get_session),
) -> ConversationsService:
    """Create ConversationsService with user context."""
    return ConversationsService(
        session=session,
        user_id=context.user_id,
        organization_id=context.organization_id,
    )


def get_limits_service(
    context: Context = Depends(get_context),
    usage_service: UsageService = Depends(get_usage_service),
) -> LimitsService:
    """Create LimitsService with usage tracking."""
    return LimitsService(
        usage_service=usage_service,
        context=context,
        settings=get_app_settings(),
    )
