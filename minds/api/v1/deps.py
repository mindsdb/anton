"""
Shared FastAPI dependency functions for API v1.

This module centralises the construction of service objects so that
endpoint modules can simply import the dependency they need instead of
re-implementing the same context-extraction / client-creation boilerplate.

All factories accept their sub-dependencies via ``Depends(...)`` so that
FastAPI caches a single ``Context`` (and MindsDB client) per request,
avoiding duplicate header parsing and request-ID generation.
"""

from fastapi import Depends, HTTPException, Request
from sqlmodel import Session

from minds.client.mindsdb import create_mindsdb_client_from_request
from minds.common.settings.app_settings import get_app_settings
from minds.db.pg_session import get_session
from minds.requests.context import Context, extract_context_from_request
from minds.services.conversations import ConversationsService
from minds.services.data_catalog.data_catalog_loader import DataCatalogLoader
from minds.services.datasources import DatasourcesService
from minds.services.limits import LimitsService
from minds.services.memory import MemoryAdminService
from minds.services.minds import MindsService
from minds.services.tree import TreeService
from minds.services.usage import UsageService

# -- Low-level building blocks ------------------------------------------------


def get_context(request: Request) -> Context:
    """
    Extract the authenticated user/org context from the incoming request.

    Args:
        request: The incoming request.

    Returns:
        The context.
    """
    return extract_context_from_request(request)


def get_mindsdb_client(request: Request, context: Context = Depends(get_context)):
    """
    Create a MindsDB SDK client from the incoming request.

    The ``Request`` is still needed to extract the bearer token from the
    ``Authorization`` header; the ``Context`` is provided via dependency
    injection so it is shared with other per-request dependencies.

    Args:
        request: The incoming request (used for the bearer token).
        context: The request-scoped Context.

    Returns:
        The MindsDB SDK client.
    """
    return create_mindsdb_client_from_request(request, context)


# -- Service factories --------------------------------------------------------


def get_minds_service(
    context: Context = Depends(get_context),
    session: Session = Depends(get_session),
    mindsdb_client=Depends(get_mindsdb_client),
) -> MindsService:
    """
    Create MindsService with user context.

    Args:
        context: The request-scoped Context.
        session: The database session.
        mindsdb_client: The MindsDB SDK client.

    Returns:
        The MindsService.
    """
    return MindsService(
        session=session,
        mindsdb_client=mindsdb_client,
        user_id=context.user_id,
        organization_id=context.organization_id,
    )


def get_conversations_service(
    context: Context = Depends(get_context),
    session: Session = Depends(get_session),
    mindsdb_client=Depends(get_mindsdb_client),
) -> ConversationsService:
    """
    Create ConversationsService with user context.

    Args:
        context: The request-scoped Context.
        session: The database session.
        mindsdb_client: The MindsDB SDK client.

    Returns:
        The ConversationsService.
    """
    return ConversationsService(
        session=session,
        mindsdb_client=mindsdb_client,
        user_id=context.user_id,
        organization_id=context.organization_id,
    )


def get_datasources_service(
    context: Context = Depends(get_context),
    session: Session = Depends(get_session),
    mindsdb_client=Depends(get_mindsdb_client),
) -> DatasourcesService:
    """
    Create DatasourcesService with user context.

    Args:
        context: The request-scoped Context.
        session: The database session.
        mindsdb_client: The MindsDB SDK client.

    Returns:
        The DatasourcesService.
    """
    return DatasourcesService(
        session=session,
        mindsdb_client=mindsdb_client,
        user_id=context.user_id,
        organization_id=context.organization_id,
    )


def get_data_catalog_loader(
    context: Context = Depends(get_context),
    session: Session = Depends(get_session),
) -> DataCatalogLoader:
    """
    Create DataCatalogLoader with user context.

    Args:
        context: The request-scoped Context.
        session: The database session.

    Returns:
        The DataCatalogLoader.
    """
    return DataCatalogLoader(
        session=session,
        organization_id=context.organization_id,
        user_id=context.user_id,
    )


def get_tree_service(
    context: Context = Depends(get_context),
    mindsdb_client=Depends(get_mindsdb_client),
) -> TreeService:
    """
    Create TreeService with user context and MindsDB client.

    Args:
        context: The request-scoped Context.
        mindsdb_client: The MindsDB SDK client.

    Returns:
        The TreeService.
    """
    return TreeService(mindsdb_client=mindsdb_client, user_id=context.user_id)


def get_usage_service(
    context: Context = Depends(get_context),
    session: Session = Depends(get_session),
) -> UsageService:
    """
    Create UsageService with user context.

    Args:
        context: The request-scoped Context.
        session: The database session.

    Returns:
        The UsageService.
    """
    return UsageService(session=session, context=context)


def require_admin(context: Context = Depends(get_context)) -> None:
    """Raise 403 if the caller does not have an admin-level role."""
    admin_roles = {"admin", "sysadmin", "mindsdb-admin"}
    if not admin_roles.intersection(context.user_roles):
        raise HTTPException(status_code=403, detail="Admin role required")


def get_memory_admin_service(
    context: Context = Depends(get_context),
    session: Session = Depends(get_session),
) -> MemoryAdminService:
    return MemoryAdminService(
        session=session,
        user_id=context.user_id,
        organization_id=context.organization_id,
    )


def get_limits_service(
    context: Context = Depends(get_context),
    minds_service: MindsService = Depends(get_minds_service),
    datasources_service: DatasourcesService = Depends(get_datasources_service),
    usage_service: UsageService = Depends(get_usage_service),
) -> LimitsService:
    """
    Create LimitsService with all required sub-services.

    Args:
        context: The request-scoped Context.
        minds_service: The MindsService dependency.
        datasources_service: The DatasourcesService dependency.
        usage_service: The UsageService dependency.

    Returns:
        The LimitsService.
    """
    return LimitsService(
        minds_service=minds_service,
        datasources_service=datasources_service,
        usage_service=usage_service,
        context=context,
        settings=get_app_settings(),
    )
