"""
Mind management endpoints for API v1.

This module contains endpoints for CRUD operations on minds (agents),
providing a clean v1 API interface for mind management.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlmodel import Session

from minds.client.mindsdb import create_mindsdb_client_from_request
from minds.common.logger import setup_logging
from minds.db.pg_session import get_session
from minds.requests.context import extract_context_from_request
from minds.schemas.minds import MindCreateRequest, MindResponse, MindUpdateRequest
from minds.services.data_catalog.data_catalog_loader import DataCatalogLoader
from minds.services.minds import MindAlreadyExistsError, MindNotFoundError, MindsService, MindsServiceError

# Set up logging
logger = setup_logging()

router = APIRouter()


def get_minds_service(request: Request, session: Session = Depends(get_session)) -> MindsService:
    """
    Dependency function to create MindsService with user context.
    """
    context = extract_context_from_request(request)
    mindsdb_client = create_mindsdb_client_from_request(request)
    return MindsService(
        session=session,
        mindsdb_client=mindsdb_client,
        user_id=context.user_id,
        tenant_id=context.tenant_id,
    )


def get_data_catalog_loader(request: Request, session: Session = Depends(get_session)) -> DataCatalogLoader:
    """
    Dependency function to create DataCatalogLoader.
    No parameters needed - it's a stateless service.
    """
    context = extract_context_from_request(request)
    return DataCatalogLoader(session=session, tenant_id=context.tenant_id)


@router.get("/")
async def list_minds(
    minds_service: MindsService = Depends(get_minds_service),
    # Optional query parameters for filtering and pagination
    provider: str | None = Query(None, description="Filter by provider (openai, google, etc.)"),
    include_deleted: bool = Query(False, description="Filter by deleted status"),
    limit: int = Query(50, le=100, ge=1, description="Maximum number of minds to return"),
    offset: int = Query(0, ge=0, description="Number of minds to skip for pagination"),
) -> list[MindResponse]:
    """
    List minds for the authenticated user with optional filtering and pagination.

    Query Parameters:
        - provider: Filter by AI provider (openai, google, etc.)
        - include_deleted: Filter by deleted status (true/false)
        - limit: Maximum number of minds to return (1-100, default: 50)
        - offset: Number of minds to skip for pagination (default: 0)

    Returns:
        List[MindResponse]: List of mind objects
    """
    logger.debug(f"List minds requested (v1) for user {minds_service.user_id} in tenant {minds_service.tenant_id}")

    try:
        minds = await minds_service.list_minds(
            provider=provider, include_deleted=include_deleted, limit=limit, offset=offset
        )

        logger.info(f"Listed minds for user {minds_service.user_id} in tenant {minds_service.tenant_id}")
        return minds
    except MindsServiceError as e:
        logger.error(
            f"Service error listing minds for user {minds_service.user_id} in tenant {minds_service.tenant_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error listing minds for user {minds_service.user_id} in tenant {minds_service.tenant_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/{mind_name}")
async def get_mind(
    mind_name: str,
    minds_service: MindsService = Depends(get_minds_service),
    with_detailed_data: bool = Query(False, description="Include detailed datasource information"),
) -> MindResponse:
    """
    Get a specific mind by name for the authenticated user.

    Args:
        mind_name (str): Name of the mind to retrieve
        with_detailed_data (bool): Include detailed datasource information

    Returns:
        MindResponse: Mind object with full details
    """
    logger.debug(
        f"Get mind requested: {mind_name} (v1) for user {minds_service.user_id} in tenant {minds_service.tenant_id}"
    )

    try:
        mind = await minds_service.get_mind(mind_name=mind_name, with_detailed_data=with_detailed_data)
        logger.info(f"Retrieved mind {mind_name} for user {minds_service.user_id} in tenant {minds_service.tenant_id}")
        return mind
    except MindNotFoundError as e:
        logger.warning(f"Mind not found for user {minds_service.user_id} in tenant {minds_service.tenant_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MindsServiceError as e:
        logger.error(
            f"Service error getting mind for user {minds_service.user_id} in tenant {minds_service.tenant_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error getting mind {mind_name} "
            f"for user {minds_service.user_id} in tenant {minds_service.tenant_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.post("/", status_code=201)
async def create_mind(
    mind_data: MindCreateRequest,
    minds_service: MindsService = Depends(get_minds_service),
    data_catalog_loader: DataCatalogLoader = Depends(get_data_catalog_loader),
) -> MindResponse:
    """
    Create a new mind for the authenticated user.

    Args:
        mind_data (MindCreateRequest): Mind creation data including name, provider, model, etc.

    Returns:
        MindResponse: Created mind object with generated ID and timestamps
    """
    logger.debug(
        f"Create mind requested: {mind_data.name} (v1) "
        f"for user {minds_service.user_id} in tenant {minds_service.tenant_id}"
    )

    try:
        mind = await minds_service.create_mind(mind_data, data_catalog_loader)
        logger.info(f"Created mind {mind_data.name} for user {minds_service.user_id}")
        return mind
    except MindAlreadyExistsError as e:
        logger.warning(f"Mind already exists for user {minds_service.user_id} in tenant {minds_service.tenant_id}: {e}")
        raise HTTPException(status_code=409, detail=str(e)) from None
    except MindsServiceError as e:
        logger.error(
            f"Service error creating mind for user {minds_service.user_id} in tenant {minds_service.tenant_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error creating mind {mind_data.name} "
            f"for user {minds_service.user_id} in tenant {minds_service.tenant_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.put("/{mind_name}")
async def update_mind(
    mind_name: str,
    mind_data: MindUpdateRequest,
    minds_service: MindsService = Depends(get_minds_service),
    data_catalog_loader: DataCatalogLoader = Depends(get_data_catalog_loader),
) -> MindResponse:
    """
    Update an existing mind for the authenticated user.

    Args:
        mind_name (str): Name of the mind to update
        mind_data (MindUpdateRequest): Updated mind data (only provided fields will be updated)

    Returns:
        MindResponse: Updated mind object with new values
    """
    logger.debug(
        f"Update mind requested: {mind_name} (v1) for user {minds_service.user_id} in tenant {minds_service.tenant_id}"
    )

    try:
        mind = await minds_service.update_mind(mind_name, mind_data, data_catalog_loader)
        logger.info(f"Updated mind {mind_name} for user {minds_service.user_id} in tenant {minds_service.tenant_id}")
        return mind

    except MindNotFoundError as e:
        logger.warning(
            f"Mind not found for update for user {minds_service.user_id} in tenant {minds_service.tenant_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MindsServiceError as e:
        logger.error(
            f"Service error updating mind for user {minds_service.user_id} in tenant {minds_service.tenant_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error updating mind {mind_name} "
            f"for user {minds_service.user_id} in tenant {minds_service.tenant_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.delete("/{mind_name}", status_code=204)
async def delete_mind(mind_name: str, minds_service: MindsService = Depends(get_minds_service)):
    """
    Delete a mind for the authenticated user (soft delete - marks as inactive).

    Args:
        mind_name (str): Name of the mind to delete

    Returns:
        None: 204 No Content on successful deletion
    """
    logger.debug(
        f"Delete mind requested: {mind_name} (v1) for user {minds_service.user_id} in tenant {minds_service.tenant_id}"
    )

    try:
        await minds_service.delete_mind(mind_name)
        logger.info(f"Deleted mind {mind_name} for user {minds_service.user_id} in tenant {minds_service.tenant_id}")
        # Return nothing for 204 No Content

    except MindNotFoundError as e:
        logger.warning(
            f"Mind not found for deletion for user {minds_service.user_id} in tenant {minds_service.tenant_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MindsServiceError as e:
        logger.error(
            f"Service error deleting mind for user {minds_service.user_id} in tenant {minds_service.tenant_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error deleting mind {mind_name} "
            f"for user {minds_service.user_id} in tenant {minds_service.tenant_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None
