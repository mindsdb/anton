"""
Mind management endpoints for API v1.

This module contains endpoints for CRUD operations on minds (agents),
providing a clean v1 API interface for mind management.
"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from minds.api.v1.deps import get_conversations_service, get_data_catalog_loader, get_limits_service, get_minds_service
from minds.common.guards import ResourceType, require_usage_available
from minds.common.llm_provider import get_supported_models_by_provider
from minds.common.logger import setup_logging
from minds.schemas.minds import MindCreateRequest, MindResponse, MindUpdateRequest
from minds.services.conversations import ConversationsService
from minds.services.data_catalog.data_catalog_loader import DataCatalogLoader
from minds.services.limits import LimitsService
from minds.services.minds import MindAlreadyExistsError, MindNotFoundError, MindsService, MindsServiceError

# Set up logging
logger = setup_logging()

router = APIRouter()


@router.get("/supported-models")
async def get_supported_models(
    minds_service: MindsService = Depends(get_minds_service),
) -> dict[str, bool | str | str | dict[str, list[str]]]:
    """
    Get the supported models for the current user/company.
    """
    logger.debug(
        f"Get supported models requested for user {minds_service.user_id} in "
        f"organization {minds_service.organization_id}"
    )
    is_model_selection_enabled, default_provider, default_model, providers_and_models = (
        get_supported_models_by_provider()
    )
    return {
        "is_model_selection_enabled": is_model_selection_enabled,
        "default_provider": default_provider,
        "default_model": default_model,
        "providers": providers_and_models,
    }


@router.get("/")
async def list_minds(
    minds_service: MindsService = Depends(get_minds_service),
    conversations_service: ConversationsService = Depends(get_conversations_service),
    # Optional query parameters for filtering and pagination
    name: str | None = Query(None, description="Filter by mind name"),
    provider: str | None = Query(None, description="Filter by provider (openai, google, etc.)"),
    is_sample: bool | None = Query(None, description="Filter by sample status"),
    include_deleted: bool = Query(False, description="Filter by deleted status"),
    limit: int = Query(50, le=100, ge=1, description="Maximum number of minds to return"),
    offset: int = Query(0, ge=0, description="Number of minds to skip for pagination"),
    with_detailed_data: bool = Query(False, description="Include detailed datasource information"),
    include_total: bool = Query(False, description="Include total count of minds in response"),
    sort_by: Literal["name", "created_at", "updated_at", "provider", "model_name"] | None = Query(
        None, description="Field to sort by (name, created_at, updated_at, provider, model_name)"
    ),
    sort_order: Literal["asc", "desc"] = Query("desc", description="Sort order (asc or desc)"),
) -> list[MindResponse] | dict[str, list[MindResponse] | int]:
    """
    List minds for the authenticated user with optional filtering, pagination, sorting, and total count.

    Query Parameters:
        - provider: Filter by AI provider (openai, google, etc.)
        - include_deleted: Filter by deleted status (true/false)
        - is_sample: Filter by sample status (true/false)
        - limit: Maximum number of minds to return (1-100, default: 50)
        - offset: Number of minds to skip for pagination (default: 0)
        - include_total: Include total count of minds in response (default: false)
        - sort_by: Field to sort by (name, created_at, updated_at, provider, model_name).
          Defaults to created_at if not specified.
        - sort_order: Sort order (asc or desc, default: desc)

    Returns:
        List[MindResponse] or dict with 'minds' and 'total': List of mind objects, optionally with total count
    """
    logger.debug(
        f"List minds requested (v1) for user {minds_service.user_id} in organization {minds_service.organization_id}"
    )

    try:
        result = await minds_service.list_minds(
            conversations_service=conversations_service,
            name=name,
            provider=provider,
            is_sample=is_sample,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
            with_detailed_data=with_detailed_data,
            include_total=include_total,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        if include_total:
            minds, total = result
            logger.info(
                f"Listed {len(minds)} minds (total: {total}) "
                f"for user {minds_service.user_id} in "
                f"organization {minds_service.organization_id} (offset={offset}, limit={limit})"
            )
            return {"minds": minds, "total": total}
        else:
            logger.info(
                f"Listed minds for user {minds_service.user_id} in organization {minds_service.organization_id}"
            )
            return result
    except MindsServiceError as e:
        logger.error(
            f"Service error listing minds for user {minds_service.user_id} in "
            f"organization {minds_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error listing minds for user {minds_service.user_id} in "
            f"organization {minds_service.organization_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/{mind_name}")
async def get_mind(
    mind_name: str,
    minds_service: MindsService = Depends(get_minds_service),
    conversations_service: ConversationsService = Depends(get_conversations_service),
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
        f"Get mind requested: {mind_name} (v1) for user {minds_service.user_id} in "
        f"organization {minds_service.organization_id}"
    )

    try:
        mind = await minds_service.get_mind(
            mind_name=mind_name, conversations_service=conversations_service, with_detailed_data=with_detailed_data
        )
        logger.info(
            f"Retrieved mind {mind_name} for user {minds_service.user_id} in "
            f"organization {minds_service.organization_id}"
        )
        return mind
    except MindNotFoundError as e:
        logger.warning(
            f"Mind not found for user {minds_service.user_id} in organization {minds_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MindsServiceError as e:
        logger.error(
            f"Service error getting mind for user {minds_service.user_id} in "
            f"organization {minds_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error getting mind {mind_name} "
            f"for user {minds_service.user_id} in organization {minds_service.organization_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.head("/{mind_name}", status_code=200)
async def check_mind_exists(
    mind_name: str,
    minds_service: MindsService = Depends(get_minds_service),
) -> None:
    """
    Check if a specific mind exists by name for the authenticated user.

    Args:
        mind_name (str): Name of the mind to check

    Returns:
        None: 200 OK if mind exists, 404 Not Found if it does not
    """
    logger.debug(
        f"Check mind existence requested: {mind_name} (v1) for "
        f"user {minds_service.user_id} in organization {minds_service.organization_id}"
    )

    try:
        await minds_service.check_mind_exists(mind_name=mind_name)
    except MindNotFoundError as e:
        logger.warning(
            f"Mind not found for user {minds_service.user_id} in organization {minds_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MindsServiceError as e:
        logger.error(
            f"Service error checking mind for user {minds_service.user_id} in "
            f"organization {minds_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error checking mind {mind_name} "
            f"for user {minds_service.user_id} in organization {minds_service.organization_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.post("/", status_code=201)
async def create_mind(
    mind_data: MindCreateRequest,
    minds_service: MindsService = Depends(get_minds_service),
    data_catalog_loader: DataCatalogLoader = Depends(get_data_catalog_loader),
    limits_service: LimitsService = Depends(get_limits_service),
) -> MindResponse:
    """
    Create a new mind for the authenticated user.

    Args:
        mind_data (MindCreateRequest): Mind creation data including name, provider, model, etc.

    Returns:
        MindResponse: Created mind object with generated ID and timestamps

    Raises:
        HTTPException: 429 if usage limit exceeded.
    """
    logger.debug(
        f"Create mind requested: {mind_data.name} (v1) "
        f"for user {minds_service.user_id} in organization {minds_service.organization_id}"
    )

    # Check usage limits before creating
    await require_usage_available(limits_service, ResourceType.MINDS)

    try:
        mind = await minds_service.create_mind(mind_data, data_catalog_loader)
        logger.info(f"Created mind {mind_data.name} for user {minds_service.user_id}")
        return mind
    except MindAlreadyExistsError as e:
        logger.warning(
            f"Mind already exists for user {minds_service.user_id} in organization {minds_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=409, detail=str(e)) from None
    except MindsServiceError as e:
        logger.error(
            f"Service error creating mind for user {minds_service.user_id} in "
            f"organization {minds_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error creating mind {mind_data.name} "
            f"for user {minds_service.user_id} in organization {minds_service.organization_id}: {e}",
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
        f"Update mind requested: {mind_name} (v1) for user {minds_service.user_id} in "
        f"organization {minds_service.organization_id}"
    )

    try:
        mind = await minds_service.update_mind(mind_name, mind_data, data_catalog_loader)
        logger.info(
            f"Updated mind {mind_name} for user {minds_service.user_id} in organization {minds_service.organization_id}"
        )
        return mind

    except MindNotFoundError as e:
        logger.warning(
            f"Mind not found for update for user {minds_service.user_id} in "
            f"organization {minds_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MindsServiceError as e:
        logger.error(
            f"Service error updating mind for user {minds_service.user_id} in "
            f"organization {minds_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error updating mind {mind_name} "
            f"for user {minds_service.user_id} in organization {minds_service.organization_id}: {e}",
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
        f"Delete mind requested: {mind_name} (v1) for user {minds_service.user_id} in "
        f"organization {minds_service.organization_id}"
    )

    try:
        await minds_service.delete_mind(mind_name)
        logger.info(
            f"Deleted mind {mind_name} for user {minds_service.user_id} in organization {minds_service.organization_id}"
        )
        # Return nothing for 204 No Content

    except MindNotFoundError as e:
        logger.warning(
            f"Mind not found for deletion for user {minds_service.user_id} in "
            f"organization {minds_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MindsServiceError as e:
        logger.error(
            f"Service error deleting mind for user {minds_service.user_id} in "
            f"organization {minds_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error deleting mind {mind_name} "
            f"for user {minds_service.user_id} in organization {minds_service.organization_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None
