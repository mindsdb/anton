"""
Datasource management endpoints for API v1.

This module contains endpoints for CRUD operations on datasources,
providing a clean v1 API interface for datasource management.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlmodel import Session

from minds.client.mindsdb import create_mindsdb_client_from_request
from minds.common.logger import setup_logging
from minds.db.pg_session import get_session
from minds.requests.context import extract_context_from_request
from minds.schemas.datasources import (
    DatasourceConnectionStatus,
    DatasourceCreateRequest,
    DatasourceDetailedResponse,
    DatasourceResponse,
    DatasourceUpdateRequest,
)
from minds.services.datasources import (
    DatasourceAlreadyExistsError,
    DatasourceNotFoundError,
    DatasourceServiceError,
    DatasourcesService,
)

# Set up logging
logger = setup_logging()

router = APIRouter()


def get_datasources_service(request: Request, session: Session = Depends(get_session)) -> DatasourcesService:
    """
    Dependency function to create DatasourcesService with user context and MindsDB client.
    """
    context = extract_context_from_request(request)
    mindsdb_client = create_mindsdb_client_from_request(request)

    return DatasourcesService(session=session, mindsdb_client=mindsdb_client, user_id=context.user_id)


@router.get("/", status_code=200)
async def list_datasources(
    datasources_service: DatasourcesService = Depends(get_datasources_service),
    # Optional query parameters for filtering and pagination
    engine: str | None = Query(None, description="Filter by database engine (postgres, mysql, etc.)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results (1-1000)"),
    offset: int = Query(0, ge=0, description="Number of results to skip for pagination"),
    with_detailed_data: bool = Query(False, description="Include connection status and detailed information"),
) -> list[DatasourceResponse | DatasourceDetailedResponse]:
    """
    List all datasources for the authenticated user.

    Provides pagination and filtering capabilities for efficient datasource management.

    Args:
        engine: Filter by database engine type
        limit: Maximum number of results to return
        offset: Number of results to skip (for pagination)
        with_detailed_data: Include connection status and additional details

    Returns:
        List of datasource objects matching the specified criteria
    """
    try:
        logger.debug(f"List datasources requested (v1) for user {datasources_service.user_id} ")

        datasources = await datasources_service.list_datasources(
            engine=engine, limit=limit, offset=offset, with_detailed_data=with_detailed_data
        )

        return datasources

    except DatasourceServiceError as e:
        logger.error(f"Service error in list_datasources: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Unexpected error in list_datasources: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/{datasource_name}", status_code=200)
async def get_datasource(
    datasource_name: str,
    datasources_service: DatasourcesService = Depends(get_datasources_service),
    with_detailed_data: bool = Query(False, description="Include connection status and detailed information"),
) -> DatasourceResponse | DatasourceDetailedResponse:
    """
    Get a specific datasource by name.

    Args:
        datasource_name: Name of the datasource to retrieve
        with_detailed_data: Include connection status and additional details

    Returns:
        Datasource details object
    """
    try:
        logger.debug(f"Get datasource requested: {datasource_name} (v1) for user {datasources_service.user_id}")

        datasource = await datasources_service.get_datasource(
            datasource_name=datasource_name, with_detailed_data=with_detailed_data
        )

        return datasource

    except DatasourceServiceError as e:
        logger.error(f"Service error in get_datasource: {str(e)}")
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e)) from None
        else:
            raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Unexpected error in get_datasource: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.post("/", status_code=201)
async def create_datasource(
    datasource_data: DatasourceCreateRequest, datasources_service: DatasourcesService = Depends(get_datasources_service)
) -> DatasourceResponse:
    """
    Create a new datasource.

    Args:
        datasource_data: Datasource creation request data

    Returns:
        Created datasource details
    """
    try:
        logger.debug(
            f"Create datasource requested: {datasource_data.name} (v1) for user {datasources_service.user_id}, "
        )

        datasource = await datasources_service.create_datasource(datasource_data)

        return datasource

    except DatasourceAlreadyExistsError as e:
        logger.error(f"Datasource already exists: {str(e)}")
        raise HTTPException(status_code=409, detail=str(e)) from None
    except DatasourceServiceError as e:
        logger.error(f"Service error in create_datasource: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Unexpected error in create_datasource: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.put("/{datasource_name}", status_code=200)
async def update_datasource(
    datasource_name: str,
    datasource_data: DatasourceUpdateRequest,
    datasources_service: DatasourcesService = Depends(get_datasources_service),
) -> DatasourceResponse:
    """
    Update an existing datasource.

    Args:
        datasource_name: Name of the datasource to update
        datasource_data: Datasource update request data

    Returns:
        Updated datasource details
    """
    try:
        logger.debug(f"Update datasource requested: {datasource_name} (v1) for user {datasources_service.user_id}")

        datasource = await datasources_service.update_datasource(datasource_name, datasource_data)

        return datasource

    except DatasourceNotFoundError as e:
        logger.error(f"Datasource not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e)) from None
    except DatasourceServiceError as e:
        logger.error(f"Service error in update_datasource: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Unexpected error in update_datasource: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.delete("/{datasource_name}", status_code=204)
async def delete_datasource(
    datasource_name: str,
    datasources_service: DatasourcesService = Depends(get_datasources_service),
    cascade: bool = Query(False, description="Remove from all minds that use this datasource"),
):
    """
    Delete a datasource (hard delete to match MindsDB behavior).

    Args:
        datasource_name: Name of the datasource to delete
        cascade: Whether to remove from all minds that use it

    Returns:
        No content (204) on successful deletion
    """
    try:
        logger.debug(
            f"Delete datasource requested: {datasource_name} (v1) \
                for user {datasources_service.user_id}, cascade={cascade}"
        )

        await datasources_service.delete_datasource(datasource_name, cascade=cascade)

        # Return 204 No Content on successful deletion
        return None

    except DatasourceNotFoundError as e:
        logger.error(f"Datasource not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e)) from None
    except DatasourceServiceError as e:
        logger.error(f"Service error in delete_datasource: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Unexpected error in delete_datasource: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.post("/{datasource_name}/test-connection", status_code=200)
async def check_datasource_connection(
    datasource_name: str, datasources_service: DatasourcesService = Depends(get_datasources_service)
) -> DatasourceConnectionStatus:
    """
    Test connection to a datasource using MindsDB.

    Args:
        datasource_name: Name of the datasource to test

    Returns:
        Connection status with success/failure and error details
    """
    try:
        logger.debug(f"Test connection requested: {datasource_name} (v1) for user {datasources_service.user_id}")

        result = await datasources_service.test_connection(datasource_name)

        return result

    except Exception as e:
        logger.error(f"Unexpected error in check_datasource_connection: {str(e)}")
        return DatasourceConnectionStatus(success=False, error_message=f"Connection test failed: {str(e)}")
