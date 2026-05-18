"""
Datasource management endpoints for API v1.

This module contains endpoints for CRUD operations on datasources,
providing a clean v1 API interface for datasource management.
"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from minds.api.v1.deps import get_datasources_service, get_limits_service
from minds.common.guards import ResourceType, require_usage_available
from minds.common.logger import get_logger
from minds.schemas.datasources import (
    ColumnResponse,
    DataCatalogResponse,
    DatasourceConnectionStatus,
    DatasourceCreateRequest,
    DatasourceDetailedResponse,
    DatasourceQueryRequest,
    DatasourceQueryResponse,
    DatasourceResponse,
    DatasourceTableSampleResponse,
    DatasourceUpdateRequest,
    TableResponse,
    UpdateColumnDescriptionRequest,
    UpdateTableDescriptionRequest,
)
from minds.services.datasources import (
    DatasourceAlreadyExistsError,
    DatasourceNotFoundError,
    DatasourceServiceError,
    DatasourcesService,
    DatasourceTableColumnNotCatalogedError,
    DatasourceTableColumnNotFoundError,
    DatasourceTableNotCatalogedError,
    DatasourceTableNotFoundError,
    InvalidDatasourceQueryError,
)
from minds.services.limits import LimitsService

# Set up logging
logger = get_logger(__name__)

router = APIRouter()


@router.get("/", status_code=200)
async def list_datasources(
    datasources_service: DatasourcesService = Depends(get_datasources_service),
    # Optional query parameters for filtering and pagination
    name: str | None = Query(None, description="Filter by datasource name"),
    engine: str | None = Query(None, description="Filter by database engine (postgres, mysql, etc.)"),
    include_deleted: bool = Query(False, description="Include deleted datasources"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results (1-1000)"),
    offset: int = Query(0, ge=0, description="Number of results to skip for pagination"),
    with_detailed_data: bool = Query(False, description="Include connection status and detailed information"),
    include_total: bool = Query(False, description="Include total count of datasources in response"),
    sort_by: Literal["name", "created_at", "updated_at", "engine"] | None = Query(
        None, description="Field to sort by (name, created_at, updated_at, engine)"
    ),
    sort_order: Literal["asc", "desc"] = Query("desc", description="Sort order (asc or desc)"),
) -> (
    list[DatasourceResponse | DatasourceDetailedResponse]
    | dict[str, list[DatasourceResponse | DatasourceDetailedResponse] | int]
):
    """
    List all datasources for the authenticated user.

    Provides pagination and filtering capabilities for efficient datasource management.

    Args:
        name: Filter by datasource name
        engine: Filter by database engine type
        include_deleted: Filter by deleted datasources
        limit: Maximum number of results to return
        offset: Number of results to skip (for pagination)
        with_detailed_data: Include connection status and other details
        include_total: Include total count of datasources in response
        sort_by: Field to sort by (name, created_at, updated_at, engine)
        sort_order: Sort order (asc or desc, default: desc)
    Returns:
        List of datasource objects matching the specified criteria
    """
    try:
        logger.debug(
            f"List datasources requested (v1) "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )

        datasources = await datasources_service.list_datasources(
            name=name,
            engine=engine,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
            with_detailed_data=with_detailed_data,
            include_total=include_total,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        if include_total:
            datasources, total = datasources
            logger.info(
                f"Listed {len(datasources)} datasources (total: {total}) "
                f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
            )
            return {"datasources": datasources, "total": total}
        else:
            logger.info(
                f"Listed datasources for user {datasources_service.user_id} in "
                f"organization {datasources_service.organization_id}"
            )
            return datasources
    except DatasourceServiceError as e:
        logger.error(
            f"Service error in list_datasources "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in list_datasources "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
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
        logger.debug(
            f"Get datasource requested: {datasource_name} (v1) "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )

        datasource = await datasources_service.get_datasource(
            datasource_name=datasource_name, with_detailed_data=with_detailed_data
        )

        logger.info(
            f"Retrieved datasource {datasource_name} "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )
        return datasource
    except DatasourceServiceError as e:
        logger.error(
            f"Service error in get_datasource "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e)) from None
        else:
            raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in get_datasource "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.head("/{datasource_name}", status_code=200)
async def check_datasource_exists(
    datasource_name: str,
    datasources_service: DatasourcesService = Depends(get_datasources_service),
):
    """
    Check if a datasource exists by name.

    Args:
        datasource_name: Name of the datasource to check

    Returns:
        200 OK if exists, 404 Not Found if it does not exist
    """
    try:
        logger.debug(
            f"Check datasource exists requested: {datasource_name} (v1) "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )

        await datasources_service.check_datasource_exists(datasource_name=datasource_name)

    except DatasourceNotFoundError as e:
        logger.error(
            f"Datasource not found "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except DatasourceServiceError as e:
        logger.error(
            f"Service error in check_datasource_exists "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in check_datasource_exists "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.post("/", status_code=201)
async def create_datasource(
    datasource_data: DatasourceCreateRequest,
    datasources_service: DatasourcesService = Depends(get_datasources_service),
    limits_service: LimitsService = Depends(get_limits_service),
) -> DatasourceResponse:
    """
    Create a new datasource.

    Args:
        datasource_data: Datasource creation request data

    Returns:
        Created datasource details

    Raises:
        HTTPException: 429 if usage limit exceeded.
    """
    # Check usage limits before creating
    await require_usage_available(limits_service, ResourceType.DATASOURCES)

    try:
        logger.debug(
            f"Create datasource requested: {datasource_data.name} (v1) "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )

        datasource = await datasources_service.create_datasource(datasource_data)

        logger.info(
            f"Created datasource {datasource_data.name} "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )
        return datasource
    except DatasourceAlreadyExistsError as e:
        logger.error(
            f"Datasource already exists "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=409, detail=str(e)) from None
    except DatasourceServiceError as e:
        logger.error(
            f"Service error in create_datasource "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in create_datasource "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
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
        logger.debug(
            f"Update datasource requested: {datasource_name} (v1) "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )

        datasource = await datasources_service.update_datasource(datasource_name, datasource_data)

        logger.info(
            f"Updated datasource {datasource_name} "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )
        return datasource
    except DatasourceNotFoundError as e:
        logger.error(
            f"Datasource not found "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except DatasourceServiceError as e:
        logger.error(
            f"Service error in update_datasource "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in update_datasource "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
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
            f"Delete datasource requested: {datasource_name} (v1) "
            f"for user {datasources_service.user_id} in "
            f"organization {datasources_service.organization_id}, cascade={cascade}"
        )

        await datasources_service.delete_datasource(datasource_name, cascade=cascade)

        logger.info(
            f"Deleted datasource {datasource_name} "
            f"for user {datasources_service.user_id} in "
            f"organization {datasources_service.organization_id}"
        )
        # Return 204 No Content on successful deletion
        return None
    except DatasourceNotFoundError as e:
        logger.error(
            f"Datasource not found "
            f"for user {datasources_service.user_id} in "
            f"organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except DatasourceServiceError as e:
        logger.error(
            f"Service error in delete_datasource "
            f"for user {datasources_service.user_id} in "
            f"organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in delete_datasource "
            f"for user {datasources_service.user_id} in "
            f"organization {datasources_service.organization_id}: {str(e)}"
        )
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
        logger.debug(
            f"Test connection requested: {datasource_name} (v1) "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )

        result = await datasources_service.test_connection(datasource_name)

        logger.info(
            f"Test connection for datasource {datasource_name} "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )
        return result
    except Exception as e:
        logger.error(
            f"Unexpected error in check_datasource_connection "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        return DatasourceConnectionStatus(success=False, error_message=f"Connection test failed: {str(e)}")


@router.get("/{datasource_name}/tables/{table_name}/sample", status_code=200)
async def get_datasource_table_sample(
    datasource_name: str,
    table_name: str,
    limit: int = Query(10, ge=1, le=1000, description="Number of sample rows to return (1-1000)"),
    datasources_service: DatasourcesService = Depends(get_datasources_service),
) -> DatasourceTableSampleResponse:
    """
    Get a sample of a table from a datasource.

    Args:
        datasource_name: Name of the datasource.
        table_name: Name of the table to sample.
        limit: Number of sample rows to return.

    Returns:
        Sample data with column names in structured format.
    """
    try:
        logger.debug(
            f"Get table sample requested: {datasource_name}.{table_name} (v1) "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )

        sample_response = await datasources_service.get_datasource_table_sample(datasource_name, table_name, limit)

        logger.info(
            f"Get table sample for datasource {datasource_name}.{table_name} "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )
        return sample_response
    except DatasourceNotFoundError as e:
        logger.error(
            f"Datasource not found "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except DatasourceServiceError as e:
        logger.error(
            f"Service error in get_datasource_table_sample "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in get_datasource_table_sample "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/{datasource_name}/tables/{table_name}/row-count", status_code=200)
async def get_datasource_table_row_count(
    datasource_name: str, table_name: str, datasources_service: DatasourcesService = Depends(get_datasources_service)
) -> int:
    """
    Get the row count of a table from a datasource.

    Args:
        datasource_name: Name of the datasource.
        table_name: Name of the table to get row count for.

    Returns:
        Number of rows in the table.
    """
    try:
        logger.debug(
            f"Get table row count requested: {datasource_name}.{table_name} (v1) "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )

        row_count = await datasources_service.get_datasource_table_row_count(datasource_name, table_name)

        logger.info(
            f"Get table row count for datasource {datasource_name}.{table_name} "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )
        return row_count
    except DatasourceNotFoundError as e:
        logger.error(
            f"Datasource not found "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except DatasourceServiceError as e:
        logger.error(
            f"Service error in get_datasource_table_row_count "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in get_datasource_table_row_count "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.post("/{datasource_name}/query", status_code=200)
async def query_datasource(
    datasource_name: str,
    request: DatasourceQueryRequest,
    datasources_service: DatasourcesService = Depends(get_datasources_service),
) -> DatasourceQueryResponse:
    """Execute a query against a datasource."""
    try:
        query_response = await datasources_service.query(datasource_name, request.query, request.native_query)

        return query_response
    except DatasourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except InvalidDatasourceQueryError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except DatasourceServiceError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/{datasource_name}/catalog", status_code=200)
async def get_datasource_catalog(
    datasource_name: str,
    datasources_service: DatasourcesService = Depends(get_datasources_service),
) -> DataCatalogResponse:
    """
    Get the aggregated data catalog for a datasource.

    Returns all cataloged tables for the given datasource.

    Args:
        datasource_name: Name of the datasource to get catalog for

    Returns:
        DataCatalogResponse with all cataloged tables for the given datasource
    """
    try:
        logger.debug(
            f"Get datasource catalog requested: {datasource_name} (v1) "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )

        data_catalog = await datasources_service.get_datasource_catalog(datasource_name=datasource_name)

        logger.info(
            f"Retrieved data catalog for datasource {datasource_name} "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )
        return data_catalog
    except DatasourceNotFoundError as e:
        logger.error(
            f"Datasource not found "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except DatasourceServiceError as e:
        logger.error(
            f"Service error in get_datasource_catalog "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in get_datasource_catalog "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/{datasource_name}/catalog/tables/{table_name}", status_code=200)
async def get_datasource_table_catalog(
    datasource_name: str,
    table_name: str,
    datasources_service: DatasourcesService = Depends(get_datasources_service),
) -> TableResponse:
    """
    Get the data catalog for a table in a datasource.
    """
    try:
        logger.debug(
            f"Get datasource table catalog requested: {datasource_name}.{table_name} (v1) "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )

        data_catalog = await datasources_service.get_datasource_table_catalog(datasource_name, table_name)

        logger.info(
            f"Retrieved data catalog for table {table_name} in datasource {datasource_name} "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )
        return data_catalog
    except DatasourceNotFoundError as e:
        logger.error(
            f"Datasource not found "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except DatasourceTableNotCatalogedError as e:
        logger.error(
            f"Table not cataloged "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=409, detail=str(e)) from None
    except DatasourceServiceError as e:
        logger.error(
            f"Service error in get_datasource_table_catalog "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in get_datasource_table_catalog "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.patch("/{datasource_name}/catalog/tables/{table_name}", status_code=200)
async def update_table_description(
    datasource_name: str,
    table_name: str,
    update_request: UpdateTableDescriptionRequest,
    datasources_service: DatasourcesService = Depends(get_datasources_service),
) -> TableResponse:
    """
    Update the description of a table in the data catalog.

    Args:
        datasource_name: Name of the datasource
        table_name: Name of the table to update
        update_request: Request with new description

    Returns:
        TableResponse with updated table information
    """
    try:
        logger.debug(
            f"Update table description requested: {datasource_name}/{table_name} (v1) "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )

        table_response = await datasources_service.update_datasource_table_catalog_description(
            datasource_name=datasource_name,
            table_name=table_name,
            description=update_request.description,
        )

        logger.info(
            f"Updated table description for {datasource_name}/{table_name} "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )
        return table_response

    except DatasourceNotFoundError as e:
        logger.error(
            f"Datasource not found "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except DatasourceTableNotFoundError as e:
        logger.error(
            f"Table not found "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except DatasourceTableNotCatalogedError as e:
        logger.error(
            f"Table not cataloged "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=409, detail=str(e)) from None
    except DatasourceServiceError as e:
        logger.error(
            f"Service error in update_table_description "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in update_table_description "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.patch("/{datasource_name}/catalog/tables/{table_name}/columns/{column_name}", status_code=200)
async def update_column_description(
    datasource_name: str,
    table_name: str,
    column_name: str,
    update_request: UpdateColumnDescriptionRequest,
    datasources_service: DatasourcesService = Depends(get_datasources_service),
) -> ColumnResponse:
    """
    Update the description of a column in the data catalog.

    Args:
        datasource_name: Name of the datasource
        table_name: Name of the table containing the column
        column_name: Name of the column to update
        update_request: Request with new description

    Returns:
        ColumnResponse with updated column information
    """
    try:
        logger.debug(
            f"Update column description requested: {datasource_name}/{table_name}/{column_name} (v1) "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )

        column_response = await datasources_service.update_datasource_table_catalog_column_description(
            datasource_name=datasource_name,
            table_name=table_name,
            column_name=column_name,
            description=update_request.description,
        )

        logger.info(
            f"Updated column description for {datasource_name}/{table_name}/{column_name} "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}"
        )
        return column_response

    except DatasourceNotFoundError as e:
        logger.error(
            f"Datasource not found "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except DatasourceTableNotFoundError as e:
        logger.error(
            f"Table not found "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except DatasourceTableNotCatalogedError as e:
        logger.error(
            f"Table not cataloged "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=409, detail=str(e)) from None
    except DatasourceTableColumnNotFoundError as e:
        logger.error(
            f"Column not found "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except DatasourceTableColumnNotCatalogedError as e:
        logger.error(
            f"Column not cataloged "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=409, detail=str(e)) from None
    except DatasourceServiceError as e:
        logger.error(
            f"Service error in update_column_description "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in update_column_description "
            f"for user {datasources_service.user_id} in organization {datasources_service.organization_id}: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None
