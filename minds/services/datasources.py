"""
Datasources service for managing datasource operations.

This service handles both internal dataource information storage and MindsDB SDK integration
for datasource management operations.
"""

from datetime import datetime, timezone
from typing import Any, Literal

import sqlglot
from mindsdb_sdk.server import Server
from mindsdb_sql_parser import ParsingException, parse_sql
from mindsdb_sql_parser.ast import Select
from sqlalchemy.orm import selectinload, with_loader_criteria
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlmodel import Session, and_, func, select

from minds.common.logger import setup_logging
from minds.common.mindsdb import extract_databases_from_select
from minds.common.utilities import safe_parse
from minds.model.data_catalog.column import Column
from minds.model.data_catalog.column_statistics import ColumnStatistics
from minds.model.data_catalog.foreign_key_constraint import ForeignKeyConstraint
from minds.model.data_catalog.primary_key_constraint import PrimaryKeyConstraint
from minds.model.data_catalog.table import Table
from minds.model.datasource import Datasource
from minds.model.mind_datasource import MindDatasource
from minds.schemas.datasources import (
    ColumnResponse,
    ColumnStatisticsResponse,
    DataCatalogResponse,
    DatasourceConnectionStatus,
    DatasourceCreateRequest,
    DatasourceDetailedResponse,
    DatasourceQueryResponse,
    DatasourceResponse,
    DatasourceTableSampleResponse,
    DatasourceUpdateRequest,
    ForeignKeyConstraintResponse,
    PrimaryKeyConstraintResponse,
    TableResponse,
)

# Set up logging
logger = setup_logging()


class DatasourceServiceError(Exception):
    """Base exception for datasource service errors."""

    pass


class DatasourceNotFoundError(DatasourceServiceError):
    """Raised when a datasource is not found."""

    pass


class DatasourceAlreadyExistsError(DatasourceServiceError):
    """Raised when trying to create a datasource that already exists."""

    pass


class DatasourceConnectionError(DatasourceServiceError):
    """Raised when datasource connection fails."""

    pass


class DatasourceTableNotFoundError(DatasourceServiceError):
    """Raised when a table is not found in a datasource."""

    pass


class DatasourceTableNotCatalogedError(DatasourceServiceError):
    """Raised when a table is not cataloged in a datasource."""

    pass


class DatasourceTableColumnNotFoundError(DatasourceServiceError):
    """Raised when a column is not found in a table catalog."""

    pass


class DatasourceTableColumnNotCatalogedError(DatasourceServiceError):
    """Raised when a column is not cataloged in a table."""

    pass


class InvalidDatasourceQueryError(DatasourceServiceError):
    """Raised when a datasource query is invalid."""

    pass


class DatasourcesService:
    """
    Service for managing datasource operations.

    Handles CRUD operations for datasources using dual storage:
    - Internal PostgreSQL storage for user attribution and fast queries
    - MindsDB database creation for connection

    Follows eager sync pattern - always syncs to MindsDB on create/update/delete.
    """

    def __init__(self, session: Session, mindsdb_client: Server, user_id: str, organization_id: str):
        """
        Initialize the datasources service.

        Args:
            session: Database session for internal storage
            mindsdb_client: MindsDB client for SDK operations
            user_id: Current user ID
        """
        self.session = session
        self.mindsdb_client = mindsdb_client
        self.user_id = user_id
        self.organization_id = organization_id
        logger.debug(f"DatasourcesService initialized for user {user_id} and organization {organization_id}")

    async def count_datasources(
        self,
        is_sample: bool | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> int:
        """
        Count non-deleted datasources for the current organization.

        This is a lightweight alternative to list_datasources() intended for
        usage tracking (e.g. limits enforcement). It runs a single COUNT query
        instead of loading full Datasource objects.

        Args:
            is_sample: When provided, only count datasources matching this sample flag.
                       Pass False to exclude sample/template datasources from the count.
            since: When provided, only count datasources created on or after this datetime.
                   Used for billing-cycle-scoped counts.
            until: When provided, only count datasources created before this datetime.
                   Used for billing-cycle-scoped counts.

        Returns:
            int: Number of datasources matching the filters.
        """
        try:
            logger.debug(
                f"Counting datasources for organization {self.organization_id} "
                f"(user_id={self.user_id}, is_sample={is_sample}, since={since}, until={until})"
            )

            # Always scope to the current organization and exclude soft-deleted records
            conditions = [
                Datasource.organization_id == self.organization_id,
                Datasource.user_id == self.user_id,
                Datasource.deleted_at.is_(None),
            ]

            # Optionally narrow to a specific user for per-user limits
            if is_sample is not None:
                conditions.append(Datasource.is_sample == is_sample)

            if since is not None:
                conditions.append(Datasource.created_at >= since)

            if until is not None:
                conditions.append(Datasource.created_at <= until)

            stmt = select(func.count(Datasource.id)).where(and_(*conditions))
            count = self.session.exec(stmt).one()

            logger.debug(
                f"Counted {count} datasources for organization {self.organization_id} "
                f"(user_id={self.user_id}, is_sample={is_sample}, since={since}, until={until})"
            )
            return count
        except Exception as e:
            logger.error(
                f"Error counting datasources for organization {self.organization_id} "
                f"(user_id={self.user_id}, is_sample={is_sample}, since={since}, until={until}): {str(e)}"
            )
            raise DatasourceServiceError(f"Failed to count datasources: {str(e)}") from None

    async def list_datasources(
        self,
        name: str | None = None,
        engine: str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
        with_detailed_data: bool = False,
        include_total: bool = False,
        sort_by: Literal["name", "created_at", "updated_at", "engine"] | None = None,
        sort_order: Literal["asc", "desc"] = "desc",
    ) -> (
        list[DatasourceResponse | DatasourceDetailedResponse]
        | tuple[list[DatasourceResponse | DatasourceDetailedResponse], int]
    ):
        """
        List datasources for the current company.

        Args:
            name: Filter by datasource name
            engine: Filter by database engine
            include_deleted: Include deleted datasources
            limit: Maximum number of results
            offset: Number of results to skip
            with_detailed_data: Include connection status and other details
            include_total: Include total count of datasources in response
            sort_by: Field to sort by (name, created_at, updated_at, engine)
            sort_order: Sort order (asc or desc, default: desc)

        Returns:
            List of datasource response objects
        """
        try:
            logger.debug(
                f"Listing datasources for user {self.user_id} in organization {self.organization_id} with filters: "
                f"name={name}, engine={engine}, include_deleted={include_deleted}, limit={limit}, offset={offset}, "
                f"sort_by={sort_by}, sort_order={sort_order}, include_total={include_total}"
            )

            # Build query conditions
            conditions = [Datasource.organization_id == self.organization_id]
            if name is not None:
                conditions.append(Datasource.name.ilike(f"%{name}%"))
            if engine is not None:
                conditions.append(Datasource.engine == engine)
            if not include_deleted:
                conditions.append(Datasource.deleted_at.is_(None))

            # Calculate total count if requested
            total_count = None
            if include_total:
                # For count, we don't need joins or options
                count_statement = (
                    select(func.count(func.distinct(Datasource.id))).select_from(Datasource).where(and_(*conditions))
                )
                total_count = self.session.exec(count_statement).one()

            # Determine sort field and order
            sort_field = Datasource.created_at  # default
            if sort_by == "name":
                sort_field = Datasource.name
            elif sort_by == "created_at":
                sort_field = Datasource.created_at
            elif sort_by == "updated_at":
                sort_field = Datasource.modified_at
            elif sort_by == "engine":
                sort_field = Datasource.engine

            order_by = sort_field.desc() if sort_order == "desc" else sort_field.asc()

            statement = select(Datasource).where(and_(*conditions)).order_by(order_by).offset(offset).limit(limit)

            datasources = self.session.exec(statement).all()

            datasources_list = []
            for datasource in datasources:
                if with_detailed_data:
                    datasource_response = await self._datasource_to_detailed_response(datasource)
                else:
                    datasource_response = self._datasource_to_response(datasource)
                datasources_list.append(datasource_response)

            logger.info(
                f"Retrieved {len(datasources)} datasources for user {self.user_id} and "
                f"organization {self.organization_id}"
            )

            if include_total:
                return datasources_list, total_count
            return datasources_list
        except Exception as e:
            logger.error(
                f"Error listing datasources for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise DatasourceServiceError(f"Failed to list datasources: {str(e)}") from None

    async def get_datasource(
        self, datasource_name: str, with_detailed_data: bool = False
    ) -> DatasourceResponse | DatasourceDetailedResponse:
        """
        Get a specific datasource by name.

        Args:
            datasource_name: Name of the datasource
            with_detailed_data: Include connection status and details

        Returns:
            Datasource response object

        Raises:
            DatasourceNotFoundError: If datasource doesn't exist
        """
        try:
            datasource_name = datasource_name.lower()
            logger.debug(
                f"Getting datasource {datasource_name} for user {self.user_id} in organization {self.organization_id}"
            )

            datasource = await self._get_datasource(datasource_name)

            if not datasource:
                raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not found")

            if with_detailed_data:
                return await self._datasource_to_detailed_response(datasource)
            else:
                return self._datasource_to_response(datasource)

        except DatasourceNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error getting datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise DatasourceServiceError(f"Failed to get datasource: {str(e)}") from None

    async def create_datasource(self, datasource_data: DatasourceCreateRequest) -> DatasourceResponse:
        """
        Create a new datasource.

        Args:
            datasource_data: Datasource creation request data

        Returns:
            Created datasource response

        Raises:
            DatasourceAlreadyExistsError: If datasource name already exists
            DatasourceConnectionError: If connection test fails
        """
        try:
            logger.debug(
                f"Creating datasource {datasource_data.name} for user {self.user_id} in "
                f"organization {self.organization_id}"
            )

            # Check if datasource already exists
            existing = await self._get_datasource(datasource_data.name)

            if existing:
                raise DatasourceAlreadyExistsError(f"Datasource with name '{datasource_data.name}' already exists")

            # Create datasource record (matching MindsDB schema)
            datasource = Datasource(
                name=datasource_data.name,
                description=datasource_data.description,
                engine=datasource_data.engine,
                user_id=self.user_id,
                organization_id=self.organization_id,
                is_sample=datasource_data.is_sample,
            )

            # Save to internal database first
            self.session.add(datasource)
            self.session.flush()

            try:
                await self._create_mindsdb_database(datasource, datasource_data.connection_data)

                self.session.commit()
                self.session.refresh(datasource)
                logger.info(
                    f"Created datasource {datasource_data.name} for user {self.user_id} in "
                    f"organization {self.organization_id}"
                )

            except DatasourceServiceError:
                # Rollback internal database if MindsDB creation fails
                self.session.rollback()
                raise

            return self._datasource_to_response(datasource)

        except DatasourceAlreadyExistsError:
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(
                f"Error creating datasource {datasource_data.name} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise DatasourceServiceError(f"Failed to create datasource: {str(e)}") from None

    async def update_datasource(
        self, datasource_name: str, datasource_data: DatasourceUpdateRequest
    ) -> DatasourceResponse:
        """
        Update an existing datasource.

        Args:
            datasource_name: Name of the datasource to update
            datasource_data: Update request data

        Returns:
            Updated datasource response

        Raises:
            DatasourceNotFoundError: If datasource doesn't exist
            DatasourceConnectionError: If connection test fails
        """
        try:
            datasource_name = datasource_name.lower()
            logger.debug(
                f"Updating datasource {datasource_name} for user {self.user_id} in organization {self.organization_id}"
            )

            # Get existing datasource
            datasource = await self._get_datasource(datasource_name)

            if not datasource:
                raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not found")

            # Update fields (using simplified schema)
            if datasource_data.description is not None:
                datasource.description = datasource_data.description

            # Save to internal database first
            self.session.flush()

            try:
                # Update in MindsDB if connection data changed
                if datasource_data.connection_data is not None:
                    await self._update_mindsdb_database(datasource, datasource_data.connection_data)

                self.session.commit()
                self.session.refresh(datasource)
                logger.info(
                    f"Updated datasource {datasource_name} for user {self.user_id} in "
                    f"organization {self.organization_id}"
                )

            except DatasourceServiceError:
                self.session.rollback()
                raise

            return self._datasource_to_response(datasource)

        except DatasourceNotFoundError:
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(
                f"Error updating datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise DatasourceServiceError(f"Failed to update datasource: {str(e)}") from None

    async def delete_datasource(self, datasource_name: str, cascade: bool = False) -> None:
        """
        Delete a datasource (hard delete on MindsDB and soft delete in internal database).

        Args:
            datasource_name: Name of the datasource to delete
            cascade: Whether to remove from all minds that use it

        Raises:
            DatasourceNotFoundError: If datasource doesn't exist
        """
        try:
            datasource_name = datasource_name.lower()
            logger.debug(
                f"Deleting datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id} (cascade={cascade})"
            )

            # Get existing datasource
            statement = (
                select(Datasource)
                .where(
                    and_(
                        Datasource.name == datasource_name,
                        Datasource.organization_id == self.organization_id,
                        Datasource.deleted_at.is_(None),
                    )
                )
                .options(
                    selectinload(Datasource.mind_datasources),
                    with_loader_criteria(
                        MindDatasource,
                        lambda cls: cls.deleted_at.is_(None),
                        include_aliases=True,
                    ),
                )
            )

            datasource = self.session.exec(statement).first()

            if not datasource:
                raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not found")

            # TODO: If cascade is True, remove from all minds that use this datasource
            if cascade:
                logger.debug(f"Cascade deletion for datasource {datasource_name} - implement mind deletion")

            datasource.deleted_at = datetime.now(timezone.utc)

            # Soft delete the datasource first because it can be rolled back
            self.session.add(datasource)
            self.session.flush()

            await self._delete_mindsdb_database(datasource_name)

            self.session.commit()

            logger.info(
                f"Deleted datasource {datasource_name} for user {self.user_id} in organization {self.organization_id}"
            )
        except DatasourceNotFoundError:
            self.session.rollback()
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(
                f"Error deleting datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise DatasourceServiceError(f"Failed to delete datasource: {str(e)}") from None

    async def test_connection(self, datasource_name: str) -> DatasourceConnectionStatus:
        """
        Test connection to a datasource using MindsDB.

        Args:
            datasource_name: Name of the datasource to test

        Returns:
            Connection status result
        """
        try:
            datasource_name = datasource_name.lower()
            logger.debug(
                f"Testing connection for datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id}"
            )

            # Get datasource to verify it exists
            _ = await self._get_datasource(datasource_name)

            # Test connection using MindsDB by trying to list tables
            databases = self.mindsdb_client.databases

            try:
                database = databases.get(datasource_name)
            except AttributeError:
                return DatasourceConnectionStatus(success=False, error_message="Datasource not found in MindsDB")

            # Try to list tables to test the connection
            # This will trigger a connection test in MindsDB
            try:
                database.tables.list()
                return DatasourceConnectionStatus(success=True, mindsdb_database=database)
            except Exception as db_error:
                return DatasourceConnectionStatus(
                    success=False, error_message=f"Connection test failed: {str(db_error)}", mindsdb_database=database
                )
        except DatasourceNotFoundError:
            return DatasourceConnectionStatus(success=False, error_message="Datasource not found")
        except Exception as e:
            logger.error(
                f"Connection test failed for {datasource_name} "
                f"for user {self.user_id} and organization {self.organization_id}: {str(e)}"
            )
            return DatasourceConnectionStatus(success=False, error_message=str(e))

    async def get_datasource_table_sample(
        self, datasource_name: str, table_name: str, limit: int = 10
    ) -> DatasourceTableSampleResponse:
        """Get a sample of a table from a datasource."""
        datasource_name = datasource_name.lower()
        logger.debug(
            f"Getting sample data for table {table_name} of datasource {datasource_name} "
            f"for user {self.user_id} in organization {self.organization_id}"
        )

        try:
            datasource = await self._get_datasource(datasource_name)

            if not datasource:
                raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not found")

            sample_query = self.mindsdb_client.databases.get(datasource_name).query(
                f"SELECT * FROM {table_name} LIMIT {limit}"
            )
            result = sample_query.fetch()

            # Convert DataFrame to structured response
            column_names = result.columns.tolist()
            data = result.values.tolist()

            return DatasourceTableSampleResponse(data=data, column_names=column_names)
        except DatasourceNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error getting sample data for table {table_name} of datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise DatasourceServiceError(f"Failed to get sample data: {str(e)}") from None

    async def get_datasource_table_row_count(self, datasource_name: str, table_name: str) -> int:
        """Get the row count of a table from a datasource."""
        datasource_name = datasource_name.lower()
        logger.debug(
            f"Getting row count for table {table_name} of datasource {datasource_name} "
            f"for user {self.user_id} in organization {self.organization_id}"
        )

        try:
            datasource = await self._get_datasource(datasource_name)

            if not datasource:
                raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not found")

            row_count_query = self.mindsdb_client.databases.get(datasource_name).query(
                f"SELECT COUNT(*) FROM {table_name}"
            )
            result = row_count_query.fetch()
            return result.values[0][0]
        except DatasourceNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error getting row count for table {table_name} of datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise DatasourceServiceError(f"Failed to get row count: {str(e)}") from None

    async def query(self, datasource_name: str, query: str, native_query: bool = True) -> DatasourceQueryResponse:
        """Execute a query against a datasource via MindsDB."""
        datasource_name = datasource_name.lower()
        logger.debug(
            f"Executing native query on datasource {datasource_name} "
            f"for user {self.user_id} in organization {self.organization_id}"
        )

        try:
            datasource = await self._get_datasource(datasource_name)
            if not datasource:
                raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not found")

            if native_query:
                # TODO: Ensure that all engines are supported by sqlglot
                # TODO: Ensure that engines are properly mapped to sqlglot dialects
                self._validate_native_query(query, datasource.engine)

                # TODO: This should be run be executed via the SDK
                # A function is not available via the SDK at the moment
                response = self.mindsdb_client.api.session.post(
                    self.mindsdb_client.api.url + "/api/sql/query",
                    json={"query": query, "context": {"db": datasource_name, "native_query": native_query}},
                )
                response.raise_for_status()
                response_json = response.json()

                # Type can be either 'table', 'ok', or 'error'
                # Since mutations are not allowed, it is sage to assume that 'ok' is not possible
                if response_json.get("type") == "error":
                    raise DatasourceServiceError(
                        f"Failed to execute query: {response_json.get('error_message')}"
                    ) from None

                return DatasourceQueryResponse(
                    data=response_json.get("data"),
                    column_names=response_json.get("column_names"),
                )

            else:
                self._validate_mindsdb_query(query, datasource_name)

                database = self.mindsdb_client.databases.get(datasource_name)
                response_df = database.query(query).fetch()

                return DatasourceQueryResponse(
                    data=response_df.values.tolist(),
                    column_names=response_df.columns.tolist(),
                )
        except (DatasourceNotFoundError, InvalidDatasourceQueryError) as ue:
            logger.error(
                f"Error executing native query on datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(ue)}"
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error executing native query on datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise DatasourceServiceError(f"Failed to execute query: {str(e)}") from None

    def _validate_native_query(self, query: str, dialect: str):
        """
        Ensure the query is:
        - Exactly one statement
        - A read-only query (SELECT / WITH / set operations)
        - Contains no DML/DDL/transaction/command nodes anywhere
        - No SELECT INTO
        - No locking reads (FOR UPDATE / LOCK ...)

        DISCLAIMER: This is a best-effort validation. It is not guaranteed to catch all invalid queries.

        Args:
            query: The query to validate.
            dialect: The dialect of the query.

        Raises:
            InvalidDatasourceQueryError: If the query is not valid.
        """
        # Map the engines to dialects that do not coincide.
        # TODO: Complete this list for all engines that need to be supported.
        ENGINE_TO_DIALECT_MAP = {"mssql": "tsql"}

        try:
            if dialect in ENGINE_TO_DIALECT_MAP:
                dialect = ENGINE_TO_DIALECT_MAP[dialect]
            statements = sqlglot.parse(query, read=dialect)
        except ParseError:
            raise InvalidDatasourceQueryError(f"Invalid query: {query}") from None

        # 1) Exactly one statement
        if len(statements) != 1:
            raise InvalidDatasourceQueryError(f"Query must contain exactly one statement: {query}") from None

        statement = statements[0]

        # 2) Root must be a query expression
        ALLOWED_ROOT_NODES = (
            exp.Select,
            exp.With,
            exp.Union,
            exp.Intersect,
            exp.Except,
            exp.Show,
            exp.Describe,
        )
        if not isinstance(statement, ALLOWED_ROOT_NODES):
            raise InvalidDatasourceQueryError(
                f"Query must be a read-only query (SELECT/WITH/set-ops): {query}"
            ) from None

        # 3) Forbid writes / DDL / transactions / commands anywhere in tree
        FORBIDDEN_NODES = (
            # DML
            exp.Insert,
            exp.Update,
            exp.Delete,
            exp.Merge,
            exp.Replace,
            # DDL
            exp.Create,
            exp.Drop,
            exp.Alter,
            exp.TruncateTable,
            exp.Comment,
            # Transactions
            exp.Commit,
            exp.Rollback,
            exp.Transaction,
            # Generic command bucket (VACUUM, ANALYZE, etc.)
            exp.Command,
            # Utility (not a query)
            exp.Use,
            exp.Set,
        )

        for node_type in FORBIDDEN_NODES:
            if statement.find(node_type):
                raise InvalidDatasourceQueryError(
                    f"Query must be read-only (found forbidden operation): {query}"
                ) from None

        # 4) Prevent SELECT INTO (table creation via SELECT)
        if statement.find(exp.Into):
            raise InvalidDatasourceQueryError(f"Query must not contain SELECT INTO: {query}") from None

        # 5) Prevent locking reads (FOR UPDATE / FOR SHARE / LOCK ...)
        LOCKING_NODES = (exp.Lock,)

        for node_type in LOCKING_NODES:
            if statement.find(node_type):
                raise InvalidDatasourceQueryError(f"Query must not take locks (e.g., FOR UPDATE): {query}") from None

    def _validate_mindsdb_query(self, query: str, datasource_name: str) -> None:
        """
        Validate a query to ensure it is a single, read-only MindsDB SELECT statement.

        Args:
            query: The query to validate.
            datasource_name: Name of the datasource.
        """
        try:
            parsed_query = parse_sql(query)
        except ParsingException as e:
            raise InvalidDatasourceQueryError(f"Invalid query: {e}") from e

        # The parser does not allow for multiple statements
        # TODO: Validate this assumption

        if not isinstance(parsed_query, Select):
            raise InvalidDatasourceQueryError(f"Query is not a SELECT statement: {parsed_query}") from None

        # Ensure that the query is referencing the correct datasource
        databases = extract_databases_from_select(parsed_query)
        if len(databases) != 1 or databases[0] != datasource_name:
            raise InvalidDatasourceQueryError(
                f"Query is not referencing the correct datasource: {parsed_query}"
            ) from None

    async def check_datasource_exists(self, datasource_name: str) -> None:
        """
        Check if a datasource exists by name.

        Args:
            datasource_name: Name of the datasource to check.

        Raises:
            DatasourceNotFoundError: If the datasource does not exist.

        Returns:
            None
        """
        datasource_name = datasource_name.lower()
        logger.debug(
            f"Checking existence of datasource {datasource_name} for user {self.user_id} in "
            f"organization {self.organization_id}"
        )

        try:
            datasource = await self._get_datasource(datasource_name)
        except Exception as e:
            logger.error(
                f"Error checking existence of datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise DatasourceServiceError(f"Failed to check datasource existence: {str(e)}") from None

        if datasource:
            logger.debug(
                f"Datasource {datasource_name} exists for user {self.user_id} in organization {self.organization_id}"
            )
            return
        else:
            logger.debug(
                f"Datasource {datasource_name} does not exist for user {self.user_id} in "
                f"organization {self.organization_id}"
            )
            raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not found")

    async def get_datasource_catalog(self, datasource_name: str) -> DataCatalogResponse:
        """
        Get the data catalog for a datasource.

        Args:
            datasource_name: Name of the datasource

        Returns:
            DataCatalogResponse with aggregated catalog information

        Raises:
            DatasourceNotFoundError: If the datasource does not exist
        """
        try:
            datasource_name = datasource_name.lower()
            logger.debug(
                f"Getting data catalog for datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id}"
            )

            statement = (
                select(Datasource)
                .where(
                    and_(
                        Datasource.name == datasource_name,
                        Datasource.user_id == self.user_id,
                        Datasource.organization_id == self.organization_id,
                        Datasource.deleted_at.is_(None),
                    )
                )
                .options(
                    # Tables
                    selectinload(Datasource.tables),
                    # Columns + statistics
                    selectinload(Datasource.tables).selectinload(Table.columns).selectinload(Column.statistics),
                    # PKs
                    selectinload(Datasource.tables)
                    .selectinload(Table.primary_key_constraints)
                    .selectinload(PrimaryKeyConstraint.column),
                    # FKs (and their referenced objects)
                    selectinload(Datasource.tables)
                    .selectinload(Table.foreign_key_constraints)
                    .selectinload(ForeignKeyConstraint.column),
                    selectinload(Datasource.tables)
                    .selectinload(Table.foreign_key_constraints)
                    .selectinload(ForeignKeyConstraint.referenced_table),
                    selectinload(Datasource.tables)
                    .selectinload(Table.foreign_key_constraints)
                    .selectinload(ForeignKeyConstraint.referenced_column),
                    # Avoid loading deleted objects (and optionally restrict tables)
                    with_loader_criteria(
                        Table,
                        lambda cls: cls.deleted_at.is_(None),
                        include_aliases=True,
                    ),
                    with_loader_criteria(
                        Column,
                        lambda cls: cls.deleted_at.is_(None),
                        include_aliases=True,
                    ),
                    with_loader_criteria(
                        ColumnStatistics,
                        lambda cls: cls.deleted_at.is_(None),
                        include_aliases=True,
                    ),
                    with_loader_criteria(
                        PrimaryKeyConstraint,
                        lambda cls: cls.deleted_at.is_(None),
                        include_aliases=True,
                    ),
                    with_loader_criteria(
                        ForeignKeyConstraint,
                        lambda cls: cls.deleted_at.is_(None),
                        include_aliases=True,
                    ),
                )
            )

            datasource = self.session.exec(statement).first()

            if not datasource:
                raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not found")

            return self._datasource_to_catalog_response(datasource)

        except DatasourceNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error getting data catalog for datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise DatasourceServiceError(f"Failed to get data catalog: {str(e)}") from None

    async def get_datasource_table_catalog(self, datasource_name: str, table_name: str) -> TableResponse:
        """
        Get the data catalog for a table in a datasource.

        Args:
            datasource_name: Name of the datasource
            table_name: Name of the table

        Returns:
            TableResponse with the data catalog for the table

        Raises:
            DatasourceNotFoundError: If the datasource does not exist
            DatasourceTableNotFoundError: If the table does not exist
        """
        try:
            datasource_name = datasource_name.lower()
            logger.debug(
                f"Getting data catalog for table {table_name} in datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id}"
            )

            table = await self._get_datasource_table_catalog(datasource_name, table_name)

            if table is None:
                # If the table is not returned, check if it exists in the datasource
                if not self._check_table_exists(datasource_name, table_name):
                    raise DatasourceTableNotFoundError(
                        f"Table '{table_name}' not found in datasource '{datasource_name}'"
                    )
                else:
                    raise DatasourceTableNotCatalogedError(
                        f"Table '{table_name}' not cataloged in datasource '{datasource_name}'"
                    )

            return self._table_to_catalog_response(table)

        except (DatasourceNotFoundError, DatasourceTableNotFoundError, DatasourceTableNotCatalogedError):
            raise
        except Exception as e:
            logger.error(
                f"Error getting data catalog for table {table_name} in datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise DatasourceServiceError(f"Failed to get table catalog: {str(e)}") from None

    async def update_datasource_table_catalog_description(
        self, datasource_name: str, table_name: str, description: str
    ) -> TableResponse:
        """
        Update the description of a table in the data catalog.

        Args:
            datasource_name: Name of the datasource
            table_name: Name of the table
            description: New description

        Returns:
            TableResponse with the updated data catalog for the table

        Raises:
            DatasourceNotFoundError: If the datasource does not exist
            DatasourceTableCatalogNotFoundError: If the table does not exist
        """
        try:
            datasource_name = datasource_name.lower()
            logger.debug(
                f"Updating description for table {table_name} in datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id}"
            )

            table = await self._get_datasource_table_catalog(datasource_name, table_name)

            if not table:
                # If the table is not returned, check if it exists in the datasource
                if not self._check_table_exists(datasource_name, table_name):
                    raise DatasourceTableNotFoundError(
                        f"Table '{table_name}' not found in datasource '{datasource_name}'"
                    )

                raise DatasourceTableNotCatalogedError(
                    f"Table '{table_name}' not cataloged in datasource '{datasource_name}'"
                )

            table.description = description
            self.session.add(table)
            self.session.commit()
            self.session.refresh(table)

            return self._table_to_catalog_response(table)

        except (DatasourceNotFoundError, DatasourceTableNotFoundError, DatasourceTableNotCatalogedError):
            raise
        except Exception as e:
            logger.error(
                f"Error updating description for table {table_name} in datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise DatasourceServiceError(f"Failed to update table catalog description: {str(e)}") from None

    async def update_datasource_table_catalog_column_description(
        self, datasource_name: str, table_name: str, column_name: str, description: str
    ) -> ColumnResponse:
        """
        Update the description of a column in the data catalog.

        Args:
            datasource_name: Name of the datasource
            table_name: Name of the table
            column_name: Name of the column
            description: New description
        """
        try:
            datasource_name = datasource_name.lower()
            logger.debug(
                f"Updating description for column {column_name} in table {table_name} in datasource {datasource_name} "
                f"for user {self.user_id} in organization {self.organization_id}"
            )

            datasource = await self._get_datasource(datasource_name=datasource_name)
            if not datasource:
                raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not found")

            # Breaking down the logic into multiple queries allows for better error reporting
            table_statement = select(Table).where(
                and_(
                    Table.datasource_id == datasource.id,
                    Table.name == table_name,
                    Table.organization_id == self.organization_id,
                    Table.deleted_at.is_(None),
                )
            )
            table = self.session.exec(table_statement).first()
            if not table:
                # If the table is not returned, check if it exists in the datasource
                if not self._check_table_exists(datasource_name, table_name):
                    raise DatasourceTableNotFoundError(
                        f"Table '{table_name}' not found in datasource '{datasource_name}'"
                    )
                else:
                    raise DatasourceTableNotCatalogedError(
                        f"Table '{table_name}' not cataloged in datasource '{datasource_name}'"
                    )

            column_statement = (
                select(Column)
                .where(
                    and_(
                        Column.table_id == table.id,
                        Column.name == column_name,
                        Column.organization_id == self.organization_id,
                        Column.deleted_at.is_(None),
                    )
                )
                .options(
                    # Statistics
                    selectinload(Column.statistics),
                    # Avoid loading deleted objects
                    with_loader_criteria(
                        ColumnStatistics,
                        lambda cls: cls.deleted_at.is_(None),
                        include_aliases=True,
                    ),
                )
            )
            column = self.session.exec(column_statement).first()
            if column is None:
                # If the column is not returned, check if it exists in the table
                if not self._check_column_exists(datasource_name, table_name, column_name):
                    raise DatasourceTableColumnNotFoundError(
                        f"Column '{column_name}' not found in table '{table_name}'"
                    )
                else:
                    raise DatasourceTableColumnNotCatalogedError(
                        f"Column '{column_name}' not cataloged in table '{table_name}'"
                    )

            column.description = description
            self.session.add(column)
            self.session.commit()
            self.session.refresh(column)

            return self._column_to_catalog_response(column)

        except (
            DatasourceNotFoundError,
            DatasourceTableNotFoundError,
            DatasourceTableNotCatalogedError,
            DatasourceTableColumnNotFoundError,
            DatasourceTableColumnNotCatalogedError,
        ):
            raise
        except Exception as e:
            logger.error(
                f"Error updating description for column {column_name} in table {table_name} "
                f"in datasource {datasource_name} for user {self.user_id} in organization {self.organization_id}: {e!s}"
            )
            raise DatasourceServiceError(f"Failed to update table catalog column description: {str(e)}") from None

    async def _get_datasource(self, datasource_name: str) -> Datasource:
        """
        Utility function to get a specific datasource by name.

        Args:
            datasource_name: Name of the datasource to get.

        Returns:
            Datasource: Datasource object.
        """
        statement = select(Datasource).where(
            and_(
                Datasource.name == datasource_name,
                Datasource.organization_id == self.organization_id,
                Datasource.deleted_at.is_(None),
            )
        )
        result = self.session.exec(statement)
        datasource = result.first()

        return datasource

    async def _get_datasource_table_catalog(self, datasource_name: str, table_name: str) -> Table:
        """
        Utility function to get a specific table catalog by name in a datasource.

        Args:
            datasource_name: Name of the datasource
            table_name: Name of the table
        """
        datasource = await self._get_datasource(datasource_name=datasource_name)

        if not datasource:
            raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not found")

        statement = (
            select(Table)
            .where(
                and_(
                    Table.datasource_id == datasource.id,
                    Table.name == table_name,
                    Table.organization_id == self.organization_id,
                    Table.deleted_at.is_(None),
                )
            )
            .options(
                # Columns + statistics
                selectinload(Table.columns).selectinload(Column.statistics),
                # PKs
                selectinload(Table.primary_key_constraints),
                # FKs (and their referenced objects)
                selectinload(Table.foreign_key_constraints).selectinload(ForeignKeyConstraint.column),
                selectinload(Table.foreign_key_constraints).selectinload(ForeignKeyConstraint.referenced_table),
                selectinload(Table.foreign_key_constraints).selectinload(ForeignKeyConstraint.referenced_column),
                # Avoid loading deleted objects
                with_loader_criteria(
                    Column,
                    lambda cls: cls.deleted_at.is_(None),
                    include_aliases=True,
                ),
                with_loader_criteria(
                    ColumnStatistics,
                    lambda cls: cls.deleted_at.is_(None),
                    include_aliases=True,
                ),
                with_loader_criteria(
                    PrimaryKeyConstraint,
                    lambda cls: cls.deleted_at.is_(None),
                    include_aliases=True,
                ),
                with_loader_criteria(
                    ForeignKeyConstraint,
                    lambda cls: cls.deleted_at.is_(None),
                    include_aliases=True,
                ),
            )
        )
        return self.session.exec(statement).first()

    def _check_table_exists(self, datasource_name: str, table_name: str) -> bool:
        """
        Check if a table exists in a datasource.

        Args:
            datasource_name: Name of the datasource
            table_name: Name of the table

        Returns:
            True if the table exists, False otherwise
        """
        available_tables = self.mindsdb_client.databases.get(datasource_name).tables.list()
        available_table_names = [table.name for table in available_tables]
        return table_name in available_table_names

    def _check_column_exists(self, datasource_name: str, table_name: str, column_name: str) -> bool:
        """
        Check if a column exists in a table.

        Args:
            datasource_name: Name of the datasource
            table_name: Name of the table
            column_name: Name of the column
        """
        # TODO: Is there a more efficient way to get the columns of a table?
        table = self.mindsdb_client.databases.get(datasource_name).tables.get(table_name)
        available_columns = table.limit(1).fetch().columns.tolist()
        return column_name in available_columns

    async def _create_mindsdb_database(self, datasource: Datasource, connection_data: dict[str, Any]) -> None:
        """Create database/integration in MindsDB."""
        try:
            logger.debug(f"Creating MindsDB database for datasource {datasource.name}")

            # Use MindsDB SDK to create database
            databases = self.mindsdb_client.databases

            # Create the database with connection validation
            databases.create(
                name=datasource.name,
                engine=datasource.engine,
                connection_args=connection_data,
            )

            logger.info(f"Created MindsDB database {datasource.name}")

        except Exception as e:
            logger.error(f"Failed to create MindsDB database {datasource.name}: {str(e)}")
            raise DatasourceServiceError(f"MindsDB database creation failed: {str(e)}") from None

    async def _update_mindsdb_database(self, datasource: Datasource, connection_data: dict[str, Any]) -> None:
        """Update database/integration (connection data) in MindsDB."""
        try:
            logger.debug(f"Updating MindsDB database for datasource {datasource.name}")

            self.mindsdb_client.databases.update(
                name=datasource.name,
                connection_args=connection_data,
            )

            logger.info(f"Updated MindsDB database {datasource.name}")
        except Exception as e:
            logger.error(f"Failed to update MindsDB database {datasource.name}: {str(e)}")
            raise DatasourceServiceError(f"MindsDB database update failed: {str(e)}") from None

    async def _delete_mindsdb_database(self, datasource_name: str) -> None:
        """Delete database/integration from MindsDB."""
        try:
            logger.debug(f"Deleting MindsDB database {datasource_name}")

            self.mindsdb_client.databases.drop(datasource_name)
            logger.info(f"Deleted MindsDB database {datasource_name}")

            logger.debug(f"Deleted MindsDB database {datasource_name}")
        except Exception as e:
            logger.error(f"Failed to delete MindsDB database {datasource_name}: {str(e)}")
            raise DatasourceServiceError(f"MindsDB database deletion failed: {str(e)}") from None

    def _datasource_to_response(self, datasource: Datasource) -> DatasourceResponse:
        """Convert Datasource model to response object."""
        return DatasourceResponse(
            id=datasource.id,
            name=datasource.name,
            description=datasource.description,
            engine=datasource.engine,
            is_sample=datasource.is_sample,
            created_at=datasource.created_at.isoformat(),
            modified_at=datasource.modified_at.isoformat(),
        )

    async def _datasource_to_detailed_response(self, datasource: Datasource) -> DatasourceDetailedResponse:
        """Convert Datasource model to detailed response object with connection status."""
        base_response = self._datasource_to_response(datasource)

        # Get real connection status from MindsDB
        connection_status = await self.test_connection(datasource.name)
        connection_data = None
        if connection_status.mindsdb_database:
            connection_data = (
                connection_status.mindsdb_database.params
                if isinstance(connection_status.mindsdb_database.params, dict)
                else safe_parse(connection_status.mindsdb_database.params)
            )

        return DatasourceDetailedResponse(
            **base_response.model_dump(), connection_data=connection_data, connection_status=connection_status
        )

    def _datasource_to_catalog_response(self, datasource: Datasource) -> DataCatalogResponse:
        """Convert Datasource model to catalog response object."""
        return DataCatalogResponse(
            datasource=self._datasource_to_response(datasource),
            tables=[self._table_to_catalog_response(table) for table in datasource.tables],
        )

    def _table_to_catalog_response(self, table: Table) -> TableResponse:
        """Convert a Table model to a catalog TableResponse."""
        return TableResponse(
            name=table.name,
            schema=table.schema,
            description=table.description,
            type=table.type,
            row_count=table.row_count,
            columns=[self._column_to_catalog_response(column) for column in table.columns],
            primary_key_constraints=[
                PrimaryKeyConstraintResponse(
                    column_name=pk_constraint.column.name,
                    ordinal_position=pk_constraint.ordinal_position,
                    constraint_name=pk_constraint.constraint_name,
                )
                for pk_constraint in table.primary_key_constraints
            ],
            foreign_key_constraints=[
                ForeignKeyConstraintResponse(
                    column_name=fk_constraint.column.name,
                    referenced_table_name=fk_constraint.referenced_table.name,
                    referenced_column_name=fk_constraint.referenced_column.name,
                    constraint_name=fk_constraint.constraint_name,
                    ordinal_position=fk_constraint.ordinal_position,
                )
                for fk_constraint in table.foreign_key_constraints
            ],
        )

    def _column_to_catalog_response(self, column: Column) -> ColumnResponse:
        """Convert a Column model to a catalog ColumnResponse."""
        return ColumnResponse(
            name=column.name,
            data_type=column.data_type,
            description=column.description,
            default_value=column.default_value,
            is_nullable=column.is_nullable,
            statistics=(
                ColumnStatisticsResponse(
                    most_common_values=column.statistics.most_common_values,
                    most_common_frequencies=column.statistics.most_common_frequencies,
                    null_percentage=column.statistics.null_percentage,
                    distinct_values_count=column.statistics.distinct_values_count,
                    min_value=column.statistics.min_value,
                    max_value=column.statistics.max_value,
                )
                if getattr(column, "statistics", None) is not None
                else None
            ),
        )
