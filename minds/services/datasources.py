"""
Datasources service for managing datasource operations.

This service handles both internal dataource information storage and MindsDB SDK integration
for datasource management operations.
"""

from mindsdb_sdk.server import Server
from sqlmodel import Session, and_, select

from minds.common.logger import setup_logging
from minds.model.datasource import Datasource
from minds.schemas.datasources import (
    DatasourceConnectionStatus,
    DatasourceCreateRequest,
    DatasourceDetailedResponse,
    DatasourceResponse,
    DatasourceUpdateRequest,
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


class DatasourcesService:
    """
    Service for managing datasource operations.

    Handles CRUD operations for datasources using dual storage:
    - Internal PostgreSQL storage for user attribution and fast queries
    - MindsDB database creation for connection

    Follows eager sync pattern - always syncs to MindsDB on create/update/delete.
    """

    def __init__(self, session: Session, mindsdb_client: Server, user_id: str):
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

    async def list_datasources(
        self, engine: str | None = None, limit: int = 100, offset: int = 0, with_detailed_data: bool = False
    ) -> list[DatasourceResponse | DatasourceDetailedResponse]:
        """
        List datasources for the current company.

        Args:
            engine: Filter by database engine
            limit: Maximum number of results
            offset: Number of results to skip
            with_detailed_data: Include connection status and other details

        Returns:
            List of datasource response objects
        """
        try:
            logger.debug(
                f"Listing datasources for user {self.user_id} with filters: "
                f"engine={engine}, limit={limit}, offset={offset}"
            )

            # Build query with filters
            statement = select(Datasource).where(Datasource.user_id == self.user_id)

            if engine is not None:
                statement = statement.where(Datasource.engine == engine)

            statement = statement.offset(offset).limit(limit)

            result = self.session.exec(statement)
            datasources = result.all()

            # Convert to response objects
            responses = []
            for datasource in datasources:
                if with_detailed_data:
                    response = await self._datasource_to_detailed_response(datasource)
                else:
                    response = self._datasource_to_response(datasource)
                responses.append(response)

            logger.info(f"Found {len(responses)} datasources for user {self.user_id}")
            return responses

        except Exception as e:
            logger.error(f"Error listing datasources: {str(e)}")
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
            logger.debug(f"Getting datasource {datasource_name} for user {self.user_id}")

            statement = select(Datasource).where(
                and_(Datasource.name == datasource_name, Datasource.user_id == self.user_id)
            )

            result = self.session.exec(statement)
            datasource = result.first()

            if not datasource:
                raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not found")

            if with_detailed_data:
                return await self._datasource_to_detailed_response(datasource)
            else:
                return self._datasource_to_response(datasource)

        except DatasourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error getting datasource {datasource_name}: {str(e)}")
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
            logger.debug(f"Creating datasource {datasource_data.name} for user {self.user_id}")

            # Check if datasource already exists
            existing = self.session.exec(
                select(Datasource).where(
                    and_(Datasource.name == datasource_data.name, Datasource.user_id == self.user_id)
                )
            ).first()

            if existing:
                raise DatasourceAlreadyExistsError(f"Datasource with name '{datasource_data.name}' already exists")

            # Create datasource record (matching MindsDB schema)
            datasource = Datasource(
                name=datasource_data.name,
                engine=datasource_data.engine,
                connection_data=datasource_data.connection_data,
                user_id=self.user_id,
            )

            # Save to internal database first
            self.session.add(datasource)
            self.session.commit()
            self.session.refresh(datasource)

            try:
                await self._create_mindsdb_database(datasource)

                logger.info(
                    f"Created datasource {datasource_data.name} for user \
                                {self.user_id} (synced to MindsDB)"
                )

            except DatasourceServiceError:
                # Rollback internal database if MindsDB creation fails
                self.session.delete(datasource)
                self.session.commit()
                raise

            return self._datasource_to_response(datasource)

        except DatasourceAlreadyExistsError:
            self.session.rollback()
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating datasource {datasource_data.name}: {str(e)}")
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
            logger.debug(f"Updating datasource {datasource_name} for user {self.user_id}")

            # Get existing datasource
            statement = select(Datasource).where(
                and_(Datasource.name == datasource_name, Datasource.user_id == self.user_id)
            )

            result = self.session.exec(statement)
            datasource = result.first()

            if not datasource:
                raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not found")

            # Store original data for rollback
            original_data = datasource.connection_data.copy() if datasource.connection_data else {}

            # Update fields (using simplified schema)
            if datasource_data.connection_data is not None:
                datasource.connection_data = datasource_data.connection_data

            # Save to internal database first
            self.session.add(datasource)
            self.session.commit()
            self.session.refresh(datasource)

            try:
                # Update in MindsDB if connection data changed
                if datasource_data.connection_data is not None:
                    await self._update_mindsdb_database(datasource)

                logger.info(f"Updated datasource {datasource_name} for user {self.user_id} (synced to MindsDB)")

            except DatasourceServiceError:
                # Rollback internal database if MindsDB update fails
                datasource.connection_data = original_data
                self.session.add(datasource)
                self.session.commit()
                raise

            return self._datasource_to_response(datasource)

        except DatasourceNotFoundError:
            self.session.rollback()
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating datasource {datasource_name}: {str(e)}")
            raise DatasourceServiceError(f"Failed to update datasource: {str(e)}") from None

    async def delete_datasource(self, datasource_name: str, cascade: bool = False) -> None:
        """
        Delete a datasource (hard delete to match MindsDB behavior).

        Args:
            datasource_name: Name of the datasource to delete
            cascade: Whether to remove from all minds that use it

        Raises:
            DatasourceNotFoundError: If datasource doesn't exist
        """
        try:
            logger.debug(f"Deleting datasource {datasource_name} for user {self.user_id} (cascade={cascade})")

            # Get existing datasource
            statement = select(Datasource).where(
                and_(Datasource.name == datasource_name, Datasource.user_id == self.user_id)
            )

            result = self.session.exec(statement)
            datasource = result.first()

            if not datasource:
                raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not found")

            # TODO: If cascade is True, remove from all minds that use this datasource
            if cascade:
                logger.debug(f"Cascade deletion for datasource {datasource_name} - implement mind updates")
            await self._delete_mindsdb_database(datasource_name)

            # Then delete from internal database
            self.session.delete(datasource)
            self.session.commit()

            logger.info(f"Deleted datasource {datasource_name} for user {self.user_id} (removed from MindsDB)")

        except DatasourceNotFoundError:
            self.session.rollback()
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting datasource {datasource_name}: {str(e)}")
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
            logger.debug(f"Testing connection for datasource {datasource_name}")

            # Get datasource to verify it exists
            _ = await self.get_datasource(datasource_name)

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
                return DatasourceConnectionStatus(success=True)
            except Exception as db_error:
                return DatasourceConnectionStatus(
                    success=False, error_message=f"Connection test failed: {str(db_error)}"
                )

        except DatasourceNotFoundError:
            return DatasourceConnectionStatus(success=False, error_message="Datasource not found")
        except Exception as e:
            logger.error(f"Connection test failed for {datasource_name}: {str(e)}")
            return DatasourceConnectionStatus(success=False, error_message=str(e))

    async def _create_mindsdb_database(self, datasource: Datasource) -> None:
        """Create database/integration in MindsDB."""
        try:
            logger.debug(f"Creating MindsDB database for datasource {datasource.name}")

            # Use MindsDB SDK to create database
            databases = self.mindsdb_client.databases

            # Create the database with connection validation
            databases.create(
                name=datasource.name,
                engine=datasource.engine,
                connection_args=datasource.connection_data,
                company_id=self.user_id,
            )

            logger.info(f"Created MindsDB database {datasource.name}")

        except Exception as e:
            logger.error(f"Failed to create MindsDB database {datasource.name}: {str(e)}")
            raise DatasourceServiceError(f"MindsDB database creation failed: {str(e)}") from None

    async def _update_mindsdb_database(self, datasource: Datasource) -> None:
        """Update database/integration in MindsDB by recreating it."""
        try:
            logger.debug(f"Updating MindsDB database for datasource {datasource.name}")

            databases = self.mindsdb_client.databases

            # MindsDB SDK doesn't have update method, so we drop and recreate
            try:
                databases.drop(datasource.name)
                logger.debug(f"Dropped existing MindsDB database {datasource.name}")
            except Exception:
                # Database might not exist, continue with creation
                logger.debug("Datasource not found. Skipping...")

            # Recreate with new parameters
            databases.create(
                name=datasource.name,
                company_id=self.user_id,
                engine=datasource.engine,
                connection_args=datasource.connection_data,
            )

            logger.info(f"Updated MindsDB database {datasource.name}")

        except Exception as e:
            logger.error(f"Failed to update MindsDB database {datasource.name}: {str(e)}")
            raise DatasourceServiceError(f"MindsDB database update failed: {str(e)}") from None

    async def _delete_mindsdb_database(self, datasource_name: str) -> None:
        """Delete database/integration from MindsDB."""
        try:
            logger.debug(f"Deleting MindsDB database {datasource_name}")

            databases = self.mindsdb_client.databases

            # Use databases.drop() method instead of database.drop()
            try:
                databases.drop(datasource_name)
                logger.info(f"Deleted MindsDB database {datasource_name}")
            except Exception:
                # Database might not exist in MindsDB, which is fine
                logger.debug(f"MindsDB database {datasource_name} was not found for deletion")

        except Exception as e:
            logger.error(f"Failed to delete MindsDB database {datasource_name}: {str(e)}")
            raise DatasourceServiceError(f"MindsDB database deletion failed: {str(e)}") from None

    async def _exists_in_mindsdb(self, datasource_name: str) -> bool:
        """Check if datasource exists in MindsDB."""
        try:
            databases = self.mindsdb_client.databases
            database = databases.get(datasource_name)
            return database is not None
        except Exception:
            return False

    def _datasource_to_response(self, datasource: Datasource) -> DatasourceResponse:
        """Convert Datasource model to response object."""
        return DatasourceResponse(
            id=datasource.id,
            name=datasource.name,
            engine=datasource.engine,
            connection_data=datasource.connection_data,
            created_at=datasource.created_on.isoformat() if datasource.created_on else None,
            is_demo=False,
        )

    async def _datasource_to_detailed_response(self, datasource: Datasource) -> DatasourceDetailedResponse:
        """Convert Datasource model to detailed response object with connection status."""
        base_response = self._datasource_to_response(datasource)

        # Get real connection status from MindsDB
        connection_status = await self.test_connection(datasource.name)

        return DatasourceDetailedResponse(**base_response.model_dump(), connection_status=connection_status)
