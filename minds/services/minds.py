"""
Minds service layer for business logic and data operations.

This module contains the MindsService class that handles all business logic
related to mind management, including CRUD operations with internal database storage.
MindsDB is only used for datasource validation, not for minds storage.
"""

from datetime import datetime, timezone

from mindsdb_sdk.server import Server
from sqlalchemy.orm import selectinload, with_loader_criteria
from sqlmodel import Session, and_, select

from minds.client.prefect import PrefectClient
from minds.common.logger import setup_logging
from minds.model.datasource import Datasource
from minds.model.mind import Mind
from minds.model.mind_datasource import DataCatalogStatus, MindDatasource
from minds.schemas.minds import (
    DatasourceConfig,
    DetailedDatasourceConfig,
    MindCreateRequest,
    MindResponse,
    MindUpdateRequest,
)
from minds.services.data_catalog.data_catalog_loader import DataCatalogLoader

# Set up logging
logger = setup_logging()


class MindsServiceError(Exception):
    """Base exception for minds service errors."""

    pass


class MindNotFoundError(MindsServiceError):
    """Raised when a mind is not found."""

    pass


class MindAlreadyExistsError(MindsServiceError):
    """Raised when trying to create a mind that already exists."""

    pass


class DatasourceNotFoundError(MindsServiceError):
    """Raised when a datasource is not found."""

    pass


class DatasourceTableNotFoundError(MindsServiceError):
    """Raised when a datasource table is not found."""

    pass


class MindsService:
    """
    Service class for mind management operations.

    This class encapsulates all business logic related to minds,
    including CRUD operations with internal database storage. MindsDB is
    only used for datasource validation, not for storing minds data.
    """

    def __init__(self, session: Session, mindsdb_client: Server, user_id: str, tenant_id: str):
        """
        Initialize the minds service.

        Args:
            session (Session): Database session
            user_id (str): Current user ID
        """
        self.session = session
        self.mindsdb_client = mindsdb_client
        self.user_id = user_id
        self.tenant_id = tenant_id
        logger.debug(f"MindsService initialized for user {user_id} and tenant {tenant_id}")

    async def list_minds(
        self,
        provider: str | None = None,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0,
        with_detailed_data: bool = False,
    ) -> list[MindResponse]:
        """
        List minds for the current user/company with optional filtering and pagination.

        Args:
            provider (Optional[str]): Filter by provider (openai, google, etc.)
            include_deleted (Optional[bool]): Filter by deleted status
            limit (int): Maximum number of minds to return (default: 50)
            offset (int): Number of minds to skip (default: 0)
            with_detailed_data (bool): Include detailed datasource base data

        Returns:
            List[MindResponse]: List of mind objects
        """
        try:
            logger.debug(
                f"Listing minds for user {self.user_id} in tenant {self.tenant_id} with filters: "
                f"provider={provider}, include_deleted={include_deleted}, limit={limit}, offset={offset}"
            )

            # Build query conditions
            conditions = [Mind.user_id == self.user_id, Mind.tenant_id == self.tenant_id]
            if provider is not None:
                conditions.append(Mind.provider == provider)
            # not sure if this is needed initially
            if not include_deleted:
                conditions.append(Mind.deleted_at.is_(None))

            statement = (
                select(Mind)
                .where(and_(*conditions))
                .order_by(Mind.created_at.desc())
                .offset(offset)
                .limit(limit)
                .options(
                    selectinload(Mind.mind_datasources.and_(MindDatasource.deleted_at.is_(None))).selectinload(
                        MindDatasource.datasource
                    )
                )
            )

            minds = self.session.exec(statement).all()

            minds_list = []
            for mind in minds:
                mind_response = await self._mind_to_response(mind, with_detailed_data=with_detailed_data)
                minds_list.append(mind_response)

            logger.info(
                f"Retrieved {len(minds_list)} minds "
                f"for user {self.user_id} in tenant {self.tenant_id} (offset={offset}, limit={limit})"
            )
            return minds_list
        except Exception as e:
            logger.error(f"Error listing minds for user {self.user_id} in tenant {self.tenant_id}: {str(e)}")
            raise MindsServiceError(f"Failed to list minds: {str(e)}") from None

    async def get_mind(self, mind_name: str, with_detailed_data: bool = False) -> MindResponse:
        """
        Get a specific mind by name.

        Args:
            mind_name (str): Name of the mind
            with_detailed_data (bool): Include detailed datasource/knowledge base data

        Returns:
            MindResponse: Mind object

        Raises:
            MindNotFoundError: If the mind doesn't exist
        """
        try:
            logger.debug(f"Getting mind {mind_name} for user {self.user_id} in tenant {self.tenant_id}")

            mind = await self._get_mind_with_datasources(mind_name)

            if not mind:
                raise MindNotFoundError(f"Mind '{mind_name}' not found")

            mind_response = await self._mind_to_response(mind, with_detailed_data=with_detailed_data)
            logger.info(f"Retrieved mind {mind_name} for user {self.user_id} in tenant {self.tenant_id}")
            return mind_response
        except MindNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error getting mind {mind_name} for user {self.user_id} in tenant {self.tenant_id}: {str(e)}")
            raise MindsServiceError(f"Failed to get mind: {str(e)}") from None

    async def get_mind_model(self, mind_name: str) -> Mind:
        """
        Get a specific mind by name as a Mind database model object.

        This method is intended for internal use when you need the actual Mind
        database model rather than the API response schema.

        Args:
            mind_name (str): Name of the mind

        Returns:
            Mind: Mind database model object

        Raises:
            MindNotFoundError: If the mind doesn't exist
        """
        try:
            logger.debug(f"Getting mind model {mind_name} for user {self.user_id} in tenant {self.tenant_id}")

            mind = await self._get_mind(mind_name)

            if not mind:
                raise MindNotFoundError(f"Mind '{mind_name}' not found")

            logger.info(f"Retrieved mind model {mind_name} for user {self.user_id} in tenant {self.tenant_id}")
            return mind
        except MindNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error getting mind model {mind_name} for user {self.user_id} in tenant {self.tenant_id}: {str(e)}"
            )
            raise MindsServiceError(f"Failed to get mind model: {str(e)}") from None

    async def create_mind(self, mind_data: MindCreateRequest, data_catalog_loader: DataCatalogLoader) -> MindResponse:
        """
        Create a new mind.

        Args:
            mind_data (MindCreateRequest): Mind creation data
            data_catalog_loader (DataCatalogLoader): Data catalog loader

        Returns:
            MindResponse: Created mind object

        Raises:
            MindAlreadyExistsError: If a mind with the same name already exists
            DatasourceNotFoundError: If any specified datasource doesn't exist
        """
        try:
            logger.debug(f"Creating mind {mind_data.name} for user {self.user_id} in tenant {self.tenant_id}")

            # Check if mind already exists in our database
            existing_mind = self.session.exec(
                select(Mind).where(
                    and_(
                        Mind.name == mind_data.name,
                        Mind.user_id == self.user_id,
                        Mind.tenant_id == self.tenant_id,
                        Mind.deleted_at.is_(None),
                    )
                )
            ).first()

            if existing_mind:
                raise MindAlreadyExistsError(f"Mind '{mind_data.name}' already exists")

            # Validate datasources exist if provided
            if mind_data.datasources:
                await self._validate_datasources(mind_data.datasources)

            new_mind = Mind(
                name=mind_data.name,
                provider=mind_data.provider,
                model_name=mind_data.model_name or "gpt-4o",  # Default model
                user_id=self.user_id,
                tenant_id=self.tenant_id,
                parameters=mind_data.parameters or {},
                deleted_at=None,
            )

            self.session.add(new_mind)
            self.session.commit()
            self.session.refresh(new_mind)

            # Add datasource relationships if provided
            if mind_data.datasources:
                await self._add_datasources_to_mind(new_mind, mind_data.datasources, data_catalog_loader)

            logger.info(f"Created mind {mind_data.name} for user {self.user_id} in tenant {self.tenant_id}")

            return await self._mind_to_response(new_mind, datasource_configs=mind_data.datasources)
        except (MindAlreadyExistsError, DatasourceNotFoundError, DatasourceTableNotFoundError):
            self.session.rollback()
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(
                f"Error creating mind {mind_data.name} for user {self.user_id} in tenant {self.tenant_id}: {str(e)}"
            )
            raise MindsServiceError(f"Failed to create mind: {str(e)}") from None

    async def update_mind(
        self, mind_name: str, mind_data: MindUpdateRequest, data_catalog_loader: DataCatalogLoader
    ) -> MindResponse:
        """
        Update an existing mind.

        Args:
            mind_name (str): Name of the mind to update
            mind_data (MindUpdateRequest): Updated mind data
            data_catalog_loader (DataCatalogLoader): Data catalog loader

        Returns:
            MindResponse: Updated mind object

        Raises:
            MindNotFoundError: If the mind doesn't exist
        """
        try:
            logger.debug(f"Updating mind {mind_name} for user {self.user_id} in tenant {self.tenant_id}")

            mind = await self._get_mind_with_datasources(mind_name)

            if not mind:
                raise MindNotFoundError(f"Mind '{mind_name}' not found")

            # Check if new name conflicts with existing minds
            if mind_data.name and mind_data.name != mind_name:
                existing_mind = await self._get_mind(mind_data.name)
                if existing_mind:
                    raise MindAlreadyExistsError(f"Mind with name '{mind_data.name}' already exists")

            datasource_configs = mind_data.datasources

            if datasource_configs:
                # Validate new datasources if provided
                await self._validate_datasources(datasource_configs)

            # Cancel any running data catalog loader flows for the mind
            await self._cancel_data_catalog_loader_flows_for_mind(mind)
            # Update the datasources associated with the mind
            await self._update_mind_datasources(mind, mind_data.datasources, data_catalog_loader)

            # Update mind fields
            if mind_data.name is not None:
                mind.name = mind_data.name
            if mind_data.provider is not None:
                mind.provider = mind_data.provider
            if mind_data.model_name is not None:
                mind.model_name = mind_data.model_name
            if mind_data.parameters is not None:
                mind.parameters = mind_data.parameters

            self.session.add(mind)
            self.session.commit()
            self.session.refresh(mind)

            logger.info(f"Updated mind {mind_name} for user {self.user_id} in tenant {self.tenant_id}")

            return await self._mind_to_response(mind, datasource_configs=datasource_configs)
        except (MindNotFoundError, MindAlreadyExistsError, DatasourceNotFoundError, DatasourceTableNotFoundError):
            self.session.rollback()
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(
                f"Error updating mind {mind_name} for user {self.user_id} in tenant {self.tenant_id}: {str(e)}"
            )
            raise MindsServiceError(f"Failed to update mind: {str(e)}") from None

    async def delete_mind(self, mind_name: str, cascade: bool = False) -> bool:
        """
        Delete a mind (soft delete).

        Args:
            mind_name (str): Name of the mind to delete
            cascade (bool): Whether to delete associated resources (placeholder for future use)

        Returns:
            bool: True if deletion was successful

        Raises:
            MindNotFoundError: If the mind doesn't exist
        """
        try:
            logger.debug(f"Deleting mind {mind_name} for user {self.user_id} in tenant {self.tenant_id}")

            mind = await self._get_mind_with_datasources(mind_name)

            if not mind:
                raise MindNotFoundError(f"Mind '{mind_name}' not found")

            # TODO: If cascade is True, delete all datasources associated with the mind
            if cascade:
                logger.debug(f"Cascade deletion requested for mind {mind_name} - implement datasource deletion")

            # Cancel any running data catalog loader flows for the mind
            await self._cancel_data_catalog_loader_flows_for_mind(mind)

            mind.deleted_at = datetime.now(timezone.utc)

            self.session.add(mind)
            self.session.commit()

            logger.info(f"Deleted mind {mind_name} for user {self.user_id} in tenant {self.tenant_id}")
            return True
        except MindNotFoundError:
            self.session.rollback()
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(
                f"Error deleting mind {mind_name} for user {self.user_id} in tenant {self.tenant_id}: {str(e)}"
            )
            raise MindsServiceError(f"Failed to delete mind: {str(e)}") from None

    async def _get_mind(self, mind_name: str) -> Mind:
        """Utility function to get a specific mind by name."""
        statement = select(Mind).where(
            and_(
                Mind.name == mind_name,
                Mind.user_id == self.user_id,
                Mind.tenant_id == self.tenant_id,
                Mind.deleted_at.is_(None),
            )
        )
        return self.session.exec(statement).first()

    async def _get_mind_with_datasources(self, mind_name: str) -> Mind:
        """Utility function to get a specific mind by name with datasources eagerly loaded."""
        statement = (
            select(Mind)
            .where(
                and_(
                    Mind.name == mind_name,
                    Mind.user_id == self.user_id,
                    Mind.deleted_at.is_(None),
                    Mind.tenant_id == self.tenant_id,
                )
            )
            .options(
                selectinload(Mind.mind_datasources),
                with_loader_criteria(
                    MindDatasource,
                    lambda cls: cls.deleted_at.is_(None),
                    include_aliases=True,
                ),
            )
        )
        return self.session.exec(statement).first()

    async def _mind_to_response(
        self, mind: Mind, datasource_configs: list[DatasourceConfig] = None, with_detailed_data: bool = False
    ) -> MindResponse:
        """
        Convert Mind database model to MindResponse object.

        Args:
            mind (Mind): Mind database model
            datasource_configs (list[DatasourceConfig]): Optional explicit datasource configs
            with_detailed_data (bool): Include detailed datasource/knowledge base data

        Returns:
            MindResponse: Mind response object
        """
        # If datasource configs are explicitly provided (e.g. on creation), use those
        # On create and update, the relationships may not be fully populated yet
        if datasource_configs is not None:
            datasources = datasource_configs
        # Get linked datasources through the many-to-many relationship
        else:
            if with_detailed_data:
                datasources = []
                for relationship in mind.mind_datasources:
                    status = await relationship.status
                    datasources.append(
                        DetailedDatasourceConfig(
                            name=relationship.datasource.name,
                            engine=relationship.datasource.engine,
                            description=relationship.datasource.description,
                            connection_data=relationship.datasource.connection_data,
                            tables=[
                                mind_datasource_table.table.name
                                for mind_datasource_table in relationship.mind_datasource_tables
                            ],
                            status=status,
                            created_at=str(relationship.datasource.created_at),
                            modified_at=str(relationship.datasource.modified_at),
                        )
                    )
            else:
                datasources = []
                for relationship in mind.mind_datasources:
                    status = await relationship.status
                    datasources.append(
                        DatasourceConfig(
                            name=relationship.datasource.name,
                            tables=[
                                mind_datasource_table.table.name
                                for mind_datasource_table in relationship.mind_datasource_tables
                            ],
                            status=status,
                        )
                    )

        return MindResponse(
            name=mind.name,
            model_name=mind.model_name,
            provider=mind.provider,
            parameters=mind.parameters or {},
            datasources=datasources,
            created_at=mind.created_at.isoformat(),
            modified_at=mind.modified_at.isoformat(),
        )

    async def _validate_datasources(self, datasource_configs: list[DatasourceConfig]) -> None:
        """
        Validate that all datasources (and tables if specified) exist in the internal database as well as the MindsDB.
        """
        for datasource_config in datasource_configs:
            datasource_name = datasource_config.name
            datasource_tables = datasource_config.tables
            try:
                logger.debug(
                    f"Validating datasource {datasource_name} for user {self.user_id} in tenant {self.tenant_id}"
                )
                datasource = self.session.exec(
                    select(Datasource).where(
                        and_(
                            Datasource.name == datasource_name,
                            Datasource.user_id == self.user_id,
                            Datasource.tenant_id == self.tenant_id,
                            Datasource.deleted_at.is_(None),
                        )
                    )
                ).first()

                if not datasource:
                    raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not found in database")

                if datasource_tables:
                    try:
                        available_tables = self.mindsdb_client.databases.get(datasource_name).tables.list()
                    except AttributeError:
                        raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not found in MindsDB") from None
                    available_table_names = [table.name for table in available_tables]

                    missing_tables = [table for table in datasource_tables if table not in available_table_names]
                    if missing_tables:
                        raise DatasourceTableNotFoundError(
                            f"Tables {missing_tables} not found in datasource '{datasource_name}'. "
                            f"Available tables: {available_table_names}"
                        )
            except (DatasourceNotFoundError, DatasourceTableNotFoundError) as e:
                logger.error(
                    f"Error validating datasource {datasource_name} "
                    f"for user {self.user_id} in tenant {self.tenant_id}: {str(e)}"
                )
                raise

    async def _add_datasources_to_mind(
        self, mind: Mind, datasource_configs: list[DatasourceConfig], data_catalog_loader: DataCatalogLoader
    ) -> None:
        """
        Add multiple datasources to a mind by creating MindDatasource relationships.

        Args:
            mind (Mind): The mind to add datasources to
            datasource_names (list[str]): List of datasource names to add
        """
        for datasource_config in datasource_configs:
            try:
                logger.debug(
                    f"Adding datasource {datasource_config.name} to mind {mind.name} "
                    f"for user {self.user_id} in tenant {self.tenant_id}"
                )
                datasource_name = datasource_config.name
                table_names = datasource_config.tables
                # Find the datasource
                datasource = self.session.exec(
                    select(Datasource).where(
                        and_(
                            Datasource.name == datasource_name,
                            Datasource.user_id == self.user_id,
                            Datasource.tenant_id == self.tenant_id,
                            Datasource.deleted_at.is_(None),
                        )
                    )
                ).first()

                if not datasource:
                    logger.warning(
                        f"Datasource '{datasource_name}' not found in database "
                        f"for user {self.user_id} in tenant {self.tenant_id}, skipping"
                    )
                    continue

                # Check if relationship already exists
                existing_relationship = self.session.exec(
                    select(MindDatasource).where(
                        and_(
                            MindDatasource.mind_id == mind.id,
                            MindDatasource.datasource_id == datasource.id,
                            MindDatasource.tenant_id == self.tenant_id,
                            MindDatasource.deleted_at.is_(None),
                        )
                    )
                ).first()

                if existing_relationship:
                    logger.debug(
                        f"Datasource {datasource_name} already linked to mind {mind.name} "
                        f"for user {self.user_id} in tenant {self.tenant_id}"
                    )
                    continue

                # Create new relationship
                mind_datasource = MindDatasource(
                    tenant_id=self.tenant_id,
                    mind_id=mind.id,
                    datasource_id=datasource.id,
                )

                self.session.add(mind_datasource)
                self.session.commit()
                self.session.refresh(mind_datasource)
                logger.debug(
                    f"Added datasource {datasource_name} to mind {mind.name} "
                    f"for user {self.user_id} in tenant {self.tenant_id}"
                )

                logger.debug(
                    f"Loading datasource {datasource_name} to the data catalog "
                    f"for user {self.user_id} in tenant {self.tenant_id}"
                )

                logger.debug(
                    f"Loading datasource {datasource_name} to the data catalog "
                    f"for user {self.user_id} in tenant {self.tenant_id}"
                )
                # No exception handling here - the code executes in the Prefect flow.
                await data_catalog_loader.load(mind_datasource, table_names)
            except Exception as e:
                logger.error(
                    f"Error adding datasource {datasource_name} to mind {mind.name} "
                    f"for user {self.user_id} in tenant {self.tenant_id}: {str(e)}"
                )

        # Commit all relationships at once
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error committing datasource relationships: {str(e)}")
            raise

    async def _update_mind_datasources(
        self, mind: Mind, new_datasource_configs: list[DatasourceConfig], data_catalog_loader: DataCatalogLoader
    ) -> None:
        """
        Update the datasources associated with a mind by replacing all relationships.

        Args:
            mind (Mind): The mind to update datasources for
            new_datasource_configs (list[DatasourceConfig]): New list of datasource configs
        """
        try:
            # Remove all existing relationships - soft delete only.
            for relationship in mind.mind_datasources:
                relationship.deleted_at = datetime.now(timezone.utc)

            self.session.flush()

            # Add new relationships
            if new_datasource_configs:
                await self._add_datasources_to_mind(mind, new_datasource_configs, data_catalog_loader)

            logger.debug(f"Updated datasources for mind {mind.name}")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating datasources for mind {mind.name}: {str(e)}")
            raise

    async def _cancel_data_catalog_loader_flows_for_mind(self, mind: Mind) -> None:
        """
        Cancel all data catalog loader flows for a mind.

        Args:
            mind (Mind): The mind to cancel data catalog loader flows for
        """
        prefect_client = PrefectClient()
        for relationship in mind.mind_datasources:
            status = await relationship.status
            if status == DataCatalogStatus.LOADING and relationship.flow_run_id:
                await prefect_client.cancel_flow_run(relationship.flow_run_id)
