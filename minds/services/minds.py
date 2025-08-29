"""
Minds service layer for business logic and data operations.

This module contains the MindsService class that handles all business logic
related to mind management, including CRUD operations with internal database storage.
MindsDB is only used for datasource validation, not for minds storage.
"""

from sqlalchemy.orm import selectinload
from sqlmodel import Session, and_, select

from minds.common.logger import setup_logging
from minds.model.datasource import Datasource
from minds.model.mind import Mind
from minds.model.mind_datasource import MindDatasource
from minds.schemas.minds import AddDatasourceRequest, MindCreateRequest, MindResponse, MindUpdateRequest

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


class MindsService:
    """
    Service class for mind management operations.

    This class encapsulates all business logic related to minds,
    including CRUD operations with internal database storage. MindsDB is
    only used for datasource validation, not for storing minds data.
    """

    def __init__(self, session: Session, user_id: str):
        """
        Initialize the minds service.

        Args:
            session (Session): Database session
            user_id (str): Current user ID
        """
        self.session = session
        self.user_id = user_id
        logger.debug(f"MindsService initialized for user {user_id}")

    @classmethod
    def create(cls, session: Session, user_id: str) -> "MindsService":
        """Factory method to create a MindsService instance."""
        return cls(session=session, user_id=user_id)

    async def list_minds(
        self,
        provider: str | None = None,
        is_active: bool | None = None,
        limit: int = 50,
        offset: int = 0,
        with_detailed_data: bool = False,
    ) -> list[MindResponse]:
        """
        List minds for the current user/company with optional filtering and pagination.

        Args:
            provider (Optional[str]): Filter by provider (openai, google, etc.)
            is_active (Optional[bool]): Filter by active status
            limit (int): Maximum number of minds to return (default: 50)
            offset (int): Number of minds to skip (default: 0)
            with_detailed_data (bool): Include detailed datasource base data

        Returns:
            List[MindResponse]: List of mind objects
        """
        try:
            logger.debug(
                f"Listing minds for user {self.user_id} with filters: "
                f"provider={provider}, is_active={is_active}, limit={limit}, offset={offset}"
            )

            # Build query conditions
            conditions = [Mind.user_id == self.user_id]
            if provider is not None:
                conditions.append(Mind.provider == provider)
            # not sure if this is needed initially
            if is_active is not None:
                conditions.append(Mind.is_active == is_active)
            else:
                # Default: only active minds unless explicitly requested
                conditions.append(Mind.is_active)

            statement = (
                select(Mind)
                .options(selectinload(Mind.mind_datasources).selectinload(MindDatasource.datasource))
                .where(and_(*conditions))
                .order_by(Mind.created_on.desc())
                .offset(offset)
                .limit(limit)
            )
            minds = self.session.exec(statement).all()

            minds_list = []
            for mind in minds:
                mind_response = self._mind_to_response(mind, with_detailed_data)
                minds_list.append(mind_response)

            logger.info(f"Retrieved {len(minds_list)} minds for user {self.user_id} (offset={offset}, limit={limit})")
            return minds_list
        except Exception as e:
            logger.error(f"Error listing minds for user {self.user_id}: {str(e)}")
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
            logger.debug(f"Getting mind {mind_name} for user {self.user_id}")

            statement = (
                select(Mind)
                .options(selectinload(Mind.mind_datasources).selectinload(MindDatasource.datasource))
                .where(and_(Mind.name == mind_name, Mind.user_id == self.user_id, Mind.is_active))
            )
            mind = self.session.exec(statement).first()

            if not mind:
                raise MindNotFoundError(f"Mind '{mind_name}' not found")

            mind_response = self._mind_to_response(mind, with_detailed_data)
            logger.info(f"Retrieved mind {mind_name} for user {self.user_id}")
            return mind_response
        except MindNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error getting mind {mind_name}: {str(e)}")
            raise MindsServiceError(f"Failed to get mind: {str(e)}") from None

    async def create_mind(self, mind_data: MindCreateRequest) -> MindResponse:
        """
        Create a new mind.

        Args:
            mind_data (MindCreateRequest): Mind creation data

        Returns:
            MindResponse: Created mind object

        Raises:
            MindAlreadyExistsError: If a mind with the same name already exists
            DatasourceNotFoundError: If any specified datasource doesn't exist
        """
        try:
            logger.debug(f"Creating mind {mind_data.name} for user {self.user_id}")

            # Check if mind already exists in our database
            existing_mind = self.session.exec(
                select(Mind).where(and_(Mind.name == mind_data.name, Mind.user_id == self.user_id, Mind.is_active))
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
                parameters=mind_data.parameters or {},
                is_active=True,
            )

            self.session.add(new_mind)
            self.session.commit()
            self.session.refresh(new_mind)

            # Add datasource relationships if provided
            if mind_data.datasources:
                await self._add_datasources_to_mind(new_mind, mind_data.datasources)

            logger.info(f"Created mind {mind_data.name} for user {self.user_id}")

            return self._mind_to_response(new_mind)

        except (MindAlreadyExistsError, DatasourceNotFoundError):
            self.session.rollback()
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error creating mind {mind_data.name}: {str(e)}")
            raise MindsServiceError(f"Failed to create mind: {str(e)}") from None

    async def update_mind(self, mind_name: str, mind_data: MindUpdateRequest) -> MindResponse:
        """
        Update an existing mind.

        Args:
            mind_name (str): Name of the mind to update
            mind_data (MindUpdateRequest): Updated mind data

        Returns:
            MindResponse: Updated mind object

        Raises:
            MindNotFoundError: If the mind doesn't exist
        """
        try:
            logger.debug(f"Updating mind {mind_name} for user {self.user_id}")

            statement = (
                select(Mind)
                .options(selectinload(Mind.mind_datasources).selectinload(MindDatasource.datasource))
                .where(and_(Mind.name == mind_name, Mind.user_id == self.user_id, Mind.is_active))
            )
            mind = self.session.exec(statement).first()

            if not mind:
                raise MindNotFoundError(f"Mind '{mind_name}' not found")

            # Check if new name conflicts with existing minds
            if mind_data.name and mind_data.name != mind_name:
                existing_mind = self.session.exec(
                    select(Mind).where(and_(Mind.name == mind_data.name, Mind.user_id == self.user_id, Mind.is_active))
                ).first()
                if existing_mind:
                    raise MindAlreadyExistsError(f"Mind with name '{mind_data.name}' already exists")

            # Validate new datasources if provided
            if mind_data.datasources is not None:
                await self._validate_datasources(mind_data.datasources)

            # Update mind fields
            if mind_data.name is not None:
                mind.name = mind_data.name
            if mind_data.provider is not None:
                mind.provider = mind_data.provider
            if mind_data.model_name is not None:
                mind.model_name = mind_data.model_name
            if mind_data.parameters is not None:
                mind.parameters = mind_data.parameters

            # Handle datasource relationships separately
            if mind_data.datasources is not None:
                await self._update_mind_datasources(mind, mind_data.datasources)

            self.session.add(mind)
            self.session.commit()
            self.session.refresh(mind)

            logger.info(f"Updated mind {mind_name} for user {self.user_id}")

            return self._mind_to_response(mind)

        except (MindNotFoundError, MindAlreadyExistsError):
            self.session.rollback()
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating mind {mind_name}: {str(e)}")
            raise MindsServiceError(f"Failed to update mind: {str(e)}") from None

    async def delete_mind(self, mind_name: str, cascade: bool = False) -> bool:
        """
        Delete a mind (soft delete by marking as inactive).

        Args:
            mind_name (str): Name of the mind to delete
            cascade (bool): Whether to delete associated resources (placeholder for future use)

        Returns:
            bool: True if deletion was successful

        Raises:
            MindNotFoundError: If the mind doesn't exist
        """
        try:
            logger.debug(f"Deleting mind {mind_name} for user {self.user_id}")

            statement = select(Mind).where(and_(Mind.name == mind_name, Mind.user_id == self.user_id, Mind.is_active))
            mind = self.session.exec(statement).first()

            if not mind:
                raise MindNotFoundError(f"Mind '{mind_name}' not found")

            if cascade:
                logger.debug(f"Cascade deletion requested for mind {mind_name}")

            mind.is_active = False

            self.session.add(mind)
            self.session.commit()

            logger.info(f"Deleted mind {mind_name} for user {self.user_id}")
            return True

        except MindNotFoundError:
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting mind {mind_name}: {str(e)}")
            raise MindsServiceError(f"Failed to delete mind: {str(e)}") from None

    async def add_datasource_to_mind(self, mind_name: str, datasource_request: AddDatasourceRequest) -> bool:
        """
        Add a datasource to a mind.

        Args:
            mind_name (str): Name of the mind
            datasource_request (AddDatasourceRequest): Datasource to add

        Returns:
            bool: True if addition was successful
        """
        try:
            logger.debug(f"Adding datasource {datasource_request.name} to mind {mind_name}")

            statement = select(Mind).where(and_(Mind.name == mind_name, Mind.user_id == self.user_id, Mind.is_active))
            mind = self.session.exec(statement).first()

            if not mind:
                raise MindNotFoundError(f"Mind '{mind_name}' not found")

            # Validate datasource exists
            await self._validate_datasources([datasource_request.name])

            # Check connection if requested
            if datasource_request.check_connection:
                await self._check_datasource_connection(datasource_request.name)

            # Find the datasource in our database
            datasource = self.session.exec(
                select(Datasource).where(
                    and_(Datasource.name == datasource_request.name, Datasource.user_id == self.user_id)
                )
            ).first()

            if not datasource:
                raise DatasourceNotFoundError(f"Datasource '{datasource_request.name}' not found")

            # Check if relationship already exists
            existing_relationship = self.session.exec(
                select(MindDatasource).where(
                    and_(MindDatasource.mind_id == mind.id, MindDatasource.datasource_id == datasource.id)
                )
            ).first()

            if existing_relationship:
                logger.info(f"Datasource {datasource_request.name} already linked to mind {mind_name}")
                return True

            # Create new relationship
            mind_datasource = MindDatasource(
                mind_id=mind.id, datasource_id=datasource.id, purpose=getattr(datasource_request, "purpose", None)
            )

            self.session.add(mind_datasource)
            self.session.commit()

            logger.info(f"Added datasource {datasource_request.name} to mind {mind_name}")
            return True

        except (MindNotFoundError, DatasourceNotFoundError):
            self.session.rollback()
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error adding datasource to mind: {str(e)}")
            raise MindsServiceError(f"Failed to add datasource: {str(e)}") from None

    async def remove_datasource_from_mind(self, mind_name: str, datasource_name: str) -> bool:
        """
        Remove a datasource from a mind.

        Args:
            mind_name (str): Name of the mind
            datasource_name (str): Name of the datasource to remove

        Returns:
            bool: True if removal was successful
        """
        try:
            logger.debug(f"Removing datasource {datasource_name} from mind {mind_name}")

            statement = select(Mind).where(and_(Mind.name == mind_name, Mind.user_id == self.user_id, Mind.is_active))
            mind = self.session.exec(statement).first()

            if not mind:
                raise MindNotFoundError(f"Mind '{mind_name}' not found")

            # Find the datasource
            datasource = self.session.exec(
                select(Datasource).where(and_(Datasource.name == datasource_name, Datasource.user_id == self.user_id))
            ).first()

            if not datasource:
                raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not found")

            # Find and remove the relationship
            relationship = self.session.exec(
                select(MindDatasource).where(
                    and_(MindDatasource.mind_id == mind.id, MindDatasource.datasource_id == datasource.id)
                )
            ).first()

            if not relationship:
                raise DatasourceNotFoundError(f"Datasource '{datasource_name}' not linked to mind '{mind_name}'")

            self.session.delete(relationship)
            self.session.commit()

            logger.info(f"Removed datasource {datasource_name} from mind {mind_name}")
            return True

        except (MindNotFoundError, DatasourceNotFoundError):
            self.session.rollback()
            raise
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error removing datasource from mind: {str(e)}")
            raise MindsServiceError(f"Failed to remove datasource: {str(e)}") from None

    def _mind_to_response(self, mind: Mind, with_detailed_data: bool = False) -> MindResponse:
        """Convert Mind database model to MindResponse object."""
        # Get linked datasources through the many-to-many relationship
        datasources = [relationship.datasource.name for relationship in mind.mind_datasources]

        # TODO: add detailed datasource data if with_detailed_data is True, this is the
        # actual DATA value in the integrations e.g without password which should be hashed
        # {"user": "", "password": "", "host": ".com", "port": "5432", "database": "demo", "schema": "demo_data"}
        return MindResponse(
            name=mind.name,
            model_name=mind.model_name,
            provider=mind.provider,
            parameters=mind.parameters or {},
            datasources=datasources,
            created_at=str(mind.created_on) if mind.created_on else "",
            updated_at=str(mind.modified_on) if mind.modified_on else "",
        )

    async def _validate_datasources(self, datasource_names: list[str]) -> None:
        """
        Validate that all datasources exist using MindsDB.

        Note: MindsDB is still used for datasource validation since datasources
        are managed by MindsDB, not stored in our internal database.
        """
        for datasource_name in datasource_names:
            try:
                # Check if datasource exists in MindsDB
                # Note: Replace with actual MindsDB SDK call
                logger.debug(f"Validating datasource: {datasource_name}")
            except Exception as e:
                logger.error(f"Error validating datasource {datasource_name}: {str(e)}")
                raise DatasourceNotFoundError(f"Datasource '{datasource_name}' validation failed") from None

    async def _check_datasource_connection(self, datasource_name: str) -> None:
        """
        Check if datasource connection is valid using MindsDB.

        Note: MindsDB is still used for connection testing since datasources
        are managed by MindsDB.
        """
        try:
            # Test datasource connection via MindsDB
            # Note: Replace with actual MindsDB SDK call when implementing
            logger.debug(f"Checking connection for datasource: {datasource_name}")
        except Exception as e:
            logger.error(f"Error checking datasource connection {datasource_name}: {str(e)}")
            raise MindsServiceError(f"Datasource connection check failed: {str(e)}") from None

    async def _add_datasources_to_mind(self, mind: Mind, datasource_names: list[str]) -> None:
        """
        Add multiple datasources to a mind by creating MindDatasource relationships.

        Args:
            mind (Mind): The mind to add datasources to
            datasource_names (list[str]): List of datasource names to add
        """
        for datasource_name in datasource_names:
            try:
                # Find the datasource
                datasource = self.session.exec(
                    select(Datasource).where(
                        and_(Datasource.name == datasource_name, Datasource.user_id == self.user_id)
                    )
                ).first()

                if not datasource:
                    logger.warning(f"Datasource '{datasource_name}' not found in database, skipping")
                    continue

                # Check if relationship already exists
                existing_relationship = self.session.exec(
                    select(MindDatasource).where(
                        and_(MindDatasource.mind_id == mind.id, MindDatasource.datasource_id == datasource.id)
                    )
                ).first()

                if existing_relationship:
                    logger.debug(f"Datasource {datasource_name} already linked to mind {mind.name}")
                    continue

                # Create new relationship
                mind_datasource = MindDatasource(mind_id=mind.id, datasource_id=datasource.id)

                self.session.add(mind_datasource)
                logger.debug(f"Added datasource {datasource_name} to mind {mind.name}")

            except Exception as e:
                logger.error(f"Error adding datasource {datasource_name} to mind {mind.name}: {str(e)}")
                # Continue with other datasources even if one fails
                continue

        # Commit all relationships at once
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error committing datasource relationships: {str(e)}")
            raise

    async def _update_mind_datasources(self, mind: Mind, new_datasource_names: list[str]) -> None:
        """
        Update the datasources associated with a mind by replacing all relationships.

        Args:
            mind (Mind): The mind to update datasources for
            new_datasource_names (list[str]): New list of datasource names
        """
        try:
            # Remove all existing relationships
            existing_relationships = self.session.exec(
                select(MindDatasource).where(MindDatasource.mind_id == mind.id)
            ).all()

            for relationship in existing_relationships:
                self.session.delete(relationship)

            # Add new relationships
            await self._add_datasources_to_mind(mind, new_datasource_names)

            logger.debug(f"Updated datasources for mind {mind.name}")

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error updating datasources for mind {mind.name}: {str(e)}")
            raise
