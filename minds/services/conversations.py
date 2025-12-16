"""
Conversations service for managing conversation operations.
"""

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from sqlalchemy.orm import selectinload, with_loader_criteria
from sqlmodel import Session, and_, func, select

from minds.common.logger import setup_logging
from minds.model.conversation import Conversation
from minds.model.message import Message
from minds.schemas.chat import Role
from minds.schemas.conversations import ConversationCreateRequest, ConversationMetadata, ConversationResponse

logger = setup_logging()


class ConversationAlreadyExistsError(Exception):
    """Exception for when a conversation already exists."""
    pass


class ConversationNotFoundError(Exception):
    """Exception for when a conversation is not found."""
    pass


class ConversationsServiceError(Exception):
    """Base exception for conversations service errors."""
    pass


class ConversationsService:
    """
    Service class for conversation management operations.
    """

    def __init__(self, session: Session, user_id: str, tenant_id: str):
        """
        Initialize the conversations service.
        """
        self.session = session
        self.user_id = user_id
        self.tenant_id = tenant_id
        logger.debug(f"ConversationsService initialized for user {user_id} and tenant {tenant_id}")

    async def list_conversations(
        self,
        topic: str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
        include_total: bool = False,
        sort_by: Literal["created_at", "updated_at"] | None = None,
        sort_order: Literal["asc", "desc"] = "desc",
    ) -> list[Conversation]:
        """
        List conversations for the current user/company with optional filtering, pagination, sorting, and total count.

        Args:
            topic: Topic of the conversation to filter by.
            include_deleted: Whether to include deleted conversations.
            limit: Maximum number of conversations to return.
            offset: Number of conversations to skip.
            include_total: Whether to include the total count of conversations.
            sort_by: Field to sort by.
            sort_order: Order to sort by.

        Returns:
            list[ConversationResponse]: List of conversations.
            tuple[list[ConversationResponse], int]: List of conversations and total count if include_total is True.

        Raises:
            ConversationsServiceError: If there is an error listing conversations.
        """
        logger.debug(f"Listing conversations for user {self.user_id} and tenant {self.tenant_id} with filters: "
                     f"topic={topic}, include_deleted={include_deleted}, limit={limit}, offset={offset}, "
                     f"include_total={include_total}, sort_by={sort_by}, sort_order={sort_order}")

        try:
            # Build query conditions
            conditions = [Conversation.user_id == self.user_id, Conversation.tenant_id == self.tenant_id]
            if topic is not None:
                conditions.append(Conversation.topic.ilike(f"%{topic}%"))
            if not include_deleted:
                conditions.append(Conversation.deleted_at.is_(None))

            # Calculate total count if requested
            total_count = None
            if include_total:
                # For count, we don't need joins or options
                count_statement = select(func.count(func.distinct(Conversation.id))).select_from(Conversation).where(and_(*conditions))
                total_count = self.session.exec(count_statement).one()

            # Determine sort field and order
            sort_field = Conversation.created_at  # default
            if sort_by:
                try:
                    sort_field = getattr(Conversation, sort_by)
                except AttributeError:
                    error_msg = f"Invalid sort_by field: {sort_by}"
                    logger.warning(error_msg)
                    raise

            order_by = sort_field.desc() if sort_order == "desc" else sort_field.asc()

            statement = select(Conversation).where(and_(*conditions)).order_by(order_by).offset(offset).limit(limit)

            conversations = self.session.exec(statement).all()

            conversations_list = []
            for conversation in conversations:
                conversation_response = await self._conversation_to_response(conversation)
                conversations_list.append(conversation_response)

            logger.info(f"Retrieved {len(conversations_list)} conversations for user {self.user_id} and tenant {self.tenant_id} (offset={offset}, limit={limit})")

            if include_total:
                return conversations_list, total_count
            return conversations_list
        except Exception as e:
            logger.error(f"Error listing conversations for user {self.user_id} in tenant {self.tenant_id}: {str(e)}")
            raise ConversationsServiceError(f"Failed to list conversations: {str(e)}") from None

    async def get_conversation(self, conversation_id: UUID) -> ConversationResponse:
        """
        Get a conversation by ID.

        Args:
            conversation_id: ID of the conversation to get.

        Returns:
            ConversationResponse: Conversation.

        Raises:
            ConversationNotFoundError: If conversation with the given ID does not exist.
            ConversationsServiceError: If there is an error getting the conversation.
        """
        logger.debug(f"Getting conversation {conversation_id} for user {self.user_id} and tenant {self.tenant_id}")

        try:
            conversation = await self._get_conversation(conversation_id)

            return await self._conversation_to_response(conversation)
        except ConversationNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error getting conversation {conversation_id} for user {self.user_id} in tenant {self.tenant_id}: {str(e)}")
            raise ConversationsServiceError(f"Failed to get conversation: {str(e)}") from None

    async def create_conversation(self, conversation_data: ConversationCreateRequest) -> ConversationResponse:
        """
        Create a new conversation.

        Args:
            conversation_data: Conversation data to create.

        Returns:
            ConversationResponse: Created conversation.

        Raises:
            ConversationAlreadyExistsError: If conversation with the same topic already exists.
            ConversationsServiceError: If there is an error creating the conversation.
        """
        logger.debug(f"Creating conversation {conversation_data.metadata.topic} for user {self.user_id} and tenant {self.tenant_id}")

        try:
            # Check if conversation already exists
            existing_conversation = self.session.exec(
                select(Conversation).where(
                    and_(
                        Conversation.topic == conversation_data.metadata.topic,
                        Conversation.user_id == self.user_id,
                        Conversation.tenant_id == self.tenant_id,
                    )
                )
            ).first()

            if existing_conversation:
                raise ConversationAlreadyExistsError(f"Conversation with topic '{conversation_data.metadata.topic}' already exists")

            new_conversation = Conversation(
                topic=conversation_data.metadata.topic,
                user_id=self.user_id,
                tenant_id=self.tenant_id,
            )

            self.session.add(new_conversation)
            self.session.flush()

            if conversation_data.items:
                new_messages = []
                for item in conversation_data.items:
                    new_message = Message(
                        tenant_id=self.tenant_id,
                        conversation_id=new_conversation.id,
                        role=item.role,
                        content=item.content,
                    )
                    new_messages.append(new_message)
                self.session.add_all(new_messages)

            self.session.commit()

            logger.info(f"Created conversation {conversation_data.metadata.topic} for user {self.user_id} and tenant {self.tenant_id}")

            return await self._conversation_to_response(new_conversation)
        except ConversationAlreadyExistsError:
            self.session.rollback()
            raise
        except Exception as e:
            logger.error(f"Error creating conversation {conversation_data.metadata.topic} for user {self.user_id} in tenant {self.tenant_id}: {str(e)}")
            self.session.rollback()
            raise ConversationsServiceError(f"Failed to create conversation: {str(e)}") from None

    async def delete_conversation(self, conversation_id: UUID) -> None:
        """
        Delete a conversation.

        Args:
            conversation_id: ID of the conversation to delete.

        Returns:
            None.

        Raises:
            ConversationNotFoundError: If conversation with the given ID does not exist.
            ConversationsServiceError: If there is an error deleting the conversation.
        """
        logger.debug(f"Deleting conversation {conversation_id} for user {self.user_id} and tenant {self.tenant_id}")

        try:
            conversation = await self._get_conversation(conversation_id)

            conversation.deleted_at = datetime.now(timezone.utc)
            self.session.add(conversation)
            self.session.commit()

            logger.info(f"Deleted conversation {conversation_id} for user {self.user_id} and tenant {self.tenant_id}")
        except ConversationNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id} for user {self.user_id} in tenant {self.tenant_id}: {str(e)}")
            raise ConversationsServiceError(f"Failed to delete conversation: {str(e)}") from None

    async def get_conversation_model_with_messages(self, conversation_id: UUID) -> Conversation:
        """
        Get a conversation by ID with its messages eagerly loaded.

        This method is intended for internal use when you need the actual Conversation database model rather than the API response schema.

        Args:
            conversation_id: ID of the conversation to get.

        Returns:
            Conversation: Conversation with its messages.

        Raises:
            ConversationNotFoundError: If conversation with the given ID does not exist.
        """
        logger.debug(f"Getting conversation {conversation_id} with messages for user {self.user_id} and tenant {self.tenant_id}")

        try:
            statement = (
                select(Conversation)
                .where(
                    and_(
                        Conversation.id == conversation_id,
                        Conversation.deleted_at.is_(None),
                        Conversation.user_id == self.user_id,
                        Conversation.tenant_id == self.tenant_id,
                    )
                )
                .options(
                    selectinload(Conversation.messages),
                    with_loader_criteria(
                        Message,
                        lambda cls: cls.deleted_at.is_(None),
                        include_aliases=True,
                    ),
                )
            )
            conversation = self.session.exec(statement).first()
            if not conversation:
                raise ConversationNotFoundError(f"Conversation with ID '{conversation_id}' not found")
            return conversation
        except ConversationNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error getting conversation {conversation_id} with messages for user {self.user_id} in tenant {self.tenant_id}: {str(e)}")
            raise ConversationsServiceError(f"Failed to get conversation with messages: {str(e)}") from None

    async def add_message_to_conversation(self, conversation_id: UUID, role: Role, content: str) -> None:
        """
        Add a message to a conversation.

        Args:
            conversation_id: ID of the conversation to add the message to.
            role: Role of the message.
            content: Content of the message.
        """
        logger.debug(f"Adding message to conversation {conversation_id} for user {self.user_id} and tenant {self.tenant_id} with role {role} and content {content}")\

        # Check if conversation exists
        conversation = await self._get_conversation(conversation_id)
        if not conversation:
            raise ConversationNotFoundError(f"Conversation with ID '{conversation_id}' not found")

        try:
            new_message = Message(
                tenant_id=self.tenant_id,
                conversation_id=conversation_id,
                role=role,
                content=content,
            )
            self.session.add(new_message)
            self.session.commit()

            logger.info(f"Added message to conversation {conversation_id} for user {self.user_id} and tenant {self.tenant_id} with role {role} and content {content}")
        except ConversationNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error adding message to conversation {conversation_id} for user {self.user_id} in tenant {self.tenant_id}: {str(e)}")
            raise ConversationsServiceError(f"Failed to add message to conversation: {str(e)}") from None

    async def _get_conversation(self, conversation_id: UUID) -> Conversation:
        """
        Utility function to get a conversation by ID.

        Args:
            conversation_id: ID of the conversation to get.

        Returns:
            Conversation: Conversation.

        Raises:
            ConversationNotFoundError: If conversation with the given ID does not exist.
        """
        statement = (
            select(Conversation)
            .where(
                and_(
                    Conversation.id == conversation_id,
                    Conversation.deleted_at.is_(None),
                    Conversation.user_id == self.user_id,
                    Conversation.tenant_id == self.tenant_id,
                )
            )
        )
        conversation = self.session.exec(statement).first()
        if not conversation:
            raise ConversationNotFoundError(f"Conversation with ID '{conversation_id}' not found")
        return conversation

    async def _conversation_to_response(self, conversation: Conversation) -> ConversationResponse:
        """
        Convert Conversation database model to ConversationResponse object.

        Args:
            conversation: Conversation database model.

        Returns:
            ConversationResponse: Conversation response object.
        """
        return ConversationResponse(
            id=conversation.id,
            metadata=ConversationMetadata(topic=conversation.topic),
            created_at=conversation.created_at.isoformat(),
            modified_at=conversation.modified_at.isoformat(),
        )
