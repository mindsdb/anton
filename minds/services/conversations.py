"""
Conversations service for managing conversation operations.
"""

from typing import Literal
from uuid import UUID

from sqlmodel import Session, and_, func, select

from minds.common.logger import setup_logging
from minds.model.conversation import Conversation
from minds.model.message import Message
from minds.schemas.conversations import ConversationCreateRequest, ConversationDetailedResponse, ConversationResponse
from minds.schemas.messages import MessageResponse

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

    async def get_conversation(self, conversation_id: UUID, with_messages: bool = False) -> ConversationResponse | ConversationDetailedResponse:
        """
        Get a conversation by ID.

        Args:
            conversation_id: ID of the conversation to get.
            with_messages: Whether to include messages in the conversation.

        Returns:
            ConversationResponse | ConversationDetailedResponse: Conversation.

        Raises:
            ConversationNotFoundError: If conversation with the given ID does not exist.
            ConversationsServiceError: If there is an error getting the conversation.
        """
        logger.debug(f"Getting conversation {conversation_id} for user {self.user_id} and tenant {self.tenant_id}")

        try:
            conversation = self.session.get(Conversation, conversation_id)
            if not conversation:
                raise ConversationNotFoundError(f"Conversation with ID '{conversation_id}' not found")

            if with_messages:
                return await self._conversation_to_detailed_response(conversation)
            else:
                return await self._conversation_to_response(conversation)
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
        logger.debug(f"Creating conversation {conversation_data.topic} for user {self.user_id} and tenant {self.tenant_id}")

        try:
            # Check if conversation already exists
            existing_conversation = self.session.exec(
                select(Conversation).where(
                    and_(
                        Conversation.topic == conversation_data.topic,
                        Conversation.user_id == self.user_id,
                        Conversation.tenant_id == self.tenant_id,
                    )
                )
            ).first()

            if existing_conversation:
                raise ConversationAlreadyExistsError(f"Conversation with topic '{conversation_data.topic}' already exists")

            new_conversation = Conversation(
                topic=conversation_data.topic,
                user_id=self.user_id,
                tenant_id=self.tenant_id,
            )

            self.session.add(new_conversation)
            self.session.commit()
            self.session.refresh(new_conversation)

            logger.info(f"Created conversation {conversation_data.topic} for user {self.user_id} and tenant {self.tenant_id}")

            return await self._conversation_to_response(new_conversation)
        except ConversationAlreadyExistsError:
            raise
        except Exception as e:
            logger.error(f"Error creating conversation {conversation_data.topic} for user {self.user_id} in tenant {self.tenant_id}: {str(e)}")
            raise ConversationsServiceError(f"Failed to create conversation: {str(e)}") from None

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
            topic=conversation.topic,
            created_at=conversation.created_at.isoformat(),
            modified_at=conversation.modified_at.isoformat(),
        )

    async def _conversation_to_detailed_response(self, conversation: Conversation) -> ConversationDetailedResponse:
        """
        Convert Conversation database model to ConversationDetailedResponse object.

        Args:
            conversation: Conversation database model.

        Returns:
            ConversationDetailedResponse: Conversation detailed response object.
        """
        return ConversationDetailedResponse(
            id=conversation.id,
            topic=conversation.topic,
            created_at=conversation.created_at.isoformat(),
            modified_at=conversation.modified_at.isoformat(),
            messages=await self._messages_to_response(conversation.messages),
        )

    async def _messages_to_response(self, messages: list[Message]) -> list[MessageResponse]:
        """
        Convert Message database model to MessageResponse object.

        Args:
            messages: List of Message database models.

        Returns:
            list[MessageResponse]: List of Message response objects.
        """
        return [MessageResponse(
            id=message.id,
            role=message.role,
            content=message.content,
            created_at=message.created_at.isoformat(),
            modified_at=message.modified_at.isoformat(),
        ) for message in messages]
