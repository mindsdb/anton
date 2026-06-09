"""
Conversations service — stateful conversation management for responses endpoint.

For inference-only implementation: stores conversation history and message state.
Chart generation, CSV export, and report serving are removed.
"""

from datetime import datetime
from uuid import UUID

from sqlmodel import Session, and_, select

from minds.common.logger import get_logger
from minds.model.conversation import Conversation
from minds.model.message import Message
from minds.schemas.chat import Role
from minds.schemas.conversations import (
    ConversationCreateRequest,
    ConversationMetadata,
    ConversationResponse,
)
from minds.schemas.messages import MessageContent, MessageResponse

logger = get_logger(__name__)


class ConversationNotFoundError(Exception):
    """Exception for when a conversation is not found."""

    pass


class MessageNotFoundError(Exception):
    """Exception for when a message is not found."""

    pass


class ConversationsService:
    """Service for stateful conversation management."""

    def __init__(self, session: Session, user_id: str, organization_id: str):
        """
        Initialize the conversations service.

        Args:
            session: Database session.
            user_id: User ID (from context).
            organization_id: Organization ID (from context).
        """
        self.session = session
        self.user_id = user_id
        self.organization_id = organization_id
        logger.debug(f"ConversationsService initialized for user {user_id} in org {organization_id}")

    async def list_conversations(self, limit: int = 50, offset: int = 0) -> list[ConversationResponse]:
        """List conversations for the user."""
        stmt = (
            select(Conversation)
            .where(
                and_(
                    Conversation.organization_id == self.organization_id,
                    Conversation.user_id == self.user_id,
                    Conversation.deleted_at.is_(None),
                )
            )
            .offset(offset)
            .limit(limit)
            .order_by(Conversation.created_at.desc())
        )
        conversations = self.session.exec(stmt).all()
        return [await self.conversation_to_response(c) for c in conversations]

    async def get_conversation(self, conversation_id: UUID) -> ConversationResponse:
        """Get a single conversation by ID."""
        conversation = await self._get_conversation(conversation_id)
        return await self.conversation_to_response(conversation)

    async def create_conversation(self, req: ConversationCreateRequest) -> ConversationResponse:
        """Create a new conversation, optionally seeded with initial messages."""
        conversation = Conversation(
            user_id=self.user_id,
            organization_id=self.organization_id,
            topic=req.metadata.topic,
        )
        self.session.add(conversation)
        self.session.flush()

        if req.items:
            self.session.add_all(
                Message(
                    conversation_id=conversation.id,
                    user_id=self.user_id,
                    organization_id=self.organization_id,
                    role=item.role,
                    content=item.content,
                )
                for item in req.items
            )

        self.session.commit()
        self.session.refresh(conversation)
        logger.debug(f"Created conversation {conversation.id}")
        return await self.conversation_to_response(conversation)

    async def delete_conversation(self, conversation_id: UUID) -> None:
        """Soft-delete a conversation."""
        conversation = await self._get_conversation(conversation_id)
        conversation.deleted_at = datetime.utcnow()
        self.session.add(conversation)
        self.session.commit()
        logger.debug(f"Deleted conversation {conversation_id}")

    async def get_conversation_messages(self, conversation_id: UUID) -> list[MessageResponse]:
        """Get all messages in a conversation."""
        await self._get_conversation(conversation_id)  # Verify access
        stmt = (
            select(Message)
            .where(
                and_(
                    Message.conversation_id == conversation_id,
                    Message.deleted_at.is_(None),
                )
            )
            .order_by(Message.created_at)
        )
        messages = self.session.exec(stmt).all()
        return [await self._message_to_response(m) for m in messages]

    async def create_conversation_message(self, conversation_id: UUID, role: Role, content: str) -> MessageResponse:
        """Create a message in a conversation."""
        await self._get_conversation(conversation_id)  # Verify access
        message = Message(
            conversation_id=conversation_id,
            user_id=self.user_id,
            organization_id=self.organization_id,
            role=role,
            content=content,
        )
        self.session.add(message)
        self.session.commit()
        self.session.refresh(message)
        logger.debug(f"Created message {message.id} in conversation {conversation_id}")
        return await self._message_to_response(message)

    async def create_conversation_message_placeholder(self, conversation_id: UUID, role: Role) -> Message:
        """Create a placeholder message (no content yet)."""
        await self._get_conversation(conversation_id)
        message = Message(
            conversation_id=conversation_id,
            user_id=self.user_id,
            organization_id=self.organization_id,
            role=role,
        )
        self.session.add(message)
        self.session.commit()
        self.session.refresh(message)
        return message

    async def update_message_content(
        self,
        message: Message,
        content: str,
        *,
        model_name: str | None = None,
        request_id: str | None = None,
        langfuse_trace_id: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> Message:
        """Write final content + per-message tracing onto a placeholder message.

        Tracing fields enable per-conversation token rollups (chat_completions
        has no conversation_id, so the message row is the join point).
        """
        message.content = content
        message.model_name = model_name
        message.request_id = request_id
        message.langfuse_trace_id = langfuse_trace_id
        message.input_tokens = input_tokens
        message.output_tokens = output_tokens
        self.session.add(message)
        self.session.commit()
        self.session.refresh(message)
        logger.debug(f"Updated message {message.id}")
        return message

    async def conversation_to_response(self, conversation: Conversation) -> ConversationResponse:
        """Convert a conversation ORM to a response DTO."""
        return ConversationResponse(
            id=conversation.id,
            metadata=ConversationMetadata(topic=conversation.topic, model_name=""),
            created_at=str(conversation.created_at) if conversation.created_at else None,
            modified_at=str(conversation.modified_at) if conversation.modified_at else None,
        )

    async def _message_to_response(self, message: Message) -> MessageResponse:
        """Convert a message ORM to a response DTO."""
        text = message.content if isinstance(message.content, str) else ""
        return MessageResponse(
            id=message.id,
            role=message.role,
            content=MessageContent(text=text),
        )

    async def _get_conversation(self, conversation_id: UUID) -> Conversation:
        """Get and verify access to a conversation."""
        stmt = select(Conversation).where(
            and_(
                Conversation.id == conversation_id,
                Conversation.organization_id == self.organization_id,
                Conversation.user_id == self.user_id,
                Conversation.deleted_at.is_(None),
            )
        )
        conversation = self.session.exec(stmt).first()
        if not conversation:
            raise ConversationNotFoundError(f"Conversation {conversation_id} not found")
        return conversation

    async def _get_message(self, conversation_id: UUID, message_id: UUID) -> Message:
        """Get and verify access to a message."""
        await self._get_conversation(conversation_id)  # Verify conversation access
        stmt = select(Message).where(
            and_(
                Message.id == message_id,
                Message.conversation_id == conversation_id,
                Message.deleted_at.is_(None),
            )
        )
        message = self.session.exec(stmt).first()
        if not message:
            raise MessageNotFoundError(f"Message {message_id} not found")
        return message
