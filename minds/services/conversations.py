"""
Conversations service for managing conversation operations.
"""

import csv
import io
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from mindsdb_sdk.server import Server
from mindsdb_sql_parser import parse_sql
from mindsdb_sql_parser.ast import Select
from mindsdb_sql_parser.exceptions import ParsingException
from sqlalchemy.exc import PendingRollbackError
from sqlmodel import Session, and_, func, select

from minds.common.logger import setup_logging
from minds.model.conversation import Conversation
from minds.model.message import Message
from minds.schemas.chat import Role
from minds.schemas.conversations import ConversationCreateRequest, ConversationMetadata, ConversationResponse
from minds.schemas.messages import MessageContent, MessageContentType, MessageResponse, MessageResultResponse

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


class MessageNotFoundError(Exception):
    """Exception for when a message is not found."""

    pass


class MessageNotAssistantError(Exception):
    """Exception for when a message is not an assistant message."""

    pass


class MessageNoSQLQueryError(Exception):
    """Exception for when a message does not have a SQL query."""

    pass


class InvalidSQLQueryError(Exception):
    """Exception for when a SQL query is invalid."""

    pass


class ConversationsService:
    """
    Service class for conversation management operations.
    """

    def __init__(self, session: Session, mindsdb_client: Server, user_id: str, tenant_id: str):
        """
        Initialize the conversations service.
        """
        self.session = session
        self.mindsdb_client = mindsdb_client
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
    ) -> tuple[list[ConversationResponse], int]:
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
            tuple[list[ConversationResponse], int]: List of conversations and total count if include_total is True.

        Raises:
            ConversationsServiceError: If there is an error listing conversations.
        """
        logger.debug(
            f"Listing conversations for user {self.user_id} and tenant {self.tenant_id} with filters: "
            f"topic={topic}, include_deleted={include_deleted}, limit={limit}, offset={offset}, "
            f"include_total={include_total}, sort_by={sort_by}, sort_order={sort_order}"
        )

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
                count_statement = (
                    select(func.count(func.distinct(Conversation.id)))
                    .select_from(Conversation)
                    .where(and_(*conditions))
                )
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

            logger.info(
                f"Retrieved {len(conversations_list)} conversations "
                f"for user {self.user_id} in tenant {self.tenant_id} (offset={offset}, limit={limit})"
            )

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
            logger.error(
                f"Error getting conversation {conversation_id} "
                f"for user {self.user_id} in tenant {self.tenant_id}: {str(e)}"
            )
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
        logger.debug(
            f"Creating conversation {conversation_data.metadata.topic} "
            f"for user {self.user_id} in tenant {self.tenant_id}"
        )

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
                raise ConversationAlreadyExistsError(
                    f"Conversation with topic '{conversation_data.metadata.topic}' already exists"
                )

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

            logger.info(
                f"Created conversation {conversation_data.metadata.topic} "
                f"for user {self.user_id} in tenant {self.tenant_id}"
            )

            return await self._conversation_to_response(new_conversation)
        except ConversationAlreadyExistsError:
            self.session.rollback()
            raise
        except Exception as e:
            logger.error(
                f"Error creating conversation {conversation_data.metadata.topic} "
                f"for user {self.user_id} in tenant {self.tenant_id}: {str(e)}"
            )
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
            logger.error(
                f"Error deleting conversation {conversation_id} "
                f"for user {self.user_id} in tenant {self.tenant_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to delete conversation: {str(e)}") from None

    async def get_conversation_messages(self, conversation_id: UUID) -> list[MessageResponse]:
        """
        Get the messages of a conversation by ID.

        Args:
            conversation_id: ID of the conversation to get the messages from.

        Returns:
            List[MessageResponse]: List of messages.
        """
        logger.debug(
            f"Getting conversation {conversation_id} with messages for user {self.user_id} and tenant {self.tenant_id}"
        )

        try:
            conversation = await self._get_conversation(conversation_id)
            if not conversation:
                raise ConversationNotFoundError(f"Conversation with ID '{conversation_id}' not found")

            statement = (
                select(Message)
                .where(
                    and_(
                        Message.conversation_id == conversation_id,
                        Message.deleted_at.is_(None),
                        Message.tenant_id == self.tenant_id,
                    )
                )
                .order_by(Message.created_at.asc())
            )
            messages = self.session.exec(statement).all()
            messages_list = []
            for message in messages:
                messages_list.append(await self._message_to_response(message))
            return messages_list
        except ConversationNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error getting conversation {conversation_id} with messages "
                f"for user {self.user_id} in tenant {self.tenant_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to get conversation with messages: {str(e)}") from None

    async def create_conversation_message(self, conversation_id: UUID, role: Role, content: str) -> MessageResponse:
        """
        Create a new message in a conversation.
        """
        logger.debug(
            f"Creating message in conversation {conversation_id} "
            f"for user {self.user_id} in tenant {self.tenant_id} with role {role} and content {content}"
        )

        try:
            conversation = await self._get_conversation(conversation_id)
            if not conversation:
                raise ConversationNotFoundError(f"Conversation with ID '{conversation_id}' not found")

            new_message = Message(
                tenant_id=self.tenant_id,
                conversation_id=conversation_id,
                role=role,
                content=content,
            )
            self.session.add(new_message)
            self.session.commit()

            return await self._message_to_response(new_message)
        except ConversationNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error creating message in conversation {conversation_id} "
                f"for user {self.user_id} in tenant {self.tenant_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to create message: {str(e)}") from None

    async def create_conversation_message_placeholder(self, conversation_id: UUID, role: Role) -> Message:
        """
        Create a message placeholder with empty content and flush to get the ID.
        The message will be updated with actual content later via the update_message_content method.

        Args:
            conversation_id: ID of the conversation to add the message to.
            role: Role of the message.

        Returns:
            Message: The created message object with generated ID.

        Raises:
            ConversationNotFoundError: If conversation with the given ID does not exist.
            ConversationsServiceError: If there is an error creating the message.
        """
        logger.debug(
            f"Creating message placeholder in conversation {conversation_id} "
            f"for user {self.user_id} and tenant {self.tenant_id} with role {role}"
        )

        # Check if conversation exists
        conversation = await self._get_conversation(conversation_id)
        if not conversation:
            raise ConversationNotFoundError(f"Conversation with ID '{conversation_id}' not found")

        try:
            new_message = Message(
                tenant_id=self.tenant_id,
                conversation_id=conversation_id,
                role=role,
                content="",  # Placeholder - will be updated later
            )
            self.session.add(new_message)
            self.session.flush()  # Flush to get the ID, but don't commit yet

            logger.info(
                f"Created message placeholder {new_message.id} in conversation {conversation_id} "
                f"for user {self.user_id} and tenant {self.tenant_id}"
            )

            return new_message  # Return the message object so caller can access .id
        except ConversationNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error creating message placeholder in conversation {conversation_id} "
                f"for user {self.user_id} in tenant {self.tenant_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to create message placeholder: {str(e)}") from None

    async def update_conversation_message_content(
        self, message: Message, content: str, sql_query: str | None = None
    ) -> MessageResponse:
        """
        Update the content of an existing message.

        Args:
            message: The message object to update.
            content: The new content for the message.
            sql_query: The SQL query for the message.

        Returns:
            MessageResponse: Updated message response object.

        Raises:
            ConversationsServiceError: If there is an error updating the message.
        """
        logger.info(f"Updating message {message.id} content for user {self.user_id} and tenant {self.tenant_id}")

        try:
            # Ensure the message object is tracked by the session
            # Merge ensures the object is properly attached and tracked for updates
            message = self.session.merge(message)

            # Modify the content and sql_query
            message.content = content
            message.sql_query = sql_query
            # Commit will automatically flush any pending changes
            self.session.commit()

            logger.info(f"Updated message {message.id} content for user {self.user_id} and tenant {self.tenant_id}")

            return await self._message_to_response(message)
        except PendingRollbackError:
            # Session was rolled back, clear the state and retry once
            self.session.rollback()
            # Re-query the message to ensure it exists in the database
            # (it may not exist if it was only flushed and not committed)
            message_from_db = await self._get_message(message.conversation_id, message.id)
            if message_from_db is None:
                # Message doesn't exist, create it
                message_from_db = Message(
                    id=message.id,
                    tenant_id=self.tenant_id,
                    conversation_id=message.conversation_id,
                    role=message.role,
                    content=content,
                    sql_query=sql_query,
                )
                self.session.add(message_from_db)
            else:
                # Message exists, update it
                message_from_db.content = content
                message_from_db.sql_query = sql_query
            self.session.commit()

            logger.info(
                f"Updated message {message_from_db.id} content "
                f"for user {self.user_id} in tenant {self.tenant_id} (after rollback recovery)"
            )

            return await self._message_to_response(message_from_db)
        except Exception as e:
            self.session.rollback()
            logger.error(
                f"Error updating message {message.id} content "
                f"for user {self.user_id} in tenant {self.tenant_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to update message content: {str(e)}") from None

    async def get_conversation_message_result(
        self,
        conversation_id: UUID,
        message_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[MessageResultResponse, int, bool]:
        """
        Get the full result set of a response generated by the agent by executing the SQL query.

        Args:
            conversation_id: ID of the conversation to get the result from.
            message_id: ID of the message to get the result from.
            limit: Maximum number of rows to return.
            offset: Number of rows to skip.

        Returns:
            tuple[MessageResultResponse, int, bool]: Message result response object,
                total number of rows, and whether the pagination is consistent.
        """
        logger.debug(
            f"Getting message result for conversation {conversation_id} and message {message_id} "
            f"for user {self.user_id} in tenant {self.tenant_id}"
        )

        try:
            conversation = await self._get_conversation(conversation_id)
            if not conversation:
                raise ConversationNotFoundError(f"Conversation with ID '{conversation_id}' not found")

            message = await self._get_message(conversation_id, message_id)
            if not message:
                raise MessageNotFoundError(f"Message with ID '{message_id}' not found")

            if message.role != Role.assistant:
                raise MessageNotAssistantError(f"Message with ID '{message_id}' is not an assistant message")

            if message.sql_query is None:
                raise MessageNoSQLQueryError(f"Message with ID '{message_id}' does not have a SQL query")

            sql_query = message.sql_query

            # Validate the SQL query to prevent SQL injection.
            parsed_sql_query = self._validate_and_parse_sql_query(sql_query)

            # First off, we need to get a count of the total number of rows in the result
            count_sql_query = f"SELECT COUNT(*) FROM ({sql_query}) AS total_rows"
            count_result = self.mindsdb_client.query(count_sql_query).fetch()
            total_rows = count_result.values[0][0]

            # Then we need to get the data from the result with pagination
            # We need to check if the SQL query has a ORDER BY clause,
            # this is to ensure that the pagination is consistent.
            is_pagination_consistent = False
            if parsed_sql_query.order_by:
                is_pagination_consistent = True

            # Validate and coerce pagination parameters to integers to avoid SQL injection.
            try:
                limit_int = int(limit)
                offset_int = int(offset)
            except (TypeError, ValueError) as err:
                raise ValueError("Invalid pagination parameters: limit and offset must be integers") from err
            if limit_int < 0 or offset_int < 0:
                raise ValueError("Invalid pagination parameters: limit and offset must be non-negative")

            # This SQL query should also be within a subquery because the query itself,
            # may have a LIMIT or OFFSET clause.
            # TODO: Are there situations where we don't need to execute the query?
            # or put it in the nested subquery?
            paginated_sql_query = f"SELECT * FROM ({sql_query}) AS paginated_rows LIMIT {limit_int} OFFSET {offset_int}"
            result = self.mindsdb_client.query(paginated_sql_query).fetch()
            # Convert DataFrame to structured response
            column_names = result.columns.tolist()
            data = result.values.tolist()

            return (
                MessageResultResponse(
                    data=data,
                    column_names=column_names,
                ),
                total_rows,
                is_pagination_consistent,
            )
        except (
            ConversationNotFoundError,
            MessageNotFoundError,
            MessageNotAssistantError,
            MessageNoSQLQueryError,
            ValueError,
        ):
            raise
        except Exception as e:
            logger.error(
                f"Error getting message result for conversation {conversation_id} and message {message_id} "
                f"for user {self.user_id} in tenant {self.tenant_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to get message result: {str(e)}") from None

    async def export_conversation_message_result(
        self,
        conversation_id: UUID,
        message_id: UUID,
    ) -> bytes:
        """
        Export the result of a message by ID.
        """
        logger.debug(
            f"Exporting message result for conversation {conversation_id} and message {message_id} "
            f"for user {self.user_id} in tenant {self.tenant_id}"
        )

        try:
            conversation = await self._get_conversation(conversation_id)
            if not conversation:
                raise ConversationNotFoundError(f"Conversation with ID '{conversation_id}' not found")

            message = await self._get_message(conversation_id, message_id)
            if not message:
                raise MessageNotFoundError(f"Message with ID '{message_id}' not found")

            if message.role != Role.assistant:
                raise MessageNotAssistantError(f"Message with ID '{message_id}' is not an assistant message")

            if message.sql_query is None:
                raise MessageNoSQLQueryError(f"Message with ID '{message_id}' does not have a SQL query")

            sql_query = message.sql_query

            # Validate the SQL query to prevent SQL injection.
            _ = self._validate_and_parse_sql_query(sql_query)

            result = self.mindsdb_client.query(sql_query).fetch()

            # Convert DataFrame to structured response
            column_names = result.columns.tolist()
            data = result.to_dict(orient="records")
            # Write to CSV
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=column_names, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)

            return buf.getvalue().encode("utf-8")
        except (ConversationNotFoundError, MessageNotFoundError, MessageNotAssistantError, MessageNoSQLQueryError):
            raise
        except Exception as e:
            logger.error(
                f"Error exporting message result for conversation {conversation_id} and message {message_id} "
                f"for user {self.user_id} in tenant {self.tenant_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to export message result: {str(e)}") from None

    def _validate_and_parse_sql_query(self, sql_query: str) -> Select:
        """
        Validate and parse the SQL query to ensure it is valid and a SELECT query.

        Args:
            sql_query: The SQL query to validate and parse.

        Raises:
            InvalidSQLQueryError: If the SQL query is invalid or not a SELECT query.

        Returns:
            Select: The parsed SQL query.
        """
        try:
            parsed_sql_query = parse_sql(sql_query)
        except ParsingException as e:
            raise InvalidSQLQueryError(f"Invalid SQL query: {e}") from e

        # Ensure the SQL query is a SELECT query to avoid SQL injection.
        # This is an extra guardrail since we control the SQL query stored in the database.
        if not isinstance(parsed_sql_query, Select):
            raise InvalidSQLQueryError(f"SQL query is not a SELECT query: {parsed_sql_query}")

        return parsed_sql_query

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
        statement = select(Conversation).where(
            and_(
                Conversation.id == conversation_id,
                Conversation.deleted_at.is_(None),
                Conversation.user_id == self.user_id,
                Conversation.tenant_id == self.tenant_id,
            )
        )
        conversation = self.session.exec(statement).first()
        if not conversation:
            raise ConversationNotFoundError(f"Conversation with ID '{conversation_id}' not found")
        return conversation

    async def _get_message(self, conversation_id: UUID, message_id: UUID) -> Message:
        """
        Utility function to get a message by ID.
        """
        statement = select(Message).where(
            and_(
                Message.id == message_id,
                Message.conversation_id == conversation_id,
                Message.deleted_at.is_(None),
                Message.tenant_id == self.tenant_id,
            )
        )
        message = self.session.exec(statement).first()
        if not message:
            raise MessageNotFoundError(f"Message with ID '{message_id}' not found")
        return message

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

    async def _message_to_response(self, message: Message) -> MessageResponse:
        """
        Convert Message database model to MessageResponse object.

        Args:
            message: Message database model.

        Returns:
            MessageResponse: Message response object.
        """
        content = MessageContent(
            type=MessageContentType.output_text if message.role == Role.assistant else MessageContentType.input_text,
            text=message.content,
        )
        return MessageResponse(
            id=message.id,
            role=message.role,
            content=content,
            created_at=message.created_at.isoformat(),
            modified_at=message.modified_at.isoformat(),
        )
