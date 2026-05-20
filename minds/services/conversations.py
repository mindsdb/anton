"""
Conversations service for managing conversation operations.
"""

import asyncio
import base64
import csv
import io
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from mindsdb_sdk.server import Server
from mindsdb_sql_parser import parse_sql
from mindsdb_sql_parser.ast import Select
from mindsdb_sql_parser.exceptions import ParsingException
from pydantic import ValidationError
from sqlalchemy.exc import PendingRollbackError
from sqlalchemy.orm import selectinload, with_loader_criteria
from sqlmodel import Session, and_, func, select

# Anton agent imports - required only for the feature for exporting reports.
from minds.agents.anton_agent.anton.backends.base import ScratchpadRuntimeFactory
from minds.agents.anton_agent.settings import AntonAgentSettings
from minds.agents.helpers import is_anton_agent

# Common imports.
from minds.common.logger import get_logger
from minds.common.mindsdb import extract_database_engines_from_select
from minds.common.settings.app_settings import get_app_settings
from minds.common.utilities import format_numeric_columns
from minds.model.conversation import Conversation
from minds.model.message import Message
from minds.model.message_event import MessageEvent
from minds.schemas.charts import (
    ChartImageResponse,
    ChartImageTokenPayload,
    ChartIntent,
    ChartResponse,
    PieIntent,
    ScatterIntent,
    XYIntent,
)
from minds.schemas.chat import Role
from minds.schemas.conversations import ConversationCreateRequest, ConversationMetadata, ConversationResponse
from minds.schemas.messages import MessageContent, MessageContentType, MessageResponse, MessageResultResponse
from minds.services.minds import MindNotFoundError, MindsService

logger = get_logger(__name__)

app_settings = get_app_settings()


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


class AgentNotAntonError(Exception):
    """Exception for when a message is not an Anton message."""

    pass


class ConversationsService:
    """
    Service class for conversation management operations.
    """

    def __init__(self, session: Session, mindsdb_client: Server, user_id: str, organization_id: str):
        """
        Initialize the conversations service.
        """
        self.session = session
        self.mindsdb_client = mindsdb_client
        self.user_id = user_id
        self.organization_id = organization_id
        logger.debug(f"ConversationsService initialized for user {user_id} and organization {organization_id}")

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
            f"Listing conversations for user {self.user_id} and organization {self.organization_id} with filters: "
            f"topic={topic}, include_deleted={include_deleted}, limit={limit}, offset={offset}, "
            f"include_total={include_total}, sort_by={sort_by}, sort_order={sort_order}"
        )

        try:
            # Build query conditions
            conditions = [Conversation.organization_id == self.organization_id, Conversation.user_id == self.user_id]
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

            statement = (
                select(Conversation)
                .options(selectinload(Conversation.mind))
                .where(and_(*conditions))
                .order_by(order_by)
                .offset(offset)
                .limit(limit)
            )

            conversations = self.session.exec(statement).all()

            conversations_list = []
            for conversation in conversations:
                conversation_response = await self.conversation_to_response(conversation)
                conversations_list.append(conversation_response)

            logger.info(
                f"Retrieved {len(conversations_list)} conversations "
                f"for user {self.user_id} in organization {self.organization_id} (offset={offset}, limit={limit})"
            )

            if include_total:
                return conversations_list, total_count
            return conversations_list
        except Exception as e:
            logger.error(
                f"Error listing conversations for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
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
        logger.debug(
            f"Getting conversation {conversation_id} for user {self.user_id} and organization {self.organization_id}"
        )

        try:
            conversation = await self._get_conversation(conversation_id)

            return await self.conversation_to_response(conversation)
        except ConversationNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error getting conversation {conversation_id} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to get conversation: {str(e)}") from None

    async def create_conversation(
        self, conversation_data: ConversationCreateRequest, mind_service: MindsService
    ) -> ConversationResponse:
        """
        Create a new conversation.

        Args:
            conversation_data: Conversation data to create.
            mind_service: MindsService instance to get the mind.

        Returns:
            ConversationResponse: Created conversation.

        Raises:
            ConversationsServiceError: If there is an error creating the conversation.
        """
        logger.debug(
            f"Creating conversation {conversation_data.metadata.topic} "
            f"for user {self.user_id} in organization {self.organization_id}"
        )

        try:
            # Check if Mind exists
            mind = await mind_service.get_mind_model(conversation_data.metadata.model_name)

            new_conversation = Conversation(
                topic=conversation_data.metadata.topic,
                user_id=self.user_id,
                organization_id=self.organization_id,
                mind_id=mind.id,
            )

            self.session.add(new_conversation)
            self.session.flush()

            if conversation_data.items:
                new_messages = []
                for item in conversation_data.items:
                    new_message = Message(
                        organization_id=self.organization_id,
                        user_id=self.user_id,
                        conversation_id=new_conversation.id,
                        role=item.role,
                        content=item.content,
                    )
                    new_messages.append(new_message)
                self.session.add_all(new_messages)

            self.session.commit()

            logger.info(
                f"Created conversation {conversation_data.metadata.topic} "
                f"for user {self.user_id} in organization {self.organization_id}"
            )

            return await self.conversation_to_response(new_conversation)
        except MindNotFoundError:
            self.session.rollback()
            raise
        except Exception as e:
            logger.error(
                f"Error creating conversation {conversation_data.metadata.topic} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
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
        logger.debug(
            f"Deleting conversation {conversation_id} for user {self.user_id} and organization {self.organization_id}"
        )

        try:
            conversation = await self._get_conversation(conversation_id)

            conversation.deleted_at = datetime.now(timezone.utc)
            self.session.add(conversation)
            self.session.commit()

            logger.info(
                f"Deleted conversation {conversation_id} for user {self.user_id} in organization {self.organization_id}"
            )
        except ConversationNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error deleting conversation {conversation_id} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to delete conversation: {str(e)}") from None

    async def get_conversation_messages(
        self,
        conversation_id: UUID,
        with_sql_query: bool = False,
        with_events: bool = False,
    ) -> list[MessageResponse]:
        """
        Get the messages of a conversation by ID.

        Args:
            conversation_id: ID of the conversation to get the messages from.
            with_sql_query: Whether to include the SQL query in the response.
            with_events: Whether to include the events in the response.

        Returns:
            List[MessageResponse]: List of messages.
        """
        logger.debug(
            f"Getting conversation {conversation_id} with messages for user {self.user_id} and "
            f"organization {self.organization_id}"
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
                        Message.organization_id == self.organization_id,
                    )
                )
                .order_by(Message.created_at.asc())
            )
            if with_events:
                statement = statement.options(
                    selectinload(Message.message_events),
                    with_loader_criteria(
                        MessageEvent,
                        lambda cls: cls.deleted_at.is_(None),
                        include_aliases=True,
                    ),
                )

            messages = self.session.exec(statement).all()
            messages_list = []
            for message in messages:
                messages_list.append(
                    await self._message_to_response(
                        message,
                        with_sql_query=with_sql_query,
                        with_events=with_events,
                    )
                )
            return messages_list
        except ConversationNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error getting conversation {conversation_id} with messages "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to get conversation with messages: {str(e)}") from None

    async def create_conversation_message(self, conversation_id: UUID, role: Role, content: str) -> MessageResponse:
        """
        Create a new message in a conversation.
        """
        logger.debug(
            f"Creating message in conversation {conversation_id} "
            f"for user {self.user_id} in organization {self.organization_id} with role {role} and content {content}"
        )

        try:
            conversation = await self._get_conversation(conversation_id)
            if not conversation:
                raise ConversationNotFoundError(f"Conversation with ID '{conversation_id}' not found")

            new_message = Message(
                organization_id=self.organization_id,
                user_id=self.user_id,
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
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
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
            f"for user {self.user_id} in organization {self.organization_id} with role {role}"
        )

        # Check if conversation exists
        conversation = await self._get_conversation(conversation_id)
        if not conversation:
            raise ConversationNotFoundError(f"Conversation with ID '{conversation_id}' not found")

        try:
            new_message = Message(
                organization_id=self.organization_id,
                user_id=self.user_id,
                conversation_id=conversation_id,
                role=role,
                content="",  # Placeholder - will be updated later
            )
            self.session.add(new_message)
            self.session.flush()  # Flush to get the ID, but don't commit yet

            logger.info(
                f"Created message placeholder {new_message.id} in conversation {conversation_id} "
                f"for user {self.user_id} in organization {self.organization_id}"
            )

            return new_message  # Return the message object so caller can access .id
        except ConversationNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Error creating message placeholder in conversation {conversation_id} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to create message placeholder: {str(e)}") from None

    async def create_conversation_message_event(
        self,
        message_id: UUID,
        sequence_number: int,
        event_data: dict,
        commit: bool = False,
    ) -> None:
        """
        Create a new message event.

        Args:
            message_id: ID of the message to add the event to.
            sequence_number: Sequence number of the event.
            event_data: Data of the event.
            commit: Whether to commit the session.

        Returns:
            None.

        Raises:
            MessageNotFoundError: If message with the given ID does not exist.
            ConversationsServiceError: If there is an error creating the message event.
        """
        logger.debug(
            f"Creating message event for message {message_id} "
            f"for user {self.user_id} in organization {self.organization_id} with event data {event_data}"
        )
        try:
            event = MessageEvent(
                message_id=message_id,
                organization_id=self.organization_id,
                user_id=self.user_id,
                sequence_number=sequence_number,
                event_data=event_data,
            )
            self.session.add(event)
            self.session.flush()  # The event will be committed when the message is committed
            if commit:
                self.session.commit()
            logger.info(
                f"Created message event {event.id} for message {message_id} "
                f"for user {self.user_id} in organization {self.organization_id}"
            )
        except Exception as e:
            logger.error(
                f"Error creating message event for message {message_id} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to create message event: {str(e)}") from None

    async def update_conversation_message_content(
        self,
        message: Message,
        content: str,
        sql_query: str | None = None,
        model_name: str | None = None,
        request_id: str | None = None,
        langfuse_trace_id: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> MessageResponse:
        """
        Update the content of an existing message.

        Args:
            message: The message object to update.
            content: The new content for the message.
            sql_query: The SQL query for the message.
            model_name: The model/mind name used for the completion.
            request_id: The request ID for tracing.
            langfuse_trace_id: The Langfuse trace ID for cross-referencing.
            input_tokens: Number of input (prompt) tokens used.
            output_tokens: Number of output (completion) tokens used.

        Returns:
            MessageResponse: Updated message response object.

        Raises:
            ConversationsServiceError: If there is an error updating the message.
        """
        logger.info(
            f"Updating message {message.id} content for user {self.user_id} in organization {self.organization_id}"
        )

        def _apply_fields(msg: Message) -> None:
            """Apply content and usage fields to a message object."""
            msg.content = content
            msg.sql_query = sql_query
            msg.model_name = model_name
            msg.request_id = request_id
            msg.langfuse_trace_id = langfuse_trace_id
            msg.input_tokens = input_tokens
            msg.output_tokens = output_tokens

        try:
            # Ensure the message object is tracked by the session
            # Merge ensures the object is properly attached and tracked for updates
            message = self.session.merge(message)

            _apply_fields(message)
            # Commit will automatically flush any pending changes
            self.session.commit()

            logger.info(
                f"Updated message {message.id} content for user {self.user_id} in organization {self.organization_id}"
            )

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
                    organization_id=self.organization_id,
                    user_id=self.user_id,
                    conversation_id=message.conversation_id,
                    role=message.role,
                    content=content,
                    sql_query=sql_query,
                    model_name=model_name,
                    request_id=request_id,
                    langfuse_trace_id=langfuse_trace_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                self.session.add(message_from_db)
            else:
                _apply_fields(message_from_db)
            self.session.commit()

            logger.info(
                f"Updated message {message_from_db.id} content "
                f"for user {self.user_id} in organization {self.organization_id} (after rollback recovery)"
            )

            return await self._message_to_response(message_from_db)
        except Exception as e:
            self.session.rollback()
            logger.error(
                f"Error updating message {message.id} content "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
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
            f"for user {self.user_id} in organization {self.organization_id}"
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

            sql_query = str(message.sql_query).strip()

            # Validate the SQL query to prevent SQL injection.
            parsed_sql_query = self._validate_and_parse_sql_query(sql_query)

            # First off, we need to get a count of the total number of rows in the result
            count_sql_query = f"SELECT COUNT(*) FROM ({sql_query}) AS total_rows"
            count_result = self.mindsdb_client.query(count_sql_query).fetch()
            total_rows = count_result.values[0][0]
            try:
                if isinstance(total_rows, str):
                    total_rows = total_rows.replace(",", "").strip()
                total_rows = int(total_rows)
            except (TypeError, ValueError):
                total_rows = None

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

            # TODO: This is a temporary solution to handle pagination for MSSQL.
            # This is because the MSSQL integration for MindsDB does not support LIMIT and OFFSET.
            # This should be fixed in MindsDB itself and removed from here.
            # Get the database engine of the query
            database_engines = extract_database_engines_from_select(
                parsed_sql_query,
                mindsdb_client=self.mindsdb_client,
                exclude_cte_names=True,
            )
            # database_engine = self.mindsdb_client.databases.get(parsed_sql_query.from_table.parts[0]).engine
            # If is is MSSQL, execute the original query without LIMIT and OFFSET
            # Execute pagination in memory.
            if "mssql" in database_engines:
                result = self.mindsdb_client.query(sql_query).fetch()
                result = result.iloc[offset_int : offset_int + limit_int]

            # For all other database engines, execute the query with LIMIT and OFFSET
            else:
                # This SQL query should also be within a subquery because the query itself,
                # may have a LIMIT or OFFSET clause.
                # TODO: Are there situations where we don't need to execute the query?
                # or put it in the nested subquery?
                paginated_sql_query = (
                    f"SELECT * FROM ({sql_query}) AS paginated_rows LIMIT {limit_int} OFFSET {offset_int}"
                )
                result = self.mindsdb_client.query(paginated_sql_query).fetch()

            # Convert DataFrame to structured response
            column_names = result.columns.tolist()
            data = result.values.tolist()
            if total_rows is None:
                total_rows = len(data)

            # Format numeric columns for display (prevents scientific notation in UI).
            # Uses a copy so raw numeric values in `data` are preserved for evaluation.
            display_data = format_numeric_columns(result.copy()).values.tolist()

            return (
                MessageResultResponse(
                    data=data,
                    column_names=column_names,
                    display_data=display_data,
                ),
                int(total_rows),
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
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
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
            f"for user {self.user_id} in organization {self.organization_id}"
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

            sql_query = str(message.sql_query).strip()

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
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to export message result: {str(e)}") from None

    async def get_conversation_message_chart(
        self,
        conversation_id: UUID,
        message_id: UUID,
        intent: XYIntent | PieIntent | ScatterIntent,
    ) -> ChartResponse:
        """
        Generate a Chart.js configuration from a message's SQL query results.

        Args:
            conversation_id: ID of the conversation.
            message_id: ID of the message with the SQL query.
            intent: Chart intent specification (XYIntent, PieIntent, or ScatterIntent).

        Returns:
            ChartResponse containing Chart.js configuration, metadata, and warnings.

        Raises:
            ConversationNotFoundError: If conversation is not found.
            MessageNotFoundError: If message is not found.
            MessageNotAssistantError: If message is not from the assistant.
            MessageNoSQLQueryError: If message has no SQL query.
            InvalidSQLQueryError: If SQL query is invalid.
            ValueError: If intent references invalid columns.
        """
        logger.debug(
            f"Getting chart config for conversation {conversation_id} and message {message_id} "
            f"for user {self.user_id} in organization {self.organization_id}"
        )

        try:
            from minds.services.chart_compiler import render_plan_to_chartjs

            plan, warnings, meta = await self._compile_conversation_message_chart(
                conversation_id=conversation_id,
                message_id=message_id,
                intent=intent,
            )
            return ChartResponse(
                config=render_plan_to_chartjs(plan),
                meta=meta,
                warnings=warnings,
            )

        except (
            ConversationNotFoundError,
            MessageNotFoundError,
            MessageNotAssistantError,
            MessageNoSQLQueryError,
            InvalidSQLQueryError,
            ValueError,
        ):
            raise
        except Exception as e:
            logger.error(
                f"Error getting chart for conversation {conversation_id} and message {message_id} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to generate chart: {str(e)}") from None

    async def check_conversation_message_report_exists(
        self,
        conversation_id: UUID,
        message_id: UUID,
    ) -> None:
        """
        Check if a report exists for a message by ID.

        Args:
            conversation_id: ID of the conversation to check the report for.
            message_id: ID of the message to check the report for.

        Returns:
            None

        Raises:
            ConversationNotFoundError: If conversation is not found.
            MessageNotFoundError: If message is not found.
            MessageNotAssistantError: If message is not from the assistant.
            ConversationsServiceError: If there is an error checking if the report exists.
            FileNotFoundError: If the report does not exist.
        """
        logger.debug(
            f"Checking if report exists for conversation {conversation_id} and message {message_id} "
            f"for user {self.user_id} in organization {self.organization_id}"
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

            if not is_anton_agent(conversation.mind):
                raise AgentNotAntonError(f"Mind {conversation.mind.name} is not using the Anton agent")

            anton_settings = AntonAgentSettings()
            exists = await ScratchpadRuntimeFactory().report_exists(
                backend=anton_settings.backend,
                organization_id=self.organization_id,
                user_id=self.user_id,
                conversation_id=conversation_id,
                message_id=message_id,
            )
            if exists:
                logger.debug(
                    f"Report exists for conversation {conversation_id} and message {message_id} "
                    f"for user {self.user_id} in organization {self.organization_id}"
                )
            else:
                logger.debug(
                    f"Report does not exist for conversation {conversation_id} and message {message_id} "
                    f"for user {self.user_id} in organization {self.organization_id}"
                )
                raise FileNotFoundError("A report is not available for this message")
        except (
            ConversationNotFoundError,
            MessageNotFoundError,
            MessageNotAssistantError,
            AgentNotAntonError,
            ValueError,
            FileNotFoundError,
        ):
            raise
        except Exception as e:
            logger.error(
                f"Error checking if report exists for conversation {conversation_id} and message {message_id} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to check if report exists: {str(e)}") from None

    async def get_conversation_message_report(
        self,
        conversation_id: UUID,
        message_id: UUID,
    ) -> str:
        """
        Get the report of a message by ID.
        This will be a file system path to the HTML file containing the report.
        Applies only to the Anton agent.

        Args:
            conversation_id: ID of the conversation to get the report from.
            message_id: ID of the message to get the report from.

        Returns:
            str: The file system path to the HTML file containing the report.
        """
        logger.debug(
            f"Getting report for conversation {conversation_id} and message {message_id} "
            f"for user {self.user_id} in organization {self.organization_id}"
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

            if not is_anton_agent(conversation.mind):
                raise AgentNotAntonError(f"Mind {conversation.mind.name} is not using the Anton agent")

            anton_settings = AntonAgentSettings()
            report = await ScratchpadRuntimeFactory().get_report(
                backend=anton_settings.backend,
                organization_id=self.organization_id,
                user_id=self.user_id,
                conversation_id=conversation_id,
                message_id=message_id,
            )
            return report

        except (
            ConversationNotFoundError,
            MessageNotFoundError,
            MessageNotAssistantError,
            AgentNotAntonError,
            ValueError,
            FileNotFoundError,
        ):
            raise
        except Exception as e:
            logger.error(
                f"Error getting report for conversation {conversation_id} and message {message_id} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to get report: {str(e)}") from None

    async def render_conversation_message_chart_png(
        self,
        conversation_id: UUID,
        message_id: UUID,
        intent: ChartIntent,
    ) -> bytes:
        """Render a conversation chart directly to PNG bytes."""
        from minds.services.chart_renderer import render_chart_image

        logger.debug(
            f"Rendering direct chart PNG for conversation {conversation_id} and message {message_id} "
            f"for user {self.user_id} in organization {self.organization_id}"
        )

        try:
            plan, _, _ = await self._compile_conversation_message_chart(
                conversation_id=conversation_id,
                message_id=message_id,
                intent=intent,
            )
            return await asyncio.to_thread(render_chart_image, plan)
        except (
            ConversationNotFoundError,
            MessageNotFoundError,
            MessageNotAssistantError,
            MessageNoSQLQueryError,
            InvalidSQLQueryError,
            ValueError,
        ):
            raise
        except Exception as e:
            logger.error(
                f"Error rendering chart PNG for conversation {conversation_id} and message {message_id} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to render chart image: {str(e)}") from None

    async def get_conversation_message_chart_image(
        self,
        conversation_id: UUID,
        message_id: UUID,
        intent: ChartIntent,
    ) -> ChartImageResponse:
        """
        Return an authenticated URL for a server-rendered chart image.

        The returned URL is an authenticated fetch endpoint that renders the
        chart on demand. Callers should not treat it as a durable, immutable
        snapshot reference.
        """

        logger.debug(
            f"Getting chart image URL for conversation {conversation_id} and message {message_id} "
            f"for user {self.user_id} in organization {self.organization_id}"
        )

        try:
            _, warnings, meta = await self._compile_conversation_message_chart(
                conversation_id=conversation_id,
                message_id=message_id,
                intent=intent,
            )
            token = self._build_chart_image_token(intent)
            image_url = f"/v1/conversations/{conversation_id}/items/{message_id}/chart?token={token}"

            return ChartImageResponse(
                image_url=image_url,
                meta=meta,
                warnings=warnings,
            )
        except (
            ConversationNotFoundError,
            MessageNotFoundError,
            MessageNotAssistantError,
            MessageNoSQLQueryError,
            InvalidSQLQueryError,
            ValueError,
        ):
            raise
        except Exception as e:
            logger.error(
                f"Error preparing chart image URL for conversation {conversation_id} and message {message_id} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to prepare chart image URL: {str(e)}") from None

    async def render_conversation_message_chart_by_token(
        self,
        conversation_id: UUID,
        message_id: UUID,
        token: str,
    ) -> bytes:
        """Render a chart image from an opaque token."""

        logger.debug(
            f"Rendering chart image from token for conversation {conversation_id} and message {message_id} "
            f"for user {self.user_id} in organization {self.organization_id}"
        )

        try:
            intent = self._parse_chart_image_token(token)
            return await self.render_conversation_message_chart_png(
                conversation_id=conversation_id,
                message_id=message_id,
                intent=intent,
            )
        except (
            ConversationNotFoundError,
            MessageNotFoundError,
            MessageNotAssistantError,
            MessageNoSQLQueryError,
            InvalidSQLQueryError,
            ValueError,
        ):
            raise
        except Exception as e:
            logger.error(
                f"Error rendering chart image from token for conversation {conversation_id} and message {message_id} "
                f"for user {self.user_id} in organization {self.organization_id}: {str(e)}"
            )
            raise ConversationsServiceError(f"Failed to render chart image: {str(e)}") from None

    async def _compile_conversation_message_chart(
        self,
        conversation_id: UUID,
        message_id: UUID,
        intent: ChartIntent,
    ) -> tuple:
        """Compile chart artifacts shared by all chart delivery endpoints."""
        from minds.services.chart_compiler import MAX_ROWS_TO_PROCESS, compile_chart

        await self._get_conversation(conversation_id)
        message = await self._get_message(conversation_id, message_id)

        if message.role != Role.assistant:
            raise MessageNotAssistantError(f"Message with ID '{message_id}' is not an assistant message")

        if message.sql_query is None:
            raise MessageNoSQLQueryError(f"Message with ID '{message_id}' does not have a SQL query")

        sql_query = str(message.sql_query).strip()
        parsed_sql_query = self._validate_and_parse_sql_query(sql_query)
        database_engines = extract_database_engines_from_select(
            parsed_sql_query,
            mindsdb_client=self.mindsdb_client,
            exclude_cte_names=True,
        )

        if "mssql" in database_engines:
            limited_sql_query = f"SELECT TOP {MAX_ROWS_TO_PROCESS} * FROM ({sql_query}) AS chart_data"
        else:
            limited_sql_query = f"SELECT * FROM ({sql_query}) AS chart_data LIMIT {MAX_ROWS_TO_PROCESS}"

        result = self.mindsdb_client.query(limited_sql_query).fetch()
        return compile_chart(result, intent)

    def _build_chart_image_token(self, intent: ChartIntent) -> str:
        payload = ChartImageTokenPayload(intent=intent).model_dump_json(exclude_none=True)
        token_bytes = base64.urlsafe_b64encode(payload.encode("utf-8"))
        return token_bytes.decode("ascii").rstrip("=")

    def _parse_chart_image_token(self, token: str) -> ChartIntent:
        try:
            padded = token + ("=" * (-len(token) % 4))
            payload = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
            return ChartImageTokenPayload.model_validate_json(payload).intent
        except (UnicodeDecodeError, ValidationError, ValueError) as e:
            raise ValueError("Invalid chart image token") from e

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
        if not sql_query or not sql_query.strip():
            raise InvalidSQLQueryError("Invalid SQL query: Empty input")

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
        statement = (
            select(Conversation)
            .options(selectinload(Conversation.mind))
            .where(
                and_(
                    Conversation.id == conversation_id,
                    Conversation.deleted_at.is_(None),
                    Conversation.organization_id == self.organization_id,
                )
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
                Message.organization_id == self.organization_id,
            )
        )
        message = self.session.exec(statement).first()
        if not message:
            raise MessageNotFoundError(f"Message with ID '{message_id}' not found")
        return message

    async def conversation_to_response(self, conversation: Conversation) -> ConversationResponse:
        """
        Convert Conversation database model to ConversationResponse object.

        Args:
            conversation: Conversation database model.

        Returns:
            ConversationResponse: Conversation response object.
        """
        return ConversationResponse(
            id=conversation.id,
            metadata=ConversationMetadata(
                topic=conversation.topic,
                model_name=conversation.mind.name,
            ),
            created_at=conversation.created_at.isoformat(),
            modified_at=conversation.modified_at.isoformat(),
        )

    async def _message_to_response(
        self, message: Message, with_sql_query: bool = False, with_events: bool = False
    ) -> MessageResponse:
        """
        Convert Message database model to MessageResponse object.

        Args:
            message: Message database model.
            with_sql_query: Whether to include the SQL query in the response.
            with_events: Whether to include the events in the response.

        Returns:
            MessageResponse: Message response object.
        """
        content = MessageContent(
            type=MessageContentType.output_text if message.role == Role.assistant else MessageContentType.input_text,
            text=str(message.content),
        )
        message_response = MessageResponse(
            id=message.id,
            role=message.role,
            content=content,
            created_at=message.created_at.isoformat(),
            modified_at=message.modified_at.isoformat(),
        )

        if with_sql_query:
            message_response.sql_query = str(message.sql_query)

        if with_events:
            message_response.events = [event.event_data for event in message.message_events]

        return message_response
