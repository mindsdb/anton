"""
Conversation management endpoints for API v1.

This module contains endpoints for CRUD operations on conversations,
providing a clean v1 API interface for conversation management.
"""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlmodel import Session

from minds.client.mindsdb import create_mindsdb_client_from_request
from minds.common.logger import setup_logging
from minds.db.pg_session import get_session
from minds.requests.context import extract_context_from_request
from minds.schemas.conversations import ConversationCreateRequest, ConversationResponse
from minds.schemas.messages import MessageResponse, MessageResultResponse
from minds.services.conversations import (
    ConversationsService,
    ConversationAlreadyExistsError,
    ConversationNotFoundError,
    ConversationsServiceError,
    MessageNotFoundError,
    MessageNotAssistantError,
    MessageNoSQLQueryError,
)

logger = setup_logging()

router = APIRouter()


def get_conversations_service(request: Request, session: Session = Depends(get_session)) -> ConversationsService:
    """
    Dependency function to create ConversationsService with user context.
    """
    context = extract_context_from_request(request)
    mindsdb_client = create_mindsdb_client_from_request(request, context)
    return ConversationsService(session=session, mindsdb_client=mindsdb_client, user_id=context.user_id, tenant_id=context.tenant_id)


@router.get("/")
async def list_conversations(
    conversations_service: ConversationsService = Depends(get_conversations_service),
    topic: str | None = Query(None, description="Filter by topic"),
    include_deleted: bool = Query(False, description="Filter by deleted status"),
    limit: int = Query(50, le=100, ge=1, description="Maximum number of conversations to return"),
    offset: int = Query(0, ge=0, description="Number of conversations to skip for pagination"),
    include_total: bool = Query(False, description="Include total count of conversations in response"),
    sort_by: Literal["topic", "created_at", "updated_at"] | None = Query(None, description="Field to sort by (topic, created_at, updated_at)"),
    sort_order: Literal["asc", "desc"] = Query("desc", description="Sort order (asc or desc)"),
) -> list[ConversationResponse] | dict[str, list[ConversationResponse] | int]:
    """
    List conversations for the authenticated user.

    Query Parameters:
        - topic: Filter by topic
        - include_deleted: Filter by deleted status
        - limit: Maximum number of conversations to return
        - offset: Number of conversations to skip for pagination
        - include_total: Include total count of conversations in response
        - sort_by: Field to sort by (created_at, updated_at)
        - sort_order: Sort order (asc or desc)

    Returns:
        List of conversations matching the specified criteria
    """
    logger.debug(f"List conversations requested (v1) for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}")

    try:
        result = await conversations_service.list_conversations(
            topic=topic,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
            include_total=include_total,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        if include_total:
            conversations, total = result
            logger.info(f"Listed {len(conversations)} conversations (total: {total}) for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}")
            return {"conversations": conversations, "total": total}
        else:
            logger.info(f"Listed conversations for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}")
            return result
    except ConversationsServiceError as e:
        logger.error(f"Service error in list_conversations for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Unexpected error in list_conversations for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: UUID,
    conversations_service: ConversationsService = Depends(get_conversations_service),
) -> ConversationResponse:
    """
    Get a conversation by ID.

    Args:
        conversation_id: ID of the conversation to get.

    Returns:
        ConversationResponse: Conversation.
    """
    logger.debug(f"Get conversation requested (v1) for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}")

    try:
        conversation = await conversations_service.get_conversation(conversation_id)
        logger.info(f"Retrieved conversation {conversation_id} for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}")
        return conversation
    except ConversationNotFoundError as e:
        logger.warning(f"Conversation not found for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from None
    except ConversationsServiceError as e:
        logger.error(f"Service error in get_conversation for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Unexpected error in get_conversation for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/{conversation_id}/items")
async def get_conversation_messages(
    conversation_id: UUID,
    conversations_service: ConversationsService = Depends(get_conversations_service),
) -> dict[str, list[MessageResponse] | str, bool]:
    """
    Get the messages of a conversation by ID.
    """
    logger.debug(f"Get conversation messages requested (v1) for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}")

    try:
        messages = await conversations_service.get_conversation_messages(conversation_id)
        return {
            "object": "list",
            "data": messages,
        }
    except ConversationsServiceError as e:
        logger.error(f"Service error in get_conversation_messages for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Unexpected error in get_conversation_messages for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.post("/")
async def create_conversation(
    conversation_data: ConversationCreateRequest,
    conversations_service: ConversationsService = Depends(get_conversations_service),
) -> ConversationResponse:
    """
    Create a new conversation.

    Request Body:
        - conversation_data: Conversation creation data, i.e., the topic of the conversation.

    Returns:
        ConversationResponse: Created conversation.
    """
    logger.debug(f"Create conversation requested (v1) for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}")
    
    try:
        conversation = await conversations_service.create_conversation(conversation_data)
        return conversation
    except ConversationAlreadyExistsError as e:
        logger.warning(f"Conversation already exists for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=409, detail=str(e)) from None
    except ConversationsServiceError as e:
        logger.error(f"Service error in create_conversation for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Unexpected error in create_conversation for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: UUID,
    conversations_service: ConversationsService = Depends(get_conversations_service),
) -> None:
    """
    Delete a conversation by ID.

    Args:
        conversation_id: ID of the conversation to delete.

    Returns:
        None: 204 No Content on successful deletion
    """
    logger.debug(f"Delete conversation requested (v1) for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}")
    try:
        await conversations_service.delete_conversation(conversation_id)
        logger.info(f"Deleted conversation {conversation_id} for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}")
        # Return nothing for 204 No Content
    except ConversationNotFoundError as e:
        logger.warning(f"Conversation not found for deletion for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from None
    except ConversationsServiceError as e:
        logger.error(f"Service error in delete_conversation for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Unexpected error in delete_conversation for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/{conversation_id}/items/{message_id}/result")
async def get_conversation_message_result(
    conversation_id: UUID,
    message_id: UUID,
    conversations_service: ConversationsService = Depends(get_conversations_service),
) -> dict[str, MessageResultResponse | int | bool]:
    """
    Get the result of a message by ID.
    """
    logger.debug(f"Get conversation message result requested (v1) for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}")

    try:
        result = await conversations_service.get_conversation_message_result(conversation_id, message_id)
        return {"data": result[0], "total": result[1], "is_pagination_consistent": result[2]}
    except ConversationNotFoundError as e:
        logger.warning(f"Conversation not found for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MessageNotFoundError as e:
        logger.warning(f"Message not found for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MessageNotAssistantError as e:
        logger.warning(f"Message is not an assistant message for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except MessageNoSQLQueryError as e:
        logger.warning(f"Message does not have a SQL query for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except ConversationsServiceError as e:
        logger.error(f"Service error in get_conversation_message_result for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Unexpected error in get_conversation_message_result for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/{conversation_id}/items/{message_id}/export")
async def export_conversation_message_result(
    conversation_id: UUID,
    message_id: UUID,
    conversations_service: ConversationsService = Depends(get_conversations_service),
) -> Response:
    """
    Export the result of a message by ID.
    """
    logger.debug(f"Export conversation message result requested (v1) for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}")

    try:
        result = await conversations_service.export_conversation_message_result(conversation_id, message_id)
        return Response(content=result, media_type="text/csv")
    except ConversationNotFoundError as e:
        logger.warning(f"Conversation not found for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MessageNotFoundError as e:
        logger.warning(f"Message not found for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MessageNotAssistantError as e:
        logger.warning(f"Message is not an assistant message for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except ConversationsServiceError as e:
        logger.error(f"Service error in export_conversation_message_result for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Unexpected error in export_conversation_message_result for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from None
