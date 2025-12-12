"""
Conversation management endpoints for API v1.

This module contains endpoints for CRUD operations on conversations,
providing a clean v1 API interface for conversation management.
"""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlmodel import Session

from minds.common.logger import setup_logging
from minds.db.pg_session import get_session
from minds.requests.context import extract_context_from_request
from minds.schemas.conversations import ConversationCreateRequest, ConversationDetailedResponse, ConversationResponse
from minds.services.conversations import ConversationsService, ConversationAlreadyExistsError, ConversationNotFoundError, ConversationsServiceError

logger = setup_logging()

router = APIRouter()


def get_conversations_service(request: Request, session: Session = Depends(get_session)) -> ConversationsService:
    """
    Dependency function to create ConversationsService with user context.
    """
    context = extract_context_from_request(request)
    return ConversationsService(session=session, user_id=context.user_id, tenant_id=context.tenant_id)


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
    with_messages: bool = Query(False, description="Include messages in the conversation"),
) -> ConversationResponse | ConversationDetailedResponse:
    """
    Get a conversation by ID.

    Query Parameters:
        - with_messages: Include messages in the conversation

    Returns:
        ConversationResponse | ConversationDetailedResponse: Conversation.
    """
    logger.debug(f"Get conversation requested (v1) for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}")

    try:
        conversation = await conversations_service.get_conversation(conversation_id, with_messages=with_messages)
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
        logger.info(f"Created conversation {conversation_data.topic} for user {conversations_service.user_id} in tenant {conversations_service.tenant_id}")
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
