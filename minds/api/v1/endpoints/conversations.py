"""
Conversation management endpoints for API v1.

This module contains endpoints for CRUD operations on conversations,
providing a clean v1 API interface for conversation management.
"""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from minds.api.v1.deps import get_conversations_service, get_minds_service
from minds.common.logger import get_logger
from minds.schemas.charts import ChartImageResponse, ChartOutputFormat, ChartRequest, ChartResponse
from minds.schemas.conversations import ConversationCreateRequest, ConversationResponse
from minds.schemas.messages import MessageResponse, MessageResultResponse
from minds.services.conversations import (
    AgentNotAntonError,
    ConversationNotFoundError,
    ConversationsService,
    ConversationsServiceError,
    InvalidSQLQueryError,
    MessageNoSQLQueryError,
    MessageNotAssistantError,
    MessageNotFoundError,
)
from minds.services.minds import MindNotFoundError, MindsService

logger = get_logger(__name__)

router = APIRouter()

# Fixed cache-control headers for chart image responses.
# "no-store" prevents browsers/proxies from caching potentially sensitive query results.
DIRECT_CHART_HEADERS = {"Cache-Control": "no-store"}
# "private" ensures the response is not cached by shared caches (CDNs, proxies).
IMAGE_URL_CHART_HEADERS = {"Cache-Control": "private, no-store"}


@router.get("/")
async def list_conversations(
    conversations_service: ConversationsService = Depends(get_conversations_service),
    topic: str | None = Query(None, description="Filter by topic"),
    # include_deleted: bool = Query(False, description="Filter by deleted status"),
    limit: int = Query(50, le=100, ge=1, description="Maximum number of conversations to return"),
    offset: int = Query(0, ge=0, description="Number of conversations to skip for pagination"),
    include_total: bool = Query(False, description="Include total count of conversations in response"),
    sort_by: Literal["topic", "created_at", "updated_at"] | None = Query(
        None, description="Field to sort by (topic, created_at, updated_at)"
    ),
    sort_order: Literal["asc", "desc"] = Query("desc", description="Sort order (asc or desc)"),
) -> list[ConversationResponse] | dict[str, list[ConversationResponse] | int]:
    """
    List conversations for the authenticated user.

    Query Parameters:
        - topic: Filter by topic
        - limit: Maximum number of conversations to return
        - offset: Number of conversations to skip for pagination
        - include_total: Include total count of conversations in response
        - sort_by: Field to sort by (created_at, updated_at)
        - sort_order: Sort order (asc or desc)

    Returns:
        List of conversations matching the specified criteria
    """
    logger.debug(
        f"List conversations requested (v1) "
        f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}"
    )

    try:
        result = await conversations_service.list_conversations(
            topic=topic,
            limit=limit,
            offset=offset,
            include_total=include_total,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        if include_total:
            conversations, total = result
            logger.info(
                f"Listed {len(conversations)} conversations (total: {total}) "
                f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}"
            )
            return {"conversations": conversations, "total": total}
        else:
            logger.info(
                f"Listed conversations "
                f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}"
            )
            return result
    except ConversationsServiceError as e:
        logger.error(
            f"Service error in list_conversations "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in list_conversations "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
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
    logger.debug(
        f"Get conversation requested (v1) "
        f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}"
    )

    try:
        conversation = await conversations_service.get_conversation(conversation_id)
        logger.info(
            f"Retrieved conversation {conversation_id} "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}"
        )
        return conversation
    except ConversationNotFoundError as e:
        logger.warning(
            f"Conversation not found "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except ConversationsServiceError as e:
        logger.error(
            f"Service error in get_conversation "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in get_conversation "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/{conversation_id}/items")
async def get_conversation_messages(
    conversation_id: UUID,
    conversations_service: ConversationsService = Depends(get_conversations_service),
    with_events: bool = Query(True, description="Whether to include events in the response"),
) -> dict[Literal["object", "data"], list[MessageResponse] | str]:
    """
    Get the messages of a conversation by ID.
    """
    logger.debug(
        f"Get conversation messages requested (v1) "
        f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}"
    )

    try:
        messages = await conversations_service.get_conversation_messages(
            conversation_id,
            with_sql_query=True,
            with_events=with_events,
        )
        return {
            "object": "list",
            "data": messages,
        }
    except ConversationsServiceError as e:
        logger.error(
            f"Service error in get_conversation_messages "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in get_conversation_messages "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.post("/")
async def create_conversation(
    conversation_data: ConversationCreateRequest,
    conversations_service: ConversationsService = Depends(get_conversations_service),
    mind_service: MindsService = Depends(get_minds_service),
) -> ConversationResponse:
    """
    Create a new conversation.

    Request Body:
        - conversation_data: Conversation creation data, i.e., the topic of the conversation.

    Returns:
        ConversationResponse: Created conversation.
    """
    logger.debug(
        f"Create conversation requested (v1) "
        f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}"
    )

    try:
        conversation = await conversations_service.create_conversation(conversation_data, mind_service)
        return conversation
    except MindNotFoundError as e:
        logger.warning(
            f"Mind not found for user {conversations_service.user_id} in "
            f"organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except ConversationsServiceError as e:
        logger.error(
            f"Service error in create_conversation "
            f"for user {conversations_service.user_id} in "
            f"organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in create_conversation "
            f"for user {conversations_service.user_id} in "
            f"organization {conversations_service.organization_id}: {e}"
        )
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
    logger.debug(
        f"Delete conversation requested (v1) "
        f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}"
    )
    try:
        await conversations_service.delete_conversation(conversation_id)
        logger.info(
            f"Deleted conversation {conversation_id} "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}"
        )
        # Return nothing for 204 No Content
    except ConversationNotFoundError as e:
        logger.warning(
            f"Conversation not found for deletion "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except ConversationsServiceError as e:
        logger.error(
            f"Service error in delete_conversation "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in delete_conversation "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/{conversation_id}/items/{message_id}/result")
async def get_conversation_message_result(
    conversation_id: UUID,
    message_id: UUID,
    limit: int = Query(100, le=1000, ge=1, description="Maximum number of rows to return"),
    offset: int = Query(0, ge=0, description="Number of rows to skip for pagination"),
    conversations_service: ConversationsService = Depends(get_conversations_service),
) -> dict[Literal["result", "total", "is_pagination_consistent"], MessageResultResponse | int | bool]:
    """
    Get the result of a message by ID.
    """
    logger.debug(
        f"Get conversation message result requested (v1) "
        f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}"
    )

    try:
        result = await conversations_service.get_conversation_message_result(
            conversation_id,
            message_id,
            limit=limit,
            offset=offset,
        )
        return {"result": result[0], "total": result[1], "is_pagination_consistent": result[2]}
    except ConversationNotFoundError as e:
        logger.warning(
            f"Conversation not found "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MessageNotFoundError as e:
        logger.warning(
            f"Message not found "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MessageNotAssistantError as e:
        logger.warning(
            f"Message is not an assistant message "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except MessageNoSQLQueryError as e:
        logger.warning(
            f"Message does not have a SQL query "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except InvalidSQLQueryError as e:
        logger.warning(
            f"Invalid SQL query "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except ValueError as e:
        logger.warning(
            f"Invalid pagination parameters "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except ConversationsServiceError as e:
        logger.error(
            f"Service error in get_conversation_message_result "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in get_conversation_message_result "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
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
    logger.debug(
        f"Export conversation message result requested (v1) "
        f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}"
    )

    try:
        result = await conversations_service.export_conversation_message_result(conversation_id, message_id)
        return Response(content=result, media_type="text/csv")
    except ConversationNotFoundError as e:
        logger.warning(
            f"Conversation not found "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MessageNotFoundError as e:
        logger.warning(
            f"Message not found "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MessageNotAssistantError as e:
        logger.warning(
            f"Message is not an assistant message "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except InvalidSQLQueryError as e:
        logger.warning(
            f"Invalid SQL query "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except ConversationsServiceError as e:
        logger.error(
            f"Service error in export_conversation_message_result "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in export_conversation_message_result "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.post(
    "/{conversation_id}/items/{message_id}/chart",
    response_model=None,
    responses={
        200: {
            "description": "Chart output whose shape depends on the `output` field in the request body.",
            "content": {
                "application/json": {
                    "description": "ChartResponse (output=chartjs) or ChartImageResponse (output=image_url)",
                },
                "image/png": {"description": "PNG bytes (output=png)"},
            },
        },
    },
)
async def generate_chart(
    conversation_id: UUID,
    message_id: UUID,
    req: ChartRequest,
    conversations_service: ConversationsService = Depends(get_conversations_service),
) -> ChartResponse | ChartImageResponse | Response:
    """
    Generate a chart from a message's SQL query results.

    The ``output`` field in the request body controls the response format:

    - **chartjs** *(default)* — Returns a complete Chart.js configuration
      for frontend rendering.
    - **png** — Compiles and server-renders the chart, returning PNG bytes
      directly.  Intended for immediate consumers (e.g. Slack upload).
    - **image_url** — Returns an authenticated URL that can be used to fetch
      a server-rendered image later via ``GET .../chart``. The URL renders
      the chart on demand when fetched.
    """
    user_id = conversations_service.user_id
    org_id = conversations_service.organization_id
    logger.debug(f"Chart requested (output={req.output.value}) for user {user_id} in organization {org_id}")

    try:
        if req.output is ChartOutputFormat.CHARTJS:
            return await conversations_service.get_conversation_message_chart(
                conversation_id,
                message_id,
                req.intent,
            )

        if req.output is ChartOutputFormat.PNG:
            image_bytes = await conversations_service.render_conversation_message_chart_png(
                conversation_id,
                message_id,
                req.intent,
            )
            return Response(
                content=image_bytes,
                media_type="image/png",
                headers=DIRECT_CHART_HEADERS,
            )

        return await conversations_service.get_conversation_message_chart_image(
            conversation_id,
            message_id,
            req.intent,
        )

    except ConversationNotFoundError as e:
        logger.warning(f"Conversation not found for user {user_id} in organization {org_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MessageNotFoundError as e:
        logger.warning(f"Message not found for user {user_id} in organization {org_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MessageNotAssistantError as e:
        logger.warning(f"Message is not an assistant message for user {user_id} in organization {org_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except MessageNoSQLQueryError as e:
        logger.warning(f"Message does not have a SQL query for user {user_id} in organization {org_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except InvalidSQLQueryError as e:
        logger.warning(f"Invalid SQL query for user {user_id} in organization {org_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except ValueError as e:
        logger.warning(f"Invalid chart request for user {user_id} in organization {org_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from None
    except ConversationsServiceError as e:
        logger.error(f"Service error in generate_chart for user {user_id} in organization {org_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from None
    except Exception as e:
        logger.error(f"Unexpected error in generate_chart for user {user_id} in organization {org_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/{conversation_id}/items/{message_id}/chart")
async def get_chart_image(
    conversation_id: UUID,
    message_id: UUID,
    token: str = Query(
        ...,
        description="Opaque chart token returned by the chart generation endpoint.",
    ),
    conversations_service: ConversationsService = Depends(get_conversations_service),
) -> Response:
    """
    Serve a server-rendered chart image for a message.

    This is the URL returned by ``POST .../chart`` when ``output`` is
    ``image_url``. The URL is authenticated and renders the chart on demand.
    """
    try:
        image_bytes = await conversations_service.render_conversation_message_chart_by_token(
            conversation_id,
            message_id,
            token,
        )
        return Response(
            content=image_bytes,
            media_type="image/png",
            headers=IMAGE_URL_CHART_HEADERS,
        )
    except (ConversationNotFoundError, MessageNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except (
        MessageNotAssistantError,
        MessageNoSQLQueryError,
        InvalidSQLQueryError,
        ValueError,
    ) as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except ConversationsServiceError as e:
        logger.error(
            f"Service error in get_chart_image "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in get_chart_for_message "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.head("/{conversation_id}/items/{message_id}/report")
async def check_conversation_message_report_exists(
    conversation_id: UUID,
    message_id: UUID,
    conversations_service: ConversationsService = Depends(get_conversations_service),
) -> None:
    """
    Check if a report exists for a message by ID.
    """
    logger.debug(
        f"Check conversation message report exists requested (v1) "
        f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}"
    )

    try:
        await conversations_service.check_conversation_message_report_exists(conversation_id, message_id)
    # FileNotFoundError is the obvious 404 here.
    except FileNotFoundError as e:
        logger.warning(
            f"Report not found "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    # The other errors here that are more related to the conversation or message,
    # will also be treated as 404s for this endpoint.
    except ConversationNotFoundError as e:
        logger.warning(
            f"Conversation not found "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MessageNotFoundError as e:
        logger.warning(
            f"Message not found "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MessageNotAssistantError as e:
        logger.warning(
            f"Message is not an assistant message "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    # This is also treated as a 404 to avoid having to make conditional calls in the UI.
    # It will be treated as a 400 in the GET endpoint below.
    except AgentNotAntonError as e:
        logger.warning(
            f"Mind is not using the Anton agent "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    # This is a 400 because the agent name is invalid, so it is a user error.
    # This is unlikely because it is checked previously.
    except ValueError as e:
        logger.warning(
            f"Invalid agent name "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    # These are unexpected errors, so we return a 500.
    except ConversationsServiceError as e:
        logger.error(
            f"Service error in check_conversation_message_report_exists "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in check_conversation_message_report_exists "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None


@router.get("/{conversation_id}/items/{message_id}/report")
async def get_conversation_message_report(
    conversation_id: UUID,
    message_id: UUID,
    conversations_service: ConversationsService = Depends(get_conversations_service),
) -> Response:
    """
    Get the report of a message by ID.
    """
    logger.debug(
        f"Get conversation message report requested (v1) "
        f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}"
    )

    try:
        report = await conversations_service.get_conversation_message_report(conversation_id, message_id)
        return Response(content=report, media_type="text/html")
    except ConversationNotFoundError as e:
        logger.warning(
            f"Conversation not found "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MessageNotFoundError as e:
        logger.warning(
            f"Message not found "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MessageNotAssistantError as e:
        logger.warning(
            f"Message is not an assistant message "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except AgentNotAntonError as e:
        logger.warning(
            f"Mind is not using the Anton agent "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except ValueError as e:
        logger.warning(
            f"Invalid agent name "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=400, detail=str(e)) from None
    except FileNotFoundError as e:
        logger.warning(
            f"Report not found "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e)) from None
    except ConversationsServiceError as e:
        logger.error(
            f"Service error in get_conversation_message_report "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e)) from None
    except Exception as e:
        logger.error(
            f"Unexpected error in get_conversation_message_report "
            f"for user {conversations_service.user_id} in organization {conversations_service.organization_id}: {e}"
        )
        raise HTTPException(status_code=500, detail="Internal server error") from None
