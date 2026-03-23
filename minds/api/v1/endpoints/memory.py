"""
Admin endpoints for managing mind(anton) memory rules and topics.

All endpoints require admin role and are scoped to a specific mind.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from minds.api.v1.deps import get_memory_admin_service, get_minds_service, require_admin
from minds.common.logger import get_logger
from minds.schemas.memory import (
    MemoryRuleCreateRequest,
    MemoryRuleResponse,
    MemoryRuleUpdateRequest,
    MemoryTopicCreateRequest,
    MemoryTopicResponse,
    MemoryTopicUpdateRequest,
)
from minds.services.memory import MemoryAdminService, MemoryConflictError, MemoryNotFoundError
from minds.services.minds import MindNotFoundError, MindsService, MindsServiceError

logger = get_logger(__name__)
router = APIRouter()


async def _resolve_mind_id(mind_name: str, minds_service: MindsService) -> UUID:
    """Look up mind_id from mind_name, raising 404 if not found."""
    try:
        mind = await minds_service.get_mind_model(mind_name=mind_name)
    except MindNotFoundError:
        raise HTTPException(status_code=404, detail=f"Mind '{mind_name}' not found") from None
    except MindsServiceError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    if mind.id is None:
        raise HTTPException(status_code=500, detail="Internal server error")
    return mind.id


@router.get("/rules", response_model=list[MemoryRuleResponse])
async def list_rules(
    mind_name: str,
    _: None = Depends(require_admin),
    minds_service: MindsService = Depends(get_minds_service),
    memory_service: MemoryAdminService = Depends(get_memory_admin_service),
) -> list[MemoryRuleResponse]:
    mind_id = await _resolve_mind_id(mind_name, minds_service)
    rules = memory_service.list_rules(mind_id)
    logger.info(f"Listed {len(rules)} memory rules for mind '{mind_name}'")
    return [MemoryRuleResponse(**r.model_dump()) for r in rules]


@router.post("/rules", response_model=MemoryRuleResponse, status_code=201)
async def create_rule(
    mind_name: str,
    body: MemoryRuleCreateRequest,
    _: None = Depends(require_admin),
    minds_service: MindsService = Depends(get_minds_service),
    memory_service: MemoryAdminService = Depends(get_memory_admin_service),
) -> MemoryRuleResponse:
    mind_id = await _resolve_mind_id(mind_name, minds_service)
    rule = memory_service.create_rule(
        mind_id=mind_id,
        rule_type=body.rule_type,
        content=body.content,
    )
    logger.info(f"Created memory rule {rule.id} for mind '{mind_name}'")
    return MemoryRuleResponse(**rule.model_dump())


@router.patch("/rules/{rule_id}", response_model=MemoryRuleResponse)
async def update_rule(
    mind_name: str,
    rule_id: UUID,
    body: MemoryRuleUpdateRequest,
    _: None = Depends(require_admin),
    minds_service: MindsService = Depends(get_minds_service),
    memory_service: MemoryAdminService = Depends(get_memory_admin_service),
) -> MemoryRuleResponse:
    mind_id = await _resolve_mind_id(mind_name, minds_service)
    try:
        rule = memory_service.update_rule(
            mind_id=mind_id,
            rule_id=rule_id,
            rule_type=body.rule_type,
            content=body.content,
        )
    except MemoryNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    logger.info(f"Updated memory rule {rule_id} for mind '{mind_name}'")
    return MemoryRuleResponse(**rule.model_dump())


@router.delete("/rules/{rule_id}", status_code=204)
async def delete_rule(
    mind_name: str,
    rule_id: UUID,
    _: None = Depends(require_admin),
    minds_service: MindsService = Depends(get_minds_service),
    memory_service: MemoryAdminService = Depends(get_memory_admin_service),
) -> None:
    mind_id = await _resolve_mind_id(mind_name, minds_service)
    try:
        memory_service.delete_rule(mind_id=mind_id, rule_id=rule_id)
    except MemoryNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    logger.info(f"Deleted memory rule {rule_id} for mind '{mind_name}'")


@router.get("/topics", response_model=list[MemoryTopicResponse])
async def list_topics(
    mind_name: str,
    _: None = Depends(require_admin),
    minds_service: MindsService = Depends(get_minds_service),
    memory_service: MemoryAdminService = Depends(get_memory_admin_service),
) -> list[MemoryTopicResponse]:
    mind_id = await _resolve_mind_id(mind_name, minds_service)
    topics = memory_service.list_topics(mind_id)
    logger.info(f"Listed {len(topics)} memory topics for mind '{mind_name}'")
    return [MemoryTopicResponse(**t.model_dump()) for t in topics]


@router.post("/topics", response_model=MemoryTopicResponse, status_code=201)
async def create_topic(
    mind_name: str,
    body: MemoryTopicCreateRequest,
    _: None = Depends(require_admin),
    minds_service: MindsService = Depends(get_minds_service),
    memory_service: MemoryAdminService = Depends(get_memory_admin_service),
) -> MemoryTopicResponse:
    mind_id = await _resolve_mind_id(mind_name, minds_service)
    try:
        topic = memory_service.create_topic(
            mind_id=mind_id,
            title=body.title,
            body=body.body,
            tags=body.tags,
            description=body.description,
        )
    except MemoryConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from None
    logger.info(f"Created memory topic {topic.id} for mind '{mind_name}'")
    return MemoryTopicResponse(**topic.model_dump())


@router.patch("/topics/{topic_id}", response_model=MemoryTopicResponse)
async def update_topic(
    mind_name: str,
    topic_id: UUID,
    body: MemoryTopicUpdateRequest,
    _: None = Depends(require_admin),
    minds_service: MindsService = Depends(get_minds_service),
    memory_service: MemoryAdminService = Depends(get_memory_admin_service),
) -> MemoryTopicResponse:
    mind_id = await _resolve_mind_id(mind_name, minds_service)
    try:
        topic = memory_service.update_topic(
            mind_id=mind_id,
            topic_id=topic_id,
            title=body.title,
            body=body.body,
            tags=body.tags,
            description=body.description,
        )
    except MemoryNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except MemoryConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from None
    logger.info(f"Updated memory topic {topic_id} for mind '{mind_name}'")
    return MemoryTopicResponse(**topic.model_dump())


@router.delete("/topics/{topic_id}", status_code=204)
async def delete_topic(
    mind_name: str,
    topic_id: UUID,
    _: None = Depends(require_admin),
    minds_service: MindsService = Depends(get_minds_service),
    memory_service: MemoryAdminService = Depends(get_memory_admin_service),
) -> None:
    mind_id = await _resolve_mind_id(mind_name, minds_service)
    try:
        memory_service.delete_topic(mind_id=mind_id, topic_id=topic_id)
    except MemoryNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    logger.info(f"Deleted memory topic {topic_id} for mind '{mind_name}'")
