"""Anton-only local search for CoWork data."""

from __future__ import annotations

from fastapi import APIRouter

from ..anton_api import conversation_manager
from .artifacts import list_artifacts
from .attachments import get_attachments
from .cowork_state import load_state
from .projects import list_projects


router = APIRouter(prefix="/v1/search", tags=["search"])


def _score(text: str, query: str) -> int:
    haystack = text.lower()
    terms = [term for term in query.lower().split() if term]
    if not terms:
        return 0
    score = 0
    for term in terms:
        if term in haystack:
            score += 10
        score += haystack.count(term)
    return score


@router.get("")
async def search_cowork(q: str = "", limit: int = 25):
    query = q.strip()
    if not query:
        return {"results": []}

    results: list[dict] = []
    conversations = conversation_manager.list_conversations(limit=500, project="all")
    for conv in conversations:
        project_label = conv.get("project") or conv.get("project_path") or ""
        text = " ".join(
            [
                conv.get("title") or "",
                conv.get("preview") or "",
                project_label,
            ]
        )
        score = _score(text, query)
        if score:
            results.append(
                {
                    "type": "task",
                    "id": conv["id"],
                    "title": conv.get("title") or "Untitled task",
                    "subtitle": project_label or "Task",
                    "route": "task",
                    "score": score,
                }
            )

    project_listing = await list_projects()
    for project in project_listing.get("projects", []):
        text = " ".join([project.get("name") or "", project.get("path") or ""])
        score = _score(text, query)
        if score:
            results.append(
                {
                    "type": "project",
                    "id": project.get("name"),
                    "title": project.get("name") or "Project",
                    "subtitle": project.get("path") or "Project",
                    "route": "project",
                    "score": score,
                }
            )

    for artifact in await list_artifacts():
        # New artifact-card shape uses `title` for the display name
        # and adds `description`. Search across title + description +
        # path + kind so a match on any of those surfaces the card.
        text = " ".join([
            artifact.get("title") or "",
            artifact.get("description") or "",
            artifact.get("path") or "",
            artifact.get("kind") or "",
        ])
        score = _score(text, query)
        if score:
            results.append(
                {
                    "type": "artifact",
                    "id": artifact.get("path") or artifact.get("id") or artifact.get("title"),
                    "title": artifact.get("title") or "Artifact",
                    "subtitle": artifact.get("description") or artifact.get("path") or "Artifact",
                    "route": "artifacts",
                    "score": score,
                }
            )

    for attachment in get_attachments():
        text = " ".join(
            [
                attachment.get("name") or "",
                attachment.get("source") or "",
                attachment.get("sourceUrl") or "",
                attachment.get("textPreview") or "",
                attachment.get("projectPath") or "",
            ]
        )
        score = _score(text, query)
        if score:
            results.append(
                {
                    "type": "attachment",
                    "id": attachment["id"],
                    "sessionId": attachment.get("sessionId"),
                    "title": attachment.get("name") or "Attachment",
                    "subtitle": attachment.get("sourceUrl") or attachment.get("source") or "Attachment",
                    "route": "attachment",
                    "score": score,
                }
            )

    state = load_state()
    for schedule in state.get("schedules", []):
        text = " ".join([schedule.get("title") or "", schedule.get("prompt") or "", schedule.get("projectPath") or ""])
        score = _score(text, query)
        if score:
            results.append(
                {
                    "type": "schedule",
                    "id": schedule["id"],
                    "title": schedule.get("title") or "Scheduled task",
                    "subtitle": schedule.get("nextRunAt") or "Schedule",
                    "route": "scheduled",
                    "score": score,
                }
            )

    for index, pin in enumerate(state.get("pins", [])):
        text = " ".join([pin.get("title") or "", pin.get("id") or "", pin.get("type") or ""])
        score = _score(text, query)
        if score:
            results.append(
                {
                    "type": "pin",
                    "id": pin.get("id"),
                    "title": pin.get("title") or pin.get("id") or "Pinned item",
                    "subtitle": f"Pinned {pin.get('type', 'item')}",
                    "route": pin.get("type") or "task",
                    "score": score + max(0, 5 - index),
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    return {"results": results[:limit]}
