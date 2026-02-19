from __future__ import annotations

from enum import Enum
from typing import Union

from pydantic import BaseModel


class Phase(str, Enum):
    MEMORY_RECALL = "memory_recall"
    PLANNING = "planning"
    SKILL_DISCOVERY = "skill_discovery"
    SKILL_BUILDING = "skill_building"
    EXECUTING = "executing"
    COMPLETE = "complete"
    FAILED = "failed"


class StatusUpdate(BaseModel):
    type: str = "status_update"
    phase: Phase
    message: str
    eta_seconds: float | None = None


class TaskComplete(BaseModel):
    type: str = "task_complete"
    summary: str


class TaskFailed(BaseModel):
    type: str = "task_failed"
    error_summary: str


class PromptUser(BaseModel):
    type: str = "prompt_user"
    question: str


AntonEvent = Union[StatusUpdate, TaskComplete, TaskFailed, PromptUser]
