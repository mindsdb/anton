from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

from pydantic import BaseModel


class SkillResult(BaseModel):
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


@dataclass
class SkillInfo:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema dict
    execute: Callable[..., Coroutine[Any, Any, SkillResult]]
    source_path: Path | None = None


# Python type -> JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _build_parameters_schema(func: Callable) -> dict[str, Any]:
    sig = inspect.signature(func)
    hints = {}
    try:
        hints = inspect.get_annotations(func, eval_str=True)
    except Exception:
        hints = getattr(func, "__annotations__", {})

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        annotation = hints.get(param_name, str)
        json_type = _TYPE_MAP.get(annotation, "string")
        properties[param_name] = {"type": json_type}

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def skill(name: str, description: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        schema = _build_parameters_schema(func)
        info = SkillInfo(
            name=name,
            description=description,
            parameters=schema,
            execute=func,
        )
        func._skill_info = info  # type: ignore[attr-defined]
        return func

    return decorator
