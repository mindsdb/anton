"""Pinning and visit-count APIs for Anton CoWork."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .cowork_state import load_state, save_state, utc_now_iso


router = APIRouter(prefix="/v1/pins", tags=["pins"])


class PinRequest(BaseModel):
    item_type: str = "task"
    item_id: str
    title: str | None = None


class ReorderRequest(BaseModel):
    item_ids: list[str]


def _pin_key(item_type: str, item_id: str) -> str:
    return f"{item_type}:{item_id}"


def is_pinned(item_id: str, item_type: str = "task") -> bool:
    key = _pin_key(item_type, item_id)
    state = load_state()
    return any(item.get("key") == key for item in state.get("pins", []))


def _upsert_pin(state: dict, item_type: str, item_id: str, title: str | None = None) -> dict:
    key = _pin_key(item_type, item_id)
    now = utc_now_iso()
    existing = next((item for item in state["pins"] if item.get("key") == key), None)
    if existing:
        existing["updatedAt"] = now
        if title:
            existing["title"] = title
        return existing
    pin = {
        "key": key,
        "type": item_type,
        "id": item_id,
        "title": title or item_id,
        "createdAt": now,
        "updatedAt": now,
    }
    state["pins"].insert(0, pin)
    return pin


@router.get("")
def list_pins():
    state = load_state()
    return {"pins": state.get("pins", [])}


@router.post("")
def pin_item(request: PinRequest):
    if request.item_type not in {"task", "project", "artifact", "schedule"}:
        raise HTTPException(status_code=400, detail="Unsupported pin type.")
    state = load_state()
    pin = _upsert_pin(state, request.item_type, request.item_id, request.title)
    save_state(state)
    return {"pin": pin, "pins": state["pins"]}


@router.delete("/{item_id}")
def unpin_task(item_id: str, item_type: str = "task"):
    state = load_state()
    key = _pin_key(item_type, item_id)
    before = len(state.get("pins", []))
    state["pins"] = [item for item in state.get("pins", []) if item.get("key") != key]
    save_state(state)
    if len(state["pins"]) == before:
        raise HTTPException(status_code=404, detail="Pin not found.")
    return {"ok": True, "pins": state["pins"]}


@router.put("/reorder")
def reorder_pins(request: ReorderRequest):
    state = load_state()
    pins_by_id = {item.get("id"): item for item in state.get("pins", [])}
    reordered = [pins_by_id[item_id] for item_id in request.item_ids if item_id in pins_by_id]
    existing_ids = {item.get("id") for item in reordered}
    reordered.extend(item for item in state.get("pins", []) if item.get("id") not in existing_ids)
    state["pins"] = reordered
    save_state(state)
    return {"pins": reordered}


@router.post("/{task_id}/visit")
def record_task_visit(task_id: str, auto_pin: bool = False, title: str | None = None):
    state = load_state()
    visits = state.setdefault("visit_counts", {})
    visits[task_id] = int(visits.get(task_id, 0)) + 1
    pin = None
    if auto_pin and visits[task_id] >= 3:
        pin = _upsert_pin(state, "task", task_id, title)
    save_state(state)
    return {"taskId": task_id, "visits": visits[task_id], "autoPinned": bool(pin), "pin": pin}
