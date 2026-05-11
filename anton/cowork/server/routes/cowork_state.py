"""Persistent local state for Anton CoWork desktop features."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_VERSION = 3


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def cowork_dir() -> Path:
    base = os.environ.get("ANTON_COWORK_STATE_DIR")
    if base:
        path = Path(base).expanduser()
    else:
        path = Path.home() / ".anton" / "cowork"
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


def state_path() -> Path:
    return cowork_dir() / "state.json"


def attachments_dir() -> Path:
    path = cowork_dir() / "attachments"
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


def backups_dir() -> Path:
    path = cowork_dir() / "backups"
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


def default_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "attachments": {},
        "pins": [],
        "preferences": {},
        "visit_counts": {},
        "schedules": [],
        "schedule_runs": {},
        "search_metadata": {},
        "utility_state": {},
        "publish_history": [],
        "migrations": [],
    }


def _backup_file(path: Path, label: str) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target = backups_dir() / f"{path.stem}.{label}.{stamp}{path.suffix}"
    try:
        shutil.copy2(path, target)
    except OSError:
        return None
    return target


def _normalise_state(data: dict[str, Any]) -> dict[str, Any]:
    state = default_state()
    previous_version = data.get("version", 0)
    state.update(data)

    expected_types = {
        "attachments": dict,
        "pins": list,
        "preferences": dict,
        "visit_counts": dict,
        "schedules": list,
        "schedule_runs": dict,
        "search_metadata": dict,
        "utility_state": dict,
        "publish_history": list,
        "migrations": list,
    }
    for key, expected in expected_types.items():
        if not isinstance(state.get(key), expected):
            state[key] = default_state()[key]

    if previous_version != STATE_VERSION:
        state["migrations"].append({
            "from": previous_version,
            "to": STATE_VERSION,
            "at": utc_now_iso(),
        })
    state["version"] = STATE_VERSION
    return state


def load_state() -> dict[str, Any]:
    path = state_path()
    if not path.exists():
        return default_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        _backup_file(path, "corrupt")
        return default_state()

    if not isinstance(data, dict):
        _backup_file(path, "invalid")
        return default_state()
    state = _normalise_state(data)
    if state != data:
        save_state(state)
    return state


def save_state(state: dict[str, Any]) -> dict[str, Any]:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    tmp.replace(path)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return state


def update_state(mutator):
    state = load_state()
    result = mutator(state)
    save_state(state)
    return result
