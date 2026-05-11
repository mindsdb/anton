"""Scratchpad runtime pool for /v1/scratchpad/*.

Manages a bounded set of named LocalScratchpadRuntime instances. Each pad
has its own venv, can execute code, install packages, and be reset/cancelled.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional


MAX_PADS = int(os.environ.get("ANTON_SERVER_MAX_PADS", "5"))


_pads: dict[str, object] = {}  # name -> LocalScratchpadRuntime
last_activity: float = time.time()


def _touch_activity() -> None:
    global last_activity
    last_activity = time.time()


def _resolve_workspace(workspace_path: Optional[str]) -> Path:
    return (
        Path(workspace_path).expanduser().resolve()
        if workspace_path
        else Path.cwd().resolve()
    )


def _resolve_coding(
    *,
    coding_provider: str,
    coding_model: str,
    coding_api_key: str,
    coding_base_url: str,
) -> tuple[str, str, str, str]:
    """Fill in any blank coding fields from AntonSettings."""
    from anton.config.settings import AntonSettings

    s = AntonSettings()
    provider = coding_provider or s.coding_provider or ""
    model = coding_model or s.coding_model or ""
    if coding_api_key:
        api_key = coding_api_key
    elif provider == "anthropic":
        api_key = s.anthropic_api_key or ""
    else:
        api_key = s.openai_api_key or ""
    base_url = coding_base_url or (s.openai_base_url or "")
    return provider, model, api_key, base_url


def _make_runtime(
    name: str,
    *,
    workspace_path: Optional[str],
    coding_provider: str,
    coding_model: str,
    coding_api_key: str,
    coding_base_url: str,
):
    from anton.core.backends.local import LocalScratchpadRuntime

    return LocalScratchpadRuntime(
        name,
        coding_provider=coding_provider,
        coding_model=coding_model,
        coding_api_key=coding_api_key,
        coding_base_url=coding_base_url,
        workspace_path=_resolve_workspace(workspace_path),
    )


def get(name: str):
    _touch_activity()
    return _pads.get(name)


def get_or_create(
    name: str,
    *,
    workspace_path: Optional[str] = None,
    coding_provider: str = "",
    coding_model: str = "",
    coding_api_key: str = "",
    coding_base_url: str = "",
):
    _touch_activity()
    if name in _pads:
        return _pads[name]
    if len(_pads) >= MAX_PADS:
        raise RuntimeError(
            f"Maximum concurrent scratchpads ({MAX_PADS}) reached. "
            f"Close an existing pad first."
        )
    provider, model, api_key, base_url = _resolve_coding(
        coding_provider=coding_provider,
        coding_model=coding_model,
        coding_api_key=coding_api_key,
        coding_base_url=coding_base_url,
    )
    pad = _make_runtime(
        name,
        workspace_path=workspace_path,
        coding_provider=provider,
        coding_model=model,
        coding_api_key=api_key,
        coding_base_url=base_url,
    )
    _pads[name] = pad
    return pad


def remove(name: str) -> None:
    _pads.pop(name, None)


def list_pads() -> list[str]:
    return list(_pads.keys())


async def close_all() -> None:
    for name in list(_pads):
        try:
            pad = _pads[name]
            await pad.close()
        except Exception:
            pass
    _pads.clear()
