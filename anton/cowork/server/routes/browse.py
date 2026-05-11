"""Browse capability gate for Anton CoWork.

Anton CoWork supports specific URL context through the attachment system. Broad
web browsing is only enabled when the installed Anton package exposes a real
browse/search tool surface that can be called from sessions.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/browse", tags=["browse"])


def _detect_browse_capability() -> dict:
    try:
        import anton
    except Exception as exc:
        logger.warning("Anton import failed: %s", exc)
        return {
            "supported": False,
            "mode": "url_context_only",
            "reason": "Anton is unavailable",
            "evidence": [],
        }

    evidence: list[str] = []
    callable_markers = {"browse", "web_search", "search_web", "fetch_web", "web_fetch"}
    module_markers = ("browse", "web", "search")

    try:
        import anton.tools as anton_tools
        for name in dir(anton_tools):
            lowered = name.lower()
            if any(marker in lowered for marker in callable_markers):
                value = getattr(anton_tools, name)
                if callable(value) or str(name).isupper():
                    evidence.append(f"anton.tools.{name}")
    except Exception:
        pass

    try:
        for module in pkgutil.walk_packages(anton.__path__, anton.__name__ + "."):
            lowered = module.name.lower()
            if any(marker in lowered for marker in module_markers):
                evidence.append(module.name)
    except Exception:
        pass

    verified = []
    for name in evidence:
        if not name.startswith("anton."):
            continue
        if name.count(".") >= 2:
            module_name = name.rsplit(".", 1)[0]
        else:
            module_name = name
        try:
            importlib.import_module(module_name)
            verified.append(name)
        except Exception:
            continue

    supported = any("browse" in item.lower() or "web" in item.lower() for item in verified)
    return {
        "supported": supported,
        "mode": "anton_browse" if supported else "url_context_only",
        "reason": (
            "Installed Anton exposes a browse/web capability."
            if supported
            else "Installed Anton does not expose a callable broad web browse tool; use URL context."
        ),
        "evidence": sorted(set(verified))[:20],
    }


@router.get("/status")
async def browse_status():
    return _detect_browse_capability()
