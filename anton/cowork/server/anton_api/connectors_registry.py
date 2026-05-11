"""Predefined connector registry.

Each .json file in `server/connectors/` is loaded once and cached.
Shape matches the cowork DataVaultForm spec, with three top-level
metadata fields the matcher reads:

    {
      "id":          "gmail",
      "label":       "Gmail",
      "aliases":     ["google mail", ...],
      "keywords":    ["email", "messaging", ...],
      "description": "...",
      "category":    "communication",
      "logo":        "mail",       // an Ico.<name> the renderer maps
      "logo_color":  "#EA4335",
      "form": { /* DataVaultForm spec — methods/fields/title/etc. */ }
    }

The matcher returns ranked candidates without needing an LLM for
exact / token-overlap hits. The optional LLM stage is wired in the
match endpoint and stays opt-in until the registry grows.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_CONNECTORS_DIR = Path(__file__).resolve().parent.parent / "connectors"
_CACHE: dict[str, dict] | None = None


def _load_all() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not _CONNECTORS_DIR.is_dir():
        return out
    for path in sorted(_CONNECTORS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # A broken file shouldn't break the whole registry.
            continue
        if not isinstance(data, dict):
            continue
        cid = data.get("id") or path.stem
        data["id"] = cid
        out[cid] = data
    return out


def all_connectors() -> dict[str, dict]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_all()
    return _CACHE


def get_connector(cid: str) -> dict | None:
    return all_connectors().get(cid)


def list_summaries() -> list[dict[str, Any]]:
    """Lightweight records suitable for picker UIs — no field schemas."""
    out: list[dict[str, Any]] = []
    for c in all_connectors().values():
        out.append({
            "id": c.get("id"),
            "label": c.get("label", c.get("id")),
            "description": c.get("description", ""),
            "category": c.get("category", "other"),
            "logo": c.get("logo"),
            "logo_color": c.get("logo_color"),
            "aliases": c.get("aliases", []),
        })
    out.sort(key=lambda x: (x.get("label") or "").lower())
    return out


def reload_connectors() -> None:
    """Force a re-read from disk. Useful in dev/testing."""
    global _CACHE
    _CACHE = None
