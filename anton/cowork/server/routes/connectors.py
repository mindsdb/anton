"""Connector registry endpoints.

  GET  /v1/connectors                → list summaries (id, label, description, category, logo)
  GET  /v1/connectors/{id}           → full spec (the literal-retrieval path; no LLM)
  POST /v1/connectors/match          → natural-language → ranked candidates
  POST /v1/connectors/{id}/save      → save a connection using the connector's own
                                       JSON-declared field schema (bypasses Anton-core's
                                       built-in datasource registry — needed for OAuth
                                       and any other auth shapes that don't fit the
                                       legacy email/password engines)

The match endpoint runs a three-stage cascade so most calls finish
without an LLM:

  Stage 1  exact match (id or alias, normalized)         → 1 candidate
  Stage 2  token-overlap scoring                         → 1 confident, else multi
  Stage 3  LLM tiebreak (Haiku, optional, future)        → not yet wired

Stages 1-2 cover the common cases — clicks from the UI, canonical
names from the chat agent — without paying any model cost.
"""
from __future__ import annotations

import re
import uuid
from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..anton_api import connectors_registry as registry


router = APIRouter(prefix="/v1/connectors", tags=["connectors"])


class MatchRequest(BaseModel):
    query: str
    max_candidates: int = Field(default=3, ge=1, le=5)


@router.get("")
def list_connectors() -> dict:
    return {"connectors": registry.list_summaries()}


@router.get("/{connector_id}")
def get_connector(connector_id: str) -> dict:
    c = registry.get_connector(connector_id)
    if not c:
        raise HTTPException(status_code=404, detail="Connector not found")
    return c


# ─── Matching helpers ──────────────────────────────────────────────


def _normalize(s: str) -> str:
    """Lowercase + collapse non-alphanumerics to single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _exact_match(query: str) -> str | None:
    nq = _normalize(query)
    if not nq:
        return None
    for c in registry.all_connectors().values():
        if _normalize(c.get("id", "")) == nq:
            return c["id"]
        for alias in c.get("aliases", []):
            if _normalize(alias) == nq:
                return c["id"]
    return None


def _token_score(query: str, c: dict) -> float:
    """Weighted token-overlap. Label > aliases > keywords > description."""
    q_tokens = set(_normalize(query).split())
    if not q_tokens:
        return 0.0

    label_tokens = set(_normalize(c.get("label", "")).split())
    alias_tokens: set[str] = set()
    for alias in c.get("aliases", []):
        alias_tokens.update(_normalize(alias).split())
    keyword_tokens = set(_normalize(" ".join(c.get("keywords", []))).split())
    desc_tokens = set(_normalize(c.get("description", "")).split())

    score = 0.0
    score += 3.0 * len(q_tokens & label_tokens)
    score += 2.5 * len(q_tokens & alias_tokens)
    score += 1.0 * len(q_tokens & keyword_tokens)
    score += 0.4 * len(q_tokens & desc_tokens)
    return score


@router.post("/match")
def match_connector(req: MatchRequest) -> dict:
    # Stage 1 — exact (id or alias, normalized).
    exact_id = _exact_match(req.query)
    if exact_id:
        return {
            "candidates": [{"id": exact_id, "confidence": 1.0}],
            "needs_clarification": False,
            "stage": "exact",
        }

    # Stage 2 — token-overlap scoring across the registry.
    scored: list[tuple[float, str]] = []
    for c in registry.all_connectors().values():
        s = _token_score(req.query, c)
        if s > 0:
            scored.append((s, c["id"]))
    scored.sort(reverse=True)

    if not scored:
        return {
            "candidates": [],
            "needs_clarification": True,
            "question": "I don't recognize that one — try the search box?",
            "stage": "no-match",
        }

    top_score = scored[0][0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0

    # Top is dominant — single confident pick. (top ≥ 2× runner-up
    # is the rule of thumb; tunable as the registry grows.)
    if runner_up == 0.0 or top_score >= runner_up * 2:
        return {
            "candidates": [{"id": scored[0][1], "confidence": 0.85}],
            "needs_clarification": False,
            "stage": "scored-single",
        }

    # Otherwise return up to N candidates with normalized confidence.
    n = min(req.max_candidates, len(scored))
    max_s = top_score
    return {
        "candidates": [
            {"id": cid, "confidence": round(s / max_s, 3)} for s, cid in scored[:n]
        ],
        "needs_clarification": True,
        "question": "Which one did you mean?",
        "stage": "scored-multi",
    }


# ─── Save endpoint ─────────────────────────────────────────────────


class SaveConnectorRequest(BaseModel):
    """Save a connection using the connector's own JSON-declared
    field schema. Validates required fields against the *picked
    method's* fields (or the spec's top-level `fields` for
    single-method specs), then writes the credentials map directly
    to the local data vault under engine = connector_id.

    This bypasses Anton-core's built-in datasource registry, which
    is what makes OAuth-shaped saves work without the legacy
    Gmail/Slack/etc. engines complaining about unfamiliar fields
    (refresh_token, client_id, etc.).
    """
    method: str | None = None
    name: str = ""
    values: dict[str, Any] = Field(default_factory=dict)


def _resolve_fields(spec: dict, method_id: str | None) -> tuple[list[dict], dict | None]:
    """Pull the field schema for the active method (or top-level
    fields for single-method specs). Returns (fields, method_def)."""
    form = spec.get("form") or {}
    methods = form.get("methods") or []
    if methods:
        if not method_id:
            return [], None
        method_def = next((m for m in methods if m.get("id") == method_id), None)
        if not method_def:
            return [], None
        return list(method_def.get("fields") or []), method_def
    return list(form.get("fields") or []), None


def _missing_required(fields: list[dict], values: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for f in fields:
        if not f.get("required"):
            continue
        v = values.get(f.get("name"))
        if v is None or (isinstance(v, str) and not v.strip()):
            out.append(f.get("name"))
    return out


@router.post("/{connector_id}/save")
def save_connector(connector_id: str, req: SaveConnectorRequest) -> dict:
    spec = registry.get_connector(connector_id)
    if not spec:
        raise HTTPException(status_code=404, detail=f"Unknown connector: {connector_id}")

    form = spec.get("form") or {}
    methods = form.get("methods") or []
    if methods and not req.method:
        raise HTTPException(
            status_code=400,
            detail="`method` is required for connectors that declare multiple auth methods.",
        )

    fields, method_def = _resolve_fields(spec, req.method)
    if methods and method_def is None:
        raise HTTPException(status_code=400, detail=f"Unknown method: {req.method}")

    missing = _missing_required(fields, req.values)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required fields: {', '.join(missing)}",
        )

    # Validation passed — persist to the local data vault. We bypass
    # `anton.utils.datasources.save_connection` (which expects an
    # engine_def from Anton-core's built-in registry) and write
    # straight through `LocalDataVault.save`, treating the connector
    # id as the engine name and the values map as the credentials
    # blob. This is what lets OAuth + service-account + any other
    # auth shape we declare in JSON land in the vault unchanged.
    try:
        from anton.core.datasources.data_vault import LocalDataVault
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Anton data vault is unavailable") from exc

    name = req.name.strip() or uuid.uuid4().hex[:8]

    # Strip empty-string values so the vault doesn't store noise.
    # Skip `None` too. Booleans / non-string values pass through.
    payload: dict[str, Any] = {}
    for k, v in (req.values or {}).items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        payload[k] = v

    # Stamp method + connector metadata so anton's downstream tools
    # can identify which connector + which auth path produced this
    # record, without having to re-read the registry.
    if req.method:
        payload["_method"] = req.method
    payload["_connector_id"] = connector_id

    # Modify-flow merge: resolves `ANTON_VAULT_KEEP` sentinels in the
    # incoming payload against the existing vault record (no-op on
    # create paths where no prior record exists), and computes the
    # secure-key set to persist. Spec-marked secrets come from the
    # connector JSON's `secret: true` field flag. Both the merge
    # helper and the `secure_keys` save kwarg ship in newer anton-
    # core; defensive try/except around each so a stale install
    # falls through to a plain save instead of crashing.
    try:
        from anton.core.datasources.data_vault import resolve_modify_merge as _merge
    except ImportError:
        _merge = None
    spec_secret_names = [
        f.get("name") for f in (fields or [])
        if f.get("secret") and f.get("name")
    ]

    vault = LocalDataVault()
    try:
        if _merge is not None:
            merged_payload, secure_keys = _merge(
                vault, connector_id, name, payload,
                spec_secret_keys=spec_secret_names,
            )
        else:
            merged_payload, secure_keys = payload, None
        # LocalDataVault.save(engine, name, values, secure_keys=…) is
        # the canonical write path. Older vaults don't accept the
        # kwarg — fall back to the positional shape on TypeError so
        # a stale anton install still saves successfully.
        try:
            vault.save(connector_id, name, merged_payload, secure_keys=secure_keys)
        except TypeError:
            vault.save(connector_id, name, merged_payload)
    except AttributeError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "LocalDataVault.save not available — Anton-core may need an update "
                f"that exposes a generic save path. ({exc})"
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save connection: {exc}") from exc

    return {
        "status": "ok",
        "engine": connector_id,
        "name": name,
        "method": req.method,
        "label": spec.get("label") or connector_id,
    }
