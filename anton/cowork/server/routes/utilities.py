"""Anton utility routes for memory, skills, data connections, and publishing."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .artifacts import _resolve_artifact_path, _scan_artifact_dirs
from .cowork_state import backups_dir, load_state, save_state, utc_now_iso
from .integrations import ensure_managed_integrations
from .settings import _get_env, get_config_status

router = APIRouter()
logger = logging.getLogger(__name__)


def _safe_text(path: Path, limit: int = 80_000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[:limit]


def _memory_roots(project_path: Optional[str] = None) -> list[tuple[str, Optional[str], Path]]:
    """Enumerate (scope, project_name, root) tuples for the memory listing.

    `scope` is "Global" or "Project". Global has no project_name. When
    `project_path` is set we list only that one project (legacy
    single-project shape). When `project_path` is None we list every
    project on disk so the UI can present a unified view of all memory
    grouped by project.
    """
    roots: list[tuple[str, Optional[str], Path]] = [
        ("Global", None, Path.home() / ".anton" / "memory"),
    ]
    if project_path:
        target = Path(project_path).expanduser().resolve()
        roots.append(("Project", target.name, target / ".anton" / "memory"))
        return roots
    try:
        from ..anton_api import projects_store
        for project in projects_store.list_projects():
            target = Path(project["path"]).expanduser().resolve()
            roots.append(("Project", project["name"], target / ".anton" / "memory"))
    except Exception:
        logger.warning("Could not enumerate projects for memory listing", exc_info=True)
    return roots


def _memory_root(scope: str, project_path: Optional[str] = None) -> Path:
    normalized = (scope or "Global").strip().lower()
    if normalized == "global":
        return Path.home() / ".anton" / "memory"
    if normalized == "project":
        if not project_path:
            raise HTTPException(status_code=400, detail="A project path is required for project memory.")
        return Path(project_path).expanduser().resolve() / ".anton" / "memory"
    raise HTTPException(status_code=400, detail="Memory scope must be Global or Project.")


def _normalise_relative_md(relative_path: str) -> Path:
    text = (relative_path or "").strip().replace("\\", "/")
    if not text:
        raise HTTPException(status_code=400, detail="Memory file path is required.")
    if text.startswith("/") or ".." in Path(text).parts:
        raise HTTPException(status_code=400, detail="Memory file path must stay inside the memory folder.")
    path = Path(text)
    if path.suffix and path.suffix.lower() != ".md":
        raise HTTPException(status_code=415, detail="Memory files must be Markdown files.")
    if not path.suffix:
        path = path.with_suffix(".md")
    return path


def _resolve_memory_path(scope: str, relative_path: str, project_path: Optional[str] = None) -> tuple[Path, Path]:
    root = _memory_root(scope, project_path)
    rel = _normalise_relative_md(relative_path)
    target = (root / rel).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Memory file must stay inside the memory folder.") from exc
    return root, target


def _backup_target(path: Path, namespace: str) -> None:
    if not path.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_root = backups_dir() / namespace
    backup_root.mkdir(parents=True, exist_ok=True)
    name = f"{path.name}.{stamp}.bak"
    try:
        shutil.copy2(path, backup_root / name)
    except OSError:
        logger.warning("Could not backup %s before mutation", path)


def _memory_file_payload(
    path: Path,
    root: Path,
    scope: str,
    project_name: Optional[str] = None,
) -> dict[str, Any]:
    text = _safe_text(path, 16_000)
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    payload: dict[str, Any] = {
        "name": path.stem.replace("-", " ").replace("_", " ").title(),
        "path": str(resolved_path),
        "relativePath": str(resolved_path.relative_to(resolved_root)),
        "scope": scope,
        "bytes": path.stat().st_size,
        "preview": "\n".join([line for line in text.splitlines() if line.strip()][:8])[:700],
        "content": text,
    }
    if project_name:
        # `projectPath` is the project root (two levels up from
        # `<project>/.anton/memory/`) — the UI passes this back to
        # save/delete so writes hit the right project regardless of
        # which project is currently "active" in the sidebar.
        payload["projectName"] = project_name
        payload["projectPath"] = str(resolved_root.parent.parent)
    return payload


@router.get("/memory")
async def list_memory(project_path: Optional[str] = None):
    sections = []
    for scope, project_name, root in _memory_roots(project_path):
        files = []
        if root.is_dir():
            for path in sorted(root.rglob("*.md")):
                if not path.is_file():
                    continue
                files.append(_memory_file_payload(path, root, scope, project_name))
        section: dict[str, Any] = {
            "scope": scope,
            "root": str(root),
            "files": files,
        }
        if project_name:
            section["projectName"] = project_name
            section["projectPath"] = str(root.parent.parent)
        sections.append(section)
    return {"sections": sections}


class MemorySaveRequest(BaseModel):
    scope: str = "Global"
    relativePath: str
    content: str
    projectPath: Optional[str] = None


@router.post("/memory")
async def save_memory(req: MemorySaveRequest):
    root, target = _resolve_memory_path(req.scope, req.relativePath, req.projectPath)
    target.parent.mkdir(parents=True, exist_ok=True)
    _backup_target(target, "memory")
    target.write_text(req.content, encoding="utf-8")
    project_name: Optional[str] = None
    if req.scope.strip().lower() == "project" and req.projectPath:
        project_name = Path(req.projectPath).expanduser().resolve().name
    return {"status": "ok", "file": _memory_file_payload(target, root, req.scope.title(), project_name)}


@router.delete("/memory")
async def delete_memory(scope: str, relative_path: str, project_path: Optional[str] = None):
    root, target = _resolve_memory_path(scope, relative_path, project_path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Memory file not found.")
    _backup_target(target, "memory")
    target.unlink()
    return {"status": "ok", "deleted": str(target), "root": str(root)}


@router.get("/skills")
async def list_skills():
    try:
        from anton.core.memory.skills import SkillStore
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Anton skill store is unavailable") from exc

    store = SkillStore()
    skills = []
    for skill in store.list_all():
        skills.append({
            "label": skill.label,
            "name": skill.name,
            "description": skill.description,
            "whenToUse": skill.when_to_use,
            "declarative": skill.declarative_md,
            "createdAt": skill.created_at,
            "provenance": skill.provenance,
            "stages": {
                "stage1": skill.stage_1_present,
                "stage2": skill.stage_2_present,
                "stage3": skill.stage_3_present,
            },
            "stats": {
                "totalRecalls": getattr(skill.stats, "total_recalls", 0),
            },
        })
    return {"skills": skills}


def _skill_label(value: str) -> str:
    label = re.sub(r"[^a-z0-9_]+", "_", (value or "").strip().lower()).strip("_")
    if not label:
        raise HTTPException(status_code=400, detail="Skill label is required.")
    return label[:80]


class SkillSaveRequest(BaseModel):
    label: str
    name: str
    description: str = ""
    whenToUse: str = ""
    declarative: str


@router.post("/skills")
async def save_skill(req: SkillSaveRequest):
    try:
        from anton.core.memory.skills import Skill, SkillStore
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Anton skill store is unavailable") from exc

    label = _skill_label(req.label)
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Skill name is required.")
    if not req.declarative.strip():
        raise HTTPException(status_code=400, detail="Skill instructions are required.")

    store = SkillStore()
    existing = store.load(label)
    skill = Skill(
        label=label,
        name=req.name.strip(),
        description=req.description.strip(),
        when_to_use=req.whenToUse.strip(),
        declarative_md=req.declarative.strip() + "\n",
        created_at=getattr(existing, "created_at", None) or utc_now_iso(),
        provenance=getattr(existing, "provenance", None) or "manual",
        stage_1_present=True,
        stage_2_present=getattr(existing, "stage_2_present", False),
        stage_3_present=getattr(existing, "stage_3_present", False),
        stats=getattr(existing, "stats", None) if existing else None,
    )
    if skill.stats is None:
        from anton.core.memory.skills import SkillStats
        skill.stats = SkillStats()
    path = store.save(skill)
    return {"status": "ok", "label": label, "path": str(path)}


@router.delete("/skills/{label}")
async def delete_skill(label: str):
    try:
        from anton.core.memory.skills import SkillStore
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Anton skill store is unavailable") from exc

    deleted = SkillStore().delete(label)
    if not deleted:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"status": "ok", "deleted": label}


class DatasourceSaveRequest(BaseModel):
    engine: str
    name: str = ""
    authMethod: str | None = None
    credentials: dict[str, str]


class DatasourceValidateRequest(BaseModel):
    engine: str
    name: str = ""
    authMethod: str | None = None
    credentials: dict[str, str] = Field(default_factory=dict)


def _datasource_fields_payload(fields) -> list[dict[str, Any]]:
    return [
            {
                "name": field.name,
                "required": field.required,
                "secret": field.secret,
                "description": field.description,
                "default": field.default,
            }
            for field in fields
    ]


def _datasource_engine_payload(engine_def) -> dict[str, Any]:
    return {
        "engine": engine_def.engine,
        "displayName": engine_def.display_name,
        "popular": engine_def.popular,
        "custom": engine_def.custom,
        "testAvailable": bool(engine_def.test_snippet),
        "nameFrom": engine_def.name_from,
        "authMethodMode": engine_def.auth_method,
        "fields": _datasource_fields_payload(engine_def.fields),
        "authMethods": [
            {
                "name": method.name,
                "display": method.display,
                "fields": _datasource_fields_payload(method.fields),
            }
            for method in engine_def.auth_methods
        ],
    }


def _datasource_registry():
    ensure_managed_integrations()
    try:
        from anton.core.datasources.datasource_registry import DatasourceRegistry
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Anton datasource registry is unavailable") from exc
    return DatasourceRegistry()


def _active_datasource_fields(engine_def, auth_method: str | None = None) -> tuple[str | None, str, list[Any]]:
    if engine_def.auth_method == "choice" and engine_def.auth_methods:
        chosen = next((method for method in engine_def.auth_methods if method.name == auth_method), None)
        if auth_method and not chosen:
            raise HTTPException(status_code=400, detail=f"Unknown auth method for {engine_def.engine}: {auth_method}")
        chosen = chosen or engine_def.auth_methods[0]
        return chosen.name, chosen.display, chosen.fields
    return None, "", engine_def.fields


def _clean_credentials(credentials: dict[str, str]) -> dict[str, str]:
    return {
        str(key).strip(): "" if value is None else str(value)
        for key, value in credentials.items()
        if str(key).strip()
    }


def _resolve_modify_merge(
    engine: str,
    name: str,
    incoming: dict[str, str],
    spec_fields: list[Any],
) -> tuple[dict[str, str], list[str]]:
    """Server-side wrapper around anton-core's `resolve_modify_merge`.

    Pulls the spec-marked secret-field names out of the engine spec
    objects (which use a `field.secret: bool` attribute), then hands
    everything off to anton-core so the merge logic stays in one
    place. See `anton.core.datasources.data_vault.resolve_modify_merge`
    for the full contract.

    Falls back to a passthrough (no merge, no secure-key tracking)
    when the installed anton-core predates `resolve_modify_merge`.
    Logs a warning so the user knows to reinstall — the modify
    flow's sentinel preservation only kicks in once anton is fresh.
    """
    try:
        from anton.core.datasources.data_vault import LocalDataVault
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Anton data vault is unavailable") from exc
    try:
        from anton.core.datasources.data_vault import resolve_modify_merge
    except ImportError:
        logger.warning(
            "anton-core has no `resolve_modify_merge` — saving without "
            "the modify-flow merge. Reinstall anton to enable sentinel-"
            "based credential preservation."
        )
        return dict(incoming), []

    spec_secret_names = [
        getattr(f, "name", "") for f in spec_fields
        if getattr(f, "secret", False) and getattr(f, "name", "")
    ]
    return resolve_modify_merge(
        LocalDataVault(),
        engine,
        name,
        incoming,
        spec_secret_keys=spec_secret_names,
    )


def _datasource_record_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Build the modify-flow read response.

    Substitutes `ANTON_VAULT_KEEP` into every secret-shaped slot so
    the renderer can pre-fill the form without ever seeing the
    underlying credential. Identity + timestamps + the secure-keys
    list pass through verbatim.
    """
    try:
        from anton.core.datasources.data_vault import ANTON_VAULT_KEEP, is_secret_key
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Anton data vault is unavailable") from exc

    raw_fields = record.get("fields") or {}
    secure_keys = record.get("secure_keys")  # may be None on legacy records
    fields_out: dict[str, str] = {}
    for key, value in raw_fields.items():
        if is_secret_key(key, secure_keys=secure_keys):
            fields_out[key] = ANTON_VAULT_KEEP
        else:
            fields_out[key] = value
    return {
        "engine": record.get("engine", ""),
        "name": record.get("name", ""),
        "createdAt": record.get("created_at"),
        "updatedAt": record.get("updated_at"),
        # Echo the secure-key set the renderer should treat as "saved · keep".
        # On legacy records (no stored list), recompute from the heuristic
        # so the renderer's UX is identical regardless of schema age.
        "secureKeys": (
            secure_keys
            if secure_keys is not None
            else sorted(k for k in raw_fields.keys() if is_secret_key(k))
        ),
        "fields": fields_out,
    }


def _validate_datasource_payload(
    engine: str,
    credentials: dict[str, str],
    auth_method: str | None = None,
) -> tuple[Any, str | None, str, list[Any], list[str]]:
    registry = _datasource_registry()
    engine_def = registry.get(engine.strip())
    if not engine_def:
        raise HTTPException(status_code=400, detail=f"Unknown datasource engine: {engine}")
    selected_auth, selected_auth_display, fields = _active_datasource_fields(engine_def, auth_method)
    missing = [
        field.name for field in fields
        if field.required and not str(credentials.get(field.name, field.default or "")).strip()
    ]
    return engine_def, selected_auth, selected_auth_display, fields, missing


@router.get("/datasources")
async def list_datasources():
    try:
        from anton.core.datasources.data_vault import LocalDataVault
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Anton data vault is unavailable") from exc

    vault = LocalDataVault()
    registry = _datasource_registry()
    connections = []
    for item in vault.list_connections():
        safe_item = {k: v for k, v in item.items() if k in {"engine", "name", "path"}}
        engine_def = registry.get(safe_item.get("engine", ""))
        if engine_def:
            safe_item["displayName"] = engine_def.display_name
            safe_item["testAvailable"] = bool(engine_def.test_snippet)
        connections.append(safe_item)

    status = get_config_status()
    return {
        "connections": connections,
        "engines": [_datasource_engine_payload(engine) for engine in registry.all_engines()],
        "configReady": status["config_ready"],
        "configError": status["config_error"],
    }


@router.get("/datasources/{engine}/{name}")
async def read_datasource(engine: str, name: str):
    """Modify-flow read: return the saved connection's non-secret
    fields verbatim, with the `ANTON_VAULT_KEEP` sentinel substituted
    into every secret slot.

    The renderer hydrates its form from this payload — non-secrets
    pre-fill the inputs, secret-slot sentinels render as empty inputs
    with a "Saved · type to replace, clear to remove" placeholder.
    On submit, any field still carrying the sentinel is resolved
    server-side against the prior record (see `_resolve_modify_merge`).
    """
    if not engine.strip() or not name.strip():
        raise HTTPException(status_code=400, detail="engine and name are required")
    try:
        from anton.core.datasources.data_vault import LocalDataVault
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Anton data vault is unavailable") from exc
    vault = LocalDataVault()
    # Older anton installs don't have `read_record` — fall back to
    # synthesizing a minimal record from `load()`. The renderer's
    # modify form still works (heuristic fills in for the missing
    # secure-keys list); only created_at / updated_at are missing.
    if hasattr(vault, "read_record"):
        record = vault.read_record(engine.strip(), name.strip())
    else:
        fields = vault.load(engine.strip(), name.strip())
        record = (
            None if fields is None
            else {"engine": engine.strip(), "name": name.strip(), "fields": fields}
        )
    if record is None:
        raise HTTPException(status_code=404, detail="Datasource connection not found")
    return _datasource_record_payload(record)


@router.post("/datasources/validate")
async def validate_datasource(req: DatasourceValidateRequest):
    credentials = req.credentials
    if not credentials and req.name.strip():
        try:
            from anton.core.datasources.data_vault import LocalDataVault
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Anton data vault is unavailable") from exc
        credentials = LocalDataVault().load(req.engine.strip(), req.name.strip()) or {}
    engine_def, auth_method, auth_display, fields, missing = _validate_datasource_payload(req.engine, credentials, req.authMethod)
    return {
        "status": "ok" if not missing else "missing_fields",
        "engine": req.engine.strip(),
        "displayName": engine_def.display_name,
        "authMethod": auth_method,
        "authDisplay": auth_display,
        "fields": _datasource_fields_payload(fields),
        "missingFields": missing,
        "testAvailable": bool(engine_def.test_snippet),
        "liveTest": "not_run",
        "message": "Required fields are present." if not missing else "Required credential fields are missing.",
    }


@router.post("/datasources")
async def save_datasource(req: DatasourceSaveRequest):
    if not req.engine.strip():
        raise HTTPException(status_code=400, detail="Datasource engine is required")
    raw_credentials = _clean_credentials(req.credentials)

    # Modify-flow merge: resolve any `ANTON_VAULT_KEEP` sentinels in
    # the incoming credentials against the existing vault record, and
    # compute the secure-key set to persist. Pure no-op for create
    # paths — there's no prior record so no sentinels survive, and
    # the secure-key set is computed from spec + heuristic alone.
    engine_def_pre, _, _, spec_fields_pre, _ = _validate_datasource_payload(
        req.engine, raw_credentials, req.authMethod
    )
    candidate_name = req.name.strip()
    credentials, merged_secure_keys = _resolve_modify_merge(
        engine=req.engine.strip(),
        name=candidate_name,
        incoming=raw_credentials,
        spec_fields=spec_fields_pre,
    )

    engine_def, _, _, fields, missing = _validate_datasource_payload(req.engine, credentials, req.authMethod)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing)}")
    known_field_names = {field.name for field in fields}
    has_user_value = any(str(credentials.get(field.name, "")).strip() for field in fields)
    has_extra_value = any(
        key not in known_field_names and str(value).strip()
        for key, value in credentials.items()
    )
    if fields and not has_user_value and not has_extra_value:
        raise HTTPException(status_code=400, detail="At least one credential value is required")

    try:
        from anton.core.datasources.data_vault import LocalDataVault
        from anton.utils.datasources import find_matching_connection, save_connection
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Anton data vault is unavailable") from exc

    try:
        vault = LocalDataVault()
        name = candidate_name or find_matching_connection(vault, engine_def, credentials) or uuid.uuid4().hex[:8]
        # If find_matching_connection picked a different name than the
        # one we merged against, the sentinel resolution we did above
        # used the wrong prior record. Re-resolve against the picked
        # name so secrets actually carry through. Cheap and rare.
        if name != candidate_name:
            credentials, merged_secure_keys = _resolve_modify_merge(
                engine=engine_def.engine,
                name=name,
                incoming=raw_credentials,
                spec_fields=spec_fields_pre,
            )
        try:
            slug = save_connection(
                vault, engine_def, name, credentials,
                secure_keys=merged_secure_keys,
            )
        except TypeError:
            # Older anton.utils.datasources.save_connection signature
            # has no `secure_keys` kwarg. Save still lands; the
            # secure-key set just isn't persisted on the record. The
            # next read falls back to the heuristic, which is good
            # enough until the user reinstalls anton.
            logger.warning(
                "anton-core save_connection has no `secure_keys` kwarg — "
                "persisting without it. Reinstall anton to enable "
                "secure-key tracking."
            )
            slug = save_connection(vault, engine_def, name, credentials)
    except Exception as exc:
        logger.exception("Datasource save failed")
        raise HTTPException(status_code=500, detail="Could not save datasource") from exc
    return {"status": "ok", "slug": slug, "engine": req.engine.strip(), "name": name}


@router.delete("/datasources/{engine}/{name}")
async def delete_datasource(engine: str, name: str):
    try:
        from anton.core.datasources.data_vault import LocalDataVault
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Anton data vault is unavailable") from exc

    try:
        from .integrations import revoke_google_token
        await asyncio.to_thread(revoke_google_token, engine, name)
    except Exception:
        pass

    deleted = LocalDataVault().delete(engine, name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Datasource connection not found")
    return {"status": "ok", "deleted": {"engine": engine, "name": name}}


def _html_artifacts() -> list[dict[str, Any]]:
    """List every HTML file under every project's `artifacts/` tree.

    Recursive — multi-file artifacts (HTML + assets) get their main
    file plus any sibling HTML pages all surfaced as publishable
    candidates. Sorted by mtime desc, capped at 40 items.

    Each entry surfaces the existing `.published.json` URL when the
    file has been published before, so the UI can show "Re-publish"
    instead of "Publish" on the next click.
    """
    out = []
    seen = set()
    for art_root in _scan_artifact_dirs():
        if not art_root.exists():
            continue
        for path in sorted(art_root.rglob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            published_path = path.parent / ".published.json"
            published = {}
            if published_path.is_file():
                try:
                    published = json.loads(published_path.read_text(encoding="utf-8")).get(path.name, {})
                except Exception:
                    published = {}
            out.append({
                "title": path.stem.replace("_", " ").replace("-", " ").title(),
                "path": str(path),
                "bytes": path.stat().st_size,
                "publishedUrl": published.get("url", "") if isinstance(published, dict) else "",
            })
    return out[:40]


@router.get("/publish")
async def list_publishable():
    state = load_state()
    return {
        "artifacts": _html_artifacts(),
        "publishReady": bool(_get_env("ANTON_MINDS_API_KEY")),
        "publishUrl": _get_env("ANTON_PUBLISH_URL", "https://4nton.ai"),
        "history": state.get("publish_history", [])[:40],
    }


class PublishRequest(BaseModel):
    path: str


@router.post("/publish")
async def publish_artifact(req: PublishRequest):
    api_key = _get_env("ANTON_MINDS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Configure ANTON_MINDS_API_KEY before publishing")

    artifact = _resolve_artifact_path(req.path)
    if artifact.suffix.lower() != ".html":
        raise HTTPException(status_code=415, detail="Only HTML artifacts can be published")

    try:
        from anton.publisher import publish
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Anton publisher is unavailable") from exc

    published_json = artifact.parent / ".published.json"
    published_map: dict[str, Any] = {}
    if published_json.is_file():
        try:
            published_map = json.loads(published_json.read_text(encoding="utf-8"))
        except Exception:
            published_map = {}
    previous = published_map.get(artifact.name)
    report_id = previous.get("report_id") if isinstance(previous, dict) else None

    try:
        result = publish(
            artifact,
            api_key=api_key,
            report_id=report_id,
            publish_url=_get_env("ANTON_PUBLISH_URL", "https://4nton.ai"),
            ssl_verify=_get_env("ANTON_MINDS_SSL_VERIFY", "true").lower() == "true",
        )
    except Exception as exc:
        logger.exception("Publishing failed")
        raise HTTPException(status_code=502, detail="Publishing failed. Check your Minds credentials and try again.") from exc

    view_url = result.get("view_url", "")
    returned_report_id = result.get("report_id", "")
    if returned_report_id:
        history_item = {
            "artifact": str(artifact),
            "artifactName": artifact.name,
            "url": view_url,
            "reportId": returned_report_id,
            "publishedAt": utc_now_iso(),
        }
        published_map[artifact.name] = {
            "report_id": returned_report_id,
            "url": view_url,
            "last_md5": result.get("md5", ""),
        }
        try:
            published_json.write_text(json.dumps(published_map, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        state = load_state()
        state["publish_history"] = [history_item, *state.get("publish_history", [])][:100]
        save_state(state)

    return {"status": "ok", "url": view_url, "result": {k: v for k, v in result.items() if k != "file_payload"}}


@router.delete("/publish")
async def unpublish_artifact(path: str = Query(..., description="Absolute path to the published HTML artifact")):
    """Tear down the published copy of an HTML artifact.

    Looks up the artifact's last-published md5 from `.published.json`,
    calls anton.publisher.unpublish, and removes the entry from the
    map so the UI no longer shows a "Published" pill.
    """
    api_key = _get_env("ANTON_MINDS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Configure ANTON_MINDS_API_KEY before unpublishing")

    artifact = _resolve_artifact_path(path)
    published_json = artifact.parent / ".published.json"
    if not published_json.is_file():
        raise HTTPException(status_code=404, detail="Artifact has no publish record")

    try:
        published_map: dict[str, Any] = json.loads(published_json.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail="Could not read publish record")

    entry = published_map.get(artifact.name)
    # Match anton's CLI: prefer the report_id, fall back to last_md5.
    # mdb.ai's `/delete/{id}` endpoint accepts the report_id directly,
    # which is what the publish response gives us; the md5 is the
    # version hash and isn't always recognized as a delete target.
    identifier = None
    if isinstance(entry, dict):
        identifier = entry.get("report_id") or entry.get("last_md5") or None
    if not identifier:
        raise HTTPException(status_code=404, detail="No published version on file")

    try:
        from anton.publisher import unpublish
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Anton publisher is unavailable") from exc

    try:
        unpublish(
            identifier,
            api_key=api_key,
            publish_url=_get_env("ANTON_PUBLISH_URL", "https://4nton.ai"),
            ssl_verify=_get_env("ANTON_MINDS_SSL_VERIFY", "true").lower() == "true",
        )
    except Exception as exc:
        logger.exception("Unpublishing failed (identifier=%s)", identifier)
        # Surface the underlying error to the client so the toast is
        # informative — it'll usually be a 401 (bad key) or a 404 (the
        # report was already deleted upstream). We map "not found"
        # specifically so the UI can treat it as success and clear the
        # local record.
        msg = str(exc) or "Unpublishing failed."
        if "404" in msg or "not found" in msg.lower():
            # Already gone upstream — clear the local record below.
            pass
        else:
            raise HTTPException(status_code=502, detail=f"Unpublishing failed: {msg}") from exc

    # Strip from the per-folder map so the artifact is no longer
    # reported as published.
    published_map.pop(artifact.name, None)
    try:
        if published_map:
            published_json.write_text(json.dumps(published_map, indent=2) + "\n", encoding="utf-8")
        else:
            published_json.unlink()
    except Exception:
        pass
    return {"status": "ok"}
