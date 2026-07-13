from __future__ import annotations

import sys
from pathlib import Path

import pytest

from anton.core.tools.generate_artifact.verifiers import evaluate_backend, verify_backend

GOOD_SOURCE = '''
import os
from mangum import Mangum
SECRETS = {"DS_POSTGRES_PROD_DB__PASSWORD": os.environ.get("DS_POSTGRES_PROD_DB__PASSWORD")}
@app.get("/api/health")
async def health():
    return {"status": "ok"}
@app.get("/api/items")
async def items():
    pw = SECRETS["DS_POSTGRES_PROD_DB__PASSWORD"]
    return []
handler = Mangum(app, lifespan="off")
'''

GOOD_INTROSPECTION = {
    "import_ok": True, "import_error": "",
    "handler_ok": True, "app_ok": True, "secrets_ok": True,
    "api_routes": ["/api/health", "/api/items"], "root_routes": [],
}
GOOD_REQS = "fastapi\nmangum\nuvicorn\npsycopg2-binary\n"


def test_good_backend_passes():
    r, ds = evaluate_backend(GOOD_INTROSPECTION, GOOD_SOURCE, GOOD_REQS)
    assert r.ok, r.errors
    assert ds == ["DS_POSTGRES_PROD_DB__PASSWORD"]


def test_import_failure_is_error():
    intro = {**GOOD_INTROSPECTION, "import_ok": False, "import_error": "boom"}
    r, _ = evaluate_backend(intro, GOOD_SOURCE, GOOD_REQS)
    assert not r.ok and any("import" in e.lower() for e in r.errors)


def test_missing_health_route_is_error():
    intro = {**GOOD_INTROSPECTION, "api_routes": ["/api/items"]}
    r, _ = evaluate_backend(intro, GOOD_SOURCE, GOOD_REQS)
    assert not r.ok and any("/api/health" in e for e in r.errors)


def test_root_route_is_error():
    intro = {**GOOD_INTROSPECTION, "root_routes": ["/login"]}
    r, _ = evaluate_backend(intro, GOOD_SOURCE, GOOD_REQS)
    assert not r.ok and any("/api/" in e for e in r.errors)


def test_missing_contract_attrs_are_errors():
    intro = {**GOOD_INTROSPECTION, "handler_ok": False, "secrets_ok": False}
    r, _ = evaluate_backend(intro, GOOD_SOURCE, GOOD_REQS)
    assert not r.ok
    assert any("handler" in e for e in r.errors)
    assert any("SECRETS" in e for e in r.errors)


def test_secret_copied_to_module_level_is_error():
    src = GOOD_SOURCE + '\nPW = SECRETS["DS_POSTGRES_PROD_DB__PASSWORD"]\n'
    r, _ = evaluate_backend(GOOD_INTROSPECTION, src, GOOD_REQS)
    assert not r.ok and any("module-level" in e.lower() for e in r.errors)


def test_missing_core_requirements_is_error():
    r, _ = evaluate_backend(GOOD_INTROSPECTION, GOOD_SOURCE, "fastapi\n")
    assert not r.ok and any("mangum" in e.lower() for e in r.errors)


# ── verify_backend (async subprocess glue) ───────────────────────────────────
# The subprocess imports backend.py, which needs fastapi + mangum in the venv
# python (here sys.executable); skip only the integration cases if either is
# absent (a module-level importorskip would skip the pure tests above too).
import importlib.util

_HAS_ASGI = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("mangum") is not None
)
_needs_asgi = pytest.mark.skipif(not _HAS_ASGI, reason="fastapi/mangum not installed in the venv")

BACKEND_OK = '''
import os, argparse
from fastapi import FastAPI
from mangum import Mangum
app = FastAPI()
SECRETS = {}
@app.get("/api/health")
async def health():
    return {"status": "ok"}
@app.get("/api/items")
async def items():
    return []
handler = Mangum(app, lifespan="off")
'''


class _FakePool:
    """Runs installs as no-ops and returns the current interpreter as the venv."""
    async def get_or_create(self, name):
        return self
    async def install_packages(self, packages):
        return "ok"
    async def venv_python(self, name):
        return sys.executable


@_needs_asgi
async def test_verify_backend_happy_path(tmp_path: Path):
    (tmp_path / "backend.py").write_text(BACKEND_OK)
    (tmp_path / "requirements.txt").write_text("fastapi\nmangum\nuvicorn\n")
    result, ds = await verify_backend(
        scratchpad_pool=_FakePool(), slug="x", artifact_path=tmp_path,
    )
    assert result.ok, result.errors
    assert ds == []


@_needs_asgi
async def test_verify_backend_reports_import_error(tmp_path: Path):
    (tmp_path / "backend.py").write_text("import definitely_not_a_real_module\n")
    (tmp_path / "requirements.txt").write_text("fastapi\nmangum\nuvicorn\n")
    result, _ = await verify_backend(
        scratchpad_pool=_FakePool(), slug="x", artifact_path=tmp_path,
    )
    assert not result.ok
    assert any("import" in e.lower() for e in result.errors)
