from __future__ import annotations

import json

import sys
from pathlib import Path

import pytest

from anton.core.tools.generate_artifact.verifiers import evaluate_backend, verify_backend, contract_operations, contract_paths

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
    # Self-evidencing message: shows what the parser actually saw.
    assert any("Parsed package names: fastapi" in e for e in r.errors)


def test_core_requirements_accept_extras_and_specifiers():
    # uvicorn[standard], version ranges, spaces, case — all valid for pip and
    # must all satisfy the core-requirement check (the trace.json regression:
    # `uvicorn[standard]` was reported as missing `uvicorn`).
    reqs = "FastAPI >= 0.100\nmangum~=0.17\nuvicorn[standard]==0.29.0  # ASGI server\n"
    r, _ = evaluate_backend(GOOD_INTROSPECTION, GOOD_SOURCE, reqs)
    assert r.ok, r.errors


# ── STATE contract (fullstack-stateful-app / fullstack-stateless-app) ────────

from anton.core.artifacts.backend_launcher import parse_requirements

STATEFUL = "fullstack-stateful-app"
STATELESS = "fullstack-stateless-app"

STATEFUL_SOURCE = '''
import os
from pathlib import Path
from mangum import Mangum
from anton_state import open_store, Collection
SECRETS = {}
STATE = None
_STATE_DIR = Path(__file__).resolve().parent

def get_store():
    return open_store(
        state=STATE,
        manifest_path=str(_STATE_DIR / "state_manifest.json"),
        local_path=str(_STATE_DIR / ".anton_state.db"),
    )

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/api/todos")
async def todos():
    items = await Collection(get_store(), "todos").list()
    return items

handler = Mangum(app, lifespan="off")
'''

STATEFUL_INTROSPECTION = {**GOOD_INTROSPECTION, "state_defined": True,
                          "api_routes": ["/api/health", "/api/todos"]}

GOOD_MANIFEST = (
    '{"version": 1, "pk": {"name": "pk", "type": "S"},'
    ' "sk": {"name": "sk", "type": "S"}, "collections": ["todos"]}'
)


def test_stateful_backend_passes_with_manifest():
    r, _ = evaluate_backend(
        STATEFUL_INTROSPECTION, STATEFUL_SOURCE, GOOD_REQS,
        artifact_type=STATEFUL, state_manifest=GOOD_MANIFEST,
    )
    assert r.ok, r.errors


def test_stateful_missing_state_slot_is_error():
    intro = {**STATEFUL_INTROSPECTION, "state_defined": False}
    r, _ = evaluate_backend(
        intro, STATEFUL_SOURCE, GOOD_REQS,
        artifact_type=STATEFUL, state_manifest=GOOD_MANIFEST,
    )
    assert not r.ok and any("STATE = None" in e for e in r.errors)


def test_stateful_missing_manifest_is_error():
    r, _ = evaluate_backend(
        STATEFUL_INTROSPECTION, STATEFUL_SOURCE, GOOD_REQS,
        artifact_type=STATEFUL, state_manifest=None,
    )
    assert not r.ok and any("state_manifest.json is missing" in e for e in r.errors)


@pytest.mark.parametrize("bad_manifest", [
    "not json at all",
    # DynamoDB-CreateTable shape: pk is required and missing at the top level.
    '{"entities": {"todo": {"attributes": ["text"]}}}',
    # GSIs are rejected in v1.
    ('{"version": 1, "pk": {"name": "pk"}, "sk": {"name": "sk"},'
     ' "gsis": [{"name": "g", "pk": {"name": "x"}}]}'),
    # collections require a sort key.
    '{"version": 1, "pk": {"name": "pk"}, "collections": ["todos"]}',
])
def test_stateful_invalid_manifest_is_error(bad_manifest):
    r, _ = evaluate_backend(
        STATEFUL_INTROSPECTION, STATEFUL_SOURCE, GOOD_REQS,
        artifact_type=STATEFUL, state_manifest=bad_manifest,
    )
    assert not r.ok
    assert any(e.startswith("state_manifest.json is invalid: ") for e in r.errors)


@pytest.mark.parametrize("line", [
    "store = open_store(state=STATE)",
    "todos = Collection(get_store(), 'todos')",
])
def test_stateful_store_built_at_import_is_error(line):
    src = STATEFUL_SOURCE + "\n" + line + "\n"
    r, _ = evaluate_backend(
        STATEFUL_INTROSPECTION, src, GOOD_REQS,
        artifact_type=STATEFUL, state_manifest=GOOD_MANIFEST,
    )
    assert not r.ok and any("import time" in e for e in r.errors)


def test_store_built_inside_route_is_not_flagged():
    # STATEFUL_SOURCE itself calls get_store() inside the /api/todos route.
    r, _ = evaluate_backend(
        STATEFUL_INTROSPECTION, STATEFUL_SOURCE, GOOD_REQS,
        artifact_type=STATEFUL, state_manifest=GOOD_MANIFEST,
    )
    assert not any("import time" in e for e in r.errors)


def test_anton_state_in_requirements_is_error_for_any_type():
    reqs = GOOD_REQS + "anton_state\n"
    for artifact_type in (STATEFUL, STATELESS, ""):
        kwargs = {"artifact_type": artifact_type} if artifact_type else {}
        if artifact_type == STATEFUL:
            kwargs["state_manifest"] = GOOD_MANIFEST
        src = STATEFUL_SOURCE if artifact_type == STATEFUL else GOOD_SOURCE
        intro = STATEFUL_INTROSPECTION if artifact_type == STATEFUL else GOOD_INTROSPECTION
        r, _ = evaluate_backend(intro, src, reqs, **kwargs)
        assert any("must not list `anton_state`" in e for e in r.errors), artifact_type


def test_stateless_importing_anton_state_is_error():
    src = GOOD_SOURCE + "\nfrom anton_state import open_store\n"
    r, _ = evaluate_backend(
        GOOD_INTROSPECTION, src, GOOD_REQS, artifact_type=STATELESS,
    )
    assert not r.ok and any("stateless backend" in e for e in r.errors)


def test_stateless_without_anton_state_passes():
    r, _ = evaluate_backend(
        GOOD_INTROSPECTION, GOOD_SOURCE, GOOD_REQS, artifact_type=STATELESS,
    )
    assert r.ok, r.errors


def test_no_artifact_type_skips_state_checks():
    """The pre-stateful call shape: no manifest, no STATE slot — still ok."""
    r, _ = evaluate_backend(GOOD_INTROSPECTION, GOOD_SOURCE, GOOD_REQS)
    assert r.ok, r.errors


def test_parse_requirements_drops_anton_state_before_install():
    text = "fastapi\nanton_state\nanton-state==0.1\nAnton_State\nmangum\n"
    assert parse_requirements(text) == ["fastapi", "mangum"]


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



# ── the API contract from the backend's side (I-34) ─────────────────────────

CONTRACT_INTROSPECTION = {
    **GOOD_INTROSPECTION,
    "api_routes": ["/api/health", "/api/rooms", "/api/rooms/{room_id}"],
    "api_operations": [
        "GET /api/health", "GET /api/rooms", "POST /api/rooms", "GET /api/rooms/{room_id}",
    ],
}


def test_contract_operations_are_read_from_openapi():
    spec = json.dumps({"paths": {
        "/api/health": {"get": {}},
        "/api/rooms": {"get": {"responses": {}}, "post": {"responses": {}}},
        "/api/rooms/{code}/": {"get": {"responses": {}}, "parameters": []},
    }})
    ops = contract_operations(spec)
    assert ops == {"GET /api/rooms", "POST /api/rooms", "GET /api/rooms/{}"}
    assert contract_paths(ops) == {"/api/rooms", "/api/rooms/{}"}


def test_no_usable_contract_means_no_comparison():
    for spec in (None, "", "not json", "{}", json.dumps({"paths": {}}),
                 json.dumps({"paths": {"/api/health": {"get": {}}}})):
        assert contract_operations(spec) is None, spec
    assert contract_paths(None) is None
    r, _ = evaluate_backend(GOOD_INTROSPECTION, GOOD_SOURCE, GOOD_REQS, api_operations=None)
    assert r.ok, r.errors


def test_backend_implementing_the_contract_passes_whatever_the_parameter_names():
    ops = {"GET /api/rooms", "POST /api/rooms", "GET /api/rooms/{}"}
    r, _ = evaluate_backend(CONTRACT_INTROSPECTION, GOOD_SOURCE, GOOD_REQS, api_operations=ops)
    assert r.ok, r.errors
    assert r.warnings == []


def test_missing_contract_operation_is_error():
    ops = {"GET /api/rooms", "POST /api/rooms", "GET /api/rooms/{}", "DELETE /api/rooms/{}"}
    r, _ = evaluate_backend(CONTRACT_INTROSPECTION, GOOD_SOURCE, GOOD_REQS, api_operations=ops)
    assert not r.ok
    assert any("does not implement `DELETE /api/rooms/{}`" in e for e in r.errors)


def test_route_outside_the_contract_is_only_a_warning():
    ops = {"GET /api/rooms", "GET /api/rooms/{}"}
    r, _ = evaluate_backend(CONTRACT_INTROSPECTION, GOOD_SOURCE, GOOD_REQS, api_operations=ops)
    assert r.ok, r.errors
    assert any("adds routes that openapi.json does not define: `POST /api/rooms`" in w
               for w in r.warnings)


def test_an_old_introspection_without_operations_reports_everything_missing():
    """A contract with no `api_operations` in the introspection is not
    silently passed: the missing key means the routes are unknown."""
    r, _ = evaluate_backend(GOOD_INTROSPECTION, GOOD_SOURCE, GOOD_REQS,
                            api_operations={"GET /api/items"})
    assert not r.ok
    assert any("does not implement `GET /api/items`" in e for e in r.errors)


@_needs_asgi
async def test_the_introspection_reports_the_routes_operations(tmp_path: Path):
    """The real subprocess script emits `METHOD /path` per API route, so the
    contract check sees the backend as FastAPI registered it."""
    (tmp_path / "backend.py").write_text(BACKEND_OK)
    (tmp_path / "requirements.txt").write_text("fastapi\nmangum\nuvicorn\n")
    ok, _ = await verify_backend(
        scratchpad_pool=_FakePool(), slug="x", artifact_path=tmp_path,
        api_operations={"GET /api/items"},
    )
    assert ok.ok, ok.errors
    bad, _ = await verify_backend(
        scratchpad_pool=_FakePool(), slug="x", artifact_path=tmp_path,
        api_operations={"GET /api/items", "POST /api/items"},
    )
    assert any("does not implement `POST /api/items`" in e for e in bad.errors)

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


# The launcher's PYTHONPATH injection (anton_state exposed via a private dir)
# must reach the introspection subprocess too. The backend asserts it at import
# time: in the dev venv `import anton_state` would succeed even without the
# injection (site-packages has it), so the assert is what actually proves the
# env went through.
BACKEND_CHECKS_INJECTION = BACKEND_OK.replace(
    "import os, argparse",
    "import os, argparse\n"
    'assert "anton_state_pp" in os.environ.get("PYTHONPATH", ""), '
    '"anton_state PYTHONPATH injection missing"\n'
    "import anton_state",
)

STATEFUL_BACKEND_OK = '''
import os, argparse
from pathlib import Path
from fastapi import FastAPI
from mangum import Mangum
from anton_state import open_store, Collection
app = FastAPI()
SECRETS = {}
STATE = None
_STATE_DIR = Path(__file__).resolve().parent

def get_store():
    return open_store(
        state=STATE,
        manifest_path=str(_STATE_DIR / "state_manifest.json"),
        local_path=str(_STATE_DIR / ".anton_state.db"),
    )

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/api/todos")
async def todos():
    return await Collection(get_store(), "todos").list()

handler = Mangum(app, lifespan="off")
'''

STATEFUL_MANIFEST = (
    '{"version": 1, "pk": {"name": "pk", "type": "S"},'
    ' "sk": {"name": "sk", "type": "S"}, "collections": ["todos"]}'
)


@_needs_asgi
async def test_verify_backend_injects_anton_state_pythonpath(tmp_path: Path):
    (tmp_path / "backend.py").write_text(BACKEND_CHECKS_INJECTION)
    (tmp_path / "requirements.txt").write_text("fastapi\nmangum\nuvicorn\n")
    result, _ = await verify_backend(
        scratchpad_pool=_FakePool(), slug="x", artifact_path=tmp_path,
    )
    assert result.ok, result.errors


@_needs_asgi
async def test_verify_backend_stateful_end_to_end(tmp_path: Path):
    (tmp_path / "backend.py").write_text(STATEFUL_BACKEND_OK)
    (tmp_path / "requirements.txt").write_text("fastapi\nmangum\nuvicorn\n")
    (tmp_path / "state_manifest.json").write_text(STATEFUL_MANIFEST)
    result, ds = await verify_backend(
        scratchpad_pool=_FakePool(), slug="x", artifact_path=tmp_path,
        artifact_type="fullstack-stateful-app",
    )
    assert result.ok, result.errors
    assert ds == []


@_needs_asgi
async def test_verify_backend_stateful_missing_manifest_fails(tmp_path: Path):
    (tmp_path / "backend.py").write_text(STATEFUL_BACKEND_OK)
    (tmp_path / "requirements.txt").write_text("fastapi\nmangum\nuvicorn\n")
    result, _ = await verify_backend(
        scratchpad_pool=_FakePool(), slug="x", artifact_path=tmp_path,
        artifact_type="fullstack-stateful-app",
    )
    assert not result.ok
    assert any("state_manifest.json is missing" in e for e in result.errors)
