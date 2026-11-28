import os
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

import anton_state
from anton.core.artifacts import backend_launcher
from anton.core.artifacts.backend_launcher import (
    _anton_state_pythonpath_dir,
    build_backend_env,
    launch_artifact_backend,
)


def test_pythonpath_dir_contains_only_anton_state():
    d = _anton_state_pythonpath_dir()
    assert os.listdir(d) == ["anton_state"]
    # resolves to the real package
    assert (Path(d) / "anton_state" / "__init__.py").resolve() == Path(
        anton_state.__file__
    ).resolve()


# ── who can create or fill the import folder ────────────────────────────────
#
# The folder sits in the temp folder, which every account shares on Linux, and
# Python imports any module on PYTHONPATH (`sitecustomize` at every start). So
# a name another account could have made first must never reach PYTHONPATH.

_posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX owners and modes")


@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    """The temp folder as `tempfile.gettempdir()` reports it, fresh per test."""
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    # raising=False: the tests below also run against code without a fallback.
    monkeypatch.setattr(backend_launcher, "_fallback_pythonpath_root", None, raising=False)
    return temp


def _names_another_account_could_take() -> list[str]:
    """Every name in the temp folder the launcher may use: the old fixed one and the per-user one."""
    return ["anton_state_pp", f"anton_state_pp-{os.getuid()}"]


@_posix_only
def test_the_import_folder_is_a_real_folder_only_this_user_can_open(temp_dir):
    old_umask = os.umask(0o022)
    try:
        folder = _anton_state_pythonpath_dir()
    finally:
        os.umask(old_umask)

    st = os.lstat(folder)
    assert stat.S_ISDIR(st.st_mode)
    assert st.st_uid == os.getuid()
    assert stat.S_IMODE(st.st_mode) == 0o700
    assert os.listdir(folder) == ["anton_state"]


@_posix_only
def test_a_world_writable_folder_planted_at_the_name_never_runs_its_code(temp_dir, tmp_path):
    """Another account's `mkdir -m 1777` plus a sitecustomize.py: nothing it holds may run."""
    marker = tmp_path / "planted-code-ran"
    planted = []
    for name in _names_another_account_could_take():
        folder = temp_dir / name
        folder.mkdir()
        os.chmod(folder, 0o1777)
        (folder / "sitecustomize.py").write_text(
            f"import pathlib\npathlib.Path({str(marker)!r}).write_text('ran')\n"
        )
        planted.append(str(folder))

    env = build_backend_env(None)
    subprocess.run([sys.executable, "-c", "pass"], env=env, check=True)

    assert not marker.exists()
    assert env["PYTHONPATH"].split(os.pathsep)[0] not in planted
    for folder in planted:
        assert sorted(os.listdir(folder)) == ["sitecustomize.py"]


@_posix_only
def test_a_symlink_planted_at_the_name_is_never_followed(temp_dir, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.chmod(elsewhere, 0o777)
    for name in _names_another_account_could_take():
        (temp_dir / name).symlink_to(elsewhere, target_is_directory=True)

    folder = _anton_state_pythonpath_dir()

    assert os.listdir(elsewhere) == []
    assert not os.path.islink(folder)
    assert elsewhere.resolve() not in Path(folder).resolve().parents
    assert Path(folder).resolve() != elsewhere.resolve()


@_posix_only
def test_a_folder_another_account_owns_is_not_used(temp_dir, monkeypatch):
    """Simulated with a stubbed uid: the folders below are this test's, so to
    that uid they belong to someone else."""
    other_uid = os.getuid() + 1
    # Only the candidate ownership is simulated. Keep this fixture's real
    # temporary parent pinned while the effective uid is mocked.
    monkeypatch.setattr(backend_launcher, "_trusted_temp_parent", lambda: temp_dir)
    monkeypatch.setattr(os, "getuid", lambda: other_uid)
    planted = []
    for name in _names_another_account_could_take():
        folder = temp_dir / name
        folder.mkdir(mode=0o700)
        planted.append(str(folder))

    folder = _anton_state_pythonpath_dir()

    assert folder not in planted
    for name in planted:
        assert os.listdir(name) == []


@_posix_only
def test_a_refused_name_gives_one_folder_for_the_life_of_the_process(temp_dir):
    """The fallback is made once and kept: backends started from it go on
    importing from it, and later launches reuse it."""
    for name in _names_another_account_could_take():
        (temp_dir / name).write_text("not a folder")

    first = _anton_state_pythonpath_dir()
    second = _anton_state_pythonpath_dir()

    assert first == second
    assert os.listdir(first) == ["anton_state"]
    assert stat.S_IMODE(os.lstat(first).st_mode) == 0o700


@_posix_only
def test_configured_temp_with_writable_ancestor_uses_trusted_parent(temp_dir, monkeypatch):
    shared = temp_dir.parent / "shared"
    child = shared / "subdirectory"
    child.mkdir(parents=True)
    shared.chmod(0o777)
    try:
        monkeypatch.setattr(tempfile, "tempdir", str(child))
        parent = backend_launcher._trusted_temp_parent()
        assert shared not in (parent, *parent.parents)
        assert parent != child
    finally:
        shared.chmod(0o700)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows profile lookup and ACLs")
def test_windows_imports_use_the_profile_despite_configured_temp(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    profile.mkdir()
    unrelated = tmp_path / "configured-temp"
    unrelated.mkdir()
    monkeypatch.setattr(backend_launcher, "_windows_profile", lambda: profile)
    monkeypatch.setattr(tempfile, "tempdir", str(unrelated))

    directory = Path(_anton_state_pythonpath_dir())

    assert directory == profile / ".anton" / "anton_state_pp"
    assert list(unrelated.iterdir()) == []


def test_build_env_prepends_isolated_dir_to_pythonpath():
    env = build_backend_env({"PYTHONPATH": "/existing"})
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == _anton_state_pythonpath_dir()
    assert "/existing" in parts


def test_build_env_without_existing_pythonpath():
    env = build_backend_env(None)
    assert env["PYTHONPATH"] == _anton_state_pythonpath_dir()


def test_build_env_merges_extra_env():
    env = build_backend_env({"DS_X__Y": "z"})
    assert env["DS_X__Y"] == "z"
    assert env["PATH"] == os.environ["PATH"]  # inherits parent env


def test_build_env_disables_bytecode_writes(monkeypatch):
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)

    assert build_backend_env(None)["PYTHONDONTWRITEBYTECODE"] == "1"


def test_ds_env_replaces_inherited_ds_vars(monkeypatch):
    """A backend must see only the datasources its artifact declared, not
    whatever DS_* happen to be in the parent process."""
    monkeypatch.setenv("DS_LEFTOVER__PASSWORD", "from-another-turn")

    env = build_backend_env(None, {"DS_DECLARED__PASSWORD": "mine"})

    assert env["DS_DECLARED__PASSWORD"] == "mine"
    assert "DS_LEFTOVER__PASSWORD" not in env
    assert env["PATH"] == os.environ["PATH"]  # non-DS_ vars still inherited


def test_ds_env_none_keeps_inherited_ds_vars(monkeypatch):
    """Callers that route DS_* through extra_env keep the old behaviour."""
    monkeypatch.setenv("DS_LEFTOVER__PASSWORD", "inherited")

    env = build_backend_env(None)

    assert env["DS_LEFTOVER__PASSWORD"] == "inherited"


def test_empty_ds_env_still_strips(monkeypatch):
    """An artifact declaring no datasources gets no DS_* at all."""
    monkeypatch.setenv("DS_LEFTOVER__PASSWORD", "inherited")

    env = build_backend_env(None, {})

    assert "DS_LEFTOVER__PASSWORD" not in env


def test_a_project_dotenv_cannot_override_a_vault_credential(monkeypatch):
    """extra_env is applied before the DS_* strip, so a project .env cannot
    replace or smuggle in a credential — same order a scratchpad uses."""
    env = build_backend_env(
        {"DS_POSTGRES_PROD__PASSWORD": "from-dotenv", "OTHER": "kept"},
        {"DS_POSTGRES_PROD__PASSWORD": "from-vault"},
    )

    assert env["DS_POSTGRES_PROD__PASSWORD"] == "from-vault"
    assert env["OTHER"] == "kept"


def test_a_project_dotenv_cannot_add_an_undeclared_ds_var():
    env = build_backend_env({"DS_SNEAKY__TOKEN": "from-dotenv"}, {})

    assert "DS_SNEAKY__TOKEN" not in env


# ── the same DS_* rule on both launch paths ─────────────────────────────────
#
# `handle_launch_backend` is what the agent calls; `generate_artifact` launches
# the backend itself at the end of its pipeline and never reaches that handler.
# Building the credential set in only one of them makes an artifact's exposure
# depend on WHO started it, which is not a property anyone would choose.

from types import SimpleNamespace
from unittest.mock import AsyncMock

from anton.core.artifacts.backend_launcher import build_datasource_env
from anton.core.artifacts.models import DatasourceRef


class _Vault:
    def __init__(self, envs: dict[str, dict[str, str]]):
        self._envs = envs

    def env_for(self, engine: str, name: str, *, flat: bool = False):
        return self._envs.get(f"{engine}-{name}")


def _ref(engine="postgres", name="prod_db") -> DatasourceRef:
    return DatasourceRef(engine=engine, name=name)


def test_declared_datasource_credentials_are_collected():
    ref = _ref()
    vault = _Vault({"postgres-prod_db": {f"{ref.env_prefix}__HOST": "db.internal"}})
    assert build_datasource_env(vault, [ref]) == {f"{ref.env_prefix}__HOST": "db.internal"}


def test_bookkeeping_fields_never_reach_the_backend():
    """The launcher drops `_`-prefixed fields itself, whatever the vault
    returns."""
    ref = _ref()
    vault = _Vault({"postgres-prod_db": {
        f"{ref.env_prefix}__HOST": "db.internal",
        f"{ref.env_prefix}___user_label": "Prod DB",
    }})
    env = build_datasource_env(vault, [ref])
    assert env == {f"{ref.env_prefix}__HOST": "db.internal"}


def test_one_unresolvable_source_does_not_deny_the_others():
    good, gone = _ref(), _ref(name="missing")
    vault = _Vault({"postgres-prod_db": {f"{good.env_prefix}__HOST": "db"}})
    env = build_datasource_env(vault, [gone, good])
    assert env == {f"{good.env_prefix}__HOST": "db"}


def test_no_vault_yields_nothing_rather_than_raising():
    assert build_datasource_env(None, [_ref()]) == {}


def test_nothing_declared_yields_an_empty_set_not_none():
    """`{}` and None mean opposite things downstream: `{}` strips the inherited
    DS_*, None leaves them. An artifact that declared no datasource must get
    the strip."""
    assert build_datasource_env(_Vault({}), []) == {}
    assert build_datasource_env(_Vault({}), None) == {}


def test_an_undeclared_variable_is_stripped_from_the_backend_env(monkeypatch):
    ref = _ref()
    monkeypatch.setenv("DS_SOMEONE_ELSE__PASSWORD", "not-yours")
    vault = _Vault({"postgres-prod_db": {f"{ref.env_prefix}__HOST": "db"}})

    env = build_backend_env(None, build_datasource_env(vault, [ref]))

    assert env[f"{ref.env_prefix}__HOST"] == "db"
    assert "DS_SOMEONE_ELSE__PASSWORD" not in env


async def test_the_generator_launches_with_only_the_declared_credentials(tmp_path, monkeypatch):
    """The pipeline's own launch must pass ds_env, not inherit the process's.

    Without it the generator's launch is the single path the declared-only
    rule does not cover — and it runs on every fullstack generation.
    """
    from anton.core.tools.generate_artifact import orchestrator
    from anton.core.tools.generate_artifact.state import GenState

    ref = _ref()
    monkeypatch.setenv("DS_SOMEONE_ELSE__PASSWORD", "not-yours")
    session = AsyncMock()
    session._data_vault = _Vault({"postgres-prod_db": {f"{ref.env_prefix}__HOST": "db"}})
    session._tracked_backends = {}
    state = GenState(
        session=session, artifact_type="fullstack-stateless-app",
        artifact_path=tmp_path, slug="s", user_request="r", agent_understanding="u",
    )
    # Patched where it is defined, not on `orchestrator`: the node imports it
    # inside the function body, so a module attribute here would never be read
    # and this test would pass on an inert patch.
    monkeypatch.setattr(
        "anton.core.tools.tool_handlers.resolve_artifact_store",
        lambda s: SimpleNamespace(
            open=lambda slug: SimpleNamespace(datasources=[ref]),
            update=lambda *a, **k: None,
        ),
    )
    seen = {}

    async def _fake_launch(**kw):
        seen.update(kw)
        return "launcher refused"  # short-circuits the rest of the node

    monkeypatch.setattr(orchestrator, "_launch_backend", _fake_launch)
    monkeypatch.setattr(orchestrator, "_tail_log", AsyncMock(return_value=""))
    monkeypatch.setattr(orchestrator, "_gen_verify_backend", AsyncMock(return_value=None))

    await orchestrator._run_and_verify_app(state)

    assert seen["ds_env"] == {f"{ref.env_prefix}__HOST": "db"}


def test_both_launch_paths_build_ds_env_through_the_shared_helper():
    """An inventory lock, not a style rule: the drift this prevents is silent
    and only visible at runtime, in a subprocess, under the right vault."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "anton"
    callers = set()
    for rel in ("core/tools/tool_handlers.py",
                "core/tools/generate_artifact/orchestrator.py"):
        tree = ast.parse((root / rel).read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "build_datasource_env"):
                callers.add(rel)
    assert callers == {
        "core/tools/tool_handlers.py",
        "core/tools/generate_artifact/orchestrator.py",
    }, f"a launch path stopped using the shared helper: {sorted(callers)}"


class _PoolWithHostPython:
    async def venv_python(self, slug):
        return sys.executable


_LOGGED_TOKEN = "FAKE0tok3nZr1xK2mN4pL7vT9wY3sB6dF"


async def test_an_early_exit_log_tail_redacts_a_logged_token(tmp_path):
    """A backend that logs a token response and dies returns its log tail as
    the tool result; the token must not ride along to the model."""
    (tmp_path / "backend.py").write_text(
        f"print({{'access_token': '{_LOGGED_TOKEN}'}}, flush=True)\n"
        "raise SystemExit(1)\n"
    )

    result = await launch_artifact_backend(
        slug="app",
        artifact_folder=tmp_path,
        scratchpad_pool=_PoolWithHostPython(),
        tracked_backends={},
    )

    assert isinstance(result, str) and "exited early" in result
    assert "access_token" in result
    assert _LOGGED_TOKEN not in result


async def test_a_not_ready_log_tail_redacts_a_logged_token(tmp_path):
    """The readiness timeout path returns the same log tail."""
    (tmp_path / "backend.py").write_text(
        "import time\n"
        f"print('Authorization: Bearer {_LOGGED_TOKEN}', flush=True)\n"
        "time.sleep(30)\n"
    )

    result = await launch_artifact_backend(
        slug="app",
        artifact_folder=tmp_path,
        scratchpad_pool=_PoolWithHostPython(),
        tracked_backends={},
        health_timeout=1.0,
    )

    assert isinstance(result, str) and "did not become ready" in result
    assert "Authorization: Bearer" in result
    assert _LOGGED_TOKEN not in result


async def test_a_log_tail_cut_between_key_and_token_still_redacts_it(tmp_path):
    """The tail is the last 2000 characters; a cut just before the token must
    not leave it without the key that identifies it."""
    filler = "x" * (2000 - len(_LOGGED_TOKEN) - len('"}') - len("\n"))
    (tmp_path / "backend.py").write_text(
        f"print('INFO got {{\"access_token\": \"{_LOGGED_TOKEN}\"}}{filler}', flush=True)\n"
        "raise SystemExit(1)\n"
    )

    result = await launch_artifact_backend(
        slug="app",
        artifact_folder=tmp_path,
        scratchpad_pool=_PoolWithHostPython(),
        tracked_backends={},
    )

    assert isinstance(result, str) and "exited early" in result
    assert _LOGGED_TOKEN not in result


async def test_the_log_tail_is_scrubbed_off_the_event_loop(tmp_path, monkeypatch):
    """The whole log is scrubbed and it grows across relaunches, so the work
    must not block the event loop."""
    loop_thread = threading.get_ident()
    scrub_threads: list[int] = []

    def recording_scrub(text):
        scrub_threads.append(threading.get_ident())
        return text

    monkeypatch.setattr(backend_launcher, "scrub_output", recording_scrub)
    (tmp_path / "backend.py").write_text("raise SystemExit(1)\n")

    await launch_artifact_backend(
        slug="app",
        artifact_folder=tmp_path,
        scratchpad_pool=_PoolWithHostPython(),
        tracked_backends={},
    )

    assert scrub_threads and loop_thread not in scrub_threads


async def test_a_launched_backend_leaves_no_bytecode_in_the_artifact_folder(tmp_path, monkeypatch):
    """`backend.py` runs with the artifact folder on sys.path, so its local
    imports would otherwise leave `__pycache__` next to them."""
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    (tmp_path / "helpers.py").write_text("VALUE = 1\n")
    (tmp_path / "routes").mkdir()
    (tmp_path / "routes" / "x.py").write_text("VALUE = 2\n")
    (tmp_path / "backend.py").write_text(
        "import helpers\nimport routes.x\nraise SystemExit(1)\n"
    )

    result = await launch_artifact_backend(
        slug="app",
        artifact_folder=tmp_path,
        scratchpad_pool=_PoolWithHostPython(),
        tracked_backends={},
    )

    assert isinstance(result, str) and "exited early" in result
    assert list(tmp_path.rglob("__pycache__")) == []
    assert list(tmp_path.rglob("*.pyc")) == []
