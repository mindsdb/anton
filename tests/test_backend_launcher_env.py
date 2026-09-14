import os
from pathlib import Path

import anton_state
from anton.core.artifacts.backend_launcher import (
    _anton_state_pythonpath_dir,
    _build_backend_env,
)


def test_pythonpath_dir_contains_only_anton_state():
    d = _anton_state_pythonpath_dir()
    assert os.listdir(d) == ["anton_state"]
    # resolves to the real package
    assert (Path(d) / "anton_state" / "__init__.py").resolve() == Path(
        anton_state.__file__
    ).resolve()


def test_build_env_prepends_isolated_dir_to_pythonpath():
    env = _build_backend_env({"PYTHONPATH": "/existing"})
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == _anton_state_pythonpath_dir()
    assert "/existing" in parts


def test_build_env_without_existing_pythonpath():
    env = _build_backend_env(None)
    assert env["PYTHONPATH"] == _anton_state_pythonpath_dir()


def test_build_env_merges_extra_env():
    env = _build_backend_env({"DS_X__Y": "z"})
    assert env["DS_X__Y"] == "z"
    assert env["PATH"] == os.environ["PATH"]  # inherits parent env


def test_ds_env_replaces_inherited_ds_vars(monkeypatch):
    """A backend must see only the datasources its artifact declared, not
    whatever DS_* happen to be in the parent process."""
    monkeypatch.setenv("DS_LEFTOVER__PASSWORD", "from-another-turn")

    env = _build_backend_env(None, {"DS_DECLARED__PASSWORD": "mine"})

    assert env["DS_DECLARED__PASSWORD"] == "mine"
    assert "DS_LEFTOVER__PASSWORD" not in env
    assert env["PATH"] == os.environ["PATH"]  # non-DS_ vars still inherited


def test_ds_env_none_keeps_inherited_ds_vars(monkeypatch):
    """Callers that route DS_* through extra_env keep the old behaviour."""
    monkeypatch.setenv("DS_LEFTOVER__PASSWORD", "inherited")

    env = _build_backend_env(None)

    assert env["DS_LEFTOVER__PASSWORD"] == "inherited"


def test_empty_ds_env_still_strips(monkeypatch):
    """An artifact declaring no datasources gets no DS_* at all."""
    monkeypatch.setenv("DS_LEFTOVER__PASSWORD", "inherited")

    env = _build_backend_env(None, {})

    assert "DS_LEFTOVER__PASSWORD" not in env


def test_a_project_dotenv_cannot_override_a_vault_credential(monkeypatch):
    """extra_env is applied before the DS_* strip, so a project .env cannot
    replace or smuggle in a credential — same order a scratchpad uses."""
    env = _build_backend_env(
        {"DS_POSTGRES_PROD__PASSWORD": "from-dotenv", "OTHER": "kept"},
        {"DS_POSTGRES_PROD__PASSWORD": "from-vault"},
    )

    assert env["DS_POSTGRES_PROD__PASSWORD"] == "from-vault"
    assert env["OTHER"] == "kept"


def test_a_project_dotenv_cannot_add_an_undeclared_ds_var():
    env = _build_backend_env({"DS_SNEAKY__TOKEN": "from-dotenv"}, {})

    assert "DS_SNEAKY__TOKEN" not in env


# ── ENG-1382: the same DS_* rule on both launch paths ───────────────────────
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
    """`TurnKeyDataVault.env_for` does not drop `_`-prefixed fields despite its
    contract, so the filter lives here."""
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

    env = _build_backend_env(None, build_datasource_env(vault, [ref]))

    assert env[f"{ref.env_prefix}__HOST"] == "db"
    assert "DS_SOMEONE_ELSE__PASSWORD" not in env


async def test_the_generator_launches_with_only_the_declared_credentials(tmp_path, monkeypatch):
    """The pipeline's own launch must pass ds_env, not inherit the process's.

    Without it the generator's launch is the single path ENG-1382 does not
    cover — and it is the path that runs on every fullstack generation.
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
        "anton.core.tools.tool_handlers._artifact_store",
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
