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
