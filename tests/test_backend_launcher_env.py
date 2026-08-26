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
