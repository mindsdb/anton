"""Which environment names agent code may see, with or without a sandbox policy.

``agent_env`` is the one filter both cell paths use. These tests feed it an
environment shaped like a desktop server's and check what survives.
"""

from __future__ import annotations

import pytest

from anton.core.backends.sandbox import agent_env

_SERVER_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/Users/person",
    "USER": "person",
    "TMPDIR": "/var/folders/xy/T/",
    "XDG_CONFIG_HOME": "/Users/person/.config",
    "SYSTEMROOT": "C:\\Windows",
    "COWORK_AUTH_TOKEN": "tok-secret",
    "DATABASE_URI": "sqlite:////Users/person/.cowork/cowork.db",
    "MINDS_API_KEY": "minds-secret",
    "ANTON_MINDS_API_KEY": "anton-minds-secret",
    "GOOGLE_DRIVE_CLIENT_SECRET": "gd-secret",
    "AWS_PROFILE": "work",
    "OPENAI_API_KEY": "coding-key",
    "DS_SERVER__PASSWORD": "server-ds-secret",
    "PYTHONNOUSERSITE": "1",
    "DYLD_INSERT_LIBRARIES": "/tmp/evil.dylib",
}

_SECRETS = (
    "COWORK_AUTH_TOKEN",
    "DATABASE_URI",
    "MINDS_API_KEY",
    "ANTON_MINDS_API_KEY",
    "GOOGLE_DRIVE_CLIENT_SECRET",
)


@pytest.mark.parametrize("host_basics", [False, True])
def test_server_secrets_and_unlisted_names_never_pass(host_basics):
    child = agent_env(_SERVER_ENV, ds_env={}, host_basics=host_basics)
    for name in (*_SECRETS, "AWS_PROFILE"):
        assert name not in child


def test_the_parents_home_temp_user_and_system_names_pass_only_with_host_basics():
    basics = ("HOME", "USER", "TMPDIR", "XDG_CONFIG_HOME", "SYSTEMROOT")

    without = agent_env(_SERVER_ENV, ds_env={}, host_basics=False)
    with_basics = agent_env(_SERVER_ENV, ds_env={}, host_basics=True)

    assert not any(name in without for name in basics)
    assert {name: with_basics[name] for name in basics} == {name: _SERVER_ENV[name] for name in basics}
    assert with_basics["PATH"] == without["PATH"] == "/usr/bin:/bin"


@pytest.mark.parametrize("host_basics", [False, True])
def test_the_pods_user_site_switch_passes(host_basics):
    assert agent_env(_SERVER_ENV, ds_env={}, host_basics=host_basics)["PYTHONNOUSERSITE"] == "1"


def test_a_loader_injection_name_never_passes_from_the_parent_or_the_dotenv():
    env = {**_SERVER_ENV, "DYLD_LIBRARY_PATH": "/project/lib"}

    child = agent_env(env, overlay_keys={"DYLD_LIBRARY_PATH"}, ds_env={}, host_basics=True)

    assert "DYLD_INSERT_LIBRARIES" not in child
    assert "DYLD_LIBRARY_PATH" not in child


def test_a_dotenv_cannot_set_the_childs_home_or_temp_folder():
    env = {"PATH": "/usr/bin", "HOME": "/project/fake-home", "TMPDIR": "/project/tmp"}

    child = agent_env(env, overlay_keys={"HOME", "TMPDIR"}, ds_env={}, host_basics=True)

    assert "HOME" not in child
    assert "TMPDIR" not in child


def test_given_connections_replace_every_inherited_ds_value():
    child = agent_env(_SERVER_ENV, ds_env={"DS_PAD__PASSWORD": "pad-own"}, host_basics=True)

    assert "DS_SERVER__PASSWORD" not in child
    assert child["DS_PAD__PASSWORD"] == "pad-own"


def test_no_connections_given_keeps_the_ds_values_already_in_the_env():
    """``child_env`` callers put the pad's own DS_* in ``env`` first and rely on this."""
    child = agent_env(_SERVER_ENV, ds_env=None)

    assert child["DS_SERVER__PASSWORD"] == "server-ds-secret"


def test_a_given_connection_value_execve_refuses_is_dropped():
    child = agent_env({"PATH": "/usr/bin"}, ds_env={"DS_BAD__PASSWORD": "a\0b", "DS_OK__HOST": "h"})

    assert "DS_BAD__PASSWORD" not in child
    assert child["DS_OK__HOST"] == "h"


def test_the_env_given_is_not_changed():
    env = dict(_SERVER_ENV)

    agent_env(env, ds_env={"DS_PAD__HOST": "h"}, host_basics=True)

    assert env == _SERVER_ENV
