"""The suite's own guards must stay armed.

conftest.py disarms things the suite is never allowed to do, and a disarmed
side effect is invisible when it works — which is exactly how it rots. Delete
the browser guard and three real `webbrowser.open` calls come back on every
local run while CI stays green, because no assertion anywhere depends on it
(ENG-1453). These tests are the thing that fails instead.
"""
from __future__ import annotations

import os
import webbrowser

import pytest


@pytest.mark.parametrize("name", ["open", "open_new", "open_new_tab"])
def test_browser_guard_is_installed(name):
    """`_no_browser_windows` must have replaced the stdlib entry point.

    Checked by provenance rather than behaviour: actually calling the real
    function is the failure mode under test, so the assertion has to hold
    before any call happens.
    """
    fn = getattr(webbrowser, name)
    assert fn.__module__ != "webbrowser", (
        f"the _no_browser_windows guard in tests/conftest.py is not covering "
        f"webbrowser.{name} — a local run will open real browser windows"
    )
    assert callable(fn)


# Deliberately NOT asserted by calling it: `webbrowser.open(...) is True` holds
# whether the guard is installed or not — the real function also returns True,
# after opening a window. Mutation-verified: with the guard neutered that
# assertion passes *and* opens a browser, making it both a blind test and an
# instance of the very problem this file exists to catch.


def test_home_guard_is_installed():
    """`_no_real_home` must have redirected `Path.home()` away from the real one.

    Same rot pattern as the browser guard, and it had already happened: on
    `origin/staging` a full run wrote `ANTON_MINDS_API_KEY=goodkey` and
    `ANTON_FIRST_RUN_DONE=true` into the developer's real `~/.anton/.env` and
    still reported all green, because the publish tests assert on a mocked
    *project* workspace and nothing ever looked at the global vault (ENG-1424).

    Asserted by provenance, not by writing a file: the failure mode is the
    write itself, so this has to hold before anything touches disk.
    """
    from pathlib import Path

    from tests.conftest import REAL_HOME

    assert str(Path.home()) != REAL_HOME, (
        "the _no_real_home guard in tests/conftest.py is not armed — "
        "tests that persist credentials will write to the real ~/.anton/.env"
    )
    assert os.environ.get("HOME") != REAL_HOME, (
        "the _no_real_home guard redirects Path.home() but not $HOME — every "
        "expanduser('~') writer, and every child process, still hits the real home"
    )
