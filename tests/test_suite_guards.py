"""The suite's own guards must stay armed.

conftest.py disarms things the suite is never allowed to do, and a disarmed
side effect is invisible when it works — which is exactly how it rots. Delete
the browser guard and three real `webbrowser.open` calls come back on every
local run while CI stays green, because no assertion anywhere depends on it
(ENG-1453). These tests are the thing that fails instead.
"""
from __future__ import annotations

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
